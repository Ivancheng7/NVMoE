use super::*;

#[derive(Default)]
pub(crate) struct MhcTelemetry {
    pub(crate) wall_ns: u64,
    pub(crate) input_alloc_h2d_sync_ns: u64,
    pub(crate) workspace_alloc_sync_ns: u64,
    pub(crate) workspace_init_gpu_ns: u64,
    pub(crate) mixing_linear_sync_ns: u64,
    pub(crate) mixing_linear_gpu_ns: u64,
    pub(crate) sinkhorn_normalize_collapse_sync_ns: u64,
    pub(crate) sinkhorn_normalize_collapse_gpu_ns: u64,
    pub(crate) output_d2h_sync_ns: u64,
    pub(crate) other_ns: u64,
}

impl MhcTelemetry {
    pub(crate) fn accounted_wall_ns(&self) -> u64 {
        self.input_alloc_h2d_sync_ns
            .saturating_add(self.workspace_alloc_sync_ns)
            .saturating_add(self.mixing_linear_sync_ns)
            .saturating_add(self.sinkhorn_normalize_collapse_sync_ns)
            .saturating_add(self.output_d2h_sync_ns)
            .saturating_add(self.other_ns)
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_l0_mhc_pre_device_into(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    input: &DeviceBuffer<u16>,
    hc_fn: &DeviceBuffer<f32>,
    hc_scale: &DeviceBuffer<f32>,
    hc_base: &DeviceBuffer<f32>,
    mixes: &mut DeviceBuffer<f32>,
    pre: &mut DeviceBuffer<f32>,
    post: &mut DeviceBuffer<f32>,
    comb: &mut DeviceBuffer<f32>,
    collapsed: &mut DeviceBuffer<u16>,
    trace: &mut DeviceBuffer<f32>,
    mut telemetry: Option<&mut MhcTelemetry>,
) -> Result<(), Box<dyn Error>> {
    if hc_fn.len() % (v4_state::MIX * v4_state::HC) != 0 {
        return Err("device mHC weight geometry mismatch".into());
    }
    let hidden = hc_fn.len() / (v4_state::MIX * v4_state::HC);
    let row_width = v4_state::HC * hidden;
    if input.is_empty()
        || !input.len().is_multiple_of(row_width)
        || mixes.len() != input.len() / row_width * v4_state::MIX
        || pre.len() != input.len() / row_width * v4_state::HC
        || post.len() != pre.len()
        || comb.len() != pre.len() * v4_state::HC
        || collapsed.len() != input.len() / v4_state::HC
        || trace.len() != input.len() / row_width * 41 * v4_state::HC * v4_state::HC
    {
        return Err("device mHC workspace contract mismatch".into());
    }
    let rows = u32::try_from(input.len() / row_width)?;
    let linear_launch =
        LaunchConfig::for_num_elems(rows.checked_mul(32).ok_or("mHC linear launch overflow")?);
    let row_launch = LaunchConfig::for_num_elems(rows);
    let profile = telemetry.is_some();
    let events = profile
        .then(|| {
            Ok::<_, Box<dyn Error>>((
                input
                    .context()
                    .new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?,
                input
                    .context()
                    .new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?,
            ))
        })
        .transpose()?;
    let mixing_started = std::time::Instant::now();
    if let Some((start, _)) = events.as_ref() {
        start.record(stream)?;
    }
    // SAFETY: each kernel owns one validated row and every output is disjoint.
    unsafe {
        module.mhc_linear_v4_bf16(stream, linear_launch, input, hc_fn, mixes)?;
    }
    if let Some((start, end)) = events.as_ref() {
        end.record(stream)?;
        end.synchronize()?;
        let telemetry = telemetry.as_deref_mut().expect("profile has mHC telemetry");
        telemetry.mixing_linear_gpu_ns = (start.elapsed_ms(end)? as f64 * 1_000_000.0) as u64;
        telemetry.mixing_linear_sync_ns =
            u64::try_from(mixing_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
    }
    let sinkhorn_started = std::time::Instant::now();
    if let Some((start, _)) = events.as_ref() {
        start.record(stream)?;
    }
    // SAFETY: each kernel owns one validated row and every output is disjoint.
    unsafe {
        module.mhc_sinkhorn_v4_bf16(
            stream, row_launch, input, mixes, hc_scale, hc_base, pre, post, comb, collapsed, trace,
        )?;
    }
    if let Some((start, end)) = events.as_ref() {
        end.record(stream)?;
        end.synchronize()?;
        let telemetry = telemetry.as_deref_mut().expect("profile has mHC telemetry");
        telemetry.sinkhorn_normalize_collapse_gpu_ns =
            (start.elapsed_ms(end)? as f64 * 1_000_000.0) as u64;
        telemetry.sinkhorn_normalize_collapse_sync_ns =
            u64::try_from(sinkhorn_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
    }
    Ok(())
}

pub(crate) fn run_l0_mhc_pre(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    input: &[u16],
    hc_fn: &DeviceBuffer<f32>,
    hc_scale: &DeviceBuffer<f32>,
    hc_base: &DeviceBuffer<f32>,
) -> Result<MhcGpuOutput, Box<dyn Error>> {
    run_l0_mhc_pre_with_arm(
        module,
        stream,
        input,
        hc_fn,
        hc_scale,
        hc_base,
        FlashMhcLinearArm::Canonical,
        false,
        None,
    )
}

pub(crate) fn run_l0_mhc_pre_with_arm(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    input: &[u16],
    hc_fn: &DeviceBuffer<f32>,
    hc_scale: &DeviceBuffer<f32>,
    hc_base: &DeviceBuffer<f32>,
    linear_arm: FlashMhcLinearArm,
    capture_rsqrt_debug: bool,
    session: Option<&FlashCublasSession>,
) -> Result<MhcGpuOutput, Box<dyn Error>> {
    if hc_fn.len() % (v4_state::MIX * v4_state::HC) != 0 {
        return Err("L0 mHC weight geometry mismatch".into());
    }
    let hidden = hc_fn.len() / (v4_state::MIX * v4_state::HC);
    let row_width = v4_state::HC * hidden;
    if input.len() % row_width != 0 || input.is_empty() {
        return Err("L0 mHC input is not complete rows".into());
    }
    let started = std::time::Instant::now();
    let profile = bench_telemetry_enabled();
    let mut telemetry = MhcTelemetry::default();
    let rows = input.len() / row_width;
    let input_started = std::time::Instant::now();
    let input = DeviceBuffer::from_host(stream, input)?;
    if profile {
        telemetry.input_alloc_h2d_sync_ns =
            u64::try_from(input_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
    }
    let workspace_started = std::time::Instant::now();
    let workspace_events = profile
        .then(|| {
            Ok::<_, Box<dyn Error>>((
                input
                    .context()
                    .new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?,
                input
                    .context()
                    .new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?,
            ))
        })
        .transpose()?;
    if let Some((start, _)) = workspace_events.as_ref() {
        start.record(stream)?;
    }
    let mut mixes = DeviceBuffer::<f32>::zeroed(stream, rows * v4_state::MIX)?;
    let mut pre = DeviceBuffer::<f32>::zeroed(stream, rows * v4_state::HC)?;
    let mut post = DeviceBuffer::<f32>::zeroed(stream, rows * v4_state::HC)?;
    let mut comb = DeviceBuffer::<f32>::zeroed(stream, rows * v4_state::HC * v4_state::HC)?;
    let mut collapsed = DeviceBuffer::<u16>::zeroed(stream, rows * hidden)?;
    let mut trace = DeviceBuffer::<f32>::zeroed(stream, rows * 41 * v4_state::HC * v4_state::HC)?;
    if let Some((start, end)) = workspace_events.as_ref() {
        end.record(stream)?;
        end.synchronize()?;
        telemetry.workspace_init_gpu_ns = (start.elapsed_ms(end)? as f64 * 1_000_000.0) as u64;
        telemetry.workspace_alloc_sync_ns =
            u64::try_from(workspace_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
    }
    let owned_session = (linear_arm == FlashMhcLinearArm::Cublas && session.is_none())
        .then(|| FlashCublasSession::new(stream))
        .transpose()?;
    let session = session.or(owned_session.as_ref());
    let cublas_buffers = if linear_arm == FlashMhcLinearArm::Canonical {
        run_l0_mhc_pre_device_into(
            module,
            stream,
            &input,
            hc_fn,
            hc_scale,
            hc_base,
            &mut mixes,
            &mut pre,
            &mut post,
            &mut comb,
            &mut collapsed,
            &mut trace,
            profile.then_some(&mut telemetry),
        )?;
        None
    } else {
        let session = session.ok_or("Flash cuBLAS mHC session is missing")?;
        let linear_events = profile
            .then(|| {
                Ok::<_, Box<dyn Error>>((
                    input
                        .context()
                        .new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?,
                    input
                        .context()
                        .new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?,
                ))
            })
            .transpose()?;
        let linear_started = std::time::Instant::now();
        if let Some((start, _)) = linear_events.as_ref() {
            start.record(stream)?;
        }
        let buffers = run_flash_mhc_cublas_linear_device_into(
            session,
            module,
            stream,
            &input,
            hc_fn,
            &mut mixes,
            rows,
            hidden,
            capture_rsqrt_debug,
        )?;
        if let Some((start, end)) = linear_events.as_ref() {
            end.record(stream)?;
            end.synchronize()?;
            telemetry.mixing_linear_gpu_ns = (start.elapsed_ms(end)? as f64 * 1_000_000.0) as u64;
            telemetry.mixing_linear_sync_ns =
                u64::try_from(linear_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        }
        let sinkhorn_events = profile
            .then(|| {
                Ok::<_, Box<dyn Error>>((
                    input
                        .context()
                        .new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?,
                    input
                        .context()
                        .new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?,
                ))
            })
            .transpose()?;
        let sinkhorn_started = std::time::Instant::now();
        if let Some((start, _)) = sinkhorn_events.as_ref() {
            start.record(stream)?;
        }
        // SAFETY: the cuBLAS arm wrote complete mixes rows; Sinkhorn owns its
        // disjoint output tensors in the same stream.
        unsafe {
            module.mhc_sinkhorn_v4_bf16(
                stream,
                LaunchConfig::for_num_elems(u32::try_from(rows)?),
                &input,
                &mixes,
                hc_scale,
                hc_base,
                &mut pre,
                &mut post,
                &mut comb,
                &mut collapsed,
                &mut trace,
            )?;
        }
        if let Some((start, end)) = sinkhorn_events.as_ref() {
            end.record(stream)?;
            end.synchronize()?;
            telemetry.sinkhorn_normalize_collapse_gpu_ns =
                (start.elapsed_ms(end)? as f64 * 1_000_000.0) as u64;
            telemetry.sinkhorn_normalize_collapse_sync_ns =
                u64::try_from(sinkhorn_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        }
        Some(buffers)
    };
    let d2h_started = std::time::Instant::now();
    let collapsed = collapsed.to_host_vec(stream)?;
    let post = post.to_host_vec(stream)?;
    let comb = comb.to_host_vec(stream)?;
    let debug = std::env::var_os("COLIBRI_V4_FLASH_PARITY_DUMP_DIR")
        .is_some()
        .then(|| {
            Ok::<_, Box<dyn Error>>(MhcDebugOutput {
                mixes: mixes.to_host_vec(stream)?,
                pre: pre.to_host_vec(stream)?,
                sinkhorn_trace: trace.to_host_vec(stream)?,
                raw_dot: cublas_buffers
                    .as_ref()
                    .map(|buffers| buffers.raw_dot.to_host_vec(stream))
                    .transpose()?,
                rsqrt: cublas_buffers
                    .as_ref()
                    .map(|buffers| buffers.rsqrt.to_host_vec(stream))
                    .transpose()?,
                rsqrt_debug: cublas_buffers
                    .as_ref()
                    .and_then(|buffers| buffers.rsqrt_debug.as_ref())
                    .map(|debug| debug.to_host_vec(stream))
                    .transpose()?,
            })
        })
        .transpose()?;
    if profile {
        telemetry.output_d2h_sync_ns =
            u64::try_from(d2h_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
    }
    telemetry.wall_ns = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
    telemetry.other_ns = telemetry
        .wall_ns
        .saturating_sub(telemetry.accounted_wall_ns());
    Ok(MhcGpuOutput {
        collapsed,
        post,
        comb,
        debug,
        telemetry,
    })
}

pub(crate) fn run_l0_rmsnorm_device_into(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    input: &DeviceBuffer<u16>,
    norm: &DeviceBuffer<u16>,
    output: &mut DeviceBuffer<u16>,
    width: usize,
) -> Result<(), Box<dyn Error>> {
    if input.is_empty()
        || !input.len().is_multiple_of(width)
        || (!norm.is_empty() && norm.len() != width)
        || output.len() != input.len()
    {
        return Err("device RMSNorm workspace contract mismatch".into());
    }
    // SAFETY: complete fixed-width rows and equally sized output were checked.
    unsafe {
        module.rmsnorm_v4_bf16(
            stream,
            LaunchConfig::for_num_elems(u32::try_from(input.len() / width)?),
            input,
            norm,
            output,
            u32::try_from(width)?,
        )?;
    }
    Ok(())
}

pub(crate) fn run_flash_learned_rmsnorm_4096_device_into(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    input: &DeviceBuffer<u16>,
    norm: &DeviceBuffer<u16>,
    output: &mut DeviceBuffer<u16>,
    trace: &mut DeviceBuffer<f32>,
) -> Result<Vec<f32>, Box<dyn Error>> {
    pub(crate) const WIDTH: usize = 4_096;
    if input.is_empty()
        || !input.len().is_multiple_of(WIDTH)
        || norm.len() != WIDTH
        || output.len() != input.len()
        || !(trace.is_empty() || trace.len() == input.len() / WIDTH * 3)
    {
        return Err("Flash learned RMSNorm 4096 workspace contract mismatch".into());
    }
    let rows = input.len() / WIDTH;
    let mut squares = DeviceBuffer::<f32>::zeroed(stream, input.len())?;
    // SAFETY: the fixed Flash contract proves contiguous [rows, 4096] BF16
    // input, one square per element, and one 512-thread reduction block per row.
    unsafe {
        module.rmsnorm_flash_square_bf16(
            stream,
            LaunchConfig::for_num_elems(u32::try_from(input.len())?),
            input,
            &mut squares,
        )?;
    }
    let traced_squares = if trace.is_empty() {
        Vec::new()
    } else {
        squares.to_host_vec(stream)?
    };
    unsafe {
        module.rmsnorm_flash_learned_4096_bf16(
            stream,
            LaunchConfig {
                grid_dim: (u32::try_from(rows)?, 1, 1),
                block_dim: (512, 1, 1),
                shared_mem_bytes: 0,
            },
            input,
            norm,
            &squares,
            output,
            trace,
        )?;
    }
    Ok(traced_squares)
}

pub(crate) fn run_l0_rmsnorm(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    input: &[u16],
    norm: &DeviceBuffer<u16>,
    width: usize,
) -> Result<Vec<u16>, Box<dyn Error>> {
    if input.is_empty() || input.len() % width != 0 || (!norm.is_empty() && norm.len() != width) {
        return Err("L0 RMSNorm shape mismatch".into());
    }
    let input = DeviceBuffer::from_host(stream, input)?;
    let mut output = DeviceBuffer::<u16>::zeroed(stream, input.len())?;
    run_l0_rmsnorm_device_into(module, stream, &input, norm, &mut output, width)?;
    Ok(output.to_host_vec(stream)?)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_l0_fp8_linear_device_into(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    input: &DeviceBuffer<u16>,
    matrix: &DeviceFp8Matrix,
    quantized: &mut DeviceBuffer<u8>,
    scales: &mut DeviceBuffer<u8>,
    dequantized: &mut DeviceBuffer<u16>,
    output_f32: &mut DeviceBuffer<f32>,
    output_bf16: &mut DeviceBuffer<u16>,
    strict_k_order: bool,
) -> Result<(), Box<dyn Error>> {
    if input.is_empty()
        || !input.len().is_multiple_of(matrix.in_features)
        || !matrix.in_features.is_multiple_of(v4_state::FP8_BLOCK)
    {
        return Err("device FP8 linear input contract mismatch".into());
    }
    let rows = input.len() / matrix.in_features;
    if quantized.len() != input.len()
        || scales.len() != rows * matrix.in_features / v4_state::FP8_BLOCK
        || !dequantized.is_empty()
        || output_f32.len() != rows * matrix.out_features
        || output_bf16.len() != output_f32.len()
    {
        return Err("device FP8 linear workspace contract mismatch".into());
    }
    // SAFETY: all activation groups and matrix/output geometries were checked.
    unsafe {
        module.quantize_v4_fp8_bf16(
            stream,
            LaunchConfig::for_num_elems(u32::try_from(scales.len())?),
            input,
            quantized,
            scales,
            dequantized,
            u32::try_from(matrix.in_features)?,
            u32::try_from(matrix.in_features)?,
            u32::try_from(v4_state::FP8_BLOCK)?,
        )?;
        launch_v4_fp8_matvec(
            module,
            stream,
            quantized,
            scales,
            &matrix.weight,
            &matrix.scale,
            output_f32,
            output_bf16,
            matrix.in_features,
            matrix.out_features,
            strict_k_order,
        )?;
    }
    Ok(())
}

pub(crate) fn run_l0_fp8_linear(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    input: &[u16],
    matrix: &DeviceFp8Matrix,
    strict_k_order: bool,
) -> Result<(Vec<f32>, Vec<u16>), Box<dyn Error>> {
    if input.is_empty()
        || input.len() % matrix.in_features != 0
        || matrix.in_features % v4_state::FP8_BLOCK != 0
    {
        return Err("L0 FP8 linear input shape mismatch".into());
    }
    let rows = input.len() / matrix.in_features;
    let input = DeviceBuffer::from_host(stream, input)?;
    let mut quantized = DeviceBuffer::<u8>::zeroed(stream, input.len())?;
    let mut scales =
        DeviceBuffer::<u8>::zeroed(stream, rows * matrix.in_features / v4_state::FP8_BLOCK)?;
    let mut dequantized = DeviceBuffer::<u16>::zeroed(stream, 0)?;
    let mut output_f32 = DeviceBuffer::<f32>::zeroed(stream, rows * matrix.out_features)?;
    let mut output_bf16 = DeviceBuffer::<u16>::zeroed(stream, rows * matrix.out_features)?;
    run_l0_fp8_linear_device_into(
        module,
        stream,
        &input,
        matrix,
        &mut quantized,
        &mut scales,
        &mut dequantized,
        &mut output_f32,
        &mut output_bf16,
        strict_k_order,
    )?;
    Ok((
        output_f32.to_host_vec(stream)?,
        output_bf16.to_host_vec(stream)?,
    ))
}

pub(crate) fn maybe_write_v4_audit_u16(
    layer: usize,
    stage: &str,
    values: &[u16],
) -> Result<(), Box<dyn Error>> {
    let Some(requested_layer) = std::env::var_os("COLIBRI_V4_AUDIT_LAYER") else {
        return Ok(());
    };
    if requested_layer.to_string_lossy().parse::<usize>()? != layer
        || std::env::var("COLIBRI_V4_AUDIT_STAGE").as_deref() != Ok(stage)
    {
        return Ok(());
    }
    let path = std::env::var_os("COLIBRI_V4_AUDIT_PATH")
        .map(std::path::PathBuf::from)
        .ok_or("set COLIBRI_V4_AUDIT_PATH when requesting a V4 audit stage")?;
    let mut output = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)?;
    for &bits in values {
        output.write_all(&bits.to_le_bytes())?;
    }
    output.sync_all()?;
    println!(
        "{{\"event\":\"v4_audit\",\"layer\":{layer},\"stage\":{},\"elements\":{},\"digest\":\"{:016x}\",\"path\":{}}}",
        serde_json::to_string(stage)?,
        values.len(),
        fnv1a_u16(values),
        serde_json::to_string(&path)?
    );
    Ok(())
}

pub(crate) fn maybe_write_v4_audit_f32(
    layer: usize,
    stage: &str,
    values: &[f32],
) -> Result<(), Box<dyn Error>> {
    let Some(requested_layer) = std::env::var_os("COLIBRI_V4_AUDIT_LAYER") else {
        return Ok(());
    };
    if requested_layer.to_string_lossy().parse::<usize>()? != layer
        || std::env::var("COLIBRI_V4_AUDIT_STAGE").as_deref() != Ok(stage)
    {
        return Ok(());
    }
    let path = std::env::var_os("COLIBRI_V4_AUDIT_PATH")
        .map(std::path::PathBuf::from)
        .ok_or("set COLIBRI_V4_AUDIT_PATH when requesting a V4 audit stage")?;
    let mut output = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)?;
    for &value in values {
        output.write_all(&value.to_bits().to_le_bytes())?;
    }
    output.sync_all()?;
    Ok(())
}

#[derive(Clone, Copy)]
pub(crate) enum FlashParityDtype {
    Bf16,
    F32,
    I32,
    U32,
    U8,
}

impl FlashParityDtype {
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Bf16 => "BF16",
            Self::F32 => "F32",
            Self::I32 => "I32",
            Self::U32 => "U32",
            Self::U8 => "U8",
        }
    }

    pub(crate) const fn bytes(self) -> usize {
        match self {
            Self::Bf16 => std::mem::size_of::<u16>(),
            Self::F32 | Self::I32 | Self::U32 => std::mem::size_of::<u32>(),
            Self::U8 => std::mem::size_of::<u8>(),
        }
    }
}

pub(crate) fn flash_parity_dump_dir(
    spec: &v4_spec::V4Spec,
    layer: usize,
    start_pos: usize,
    rows: usize,
) -> Result<Option<std::path::PathBuf>, Box<dyn Error>> {
    if spec != &v4_spec::FLASH_0731 || start_pos != 0 || rows != 1 {
        return Ok(None);
    }
    let target_layer = std::env::var("COLIBRI_V4_FLASH_PARITY_STOP_AFTER_LAYER")
        .ok()
        .map(|value| {
            value
                .parse::<usize>()
                .map_err(|_| "COLIBRI_V4_FLASH_PARITY_STOP_AFTER_LAYER must be a layer index")
        })
        .transpose()?;
    if target_layer != Some(layer) {
        return Ok(None);
    }
    let Some(path) = std::env::var_os("COLIBRI_V4_FLASH_PARITY_DUMP_DIR") else {
        return Ok(None);
    };
    if path.is_empty() {
        return Err("COLIBRI_V4_FLASH_PARITY_DUMP_DIR must not be empty".into());
    }
    let path = std::path::PathBuf::from(path);
    std::fs::create_dir_all(&path)?;
    Ok(Some(path))
}

/// Narrow router-only dump for a later parity-stop layer.  The broader L0
/// dump remains intentionally L0-only because its attention/expert tensors
/// are not evidence for a later router mismatch.
pub(crate) fn flash_router_trace_dump_dir(
    spec: &v4_spec::V4Spec,
    layer: usize,
    start_pos: usize,
    rows: usize,
) -> Result<Option<std::path::PathBuf>, Box<dyn Error>> {
    if spec != &v4_spec::FLASH_0731 || start_pos != 0 || rows != 1 {
        return Ok(None);
    }
    let Some(path) = std::env::var_os("COLIBRI_V4_FLASH_PARITY_DUMP_DIR") else {
        return Ok(None);
    };
    let target_layer = std::env::var("COLIBRI_V4_FLASH_PARITY_STOP_AFTER_LAYER")
        .ok()
        .map(|value| {
            value
                .parse::<usize>()
                .map_err(|_| "COLIBRI_V4_FLASH_PARITY_STOP_AFTER_LAYER must be a layer index")
        })
        .transpose()?;
    if target_layer != Some(layer) {
        return Ok(None);
    }
    if path.is_empty() {
        return Err("COLIBRI_V4_FLASH_PARITY_DUMP_DIR must not be empty".into());
    }
    let path = std::path::PathBuf::from(path);
    std::fs::create_dir_all(&path)?;
    Ok(Some(path))
}

pub(crate) fn flash_parity_manifest(
    name: &str,
    dtype: FlashParityDtype,
    shape: &[usize],
    byte_len: usize,
    sha256: &str,
) -> Result<serde_json::Value, Box<dyn Error>> {
    let elements = shape.iter().try_fold(1_usize, |product, &dimension| {
        product
            .checked_mul(dimension)
            .ok_or("Flash parity dump shape overflows")
    })?;
    if elements
        .checked_mul(dtype.bytes())
        .ok_or("Flash parity dump byte length overflows")?
        != byte_len
    {
        return Err(format!(
            "Flash parity dump {name} shape/dtype does not match {byte_len} bytes"
        )
        .into());
    }
    Ok(serde_json::json!({
        "schema": "colibri-flash-parity-array-v1",
        "name": name,
        "layout": "row_major",
        "endianness": "little",
        "dtype": dtype.name(),
        "shape": shape,
        "elements": elements,
        "byte_len": byte_len,
        "sha256": sha256,
        "binary": format!("{name}.bin"),
    }))
}

pub(crate) fn flash_parity_dump_bytes(
    dir: &std::path::Path,
    name: &str,
    dtype: FlashParityDtype,
    shape: &[usize],
    bytes: &[u8],
) -> Result<(), Box<dyn Error>> {
    let binary = dir.join(format!("{name}.bin"));
    let mut output = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&binary)?;
    output.write_all(bytes)?;
    output.sync_all()?;
    let manifest = flash_parity_manifest(name, dtype, shape, bytes.len(), &sha256_file(&binary)?)?;
    let manifest_path = dir.join(format!("{name}.manifest.json"));
    let mut manifest_output = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(manifest_path)?;
    serde_json::to_writer_pretty(&mut manifest_output, &manifest)?;
    manifest_output.write_all(b"\n")?;
    manifest_output.sync_all()?;
    Ok(())
}

pub(crate) fn flash_parity_dump_u16(
    dir: &std::path::Path,
    name: &str,
    shape: &[usize],
    values: &[u16],
) -> Result<(), Box<dyn Error>> {
    let bytes = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    flash_parity_dump_bytes(dir, name, FlashParityDtype::Bf16, shape, &bytes)
}

pub(crate) fn flash_parity_dump_f32(
    dir: &std::path::Path,
    name: &str,
    shape: &[usize],
    values: &[f32],
) -> Result<(), Box<dyn Error>> {
    let bytes = values
        .iter()
        .flat_map(|value| value.to_bits().to_le_bytes())
        .collect::<Vec<_>>();
    flash_parity_dump_bytes(dir, name, FlashParityDtype::F32, shape, &bytes)
}

pub(crate) fn flash_parity_dump_i32(
    dir: &std::path::Path,
    name: &str,
    shape: &[usize],
    values: &[i32],
) -> Result<(), Box<dyn Error>> {
    let bytes = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    flash_parity_dump_bytes(dir, name, FlashParityDtype::I32, shape, &bytes)
}

pub(crate) fn flash_parity_dump_u32(
    dir: &std::path::Path,
    name: &str,
    shape: &[usize],
    values: &[u32],
) -> Result<(), Box<dyn Error>> {
    let bytes = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    flash_parity_dump_bytes(dir, name, FlashParityDtype::U32, shape, &bytes)
}

pub(crate) fn flash_parity_dump_u8(
    dir: &std::path::Path,
    name: &str,
    shape: &[usize],
    values: &[u8],
) -> Result<(), Box<dyn Error>> {
    flash_parity_dump_bytes(dir, name, FlashParityDtype::U8, shape, values)
}

pub(crate) fn run_l0_rope(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    input: &[u16],
    freqs: &[f32],
    heads: usize,
    inverse: bool,
) -> Result<Vec<u16>, Box<dyn Error>> {
    run_rope(
        module,
        stream,
        input,
        freqs,
        heads,
        v4_state::Q_HEAD_DIM,
        inverse,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_rope(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    input: &[u16],
    freqs: &[f32],
    heads: usize,
    head_dim: usize,
    inverse: bool,
) -> Result<Vec<u16>, Box<dyn Error>> {
    let row_width = heads * head_dim;
    if input.is_empty()
        || input.len() % row_width != 0
        || head_dim < v4_state::Q_ROPE_DIM
        || freqs.len() != input.len() / row_width * v4_state::Q_ROPE_DIM
    {
        return Err("L0 RoPE shape mismatch".into());
    }
    let mut freqs = freqs.to_vec();
    if inverse {
        for sine in freqs.iter_mut().skip(1).step_by(2) {
            *sine = -*sine;
        }
    }
    let input = DeviceBuffer::from_host(stream, input)?;
    let freqs = DeviceBuffer::from_host(stream, &freqs)?;
    let mut output = DeviceBuffer::<u16>::zeroed(stream, input.len())?;
    run_rope_device_into(module, stream, &input, &freqs, &mut output, heads, head_dim)?;
    Ok(output.to_host_vec(stream)?)
}

pub(crate) fn run_rope_device_into(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    input: &DeviceBuffer<u16>,
    freqs: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<u16>,
    heads: usize,
    head_dim: usize,
) -> Result<(), Box<dyn Error>> {
    let row_width = heads * head_dim;
    if input.is_empty()
        || !input.len().is_multiple_of(row_width)
        || head_dim < v4_state::Q_ROPE_DIM
        || freqs.len() != input.len() / row_width * v4_state::Q_ROPE_DIM
        || output.len() != input.len()
    {
        return Err("device RoPE workspace contract mismatch".into());
    }
    // SAFETY: kernel checks the same fixed head and frequency geometry.
    unsafe {
        module.rope_v4_bf16(
            stream,
            LaunchConfig::for_num_elems(u32::try_from(output.len())?),
            input,
            freqs,
            output,
            u32::try_from(heads)?,
            u32::try_from(head_dim)?,
        )?;
    }
    Ok(())
}

pub(crate) fn run_l0_kv_quantize_device_into(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    input: &DeviceBuffer<u16>,
    empty: &mut DeviceBuffer<u8>,
    scales: &mut DeviceBuffer<u8>,
    output: &mut DeviceBuffer<u16>,
) -> Result<(), Box<dyn Error>> {
    if input.is_empty()
        || !input.len().is_multiple_of(v4_state::Q_HEAD_DIM)
        || !empty.is_empty()
        || scales.len() != input.len() / v4_state::Q_HEAD_DIM * 7
        || output.len() != input.len()
    {
        return Err("device KV quantization workspace contract mismatch".into());
    }
    // SAFETY: the full-row copy preserves the 64-value RoPE tail; the kernel
    // then overwrites exactly seven complete 64-value non-RoPE groups.
    unsafe {
        cuda_core::memory::memcpy_dtod_async(
            output.cu_deviceptr(),
            input.cu_deviceptr(),
            input.num_bytes(),
            stream.cu_stream(),
        )?;
        module.quantize_v4_fp8_bf16(
            stream,
            LaunchConfig::for_num_elems(u32::try_from(scales.len())?),
            input,
            empty,
            scales,
            output,
            u32::try_from(v4_state::Q_HEAD_DIM)?,
            u32::try_from(v4_state::Q_HEAD_DIM - v4_state::Q_ROPE_DIM)?,
            64,
        )?;
    }
    Ok(())
}

pub(crate) fn run_l0_kv_quantize(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    input: &[u16],
) -> Result<Vec<u16>, Box<dyn Error>> {
    if input.is_empty() || input.len() % v4_state::Q_HEAD_DIM != 0 {
        return Err("L0 KV quantization shape mismatch".into());
    }
    let rows = input.len() / v4_state::Q_HEAD_DIM;
    let mut output = DeviceBuffer::from_host(stream, input)?;
    let input = DeviceBuffer::from_host(stream, input)?;
    let mut empty = DeviceBuffer::<u8>::zeroed(stream, 0)?;
    let mut scales = DeviceBuffer::<u8>::zeroed(stream, rows * 7)?;
    run_l0_kv_quantize_device_into(module, stream, &input, &mut empty, &mut scales, &mut output)?;
    Ok(output.to_host_vec(stream)?)
}

pub(crate) fn run_layer_compressor(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    input: &[u16],
    weights: &LayerDeviceWeights,
    start_pos: usize,
    cublas: Option<&cublas_exact::CublasExact>,
    strict_scratch: Option<&mut StrictCompressorScratch>,
) -> Result<v4_state::CompressorPrepared, Box<dyn Error>> {
    let ratio = weights.compression_ratio.get();
    if weights.compressor_ape.len() % ratio != 0 {
        return Err("layer Compressor APE shape mismatch".into());
    }
    let outputs = weights.compressor_ape.len() / ratio;
    if weights.compressor_wkv.len() != outputs * weights.spec.hidden
        || weights.compressor_wgate.len() != outputs * weights.spec.hidden
    {
        return Err("layer Compressor weight shape mismatch".into());
    }
    let probe = flash_compressor_probe_enabled();
    if probe {
        eprintln!(
            "COMPPROBE_BRANCH outputs={} ratio={} strict={} scratch={} input_len={} \
             arena={} pool_reuse={}",
            outputs,
            ratio,
            strict_compressor_enabled(),
            strict_scratch.is_some(),
            input.len(),
            strict_scratch
                .as_ref()
                .map(|scratch| scratch.hidden())
                .unwrap_or(0),
            strict_scratch
                .as_ref()
                .is_some_and(|scratch| input.len() == scratch.hidden()),
        );
    }
    if strict_compressor_enabled() {
        // The pool is sized for exactly one position (the width it was built
        // with); the chunked prefill pass hands the compressor several positions
        // at once, so that pass keeps the per-row implementation below.
        let scratch = strict_scratch.filter(|scratch| input.len() == scratch.hidden());
        if let Some(scratch) = scratch {
            return scratch.run_from_device(
                stream,
                input,
                &weights.compressor_wkv,
                &weights.compressor_wgate,
                &weights.compressor_ape,
                outputs,
                ratio,
                start_pos,
            );
        }
        // The pinned official oracle dispatches this projection through cuBLAS
        // SGEMM.  Its reduction tree is observable in the committed FP32 state,
        // so the strict trace path must use the same pinned library.
        let owned;
        let cublas = if let Some(cublas) = cublas {
            cublas
        } else {
            owned = cublas_exact::CublasExact::load()?;
            &owned
        };
        return run_compressor_official_cublas_from_device(
            &cublas,
            stream,
            input,
            &weights.compressor_wkv,
            &weights.compressor_wgate,
            &weights.compressor_ape,
            outputs,
            ratio,
            start_pos,
        );
    }
    run_compressor(
        module,
        stream,
        input,
        &weights.compressor_wkv,
        &weights.compressor_wgate,
        &weights.compressor_ape,
        outputs,
        ratio,
        start_pos,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_compressor(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    input: &[u16],
    wkv: &DeviceBuffer<u16>,
    wgate: &DeviceBuffer<u16>,
    ape: &DeviceBuffer<f32>,
    outputs: usize,
    ratio: usize,
    start_pos: usize,
) -> Result<v4_state::CompressorPrepared, Box<dyn Error>> {
    let hidden = wkv.len().checked_div(outputs).unwrap_or(0);
    if input.is_empty()
        || hidden == 0
        || input.len() % hidden != 0
        || outputs == 0
        || wkv.len() != outputs * hidden
        || wgate.len() != outputs * hidden
        || ape.len() != ratio * outputs
    {
        return Err("layer Compressor input/weight geometry mismatch".into());
    }
    let rows = input.len() / hidden;
    let input = DeviceBuffer::from_host(stream, input)?;
    let mut kv = DeviceBuffer::<f32>::zeroed(stream, rows * outputs)?;
    let mut score = DeviceBuffer::<f32>::zeroed(stream, rows * outputs)?;
    let mut score_plus_ape = DeviceBuffer::<f32>::zeroed(stream, rows * outputs)?;
    run_compressor_device_into(
        module,
        stream,
        &input,
        wkv,
        wgate,
        ape,
        &mut kv,
        &mut score,
        &mut score_plus_ape,
        outputs,
        ratio,
        start_pos,
    )?;
    Ok(v4_state::CompressorPrepared {
        kv_f32: kv.to_host_vec(stream)?,
        score_f32: score.to_host_vec(stream)?,
        score_plus_ape_f32: score_plus_ape.to_host_vec(stream)?,
    })
}

pub(crate) fn run_l0_compressor_official_cublas(
    cublas: &cublas_exact::CublasExact,
    stream: &CudaStream,
    input_bf16: &[u16],
    weights: &v4_state::CompressorWeights,
) -> Result<v4_state::CompressorPrepared, Box<dyn Error>> {
    run_compressor_official_cublas_host(
        cublas,
        stream,
        input_bf16,
        &weights.wkv_bf16,
        &weights.wgate_bf16,
        &weights.ape_f32,
        v4_state::Q_HEAD_DIM,
        v4_state::COMPRESS_RATIO,
        0,
    )
}

pub(crate) struct StrictCompressorArena {
    pub(crate) hidden: usize,
    pub(crate) outputs: usize,
    pub(crate) ratio: usize,
    pub(crate) wkv_bf16: Vec<u16>,
    pub(crate) wgate_bf16: Vec<u16>,
    pub(crate) ape_f32: Vec<f32>,
    pub(crate) input_f32: Vec<f32>,
    pub(crate) wkv_f32: Vec<f32>,
    pub(crate) wgate_f32: Vec<f32>,
    pub(crate) kv_f32: Vec<f32>,
    pub(crate) score_f32: Vec<f32>,
    pub(crate) score_plus_ape_f32: Vec<f32>,
    pub(crate) input_device: DeviceBuffer<f32>,
    pub(crate) wkv_device: DeviceBuffer<f32>,
    pub(crate) wgate_device: DeviceBuffer<f32>,
    pub(crate) workspace: DeviceBuffer<u8>,
    pub(crate) kv_device: DeviceBuffer<f32>,
    pub(crate) score_device: DeviceBuffer<f32>,
}

impl StrictCompressorArena {
    pub(crate) fn new(
        stream: &CudaStream,
        hidden: usize,
        outputs: usize,
        ratio: usize,
    ) -> Result<Self, Box<dyn Error>> {
        let weights = outputs
            .checked_mul(hidden)
            .ok_or("strict Compressor arena weight size overflow")?;
        let ape = outputs
            .checked_mul(ratio)
            .ok_or("strict Compressor arena APE size overflow")?;
        Ok(Self {
            hidden,
            outputs,
            ratio,
            wkv_bf16: vec![0; weights],
            wgate_bf16: vec![0; weights],
            ape_f32: vec![0.0; ape],
            input_f32: vec![0.0; hidden],
            wkv_f32: vec![0.0; weights],
            wgate_f32: vec![0.0; weights],
            kv_f32: vec![0.0; outputs],
            score_f32: vec![0.0; outputs],
            score_plus_ape_f32: vec![0.0; outputs],
            input_device: DeviceBuffer::zeroed(stream, hidden)?,
            wkv_device: DeviceBuffer::zeroed(stream, weights)?,
            wgate_device: DeviceBuffer::zeroed(stream, weights)?,
            workspace: DeviceBuffer::zeroed(stream, cublas_exact::WORKSPACE_BYTES)?,
            kv_device: DeviceBuffer::zeroed(stream, outputs)?,
            score_device: DeviceBuffer::zeroed(stream, outputs)?,
        })
    }

    pub(crate) fn host_bytes_for(
        hidden: usize,
        outputs: usize,
        ratio: usize,
    ) -> Result<usize, Box<dyn Error>> {
        let weights = outputs
            .checked_mul(hidden)
            .ok_or("strict Compressor host arena weight size overflow")?;
        weights
            .checked_mul(2 * std::mem::size_of::<u16>() + 2 * std::mem::size_of::<f32>())
            .and_then(|bytes| bytes.checked_add(hidden * std::mem::size_of::<f32>()))
            .and_then(|bytes| bytes.checked_add(outputs * ratio * std::mem::size_of::<f32>()))
            .and_then(|bytes| bytes.checked_add(outputs * 3 * std::mem::size_of::<f32>()))
            .ok_or_else(|| "strict Compressor host arena size overflow".into())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_from_device(
        &mut self,
        cublas: &cublas_exact::CublasExact,
        stream: &CudaStream,
        input_bf16: &[u16],
        wkv: &DeviceBuffer<u16>,
        wgate: &DeviceBuffer<u16>,
        ape: &DeviceBuffer<f32>,
        outputs: usize,
        ratio: usize,
        start_pos: usize,
    ) -> Result<v4_state::CompressorPrepared, Box<dyn Error>> {
        if input_bf16.len() != self.hidden
            || outputs != self.outputs
            || ratio != self.ratio
            || wkv.len() != self.wkv_bf16.len()
            || wgate.len() != self.wgate_bf16.len()
            || ape.len() != self.ape_f32.len()
        {
            return Err("strict Compressor reusable arena geometry mismatch".into());
        }
        // Segment timing for the compressor tier investigation. `Instant` is not
        // a sync, and nine pairs of marks are ~0.6 us against a ~10 ms call, so
        // this stays unconditional and only the print is gated: a probed and an
        // unprobed run then share one code path.
        let mut probe_laps = [0_u128; 9];
        let probe_start = std::time::Instant::now();
        let mut probe_mark = probe_start;
        wkv.copy_to_host(stream, &mut self.wkv_bf16)?;
        wgate.copy_to_host(stream, &mut self.wgate_bf16)?;
        ape.copy_to_host(stream, &mut self.ape_f32)?;
        probe_laps[0] = probe_mark.elapsed().as_micros();
        probe_mark = std::time::Instant::now();
        for (dst, &src) in self.input_f32.iter_mut().zip(input_bf16) {
            *dst = f32::from_bits(u32::from(src) << 16);
        }
        probe_laps[1] = probe_mark.elapsed().as_micros();
        probe_mark = std::time::Instant::now();
        for (dst, &src) in self.wkv_f32.iter_mut().zip(&self.wkv_bf16) {
            *dst = f32::from_bits(u32::from(src) << 16);
        }
        probe_laps[2] = probe_mark.elapsed().as_micros();
        probe_mark = std::time::Instant::now();
        for (dst, &src) in self.wgate_f32.iter_mut().zip(&self.wgate_bf16) {
            *dst = f32::from_bits(u32::from(src) << 16);
        }
        probe_laps[3] = probe_mark.elapsed().as_micros();
        probe_mark = std::time::Instant::now();
        self.input_device.copy_from_host(stream, &self.input_f32)?;
        self.wkv_device.copy_from_host(stream, &self.wkv_f32)?;
        self.wgate_device.copy_from_host(stream, &self.wgate_f32)?;
        probe_laps[4] = probe_mark.elapsed().as_micros();
        probe_mark = std::time::Instant::now();
        let device_const = |pointer| pointer as usize as *const std::ffi::c_void;
        let device_mut = |pointer| pointer as usize as *mut std::ffi::c_void;
        let stream_pointer = stream.cu_stream().cast::<std::ffi::c_void>();
        cublas.sgemm_mn1(
            stream_pointer,
            device_mut(self.workspace.cu_deviceptr()),
            self.workspace.num_bytes(),
            device_const(self.wkv_device.cu_deviceptr()),
            device_const(self.input_device.cu_deviceptr()),
            device_mut(self.kv_device.cu_deviceptr()),
            outputs,
            self.hidden,
        )?;
        probe_laps[5] = probe_mark.elapsed().as_micros();
        probe_mark = std::time::Instant::now();
        cublas.sgemm_mn1(
            stream_pointer,
            device_mut(self.workspace.cu_deviceptr()),
            self.workspace.num_bytes(),
            device_const(self.wgate_device.cu_deviceptr()),
            device_const(self.input_device.cu_deviceptr()),
            device_mut(self.score_device.cu_deviceptr()),
            outputs,
            self.hidden,
        )?;
        probe_laps[6] = probe_mark.elapsed().as_micros();
        probe_mark = std::time::Instant::now();
        self.kv_device.copy_to_host(stream, &mut self.kv_f32)?;
        self.score_device
            .copy_to_host(stream, &mut self.score_f32)?;
        probe_laps[7] = probe_mark.elapsed().as_micros();
        probe_mark = std::time::Instant::now();
        let ape_offset = start_pos
            .checked_rem(ratio)
            .and_then(|row| row.checked_mul(outputs))
            .ok_or("strict Compressor APE offset overflow")?;
        for ((dst, &score), &ape) in self
            .score_plus_ape_f32
            .iter_mut()
            .zip(&self.score_f32)
            .zip(&self.ape_f32[ape_offset..][..outputs])
        {
            *dst = score + ape;
        }
        if self
            .kv_f32
            .iter()
            .chain(&self.score_f32)
            .chain(&self.score_plus_ape_f32)
            .any(|value| !value.is_finite())
        {
            return Err("strict cuBLAS Compressor produced non-finite output".into());
        }
        probe_laps[8] = probe_mark.elapsed().as_micros();
        if flash_compressor_probe_enabled() {
            eprintln!(
                "COMPPROBE outputs={} ratio={} d2h_w={} expand_in={} expand_wkv={} \
                 expand_wgate={} h2d_w={} gemm1={} gemm2={} d2h_out={} tail={} total={}",
                outputs,
                ratio,
                probe_laps[0],
                probe_laps[1],
                probe_laps[2],
                probe_laps[3],
                probe_laps[4],
                probe_laps[5],
                probe_laps[6],
                probe_laps[7],
                probe_laps[8],
                probe_start.elapsed().as_micros(),
            );
        }
        Ok(v4_state::CompressorPrepared {
            kv_f32: self.kv_f32.clone(),
            score_f32: self.score_f32.clone(),
            score_plus_ape_f32: self.score_plus_ape_f32.clone(),
        })
    }
}

pub(crate) struct StrictCompressorScratch {
    pub(crate) hca_cublas: cublas_exact::CublasExact,
    pub(crate) csa_cublas: cublas_exact::CublasExact,
    pub(crate) hca: StrictCompressorArena,
    pub(crate) csa: StrictCompressorArena,
    pub(crate) csa_indexer: StrictCompressorArena,
}

impl StrictCompressorScratch {
    pub(crate) fn new(stream: &CudaStream, hidden: usize) -> Result<Self, Box<dyn Error>> {
        Ok(Self {
            hca_cublas: cublas_exact::CublasExact::load()?,
            csa_cublas: cublas_exact::CublasExact::load()?,
            hca: StrictCompressorArena::new(stream, hidden, 512, 128)?,
            csa: StrictCompressorArena::new(stream, hidden, 1_024, 4)?,
            csa_indexer: StrictCompressorArena::new(stream, hidden, 256, 4)?,
        })
    }

    pub(crate) fn host_bytes(hidden: usize) -> Result<usize, Box<dyn Error>> {
        [
            StrictCompressorArena::host_bytes_for(hidden, 512, 128)?,
            StrictCompressorArena::host_bytes_for(hidden, 1_024, 4)?,
            StrictCompressorArena::host_bytes_for(hidden, 256, 4)?,
        ]
        .into_iter()
        .try_fold(0_usize, |sum, bytes| sum.checked_add(bytes))
        .ok_or_else(|| "strict Compressor scratch host size overflow".into())
    }

    /// The width every one of the three pools is sized for. Callers compare
    /// against this instead of a model constant, so the pool is used exactly
    /// when the compressor hands it a geometry it was built for.
    pub(crate) fn hidden(&self) -> usize {
        self.hca.hidden
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_from_device(
        &mut self,
        stream: &CudaStream,
        input_bf16: &[u16],
        wkv: &DeviceBuffer<u16>,
        wgate: &DeviceBuffer<u16>,
        ape: &DeviceBuffer<f32>,
        outputs: usize,
        ratio: usize,
        start_pos: usize,
    ) -> Result<v4_state::CompressorPrepared, Box<dyn Error>> {
        match (outputs, ratio) {
            (512, 128) => self.hca.run_from_device(
                &self.hca_cublas,
                stream,
                input_bf16,
                wkv,
                wgate,
                ape,
                outputs,
                ratio,
                start_pos,
            ),
            (1_024, 4) => self.csa.run_from_device(
                &self.csa_cublas,
                stream,
                input_bf16,
                wkv,
                wgate,
                ape,
                outputs,
                ratio,
                start_pos,
            ),
            (256, 4) => self.csa_indexer.run_from_device(
                &self.csa_cublas,
                stream,
                input_bf16,
                wkv,
                wgate,
                ape,
                outputs,
                ratio,
                start_pos,
            ),
            _ => Err("strict Compressor scratch received unsupported layout".into()),
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_compressor_official_cublas_from_device(
    cublas: &cublas_exact::CublasExact,
    stream: &CudaStream,
    input: &[u16],
    wkv: &DeviceBuffer<u16>,
    wgate: &DeviceBuffer<u16>,
    ape: &DeviceBuffer<f32>,
    outputs: usize,
    ratio: usize,
    start_pos: usize,
) -> Result<v4_state::CompressorPrepared, Box<dyn Error>> {
    run_compressor_official_cublas_host(
        cublas,
        stream,
        input,
        &wkv.to_host_vec(stream)?,
        &wgate.to_host_vec(stream)?,
        &ape.to_host_vec(stream)?,
        outputs,
        ratio,
        start_pos,
    )
}

pub(crate) fn official_cublas_compressor_geometry(
    input_len: usize,
    wkv_len: usize,
    wgate_len: usize,
    ape_len: usize,
    outputs: usize,
    ratio: usize,
) -> Option<(usize, usize)> {
    let hidden = wkv_len.checked_div(outputs)?;
    let rows = input_len.checked_div(hidden)?;
    (hidden != 0
        && rows != 0
        && input_len == rows.checked_mul(hidden)?
        && wkv_len == outputs.checked_mul(hidden)?
        && wgate_len == outputs.checked_mul(hidden)?
        && ape_len == ratio.checked_mul(outputs)?
        && matches!(ratio, 4 | 128))
    .then_some((hidden, rows))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_compressor_official_cublas_host(
    cublas: &cublas_exact::CublasExact,
    stream: &CudaStream,
    input_bf16: &[u16],
    wkv_bf16: &[u16],
    wgate_bf16: &[u16],
    ape_f32: &[f32],
    outputs: usize,
    ratio: usize,
    start_pos: usize,
) -> Result<v4_state::CompressorPrepared, Box<dyn Error>> {
    let (hidden, rows) = official_cublas_compressor_geometry(
        input_bf16.len(),
        wkv_bf16.len(),
        wgate_bf16.len(),
        ape_f32.len(),
        outputs,
        ratio,
    )
    .ok_or("official cuBLAS Compressor geometry mismatch")?;
    let to_f32 = |values: &[u16]| {
        values
            .iter()
            .map(|&bits| f32::from_bits(u32::from(bits) << 16))
            .collect::<Vec<_>>()
    };
    let probe = flash_compressor_probe_enabled();
    let mut probe_laps = [0_u128; 10];
    let probe_start = std::time::Instant::now();
    let mut probe_mark = probe_start;
    let wkv_f32_host = to_f32(wkv_bf16);
    probe_laps[0] = probe_mark.elapsed().as_micros();
    probe_mark = std::time::Instant::now();
    let wgate_f32_host = to_f32(wgate_bf16);
    probe_laps[1] = probe_mark.elapsed().as_micros();
    probe_mark = std::time::Instant::now();
    let wkv = DeviceBuffer::from_host(stream, &wkv_f32_host)?;
    probe_laps[2] = probe_mark.elapsed().as_micros();
    probe_mark = std::time::Instant::now();
    let wgate = DeviceBuffer::from_host(stream, &wgate_f32_host)?;
    probe_laps[3] = probe_mark.elapsed().as_micros();
    probe_mark = std::time::Instant::now();
    let workspace = DeviceBuffer::<u8>::zeroed(stream, cublas_exact::WORKSPACE_BYTES)?;
    probe_laps[6] = probe_mark.elapsed().as_micros();
    probe_mark = std::time::Instant::now();
    let device_const = |pointer| pointer as usize as *const std::ffi::c_void;
    let device_mut = |pointer| pointer as usize as *mut std::ffi::c_void;
    let stream_pointer = stream.cu_stream().cast::<std::ffi::c_void>();
    let mut kv_f32 = Vec::with_capacity(rows * outputs);
    let mut score_f32 = Vec::with_capacity(rows * outputs);
    let mut probe_row_upload_ns = 0_u128;
    let mut probe_row_alloc_ns = 0_u128;
    let mut probe_gemm1_ns = 0_u128;
    let mut probe_gemm2_ns = 0_u128;
    let mut probe_row_d2h_ns = 0_u128;
    for input_bf16 in input_bf16.chunks_exact(hidden) {
        let row_mark = std::time::Instant::now();
        let input = DeviceBuffer::from_host(stream, &to_f32(input_bf16))?;
        let kv = DeviceBuffer::<f32>::zeroed(stream, outputs)?;
        let score = DeviceBuffer::<f32>::zeroed(stream, outputs)?;
        probe_row_upload_ns += row_mark.elapsed().as_nanos();
        let row_mark = std::time::Instant::now();
        cublas.sgemm_mn1(
            stream_pointer,
            device_mut(workspace.cu_deviceptr()),
            workspace.num_bytes(),
            device_const(wkv.cu_deviceptr()),
            device_const(input.cu_deviceptr()),
            device_mut(kv.cu_deviceptr()),
            outputs,
            hidden,
        )?;
        probe_gemm1_ns += row_mark.elapsed().as_nanos();
        let row_mark = std::time::Instant::now();
        cublas.sgemm_mn1(
            stream_pointer,
            device_mut(workspace.cu_deviceptr()),
            workspace.num_bytes(),
            device_const(wgate.cu_deviceptr()),
            device_const(input.cu_deviceptr()),
            device_mut(score.cu_deviceptr()),
            outputs,
            hidden,
        )?;
        probe_gemm2_ns += row_mark.elapsed().as_nanos();
        let row_mark = std::time::Instant::now();
        kv_f32.extend(kv.to_host_vec(stream)?);
        score_f32.extend(score.to_host_vec(stream)?);
        probe_row_d2h_ns += row_mark.elapsed().as_nanos();
    }
    probe_laps[4] = probe_mark.elapsed().as_micros();
    probe_mark = std::time::Instant::now();
    let score_plus_ape_f32 = score_f32
        .chunks_exact(outputs)
        .flat_map(|scores| {
            scores
                .iter()
                .zip(&ape_f32[start_pos % ratio * outputs..][..outputs])
                .map(|(&score, &ape)| score + ape)
        })
        .collect::<Vec<_>>();
    if kv_f32
        .iter()
        .chain(&score_f32)
        .chain(&score_plus_ape_f32)
        .any(|value| !value.is_finite())
    {
        return Err("official cuBLAS L0 Compressor produced non-finite output".into());
    }
    probe_laps[5] = probe_mark.elapsed().as_micros();
    if probe {
        eprintln!(
            "COMPPROBE_HOST outputs={} ratio={} rows={} expand_wkv={} expand_wgate={} \
             h2d_wkv={} h2d_wgate={} workspace_alloc={} row_loop={} tail={} row_upload={} \
             row_alloc={} gemm1={} gemm2={} row_d2h={} total={}",
            outputs,
            ratio,
            rows,
            probe_laps[0],
            probe_laps[1],
            probe_laps[2],
            probe_laps[3],
            probe_laps[6],
            probe_laps[4],
            probe_laps[5],
            probe_row_upload_ns / 1_000,
            probe_row_alloc_ns / 1_000,
            probe_gemm1_ns / 1_000,
            probe_gemm2_ns / 1_000,
            probe_row_d2h_ns / 1_000,
            probe_start.elapsed().as_micros(),
        );
    }
    Ok(v4_state::CompressorPrepared {
        kv_f32,
        score_f32,
        score_plus_ape_f32,
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_compressor_device_into(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    input: &DeviceBuffer<u16>,
    wkv: &DeviceBuffer<u16>,
    wgate: &DeviceBuffer<u16>,
    ape: &DeviceBuffer<f32>,
    kv: &mut DeviceBuffer<f32>,
    score: &mut DeviceBuffer<f32>,
    score_plus_ape: &mut DeviceBuffer<f32>,
    outputs: usize,
    ratio: usize,
    start_pos: usize,
) -> Result<(), Box<dyn Error>> {
    let hidden = wkv.len().checked_div(outputs).unwrap_or(0);
    if !compressor_workspace_contract_matches(
        input.len(),
        wkv.len(),
        wgate.len(),
        ape.len(),
        kv.len(),
        score.len(),
        score_plus_ape.len(),
        outputs,
        ratio,
    ) {
        return Err("device Compressor workspace contract mismatch".into());
    }
    // SAFETY: the explicit layer geometry and output rows are disjoint.
    unsafe {
        module.compressor_v4_bf16_f32(
            stream,
            LaunchConfig::for_num_elems(u32::try_from(kv.len())?),
            input,
            wkv,
            wgate,
            ape,
            kv,
            score,
            score_plus_ape,
            u32::try_from(start_pos)?,
            u32::try_from(hidden)?,
            u32::try_from(outputs)?,
            u32::try_from(ratio)?,
        )?;
    }
    Ok(())
}

pub(crate) fn compressor_workspace_contract_matches(
    input_elements: usize,
    wkv_elements: usize,
    wgate_elements: usize,
    ape_elements: usize,
    kv_elements: usize,
    score_elements: usize,
    score_plus_ape_elements: usize,
    outputs: usize,
    ratio: usize,
) -> bool {
    let hidden = wkv_elements.checked_div(outputs).unwrap_or(0);
    input_elements != 0
        && hidden != 0
        && input_elements.is_multiple_of(hidden)
        && outputs != 0
        && wkv_elements == outputs * hidden
        && wgate_elements == outputs * hidden
        && ape_elements == ratio * outputs
        && kv_elements == input_elements / hidden * outputs
        && score_elements == kv_elements
        && score_plus_ape_elements == kv_elements
}

pub(crate) fn run_generic_main_compressor_device(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    scratch: &mut DeviceRequestScratch,
    weights: &LayerDeviceWeights,
    path: GenericStationaryPath,
    position: usize,
) -> Result<(), Box<dyn Error>> {
    if scratch.exact_compressor.is_some() {
        let DeviceRequestScratch {
            attn_input,
            compressor_kv,
            compressor_score,
            compressor_score_plus_ape,
            hca_compressor_kv,
            hca_compressor_score,
            hca_compressor_score_plus_ape,
            exact_compressor,
            ..
        } = scratch;
        let strict = exact_compressor
            .as_mut()
            .ok_or("strict generic Compressor workspace disappeared")?;
        return match path {
            GenericStationaryPath::Csa4 => {
                let refresh_weights = strict.begin_main_refresh(path, weights.layer);
                let result = run_exact_compressor_buffers_device_into(
                    module,
                    stream,
                    attn_input,
                    &weights.compressor_wkv,
                    &weights.compressor_wgate,
                    &weights.compressor_ape,
                    compressor_kv,
                    compressor_score,
                    compressor_score_plus_ape,
                    &strict.cublas,
                    &strict.workspace,
                    &mut strict.input,
                    &mut strict.csa_wkv,
                    &mut strict.csa_wgate,
                    path.ratio(),
                    position,
                    refresh_weights,
                );
                if result.is_ok() && refresh_weights {
                    strict.finish_main_refresh(path, weights.layer, refresh_weights);
                } else if result.is_err() {
                    strict.invalidate();
                }
                result
            }
            GenericStationaryPath::Hca128 => {
                let refresh_weights = strict.begin_main_refresh(path, weights.layer);
                let result = run_exact_compressor_buffers_device_into(
                    module,
                    stream,
                    attn_input,
                    &weights.compressor_wkv,
                    &weights.compressor_wgate,
                    &weights.compressor_ape,
                    hca_compressor_kv,
                    hca_compressor_score,
                    hca_compressor_score_plus_ape,
                    &strict.cublas,
                    &strict.workspace,
                    &mut strict.input,
                    &mut strict.hca_wkv,
                    &mut strict.hca_wgate,
                    path.ratio(),
                    position,
                    refresh_weights,
                );
                if result.is_ok() && refresh_weights {
                    strict.finish_main_refresh(path, weights.layer, refresh_weights);
                } else if result.is_err() {
                    strict.invalidate();
                }
                result
            }
        };
    }
    match path {
        GenericStationaryPath::Csa4 => run_compressor_device_into(
            module,
            stream,
            &scratch.attn_input,
            &weights.compressor_wkv,
            &weights.compressor_wgate,
            &weights.compressor_ape,
            &mut scratch.compressor_kv,
            &mut scratch.compressor_score,
            &mut scratch.compressor_score_plus_ape,
            path.compressor_width(),
            path.ratio(),
            position,
        ),
        GenericStationaryPath::Hca128 => run_compressor_device_into(
            module,
            stream,
            &scratch.attn_input,
            &weights.compressor_wkv,
            &weights.compressor_wgate,
            &weights.compressor_ape,
            &mut scratch.hca_compressor_kv,
            &mut scratch.hca_compressor_score,
            &mut scratch.hca_compressor_score_plus_ape,
            path.compressor_width(),
            path.ratio(),
            position,
        ),
    }
}

pub(crate) fn run_generic_compressor_pool_device(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    scratch: &mut DeviceRequestScratch,
    path: GenericStationaryPath,
) -> Result<(), Box<dyn Error>> {
    match path {
        GenericStationaryPath::Csa4 => run_compressor_pool_ratio4_device_into(
            module,
            stream,
            &scratch.candidate_tail_kv,
            &scratch.candidate_tail_score,
            &mut scratch.compressed_pool,
            v4_state::Q_HEAD_DIM,
        ),
        GenericStationaryPath::Hca128 => run_compressor_pool_ratio128_device_into(
            module,
            stream,
            &scratch.candidate_hca_tail_kv,
            &scratch.candidate_hca_tail_score,
            &mut scratch.compressed_pool,
        ),
    }
}

pub(crate) fn run_generic_indexer_compressor_device(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    scratch: &mut DeviceRequestScratch,
    layer: usize,
    indexer: &DeviceLayerIndexer,
    position: usize,
) -> Result<(), Box<dyn Error>> {
    let DeviceRequestScratch {
        attn_input,
        indexer_compressor_kv,
        indexer_compressor_score,
        indexer_compressor_score_plus_ape,
        exact_compressor,
        ..
    } = scratch;
    let kv = indexer_compressor_kv
        .as_mut()
        .ok_or("generic CSA indexer scratch disappeared")?;
    let score = indexer_compressor_score
        .as_mut()
        .ok_or("generic CSA indexer scratch disappeared")?;
    let score_plus_ape = indexer_compressor_score_plus_ape
        .as_mut()
        .ok_or("generic CSA indexer scratch disappeared")?;
    if let Some(strict) = exact_compressor.as_mut() {
        let refresh_weights = strict.begin_indexer_refresh(layer);
        let result = run_exact_compressor_buffers_device_into(
            module,
            stream,
            attn_input,
            &indexer.compressor_wkv,
            &indexer.compressor_wgate,
            &indexer.compressor_ape,
            kv,
            score,
            score_plus_ape,
            &strict.cublas,
            &strict.workspace,
            &mut strict.input,
            &mut strict.csa_indexer_wkv,
            &mut strict.csa_indexer_wgate,
            4,
            position,
            refresh_weights,
        );
        if result.is_ok() && refresh_weights {
            strict.finish_indexer_refresh(layer, refresh_weights);
        } else if result.is_err() {
            strict.invalidate();
        }
        return result;
    }
    run_compressor_device_into(
        module,
        stream,
        attn_input,
        &indexer.compressor_wkv,
        &indexer.compressor_wgate,
        &indexer.compressor_ape,
        kv,
        score,
        score_plus_ape,
        256,
        4,
        position,
    )
}

pub(crate) fn l0_pos0_compressor_probe_enabled(layer: usize, start_pos: usize) -> bool {
    layer == 0
        && start_pos == 0
        && std::env::var_os("COLIBRI_V4_L0_POS0_CUTPOINT").as_deref()
            == Some(std::ffi::OsStr::new("1"))
}

pub(crate) fn strict_compressor_enabled() -> bool {
    strict_compressor_enabled_with(|name| {
        std::env::var_os(name).as_deref() == Some(std::ffi::OsStr::new("1"))
    }) || flash_runtime_strict_compressor_enabled(
        std::env::var_os("COLIBRI_V4_FLASH_BASE").as_deref() == Some(std::ffi::OsStr::new("1"))
            || std::env::var_os("COLIBRI_V4_FLASH_DSPARK").as_deref()
                == Some(std::ffi::OsStr::new("1"))
            || std::env::var_os("COLIBRI_V4_FLASH_SERVE").as_deref()
                == Some(std::ffi::OsStr::new("1")),
        std::env::var_os("COLIBRI_V4_FLASH_MHC_LINEAR_ARM").as_deref()
            == Some(std::ffi::OsStr::new("cublas")),
        std::env::var_os("COLIBRI_V4_FLASH_ROUTER_LINEAR_ARM").as_deref()
            == Some(std::ffi::OsStr::new("cublas")),
    )
}

pub(crate) fn flash_runtime_strict_compressor_enabled(
    flash_runtime: bool,
    mhc_cublas: bool,
    router_cublas: bool,
) -> bool {
    flash_exact_runtime_requires_strict_compressor(flash_runtime, mhc_cublas, router_cublas)
}

pub(crate) fn strict_compressor_enabled_with(enabled: impl Fn(&str) -> bool) -> bool {
    [
        "COLIBRI_V4_FULL61_B1",
        "COLIBRI_V4_B2_GENERATE",
        "COLIBRI_V4_B4_GENERATE",
        "COLIBRI_V4_B4_GREEDY_ABBA_BENCH",
        "COLIBRI_V4_DEVICE_TOKEN_STATE_GATE",
        "COLIBRI_V4_DEVICE_ATTENTION_PREFLIGHT_GATE",
    ]
    .into_iter()
    .any(enabled)
}

pub(crate) fn flash_exact_runtime_requires_strict_compressor(
    flash_runtime: bool,
    mhc_cublas: bool,
    router_cublas: bool,
) -> bool {
    flash_runtime && (mhc_cublas || router_cublas)
}

pub(crate) fn print_l0_pos0_compressor_probe(source: &str, kv: &[f32], score: &[f32], score_plus_ape: &[f32]) {
    let digests = (fnv1a_f32(kv), fnv1a_f32(score), fnv1a_f32(score_plus_ape));
    println!(
        "COLIBRI_L0_POS0_COMPRESSOR_PRECOMMIT={}",
        serde_json::json!({
            "source": source,
            "kv_f32_digest": format!("{:016x}", digests.0),
            "score_f32_digest": format!("{:016x}", digests.1),
            "score_plus_ape_f32_digest": format!("{:016x}", digests.2),
        })
    );
}

pub(crate) fn run_bf16_linear_f32(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    input: &[u16],
    weight: &DeviceBuffer<u16>,
    in_features: usize,
    out_features: usize,
) -> Result<Vec<f32>, Box<dyn Error>> {
    if input.is_empty()
        || input.len() % in_features != 0
        || weight.len() != in_features * out_features
    {
        return Err("layer BF16 projection geometry mismatch".into());
    }
    let input = DeviceBuffer::from_host(stream, input)?;
    let mut output = DeviceBuffer::<f32>::zeroed(stream, input.len() / in_features * out_features)?;
    // SAFETY: one thread owns each output and scans a complete validated row.
    unsafe {
        module.linear_v4_bf16_f32(
            stream,
            LaunchConfig::for_num_elems(u32::try_from(output.len())?),
            &input,
            weight,
            &mut output,
            u32::try_from(in_features)?,
            u32::try_from(out_features)?,
        )?;
    }
    Ok(output.to_host_vec(stream)?)
}

pub(crate) struct CsaIndexerPrepared {
    pub(crate) q_f32: Vec<f32>,
    pub(crate) weights_f32: Vec<f32>,
    pub(crate) compressor: v4_state::CompressorPrepared,
}

pub(crate) fn run_csa_indexer_projections(
    spec: &'static v4_spec::V4Spec,
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    attention_input: &[u16],
    q_lora: &[u16],
    freqs: &[f32],
    indexer: &DeviceLayerIndexer,
    start_pos: usize,
    cublas: Option<&cublas_exact::CublasExact>,
    strict_scratch: Option<&mut StrictCompressorScratch>,
) -> Result<CsaIndexerPrepared, Box<dyn Error>> {
    let rows = attention_input.len() / spec.hidden;
    if rows == 0
        || attention_input.len() != rows * spec.hidden
        || q_lora.len() != rows * spec.q_lora_rank
        || freqs.len() != rows * v4_state::Q_ROPE_DIM
    {
        return Err("CSA indexer projection input contract mismatch".into());
    }
    let (_, q) = run_l0_fp8_linear(module, stream, q_lora, &indexer.wq_b, true)?;
    let q = run_rope(module, stream, &q, freqs, 64, 128, false)?;
    let q_f32 = v4_state::rotate_fp4_sim_bf16(&q, rows * 64, 128)?;
    let mut weights_f32 = run_bf16_linear_f32(
        module,
        stream,
        attention_input,
        &indexer.weights_proj,
        spec.hidden,
        64,
    )?;
    let scale = 1.0_f32 / (128.0_f32 * 64.0).sqrt();
    for weight in &mut weights_f32 {
        *weight *= scale;
    }
    let compressor = if strict_compressor_enabled() {
        // Single position only, same reason as the main compressor above: the
        // pool holds one row's worth of buffers.
        let scratch = strict_scratch.filter(|scratch| attention_input.len() == scratch.hidden());
        if let Some(scratch) = scratch {
            scratch.run_from_device(
                stream,
                attention_input,
                &indexer.compressor_wkv,
                &indexer.compressor_wgate,
                &indexer.compressor_ape,
                256,
                4,
                start_pos,
            )?
        } else {
            let owned;
            let cublas = if let Some(cublas) = cublas {
                cublas
            } else {
                owned = cublas_exact::CublasExact::load()?;
                &owned
            };
            run_compressor_official_cublas_from_device(
                cublas,
                stream,
                attention_input,
                &indexer.compressor_wkv,
                &indexer.compressor_wgate,
                &indexer.compressor_ape,
                256,
                4,
                start_pos,
            )?
        }
    } else {
        run_compressor(
            module,
            stream,
            attention_input,
            &indexer.compressor_wkv,
            &indexer.compressor_wgate,
            &indexer.compressor_ape,
            256,
            4,
            start_pos,
        )?
    };
    Ok(CsaIndexerPrepared {
        q_f32,
        weights_f32,
        compressor,
    })
}

pub(crate) fn run_indexer_topk_gpu(
    spec: &'static v4_spec::V4Spec,
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    q_f32: &[f32],
    weights_f32: &[f32],
    kv_bf16: &[u16],
    rows: usize,
    valid_keys: &[usize],
    offset: usize,
) -> Result<(Vec<i32>, u64), Box<dyn Error>> {
    let started = std::time::Instant::now();
    let keys = kv_bf16.len() / 128;
    if rows == 0
        || keys == 0
        || kv_bf16.len() != keys * 128
        || q_f32.len() != rows * 64 * 128
        || weights_f32.len() != rows * 64
        || valid_keys.len() != rows
        || valid_keys.iter().any(|&valid| valid > keys)
    {
        return Err("CSA GPU indexer top-k input contract mismatch".into());
    }
    let kv_f32 = kv_bf16
        .iter()
        .map(|&bits| v4_router::bf16_to_f32(bits))
        .collect::<Vec<_>>();
    let q = DeviceBuffer::from_host(stream, q_f32)?;
    let kv = DeviceBuffer::from_host(stream, &kv_f32)?;
    let weights = DeviceBuffer::from_host(stream, weights_f32)?;
    let mut scores = DeviceBuffer::<f32>::zeroed(stream, rows * keys)?;
    // SAFETY: score geometry is fully checked and each thread owns one row/key.
    unsafe {
        module.indexer_scores_v4_f32(
            stream,
            LaunchConfig::for_num_elems(u32::try_from(scores.len())?),
            &q,
            &kv,
            &weights,
            &mut scores,
            u32::try_from(rows)?,
            64,
            128,
            u32::try_from(keys)?,
        )?;
    }
    let topk = keys.min(spec.index_topk);
    let valid = DeviceBuffer::from_host(
        stream,
        &valid_keys
            .iter()
            .map(|&value| u32::try_from(value))
            .collect::<Result<Vec<_>, _>>()?,
    )?;
    let mut selected = DeviceBuffer::<i32>::zeroed(stream, rows * topk)?;
    // SAFETY: deterministic rank scans are bounded by validated score/key counts.
    unsafe {
        module.indexer_topk_v4_f32(
            stream,
            LaunchConfig::for_num_elems(u32::try_from(selected.len())?),
            &scores,
            &valid,
            &mut selected,
            u32::try_from(rows)?,
            u32::try_from(keys)?,
            u32::try_from(topk)?,
            u32::try_from(offset)?,
        )?;
    }
    let output = selected.to_host_vec(stream)?;
    Ok((
        output,
        started.elapsed().as_nanos().try_into().unwrap_or(u64::MAX),
    ))
}

pub(crate) fn run_l0_sparse_attention(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    q: &[u16],
    kv: &[u16],
    attn_sink: &DeviceBuffer<f32>,
    start_pos: usize,
) -> Result<Vec<u16>, Box<dyn Error>> {
    let heads = attn_sink.len();
    let row_width = heads * v4_state::Q_HEAD_DIM;
    if q.is_empty() || q.len() % row_width != 0 {
        return Err("L0 sparse attention query shape mismatch".into());
    }
    let rows = q.len() / row_width;
    let topk = v4_state::build_hca_sparse_topk(rows, start_pos, v4_state::KV_WINDOW, 128)?;
    run_sparse_attention(module, stream, q, kv, attn_sink, &topk)
}

pub(crate) fn run_sparse_attention(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    q: &[u16],
    kv: &[u16],
    attn_sink: &DeviceBuffer<f32>,
    topk: &v4_state::SparseTopk,
) -> Result<Vec<u16>, Box<dyn Error>> {
    let row_width = attn_sink.len() * v4_state::Q_HEAD_DIM;
    let rows = q.len() / row_width;
    if q.is_empty()
        || q.len() % row_width != 0
        || topk.rows != rows
        || topk.indices.len() != rows * topk.per_row
        || kv.len() % v4_state::Q_HEAD_DIM != 0
    {
        return Err("layer sparse attention geometry mismatch".into());
    }
    let q = DeviceBuffer::from_host(stream, q)?;
    let kv = DeviceBuffer::from_host(stream, kv)?;
    let topk_device = DeviceBuffer::from_host(stream, &topk.indices)?;
    let mut output = DeviceBuffer::<u16>::zeroed(stream, q.len())?;
    run_sparse_attention_device_into(
        module,
        stream,
        &q,
        &kv,
        attn_sink,
        &topk_device,
        &mut output,
        topk.per_row,
    )?;
    Ok(output.to_host_vec(stream)?)
}

pub(crate) fn sparse_attention_arm_from_env() -> Result<SparseAttentionArm, Box<dyn Error>> {
    match std::env::var_os("COLIBRI_V4_SPARSE_ATTN_ARM") {
        None => Ok(SparseAttentionArm::Sm89Crossover),
        Some(value) if value == "reference" => Ok(SparseAttentionArm::Reference),
        Some(value) if value == "score_reuse" => Ok(SparseAttentionArm::ScoreReuse),
        Some(value) if value == "sm89_crossover" => Ok(SparseAttentionArm::Sm89Crossover),
        Some(_) => Err(
            "COLIBRI_V4_SPARSE_ATTN_ARM must be reference, score_reuse, or sm89_crossover".into(),
        ),
    }
}

pub(crate) fn run_sparse_attention_score_reuse(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    q: &[u16],
    kv: &[u16],
    attn_sink: &DeviceBuffer<f32>,
    topk: &v4_state::SparseTopk,
) -> Result<Vec<u16>, Box<dyn Error>> {
    let heads = attn_sink.len();
    let row_width = heads * v4_state::Q_HEAD_DIM;
    let rows = q.len() / row_width;
    if q.is_empty()
        || q.len() % row_width != 0
        || topk.rows != rows
        || topk.indices.len() != rows * topk.per_row
        || kv.len() % v4_state::Q_HEAD_DIM != 0
    {
        return Err("score-reuse sparse attention geometry mismatch".into());
    }
    let q = DeviceBuffer::from_host(stream, q)?;
    let kv = DeviceBuffer::from_host(stream, kv)?;
    let topk_device = DeviceBuffer::from_host(stream, &topk.indices)?;
    let mut output = DeviceBuffer::<u16>::zeroed(stream, q.len())?;
    let mut scratch =
        SparseAttentionScoreReuseScratch::new_for_heads(stream, rows, heads, topk.per_row)?;
    run_sparse_attention_score_reuse_device_into(
        module,
        stream,
        &q,
        &kv,
        attn_sink,
        &topk_device,
        &mut output,
        &mut scratch,
        topk.per_row,
    )?;
    Ok(output.to_host_vec(stream)?)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_sparse_attention_device_into(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    q: &DeviceBuffer<u16>,
    kv: &DeviceBuffer<u16>,
    attn_sink: &DeviceBuffer<f32>,
    topk: &DeviceBuffer<i32>,
    output: &mut DeviceBuffer<u16>,
    topk_per_row: usize,
) -> Result<(), Box<dyn Error>> {
    let heads = attn_sink.len();
    let row_width = heads * v4_state::Q_HEAD_DIM;
    if q.is_empty()
        || !q.len().is_multiple_of(row_width)
        || !kv.len().is_multiple_of(v4_state::Q_HEAD_DIM)
        || topk_per_row == 0
        || topk.len() < q.len() / row_width * topk_per_row
        || output.len() != q.len()
    {
        return Err("device sparse attention workspace contract mismatch".into());
    }
    // SAFETY: one output thread scans only validated top-k cache indices.
    unsafe {
        module.sparse_attn_v4_bf16(
            stream,
            LaunchConfig::for_num_elems(u32::try_from(output.len())?),
            q,
            kv,
            attn_sink,
            topk,
            output,
            u32::try_from(topk_per_row)?,
            u32::try_from(heads)?,
            u32::try_from(v4_state::Q_HEAD_DIM)?,
        )?;
    }
    Ok(())
}

pub(crate) struct SparseAttentionScoreReuseScratch {
    pub(crate) scores: DeviceBuffer<f32>,
    pub(crate) probabilities: DeviceBuffer<f32>,
    pub(crate) denominators: DeviceBuffer<f32>,
    pub(crate) valid_counts: DeviceBuffer<u32>,
}

impl SparseAttentionScoreReuseScratch {
    pub(crate) fn new(stream: &CudaStream, rows: usize, topk_per_row: usize) -> Result<Self, Box<dyn Error>> {
        Self::new_for_heads(stream, rows, v4_state::Q_HEADS, topk_per_row)
    }

    pub(crate) fn new_for_heads(
        stream: &CudaStream,
        rows: usize,
        heads: usize,
        topk_per_row: usize,
    ) -> Result<Self, Box<dyn Error>> {
        let score_len = rows
            .checked_mul(heads)
            .and_then(|value| value.checked_mul(topk_per_row))
            .ok_or("sparse attention score scratch length overflow")?;
        let head_len = rows
            .checked_mul(heads)
            .ok_or("sparse attention head scratch length overflow")?;
        Ok(Self {
            scores: DeviceBuffer::zeroed(stream, score_len)?,
            probabilities: DeviceBuffer::zeroed(stream, score_len)?,
            denominators: DeviceBuffer::zeroed(stream, head_len)?,
            valid_counts: DeviceBuffer::zeroed(stream, head_len)?,
        })
    }

    pub(crate) fn num_bytes(&self) -> usize {
        self.scores.num_bytes()
            + self.probabilities.num_bytes()
            + self.denominators.num_bytes()
            + self.valid_counts.num_bytes()
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_sparse_attention_reference_state_device_into(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    q: &DeviceBuffer<u16>,
    kv: &DeviceBuffer<u16>,
    attn_sink: &DeviceBuffer<f32>,
    topk: &DeviceBuffer<i32>,
    scratch: &mut SparseAttentionScoreReuseScratch,
    topk_per_row: usize,
) -> Result<(), Box<dyn Error>> {
    let heads = attn_sink.len();
    let row_width = heads * v4_state::Q_HEAD_DIM;
    let rows = q.len() / row_width;
    let score_len = rows
        .checked_mul(heads)
        .and_then(|value| value.checked_mul(topk_per_row))
        .ok_or("sparse attention reference-state length overflow")?;
    let head_len = rows
        .checked_mul(heads)
        .ok_or("sparse attention reference-state head length overflow")?;
    if q.is_empty()
        || q.len() != rows * row_width
        || !kv.len().is_multiple_of(v4_state::Q_HEAD_DIM)
        || attn_sink.len() != heads
        || topk_per_row == 0
        || topk.len() < rows * topk_per_row
        || scratch.scores.len() < score_len
        || scratch.probabilities.len() < score_len
        || scratch.denominators.len() != head_len
        || scratch.valid_counts.len() != head_len
    {
        return Err("reference-state sparse attention workspace contract mismatch".into());
    }
    // SAFETY: this probe is scalar-equivalent instrumentation: one head
    // preserves both score scans and all original floating-point orders.
    unsafe {
        module.sparse_attn_v4_reference_state(
            stream,
            LaunchConfig::for_num_elems(u32::try_from(head_len)?),
            q,
            kv,
            attn_sink,
            topk,
            &mut scratch.scores,
            &mut scratch.probabilities,
            &mut scratch.denominators,
            &mut scratch.valid_counts,
            u32::try_from(topk_per_row)?,
            u32::try_from(heads)?,
            u32::try_from(v4_state::Q_HEAD_DIM)?,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_sparse_attention_score_reuse_device_into(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    q: &DeviceBuffer<u16>,
    kv: &DeviceBuffer<u16>,
    attn_sink: &DeviceBuffer<f32>,
    topk: &DeviceBuffer<i32>,
    output: &mut DeviceBuffer<u16>,
    scratch: &mut SparseAttentionScoreReuseScratch,
    topk_per_row: usize,
) -> Result<(), Box<dyn Error>> {
    let heads = attn_sink.len();
    let row_width = heads * v4_state::Q_HEAD_DIM;
    let rows = q.len() / row_width;
    let score_len = rows
        .checked_mul(heads)
        .and_then(|value| value.checked_mul(topk_per_row))
        .ok_or("sparse attention score launch length overflow")?;
    let head_len = rows
        .checked_mul(heads)
        .ok_or("sparse attention probability launch length overflow")?;
    if !sparse_attention_score_reuse_workspace_matches(
        q.len(),
        kv.len(),
        attn_sink.len(),
        topk.len(),
        output.len(),
        scratch.scores.len(),
        scratch.probabilities.len(),
        scratch.denominators.len(),
        scratch.valid_counts.len(),
        topk_per_row,
        heads,
    ) {
        return Err("score-reuse sparse attention workspace contract mismatch".into());
    }
    // SAFETY: each phase has a disjoint output; the next phase consumes the
    // preceding phase in the same stream, preserving the scalar reference's
    // score, max, probability, denominator, and numerator orders.
    unsafe {
        module.sparse_attn_v4_scores(
            stream,
            LaunchConfig::for_num_elems(u32::try_from(score_len)?),
            q,
            kv,
            topk,
            &mut scratch.scores,
            u32::try_from(topk_per_row)?,
            u32::try_from(heads)?,
            u32::try_from(v4_state::Q_HEAD_DIM)?,
        )?;
        module.sparse_attn_v4_probabilities(
            stream,
            LaunchConfig::for_num_elems(u32::try_from(head_len)?),
            &scratch.scores,
            kv,
            attn_sink,
            topk,
            &mut scratch.probabilities,
            &mut scratch.denominators,
            &mut scratch.valid_counts,
            u32::try_from(topk_per_row)?,
            u32::try_from(heads)?,
            u32::try_from(v4_state::Q_HEAD_DIM)?,
        )?;
        module.sparse_attn_v4_values(
            stream,
            LaunchConfig::for_num_elems(u32::try_from(output.len())?),
            kv,
            topk,
            &scratch.probabilities,
            &scratch.denominators,
            &scratch.valid_counts,
            output,
            u32::try_from(topk_per_row)?,
            u32::try_from(heads)?,
            u32::try_from(v4_state::Q_HEAD_DIM)?,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn sparse_attention_score_reuse_workspace_matches(
    q_elements: usize,
    kv_elements: usize,
    attn_sink_elements: usize,
    topk_capacity: usize,
    output_elements: usize,
    score_capacity: usize,
    probability_capacity: usize,
    denominator_elements: usize,
    valid_count_elements: usize,
    active_topk_per_row: usize,
    heads: usize,
) -> bool {
    let row_width = heads * v4_state::Q_HEAD_DIM;
    let rows = q_elements / row_width;
    let Some(active_topk) = rows.checked_mul(active_topk_per_row) else {
        return false;
    };
    let Some(active_scores) = rows
        .checked_mul(heads)
        .and_then(|value| value.checked_mul(active_topk_per_row))
    else {
        return false;
    };
    let Some(head_elements) = rows.checked_mul(heads) else {
        return false;
    };
    q_elements != 0
        && q_elements == rows * row_width
        && kv_elements.is_multiple_of(v4_state::Q_HEAD_DIM)
        && attn_sink_elements == heads
        && active_topk_per_row != 0
        && topk_capacity >= active_topk
        && output_elements == q_elements
        && score_capacity >= active_scores
        && probability_capacity >= active_scores
        && denominator_elements == head_elements
        && valid_count_elements == head_elements
}

pub(crate) fn run_l0_wo_a_device_into(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    input: &DeviceBuffer<u16>,
    weight: &DeviceBuffer<u16>,
    output: &mut DeviceBuffer<u16>,
) -> Result<(), Box<dyn Error>> {
    run_l0_wo_a_device_into_for(&v4_spec::PRO, module, stream, input, weight, output)
}

pub(crate) fn run_l0_wo_a_device_into_for(
    spec: &v4_spec::V4Spec,
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    input: &DeviceBuffer<u16>,
    weight: &DeviceBuffer<u16>,
    output: &mut DeviceBuffer<u16>,
) -> Result<(), Box<dyn Error>> {
    let input_width = spec.q_heads * spec.q_head_dim;
    let output_width = spec.o_groups * spec.o_lora_rank;
    let weight_elements = output_width * input_width / spec.o_groups;
    if input.is_empty()
        || !input.len().is_multiple_of(input_width)
        || weight.len() != weight_elements
        || output.len() != input.len() / input_width * output_width
    {
        return Err("device wo_a workspace contract mismatch".into());
    }
    // SAFETY: resident grouped matrix and fixed row widths were checked.
    unsafe {
        module.wo_a_v4_bf16(
            stream,
            LaunchConfig::for_num_elems(u32::try_from(output.len())?),
            input,
            weight,
            output,
            u32::try_from(spec.q_heads)?,
            u32::try_from(spec.q_head_dim)?,
            u32::try_from(spec.o_groups)?,
            u32::try_from(spec.o_lora_rank)?,
        )?;
    }
    Ok(())
}

pub(crate) fn run_l0_wo_a(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    input: &[u16],
    weight: &DeviceBuffer<u16>,
) -> Result<Vec<u16>, Box<dyn Error>> {
    run_l0_wo_a_for(&v4_spec::PRO, module, stream, input, weight)
}

pub(crate) fn run_l0_wo_a_for(
    spec: &v4_spec::V4Spec,
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    input: &[u16],
    weight: &DeviceBuffer<u16>,
) -> Result<Vec<u16>, Box<dyn Error>> {
    let input_width = spec.q_heads * spec.q_head_dim;
    let output_width = spec.o_groups * spec.o_lora_rank;
    if input.is_empty() || input.len() % input_width != 0 {
        return Err("L0 wo_a input shape mismatch".into());
    }
    let input = DeviceBuffer::from_host(stream, input)?;
    let mut output = DeviceBuffer::<u16>::zeroed(stream, input.len() / input_width * output_width)?;
    run_l0_wo_a_device_into_for(spec, module, stream, &input, weight, &mut output)?;
    Ok(output.to_host_vec(stream)?)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_l0_hc_post_device_into(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    branch: &DeviceBuffer<u16>,
    residual: &DeviceBuffer<u16>,
    post: &DeviceBuffer<f32>,
    comb: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<u16>,
) -> Result<(), Box<dyn Error>> {
    if post.len() % v4_state::HC != 0 {
        return Err("device HC post metadata mismatch".into());
    }
    let rows = post.len() / v4_state::HC;
    if rows == 0 || branch.len() % rows != 0 {
        return Err("device HC post row mismatch".into());
    }
    if branch.is_empty()
        || residual.len() != branch.len() * v4_state::HC
        || post.len() != rows * v4_state::HC
        || comb.len() != post.len() * v4_state::HC
        || output.len() != residual.len()
    {
        return Err("device HC post workspace contract mismatch".into());
    }
    // SAFETY: complete row/HC shapes prove unique output ownership.
    unsafe {
        module.hc_post_v4_bf16(
            stream,
            LaunchConfig::for_num_elems(u32::try_from(output.len())?),
            branch,
            residual,
            post,
            comb,
            output,
        )?;
    }
    Ok(())
}

pub(crate) fn run_l0_hc_post(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    branch: &[u16],
    residual: &[u16],
    post: &[f32],
    comb: &[f32],
) -> Result<Vec<u16>, Box<dyn Error>> {
    if post.len() % v4_state::HC != 0 {
        return Err("L0 HC post metadata mismatch".into());
    }
    let rows = post.len() / v4_state::HC;
    if branch.is_empty()
        || rows == 0
        || branch.len() % rows != 0
        || residual.len() != branch.len() * v4_state::HC
    {
        return Err("L0 HC post shape mismatch".into());
    }
    let branch = DeviceBuffer::from_host(stream, branch)?;
    let residual = DeviceBuffer::from_host(stream, residual)?;
    let post = DeviceBuffer::from_host(stream, post)?;
    let comb = DeviceBuffer::from_host(stream, comb)?;
    let mut output = DeviceBuffer::<u16>::zeroed(stream, residual.len())?;
    run_l0_hc_post_device_into(
        module,
        stream,
        &branch,
        &residual,
        &post,
        &comb,
        &mut output,
    )?;
    Ok(output.to_host_vec(stream)?)
}

pub(crate) fn run_l0_hc_post_trace(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    branch: &[u16],
    residual: &[u16],
    post: &[f32],
    comb: &[f32],
    audit_index: usize,
) -> Result<Vec<u32>, Box<dyn Error>> {
    pub(crate) const WORDS: usize = 21;
    if audit_index >= residual.len() {
        return Err("Flash HC post trace index is outside residual".into());
    }
    let branch = DeviceBuffer::from_host(stream, branch)?;
    let residual = DeviceBuffer::from_host(stream, residual)?;
    let post = DeviceBuffer::from_host(stream, post)?;
    let comb = DeviceBuffer::from_host(stream, comb)?;
    let mut audit = DeviceBuffer::<u32>::zeroed(stream, WORDS)?;
    // SAFETY: the trace kernel repeats the checked HC-post geometry for one index.
    unsafe {
        module.hc_post_v4_bf16_trace(
            stream,
            LaunchConfig::for_num_elems(1),
            &branch,
            &residual,
            &post,
            &comb,
            u32::try_from(audit_index)?,
            &mut audit,
        )?;
    }
    Ok(audit.to_host_vec(stream)?)
}

pub(crate) fn run_layer_router(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    session: Option<&FlashCublasSession>,
    input: &[u16],
    weights: &LayerDeviceWeights,
    hash_ids: &[i64],
    capture_debug: bool,
) -> Result<LayerRouterRun, Box<dyn Error>> {
    let spec = weights.spec;
    let score_arm = flash_router_score_arm(spec)?;
    let linear_arm = flash_router_linear_arm(spec)?;
    if input.is_empty() || !input.len().is_multiple_of(spec.hidden) {
        return Err("layer router host input mismatch".into());
    }
    let rows = input.len() / spec.hidden;
    if (weights.hash_router && hash_ids.len() != rows * spec.experts_per_token)
        || (!weights.hash_router && !hash_ids.is_empty())
    {
        return Err("layer router mode/input mismatch".into());
    }
    let profile = bench_telemetry_enabled();
    let owned_session = (linear_arm == FlashMhcLinearArm::Cublas && session.is_none())
        .then(|| FlashCublasSession::new(stream))
        .transpose()?;
    let session = session.or(owned_session.as_ref());
    let mut telemetry = RouterTelemetry::default();
    let alloc_h2d_started = std::time::Instant::now();
    let input = DeviceBuffer::from_host(stream, input)?;
    let hash_ids_device = DeviceBuffer::from_host(stream, hash_ids)?;
    if profile {
        telemetry.alloc_h2d_sync_ns =
            u64::try_from(alloc_h2d_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
    }
    let workspace_alloc_started = std::time::Instant::now();
    let mut normalized_device = DeviceBuffer::<u16>::zeroed(stream, input.len())?;
    let mut logits = DeviceBuffer::<f32>::zeroed(stream, rows * spec.routed_experts)?;
    let mut ids = DeviceBuffer::<i32>::zeroed(stream, rows * spec.experts_per_token)?;
    let mut route_weights = DeviceBuffer::<f32>::zeroed(stream, rows * spec.experts_per_token)?;
    if profile {
        telemetry.workspace_alloc_ns =
            u64::try_from(workspace_alloc_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
    }
    let kernel_select_started = std::time::Instant::now();
    let events = profile
        .then(|| {
            Ok::<_, Box<dyn Error>>((
                input
                    .context()
                    .new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?,
                input
                    .context()
                    .new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?,
            ))
        })
        .transpose()?;
    if let Some((start, _)) = events.as_ref() {
        start.record(stream)?;
    }
    let mut router_debug = if capture_debug {
        Some((
            DeviceBuffer::<f32>::zeroed(stream, rows * 3)?,
            Vec::<f32>::new(),
            DeviceBuffer::<f32>::zeroed(stream, logits.len())?,
            DeviceBuffer::<f32>::zeroed(stream, logits.len())?,
            DeviceBuffer::<f32>::zeroed(stream, ids.len())?,
            DeviceBuffer::<f32>::zeroed(stream, ids.len())?,
        ))
    } else {
        None
    };
    let mut empty_rmsnorm_trace = DeviceBuffer::<f32>::zeroed(stream, 0)?;
    let mut empty_rmsnorm_squares = Vec::new();
    let mut empty_softplus = DeviceBuffer::<f32>::zeroed(stream, 0)?;
    let mut empty_post_scores = DeviceBuffer::<f32>::zeroed(stream, 0)?;
    let mut empty_selected_scores = DeviceBuffer::<f32>::zeroed(stream, 0)?;
    let mut empty_denom_steps = DeviceBuffer::<f32>::zeroed(stream, 0)?;
    let (
        trace_rmsnorm,
        trace_rmsnorm_squares,
        trace_softplus,
        trace_post_scores,
        trace_selected_scores,
        trace_denom_steps,
    ) = match router_debug.as_mut() {
        Some((rmsnorm, rmsnorm_squares, softplus, post_scores, selected_scores, denom_steps)) => (
            rmsnorm,
            rmsnorm_squares,
            softplus,
            post_scores,
            selected_scores,
            denom_steps,
        ),
        None => (
            &mut empty_rmsnorm_trace,
            &mut empty_rmsnorm_squares,
            &mut empty_softplus,
            &mut empty_post_scores,
            &mut empty_selected_scores,
            &mut empty_denom_steps,
        ),
    };
    run_layer_router_device_into(
        module,
        stream,
        &input,
        weights,
        &hash_ids_device,
        &mut normalized_device,
        &mut logits,
        &mut ids,
        &mut route_weights,
        session,
        score_arm,
        trace_rmsnorm,
        trace_rmsnorm_squares,
        trace_softplus,
        trace_post_scores,
        trace_selected_scores,
        trace_denom_steps,
    )?;
    if let Some((start, end)) = events.as_ref() {
        end.record(stream)?;
        end.synchronize()?;
        telemetry.kernel_select_gpu_ns = (start.elapsed_ms(end)? as f64 * 1_000_000.0) as u64;
        telemetry.kernel_select_sync_ns =
            u64::try_from(kernel_select_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
    }
    let d2h_started = std::time::Instant::now();
    let normalized = normalized_device.to_host_vec(stream)?;
    let routes = v4_router::RouterSelection {
        ids: ids.to_host_vec(stream)?,
        weights: route_weights.to_host_vec(stream)?,
    };
    let debug = if let Some((
        rmsnorm,
        rmsnorm_squares,
        softplus,
        post_scores,
        selected_scores,
        denom_steps,
    )) = router_debug.take()
    {
        Some(RouterDebug {
            rmsnorm: rmsnorm.to_host_vec(stream)?,
            rmsnorm_squares,
            logits: logits.to_host_vec(stream)?,
            softplus: softplus.to_host_vec(stream)?,
            post_scores: post_scores.to_host_vec(stream)?,
            selected_scores: selected_scores.to_host_vec(stream)?,
            denom_steps: denom_steps.to_host_vec(stream)?,
        })
    } else {
        None
    };
    if profile {
        telemetry.d2h_sync_ns = u64::try_from(d2h_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
    }
    Ok(LayerRouterRun {
        input: normalized,
        routes,
        debug,
        telemetry,
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_layer_router_device_into(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    input: &DeviceBuffer<u16>,
    weights: &LayerDeviceWeights,
    hash_ids: &DeviceBuffer<i64>,
    normalized: &mut DeviceBuffer<u16>,
    logits: &mut DeviceBuffer<f32>,
    ids: &mut DeviceBuffer<i32>,
    route_weights: &mut DeviceBuffer<f32>,
    session: Option<&FlashCublasSession>,
    score_arm: FlashRouterScoreArm,
    trace_rmsnorm: &mut DeviceBuffer<f32>,
    trace_rmsnorm_squares: &mut Vec<f32>,
    trace_softplus: &mut DeviceBuffer<f32>,
    trace_post_scores: &mut DeviceBuffer<f32>,
    trace_selected_scores: &mut DeviceBuffer<f32>,
    trace_denom_steps: &mut DeviceBuffer<f32>,
) -> Result<(), Box<dyn Error>> {
    let spec = weights.spec;
    if input.is_empty() || !input.len().is_multiple_of(spec.hidden) {
        return Err("device router input contract mismatch".into());
    }
    let rows = input.len() / spec.hidden;
    if normalized.len() != input.len()
        || logits.len() != rows * spec.routed_experts
        || ids.len() != rows * spec.experts_per_token
        || route_weights.len() != ids.len()
        || (weights.hash_router && hash_ids.len() != ids.len())
        || (!weights.hash_router && !hash_ids.is_empty())
    {
        return Err("device router workspace contract mismatch".into());
    }
    if spec == &v4_spec::FLASH_0731
        && !weights.hash_router
        && spec.hidden == 4_096
        && weights.ffn_norm.len() == 4_096
    {
        *trace_rmsnorm_squares = run_flash_learned_rmsnorm_4096_device_into(
            module,
            stream,
            input,
            &weights.ffn_norm,
            normalized,
            trace_rmsnorm,
        )?;
    } else {
        run_l0_rmsnorm_device_into(
            module,
            stream,
            input,
            &weights.ffn_norm,
            normalized,
            spec.hidden,
        )?;
    }
    let launch = LaunchConfig::for_num_elems(u32::try_from(rows)?);
    let _router_cublas = match flash_router_linear_arm(spec)? {
        FlashMhcLinearArm::Canonical => {
            // SAFETY: the fixed gate matrix satisfies the canonical gate kernel.
            unsafe {
                module.gate_logits_v4_bf16(
                    stream,
                    LaunchConfig::for_num_elems(u32::try_from(logits.len())?),
                    normalized,
                    &weights.gate,
                    logits,
                    u32::try_from(spec.hidden)?,
                    u32::try_from(spec.routed_experts)?,
                )?;
            }
            None
        }
        FlashMhcLinearArm::Cublas => Some(run_flash_router_cublas_linear_device_into(
            session.ok_or("Flash cuBLAS router session is missing")?,
            module,
            stream,
            normalized,
            &weights.gate,
            logits,
            rows,
            spec.hidden,
            spec.routed_experts,
        )?),
    };
    // SAFETY: both router-linear arms produce complete logits under the same
    // fixed gate geometry; selection retains the existing score/top-k path.
    unsafe {
        module.select_v4_router(
            stream,
            launch,
            logits,
            &weights.router_bias,
            hash_ids,
            ids,
            route_weights,
            trace_softplus,
            trace_post_scores,
            trace_selected_scores,
            trace_denom_steps,
            u32::from(weights.hash_router),
            u32::try_from(spec.routed_experts)?,
            spec.route_scale(),
            score_arm.kernel_flag(),
        )?;
    }
    Ok(())
}

pub(crate) fn run_layer_shared_expert(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    input: &[u16],
    weights: &LayerDeviceWeights,
) -> Result<Vec<f32>, Box<dyn Error>> {
    Ok(run_layer_shared_expert_with_bf16(module, stream, input, weights)?.0)
}

pub(crate) fn run_layer_shared_expert_with_bf16(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    input: &[u16],
    weights: &LayerDeviceWeights,
) -> Result<(Vec<f32>, Vec<u16>), Box<dyn Error>> {
    let (_, gate) = run_l0_fp8_linear(module, stream, input, &weights.shared_w1, false)?;
    let (_, up) = run_l0_fp8_linear(module, stream, input, &weights.shared_w3, false)?;
    let gate = DeviceBuffer::from_host(stream, &gate)?;
    let up = DeviceBuffer::from_host(stream, &up)?;
    let mut activated = DeviceBuffer::<u16>::zeroed(stream, gate.len())?;
    launch_v4_swiglu_route(module, stream, &gate, &up, &mut activated, 1.0)?;
    let activated = activated.to_host_vec(stream)?;
    let (_, output) = run_l0_fp8_linear(module, stream, &activated, &weights.shared_w2, false)?;
    Ok((
        output.iter().copied().map(v4_router::bf16_to_f32).collect(),
        output,
    ))
}

