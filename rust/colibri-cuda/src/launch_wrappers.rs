use super::*;

pub(crate) fn launch_v4_activation_quantize(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    input: &DeviceBuffer<u16>,
    output: &mut DeviceBuffer<f32>,
) -> Result<(), Box<dyn Error>> {
    const BLOCK: usize = 128;
    if input.len() == 0 || input.len() % BLOCK != 0 || output.len() != input.len() {
        return Err("V4 activation quantization requires complete 128-value groups".into());
    }
    let elements = u32::try_from(input.len())?;
    // SAFETY: the validated launch assigns one output element to each thread;
    // each thread scans its complete 128-value source group before writing.
    unsafe {
        module.quantize_v4_activation(stream, LaunchConfig::for_num_elems(elements), input, output)
    }?;
    Ok(())
}

pub(crate) fn launch_v4_activation_quantize_raw(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    input: &DeviceBuffer<u16>,
    output: &mut DeviceBuffer<u8>,
    scales: &mut DeviceBuffer<u8>,
) -> Result<(), Box<dyn Error>> {
    const BLOCK: usize = v4_state::FP8_BLOCK;
    if input.is_empty()
        || input.len() % BLOCK != 0
        || output.len() != input.len()
        || scales.len() != input.len() / BLOCK
    {
        return Err("Flash FP4 raw activation buffers require complete 128-value groups".into());
    }
    // SAFETY: one thread owns one complete 128-value raw-FP8 group.
    unsafe {
        module.quantize_v4_activation_raw(
            stream,
            LaunchConfig::for_num_elems(u32::try_from(scales.len())?),
            input,
            output,
            scales,
        )
    }?;
    Ok(())
}

/// Materializes the raw FP8/E8M0 view of a live BF16 device buffer for the
/// Flash L0 parity harness.  This is deliberately debug-only: production
/// expert GEMMs retain their existing dequantized-F32 input path.
pub(crate) fn flash_parity_capture_activation_fp8(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    input: &DeviceBuffer<u16>,
) -> Result<(Vec<u8>, Vec<u8>), Box<dyn Error>> {
    const BLOCK: usize = v4_state::FP8_BLOCK;
    if input.is_empty() || input.len() % BLOCK != 0 {
        return Err("Flash parity FP8 trace requires complete 128-value groups".into());
    }
    let mut fp8 = DeviceBuffer::<u8>::zeroed(stream, input.len())?;
    let mut scales = DeviceBuffer::<u8>::zeroed(stream, input.len() / BLOCK)?;
    let mut no_dequantized = DeviceBuffer::<u16>::zeroed(stream, 0)?;
    // SAFETY: one launch thread owns each complete 128-value group; this only
    // snapshots the live source buffer and never feeds a model kernel.
    unsafe {
        module.quantize_v4_fp8_bf16(
            stream,
            LaunchConfig::for_num_elems(u32::try_from(scales.len())?),
            input,
            &mut fp8,
            &mut scales,
            &mut no_dequantized,
            u32::try_from(input.len())?,
            u32::try_from(input.len())?,
            u32::try_from(BLOCK)?,
        )?;
    }
    Ok((fp8.to_host_vec(stream)?, scales.to_host_vec(stream)?))
}

pub(crate) fn launch_v4_fp4_matvec(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    input: &DeviceBuffer<f32>,
    packed: &DeviceBuffer<u8>,
    scales: &DeviceBuffer<u8>,
    output: &mut DeviceBuffer<u16>,
    rows: usize,
    out_features: usize,
    in_features: usize,
) -> Result<(), Box<dyn Error>> {
    if rows == 0
        || rows > v4_expert::V4_MAX_EXPERT_ROWS
        || in_features % 32 != 0
        || input.len() != rows * in_features
        || packed.len() != out_features * in_features / 2
        || scales.len() != out_features * in_features / 32
        || output.len() != rows * out_features
    {
        return Err("V4 FP4 matvec buffers disagree with the fixed expert layout".into());
    }
    let threads = u32::try_from(
        output
            .len()
            .checked_mul(32)
            .ok_or("V4 matvec launch overflow")?,
    )?;
    // SAFETY: each complete warp owns one bounded output element. Host validation
    // proves the packed-two-value and one-E8M0-per-32-K indexing used by the kernel.
    unsafe {
        module.matvec_v4_fp4(
            stream,
            LaunchConfig::for_num_elems(threads),
            input,
            packed,
            scales,
            output,
            u32::try_from(in_features)?,
            u32::try_from(out_features)?,
        )
    }?;
    Ok(())
}

pub(crate) fn launch_v4_fp4_matvec_block32(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    input: &DeviceBuffer<u8>,
    input_scales: &DeviceBuffer<u8>,
    packed: &DeviceBuffer<u8>,
    scales: &DeviceBuffer<u8>,
    output: &mut DeviceBuffer<u16>,
    rows: usize,
    out_features: usize,
    in_features: usize,
) -> Result<(), Box<dyn Error>> {
    const WEIGHT_BLOCK: usize = 32;
    const ACTIVATION_BLOCK: usize = v4_state::FP8_BLOCK;
    if rows == 0
        || rows > v4_expert::V4_MAX_EXPERT_ROWS
        || in_features % ACTIVATION_BLOCK != 0
        || input.len() != rows * in_features
        || input_scales.len() != rows * in_features / ACTIVATION_BLOCK
        || packed.len() != out_features * in_features / 2
        || scales.len() != out_features * in_features / WEIGHT_BLOCK
        || output.len() != rows * out_features
    {
        return Err("Flash FP4 block32 buffers disagree with the expert layout".into());
    }
    let threads = u32::try_from(
        output
            .len()
            .checked_mul(32)
            .ok_or("Flash FP4 block32 launch overflow")?,
    )?;
    // SAFETY: one complete warp owns each output; each 32-K block is reduced
    // before lane 0 applies the corresponding activation and weight scales.
    unsafe {
        module.matvec_v4_fp4_block32(
            stream,
            LaunchConfig::for_num_elems(threads),
            input,
            input_scales,
            packed,
            scales,
            output,
            u32::try_from(in_features)?,
            u32::try_from(out_features)?,
        )
    }?;
    Ok(())
}

pub(crate) fn launch_v4_fp4_matvec_mma(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    input: &DeviceBuffer<u8>,
    input_scales: &DeviceBuffer<u8>,
    packed: &DeviceBuffer<u8>,
    scales: &DeviceBuffer<u8>,
    output: &mut DeviceBuffer<u16>,
    rows: usize,
    out_features: usize,
    in_features: usize,
) -> Result<(), Box<dyn Error>> {
    if rows == 0
        || rows > v4_expert::V4_MAX_EXPERT_ROWS
        || in_features % v4_state::FP8_BLOCK != 0
        || out_features % 8 != 0
        || input.len() != rows * in_features
        || input_scales.len() != rows * in_features / v4_state::FP8_BLOCK
        || packed.len() != out_features * in_features / 2
        || scales.len() != out_features * in_features / 32
        || output.len() != rows * out_features
    {
        return Err("Flash FP8-MMA buffers disagree with the expert layout".into());
    }
    let threads = rows
        .checked_mul(out_features / 8)
        .and_then(|tiles| tiles.checked_mul(32))
        .ok_or("Flash FP8-MMA launch overflow")?;
    // SAFETY: the locked kernel maps one full warp to an eight-wide output tile;
    // host validation proves every FP8/FP4 packed span and E8M0 scale span.
    unsafe {
        module.matvec_v4_fp4_mma(
            stream,
            LaunchConfig::for_num_elems(u32::try_from(threads)?),
            input,
            input_scales,
            packed,
            scales,
            output,
            u32::try_from(in_features)?,
            u32::try_from(out_features)?,
        )
    }?;
    Ok(())
}

pub(crate) fn launch_v4_fp4_tilelang_ref(
    reference: &v4_tilelang_ref::TilelangFp4Reference,
    stream: &CudaStream,
    input: &DeviceBuffer<u8>,
    input_scales: &DeviceBuffer<u8>,
    packed: &DeviceBuffer<u8>,
    scales: &DeviceBuffer<u8>,
    output: &mut DeviceBuffer<u16>,
    rows: usize,
    out_features: usize,
    in_features: usize,
) -> Result<(), Box<dyn Error>> {
    if rows == 0
        || rows > v4_expert::V4_MAX_EXPERT_ROWS
        || input.len() != rows * in_features
        || input_scales.len() != rows * in_features / v4_state::FP8_BLOCK
        || packed.len() != out_features * in_features / 2
        || scales.len() != out_features * in_features / 32
        || output.len() != rows * out_features
    {
        return Err("TileLang reference buffers disagree with the locked expert layout".into());
    }
    let kernel = reference.for_dimensions(out_features, in_features)?;
    let (block_x, block_y, block_z) = kernel.block();
    let mut input_ptr = input.cu_deviceptr();
    let mut packed_ptr = packed.cu_deviceptr();
    let mut output_ptr = output.cu_deviceptr();
    let mut input_scales_ptr = input_scales.cu_deviceptr();
    let mut scales_ptr = scales.cu_deviceptr();
    let mut rows_u32 = u32::try_from(rows)?;
    let mut arguments = [
        (&mut input_ptr as *mut _) as *mut std::ffi::c_void,
        (&mut packed_ptr as *mut _) as *mut std::ffi::c_void,
        (&mut output_ptr as *mut _) as *mut std::ffi::c_void,
        (&mut input_scales_ptr as *mut _) as *mut std::ffi::c_void,
        (&mut scales_ptr as *mut _) as *mut std::ffi::c_void,
        (&mut rows_u32 as *mut _) as *mut std::ffi::c_void,
    ];
    // SAFETY: `v4_tilelang_ref` has already validated the exact raw ABI and
    // PTX identity. These six local values remain alive through submission.
    unsafe {
        cuda_core::launch_kernel_on_stream(
            kernel.function(),
            (kernel.grid_x(), kernel.grid_y_for_rows(rows)?, 1),
            (block_x, block_y, block_z),
            kernel.dynamic_shared_bytes(),
            stream,
            &mut arguments,
        )
    }?;
    Ok(())
}

pub(crate) fn launch_v4_swiglu_route(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    gate: &DeviceBuffer<u16>,
    up: &DeviceBuffer<u16>,
    output: &mut DeviceBuffer<u16>,
    route_weight: f32,
) -> Result<(), Box<dyn Error>> {
    if !route_weight.is_finite()
        || gate.len() == 0
        || up.len() != gate.len()
        || output.len() != gate.len()
    {
        return Err("V4 SwiGLU route buffers are invalid".into());
    }
    // SAFETY: all three equally sized buffers are validated; every thread
    // writes one output. This is the official FP32 SwiGLU/route then BF16 cast boundary.
    unsafe {
        module.swiglu_route_bf16(
            stream,
            LaunchConfig::for_num_elems(u32::try_from(gate.len())?),
            gate,
            up,
            output,
            route_weight,
        )
    }?;
    Ok(())
}

pub(crate) fn launch_v4_bf16_to_f32(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    input: &DeviceBuffer<u16>,
    output: &mut DeviceBuffer<f32>,
) -> Result<(), Box<dyn Error>> {
    if input.len() == 0 || output.len() != input.len() {
        return Err("V4 routed contribution output buffers disagree".into());
    }
    // SAFETY: validation proves one output slot per BF16 input element.
    unsafe {
        module.bf16_to_f32_rows(
            stream,
            LaunchConfig::for_num_elems(u32::try_from(input.len())?),
            input,
            output,
        )
    }?;
    Ok(())
}

pub(crate) fn launch_e4m3fn_to_bf16(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    input: &DeviceBuffer<u8>,
    output: &mut DeviceBuffer<u16>,
) -> Result<(), Box<dyn Error>> {
    if output.len() < input.len() {
        return Err("E4M3FN decode scratch is too small".into());
    }
    unsafe {
        module.e4m3fn_to_bf16(
            stream,
            LaunchConfig::for_num_elems(u32::try_from(input.len())?),
            input,
            output,
        )?;
    }
    Ok(())
}

pub(crate) fn assert_v4_expert_close(actual: &[f32], expected: &[f32]) {
    let maximum_error = v4_expert_maximum_error(actual, expected);
    println!("V4 expert-0 maximum absolute CPU/GPU error: {maximum_error:.6e}");
}

pub(crate) fn v4_expert_maximum_error(actual: &[f32], expected: &[f32]) -> f32 {
    assert_eq!(actual.len(), expected.len(), "V4 expert output length");
    let mut maximum_error = 0.0_f32;
    for (actual, expected) in actual.iter().zip(expected) {
        maximum_error = maximum_error.max((actual - expected).abs());
        let tolerance = 3e-3 * expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= tolerance,
            "V4 expert output differs: expected {expected}, got {actual}, tolerance {tolerance}"
        );
    }
    maximum_error
}
