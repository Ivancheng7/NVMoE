use super::*;

pub(crate) const FLASH_FINAL_HEAD_CAPTURE_SCHEMA: &str = "colibri-flash-0731-final-head-capture-v1";
pub(crate) const FLASH_FINAL_HEAD_RUST_SCHEMA: &str = "colibri-flash-0731-final-head-rust-v1";

pub(crate) fn final_head_u16_bytes(values: &[u16]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

pub(crate) fn final_head_f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_bits().to_le_bytes())
        .collect()
}

pub(crate) fn final_head_u32_bytes(values: &[u32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

pub(crate) fn final_head_capture_config() -> serde_json::Value {
    serde_json::json!({
        "base": std::env::var("COLIBRI_V4_FLASH_BASE").ok(),
        "prefetch": std::env::var("COLIBRI_V4_FLASH_PREFETCH").ok(),
        "stream_read_mode": std::env::var("COLIBRI_V4_STREAM_READ_MODE").ok(),
        "mhc_linear": std::env::var("COLIBRI_V4_FLASH_MHC_LINEAR_ARM").ok(),
        "router_linear": std::env::var("COLIBRI_V4_FLASH_ROUTER_LINEAR_ARM").ok(),
        "router_score": std::env::var("COLIBRI_V4_FLASH_ROUTER_SCORE_ARM").ok(),
        "fp4_accum": std::env::var("COLIBRI_V4_FLASH_FP4_ACCUM_ARM").ok(),
    })
}

pub(crate) fn write_final_head_file(
    path: std::path::PathBuf,
    bytes: &[u8],
) -> Result<(), Box<dyn Error>> {
    let mut output = PendingResultFile::reserve(path)?;
    output.file_mut().write_all(bytes)?;
    output.commit()
}

pub(crate) fn final_head_dump_dir() -> Result<Option<std::path::PathBuf>, Box<dyn Error>> {
    if std::env::var_os("COLIBRI_V4_FLASH_SERVE").as_deref() == Some(std::ffi::OsStr::new("1")) {
        return Ok(None);
    }
    match std::env::var_os("COLIBRI_V4_FLASH_FINAL_HEAD_DUMP_DIR") {
        None => Ok(None),
        Some(path) if path.is_empty() => {
            Err("COLIBRI_V4_FLASH_FINAL_HEAD_DUMP_DIR must not be empty".into())
        }
        Some(path) => Ok(Some(path.into())),
    }
}

pub(crate) fn write_final_head_input(
    directory: &std::path::Path,
    hidden: &[u16],
) -> Result<(), Box<dyn Error>> {
    if hidden.len() != v4_state::HC * v4_spec::FLASH_0731.hidden {
        return Err("Flash final-head capture hidden shape mismatch".into());
    }
    let bytes = final_head_u16_bytes(hidden);
    let hidden_path = directory.join("final_hidden_bf16.bin");
    write_final_head_file(hidden_path.clone(), &bytes)?;
    let binary = std::env::current_exe()?.canonicalize()?;
    let mut manifest = PendingResultFile::reserve(directory.join("manifest.json"))?;
    serde_json::to_writer_pretty(
        manifest.file_mut(),
        &serde_json::json!({
            "schema": FLASH_FINAL_HEAD_CAPTURE_SCHEMA,
            "model_revision": v4_spec::FLASH_0731.revision,
            "shape": [1, v4_state::HC, v4_spec::FLASH_0731.hidden],
            "dtype": "BF16",
            "byte_order": "little",
            "hidden": {
                "file": "final_hidden_bf16.bin", "bytes": bytes.len(),
                "fnv1a64": format!("{:016x}", fnv1a_u16(hidden)),
                "sha256": sha256_file(&hidden_path)?,
            },
            "binary": {"path": binary, "sha256": sha256_file(&std::env::current_exe()?)?},
            "config": final_head_capture_config(),
        }),
    )?;
    manifest.file_mut().write_all(b"\n")?;
    manifest.commit()
}

pub(crate) fn write_final_head_output(
    directory: &std::path::Path,
    hidden: &[u16],
    cpu: &v4_state::FinalHeadCpuRun,
    mode: &str,
) -> Result<(), Box<dyn Error>> {
    let root = directory.join("rust-head");
    let collapsed = final_head_u16_bytes(&cpu.output.collapsed_bf16);
    let normalized = final_head_u16_bytes(&cpu.output.normalized_bf16);
    let logits = final_head_f32_bytes(&cpu.output.logits_f32);
    let collapsed_path = root.join("collapsed_bf16.bin");
    let normalized_path = root.join("normalized_bf16.bin");
    let logits_path = root.join("logits_f32.bin");
    write_final_head_file(collapsed_path.clone(), &collapsed)?;
    write_final_head_file(normalized_path.clone(), &normalized)?;
    write_final_head_file(logits_path.clone(), &logits)?;
    let mut manifest = PendingResultFile::reserve(root.join("manifest.json"))?;
    serde_json::to_writer_pretty(
        manifest.file_mut(),
        &serde_json::json!({
            "schema": FLASH_FINAL_HEAD_RUST_SCHEMA, "mode": mode,
            "model_revision": v4_spec::FLASH_0731.revision,
            "input_hidden_fnv1a64": format!("{:016x}", fnv1a_u16(hidden)),
            "collapsed_bf16": {"file":"collapsed_bf16.bin","shape":[1,v4_spec::FLASH_0731.hidden],"fnv1a64":format!("{:016x}",fnv1a_u16(&cpu.output.collapsed_bf16)),"sha256":sha256_file(&collapsed_path)?},
            "normalized_bf16": {"file":"normalized_bf16.bin","shape":[1,v4_spec::FLASH_0731.hidden],"fnv1a64":format!("{:016x}",fnv1a_u16(&cpu.output.normalized_bf16)),"sha256":sha256_file(&normalized_path)?},
            "logits_f32": {"file":"logits_f32.bin","shape":[1,v4_spec::FLASH_0731.vocab],"fnv1a64":format!("{:016x}",fnv1a_f32(&cpu.output.logits_f32)),"sha256":sha256_file(&logits_path)?},
            "argmax": cpu.output.argmax[0],
        }),
    )?;
    manifest.file_mut().write_all(b"\n")?;
    manifest.commit()
}

pub(crate) fn final_head_trace_float(value: f32) -> serde_json::Value {
    serde_json::json!({"bits": format!("0x{:08x}", value.to_bits()), "value": value})
}

/// A head-only diagnostic sidecar.  It deliberately does not alter the capture
/// schema or the production final-head call: fields marked `scalar_replay` are
/// recomputed from the locked input with the same scalar Rust formula solely to
/// make a reduction boundary inspectable.
pub(crate) fn write_final_head_replay_trace(
    directory: &std::path::Path,
    hidden: &[u16],
    weights: &v4_state::FinalHeadWeights,
    cpu: &v4_state::FinalHeadCpuRun,
) -> Result<(), Box<dyn Error>> {
    pub(crate) const COLLAPSE_COLUMN: usize = 3661;
    let width = v4_state::HC * v4_spec::FLASH_0731.hidden;
    if hidden.len() != width || COLLAPSE_COLUMN >= v4_spec::FLASH_0731.hidden {
        return Err("Flash final-head replay trace shape mismatch".into());
    }
    // This invokes the production gate once and makes the resulting pre4 bits
    // explicit.  The individual dot/partial fields below cannot be recovered
    // from that API and are therefore labelled scalar_replay.
    let pre_actual = v4_state::final_hc_gate_cpu(
        hidden,
        1,
        v4_state::HC,
        v4_spec::FLASH_0731.hidden,
        &weights.hc_fn_f32,
        weights.hc_scale_f32,
        &weights.hc_base_f32,
    )?;
    let square_sum = hidden
        .iter()
        .map(|&bits| {
            let value = v4_router::bf16_to_f32(bits);
            value * value
        })
        .sum::<f32>();
    let mean_plus_eps = square_sum / width as f32 + 1e-6_f32;
    let rsqrt = 1.0_f32 / mean_plus_eps.sqrt();
    let mut gates = Vec::with_capacity(v4_state::HC);
    for copy in 0..v4_state::HC {
        let mut dot = 0.0_f32;
        for column in 0..width {
            dot +=
                v4_router::bf16_to_f32(hidden[column]) * weights.hc_fn_f32[copy * width + column];
        }
        let scaled = dot * rsqrt;
        let affine = scaled * weights.hc_scale_f32 + weights.hc_base_f32[copy];
        let pre_replay = 1.0_f32 / (1.0_f32 + (-affine).exp()) + 1e-6_f32;
        gates.push(serde_json::json!({
            "copy": copy,
            "raw_dot_scalar_replay": final_head_trace_float(dot),
            "scaled_scalar_replay": final_head_trace_float(scaled),
            "affine_scalar_replay": final_head_trace_float(affine),
            "pre_scalar_replay": final_head_trace_float(pre_replay),
            "pre_actual": final_head_trace_float(pre_actual[copy]),
        }));
    }
    let mut partial = 0.0_f32;
    let mut collapse_terms = Vec::with_capacity(v4_state::HC);
    for copy in 0..v4_state::HC {
        let input_bits = hidden[copy * v4_spec::FLASH_0731.hidden + COLLAPSE_COLUMN];
        let input = v4_router::bf16_to_f32(input_bits);
        let product = pre_actual[copy] * input;
        partial += product;
        collapse_terms.push(serde_json::json!({
            "copy": copy,
            "input_bf16_bits": format!("0x{input_bits:04x}"),
            "input_f32": final_head_trace_float(input),
            "pre_actual": final_head_trace_float(pre_actual[copy]),
            "product_scalar_replay": final_head_trace_float(product),
            "partial_scalar_replay": final_head_trace_float(partial),
        }));
    }
    let trace = serde_json::json!({
        "schema": "colibri-flash-0731-final-head-replay-trace-v1",
        "scope": "head_only",
        "gate": {
            "square_sum_scalar_replay": final_head_trace_float(square_sum),
            "mean_plus_eps_scalar_replay": final_head_trace_float(mean_plus_eps),
            "rsqrt_scalar_replay": final_head_trace_float(rsqrt),
            "pre4_actual": pre_actual.iter().map(|&value| final_head_trace_float(value)).collect::<Vec<_>>(),
            "copies": gates,
        },
        "collapse_index_3661": {
            "terms": collapse_terms,
            "collapsed_bf16_actual": format!("0x{:04x}", cpu.output.collapsed_bf16[COLLAPSE_COLUMN]),
            "partials": "scalar_replay; production final_hc_head_cpu does not expose internal partials",
        },
    });
    let mut output =
        PendingResultFile::reserve(directory.join("rust-head").join("head-trace.json"))?;
    serde_json::to_writer_pretty(output.file_mut(), &trace)?;
    output.file_mut().write_all(b"\n")?;
    output.commit()
}

pub(crate) fn flash_final_head_candidate_sm120() -> Result<String, Box<dyn Error>> {
    let output = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader,nounits"])
        .output()?;
    if !output.status.success() {
        return Err("Flash final-head GPU candidate requires nvidia-smi compute_cap".into());
    }
    let capability = std::str::from_utf8(&output.stdout)?.trim();
    if capability != "12.0" {
        return Err(
            format!("Flash final-head GPU candidate requires SM 12.0, got {capability}").into(),
        );
    }
    Ok(capability.into())
}

pub(crate) fn validate_flash_final_head_candidate_config(
    manifest: &serde_json::Value,
) -> Result<(), Box<dyn Error>> {
    let config = manifest["config"]
        .as_object()
        .ok_or("Flash final-head candidate input lacks config")?;
    for (name, expected) in [
        ("base", "1"),
        ("prefetch", "1"),
        ("stream_read_mode", "buffered"),
        ("mhc_linear", "cublas"),
        ("router_linear", "cublas"),
        ("router_score", "libdevice"),
        ("fp4_accum", "fp8_mma"),
    ] {
        if config.get(name).and_then(serde_json::Value::as_str) != Some(expected) {
            return Err(
                format!("Flash final-head candidate requires captured {name}={expected}").into(),
            );
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum FlashFinalHeadArm {
    Cpu,
    GpuExact,
}

pub(crate) fn flash_final_head_arm_for(
    spec: &v4_spec::V4Spec,
    value: Option<&std::ffi::OsStr>,
) -> Result<FlashFinalHeadArm, Box<dyn Error>> {
    if spec != &v4_spec::FLASH_0731 {
        if value.is_some() {
            return Err("COLIBRI_V4_FLASH_FINAL_HEAD_ARM is Flash-0731 base only".into());
        }
        return Ok(FlashFinalHeadArm::Cpu);
    }
    match value {
        None => Ok(FlashFinalHeadArm::Cpu),
        Some(value) if value == "cpu" => Ok(FlashFinalHeadArm::Cpu),
        Some(value) if value == "gpu_exact" => Ok(FlashFinalHeadArm::GpuExact),
        Some(_) => Err("COLIBRI_V4_FLASH_FINAL_HEAD_ARM must be cpu or gpu_exact".into()),
    }
}

pub(crate) fn flash_final_head_arm(
    spec: &v4_spec::V4Spec,
) -> Result<FlashFinalHeadArm, Box<dyn Error>> {
    flash_final_head_arm_for(
        spec,
        std::env::var_os("COLIBRI_V4_FLASH_FINAL_HEAD_ARM").as_deref(),
    )
}

pub(crate) fn flash_final_head_gpu_production_dir() -> Result<std::path::PathBuf, Box<dyn Error>> {
    let result = std::env::var_os("COLIBRI_V4_RESULT_JSONL")
        .map(std::path::PathBuf::from)
        .ok_or("set COLIBRI_V4_RESULT_JSONL for Flash GPU exact final head")?;
    let parent = result
        .parent()
        .ok_or("Flash GPU exact result path has no parent")?;
    Ok(parent.join("flash-final-head-gpu-exact"))
}

pub(crate) const FLASH_FINAL_HEAD_CANDIDATE_HEADROOM_BYTES: usize = 512 * 1024 * 1024;

pub(crate) fn final_head_candidate_observe_device_free(
    minimum: &mut usize,
) -> Result<(), Box<dyn Error>> {
    *minimum = (*minimum).min(device_available_bytes()?);
    Ok(())
}

pub(crate) fn final_head_candidate_memory_gate(
    stage: &str,
    required_device_bytes: usize,
    required_host_bytes: usize,
) -> Result<(usize, usize), Box<dyn Error>> {
    let free_device = device_available_bytes()?;
    let free_host = host_available_bytes()?;
    let device_required = required_device_bytes
        .checked_add(FLASH_FINAL_HEAD_CANDIDATE_HEADROOM_BYTES)
        .ok_or("Flash final-head candidate device capacity overflow")?;
    if free_device < device_required || free_host < required_host_bytes {
        return Err(format!(
            "Flash final-head candidate {stage} capacity requires device={device_required} (payload={required_device_bytes} headroom={FLASH_FINAL_HEAD_CANDIDATE_HEADROOM_BYTES}) and host={required_host_bytes}, available device={free_device} host={free_host}"
        )
        .into());
    }
    Ok((free_device, free_host))
}

pub(crate) fn final_head_candidate_official_tensor(
    directory: &std::path::Path,
    manifest: &serde_json::Value,
    tensor: &str,
    filename: &str,
    dtype: &str,
    expected_bytes: usize,
) -> Result<Vec<u8>, Box<dyn Error>> {
    let entry = &manifest["tensors"][tensor];
    if entry["file"] != filename || entry["dtype"] != dtype {
        return Err(format!("Flash final-head official {tensor} manifest mismatch").into());
    }
    let bytes = std::fs::read(directory.join(filename))?;
    if bytes.len() != expected_bytes
        || sha256_bytes(&bytes)? != entry["sha256"].as_str().unwrap_or_default()
    {
        return Err(format!("Flash final-head official {tensor} bytes mismatch").into());
    }
    Ok(bytes)
}

pub(crate) fn final_head_candidate_require_exact(
    stage: &str,
    actual: &[u8],
    expected: &[u8],
    element_bytes: usize,
) -> Result<(), Box<dyn Error>> {
    if element_bytes == 0 || actual.len() != expected.len() || actual.len() % element_bytes != 0 {
        return Err(format!("Flash final-head {stage} comparison shape mismatch").into());
    }
    if let Some(byte) = actual
        .iter()
        .zip(expected)
        .position(|(left, right)| left != right)
    {
        let element = byte / element_bytes;
        let start = element * element_bytes;
        return Err(format!(
            "Flash final-head {stage} first divergence element={element} byte={byte} actual={:02x?} expected={:02x?}",
            &actual[start..start + element_bytes],
            &expected[start..start + element_bytes],
        )
        .into());
    }
    Ok(())
}

pub(crate) struct FlashFinalHeadGpuExactRun {
    pub(crate) output: v4_state::FinalHeadOutput,
    pub(crate) source_bytes: usize,
    pub(crate) timing: serde_json::Value,
    pub(crate) memory: serde_json::Value,
}

pub(crate) fn step_flash_final_head_gpu_exact(
    output_dir: &std::path::Path,
    verification_manifest: Option<&serde_json::Value>,
    hidden: &[u16],
    weights: &v4_state::FinalHeadWeights,
) -> Result<FlashFinalHeadGpuExactRun, Box<dyn Error>> {
    if let Some(manifest) = verification_manifest {
        validate_flash_final_head_candidate_config(manifest)?;
    }
    let sm = flash_final_head_candidate_sm120()?;
    if hidden.len() != v4_state::HC * v4_spec::FLASH_0731.hidden
        || weights.hc_fn_f32.len() != v4_state::HC * hidden.len()
        || weights.hc_base_f32.len() != v4_state::HC
    {
        return Err("Flash final-head GPU candidate locked geometry mismatch".into());
    }
    let official = if verification_manifest.is_some() {
        let directory = std::env::var_os("COLIBRI_V4_FLASH_FINAL_HEAD_OFFICIAL_DIR")
            .map(std::path::PathBuf::from)
            .ok_or("set COLIBRI_V4_FLASH_FINAL_HEAD_OFFICIAL_DIR for GPU exact comparison")?;
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(directory.join("manifest.json"))?)?;
        if manifest["schema"] != "colibri-flash-0731-final-head-official-v1"
            || manifest["model_revision"] != v4_spec::FLASH_0731.revision
            || manifest["input_hidden_fnv1a64"] != format!("{:016x}", fnv1a_u16(hidden))
        {
            return Err("Flash final-head official comparison manifest mismatch".into());
        }
        Some((directory, manifest))
    } else {
        None
    };
    let hidden_bytes = hidden
        .len()
        .checked_mul(std::mem::size_of::<u16>())
        .ok_or("Flash final-head hidden byte count overflow")?;
    let gate_weight_bytes = weights
        .hc_fn_f32
        .len()
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or("Flash final-head gate byte count overflow")?;
    let base_device_payload = hidden_bytes
        .checked_add(gate_weight_bytes)
        .and_then(|bytes| bytes.checked_add(hidden.len() * std::mem::size_of::<f32>()))
        .and_then(|bytes| bytes.checked_add(hidden.len() * std::mem::size_of::<f32>()))
        .and_then(|bytes| bytes.checked_add(cublas_exact::WORKSPACE_BYTES))
        .and_then(|bytes| bytes.checked_add(64 * 1024))
        .ok_or("Flash final-head base device capacity overflow")?;
    let context = CudaContext::new(0)?;
    let (device_free_before, host_free_before) =
        final_head_candidate_memory_gate("gate", base_device_payload, 0)?;
    let mut device_free_min = device_free_before;
    let total_started = std::time::Instant::now();
    let stream = context.default_stream();
    let module = kernels::load(&context)?;
    let gate_h2d_started = std::time::Instant::now();
    let input = DeviceBuffer::from_host(&stream, hidden)?;
    let hc_fn = DeviceBuffer::from_host(&stream, &weights.hc_fn_f32)?;
    let hc_scale = DeviceBuffer::from_host(&stream, &[weights.hc_scale_f32])?;
    let hc_base = DeviceBuffer::from_host(&stream, &weights.hc_base_f32)?;
    let mut input_f32 = DeviceBuffer::<f32>::zeroed(&stream, hidden.len())?;
    let mut squares = DeviceBuffer::<f32>::zeroed(&stream, hidden.len())?;
    let raw_dot = DeviceBuffer::<f32>::zeroed(&stream, v4_state::HC)?;
    let mut rsqrt = DeviceBuffer::<f32>::zeroed(&stream, 1)?;
    let mut rms_debug = DeviceBuffer::<u32>::zeroed(&stream, FLASH_MHC_RSQRT_DEBUG_FIELDS)?;
    let mut pre = DeviceBuffer::<f32>::zeroed(&stream, v4_state::HC)?;
    let mut collapsed = DeviceBuffer::<u16>::zeroed(&stream, v4_spec::FLASH_0731.hidden)?;
    let cublas = cublas_exact::CublasExact::load()?;
    let workspace = DeviceBuffer::<u8>::zeroed(&stream, cublas_exact::WORKSPACE_BYTES)?;
    stream.synchronize()?;
    final_head_candidate_observe_device_free(&mut device_free_min)?;
    let gate_h2d_ms = gate_h2d_started.elapsed().as_secs_f64() * 1_000.0;
    let gate_kernel_started = std::time::Instant::now();
    launch_v4_bf16_to_f32(&module, &stream, &input, &mut input_f32)?;
    let device_const = |pointer: u64| pointer as usize as *const std::ffi::c_void;
    let device_mut = |pointer: u64| pointer as usize as *mut std::ffi::c_void;
    cublas.sgemm_mn1(
        stream.cu_stream().cast::<std::ffi::c_void>(),
        device_mut(workspace.cu_deviceptr()),
        workspace.num_bytes(),
        device_const(hc_fn.cu_deviceptr()),
        device_const(input_f32.cu_deviceptr()),
        device_mut(raw_dot.cu_deviceptr()),
        v4_state::HC,
        hidden.len(),
    )?;
    // SAFETY: fixed 16K gate geometry, locked BF16 input, and disjoint output
    // lanes are established above.  This path is replay-only.
    unsafe {
        module.rmsnorm_flash_square_bf16(
            &stream,
            LaunchConfig::for_num_elems(u32::try_from(hidden.len())?),
            &input,
            &mut squares,
        )?;
        module.mhc_flash_reduce_rsqrt_16384(
            &stream,
            LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (512, 1, 1),
                shared_mem_bytes: 0,
            },
            &squares,
            &mut rsqrt,
            &mut rms_debug,
        )?;
        module.flash_final_head_sigmoid_collapse4(
            &stream,
            LaunchConfig::for_num_elems(u32::try_from(v4_spec::FLASH_0731.hidden)?),
            &input,
            &raw_dot,
            &rsqrt,
            &hc_scale,
            &hc_base,
            &mut pre,
            &mut collapsed,
        )?;
    }
    stream.synchronize()?;
    final_head_candidate_observe_device_free(&mut device_free_min)?;
    let gate_kernel_ms = gate_kernel_started.elapsed().as_secs_f64() * 1_000.0;
    let gate_d2h_started = std::time::Instant::now();
    let raw_dot = raw_dot.to_host_vec(&stream)?;
    let rsqrt = rsqrt.to_host_vec(&stream)?;
    let rms_debug = rms_debug.to_host_vec(&stream)?;
    let pre = pre.to_host_vec(&stream)?;
    let collapsed = collapsed.to_host_vec(&stream)?;
    let gate_d2h_ms = gate_d2h_started.elapsed().as_secs_f64() * 1_000.0;
    if rsqrt.len() != 1 || rms_debug.len() != FLASH_MHC_RSQRT_DEBUG_FIELDS {
        return Err("Flash final-head GPU candidate trace shape mismatch".into());
    }
    let root = output_dir.join("gpu-exact-candidate");
    let raw_dot_path = root.join("raw_dot_f32.bin");
    let pre_path = root.join("pre_f32.bin");
    let collapsed_path = root.join("collapsed_bf16.bin");
    let rms_path = root.join("rms_debug_u32.bin");
    write_final_head_file(raw_dot_path.clone(), &final_head_f32_bytes(&raw_dot))?;
    write_final_head_file(pre_path.clone(), &final_head_f32_bytes(&pre))?;
    write_final_head_file(collapsed_path.clone(), &final_head_u16_bytes(&collapsed))?;
    write_final_head_file(rms_path.clone(), &final_head_u32_bytes(&rms_debug))?;
    if let Some((directory, manifest)) = official.as_ref() {
        let expected = final_head_candidate_official_tensor(
            directory,
            manifest,
            "collapsed_bf16",
            "collapsed_bf16.bin",
            "BF16",
            collapsed.len() * std::mem::size_of::<u16>(),
        )?;
        final_head_candidate_require_exact(
            "collapsed_bf16",
            &final_head_u16_bytes(&collapsed),
            &expected,
            std::mem::size_of::<u16>(),
        )?;
    }

    let norm_h2d_started = std::time::Instant::now();
    let norm_weight = DeviceBuffer::from_host(&stream, &weights.norm_bf16)?;
    let collapsed_device = DeviceBuffer::from_host(&stream, &collapsed)?;
    let mut normalized_device = DeviceBuffer::<u16>::zeroed(&stream, v4_spec::FLASH_0731.hidden)?;
    let mut norm_trace = DeviceBuffer::<f32>::zeroed(&stream, 0)?;
    stream.synchronize()?;
    final_head_candidate_observe_device_free(&mut device_free_min)?;
    let norm_h2d_ms = norm_h2d_started.elapsed().as_secs_f64() * 1_000.0;
    let norm_kernel_started = std::time::Instant::now();
    step_flash_learned_rmsnorm_4096_device_into(
        &module,
        &stream,
        &collapsed_device,
        &norm_weight,
        &mut normalized_device,
        &mut norm_trace,
    )?;
    stream.synchronize()?;
    final_head_candidate_observe_device_free(&mut device_free_min)?;
    let norm_kernel_ms = norm_kernel_started.elapsed().as_secs_f64() * 1_000.0;
    let norm_d2h_started = std::time::Instant::now();
    let normalized = normalized_device.to_host_vec(&stream)?;
    let norm_d2h_ms = norm_d2h_started.elapsed().as_secs_f64() * 1_000.0;
    let normalized_path = root.join("normalized_bf16.bin");
    let normalized_bytes = final_head_u16_bytes(&normalized);
    write_final_head_file(normalized_path.clone(), &normalized_bytes)?;
    if let Some((directory, manifest)) = official.as_ref() {
        let expected = final_head_candidate_official_tensor(
            directory,
            manifest,
            "normalized_bf16",
            "normalized_bf16.bin",
            "BF16",
            normalized_bytes.len(),
        )?;
        final_head_candidate_require_exact(
            "normalized_bf16",
            &normalized_bytes,
            &expected,
            std::mem::size_of::<u16>(),
        )?;
    }

    let head_elements = v4_spec::FLASH_0731
        .vocab
        .checked_mul(v4_spec::FLASH_0731.hidden)
        .ok_or("Flash final-head full head element count overflow")?;
    let head_source_bytes = head_elements
        .checked_mul(std::mem::size_of::<u16>())
        .ok_or("Flash final-head full head source byte count overflow")?;
    let head_f32_bytes = head_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or("Flash final-head full head FP32 byte count overflow")?;
    let logits_bytes = v4_spec::FLASH_0731
        .vocab
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or("Flash final-head logits byte count overflow")?;
    let full_device_payload = base_device_payload
        .checked_add(head_source_bytes)
        .and_then(|bytes| bytes.checked_add(head_f32_bytes))
        .and_then(|bytes| bytes.checked_add(logits_bytes))
        .ok_or("Flash final-head full head device capacity overflow")?;
    final_head_candidate_memory_gate("full_head", full_device_payload, head_source_bytes)?;
    let head_load_started = std::time::Instant::now();
    let mut head_bf16 = vec![0_u16; head_elements];
    // SAFETY: `u16` is two bytes and the vec owns exactly `head_source_bytes`
    // contiguous bytes; the pinned checkpoint is little-endian BF16.
    let head_bytes = unsafe {
        std::slice::from_raw_parts_mut(head_bf16.as_mut_ptr().cast::<u8>(), head_source_bytes)
    };
    weights.read_head_weight_bytes(0, v4_spec::FLASH_0731.vocab, head_bytes)?;
    let head_load_ms = head_load_started.elapsed().as_secs_f64() * 1_000.0;
    let logits_h2d_started = std::time::Instant::now();
    let head_bf16_device = DeviceBuffer::from_host(&stream, &head_bf16)?;
    drop(head_bf16);
    let mut head_device = DeviceBuffer::<f32>::zeroed(&stream, head_elements)?;
    let logits_device = DeviceBuffer::<f32>::zeroed(&stream, v4_spec::FLASH_0731.vocab)?;
    let mut normalized_f32 = DeviceBuffer::<f32>::zeroed(&stream, v4_spec::FLASH_0731.hidden)?;
    launch_v4_bf16_to_f32(&module, &stream, &head_bf16_device, &mut head_device)?;
    launch_v4_bf16_to_f32(&module, &stream, &normalized_device, &mut normalized_f32)?;
    stream.synchronize()?;
    drop(head_bf16_device);
    final_head_candidate_observe_device_free(&mut device_free_min)?;
    let logits_h2d_ms = logits_h2d_started.elapsed().as_secs_f64() * 1_000.0;
    let logits_kernel_started = std::time::Instant::now();
    cublas.sgemm_mn1(
        stream.cu_stream().cast::<std::ffi::c_void>(),
        device_mut(workspace.cu_deviceptr()),
        workspace.num_bytes(),
        device_const(head_device.cu_deviceptr()),
        device_const(normalized_f32.cu_deviceptr()),
        device_mut(logits_device.cu_deviceptr()),
        v4_spec::FLASH_0731.vocab,
        v4_spec::FLASH_0731.hidden,
    )?;
    stream.synchronize()?;
    final_head_candidate_observe_device_free(&mut device_free_min)?;
    let logits_kernel_ms = logits_kernel_started.elapsed().as_secs_f64() * 1_000.0;
    let logits_d2h_started = std::time::Instant::now();
    let logits = logits_device.to_host_vec(&stream)?;
    let logits_d2h_ms = logits_d2h_started.elapsed().as_secs_f64() * 1_000.0;
    let logits_path = root.join("logits_f32.bin");
    let logits_bytes_raw = final_head_f32_bytes(&logits);
    write_final_head_file(logits_path.clone(), &logits_bytes_raw)?;
    if let Some((directory, manifest)) = official.as_ref() {
        let expected = final_head_candidate_official_tensor(
            directory,
            manifest,
            "logits_f32",
            "logits_f32.bin",
            "F32",
            logits_bytes_raw.len(),
        )?;
        final_head_candidate_require_exact(
            "logits_f32",
            &logits_bytes_raw,
            &expected,
            std::mem::size_of::<f32>(),
        )?;
    }
    let (&first_logit, tail) = logits
        .split_first()
        .ok_or("Flash final-head logits are empty")?;
    let mut argmax = 0_usize;
    let mut max_logit = first_logit;
    for (offset, &logit) in tail.iter().enumerate() {
        if logit > max_logit {
            argmax = offset + 1;
            max_logit = logit;
        }
    }
    if let Some((_, manifest)) = official.as_ref() {
        if manifest["argmax"].as_u64() != Some(u64::try_from(argmax)?) {
            return Err(format!("Flash final-head argmax mismatch actual={argmax}").into());
        }
    }
    let timing = serde_json::json!({"gate_h2d":gate_h2d_ms, "gate_kernel":gate_kernel_ms, "gate_d2h":gate_d2h_ms, "norm_h2d":norm_h2d_ms, "norm_kernel":norm_kernel_ms, "norm_d2h":norm_d2h_ms, "head_load_and_promote":head_load_ms, "logits_h2d":logits_h2d_ms, "logits_kernel":logits_kernel_ms, "logits_d2h":logits_d2h_ms, "total":total_started.elapsed().as_secs_f64()*1_000.0});
    let memory = serde_json::json!({"device_free_before":device_free_before, "device_free_min":device_free_min, "device_peak_observed":device_free_before.saturating_sub(device_free_min), "host_free_before":host_free_before, "full_head_source":head_source_bytes, "full_head_fp32":head_f32_bytes, "headroom":FLASH_FINAL_HEAD_CANDIDATE_HEADROOM_BYTES});
    let provenance = cublas.provenance();
    let mut output = PendingResultFile::reserve(root.join("manifest.json"))?;
    serde_json::to_writer_pretty(
        output.file_mut(),
        &serde_json::json!({
            "schema": "colibri-flash-0731-final-head-gpu-exact-v1",
            "scope": if official.is_some() { "head_only_replay_with_official_comparison" } else { "runtime_gpu_exact_without_per_input_oracle" }, "arm": "gpu_exact",
            "input_hidden_fnv1a64": format!("{:016x}", fnv1a_u16(hidden)),
            "sm": sm,
            "binary": {"path": std::env::current_exe()?.canonicalize()?, "sha256": sha256_file(&std::env::current_exe()?)?},
            "cublas": {"sha256": provenance.actual_sha256, "version": provenance.version, "math_mode": "CUBLAS_DEFAULT_MATH", "tf32": "off", "gate":{"m":4, "n":1, "k":16384}, "head":{"m":129280, "n":1, "k":4096}},
            "config": verification_manifest.map(|manifest| manifest["config"].clone()).unwrap_or(serde_json::Value::Null),
            "official_comparison": official.as_ref().map(|(directory, _)| serde_json::json!({"directory": directory, "collapsed_bf16":"exact", "normalized_bf16":"exact", "logits_f32":"exact", "argmax":argmax})).unwrap_or_else(|| serde_json::json!({"status":"not_requested","reason":"runtime input has no independent official oracle"})),
            "timing_ms": timing.clone(),
            "memory_bytes": memory.clone(),
            "rms": {"debug_fields": ["square_sum", "mean_plus_eps", "newton_rsqrt", "libdevice_sqrt_rsqrt", "libdevice_rsqrt"], "debug_bits": rms_debug.iter().map(|value| format!("0x{value:08x}")).collect::<Vec<_>>(), "rsqrt_bits": format!("0x{:08x}", rsqrt[0].to_bits())},
            "raw_dot_f32": {"file":"raw_dot_f32.bin", "bits": raw_dot.iter().map(|value| format!("0x{:08x}",value.to_bits())).collect::<Vec<_>>(), "sha256":sha256_file(&raw_dot_path)?},
            "pre_f32": {"file":"pre_f32.bin", "bits": pre.iter().map(|value| format!("0x{:08x}",value.to_bits())).collect::<Vec<_>>(), "sha256":sha256_file(&pre_path)?},
            "collapsed_bf16": {"file":"collapsed_bf16.bin", "fnv1a64":format!("{:016x}",fnv1a_u16(&collapsed)), "sha256":sha256_file(&collapsed_path)?},
            "normalized_bf16": {"file":"normalized_bf16.bin", "fnv1a64":format!("{:016x}",fnv1a_u16(&normalized)), "sha256":sha256_file(&normalized_path)?},
            "logits_f32": {"file":"logits_f32.bin", "fnv1a64":format!("{:016x}",fnv1a_f32(&logits)), "sha256":sha256_file(&logits_path)?},
            "rms_debug_u32": {"file":"rms_debug_u32.bin", "sha256":sha256_file(&rms_path)?},
        }),
    )?;
    output.file_mut().write_all(b"\n")?;
    output.commit()?;
    Ok(FlashFinalHeadGpuExactRun {
        output: v4_state::FinalHeadOutput {
            collapsed_bf16: collapsed,
            normalized_bf16: normalized,
            logits_f32: logits,
            argmax: vec![argmax],
        },
        source_bytes: head_source_bytes,
        timing,
        memory,
    })
}

// DSpark validates up to six base targets in one round.  The exact output
// head is immutable, so keep its BF16-to-FP32 promotion and cuBLAS workspace
// for that round's lifetime instead of rebuilding a multi-GiB context per
// target.  Per-input gate/norm/logit buffers remain call-local.
pub(crate) struct FlashDsparkFinalHeadGpuWorkspace {
    stream: std::sync::Arc<CudaStream>,
    module: kernels::LoadedModule,
    hc_fn: DeviceBuffer<f32>,
    hc_scale: DeviceBuffer<f32>,
    hc_base: DeviceBuffer<f32>,
    norm: DeviceBuffer<u16>,
    head: DeviceBuffer<f32>,
    cublas: cublas_exact::CublasExact,
    cublas_workspace: DeviceBuffer<u8>,
    head_source_bytes: usize,
    head_f32_bytes: usize,
    device_free_after_load: usize,
    _context: std::sync::Arc<CudaContext>,
}

impl FlashDsparkFinalHeadGpuWorkspace {
    pub(crate) fn new(weights: &v4_state::FinalHeadWeights) -> Result<Self, Box<dyn Error>> {
        let head_elements = v4_spec::FLASH_0731
            .vocab
            .checked_mul(v4_spec::FLASH_0731.hidden)
            .ok_or("Flash DSpark final-head element count overflow")?;
        let head_source_bytes = head_elements
            .checked_mul(std::mem::size_of::<u16>())
            .ok_or("Flash DSpark final-head source byte count overflow")?;
        let head_f32_bytes = head_elements
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or("Flash DSpark final-head FP32 byte count overflow")?;
        let context = CudaContext::new(0)?;
        let stream = context.default_stream();
        let module = kernels::load(&context)?;
        let gate_bytes = weights
            .hc_fn_f32
            .len()
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or("Flash DSpark gate byte count overflow")?;
        let payload = head_source_bytes
            .checked_add(head_f32_bytes)
            .and_then(|bytes| bytes.checked_add(gate_bytes))
            .and_then(|bytes| bytes.checked_add(cublas_exact::WORKSPACE_BYTES))
            .and_then(|bytes| bytes.checked_add(64 * 1024))
            .ok_or("Flash DSpark final-head device capacity overflow")?;
        final_head_candidate_memory_gate("dspark_workspace", payload, head_source_bytes)?;
        let mut head_bf16 = vec![0_u16; head_elements];
        // SAFETY: the owning u16 vector has exactly the checkpoint's BF16 byte
        // extent and the source checkpoint is little-endian.
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(head_bf16.as_mut_ptr().cast::<u8>(), head_source_bytes)
        };
        weights.read_head_weight_bytes(0, v4_spec::FLASH_0731.vocab, bytes)?;
        let head_bf16_device = DeviceBuffer::from_host(&stream, &head_bf16)?;
        drop(head_bf16);
        let mut head = DeviceBuffer::<f32>::zeroed(&stream, head_elements)?;
        let hc_fn = DeviceBuffer::from_host(&stream, &weights.hc_fn_f32)?;
        let hc_scale = DeviceBuffer::from_host(&stream, &[weights.hc_scale_f32])?;
        let hc_base = DeviceBuffer::from_host(&stream, &weights.hc_base_f32)?;
        let norm = DeviceBuffer::from_host(&stream, &weights.norm_bf16)?;
        let cublas = cublas_exact::CublasExact::load()?;
        let cublas_workspace = DeviceBuffer::<u8>::zeroed(&stream, cublas_exact::WORKSPACE_BYTES)?;
        launch_v4_bf16_to_f32(&module, &stream, &head_bf16_device, &mut head)?;
        stream.synchronize()?;
        drop(head_bf16_device);
        Ok(Self {
            stream,
            module,
            hc_fn,
            hc_scale,
            hc_base,
            norm,
            head,
            cublas,
            cublas_workspace,
            head_source_bytes,
            head_f32_bytes,
            device_free_after_load: device_available_bytes()?,
            _context: context,
        })
    }

    pub(crate) fn run(
        &self,
        output_dir: &std::path::Path,
        verification_manifest: Option<&serde_json::Value>,
        hidden: &[u16],
    ) -> Result<FlashFinalHeadGpuExactRun, Box<dyn Error>> {
        self.run_inner(Some(output_dir), verification_manifest, hidden)
    }

    // The serving and DSpark paths need the same exact arithmetic, but no
    // per-token artifacts or capture-only provenance contract.
    pub(crate) fn step_runtime(
        &self,
        hidden: &[u16],
    ) -> Result<FlashFinalHeadGpuExactRun, Box<dyn Error>> {
        self.run_inner(None, None, hidden)
    }

    pub(crate) fn run_inner(
        &self,
        output_dir: Option<&std::path::Path>,
        verification_manifest: Option<&serde_json::Value>,
        hidden: &[u16],
    ) -> Result<FlashFinalHeadGpuExactRun, Box<dyn Error>> {
        if let Some(manifest) = verification_manifest {
            validate_flash_final_head_candidate_config(manifest)?;
        }
        if hidden.len() != v4_state::HC * v4_spec::FLASH_0731.hidden {
            return Err("Flash DSpark GPU final-head hidden geometry mismatch".into());
        }
        let official = if verification_manifest.is_some() {
            let directory = std::env::var_os("COLIBRI_V4_FLASH_FINAL_HEAD_OFFICIAL_DIR")
                .map(std::path::PathBuf::from)
                .ok_or("set COLIBRI_V4_FLASH_FINAL_HEAD_OFFICIAL_DIR for GPU exact comparison")?;
            let manifest: serde_json::Value =
                serde_json::from_slice(&std::fs::read(directory.join("manifest.json"))?)?;
            if manifest["schema"] != "colibri-flash-0731-final-head-official-v1"
                || manifest["model_revision"] != v4_spec::FLASH_0731.revision
                || manifest["input_hidden_fnv1a64"] != format!("{:016x}", fnv1a_u16(hidden))
            {
                return Err("Flash DSpark final-head official comparison manifest mismatch".into());
            }
            Some((directory, manifest))
        } else {
            None
        };
        let total_started = std::time::Instant::now();
        let free_before = device_available_bytes()?;
        let mut free_min = free_before;
        let input = DeviceBuffer::from_host(&self.stream, hidden)?;
        let mut input_f32 = DeviceBuffer::<f32>::zeroed(&self.stream, hidden.len())?;
        let mut squares = DeviceBuffer::<f32>::zeroed(&self.stream, hidden.len())?;
        let mut raw_dot = DeviceBuffer::<f32>::zeroed(&self.stream, v4_state::HC)?;
        let mut rsqrt = DeviceBuffer::<f32>::zeroed(&self.stream, 1)?;
        let mut rms_debug =
            DeviceBuffer::<u32>::zeroed(&self.stream, FLASH_MHC_RSQRT_DEBUG_FIELDS)?;
        let mut pre = DeviceBuffer::<f32>::zeroed(&self.stream, v4_state::HC)?;
        let mut collapsed_device =
            DeviceBuffer::<u16>::zeroed(&self.stream, v4_spec::FLASH_0731.hidden)?;
        let gate_started = std::time::Instant::now();
        launch_v4_bf16_to_f32(&self.module, &self.stream, &input, &mut input_f32)?;
        let device_const = |pointer: u64| pointer as usize as *const std::ffi::c_void;
        let device_mut = |pointer: u64| pointer as usize as *mut std::ffi::c_void;
        self.cublas.sgemm_mn1(
            self.stream.cu_stream().cast::<std::ffi::c_void>(),
            device_mut(self.cublas_workspace.cu_deviceptr()),
            self.cublas_workspace.num_bytes(),
            device_const(self.hc_fn.cu_deviceptr()),
            device_const(input_f32.cu_deviceptr()),
            device_mut(raw_dot.cu_deviceptr()),
            v4_state::HC,
            hidden.len(),
        )?;
        // SAFETY: fixed four-lane Flash geometry and disjoint device buffers.
        unsafe {
            self.module.rmsnorm_flash_square_bf16(
                &self.stream,
                LaunchConfig::for_num_elems(u32::try_from(hidden.len())?),
                &input,
                &mut squares,
            )?;
            self.module.mhc_flash_reduce_rsqrt_16384(
                &self.stream,
                LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (512, 1, 1),
                    shared_mem_bytes: 0,
                },
                &squares,
                &mut rsqrt,
                &mut rms_debug,
            )?;
            self.module.flash_final_head_sigmoid_collapse4(
                &self.stream,
                LaunchConfig::for_num_elems(u32::try_from(v4_spec::FLASH_0731.hidden)?),
                &input,
                &raw_dot,
                &rsqrt,
                &self.hc_scale,
                &self.hc_base,
                &mut pre,
                &mut collapsed_device,
            )?;
        }
        self.stream.synchronize()?;
        final_head_candidate_observe_device_free(&mut free_min)?;
        let gate_ms = gate_started.elapsed().as_secs_f64() * 1_000.0;
        let raw_dot = raw_dot.to_host_vec(&self.stream)?;
        let rsqrt = rsqrt.to_host_vec(&self.stream)?;
        let rms_debug = rms_debug.to_host_vec(&self.stream)?;
        let pre = pre.to_host_vec(&self.stream)?;
        let collapsed = collapsed_device.to_host_vec(&self.stream)?;
        let norm_started = std::time::Instant::now();
        let mut normalized_device =
            DeviceBuffer::<u16>::zeroed(&self.stream, v4_spec::FLASH_0731.hidden)?;
        let mut norm_trace = DeviceBuffer::<f32>::zeroed(&self.stream, 0)?;
        step_flash_learned_rmsnorm_4096_device_into(
            &self.module,
            &self.stream,
            &collapsed_device,
            &self.norm,
            &mut normalized_device,
            &mut norm_trace,
        )?;
        self.stream.synchronize()?;
        final_head_candidate_observe_device_free(&mut free_min)?;
        let norm_ms = norm_started.elapsed().as_secs_f64() * 1_000.0;
        let normalized = normalized_device.to_host_vec(&self.stream)?;
        let logits_started = std::time::Instant::now();
        let mut normalized_f32 =
            DeviceBuffer::<f32>::zeroed(&self.stream, v4_spec::FLASH_0731.hidden)?;
        let mut logits_device =
            DeviceBuffer::<f32>::zeroed(&self.stream, v4_spec::FLASH_0731.vocab)?;
        launch_v4_bf16_to_f32(
            &self.module,
            &self.stream,
            &normalized_device,
            &mut normalized_f32,
        )?;
        self.cublas.sgemm_mn1(
            self.stream.cu_stream().cast::<std::ffi::c_void>(),
            device_mut(self.cublas_workspace.cu_deviceptr()),
            self.cublas_workspace.num_bytes(),
            device_const(self.head.cu_deviceptr()),
            device_const(normalized_f32.cu_deviceptr()),
            device_mut(logits_device.cu_deviceptr()),
            v4_spec::FLASH_0731.vocab,
            v4_spec::FLASH_0731.hidden,
        )?;
        self.stream.synchronize()?;
        final_head_candidate_observe_device_free(&mut free_min)?;
        let logits_ms = logits_started.elapsed().as_secs_f64() * 1_000.0;
        let logits = logits_device.to_host_vec(&self.stream)?;
        let (&first_logit, tail) = logits
            .split_first()
            .ok_or("Flash DSpark final-head logits are empty")?;
        let mut argmax = 0_usize;
        let mut max_logit = first_logit;
        for (offset, &logit) in tail.iter().enumerate() {
            if logit > max_logit {
                argmax = offset + 1;
                max_logit = logit;
            }
        }
        if let Some((directory, manifest)) = official.as_ref() {
            let expected = final_head_candidate_official_tensor(
                directory,
                manifest,
                "collapsed_bf16",
                "collapsed_bf16.bin",
                "BF16",
                collapsed.len() * std::mem::size_of::<u16>(),
            )?;
            final_head_candidate_require_exact(
                "collapsed_bf16",
                &final_head_u16_bytes(&collapsed),
                &expected,
                std::mem::size_of::<u16>(),
            )?;
            let expected = final_head_candidate_official_tensor(
                directory,
                manifest,
                "normalized_bf16",
                "normalized_bf16.bin",
                "BF16",
                normalized.len() * std::mem::size_of::<u16>(),
            )?;
            final_head_candidate_require_exact(
                "normalized_bf16",
                &final_head_u16_bytes(&normalized),
                &expected,
                std::mem::size_of::<u16>(),
            )?;
            let expected = final_head_candidate_official_tensor(
                directory,
                manifest,
                "logits_f32",
                "logits_f32.bin",
                "F32",
                logits.len() * std::mem::size_of::<f32>(),
            )?;
            final_head_candidate_require_exact(
                "logits_f32",
                &final_head_f32_bytes(&logits),
                &expected,
                std::mem::size_of::<f32>(),
            )?;
            if manifest["argmax"].as_u64() != Some(u64::try_from(argmax)?) {
                return Err(
                    format!("Flash DSpark final-head argmax mismatch actual={argmax}").into(),
                );
            }
        }
        let timing = serde_json::json!({"gate_kernel":gate_ms, "norm_kernel":norm_ms, "logits_kernel":logits_ms, "total":total_started.elapsed().as_secs_f64()*1_000.0});
        let memory = serde_json::json!({"workspace_device_free_after_load":self.device_free_after_load, "call_device_free_before":free_before, "call_device_free_min":free_min, "call_device_peak_observed":free_before.saturating_sub(free_min), "full_head_source":self.head_source_bytes, "full_head_fp32":self.head_f32_bytes, "lifetime":"dspark_run"});
        if let Some(output_dir) = output_dir {
            let root = output_dir.join("gpu-exact-candidate");
            let raw_dot_path = root.join("raw_dot_f32.bin");
            let pre_path = root.join("pre_f32.bin");
            let collapsed_path = root.join("collapsed_bf16.bin");
            let normalized_path = root.join("normalized_bf16.bin");
            let logits_path = root.join("logits_f32.bin");
            let rms_path = root.join("rms_debug_u32.bin");
            write_final_head_file(raw_dot_path.clone(), &final_head_f32_bytes(&raw_dot))?;
            write_final_head_file(pre_path.clone(), &final_head_f32_bytes(&pre))?;
            write_final_head_file(collapsed_path.clone(), &final_head_u16_bytes(&collapsed))?;
            write_final_head_file(normalized_path.clone(), &final_head_u16_bytes(&normalized))?;
            write_final_head_file(logits_path.clone(), &final_head_f32_bytes(&logits))?;
            write_final_head_file(rms_path.clone(), &final_head_u32_bytes(&rms_debug))?;
            let provenance = self.cublas.provenance();
            let mut output = PendingResultFile::reserve(root.join("manifest.json"))?;
            serde_json::to_writer_pretty(
                output.file_mut(),
                &serde_json::json!({
                    "schema":"colibri-flash-0731-final-head-gpu-exact-v1",
                    "scope":if official.is_some() { "head_only_replay_with_official_comparison" } else { "runtime_gpu_exact_without_per_input_oracle" },
                    "arm":"gpu_exact",
                    "input_hidden_fnv1a64":format!("{:016x}", fnv1a_u16(hidden)),
                    "sm":flash_final_head_candidate_sm120()?,
                    "binary":{"path":std::env::current_exe()?.canonicalize()?, "sha256":sha256_file(&std::env::current_exe()?)?},
                    "cublas":{"sha256":provenance.actual_sha256, "version":provenance.version, "math_mode":"CUBLAS_DEFAULT_MATH", "tf32":"off", "gate":{"m":4,"n":1,"k":16384}, "head":{"m":129280,"n":1,"k":4096}},
                    "config":verification_manifest.map(|manifest| manifest["config"].clone()).unwrap_or(serde_json::Value::Null),
                    "official_comparison":official.as_ref().map(|(directory, _)| serde_json::json!({"directory":directory,"collapsed_bf16":"exact","normalized_bf16":"exact","logits_f32":"exact","argmax":argmax})).unwrap_or_else(|| serde_json::json!({"status":"not_requested","reason":"runtime input has no independent official oracle"})),
                    "timing_ms":timing.clone(), "memory_bytes":memory.clone(),
                    "rms":{"debug_fields":["square_sum","mean_plus_eps","newton_rsqrt","libdevice_sqrt_rsqrt","libdevice_rsqrt"], "debug_bits":rms_debug.iter().map(|value| format!("0x{value:08x}")).collect::<Vec<_>>(), "rsqrt_bits":format!("0x{:08x}", rsqrt[0].to_bits())},
                    "raw_dot_f32":{"file":"raw_dot_f32.bin", "bits":raw_dot.iter().map(|value| format!("0x{:08x}", value.to_bits())).collect::<Vec<_>>(), "sha256":sha256_file(&raw_dot_path)?},
                    "pre_f32":{"file":"pre_f32.bin", "bits":pre.iter().map(|value| format!("0x{:08x}", value.to_bits())).collect::<Vec<_>>(), "sha256":sha256_file(&pre_path)?},
                    "collapsed_bf16":{"file":"collapsed_bf16.bin", "fnv1a64":format!("{:016x}",fnv1a_u16(&collapsed)), "sha256":sha256_file(&collapsed_path)?},
                    "normalized_bf16":{"file":"normalized_bf16.bin", "fnv1a64":format!("{:016x}",fnv1a_u16(&normalized)), "sha256":sha256_file(&normalized_path)?},
                    "logits_f32":{"file":"logits_f32.bin", "fnv1a64":format!("{:016x}",fnv1a_f32(&logits)), "sha256":sha256_file(&logits_path)?},
                    "rms_debug_u32":{"file":"rms_debug_u32.bin", "sha256":sha256_file(&rms_path)?}
                }),
            )?;
            output.file_mut().write_all(b"\n")?;
            output.commit()?;
        }
        Ok(FlashFinalHeadGpuExactRun {
            output: v4_state::FinalHeadOutput {
                collapsed_bf16: collapsed,
                normalized_bf16: normalized,
                logits_f32: logits,
                argmax: vec![argmax],
            },
            source_bytes: self.head_source_bytes,
            timing,
            memory,
        })
    }
}

pub(crate) fn run_flash_final_head_replay() -> Result<(), Box<dyn Error>> {
    let input_dir = std::env::var_os("COLIBRI_V4_FLASH_FINAL_HEAD_DUMP_DIR")
        .map(std::path::PathBuf::from)
        .ok_or("set COLIBRI_V4_FLASH_FINAL_HEAD_DUMP_DIR to a captured input")?;
    let output_dir = std::env::var_os("COLIBRI_V4_FLASH_FINAL_HEAD_REPLAY_DIR")
        .map(std::path::PathBuf::from)
        .ok_or("set COLIBRI_V4_FLASH_FINAL_HEAD_REPLAY_DIR")?;
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(input_dir.join("manifest.json"))?)?;
    if manifest["schema"] != FLASH_FINAL_HEAD_CAPTURE_SCHEMA
        || manifest["model_revision"] != v4_spec::FLASH_0731.revision
        || manifest["shape"] != serde_json::json!([1, v4_state::HC, v4_spec::FLASH_0731.hidden])
    {
        return Err("Flash final-head replay input manifest mismatch".into());
    }
    let bytes = std::fs::read(input_dir.join("final_hidden_bf16.bin"))?;
    if bytes.len() != v4_state::HC * v4_spec::FLASH_0731.hidden * 2
        || sha256_bytes(&bytes)? != manifest["hidden"]["sha256"].as_str().unwrap_or_default()
    {
        return Err("Flash final-head replay input bytes mismatch".into());
    }
    let hidden = bytes
        .chunks_exact(2)
        .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
        .collect::<Vec<_>>();
    if format!("{:016x}", fnv1a_u16(&hidden))
        != manifest["hidden"]["fnv1a64"].as_str().unwrap_or_default()
    {
        return Err("Flash final-head replay input digest mismatch".into());
    }
    let weights = v4_state::load_final_head_for(
        &v4_spec::FLASH_0731,
        &v4_model_dir_for(&v4_spec::FLASH_0731)?,
    )?;
    let replay_arm = match std::env::var_os("COLIBRI_V4_FLASH_FINAL_HEAD_REPLAY_ARM") {
        None => None,
        Some(value) => Some(
            value
                .into_string()
                .map_err(|_| "COLIBRI_V4_FLASH_FINAL_HEAD_REPLAY_ARM must be valid UTF-8")?,
        ),
    };
    match replay_arm.as_deref() {
        None | Some("cpu") => {}
        Some("gpu_exact_candidate") => {
            FlashDsparkFinalHeadGpuWorkspace::new(&weights)?.run(
                &output_dir,
                Some(&manifest),
                &hidden,
            )?;
            println!(
                "{{\"status\":\"PASS\",\"mode\":\"flash-final-head-gpu-exact-candidate\",\"output\":{}}}",
                serde_json::to_string(&output_dir)?
            );
            return Ok(());
        }
        Some(value) => {
            return Err(format!("unknown COLIBRI_V4_FLASH_FINAL_HEAD_REPLAY_ARM={value}").into());
        }
    }
    let cpu = v4_state::final_checkpoint_head_cpu_timed(&weights, &hidden, 1)?;
    write_final_head_output(&output_dir, &hidden, &cpu, "head_only_replay")?;
    write_final_head_replay_trace(&output_dir, &hidden, &weights, &cpu)?;
    println!(
        "{{\"status\":\"PASS\",\"mode\":\"flash-final-head-replay\",\"output\":{}}}",
        serde_json::to_string(&output_dir)?
    );
    Ok(())
}
