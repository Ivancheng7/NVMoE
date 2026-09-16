use super::*;

#[derive(Clone)]
pub(crate) struct L0Persistent {
    pub(crate) kv: v4_state::V4KvCache,
    pub(crate) compressor: v4_state::V4CompressorState,
}

impl L0Persistent {
    pub(crate) fn new() -> Result<Self, Box<dyn Error>> {
        Ok(Self {
            kv: v4_state::V4KvCache::new(),
            compressor: v4_state::V4CompressorState::new_decode(
                v4_state::COMPRESS_RATIO,
                v4_state::Q_HEAD_DIM,
            )?,
        })
    }

    pub(crate) fn bits_equal(&self, other: &Self) -> bool {
        self.kv.bits_equal(&other.kv) && self.compressor.bits_equal(&other.compressor)
    }
}

pub(crate) struct L0Run {
    pub(crate) output: Vec<u16>,
    pub(crate) routes: v4_router::RouterSelection,
    pub(crate) prefixes: Vec<L0Persistent>,
    pub(crate) persistent: L0Persistent,
    pub(crate) stage_hashes: Vec<(&'static str, u64)>,
    pub(crate) expert_union: Vec<u16>,
    pub(crate) expert_bytes: usize,
    pub(crate) expert_alloc_ms: f64,
    pub(crate) expert_read_ms: f64,
    pub(crate) expert_setup_ms: f64,
    pub(crate) expert_setup_samples_ms: Vec<f64>,
    pub(crate) expert_h2d_samples_ms: Vec<f64>,
    pub(crate) expert_kernel_ms: f64,
    pub(crate) dynamic_bytes: usize,
    pub(crate) elapsed_ms: f64,
}

pub(crate) fn fnv1a_u16(values: &[u16]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for value in values {
        for byte in value.to_le_bytes() {
            hash = (hash ^ byte as u64).wrapping_mul(0x100_0000_01b3);
        }
    }
    hash
}

pub(crate) fn fnv1a_bytes(values: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for &byte in values {
        hash = (hash ^ u64::from(byte)).wrapping_mul(0x100_0000_01b3);
    }
    hash
}

pub(crate) fn normalized_config_fingerprint(
    config: serde_json::Value,
) -> Result<serde_json::Value, Box<dyn Error>> {
    if !config.is_object() {
        return Err("normalized relevant configuration must be a JSON object".into());
    }
    let canonical_json = serde_json::to_string(&config)?;
    Ok(serde_json::json!({
        "algorithm": "fnv1a64",
        "digest": format!("{:016x}", fnv1a_bytes(canonical_json.as_bytes())),
        "canonical_json": canonical_json,
        "normalized": config,
    }))
}

pub(crate) fn fnv1a_i32(values: &[i32]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for value in values {
        for byte in value.to_le_bytes() {
            hash = (hash ^ byte as u64).wrapping_mul(0x100_0000_01b3);
        }
    }
    hash
}

pub(crate) fn fnv1a_f32(values: &[f32]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for value in values {
        for byte in value.to_bits().to_le_bytes() {
            hash = (hash ^ byte as u64).wrapping_mul(0x100_0000_01b3);
        }
    }
    hash
}

pub(crate) fn p95_f64(values: &[f64]) -> f64 {
    let mut values = values.to_vec();
    values.sort_by(f64::total_cmp);
    values[(values.len() * 95).div_ceil(100) - 1]
}

pub(crate) fn l0_persistent_digest(state: &L0Persistent) -> u64 {
    let mut hash = fnv1a_u16(state.kv.as_slice());
    for value in state.compressor.kv().iter().chain(state.compressor.score()) {
        for byte in value.to_bits().to_le_bytes() {
            hash = (hash ^ byte as u64).wrapping_mul(0x100_0000_01b3);
        }
    }
    hash
}

/// Actual host-to-device bytes issued by one host-boundary L0 P0 run.
/// CUDA memset/allocations and device-to-host trace copies are not H2D.
pub(crate) fn l0_dynamic_h2d_bytes(rows: usize) -> usize {
    let bf16 = std::mem::size_of::<u16>();
    let f32_bytes = std::mem::size_of::<f32>();
    let i32_bytes = std::mem::size_of::<i32>();
    let i64_bytes = std::mem::size_of::<i64>();
    let h = v4_state::HIDDEN;
    let hc_state = rows * v4_state::HC * h;
    let q = rows * v4_state::Q_HEADS * v4_state::Q_HEAD_DIM;
    let kv = rows * v4_state::Q_HEAD_DIM;
    let q_lora = rows * 1_536;
    let o_lora = rows * v4_state::O_GROUPS * v4_state::O_LORA_RANK;
    let expert_intermediate = rows * v4_expert::V4_EXPERT_INTERMEDIATE;
    let post = rows * v4_state::HC;
    let comb = rows * v4_state::HC * v4_state::HC;
    let freqs = rows * v4_state::Q_ROPE_DIM;
    let cache = v4_state::KV_CACHE_POSITIONS * v4_state::Q_HEAD_DIM;
    let topk = rows * 128;

    hc_state * bf16 // attention mHC
        + rows * h * bf16 // attention RMSNorm
        + rows * h * bf16 // wq_a
        + q_lora * bf16 // q_norm
        + q_lora * bf16 // wq_b
        + q * bf16 // per-head RMSNorm
        + q * bf16 + freqs * f32_bytes // forward q RoPE
        + rows * h * bf16 // wkv
        + kv * bf16 // kv_norm
        + kv * bf16 + freqs * f32_bytes // forward kv RoPE
        + kv * bf16 * 2 // fake-quant source and initialized destination
        + rows * h * bf16 // Compressor input
        + q * bf16 + cache * bf16 + topk * i32_bytes // sparse attention
        + q * bf16 + freqs * f32_bytes // inverse RoPE
        + q * bf16 // wo_a
        + o_lora * bf16 // wo_b
        + rows * h * bf16 + hc_state * bf16 + post * f32_bytes + comb * f32_bytes // attention HC post
        + hc_state * bf16 // FFN mHC
        + rows * h * bf16 // FFN RMSNorm
        + rows * h * bf16 + v4_router::V4_ROUTED_EXPERTS * f32_bytes
        + rows * v4_router::V4_TOPK * i64_bytes // gate projection/select
        + rows * h * bf16 * 2 // shared w1/w3
        + expert_intermediate * bf16 * 3 // shared gate/up uploads and w2 input
        + rows * v4_router::V4_TOPK * h * bf16 // routed expert inputs
        + rows * h * bf16 + hc_state * bf16 + post * f32_bytes + comb * f32_bytes
    // final HC post
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn benchmark_v4_fp8_matvec(
    context: &std::sync::Arc<CudaContext>,
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    input: &DeviceBuffer<u8>,
    input_scales: &DeviceBuffer<u8>,
    weight: &DeviceBuffer<u8>,
    weight_scales: &DeviceBuffer<u8>,
    output_f32: &mut DeviceBuffer<f32>,
    output_bf16: &mut DeviceBuffer<u16>,
    in_features: usize,
    out_features: usize,
    strict_k_order: bool,
    requests: usize,
) -> Result<Vec<u64>, Box<dyn Error>> {
    let start = context.new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
    let end = context.new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
    for _ in 0..5 {
        // SAFETY: this helper preserves the fixed validated buffers.
        unsafe {
            launch_v4_fp8_matvec(
                module,
                stream,
                input,
                input_scales,
                weight,
                weight_scales,
                output_f32,
                output_bf16,
                in_features,
                out_features,
                strict_k_order,
            )
        }?;
    }
    end.record(stream)?;
    end.synchronize()?;
    let mut samples = Vec::with_capacity(requests);
    for _ in 0..requests {
        start.record(stream)?;
        // SAFETY: this helper preserves the fixed validated buffers.
        unsafe {
            launch_v4_fp8_matvec(
                module,
                stream,
                input,
                input_scales,
                weight,
                weight_scales,
                output_f32,
                output_bf16,
                in_features,
                out_features,
                strict_k_order,
            )
        }?;
        end.record(stream)?;
        end.synchronize()?;
        samples.push((start.elapsed_ms(&end)? as f64 * 1_000_000.0) as u64);
    }
    Ok(samples)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn benchmark_v4_kv_ready(
    context: &std::sync::Arc<CudaContext>,
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    input: &DeviceBuffer<u16>,
    freqs: &DeviceBuffer<f32>,
    rope: &mut DeviceBuffer<u16>,
    ready: &mut DeviceBuffer<u16>,
    empty_fp8: &mut DeviceBuffer<u8>,
    scales: &mut DeviceBuffer<u8>,
    topk: &mut DeviceBuffer<i32>,
    requests: usize,
) -> Result<Vec<u64>, Box<dyn Error>> {
    let start = context.new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
    let end = context.new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
    for _ in 0..5 {
        // SAFETY: benchmark reuses only the validated resident Phase A buffers.
        unsafe {
            launch_v4_kv_ready(
                module, stream, input, freqs, rope, ready, empty_fp8, scales, topk,
            )
        }?;
    }
    end.record(stream)?;
    end.synchronize()?;
    let mut samples = Vec::with_capacity(requests);
    for _ in 0..requests {
        start.record(stream)?;
        // SAFETY: benchmark reuses only the validated resident Phase A buffers.
        unsafe {
            launch_v4_kv_ready(
                module, stream, input, freqs, rope, ready, empty_fp8, scales, topk,
            )
        }?;
        end.record(stream)?;
        end.synchronize()?;
        samples.push((start.elapsed_ms(&end)? as f64 * 1_000_000.0) as u64);
    }
    Ok(samples)
}

pub(crate) fn benchmark_v4_kv_commit(
    context: &std::sync::Arc<CudaContext>,
    stream: &CudaStream,
    ready: &DeviceBuffer<u16>,
    cache: &DeviceBuffer<u16>,
    bytes: usize,
    requests: usize,
) -> Result<Vec<u64>, Box<dyn Error>> {
    let start = context.new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
    let end = context.new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
    for _ in 0..5 {
        // SAFETY: Phase B preflight already proved this fixed contiguous copy.
        unsafe {
            cuda_core::memory::memcpy_dtod_async(
                cache.cu_deviceptr(),
                ready.cu_deviceptr(),
                bytes,
                stream.cu_stream(),
            )?;
        }
    }
    end.record(stream)?;
    end.synchronize()?;
    let mut samples = Vec::with_capacity(requests);
    for _ in 0..requests {
        start.record(stream)?;
        // SAFETY: Phase B preflight already proved this fixed contiguous copy.
        unsafe {
            cuda_core::memory::memcpy_dtod_async(
                cache.cu_deviceptr(),
                ready.cu_deviceptr(),
                bytes,
                stream.cu_stream(),
            )?;
        }
        end.record(stream)?;
        end.synchronize()?;
        samples.push((start.elapsed_ms(&end)? as f64 * 1_000_000.0) as u64);
    }
    Ok(samples)
}

pub(crate) fn report_v4_state_u8(name: &str, actual: &[u8], expected: &[u8]) -> usize {
    assert_eq!(actual.len(), expected.len(), "{name}: length");
    let mismatches = actual
        .iter()
        .zip(expected)
        .filter(|(actual, expected)| actual != expected)
        .count();
    println!("{name}: bit_mismatches={mismatches}");
    mismatches
}

pub(crate) fn report_v4_state_i32(name: &str, actual: &[i32], expected: &[i32]) -> usize {
    assert_eq!(actual.len(), expected.len(), "{name}: length");
    let mismatches = actual
        .iter()
        .zip(expected)
        .filter(|(actual, expected)| actual != expected)
        .count();
    println!("{name}: mismatches={mismatches}");
    mismatches
}

pub(crate) fn run_real_v4_router_slice_if_available(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
) -> Result<(), Box<dyn Error>> {
    let root = v4_model_dir()?;
    let hash_path = root.join("model-00002-of-00064.safetensors");
    let learned_path = root.join("model-00005-of-00064.safetensors");
    if !hash_path.is_file() || !learned_path.is_file() {
        return Err(format!(
            "V4 L0/L3 router shards are unavailable under {}",
            root.display()
        )
        .into());
    }

    let rows = 4; // B=2, S=2 in the fixture's flattened reference order.
    let hash_input_ids = [0_i64, 1, 2, 129_279];
    let hash = safetensors::ShardFile::open(hash_path)?;
    let hash_input = deterministic_v4_bf16(rows, 7);
    let hash_norm = read_v4_bf16(&hash, "layers.0.ffn_norm.weight", &[7_168])?;
    let hash_gate = read_v4_bf16(&hash, "layers.0.ffn.gate.weight", &[384, 7_168])?;
    let hash_ids = read_v4_hash_rows(&hash, "layers.0.ffn.gate.tid2eid", &hash_input_ids)?;
    let hash_normalized = v4_router::rmsnorm_bf16(&hash_input, &hash_norm)?;
    let hash_logits = v4_router::project_gate_logits(&hash_normalized, &hash_gate)?;
    let hash_expected = v4_router::select_hash(&hash_logits, &hash_ids)?;
    let (hash_actual_normalized, hash_actual) = run_v4_router_gpu_oracle(
        module,
        stream,
        &hash_input,
        &hash_norm,
        &hash_logits,
        &vec![0.0; v4_router::V4_ROUTED_EXPERTS],
        &hash_ids,
        true,
    )?;
    assert_eq!(hash_actual_normalized, hash_normalized, "L0 BF16 RMSNorm");
    assert_router_selection_close("L0 V4 hash router", &hash_actual, &hash_expected);

    let learned = safetensors::ShardFile::open(learned_path)?;
    let learned_input = deterministic_v4_bf16(rows, 31);
    let learned_norm = read_v4_bf16(&learned, "layers.3.ffn_norm.weight", &[7_168])?;
    let learned_gate = read_v4_bf16(&learned, "layers.3.ffn.gate.weight", &[384, 7_168])?;
    let learned_bias = read_v4_f32(&learned, "layers.3.ffn.gate.bias", &[384])?;
    let learned_normalized = v4_router::rmsnorm_bf16(&learned_input, &learned_norm)?;
    let learned_logits = v4_router::project_gate_logits(&learned_normalized, &learned_gate)?;
    let learned_expected = v4_router::select_learned(&learned_logits, &learned_bias)?;
    let (learned_actual_normalized, learned_actual) = run_v4_router_gpu_oracle(
        module,
        stream,
        &learned_input,
        &learned_norm,
        &learned_logits,
        &learned_bias,
        &[],
        false,
    )?;
    assert_eq!(
        learned_actual_normalized, learned_normalized,
        "L3 BF16 RMSNorm"
    );
    assert_router_selection_close("L3 V4 learned router", &learned_actual, &learned_expected);
    println!(
        "PASS: real V4 L0 hash and L3 learned router ranges, B=2/S=2, match CPU and GPU selection"
    );
    Ok(())
}
