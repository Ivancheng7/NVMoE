use super::*;

pub(crate) struct MhcGpuOutput {
    pub(crate) collapsed: Vec<u16>,
    pub(crate) post: Vec<f32>,
    pub(crate) comb: Vec<f32>,
    pub(crate) debug: Option<MhcDebugOutput>,
    pub(crate) telemetry: MhcTelemetry,
}

pub(crate) struct MhcDebugOutput {
    pub(crate) mixes: Vec<f32>,
    pub(crate) pre: Vec<f32>,
    pub(crate) sinkhorn_trace: Vec<f32>,
    pub(crate) raw_dot: Option<Vec<f32>>,
    pub(crate) rsqrt: Option<Vec<f32>>,
    pub(crate) rsqrt_debug: Option<Vec<u32>>,
}

pub(crate) const FLASH_MHC_RSQRT_DEBUG_FIELDS: usize = 5;
pub(crate) const FLASH_MHC_RSQRT_COMMITTED_CANDIDATE: usize = 4;
pub(crate) const FLASH_MHC_RSQRT_COMMITTED_FIELD: &str = "libdevice_rsqrt_f32_bits";

pub(crate) fn flash_mhc_rsqrt_debug_fields() -> [&'static str; FLASH_MHC_RSQRT_DEBUG_FIELDS] {
    [
        "square_sum_f32_bits",
        "mean_square_plus_eps_f32_bits",
        "newton_candidate_f32_bits",
        "libdevice_sqrt_f32_bits",
        "libdevice_rsqrt_f32_bits",
    ]
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum FlashMhcLinearArm {
    Canonical,
    Cublas,
}

impl FlashMhcLinearArm {
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Canonical => "canonical",
            Self::Cublas => "cublas",
        }
    }
}

pub(crate) fn flash_mhc_linear_arm_for(
    spec: &v4_spec::V4Spec,
    value: Option<&std::ffi::OsStr>,
) -> Result<FlashMhcLinearArm, Box<dyn Error>> {
    if spec != &v4_spec::FLASH_0731 {
        return Ok(FlashMhcLinearArm::Canonical);
    }
    match value {
        None => Ok(FlashMhcLinearArm::Canonical),
        Some(value) if value == "canonical" => Ok(FlashMhcLinearArm::Canonical),
        Some(value) if value == "cublas" => Ok(FlashMhcLinearArm::Cublas),
        Some(_) => {
            Err("COLIBRI_V4_FLASH_MHC_LINEAR_ARM must be canonical or cublas for Flash-0731".into())
        }
    }
}

pub(crate) fn flash_mhc_linear_arm(spec: &v4_spec::V4Spec) -> Result<FlashMhcLinearArm, Box<dyn Error>> {
    let value = std::env::var_os("COLIBRI_V4_FLASH_MHC_LINEAR_ARM");
    flash_mhc_linear_arm_for(spec, value.as_deref())
}

pub(crate) fn flash_router_linear_arm_for(
    spec: &v4_spec::V4Spec,
    value: Option<&std::ffi::OsStr>,
) -> Result<FlashMhcLinearArm, Box<dyn Error>> {
    if spec != &v4_spec::FLASH_0731 {
        return Ok(FlashMhcLinearArm::Canonical);
    }
    match value {
        None => Ok(FlashMhcLinearArm::Canonical),
        Some(value) if value == "canonical" => Ok(FlashMhcLinearArm::Canonical),
        Some(value) if value == "cublas" => Ok(FlashMhcLinearArm::Cublas),
        Some(_) => Err(
            "COLIBRI_V4_FLASH_ROUTER_LINEAR_ARM must be canonical or cublas for Flash-0731".into(),
        ),
    }
}

pub(crate) fn flash_router_linear_arm(spec: &v4_spec::V4Spec) -> Result<FlashMhcLinearArm, Box<dyn Error>> {
    let value = std::env::var_os("COLIBRI_V4_FLASH_ROUTER_LINEAR_ARM");
    flash_router_linear_arm_for(spec, value.as_deref())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum FlashRouterScoreArm {
    Canonical,
    Libdevice,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum FlashFp4AccumArm {
    Current,
    Block32,
    TilelangRef,
    Fp8Mma,
}

impl FlashFp4AccumArm {
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Current => "current",
            Self::Block32 => "block32",
            Self::TilelangRef => "tilelang_ref",
            Self::Fp8Mma => "fp8_mma",
        }
    }

    pub(crate) const fn uses_raw_fp8(self) -> bool {
        matches!(self, Self::Block32 | Self::TilelangRef | Self::Fp8Mma)
    }
}

pub(crate) fn flash_fp4_accum_arm_for(
    spec: &v4_spec::V4Spec,
    value: Option<&std::ffi::OsStr>,
) -> Result<FlashFp4AccumArm, Box<dyn Error>> {
    if spec != &v4_spec::FLASH_0731 {
        return Ok(FlashFp4AccumArm::Current);
    }
    match value {
        // sm_120 business baseline: keep the raw-FP8/block32 path as the
        // reproducible default.  The WMMA and cached TileLang paths remain
        // explicit experiments until they have stable target-GPU evidence.
        None => Ok(FlashFp4AccumArm::Block32),
        Some(value) if value == "current" => Ok(FlashFp4AccumArm::Current),
        Some(value) if value == "block32" => Ok(FlashFp4AccumArm::Block32),
        Some(value) if value == "tilelang_ref" => Ok(FlashFp4AccumArm::TilelangRef),
        Some(value) if value == "fp8_mma" => Ok(FlashFp4AccumArm::Fp8Mma),
        Some(_) => {
            Err("COLIBRI_V4_FLASH_FP4_ACCUM_ARM must be current, block32, tilelang_ref, or fp8_mma for Flash-0731".into())
        }
    }
}

pub(crate) fn flash_fp4_accum_arm(spec: &v4_spec::V4Spec) -> Result<FlashFp4AccumArm, Box<dyn Error>> {
    let value = std::env::var_os("COLIBRI_V4_FLASH_FP4_ACCUM_ARM");
    flash_fp4_accum_arm_for(spec, value.as_deref())
}

impl FlashRouterScoreArm {
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Canonical => "canonical",
            Self::Libdevice => "libdevice",
        }
    }

    pub(crate) const fn kernel_flag(self) -> u32 {
        match self {
            Self::Canonical => 0,
            Self::Libdevice => 1,
        }
    }
}

pub(crate) fn flash_router_score_arm_for(
    spec: &v4_spec::V4Spec,
    value: Option<&std::ffi::OsStr>,
) -> Result<FlashRouterScoreArm, Box<dyn Error>> {
    if spec != &v4_spec::FLASH_0731 {
        return Ok(FlashRouterScoreArm::Canonical);
    }
    match value {
        None => Ok(FlashRouterScoreArm::Canonical),
        Some(value) if value == "canonical" => Ok(FlashRouterScoreArm::Canonical),
        Some(value) if value == "libdevice" => Ok(FlashRouterScoreArm::Libdevice),
        Some(_) => Err(
            "COLIBRI_V4_FLASH_ROUTER_SCORE_ARM must be canonical or libdevice for Flash-0731"
                .into(),
        ),
    }
}

pub(crate) fn flash_router_score_arm(spec: &v4_spec::V4Spec) -> Result<FlashRouterScoreArm, Box<dyn Error>> {
    let value = std::env::var_os("COLIBRI_V4_FLASH_ROUTER_SCORE_ARM");
    flash_router_score_arm_for(spec, value.as_deref())
}

pub(crate) fn flash_router_score_provenance(arm: FlashRouterScoreArm) -> serde_json::Value {
    match arm {
        FlashRouterScoreArm::Canonical => serde_json::json!({
            "arm": arm.name(),
        }),
        FlashRouterScoreArm::Libdevice => serde_json::json!({
            "arm": arm.name(),
            "softplus": "x > 20 ? x : __nv_log1pf(__nv_expf(x))",
            "sqrt": "__nv_sqrtf",
            "device_symbol_execution": "requires GPU parity gate",
        }),
    }
}

#[cfg(test)]
pub(crate) fn flash_router_libdevice_score_host_reference(logit: f32) -> f32 {
    let softplus = if logit > 20.0 {
        logit
    } else {
        libm::log1pf(libm::expf(logit))
    };
    libm::sqrtf(softplus)
}

pub(crate) struct FlashCublasSession {
    pub(crate) cublas: cublas_exact::CublasExact,
    pub(crate) workspace: DeviceBuffer<u8>,
}

impl FlashCublasSession {
    pub(crate) fn new(stream: &CudaStream) -> Result<Self, Box<dyn Error>> {
        Ok(Self {
            // Keep the verified object mapped for the full request.  Reopening it
            // cannot improve numerical behavior and used to repeat a 105 MiB SHA.
            cublas: cublas_exact::CublasExact::load()?,
            workspace: DeviceBuffer::zeroed(stream, cublas_exact::WORKSPACE_BYTES)?,
        })
    }
}

pub(crate) fn flash_mhc_cublas_provenance(
    cublas: &cublas_exact::CublasExact,
) -> Result<serde_json::Value, Box<dyn Error>> {
    let mut provenance = flash_cublas_sgemm_provenance(
        cublas,
        v4_state::MIX,
        v4_state::HC * v4_spec::FLASH_0731.hidden,
    )?;
    provenance
        .as_object_mut()
        .ok_or("Flash cuBLAS provenance must be an object")?
        .insert(
            "rsqrt".into(),
            serde_json::json!({
                "actual": "direct __nv_rsqrtf(mean_square_plus_eps)",
                "actual_instruction": "shared device helper calling libdevice __nv_rsqrtf; this is debug candidate 4",
                "committed_debug_candidate": FLASH_MHC_RSQRT_COMMITTED_FIELD,
                "debug_candidates": flash_mhc_rsqrt_debug_fields(),
                "debug_candidate_notes": {
                    "libdevice_sqrt_f32_bits": "1.0 / core::f32::math::sqrt(x), lowered by cuda-oxide to libdevice __nv_sqrtf",
                    "libdevice_rsqrt_f32_bits": "direct #[device] extern C call to libdevice __nv_rsqrtf(x), also the committed Flash cuBLAS-arm normalization",
                },
            }),
        );
    Ok(provenance)
}

pub(crate) fn flash_router_cublas_provenance(
    cublas: &cublas_exact::CublasExact,
) -> Result<serde_json::Value, Box<dyn Error>> {
    flash_cublas_sgemm_provenance(
        cublas,
        v4_spec::FLASH_0731.routed_experts,
        v4_spec::FLASH_0731.hidden,
    )
}

pub(crate) fn flash_cublas_sgemm_provenance(
    cublas: &cublas_exact::CublasExact,
    m: usize,
    k: usize,
) -> Result<serde_json::Value, Box<dyn Error>> {
    let provenance = cublas.provenance();
    Ok(serde_json::json!({
        "abi": "cublasSgemm_v2",
        "resolved_path": provenance.resolved_path,
        "dladdr_path": provenance.dladdr_path,
        "maps_path": provenance.maps_path,
        "sha256": provenance.actual_sha256,
        "version": provenance.version,
        "math_mode": "CUBLAS_DEFAULT_MATH",
        "tf32": "off",
        "trans_a": "T",
        "trans_b": "N",
        "m": m,
        "n": 1,
        "k": k,
        "alpha": 1.0_f32,
        "beta": 0.0_f32,
        "workspace_bytes": cublas_exact::WORKSPACE_BYTES,
    }))
}

pub(crate) struct MhcCublasBuffers {
    pub(crate) _input_f32: DeviceBuffer<f32>,
    pub(crate) _squares: DeviceBuffer<f32>,
    pub(crate) raw_dot: DeviceBuffer<f32>,
    pub(crate) rsqrt: DeviceBuffer<f32>,
    pub(crate) rsqrt_debug: Option<DeviceBuffer<u32>>,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_flash_mhc_cublas_linear_device_into(
    session: &FlashCublasSession,
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    input: &DeviceBuffer<u16>,
    hc_fn: &DeviceBuffer<f32>,
    mixes: &mut DeviceBuffer<f32>,
    rows: usize,
    hidden: usize,
    capture_rsqrt_debug: bool,
) -> Result<MhcCublasBuffers, Box<dyn Error>> {
    let k = v4_state::HC
        .checked_mul(hidden)
        .ok_or("Flash mHC cuBLAS K overflow")?;
    if hidden != v4_spec::FLASH_0731.hidden
        || k != 16_384
        || rows == 0
        || input.len() != rows * k
        || hc_fn.len() != v4_state::MIX * k
        || mixes.len() != rows * v4_state::MIX
    {
        return Err("Flash mHC cuBLAS SGEMM geometry must be rows x 24 x 16384".into());
    }
    let mut input_f32 = DeviceBuffer::<f32>::zeroed(stream, input.len())?;
    let mut squares = DeviceBuffer::<f32>::zeroed(stream, input.len())?;
    let raw_dot = DeviceBuffer::<f32>::zeroed(stream, mixes.len())?;
    let mut rsqrt = DeviceBuffer::<f32>::zeroed(stream, rows)?;
    let mut rsqrt_debug = capture_rsqrt_debug
        .then(|| DeviceBuffer::<u32>::zeroed(stream, rows * FLASH_MHC_RSQRT_DEBUG_FIELDS))
        .transpose()?;
    launch_v4_bf16_to_f32(module, stream, input, &mut input_f32)?;
    let device_const = |pointer: u64| pointer as usize as *const std::ffi::c_void;
    let device_mut = |pointer: u64| pointer as usize as *mut std::ffi::c_void;
    for row in 0..rows {
        let input_offset = row
            .checked_mul(k)
            .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
            .ok_or("Flash mHC cuBLAS input offset overflow")?;
        let output_offset = row
            .checked_mul(v4_state::MIX)
            .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
            .ok_or("Flash mHC cuBLAS output offset overflow")?;
        session.cublas.sgemm_mn1(
            stream.cu_stream().cast::<std::ffi::c_void>(),
            device_mut(session.workspace.cu_deviceptr()),
            session.workspace.num_bytes(),
            device_const(hc_fn.cu_deviceptr()),
            device_const(
                input_f32
                    .cu_deviceptr()
                    .checked_add(u64::try_from(input_offset)?)
                    .ok_or("Flash mHC cuBLAS input pointer overflow")?,
            ),
            device_mut(
                raw_dot
                    .cu_deviceptr()
                    .checked_add(u64::try_from(output_offset)?)
                    .ok_or("Flash mHC cuBLAS output pointer overflow")?,
            ),
            v4_state::MIX,
            k,
        )?;
    }
    let mut empty_debug = DeviceBuffer::<u32>::zeroed(stream, 0)?;
    let debug = rsqrt_debug.as_mut().unwrap_or(&mut empty_debug);
    // SAFETY: the square launch owns each input lane, the fixed reduction owns
    // one complete 16,384-wide row, and the final scale owns 24 disjoint dots.
    unsafe {
        module.rmsnorm_flash_square_bf16(
            stream,
            LaunchConfig::for_num_elems(u32::try_from(input.len())?),
            input,
            &mut squares,
        )?;
        module.mhc_flash_reduce_rsqrt_16384(
            stream,
            LaunchConfig {
                grid_dim: (u32::try_from(rows)?, 1, 1),
                block_dim: (512, 1, 1),
                shared_mem_bytes: 0,
            },
            &squares,
            &mut rsqrt,
            debug,
        )?;
        module.mhc_scale_raw_dot_by_rsqrt_v4(
            stream,
            LaunchConfig::for_num_elems(u32::try_from(rows)?),
            &raw_dot,
            &rsqrt,
            mixes,
        )?;
    }
    Ok(MhcCublasBuffers {
        _input_f32: input_f32,
        _squares: squares,
        raw_dot,
        rsqrt,
        rsqrt_debug,
    })
}

pub(crate) struct RouterCublasBuffers {
    pub(crate) _input_f32: DeviceBuffer<f32>,
    pub(crate) _gate_f32: DeviceBuffer<f32>,
}

pub(crate) fn run_flash_router_cublas_linear_device_into(
    session: &FlashCublasSession,
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    input: &DeviceBuffer<u16>,
    gate: &DeviceBuffer<u16>,
    logits: &mut DeviceBuffer<f32>,
    rows: usize,
    hidden: usize,
    experts: usize,
) -> Result<RouterCublasBuffers, Box<dyn Error>> {
    if hidden != v4_spec::FLASH_0731.hidden
        || experts != v4_spec::FLASH_0731.routed_experts
        || rows == 0
        || input.len() != rows * hidden
        || gate.len() != experts * hidden
        || logits.len() != rows * experts
    {
        return Err("Flash router cuBLAS SGEMM geometry must be rows x 256 x 4096".into());
    }
    let mut input_f32 = DeviceBuffer::<f32>::zeroed(stream, input.len())?;
    let mut gate_f32 = DeviceBuffer::<f32>::zeroed(stream, gate.len())?;
    // ponytail: converts the 256x4096 gate per router call; merge into an
    // existing fixed-weight cache only after this arm proves parity and a
    // profile attributes material time to conversion.
    launch_v4_bf16_to_f32(module, stream, input, &mut input_f32)?;
    launch_v4_bf16_to_f32(module, stream, gate, &mut gate_f32)?;
    let device_const = |pointer: u64| pointer as usize as *const std::ffi::c_void;
    let device_mut = |pointer: u64| pointer as usize as *mut std::ffi::c_void;
    for row in 0..rows {
        let input_offset = row
            .checked_mul(hidden)
            .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
            .ok_or("Flash router cuBLAS input offset overflow")?;
        let output_offset = row
            .checked_mul(experts)
            .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
            .ok_or("Flash router cuBLAS output offset overflow")?;
        session.cublas.sgemm_mn1(
            stream.cu_stream().cast::<std::ffi::c_void>(),
            device_mut(session.workspace.cu_deviceptr()),
            session.workspace.num_bytes(),
            device_const(gate_f32.cu_deviceptr()),
            device_const(
                input_f32
                    .cu_deviceptr()
                    .checked_add(u64::try_from(input_offset)?)
                    .ok_or("Flash router cuBLAS input pointer overflow")?,
            ),
            device_mut(
                logits
                    .cu_deviceptr()
                    .checked_add(u64::try_from(output_offset)?)
                    .ok_or("Flash router cuBLAS output pointer overflow")?,
            ),
            experts,
            hidden,
        )?;
    }
    Ok(RouterCublasBuffers {
        _input_f32: input_f32,
        _gate_f32: gate_f32,
    })
}

#[derive(Clone, Default)]
pub(crate) struct AttentionProfileStep {
    pub(crate) sync_wall_ns: u64,
    pub(crate) gpu_event_ns: u64,
}

#[derive(Clone, Default)]
pub(crate) struct AttentionProfile {
    pub(crate) mhc: AttentionProfileStep,
    pub(crate) attn_norm: AttentionProfileStep,
    pub(crate) q_a_projection: AttentionProfileStep,
    pub(crate) q_a_norm: AttentionProfileStep,
    pub(crate) q_b_projection: AttentionProfileStep,
    pub(crate) q_b_norm: AttentionProfileStep,
    pub(crate) q_rope: AttentionProfileStep,
    pub(crate) kv_projection: AttentionProfileStep,
    pub(crate) kv_norm: AttentionProfileStep,
    pub(crate) kv_rope_quantize: AttentionProfileStep,
    pub(crate) compressor: AttentionProfileStep,
    pub(crate) indexer: AttentionProfileStep,
    pub(crate) commit_topk: AttentionProfileStep,
    pub(crate) sparse_score: AttentionProfileStep,
    pub(crate) sparse_inverse_rope: AttentionProfileStep,
    pub(crate) wo_a: AttentionProfileStep,
    pub(crate) wo_b: AttentionProfileStep,
    pub(crate) hc_post: AttentionProfileStep,
}

impl AttentionProfile {
    pub(crate) fn step_json(step: &AttentionProfileStep) -> serde_json::Value {
        serde_json::json!({"sync_wall_ns": step.sync_wall_ns, "gpu_event_ns": step.gpu_event_ns})
    }

    pub(crate) fn json(&self) -> serde_json::Value {
        serde_json::json!({
            "mhc": Self::step_json(&self.mhc),
            "attn_norm": Self::step_json(&self.attn_norm),
            "q_a_projection": Self::step_json(&self.q_a_projection),
            "q_a_norm": Self::step_json(&self.q_a_norm),
            "q_b_projection": Self::step_json(&self.q_b_projection),
            "q_b_norm": Self::step_json(&self.q_b_norm),
            "q_rope": Self::step_json(&self.q_rope),
            "kv_projection": Self::step_json(&self.kv_projection),
            "kv_norm": Self::step_json(&self.kv_norm),
            "kv_rope_quantize": Self::step_json(&self.kv_rope_quantize),
            "compressor": Self::step_json(&self.compressor),
            "indexer": Self::step_json(&self.indexer),
            "commit_topk": Self::step_json(&self.commit_topk),
            "sparse_score": Self::step_json(&self.sparse_score),
            "sparse_inverse_rope": Self::step_json(&self.sparse_inverse_rope),
            "wo_a": Self::step_json(&self.wo_a),
            "wo_b": Self::step_json(&self.wo_b),
            "hc_post": Self::step_json(&self.hc_post),
        })
    }
}

#[derive(Default)]
pub(crate) struct LayerStageTelemetry {
    pub(crate) layer_prepare_ns: u64,
    pub(crate) layer_prepare_count: usize,
    pub(crate) attention_ns: u64,
    pub(crate) device_state_setup_ns: u64,
    pub(crate) device_state_setup_count: usize,
    pub(crate) device_state_reset_cuda_ns: u64,
    pub(crate) device_state_reset_calls: usize,
    pub(crate) attention_audit_oracle_ns: u64,
    pub(crate) attention_hidden_h2d_count: usize,
    pub(crate) attention_hidden_h2d_bytes: usize,
    pub(crate) attention_rope_lut_d2d_count: usize,
    pub(crate) attention_rope_lut_d2d_bytes: usize,
    pub(crate) attention_terminal_d2h_count: usize,
    pub(crate) attention_terminal_d2h_bytes: usize,
    pub(crate) attention_intermediate_or_audit_d2h_count: usize,
    pub(crate) attention_intermediate_or_audit_d2h_bytes: usize,
    pub(crate) attention_steady_device_temporary_alloc_count: usize,
    pub(crate) attention_profile: Option<AttentionProfile>,
    pub(crate) router_ns: u64,
    pub(crate) router_mhc_ns: u64,
    pub(crate) router_mhc_input_alloc_h2d_sync_ns: u64,
    pub(crate) router_mhc_workspace_alloc_sync_ns: u64,
    pub(crate) router_mhc_workspace_init_gpu_ns: u64,
    pub(crate) router_mhc_mixing_linear_sync_ns: u64,
    pub(crate) router_mhc_mixing_linear_gpu_ns: u64,
    pub(crate) router_mhc_sinkhorn_normalize_collapse_sync_ns: u64,
    pub(crate) router_mhc_sinkhorn_normalize_collapse_gpu_ns: u64,
    pub(crate) router_mhc_output_d2h_sync_ns: u64,
    pub(crate) router_mhc_other_ns: u64,
    pub(crate) router_host_prep_ns: u64,
    pub(crate) router_alloc_h2d_sync_ns: u64,
    pub(crate) router_workspace_alloc_ns: u64,
    pub(crate) router_kernel_select_sync_ns: u64,
    pub(crate) router_kernel_select_gpu_ns: u64,
    pub(crate) router_d2h_sync_ns: u64,
    pub(crate) router_other_ns: u64,
    pub(crate) shared_ns: u64,
    /// Time from the shared window's start until the shared MLP returned on the
    /// main thread. The window also covers the routed read workers, so
    /// `shared_ns - shared_main_ns` is the tail spent joining them (normally the
    /// slowest reader). Kept so the shared row can carry its own breakdown
    /// instead of the phase table attributing the reads to `shared`.
    pub(crate) shared_main_ns: u64,
    pub(crate) shared_read_workers: usize,
    pub(crate) shared_read_worker_sum_ns: u64,
    pub(crate) shared_read_worker_max_ns: u64,
    pub(crate) routed_ns: u64,
    pub(crate) routed_prefetched: bool,
    pub(crate) routed_read_ns: u64,
    pub(crate) routed_alloc_ns: u64,
    pub(crate) routed_setup_ns: u64,
    pub(crate) routed_h2d_ns: u64,
    pub(crate) routed_kernel_ns: u64,
    pub(crate) commit_ns: u64,
    pub(crate) routed_source_bytes: usize,
    pub(crate) routed_h2d_bytes: usize,
    pub(crate) routed_expert_loads: usize,
    /// Route slots served from the identity cache instead of a disk read, when
    /// `COLIBRI_V4_ROUTED_EXPERT_CACHE` is on. 0 on every other path.
    pub(crate) routed_cache_hits: usize,
    pub(crate) routed_parallel_batch_wall_ns: Option<u64>,
    pub(crate) routed_parallel_read_horizon_ns: Option<u64>,
}

impl LayerStageTelemetry {
    pub(crate) fn router_accounted_ns(&self) -> u64 {
        self.router_mhc_ns
            .saturating_add(self.router_host_prep_ns)
            .saturating_add(self.router_alloc_h2d_sync_ns)
            .saturating_add(self.router_workspace_alloc_ns)
            .saturating_add(self.router_kernel_select_sync_ns)
            .saturating_add(self.router_d2h_sync_ns)
            .saturating_add(self.router_other_ns)
    }
}

impl std::ops::AddAssign for LayerStageTelemetry {
    /// Folds one position's stage telemetry into a layer-pass accumulator. This
    /// is what makes a chunked prompt pass report a wall that covers every
    /// position it executed instead of only the last one. `routed_prefetched`
    /// is the AND of its inputs: a pass that mixed prefetched and synchronous
    /// routed loads reports "not prefetched" and fails the nested-scope guard
    /// loudly rather than claiming a background read it did not have.
    fn add_assign(&mut self, other: Self) {
        macro_rules! fold {
            ($($field:ident),* $(,)?) => {
                $(self.$field = self.$field.saturating_add(other.$field);)*
            };
        }
        fold!(
            layer_prepare_ns,
            layer_prepare_count,
            attention_ns,
            device_state_setup_ns,
            device_state_setup_count,
            device_state_reset_cuda_ns,
            device_state_reset_calls,
            attention_audit_oracle_ns,
            attention_hidden_h2d_count,
            attention_hidden_h2d_bytes,
            attention_rope_lut_d2d_count,
            attention_rope_lut_d2d_bytes,
            attention_terminal_d2h_count,
            attention_terminal_d2h_bytes,
            attention_intermediate_or_audit_d2h_count,
            attention_intermediate_or_audit_d2h_bytes,
            attention_steady_device_temporary_alloc_count,
            router_ns,
            router_mhc_ns,
            router_mhc_input_alloc_h2d_sync_ns,
            router_mhc_workspace_alloc_sync_ns,
            router_mhc_workspace_init_gpu_ns,
            router_mhc_mixing_linear_sync_ns,
            router_mhc_mixing_linear_gpu_ns,
            router_mhc_sinkhorn_normalize_collapse_sync_ns,
            router_mhc_sinkhorn_normalize_collapse_gpu_ns,
            router_mhc_output_d2h_sync_ns,
            router_mhc_other_ns,
            router_host_prep_ns,
            router_alloc_h2d_sync_ns,
            router_workspace_alloc_ns,
            router_kernel_select_sync_ns,
            router_kernel_select_gpu_ns,
            router_d2h_sync_ns,
            router_other_ns,
            shared_ns,
            shared_main_ns,
            shared_read_workers,
            shared_read_worker_sum_ns,
            shared_read_worker_max_ns,
            routed_ns,
            routed_read_ns,
            routed_alloc_ns,
            routed_setup_ns,
            routed_h2d_ns,
            routed_kernel_ns,
            commit_ns,
            routed_source_bytes,
            routed_h2d_bytes,
            routed_expert_loads,
            routed_cache_hits,
        );
        self.routed_prefetched = self.routed_prefetched && other.routed_prefetched;
        if other.attention_profile.is_some() {
            self.attention_profile = other.attention_profile;
        }
        self.routed_parallel_batch_wall_ns = fold_option_ns(
            self.routed_parallel_batch_wall_ns,
            other.routed_parallel_batch_wall_ns,
        );
        self.routed_parallel_read_horizon_ns = fold_option_ns(
            self.routed_parallel_read_horizon_ns,
            other.routed_parallel_read_horizon_ns,
        );
    }
}

fn fold_option_ns(left: Option<u64>, right: Option<u64>) -> Option<u64> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left.saturating_add(right)),
        (Some(value), None) | (None, Some(value)) => Some(value),
        (None, None) => None,
    }
}
