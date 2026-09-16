use super::*;

pub(crate) const V4_QD4_LANES: usize = 4;
pub(crate) const V4_DEVICE_CAPACITY_HEADROOM_BYTES: usize = 2 * 1024 * 1024 * 1024;
// MTP P0 owns only its sliding KV persistently.  The routed FFN uses one
// ordinary ExpertSlot at a time and never enters the teacher cache.
pub(crate) const MTP1_DRAFT_PERSISTENT_BYTES: usize = 131_072;
// These are the candidate-owned allocations which outlive the fixed-weight
// upload through the streamed head.  They are deliberately separate from the
// fixed tensor ledger: the candidate keeps its sliding-KV input, raw-window
// attention cache, Q, sparse output, and score-reuse scratch live while the
// fixed block is resident and while the final head is streamed.
pub(crate) const MTP1_DRAFT_ENORM_HNORM_BYTES: usize = 2 * v4_state::HIDDEN * std::mem::size_of::<u16>();
pub(crate) const MTP1_DRAFT_KV_READY_BYTES: usize = v4_state::Q_HEAD_DIM * std::mem::size_of::<u16>();
pub(crate) const MTP1_DRAFT_Q_BYTES: usize =
    v4_state::Q_HEADS * v4_state::Q_HEAD_DIM * std::mem::size_of::<u16>();
pub(crate) const MTP1_DRAFT_CACHE_BYTES: usize =
    v4_state::KV_WINDOW * v4_state::Q_HEAD_DIM * std::mem::size_of::<u16>();
pub(crate) const MTP1_DRAFT_TOPK_BYTES: usize = v4_state::KV_WINDOW * std::mem::size_of::<i32>();
pub(crate) const MTP1_DRAFT_SPARSE_BYTES: usize =
    v4_state::Q_HEADS * v4_state::Q_HEAD_DIM * std::mem::size_of::<u16>();
pub(crate) const MTP1_DRAFT_SCORE_REUSE_BYTES: usize =
    2 * v4_state::Q_HEADS * v4_state::KV_WINDOW * std::mem::size_of::<f32>()
        + v4_state::Q_HEADS * (std::mem::size_of::<f32>() + std::mem::size_of::<u32>());
pub(crate) const MTP1_DRAFT_CANDIDATE_LOCAL_BYTES: usize = MTP1_DRAFT_ENORM_HNORM_BYTES
    + MTP1_DRAFT_KV_READY_BYTES
    + MTP1_DRAFT_Q_BYTES
    + MTP1_DRAFT_CACHE_BYTES
    + MTP1_DRAFT_TOPK_BYTES
    + MTP1_DRAFT_SPARSE_BYTES
    + MTP1_DRAFT_SCORE_REUSE_BYTES;
pub(crate) const MTP1_DRAFT_TRANSIENT_EXPERT_SLOT_BYTES: usize = 35_094_528;
pub(crate) const MTP1_DRAFT_EXTRA_CAPACITY_BYTES: usize = 43_230_456;
pub(crate) const MTP1_DRAFT_DEFAULT_MAX_SCORED_TRANSITIONS: usize = 8;

pub(crate) fn mtp1_draft_peak_bytes(
    persistent_bytes: usize,
    fixed_bytes: usize,
    e_projection_bytes: usize,
    h_projection_bytes: usize,
    expert_slot_bytes: usize,
    lm_head_bytes: usize,
) -> Result<usize, Box<dyn Error>> {
    let add = |left: usize, right: usize| {
        left.checked_add(right)
            .ok_or("MTP1 draft peak ledger overflow")
    };
    let candidate_base = add(persistent_bytes, MTP1_DRAFT_CANDIDATE_LOCAL_BYTES)?;
    let projection_peak = add(candidate_base, e_projection_bytes.max(h_projection_bytes))?;
    let fixed_expert_peak = add(add(candidate_base, fixed_bytes)?, expert_slot_bytes)?;
    let lm_head_peak = add(candidate_base, lm_head_bytes)?;
    Ok(projection_peak.max(fixed_expert_peak).max(lm_head_peak))
}

pub(crate) fn mtp1_draft_fixed_device_bytes(
    host: &v4_state::MtpDraftWeights,
) -> Result<usize, Box<dyn Error>> {
    let fp8 = |matrix: &v4_state::Fp8Matrix| matrix.weight.len().checked_add(matrix.scale.len());
    let bytes = [
        host.hc_attn_fn
            .len()
            .checked_mul(std::mem::size_of::<f32>()),
        host.hc_attn_base
            .len()
            .checked_mul(std::mem::size_of::<f32>()),
        host.hc_attn_scale
            .len()
            .checked_mul(std::mem::size_of::<f32>()),
        host.attn_norm.len().checked_mul(std::mem::size_of::<u16>()),
        fp8(&host.wq_a.matrix),
        host.wq_a.norm.len().checked_mul(std::mem::size_of::<u16>()),
        fp8(&host.wq_b),
        fp8(&host.wkv.matrix),
        host.wkv.norm.len().checked_mul(std::mem::size_of::<u16>()),
        host.attn_sink.len().checked_mul(std::mem::size_of::<f32>()),
        host.wo_a
            .weight
            .len()
            .checked_mul(std::mem::size_of::<u16>()),
        // Upload converts FP8 wo_a while the raw input remains live.
        Some(host.wo_a.weight.len()),
        fp8(&host.wo_b),
        host.hc_ffn_fn.len().checked_mul(std::mem::size_of::<f32>()),
        host.hc_ffn_base
            .len()
            .checked_mul(std::mem::size_of::<f32>()),
        host.hc_ffn_scale
            .len()
            .checked_mul(std::mem::size_of::<f32>()),
        host.ffn_norm.len().checked_mul(std::mem::size_of::<u16>()),
        host.gate.len().checked_mul(std::mem::size_of::<u16>()),
        host.router_bias
            .len()
            .checked_mul(std::mem::size_of::<f32>()),
        fp8(&host.shared.w1),
        fp8(&host.shared.w2),
        fp8(&host.shared.w3),
    ];
    bytes.into_iter().try_fold(0usize, |total, bytes| {
        total
            .checked_add(bytes.ok_or("MTP1 fixed device ledger overflow")?)
            .ok_or("MTP1 fixed device ledger overflow".into())
    })
}

pub(crate) fn mtp1_draft_fixed_device_bytes_from_geometry() -> Result<usize, Box<dyn Error>> {
    let fp8 = |input: usize, output: usize| {
        input
            .checked_mul(output)
            .and_then(|weight| {
                output
                    .div_ceil(v4_state::FP8_BLOCK)
                    .checked_mul(input / v4_state::FP8_BLOCK)
                    .and_then(|scale| weight.checked_add(scale))
            })
            .ok_or("MTP1 geometry fixed ledger overflow")
    };
    [
        v4_state::MIX * v4_state::HC * v4_state::HIDDEN * 4,
        v4_state::MIX * 4,
        3 * 4,
        v4_state::HIDDEN * 2,
        fp8(v4_state::HIDDEN, V4_Q_LORA_RANK)?,
        V4_Q_LORA_RANK * 2,
        fp8(V4_Q_LORA_RANK, v4_state::Q_HEADS * v4_state::Q_HEAD_DIM)?,
        fp8(v4_state::HIDDEN, v4_state::Q_HEAD_DIM)?,
        v4_state::Q_HEAD_DIM * 2,
        v4_state::Q_HEADS * 4,
        (v4_state::Q_HEADS * v4_state::Q_HEAD_DIM / v4_state::O_GROUPS)
            * (v4_state::O_GROUPS * v4_state::O_LORA_RANK)
            * 3,
        fp8(v4_state::O_GROUPS * v4_state::O_LORA_RANK, v4_state::HIDDEN)?,
        v4_state::MIX * v4_state::HC * v4_state::HIDDEN * 4,
        v4_state::MIX * 4,
        3 * 4,
        v4_state::HIDDEN * 2,
        v4_router::V4_ROUTED_EXPERTS * v4_state::HIDDEN * 2,
        v4_router::V4_ROUTED_EXPERTS * 4,
        fp8(v4_state::HIDDEN, v4_expert::V4_EXPERT_INTERMEDIATE)?,
        fp8(v4_expert::V4_EXPERT_INTERMEDIATE, v4_state::HIDDEN)?,
        fp8(v4_state::HIDDEN, v4_expert::V4_EXPERT_INTERMEDIATE)?,
    ]
    .into_iter()
    .try_fold(0usize, |total, bytes| {
        total
            .checked_add(bytes)
            .ok_or("MTP1 geometry fixed ledger overflow".into())
    })
}

pub(crate) fn mtp1_stationary_peak_reserve_bytes(
    model_dir: &std::path::Path,
    borrows_qd4_lane: bool,
) -> Result<usize, Box<dyn Error>> {
    let host = v4_state::load_mtp_draft_weights(model_dir)?;
    mtp1_draft_peak_bytes(
        MTP1_DRAFT_PERSISTENT_BYTES,
        mtp1_draft_fixed_device_bytes(&host)?,
        host.e_proj.source_bytes(),
        host.h_proj.source_bytes(),
        usize::from(!borrows_qd4_lane) * MTP1_DRAFT_TRANSIENT_EXPERT_SLOT_BYTES,
        CERTIFIED_ARGMAX_DEVICE_TEMP_BYTES,
    )
}

/// One extra routed slot is needed only when stationary MTP cannot borrow a
/// drained QD4 lane.  Keep its pinned staging reservation beside the raw-cache
/// host selection, rather than discovering it after that cache has filled RAM.
pub(crate) fn mtp1_stationary_pinned_host_reserve_bytes(borrows_qd4_lane: bool) -> usize {
    if borrows_qd4_lane {
        0
    } else {
        V4_QD4_PINNED_HOST_BYTES / V4_QD4_LANES
    }
}

pub(crate) fn generic_stationary_persistent_reserve_bytes(
    manifests: &[v4_state::LayerExecutionManifest],
    plan: v4_state::LayerStationaryRequestPlan,
) -> Result<usize, Box<dyn Error>> {
    let bytes = |elements: usize, element_bytes: usize| {
        elements
            .checked_mul(element_bytes)
            .ok_or("generic stationary persistent reserve overflow")
    };
    let state_bytes = |path: GenericStationaryPath| -> Result<usize, Box<dyn Error>> {
        let raw = bytes(v4_state::KV_WINDOW * v4_state::Q_HEAD_DIM, 2)?;
        let tails = bytes(path.tail_elements(), 4)?
            .checked_mul(2)
            .ok_or("generic stationary persistent tail reserve overflow")?;
        let compressed = bytes(path.compressed_rows(plan) * v4_state::Q_HEAD_DIM, 2)?;
        let indexer = if path.is_csa() {
            let indexer_kv = bytes(plan.csa_compressed_rows * 128, 2)?;
            bytes(8 * 256, 4)?
                .checked_mul(2)
                .and_then(|value| value.checked_add(indexer_kv))
                .ok_or("generic stationary indexer reserve overflow")?
        } else {
            0
        };
        raw.checked_add(tails)
            .and_then(|value| value.checked_add(compressed))
            .and_then(|value| value.checked_add(indexer))
            .ok_or("generic stationary persistent reserve overflow".into())
    };
    manifests.iter().try_fold(0usize, |total, manifest| {
        total
            .checked_add(state_bytes(GenericStationaryPath::from_ratio(
                manifest.compression_ratio,
            ))?)
            .ok_or("generic stationary persistent reserve overflow".into())
    })
}

pub(crate) fn generic_stationary_scratch_reserve_bytes(
    plan: v4_state::LayerStationaryRequestPlan,
) -> Result<usize, Box<dyn Error>> {
    generic_stationary_scratch_reserve_bytes_with_strict(plan, strict_compressor_enabled())
}

pub(crate) fn generic_stationary_scratch_reserve_bytes_with_strict(
    plan: v4_state::LayerStationaryRequestPlan,
    strict: bool,
) -> Result<usize, Box<dyn Error>> {
    DeviceRequestScratchGeometry::new(plan, strict)?.num_bytes()
}

pub(crate) fn stationary_mtp1_target_strict_scratch_contract(
    plan: v4_state::LayerStationaryRequestPlan,
    target_strict: bool,
    pinned_cublas: Option<&std::path::Path>,
) -> Result<usize, Box<dyn Error>> {
    if target_strict && pinned_cublas.is_none() {
        return Err(
            "strict stationary MTP capacity requires pinned COLIBRI_CUBLAS_EXACT_LIB".into(),
        );
    }
    generic_stationary_scratch_reserve_bytes_with_strict(plan, target_strict)
}

#[derive(Default)]
pub(crate) struct StationaryMtpCapacityProbe {
    pub(crate) device_u8: Vec<DeviceBuffer<u8>>,
    pub(crate) device_u16: Vec<DeviceBuffer<u16>>,
    pub(crate) device_f32: Vec<DeviceBuffer<f32>>,
    pub(crate) device_i32: Vec<DeviceBuffer<i32>>,
    pub(crate) device_u32: Vec<DeviceBuffer<u32>>,
    pub(crate) pinned_u8: Vec<PinnedHostBuffer<u8>>,
    pub(crate) pinned_u16: Vec<PinnedHostBuffer<u16>>,
    pub(crate) pinned_f32: Vec<PinnedHostBuffer<f32>>,
    pub(crate) host: Vec<Vec<u8>>,
    pub(crate) raw_host: Vec<Vec<u8>>,
    pub(crate) generic_scratch: Option<DeviceRequestScratch>,
}

pub(crate) const STATIONARY_MTP1_CAPACITY_PREFLIGHT_SCHEMA: &str = "colibri-v4-stationary-mtp1-capacity-v2";
pub(crate) const STATIONARY_MTP1_CAPACITY_PREFLIGHT_TARGET_STRICT: bool = true;

impl StationaryMtpCapacityProbe {
    pub(crate) fn device_bytes(&self) -> usize {
        self.device_u8
            .iter()
            .map(DeviceBuffer::num_bytes)
            .sum::<usize>()
            + self
                .device_u16
                .iter()
                .map(DeviceBuffer::num_bytes)
                .sum::<usize>()
            + self
                .device_f32
                .iter()
                .map(DeviceBuffer::num_bytes)
                .sum::<usize>()
            + self
                .device_i32
                .iter()
                .map(DeviceBuffer::num_bytes)
                .sum::<usize>()
            + self
                .device_u32
                .iter()
                .map(DeviceBuffer::num_bytes)
                .sum::<usize>()
            + self
                .generic_scratch
                .as_ref()
                .map_or(0, DeviceRequestScratch::num_bytes)
    }

    pub(crate) fn pinned_bytes(&self) -> usize {
        self.pinned_u8
            .iter()
            .map(PinnedHostBuffer::num_bytes)
            .sum::<usize>()
            + self
                .pinned_u16
                .iter()
                .map(PinnedHostBuffer::num_bytes)
                .sum::<usize>()
            + self
                .pinned_f32
                .iter()
                .map(PinnedHostBuffer::num_bytes)
                .sum::<usize>()
    }

    pub(crate) fn raw_host_bytes(&self) -> usize {
        self.raw_host.iter().map(Vec::len).sum()
    }

    pub(crate) fn host_bytes(&self) -> usize {
        self.host.iter().map(Vec::len).sum()
    }

    pub(crate) fn u8(&mut self, stream: &CudaStream, elements: usize) -> Result<(), Box<dyn Error>> {
        self.device_u8.push(DeviceBuffer::zeroed(stream, elements)?);
        Ok(())
    }

    pub(crate) fn u16(&mut self, stream: &CudaStream, elements: usize) -> Result<(), Box<dyn Error>> {
        self.device_u16
            .push(DeviceBuffer::zeroed(stream, elements)?);
        Ok(())
    }

    pub(crate) fn f32(&mut self, stream: &CudaStream, elements: usize) -> Result<(), Box<dyn Error>> {
        self.device_f32
            .push(DeviceBuffer::zeroed(stream, elements)?);
        Ok(())
    }

    pub(crate) fn i32(&mut self, stream: &CudaStream, elements: usize) -> Result<(), Box<dyn Error>> {
        self.device_i32
            .push(DeviceBuffer::zeroed(stream, elements)?);
        Ok(())
    }

    pub(crate) fn u32(&mut self, stream: &CudaStream, elements: usize) -> Result<(), Box<dyn Error>> {
        self.device_u32
            .push(DeviceBuffer::zeroed(stream, elements)?);
        Ok(())
    }
}

pub(crate) fn generic_stationary_audit_mirror_host_bytes(
    manifests: &[v4_state::LayerExecutionManifest],
    plan: v4_state::LayerStationaryRequestPlan,
) -> Result<usize, Box<dyn Error>> {
    let compressor_bytes = |ratio: usize, width: usize| {
        let coefficient = 1 + usize::from(ratio == 4);
        coefficient
            .checked_mul(ratio)
            .and_then(|slots| slots.checked_mul(coefficient))
            .and_then(|elements| elements.checked_mul(width))
            .and_then(|elements| elements.checked_mul(2 * std::mem::size_of::<f32>()))
            .ok_or("generic stationary audit mirror compressor ledger overflow")
    };
    let mut total = plan.token_scratch_bytes;
    for manifest in manifests {
        let ratio = manifest.compression_ratio.get();
        let compressed_rows = if ratio == 4 {
            plan.csa_compressed_rows
        } else {
            plan.hca_compressed_rows
        };
        let layer = plan
            .raw_kv_rows
            .checked_mul(v4_state::Q_HEAD_DIM)
            .and_then(|elements| elements.checked_mul(std::mem::size_of::<u16>()))
            .and_then(|bytes| {
                bytes.checked_add(
                    compressed_rows
                        .checked_mul(v4_state::Q_HEAD_DIM)
                        .and_then(|elements| elements.checked_mul(std::mem::size_of::<u16>()))?,
                )
            })
            .and_then(|bytes| {
                (ratio == 4)
                    .then(|| {
                        compressed_rows
                            .checked_mul(128)
                            .and_then(|elements| elements.checked_mul(std::mem::size_of::<u16>()))
                    })
                    .flatten()
                    .map_or(Some(bytes), |indexer| bytes.checked_add(indexer))
            })
            .and_then(|bytes| {
                bytes.checked_add(compressor_bytes(ratio, v4_state::Q_HEAD_DIM).ok()?)
            })
            .and_then(|bytes| {
                if ratio == 4 {
                    bytes.checked_add(compressor_bytes(4, 128).ok()?)
                } else {
                    Some(bytes)
                }
            })
            .ok_or("generic stationary audit mirror layer ledger overflow")?;
        total = total
            .checked_add(layer)
            .ok_or("generic stationary audit mirror ledger overflow")?;
    }
    Ok(total)
}

pub(crate) fn probe_mtp1_candidate_locals(
    probe: &mut StationaryMtpCapacityProbe,
    stream: &CudaStream,
) -> Result<(), Box<dyn Error>> {
    // Keep this list in the same ownership shape as run_mtp1_draft_candidate.
    // `q`, cache, sparse, and score scratch remain live until after the head.
    probe.u16(stream, v4_state::HIDDEN)?; // enorm
    probe.u16(stream, v4_state::HIDDEN)?; // hnorm
    probe.u16(stream, v4_state::Q_HEAD_DIM)?; // kv_ready
    probe.u16(stream, v4_state::Q_HEADS * v4_state::Q_HEAD_DIM)?; // q
    probe.u16(stream, v4_state::KV_WINDOW * v4_state::Q_HEAD_DIM)?; // cache
    probe.i32(stream, v4_state::KV_WINDOW)?; // topk
    probe.u16(stream, v4_state::Q_HEADS * v4_state::Q_HEAD_DIM)?; // sparse
    probe.f32(stream, 2 * v4_state::Q_HEADS * v4_state::KV_WINDOW)?; // scores/probabilities
    probe.f32(stream, v4_state::Q_HEADS)?; // denominators
    probe.u32(stream, v4_state::Q_HEADS)?; // valid_counts
    let bytes = MTP1_DRAFT_ENORM_HNORM_BYTES
        + MTP1_DRAFT_KV_READY_BYTES
        + MTP1_DRAFT_Q_BYTES
        + MTP1_DRAFT_CACHE_BYTES
        + MTP1_DRAFT_TOPK_BYTES
        + MTP1_DRAFT_SPARSE_BYTES
        + MTP1_DRAFT_SCORE_REUSE_BYTES;
    if bytes != MTP1_DRAFT_CANDIDATE_LOCAL_BYTES {
        return Err("MTP1 candidate-local allocation ledger drifted".into());
    }
    Ok(())
}

pub(crate) fn stationary_mtp1_capacity_preflight_rejects_execution_flags() -> Result<(), Box<dyn Error>> {
    for name in [
        "COLIBRI_V4_FULL61_B1",
        "COLIBRI_V4_LAYER_STATIONARY_PREFILL",
        "COLIBRI_V4_GENERIC_DEVICE_STATIONARY",
        "COLIBRI_V4_STATIONARY_MTP1_DRAFT",
        "COLIBRI_V4_MTP1_DRAFT_P0",
        "COLIBRI_V4_ROUTED_QD4",
        "COLIBRI_V4_PHASE_PROFILE",
        "COLIBRI_V4_RESULT_JSONL",
        "COLIBRI_V4_PROMPT_TEXT",
        "COLIBRI_V4_CHAT_USER",
        "COLIBRI_V4_TOKEN_IDS",
        "COLIBRI_V4_TOKEN_ID",
        "COLIBRI_V4_GREEDY_STEPS",
    ] {
        if std::env::var_os(name).is_some() {
            return Err(format!(
                "COLIBRI_V4_STATIONARY_MTP1_CAPACITY_PREFLIGHT=1 rejects execution flag {name}"
            )
            .into());
        }
    }
    Ok(())
}

pub(crate) fn stationary_mtp1_preflight_plan(
    prompt_rows: Option<&std::ffi::OsStr>,
    max_new_tokens: Option<&std::ffi::OsStr>,
) -> Result<v4_state::LayerStationaryRequestPlan, Box<dyn Error>> {
    let (prompt_rows, max_new_tokens) = match (prompt_rows, max_new_tokens) {
        (None, None) => (5, 3),
        (Some(prompt_rows), Some(max_new_tokens)) => {
            let parse = |name: &str, value: &std::ffi::OsStr| -> Result<usize, Box<dyn Error>> {
                value
                    .to_str()
                    .ok_or_else(|| format!("{name} must be valid UTF-8"))?
                    .parse::<usize>()
                    .map_err(|_| format!("{name} must be a base-10 usize").into())
            };
            (
                parse("COLIBRI_V4_STATIONARY_PREFLIGHT_PROMPT_ROWS", prompt_rows)?,
                parse(
                    "COLIBRI_V4_STATIONARY_PREFLIGHT_MAX_NEW_TOKENS",
                    max_new_tokens,
                )?,
            )
        }
        _ => {
            return Err(
                "COLIBRI_V4_STATIONARY_PREFLIGHT_PROMPT_ROWS and COLIBRI_V4_STATIONARY_PREFLIGHT_MAX_NEW_TOKENS must be set together"
                    .into(),
            );
        }
    };
    if prompt_rows == 0 {
        return Err("stationary preflight prompt rows must be nonzero".into());
    }
    v4_state::LayerStationaryRequestPlan::new(prompt_rows, max_new_tokens)
}

pub(crate) fn probe_generic_stationary_resources(
    probe: &mut StationaryMtpCapacityProbe,
    stream: &CudaStream,
    manifests: &[v4_state::LayerExecutionManifest],
    plan: v4_state::LayerStationaryRequestPlan,
    strict: bool,
) -> Result<(), Box<dyn Error>> {
    for manifest in manifests {
        let path = GenericStationaryPath::from_ratio(manifest.compression_ratio);
        probe.u16(stream, v4_state::KV_WINDOW * v4_state::Q_HEAD_DIM)?;
        probe.f32(stream, path.tail_elements())?;
        probe.f32(stream, path.tail_elements())?;
        probe.u16(stream, path.compressed_rows(plan) * v4_state::Q_HEAD_DIM)?;
        if path.is_csa() {
            probe.f32(stream, 8 * 256)?;
            probe.f32(stream, 8 * 256)?;
            probe.u16(stream, plan.csa_compressed_rows * 128)?;
        }
    }
    probe.generic_scratch = Some(DeviceRequestScratch::new_with_strict(stream, plan, strict)?);
    Ok(())
}

pub(crate) fn probe_qd4_lanes(
    probe: &mut StationaryMtpCapacityProbe,
    context: &std::sync::Arc<CudaContext>,
    stream: &CudaStream,
) -> Result<(), Box<dyn Error>> {
    for _ in 0..V4_QD4_LANES {
        for bytes in [
            v4_expert::V4_W1_BYTES,
            v4_expert::V4_W1_SCALE_BYTES,
            v4_expert::V4_W2_BYTES,
            v4_expert::V4_W2_SCALE_BYTES,
            v4_expert::V4_W1_BYTES,
            v4_expert::V4_W1_SCALE_BYTES,
        ] {
            probe
                .pinned_u8
                .push(PinnedHostBuffer::zeroed(context, bytes)?);
            probe.u8(stream, bytes)?;
        }
        probe
            .pinned_u16
            .push(PinnedHostBuffer::zeroed(context, v4_expert::V4_HIDDEN)?);
        probe
            .pinned_f32
            .push(PinnedHostBuffer::zeroed(context, v4_expert::V4_HIDDEN)?);
        probe.u16(stream, v4_expert::V4_HIDDEN)?;
        probe.f32(stream, v4_expert::V4_HIDDEN)?;
        for elements in [v4_expert::V4_EXPERT_INTERMEDIATE; 3] {
            probe.u16(stream, elements)?;
        }
        probe.f32(stream, v4_expert::V4_EXPERT_INTERMEDIATE)?;
        probe.u16(stream, v4_expert::V4_HIDDEN)?;
        probe.f32(stream, v4_expert::V4_HIDDEN)?;
    }
    Ok(())
}

/// Builds the same request-shared scratch constructor used by generic Full61
/// execution, but stops before any layer row or kernel launch.  Strict target
/// mode is explicit because this isolated process intentionally has no Full61
/// execution flags set.
pub(crate) fn run_stationary_mtp1_scratch_constructor_preflight() -> Result<bool, Box<dyn Error>> {
    if !optional_one_flag("COLIBRI_V4_STATIONARY_MTP1_SCRATCH_CONSTRUCTOR_PREFLIGHT")? {
        return Ok(false);
    }
    stationary_mtp1_capacity_preflight_rejects_execution_flags()?;
    let plan = stationary_mtp1_preflight_plan(
        std::env::var_os("COLIBRI_V4_STATIONARY_PREFLIGHT_PROMPT_ROWS").as_deref(),
        std::env::var_os("COLIBRI_V4_STATIONARY_PREFLIGHT_MAX_NEW_TOKENS").as_deref(),
    )?;
    let target_strict_compressor = true;
    let runtime_strict_compressor = strict_compressor_enabled();
    let pinned_cublas = std::env::var_os("COLIBRI_CUBLAS_EXACT_LIB").map(std::path::PathBuf::from);
    let expected_bytes = stationary_mtp1_target_strict_scratch_contract(
        plan,
        target_strict_compressor,
        pinned_cublas.as_deref(),
    )?;
    drop(cublas_exact::CublasExact::load()?);
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let scratch = DeviceRequestScratch::new_with_strict(&stream, plan, target_strict_compressor)?;
    let actual_bytes = scratch.num_bytes();
    if actual_bytes != expected_bytes {
        return Err("constructor preflight reached an inconsistent scratch total".into());
    }
    println!(
        "{}",
        serde_json::json!({
            "event": "stationary_mtp1_scratch_constructor_preflight",
            "model_execution": false,
            "layer_execution": false,
            "kernel_launches": 0,
            "metadata_shards_opened": 0,
            "tensor_payload_bytes_read": 0,
            "target": {"prompt_rows": plan.prompt_rows, "max_new_tokens": plan.max_new_tokens, "context_rows": plan.context_rows, "csa_compressed_rows": plan.csa_compressed_rows, "hca_compressed_rows": plan.hca_compressed_rows, "generic_device_stationary": true, "routed_qd4": true, "generic_state_audit": true, "stationary_mtp1_draft": true},
            "target_strict_compressor": target_strict_compressor,
            "runtime_strict_compressor": runtime_strict_compressor,
            "expected_bytes": expected_bytes,
            "actual_bytes": actual_bytes,
            "expected_breakdown": DeviceRequestScratchGeometry::new(plan, target_strict_compressor)?.byte_breakdown()?.json(),
            "actual_breakdown": scratch.byte_breakdown().json(),
        })
    );
    drop(scratch);
    stream.synchronize()?;
    Ok(true)
}
