use super::*;

pub(crate) struct MtpDraftRun {
    pub(crate) candidate: i64,
    pub(crate) routes: v4_router::RouterSelection,
    pub(crate) source_bytes: usize,
    pub(crate) routed_source_bytes: usize,
    pub(crate) source_read_ns: u64,
    pub(crate) h2d_ns: u64,
    pub(crate) h2d_bytes: usize,
    pub(crate) routed_h2d_bytes: usize,
    pub(crate) kernel_ns: u64,
    pub(crate) head_read_ns: u64,
    pub(crate) head_h2d_ns: u64,
    pub(crate) head_kernel_ns: u64,
    pub(crate) head_d2h_ns: u64,
    pub(crate) head_wall_ns: u64,
    pub(crate) total_ns: u64,
    pub(crate) persistent_bytes: usize,
    pub(crate) transient_expert_peak_bytes: usize,
    pub(crate) device_peak_bytes: usize,
    pub(crate) borrowed_qd4_lane: bool,
    pub(crate) kv_position: usize,
    pub(crate) kv_digest: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Mtp1Qd4LoanTelemetry {
    pub(crate) normal_lanes_before: usize,
    pub(crate) normal_lanes_after: usize,
    pub(crate) loan_depth_before: usize,
    pub(crate) loan_depth_peak: usize,
    pub(crate) loan_depth_final: usize,
    pub(crate) event_drained: bool,
}

pub(crate) struct Mtp1Qd4Attempt<T> {
    pub(crate) result: Result<T, Box<dyn Error>>,
    pub(crate) loan: Mtp1Qd4LoanTelemetry,
}

pub(crate) fn mtp1_qd4_telemetry_fields(
    route_ids: Option<&[i32]>,
    routed_source_bytes: Option<usize>,
    routed_h2d_bytes: Option<usize>,
    loan: Mtp1Qd4LoanTelemetry,
) -> Result<serde_json::Map<String, serde_json::Value>, Box<dyn Error>> {
    let mut fields = serde_json::Map::new();
    fields.insert(
        "qd4_normal_lanes_before".into(),
        serde_json::json!(loan.normal_lanes_before),
    );
    fields.insert(
        "qd4_normal_lanes_after".into(),
        serde_json::json!(loan.normal_lanes_after),
    );
    fields.insert(
        "qd4_loan_depth_before".into(),
        serde_json::json!(loan.loan_depth_before),
    );
    fields.insert(
        "qd4_loan_depth_peak".into(),
        serde_json::json!(loan.loan_depth_peak),
    );
    fields.insert(
        "qd4_loan_depth_final".into(),
        serde_json::json!(loan.loan_depth_final),
    );
    fields.insert(
        "qd4_loan_event_drained".into(),
        serde_json::json!(loan.event_drained),
    );
    if let Some(route_ids) = route_ids {
        let source_bytes =
            routed_source_bytes.ok_or("MTP routed source byte telemetry disappeared")?;
        let h2d_bytes = routed_h2d_bytes.ok_or("MTP routed H2D byte telemetry disappeared")?;
        if source_bytes % v4_expert::V4_ROUTED_EXPERT_BYTES != 0 {
            return Err("MTP routed source bytes do not contain whole expert loads".into());
        }
        if source_bytes != h2d_bytes {
            return Err("MTP routed source and H2D bytes diverged".into());
        }
        fields.insert("route_ids".into(), serde_json::json!(route_ids));
        fields.insert(
            "routed_expert_loads".into(),
            serde_json::json!(source_bytes / v4_expert::V4_ROUTED_EXPERT_BYTES),
        );
        fields.insert(
            "routed_source_bytes".into(),
            serde_json::json!(source_bytes),
        );
        fields.insert("routed_h2d_bytes".into(), serde_json::json!(h2d_bytes));
    } else if routed_source_bytes.is_some() || routed_h2d_bytes.is_some() {
        return Err("MTP routed bytes require route IDs".into());
    }
    Ok(fields)
}

pub(crate) fn append_mtp1_qd4_telemetry(
    record: &mut serde_json::Value,
    route_ids: Option<&[i32]>,
    routed_source_bytes: Option<usize>,
    routed_h2d_bytes: Option<usize>,
    loan: Mtp1Qd4LoanTelemetry,
) -> Result<(), Box<dyn Error>> {
    record
        .as_object_mut()
        .ok_or("MTP telemetry record must be a JSON object")?
        .extend(mtp1_qd4_telemetry_fields(
            route_ids,
            routed_source_bytes,
            routed_h2d_bytes,
            loan,
        )?);
    Ok(())
}

pub(crate) struct MtpDraftPrefixRun {
    pub(crate) persistent_bytes: usize,
    pub(crate) kv_position: usize,
    pub(crate) kv_digest: u64,
}

pub(crate) enum MtpDraftStep {
    Prefix(Result<MtpDraftPrefixRun, Box<dyn Error>>),
    Candidate(Result<MtpDraftRun, Box<dyn Error>>),
}

pub(crate) fn mtp1_requires_full_candidate(prompt_len: usize, token_position: usize) -> bool {
    token_position + 1 >= prompt_len
}

/// One official MTP-1 row consumes the teacher's final hidden row and token at
/// `candidate_pos`, appends its KV at that same position, then predicts the
/// teacher's token at `candidate_pos + 1`.  Keep this separate from the P0
/// score cap: stationary acceptance never changes which rows are causal.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Mtp1Alignment {
    pub(crate) input_pos: usize,
    pub(crate) teacher_hidden_pos: usize,
    pub(crate) kv_write_pos: usize,
    pub(crate) candidate_pos: usize,
    pub(crate) teacher_target_pos: usize,
    pub(crate) prompt_forced: bool,
}

pub(crate) fn mtp1_alignment(
    prompt_len: usize,
    candidate_pos: usize,
) -> Result<Mtp1Alignment, Box<dyn Error>> {
    if prompt_len == 0 {
        return Err("MTP1 alignment needs a nonempty prompt".into());
    }
    let teacher_target_pos = candidate_pos
        .checked_add(1)
        .ok_or("MTP1 candidate position overflow")?;
    Ok(Mtp1Alignment {
        input_pos: candidate_pos,
        teacher_hidden_pos: candidate_pos,
        kv_write_pos: candidate_pos,
        candidate_pos,
        teacher_target_pos,
        prompt_forced: teacher_target_pos < prompt_len,
    })
}

pub(crate) fn stationary_mtp1_enabled(generic_stationary: bool, flag: bool) -> Result<bool, Box<dyn Error>> {
    if flag && !generic_stationary {
        return Err("COLIBRI_V4_STATIONARY_MTP1_DRAFT requires generic device stationary".into());
    }
    Ok(flag)
}

pub(crate) fn mtp1_can_borrow_qd4_lane(worker_drained: bool, qd4_lanes: usize) -> bool {
    worker_drained && qd4_lanes == V4_QD4_LANES
}

pub(crate) fn run_with_mtp1_qd4_loan<T>(
    mut qd4_lanes: Option<&mut [ExpertSlot<'_>]>,
    worker_drained: bool,
    run: impl FnOnce(Option<&mut ExpertSlot<'_>>) -> Result<T, Box<dyn Error>>,
) -> Mtp1Qd4Attempt<T> {
    let normal_lanes_before = qd4_lanes.as_deref().map_or(0, <[ExpertSlot]>::len);
    let lanes_present = qd4_lanes.is_some();
    let borrowed = lanes_present && mtp1_can_borrow_qd4_lane(worker_drained, normal_lanes_before);
    let mut event_drained = !lanes_present;
    let result = if lanes_present && !borrowed {
        Err("stationary MTP candidate requires a drained normal four-lane QD4 set".into())
    } else if let Some(lanes) = qd4_lanes.as_deref_mut() {
        let lane = &mut lanes[0];
        let result = run(Some(lane));
        let drain = lane.finish_qd4_loan();
        event_drained = drain.is_ok();
        match (result, drain) {
            (Ok(value), Ok(true)) => Ok(value),
            (Ok(_), Ok(false)) => Err("stationary MTP QD4 loan event did not drain".into()),
            (Ok(_), Err(error)) => Err(error),
            (Err(error), Ok(_)) => Err(error),
            (Err(error), Err(drain)) => {
                Err(format!("{error}; stationary MTP QD4 loan drain also failed: {drain}").into())
            }
        }
    } else {
        run(None)
    };
    let normal_lanes_after = qd4_lanes.as_deref().map_or(0, <[ExpertSlot]>::len);
    Mtp1Qd4Attempt {
        result,
        loan: Mtp1Qd4LoanTelemetry {
            normal_lanes_before,
            normal_lanes_after,
            loan_depth_before: 0,
            loan_depth_peak: usize::from(borrowed),
            loan_depth_final: 0,
            event_drained,
        },
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn advance_mtp1_draft_state(
    module: &kernels::LoadedModule,
    stream: &std::sync::Arc<CudaStream>,
    model_dir: &std::path::Path,
    teacher_hc_hidden: &[u16],
    input_id: i64,
    start_pos: usize,
    state: &mut MtpDraftState,
) -> Result<MtpDraftPrefixRun, Box<dyn Error>> {
    if state.position != start_pos || teacher_hc_hidden.len() != v4_state::HC * v4_state::HIDDEN {
        return Err("MTP1 draft state position or teacher hidden-row contract mismatch".into());
    }
    let host = v4_state::load_mtp_draft_prefix_weights(model_dir)?;
    let weights = upload_mtp_draft_prefix_weights(stream, &host)?;
    let embedding = load_mtp_embedding_row(model_dir, input_id)?;
    let enorm = DeviceBuffer::from_host(stream, &host.enorm)?;
    let hnorm = DeviceBuffer::from_host(stream, &host.hnorm)?;
    let e_input = run_l0_rmsnorm(module, stream, &embedding, &enorm, v4_state::HIDDEN)?;
    let e_projection = host.e_proj.load(&host.shard)?;
    let e_projection = DeviceFp8Matrix::new(stream, &e_projection)?;
    let e = run_l0_fp8_linear(module, stream, &e_input, &e_projection, false)?.1;
    let h_input = run_l0_rmsnorm(module, stream, teacher_hc_hidden, &hnorm, v4_state::HIDDEN)?;
    let h_projection = host.h_proj.load(&host.shard)?;
    let h_projection = DeviceFp8Matrix::new(stream, &h_projection)?;
    let h = run_l0_fp8_linear(module, stream, &h_input, &h_projection, false)?.1;
    let mut residual = Vec::with_capacity(teacher_hc_hidden.len());
    for h_row in h.chunks_exact(v4_state::HIDDEN) {
        residual.extend(h_row.iter().zip(&e).map(|(&h, &e)| {
            v4_router::f32_to_bf16_rne(v4_router::bf16_to_f32(h) + v4_router::bf16_to_f32(e))
        }));
    }
    let attn_hc = run_l0_mhc_pre(
        module,
        stream,
        &residual,
        &weights.hc_attn_fn,
        &weights.hc_attn_scale,
        &weights.hc_attn_base,
    )?;
    let attn_input = run_l0_rmsnorm(
        module,
        stream,
        &attn_hc.collapsed,
        &weights.attn_norm,
        v4_state::HIDDEN,
    )?;
    let (_, kv_raw) = run_l0_fp8_linear(module, stream, &attn_input, &weights.wkv, false)?;
    let kv_norm = run_l0_rmsnorm(
        module,
        stream,
        &kv_raw,
        &weights.kv_norm,
        v4_state::Q_HEAD_DIM,
    )?;
    let freqs = v4_state::attention_rope_freqs(&v4_spec::PRO, v4_state::V4_LAYER_COUNT, start_pos)?;
    let kv_rope = run_l0_rope(module, stream, &kv_norm, &freqs, 1, false)?;
    let kv = run_l0_kv_quantize(module, stream, &kv_rope)?;
    let kv_ready = DeviceBuffer::from_host(stream, &kv)?;
    state.commit_kv_row(stream, &kv_ready)?;
    Ok(MtpDraftPrefixRun {
        persistent_bytes: state.persistent_bytes(),
        kv_position: state.position,
        kv_digest: state.digest(stream)?,
    })
}

pub(crate) fn load_mtp_embedding_row(
    model_dir: &std::path::Path,
    input_id: i64,
) -> Result<Vec<u16>, Box<dyn Error>> {
    v4_state::validate_ids(&[input_id])?;
    let shard = safetensors::ShardFile::open(model_dir.join("model-00001-of-00064.safetensors"))?;
    let tensor = shard.tensor("embed.weight")?;
    if tensor.dtype != safetensors::TensorDtype::Bf16
        || tensor.shape != [v4_state::VOCAB as u64, v4_state::HIDDEN as u64]
    {
        return Err("MTP shared embedding tensor contract mismatch".into());
    }
    let mut embedding = vec![0_u16; v4_state::HIDDEN];
    let bytes = unsafe {
        std::slice::from_raw_parts_mut(embedding.as_mut_ptr().cast(), embedding.len() * 2)
    };
    shard.read_at(
        "embed.weight",
        u64::try_from(usize::try_from(input_id)? * v4_state::HIDDEN * 2)?,
        bytes,
    )?;
    Ok(embedding)
}
