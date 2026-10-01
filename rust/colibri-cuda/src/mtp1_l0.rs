use super::*;

pub(crate) fn run_stationary_mtp1_capacity_preflight() -> Result<bool, Box<dyn Error>> {
    if !optional_one_flag("COLIBRI_V4_STATIONARY_MTP1_CAPACITY_PREFLIGHT")? {
        return Ok(false);
    }
    stationary_mtp1_capacity_preflight_rejects_execution_flags()?;
    let root = v4_model_dir()?;
    let manifests = v4_state::serial_layer_manifests(0)?;
    let plan = stationary_mtp1_preflight_plan(
        std::env::var_os("COLIBRI_V4_STATIONARY_PREFLIGHT_PROMPT_ROWS").as_deref(),
        std::env::var_os("COLIBRI_V4_STATIONARY_PREFLIGHT_MAX_NEW_TOKENS").as_deref(),
    )?;
    // This mode may be combined with the audit switch only to account for its
    // independent host mirror.  It still rejects every execution switch and
    // returns before loading a CUDA module or launching a model kernel.
    let generic_state_audit = optional_one_flag("COLIBRI_V4_GENERIC_DEVICE_STATE_AUDIT")?;
    let static_persistent = generic_stationary_persistent_reserve_bytes(&manifests, plan)?;
    let isolation_strict_compressor = strict_compressor_enabled();
    let target_strict_compressor = STATIONARY_MTP1_CAPACITY_PREFLIGHT_TARGET_STRICT;
    let pinned_cublas = std::env::var_os("COLIBRI_CUBLAS_EXACT_LIB").map(std::path::PathBuf::from);
    // Full61 is intentionally rejected by this isolated mode.  Its local
    // switch is therefore false, while the target capacity remains the strict
    // cuBLAS production configuration.
    if target_strict_compressor {
        drop(cublas_exact::CublasExact::load()?);
    }
    let static_scratch = stationary_mtp1_target_strict_scratch_contract(
        plan,
        target_strict_compressor,
        pinned_cublas.as_deref(),
    )?;
    let audit_mirror_host_bytes = generic_state_audit
        .then(|| generic_stationary_audit_mirror_host_bytes(&manifests, plan))
        .transpose()?
        .unwrap_or(0);
    let mtp_fixed = mtp1_draft_fixed_device_bytes_from_geometry()?;
    let mtp_projection =
        v4_state::HIDDEN * v4_state::HIDDEN + (v4_state::HIDDEN / v4_state::FP8_BLOCK).pow(2);
    let mtp_head = CERTIFIED_ARGMAX_DEVICE_TEMP_BYTES;
    let mtp_static_peak = mtp1_draft_peak_bytes(
        MTP1_DRAFT_PERSISTENT_BYTES,
        mtp_fixed,
        mtp_projection,
        mtp_projection,
        0,
        mtp_head,
    )?;
    let raw_plans =
        stationary_raw_layer_plans(&root, &(0..v4_state::V4_LAYER_COUNT).collect::<Vec<_>>())?;
    let requested_device = parse_gib_budget("COLIBRI_V4_STATIONARY_RAW_DEVICE_RETENTION_GIB")?;
    let requested_host = parse_gib_budget("COLIBRI_V4_STATIONARY_RAW_HOST_RETENTION_GIB")?;
    let raw_device_budget = requested_device
        .checked_sub(PINNED_FIXED_PIPELINE_WORKER_DEVICE_BYTES)
        .ok_or("capacity preflight raw device budget is below pinned worker need")?;
    let selected_device = select_fixed_cache_layers(
        &raw_plans,
        raw_device_budget,
        &vec![false; raw_plans.len()],
        true,
        true,
    );
    let selected_host =
        select_fixed_cache_layers(&raw_plans, requested_host, &selected_device, false, true);
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let device_before = device_available_bytes()?;
    let mut device_min = device_before;
    let mut probe = StationaryMtpCapacityProbe::default();
    probe_generic_stationary_resources(
        &mut probe,
        &stream,
        &manifests,
        plan,
        target_strict_compressor,
    )?;
    let generic_scratch_actual_bytes = probe
        .generic_scratch
        .as_ref()
        .ok_or("capacity preflight did not retain the generic request scratch")?
        .num_bytes();
    if generic_scratch_actual_bytes != static_scratch {
        return Err(format!(
            "generic stationary real scratch {generic_scratch_actual_bytes} != geometry ledger {static_scratch}"
        )
        .into());
    }
    let generic_probe_bytes = probe.device_bytes();
    let generic_ledger_bytes = static_persistent
        .checked_add(static_scratch)
        .ok_or("generic stationary capacity preflight ledger overflow")?;
    if generic_probe_bytes != generic_ledger_bytes {
        return Err(format!(
            "generic stationary typed preflight allocation {generic_probe_bytes} != geometry ledger {generic_ledger_bytes}"
        )
        .into());
    }
    probe_qd4_lanes(&mut probe, &context, &stream)?;
    // A stationary MTP candidate loans one already resident lane after the
    // normal QD4 request is drained.  The loan has no extra expert allocation;
    // its ownership depth is one and the normal four-lane configuration must
    // be intact immediately after the scoped candidate ends.
    let qd4_normal_lanes_before = qd4_lane_count(true);
    let qd4_loan_depth = usize::from(mtp1_can_borrow_qd4_lane(true, qd4_normal_lanes_before));
    if qd4_loan_depth != 1 {
        return Err(
            "MTP capacity preflight could not establish exactly one drained QD4 loan".into(),
        );
    }
    let loan_drained = context.new_event(None)?;
    loan_drained.record(&stream)?;
    loan_drained.synchronize()?;
    let qd4_normal_lanes_after = qd4_lane_count(true);
    if qd4_normal_lanes_after != qd4_normal_lanes_before {
        return Err("MTP capacity preflight QD4 loan changed normal teacher lane count".into());
    }
    let raw_generation_before = vec![0_u64; raw_plans.len()];
    for (selected, layer) in selected_device.iter().zip(&raw_plans) {
        if *selected {
            probe.u8(&stream, layer.raw_device_bytes)?;
        }
    }
    for (selected, layer) in selected_host.iter().zip(&raw_plans) {
        if *selected {
            probe.raw_host.push(vec![0_u8; layer.raw_host_bytes]);
        }
    }
    if audit_mirror_host_bytes != 0 {
        probe.host.push(vec![0_u8; audit_mirror_host_bytes]);
    }
    for _ in 0..PINNED_FIXED_PIPELINE_WORKER_SLOTS {
        probe.pinned_u8.push(PinnedHostBuffer::zeroed(
            &context,
            PHASE1_CSA_PAYLOAD_BYTES,
        )?);
        probe.u16(&stream, RAW_FIXED_SCRATCH_BYTES / 2)?;
    }
    stream.synchronize()?;
    device_min = device_min.min(device_available_bytes()?);
    let base_device_bytes = probe.device_bytes();
    let base_pinned_bytes = probe.pinned_bytes();
    let base_raw_host_bytes = probe.raw_host_bytes();
    let base_host_bytes = probe.host_bytes();
    // Two replies can be prepared at once.  Their decoded runtimes are owned
    // by PinnedFixedPrepared::lease, not by the pinned host buffers.  They are
    // dropped/restored before the MTP cutpoint, so this is a separate maximum
    // rather than an MTP overlap.
    let prepared_runtimes = (0..PINNED_FIXED_PIPELINE_WORKER_SLOTS)
        .map(|_| DeviceBuffer::<u8>::zeroed(&stream, PHASE1_CSA_PAYLOAD_BYTES))
        .collect::<Result<Vec<_>, _>>()?;
    stream.synchronize()?;
    device_min = device_min.min(device_available_bytes()?);
    let worker_prepared_peak_bytes = base_device_bytes
        .checked_add(
            prepared_runtimes
                .iter()
                .map(DeviceBuffer::num_bytes)
                .sum::<usize>(),
        )
        .ok_or("stationary MTP worker prepared peak overflow")?;
    if worker_prepared_peak_bytes - base_device_bytes
        != PINNED_FIXED_PIPELINE_WORKER_PREPARED_DEVICE_BYTES
    {
        return Err("stationary MTP worker prepared-runtime ledger drifted".into());
    }
    drop(prepared_runtimes);
    stream.synchronize()?;
    let mtp_persistent = DeviceBuffer::<u16>::zeroed(&stream, MTP1_DRAFT_PERSISTENT_BYTES / 2)?;
    probe_mtp1_candidate_locals(&mut probe, &stream)?;
    let candidate_local_device_bytes = probe
        .device_bytes()
        .checked_sub(base_device_bytes)
        .ok_or("MTP candidate-local typed allocation underflow")?;
    if candidate_local_device_bytes != MTP1_DRAFT_CANDIDATE_LOCAL_BYTES {
        return Err(format!("MTP candidate-local typed allocation {candidate_local_device_bytes} != ledger {MTP1_DRAFT_CANDIDATE_LOCAL_BYTES}").into());
    }
    // The streamed head holds pageable chunk staging in addition to its device
    // buffers; account for it explicitly because it is request-local.
    probe
        .host
        .push(vec![0_u8; CERTIFIED_ARGMAX_HOST_TEMP_BYTES]);
    let projection = DeviceBuffer::<u8>::zeroed(&stream, mtp_projection)?;
    stream.synchronize()?;
    device_min = device_min.min(device_available_bytes()?);
    let projection_peak_bytes =
        probe.device_bytes() + mtp_persistent.num_bytes() + projection.num_bytes();
    drop(projection);
    let fixed = DeviceBuffer::<u8>::zeroed(&stream, mtp_fixed)?;
    stream.synchronize()?;
    device_min = device_min.min(device_available_bytes()?);
    let fixed_peak_bytes = probe.device_bytes() + mtp_persistent.num_bytes() + fixed.num_bytes();
    drop(fixed);
    let head_input = DeviceBuffer::<u16>::zeroed(&stream, v4_state::HIDDEN)?;
    let head_weight =
        DeviceBuffer::<u16>::zeroed(&stream, CERTIFIED_ARGMAX_CHUNK_ROWS * v4_state::HIDDEN)?;
    let head_output = DeviceBuffer::<f32>::zeroed(&stream, CERTIFIED_ARGMAX_CHUNK_ROWS)?;
    let head_sumabs = DeviceBuffer::<f32>::zeroed(&stream, CERTIFIED_ARGMAX_CHUNK_ROWS)?;
    let head_status = DeviceBuffer::<u32>::zeroed(&stream, CERTIFIED_ARGMAX_CHUNK_ROWS)?;
    stream.synchronize()?;
    device_min = device_min.min(device_available_bytes()?);
    let head_peak_bytes = probe.device_bytes()
        + mtp_persistent.num_bytes()
        + head_input.num_bytes()
        + head_weight.num_bytes()
        + head_output.num_bytes()
        + head_sumabs.num_bytes()
        + head_status.num_bytes();
    let measured_mtp_peak = projection_peak_bytes
        .max(fixed_peak_bytes)
        .max(head_peak_bytes)
        - base_device_bytes;
    if measured_mtp_peak != mtp_static_peak {
        return Err(format!("MTP capacity preflight static peak {mtp_static_peak} != typed-allocation peak {measured_mtp_peak}").into());
    }
    let raw_generation_after = vec![0_u64; raw_plans.len()];
    if raw_generation_after != raw_generation_before {
        return Err("MTP capacity preflight changed raw-cache generation without a request".into());
    }
    drop((
        head_input,
        head_weight,
        head_output,
        head_sumabs,
        head_status,
        mtp_persistent,
        probe,
    ));
    stream.synchronize()?;
    let device_after = device_available_bytes()?;
    if device_after != device_before {
        return Err(format!("stationary MTP capacity preflight device recovery mismatch: before={device_before} after={device_after}").into());
    }
    let static_total_peak_bytes = worker_prepared_peak_bytes.max(
        base_device_bytes
            .checked_add(mtp_static_peak)
            .ok_or("stationary MTP capacity preflight total peak overflow")?,
    );
    let measured_total_peak_bytes = device_before
        .checked_sub(device_min)
        .ok_or("stationary MTP capacity preflight free-memory underflow")?;
    let allocator_fragmentation_bytes = measured_total_peak_bytes
        .checked_sub(static_total_peak_bytes)
        .ok_or("stationary MTP capacity preflight measured peak fell below typed ledger")?;
    println!(
        "{}",
        serde_json::json!({
            "event":"stationary_mtp1_capacity_preflight", "schema":STATIONARY_MTP1_CAPACITY_PREFLIGHT_SCHEMA, "model_execution":false, "layer_execution":false, "kernel_launches":0, "metadata_shards_opened":raw_plans.len(), "tensor_payload_bytes_read":0,
            "request_plan":{"prompt_rows":plan.prompt_rows,"max_new_tokens":plan.max_new_tokens,"context_rows":plan.context_rows,"csa_compressed_rows":plan.csa_compressed_rows,"hca_compressed_rows":plan.hca_compressed_rows},
            "generic_persistent_bytes":static_persistent, "generic_scratch_bytes":static_scratch, "generic_scratch_actual_bytes":generic_scratch_actual_bytes, "generic_scratch_instances":1, "target_strict_compressor":target_strict_compressor, "runtime_isolation_strict_compressor":isolation_strict_compressor,
            "allocation_order":["generic_persistent_and_real_request_scratch","qd4_lanes","raw_device_retention","raw_host_retention","fixed_worker_scratch","prepared_runtimes_separate_peak_then_drop","mtp_persistent_and_candidate_locals","projection_fixed_head_mutually_exclusive_peaks"],
            "recovery_order":["prepared_runtimes_drop_before_mtp","projection_drop","fixed_drop","head_mtp_and_capacity_probe_drop","stream_synchronize","device_free_exact_check"],
            "qd4_device_bytes":V4_QD4_DEVICE_BYTES, "qd4_pinned_host_bytes":V4_QD4_PINNED_HOST_BYTES,
            "worker_device_bytes":PINNED_FIXED_PIPELINE_WORKER_DEVICE_BYTES, "worker_scratch_device_bytes":PINNED_FIXED_PIPELINE_WORKER_SCRATCH_DEVICE_BYTES, "worker_prepared_runtime_device_bytes":PINNED_FIXED_PIPELINE_WORKER_PREPARED_DEVICE_BYTES, "worker_pinned_host_bytes":PINNED_FIXED_PIPELINE_WORKER_PINNED_HOST_BYTES,
            "raw_device_selected_bytes":raw_plans.iter().zip(&selected_device).filter_map(|(plan, selected)| selected.then_some(plan.raw_device_bytes)).sum::<usize>(),
            "raw_host_selected_bytes":base_raw_host_bytes, "raw_device_selected_layers":selected_device.iter().filter(|&&v|v).count(), "raw_host_selected_layers":selected_host.iter().filter(|&&v|v).count(),
            "mtp":{"persistent_bytes":MTP1_DRAFT_PERSISTENT_BYTES,"candidate_local_bytes":MTP1_DRAFT_CANDIDATE_LOCAL_BYTES,"candidate_locals":{"enorm_hnorm":MTP1_DRAFT_ENORM_HNORM_BYTES,"kv_ready":MTP1_DRAFT_KV_READY_BYTES,"q":MTP1_DRAFT_Q_BYTES,"cache":MTP1_DRAFT_CACHE_BYTES,"topk":MTP1_DRAFT_TOPK_BYTES,"sparse":MTP1_DRAFT_SPARSE_BYTES,"score_reuse":MTP1_DRAFT_SCORE_REUSE_BYTES},"projection_peak_bytes":mtp_projection+MTP1_DRAFT_PERSISTENT_BYTES+MTP1_DRAFT_CANDIDATE_LOCAL_BYTES,"fixed_peak_bytes":mtp_fixed+MTP1_DRAFT_PERSISTENT_BYTES+MTP1_DRAFT_CANDIDATE_LOCAL_BYTES,"head_peak_bytes":mtp_head+MTP1_DRAFT_PERSISTENT_BYTES+MTP1_DRAFT_CANDIDATE_LOCAL_BYTES,"head_host_staging_bytes":CERTIFIED_ARGMAX_HOST_TEMP_BYTES,"measured_peak_bytes":measured_mtp_peak,"borrowed_qd4_expert_incremental_bytes":0,"qd4_loan_depth":qd4_loan_depth,"qd4_loan_event_drained":true,"qd4_normal_lanes_before":qd4_normal_lanes_before,"qd4_normal_lanes_after":qd4_normal_lanes_after,"head_streams_per_candidate":2},
            "worker_prepared_peak_bytes":worker_prepared_peak_bytes,"mtp_prepared_runtime_overlap_bytes":0,"base_device_bytes":base_device_bytes,"base_pinned_host_bytes":base_pinned_bytes,"base_host_bytes":base_host_bytes,"generic_audit_mirror_host_bytes":audit_mirror_host_bytes,"raw_cache_generation_before":raw_generation_before,"raw_cache_generation_after":raw_generation_after,"static_total_peak_bytes":static_total_peak_bytes,"measured_total_peak_bytes":measured_total_peak_bytes,"allocator_fragmentation_bytes":allocator_fragmentation_bytes,"device_free_before":device_before,"device_free_min":device_min,"device_free_after":device_after,"recovered_exact":true
        })
    );
    Ok(true)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn step_mtp1_draft_candidate(
    context: &std::sync::Arc<CudaContext>,
    module: &kernels::LoadedModule,
    stream: &std::sync::Arc<CudaStream>,
    model_dir: &std::path::Path,
    final_head: &v4_state::FinalHeadWeights,
    mut borrowed_qd4_lane: Option<&mut ExpertSlot<'_>>,
    capacity_limit: Option<usize>,
    teacher_hc_hidden: &[u16],
    input_id: i64,
    start_pos: usize,
    state: &mut MtpDraftState,
) -> Result<MtpDraftRun, Box<dyn Error>> {
    if state.position != start_pos || teacher_hc_hidden.len() != v4_state::HC * v4_state::HIDDEN {
        return Err("MTP1 draft state position or teacher hidden-row contract mismatch".into());
    }
    let started = std::time::Instant::now();
    let source_started = std::time::Instant::now();
    let host = v4_state::load_mtp_draft_weights(model_dir)?;
    let mut source_read_ns = u64::try_from(source_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
    let mut h2d_ns = 0_u64;
    let mut h2d_bytes = 0_usize;
    let embedding_started = std::time::Instant::now();
    let embedding = load_mtp_embedding_row(model_dir, input_id)?;
    source_read_ns = source_read_ns
        .saturating_add(u64::try_from(embedding_started.elapsed().as_nanos()).unwrap_or(u64::MAX));
    let enorm = DeviceBuffer::from_host(stream, &host.enorm)?;
    let hnorm = DeviceBuffer::from_host(stream, &host.hnorm)?;
    let e_input = step_l0_rmsnorm(module, stream, &embedding, &enorm, v4_state::HIDDEN)?;
    let (e, e_projection_bytes) = {
        let projection_read_started = std::time::Instant::now();
        let projection = host.e_proj.load(&host.shard)?;
        source_read_ns = source_read_ns.saturating_add(
            u64::try_from(projection_read_started.elapsed().as_nanos()).unwrap_or(u64::MAX),
        );
        let projection_h2d_started = std::time::Instant::now();
        let projection = DeviceFp8Matrix::new(stream, &projection)?;
        stream.synchronize()?;
        h2d_ns = h2d_ns.saturating_add(
            u64::try_from(projection_h2d_started.elapsed().as_nanos()).unwrap_or(u64::MAX),
        );
        let bytes = projection.num_bytes();
        h2d_bytes = h2d_bytes.saturating_add(bytes);
        (
            step_l0_fp8_linear(module, stream, &e_input, &projection, false)?.1,
            bytes,
        )
    };
    let h_input = step_l0_rmsnorm(module, stream, teacher_hc_hidden, &hnorm, v4_state::HIDDEN)?;
    let (h, h_projection_bytes) = {
        let projection_read_started = std::time::Instant::now();
        let projection = host.h_proj.load(&host.shard)?;
        source_read_ns = source_read_ns.saturating_add(
            u64::try_from(projection_read_started.elapsed().as_nanos()).unwrap_or(u64::MAX),
        );
        let projection_h2d_started = std::time::Instant::now();
        let projection = DeviceFp8Matrix::new(stream, &projection)?;
        stream.synchronize()?;
        h2d_ns = h2d_ns.saturating_add(
            u64::try_from(projection_h2d_started.elapsed().as_nanos()).unwrap_or(u64::MAX),
        );
        let bytes = projection.num_bytes();
        h2d_bytes = h2d_bytes.saturating_add(bytes);
        (
            step_l0_fp8_linear(module, stream, &h_input, &projection, false)?.1,
            bytes,
        )
    };
    // e/h projections have drained before the full MTP block is resident.
    let fixed_h2d_started = std::time::Instant::now();
    let weights = upload_mtp_draft_weights(module, stream, &host)?;
    stream.synchronize()?;
    let fixed_h2d_ns = u64::try_from(fixed_h2d_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
    h2d_ns = h2d_ns.saturating_add(fixed_h2d_ns);
    h2d_bytes = h2d_bytes.saturating_add(weights.num_bytes());
    let peak_bytes = mtp1_draft_peak_bytes(
        state.persistent_bytes(),
        weights.num_bytes(),
        e_projection_bytes,
        h_projection_bytes,
        usize::from(borrowed_qd4_lane.is_none()) * MTP1_DRAFT_TRANSIENT_EXPERT_SLOT_BYTES,
        CERTIFIED_ARGMAX_DEVICE_TEMP_BYTES,
    )?;
    if capacity_limit.is_some_and(|limit| peak_bytes > limit) {
        return Err(format!(
            "MTP1 draft peak {peak_bytes} exceeds capacity {}",
            capacity_limit.expect("checked above")
        )
        .into());
    }
    let mut residual = Vec::with_capacity(teacher_hc_hidden.len());
    for h_row in h.chunks_exact(v4_state::HIDDEN) {
        residual.extend(h_row.iter().zip(&e).map(|(&h, &e)| {
            v4_router::f32_to_bf16_rne(v4_router::bf16_to_f32(h) + v4_router::bf16_to_f32(e))
        }));
    }
    let attn_hc = step_l0_mhc_pre(
        module,
        stream,
        &residual,
        &weights.hc_attn_fn,
        &weights.hc_attn_scale,
        &weights.hc_attn_base,
    )?;
    let attn_input = step_l0_rmsnorm(
        module,
        stream,
        &attn_hc.collapsed,
        &weights.attn_norm,
        v4_state::HIDDEN,
    )?;
    let (_, q_lora_raw) = step_l0_fp8_linear(module, stream, &attn_input, &weights.wq_a, false)?;
    let q_lora = step_l0_rmsnorm(
        module,
        stream,
        &q_lora_raw,
        &weights.q_norm,
        weights.wq_a.out_features,
    )?;
    let (_, q_raw) = step_l0_fp8_linear(module, stream, &q_lora, &weights.wq_b, true)?;
    let empty_norm = DeviceBuffer::<u16>::zeroed(stream, 0)?;
    let q_norm = step_l0_rmsnorm(module, stream, &q_raw, &empty_norm, v4_state::Q_HEAD_DIM)?;
    let freqs = v4_state::attention_rope_freqs(&v4_spec::PRO, v4_state::V4_LAYER_COUNT, start_pos)?;
    let q = step_l0_rope(module, stream, &q_norm, &freqs, v4_state::Q_HEADS, false)?;
    let (_, kv_raw) = step_l0_fp8_linear(module, stream, &attn_input, &weights.wkv, false)?;
    let kv_norm = step_l0_rmsnorm(
        module,
        stream,
        &kv_raw,
        &weights.kv_norm,
        v4_state::Q_HEAD_DIM,
    )?;
    let kv_rope = step_l0_rope(module, stream, &kv_norm, &freqs, 1, false)?;
    let kv = step_l0_kv_quantize(module, stream, &kv_rope)?;
    let kv_ready = DeviceBuffer::from_host(stream, &kv)?;
    let q = DeviceBuffer::from_host(stream, &q)?;
    let cache = DeviceBuffer::zeroed(stream, v4_state::KV_WINDOW * v4_state::Q_HEAD_DIM)?;
    unsafe {
        copy_device_range(
            stream,
            &cache,
            0,
            &state.kv_cache,
            0,
            v4_state::KV_WINDOW * v4_state::Q_HEAD_DIM,
        )?;
        copy_device_row(
            stream,
            &cache,
            v4_state::mtp1_swa::mtp1_swa_attention_state_plan(v4_state::mtp1_swa::MTP1_SWA_LAYER)?
                .slot_for_position(start_pos)?,
            &kv_ready,
        )?;
    }
    let (topk, _) = v4_state::window_topk_cpu(1, start_pos, v4_state::KV_WINDOW)?;
    let topk = DeviceBuffer::from_host(stream, &topk)?;
    let mut sparse = DeviceBuffer::zeroed(stream, v4_state::Q_HEADS * v4_state::Q_HEAD_DIM)?;
    let mut sparse_scratch = SparseAttentionScoreReuseScratch::new(stream, 1, topk.len())?;
    run_device_raw_window_attention(
        module,
        stream,
        &q,
        &cache,
        &weights.attn_sink,
        &topk,
        &mut sparse,
        &mut sparse_scratch,
    )?;
    let sparse = sparse.to_host_vec(stream)?;
    let sparse = step_l0_rope(module, stream, &sparse, &freqs, v4_state::Q_HEADS, true)?;
    let wo_a = run_l0_wo_a(module, stream, &sparse, &weights.wo_a)?;
    let (_, attention) = step_l0_fp8_linear(module, stream, &wo_a, &weights.wo_b, false)?;
    let after_attention = step_l0_hc_post(
        module,
        stream,
        &attention,
        &residual,
        &attn_hc.post,
        &attn_hc.comb,
    )?;
    let ffn_hc = step_l0_mhc_pre(
        module,
        stream,
        &after_attention,
        &weights.hc_ffn_fn,
        &weights.hc_ffn_scale,
        &weights.hc_ffn_base,
    )?;
    let router = step_layer_router(
        module,
        stream,
        None,
        &ffn_hc.collapsed,
        &weights,
        &[],
        false,
    )?;
    let shared = step_layer_shared_expert(module, stream, &router.input, &weights)?;
    let mut contributions = Vec::with_capacity(v4_router::V4_TOPK);
    let mut expert_bytes = 0_usize;
    let mut expert_ids = router
        .routes
        .ids
        .iter()
        .map(|&route| u16::try_from(route))
        .collect::<Result<Vec<_>, _>>()?;
    expert_ids.sort_unstable();
    expert_ids.dedup();
    for expert_id in expert_ids {
        if let Some(slot) = borrowed_qd4_lane.as_deref_mut() {
            let plan = v4_expert::RoutedExpertRangePlan::new_mtp(&host.shard, expert_id)?;
            for (&route, &weight) in router.routes.ids.iter().zip(&router.routes.weights) {
                if u16::try_from(route)? == expert_id {
                    let mut output = vec![0.0_f32; v4_expert::V4_HIDDEN];
                    let lane = slot.load_range_execute_into(
                        &host.shard,
                        &plan,
                        &router.input,
                        weight,
                        &mut output,
                        None,
                        None,
                    )?;
                    source_read_ns = source_read_ns.saturating_add(lane.read_ns);
                    h2d_ns = h2d_ns.saturating_add(lane.h2d_ns);
                    expert_bytes = expert_bytes.saturating_add(v4_expert::V4_ROUTED_EXPERT_BYTES);
                    h2d_bytes = h2d_bytes.saturating_add(v4_expert::V4_ROUTED_EXPERT_BYTES);
                    contributions.push((expert_id, output));
                }
            }
        } else {
            let expert = v4_expert::RoutedExpert::load_mtp(&host.shard, expert_id)?;
            source_read_ns = source_read_ns.saturating_add((expert.read_ms * 1_000_000.0) as u64);
            expert_bytes = expert_bytes.saturating_add(expert.h2d_bytes());
            let mut slot = ExpertSlot::new(context, module, stream, &expert)?;
            h2d_ns = h2d_ns.saturating_add((slot.weight_h2d_ms * 1_000_000.0) as u64);
            h2d_bytes = h2d_bytes.saturating_add(expert.h2d_bytes());
            for (&route, &weight) in router.routes.ids.iter().zip(&router.routes.weights) {
                if u16::try_from(route)? == expert_id {
                    contributions.push((expert_id, slot.execute(&router.input, weight)?.0));
                }
            }
        }
    }
    contributions.sort_unstable_by_key(|(expert, _)| *expert);
    let routed = contributions
        .iter()
        .map(|(expert, output)| (*expert, output.as_slice()))
        .collect::<Vec<_>>();
    let moe = v4_expert::reduce_moe_canonical(&routed, &shared)?;
    let output = step_l0_hc_post(
        module,
        stream,
        &moe,
        &after_attention,
        &ffn_hc.post,
        &ffn_hc.comb,
    )?;
    let pre = v4_state::final_hc_gate_cpu(
        &output,
        1,
        v4_state::HC,
        v4_state::HIDDEN,
        &host.hc_head_fn,
        host.hc_head_scale,
        &host.hc_head_base,
    )?;
    let normalized = v4_state::final_hc_head_cpu(
        &output,
        1,
        v4_state::HC,
        v4_state::HIDDEN,
        &pre,
        &host
            .norm
            .iter()
            .map(|&bits| v4_router::bf16_to_f32(bits))
            .collect::<Vec<_>>(),
        &vec![0.0; v4_state::HIDDEN],
        1,
    )?
    .normalized_bf16;
    // The full MTP block and routed slot are no longer needed by the streamed
    // LM head, so the head has no fixed-weight overlap.
    drop(weights);
    let head =
        run_streamed_lm_head_gpu_normalized(context, module, stream, final_head, &normalized)?;
    let head_read_ns = head.read_ns;
    source_read_ns = source_read_ns.saturating_add(head.read_ns);
    h2d_ns = h2d_ns.saturating_add(head.h2d_ns);
    h2d_bytes = h2d_bytes.saturating_add(
        v4_state::HIDDEN * std::mem::size_of::<u16>()
            + v4_state::VOCAB * v4_state::HIDDEN * std::mem::size_of::<u16>(),
    );
    state.commit_kv_row(stream, &kv_ready)?;
    let total_ns = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
    Ok(MtpDraftRun {
        candidate: i64::try_from(head.candidate)?,
        routes: router.routes,
        source_bytes: host
            .fixed_source_bytes
            .saturating_add(host.e_proj.source_bytes())
            .saturating_add(host.h_proj.source_bytes())
            .saturating_add(expert_bytes)
            .saturating_add(v4_state::HIDDEN * std::mem::size_of::<u16>())
            .saturating_add(v4_state::VOCAB * v4_state::HIDDEN * std::mem::size_of::<u16>()),
        routed_source_bytes: expert_bytes,
        source_read_ns,
        h2d_ns,
        h2d_bytes,
        routed_h2d_bytes: expert_bytes,
        kernel_ns: total_ns
            .saturating_sub(source_read_ns)
            .saturating_sub(h2d_ns)
            .saturating_sub(head.d2h_ns),
        head_read_ns,
        head_h2d_ns: head.h2d_ns,
        head_kernel_ns: head.kernel_ns,
        head_d2h_ns: head.d2h_ns,
        head_wall_ns: head.wall_ns,
        total_ns,
        persistent_bytes: state.persistent_bytes(),
        transient_expert_peak_bytes: usize::from(borrowed_qd4_lane.is_none())
            * MTP1_DRAFT_TRANSIENT_EXPERT_SLOT_BYTES,
        device_peak_bytes: peak_bytes,
        borrowed_qd4_lane: borrowed_qd4_lane.is_some(),
        kv_position: state.position,
        kv_digest: state.digest(stream)?,
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_real_l0_rows(
    context: &std::sync::Arc<CudaContext>,
    module: &kernels::LoadedModule,
    stream: &std::sync::Arc<CudaStream>,
    weights: &LayerDeviceWeights,
    expert_shard: &safetensors::ShardFile,
    embedding: &[u16],
    freqs: &[f32],
    hash_ids: &[i64],
    start_pos: usize,
    base: &L0Persistent,
) -> Result<L0Run, Box<dyn Error>> {
    let rows = embedding.len() / v4_state::HIDDEN;
    if rows == 0
        || rows > 2
        || embedding.len() != rows * v4_state::HIDDEN
        || freqs.len() != rows * v4_state::Q_ROPE_DIM
        || (weights.hash_router && hash_ids.len() != rows * v4_router::V4_TOPK)
        || (!weights.hash_router && !hash_ids.is_empty())
    {
        return Err(format!(
            "layer {} runner accepts only complete K<=2 rows with router-mode-matched ids",
            weights.layer
        )
        .into());
    }
    if weights.compression_ratio == v4_state::CompressionRatio::Csa4 {
        return Err(format!(
            "layer {} CSA semantic gate: indexer q/weight projection, indexer compressor commit, and learned compressed-KV selection are not yet joined to the common runner",
            weights.layer
        )
        .into());
    }
    if start_pos / weights.compression_ratio.get()
        != (start_pos + rows) / weights.compression_ratio.get()
    {
        return Err(format!(
            "layer {} HCA semantic gate: compression-boundary pooled KV finalize/norm/RoPE/quantized cache commit is not yet joined to the common runner",
            weights.layer
        )
        .into());
    }
    let started = std::time::Instant::now();
    let mut residual = Vec::with_capacity(rows * v4_state::HC * v4_state::HIDDEN);
    for row in embedding.chunks_exact(v4_state::HIDDEN) {
        for _ in 0..v4_state::HC {
            residual.extend_from_slice(row);
        }
    }

    let attn_hc = step_l0_mhc_pre(
        module,
        stream,
        &residual,
        &weights.hc_attn_fn,
        &weights.hc_attn_scale,
        &weights.hc_attn_base,
    )?;
    let attn_input = step_l0_rmsnorm(
        module,
        stream,
        &attn_hc.collapsed,
        &weights.attn_norm,
        v4_state::HIDDEN,
    )?;

    let (_, q_lora_raw) = step_l0_fp8_linear(module, stream, &attn_input, &weights.wq_a, false)?;
    maybe_write_v4_audit_u16(weights.layer, "q_lora_raw", &q_lora_raw)?;
    let q_lora = step_l0_rmsnorm(
        module,
        stream,
        &q_lora_raw,
        &weights.q_norm,
        weights.wq_a.out_features,
    )?;
    let (_, q_raw) = step_l0_fp8_linear(module, stream, &q_lora, &weights.wq_b, true)?;
    maybe_write_v4_audit_u16(weights.layer, "q_raw", &q_raw)?;
    let unit_norm = DeviceBuffer::<u16>::zeroed(stream, 0)?;
    let q_norm = step_l0_rmsnorm(module, stream, &q_raw, &unit_norm, v4_state::Q_HEAD_DIM)?;
    maybe_write_v4_audit_u16(weights.layer, "q_norm", &q_norm)?;
    let q = step_l0_rope(module, stream, &q_norm, freqs, v4_state::Q_HEADS, false)?;

    let (_, kv_raw) = step_l0_fp8_linear(module, stream, &attn_input, &weights.wkv, false)?;
    let kv_norm = step_l0_rmsnorm(
        module,
        stream,
        &kv_raw,
        &weights.kv_norm,
        v4_state::Q_HEAD_DIM,
    )?;
    let kv_rope = step_l0_rope(module, stream, &kv_norm, freqs, 1, false)?;
    let kv = step_l0_kv_quantize(module, stream, &kv_rope)?;
    let compressor =
        step_layer_compressor(module, stream, &attn_input, weights, start_pos, None, None)?;

    let mut persistent = base.clone();
    let mut prefixes = Vec::with_capacity(rows);
    for row in 0..rows {
        let kv_row = &kv[row * v4_state::Q_HEAD_DIM..][..v4_state::Q_HEAD_DIM];
        persistent.kv = persistent.kv.prepare_decode_row(kv_row, start_pos + row)?;
        let range = row * v4_state::Q_HEAD_DIM..(row + 1) * v4_state::Q_HEAD_DIM;
        persistent.compressor = persistent.compressor.prepare_decode_row(
            &v4_state::CompressorPrepared {
                kv_f32: compressor.kv_f32[range.clone()].to_vec(),
                score_f32: compressor.score_f32[range.clone()].to_vec(),
                score_plus_ape_f32: compressor.score_plus_ape_f32[range].to_vec(),
            },
            start_pos + row,
        )?;
        prefixes.push(persistent.clone());
    }

    let sparse = run_l0_sparse_attention(
        module,
        stream,
        &q,
        persistent.kv.as_slice(),
        &weights.attn_sink,
        start_pos,
    )?;
    let sparse = step_l0_rope(module, stream, &sparse, freqs, v4_state::Q_HEADS, true)?;
    let wo_a = run_l0_wo_a(module, stream, &sparse, &weights.wo_a)?;
    maybe_write_v4_audit_u16(weights.layer, "wo_a", &wo_a)?;
    let (_, attention) = step_l0_fp8_linear(module, stream, &wo_a, &weights.wo_b, false)?;
    maybe_write_v4_audit_u16(weights.layer, "attention", &attention)?;
    let after_attention = step_l0_hc_post(
        module,
        stream,
        &attention,
        &residual,
        &attn_hc.post,
        &attn_hc.comb,
    )?;
    maybe_write_v4_audit_u16(weights.layer, "attention_post", &after_attention)?;

    let ffn_hc = step_l0_mhc_pre(
        module,
        stream,
        &after_attention,
        &weights.hc_ffn_fn,
        &weights.hc_ffn_scale,
        &weights.hc_ffn_base,
    )?;
    let router = step_layer_router(
        module,
        stream,
        None,
        &ffn_hc.collapsed,
        weights,
        hash_ids,
        false,
    )?;
    let ffn_input = router.input;
    let routes = router.routes;
    maybe_write_v4_audit_u16(weights.layer, "ffn_input", &ffn_input)?;
    let shared = step_layer_shared_expert(module, stream, &ffn_input, weights)?;

    let mut expert_union = routes
        .ids
        .iter()
        .map(|&id| u16::try_from(id))
        .collect::<Result<Vec<_>, _>>()?;
    expert_union.sort_unstable();
    expert_union.dedup();
    let mut contributions: Vec<Vec<Option<Vec<f32>>>> = (0..rows)
        .map(|_| (0..v4_router::V4_TOPK).map(|_| None).collect())
        .collect();
    let mut expert_bytes = 0;
    let mut expert_alloc_ms = 0.0;
    let mut expert_read_ms = 0.0;
    let mut expert_setup_ms = 0.0;
    let mut expert_setup_samples_ms = Vec::with_capacity(expert_union.len());
    let mut expert_h2d_samples_ms = Vec::with_capacity(expert_union.len());
    let mut expert_kernel_ms = 0.0;
    for &expert_id in &expert_union {
        let expert = v4_expert::RoutedExpert::load_layer(expert_shard, weights.layer, expert_id)?;
        expert_alloc_ms += expert.alloc_ms;
        expert_read_ms += expert.read_ms;
        expert_bytes += expert.h2d_bytes();
        let expert_setup_started = std::time::Instant::now();
        let mut slot = ExpertSlot::new(context, module, stream, &expert)?;
        let setup_ms = expert_setup_started.elapsed().as_secs_f64() * 1_000.0;
        expert_setup_ms += setup_ms;
        expert_setup_samples_ms.push(setup_ms);
        expert_h2d_samples_ms.push(slot.weight_h2d_ms);
        for row in 0..rows {
            for route_slot in 0..v4_router::V4_TOPK {
                let route = row * v4_router::V4_TOPK + route_slot;
                if routes.ids[route] == i32::from(expert_id) {
                    let input = &ffn_input[row * v4_state::HIDDEN..][..v4_state::HIDDEN];
                    let (output, kernel_ns) = slot.execute(input, routes.weights[route])?;
                    expert_kernel_ms += kernel_ns as f64 / 1_000_000.0;
                    contributions[row][route_slot] = Some(output);
                }
            }
        }
    }

    let mut moe = Vec::with_capacity(rows * v4_state::HIDDEN);
    for row in 0..rows {
        let routed = (0..v4_router::V4_TOPK)
            .map(|slot| {
                Ok((
                    u16::try_from(routes.ids[row * v4_router::V4_TOPK + slot])?,
                    contributions[row][slot]
                        .as_deref()
                        .ok_or("missing routed expert contribution")?,
                ))
            })
            .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
        moe.extend(v4_expert::reduce_moe_canonical(
            &routed,
            &shared[row * v4_state::HIDDEN..][..v4_state::HIDDEN],
        )?);
    }
    let output = step_l0_hc_post(
        module,
        stream,
        &moe,
        &after_attention,
        &ffn_hc.post,
        &ffn_hc.comb,
    )?;
    maybe_write_v4_audit_u16(weights.layer, "hidden", &output)?;
    let mut route_hash = fnv1a_i32(&routes.ids);
    for byte in fnv1a_f32(&routes.weights).to_le_bytes() {
        route_hash = (route_hash ^ byte as u64).wrapping_mul(0x100_0000_01b3);
    }
    let stage_hashes = vec![
        ("attn_hc", fnv1a_u16(&attn_hc.collapsed)),
        ("attn_norm", fnv1a_u16(&attn_input)),
        ("q_lora_raw", fnv1a_u16(&q_lora_raw)),
        ("q_lora_norm", fnv1a_u16(&q_lora)),
        ("q_raw", fnv1a_u16(&q_raw)),
        ("q_norm", fnv1a_u16(&q_norm)),
        ("q", fnv1a_u16(&q)),
        ("kv_raw", fnv1a_u16(&kv_raw)),
        ("kv_norm", fnv1a_u16(&kv_norm)),
        ("kv_rope", fnv1a_u16(&kv_rope)),
        ("kv", fnv1a_u16(&kv)),
        ("compressor", fnv1a_f32(&compressor.score_plus_ape_f32)),
        ("sparse_attn", fnv1a_u16(&sparse)),
        ("attn_out", fnv1a_u16(&attention)),
        ("attn_post", fnv1a_u16(&after_attention)),
        ("ffn_hc", fnv1a_u16(&ffn_hc.collapsed)),
        ("routes", route_hash),
        ("moe", fnv1a_u16(&moe)),
        ("l0_out", fnv1a_u16(&output)),
        ("state", l0_persistent_digest(&persistent)),
    ];
    let dynamic_bytes = l0_dynamic_h2d_bytes(rows);
    Ok(L0Run {
        output,
        routes,
        prefixes,
        persistent,
        stage_hashes,
        expert_union,
        expert_bytes,
        expert_alloc_ms,
        expert_read_ms,
        expert_setup_ms,
        expert_setup_samples_ms,
        expert_h2d_samples_ms,
        expert_kernel_ms,
        dynamic_bytes,
        elapsed_ms: started.elapsed().as_secs_f64() * 1_000.0,
    })
}

pub(crate) fn run_real_l0_k2_if_available(
    context: &std::sync::Arc<CudaContext>,
    module: &kernels::LoadedModule,
    stream: &std::sync::Arc<CudaStream>,
) -> Result<(), Box<dyn Error>> {
    let root = v4_model_dir()?;
    let test_started = std::time::Instant::now();
    let token_ids = [0_i64, 1];
    let state_load_started = std::time::Instant::now();
    let state = v4_state::load(&root, &token_ids)?;
    let state_load_ms = state_load_started.elapsed().as_secs_f64() * 1_000.0;
    let projection_load_started = std::time::Instant::now();
    let projections = v4_state::load_fp8_projections(&root)?;
    let projection_load_ms = projection_load_started.elapsed().as_secs_f64() * 1_000.0;
    let compressor_load_started = std::time::Instant::now();
    let compressor = v4_state::load_compressor_weights(&root)?;
    let compressor_load_ms = compressor_load_started.elapsed().as_secs_f64() * 1_000.0;
    let tail = v4_state::load_l0_tail(&root)?;
    let fixture = v4_state::fp8_projection_fixture()?;
    let freqs = &fixture.freqs_cis_f32_pairs[..2 * v4_state::Q_ROPE_DIM];
    let expert_shard = safetensors::ShardFile::open(root.join("model-00002-of-00064.safetensors"))?;
    let hash_load_started = std::time::Instant::now();
    let hash_ids = read_v4_hash_rows(&expert_shard, "layers.0.ffn.gate.tid2eid", &token_ids)?;
    let hash_load_ms = hash_load_started.elapsed().as_secs_f64() * 1_000.0;

    let upload_started = std::time::Instant::now();
    let weights = LayerDeviceWeights::new_l0(stream, &state, &projections, &compressor, &tail)?;
    stream.synchronize()?;
    let upload_ms = upload_started.elapsed().as_secs_f64() * 1_000.0;
    let dense_h2d = weights.dense_num_bytes();
    let shared_h2d = weights.shared_num_bytes();
    let resident_bytes = dense_h2d + shared_h2d;
    let embedding_source_bytes = state.embedding.len() * std::mem::size_of::<u16>();
    let fixed_source_bytes = state
        .range_bytes
        .checked_sub(embedding_source_bytes)
        .ok_or("L0 fixed source byte underflow")?
        + projections.range_bytes
        + compressor.range_bytes
        + tail.source_range_bytes;
    let base = L0Persistent::new()?;
    let k2 = run_real_l0_rows(
        context,
        module,
        stream,
        &weights,
        &expert_shard,
        &state.embedding,
        freqs,
        &hash_ids,
        0,
        &base,
    )?;
    let row_freqs = v4_state::Q_ROPE_DIM;
    let row_routes = v4_router::V4_TOPK;
    let s10 = run_real_l0_rows(
        context,
        module,
        stream,
        &weights,
        &expert_shard,
        &state.embedding[..v4_state::HIDDEN],
        &freqs[..row_freqs],
        &hash_ids[..row_routes],
        0,
        &base,
    )?;
    let s11 = run_real_l0_rows(
        context,
        module,
        stream,
        &weights,
        &expert_shard,
        &state.embedding[v4_state::HIDDEN..],
        &freqs[row_freqs..],
        &hash_ids[row_routes..],
        1,
        &s10.persistent,
    )?;

    let output_row = v4_state::HC * v4_state::HIDDEN;
    if k2.output[..output_row] != s10.output
        || k2.output[output_row..] != s11.output
        || k2.routes.ids[..row_routes] != s10.routes.ids
        || k2.routes.ids[row_routes..] != s11.routes.ids
        || !k2.routes.weights[..row_routes]
            .iter()
            .zip(&s10.routes.weights)
            .all(|(left, right)| left.to_bits() == right.to_bits())
        || !k2.routes.weights[row_routes..]
            .iter()
            .zip(&s11.routes.weights)
            .all(|(left, right)| left.to_bits() == right.to_bits())
        || k2.prefixes.len() != 2
        || !k2.prefixes[0].bits_equal(&s10.persistent)
        || !k2.prefixes[1].bits_equal(&s11.persistent)
        || !k2.persistent.bits_equal(&s11.persistent)
    {
        return Err("real full-L0 K2 differs from two sequential S=1 rows".into());
    }

    let exact = |left: &L0Persistent, right: &L0Persistent| left.bits_equal(right);
    let prepared = || {
        v4_state::K2Prepared::prepare(
            &base,
            |state| {
                state.clone_from(&k2.prefixes[0]);
                Ok(())
            },
            |state| {
                state.clone_from(&k2.prefixes[1]);
                Ok(())
            },
        )
    };
    let sequential = [&s10.persistent, &s11.persistent];
    let full = prepared()?.validate(sequential, [0, 1], [0, 1], exact)?;
    if full.accepted_rows() != 2 {
        return Err("real full-L0 K2 full acceptance changed".into());
    }
    let mut r0 = base.clone();
    prepared()?
        .validate(sequential, [0, 1], [0, 1], exact)?
        .commit_prefix(&mut r0, 0, exact)?;
    let mut r1 = base.clone();
    prepared()?
        .validate(sequential, [0, 1], [0, 1], exact)?
        .commit_prefix(&mut r1, 1, exact)?;
    let mut r2 = base.clone();
    full.commit_prefix(&mut r2, 2, exact)?;
    if !r0.bits_equal(&base) || !r1.bits_equal(&s10.persistent) || !r2.bits_equal(&s11.persistent) {
        return Err("real full-L0 K2 r0/r1/r2 commit mismatch".into());
    }
    let rejected = prepared()?.validate(sequential, [0, 1], [0, 2], exact)?;
    let mut rejected_state = base.clone();
    if !matches!(
        rejected.commit_prefix(&mut rejected_state, 2, exact),
        Err(v4_state::K2ContractError::RejectedSuffix)
    ) {
        return Err("real full-L0 K2 accepted a rejected suffix".into());
    }
    rejected.commit_prefix(&mut rejected_state, 1, exact)?;
    let mut dirty = s10.persistent.clone();
    if !matches!(
        prepared()?
            .validate(sequential, [0, 1], [0, 1], exact)?
            .commit_prefix(&mut dirty, 1, exact),
        Err(v4_state::K2ContractError::DirtyAuthoritativeState)
    ) {
        return Err("real full-L0 K2 accepted dirty authoritative state".into());
    }

    for (run_name, run) in [("K2", &k2), ("S1_r0", &s10), ("S1_r1", &s11)] {
        let hashes = run
            .stage_hashes
            .iter()
            .map(|(stage, hash)| format!("{stage}={hash:016x}"))
            .collect::<Vec<_>>()
            .join(",");
        println!(
            "V4_L0_TRACE run={run_name} routes={:?} union={:?} hashes=[{hashes}]",
            run.routes.ids, run.expert_union
        );
    }
    let s1_time = s10.elapsed_ms + s11.elapsed_ms;
    let s1_dynamic = s10.dynamic_bytes + s11.dynamic_bytes;
    let s1_expert = s10.expert_bytes + s11.expert_bytes;
    let gate = v4_state::evaluate_k2_renewal_gate(v4_state::K2GateInput {
        draft_matches: 1,
        trials: 1,
        t1_lower: (s1_time / 2.0).max(f64::EPSILON),
        t2_upper: k2.elapsed_ms.max(f64::EPSILON),
        b1_lower: ((resident_bytes + s1_expert) as f64 / 2.0).max(1.0),
        b2_upper: (resident_bytes + k2.expert_bytes) as f64,
    });
    println!(
        "V4_L0_K2_TELEMETRY rows=2 token_ids={token_ids:?} q_sample=1/1 t1_s1x2_wall_total={s1_time:.3}ms t1_per_row_mean={:.3}ms t2_k2_wall={:.3}ms D_actual_h2d_k2={} D_actual_h2d_s1x2={} P_actual_h2d_k2={} P_actual_h2d_s1x2={} P_dense_fixed={} P_shared_fixed={} P_routed_k2_union={} P_routed_s1x2_actual={} loader_source_k2={} loader_source_s1x2={} loader_fixed_source={} loader_embedding_source={} actual_test_weight_h2d={} expert_union_k2={:?} state_digest={:016x} fixed_weight_upload={upload_ms:.3}ms gate={}",
        s1_time / 2.0,
        k2.elapsed_ms,
        k2.dynamic_bytes,
        s1_dynamic,
        resident_bytes + k2.expert_bytes,
        resident_bytes + s1_expert,
        dense_h2d,
        shared_h2d,
        k2.expert_bytes,
        s1_expert,
        fixed_source_bytes + embedding_source_bytes + k2.expert_bytes,
        fixed_source_bytes + embedding_source_bytes + s1_expert,
        fixed_source_bytes,
        embedding_source_bytes,
        resident_bytes + k2.expert_bytes + s1_expert,
        k2.expert_union,
        l0_persistent_digest(&k2.persistent),
        gate.reason,
    );
    println!(
        "V4_L0_BYTE_DEFINITION D=actual dynamic host-to-device payloads issued by the host-boundary runner; P=dense fixed H2D + shared fixed H2D + routed expert H2D. S1x2 routed bytes are the two observed S1 loads summed without cross-run dedup; K2 routed bytes are its sorted expert union loaded once. loader_source counts pinned shard bytes before wo_a FP8-to-BF16 expansion."
    );
    let fixed_read_alloc_ms = state_load_ms
        + projection_load_ms
        + compressor_load_ms
        + (tail.load_elapsed_ms - tail.wo_a_decode_ms)
        + hash_load_ms;
    let mut s1_h2d_samples = s10.expert_h2d_samples_ms.clone();
    s1_h2d_samples.extend_from_slice(&s11.expert_h2d_samples_ms);
    let mut s1_setup_samples = s10.expert_setup_samples_ms.clone();
    s1_setup_samples.extend_from_slice(&s11.expert_setup_samples_ms);
    let k2_read_gbps = k2.expert_bytes as f64 / (k2.expert_read_ms * 1_000_000.0);
    let s1_read_gbps = s1_expert as f64 / ((s10.expert_read_ms + s11.expert_read_ms) * 1_000_000.0);
    println!(
        "V4_L0_IO_BREAKDOWN fixed_read_alloc={fixed_read_alloc_ms:.3}ms wo_a_decode={:.3}ms routed_host_alloc_k2={:.3}ms routed_host_alloc_s1x2={:.3}ms pinned_device_alloc_h2d_k2={:.3}ms pinned_device_alloc_h2d_s1x2={:.3}ms setup_p95_k2={:.3}ms setup_p95_s1x2={:.3}ms allocation_free_h2d_p95_k2={:.3}ms allocation_free_h2d_p95_s1x2={:.3}ms routed_read_k2={:.3}ms routed_read_s1x2={:.3}ms routed_read_k2_gbps={k2_read_gbps:.3} routed_read_s1x2_gbps={s1_read_gbps:.3} routed_decode_k2=0.000ms routed_decode_s1x2=0.000ms routed_kernel_k2={:.3}ms routed_kernel_s1x2={:.3}ms runner_e2e_k2={:.3}ms runner_e2e_s1x2={s1_time:.3}ms full_test_e2e={:.3}ms",
        tail.wo_a_decode_ms,
        k2.expert_alloc_ms,
        s10.expert_alloc_ms + s11.expert_alloc_ms,
        k2.expert_setup_ms,
        s10.expert_setup_ms + s11.expert_setup_ms,
        p95_f64(&k2.expert_setup_samples_ms),
        p95_f64(&s1_setup_samples),
        p95_f64(&k2.expert_h2d_samples_ms),
        p95_f64(&s1_h2d_samples),
        k2.expert_read_ms,
        s10.expert_read_ms + s11.expert_read_ms,
        k2.expert_kernel_ms,
        s10.expert_kernel_ms + s11.expert_kernel_ms,
        k2.elapsed_ms,
        test_started.elapsed().as_secs_f64() * 1_000.0,
    );
    println!(
        "PASS: real V4 full L0 K2 equals two S=1 rows exactly; output={:016x}, routes/weights=exact, persistent=exact, r0/r1/r2/reject/dirty=exact; one sample is telemetry only",
        fnv1a_u16(&k2.output),
    );
    Ok(())
}

pub(crate) fn run_k2_p0_host_contract() -> Result<u64, Box<dyn Error>> {
    type HostState = (
        v4_state::V4KvCache,
        [u32; v4_state::HC],
        v4_state::V4CompressorState,
        Option<v4_state::V4CompressorState>,
    );

    fn exact(left: &HostState, right: &HostState) -> bool {
        left.0.bits_equal(&right.0)
            && left.1 == right.1
            && left.2.bits_equal(&right.2)
            && match (&left.3, &right.3) {
                (Some(left), Some(right)) => left.bits_equal(right),
                (None, None) => true,
                _ => false,
            }
    }

    fn projected(width: usize, marker: f32) -> v4_state::CompressorPrepared {
        v4_state::CompressorPrepared {
            kv_f32: vec![marker; width],
            score_f32: vec![marker + 8.0; width],
            score_plus_ape_f32: vec![marker + 16.0; width],
        }
    }

    fn apply(
        state: &mut HostState,
        ratio: usize,
        start_pos: usize,
        marker: u16,
    ) -> Result<(), Box<dyn Error>> {
        let row = vec![v4_router::f32_to_bf16_rne(marker as f32); v4_state::Q_HEAD_DIM];
        state.0 = state.0.prepare_decode_row(&row, start_pos)?;
        if start_pos
            .checked_add(1)
            .ok_or("K2 host position overflow")?
            .is_multiple_of(ratio)
        {
            let compressed =
                vec![v4_router::f32_to_bf16_rne((marker + 32) as f32); v4_state::Q_HEAD_DIM];
            state.0 = state
                .0
                .prepare_compressed_row(&compressed, start_pos / ratio)?;
        }
        state.1[0] = state.1[0].wrapping_mul(17).wrapping_add(marker as u32);
        let width = if ratio == 4 { 4 } else { 2 };
        state.2 = state
            .2
            .prepare_decode_row(&projected(width, marker as f32), start_pos)?;
        if let Some(indexer) = &state.3 {
            state.3 =
                Some(indexer.prepare_decode_row(&projected(2, marker as f32 + 64.0), start_pos)?);
        }
        Ok(())
    }

    fn boundary(ratio: usize, start_pos: usize) -> Result<HostState, Box<dyn Error>> {
        let base = (
            v4_state::V4KvCache::new(),
            [0; v4_state::HC],
            v4_state::V4CompressorState::new_decode(ratio, 2)?,
            if ratio == 4 {
                Some(v4_state::V4CompressorState::new_decode(4, 1)?)
            } else {
                None
            },
        );
        let mut sequential0 = base.clone();
        apply(&mut sequential0, ratio, start_pos, 3)?;
        let mut sequential1 = sequential0.clone();
        apply(&mut sequential1, ratio, start_pos + 1, 5)?;
        // ponytail: full state snapshots are the P0 correctness ceiling; move
        // to delta pages only after this exact contract passes the byte gate.
        let validated = v4_state::K2Prepared::prepare(
            &base,
            |state| apply(state, ratio, start_pos, 3),
            |state| apply(state, ratio, start_pos + 1, 5),
        )?
        .validate([&sequential0, &sequential1], [7, 11], [7, 11], exact)?;
        if validated.accepted_rows() != 2 {
            return Err("K2 host full-accept prefix changed".into());
        }
        let mut authoritative = base;
        validated.commit_prefix(&mut authoritative, 2, exact)?;
        if !exact(&authoritative, &sequential1) {
            return Err("K2 host commit differs from sequential S=1".into());
        }
        Ok(authoritative)
    }

    fn hash_state(state: &HostState, mut hash: u64) -> u64 {
        for &bits in state.0.as_slice() {
            for byte in bits.to_le_bytes() {
                hash = (hash ^ byte as u64).wrapping_mul(0x100_0000_01b3);
            }
        }
        for &value in &state.1 {
            for byte in value.to_le_bytes() {
                hash = (hash ^ byte as u64).wrapping_mul(0x100_0000_01b3);
            }
        }
        for compressor in std::iter::once(&state.2).chain(state.3.iter()) {
            for &value in compressor.kv().iter().chain(compressor.score()) {
                for byte in value.to_bits().to_le_bytes() {
                    hash = (hash ^ byte as u64).wrapping_mul(0x100_0000_01b3);
                }
            }
        }
        hash
    }

    let ratio4 = boundary(4, 3)?;
    let ratio128 = boundary(128, 127)?;
    let mut hash = hash_state(&ratio4, 0xcbf2_9ce4_8422_2325);
    hash = hash_state(&ratio128, hash);

    let execution_order = [6_u16, 3, 1, 5, 2, 4];
    let contributions: Vec<_> = execution_order
        .iter()
        .map(|&expert| (expert, vec![expert as f32 / 8.0; v4_expert::V4_HIDDEN]))
        .collect();
    let routed: Vec<_> = contributions
        .iter()
        .map(|(expert, values)| (*expert, values.as_slice()))
        .collect();
    let shared = vec![2.0_f32; v4_expert::V4_HIDDEN];
    let canonical = v4_expert::reduce_moe_canonical(&routed, &shared)?;
    for bits in canonical {
        for byte in bits.to_le_bytes() {
            hash = (hash ^ byte as u64).wrapping_mul(0x100_0000_01b3);
        }
    }
    Ok(hash)
}

pub(crate) fn run_real_l0_compressor_cublas_exact_if_available(
    module: &kernels::LoadedModule,
    stream: &std::sync::Arc<CudaStream>,
) -> Result<(), Box<dyn Error>> {
    const OFFICIAL_RAW_KV: u64 = 0x7f8a_ba0e_84d6_9c22;
    const OFFICIAL_RAW_SCORE_APE: u64 = 0x2b84_6a05_10d4_4fbc;
    const OFFICIAL_STATE_KV: u64 = 0x9af7_3870_9362_5c22;
    const OFFICIAL_STATE_SCORE: u64 = 0x2589_1515_a4d6_ffbc;
    const REFERENCE_RAW_KV: u64 = 0xad53_d3f6_c10b_98cd;
    const REFERENCE_RAW_SCORE_APE: u64 = 0x0052_3bf6_da39_b612;
    const REFERENCE_STATE_KV: u64 = 0x607f_82ef_92d9_f8cd;
    const REFERENCE_STATE_SCORE: u64 = 0x2380_0c9d_021c_6612;

    let cublas = cublas_exact::CublasExact::load()?;
    let root = v4_model_dir()?;
    let state = v4_state::load(&root, &[0])?;
    let compressor_weights = v4_state::load_compressor_weights(&root)?;
    let mut residual = Vec::with_capacity(v4_state::HC * v4_state::HIDDEN);
    for _ in 0..v4_state::HC {
        residual.extend_from_slice(&state.embedding);
    }
    let hc_fn = DeviceBuffer::from_host(stream, &state.hc_fn)?;
    let hc_scale = DeviceBuffer::from_host(stream, &state.hc_scale)?;
    let hc_base = DeviceBuffer::from_host(stream, &state.hc_base)?;
    let norm = DeviceBuffer::from_host(stream, &state.attn_norm)?;
    let hc = step_l0_mhc_pre(module, stream, &residual, &hc_fn, &hc_scale, &hc_base)?;
    let input = step_l0_rmsnorm(module, stream, &hc.collapsed, &norm, v4_state::HIDDEN)?;

    let wkv = DeviceBuffer::from_host(stream, &compressor_weights.wkv_bf16)?;
    let wgate = DeviceBuffer::from_host(stream, &compressor_weights.wgate_bf16)?;
    let ape = DeviceBuffer::from_host(stream, &compressor_weights.ape_f32)?;
    let reference = run_compressor(
        module,
        stream,
        &input,
        &wkv,
        &wgate,
        &ape,
        v4_state::Q_HEAD_DIM,
        v4_state::COMPRESS_RATIO,
        0,
    )?;
    let official = run_l0_compressor_official_cublas(&cublas, stream, &input, &compressor_weights)?;
    let official_repeat =
        run_l0_compressor_official_cublas(&cublas, stream, &input, &compressor_weights)?;
    let prepared_equal = |left: &v4_state::CompressorPrepared,
                          right: &v4_state::CompressorPrepared| {
        left.kv_f32
            .iter()
            .chain(&left.score_f32)
            .chain(&left.score_plus_ape_f32)
            .zip(
                right
                    .kv_f32
                    .iter()
                    .chain(&right.score_f32)
                    .chain(&right.score_plus_ape_f32),
            )
            .all(|(left, right)| left.to_bits() == right.to_bits())
    };
    if !prepared_equal(&official, &official_repeat) {
        return Err("official cuBLAS L0 Compressor repeat changed raw bits".into());
    }
    let state_digests = |prepared: &v4_state::CompressorPrepared| {
        let state = v4_state::V4CompressorState::new_decode(
            v4_state::COMPRESS_RATIO,
            v4_state::Q_HEAD_DIM,
        )?
        .prepare_decode_row(prepared, 0)?;
        Ok::<_, Box<dyn Error>>((fnv1a_f32(state.kv()), fnv1a_f32(state.score())))
    };
    let reference_raw = (
        fnv1a_f32(&reference.kv_f32),
        fnv1a_f32(&reference.score_plus_ape_f32),
    );
    let official_raw = (
        fnv1a_f32(&official.kv_f32),
        fnv1a_f32(&official.score_plus_ape_f32),
    );
    let reference_state = state_digests(&reference)?;
    let official_state = state_digests(&official)?;
    if reference_raw != (REFERENCE_RAW_KV, REFERENCE_RAW_SCORE_APE)
        || reference_state != (REFERENCE_STATE_KV, REFERENCE_STATE_SCORE)
    {
        return Err(format!(
            "reference Compressor arm drifted: raw={reference_raw:016x?}, state={reference_state:016x?}"
        )
        .into());
    }
    if official_raw != (OFFICIAL_RAW_KV, OFFICIAL_RAW_SCORE_APE)
        || official_state != (OFFICIAL_STATE_KV, OFFICIAL_STATE_SCORE)
    {
        return Err(format!(
            "official cuBLAS Compressor arm mismatch: raw={official_raw:016x?}, state={official_state:016x?}"
        )
        .into());
    }
    if prepared_equal(&reference, &official) {
        return Err("Compressor A/B fixture no longer distinguishes reference and cuBLAS".into());
    }
    let provenance = cublas.provenance();
    println!(
        "CUBLAS_EXACT_PROVENANCE resolved={} version={} dladdr={} proc_maps={} sha256={} workspace_bytes={}",
        provenance.resolved_path.display(),
        provenance.version,
        provenance.dladdr_path.display(),
        provenance.maps_path.display(),
        provenance.actual_sha256,
        cublas_exact::WORKSPACE_BYTES,
    );
    println!(
        "CUBLAS_COMPRESSOR_AB reference_raw={reference_raw:016x?} reference_state={reference_state:016x?} official_raw={official_raw:016x?} official_state={official_state:016x?}"
    );
    println!(
        "PASS: official cuBLAS L0 Compressor raw/state digests match the pinned official oracle exactly; reference arm remains pinned and distinct"
    );
    Ok(())
}

pub(crate) fn run_real_v4_compressor_slice_if_available(
    module: &kernels::LoadedModule,
    stream: &std::sync::Arc<CudaStream>,
) -> Result<(), Box<dyn Error>> {
    let root = v4_model_dir()?;
    let fixture = v4_state::fp8_projection_fixture()?;
    let weights = v4_state::load_compressor_weights(&root)?;
    if weights.range_bytes != 14_942_208 {
        return Err("V4 Compressor pinned source range changed".into());
    }
    let prepared = v4_state::prepare_compressor_cpu(&fixture.input_bf16, &weights)?;
    let exact = |left: &[f32], right: &[f32]| {
        left.len() == right.len()
            && left
                .iter()
                .zip(right)
                .all(|(left, right)| left.to_bits() == right.to_bits())
    };
    let prepared_exact = |kv: &[f32], score: &[f32], score_plus_ape: &[f32]| {
        kv.iter()
            .chain(score)
            .chain(score_plus_ape)
            .all(|value| value.is_finite())
            && exact(kv, &fixture.compressor_kv_f32)
            && exact(score, &fixture.compressor_score_f32)
            && exact(score_plus_ape, &fixture.compressor_score_plus_ape_f32)
    };
    if !prepared_exact(
        &prepared.kv_f32,
        &prepared.score_f32,
        &prepared.score_plus_ape_f32,
    ) {
        return Err("V4 Compressor CPU prepare exact gate failed".into());
    }

    let input = DeviceBuffer::from_host(stream, &fixture.input_bf16)?;
    let wkv = DeviceBuffer::from_host(stream, &weights.wkv_bf16)?;
    let wgate = DeviceBuffer::from_host(stream, &weights.wgate_bf16)?;
    let ape = DeviceBuffer::from_host(stream, &weights.ape_f32)?;
    let mut kv = DeviceBuffer::<f32>::zeroed(stream, prepared.kv_f32.len())?;
    let mut score = DeviceBuffer::<f32>::zeroed(stream, prepared.score_f32.len())?;
    let mut score_plus_ape =
        DeviceBuffer::<f32>::zeroed(stream, prepared.score_plus_ape_f32.len())?;
    // SAFETY: the fixed launch and all seven buffers satisfy the kernel's
    // complete L0 start_pos=0 shapes; every thread owns one output element.
    unsafe {
        module.compressor_v4_bf16_f32(
            stream,
            LaunchConfig::for_num_elems(u32::try_from(kv.len())?),
            &input,
            &wkv,
            &wgate,
            &ape,
            &mut kv,
            &mut score,
            &mut score_plus_ape,
            0,
            u32::try_from(v4_state::HIDDEN)?,
            u32::try_from(v4_state::Q_HEAD_DIM)?,
            u32::try_from(v4_state::COMPRESS_RATIO)?,
        )?;
    }
    let kv_host = kv.to_host_vec(stream)?;
    let score_host = score.to_host_vec(stream)?;
    let score_plus_ape_host = score_plus_ape.to_host_vec(stream)?;
    let (kv_mismatches, _) = report_v4_state_f32(
        "Compressor kv GPU/fixture",
        &kv_host,
        &fixture.compressor_kv_f32,
    );
    let (score_mismatches, _) = report_v4_state_f32(
        "Compressor score GPU/fixture",
        &score_host,
        &fixture.compressor_score_f32,
    );
    let (score_ape_mismatches, _) = report_v4_state_f32(
        "Compressor score+ape GPU/fixture",
        &score_plus_ape_host,
        &fixture.compressor_score_plus_ape_f32,
    );
    if kv_mismatches != 0
        || score_mismatches != 0
        || score_ape_mismatches != 0
        || !prepared_exact(&kv_host, &score_host, &score_plus_ape_host)
    {
        return Err("V4 Compressor GPU prepare exact gate failed before state write".into());
    }

    let write_elements = v4_state::preflight_compressor_commit(
        fixture.compressor_kv_state_initial_f32.len(),
        fixture.compressor_score_state_initial_f32.len(),
        kv.len(),
        score_plus_ape.len(),
        1,
        0,
        v4_state::IDS.len(),
        v4_state::COMPRESS_RATIO,
        false,
    )?;
    let write_bytes = write_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or("V4 Compressor commit byte overflow")?;
    let mut state_kv = DeviceBuffer::from_host(stream, &fixture.compressor_kv_state_initial_f32)?;
    let mut state_score =
        DeviceBuffer::from_host(stream, &fixture.compressor_score_state_initial_f32)?;
    let initial_kv = state_kv.to_host_vec(stream)?;
    let initial_score = state_score.to_host_vec(stream)?;
    if !exact(&initial_kv, &fixture.compressor_kv_state_initial_f32)
        || !exact(&initial_score, &fixture.compressor_score_state_initial_f32)
    {
        return Err("V4 Compressor GPU initial state exact gate failed".into());
    }

    let mut scratch_kv =
        unsafe { DeviceBuffer::<f32>::uninitialized_async(stream, state_kv.len())? };
    let mut scratch_score =
        unsafe { DeviceBuffer::<f32>::uninitialized_async(stream, state_score.len())? };
    scratch_kv.copy_from_device_async(&state_kv, stream)?;
    scratch_score.copy_from_device_async(&state_score, stream)?;
    // SAFETY: preflight proves both prefix copies have the same 8,192-byte
    // range and fit the paired scratch states. Authoritative buffers are untouched.
    unsafe {
        cuda_core::memory::memcpy_dtod_async(
            scratch_kv.cu_deviceptr(),
            kv.cu_deviceptr(),
            write_bytes,
            stream.cu_stream(),
        )?;
        cuda_core::memory::memcpy_dtod_async(
            scratch_score.cu_deviceptr(),
            score_plus_ape.cu_deviceptr(),
            write_bytes,
            stream.cu_stream(),
        )?;
    }
    let scratch_kv_host = scratch_kv.to_host_vec(stream)?;
    let scratch_score_host = scratch_score.to_host_vec(stream)?;
    let (state_kv_mismatches, _) = report_v4_state_f32(
        "Compressor KV scratch GPU/fixture",
        &scratch_kv_host,
        &fixture.compressor_kv_state_after_f32,
    );
    let (state_score_mismatches, _) = report_v4_state_f32(
        "Compressor score scratch GPU/fixture",
        &scratch_score_host,
        &fixture.compressor_score_state_after_f32,
    );
    if state_kv_mismatches != 0 || state_score_mismatches != 0 {
        return Err("V4 Compressor paired GPU scratch gate failed".into());
    }
    std::mem::swap(&mut state_kv, &mut scratch_kv);
    std::mem::swap(&mut state_score, &mut scratch_score);
    let authoritative_kv = state_kv.to_host_vec(stream)?;
    let authoritative_score = state_score.to_host_vec(stream)?;

    scratch_kv.copy_from_device_async(&state_kv, stream)?;
    scratch_score.copy_from_device_async(&state_score, stream)?;
    unsafe {
        cuda_core::memory::memcpy_dtod_async(
            scratch_kv.cu_deviceptr(),
            kv.cu_deviceptr(),
            write_bytes,
            stream.cu_stream(),
        )?;
        cuda_core::memory::memcpy_dtod_async(
            scratch_score.cu_deviceptr(),
            score_plus_ape.cu_deviceptr(),
            write_bytes,
            stream.cu_stream(),
        )?;
    }
    let repeat_kv = scratch_kv.to_host_vec(stream)?;
    let repeat_score = scratch_score.to_host_vec(stream)?;
    let repeat_idempotent =
        exact(&repeat_kv, &authoritative_kv) && exact(&repeat_score, &authoritative_score);
    if !repeat_idempotent {
        return Err("V4 Compressor repeated GPU commit was not byte-idempotent".into());
    }
    std::mem::swap(&mut state_kv, &mut scratch_kv);
    std::mem::swap(&mut state_score, &mut scratch_score);

    let before_reject_kv = state_kv.to_host_vec(stream)?;
    let before_reject_score = state_score.to_host_vec(stream)?;
    let invalid_rejected = v4_state::preflight_compressor_commit(
        state_kv.len(),
        state_score.len(),
        kv.len(),
        score_plus_ape.len(),
        1,
        usize::MAX,
        v4_state::IDS.len(),
        v4_state::COMPRESS_RATIO,
        false,
    )
    .is_err();
    let mut conflict = score_plus_ape_host.clone();
    conflict[0] = f32::from_bits(conflict[0].to_bits() ^ 1);
    let conflict_rejected = !prepared_exact(&kv_host, &score_host, &conflict);
    let mut non_finite = kv_host.clone();
    non_finite[0] = f32::NAN;
    let non_finite_rejected = !prepared_exact(&non_finite, &score_host, &score_plus_ape_host);
    let after_reject_kv = state_kv.to_host_vec(stream)?;
    let after_reject_score = state_score.to_host_vec(stream)?;
    let reject_unchanged = exact(&before_reject_kv, &after_reject_kv)
        && exact(&before_reject_score, &after_reject_score);
    let cache_before = fixture.cache_after_bf16.clone();
    let cache_unchanged = fixture.cache_after_bf16 == cache_before;
    println!(
        "Compressor commit invariants: bytes={write_bytes}, repeat_idempotent={repeat_idempotent}, invalid_rejected={invalid_rejected}, conflict_rejected={conflict_rejected}, non_finite_rejected={non_finite_rejected}, reject_unchanged={reject_unchanged}, AB_cache_unchanged={cache_unchanged}"
    );
    if !invalid_rejected
        || !conflict_rejected
        || !non_finite_rejected
        || !reject_unchanged
        || !cache_unchanged
    {
        return Err("V4 Compressor GPU fail-closed gate failed".into());
    }
    println!(
        "PASS: V4 L0 Compressor C exact prepare -> paired scratch validate -> authoritative commit; stop before attention"
    );
    let k2_contract_hash = run_k2_p0_host_contract()?;
    println!(
        "K2_HOST_CONTRACT ratio4_start_pos=3 ratio128_start_pos=127 row_equivalence=true partial_commit=0 canonical_expert_order=true shared_last=true fnv1a={k2_contract_hash:016x}"
    );
    // This harness has no real K2 timing/byte samples yet. Exact parity above
    // is the hard prerequisite; the robust minimax gate must therefore choose AR.
    let renewal = v4_state::evaluate_k2_renewal_gate(v4_state::K2GateInput {
        draft_matches: 0,
        trials: 0,
        t1_lower: f64::NAN,
        t2_upper: f64::NAN,
        b1_lower: f64::NAN,
        b2_upper: f64::NAN,
    });
    println!(
        "K2_SAFE_MINIMAX exact_prerequisite=true samples=0 q_lower={:?} committed_rows_lower={:?} time_ratio_upper={:?} bytes_per_row_upper={:?} decision={} reason={}",
        renewal.q_lower,
        renewal.committed_rows_lower,
        renewal.time_ratio_upper,
        renewal.bytes_per_row_upper,
        if renewal.admit { "K2" } else { "AR" },
        renewal.reason,
    );
    if renewal.admit {
        return Err("unmeasured K2 robust gate admitted production execution".into());
    }
    Ok(())
}
