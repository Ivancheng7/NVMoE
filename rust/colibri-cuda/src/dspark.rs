use super::*;

pub(crate) fn step_device_attention_row(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    runtime: &LoadedLayerRuntime,
    state: &mut DeviceTokenState,
    rope_lut: &DeviceRequestRopeLut,
    hc_hidden: &[u16],
    start_pos: usize,
    observe_topk: bool,
) -> Result<RealAttentionRun, Box<dyn Error>> {
    let started = std::time::Instant::now();
    let rope_lut_d2d_count = step_device_token_state_commit_row(
        module, stream, runtime, state, rope_lut, hc_hidden, start_pos,
    )?;
    let commit_ns = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
    let (output, topk) = step_device_attention_after_commit(
        module,
        stream,
        runtime,
        state,
        rope_lut,
        start_pos,
        observe_topk,
    )?;
    let attention_topk_digest = topk.as_deref().map_or(0, fnv1a_i32);
    let attention_topk_per_row = topk.as_ref().map_or(0, Vec::len);
    let audit_topk_bytes = topk.as_ref().map_or(0, Vec::len) * std::mem::size_of::<i32>();
    let attention_ns = u64::try_from(started.elapsed().as_nanos())
        .unwrap_or(u64::MAX)
        .saturating_sub(commit_ns);
    Ok(RealAttentionRun {
        output,
        topk,
        stage_digests: Vec::new(),
        attention_topk_digest,
        attention_topk_per_row,
        indexer_topk_gpu_ns: 0,
        attention_ns,
        commit_ns,
        device_transfer: Some(DeviceAttentionTransfer {
            h2d_count: 1,
            h2d_bytes: state.hc_hidden.num_bytes(),
            rope_lut_d2d_count: rope_lut_d2d_count + 1,
            rope_lut_d2d_bytes: (rope_lut_d2d_count + 1) * state.freqs.num_bytes(),
            terminal_d2h_count: 1,
            terminal_d2h_bytes: state.after_attention.num_bytes(),
            intermediate_or_audit_d2h_count: usize::from(observe_topk),
            intermediate_or_audit_d2h_bytes: audit_topk_bytes,
        }),
        profile: None,
    })
}

pub(crate) fn flash_dspark_main_x(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    stage0: &LoadedDsparkStage,
    main_hidden: &[u16],
) -> Result<Vec<u16>, Box<dyn Error>> {
    let spec = &v4_spec::FLASH_0731;
    if stage0.stage != 0 || main_hidden.len() != 3 * spec.hidden {
        return Err("Flash-0731 DSpark main hidden contract mismatch".into());
    }
    let projection = stage0
        .main_proj
        .as_ref()
        .ok_or("Flash-0731 stage 0 main_proj is missing")?;
    let norm = stage0
        .main_norm
        .as_ref()
        .ok_or("Flash-0731 stage 0 main_norm is missing")?;
    let projected = step_l0_fp8_linear(module, stream, main_hidden, projection, false)?.1;
    step_l0_rmsnorm(module, stream, &projected, norm, spec.hidden)
}

pub(crate) fn flash_dspark_context_kv(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    stage: &LoadedDsparkStage,
    main_x: &[u16],
    position: usize,
) -> Result<Vec<u16>, Box<dyn Error>> {
    let spec = &v4_spec::FLASH_0731;
    if main_x.len() != spec.hidden {
        return Err("Flash-0731 DSpark main_x contract mismatch".into());
    }
    let kv = step_l0_fp8_linear(module, stream, main_x, &stage.weights.wkv, false)?.1;
    let kv = step_l0_rmsnorm(module, stream, &kv, &stage.weights.kv_norm, spec.q_head_dim)?;
    let kv = step_l0_rope(
        module,
        stream,
        &kv,
        &v4_state::attention_rope_freqs(spec, stage.weights.layer, position)?,
        1,
        false,
    )?;
    step_l0_kv_quantize(module, stream, &kv)
}

pub(crate) fn advance_flash_dspark_context(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    stages: &[LoadedDsparkStage; v4_state::DSPARK_STAGES],
    states: &mut [v4_state::V4LayerPersistentState; v4_state::DSPARK_STAGES],
    main_x: &[u16],
    position: usize,
) -> Result<(), Box<dyn Error>> {
    for (stage, state) in stages.iter().zip(states) {
        let kv = flash_dspark_context_kv(module, stream, stage, main_x, position)?;
        state.append_kv_row(position, &kv)?;
    }
    Ok(())
}

pub(crate) fn flash_dspark_moe(
    context: &std::sync::Arc<CudaContext>,
    module: &kernels::LoadedModule,
    stream: &std::sync::Arc<CudaStream>,
    stage: &LoadedDsparkStage,
    after_attention: &[u16],
) -> Result<Vec<u16>, Box<dyn Error>> {
    let spec = &v4_spec::FLASH_0731;
    let rows = after_attention.len() / (v4_state::HC * spec.hidden);
    if rows != v4_state::DSPARK_B5_DRAFT_TOKENS {
        return Err("Flash-0731 DSpark MoE requires exactly five rows".into());
    }
    let ffn_hc = step_l0_mhc_pre(
        module,
        stream,
        after_attention,
        &stage.weights.hc_ffn_fn,
        &stage.weights.hc_ffn_scale,
        &stage.weights.hc_ffn_base,
    )?;
    let router = step_layer_router(
        module,
        stream,
        None,
        &ffn_hc.collapsed,
        &stage.weights,
        &[],
        false,
    )?;
    let shared = step_layer_shared_expert(module, stream, &router.input, &stage.weights)?;
    let mut contributions = (0..rows)
        .map(|_| {
            (0..spec.experts_per_token)
                .map(|_| None)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<Vec<Option<Vec<f32>>>>>();
    let mut experts = router
        .routes
        .ids
        .iter()
        .map(|&expert| u16::try_from(expert))
        .collect::<Result<Vec<_>, _>>()?;
    experts.sort_unstable();
    experts.dedup();
    for expert_id in experts {
        let expert =
            v4_expert::RoutedExpert::load_dspark(&stage.expert_shard, stage.stage, expert_id)?;
        let mut slot = ExpertSlot::new(context, module, stream, &expert)?;
        for (row, row_contributions) in contributions.iter_mut().enumerate() {
            for (route_slot, contribution) in row_contributions.iter_mut().enumerate() {
                let route = row * spec.experts_per_token + route_slot;
                if router.routes.ids[route] == i32::from(expert_id) {
                    *contribution = Some(
                        slot.execute(
                            &router.input[row * spec.hidden..(row + 1) * spec.hidden],
                            router.routes.weights[route],
                        )?
                        .0,
                    );
                }
            }
        }
    }
    let mut moe = Vec::with_capacity(rows * spec.hidden);
    for (row, row_contributions) in contributions.iter().enumerate() {
        let routed = row_contributions
            .iter()
            .enumerate()
            .map(|(slot, output)| {
                Ok((
                    u16::try_from(router.routes.ids[row * spec.experts_per_token + slot])?,
                    output
                        .as_deref()
                        .ok_or("Flash-0731 DSpark routed contribution is missing")?,
                ))
            })
            .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
        moe.extend(v4_expert::reduce_moe_canonical_for(
            spec,
            &routed,
            &shared[row * spec.hidden..(row + 1) * spec.hidden],
        )?);
    }
    step_l0_hc_post(
        module,
        stream,
        &moe,
        after_attention,
        &ffn_hc.post,
        &ffn_hc.comb,
    )
}

pub(crate) fn step_flash_dspark_stage(
    context: &std::sync::Arc<CudaContext>,
    module: &kernels::LoadedModule,
    stream: &std::sync::Arc<CudaStream>,
    stage: &LoadedDsparkStage,
    state: &mut v4_state::V4LayerPersistentState,
    x: &[u16],
    main_x: &[u16],
    start_pos: usize,
) -> Result<Vec<u16>, Box<dyn Error>> {
    let spec = &v4_spec::FLASH_0731;
    let rows = v4_state::DSPARK_B5_DRAFT_TOKENS;
    if x.len() != rows * v4_state::HC * spec.hidden
        || state.layer() != spec.layers + stage.stage
        || state.committed_kv_rows() != start_pos
    {
        return Err("Flash-0731 DSpark stage state/input contract mismatch".into());
    }
    let attn_hc = step_l0_mhc_pre(
        module,
        stream,
        x,
        &stage.weights.hc_attn_fn,
        &stage.weights.hc_attn_scale,
        &stage.weights.hc_attn_base,
    )?;
    let attn_input = step_l0_rmsnorm(
        module,
        stream,
        &attn_hc.collapsed,
        &stage.weights.attn_norm,
        spec.hidden,
    )?;
    let q_lora = step_l0_fp8_linear(module, stream, &attn_input, &stage.weights.wq_a, false)?.1;
    let q_lora = step_l0_rmsnorm(
        module,
        stream,
        &q_lora,
        &stage.weights.q_norm,
        spec.q_lora_rank,
    )?;
    let q = step_l0_fp8_linear(module, stream, &q_lora, &stage.weights.wq_b, true)?.1;
    let unit_norm = DeviceBuffer::<u16>::zeroed(stream, 0)?;
    let q = step_l0_rmsnorm(module, stream, &q, &unit_norm, spec.q_head_dim)?;
    let mut query_freqs = Vec::with_capacity(rows * v4_state::Q_ROPE_DIM);
    for offset in 1..=rows {
        query_freqs.extend(v4_state::attention_rope_freqs(
            spec,
            stage.weights.layer,
            start_pos + offset,
        )?);
    }
    let q = step_l0_rope(module, stream, &q, &query_freqs, spec.q_heads, false)?;
    let query_kv = step_l0_fp8_linear(module, stream, &attn_input, &stage.weights.wkv, false)?.1;
    let query_kv = step_l0_rmsnorm(
        module,
        stream,
        &query_kv,
        &stage.weights.kv_norm,
        spec.q_head_dim,
    )?;
    let query_kv = step_l0_rope(module, stream, &query_kv, &query_freqs, 1, false)?;
    let query_kv = step_l0_kv_quantize(module, stream, &query_kv)?;
    let main_kv = flash_dspark_context_kv(module, stream, stage, main_x, start_pos)?;
    let mut cache = state.attention_cache_bf16();
    let slot = start_pos % v4_state::KV_WINDOW;
    cache[slot * spec.q_head_dim..(slot + 1) * spec.q_head_dim].copy_from_slice(&main_kv);
    cache.extend_from_slice(&query_kv);
    let topk = v4_state::flash_dspark_b5_topk(start_pos)?;
    let sparse = step_sparse_attention(module, stream, &q, &cache, &stage.weights.attn_sink, &topk)?;
    let sparse = step_l0_rope(module, stream, &sparse, &query_freqs, spec.q_heads, true)?;
    let wo_a = step_l0_wo_a_for(spec, module, stream, &sparse, &stage.weights.wo_a)?;
    let attention = step_l0_fp8_linear(module, stream, &wo_a, &stage.weights.wo_b, false)?.1;
    let after_attention =
        step_l0_hc_post(module, stream, &attention, x, &attn_hc.post, &attn_hc.comb)?;
    let output = flash_dspark_moe(context, module, stream, stage, &after_attention)?;
    state.append_kv_row(start_pos, &main_kv)?;
    Ok(output)
}
