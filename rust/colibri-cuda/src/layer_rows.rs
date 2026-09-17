use super::*;

/// How many threads the Flash B1 routed prefetch uses to read this layer's
/// experts: one per routed expert, which is the shape the disk serves well
/// (3.7 GB/s against 1.35 GB/s single-threaded, 2026-09-16 measurements).
///
/// This is only worth having next to the resident fixed payload. Without it the
/// fixed stream takes back whatever the routed read gives up and the wall does
/// not move (91.4 s against 91.9 s for eight steps); with it the same change is
/// 73.1 s against 51.2 s.
pub(crate) fn routed_read_threads() -> usize {
    std::env::var("COLIBRI_V4_ROUTED_READ_THREADS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|threads| (1..=16).contains(threads))
        .unwrap_or(6)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_real_layer_attention_rows(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    flash_cublas: Option<&FlashCublasSession>,
    runtime: &LoadedLayerRuntime,
    hc_hidden: &[u16],
    start_pos: usize,
    state: &mut v4_state::V4LayerPersistentState,
    cublas: Option<&cublas_exact::CublasExact>,
    mut strict_scratch: Option<&mut StrictCompressorScratch>,
) -> Result<RealAttentionRun, Box<dyn Error>> {
    let weights = &runtime.weights;
    let spec = weights.spec;
    let mhc_linear_arm = flash_mhc_linear_arm(spec)?;
    let row_width = v4_state::HC * spec.hidden;
    let rows = hc_hidden.len() / row_width;
    if rows == 0
        || rows > 2
        || hc_hidden.len() != rows * row_width
        || state.layer() != weights.layer
        || state.committed_kv_rows() != start_pos
    {
        return Err(format!(
            "layer {} real attention input/state contract mismatch",
            weights.layer
        )
        .into());
    }
    let attention_started = std::time::Instant::now();
    let profile_enabled = bench_telemetry_enabled();
    let mut profile = AttentionProfile::default();
    macro_rules! profile_step {
        ($field:ident, $operation:expr) => {{
            let started = std::time::Instant::now();
            let events = profile_enabled
                .then(|| {
                    Ok::<_, Box<dyn Error>>((
                        stream
                            .context()
                            .new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?,
                        stream
                            .context()
                            .new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?,
                    ))
                })
                .transpose()?;
            if let Some((start, _)) = events.as_ref() {
                start.record(stream)?;
            }
            let value = $operation?;
            if let Some((start, end)) = events.as_ref() {
                end.record(stream)?;
                end.synchronize()?;
                profile.$field.gpu_event_ns = profile
                    .$field
                    .gpu_event_ns
                    .saturating_add((start.elapsed_ms(end)? as f64 * 1_000_000.0) as u64);
                profile.$field.sync_wall_ns = profile.$field.sync_wall_ns.saturating_add(
                    u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
                );
            }
            value
        }};
    }
    let capture_stages = std::env::var_os("COLIBRI_V4_WAVEFRONT_ATTENTION_BENCH").is_some();
    let parity_dump = flash_parity_dump_dir(spec, weights.layer, start_pos, rows)?;
    if let Some(dir) = parity_dump.as_deref() {
        flash_parity_dump_u16(
            dir,
            "input_hidden_bf16",
            &[rows, v4_state::HC, spec.hidden],
            hc_hidden,
        )?;
    }
    let mut stage_digests = Vec::new();
    let mut freqs = Vec::with_capacity(rows * v4_state::Q_ROPE_DIM);
    for row in 0..rows {
        freqs.extend(v4_state::attention_rope_freqs(
            spec,
            weights.layer,
            start_pos + row,
        )?);
    }
    if optional_one_flag("COLIBRI_V4_ROPE_TRACE")?
        && start_pos == 1
        && rows == 1
        && weights.layer <= 2
    {
        let ratio = v4_state::compression_ratio_for_spec_layer(spec, weights.layer)?;
        let (scheme, theta, yarn_factor) = match ratio {
            v4_state::CompressionRatio::WindowOnly => ("base", 10_000, None),
            v4_state::CompressionRatio::Csa4 | v4_state::CompressionRatio::Hca128 => {
                ("compressed", 160_000, Some(16))
            }
        };
        eprintln!(
            "{}",
            serde_json::json!({
                "event": "v4-rope-trace",
                "layer": weights.layer,
                "position": start_pos,
                "compression_ratio": ratio.get(),
                "scheme": scheme,
                "theta": theta,
                "yarn_factor": yarn_factor,
            })
        );
    }
    let attn_hc = profile_step!(
        mhc,
        run_l0_mhc_pre_with_arm(
            module,
            stream,
            hc_hidden,
            &weights.hc_attn_fn,
            &weights.hc_attn_scale,
            &weights.hc_attn_base,
            mhc_linear_arm,
            parity_dump.is_some(),
            flash_cublas,
        )
    );
    maybe_write_v4_audit_f32(weights.layer, "attn_hc_post", &attn_hc.post)?;
    maybe_write_v4_audit_f32(weights.layer, "attn_hc_comb", &attn_hc.comb)?;
    if let Some(dir) = parity_dump.as_deref() {
        let debug = attn_hc
            .debug
            .as_ref()
            .ok_or("Flash parity dump did not retain mHC debug state")?;
        flash_parity_dump_f32(
            dir,
            "attn_hc_mixes_f32",
            &[rows, v4_state::MIX],
            &debug.mixes,
        )?;
        flash_parity_dump_f32(
            dir,
            "attn_hc_sinkhorn_trace_f32",
            &[rows, 41, v4_state::HC, v4_state::HC],
            &debug.sinkhorn_trace,
        )?;
        flash_parity_dump_f32(dir, "attn_hc_pre_f32", &[rows, v4_state::HC], &debug.pre)?;
        flash_parity_dump_f32(
            dir,
            "attn_hc_post_f32",
            &[rows, v4_state::HC],
            &attn_hc.post,
        )?;
        flash_parity_dump_f32(
            dir,
            "attn_hc_comb_f32",
            &[rows, v4_state::HC, v4_state::HC],
            &attn_hc.comb,
        )?;
        if mhc_linear_arm == FlashMhcLinearArm::Cublas {
            let raw_dot = debug
                .raw_dot
                .as_deref()
                .ok_or("Flash cuBLAS mHC parity dump did not retain raw dots")?;
            let rsqrt = debug
                .rsqrt
                .as_deref()
                .ok_or("Flash cuBLAS mHC parity dump did not retain rsqrt")?;
            flash_parity_dump_f32(dir, "hc_raw_dot_f32", &[rows, v4_state::MIX], raw_dot)?;
            flash_parity_dump_f32(dir, "hc_rsqrt_f32", &[rows], rsqrt)?;
            let rsqrt_debug = debug
                .rsqrt_debug
                .as_deref()
                .ok_or("Flash cuBLAS mHC parity dump did not retain rsqrt debug words")?;
            if rsqrt_debug.len() != rows * FLASH_MHC_RSQRT_DEBUG_FIELDS {
                return Err("Flash cuBLAS mHC rsqrt debug geometry mismatch".into());
            }
            flash_parity_dump_u32(
                dir,
                "hc_rsqrt_debug_u32",
                &[rows, FLASH_MHC_RSQRT_DEBUG_FIELDS],
                rsqrt_debug,
            )?;
            flash_parity_dump_f32(
                dir,
                "hc_scaled_mixes_f32",
                &[rows, v4_state::MIX],
                &debug.mixes,
            )?;
        }
    }
    if capture_stages {
        stage_digests.extend([
            ("mhc_collapsed", fnv1a_u16(&attn_hc.collapsed)),
            ("mhc_post", fnv1a_f32(&attn_hc.post)),
            ("mhc_comb", fnv1a_f32(&attn_hc.comb)),
        ]);
    }
    let attn_input =
        if spec == &v4_spec::FLASH_0731 && spec.hidden == 4_096 && weights.attn_norm.len() == 4_096
        {
            let input = DeviceBuffer::from_host(stream, &attn_hc.collapsed)?;
            let mut output = DeviceBuffer::<u16>::zeroed(stream, input.len())?;
            let mut trace = DeviceBuffer::<f32>::zeroed(stream, 0)?;
            profile_step!(
                attn_norm,
                run_flash_learned_rmsnorm_4096_device_into(
                    module,
                    stream,
                    &input,
                    &weights.attn_norm,
                    &mut output,
                    &mut trace,
                )
            );
            profile_step!(attn_norm, output.to_host_vec(stream))
        } else {
            profile_step!(
                attn_norm,
                run_l0_rmsnorm(
                    module,
                    stream,
                    &attn_hc.collapsed,
                    &weights.attn_norm,
                    spec.hidden,
                )
            )
        };
    if let Some(dir) = parity_dump.as_deref() {
        flash_parity_dump_u16(dir, "attn_norm_bf16", &[rows, spec.hidden], &attn_input)?;
    }
    if capture_stages {
        stage_digests.push(("attn_input", fnv1a_u16(&attn_input)));
    }
    let (_, q_lora_raw) = profile_step!(
        q_a_projection,
        run_l0_fp8_linear(module, stream, &attn_input, &weights.wq_a, false)
    );
    if let Some(dir) = parity_dump.as_deref() {
        flash_parity_dump_u16(
            dir,
            "q_a_linear_bf16",
            &[rows, spec.q_lora_rank],
            &q_lora_raw,
        )?;
    }
    if capture_stages {
        stage_digests.push(("q_lora_raw", fnv1a_u16(&q_lora_raw)));
    }
    maybe_write_v4_audit_u16(weights.layer, "q_lora_raw", &q_lora_raw)?;
    let q_lora = profile_step!(
        q_a_norm,
        run_l0_rmsnorm(
            module,
            stream,
            &q_lora_raw,
            &weights.q_norm,
            weights.wq_a.out_features,
        )
    );
    if let Some(dir) = parity_dump.as_deref() {
        flash_parity_dump_u16(dir, "q_a_norm_bf16", &[rows, spec.q_lora_rank], &q_lora)?;
    }
    if capture_stages {
        stage_digests.push(("q_lora", fnv1a_u16(&q_lora)));
    }
    let (_, q_raw) = profile_step!(
        q_b_projection,
        run_l0_fp8_linear(module, stream, &q_lora, &weights.wq_b, true)
    );
    if let Some(dir) = parity_dump.as_deref() {
        flash_parity_dump_u16(
            dir,
            "q_b_logical_bf16",
            &[rows, spec.q_heads, spec.q_head_dim],
            &q_raw,
        )?;
    }
    if capture_stages {
        stage_digests.push(("q_raw", fnv1a_u16(&q_raw)));
    }
    maybe_write_v4_audit_u16(weights.layer, "q_raw", &q_raw)?;
    let unit_norm = DeviceBuffer::<u16>::zeroed(stream, 0)?;
    let q = profile_step!(
        q_b_norm,
        run_l0_rmsnorm(module, stream, &q_raw, &unit_norm, v4_state::Q_HEAD_DIM)
    );
    if let Some(dir) = parity_dump.as_deref() {
        flash_parity_dump_u16(
            dir,
            "q_b_norm_bf16",
            &[rows, spec.q_heads, spec.q_head_dim],
            &q,
        )?;
    }
    if capture_stages {
        stage_digests.push(("q_norm", fnv1a_u16(&q)));
    }
    maybe_write_v4_audit_u16(weights.layer, "q_norm", &q)?;
    let q = profile_step!(
        q_rope,
        run_l0_rope(module, stream, &q, &freqs, spec.q_heads, false)
    );
    if let Some(dir) = parity_dump.as_deref() {
        flash_parity_dump_u16(
            dir,
            "q_rope_bf16",
            &[rows, spec.q_heads, spec.q_head_dim],
            &q,
        )?;
    }
    if capture_stages {
        stage_digests.push(("q_rope", fnv1a_u16(&q)));
    }
    let (_, kv) = profile_step!(
        kv_projection,
        run_l0_fp8_linear(module, stream, &attn_input, &weights.wkv, false)
    );
    if let Some(dir) = parity_dump.as_deref() {
        flash_parity_dump_u16(dir, "wkv_linear_bf16", &[rows, spec.q_head_dim], &kv)?;
    }
    if capture_stages {
        stage_digests.push(("kv_raw", fnv1a_u16(&kv)));
    }
    let kv = profile_step!(
        kv_norm,
        run_l0_rmsnorm(module, stream, &kv, &weights.kv_norm, v4_state::Q_HEAD_DIM)
    );
    if let Some(dir) = parity_dump.as_deref() {
        flash_parity_dump_u16(dir, "kv_norm_bf16", &[rows, spec.q_head_dim], &kv)?;
    }
    if capture_stages {
        stage_digests.push(("kv_norm", fnv1a_u16(&kv)));
    }
    let kv = profile_step!(
        kv_rope_quantize,
        run_l0_rope(module, stream, &kv, &freqs, 1, false)
    );
    if let Some(dir) = parity_dump.as_deref() {
        flash_parity_dump_u16(dir, "k_rope_bf16", &[rows, 1, spec.q_head_dim], &kv)?;
    }
    if capture_stages {
        stage_digests.push(("kv_rope", fnv1a_u16(&kv)));
    }
    let kv = profile_step!(kv_rope_quantize, run_l0_kv_quantize(module, stream, &kv));
    if let Some(dir) = parity_dump.as_deref() {
        flash_parity_dump_u16(dir, "cache_kv_row_bf16", &[rows, spec.q_head_dim], &kv)?;
    }
    if capture_stages {
        stage_digests.push(("kv_ready", fnv1a_u16(&kv)));
    }
    let compressor = if weights.compression_ratio == v4_state::CompressionRatio::WindowOnly {
        None
    } else {
        Some(profile_step!(
            compressor,
            run_layer_compressor(
                module,
                stream,
                &attn_input,
                weights,
                start_pos,
                cublas,
                strict_scratch.as_deref_mut(),
            )
        ))
    };
    if let Some(compressor) = compressor.as_ref() {
        if let Some(dir) = parity_dump
            .as_deref()
            .filter(|_| weights.layer == 2 && start_pos == 0)
        {
            let width = compressor.kv_f32.len() / rows;
            if width == 0 || compressor.kv_f32.len() != rows * width {
                return Err("Flash L2 main Compressor precommit geometry mismatch".into());
            }
            flash_parity_dump_f32(
                dir,
                "main_compressor_kv_precommit_f32",
                &[rows, 1, width],
                &compressor.kv_f32,
            )?;
            flash_parity_dump_f32(
                dir,
                "main_compressor_score_precommit_f32",
                &[rows, 1, width],
                &compressor.score_plus_ape_f32,
            )?;
        }
        if l0_pos0_compressor_probe_enabled(weights.layer, start_pos) {
            print_l0_pos0_compressor_probe(
                "host_pre_commit",
                &compressor.kv_f32,
                &compressor.score_f32,
                &compressor.score_plus_ape_f32,
            );
        }
        if capture_stages {
            stage_digests.extend([
                ("compressor_kv", fnv1a_f32(&compressor.kv_f32)),
                ("compressor_score", fnv1a_f32(&compressor.score_f32)),
                (
                    "compressor_score_plus_ape",
                    fnv1a_f32(&compressor.score_plus_ape_f32),
                ),
            ]);
        }
    }
    let indexer = if let Some(indexer) = weights.indexer.as_ref() {
        Some(profile_step!(
            indexer,
            run_csa_indexer_projections(
                spec,
                module,
                stream,
                &attn_input,
                &q_lora,
                &freqs,
                indexer,
                start_pos,
                cublas,
                strict_scratch.as_deref_mut(),
            )
        ))
    } else {
        None
    };
    if let Some((dir, indexer)) = parity_dump
        .as_deref()
        .filter(|_| weights.layer == 2 && start_pos == 0)
        .zip(indexer.as_ref())
    {
        let width = indexer.compressor.kv_f32.len() / rows;
        if width == 0 || indexer.compressor.kv_f32.len() != rows * width {
            return Err("Flash L2 indexer Compressor precommit geometry mismatch".into());
        }
        flash_parity_dump_f32(
            dir,
            "indexer_compressor_kv_precommit_f32",
            &[rows, 1, width],
            &indexer.compressor.kv_f32,
        )?;
        flash_parity_dump_f32(
            dir,
            "indexer_compressor_score_precommit_f32",
            &[rows, 1, width],
            &indexer.compressor.score_plus_ape_f32,
        )?;
    }
    let commit_started = std::time::Instant::now();
    let (topk, indexer_topk_gpu_ns) = profile_step!(
        commit_topk,
        commit_layer_attention_state(
            module,
            stream,
            weights,
            state,
            &kv,
            compressor.as_ref(),
            indexer.as_ref(),
            start_pos,
        )
    );
    let commit_ns = u64::try_from(commit_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
    maybe_write_v4_audit_f32(
        weights.layer,
        "main_compressor_kv_state",
        state.compressor().kv(),
    )?;
    maybe_write_v4_audit_f32(
        weights.layer,
        "main_compressor_score_state",
        state.compressor().score(),
    )?;
    let attention_topk_digest = fnv1a_i32(&topk.indices);
    let cache = state.attention_cache_bf16();
    if let Some(dir) = parity_dump.as_deref() {
        flash_parity_dump_u16(
            dir,
            "cache_kv_bf16",
            &[cache.len() / spec.q_head_dim, spec.q_head_dim],
            &cache,
        )?;
    }
    let sparse = match sparse_attention_arm_from_env()? {
        SparseAttentionArm::Reference => {
            profile_step!(
                sparse_score,
                run_sparse_attention(module, stream, &q, &cache, &weights.attn_sink, &topk)
            )
        }
        SparseAttentionArm::ScoreReuse => {
            profile_step!(
                sparse_score,
                run_sparse_attention_score_reuse(
                    module,
                    stream,
                    &q,
                    &cache,
                    &weights.attn_sink,
                    &topk
                )
            )
        }
        SparseAttentionArm::Sm89Crossover if topk.per_row >= SM89_SCORE_REUSE_MIN_TOPK => {
            profile_step!(
                sparse_score,
                run_sparse_attention_score_reuse(
                    module,
                    stream,
                    &q,
                    &cache,
                    &weights.attn_sink,
                    &topk
                )
            )
        }
        SparseAttentionArm::Sm89Crossover => {
            profile_step!(
                sparse_score,
                run_sparse_attention(module, stream, &q, &cache, &weights.attn_sink, &topk)
            )
        }
    };
    if let Some(dir) = parity_dump.as_deref() {
        flash_parity_dump_u16(
            dir,
            "attn_context_bf16",
            &[rows, spec.q_heads, spec.q_head_dim],
            &sparse,
        )?;
    }
    if capture_stages {
        stage_digests.push(("sparse", fnv1a_u16(&sparse)));
    }
    let sparse = profile_step!(
        sparse_inverse_rope,
        run_l0_rope(module, stream, &sparse, &freqs, spec.q_heads, true)
    );
    if let Some(dir) = parity_dump.as_deref() {
        flash_parity_dump_u16(
            dir,
            "attn_context_inverse_rope_bf16",
            &[rows, spec.q_heads, spec.q_head_dim],
            &sparse,
        )?;
    }
    if capture_stages {
        stage_digests.push(("sparse_inverse", fnv1a_u16(&sparse)));
    }
    let wo_a = profile_step!(
        wo_a,
        run_l0_wo_a_for(spec, module, stream, &sparse, &weights.wo_a)
    );
    if let Some(dir) = parity_dump.as_deref() {
        flash_parity_dump_u16(
            dir,
            "wo_a_logical_bf16",
            &[rows, spec.o_groups, spec.o_lora_rank],
            &wo_a,
        )?;
    }
    if capture_stages {
        stage_digests.push(("wo_a", fnv1a_u16(&wo_a)));
    }
    maybe_write_v4_audit_u16(weights.layer, "wo_a", &wo_a)?;
    let (_, attention) = profile_step!(
        wo_b,
        run_l0_fp8_linear(module, stream, &wo_a, &weights.wo_b, false)
    );
    if let Some(dir) = parity_dump.as_deref() {
        flash_parity_dump_u16(dir, "wo_b_hidden_bf16", &[rows, spec.hidden], &attention)?;
    }
    if capture_stages {
        stage_digests.push(("attention", fnv1a_u16(&attention)));
    }
    if let Some(dir) = parity_dump.as_deref() {
        let audit = run_l0_hc_post_trace(
            module,
            stream,
            &attention,
            hc_hidden,
            &attn_hc.post,
            &attn_hc.comb,
            16_102,
        )?;
        flash_parity_dump_u32(dir, "attn_hc_post_index16102_u32", &[audit.len()], &audit)?;
    }
    maybe_write_v4_audit_u16(weights.layer, "attention", &attention)?;
    let output = profile_step!(
        hc_post,
        run_l0_hc_post(
            module,
            stream,
            &attention,
            hc_hidden,
            &attn_hc.post,
            &attn_hc.comb,
        )
    );
    if let Some(dir) = parity_dump.as_deref() {
        flash_parity_dump_u16(
            dir,
            "post_attn_hidden_bf16",
            &[rows, v4_state::HC, spec.hidden],
            &output,
        )?;
    }
    if capture_stages {
        stage_digests.push(("attention_post", fnv1a_u16(&output)));
    }
    maybe_write_v4_audit_u16(weights.layer, "attention_post", &output)?;
    Ok(RealAttentionRun {
        output,
        topk: Some(topk.indices.clone()),
        stage_digests,
        attention_topk_digest,
        attention_topk_per_row: topk.per_row,
        indexer_topk_gpu_ns,
        attention_ns: u64::try_from(attention_started.elapsed().as_nanos())
            .unwrap_or(u64::MAX)
            .saturating_sub(commit_ns),
        commit_ns,
        device_transfer: None,
        profile: profile_enabled.then_some(profile),
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_real_layer_rows<'module, 'slots, 'lane>(
    context: &std::sync::Arc<CudaContext>,
    module: &'module kernels::LoadedModule,
    stream: &std::sync::Arc<CudaStream>,
    flash_cublas: Option<&FlashCublasSession>,
    runtime: &LoadedLayerRuntime,
    qd4_lanes: Option<&mut [ExpertSlot<'lane>]>,
    qd4_immutable_cache: Option<&mut Qd4ImmutableExpertCache<'lane>>,
    mut expert_slots: Option<&'slots mut BTreeMap<u16, ExpertSlot<'module>>>,
    hash_ids: &[i64],
    hc_hidden: &[u16],
    token_ids: &[i64],
    start_pos: usize,
    state: Option<&mut v4_state::V4LayerPersistentState>,
    device_state: Option<&mut DeviceTokenState>,
    device_rope_lut: Option<&DeviceRequestRopeLut>,
    device_observe_topk: bool,
    critical_path_origin: Option<std::time::Instant>,
    capture_routed_contributions: bool,
    generic_device: Option<(
        &mut DeviceLayerPersistentState,
        &mut DeviceRequestScratch,
        Option<&[u16]>,
    )>,
    strict_scratch: Option<&mut StrictCompressorScratch>,
    mut routed_slot_scratch: Option<&mut RoutedSlotScratch<'module>>,
) -> Result<RealLayerRun, Box<dyn Error>> {
    let weights = &runtime.weights;
    let spec = weights.spec;
    let mhc_linear_arm = flash_mhc_linear_arm(spec)?;
    let row_width = v4_state::HC * spec.hidden;
    let rows = hc_hidden.len() / row_width;
    if rows == 0
        || rows > 2
        || hc_hidden.len() != rows * row_width
        || token_ids.len() != rows
        || device_state.is_none()
            && generic_device.is_none()
            && state.as_ref().is_none_or(|state| {
                state.layer() != weights.layer || state.committed_kv_rows() != start_pos
            })
    {
        return Err(format!(
            "layer {} real runner input/state contract mismatch",
            weights.layer
        )
        .into());
    }
    if capture_routed_contributions && rows != 1 {
        return Err("routed contribution audit requires exactly one row".into());
    }
    if device_state.is_some() && generic_device.is_some() {
        return Err("real layer runner cannot mix frozen and generic device state".into());
    }
    let parity_dump = flash_parity_dump_dir(spec, weights.layer, start_pos, rows)?;
    if parity_dump.is_some() && qd4_lanes.is_some() {
        return Err("Flash parity dump requires the single-stream routed path".into());
    }
    let mut stage = LayerStageTelemetry::default();
    let mut critical_path_timeline = critical_path_origin.map(CriticalPathTimeline::new);
    if let Some(timeline) = critical_path_timeline.as_mut() {
        timeline.attention_start_ns = timeline.now();
    }
    let mut generic_prepared = None;
    let attention = if let Some((state, scratch, audit_expected_compressed_prefix)) = generic_device
    {
        let prepared = prepare_generic_device_attention_row(
            module,
            stream,
            runtime,
            state,
            scratch,
            hc_hidden,
            start_pos,
            audit_expected_compressed_prefix,
        )?;
        let attention = prepared.attention.clone();
        generic_prepared = Some((state, scratch, prepared, audit_expected_compressed_prefix));
        attention
    } else if let Some(device_state) = device_state {
        run_device_attention_row(
            module,
            stream,
            runtime,
            device_state,
            device_rope_lut.ok_or("device attention row is missing request RoPE LUT")?,
            hc_hidden,
            start_pos,
            device_observe_topk,
        )?
    } else {
        run_real_layer_attention_rows(
            module,
            stream,
            flash_cublas,
            runtime,
            hc_hidden,
            start_pos,
            state.ok_or("host layer runner is missing persistent state")?,
            flash_cublas.map(|session| &session.cublas),
            strict_scratch,
        )?
    };
    let after_attention = attention.output;
    let attention_output = after_attention.clone();
    let attention_topk = attention.topk;
    let device_transfer = attention.device_transfer;
    let attention_topk_digest = attention.attention_topk_digest;
    let attention_topk_per_row = attention.attention_topk_per_row;
    let indexer_topk_gpu_ns = attention.indexer_topk_gpu_ns;
    stage.attention_ns = attention.attention_ns;
    stage.commit_ns = attention.commit_ns;
    stage.attention_profile = attention.profile;
    if let Some(transfer) = device_transfer.as_ref() {
        stage.attention_hidden_h2d_count = transfer.h2d_count;
        stage.attention_hidden_h2d_bytes = transfer.h2d_bytes;
        stage.attention_rope_lut_d2d_count = transfer.rope_lut_d2d_count;
        stage.attention_rope_lut_d2d_bytes = transfer.rope_lut_d2d_bytes;
        stage.attention_terminal_d2h_count = transfer.terminal_d2h_count;
        stage.attention_terminal_d2h_bytes = transfer.terminal_d2h_bytes;
        stage.attention_intermediate_or_audit_d2h_count = transfer.intermediate_or_audit_d2h_count;
        stage.attention_intermediate_or_audit_d2h_bytes = transfer.intermediate_or_audit_d2h_bytes;
    }
    if let Some(timeline) = critical_path_timeline.as_mut() {
        timeline.attention_end_ns = timeline.now();
        timeline.attention_commit_ns = attention.commit_ns;
        timeline.router_start_ns = timeline.now();
    }
    let router_started = std::time::Instant::now();
    let ffn_hc = run_l0_mhc_pre_with_arm(
        module,
        stream,
        &after_attention,
        &weights.hc_ffn_fn,
        &weights.hc_ffn_scale,
        &weights.hc_ffn_base,
        mhc_linear_arm,
        false,
        flash_cublas,
    )?;
    if let Some(dir) = parity_dump.as_deref() {
        let debug = ffn_hc
            .debug
            .as_ref()
            .ok_or("Flash ffn mHC coefficient trace was not retained")?;
        if debug.pre.len() != v4_state::HC {
            return Err("Flash ffn mHC coefficient trace must be one four-lane row".into());
        }
        flash_parity_dump_u16(
            dir,
            "ffn_hc_collapsed_bf16",
            &[rows, spec.hidden],
            &ffn_hc.collapsed,
        )?;
        flash_parity_dump_f32(
            dir,
            "ffn_hc_pre_for_hidden124_f32",
            &[v4_state::HC],
            &debug.pre,
        )?;
        flash_parity_dump_f32(dir, "ffn_hc_post_f32", &[rows, v4_state::HC], &ffn_hc.post)?;
        flash_parity_dump_f32(
            dir,
            "ffn_hc_comb_f32",
            &[rows, v4_state::HC, v4_state::HC],
            &ffn_hc.comb,
        )?;
    }
    stage.router_mhc_ns = ffn_hc.telemetry.wall_ns;
    stage.router_mhc_input_alloc_h2d_sync_ns = ffn_hc.telemetry.input_alloc_h2d_sync_ns;
    stage.router_mhc_workspace_alloc_sync_ns = ffn_hc.telemetry.workspace_alloc_sync_ns;
    stage.router_mhc_workspace_init_gpu_ns = ffn_hc.telemetry.workspace_init_gpu_ns;
    stage.router_mhc_mixing_linear_sync_ns = ffn_hc.telemetry.mixing_linear_sync_ns;
    stage.router_mhc_mixing_linear_gpu_ns = ffn_hc.telemetry.mixing_linear_gpu_ns;
    stage.router_mhc_sinkhorn_normalize_collapse_sync_ns =
        ffn_hc.telemetry.sinkhorn_normalize_collapse_sync_ns;
    stage.router_mhc_sinkhorn_normalize_collapse_gpu_ns =
        ffn_hc.telemetry.sinkhorn_normalize_collapse_gpu_ns;
    stage.router_mhc_output_d2h_sync_ns = ffn_hc.telemetry.output_d2h_sync_ns;
    stage.router_mhc_other_ns = ffn_hc.telemetry.other_ns;
    let router_host_prep_started = std::time::Instant::now();
    let hash_ids = if weights.hash_router {
        if hash_ids.len() != rows * spec.experts_per_token {
            return Err(format!("layer {} hash router rows disappeared", weights.layer).into());
        }
        hash_ids.to_vec()
    } else {
        Vec::new()
    };
    stage.router_host_prep_ns =
        u64::try_from(router_host_prep_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
    let router_parity_dump = flash_router_trace_dump_dir(spec, weights.layer, start_pos, rows)?;
    let router = run_layer_router(
        module,
        stream,
        flash_cublas,
        &ffn_hc.collapsed,
        weights,
        &hash_ids,
        router_parity_dump.is_some(),
    )?;
    if let Some(dir) = router_parity_dump.as_deref() {
        let debug = router
            .debug
            .as_ref()
            .ok_or("Flash parity dump did not retain GPU router debug buffers")?;
        let has_learned_router_rmsnorm = !weights.hash_router;
        if debug.rmsnorm.len() != rows * 3
            || (has_learned_router_rmsnorm
                && debug.rmsnorm_squares.len() != rows * spec.hidden)
            || debug.logits.len() != rows * spec.routed_experts
            || debug.softplus.len() != debug.logits.len()
            || debug.post_scores.len() != debug.logits.len()
            || debug.selected_scores.len() != rows * spec.experts_per_token
            || debug.denom_steps.len() != rows * spec.experts_per_token
        {
            return Err("Flash parity router debug geometry mismatch".into());
        }
        flash_parity_dump_u16(
            dir,
            "router_input_bf16",
            &[rows, spec.hidden],
            &router.input,
        )?;
        if has_learned_router_rmsnorm {
            flash_parity_dump_f32(
                dir,
                "router_rmsnorm_square_f32",
                &[rows, spec.hidden],
                &debug.rmsnorm_squares,
            )?;
            flash_parity_dump_f32(
                dir,
                "router_rmsnorm_sum_sq_f32",
                &[rows],
                &debug.rmsnorm[0..rows],
            )?;
            flash_parity_dump_f32(
                dir,
                "router_rmsnorm_mean_f32",
                &[rows],
                &debug.rmsnorm[rows..rows * 2],
            )?;
            flash_parity_dump_f32(
                dir,
                "router_rmsnorm_inv_f32",
                &[rows],
                &debug.rmsnorm[rows * 2..rows * 3],
            )?;
        }
        flash_parity_dump_f32(
            dir,
            "router_gate_logits_f32",
            &[rows, spec.routed_experts],
            &debug.logits,
        )?;
        flash_parity_dump_f32(
            dir,
            "router_softplus_f32",
            &[rows, spec.routed_experts],
            &debug.softplus,
        )?;
        flash_parity_dump_f32(
            dir,
            "router_post_score_f32",
            &[rows, spec.routed_experts],
            &debug.post_scores,
        )?;
        flash_parity_dump_f32(
            dir,
            "selected_scores_f32",
            &[rows, spec.experts_per_token],
            &debug.selected_scores,
        )?;
        flash_parity_dump_f32(
            dir,
            "route_denom_steps_f32",
            &[rows, spec.experts_per_token],
            &debug.denom_steps,
        )?;
        let route_denoms = debug
            .denom_steps
            .chunks_exact(spec.experts_per_token)
            .map(|steps| steps[spec.experts_per_token - 1])
            .collect::<Vec<_>>();
        flash_parity_dump_f32(dir, "route_denom_f32", &[rows], &route_denoms)?;
        flash_parity_dump_i32(
            dir,
            "route_ids_i32",
            &[rows, spec.experts_per_token],
            &router.routes.ids,
        )?;
        flash_parity_dump_f32(
            dir,
            "route_weights_f32",
            &[rows, spec.experts_per_token],
            &router.routes.weights,
        )?;
    }
    let ffn_input = router.input;
    let routes = router.routes;
    stage.router_alloc_h2d_sync_ns = router.telemetry.alloc_h2d_sync_ns;
    stage.router_workspace_alloc_ns = router.telemetry.workspace_alloc_ns;
    stage.router_kernel_select_sync_ns = router.telemetry.kernel_select_sync_ns;
    stage.router_kernel_select_gpu_ns = router.telemetry.kernel_select_gpu_ns;
    stage.router_d2h_sync_ns = router.telemetry.d2h_sync_ns;
    maybe_write_v4_audit_u16(weights.layer, "ffn_input", &ffn_input)?;
    stage.router_ns = u64::try_from(router_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
    stage.router_other_ns = stage.router_ns.saturating_sub(stage.router_accounted_ns());
    if let Some(timeline) = critical_path_timeline.as_mut() {
        timeline.router_end_ns = timeline.now();
        timeline.shared_start_ns = timeline.now();
    }
    let flash_routed_prefetch =
        flash_b1_prefetch_enabled(spec, rows)? && qd4_lanes.is_none() && expert_slots.is_none();
    let mut prefetched_routed_experts = None;
    // The shared window runs the shared MLP on this thread while the routed
    // read workers fetch this layer's experts, so its wall is whichever of the
    // two finishes last. Time both from the inside to tell them apart.
    let shared_probe = flash_shared_probe_enabled();
    let mut shared_probe_mlp_us = 0_u128;
    let mut shared_probe_worker_sum_us = 0_u128;
    let mut shared_probe_worker_max_us = 0_u128;
    let mut shared_probe_workers = 0_usize;
    let shared_started = std::time::Instant::now();
    let (shared, shared_bf16) = if flash_routed_prefetch {
        if routes.ids.len() != spec.experts_per_token {
            return Err("Flash B1 routed prefetch requires exactly one route row".into());
        }
        let route_ids = routes.ids.clone();
        let read_threads = routed_read_threads();
        let loaded = std::thread::scope(|scope| -> Result<_, Box<dyn Error>> {
            // One worker per read thread; the chunk size keeps the experts in
            // their original order when the chunks are concatenated back.
            let chunk_size = route_ids.len().div_ceil(read_threads);
            let workers = route_ids
                .chunks(chunk_size)
                .map(|chunk| {
                    scope.spawn(move || {
                        let worker_started = std::time::Instant::now();
                        let experts = chunk
                            .iter()
                            .map(|expert_id| {
                                let expert_id =
                                    u16::try_from(*expert_id).map_err(|error| error.to_string())?;
                                let expert = v4_expert::RoutedExpert::load_layer_for(
                                    spec,
                                    &runtime.expert_shard,
                                    weights.layer,
                                    expert_id,
                                )
                                .map_err(|error| error.to_string())?;
                                Ok::<_, String>((expert_id, expert))
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        let worker_us =
                            u128::try_from(worker_started.elapsed().as_micros()).unwrap_or(u128::MAX);
                        Ok::<_, String>((experts, worker_us))
                    })
                })
                .collect::<Vec<_>>();
            let shared = if parity_dump.is_some() {
                run_layer_shared_expert_with_bf16(module, stream, &ffn_input, weights)?
            } else {
                (
                    run_layer_shared_expert(module, stream, &ffn_input, weights)?,
                    Vec::new(),
                )
            };
            shared_probe_mlp_us =
                u128::try_from(shared_started.elapsed().as_micros()).unwrap_or(u128::MAX);
            let mut experts = Vec::with_capacity(route_ids.len());
            for worker in workers {
                let (loaded, worker_us) = worker
                    .join()
                    .map_err(|_| "Flash B1 routed prefetch worker panicked")?
                    .map_err(|error| format!("Flash B1 routed prefetch worker failed: {error}"))?;
                shared_probe_worker_sum_us = shared_probe_worker_sum_us.saturating_add(worker_us);
                shared_probe_worker_max_us = shared_probe_worker_max_us.max(worker_us);
                shared_probe_workers += 1;
                experts.extend(loaded);
            }
            Ok((shared, experts))
        })?;
        prefetched_routed_experts = Some(loaded.1);
        loaded.0
    } else {
        let shared = if parity_dump.is_some() {
            run_layer_shared_expert_with_bf16(module, stream, &ffn_input, weights)?
        } else {
            (
                run_layer_shared_expert(module, stream, &ffn_input, weights)?,
                Vec::new(),
            )
        };
        shared_probe_mlp_us =
            u128::try_from(shared_started.elapsed().as_micros()).unwrap_or(u128::MAX);
        shared
    };
    if let Some(dir) = parity_dump.as_deref() {
        flash_parity_dump_u16(dir, "shared_out_bf16", &[rows, spec.hidden], &shared_bf16)?;
    }
    stage.shared_ns = u64::try_from(shared_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
    if shared_probe {
        eprintln!(
            "SHAREDPROBE layer={} rows={} prefetch={} window_us={} mlp_us={} workers={} \
             worker_sum_us={} worker_max_us={} join_wait_us={}",
            weights.layer,
            rows,
            flash_routed_prefetch,
            stage.shared_ns / 1_000,
            shared_probe_mlp_us,
            shared_probe_workers,
            shared_probe_worker_sum_us,
            shared_probe_worker_max_us,
            stage.shared_ns / 1_000 - shared_probe_mlp_us,
        );
    }
    if let Some(timeline) = critical_path_timeline.as_mut() {
        timeline.shared_end_ns = timeline.now();
    }
    maybe_run_routed_qd4_benchmark(
        context,
        module,
        stream,
        &runtime.expert_shard,
        weights.layer,
        &ffn_input,
        &routes,
        &shared,
    )?;
    let routed_started = std::time::Instant::now();
    if let Some(timeline) = critical_path_timeline.as_mut() {
        timeline.routed_start_ns = timeline.now();
    }
    let (moe, routed_contributions) = if let Some(lanes) = qd4_lanes {
        if rows != 1 {
            return Err("QD4 full inference currently requires B=1/S=1".into());
        }
        let immutable_cache_enabled = qd4_immutable_cache.is_some();
        let plans: [v4_expert::RoutedExpertRangePlan; v4_router::V4_TOPK] = routes
            .ids
            .iter()
            .map(|&expert| {
                v4_expert::RoutedExpertRangePlan::new(
                    &runtime.expert_shard,
                    weights.layer,
                    u16::try_from(expert)?,
                )
            })
            .collect::<Result<Vec<_>, Box<dyn Error>>>()?
            .try_into()
            .map_err(|_| "QD4 route plan count changed")?;
        let qd4 = run_qd4_routed_once(
            lanes,
            qd4_immutable_cache,
            &runtime.expert_shard,
            weights.layer,
            &ffn_input,
            &routes,
            &shared,
            &plans,
            critical_path_origin,
            capture_routed_contributions,
        )?;
        // QD4 lanes overlap: the phase-profile child is their elapsed wall
        // scope, not the six lane-duration sum.
        stage.routed_read_ns = qd4.sample.read_wall_ns;
        stage.routed_setup_ns = 0;
        stage.routed_h2d_ns = qd4.sample.h2d_ns;
        stage.routed_kernel_ns = qd4.sample.kernel_ns;
        stage.routed_source_bytes = qd4
            .cache_miss_count
            .checked_mul(v4_expert::V4_ROUTED_EXPERT_BYTES)
            .ok_or("QD4 immutable cache routed source byte count overflow")?;
        if qd4.cache_miss_count == 0 && !immutable_cache_enabled {
            stage.routed_source_bytes = v4_expert::V4_ROUTED_EXPERT_BYTES * v4_router::V4_TOPK;
        }
        stage.routed_h2d_bytes = stage.routed_source_bytes;
        stage.routed_expert_loads = stage.routed_source_bytes / v4_expert::V4_ROUTED_EXPERT_BYTES;
        stage.routed_parallel_batch_wall_ns = Some(qd4.sample.inclusive_ns);
        stage.routed_parallel_read_horizon_ns = Some(qd4.sample.read_wall_ns);
        if let Some(timeline) = critical_path_timeline.as_mut() {
            timeline.routed = qd4.sample.timeline.clone();
        }
        (qd4.output, qd4.contributions)
    } else {
        let mut contributions: Vec<Vec<Option<Vec<f32>>>> = (0..rows)
            .map(|_| (0..spec.experts_per_token).map(|_| None).collect())
            .collect();
        let mut flash_traces = parity_dump.as_ref().map(|_| {
            (0..spec.experts_per_token)
                .map(|_| None)
                .collect::<Vec<Option<FlashExpertTrace>>>()
        });
        if let Some(experts) = prefetched_routed_experts.take() {
            if rows != 1 || experts.len() != spec.experts_per_token {
                return Err("Flash B1 routed prefetch returned an incomplete route row".into());
            }
            stage.routed_prefetched = true;
            // The experts this layer visits all share one shape, so keep one
            // slot and restage it instead of page-locking six sets of staging
            // buffers. A request-scoped scratch keeps that slot across layers
            // too: the page-lock is 17 ms and would otherwise be paid once per
            // layer instead of once per request.
            let mut recycled_slot = match routed_slot_scratch.as_deref_mut() {
                Some(scratch) => scratch.take_slot(),
                None => None,
            };
            for (route_slot, (expert_id, expert)) in experts.into_iter().enumerate() {
                if routes.ids[route_slot] != i32::from(expert_id) {
                    return Err("Flash B1 routed prefetch changed canonical route order".into());
                }
                stage.routed_read_ns = stage
                    .routed_read_ns
                    .saturating_add((expert.read_ms * 1_000_000.0) as u64);
                stage.routed_alloc_ns = stage
                    .routed_alloc_ns
                    .saturating_add((expert.alloc_ms * 1_000_000.0) as u64);
                stage.routed_source_bytes =
                    stage.routed_source_bytes.saturating_add(expert.h2d_bytes());
                stage.routed_expert_loads = stage.routed_expert_loads.saturating_add(1);
                let setup_started = std::time::Instant::now();
                let mut slot = ExpertSlot::new_or_restage(
                    recycled_slot.take(),
                    context,
                    module,
                    stream,
                    &expert,
                )?;
                stage.routed_setup_ns = stage.routed_setup_ns.saturating_add(
                    u64::try_from(setup_started.elapsed().as_nanos()).unwrap_or(u64::MAX),
                );
                stage.routed_h2d_ns = stage
                    .routed_h2d_ns
                    .saturating_add((slot.weight_h2d_ms * 1_000_000.0) as u64);
                stage.routed_h2d_bytes = stage.routed_h2d_bytes.saturating_add(expert.h2d_bytes());
                let input = &ffn_input[..spec.hidden];
                let (output, kernel_ns, trace) = if parity_dump.is_some() {
                    let (output, kernel_ns, trace) =
                        slot.execute_with_flash_trace(input, routes.weights[route_slot])?;
                    (output, kernel_ns, Some(trace))
                } else {
                    let (output, kernel_ns) = slot.execute(input, routes.weights[route_slot])?;
                    (output, kernel_ns, None)
                };
                stage.routed_kernel_ns = stage.routed_kernel_ns.saturating_add(kernel_ns);
                contributions[0][route_slot] = Some(output);
                if let Some(traces) = flash_traces.as_mut() {
                    traces[route_slot] = trace;
                }
                recycled_slot = Some(slot);
            }
            if let (Some(scratch), Some(slot)) =
                (routed_slot_scratch.as_deref_mut(), recycled_slot.take())
            {
                scratch.put_slot(slot);
            }
        } else {
            let mut expert_union = routes
                .ids
                .iter()
                .map(|&id| u16::try_from(id))
                .collect::<Result<Vec<_>, _>>()?;
            expert_union.sort_unstable();
            expert_union.dedup();
            for expert_id in expert_union {
                let mut temporary_slot = None;
                let mut load_slot = || -> Result<ExpertSlot<'module>, Box<dyn Error>> {
                    let expert = v4_expert::RoutedExpert::load_layer_for(
                        spec,
                        &runtime.expert_shard,
                        weights.layer,
                        expert_id,
                    )?;
                    stage.routed_read_ns = stage
                        .routed_read_ns
                        .saturating_add((expert.read_ms * 1_000_000.0) as u64);
                    stage.routed_alloc_ns = stage
                        .routed_alloc_ns
                        .saturating_add((expert.alloc_ms * 1_000_000.0) as u64);
                    stage.routed_source_bytes =
                        stage.routed_source_bytes.saturating_add(expert.h2d_bytes());
                    stage.routed_expert_loads = stage.routed_expert_loads.saturating_add(1);
                    let setup_started = std::time::Instant::now();
                    let slot = ExpertSlot::new(context, module, stream, &expert)?;
                    stage.routed_setup_ns = stage.routed_setup_ns.saturating_add(
                        u64::try_from(setup_started.elapsed().as_nanos()).unwrap_or(u64::MAX),
                    );
                    stage.routed_h2d_ns = stage
                        .routed_h2d_ns
                        .saturating_add((slot.weight_h2d_ms * 1_000_000.0) as u64);
                    stage.routed_h2d_bytes =
                        stage.routed_h2d_bytes.saturating_add(expert.h2d_bytes());
                    Ok(slot)
                };
                let slot = if let Some(expert_slots) = expert_slots.as_deref_mut() {
                    if expert_slots.contains_key(&expert_id) {
                        expert_slots
                            .get_mut(&expert_id)
                            .ok_or("retained expert slot disappeared")?
                    } else if should_retain_layer_expert(expert_slots.len()) {
                        expert_slots.insert(expert_id, load_slot()?);
                        expert_slots
                            .get_mut(&expert_id)
                            .ok_or("expert slot disappeared after insertion")?
                    } else {
                        temporary_slot.insert(load_slot()?)
                    }
                } else {
                    temporary_slot.insert(load_slot()?)
                };
                drop(load_slot);
                for row in 0..rows {
                    for route_slot in 0..spec.experts_per_token {
                        let route = row * spec.experts_per_token + route_slot;
                        if routes.ids[route] == i32::from(expert_id) {
                            let input = &ffn_input[row * spec.hidden..(row + 1) * spec.hidden];
                            contributions[row][route_slot] = Some({
                                let (output, kernel_ns, trace) = if parity_dump.is_some() {
                                    let (output, kernel_ns, trace) = slot
                                        .execute_with_flash_trace(input, routes.weights[route])?;
                                    (output, kernel_ns, Some(trace))
                                } else {
                                    let (output, kernel_ns) =
                                        slot.execute(input, routes.weights[route])?;
                                    (output, kernel_ns, None)
                                };
                                stage.routed_kernel_ns =
                                    stage.routed_kernel_ns.saturating_add(kernel_ns);
                                if let Some(traces) = flash_traces.as_mut() {
                                    if row != 0 {
                                        return Err(
                                            "Flash parity dump requires exactly one row".into()
                                        );
                                    }
                                    traces[route_slot] = trace;
                                }
                                output
                            });
                        }
                    }
                }
            }
        }
        let mut moe = Vec::with_capacity(rows * spec.hidden);
        let mut routed_sum = parity_dump
            .as_ref()
            .map(|_| Vec::with_capacity(rows * spec.hidden));
        for row in 0..rows {
            let routed = (0..spec.experts_per_token)
                .map(|slot| {
                    Ok((
                        u16::try_from(routes.ids[row * spec.experts_per_token + slot])?,
                        contributions[row][slot]
                            .as_deref()
                            .ok_or("missing routed expert contribution")?,
                    ))
                })
                .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
            if let Some(routed_sum) = routed_sum.as_mut() {
                let (row_moe, row_routed_sum) =
                    v4_expert::reduce_moe_canonical_with_routed_sum_for(
                        spec,
                        &routed,
                        &shared[row * spec.hidden..(row + 1) * spec.hidden],
                    )?;
                moe.extend(row_moe);
                routed_sum.extend(row_routed_sum);
            } else {
                moe.extend(v4_expert::reduce_moe_canonical_for(
                    spec,
                    &routed,
                    &shared[row * spec.hidden..(row + 1) * spec.hidden],
                )?);
            }
        }
        if let Some(dir) = parity_dump.as_deref() {
            let traces = flash_traces
                .ok_or("Flash parity expert traces were not initialized")?
                .into_iter()
                .collect::<Option<Vec<_>>>()
                .ok_or("Flash parity route slot trace is incomplete")?;
            let mut input_fp8 = Vec::with_capacity(spec.experts_per_token * spec.hidden);
            let mut input_scale =
                Vec::with_capacity(spec.experts_per_token * spec.hidden / v4_state::FP8_BLOCK);
            let mut w1 = Vec::with_capacity(spec.experts_per_token * spec.moe_intermediate);
            let mut w3 = Vec::with_capacity(spec.experts_per_token * spec.moe_intermediate);
            let mut swiglu = Vec::with_capacity(spec.experts_per_token * spec.moe_intermediate);
            let mut w2_input_fp8 =
                Vec::with_capacity(spec.experts_per_token * spec.moe_intermediate);
            let mut w2_input_scale = Vec::with_capacity(
                spec.experts_per_token * spec.moe_intermediate / v4_state::FP8_BLOCK,
            );
            let mut expert_out = Vec::with_capacity(spec.experts_per_token * spec.hidden);
            for trace in traces {
                input_fp8.extend(trace.input_fp8);
                input_scale.extend(trace.input_scale);
                w1.extend(trace.w1);
                w3.extend(trace.w3);
                swiglu.extend(trace.swiglu);
                w2_input_fp8.extend(trace.w2_input_fp8);
                w2_input_scale.extend(trace.w2_input_scale);
                expert_out.extend(trace.output);
            }
            flash_parity_dump_u8(
                dir,
                "expert_input_fp8_e4m3",
                &[rows, spec.experts_per_token, spec.hidden],
                &input_fp8,
            )?;
            flash_parity_dump_u8(
                dir,
                "expert_input_scale_e8m0",
                &[
                    rows,
                    spec.experts_per_token,
                    spec.hidden / v4_state::FP8_BLOCK,
                ],
                &input_scale,
            )?;
            flash_parity_dump_u16(
                dir,
                "routed_w1_bf16",
                &[rows, spec.experts_per_token, spec.moe_intermediate],
                &w1,
            )?;
            flash_parity_dump_u16(
                dir,
                "routed_w3_bf16",
                &[rows, spec.experts_per_token, spec.moe_intermediate],
                &w3,
            )?;
            flash_parity_dump_u16(
                dir,
                "swiglu_bf16",
                &[rows, spec.experts_per_token, spec.moe_intermediate],
                &swiglu,
            )?;
            flash_parity_dump_u8(
                dir,
                "w2_input_fp8_e4m3",
                &[rows, spec.experts_per_token, spec.moe_intermediate],
                &w2_input_fp8,
            )?;
            flash_parity_dump_u8(
                dir,
                "w2_input_scale_e8m0",
                &[
                    rows,
                    spec.experts_per_token,
                    spec.moe_intermediate / v4_state::FP8_BLOCK,
                ],
                &w2_input_scale,
            )?;
            flash_parity_dump_u16(
                dir,
                "routed_expert_out_bf16",
                &[rows, spec.experts_per_token, spec.hidden],
                &expert_out,
            )?;
            flash_parity_dump_f32(
                dir,
                "routed_sum_f32",
                &[rows, spec.hidden],
                routed_sum
                    .as_deref()
                    .ok_or("Flash parity routed sum disappeared")?,
            )?;
        }
        let audit = if capture_routed_contributions {
            Some(
                (0..spec.experts_per_token)
                    .map(|slot| {
                        contributions[0][slot]
                            .take()
                            .ok_or("routed contribution disappeared before audit")
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            )
        } else {
            None
        };
        (moe, audit)
    };
    if let Some(dir) = parity_dump.as_deref() {
        flash_parity_dump_u16(dir, "moe_out_bf16", &[rows, spec.hidden], &moe)?;
    }
    if let Some(timeline) = critical_path_timeline.as_mut() {
        timeline.routed_end_ns = timeline.now();
        timeline.hc_post_start_ns = timeline.now();
    }
    let output = run_l0_hc_post(
        module,
        stream,
        &moe,
        &after_attention,
        &ffn_hc.post,
        &ffn_hc.comb,
    )?;
    if let Some(dir) = parity_dump.as_deref() {
        flash_parity_dump_u16(
            dir,
            "block_output_hidden_bf16",
            &[rows, v4_state::HC, spec.hidden],
            &output,
        )?;
    }
    maybe_write_v4_audit_u16(weights.layer, "hidden", &output)?;
    if let Some((state, scratch, prepared, audit_expected_compressed_prefix)) = generic_prepared {
        stage.routed_ns = u64::try_from(routed_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        let commit_started = std::time::Instant::now();
        let committed = commit_generic_device_attention_row(state, scratch, prepared, stream)?;
        if let Some(expected_prefix) = audit_expected_compressed_prefix {
            let actual = state.compressed_kv.to_host_vec(stream)?;
            let actual_prefix = actual
                .get(..expected_prefix.len())
                .ok_or("generic post-commit boundary audit device prefix is incomplete")?;
            check_compressed_boundary_bf16(
                "post_commit_compressed_prefix",
                weights.layer,
                start_pos,
                0,
                v4_state::Q_HEAD_DIM,
                expected_prefix,
                actual_prefix,
            )?;
        }
        stage.commit_ns = u64::try_from(commit_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        if committed.output != after_attention {
            return Err("generic stationary attention commit output drifted".into());
        }
    } else {
        stage.routed_ns = u64::try_from(routed_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
    }
    if let Some(timeline) = critical_path_timeline.as_mut() {
        timeline.hc_post_end_ns = timeline.now();
        timeline.output_ready_ns = timeline.now();
    }
    let routed_audit = routed_contributions.map(|contributions| RoutedExecutionAudit {
        after_attention_digest: fnv1a_u16(&after_attention),
        ffn_input_digest: fnv1a_u16(&ffn_input),
        shared_digest: fnv1a_f32(&shared),
        moe_digest: fnv1a_u16(&moe),
        contributions,
    });
    Ok(RealLayerRun {
        output,
        attention_output,
        routes,
        attention_topk,
        attention_topk_digest,
        attention_topk_per_row,
        device_transfer,
        indexer_topk_gpu_ns,
        stage,
        critical_path_timeline,
        routed_audit,
    })
}
