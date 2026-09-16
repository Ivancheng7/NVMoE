use super::*;

#[allow(clippy::too_many_arguments)]
pub(crate) fn commit_layer_attention_state(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    weights: &LayerDeviceWeights,
    state: &mut v4_state::V4LayerPersistentState,
    kv: &[u16],
    compressor: Option<&v4_state::CompressorPrepared>,
    indexer: Option<&CsaIndexerPrepared>,
    start_pos: usize,
) -> Result<(v4_state::SparseTopk, u64), Box<dyn Error>> {
    let rows = kv.len() / v4_state::Q_HEAD_DIM;
    if weights.compression_ratio == v4_state::CompressionRatio::WindowOnly {
        if rows == 0
            || kv.len() != rows * v4_state::Q_HEAD_DIM
            || indexer.is_some()
            || compressor.is_some()
        {
            return Err(format!(
                "layer {} window-only attention state input contract mismatch",
                weights.layer
            )
            .into());
        }
        for row in 0..rows {
            state.append_kv_row(
                start_pos + row,
                &kv[row * v4_state::Q_HEAD_DIM..(row + 1) * v4_state::Q_HEAD_DIM],
            )?;
        }
        let (indices, per_row) = v4_state::window_topk_cpu(rows, start_pos, v4_state::KV_WINDOW)?;
        return Ok((
            v4_state::SparseTopk {
                indices,
                rows,
                per_row,
                window_per_row: per_row,
                compressed_per_row: 0,
            },
            0,
        ));
    }
    let compressor = compressor.ok_or("compressed attention omitted compressor projections")?;
    let compressor_width = if weights.compression_ratio == v4_state::CompressionRatio::Csa4 {
        1_024
    } else {
        v4_state::Q_HEAD_DIM
    };
    if rows == 0
        || kv.len() != rows * v4_state::Q_HEAD_DIM
        || compressor.kv_f32.len() != rows * compressor_width
        || compressor.score_f32.len() != rows * compressor_width
        || compressor.score_plus_ape_f32.len() != rows * compressor_width
        || (weights.compression_ratio == v4_state::CompressionRatio::Csa4) != indexer.is_some()
    {
        return Err(format!(
            "layer {} attention state input contract mismatch",
            weights.layer
        )
        .into());
    }
    let ratio = weights.compression_ratio.get();
    for row in 0..rows {
        let absolute = start_pos + row;
        state.append_kv_row(
            absolute,
            &kv[row * v4_state::Q_HEAD_DIM..(row + 1) * v4_state::Q_HEAD_DIM],
        )?;
        let main_range = row * compressor_width..(row + 1) * compressor_width;
        let main_pool = state.commit_compressor_row_and_pool(
            absolute,
            &v4_state::CompressorPrepared {
                kv_f32: compressor.kv_f32[main_range.clone()].to_vec(),
                score_f32: compressor.score_f32[main_range.clone()].to_vec(),
                score_plus_ape_f32: compressor.score_plus_ape_f32[main_range].to_vec(),
            },
        )?;
        let index_pool = if let Some(indexer) = indexer {
            let range = row * 256..(row + 1) * 256;
            state.commit_indexer_row_and_pool(
                absolute,
                &v4_state::CompressorPrepared {
                    kv_f32: indexer.compressor.kv_f32[range.clone()].to_vec(),
                    score_f32: indexer.compressor.score_f32[range.clone()].to_vec(),
                    score_plus_ape_f32: indexer.compressor.score_plus_ape_f32[range].to_vec(),
                },
            )?
        } else {
            None
        };
        match (main_pool, index_pool) {
            (None, None) => {}
            (Some(main), None)
                if weights.compression_ratio == v4_state::CompressionRatio::Hca128 =>
            {
                let main = main
                    .into_iter()
                    .map(v4_router::f32_to_bf16_rne)
                    .collect::<Vec<_>>();
                let main = run_l0_rmsnorm(
                    module,
                    stream,
                    &main,
                    &weights.compressor_norm,
                    v4_state::Q_HEAD_DIM,
                )?;
                let position = absolute + 1 - ratio;
                let main = run_l0_rope(
                    module,
                    stream,
                    &main,
                    &v4_state::compressed_rope_freqs(position),
                    1,
                    false,
                )?;
                let main = run_l0_kv_quantize(module, stream, &main)?;
                state.append_compressed_kv_row(absolute / ratio, &main)?;
            }
            (Some(main), Some(index))
                if weights.compression_ratio == v4_state::CompressionRatio::Csa4 =>
            {
                let indexer_weights = weights
                    .indexer
                    .as_ref()
                    .ok_or("CSA indexer device weights disappeared")?;
                let position = absolute + 1 - ratio;
                let freqs = v4_state::compressed_rope_freqs(position);
                let main = main
                    .into_iter()
                    .map(v4_router::f32_to_bf16_rne)
                    .collect::<Vec<_>>();
                let main = run_l0_rmsnorm(
                    module,
                    stream,
                    &main,
                    &weights.compressor_norm,
                    v4_state::Q_HEAD_DIM,
                )?;
                let main = run_l0_rope(module, stream, &main, &freqs, 1, false)?;
                let main = run_l0_kv_quantize(module, stream, &main)?;
                state.append_compressed_kv_row(absolute / ratio, &main)?;

                let index = index
                    .into_iter()
                    .map(v4_router::f32_to_bf16_rne)
                    .collect::<Vec<_>>();
                let index = run_l0_rmsnorm(
                    module,
                    stream,
                    &index,
                    &indexer_weights.compressor_norm,
                    128,
                )?;
                let index = run_rope(module, stream, &index, &freqs, 1, 128, false)?;
                let index = v4_state::rotate_fp4_sim_bf16(&index, 1, 128)?
                    .into_iter()
                    .map(v4_router::f32_to_bf16_rne)
                    .collect::<Vec<_>>();
                state.append_indexer_kv_row(absolute / ratio, &index)?;
            }
            _ => {
                return Err(format!(
                    "layer {} compressor/indexer boundary commit diverged",
                    weights.layer
                )
                .into());
            }
        }
    }

    if weights.compression_ratio == v4_state::CompressionRatio::Hca128 {
        return Ok((
            v4_state::build_hca_sparse_topk(rows, start_pos, v4_state::KV_WINDOW, ratio)?,
            0,
        ));
    }
    let (window, window_per_row) = v4_state::window_topk_cpu(rows, start_pos, v4_state::KV_WINDOW)?;
    let keys = state.indexer_kv_bf16().len() / 128;
    if keys == 0 {
        return Ok((
            v4_state::SparseTopk {
                indices: window,
                rows,
                per_row: window_per_row,
                window_per_row,
                compressed_per_row: 0,
            },
            0,
        ));
    }
    let indexer = indexer.ok_or("CSA indexer projection result disappeared")?;
    let valid_keys = (0..rows)
        .map(|row| (start_pos + row + 1) / ratio)
        .collect::<Vec<_>>();
    let (compressed, indexer_topk_gpu_ns) = run_indexer_topk_gpu(
        weights.spec,
        module,
        stream,
        &indexer.q_f32,
        &indexer.weights_f32,
        state.indexer_kv_bf16(),
        rows,
        &valid_keys,
        v4_state::KV_WINDOW,
    )?;
    let compressed_per_row = compressed.len() / rows;
    let mut indices = Vec::with_capacity(rows * (window_per_row + compressed_per_row));
    for row in 0..rows {
        indices.extend_from_slice(&window[row * window_per_row..(row + 1) * window_per_row]);
        indices.extend_from_slice(
            &compressed[row * compressed_per_row..(row + 1) * compressed_per_row],
        );
    }
    Ok((
        v4_state::SparseTopk {
            indices,
            rows,
            per_row: window_per_row + compressed_per_row,
            window_per_row,
            compressed_per_row,
        },
        indexer_topk_gpu_ns,
    ))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_targeted_attention_audit(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    runtime: &LoadedLayerRuntime,
    hc_hidden: &[u16],
    position: usize,
    host_state: &v4_state::V4LayerPersistentState,
    device_state: &DeviceTokenState,
    expected_topk: &[i32],
    precommit_host: &PersistentStateAudit,
    precommit_device: &PersistentStateAudit,
) -> Result<bool, Box<dyn Error>> {
    if runtime.weights.layer != 2
        || position != 3
        || host_state.committed_kv_rows() != 4
        || device_state.committed_rows != 4
        || expected_topk.len() != v4_state::KV_WINDOW + 1
    {
        return Err("targeted attention audit reached the wrong layer/position geometry".into());
    }
    let weights = &runtime.weights;
    let mut stages = vec![
        targeted_stage_u16("precommit.kv", &precommit_host.kv, &precommit_device.kv),
        targeted_stage_u16(
            "precommit.compressed_kv",
            &precommit_host.compressed_kv,
            &precommit_device.compressed_kv,
        ),
        targeted_stage_u16(
            "precommit.indexer_kv",
            &precommit_host.indexer_kv,
            &precommit_device.indexer_kv,
        ),
        targeted_stage_f32(
            "precommit.main_compressor_kv",
            &precommit_host.main_kv,
            &precommit_device.main_kv,
        ),
        targeted_stage_f32(
            "precommit.main_compressor_score",
            &precommit_host.main_score,
            &precommit_device.main_score,
        ),
    ];
    if let (Some(expected), Some(actual)) = (
        precommit_host.indexer_main_kv.as_deref(),
        precommit_device.indexer_main_kv.as_deref(),
    ) {
        stages.push(targeted_stage_f32(
            "precommit.indexer_compressor_kv",
            expected,
            actual,
        ));
    }
    if let (Some(expected), Some(actual)) = (
        precommit_host.indexer_main_score.as_deref(),
        precommit_device.indexer_main_score.as_deref(),
    ) {
        stages.push(targeted_stage_f32(
            "precommit.indexer_compressor_score",
            expected,
            actual,
        ));
    }

    let freqs = v4_state::compressed_rope_freqs(position);
    let attn_hc = run_l0_mhc_pre(
        module,
        stream,
        hc_hidden,
        &weights.hc_attn_fn,
        &weights.hc_attn_scale,
        &weights.hc_attn_base,
    )?;
    let actual_hc_hidden = device_state.hc_hidden.to_host_vec(stream)?;
    let actual_post = device_state.post.to_host_vec(stream)?;
    let actual_comb = device_state.comb.to_host_vec(stream)?;
    stages.extend([
        targeted_stage_u16("input.hc_hidden", hc_hidden, &actual_hc_hidden),
        targeted_stage_f32("input.hc_post", &attn_hc.post, &actual_post),
        targeted_stage_f32("input.hc_comb", &attn_hc.comb, &actual_comb),
    ]);
    let attn_input = run_l0_rmsnorm(
        module,
        stream,
        &attn_hc.collapsed,
        &weights.attn_norm,
        v4_state::HIDDEN,
    )?;
    let (_, q_lora_raw) = run_l0_fp8_linear(module, stream, &attn_input, &weights.wq_a, false)?;
    let q_lora = run_l0_rmsnorm(
        module,
        stream,
        &q_lora_raw,
        &weights.q_norm,
        weights.wq_a.out_features,
    )?;
    let (_, q_raw) = run_l0_fp8_linear(module, stream, &q_lora, &weights.wq_b, true)?;
    let unit_norm = DeviceBuffer::<u16>::zeroed(stream, 0)?;
    let q_norm = run_l0_rmsnorm(module, stream, &q_raw, &unit_norm, v4_state::Q_HEAD_DIM)?;
    let q_rope = run_l0_rope(module, stream, &q_norm, &freqs, v4_state::Q_HEADS, false)?;
    stages.extend([
        targeted_stage_u16(
            "input.attn_input",
            &attn_input,
            &device_state.attn_input.to_host_vec(stream)?,
        ),
        targeted_stage_u16(
            "input.q_rope",
            &q_rope,
            &device_state.q_rope.to_host_vec(stream)?,
        ),
    ]);

    let cache = host_state.attention_cache_bf16();
    let actual_cache = device_state.attention_cache.to_host_vec(stream)?;
    stages.push(targeted_stage_u16(
        "input.attention_cache",
        &cache,
        &actual_cache[..cache.len()],
    ));
    let actual_topk = device_state.topk_csa[0].to_host_vec(stream)?;
    stages.push(targeted_stage_i32(
        "input.topk",
        expected_topk,
        &actual_topk,
    ));

    let topk = v4_state::SparseTopk {
        indices: expected_topk.to_vec(),
        rows: 1,
        per_row: expected_topk.len(),
        window_per_row: v4_state::KV_WINDOW,
        compressed_per_row: 1,
    };
    let reference_sparse =
        run_sparse_attention(module, stream, &q_rope, &cache, &weights.attn_sink, &topk)?;
    let q_device = DeviceBuffer::from_host(stream, &q_rope)?;
    let cache_device = DeviceBuffer::from_host(stream, &cache)?;
    let topk_device = DeviceBuffer::from_host(stream, expected_topk)?;
    let mut reference_state =
        SparseAttentionScoreReuseScratch::new(stream, 1, expected_topk.len())?;
    run_sparse_attention_reference_state_device_into(
        module,
        stream,
        &q_device,
        &cache_device,
        &weights.attn_sink,
        &topk_device,
        &mut reference_state,
        expected_topk.len(),
    )?;
    let logical_scores = v4_state::Q_HEADS * expected_topk.len();
    let candidate_state = &device_state.sparse_csa;
    let reference_scores = reference_state.scores.to_host_vec(stream)?;
    let candidate_scores = candidate_state.scores.to_host_vec(stream)?;
    let reference_probabilities = reference_state.probabilities.to_host_vec(stream)?;
    let candidate_probabilities = candidate_state.probabilities.to_host_vec(stream)?;
    stages.extend([
        targeted_stage_f32(
            "scores",
            &reference_scores[..logical_scores],
            &candidate_scores[..logical_scores],
        ),
        targeted_stage_f32(
            "probabilities",
            &reference_probabilities[..logical_scores],
            &candidate_probabilities[..logical_scores],
        ),
        targeted_stage_f32(
            "denominators",
            &reference_state.denominators.to_host_vec(stream)?,
            &candidate_state.denominators.to_host_vec(stream)?,
        ),
        targeted_stage_u32(
            "valid_counts",
            &reference_state.valid_counts.to_host_vec(stream)?,
            &candidate_state.valid_counts.to_host_vec(stream)?,
        ),
        targeted_stage_u16(
            "sparse",
            &reference_sparse,
            &device_state.sparse.to_host_vec(stream)?,
        ),
    ]);

    let sparse_inverse = run_l0_rope(
        module,
        stream,
        &reference_sparse,
        &freqs,
        v4_state::Q_HEADS,
        true,
    )?;
    let wo_a = run_l0_wo_a(module, stream, &sparse_inverse, &weights.wo_a)?;
    let (attention_f32, attention) =
        run_l0_fp8_linear(module, stream, &wo_a, &weights.wo_b, false)?;
    let attention_post = run_l0_hc_post(
        module,
        stream,
        &attention,
        hc_hidden,
        &attn_hc.post,
        &attn_hc.comb,
    )?;
    stages.extend([
        targeted_stage_u16(
            "sparse_inverse",
            &sparse_inverse,
            &device_state.sparse_inverse.to_host_vec(stream)?,
        ),
        targeted_stage_u16("wo_a_bf16", &wo_a, &device_state.wo_a.to_host_vec(stream)?),
        serde_json::json!({
            "stage": "wo_b_weights",
            "exact": true,
            "basis": "reference and candidate share the same loaded DeviceFp8Matrix",
            "bytes": weights.wo_b.num_bytes(),
        }),
        targeted_stage_f32(
            "attention_f32",
            &attention_f32,
            &device_state.attention_f32.to_host_vec(stream)?,
        ),
        targeted_stage_u16(
            "attention_bf16",
            &attention,
            &device_state.attention.to_host_vec(stream)?,
        ),
        targeted_stage_u16(
            "attention_post",
            &attention_post,
            &device_state.after_attention.to_host_vec(stream)?,
        ),
    ]);
    let first_divergence = stages
        .iter()
        .find(|stage| stage.get("exact").and_then(serde_json::Value::as_bool) == Some(false))
        .and_then(|stage| stage.get("stage"))
        .cloned();
    let exact = first_divergence.is_none();
    let cublas = cublas_exact::CublasExact::load()?;
    let cublas_provenance = cublas.provenance();
    let artifact = serde_json::json!({
        "schema": "colibri-v4-targeted-attention-audit-v1",
        "status": if exact { "PASS" } else { "FAIL" },
        "provenance": benchmark_provenance()?,
        "model_revision": v4_state::REVISION,
        "configuration": {
            "chat_user": "hi",
            "executed_layers": 3,
            "target": {"layer": 2, "position": 3},
            "device_attention_audit": true,
            "sparse_attention_arm": "sm89_crossover",
            "logical_topk": expected_topk.len(),
            "score_scratch_capacity_topk": candidate_state.scores.len() / v4_state::Q_HEADS,
            "reference": "scalar sparse_attn_v4_bf16 plus reference-state probe",
            "candidate": "sparse_attn_v4_scores+probabilities+values",
            "same_process_same_precommit_state": true,
            "cublas": {
                "sha256": cublas_provenance.actual_sha256,
                "version": cublas_provenance.version,
            },
        },
        "first_divergence": first_divergence,
        "stages": stages,
        "process": {
            "exit_status": 0,
            "peak_host_rss_bytes": process_peak_rss_bytes()?,
        },
    });
    let path = std::path::PathBuf::from(
        std::env::var_os("COLIBRI_V4_TARGETED_ATTENTION_AUDIT_PATH")
            .ok_or("targeted attention audit artifact path disappeared")?,
    );
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut output = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)?;
    serde_json::to_writer_pretty(&mut output, &artifact)?;
    writeln!(output)?;
    output.sync_all()?;
    println!(
        "{{\"status\":\"{}\",\"mode\":\"v4-targeted-attention-audit\",\"layer\":2,\"position\":3,\"logical_topk\":{},\"first_divergence\":{},\"artifact\":{}}}",
        if exact { "PASS" } else { "FAIL" },
        expected_topk.len(),
        serde_json::to_string(&first_divergence)?,
        serde_json::to_string(&path)?,
    );
    Ok(exact)
}

pub(crate) fn run_sparse_attention_topk_sweep(
    context: &std::sync::Arc<CudaContext>,
    module: &kernels::LoadedModule,
    stream: &std::sync::Arc<CudaStream>,
) -> Result<(), Box<dyn Error>> {
    const TOPKS: [usize; 8] = [1, 2, 4, 8, 16, 32, 64, 129];
    const PAIRS: usize = 15;
    let result_path = std::env::var_os("COLIBRI_V4_SPARSE_TOPK_SWEEP_RESULT_JSONL")
        .map(std::path::PathBuf::from)
        .ok_or("set COLIBRI_V4_SPARSE_TOPK_SWEEP_RESULT_JSONL")?;
    if let Some(parent) = result_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let q_host = (0..v4_state::Q_HEADS * v4_state::Q_HEAD_DIM)
        .map(|index| v4_router::f32_to_bf16_rne(((index % 29) as f32 - 14.0) * 0.007_812_5))
        .collect::<Vec<_>>();
    let kv_host = (0..129 * v4_state::Q_HEAD_DIM)
        .map(|index| v4_router::f32_to_bf16_rne(((index % 31) as f32 - 15.0) * 0.003_906_25))
        .collect::<Vec<_>>();
    let sink_host = (0..v4_state::Q_HEADS)
        .map(|head| -1.0_f32 - (head % 7) as f32 * 0.03125)
        .collect::<Vec<_>>();
    let q = DeviceBuffer::from_host(stream, &q_host)?;
    let kv = DeviceBuffer::from_host(stream, &kv_host)?;
    let sink = DeviceBuffer::from_host(stream, &sink_host)?;
    let start = context.new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
    let end = context.new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
    let mut output = std::fs::File::create(&result_path)?;
    writeln!(
        output,
        "{}",
        serde_json::json!({
            "event": "run",
            "schema": "colibri-v4-sparse-score-reuse-topk-sweep-v1",
            "provenance": benchmark_provenance()?,
            "gpu_timing": "CUDA events; output/raw-state downloads occur after each timed pair",
            "geometries": ["L3_HCA: q_heads=128,q_head_dim=512", "L4_CSA: q_heads=128,q_head_dim=512"],
            "topks": TOPKS,
            "pairs": PAIRS,
            "arm_order_schedule": "AB/BA alternating; A=scalar sparse_attn_v4_bf16; B=sparse_attn_v4_scores+probabilities+values",
            "raw_exactness": "BF16 output plus raw f32 scores/probabilities/denominators and u32 valid_counts"
        })
    )?;

    for geometry in ["L3_HCA", "L4_CSA"] {
        for topk_per_row in TOPKS {
            let topk_host = (0..topk_per_row)
                .map(i32::try_from)
                .collect::<Result<Vec<_>, _>>()?;
            let topk = DeviceBuffer::from_host(stream, &topk_host)?;
            let mut reference = DeviceBuffer::<u16>::zeroed(stream, q_host.len())?;
            let mut candidate = DeviceBuffer::<u16>::zeroed(stream, q_host.len())?;
            let mut reference_state =
                SparseAttentionScoreReuseScratch::new(stream, 1, topk_per_row)?;
            let mut candidate_state =
                SparseAttentionScoreReuseScratch::new(stream, 1, topk_per_row)?;
            for _ in 0..2 {
                run_sparse_attention_device_into(
                    module,
                    stream,
                    &q,
                    &kv,
                    &sink,
                    &topk,
                    &mut reference,
                    topk_per_row,
                )?;
                run_sparse_attention_score_reuse_device_into(
                    module,
                    stream,
                    &q,
                    &kv,
                    &sink,
                    &topk,
                    &mut candidate,
                    &mut candidate_state,
                    topk_per_row,
                )?;
            }
            end.record(stream)?;
            end.synchronize()?;
            let mut reference_samples = Vec::with_capacity(PAIRS);
            let mut candidate_samples = Vec::with_capacity(PAIRS);
            for pair in 0..PAIRS {
                let reference_first = pair.is_multiple_of(2);
                let mut reference_ns = 0_u64;
                let mut candidate_ns = 0_u64;
                for arm in if reference_first {
                    [
                        SparseAttentionArm::Reference,
                        SparseAttentionArm::ScoreReuse,
                    ]
                } else {
                    [
                        SparseAttentionArm::ScoreReuse,
                        SparseAttentionArm::Reference,
                    ]
                } {
                    start.record(stream)?;
                    match arm {
                        SparseAttentionArm::Reference => run_sparse_attention_device_into(
                            module,
                            stream,
                            &q,
                            &kv,
                            &sink,
                            &topk,
                            &mut reference,
                            topk_per_row,
                        )?,
                        SparseAttentionArm::ScoreReuse | SparseAttentionArm::Sm89Crossover => {
                            run_sparse_attention_score_reuse_device_into(
                                module,
                                stream,
                                &q,
                                &kv,
                                &sink,
                                &topk,
                                &mut candidate,
                                &mut candidate_state,
                                topk_per_row,
                            )?
                        }
                    }
                    end.record(stream)?;
                    end.synchronize()?;
                    let elapsed = (start.elapsed_ms(&end)? as f64 * 1_000_000.0) as u64;
                    match arm {
                        SparseAttentionArm::Reference => reference_ns = elapsed,
                        SparseAttentionArm::ScoreReuse | SparseAttentionArm::Sm89Crossover => {
                            candidate_ns = elapsed
                        }
                    }
                }
                run_sparse_attention_reference_state_device_into(
                    module,
                    stream,
                    &q,
                    &kv,
                    &sink,
                    &topk,
                    &mut reference_state,
                    topk_per_row,
                )?;
                let reference_output = reference.to_host_vec(stream)?;
                let candidate_output = candidate.to_host_vec(stream)?;
                if reference_output != candidate_output {
                    return Err(format!(
                        "{geometry} topk={topk_per_row} pair={pair}: BF16 output mismatch"
                    )
                    .into());
                }
                let reference_scores = reference_state.scores.to_host_vec(stream)?;
                let candidate_scores = candidate_state.scores.to_host_vec(stream)?;
                let reference_probabilities = reference_state.probabilities.to_host_vec(stream)?;
                let candidate_probabilities = candidate_state.probabilities.to_host_vec(stream)?;
                let reference_denominators = reference_state.denominators.to_host_vec(stream)?;
                let candidate_denominators = candidate_state.denominators.to_host_vec(stream)?;
                let reference_valid_counts = reference_state.valid_counts.to_host_vec(stream)?;
                let candidate_valid_counts = candidate_state.valid_counts.to_host_vec(stream)?;
                let raw_exact = reference_scores
                    .iter()
                    .zip(&candidate_scores)
                    .chain(reference_probabilities.iter().zip(&candidate_probabilities))
                    .chain(reference_denominators.iter().zip(&candidate_denominators))
                    .all(|(left, right)| left.to_bits() == right.to_bits())
                    && reference_valid_counts == candidate_valid_counts;
                if !raw_exact {
                    return Err(format!(
                        "{geometry} topk={topk_per_row} pair={pair}: raw state mismatch"
                    )
                    .into());
                }
                reference_samples.push(reference_ns);
                candidate_samples.push(candidate_ns);
                writeln!(
                    output,
                    "{}",
                    serde_json::json!({
                        "event": "pair",
                        "geometry": geometry,
                        "topk": topk_per_row,
                        "pair": pair,
                        "arm_order": if reference_first { "AB" } else { "BA" },
                        "reference_cuda_event_ns": reference_ns,
                        "candidate_cuda_event_ns": candidate_ns,
                        "speedup": reference_ns as f64 / candidate_ns as f64,
                        "exact": true,
                        "output_bf16_digest": format!("{:016x}", fnv1a_u16(&candidate_output)),
                        "scores_f32_digest": format!("{:016x}", fnv1a_f32(&candidate_scores)),
                        "probabilities_f32_digest": format!("{:016x}", fnv1a_f32(&candidate_probabilities)),
                        "denominators_f32_digest": format!("{:016x}", fnv1a_f32(&candidate_denominators)),
                        "valid_counts_digest": format!("{:016x}", fnv1a_i32(&candidate_valid_counts.iter().map(|&value| value as i32).collect::<Vec<_>>())),
                    })
                )?;
            }
            let (reference_p50_ms, reference_p95_ms) = median_p95_ms(&reference_samples);
            let (candidate_p50_ms, candidate_p95_ms) = median_p95_ms(&candidate_samples);
            writeln!(
                output,
                "{}",
                serde_json::json!({
                    "event": "summary",
                    "geometry": geometry,
                    "topk": topk_per_row,
                    "pairs": PAIRS,
                    "exact": true,
                    "reference_cuda_event_p50_ms": reference_p50_ms,
                    "reference_cuda_event_p95_ms": reference_p95_ms,
                    "candidate_cuda_event_p50_ms": candidate_p50_ms,
                    "candidate_cuda_event_p95_ms": candidate_p95_ms,
                    "speedup_p50": reference_p50_ms / candidate_p50_ms,
                    "speedup_p95": reference_p95_ms / candidate_p95_ms,
                    "candidate_p95_wins": candidate_p95_ms < reference_p95_ms,
                })
            )?;
        }
    }
    println!(
        "{{\"status\":\"PASS\",\"mode\":\"v4-sparse-topk-sweep\",\"result_jsonl\":{}}}",
        serde_json::to_string(&result_path)?
    );
    Ok(())
}

pub(crate) fn run_real_v4_wavefront_attention_bench(
    context: &std::sync::Arc<CudaContext>,
    module: &kernels::LoadedModule,
    stream: &std::sync::Arc<CudaStream>,
) -> Result<(), Box<dyn Error>> {
    const PAIRS: usize = 30;
    const REQUIRED_P50_SPEEDUP: f64 = 1.0 / 0.95;
    let result_path = std::env::var_os("COLIBRI_V4_WAVEFRONT_ATTENTION_RESULT_JSONL")
        .map(std::path::PathBuf::from)
        .ok_or("set COLIBRI_V4_WAVEFRONT_ATTENTION_RESULT_JSONL")?;
    if let Some(parent) = result_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let model_dir = v4_model_dir()?;
    let token_ids = [0_i64, 1_i64];
    let mut lanes = (0..V4_QD4_LANES)
        .map(|_| {
            let lane_stream = context.new_stream()?;
            ExpertSlot::new_empty(context, module, &lane_stream)
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    let inputs =
        prepare_two_token_prefix(context, module, stream, &model_dir, token_ids, &mut lanes)?;
    let sparse_arm = sparse_attention_arm_from_env()?;
    let rope_lut = DeviceRequestRopeLut::new(stream, V4_REQUEST_ROPE_LUT_ROWS)?;
    let telemetry_before = gpu_telemetry_snapshot()?;
    let mut output = std::fs::File::create(&result_path)?;
    writeln!(
        output,
        "{}",
        serde_json::json!({
            "event": "run",
            "schema": "colibri-v4-device-attention-gate-v1",
            "provenance": benchmark_provenance()?,
            "model_revision": v4_state::REVISION,
            "layers": [3, 4],
            "positions": [0, 1],
            "pairs": PAIRS,
            "arm_order_schedule": "AB/BA alternating; A=current host attention runner; B=existing device-resident attention chain",
            "sparse_attention_arm": format!("{sparse_arm:?}"),
            "timing": "monotonic host wall is the gate; CUDA events are diagnostics; state downloads are post-timing exactness audit only",
            "request_rope_lut": {
                "h2d_count": 1,
                "h2d_bytes": rope_lut.num_bytes(),
            },
            "gate": {
                "exact": "30/30 pairs per layer; both positions; attention_post BF16, top-k, KV, main/indexer compressor state, and all comparable stage digests",
                "p50_speedup_at_least": REQUIRED_P50_SPEEDUP,
                "p95_nonregression": true,
                "candidate_transfer_contract": "request-global RoPE LUT: 1 H2D / 2,560 B; per two positions: 2 hidden H2D, 4 LUT-row D2D, 2 terminal hidden D2H, zero temporary allocations; no intermediate D2H"
            },
            "gpu_before": telemetry_before,
        })
    )?;
    let mut overall_go = true;
    for layer in [3_usize, 4] {
        let (runtime, _) = load_layer_runtime(&model_dir, layer, &token_ids, stream)?;
        let mut host_wall_samples = Vec::with_capacity(PAIRS);
        let mut device_wall_samples = Vec::with_capacity(PAIRS);
        let mut host_cuda_event_samples = Vec::with_capacity(PAIRS);
        let mut device_cuda_event_samples = Vec::with_capacity(PAIRS);
        let mut exact_pairs = 0_usize;
        for pair in 0..PAIRS {
            let host_first = pair.is_multiple_of(2);
            let ((host, host_wall_ns, host_cuda_event_ns), (device, device_wall_ns, device_runs)) =
                if host_first {
                    let host =
                        run_attention_arm_a_timed(context, module, stream, &runtime, &inputs)?;
                    let mut state = DeviceTokenState::new(stream, &runtime.weights)?;
                    let device = run_attention_arm_b(
                        module, stream, &runtime, &inputs, &mut state, &rope_lut, sparse_arm,
                    )?;
                    (host, device)
                } else {
                    let mut state = DeviceTokenState::new(stream, &runtime.weights)?;
                    let device = run_attention_arm_b(
                        module, stream, &runtime, &inputs, &mut state, &rope_lut, sparse_arm,
                    )?;
                    let host =
                        run_attention_arm_a_timed(context, module, stream, &runtime, &inputs)?;
                    (host, device)
                };
            check_attention_audit(layer, &host, &device)?;
            exact_pairs += 1;
            let h2d_count = device_runs.iter().map(|run| run.h2d_count).sum::<usize>();
            let h2d_bytes = device_runs.iter().map(|run| run.h2d_bytes).sum::<usize>();
            let rope_lut_d2d_count = device_runs
                .iter()
                .map(|run| run.rope_lut_d2d_count)
                .sum::<usize>();
            let rope_lut_d2d_bytes = device_runs
                .iter()
                .map(|run| run.rope_lut_d2d_bytes)
                .sum::<usize>();
            let d2h_count = device_runs.iter().map(|run| run.d2h_count).sum::<usize>();
            let d2h_bytes = device_runs.iter().map(|run| run.d2h_bytes).sum::<usize>();
            let alloc_count = device_runs.iter().map(|run| run.alloc_count).sum::<usize>();
            let device_cuda_event_ns = device_runs
                .iter()
                .fold(0_u64, |sum, run| sum.saturating_add(run.cuda_event_ns));
            if h2d_count != 2
                || h2d_bytes != 2 * v4_state::HC * v4_state::HIDDEN * 2
                || rope_lut_d2d_count != 4
                || rope_lut_d2d_bytes != 4 * v4_state::Q_ROPE_DIM * std::mem::size_of::<f32>()
                || d2h_count != 2
                || d2h_bytes != 2 * v4_state::HC * v4_state::HIDDEN * 2
                || alloc_count != 0
            {
                return Err(format!(
                    "device attention layer {layer} transfer/allocation contract failed"
                )
                .into());
            }
            host_wall_samples.push(host_wall_ns);
            device_wall_samples.push(device_wall_ns);
            host_cuda_event_samples.push(host_cuda_event_ns);
            device_cuda_event_samples.push(device_cuda_event_ns);
            writeln!(
                output,
                "{}",
                serde_json::json!({
                    "event": "pair",
                    "layer": layer,
                    "pair": pair,
                    "arm_order": if host_first { "AB" } else { "BA" },
                    "exact": true,
                    "host_wall_ns": host_wall_ns,
                    "device_wall_ns": device_wall_ns,
                    "wall_speedup": host_wall_ns as f64 / device_wall_ns as f64,
                    "host_cuda_event_ns": host_cuda_event_ns,
                    "device_cuda_event_ns": device_cuda_event_ns,
                    "device_h2d_count": h2d_count,
                    "device_h2d_bytes": h2d_bytes,
                    "device_request_rope_lut_d2d_count": rope_lut_d2d_count,
                    "device_request_rope_lut_d2d_bytes": rope_lut_d2d_bytes,
                    "device_terminal_d2h_count": d2h_count,
                    "device_terminal_d2h_bytes": d2h_bytes,
                    "device_intermediate_d2h_count": 0,
                    "device_temporary_alloc_count": alloc_count,
                    "attention_post_digests": device.outputs.iter().map(|row| format!("{:016x}", fnv1a_u16(row))).collect::<Vec<_>>(),
                    "topk_digests": device.topk_digests.iter().map(|digest| format!("{digest:016x}")).collect::<Vec<_>>(),
                    "kv_digest": format!("{:016x}", fnv1a_u16(&device.kv)),
                    "main_compressor_kv_digest": format!("{:016x}", fnv1a_f32(&device.main_kv)),
                    "main_compressor_score_digest": format!("{:016x}", fnv1a_f32(&device.main_score)),
                    "indexer_compressor_kv_digest": device.indexer_kv.as_ref().map(|value| format!("{:016x}", fnv1a_f32(value))),
                    "indexer_compressor_score_digest": device.indexer_score.as_ref().map(|value| format!("{:016x}", fnv1a_f32(value))),
                })
            )?;
        }
        let host_p50 = median_u64(&host_wall_samples);
        let device_p50 = median_u64(&device_wall_samples);
        let (_, host_p95) = median_p95_ms(&host_wall_samples);
        let (_, device_p95) = median_p95_ms(&device_wall_samples);
        let p50_go = host_p50 as f64 / device_p50 as f64 >= REQUIRED_P50_SPEEDUP;
        let p95_go = device_p95 <= host_p95;
        let layer_status = if exact_pairs == PAIRS && p50_go && p95_go {
            "GO"
        } else {
            "STOP"
        };
        overall_go &= layer_status == "GO";
        writeln!(
            output,
            "{}",
            serde_json::json!({
                "event": "layer_summary",
                "layer": layer,
                "status": layer_status,
                "exact_pairs": exact_pairs,
                "host_wall_p50_ns": host_p50,
                "device_wall_p50_ns": device_p50,
                "host_wall_p95_ms": host_p95,
                "device_wall_p95_ms": device_p95,
                "wall_speedup_p50": host_p50 as f64 / device_p50 as f64,
                "host_cuda_event_p50_ns": median_u64(&host_cuda_event_samples),
                "device_cuda_event_p50_ns": median_u64(&device_cuda_event_samples),
                "p50_gate": p50_go,
                "p95_gate": p95_go,
                "required_p50_speedup": REQUIRED_P50_SPEEDUP,
                "pairs": PAIRS,
            })
        )?;
    }
    let telemetry_after = gpu_telemetry_snapshot()?;
    let status = if overall_go { "GO" } else { "STOP" };
    writeln!(
        output,
        "{}",
        serde_json::json!({
            "event": "summary",
            "status": status,
            "full61_promotion": false,
            "gpu_after": telemetry_after,
            "result": "local L3/L4 position-0/1 gate only",
        })
    )?;
    output.sync_all()?;
    println!(
        "{{\"status\":\"{status}\",\"mode\":\"v4-device-attention-gate\",\"result_jsonl\":{}}}",
        serde_json::to_string(&result_path)?
    );
    Ok(())
}

pub(crate) fn run_pinned_fixed_worker_boundary_gate(
    context: &std::sync::Arc<CudaContext>,
    module: &kernels::LoadedModule,
    stream: &std::sync::Arc<CudaStream>,
) -> Result<(), Box<dyn Error>> {
    const LAYERS: [usize; 3] = [0, 1, 2];
    let result_path = std::env::var_os("COLIBRI_V4_PINNED_FIXED_WORKER_BOUNDARY_RESULT_JSONL")
        .map(std::path::PathBuf::from)
        .ok_or("set COLIBRI_V4_PINNED_FIXED_WORKER_BOUNDARY_RESULT_JSONL")?;
    if std::env::var_os("COLIBRI_V4_FULL61_B1").is_some()
        || std::env::var_os("COLIBRI_V4_PHASE_PROFILE").is_some()
        || std::env::var_os("COLIBRI_V4_LAYER_STATIONARY_PREFILL").is_some()
        || std::env::var_os("COLIBRI_V4_LAYER_STATIONARY_FIXED_PREFETCH").is_some()
        || std::env::var_os("COLIBRI_V4_GREEDY_STEPS").is_some()
        || std::env::var_os("COLIBRI_V4_DEVICE_ATTENTION_PRODUCTION").is_some()
        || std::env::var_os("COLIBRI_V4_DEVICE_ATTENTION_AUDIT").is_some()
    {
        return Err(
            "pinned fixed worker boundary gate is isolated from full61, D, and hot modes".into(),
        );
    }
    if !optional_one_flag("COLIBRI_V4_PINNED_FIXED_PIPELINE")? || !raw_wo_a_gpu_decode_enabled()? {
        return Err(
            "pinned fixed worker boundary gate requires pinned pipeline and raw wo_a decode".into(),
        );
    }
    if v4_state::compression_ratio_for_layer(LAYERS[0])? != v4_state::CompressionRatio::Hca128
        || v4_state::compression_ratio_for_layer(LAYERS[1])? != v4_state::CompressionRatio::Hca128
        || v4_state::compression_ratio_for_layer(LAYERS[2])? != v4_state::CompressionRatio::Csa4
    {
        return Err("pinned fixed worker boundary HCA/CSA layer selection drift".into());
    }
    let mut result = PendingResultFile::reserve(result_path)?;
    let model_dir = v4_model_dir()?;
    let before_bytes = device_available_bytes()?;
    let transfer = context.new_stream()?;
    let mut cache =
        FixedLayerCache::stationary_raw_from_env(&model_dir, &LAYERS, 0, 0, 0, 0, stream)?;
    let l2_index = cache
        .plan
        .iter()
        .position(|plan| plan.layer == LAYERS[2])
        .ok_or("layer 2 is outside the pinned fixed retention plan")?;
    if cache.device_selected[l2_index] || !cache.host_selected[l2_index] {
        return Err("pinned fixed worker boundary gate requires layer 2 host retention".into());
    }
    let (l2_host, mut l2_origin) = load_layer_host_raw_wo_a(&model_dir, LAYERS[2], &[0])?;
    let (l2_runtime, l2_upload, l2_raw) =
        upload_layer_runtime_raw_wo_a(&model_dir, LAYERS[2], &l2_host, module, stream)?;
    l2_origin.fixed_h2d_ns = l2_upload.fixed_h2d_ns;
    l2_origin.fixed_h2d_bytes = l2_upload.fixed_h2d_bytes;
    l2_origin.raw_h2d_ns = l2_upload.raw_h2d_ns;
    l2_origin.raw_h2d_bytes = l2_upload.raw_h2d_bytes;
    cache.capture_stationary_raw(
        &model_dir, LAYERS[2], 1, l2_host, l2_runtime, l2_raw, &l2_origin, stream,
    )?;
    stream.synchronize()?;
    let after_origin_setup_bytes = device_available_bytes()?;
    let first_scratch = cache
        .raw_scratch
        .take()
        .ok_or("request phase setup: pinned fixed cache has no decode scratch")?;
    let second_scratch =
        DeviceBuffer::<u16>::zeroed(stream, RAW_FIXED_SCRATCH_BYTES / std::mem::size_of::<u16>())?;
    let slots = vec![
        PinnedFixedSlot::new(context, first_scratch)?,
        PinnedFixedSlot::new(context, second_scratch)?,
    ];
    let started = std::time::Instant::now();
    let (cache, layer_records) = std::thread::scope(|scope| -> Result<_, Box<dyn Error>> {
        let (request_tx, request_rx) = std::sync::mpsc::sync_channel::<PinnedFixedRequest>(2);
        let (reply_tx, reply_rx) = std::sync::mpsc::sync_channel::<PinnedFixedReply>(2);
        let main = std::sync::Arc::clone(stream);
        let transfer_worker = std::sync::Arc::clone(&transfer);
        let mut worker = Some(scope.spawn(move || {
            run_pinned_fixed_request_worker(
                cache,
                slots,
                &model_dir,
                module,
                &main,
                &transfer_worker,
                request_rx,
                reply_tx,
                false,
            )
        }));
        let mut request_tx = Some(request_tx);
        for (index, &layer) in LAYERS[..2].iter().enumerate() {
            request_tx
                .as_ref()
                .ok_or("request phase prepost: pinned fixed sender disappeared")?
                .send((None, Some((layer, 0, LAYERS.get(index + 1).copied()))))
                .map_err(|_| {
                    format!("layer {layer} phase prepost: pinned fixed worker receiver closed")
                })?;
        }
        let mut records = Vec::with_capacity(LAYERS.len());
        for (layer_position, &expected_layer) in LAYERS.iter().enumerate() {
            let prepare_started_ns =
                u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
            let reply = match reply_rx.recv() {
                Ok(reply) => reply,
                Err(_) => {
                    drop(request_tx.take());
                    return match worker
                            .take()
                            .ok_or("pinned fixed worker handle disappeared")?
                            .join()
                        {
                            Ok(Err(message)) => Err(format!(
                                "pinned fixed worker failed before layer {expected_layer} preparation: {message}"
                            )
                            .into()),
                            Ok(Ok(_)) => Err(format!(
                                "pinned fixed worker response channel closed before layer {expected_layer} preparation after a successful worker exit"
                            )
                            .into()),
                            Err(_) => Err(format!(
                                "pinned fixed worker panicked before layer {expected_layer} preparation"
                            )
                            .into()),
                        };
                }
            };
            let mut prepared = reply
                .map_err(|message| -> Box<dyn Error> { message.into() })?
                .ok_or("pinned fixed worker returned an early drain acknowledgement")?;
            if prepared.lease.layer != expected_layer {
                return Err(format!(
                    "pinned fixed worker returned layer {}, expected {expected_layer}",
                    prepared.lease.layer
                )
                .into());
            }
            let ready_received_ns = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
            prepared.wait_ready(stream)?;
            let ready_wait_done_ns =
                u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
            let actual_ratio = prepared.runtime().weights.compression_ratio;
            if actual_ratio != v4_state::compression_ratio_for_layer(expected_layer)? {
                return Err(format!(
                    "layer {expected_layer} phase ready: runtime compression ratio drift"
                )
                .into());
            }
            let generation = prepared.generation;
            let expected_generation = if layer_position < 2 { 1 } else { 2 };
            if generation != expected_generation {
                return Err(format!(
                    "layer {expected_layer} slot generation {generation} != expected {expected_generation}"
                )
                .into());
            }
            let free_at_ready_bytes = device_available_bytes()?;
            let telemetry = std::mem::take(&mut prepared.telemetry);
            let main_exposed_prepare_ns = ready_wait_done_ns
                .checked_sub(prepare_started_ns)
                .ok_or("pinned fixed gate prepare span is not monotonic")?;
            let fixed_accounting =
                fixed_prepare_dual_accounting(&telemetry, main_exposed_prepare_ns)?;
            prepared.record_runtime_done(stream).map_err(|error| {
                format!("layer {expected_layer} phase runtime_done_record: {error}")
            })?;
            let runtime_done_recorded_ns =
                u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
            let next = LAYERS
                .get(layer_position + 2)
                .map(|&layer| (layer, 0, LAYERS.get(layer_position + 3).copied()));
            request_tx
                .as_ref()
                .ok_or("request phase runtime_done: pinned fixed sender disappeared")?
                .send((Some(prepared), next))
                .map_err(|_| {
                    format!(
                        "layer {expected_layer} phase runtime_done_send: pinned fixed worker receiver closed"
                    )
                })?;
            records.push(serde_json::json!({
                "layer": expected_layer,
                "compression_ratio": actual_ratio.get(),
                "slot_generation": generation,
                "worker_command": {
                    "completed_layer": expected_layer,
                    "finish_before_prepare": true,
                    "next_layer": next.map(|(layer, _, _)| layer),
                },
                "phase": {
                    "prepare_started_ns": prepare_started_ns,
                    "ready_received_ns": ready_received_ns,
                    "ready_wait_done_ns": ready_wait_done_ns,
                    "runtime_done_recorded_ns": runtime_done_recorded_ns,
                },
                "fixed_accounting": fixed_accounting,
                "fixed_source_bytes": telemetry.fixed_source_bytes,
                "fixed_h2d_bytes": telemetry.fixed_h2d_bytes,
                "raw_h2d_bytes": telemetry.raw_h2d_bytes,
                "fixed_h2d_ns": telemetry.fixed_h2d_ns,
                "fixed_decode_ns": telemetry.fixed_decode_ns,
                "cache_status": telemetry.cache_status,
                "device_available_at_ready_bytes": free_at_ready_bytes,
            }));
        }
        let drained = match reply_rx.recv() {
            Ok(reply) => reply,
            Err(_) => {
                drop(request_tx.take());
                return match worker
                        .take()
                        .ok_or("pinned fixed worker handle disappeared")?
                        .join()
                    {
                        Ok(Err(message)) => Err(format!(
                            "pinned fixed worker failed before request drain: {message}"
                        )
                        .into()),
                        Ok(Ok(_)) => Err(
                            "pinned fixed worker response channel closed before request drain after a successful worker exit"
                                .into(),
                        ),
                        Err(_) => Err(
                            "pinned fixed worker panicked before request drain".into(),
                        ),
                    };
            }
        };
        if drained
            .map_err(|message| -> Box<dyn Error> { message.into() })?
            .is_some()
        {
            return Err("pinned fixed worker returned a layer instead of drain".into());
        }
        drop(request_tx.take());
        let cache = worker
            .take()
            .ok_or("pinned fixed worker handle disappeared")?
            .join()
            .map_err(|_| "pinned fixed worker panicked during request shutdown")?
            .map_err(|message| -> Box<dyn Error> {
                format!("pinned fixed worker failed during request shutdown: {message}").into()
            })?;
        Ok((cache, records))
    })?;
    let after_drain_bytes = device_available_bytes()?;
    let generation_restore = serde_json::json!({
        "raw_scratch_restored": cache.raw_scratch.is_some(),
        "raw_scratch_generation": cache.raw_scratch_generation,
        "layer_device_generations": LAYERS.map(|layer| cache.raw_device_generation[layer]),
        "layer_host_generations": LAYERS.map(|layer| cache.raw_host_generation[layer]),
        "device_hits": cache.device_hits,
        "host_hits": cache.host_hits,
        "misses": cache.misses,
    });
    drop(cache);
    drop(transfer);
    stream.synchronize()?;
    let after_release_bytes = device_available_bytes()?;
    if after_release_bytes < before_bytes {
        return Err(format!(
            "pinned fixed worker boundary left GPU residue: before={before_bytes} after_release={after_release_bytes}"
        )
        .into());
    }
    writeln!(
        result.file_mut(),
        "{}",
        serde_json::json!({
            "event": "run",
            "schema": "colibri-v4-pinned-fixed-worker-boundary-v2",
            "provenance": benchmark_provenance()?,
            "model_revision": v4_state::REVISION,
            "layers": LAYERS,
            "slot_count": 2,
            "request_channel_capacity": 2,
            "reply_channel_capacity": 2,
            "production_worker_loop": "run_pinned_fixed_request_worker",
            "compute": "none; main records runtime_done immediately after ready",
            "host_origin_setup": {
                "layer": LAYERS[2],
                "fixed_source_bytes": l2_origin.fixed_source_bytes,
                "fixed_h2d_bytes": l2_origin.fixed_h2d_bytes,
                "raw_h2d_bytes": l2_origin.raw_h2d_bytes,
                "device_available_after_bytes": after_origin_setup_bytes,
            },
            "ledger": {
                "hca_device_token_state_bytes": V4_DEVICE_TOKEN_STATE_HCA_BYTES,
                "csa_device_token_state_bytes": V4_DEVICE_TOKEN_STATE_MAX_BYTES,
                "full61_device_token_state_bytes": 31 * V4_DEVICE_TOKEN_STATE_HCA_BYTES
                    + 30 * V4_DEVICE_TOKEN_STATE_MAX_BYTES,
            },
            "layers_observed": layer_records,
            "generation_restore": generation_restore,
            "drain_status": "PASS",
            "join_status": "PASS",
            "cuMemGetInfo": {
                "available_before_bytes": before_bytes,
                "available_after_drain_bytes": after_drain_bytes,
                "available_after_release_bytes": after_release_bytes,
                "recovery_exact": after_release_bytes == before_bytes,
            },
        })
    )?;
    result.commit()?;
    println!(
        "{{\"status\":\"PASS\",\"mode\":\"v4-pinned-fixed-worker-boundary\",\"result_jsonl\":{}}}",
        serde_json::to_string(result.path())?
    );
    Ok(())
}

pub(crate) fn run_device_attention_preflight_gate(
    context: &std::sync::Arc<CudaContext>,
    module: &kernels::LoadedModule,
    stream: &std::sync::Arc<CudaStream>,
) -> Result<(), Box<dyn Error>> {
    let result_path = std::env::var_os("COLIBRI_V4_DEVICE_ATTENTION_PREFLIGHT_RESULT_JSONL")
        .map(std::path::PathBuf::from)
        .ok_or("set COLIBRI_V4_DEVICE_ATTENTION_PREFLIGHT_RESULT_JSONL")?;
    if std::env::var_os("COLIBRI_V4_FULL61_B1").is_some()
        || std::env::var_os("COLIBRI_V4_LAYER_STATIONARY_PREFILL").is_some()
        || std::env::var_os("COLIBRI_V4_ROUTED_QD4").is_some()
        || std::env::var_os("COLIBRI_V4_ROUTED_QD4_BENCH").is_some()
        || std::env::var_os("COLIBRI_V4_QD4_IMMUTABLE_EXPERT_CACHE").is_some()
    {
        return Err("device-attention preflight gate is isolated from full61, stationary, and production QD4 modes".into());
    }
    let mut result = PendingResultFile::reserve(result_path)?;
    let model_dir = v4_model_dir()?;
    let plan =
        device_attention_capacity_plan(&model_dir, true, false, V4_REQUEST_ROPE_LUT_ROWS, false)?;
    let relevant_config_fingerprint = normalized_config_fingerprint(serde_json::json!({
        "allocation_scope": "qd4_4_lanes+max_csa_fixed+request_rope_lut+max_csa_device_state+hca_device_state+tid2eid_transient+headroom",
        "headroom_bytes": plan.headroom_bytes,
        "max_device_token_state_bytes": plan.max_device_token_state_bytes,
        "hca_device_token_state_bytes": plan.hca_device_token_state_bytes,
        "max_fixed_device_bytes": plan.max_fixed_device_bytes,
        "model_revision": v4_state::REVISION,
        "qd4": true,
        "request_rope_lut_bytes": plan.rope_lut_bytes,
        "tid2eid_transient_max_bytes": plan.tid2eid_transient_max_bytes,
    }))?;
    let before_bytes = device_available_bytes()?;
    if before_bytes < plan.required_bytes {
        return Err(format!(
            "device-attention preflight requires {} bytes but cuMemGetInfo reports {} free",
            plan.required_bytes, before_bytes
        )
        .into());
    }
    let mut samples = vec![serde_json::json!({
        "stage": "before",
        "available_bytes": before_bytes,
    })];
    let mut min_available_bytes = before_bytes;
    let mut lane_streams = Vec::with_capacity(V4_QD4_LANES);
    for _ in 0..V4_QD4_LANES {
        lane_streams.push(context.new_stream()?);
    }
    let lanes = lane_streams
        .iter()
        .map(|lane_stream| ExpertSlot::new_empty(context, module, lane_stream))
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    for lane_stream in &lane_streams {
        lane_stream.synchronize()?;
    }
    let after_qd4_bytes = device_available_bytes()?;
    min_available_bytes = min_available_bytes.min(after_qd4_bytes);
    samples.push(serde_json::json!({
        "stage": "after_qd4_lanes",
        "available_bytes": after_qd4_bytes,
    }));
    let (hca_runtime, _) = load_layer_runtime(&model_dir, 3, &[0], stream)?;
    let hca_state = DeviceTokenState::new(stream, &hca_runtime.weights)?;
    if hca_state.num_bytes() != plan.hca_device_token_state_bytes {
        return Err(format!(
            "HCA DeviceTokenState bytes {} != planned HCA maximum {}",
            hca_state.num_bytes(),
            plan.hca_device_token_state_bytes
        )
        .into());
    }
    drop(hca_runtime);
    stream.synchronize()?;
    let after_hca_state_bytes = device_available_bytes()?;
    min_available_bytes = min_available_bytes.min(after_hca_state_bytes);
    samples.push(serde_json::json!({
        "stage": "after_hca_device_token_state",
        "available_bytes": after_hca_state_bytes,
    }));
    let (runtime, _) = load_layer_runtime(&model_dir, 4, &[0], stream)?;
    if runtime.weights.num_bytes() != plan.max_fixed_device_bytes {
        return Err(format!(
            "CSA layer 4 fixed bytes {} != metadata maximum {}",
            runtime.weights.num_bytes(),
            plan.max_fixed_device_bytes
        )
        .into());
    }
    stream.synchronize()?;
    let after_fixed_bytes = device_available_bytes()?;
    min_available_bytes = min_available_bytes.min(after_fixed_bytes);
    samples.push(serde_json::json!({
        "stage": "after_max_csa_fixed_runtime",
        "available_bytes": after_fixed_bytes,
    }));
    let rope_lut = DeviceRequestRopeLut::new(stream, V4_REQUEST_ROPE_LUT_ROWS)?;
    if rope_lut.num_bytes() != plan.rope_lut_bytes {
        return Err("request RoPE LUT allocation does not match the capacity ledger".into());
    }
    stream.synchronize()?;
    let after_rope_lut_bytes = device_available_bytes()?;
    min_available_bytes = min_available_bytes.min(after_rope_lut_bytes);
    samples.push(serde_json::json!({
        "stage": "after_request_rope_lut",
        "available_bytes": after_rope_lut_bytes,
    }));
    let state = DeviceTokenState::new(stream, &runtime.weights)?;
    if state.num_bytes() != plan.max_device_token_state_bytes {
        return Err(format!(
            "CSA DeviceTokenState bytes {} != planned maximum {}",
            state.num_bytes(),
            plan.max_device_token_state_bytes
        )
        .into());
    }
    let exact_compressor_scratch_bytes = state.exact_compressor_bytes();
    stream.synchronize()?;
    let after_state_bytes = device_available_bytes()?;
    min_available_bytes = min_available_bytes.min(after_state_bytes);
    samples.push(serde_json::json!({
        "stage": "after_max_csa_device_token_state",
        "available_bytes": after_state_bytes,
    }));
    let tid2eid = DeviceBuffer::from_host(stream, &[0_i64; v4_router::V4_TOPK])?;
    if tid2eid.num_bytes() != plan.tid2eid_transient_max_bytes {
        return Err("tid2eid transient allocation does not match the capacity ledger".into());
    }
    stream.synchronize()?;
    let after_lut_bytes = device_available_bytes()?;
    min_available_bytes = min_available_bytes.min(after_lut_bytes);
    samples.push(serde_json::json!({
        "stage": "after_tid2eid_transient",
        "available_bytes": after_lut_bytes,
    }));
    let pinned_fixed_pipeline = optional_one_flag("COLIBRI_V4_PINNED_FIXED_PIPELINE")?;
    let pinned_fixed_diagnostic = if pinned_fixed_pipeline {
        let transfer = context.new_stream()?;
        let mut cache =
            FixedLayerCache::stationary_raw_from_env(&model_dir, &[0], 0, 0, 0, 0, stream)?;
        let scratch = cache
            .raw_scratch
            .take()
            .ok_or("layer 0 phase diagnostic_setup: pinned fixed cache has no decode scratch")?;
        let slot = PinnedFixedSlot::new(context, scratch)?;
        let before_prepare_bytes = device_available_bytes()?;
        let mut prepared =
            cache.prepare_stationary_raw_pinned(&model_dir, 0, 0, module, &transfer, slot)?;
        prepared.wait_ready(stream)?;
        let telemetry = std::mem::take(&mut prepared.telemetry);
        let mut slot = prepared.finish(&mut cache, stream, &transfer)?;
        transfer.synchronize()?;
        cache.raw_scratch = slot.scratch.take();
        let after_finish_bytes = device_available_bytes()?;
        Some(serde_json::json!({
            "layer": 0,
            "token_id": 0,
            "status": "PASS",
            "boundary": "typed prepare/upload/ready/finish",
            "cache_status": telemetry.cache_status,
            "fixed_source_bytes": telemetry.fixed_source_bytes,
            "fixed_h2d_bytes": telemetry.fixed_h2d_bytes,
            "raw_h2d_bytes": telemetry.raw_h2d_bytes,
            "fixed_h2d_ns": telemetry.fixed_h2d_ns,
            "fixed_decode_ns": telemetry.fixed_decode_ns,
            "device_available_before_prepare_bytes": before_prepare_bytes,
            "device_available_after_finish_bytes": after_finish_bytes,
        }))
    } else {
        None
    };
    drop(tid2eid);
    drop(state);
    drop(hca_state);
    drop(rope_lut);
    drop(runtime);
    drop(lanes);
    for lane_stream in &lane_streams {
        lane_stream.synchronize()?;
    }
    stream.synchronize()?;
    drop(lane_streams);
    let after_release_bytes = device_available_bytes()?;
    if after_release_bytes < before_bytes {
        return Err(format!(
            "device-attention preflight left GPU residue: before={} after_release={}",
            before_bytes, after_release_bytes
        )
        .into());
    }
    samples.push(serde_json::json!({
        "stage": "after_release",
        "available_bytes": after_release_bytes,
    }));
    let output = result.file_mut();
    writeln!(
        output,
        "{}",
        serde_json::json!({
            "event": "run",
            "schema": "colibri-v4-device-attention-capacity-preflight-v1",
            "provenance": benchmark_provenance()?,
            "relevant_config_fingerprint": relevant_config_fingerprint,
            "model_revision": v4_state::REVISION,
            "allocation_scope": "QD4 four lanes, maximum CSA fixed runtime, one request-global RoPE LUT, maximum CSA DeviceTokenState/strict scratch plus one HCA DeviceTokenState/strict scratch, then one true-peak tid2eid row; no layer execution or full61",
            "checked_sum_bytes": plan.required_bytes,
            "qd4_device_bytes": plan.qd4_device_bytes,
            "max_fixed_layer": plan.fixed_layer,
            "allocated_csa_layer": 4,
            "max_fixed_device_bytes": plan.max_fixed_device_bytes,
            "max_device_token_state_bytes": plan.max_device_token_state_bytes,
            "hca_device_token_state_bytes": plan.hca_device_token_state_bytes,
            "actual_exact_compressor_scratch_bytes": exact_compressor_scratch_bytes,
            "rope_lut_bytes": plan.rope_lut_bytes,
            "rope_lut_status": "device_request_global",
            "tid2eid_transient_max_bytes": plan.tid2eid_transient_max_bytes,
            "headroom_bytes": plan.headroom_bytes,
            "pinned_fixed_diagnostic": pinned_fixed_diagnostic,
            "cuMemGetInfo": {
                "available_before_bytes": before_bytes,
                "available_after_release_bytes": after_release_bytes,
                "available_min_bytes": min_available_bytes,
                "recovery_contract": "after_release >= before; a larger value is tolerated because unrelated CUDA allocations may have been released",
                "recovery_exact": after_release_bytes == before_bytes,
                "samples": samples,
            },
        })
    )?;
    result.commit()?;
    println!(
        "{{\"status\":\"PASS\",\"mode\":\"v4-device-attention-capacity-preflight\",\"result_jsonl\":{}}}",
        serde_json::to_string(result.path())?
    );
    Ok(())
}

pub(crate) fn run_real_v4_device_token_state_gate(
    context: &std::sync::Arc<CudaContext>,
    module: &kernels::LoadedModule,
    stream: &std::sync::Arc<CudaStream>,
) -> Result<(), Box<dyn Error>> {
    let result_path = std::env::var_os("COLIBRI_V4_DEVICE_TOKEN_STATE_RESULT_JSONL")
        .map(std::path::PathBuf::from)
        .ok_or("set COLIBRI_V4_DEVICE_TOKEN_STATE_RESULT_JSONL")?;
    if std::env::var_os("COLIBRI_V4_FULL61_B1").is_some()
        || std::env::var_os("COLIBRI_V4_LAYER_STATIONARY_PREFILL").is_some()
        || std::env::var_os("COLIBRI_V4_ROUTED_QD4").is_some()
        || std::env::var_os("COLIBRI_V4_ROUTED_QD4_BENCH").is_some()
    {
        return Err(
            "device token-state gate is isolated from full61, stationary, and QD4 modes".into(),
        );
    }
    let mut result = PendingResultFile::reserve(result_path)?;
    let model_dir = v4_model_dir()?;
    let raw_wo_a_gate = optional_one_flag("COLIBRI_V4_RAW_WO_A_GATE")?;
    let hca_reuse_gate = optional_one_flag("COLIBRI_V4_HCA_REUSE_GATE")?;
    if hca_reuse_gate && !raw_wo_a_gate {
        return Err("HCA reuse gate requires raw wo_a GPU decode".into());
    }
    let layers: &[usize] = if hca_reuse_gate {
        &[0, 1]
    } else if raw_wo_a_gate {
        &[3, 4]
    } else {
        &[0, 3, 2, 4]
    };
    let capacity = validate_device_attention_capacity(
        &model_dir,
        false,
        false,
        V4_REQUEST_ROPE_LUT_ROWS,
        false,
        device_available_bytes()?,
    )?;
    let token_ids = qd4_gate_prompt_ids(&model_dir)?;
    let inputs =
        prepare_qd4_stationary_gate_inputs(context, module, stream, &model_dir, &token_ids, 4)?;
    let gate_cublas = cublas_exact::CublasExact::load()?;
    let rope_lut = DeviceRequestRopeLut::new(stream, V4_REQUEST_ROPE_LUT_ROWS)?;
    let cublas_provenance = gate_cublas.provenance();
    let sparse_attention_arm = match sparse_attention_arm_from_env()? {
        SparseAttentionArm::Reference => "reference:sparse_attn_v4_bf16",
        SparseAttentionArm::ScoreReuse => "score_reuse:sparse_attn_v4_scores+probabilities+values",
        SparseAttentionArm::Sm89Crossover => "sm89_crossover:K<4=reference,K>=4=score_reuse",
    };
    let relevant_config_fingerprint = normalized_config_fingerprint(serde_json::json!({
        "candidate": "device_authoritative_strict_cublas",
        "host_control": "independent_official_cublas",
        "layers": layers,
        "model_revision": v4_state::REVISION,
        "positions": [0, 1, 2, 3, 4],
        "prompt_ids": token_ids,
        "request_rope_lut_rows": V4_REQUEST_ROPE_LUT_ROWS,
        "sparse_attention_arm": sparse_attention_arm,
    }))?;
    let output = result.file_mut();
    writeln!(
        output,
        "{}",
        serde_json::json!({
            "event": "run",
            "schema": "colibri-v4-device-token-state-pos4-gate-v3",
            "provenance": benchmark_provenance()?,
            "relevant_config_fingerprint": relevant_config_fingerprint,
            "model_revision": v4_state::REVISION,
            "prompt": "official single-user chat hi",
            "prompt_ids": token_ids,
            "layers": layers,
            "positions": [0, 1, 2, 3, 4],
            "candidate": "device-authoritative persistent-state staging with strict cuBLAS main and indexer Compressor projections",
            "host_control": "strict cuBLAS host attention runner, used as the independent canonical persistent-state oracle for main and indexer Compressor projections",
            "strict_cublas": {
                "resolved_path": cublas_provenance.resolved_path,
                "sha256": cublas_provenance.actual_sha256,
                "abi": "cublasSgemm_v2",
                "version": cublas_provenance.version,
            },
            "gate": {
                "exact": if hca_reuse_gate { "every position: BF16 KV, compressed KV/indexer KV, complete attention_post BF16, sparse top-k, and all defined HCA persistent compressor FP32 slots" } else { "every position: BF16 KV, compressed KV/indexer KV, complete attention_post BF16, and sparse top-k; CSA additionally checks all persistent compressor FP32 bits because its boundary reads them, while HCA's unwritten compressor tail is deliberately not observable state" },
                "transaction": "candidate device state is audited before any production state is exposed; failure returns an error without host-state mutation",
                "scope": if hca_reuse_gate { "isolated L0 to L1 HCA continuous-reuse five-input pos0..4 gate using raw-FP8 wo_a H2D plus existing GPU decode; passing does not claim full61 production support" } else if raw_wo_a_gate { "isolated L3 HCA and L4 CSA five-input pos0..4 gate using raw-FP8 wo_a H2D plus existing GPU decode; passing does not claim full61 production support" } else { "isolated L0/L3 HCA and L2/L4 CSA five-input pos0..4 gate only; each layout reuses one candidate across layers; passing does not claim full61 production support" },
                "transfer": "one request-global RoPE LUT H2D plus per-row hidden H2D; forward/inverse RoPE rows are scoped device-to-device copies; all persistent writes and boundary transforms are device-side; strict audits use post-run D2H"
            }
        })
    )?;
    let mut hca_candidate = None;
    let mut csa_candidate = None;
    let mut hca_state_allocations = 0_usize;
    let mut csa_state_allocations = 0_usize;
    for &layer in layers {
        let layer_inputs = inputs
            .get(&layer)
            .ok_or("real prefix did not retain device-token-state gate inputs")?;
        if layer_inputs.len() < 5 {
            return Err("real prefix produced fewer than five input rows".into());
        }
        let (runtime, raw_upload) = if raw_wo_a_gate {
            let (host, _) = load_layer_host_raw_wo_a(&model_dir, layer, &token_ids)?;
            let raw_bytes = host
                .tail
                .wo_a_raw
                .as_ref()
                .ok_or("raw wo_a gate host payload disappeared")?
                .len();
            let host_payload_bytes = host.num_bytes();
            let (runtime, telemetry, _raw_wo_a) =
                upload_layer_runtime_raw_wo_a(&model_dir, layer, &host, module, stream)?;
            let decoded_bytes = runtime.weights.wo_a.num_bytes();
            if telemetry.raw_h2d_bytes != raw_bytes
                || telemetry.fixed_h2d_bytes.checked_add(raw_bytes)
                    != Some(runtime.weights.num_bytes())
                || decoded_bytes
                    != raw_bytes
                        .checked_mul(2)
                        .ok_or("raw wo_a decoded byte overflow")?
            {
                return Err(format!(
                    "raw wo_a physical transfer contract failed for layer {layer}"
                )
                .into());
            }
            writeln!(
                output,
                "{}",
                serde_json::json!({
                    "event": "raw_wo_a_transfer",
                    "layer": layer,
                    "host_payload_bytes": host_payload_bytes,
                    "raw_h2d_bytes": telemetry.raw_h2d_bytes,
                    "decoded_device_bytes": decoded_bytes,
                    "total_fixed_h2d_bytes": telemetry.fixed_h2d_bytes,
                    "physical_contract": "raw H2D plus all non-wo_a uploads; decoded wo_a remains resident as BF16",
                    "exact": true,
                })
            )?;
            (runtime, true)
        } else {
            (
                load_layer_runtime(&model_dir, layer, &token_ids, stream)?.0,
                false,
            )
        };
        let mut canonical = v4_state::V4SerialPersistentState::new(5)?;
        let slot = if runtime.weights.compression_ratio == v4_state::CompressionRatio::Csa4 {
            &mut csa_candidate
        } else {
            &mut hca_candidate
        };
        let reused = slot.is_some();
        if !reused {
            *slot = Some(DeviceTokenState::new(stream, &runtime.weights)?);
            if runtime.weights.compression_ratio == v4_state::CompressionRatio::Csa4 {
                csa_state_allocations += 1;
            } else {
                hca_state_allocations += 1;
            }
        }
        let candidate = slot
            .as_mut()
            .ok_or("device token-state gate candidate cache slot disappeared")?;
        let _ = candidate.reset_for_layer(module, stream, &runtime.weights)?;
        if candidate.indexer_valid.to_host_vec(stream)? != [1] {
            return Err("device token-state reset did not restore indexer validity".into());
        }
        writeln!(
            output,
            "{}",
            serde_json::json!({
                "event": "device_memory",
                "layer": layer,
                "compression_ratio": runtime.weights.compression_ratio.get(),
                "reused_layout_state": reused,
                "device_token_state_bytes": candidate.num_bytes(),
                "exact_compressor_scratch_bytes": candidate.exact_compressor_bytes(),
            })
        )?;
        for position in 0..5 {
            let mut host_attention = None;
            canonical.transaction(layer, |state| {
                let run = run_real_layer_attention_rows(
                    module,
                    stream,
                    None,
                    &runtime,
                    &layer_inputs[position],
                    position,
                    state,
                    Some(&gate_cublas),
                    None,
                )?;
                host_attention = Some((
                    run.output,
                    run.topk
                        .ok_or("strict token-state host oracle omitted sparse top-k")?,
                ));
                Ok(())
            })?;
            let expected = host_persistent_state_audit(
                canonical
                    .layer(layer)
                    .ok_or("canonical token-state gate layer disappeared")?,
            );
            run_device_token_state_commit_row(
                module,
                stream,
                &runtime,
                candidate,
                &rope_lut,
                &layer_inputs[position],
                position,
            )?;
            let actual = device_persistent_state_audit(stream, &candidate)?;
            let require_compressor_state = hca_reuse_gate
                || runtime.weights.compression_ratio == v4_state::CompressionRatio::Csa4;
            check_persistent_state_audit_with_compressor_state(
                layer,
                position,
                &expected,
                &actual,
                require_compressor_state,
            )?;
            let attention = {
                let (output, topk) = run_device_attention_after_commit(
                    module, stream, &runtime, candidate, &rope_lut, position, true,
                )?;
                let (expected_output, expected_topk) =
                    host_attention.ok_or("canonical attention result disappeared")?;
                if let Some(index) = first_mismatch(&expected_output, &output) {
                    return Err(format!(
                        "device L{} pos{position} attention output first mismatch at {index}",
                        layer,
                    )
                    .into());
                }
                let topk = topk.ok_or("strict token-state gate omitted top-k D2H")?;
                if let Some(index) = first_mismatch(&expected_topk, &topk) {
                    return Err(format!(
                        "device L{layer} pos{position} sparse top-k first mismatch at {index}"
                    )
                    .into());
                }
                let expected_topk_digest = fnv1a_i32(&expected_topk);
                let actual_topk_digest = fnv1a_i32(&topk);
                debug_assert_eq!(expected_topk_digest, actual_topk_digest);
                let after_attention_state = device_persistent_state_audit(stream, &candidate)?;
                check_persistent_state_audit_with_compressor_state(
                    layer,
                    position,
                    &expected,
                    &after_attention_state,
                    require_compressor_state,
                )?;
                Some(serde_json::json!({
                    "exact": true,
                    "attention_post_bf16_digest": format!("{:016x}", fnv1a_u16(&output)),
                    "sparse_topk_digest": format!("{actual_topk_digest:016x}"),
                    "sparse_topk": topk,
                }))
            };
            writeln!(
                output,
                "{}",
                serde_json::json!({
                    "event": "position",
                    "layer": layer,
                    "position": position,
                    "exact": true,
                    "persistent_state": persistent_state_audit_json(&actual),
                    "candidate_committed_rows": candidate.committed_rows,
                    "candidate_compressed_rows": candidate.compressed_rows,
                    "complete_attention": attention,
                })
            )?;
        }
        writeln!(
            output,
            "{}",
            serde_json::json!({
                "event": "layer_summary",
                "layer": layer,
                "status": "GO",
                "exact_positions": 5,
                "raw_wo_a_gpu_decode": raw_upload,
                "complete_attention_claim": true,
            })
        )?;
    }
    writeln!(
        output,
        "{}",
        serde_json::json!({
            "event": "summary",
            "status": "GO",
            "state_allocations": {"hca": hca_state_allocations, "csa": csa_state_allocations},
            "strict_cublas_handles": {"hca": hca_state_allocations, "csa": csa_state_allocations},
            "capacity_guard": {
                "required_bytes": capacity.required_bytes,
                "max_csa_state_bytes": capacity.max_device_token_state_bytes,
                "hca_state_bytes": capacity.hca_device_token_state_bytes,
                "headroom_bytes": capacity.headroom_bytes,
            },
            "result": if hca_reuse_gate { "raw-FP8 wo_a H2D plus GPU decode passed device-authoritative L0 to L1 HCA continuous-reuse persistent state plus complete attention pos0..4 exact output/top-k/state audit" } else if raw_wo_a_gate { "raw-FP8 wo_a H2D plus GPU decode passed device-authoritative L3 HCA and L4 CSA persistent state plus complete attention pos0..4 exact output/top-k/state audit" } else { "device-authoritative L0/L3 HCA and L2/L4 CSA persistent state plus complete attention passed pos0..4 exact output/top-k/state audit" },
            "complete_attention_claim": true,
            "full61_production_claim": false,
        })
    )?;
    output.sync_all()?;
    result.commit()?;
    println!(
        "{{\"status\":\"GO\",\"mode\":\"v4-device-token-state-pos4-gate\",\"result_jsonl\":{}}}",
        serde_json::to_string(result.path())?
    );
    Ok(())
}

pub(crate) fn run_real_v4_cache_off_bench() -> Result<(), Box<dyn Error>> {
    let repeats = std::env::var("COLIBRI_V4_CACHE_OFF_REPEATS")
        .unwrap_or_else(|_| "30".into())
        .parse::<usize>()?;
    if repeats != 3 && repeats < 30 {
        return Err("COLIBRI_V4_CACHE_OFF_REPEATS must be 3 for smoke or at least 30".into());
    }
    let result_path = std::env::var_os("COLIBRI_V4_CACHE_OFF_RESULT_JSONL")
        .map(std::path::PathBuf::from)
        .ok_or("set COLIBRI_V4_CACHE_OFF_RESULT_JSONL")?;
    if let Some(parent) = result_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let model_dir = v4_model_dir()?;
    let mut sources = [
        CacheOffSource::new(&model_dir, 3)?,
        CacheOffSource::new(&model_dir, 4)?,
    ];
    let mut output = std::fs::File::create(&result_path)?;
    writeln!(
        output,
        "{}",
        serde_json::json!({
            "event": "run",
            "schema": "colibri-v4-cache-off-stage0-v1",
            "provenance": benchmark_provenance()?,
            "model_mount": model_mount_provenance(&model_dir)?,
            "model_revision": v4_state::REVISION,
            "layers": sources.iter().map(|source| source.layer).collect::<Vec<_>>(),
            "expert_id": 0,
            "repeats": repeats,
            "arm_order_schedule": "BF/FB; every arm is range-evicted before timing",
            "gate": "exact digest; cache-off median wall <= buffered*1.10; lower Cached delta and no worse MemAvailable delta",
        })
    )?;
    let modes = [
        safetensors::RangeReadMode::Buffered,
        safetensors::RangeReadMode::Fadvise,
    ];
    let name = |mode| match mode {
        safetensors::RangeReadMode::Buffered => "buffered",
        safetensors::RangeReadMode::Fadvise => "fadvise",
        safetensors::RangeReadMode::Direct => "direct",
    };
    let mut samples = [Vec::with_capacity(repeats), Vec::with_capacity(repeats)];
    let mut oracle = None;
    for repeat in 0..repeats {
        let order = if b2_serial_first(repeat) {
            [0_usize, 1]
        } else {
            [1_usize, 0]
        };
        for &mode_index in &order {
            let sample = run_cache_off_arm(&mut sources, modes[mode_index])?;
            if oracle.is_some_and(|digest| digest != sample.digest) {
                return Err(format!(
                    "cache-off repeat {repeat} {} changed canonical bytes",
                    name(modes[mode_index])
                )
                .into());
            }
            oracle.get_or_insert(sample.digest);
            writeln!(
                output,
                "{}",
                serde_json::json!({
                    "event": "sample",
                    "repeat": repeat,
                    "arm_order": if order[0] == 0 { "BF" } else { "FB" },
                    "mode": name(modes[mode_index]),
                    "read_wall_ns": sample.read_wall_ns,
                    "logical_bytes": sample.logical_bytes,
                    "physical_bytes": sample.physical_bytes,
                    "physical_overread_bytes": sample.physical_bytes - sample.logical_bytes,
                    "digest": format!("{:016x}", sample.digest),
                    "mem_available_delta_bytes": sample.available_delta_bytes,
                    "cached_delta_bytes": sample.cached_delta_bytes,
                })
            )?;
            samples[mode_index].push(sample);
        }
    }
    let wall = samples.each_ref().map(|values| {
        median_u64(
            &values
                .iter()
                .map(|sample| sample.read_wall_ns)
                .collect::<Vec<_>>(),
        )
    });
    let cached = samples.each_ref().map(|values| {
        median_i64(
            &values
                .iter()
                .map(|sample| sample.cached_delta_bytes)
                .collect::<Vec<_>>(),
        )
    });
    let available = samples.each_ref().map(|values| {
        median_i64(
            &values
                .iter()
                .map(|sample| sample.available_delta_bytes)
                .collect::<Vec<_>>(),
        )
    });
    let fadvise_pass = wall[1] <= wall[0].saturating_mul(110) / 100
        && cached[1] < cached[0]
        && available[1] >= available[0];
    let status = if fadvise_pass { "PASS" } else { "KILL" };
    writeln!(
        output,
        "{}",
        serde_json::json!({
            "event": "summary",
            "status": status,
            "canonical_exact": true,
            "buffered": {"wall_p50_ns": wall[0], "cached_delta_p50_bytes": cached[0], "mem_available_delta_p50_bytes": available[0]},
            "fadvise": {"wall_p50_ns": wall[1], "wall_ratio": wall[1] as f64 / wall[0] as f64, "cached_delta_p50_bytes": cached[1], "mem_available_delta_p50_bytes": available[1], "gate_pass": fadvise_pass},
            "direct": serde_json::Value::Null,
            "logical_bytes_per_arm": samples[0][0].logical_bytes,
            "direct_physical_bytes_per_arm": serde_json::Value::Null,
            "direct_overread_bytes_per_arm": serde_json::Value::Null,
            "repeats": repeats,
        })
    )?;
    println!(
        "{{\"status\":\"{status}\",\"mode\":\"v4-cache-off-stage0\",\"result_jsonl\":{}}}",
        serde_json::to_string(&result_path)?
    );
    Ok(())
}

pub(crate) fn sparse_attention_score_reuse_k128_attention_post_microgate() {
    fn assert_bf16_stage(stage: &str, actual: &[u16], expected: &[u16]) {
        assert_eq!(actual.len(), expected.len(), "{stage}: length");
        if let Some((index, (&actual, &expected))) = actual
            .iter()
            .zip(expected)
            .enumerate()
            .find(|(_, (actual, expected))| actual != expected)
        {
            panic!(
                "{stage}: BF16 mismatch at {index}: {actual:04x} != {expected:04x}, delta={}",
                (v4_router::bf16_to_f32(actual) - v4_router::bf16_to_f32(expected)).abs()
            );
        }
    }

    let topk_host = v4_state::window_topk_cpu(1, 3, v4_state::KV_WINDOW)
        .unwrap()
        .0;
    assert_eq!(topk_host.len(), v4_state::KV_WINDOW);
    assert_eq!(&topk_host[..5], &[0, 1, 2, 3, -1]);
    let q_host = (0..v4_state::Q_HEADS * v4_state::Q_HEAD_DIM)
        .map(|index| {
            let value =
                ((index.wrapping_mul(73).wrapping_add(19) % 257) as f32 - 128.0) * 0.007_812_5;
            v4_router::f32_to_bf16_rne(value)
        })
        .collect::<Vec<_>>();
    let kv_host = (0..v4_state::KV_WINDOW * v4_state::Q_HEAD_DIM)
        .map(|index| {
            let value =
                ((index.wrapping_mul(151).wrapping_add(41) % 509) as f32 - 254.0) * 0.003_906_25;
            v4_router::f32_to_bf16_rne(value)
        })
        .collect::<Vec<_>>();
    let sink_host = (0..v4_state::Q_HEADS)
        .map(|head| -1.0_f32 - (head % 11) as f32 * 0.0625)
        .collect::<Vec<_>>();
    let context = CudaContext::new(0).expect("create CUDA context");
    let stream = context.default_stream();
    let module = kernels::load(&context).expect("load embedded Rust PTX");
    let q = DeviceBuffer::from_host(&stream, &q_host).unwrap();
    let kv = DeviceBuffer::from_host(&stream, &kv_host).unwrap();
    let sink = DeviceBuffer::from_host(&stream, &sink_host).unwrap();
    let topk = DeviceBuffer::from_host(&stream, &topk_host).unwrap();
    let mut reference = DeviceBuffer::<u16>::zeroed(&stream, q_host.len()).unwrap();
    let mut candidate = DeviceBuffer::<u16>::zeroed(&stream, q_host.len()).unwrap();
    let mut reference_state =
        SparseAttentionScoreReuseScratch::new(&stream, 1, v4_state::KV_WINDOW).unwrap();
    let mut candidate_state =
        SparseAttentionScoreReuseScratch::new(&stream, 1, v4_state::KV_WINDOW).unwrap();
    run_sparse_attention_device_into(
        &module,
        &stream,
        &q,
        &kv,
        &sink,
        &topk,
        &mut reference,
        v4_state::KV_WINDOW,
    )
    .unwrap();
    run_sparse_attention_reference_state_device_into(
        &module,
        &stream,
        &q,
        &kv,
        &sink,
        &topk,
        &mut reference_state,
        v4_state::KV_WINDOW,
    )
    .unwrap();
    run_sparse_attention_score_reuse_device_into(
        &module,
        &stream,
        &q,
        &kv,
        &sink,
        &topk,
        &mut candidate,
        &mut candidate_state,
        v4_state::KV_WINDOW,
    )
    .unwrap();
    assert_eq!(
        candidate_state.scores.to_host_vec(&stream).unwrap(),
        reference_state.scores.to_host_vec(&stream).unwrap(),
        "scores"
    );
    assert_eq!(
        candidate_state.probabilities.to_host_vec(&stream).unwrap(),
        reference_state.probabilities.to_host_vec(&stream).unwrap(),
        "probabilities"
    );
    assert_eq!(
        candidate_state.denominators.to_host_vec(&stream).unwrap(),
        reference_state.denominators.to_host_vec(&stream).unwrap(),
        "denominators"
    );
    assert_eq!(
        candidate_state.valid_counts.to_host_vec(&stream).unwrap(),
        reference_state.valid_counts.to_host_vec(&stream).unwrap(),
        "valid counts"
    );
    let reference_sparse = reference.to_host_vec(&stream).unwrap();
    assert_bf16_stage(
        "sparse",
        &candidate.to_host_vec(&stream).unwrap(),
        &reference_sparse,
    );

    let mut topk_129_host = topk_host.clone();
    topk_129_host.push(-1);
    let topk_129 = DeviceBuffer::from_host(&stream, &topk_129_host).unwrap();
    let mut reference_129 = DeviceBuffer::<u16>::zeroed(&stream, q_host.len()).unwrap();
    let mut candidate_129 = DeviceBuffer::<u16>::zeroed(&stream, q_host.len()).unwrap();
    let mut reference_state_130 = SparseAttentionScoreReuseScratch::new(&stream, 1, 130).unwrap();
    let mut candidate_state_130 = SparseAttentionScoreReuseScratch::new(&stream, 1, 130).unwrap();
    run_sparse_attention_device_into(
        &module,
        &stream,
        &q,
        &kv,
        &sink,
        &topk_129,
        &mut reference_129,
        129,
    )
    .unwrap();
    run_sparse_attention_reference_state_device_into(
        &module,
        &stream,
        &q,
        &kv,
        &sink,
        &topk_129,
        &mut reference_state_130,
        129,
    )
    .unwrap();
    run_sparse_attention_score_reuse_device_into(
        &module,
        &stream,
        &q,
        &kv,
        &sink,
        &topk_129,
        &mut candidate_129,
        &mut candidate_state_130,
        129,
    )
    .unwrap();
    let logical_score_len = v4_state::Q_HEADS * 129;
    assert_eq!(
        &candidate_state_130.scores.to_host_vec(&stream).unwrap()[..logical_score_len],
        &reference_state_130.scores.to_host_vec(&stream).unwrap()[..logical_score_len],
        "K129 scores in K130 scratch"
    );
    assert_eq!(
        &candidate_state_130
            .probabilities
            .to_host_vec(&stream)
            .unwrap()[..logical_score_len],
        &reference_state_130
            .probabilities
            .to_host_vec(&stream)
            .unwrap()[..logical_score_len],
        "K129 probabilities in K130 scratch"
    );
    assert_bf16_stage(
        "K129 sparse in K130 scratch",
        &candidate_129.to_host_vec(&stream).unwrap(),
        &reference_129.to_host_vec(&stream).unwrap(),
    );

    let forward_freqs = (0..v4_state::Q_ROPE_DIM / 2)
        .flat_map(|pair| {
            if pair.is_multiple_of(2) {
                [0.875_f32, 0.5]
            } else {
                [0.625_f32, -0.75]
            }
        })
        .collect::<Vec<_>>();
    let mut inverse_freqs_host = forward_freqs.clone();
    for sine in inverse_freqs_host.iter_mut().skip(1).step_by(2) {
        *sine = -*sine;
    }
    let reference_inverse = run_l0_rope(
        &module,
        &stream,
        &reference_sparse,
        &forward_freqs,
        v4_state::Q_HEADS,
        true,
    )
    .unwrap();
    let inverse_freqs = DeviceBuffer::from_host(&stream, &inverse_freqs_host).unwrap();
    let mut candidate_inverse = DeviceBuffer::<u16>::zeroed(&stream, q_host.len()).unwrap();
    run_rope_device_into(
        &module,
        &stream,
        &candidate,
        &inverse_freqs,
        &mut candidate_inverse,
        v4_state::Q_HEADS,
        v4_state::Q_HEAD_DIM,
    )
    .unwrap();
    assert_bf16_stage(
        "sparse_inverse",
        &candidate_inverse.to_host_vec(&stream).unwrap(),
        &reference_inverse,
    );

    let group_input = v4_state::Q_HEADS * v4_state::Q_HEAD_DIM / v4_state::O_GROUPS;
    let wo_a_width = v4_state::O_GROUPS * v4_state::O_LORA_RANK;
    let mut wo_a_host = vec![0_u16; wo_a_width * group_input];
    for output in 0..wo_a_width {
        wo_a_host[output * group_input + output.wrapping_mul(17) % group_input] =
            v4_router::f32_to_bf16_rne(1.0);
    }
    let wo_a = DeviceBuffer::from_host(&stream, &wo_a_host).unwrap();
    drop(wo_a_host);
    let reference_wo_a = run_l0_wo_a(&module, &stream, &reference_inverse, &wo_a).unwrap();
    let mut candidate_wo_a = DeviceBuffer::<u16>::zeroed(&stream, wo_a_width).unwrap();
    run_l0_wo_a_device_into(
        &module,
        &stream,
        &candidate_inverse,
        &wo_a,
        &mut candidate_wo_a,
    )
    .unwrap();
    assert_bf16_stage(
        "wo_a",
        &candidate_wo_a.to_host_vec(&stream).unwrap(),
        &reference_wo_a,
    );

    let mut wo_b_host = vec![0_u8; v4_state::HIDDEN * wo_a_width];
    let one_fp8 = v4_state::E4M3Fn::encode(1.0).unwrap().0;
    for output in 0..v4_state::HIDDEN {
        wo_b_host[output * wo_a_width + output.wrapping_mul(29) % wo_a_width] = one_fp8;
    }
    let wo_b = DeviceFp8Matrix {
        weight: DeviceBuffer::from_host(&stream, &wo_b_host).unwrap(),
        scale: DeviceBuffer::from_host(
            &stream,
            &vec![
                127_u8;
                v4_state::HIDDEN.div_ceil(v4_state::FP8_BLOCK)
                    * (wo_a_width / v4_state::FP8_BLOCK)
            ],
        )
        .unwrap(),
        in_features: wo_a_width,
        out_features: v4_state::HIDDEN,
    };
    drop(wo_b_host);
    let mut candidate_quantized = DeviceBuffer::<u8>::zeroed(&stream, wo_a_width).unwrap();
    let mut candidate_scales =
        DeviceBuffer::<u8>::zeroed(&stream, wo_a_width / v4_state::FP8_BLOCK).unwrap();
    let mut empty_candidate = DeviceBuffer::<u16>::zeroed(&stream, 0).unwrap();
    let mut candidate_attention_f32 =
        DeviceBuffer::<f32>::zeroed(&stream, v4_state::HIDDEN).unwrap();
    let mut candidate_attention = DeviceBuffer::<u16>::zeroed(&stream, v4_state::HIDDEN).unwrap();
    let (reference_attention_f32, reference_attention) =
        run_l0_fp8_linear(&module, &stream, &reference_wo_a, &wo_b, false).unwrap();
    run_l0_fp8_linear_device_into(
        &module,
        &stream,
        &candidate_wo_a,
        &wo_b,
        &mut candidate_quantized,
        &mut candidate_scales,
        &mut empty_candidate,
        &mut candidate_attention_f32,
        &mut candidate_attention,
        false,
    )
    .unwrap();
    assert_eq!(
        candidate_attention_f32.to_host_vec(&stream).unwrap(),
        reference_attention_f32,
        "attention f32"
    );
    assert_bf16_stage(
        "attention",
        &candidate_attention.to_host_vec(&stream).unwrap(),
        &reference_attention,
    );

    let residual_host = vec![0_u16; v4_state::HC * v4_state::HIDDEN];
    let post_host = [1.0_f32; v4_state::HC];
    let comb_host = [0.0_f32; v4_state::HC * v4_state::HC];
    let reference_post = run_l0_hc_post(
        &module,
        &stream,
        &reference_attention,
        &residual_host,
        &post_host,
        &comb_host,
    )
    .unwrap();
    let residual = DeviceBuffer::from_host(&stream, &residual_host).unwrap();
    let post = DeviceBuffer::from_host(&stream, &post_host).unwrap();
    let comb = DeviceBuffer::from_host(&stream, &comb_host).unwrap();
    let mut candidate_post =
        DeviceBuffer::<u16>::zeroed(&stream, v4_state::HC * v4_state::HIDDEN).unwrap();
    run_l0_hc_post_device_into(
        &module,
        &stream,
        &candidate_attention,
        &residual,
        &post,
        &comb,
        &mut candidate_post,
    )
    .unwrap();
    assert_bf16_stage(
        "attention_post",
        &candidate_post.to_host_vec(&stream).unwrap(),
        &reference_post,
    );
    println!(
        "{{\"status\":\"PASS\",\"mode\":\"v4-sparse-k128-attention-post-microgate\",\"topk_per_row\":128,\"valid_keys\":4,\"max_scratch_case\":{{\"logical_topk\":129,\"capacity_topk\":130}},\"stages\":[\"scores\",\"probabilities\",\"denominators\",\"valid_counts\",\"sparse\",\"sparse_inverse\",\"wo_a\",\"attention_f32\",\"attention_bf16\",\"attention_post\"]}}"
    );
}
