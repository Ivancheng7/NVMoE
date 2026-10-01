use super::*;

pub(crate) fn run_real_v4_full61_b1(
    context: &std::sync::Arc<CudaContext>,
    module: &kernels::LoadedModule,
    stream: &std::sync::Arc<CudaStream>,
    mut preflight: Full61Preflight,
) -> Result<(), Box<dyn Error>> {
    let prompt_text = match std::env::var("COLIBRI_V4_PROMPT_TEXT") {
        Ok(value) => Some(value),
        Err(std::env::VarError::NotPresent) => None,
        Err(error) => return Err(error.into()),
    };
    let chat_user = match std::env::var("COLIBRI_V4_CHAT_USER") {
        Ok(value) => Some(value),
        Err(std::env::VarError::NotPresent) => None,
        Err(error) => return Err(error.into()),
    };
    if chat_input_conflict(
        chat_user.is_some(),
        prompt_text.is_some(),
        std::env::var_os("COLIBRI_V4_TOKEN_IDS").is_some()
            || std::env::var_os("COLIBRI_V4_TOKEN_ID").is_some(),
    ) {
        return Err(
            "COLIBRI_V4_CHAT_USER is mutually exclusive with raw V4 prompt and token ID inputs"
                .into(),
        );
    }
    if prompt_text.is_some()
        && (std::env::var_os("COLIBRI_V4_TOKEN_IDS").is_some()
            || std::env::var_os("COLIBRI_V4_TOKEN_ID").is_some())
    {
        return Err("COLIBRI_V4_PROMPT_TEXT is mutually exclusive with V4 token ID inputs".into());
    }
    let mut tokenizer = None;
    let mut token_ids = if let Some(user) = chat_user.as_deref() {
        let tokenizer_root = v4_model_dir()?;
        let loaded = v4_tokenizer::V4Tokenizer::open(&tokenizer_root)?;
        let prompt = v4_chat::format_single_user(user, v4_chat::ThinkingMode::Chat);
        let ids = loaded.encode_raw(&prompt)?;
        tokenizer = Some(loaded);
        ids
    } else if let Some(text) = prompt_text.as_deref() {
        let tokenizer_root = v4_model_dir()?;
        let loaded = v4_tokenizer::V4Tokenizer::open(&tokenizer_root)?;
        let ids = loaded.encode_raw(text)?;
        tokenizer = Some(loaded);
        ids
    } else {
        match std::env::var("COLIBRI_V4_TOKEN_IDS") {
            Ok(value) => value
                .split(',')
                .map(str::trim)
                .map(str::parse::<i64>)
                .collect::<Result<Vec<_>, _>>()?,
            Err(std::env::VarError::NotPresent) => vec![
                std::env::var("COLIBRI_V4_TOKEN_ID")
                    .map_err(|_| "set COLIBRI_V4_TOKEN_ID or COLIBRI_V4_TOKEN_IDS")?
                    .parse::<i64>()?,
            ],
            Err(error) => return Err(error.into()),
        }
    };
    if token_ids.len() > v4_state::KV_WINDOW {
        return Err(if chat_user.is_some() {
            "COLIBRI_V4_CHAT_USER encodes beyond the existing V4 KV window"
        } else if prompt_text.is_some() {
            "COLIBRI_V4_PROMPT_TEXT encodes beyond the existing V4 KV window"
        } else {
            "COLIBRI_V4_TOKEN_IDS exceeds the existing V4 KV window"
        }
        .into());
    }
    v4_state::validate_ids(&token_ids)?;
    let prompt_len = token_ids.len();
    let greedy_steps = match std::env::var("COLIBRI_V4_GREEDY_STEPS") {
        Ok(value) => value.parse::<usize>()?,
        Err(std::env::VarError::NotPresent) => 0,
        Err(error) => return Err(error.into()),
    };
    let root = v4_model_dir()?;
    let request_started = std::time::Instant::now();
    let max_layers = match std::env::var("COLIBRI_V4_MAX_LAYERS") {
        Ok(value) => value.parse::<usize>()?,
        Err(std::env::VarError::NotPresent) => v4_state::V4_LAYER_COUNT,
        Err(error) => return Err(error.into()),
    };
    if !(1..=v4_state::V4_LAYER_COUNT).contains(&max_layers) {
        return Err("COLIBRI_V4_MAX_LAYERS must be in 1..=61".into());
    }
    if greedy_steps != 0
        && max_layers != v4_state::V4_LAYER_COUNT
        && !optional_one_flag("COLIBRI_V4_TARGETED_ATTENTION_AUDIT")?
    {
        return Err("COLIBRI_V4_GREEDY_STEPS requires COLIBRI_V4_MAX_LAYERS=61".into());
    }
    let critical_path_profile = match std::env::var_os("COLIBRI_V4_CRITICAL_PATH_PROFILE") {
        None => false,
        Some(value) if value == "1" => true,
        Some(_) => {
            return Err("COLIBRI_V4_CRITICAL_PATH_PROFILE must be exactly 1 when set".into());
        }
    };
    if preflight.qd4_bench {
        if token_ids.len() != 1
            || max_layers < 5
            || parse_gib_budget("COLIBRI_V4_DEVICE_FIXED_CACHE_GIB")? != 0
            || parse_gib_budget("COLIBRI_V4_HOST_FIXED_CACHE_GIB")? != 0
        {
            return Err("QD4 benchmark requires one token, layers 0..=4, and M2 cache off".into());
        }
        let path = std::env::var_os("COLIBRI_V4_ROUTED_QD4_RESULT_JSONL")
            .map(std::path::PathBuf::from)
            .ok_or("set COLIBRI_V4_ROUTED_QD4_RESULT_JSONL")?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut result = std::fs::File::create(path)?;
        writeln!(
            result,
            "{}",
            serde_json::json!({
                "event": "run",
                "schema": "colibri-v4-routed-qd4-v2",
                "provenance": benchmark_provenance()?,
                "token_e2e_ns": serde_json::Value::Null,
                "model_revision": v4_state::REVISION,
            })
        )?;
    }
    let qd4_enabled = preflight.qd4_enabled;
    if critical_path_profile
        && (token_ids.as_slice() != [0, 1]
            || max_layers != 5
            || !qd4_enabled
            || !bench_telemetry_enabled())
    {
        return Err(
            "critical-path profile requires token IDs 0,1, max_layers=5, QD4=1, and bench telemetry"
                .into(),
        );
    }
    let sparse_attention_arm = sparse_attention_arm_from_env()?;
    let mut manifests = v4_state::serial_layer_manifests(0)?;
    manifests.truncate(max_layers);
    let mut result = match preflight.result.take() {
        Some(result) => result,
        None => PendingResultFile::reserve(
            root.join("oracles")
                .join(format!("v4_full61_token_{}_pos_0.rust.jsonl", token_ids[0])),
        )?,
    };
    let result_path = result.path().to_owned();
    let max_forward_tokens = greedy_forward_tokens(prompt_len, greedy_steps)?;
    let layer_stationary_prefill = preflight.layer_stationary_prefill;
    let device_short_decode = optional_one_flag("COLIBRI_V4_DEVICE_ONE_TOKEN_DECODE")?;
    let mtp1_draft_p0 = optional_one_flag("COLIBRI_V4_MTP1_DRAFT_P0")?;
    let mtp1_max_scored_transitions = mtp1_draft_p0
        .then(mtp1_draft_max_scored_transitions)
        .transpose()?
        .unwrap_or(0);
    if mtp1_draft_p0
        && (layer_stationary_prefill
            || max_layers != v4_state::V4_LAYER_COUNT
            || qd4_enabled
            || parse_gib_budget("COLIBRI_V4_DEVICE_FIXED_CACHE_GIB")? != 0)
    {
        return Err(
            "COLIBRI_V4_MTP1_DRAFT_P0 requires non-stationary full61, QD4/cache-device-off teacher execution"
                .into(),
        );
    }
    let layer_stationary_expert_reuse = preflight.layer_stationary_expert_reuse;
    let phase_profile = preflight.phase_profile;
    if layer_stationary_prefill {
        if optional_one_flag("COLIBRI_V4_GENERIC_DEVICE_STATIONARY")? {
            if prompt_len == 0 || greedy_steps == 0 || max_layers != v4_state::V4_LAYER_COUNT {
                return Err("generic device stationary requires a nonempty full61 prompt and COLIBRI_V4_GREEDY_STEPS>=1".into());
            }
        } else {
            validate_layer_stationary_prefill(
                prompt_len,
                greedy_steps,
                max_layers,
                true,
                preflight.qd4_bench,
                device_short_decode,
            )?;
        }
        return run_real_v4_layer_stationary_prefill(
            context,
            module,
            stream,
            &root,
            &token_ids,
            prompt_len,
            &manifests,
            chat_user.as_deref(),
            prompt_text.as_deref(),
            tokenizer.as_ref(),
            sparse_attention_arm,
            layer_stationary_expert_reuse,
            qd4_enabled,
            preflight.qd4_immutable_expert_cache,
            phase_profile,
            preflight.fixed_prefetch,
            preflight.certified_argmax_audit,
            device_short_decode,
            greedy_steps,
            result,
        );
    }
    if device_short_decode {
        return Err(
            "COLIBRI_V4_DEVICE_ONE_TOKEN_DECODE=1 enables short device decode and requires layer-stationary prefill".into(),
        );
    }
    let cublas = cublas_exact::CublasExact::load()?;
    let cublas_provenance = cublas.provenance().clone();
    drop(cublas);
    let qd4_device_reserve_bytes = usize::from(qd4_enabled) * V4_QD4_DEVICE_BYTES;
    let qd4_host_reserve_bytes = usize::from(qd4_enabled) * V4_QD4_PINNED_HOST_BYTES;
    let mut cache = FixedLayerCache::from_env(
        &root,
        max_layers,
        qd4_device_reserve_bytes,
        qd4_host_reserve_bytes,
        stream,
    )?;
    let qd4_setup_started = std::time::Instant::now();
    let mut qd4_lanes = if qd4_enabled {
        Some(
            (0..4)
                .map(|_| {
                    let lane_stream = context.new_stream()?;
                    ExpertSlot::new_empty(context, module, &lane_stream)
                })
                .collect::<Result<Vec<_>, Box<dyn Error>>>()?,
        )
    } else {
        None
    };
    let qd4_startup_setup_ns = if qd4_enabled {
        u64::try_from(qd4_setup_started.elapsed().as_nanos()).unwrap_or(u64::MAX)
    } else {
        0
    };
    let mut persistent = v4_state::V4SerialPersistentState::new(max_forward_tokens)?;
    let mut layer_records = Vec::with_capacity(max_layers * max_forward_tokens);
    let mut token_records = Vec::with_capacity(max_forward_tokens);
    let mut token_e2e_ns = Vec::with_capacity(max_forward_tokens);
    let mut last_hidden = Vec::new();
    let final_weights = (greedy_steps != 0)
        .then(|| v4_state::load_final_head(&root))
        .transpose()?;
    let mut greedy_finals = Vec::with_capacity(max_forward_tokens);
    let mut generated_ids = Vec::with_capacity(greedy_steps);
    let mut greedy_stop_reason = None;
    // A draft state is created once for this request and never aliases teacher
    // state.  It stays alive across prompt and greedy positions.
    let mut mtp1_draft_state = mtp1_draft_p0
        .then(|| MtpDraftState::new(stream))
        .transpose()?;
    let mut mtp1_draft_disabled = None::<String>;
    let mut mtp1_scored_transitions = 0_usize;
    let mut mtp1_draft_records = Vec::with_capacity(max_forward_tokens);
    for token_position in 0..max_forward_tokens {
        let token_id = token_ids[token_position];
        let embedding = v4_state::load(&root, &[token_id])?.embedding;
        let mut hc_hidden = Vec::with_capacity(v4_state::HC * v4_state::HIDDEN);
        for _ in 0..v4_state::HC {
            hc_hidden.extend_from_slice(&embedding);
        }
        let token_started = std::time::Instant::now();
        let mut current_records = Vec::with_capacity(max_layers);
        let (hidden, telemetry) = v4_state::run_serial_layer_transactions(
            &mut persistent,
            hc_hidden,
            &manifests,
            |manifest, hidden, layer_state| {
                let timeline_origin = critical_path_profile
                    .then_some(())
                    .filter(|_| token_position == 1 && matches!(manifest.layer, 3 | 4))
                    .map(|_| std::time::Instant::now());
                let hash_started = std::time::Instant::now();
                let hash_ids = load_layer_hash_ids(&root, manifest.layer, &[token_id])?;
                let dynamic_hash_read_ns =
                    u64::try_from(hash_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
                let (run, fixed) = cache.access(
                    &root,
                    manifest.layer,
                    &[token_id],
                    module,
                    stream,
                    |runtime| {
                        let qd4 = qd4_lanes.as_deref_mut();
                        step_real_layer_rows(
                            context,
                            module,
                            stream,
                            None,
                            runtime,
                            qd4,
                            None,
                            None,
                            &hash_ids,
                            hidden,
                            &[token_id],
                            token_position,
                            Some(layer_state),
                            None,
                            None,
                            false,
                            timeline_origin,
                            false,
                            None,
                            None,
                        None,
                        None, // attention device chain (not this path)
                        )
                    },
                )?;
                if run.routes.ids.len() != v4_router::V4_TOPK {
                    return Err(format!(
                        "layer {} router produced incomplete B=1 routes",
                        manifest.layer
                    )
                    .into());
                }
                let main = layer_state.compressor();
                let indexer = layer_state.indexer();
                let mut record = serde_json::json!({
                    "event": "layer",
                    "token_position": token_position,
                    "token_id": token_id,
                    "layer": manifest.layer,
                    "compression_ratio": manifest.compression_ratio.get(),
                    "hidden_bf16_digest": format!("{:016x}", fnv1a_u16(&run.output)),
                    "kv_cache_bf16_digest": format!("{:016x}", fnv1a_u16(layer_state.kv_bf16())),
                    "main_compressed_kv_bf16_digest": format!("{:016x}", fnv1a_u16(layer_state.compressed_kv_bf16())),
                    "main_compressor_kv_state_f32_digest": format!("{:016x}", fnv1a_f32(main.kv())),
                    "main_compressor_score_state_f32_digest": format!("{:016x}", fnv1a_f32(main.score())),
                    "indexer_compressed_kv_bf16_digest": format!("{:016x}", fnv1a_u16(layer_state.indexer_kv_bf16())),
                    "indexer_compressor_kv_state_f32_digest": indexer.map(|state| format!("{:016x}", fnv1a_f32(state.kv()))),
                    "indexer_compressor_score_state_f32_digest": indexer.map(|state| format!("{:016x}", fnv1a_f32(state.score()))),
                    "route_ids": &run.routes.ids,
                    "route_weights_f32": &run.routes.weights,
                    "attention_topk_digest": format!("{:016x}", run.attention_topk_digest),
                    "attention_topk_per_row": run.attention_topk_per_row,
                    "indexer_topk_gpu_ns": run.indexer_topk_gpu_ns,
                    "source_range_bytes": fixed.fixed_source_bytes,
                });
                if bench_telemetry_enabled() {
                    record
                        .as_object_mut()
                        .ok_or("layer telemetry record is not an object")?
                        .insert(
                            "bench".into(),
                            serde_json::json!({
                                "fixed_cache_status": fixed.cache_status,
                                "fixed_read_ns": if matches!(fixed.cache_status, "raw_device_hit" | "raw_host_hit") { Some(0_u64) } else { None },
                                "fixed_alloc_ns": if fixed.cache_status == "raw_device_hit" { Some(0_u64) } else { None },
                                "raw_decode_alloc_ns": if matches!(fixed.cache_status, "raw_device_hit" | "raw_host_hit") { Some(0_u64) } else { None },
                                "fixed_read_alloc_ns": fixed.fixed_read_alloc_ns,
                                "fixed_decode_ns": fixed.fixed_decode_ns,
                                "fixed_h2d_ns": fixed.fixed_h2d_ns,
                                "fixed_h2d_bytes": fixed.fixed_h2d_bytes,
                                "raw_read_ns": fixed.raw_read_ns,
                                "raw_source_bytes": fixed.raw_source_bytes,
                                "raw_h2d_ns": fixed.raw_h2d_ns,
                                "raw_h2d_bytes": fixed.raw_h2d_bytes,
                                "dynamic_hash_read_ns": dynamic_hash_read_ns,
                                "attention_ns": run.stage.attention_ns,
                                "router_ns": run.stage.router_ns,
                                "router_mhc_ns": run.stage.router_mhc_ns,
                                "router_mhc_input_alloc_h2d_sync_ns": run.stage.router_mhc_input_alloc_h2d_sync_ns,
                                "router_mhc_workspace_alloc_sync_ns": run.stage.router_mhc_workspace_alloc_sync_ns,
                                "router_mhc_workspace_init_gpu_ns": run.stage.router_mhc_workspace_init_gpu_ns,
                                "router_mhc_mixing_linear_sync_ns": run.stage.router_mhc_mixing_linear_sync_ns,
                                "router_mhc_mixing_linear_gpu_ns": run.stage.router_mhc_mixing_linear_gpu_ns,
                                "router_mhc_sinkhorn_normalize_collapse_sync_ns": run.stage.router_mhc_sinkhorn_normalize_collapse_sync_ns,
                                "router_mhc_sinkhorn_normalize_collapse_gpu_ns": run.stage.router_mhc_sinkhorn_normalize_collapse_gpu_ns,
                                "router_mhc_output_d2h_sync_ns": run.stage.router_mhc_output_d2h_sync_ns,
                                "router_mhc_other_ns": run.stage.router_mhc_other_ns,
                                "router_host_prep_ns": run.stage.router_host_prep_ns,
                                "router_alloc_h2d_sync_ns": run.stage.router_alloc_h2d_sync_ns,
                                "router_workspace_alloc_ns": run.stage.router_workspace_alloc_ns,
                                "router_kernel_select_sync_ns": run.stage.router_kernel_select_sync_ns,
                                "router_kernel_select_gpu_ns": run.stage.router_kernel_select_gpu_ns,
                                "router_d2h_sync_ns": run.stage.router_d2h_sync_ns,
                                "router_other_ns": run.stage.router_other_ns,
                                "shared_ns": run.stage.shared_ns,
                                "shared_main_ns": run.stage.shared_main_ns,
                                "shared_read_workers": run.stage.shared_read_workers,
                                "shared_read_worker_sum_ns": run.stage.shared_read_worker_sum_ns,
                                "shared_read_worker_max_ns": run.stage.shared_read_worker_max_ns,
                                "routed_ns": run.stage.routed_ns,
                                "routed_read_ns": run.stage.routed_read_ns,
                                "routed_alloc_ns": run.stage.routed_alloc_ns,
                                "routed_setup_ns": run.stage.routed_setup_ns,
                                "routed_h2d_ns": run.stage.routed_h2d_ns,
                                "routed_kernel_ns": run.stage.routed_kernel_ns,
                                "commit_ns": run.stage.commit_ns,
                                "routed_source_bytes": run.stage.routed_source_bytes,
                                "routed_h2d_bytes": run.stage.routed_h2d_bytes,
                            }),
                        );
                }
                if let Some(timeline) = run.critical_path_timeline {
                    record
                        .as_object_mut()
                        .ok_or("layer timeline record is not an object")?
                        .insert("critical_path_timeline".into(), timeline.as_json());
                }
                current_records.push(record);
                Ok(v4_state::TransactionalLayerOutput {
                    next_hidden: run.output,
                    source_range_bytes: manifest.contract_range_bytes,
                    indexer_topk_cpu_ns: 0,
                })
            },
        )?;
        if telemetry.len() != max_layers {
            return Err("V4 B=1 prefix committed fewer layers than requested".into());
        }
        for (record, telemetry) in current_records.iter_mut().zip(&telemetry) {
            let object = record
                .as_object_mut()
                .ok_or("full61 layer telemetry record is not an object")?;
            object.insert(
                "manifest_contract_range_bytes".into(),
                telemetry.source_range_bytes.into(),
            );
            object.insert("layer_elapsed_ns".into(), telemetry.layer_elapsed_ns.into());
            object.insert("kv_rows_before".into(), telemetry.kv_rows_before.into());
            object.insert("kv_rows_after".into(), telemetry.kv_rows_after.into());
            object.insert(
                "indexer_topk_cpu_ns".into(),
                telemetry.indexer_topk_cpu_ns.into(),
            );
            if let Some(timeline) = object
                .get_mut("critical_path_timeline")
                .and_then(serde_json::Value::as_object_mut)
            {
                let output_ready_ns = timeline
                    .get("output_ready_ns")
                    .and_then(serde_json::Value::as_u64)
                    .ok_or("critical-path timeline omitted output ready timestamp")?;
                timeline.insert(
                    "outer_transaction_commit_and_unprofiled_residual_ns".into(),
                    telemetry
                        .layer_elapsed_ns
                        .saturating_sub(output_ready_ns)
                        .into(),
                );
                timeline.insert(
                    "outer_layer_elapsed_ns".into(),
                    telemetry.layer_elapsed_ns.into(),
                );
            }
        }
        let elapsed_ns = u64::try_from(token_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        token_e2e_ns.push(elapsed_ns);
        token_records.push(serde_json::json!({
            "event": "token",
            "token_position": token_position,
            "token_id": token_id,
            "e2e_ns": elapsed_ns,
            "hidden_bf16_digest": format!("{:016x}", fnv1a_u16(&hidden)),
            "device_cache_resident_bytes": cache.device_resident_bytes,
            "host_cache_resident_bytes": cache.host_resident_bytes,
            "device_cache_hits": cache.device_hits,
            "host_cache_hits": cache.host_hits,
            "cache_misses": cache.misses,
        }));
        layer_records.extend(current_records);
        last_hidden = hidden;
        let mtp1_draft = if mtp1_draft_p0 {
            let teacher_digest_before = fnv1a_u16(&last_hidden);
            let draft = match mtp1_draft_state.as_mut() {
                Some(state) if mtp1_requires_full_candidate(prompt_len, token_position) => {
                    MtpDraftStep::Candidate(step_mtp1_draft_candidate(
                        context,
                        module,
                        stream,
                        &root,
                        final_weights
                            .as_ref()
                            .ok_or("MTP1 draft candidate has no request-scoped final head")?,
                        None,
                        Some(MTP1_DRAFT_EXTRA_CAPACITY_BYTES),
                        &last_hidden,
                        token_id,
                        token_position,
                        state,
                    ))
                }
                Some(state) => MtpDraftStep::Prefix(advance_mtp1_draft_state(
                    module,
                    stream,
                    &root,
                    &last_hidden,
                    token_id,
                    token_position,
                    state,
                )),
                None => MtpDraftStep::Candidate(Err(std::io::Error::other(
                    mtp1_draft_disabled
                        .as_deref()
                        .unwrap_or("MTP1 draft state is unavailable"),
                )
                .into())),
            };
            let teacher_digest_after = fnv1a_u16(&last_hidden);
            if teacher_digest_after != teacher_digest_before {
                return Err("MTP1 draft mutated teacher hidden state".into());
            }
            Some((teacher_digest_before, teacher_digest_after, draft))
        } else {
            None
        };
        if b1_needs_final_head(prompt_len, token_position) {
            if let Some(weights) = &final_weights {
                let cpu = v4_state::final_checkpoint_head_cpu_timed(weights, &last_hidden, 1)?;
                let next_id = i64::try_from(cpu.output.argmax[0])?;
                match greedy_argmax_decision(
                    prompt_len,
                    token_position,
                    generated_ids.len(),
                    greedy_steps,
                    next_id,
                ) {
                    GreedyDecision::IgnorePrompt => {}
                    GreedyDecision::EmitAndContinue => {
                        generated_ids.push(next_id);
                        token_ids.push(next_id);
                    }
                    GreedyDecision::EmitAndStop(reason) => {
                        generated_ids.push(next_id);
                        token_ids.push(next_id);
                        greedy_stop_reason = Some(reason);
                    }
                }
                greedy_finals.push(serde_json::json!({
                "event": "final",
                "position": token_position,
                "input_id": token_id,
                "input_kind": if token_position < prompt_len { "prompt" } else { "generated" },
                "state": "V4SerialPersistentState",
                "head": "canonical_cpu_streamed_bf16",
                "collapsed_bf16_digest": format!("{:016x}", fnv1a_u16(&cpu.output.collapsed_bf16)),
                "normalized_bf16_digest": format!("{:016x}", fnv1a_u16(&cpu.output.normalized_bf16)),
                "logits_f32_digest": format!("{:016x}", fnv1a_f32(&cpu.output.logits_f32)),
                "argmax": cpu.output.argmax[0],
                "status": "PASS",
            }));
            }
        }
        if let Some((teacher_digest_before, teacher_digest_after, draft)) = mtp1_draft {
            let target_position = token_position + 1;
            let target_id = token_ids.get(target_position).copied();
            let (alignment, scored, exclusion) = match target_id {
                None => ("target_unavailable", false, Some("no_next_teacher_token")),
                Some(_) if target_position < prompt_len => {
                    ("prompt_forced", false, Some("prompt_forced_transition"))
                }
                Some(_) if token_position + 1 == prompt_len => {
                    if mtp1_scored_transitions < mtp1_max_scored_transitions {
                        ("last_prompt_to_first_greedy", true, None)
                    } else {
                        (
                            "last_prompt_to_first_greedy",
                            false,
                            Some("per_prompt_score_cap"),
                        )
                    }
                }
                Some(_) if mtp1_scored_transitions < mtp1_max_scored_transitions => {
                    ("greedy_to_greedy", true, None)
                }
                Some(_) => ("greedy_to_greedy", false, Some("per_prompt_score_cap")),
            };
            match draft {
                MtpDraftStep::Prefix(Ok(draft)) => {
                    if draft.persistent_bytes != MTP1_DRAFT_PERSISTENT_BYTES {
                        return Err("MTP1 draft prefix capacity assertion failed".into());
                    }
                    mtp1_draft_records.push(serde_json::json!({
                        "event": "mtp1_draft",
                        "status": "DRAFT_STATE_ADVANCED",
                        "candidate_pos": token_position,
                        "target_pos": target_position,
                        "input_id": token_id,
                        "candidate": serde_json::Value::Null,
                        "target_teacher_token": target_id,
                        "alignment": alignment,
                        "scored": false,
                        "match": serde_json::Value::Null,
                        "exclusion": "prompt_forced_transition",
                        "mtp_kv_persistent_bytes": draft.persistent_bytes,
                        "mtp_kv_position": draft.kv_position,
                        "mtp_kv_digest": format!("{:016x}", draft.kv_digest),
                        "teacher_digest_before": format!("{teacher_digest_before:016x}"),
                        "teacher_digest_after": format!("{teacher_digest_after:016x}"),
                    }));
                }
                MtpDraftStep::Candidate(Ok(draft)) => {
                    if draft.persistent_bytes != MTP1_DRAFT_PERSISTENT_BYTES
                        || draft.persistent_bytes + draft.transient_expert_peak_bytes
                            > MTP1_DRAFT_EXTRA_CAPACITY_BYTES
                    {
                        return Err("MTP1 draft capacity assertion failed".into());
                    }
                    let matched = target_id.map(|target| draft.candidate == target);
                    if scored {
                        mtp1_scored_transitions += 1;
                    }
                    mtp1_draft_records.push(serde_json::json!({
                        "event": "mtp1_draft",
                        "status": "DRAFT_ALIGNED",
                        "candidate_pos": token_position,
                        "target_pos": target_position,
                        "input_id": token_id,
                        "candidate": draft.candidate,
                        "target_teacher_token": target_id,
                        "alignment": alignment,
                        "scored": scored,
                        "match": matched,
                        "exclusion": exclusion,
                        "scored_transition_index": scored.then_some(mtp1_scored_transitions),
                        "route_ids": draft.routes.ids,
                        "route_weights_f32": draft.routes.weights,
                        "source_bytes": draft.source_bytes,
                        "source_read_ns": draft.source_read_ns,
                        "h2d_ns": draft.h2d_ns,
                        "h2d_bytes": draft.h2d_bytes,
                        "kernel_ns": draft.kernel_ns,
                        "head_read_ns": draft.head_read_ns,
                        "head_h2d_ns": draft.head_h2d_ns,
                        "head_kernel_ns": draft.head_kernel_ns,
                        "head_d2h_ns": draft.head_d2h_ns,
                        "head_wall_ns": draft.head_wall_ns,
                        "total_ns": draft.total_ns,
                        "mtp_kv_persistent_bytes": draft.persistent_bytes,
                        "mtp_kv_position": draft.kv_position,
                        "mtp_kv_digest": format!("{:016x}", draft.kv_digest),
                        "transient_expert_peak_bytes": draft.transient_expert_peak_bytes,
                        "extra_capacity_limit_bytes": MTP1_DRAFT_EXTRA_CAPACITY_BYTES,
                        "expert_lifecycle": "single transient ExpertSlot; expert IDs ascending; drop before next",
                        "teacher_digest_before": format!("{teacher_digest_before:016x}"),
                        "teacher_digest_after": format!("{teacher_digest_after:016x}"),
                    }));
                }
                MtpDraftStep::Prefix(Err(error)) | MtpDraftStep::Candidate(Err(error)) => {
                    let reason = error.to_string();
                    mtp1_draft_disabled = Some(reason.clone());
                    mtp1_draft_state = None;
                    mtp1_draft_records.push(serde_json::json!({
                        "event": "mtp1_draft",
                        "status": "DRAFT_FAIL",
                        "candidate_pos": token_position,
                        "target_pos": target_position,
                        "input_id": token_id,
                        "target_teacher_token": target_id,
                        "alignment": alignment,
                        "scored": false,
                        "match": serde_json::Value::Null,
                        "exclusion": "draft_failed",
                        "reason": reason,
                        "teacher_digest_before": format!("{teacher_digest_before:016x}"),
                        "teacher_digest_after": format!("{teacher_digest_after:016x}"),
                    }));
                }
            }
        }
        if greedy_stop_reason.is_some() {
            break;
        }
    }
    let forward_input_len = token_records.len();
    let greedy_stop_reason = if greedy_steps == 0 {
        None
    } else {
        Some(
            match greedy_stop_reason.ok_or("greedy decoding ended without a stop reason")? {
                GreedyStopReason::Eos => "eos",
                GreedyStopReason::MaxNewTokens => "max_new_tokens",
            },
        )
    };
    let mut request_e2e_ns = None;
    let mut final_head_header = serde_json::Value::Null;
    let mut final_record = None;
    let mut argmax = None;
    if max_layers == v4_state::V4_LAYER_COUNT && greedy_steps == 0 {
        let load_started = std::time::Instant::now();
        let final_weights = v4_state::load_final_head(&root)?;
        let fixed_read_ns = u64::try_from(load_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        let cpu = v4_state::final_checkpoint_head_cpu_timed(&final_weights, &last_hidden, 1)?;
        let cpu_final_wall_ns = fixed_read_ns.saturating_add(cpu.wall_ns);
        request_e2e_ns =
            Some(u64::try_from(request_started.elapsed().as_nanos()).unwrap_or(u64::MAX));
        final_head_header = serde_json::json!({
            "mode": "canonical_cpu_streamed_bf16",
            "fixed_read_ns": fixed_read_ns,
            "cpu_oracle": {
                "wall_ns": cpu_final_wall_ns,
                "streamed_weight_read_ns": cpu.read_ns,
            },
            "h2d_ns": 0,
            "kernel_ns": 0,
            "stage1_gpu_status": "KILL",
        });
        argmax = Some(cpu.output.argmax[0]);
        final_record = Some(serde_json::json!({
            "event": "final",
            "collapsed_bf16_digest": format!("{:016x}", fnv1a_u16(&cpu.output.collapsed_bf16)),
            "normalized_bf16_digest": format!("{:016x}", fnv1a_u16(&cpu.output.normalized_bf16)),
            "logits_f32_digest": format!("{:016x}", fnv1a_f32(&cpu.output.logits_f32)),
            "argmax": cpu.output.argmax[0],
            "status": "PASS",
        }));
    } else if greedy_steps != 0 {
        request_e2e_ns = Some(token_e2e_ns.iter().copied().sum());
        final_head_header = serde_json::json!({
            "mode": "canonical_cpu_streamed_bf16",
            "invocation": "after_each_forward",
            "weight_shard": "model-00063-of-00064.safetensors",
            "output": "per-forward final records"
        });
    }
    let mut provenance = benchmark_provenance()?;
    provenance
        .as_object_mut()
        .ok_or("benchmark provenance is not an object")?
        .insert(
            "cublas_exact".into(),
            serde_json::json!({
                "resolved_path": cublas_provenance.resolved_path,
                "sha256": cublas_provenance.actual_sha256,
                "version": cublas_provenance.version,
                "dladdr_path": cublas_provenance.dladdr_path,
                "proc_maps_path": cublas_provenance.maps_path,
            }),
        );
    let mut run_header = serde_json::json!({
        "event": "run",
        "schema": if greedy_steps != 0 {
            "colibri-v4-greedy-id-trace-v1"
        } else if token_ids.len() == 1 && cache.plan.is_empty() {
            "colibri-v4-full61-trace-v1"
        } else {
            "colibri-v4-fixed-cache-trace-v1"
        },
        "implementation": "rust-cuda",
        "provenance": provenance,
        "token_e2e_ns": token_e2e_ns,
        "request_e2e_ns": request_e2e_ns,
        "request_e2e_definition": "token input through canonical CPU-streamed final logits and argmax",
        "final_head": final_head_header,
        "model_revision": v4_state::REVISION,
        "token_id": if token_ids.len() == 1 { Some(token_ids[0]) } else { None },
        "token_ids": &token_ids[..forward_input_len],
        "emitted_token_ids": &token_ids,
        "prompt_ids": &token_ids[..prompt_len],
        "greedy_steps": greedy_steps,
        "generated_ids": &generated_ids,
        "generation_mode": if greedy_steps == 0 { "teacher_forced" } else { "id_only_greedy_argmax" },
        "start_pos": 0,
        "layers": v4_state::V4_LAYER_COUNT,
        "executed_layers": max_layers,
        "model_dir": root,
        "device_fixed_cache_budget_bytes": cache.device_budget_bytes,
        "host_fixed_cache_budget_bytes": cache.host_budget_bytes,
        "fixed_read_alloc_split": "unavailable",
        "raw_fixed_cache": cache.raw_mode,
        "raw_fixed_scratch_bytes": cache.raw_scratch.as_ref().map_or(0, DeviceBuffer::num_bytes),
        "sparse_attention_arm": match sparse_attention_arm {
            SparseAttentionArm::Reference => "reference:sparse_attn_v4_bf16",
            SparseAttentionArm::ScoreReuse => "score_reuse:sparse_attn_v4_scores+probabilities+values",
            SparseAttentionArm::Sm89Crossover => {
                "sm89_crossover:K<4=reference,K>=4=score_reuse"
            }
        },
        "routed_qd4": qd4_enabled,
        "routed_qd4_startup_setup_ns": qd4_startup_setup_ns,
        "routed_qd4_pinned_host_bytes": qd4_host_reserve_bytes,
        "routed_qd4_device_bytes": qd4_device_reserve_bytes,
        "mtp1_draft_acceptance": {
            "enabled": mtp1_draft_p0,
            "persistent_state_scope": "single_request",
            "candidate_target_alignment": "candidate_pos predicts teacher token at candidate_pos + 1",
            "prompt_forced_scored": false,
            "max_scored_transitions_per_prompt": mtp1_max_scored_transitions,
            "scored_transitions": mtp1_scored_transitions,
        },
        "mtp1_draft_p0": &mtp1_draft_records,
    });
    if prompt_text.is_some() || chat_user.is_some() {
        let tokenizer = tokenizer
            .as_ref()
            .ok_or("raw V4 text prompt has no loaded tokenizer")?;
        let generated_for_decode = generated_ids
            .iter()
            .copied()
            .filter(|&id| id != V4_EOS_TOKEN_ID)
            .collect::<Vec<_>>();
        let cumulative_for_decode = token_ids
            .iter()
            .copied()
            .filter(|&id| id != V4_EOS_TOKEN_ID)
            .collect::<Vec<_>>();
        let header = run_header
            .as_object_mut()
            .ok_or("V4 B1 run header is not an object")?;
        header.insert(
            "encoded_prompt_ids".into(),
            serde_json::json!(&token_ids[..prompt_len]),
        );
        header.insert(
            "generated_only_decoded_text".into(),
            serde_json::json!(tokenizer.decode(&generated_for_decode)?),
        );
        header.insert(
            "cumulative_decoded_text".into(),
            serde_json::json!(tokenizer.decode(&cumulative_for_decode)?),
        );
        if let Some(text) = prompt_text.as_deref() {
            header.insert("prompt_text".into(), serde_json::json!(text));
            header.insert(
                "tokenizer".into(),
                serde_json::json!({
                    "mode": "raw",
                    "chat_template": false,
                    "add_special_tokens": false,
                }),
            );
        } else {
            header.insert(
                "chat_user".into(),
                serde_json::json!(chat_user.as_deref().ok_or("missing V4 chat user")?),
            );
            header.insert(
                "prompt_encoding".into(),
                serde_json::json!({
                    "name": "official_encoding_dsv4",
                    "mode": "single_user",
                    "thinking_mode": "chat",
                    "jinja_template": false,
                    "tools": false,
                    "multiturn": false,
                    "add_special_tokens": false,
                }),
            );
        }
    }
    if let Some(stop_reason) = greedy_stop_reason {
        let header = run_header
            .as_object_mut()
            .ok_or("V4 B1 run header is not an object")?;
        header.insert("stop_reason".into(), serde_json::json!(stop_reason));
        header.insert("eos_token_id".into(), serde_json::json!(V4_EOS_TOKEN_ID));
        header.insert("max_new_tokens".into(), serde_json::json!(greedy_steps));
        header.insert(
            "actual_generated_count".into(),
            serde_json::json!(generated_ids.len()),
        );
    }
    writeln!(result.file_mut(), "{run_header}")?;
    for (index, plan) in cache.plan.iter().enumerate() {
        let tier = if cache.device_selected[index] {
            if cache.raw_mode {
                "raw_device"
            } else {
                "device"
            }
        } else if cache.host_selected[index] {
            if cache.raw_mode { "raw_host" } else { "host" }
        } else {
            "uncached"
        };
        for tensor in &plan.tensors {
            writeln!(
                result.file_mut(),
                "{}",
                serde_json::json!({
                    "event": "fixed_cache_manifest",
                    "revision": v4_state::REVISION,
                    "layer": plan.layer,
                    "tensor": tensor.name,
                    "dtype": tensor.dtype,
                    "shape": tensor.shape,
                    "source_bytes": tensor.source_bytes,
                    "decoded_bytes": tensor.decoded_bytes,
                    "expected_device_saved_ns_per_byte": plan.device_saved_ns as f64 / plan.device_bytes as f64,
                    "expected_host_saved_ns_per_byte": plan.host_saved_ns as f64 / plan.host_bytes as f64,
                    "tier": tier,
                })
            )?;
        }
    }
    let mut token_record = token_records.iter();
    for (position, record) in layer_records.iter().enumerate() {
        writeln!(result.file_mut(), "{record}")?;
        if token_ids.len() > 1 && (position + 1).is_multiple_of(max_layers) {
            writeln!(
                result.file_mut(),
                "{}",
                token_record
                    .next()
                    .ok_or("missing token telemetry record")?
            )?;
        }
    }
    for final_record in greedy_finals {
        writeln!(result.file_mut(), "{final_record}")?;
    }
    if let Some(final_record) = final_record {
        writeln!(result.file_mut(), "{final_record}")?;
    }
    result.commit()?;
    if let Some(stop_reason) = greedy_stop_reason {
        if let Some(text) = prompt_text.as_deref() {
            println!(
                "{{\"status\":\"PASS\",\"mode\":\"v4-b1-prefix\",\"tokenizer\":\"raw\",\"chat_template\":false,\"prompt_text\":{},\"token_ids\":{},\"generated_ids\":{},\"layers\":{},\"argmax\":{},\"stop_reason\":{},\"eos_token_id\":{},\"max_new_tokens\":{},\"actual_generated_count\":{},\"result_jsonl\":{}}}",
                serde_json::to_string(text)?,
                serde_json::to_string(&token_ids)?,
                serde_json::to_string(&generated_ids)?,
                max_layers,
                serde_json::to_string(&generated_ids.last())?,
                serde_json::to_string(stop_reason)?,
                V4_EOS_TOKEN_ID,
                greedy_steps,
                generated_ids.len(),
                serde_json::to_string(&result_path)?
            );
        } else if let Some(user) = chat_user.as_deref() {
            println!(
                "{{\"status\":\"PASS\",\"mode\":\"v4-b1-prefix\",\"prompt_encoding\":\"official_encoding_dsv4\",\"chat_mode\":\"single_user\",\"jinja_template\":false,\"tools\":false,\"multiturn\":false,\"chat_user\":{},\"token_ids\":{},\"generated_ids\":{},\"layers\":{},\"argmax\":{},\"stop_reason\":{},\"eos_token_id\":{},\"max_new_tokens\":{},\"actual_generated_count\":{},\"result_jsonl\":{}}}",
                serde_json::to_string(user)?,
                serde_json::to_string(&token_ids)?,
                serde_json::to_string(&generated_ids)?,
                max_layers,
                serde_json::to_string(&generated_ids.last())?,
                serde_json::to_string(stop_reason)?,
                V4_EOS_TOKEN_ID,
                greedy_steps,
                generated_ids.len(),
                serde_json::to_string(&result_path)?
            );
        } else {
            println!(
                "{{\"status\":\"PASS\",\"mode\":\"v4-b1-prefix\",\"token_ids\":{},\"generated_ids\":{},\"layers\":{},\"argmax\":{},\"stop_reason\":{},\"eos_token_id\":{},\"max_new_tokens\":{},\"actual_generated_count\":{},\"result_jsonl\":{}}}",
                serde_json::to_string(&token_ids)?,
                serde_json::to_string(&generated_ids)?,
                max_layers,
                serde_json::to_string(&generated_ids.last())?,
                serde_json::to_string(stop_reason)?,
                V4_EOS_TOKEN_ID,
                greedy_steps,
                generated_ids.len(),
                serde_json::to_string(&result_path)?
            );
        }
    } else if let Some(text) = prompt_text {
        println!(
            "{{\"status\":\"PASS\",\"mode\":\"v4-b1-prefix\",\"tokenizer\":\"raw\",\"chat_template\":false,\"prompt_text\":{},\"token_ids\":{},\"generated_ids\":{},\"layers\":{},\"argmax\":{},\"result_jsonl\":{}}}",
            serde_json::to_string(&text)?,
            serde_json::to_string(&token_ids)?,
            serde_json::to_string(&generated_ids)?,
            max_layers,
            serde_json::to_string(&argmax)?,
            serde_json::to_string(&result_path)?
        );
    } else if let Some(user) = chat_user.as_deref() {
        println!(
            "{{\"status\":\"PASS\",\"mode\":\"v4-b1-prefix\",\"prompt_encoding\":\"official_encoding_dsv4\",\"chat_mode\":\"single_user\",\"jinja_template\":false,\"tools\":false,\"multiturn\":false,\"chat_user\":{},\"token_ids\":{},\"generated_ids\":{},\"layers\":{},\"argmax\":{},\"result_jsonl\":{}}}",
            serde_json::to_string(user)?,
            serde_json::to_string(&token_ids)?,
            serde_json::to_string(&generated_ids)?,
            max_layers,
            serde_json::to_string(&argmax)?,
            serde_json::to_string(&result_path)?
        );
    } else {
        println!(
            "{{\"status\":\"PASS\",\"mode\":\"v4-b1-prefix\",\"token_ids\":{},\"generated_ids\":{},\"layers\":{},\"argmax\":{},\"result_jsonl\":{}}}",
            serde_json::to_string(&token_ids)?,
            serde_json::to_string(&generated_ids)?,
            max_layers,
            serde_json::to_string(&argmax)?,
            serde_json::to_string(&result_path)?
        );
    }
    Ok(())
}
