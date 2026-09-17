use super::*;

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_real_v4_layer_stationary_prefill(
    context: &std::sync::Arc<CudaContext>,
    module: &kernels::LoadedModule,
    stream: &std::sync::Arc<CudaStream>,
    root: &std::path::Path,
    token_ids: &[i64],
    prompt_len: usize,
    manifests: &[v4_state::LayerExecutionManifest],
    chat_user: Option<&str>,
    prompt_text: Option<&str>,
    tokenizer: Option<&v4_tokenizer::V4Tokenizer>,
    sparse_attention_arm: SparseAttentionArm,
    expert_slot_reuse: bool,
    qd4_enabled: bool,
    qd4_immutable_expert_cache: bool,
    phase_profile: bool,
    fixed_prefetch: bool,
    certified_argmax_audit: bool,
    device_short_decode: bool,
    greedy_steps: usize,
    mut result: PendingResultFile,
) -> Result<(), Box<dyn Error>> {
    let targeted_attention_audit = optional_one_flag("COLIBRI_V4_TARGETED_ATTENTION_AUDIT")?;
    if token_ids.len() != prompt_len
        || if targeted_attention_audit {
            manifests.len() != 3
        } else {
            manifests.len() != v4_state::V4_LAYER_COUNT
        }
    {
        return Err(
            "layer-stationary prefill received a non-prompt token or partial manifest".into(),
        );
    }
    if expert_slot_reuse && qd4_enabled {
        return Err("layer-stationary expert reuse and QD4 lanes must not coexist".into());
    }
    if qd4_immutable_expert_cache && !qd4_enabled {
        return Err("QD4 immutable expert cache requires persistent QD4 lanes".into());
    }
    let device_attention = optional_one_flag("COLIBRI_V4_DEVICE_ATTENTION_PRODUCTION")?;
    let generic_device_stationary = optional_one_flag("COLIBRI_V4_GENERIC_DEVICE_STATIONARY")?;
    let stationary_mtp1_draft = optional_one_flag("COLIBRI_V4_STATIONARY_MTP1_DRAFT")?;
    let generic_device_state_audit = optional_one_flag("COLIBRI_V4_GENERIC_DEVICE_STATE_AUDIT")?;
    let device_attention_audit = optional_one_flag("COLIBRI_V4_DEVICE_ATTENTION_AUDIT")?;
    if device_attention_audit && !device_attention {
        return Err(
            "COLIBRI_V4_DEVICE_ATTENTION_AUDIT=1 requires COLIBRI_V4_DEVICE_ATTENTION_PRODUCTION=1"
                .into(),
        );
    }
    if generic_device_stationary
        && (device_attention || device_attention_audit || device_short_decode || greedy_steps == 0)
    {
        return Err("generic device stationary requires full61 with device attention, audit, and short decode off plus greedy_steps >= 1".into());
    }
    if generic_device_state_audit && !generic_device_stationary {
        return Err(
            "generic device state audit requires COLIBRI_V4_GENERIC_DEVICE_STATIONARY=1".into(),
        );
    }
    stationary_mtp1_enabled(generic_device_stationary, stationary_mtp1_draft)?;
    if device_short_decode && !device_attention {
        return Err("short device decode requires device-attention production".into());
    }
    let l0_cutpoint = optional_one_flag("COLIBRI_V4_L0_POS0_CUTPOINT")?;
    let raw_wo_a_gpu_decode = raw_wo_a_gpu_decode_enabled()?;
    let fixed_prefetch_wo_a_arm = fixed_prefetch_wo_a_decode_arm(
        fixed_prefetch,
        device_attention || generic_device_stationary,
        raw_wo_a_gpu_decode,
    );
    let fixed_prefetch_raw_wo_a = fixed_prefetch_wo_a_arm == "raw_fp8_gpu_decode";
    let pinned_fixed_pipeline = optional_one_flag("COLIBRI_V4_PINNED_FIXED_PIPELINE")?;
    let bounded_readahead = pinned_fixed_pipeline
        && cfg!(target_os = "linux")
        && parse_default_true_zero_one_flag(
            "COLIBRI_V4_BOUNDED_READAHEAD",
            std::env::var_os("COLIBRI_V4_BOUNDED_READAHEAD").as_deref(),
        )?;
    if device_short_decode && !fixed_prefetch_raw_wo_a {
        return Err("short device decode requires raw wo_a fixed prefetch".into());
    }
    if pinned_fixed_pipeline
        && (!(device_short_decode || generic_device_stationary) || !fixed_prefetch_raw_wo_a)
    {
        return Err(
            "pinned fixed pipeline requires short device decode and raw wo_a fixed prefetch".into(),
        );
    }
    if generic_device_stationary
        && (!fixed_prefetch || !fixed_prefetch_raw_wo_a || !pinned_fixed_pipeline)
    {
        return Err("generic device stationary requires raw wo_a fixed prefetch and the existing pinned two-slot worker".into());
    }
    if certified_argmax_audit
        && (targeted_attention_audit
            || !device_attention
            || !device_attention_audit
            || !device_short_decode
            || !pinned_fixed_pipeline
            || prompt_len != 5
            || greedy_steps != 3)
    {
        return Err(
            "certified argmax audit requires the D-exact path: full61 device-attention production plus independent audit, pinned fixed pipeline, prompt positions 0..6, and greedy_steps=3"
                .into(),
        );
    }
    if device_short_decode && phase_profile && device_attention_audit {
        return Err("short device phase profile requires production non-audit attention".into());
    }
    if targeted_attention_audit
        && (chat_user != Some("hi")
            || prompt_len != 5
            || greedy_steps != 1
            || !fixed_prefetch
            || !qd4_enabled
            || !device_attention
            || !device_attention_audit
            || !raw_wo_a_gpu_decode
            || device_short_decode
            || sparse_attention_arm != SparseAttentionArm::Sm89Crossover
            || std::env::var_os("COLIBRI_V4_TARGETED_ATTENTION_AUDIT_PATH").is_none())
    {
        return Err("targeted attention audit requires chat_user=hi, five-token prefill, layers 0..=2, greedy_steps=1, fixed prefetch, QD4, audited production attention, raw wo_a decode, sm89 crossover, no short decode, and an artifact path".into());
    }
    if qd4_enabled {
        if device_attention {
            if host_available_bytes()?
                < V4_QD4_PINNED_HOST_BYTES
                    + usize::from(qd4_immutable_expert_cache)
                        * V4_QD4_IMMUTABLE_EXPERT_CACHE_HOST_BYTES
            {
                return Err("persistent QD4 lanes lack the explicit pinned-host reserve".into());
            }
        } else {
            if !qd4_immutable_expert_cache {
                validate_qd4_lane_reserve(host_available_bytes()?, device_available_bytes()?)?;
            }
            let required_host = V4_QD4_PINNED_HOST_BYTES
                + usize::from(qd4_immutable_expert_cache)
                    * V4_QD4_IMMUTABLE_EXPERT_CACHE_HOST_BYTES;
            let required_device = V4_QD4_DEVICE_BYTES
                + usize::from(qd4_immutable_expert_cache)
                    * V4_QD4_IMMUTABLE_EXPERT_CACHE_DEVICE_BYTES;
            if host_available_bytes()? < required_host
                || device_available_bytes()? < required_device
            {
                return Err("persistent QD4 lanes plus immutable cache lack the explicit pinned-host/device reserve".into());
            }
        }
    }
    if device_attention && prompt_len != 5 {
        return Err(
            "device attention semantic pilot requires exactly five prompt positions".into(),
        );
    }
    let certified_argmax_production =
        certified_argmax_production_enabled(device_attention, greedy_steps)?;
    let short_decode_rows =
        short_device_decode_rows(prompt_len, greedy_steps, device_short_decode)?;
    let strict_audit_scratch_required =
        strict_compressor_enabled() && (!device_attention || device_attention_audit);
    let strict_audit_arena_bytes = strict_audit_scratch_required
        .then(|| StrictCompressorScratch::host_bytes(v4_state::HIDDEN))
        .transpose()?
        .unwrap_or(0);
    let device_attention_capacity = device_attention
        .then(|| {
            validate_device_attention_capacity(
                root,
                qd4_enabled,
                qd4_immutable_expert_cache,
                short_decode_rows,
                certified_argmax_production,
                device_available_bytes()?,
            )
        })
        .transpose()?;
    let fixed_prefetch_max_future_host_bytes = fixed_prefetch
        .then(|| {
            validate_fixed_prefetch_window(
                root,
                manifests,
                qd4_enabled,
                qd4_immutable_expert_cache,
                strict_audit_arena_bytes,
            )
        })
        .transpose()?;
    let stationary_mtp1_peak_reserve_bytes = stationary_mtp1_draft
        .then(|| mtp1_stationary_peak_reserve_bytes(root, qd4_enabled))
        .transpose()?
        .unwrap_or(0);
    let generic_stationary_reserve_plan = generic_device_stationary
        .then(|| v4_state::LayerStationaryRequestPlan::new(prompt_len, greedy_steps))
        .transpose()?;
    let generic_stationary_device_reserve_bytes = generic_stationary_reserve_plan
        .map(|plan| {
            generic_stationary_persistent_reserve_bytes(manifests, plan)?
                .checked_add(generic_stationary_scratch_reserve_bytes(plan)?)
                .ok_or_else(|| -> Box<dyn Error> {
                    "generic stationary device reserve overflow".into()
                })
        })
        .transpose()?
        .unwrap_or(0);
    let request_started = std::time::Instant::now();
    let stationary_host_reserve_bytes = fixed_prefetch_max_future_host_bytes
        .unwrap_or(0)
        .checked_mul(FIXED_PREFETCH_WINDOW_DEPTH)
        .and_then(|bytes| bytes.checked_add(usize::from(qd4_enabled) * V4_QD4_PINNED_HOST_BYTES))
        .and_then(|bytes| {
            bytes.checked_add(
                usize::from(qd4_immutable_expert_cache) * V4_QD4_IMMUTABLE_EXPERT_CACHE_HOST_BYTES,
            )
        })
        .and_then(|bytes| bytes.checked_add(strict_audit_arena_bytes))
        .and_then(|bytes| {
            bytes.checked_add(
                device_attention_capacity
                    .as_ref()
                    .map_or(0, |plan| plan.certified_argmax_host_temp_bytes),
            )
        })
        .and_then(|bytes| bytes.checked_add(mtp1_stationary_pinned_host_reserve_bytes(qd4_enabled)))
        .and_then(|bytes| {
            bytes.checked_add(
                usize::from(generic_device_stationary && pinned_fixed_pipeline)
                    * PINNED_FIXED_PIPELINE_WORKER_PINNED_HOST_BYTES,
            )
        })
        .ok_or("stationary raw host reserve overflow")?;
    let stationary_layers = manifests
        .iter()
        .map(|manifest| manifest.layer)
        .collect::<Vec<_>>();
    let decode_retention_overlay = parse_default_true_zero_one_flag(
        "COLIBRI_V4_DECODE_RETENTION_OVERLAY",
        std::env::var_os("COLIBRI_V4_DECODE_RETENTION_OVERLAY").as_deref(),
    )?;
    let mut stationary_raw_retention = (device_short_decode || generic_device_stationary)
        .then(|| {
            FixedLayerCache::stationary_raw_from_env(
                root,
                &stationary_layers,
                device_attention_capacity
                    .as_ref()
                    .map_or(0, |plan| plan.required_bytes)
                    .checked_add(
                        usize::from(generic_device_stationary && qd4_enabled) * V4_QD4_DEVICE_BYTES,
                    )
                    .and_then(|bytes| {
                        bytes.checked_add(
                            usize::from(generic_device_stationary && qd4_immutable_expert_cache)
                                * V4_QD4_IMMUTABLE_EXPERT_CACHE_DEVICE_BYTES,
                        )
                    })
                    .and_then(|bytes| bytes.checked_add(stationary_mtp1_peak_reserve_bytes))
                    .and_then(|bytes| bytes.checked_add(generic_stationary_device_reserve_bytes))
                    .ok_or("stationary raw device reserve overflow")?,
                stationary_host_reserve_bytes,
                usize::from(
                    generic_device_stationary
                        && qd4_immutable_expert_cache
                        && decode_retention_overlay,
                ) * V4_QD4_IMMUTABLE_EXPERT_CACHE_DEVICE_BYTES,
                usize::from(
                    generic_device_stationary
                        && qd4_immutable_expert_cache
                        && decode_retention_overlay,
                ) * V4_QD4_IMMUTABLE_EXPERT_CACHE_HOST_BYTES,
                stream,
            )
        })
        .transpose()?;
    let device_rope_lut_started = std::time::Instant::now();
    let device_rope_lut = device_attention
        .then(|| DeviceRequestRopeLut::new(stream, short_decode_rows))
        .transpose()?;
    let device_rope_lut_setup_ns = device_attention
        .then(|| u64::try_from(device_rope_lut_started.elapsed().as_nanos()).unwrap_or(u64::MAX))
        .unwrap_or(0);
    let mut strict_scratch = strict_audit_scratch_required
        .then(|| StrictCompressorScratch::new(stream, v4_state::HIDDEN))
        .transpose()?;
    let mut hidden = token_ids
        .iter()
        .map(|&token_id| {
            let embedding = v4_state::load(root, &[token_id])?.embedding;
            let mut row = Vec::with_capacity(v4_state::HC * v4_state::HIDDEN);
            for _ in 0..v4_state::HC {
                row.extend_from_slice(&embedding);
            }
            Ok::<_, Box<dyn Error>>(row)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let host_persistent_rows = if generic_device_state_audit {
        prompt_len
            .checked_add(greedy_steps)
            .ok_or("generic device state audit host context overflow")?
    } else if device_short_decode && device_attention_audit {
        short_decode_rows
    } else {
        prompt_len
    };
    let mut persistent = (generic_device_state_audit
        || (!generic_device_stationary && (!device_attention || device_attention_audit)))
        .then(|| v4_state::V4SerialPersistentState::new(host_persistent_rows))
        .transpose()?;
    let generic_plan = generic_stationary_reserve_plan;
    let mut generic_states = generic_device_stationary.then(|| {
        std::iter::repeat_with(|| None)
            .take(v4_state::V4_LAYER_COUNT)
            .collect::<Vec<Option<DeviceLayerPersistentState>>>()
    });
    let mut generic_scratch = None;
    let mut generic_lifecycle = generic_device_stationary
        .then(|| GenericStationaryLifecycle::new(v4_state::V4_LAYER_COUNT))
        .transpose()?;
    let mut generic_device_state_audit_heads = Vec::new();
    let mut generic_device_state_audit_wall_ns = 0_u64;
    let mut layer_records = (0..prompt_len)
        .map(|_| Vec::with_capacity(manifests.len()))
        .collect::<Vec<Vec<serde_json::Value>>>();
    let mut fixed_source_bytes = 0_usize;
    let mut fixed_h2d_bytes = 0_usize;
    let mut routed_source_bytes = 0_usize;
    let mut routed_h2d_bytes = 0_usize;
    let mut routed_expert_loads = 0_usize;
    let mut device_attention_dispatches = 0_usize;
    let device_attention_host_fallbacks = 0_usize;
    let mut device_attention_h2d_count = usize::from(device_attention);
    let mut device_attention_h2d_bytes = device_rope_lut
        .as_ref()
        .map_or(0, DeviceRequestRopeLut::num_bytes);
    let mut device_attention_rope_lut_d2d_count = 0_usize;
    let mut device_attention_rope_lut_d2d_bytes = 0_usize;
    let mut device_attention_terminal_d2h_count = 0_usize;
    let mut device_attention_terminal_d2h_bytes = 0_usize;
    let mut device_attention_audit_d2h_count = 0_usize;
    let mut device_attention_audit_d2h_bytes = 0_usize;
    let mut phase_profile_top_level_ns = 0_u64;
    let mut phase_profile_layer_outer_ns = 0_u64;
    let mut phase_profile_final_outer_ns = 0_u64;
    let mut one_token_decode_phase_profile_layer_outer_ns = 0_u64;
    let mut one_token_decode_phase_profile_final_outer_ns = 0_u64;
    let mut fixed_prefetch_records = Vec::with_capacity(manifests.len().saturating_sub(1));
    let mut targeted_attention_audit_complete = false;
    let mut device_states = device_attention.then(|| {
        std::iter::repeat_with(|| None)
            .take(if device_short_decode {
                v4_state::V4_LAYER_COUNT
            } else {
                2
            })
            .collect::<Vec<Option<DeviceTokenState>>>()
    });
    let qd4_setup_started = std::time::Instant::now();
    let mut qd4_lanes = if qd4_enabled {
        Some(
            (0..qd4_lane_count(true))
                .map(|_| {
                    let lane_stream = context.new_stream()?;
                    ExpertSlot::new_empty(context, module, &lane_stream)
                })
                .collect::<Result<Vec<_>, Box<dyn Error>>>()?,
        )
    } else {
        None
    };
    let mut qd4_immutable_cache = qd4_immutable_expert_cache
        .then(|| Qd4ImmutableExpertCache::new(context, module))
        .transpose()?;
    let mut qd4_immutable_cache_telemetry = None;
    let mut generic_prefill_peak = None::<(usize, usize, usize)>;
    let mut generic_decode_overlay_peak = None::<(usize, usize, usize)>;
    let qd4_startup_setup_ns = qd4_enabled
        .then(|| u64::try_from(qd4_setup_started.elapsed().as_nanos()).unwrap_or(u64::MAX))
        .unwrap_or(0);

    std::thread::scope(|scope| -> Result<(), Box<dyn Error>> {
        let mut prefetch_window: VecDeque<FixedPrefetchTask<'_>> = VecDeque::new();
        let mut next_prefetch_index = 0_usize;
        let scoped_result = (|| -> Result<(), Box<dyn Error>> {
            while fixed_prefetch
                && prefetch_window.len() < FIXED_PREFETCH_WINDOW_DEPTH
                && next_prefetch_index < manifests.len()
            {
                let next_layer = manifests[next_prefetch_index].layer;
                let worker_root = root.to_owned();
                let worker_start_ns = timeline_elapsed_ns(request_started);
                let worker = scope.spawn(move || {
                    let (host, telemetry) = if fixed_prefetch_raw_wo_a {
                        load_layer_host_raw_wo_a(&worker_root, next_layer, token_ids)
                    } else {
                        load_layer_host(&worker_root, next_layer, token_ids)
                    }
                    .map_err(|error| error.to_string())?;
                    let worker_end_ns = timeline_elapsed_ns(request_started);
                    Ok(FixedPrefetchedHost {
                        layer: next_layer,
                        host_bytes: host.num_bytes(),
                        host,
                        telemetry,
                        worker_start_ns,
                        worker_end_ns,
                    })
                });
                prefetch_window.push_back(FixedPrefetchTask {
                    layer: next_layer,
                    transition: None,
                    worker,
                });
                next_prefetch_index += 1;
            }
            for (manifest_index, manifest) in manifests.iter().enumerate() {
                let fixed_scope_started_ns =
                    phase_profile.then(|| timeline_elapsed_ns(request_started));
                let layer_prepare_started = phase_profile.then(std::time::Instant::now);
                let mut prefetched_current = false;
                let (runtime, fixed, retention_host, retention_raw_wo_a) = if fixed_prefetch {
                    let consume_start_ns = timeline_elapsed_ns(request_started);
                    let task = prefetch_window
                        .pop_front()
                        .ok_or("bounded fixed prefetch window is empty at consume")?;
                    if task.layer != manifest.layer {
                        return Err("bounded fixed prefetch window lost layer order".into());
                    }
                    let compute_end_ns = task
                        .transition
                        .as_ref()
                        .map(|transition| transition.compute_end_ns);
                    let prefetched = task
                        .worker
                        .join()
                        .map_err(|_| "layer-stationary fixed prefetch worker panicked")?
                        .map_err(|error| {
                            format!("layer-stationary fixed prefetch worker failed: {error}")
                        })?;
                    if manifest_index > 0 {
                        let transition = task.transition.ok_or(
                        "bounded fixed prefetch reached consume before the preceding layer closed",
                    )?;
                        fixed_prefetch_records.push(fixed_prefetch_transition_json(
                            transition.from_layer,
                            prefetched.layer,
                            &prefetched.telemetry,
                            prefetched.worker_start_ns,
                            prefetched.worker_end_ns,
                            prefetched.host_bytes,
                            transition.compute_start_ns,
                            compute_end_ns
                                .ok_or("bounded fixed prefetch transition lacks compute end")?,
                            consume_start_ns,
                            FIXED_PREFETCH_WINDOW_DEPTH,
                        )?);
                    } else if compute_end_ns.is_some() {
                        return Err(
                            "bounded fixed prefetch startup task unexpectedly has a transition"
                                .into(),
                        );
                    }
                    let h2d_started = std::time::Instant::now();
                    let (runtime, upload, raw_wo_a) = if fixed_prefetch_raw_wo_a {
                        let (runtime, upload, raw_wo_a) = upload_layer_runtime_raw_wo_a(
                            root,
                            manifest.layer,
                            &prefetched.host,
                            module,
                            stream,
                        )?;
                        (runtime, upload, Some(raw_wo_a))
                    } else {
                        let runtime =
                            upload_layer_runtime(root, manifest.layer, &prefetched.host, stream)?;
                        if bench_telemetry_enabled() {
                            stream.synchronize()?;
                        }
                        (
                            runtime,
                            FixedLoadTelemetry {
                                fixed_h2d_ns: u64::try_from(h2d_started.elapsed().as_nanos())
                                    .unwrap_or(u64::MAX),
                                fixed_h2d_bytes: 0,
                                cache_status: "prefetched_blocking_h2d",
                                ..FixedLoadTelemetry::default()
                            },
                            None,
                        )
                    };
                    let mut telemetry = prefetched.telemetry;
                    telemetry.fixed_h2d_ns = upload.fixed_h2d_ns;
                    telemetry.fixed_h2d_bytes = if fixed_prefetch_raw_wo_a {
                        upload.fixed_h2d_bytes
                    } else {
                        runtime.weights.num_bytes()
                    };
                    telemetry.fixed_decode_ns = upload.fixed_decode_ns;
                    telemetry.raw_h2d_ns = upload.raw_h2d_ns;
                    telemetry.raw_h2d_bytes = upload.raw_h2d_bytes;
                    prefetched_current = true;
                    let retention_host = fixed_prefetch_raw_wo_a.then_some(prefetched.host);
                    (runtime, telemetry, retention_host, raw_wo_a)
                } else {
                    let (runtime, fixed) =
                        load_layer_runtime(root, manifest.layer, token_ids, stream)?;
                    (runtime, fixed, None, None)
                };
                fixed_source_bytes = fixed_source_bytes.saturating_add(fixed.fixed_source_bytes);
                fixed_h2d_bytes = fixed_h2d_bytes.saturating_add(fixed.fixed_h2d_bytes);
                let blocking_fixed = prefetched_current.then(|| FixedLoadTelemetry {
                    fixed_decode_ns: fixed.fixed_decode_ns,
                    fixed_h2d_ns: fixed.fixed_h2d_ns,
                    fixed_h2d_bytes: fixed.fixed_h2d_bytes,
                    fixed_source_bytes: fixed.fixed_source_bytes,
                    raw_h2d_ns: fixed.raw_h2d_ns,
                    raw_h2d_bytes: fixed.raw_h2d_bytes,
                    cache_status: "prefetched_blocking_h2d",
                    ..FixedLoadTelemetry::default()
                });
                let hash_ids = load_layer_hash_ids(root, manifest.layer, token_ids)?;
                let layer_prepare_ns = layer_prepare_started
                    .map(|started| u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX))
                    .unwrap_or(0);
                while fixed_prefetch
                    && prefetch_window.len() < FIXED_PREFETCH_WINDOW_DEPTH
                    && next_prefetch_index < manifests.len()
                {
                    let next_layer = manifests[next_prefetch_index].layer;
                    let worker_root = root.to_owned();
                    let worker_start_ns = timeline_elapsed_ns(request_started);
                    let worker = scope.spawn(move || {
                        let (host, telemetry) = if fixed_prefetch_raw_wo_a {
                            load_layer_host_raw_wo_a(&worker_root, next_layer, token_ids)
                        } else {
                            load_layer_host(&worker_root, next_layer, token_ids)
                        }
                        .map_err(|error| error.to_string())?;
                        let worker_end_ns = timeline_elapsed_ns(request_started);
                        Ok(FixedPrefetchedHost {
                            layer: next_layer,
                            host_bytes: host.num_bytes(),
                            host,
                            telemetry,
                            worker_start_ns,
                            worker_end_ns,
                        })
                    });
                    prefetch_window.push_back(FixedPrefetchTask {
                        layer: next_layer,
                        transition: None,
                        worker,
                    });
                    next_prefetch_index += 1;
                }
                let compute_start_ns = timeline_elapsed_ns(request_started);
                if let Some(cache) = qd4_immutable_cache.as_mut() {
                    cache.begin_layer(manifest.layer)?;
                }
                // ponytail: retain at most the laptop-safe per-layer cap; add eviction
                // only if profiling shows reloads beyond that cap dominate.
                let mut expert_slots = expert_slot_reuse.then(BTreeMap::new);
                let device_state_started = std::time::Instant::now();
                let mut device_state_reset_cuda_ns = 0_u64;
                let mut device_state_reset_calls = 0_usize;
                let mut device_state = None;
                if device_attention {
                    let states = device_states
                        .as_mut()
                        .ok_or("device state cache disappeared during production prefill")?;
                    let state_index = if device_short_decode {
                        manifest.layer
                    } else if manifest.compression_ratio == v4_state::CompressionRatio::Csa4 {
                        1
                    } else {
                        0
                    };
                    let slot = states
                        .get_mut(state_index)
                        .ok_or("device state cache index is outside its declared layout")?;
                    if slot.is_none() {
                        *slot = Some(DeviceTokenState::new(stream, &runtime.weights)?);
                    }
                    let state = slot
                        .as_mut()
                        .ok_or("device state cache slot disappeared during production prefill")?;
                    let csa = manifest.compression_ratio == v4_state::CompressionRatio::Csa4;
                    let reset_start = (!device_short_decode && csa)
                        .then(|| {
                            stream
                                .context()
                                .new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))
                        })
                        .transpose()?;
                    if let Some(start) = reset_start.as_ref() {
                        start.record(stream)?;
                    }
                    if !device_short_decode {
                        device_state_reset_calls =
                            state.reset_for_layer(module, stream, &runtime.weights)?;
                    }
                    if let Some(start) = reset_start.as_ref() {
                        let reset_end = stream
                            .context()
                            .new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
                        reset_end.record(stream)?;
                        reset_end.synchronize()?;
                        device_state_reset_cuda_ns =
                            (start.elapsed_ms(&reset_end)? as f64 * 1_000_000.0) as u64;
                    }
                    device_state = Some(state);
                }
                let device_state_setup_ns = device_attention
                    .then(|| {
                        u64::try_from(device_state_started.elapsed().as_nanos()).unwrap_or(u64::MAX)
                    })
                    .unwrap_or(0);
                for token_position in 0..prompt_len {
                    let layer_scope_started_ns = phase_profile.then(|| {
                        if token_position == 0 {
                            fixed_scope_started_ns
                                .expect("profiled fixed scope must have a request-relative start")
                        } else {
                            timeline_elapsed_ns(request_started)
                        }
                    });
                    let token_id = token_ids[token_position];
                    let row_hash_ids = if runtime.weights.hash_router {
                        &hash_ids[token_position * v4_router::V4_TOPK
                            ..(token_position + 1) * v4_router::V4_TOPK]
                    } else {
                        &[]
                    };
                    let kv_rows_before = if let Some(states) = generic_states.as_ref() {
                        states
                            .get(manifest.layer)
                            .and_then(Option::as_ref)
                            .map_or(0, |state| state.committed_rows)
                    } else {
                        device_state.as_ref().map_or_else(
                            || {
                                persistent
                                    .as_ref()
                                    .and_then(|state| state.layer(manifest.layer))
                                    .map_or(0, v4_state::V4LayerPersistentState::committed_kv_rows)
                            },
                            |state| state.committed_rows,
                        )
                    };
                    let cutpoint_pre_hidden =
                        l0_cutpoint.then(|| format!("{:016x}", fnv1a_u16(&hidden[token_position])));
                    let mut observed_state = None;
                    let mut completed = None;
                    if generic_device_stationary {
                        run_generic_device_state_audit_mirror(
                            generic_device_state_audit,
                            persistent.as_mut(),
                            manifest.layer,
                            &hidden[token_position],
                            |host_state, mirror_hidden| {
                                run_real_layer_attention_rows(
                                    module,
                                    stream,
                                    None,
                                    &runtime,
                                    mirror_hidden,
                                    token_position,
                                    host_state,
                                    None,
                                    strict_scratch.as_mut(),
                                )?;
                                Ok(())
                            },
                        )?;
                        let audit_expected_compressed_prefix = if generic_device_state_audit
                            && (token_position + 1).is_multiple_of(manifest.compression_ratio.get())
                        {
                            Some(
                                persistent
                                    .as_ref()
                                    .and_then(|host| host.layer(manifest.layer))
                                    .ok_or("generic boundary audit host mirror lost its layer")?
                                    .compressed_kv_bf16(),
                            )
                        } else {
                            None
                        };
                        let plan =
                            generic_plan.ok_or("generic stationary request plan disappeared")?;
                        let states = generic_states
                            .as_mut()
                            .ok_or("generic stationary stores disappeared during prefill")?;
                        let state = states
                            .get_mut(manifest.layer)
                            .ok_or("generic stationary layer is outside its 61 stores")?;
                        if state.is_none() {
                            *state = Some(DeviceLayerPersistentState::new(
                                stream,
                                &runtime.weights,
                                plan,
                            )?);
                        }
                        let state = state
                            .as_mut()
                            .ok_or("generic stationary layer store disappeared")?;
                        if generic_scratch.is_none() {
                            let scratch =
                                DeviceRequestScratch::new(stream, &runtime.weights, plan)?;
                            if scratch.num_bytes()
                                != generic_stationary_scratch_reserve_bytes(plan)?
                            {
                                return Err("generic stationary scratch allocation diverged from pre-cache peak ledger".into());
                            }
                            generic_scratch = Some(scratch);
                        }
                        let scratch = generic_scratch
                            .as_mut()
                            .ok_or("generic stationary scratch disappeared")?;
                        let lifecycle = generic_lifecycle
                            .as_mut()
                            .ok_or("generic stationary lifecycle disappeared")?;
                        lifecycle.begin_layer(manifest.layer, token_position)?;
                        let run = run_real_layer_rows(
                            context,
                            module,
                            stream,
                            None,
                            &runtime,
                            qd4_lanes.as_deref_mut(),
                            qd4_immutable_cache.as_mut(),
                            None,
                            row_hash_ids,
                            &hidden[token_position],
                            &[token_id],
                            token_position,
                            None,
                            None,
                            None,
                            false,
                            None,
                            false,
                            Some((state, scratch, audit_expected_compressed_prefix)),
                            None,
                        None,
                        );
                        let run = match run {
                            Ok(run) => run,
                            Err(error) => {
                                lifecycle.poison();
                                return Err(error);
                            }
                        };
                        lifecycle.commit_layer(manifest.layer, token_position)?;
                        completed = Some(run);
                    } else if device_attention && !device_attention_audit {
                        completed = Some(run_real_layer_rows(
                            context,
                            module,
                            stream,
                            None,
                            &runtime,
                            qd4_lanes.as_deref_mut(),
                            qd4_immutable_cache.as_mut(),
                            expert_slots.as_mut(),
                            row_hash_ids,
                            &hidden[token_position],
                            &[token_id],
                            token_position,
                            None,
                            device_state.as_deref_mut(),
                            device_rope_lut.as_ref(),
                            false,
                            None,
                            false,
                            None,
                            None,
                        None,
                        )?);
                    } else {
                        persistent
                        .as_mut()
                        .ok_or("host persistent state is unavailable outside device production")?
                        .transaction(manifest.layer, |layer_state| {
                            let targeted_position = targeted_attention_audit
                                && manifest.layer == 2
                                && token_position == 3;
                            let targeted_precommit = targeted_position
                                .then(|| {
                                    let host = host_persistent_state_audit(layer_state);
                                    let state = device_state
                                        .as_deref()
                                        .ok_or("targeted audit device state disappeared")?;
                                    let device = device_persistent_state_audit(stream, state)?;
                                    Ok::<_, Box<dyn Error>>((host, device))
                                })
                                .transpose()?;
                            let expected_attention = if device_attention {
                                let oracle_started = std::time::Instant::now();
                                let oracle = run_real_layer_attention_rows(
                                    module,
                                    stream,
                                    None,
                                    &runtime,
                                    &hidden[token_position],
                                    token_position,
                                    layer_state,
                                    None,
                                    strict_scratch.as_mut(),
                                )?;
                                Some((
                                    oracle.output,
                                    oracle.topk.ok_or("strict audit host oracle omitted top-k")?,
                                    host_persistent_state_audit(layer_state),
                                    u64::try_from(oracle_started.elapsed().as_nanos())
                                        .unwrap_or(u64::MAX),
                                ))
                            } else {
                                None
                            };
                            let mut run = run_real_layer_rows(
                                context,
                                module,
                                stream,
                                None,
                                &runtime,
                                qd4_lanes.as_deref_mut(),
                                qd4_immutable_cache.as_mut(),
                                expert_slots.as_mut(),
                                row_hash_ids,
                                &hidden[token_position],
                                &[token_id],
                                token_position,
                                Some(layer_state),
                                device_state.as_deref_mut(),
                                device_rope_lut.as_ref(),
                                device_attention_audit,
                                None,
                                false,
                                None,
                                strict_scratch.as_mut(),
                            None,
                            )?;
                            if let Some((expected_output, expected_topk, expected_state, oracle_ns)) =
                                expected_attention
                            {
                                let device_state =
                                    device_state.as_ref().ok_or("device audit state disappeared")?;
                                let actual_state = device_persistent_state_audit(stream, device_state)?;
                                let (audit_d2h_count, audit_d2h_bytes) =
                                    device_persistent_state_audit_d2h_transfer(device_state);
                                let transfer = run
                                    .device_transfer
                                    .as_mut()
                                    .ok_or("device audit omitted transfer telemetry")?;
                                transfer.intermediate_or_audit_d2h_count = transfer
                                    .intermediate_or_audit_d2h_count
                                    .saturating_add(audit_d2h_count);
                                transfer.intermediate_or_audit_d2h_bytes = transfer
                                    .intermediate_or_audit_d2h_bytes
                                    .saturating_add(audit_d2h_bytes);
                                run.stage.attention_intermediate_or_audit_d2h_count = run
                                    .stage
                                    .attention_intermediate_or_audit_d2h_count
                                    .saturating_add(audit_d2h_count);
                                run.stage.attention_intermediate_or_audit_d2h_bytes = run
                                    .stage
                                    .attention_intermediate_or_audit_d2h_bytes
                                    .saturating_add(audit_d2h_bytes);
                                check_persistent_state_audit(
                                    manifest.layer,
                                    token_position,
                                    &expected_state,
                                    &actual_state,
                                )?;
                                let actual_topk = run
                                    .attention_topk
                                    .as_deref()
                                    .ok_or("device audit omitted sparse top-k D2H")?;
                                if targeted_position {
                                    let (precommit_host, precommit_device) = targeted_precommit
                                        .as_ref()
                                        .ok_or("targeted audit omitted precommit state")?;
                                    let _ = run_targeted_attention_audit(
                                        module,
                                        stream,
                                        &runtime,
                                        &hidden[token_position],
                                        token_position,
                                        layer_state,
                                        device_state,
                                        &expected_topk,
                                        precommit_host,
                                        precommit_device,
                                    )?;
                                    targeted_attention_audit_complete = true;
                                } else {
                                    if let Some(index) =
                                        first_mismatch(&expected_output, &run.attention_output)
                                    {
                                        return Err(format!(
                                            "device audit layer {} position {} attention_post first mismatch at {index}",
                                            manifest.layer, token_position
                                        )
                                        .into());
                                    }
                                    if actual_topk != expected_topk.as_slice() {
                                        return Err(format!(
                                            "device audit layer {} position {} sparse top-k raw bits mismatch",
                                            manifest.layer, token_position
                                        )
                                        .into());
                                    }
                                }
                                run.stage.attention_audit_oracle_ns = oracle_ns;
                                observed_state = Some(actual_state);
                            }
                            completed = Some(run);
                            Ok(())
                        })?;
                    }
                    let mut run = completed
                        .ok_or("layer-stationary prefill transaction produced no output")?;
                    if run.routes.ids.len() != v4_router::V4_TOPK {
                        return Err(format!(
                            "layer {} router produced incomplete causal prefill routes",
                            manifest.layer
                        )
                        .into());
                    }
                    if token_position == 0 && device_attention {
                        run.stage.device_state_setup_ns = device_state_setup_ns;
                        run.stage.device_state_setup_count = 1;
                        run.stage.device_state_reset_cuda_ns = device_state_reset_cuda_ns;
                        run.stage.device_state_reset_calls = device_state_reset_calls;
                    }
                    if token_position == 0 && phase_profile {
                        run.stage.layer_prepare_ns = layer_prepare_ns;
                        run.stage.layer_prepare_count = 1;
                    }
                    let kv_rows_after = if let Some(states) = generic_states.as_ref() {
                        states
                            .get(manifest.layer)
                            .and_then(Option::as_ref)
                            .map_or(0, |state| state.committed_rows)
                    } else {
                        device_state.as_ref().map_or_else(
                            || {
                                persistent
                                    .as_ref()
                                    .and_then(|state| state.layer(manifest.layer))
                                    .map_or(0, v4_state::V4LayerPersistentState::committed_kv_rows)
                            },
                            |state| state.committed_rows,
                        )
                    };
                    if kv_rows_after != kv_rows_before + 1 {
                        return Err(
                            "layer-stationary prefill committed a noncontiguous KV row".into()
                        );
                    }
                    let state_digests = match persistent_state_digest_source(
                        observed_state.is_some(),
                        generic_device_stationary,
                        persistent.is_some(),
                    ) {
                        PersistentStateDigestSource::ObservedDevice => {
                            let state = observed_state
                                .as_ref()
                                .expect("observed device state disappeared before digesting");
                            Some((
                                fnv1a_u16(&state.kv),
                                fnv1a_u16(&state.compressed_kv),
                                fnv1a_f32(&state.main_kv),
                                fnv1a_f32(&state.main_score),
                                fnv1a_u16(&state.indexer_kv),
                                state.indexer_main_kv.as_deref().map(fnv1a_f32),
                                state.indexer_main_score.as_deref().map(fnv1a_f32),
                            ))
                        }
                        PersistentStateDigestSource::HostPersistent => {
                            let state = persistent
                                .as_ref()
                                .and_then(|state| state.layer(manifest.layer))
                                .ok_or("host layer state disappeared after commit")?;
                            let main = state.compressor();
                            let indexer = state.indexer();
                            Some((
                                fnv1a_u16(state.kv_bf16()),
                                fnv1a_u16(state.compressed_kv_bf16()),
                                fnv1a_f32(main.kv()),
                                fnv1a_f32(main.score()),
                                fnv1a_u16(state.indexer_kv_bf16()),
                                indexer.map(|value| fnv1a_f32(value.kv())),
                                indexer.map(|value| fnv1a_f32(value.score())),
                            ))
                        }
                        PersistentStateDigestSource::Unobserved => None,
                    };
                    let topk_observed = run.attention_topk.is_some();
                    let record = serde_json::json!({
                        "event": "layer",
                        "token_position": token_position,
                        "token_id": token_id,
                        "layer": manifest.layer,
                        "compression_ratio": manifest.compression_ratio.get(),
                        "hidden_bf16_digest": format!("{:016x}", fnv1a_u16(&run.output)),
                        "persistent_state_observed": state_digests.is_some(),
                        "kv_cache_bf16_digest": state_digests.as_ref().map(|value| format!("{:016x}", value.0)),
                        "main_compressed_kv_bf16_digest": state_digests.as_ref().map(|value| format!("{:016x}", value.1)),
                        "main_compressor_kv_state_f32_digest": state_digests.as_ref().map(|value| format!("{:016x}", value.2)),
                        "main_compressor_score_state_f32_digest": state_digests.as_ref().map(|value| format!("{:016x}", value.3)),
                        "indexer_compressed_kv_bf16_digest": state_digests.as_ref().map(|value| format!("{:016x}", value.4)),
                        "indexer_compressor_kv_state_f32_digest": state_digests.as_ref().and_then(|value| value.5.map(|digest| format!("{digest:016x}"))),
                        "indexer_compressor_score_state_f32_digest": state_digests.as_ref().and_then(|value| value.6.map(|digest| format!("{digest:016x}"))),
                        "route_ids": &run.routes.ids,
                        "route_weights_f32": &run.routes.weights,
                        "topk_observed": topk_observed,
                        "attention_topk_digest": topk_observed.then(|| format!("{:016x}", run.attention_topk_digest)),
                        "attention_topk_per_row": topk_observed.then_some(run.attention_topk_per_row),
                        "manifest_contract_range_bytes": manifest.contract_range_bytes,
                        "kv_rows_before": kv_rows_before,
                        "kv_rows_after": kv_rows_after,
                    });
                    let (next_hidden, mut record, stage, indexer_topk_gpu_ns, transfer) = (
                        run.output,
                        record,
                        run.stage,
                        run.indexer_topk_gpu_ns,
                        run.device_transfer,
                    );
                    if device_attention || generic_device_stationary {
                        let transfer = transfer
                            .ok_or("device attention dispatch omitted transfer telemetry")?;
                        device_attention_dispatches = device_attention_dispatches.saturating_add(1);
                        device_attention_h2d_count =
                            device_attention_h2d_count.saturating_add(transfer.h2d_count);
                        device_attention_h2d_bytes =
                            device_attention_h2d_bytes.saturating_add(transfer.h2d_bytes);
                        device_attention_rope_lut_d2d_count = device_attention_rope_lut_d2d_count
                            .saturating_add(transfer.rope_lut_d2d_count);
                        device_attention_rope_lut_d2d_bytes = device_attention_rope_lut_d2d_bytes
                            .saturating_add(transfer.rope_lut_d2d_bytes);
                        device_attention_terminal_d2h_count = device_attention_terminal_d2h_count
                            .saturating_add(transfer.terminal_d2h_count);
                        device_attention_terminal_d2h_bytes = device_attention_terminal_d2h_bytes
                            .saturating_add(transfer.terminal_d2h_bytes);
                        device_attention_audit_d2h_count = device_attention_audit_d2h_count
                            .saturating_add(transfer.intermediate_or_audit_d2h_count);
                        device_attention_audit_d2h_bytes = device_attention_audit_d2h_bytes
                            .saturating_add(transfer.intermediate_or_audit_d2h_bytes);
                    } else if transfer.is_some() {
                        return Err(
                            "host attention unexpectedly reported device transfer telemetry".into(),
                        );
                    }
                    routed_source_bytes =
                        routed_source_bytes.saturating_add(stage.routed_source_bytes);
                    routed_h2d_bytes = routed_h2d_bytes.saturating_add(stage.routed_h2d_bytes);
                    routed_expert_loads =
                        routed_expert_loads.saturating_add(stage.routed_expert_loads);
                    if let Some(request_start_ns) = layer_scope_started_ns {
                        let request_end_ns = timeline_elapsed_ns(request_started);
                        let fixed_scope = (token_position == 0)
                            .then_some(blocking_fixed.as_ref().unwrap_or(&fixed));
                        let (profile, attributed_ns, outer_ns) =
                            stationary_layer_phase_profile_json(
                                request_start_ns,
                                request_end_ns,
                                fixed_scope,
                                &stage,
                                indexer_topk_gpu_ns,
                                qd4_enabled,
                            )?;
                        record
                            .as_object_mut()
                            .ok_or("stationary layer record is not an object")?
                            .insert("phase_profile".into(), profile);
                        phase_profile_top_level_ns =
                            phase_profile_top_level_ns.saturating_add(attributed_ns);
                        phase_profile_layer_outer_ns =
                            phase_profile_layer_outer_ns.saturating_add(outer_ns);
                    }
                    hidden[token_position] = next_hidden;
                    layer_records[token_position].push(record);
                    if targeted_attention_audit_complete {
                        return Ok(());
                    }
                    if l0_cutpoint && manifest.layer == 0 && token_position == 0 {
                        println!(
                            "COLIBRI_L0_POS0_CUTPOINT={}",
                            serde_json::json!({
                                "pre_hidden_bf16_digest": cutpoint_pre_hidden,
                                "start_pos": token_position,
                                "compression_ratio": manifest.compression_ratio.get(),
                                "runtime_weight_bytes": runtime.weights.num_bytes(),
                                "pre_committed_rows": kv_rows_before,
                                "record": layer_records[token_position].last(),
                            })
                        );
                        return Err("L0 pos0 cutpoint reached after the production record".into());
                    }
                }
                if let Some(cache) = qd4_immutable_cache.as_mut() {
                    cache.finish_layer(manifest.layer)?;
                }
                drop(expert_slots);
                if let Some(cache) = stationary_raw_retention.as_mut() {
                    cache.capture_stationary_raw(
                        root,
                        manifest.layer,
                        prompt_len,
                        retention_host
                            .ok_or("stationary raw retention lost its prefetched host payload")?,
                        runtime,
                        retention_raw_wo_a
                            .ok_or("stationary raw retention lost its uploaded raw payload")?,
                        &fixed,
                        stream,
                    )?;
                } else {
                    drop(runtime);
                }
                if fixed_prefetch && manifest_index + 1 < manifests.len() {
                    let next_layer = manifests[manifest_index + 1].layer;
                    let task = prefetch_window
                        .iter_mut()
                        .find(|task| task.layer == next_layer)
                        .ok_or("bounded fixed prefetch lost the next ordered layer")?;
                    task.transition = Some(FixedPrefetchTransition {
                        from_layer: manifest.layer,
                        compute_start_ns,
                        compute_end_ns: timeline_elapsed_ns(request_started),
                    });
                }
            }
            if !prefetch_window.is_empty() {
                return Err("bounded fixed prefetch left an unconsumed worker".into());
            }
            Ok(())
        })();
        while let Some(task) = prefetch_window.pop_front() {
            let _ = task.worker.join();
        }
        scoped_result
    })?;
    if targeted_attention_audit {
        if !targeted_attention_audit_complete {
            return Err("targeted attention audit did not reach layer 2 position 3".into());
        }
        return Ok(());
    }
    if fixed_prefetch && fixed_prefetch_records.len() != manifests.len().saturating_sub(1) {
        return Err("layer-stationary fixed prefetch did not record every transition".into());
    }
    if !generic_device_stationary
        && (persistent
            .as_ref()
            .is_some_and(|state| state.allocated_layers() != v4_state::V4_LAYER_COUNT)
            || layer_records
                .iter()
                .any(|records| records.len() != v4_state::V4_LAYER_COUNT))
    {
        return Err("layer-stationary prefill did not commit every layer/position pair".into());
    }
    if generic_device_stationary && qd4_immutable_expert_cache {
        let cache = qd4_immutable_cache
            .take()
            .ok_or("generic stationary lost its immutable-cache owner after prefill")?;
        if cache.layer.is_some()
            || !cache.resident.is_empty()
            || cache.free.len() != V4_QD4_IMMUTABLE_EXPERT_CACHE_SLOTS
        {
            return Err("generic stationary immutable cache was not drained before handoff".into());
        }
        qd4_immutable_cache_telemetry = Some(cache.telemetry);
        drop(cache);
    }
    if generic_device_stationary {
        let lifecycle = generic_lifecycle
            .as_mut()
            .ok_or("generic stationary lifecycle disappeared after prefill")?;
        let persistent_device_bytes = generic_states
            .as_ref()
            .ok_or("generic stationary stores disappeared after prefill")?
            .iter()
            .map(|state| {
                state
                    .as_ref()
                    .map_or(0, DeviceLayerPersistentState::num_bytes)
            })
            .try_fold(0_usize, |total, bytes| total.checked_add(bytes))
            .ok_or("generic stationary device ledger overflow")?;
        let raw_cache_device_bytes = stationary_raw_retention
            .as_ref()
            .map(|cache| {
                cache
                    .plan
                    .iter()
                    .zip(&cache.device_selected)
                    .filter_map(|(plan, selected)| selected.then_some(plan.raw_device_bytes))
                    .sum::<usize>()
            })
            .unwrap_or(0);
        let raw_cache_host_bytes = stationary_raw_retention
            .as_ref()
            .map(|cache| {
                cache
                    .plan
                    .iter()
                    .zip(&cache.host_selected)
                    .filter_map(|(plan, selected)| selected.then_some(plan.raw_host_bytes))
                    .sum::<usize>()
            })
            .unwrap_or(0);
        let generic_scratch_bytes = generic_scratch
            .as_ref()
            .map_or(0, DeviceRequestScratch::num_bytes);
        let base_device_bytes = persistent_device_bytes
            .checked_add(generic_scratch_bytes)
            .and_then(|bytes| bytes.checked_add(usize::from(qd4_enabled) * V4_QD4_DEVICE_BYTES))
            .and_then(|bytes| {
                bytes.checked_add(
                    usize::from(qd4_immutable_expert_cache)
                        * V4_QD4_IMMUTABLE_EXPERT_CACHE_DEVICE_BYTES,
                )
            })
            .and_then(|bytes| bytes.checked_add(raw_cache_device_bytes))
            // Scratch remains allocated after the worker drain; prepared
            // runtimes are accounted as a separate maximum below.
            .and_then(|bytes| bytes.checked_add(PINNED_FIXED_PIPELINE_WORKER_SCRATCH_DEVICE_BYTES))
            .ok_or("generic stationary lifecycle base device ledger overflow")?;
        let device_bytes = base_device_bytes
            .checked_add(stationary_mtp1_peak_reserve_bytes)
            .ok_or("generic stationary lifecycle MTP device ledger overflow")?
            .max(
                base_device_bytes
                    .checked_add(PINNED_FIXED_PIPELINE_WORKER_PREPARED_DEVICE_BYTES)
                    .ok_or("generic stationary lifecycle prepared device ledger overflow")?,
            );
        let audit_mirror_host_bytes = if generic_device_state_audit {
            generic_stationary_audit_mirror_host_bytes(
                manifests,
                generic_plan.ok_or("generic stationary audit has no request plan")?,
            )?
        } else {
            0
        };
        let host_bytes = raw_cache_host_bytes
            .checked_add(audit_mirror_host_bytes)
            .and_then(|bytes| {
                bytes.checked_add(
                    usize::from(stationary_mtp1_draft) * CERTIFIED_ARGMAX_HOST_TEMP_BYTES,
                )
            })
            .ok_or("generic stationary lifecycle host ledger overflow")?;
        let pinned_bytes = usize::from(qd4_enabled) * V4_QD4_PINNED_HOST_BYTES
            + usize::from(qd4_immutable_expert_cache) * V4_QD4_IMMUTABLE_EXPERT_CACHE_HOST_BYTES
            + PINNED_FIXED_PIPELINE_WORKER_PINNED_HOST_BYTES
            + mtp1_stationary_pinned_host_reserve_bytes(qd4_enabled);
        generic_prefill_peak = Some((device_bytes, host_bytes, pinned_bytes));
        lifecycle.observe_peaks(device_bytes, host_bytes, pinned_bytes);
        if lifecycle.telemetry.layer_commits != prompt_len * v4_state::V4_LAYER_COUNT {
            return Err(
                "generic stationary lifecycle commit ledger diverged from prefill rows".into(),
            );
        }
        lifecycle.assert_drained()?;
    }
    let initial_teacher_state_before_mtp = if generic_device_state_audit {
        let (audit, snapshot) = generic_device_state_audit_head(
            stream,
            generic_states
                .as_ref()
                .ok_or("generic device state audit stores disappeared after prefill")?,
            persistent
                .as_ref()
                .ok_or("generic device state audit host mirror disappeared after prefill")?,
            generic_lifecycle
                .as_ref()
                .ok_or("generic device state audit lifecycle disappeared after prefill")?,
            generic_scratch
                .as_ref()
                .ok_or("generic device state audit scratch disappeared after prefill")?,
            prompt_len - 1,
        )
        .map_err(|error| -> Box<dyn Error> {
            format!("pre-MTP generic device state audit failed: {error}").into()
        })?;
        generic_device_state_audit_wall_ns = generic_device_state_audit_wall_ns
            .saturating_add(audit["audit_wall_ns"].as_u64().unwrap_or(u64::MAX));
        generic_device_state_audit_heads.push(audit);
        Some(snapshot)
    } else {
        None
    };
    if device_attention {
        let expected_dispatches = prompt_len * v4_state::V4_LAYER_COUNT;
        let expected_csa_boundary_copies = manifests
            .iter()
            .filter(|manifest| manifest.compression_ratio == v4_state::CompressionRatio::Csa4)
            .count();
        let expected_rope_lut_d2d_count = expected_dispatches
            .checked_mul(2)
            .and_then(|count| count.checked_add(expected_csa_boundary_copies))
            .ok_or("device RoPE LUT D2D count overflow")?;
        let rope_row_bytes = v4_state::Q_ROPE_DIM * std::mem::size_of::<f32>();
        let expected_h2d_count = expected_dispatches + 1;
        let expected_h2d_bytes = device_attention_expected_h2d_bytes(
            device_rope_lut
                .as_ref()
                .map_or(0, DeviceRequestRopeLut::num_bytes),
            expected_dispatches,
        )?;
        if device_attention_dispatches != expected_dispatches
            || device_attention_host_fallbacks != 0
            || device_attention_h2d_count != expected_h2d_count
            || device_attention_h2d_bytes != expected_h2d_bytes
            || device_attention_rope_lut_d2d_count != expected_rope_lut_d2d_count
            || device_attention_rope_lut_d2d_bytes != expected_rope_lut_d2d_count * rope_row_bytes
            || device_attention_terminal_d2h_count != expected_dispatches
            || device_attention_terminal_d2h_bytes
                != expected_dispatches
                    * v4_state::HC
                    * v4_state::HIDDEN
                    * std::mem::size_of::<u16>()
            || (!device_attention_audit
                && (device_attention_audit_d2h_count != 0 || device_attention_audit_d2h_bytes != 0))
        {
            return Err("device attention measured transfer/dispatch accounting diverged from the strict five-token contract".into());
        }
    }
    if qd4_enabled && !generic_device_stationary {
        let expected_loads = qd4_expected_routed_loads(manifests.len(), prompt_len)?;
        if let Some(cache) = qd4_immutable_cache.as_ref() {
            let telemetry = &cache.telemetry;
            let expected_d2d_bytes = expected_loads
                .checked_mul(v4_expert::V4_ROUTED_EXPERT_BYTES)
                .ok_or("QD4 immutable expert cache D2D byte count overflow")?;
            if telemetry.route_executions != expected_loads
                || telemetry.source_loads != telemetry.expert_union_loads
                || telemetry.source_bytes
                    != telemetry
                        .source_loads
                        .checked_mul(v4_expert::V4_ROUTED_EXPERT_BYTES)
                        .ok_or("QD4 immutable expert cache source byte count overflow")?
                || telemetry.h2d_loads != telemetry.source_loads
                || telemetry.h2d_bytes != telemetry.source_bytes
                || telemetry.hits != expected_loads.saturating_sub(telemetry.source_loads)
                || telemetry.d2d_copies != expected_loads
                || telemetry.d2d_bytes != expected_d2d_bytes
                || telemetry.max_occupancy > V4_QD4_IMMUTABLE_EXPERT_CACHE_SLOTS
                || telemetry.completed_layers != manifests.len()
                || telemetry.worker_errors != 0
                || telemetry.fallbacks != 0
                || routed_expert_loads != telemetry.source_loads
                || routed_source_bytes != telemetry.source_bytes
                || routed_h2d_bytes != telemetry.h2d_bytes
            {
                return Err(
                    "stationary QD4 immutable cache physical telemetry did not close".into(),
                );
            }
        } else {
            let expected_bytes = expected_loads
                .checked_mul(v4_expert::V4_ROUTED_EXPERT_BYTES)
                .ok_or("QD4 routed byte count overflow")?;
            if routed_expert_loads != expected_loads
                || routed_source_bytes != expected_bytes
                || routed_h2d_bytes != expected_bytes
            {
                return Err(
                    "stationary QD4 routed load/byte accounting diverged from its six-route contract"
                        .into(),
                );
            }
        }
    }

    if certified_argmax_production
        && device_short_decode
        && parse_gib_budget("COLIBRI_V4_STATIONARY_RAW_DEVICE_RETENTION_GIB")?
            == 7 * 512 * 1024 * 1024
        && stationary_raw_retention.as_ref().map(|cache| {
            cache
                .raw_device
                .iter()
                .filter(|entry| entry.is_some())
                .count()
        }) != Some(8)
    {
        return Err(
            "certified final-head ledger reduced the required eight device retention entries"
                .into(),
        );
    }
    let mut stationary_mtp1_state = None;
    let mut stationary_mtp1_disabled = None::<String>;
    let mut stationary_mtp1_records = Vec::new();
    if stationary_mtp1_draft {
        generic_lifecycle
            .as_ref()
            .ok_or("stationary MTP lost generic lifecycle")?
            .assert_drained()?;
        stationary_mtp1_state = match MtpDraftState::new(stream) {
            Ok(state) => Some(state),
            Err(error) => {
                stationary_mtp1_disabled = Some(error.to_string());
                None
            }
        };
        for position in 0..prompt_len.saturating_sub(1) {
            let alignment = mtp1_alignment(prompt_len, position)?;
            let Some(state) = stationary_mtp1_state.as_mut() else {
                break;
            };
            match advance_mtp1_draft_state(
                module,
                stream,
                root,
                &hidden[position],
                token_ids[position],
                alignment.kv_write_pos,
                state,
            ) {
                Ok(draft) => stationary_mtp1_records.push(serde_json::json!({
                    "event": "mtp1_draft", "status": "DRAFT_STATE_ADVANCED",
                    "input_pos": alignment.input_pos, "teacher_hidden_pos": alignment.teacher_hidden_pos,
                    "candidate_pos": alignment.candidate_pos, "target_pos": alignment.teacher_target_pos,
                    "input_id": token_ids[position], "candidate": serde_json::Value::Null,
                    "target_teacher_token": token_ids.get(alignment.teacher_target_pos),
                    "scored": false, "prompt_forced": alignment.prompt_forced,
                    "mtp_kv_position": draft.kv_position, "mtp_kv_digest": format!("{:016x}", draft.kv_digest),
                    "mtp_kv_persistent_bytes": draft.persistent_bytes,
                })),
                Err(error) => {
                    let reason = error.to_string();
                    stationary_mtp1_records.push(serde_json::json!({
                        "event": "mtp1_draft", "status": "DRAFT_FAIL",
                        "candidate_pos": alignment.candidate_pos, "target_pos": alignment.teacher_target_pos,
                        "scored": false, "reason": reason,
                    }));
                    stationary_mtp1_disabled = Some(error.to_string());
                    stationary_mtp1_state = None;
                }
            }
        }
    }
    let final_scope_started_ns = phase_profile.then(|| timeline_elapsed_ns(request_started));
    let final_load_started = phase_profile.then(std::time::Instant::now);
    let final_weights = v4_state::load_final_head(root)?;
    let final_fixed_read_ns = final_load_started
        .map(|started| u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX));
    let final_decision = run_id_only_final_head(
        context,
        module,
        stream,
        &final_weights,
        &hidden[prompt_len - 1],
        certified_argmax_production,
        certified_argmax_audit,
    )?;
    let final_decision_elapsed_ns = timeline_elapsed_ns(request_started);
    if let Some(state) = stationary_mtp1_state.as_mut() {
        let position = prompt_len - 1;
        let alignment = mtp1_alignment(prompt_len, position)?;
        let Mtp1Qd4Attempt { result, loan } =
            run_with_mtp1_qd4_loan(qd4_lanes.as_deref_mut(), true, |borrowed_qd4_lane| {
                run_mtp1_draft_candidate(
                    context,
                    module,
                    stream,
                    root,
                    &final_weights,
                    borrowed_qd4_lane,
                    None,
                    &hidden[position],
                    token_ids[position],
                    alignment.kv_write_pos,
                    state,
                )
            });
        match result {
            Ok(draft) => {
                let mut record = serde_json::json!({
                    "event": "mtp1_draft", "status": "DRAFT_ALIGNED",
                    "input_pos": alignment.input_pos, "teacher_hidden_pos": alignment.teacher_hidden_pos,
                    "candidate_pos": alignment.candidate_pos, "target_pos": alignment.teacher_target_pos,
                    "input_id": token_ids[position], "candidate": draft.candidate,
                    "target_teacher_token": final_decision.token, "scored": true,
                    "prompt_forced": alignment.prompt_forced,
                    "match": draft.candidate == i64::try_from(final_decision.token)?,
                    "source_bytes": draft.source_bytes, "source_read_ns": draft.source_read_ns,
                    "h2d_ns": draft.h2d_ns, "h2d_bytes": draft.h2d_bytes,
                    "kernel_ns": draft.kernel_ns, "head_read_ns": draft.head_read_ns,
                    "head_h2d_ns": draft.head_h2d_ns, "head_kernel_ns": draft.head_kernel_ns,
                    "head_d2h_ns": draft.head_d2h_ns, "head_wall_ns": draft.head_wall_ns,
                    "total_ns": draft.total_ns, "mtp_kv_position": draft.kv_position,
                    "mtp_kv_digest": format!("{:016x}", draft.kv_digest),
                    "mtp_kv_persistent_bytes": draft.persistent_bytes,
                    "transient_expert_peak_bytes": draft.transient_expert_peak_bytes,
                    "device_peak_bytes": draft.device_peak_bytes,
                    "borrowed_qd4_lane": draft.borrowed_qd4_lane,
                });
                append_mtp1_qd4_telemetry(
                    &mut record,
                    Some(&draft.routes.ids),
                    Some(draft.routed_source_bytes),
                    Some(draft.routed_h2d_bytes),
                    loan,
                )?;
                stationary_mtp1_records.push(record);
            }
            Err(error) => {
                let reason = error.to_string();
                let mut record = serde_json::json!({
                    "event": "mtp1_draft", "status": "DRAFT_FAIL",
                    "candidate_pos": alignment.candidate_pos, "target_pos": alignment.teacher_target_pos,
                    "scored": false, "reason": reason,
                });
                append_mtp1_qd4_telemetry(&mut record, None, None, None, loan)?;
                stationary_mtp1_records.push(record);
                stationary_mtp1_disabled = Some(error.to_string());
                stationary_mtp1_state = None;
            }
        }
    }
    if stationary_mtp1_draft {
        if let Some(expected) = initial_teacher_state_before_mtp.as_deref() {
            let audit_wall_ns = check_generic_device_state_unchanged(
                stream,
                generic_states.as_ref().ok_or(
                    "post-MTP teacher-state clobber: stores disappeared after initial draft",
                )?,
                expected,
                prompt_len - 1,
            )?;
            generic_device_state_audit_wall_ns =
                generic_device_state_audit_wall_ns.saturating_add(audit_wall_ns);
        }
    }
    let (final_head_wall_ns, final_head_read_ns) = final_head_profile_metrics(&final_decision);
    let mut final_phase_profile = None;
    if let Some(request_start_ns) = final_scope_started_ns {
        let request_end_ns = timeline_elapsed_ns(request_started);
        let (profile, attributed_ns) = stationary_final_phase_profile_json(
            request_start_ns,
            request_end_ns,
            final_fixed_read_ns.expect("profiled final load must have a duration"),
            final_head_wall_ns,
            final_head_read_ns,
            qd4_enabled,
        )?;
        final_phase_profile = Some(profile);
        phase_profile_top_level_ns = phase_profile_top_level_ns.saturating_add(attributed_ns);
        phase_profile_final_outer_ns = request_end_ns - request_start_ns;
    }
    let argmax = final_decision.token;
    let generated_id = i64::try_from(argmax)?;
    let mut stop_reason = if generated_id == V4_EOS_TOKEN_ID {
        "eos"
    } else {
        "max_new_tokens"
    };
    let mut short_decode_records = Vec::new();
    let mut short_decode_finals = Vec::new();
    let mut short_decode_fixed_source_bytes = 0_usize;
    let mut short_decode_fixed_h2d_bytes = 0_usize;
    let mut short_decode_routed_source_bytes = 0_usize;
    let mut short_decode_routed_h2d_bytes = 0_usize;
    let mut generated_ids = if generic_device_stationary {
        Vec::new()
    } else {
        vec![generated_id]
    };
    if generic_device_stationary {
        let mut pipeline_cache = stationary_raw_retention
            .take()
            .ok_or("generic stationary lost its request-local fixed retention")?;
        let planned_raw_slots = pipeline_cache
            .device_selected
            .iter()
            .chain(&pipeline_cache.host_selected)
            .filter(|&&selected| selected)
            .count();
        let retained_raw_slots = pipeline_cache
            .raw_device
            .iter()
            .filter(|entry| entry.is_some())
            .count()
            .saturating_add(
                pipeline_cache
                    .raw_host
                    .iter()
                    .filter(|entry| entry.is_some())
                    .count(),
            );
        if planned_raw_slots == 0
            || retained_raw_slots != planned_raw_slots
            || pipeline_cache.raw_scratch.is_none()
            || pipeline_cache
                .raw_device_generation
                .iter()
                .any(|&value| value != 0)
            || pipeline_cache
                .raw_host_generation
                .iter()
                .any(|&value| value != 0)
            || pipeline_cache.raw_scratch_generation != 0
        {
            return Err("generic stationary pinned decode retention was not fully captured before worker start".into());
        }
        let transfer = context.new_stream()?;
        let first_scratch = pipeline_cache
            .raw_scratch
            .take()
            .ok_or("generic stationary pinned worker lost its retained decode scratch")?;
        let second_scratch = DeviceBuffer::<u16>::zeroed(
            stream,
            RAW_FIXED_SCRATCH_BYTES / std::mem::size_of::<u16>(),
        )?;
        let slots = vec![
            PinnedFixedSlot::new(context, first_scratch)?,
            PinnedFixedSlot::new(context, second_scratch)?,
        ];
        let restored_pipeline_cache = std::thread::scope(
            |scope| -> Result<FixedLayerCache, Box<dyn Error>> {
                let (request_tx, request_rx) =
                    std::sync::mpsc::sync_channel::<PinnedFixedRequest>(2);
                let (reply_tx, reply_rx) = std::sync::mpsc::sync_channel::<PinnedFixedReply>(2);
                let main = std::sync::Arc::clone(stream);
                let transfer_worker = std::sync::Arc::clone(&transfer);
                let mut worker = Some(scope.spawn(move || {
                    run_pinned_fixed_request_worker(
                        pipeline_cache,
                        slots,
                        root,
                        module,
                        &main,
                        &transfer_worker,
                        request_rx,
                        reply_tx,
                        bounded_readahead,
                    )
                }));
                let mut request_tx = Some(request_tx);
                let mut decision = greedy_argmax_decision(
                    prompt_len,
                    prompt_len - 1,
                    0,
                    greedy_steps,
                    generated_id,
                );
                let mut input_id = generated_id;
                let decode_result = (|| -> Result<(), Box<dyn Error>> {
                    loop {
                        match decision {
                            GreedyDecision::IgnorePrompt => {
                                return Err(
                                    "generic stationary final head ignored its prompt boundary"
                                        .into(),
                                );
                            }
                            GreedyDecision::EmitAndStop(reason) => {
                                generated_ids.push(input_id);
                                stop_reason = match reason {
                                    GreedyStopReason::Eos => "eos",
                                    GreedyStopReason::MaxNewTokens => "max_new_tokens",
                                };
                                break;
                            }
                            GreedyDecision::EmitAndContinue => generated_ids.push(input_id),
                        }
                        let token_position = prompt_len
                            .checked_add(generated_ids.len() - 1)
                            .ok_or("generic stationary decode position overflow")?;
                        let first = manifests
                            .first()
                            .ok_or("generic stationary decode has no layers")?;
                        let second = manifests
                            .get(1)
                            .ok_or("generic stationary decode has one layer")?;
                        request_tx
                            .as_ref()
                            .ok_or("generic stationary pinned sender disappeared")?
                            .send((None, Some((first.layer, input_id, None))))?;
                        request_tx
                            .as_ref()
                            .ok_or("generic stationary pinned sender disappeared")?
                            .send((
                                None,
                                Some((
                                    second.layer,
                                    input_id,
                                    manifests.get(2).map(|next| next.layer),
                                )),
                            ))?;
                        let embedding = v4_state::load(root, &[input_id])?.embedding;
                        let mut decode_hidden = Vec::with_capacity(v4_state::HC * v4_state::HIDDEN);
                        for _ in 0..v4_state::HC {
                            decode_hidden.extend_from_slice(&embedding);
                        }
                        for (layer_position, manifest) in manifests.iter().enumerate() {
                            let layer_scope_started_ns =
                                phase_profile.then(|| timeline_elapsed_ns(request_started));
                            let layer_prepare_started = phase_profile.then(std::time::Instant::now);
                            let reply = reply_rx.recv().map_err(|_| {
                            "generic stationary pinned worker response channel closed before layer preparation"
                        })?;
                            let mut prepared = reply
                            .map_err(|message| -> Box<dyn Error> { message.into() })?
                            .ok_or("generic stationary pinned worker returned an early drain acknowledgement")?;
                            if prepared.lease.layer != manifest.layer {
                                return Err(format!(
                                "generic stationary pinned worker returned layer {}, expected {}",
                                prepared.lease.layer, manifest.layer
                            )
                                .into());
                            }
                            prepared.wait_ready(stream)?;
                            let fixed = std::mem::take(&mut prepared.telemetry);
                            fixed_source_bytes =
                                fixed_source_bytes.saturating_add(fixed.fixed_source_bytes);
                            fixed_h2d_bytes = fixed_h2d_bytes.saturating_add(fixed.fixed_h2d_bytes);
                            let hash_ids = load_layer_hash_ids(root, manifest.layer, &[input_id])?;
                            let runtime = prepared.runtime();
                            let layer_prepare_ns = layer_prepare_started
                                .map(|started| {
                                    u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX)
                                })
                                .unwrap_or(0);
                            run_generic_device_state_audit_mirror(
                                generic_device_state_audit,
                                persistent.as_mut(),
                                manifest.layer,
                                &decode_hidden,
                                |host_state, mirror_hidden| {
                                    run_real_layer_attention_rows(
                                        module,
                                        stream,
                                        None,
                                        runtime,
                                        mirror_hidden,
                                        token_position,
                                        host_state,
                                        None,
                                        strict_scratch.as_mut(),
                                    )?;
                                    Ok(())
                                },
                            )?;
                            let audit_expected_compressed_prefix = if generic_device_state_audit
                                && (token_position + 1)
                                    .is_multiple_of(manifest.compression_ratio.get())
                            {
                                Some(
                                    persistent
                                        .as_ref()
                                        .and_then(|host| host.layer(manifest.layer))
                                        .ok_or("generic boundary audit host mirror lost its layer")?
                                        .compressed_kv_bf16(),
                                )
                            } else {
                                None
                            };
                            let states = generic_states
                                .as_mut()
                                .ok_or("generic stationary stores disappeared during decode")?;
                            let state = states
                                .get_mut(manifest.layer)
                                .and_then(Option::as_mut)
                                .ok_or("generic stationary decode layer store disappeared")?;
                            let scratch = generic_scratch
                                .as_mut()
                                .ok_or("generic stationary scratch disappeared during decode")?;
                            let lifecycle = generic_lifecycle
                                .as_mut()
                                .ok_or("generic stationary lifecycle disappeared during decode")?;
                            lifecycle.begin_layer(manifest.layer, token_position)?;
                            lifecycle.acquire_lease();
                            let before = state.committed_rows;
                            let run = {
                                let runtime = prepared.runtime();
                                run_real_layer_rows(
                                    context,
                                    module,
                                    stream,
                                    None,
                                    runtime,
                                    qd4_lanes.as_deref_mut(),
                                    None,
                                    None,
                                    if runtime.weights.hash_router {
                                        &hash_ids
                                    } else {
                                        &[]
                                    },
                                    &decode_hidden,
                                    &[input_id],
                                    token_position,
                                    None,
                                    None,
                                    None,
                                    false,
                                    None,
                                    false,
                                    Some((state, scratch, audit_expected_compressed_prefix)),
                                    None,
                                None,
                                )
                            };
                            let mut run = match run {
                                Ok(run) => run,
                                Err(error) => {
                                    lifecycle.release_lease()?;
                                    lifecycle.poison();
                                    return Err(error);
                                }
                            };
                            if phase_profile {
                                run.stage.layer_prepare_ns = layer_prepare_ns;
                                run.stage.layer_prepare_count = 1;
                            }
                            if let Err(error) =
                                lifecycle.commit_layer(manifest.layer, token_position)
                            {
                                lifecycle.release_lease()?;
                                lifecycle.poison();
                                return Err(error);
                            }
                            if state.committed_rows != before + 1 {
                                lifecycle.release_lease()?;
                                lifecycle.poison();
                                return Err(
                                    "generic stationary decode committed a noncontiguous KV row"
                                        .into(),
                                );
                            }
                            if let Err(error) = prepared.record_runtime_done(stream) {
                                lifecycle.release_lease()?;
                                lifecycle.poison();
                                return Err(error);
                            }
                            let next = manifests.get(layer_position + 2).map(|next| {
                                (
                                    next.layer,
                                    input_id,
                                    manifests.get(layer_position + 3).map(|next| next.layer),
                                )
                            });
                            let tx = match request_tx.as_ref() {
                                Some(tx) => tx,
                                None => {
                                    lifecycle.release_lease()?;
                                    lifecycle.poison();
                                    return Err(
                                        "generic stationary pinned sender disappeared".into()
                                    );
                                }
                            };
                            if let Err(error) = tx.send((Some(prepared), next)) {
                                lifecycle.release_lease()?;
                                lifecycle.poison();
                                return Err(error.into());
                            }
                            lifecycle.release_lease()?;
                            routed_source_bytes =
                                routed_source_bytes.saturating_add(run.stage.routed_source_bytes);
                            routed_h2d_bytes =
                                routed_h2d_bytes.saturating_add(run.stage.routed_h2d_bytes);
                            routed_expert_loads =
                                routed_expert_loads.saturating_add(run.stage.routed_expert_loads);
                            let mut record = serde_json::json!({
                                "event": "layer", "token_position": token_position, "token_id": input_id,
                                "layer": manifest.layer, "generic_stationary": true,
                                "kv_rows_before": before, "kv_rows_after": state.committed_rows,
                                "fixed_source_bytes": fixed.fixed_source_bytes,
                                "fixed_h2d_bytes": fixed.fixed_h2d_bytes,
                                "fixed_cache_status": fixed.cache_status,
                                "hidden_bf16_digest": format!("{:016x}", fnv1a_u16(&run.output)),
                            });
                            if let Some(request_start_ns) = layer_scope_started_ns {
                                let request_end_ns = timeline_elapsed_ns(request_started);
                                let (profile, attributed_ns, outer_ns) =
                                    stationary_layer_phase_profile_json(
                                        request_start_ns,
                                        request_end_ns,
                                        Some(&fixed),
                                        &run.stage,
                                        run.indexer_topk_gpu_ns,
                                        qd4_enabled,
                                    )?;
                                record
                                    .as_object_mut()
                                    .ok_or("generic decode layer record is not an object")?
                                    .insert("phase_profile".into(), profile);
                                phase_profile_top_level_ns =
                                    phase_profile_top_level_ns.saturating_add(attributed_ns);
                                one_token_decode_phase_profile_layer_outer_ns =
                                    one_token_decode_phase_profile_layer_outer_ns
                                        .saturating_add(outer_ns);
                            }
                            short_decode_records.push(record);
                            decode_hidden = run.output;
                        }
                        let drained = reply_rx.recv().map_err(|_| {
                        "generic stationary pinned worker response channel closed before token drain"
                    })?;
                        let worker_drained = drained
                            .map_err(|message| -> Box<dyn Error> { message.into() })?
                            .is_none();
                        if !worker_drained {
                            return Err(
                            "generic stationary pinned worker returned a layer instead of drain"
                                .into(),
                        );
                        }
                        generic_lifecycle
                            .as_ref()
                            .ok_or("generic stationary lifecycle disappeared at token drain")?
                            .assert_drained()?;
                        let teacher_state_before_mtp = if generic_device_state_audit {
                            let (audit, snapshot) = generic_device_state_audit_head(
                                stream,
                                generic_states.as_ref().ok_or(
                                    "generic device state audit stores disappeared during decode",
                                )?,
                                persistent.as_ref().ok_or(
                                    "generic device state audit host mirror disappeared during decode",
                                )?,
                            generic_lifecycle.as_ref().ok_or(
                                "generic device state audit lifecycle disappeared during decode",
                            )?,
                            generic_scratch.as_ref().ok_or(
                                "generic device state audit scratch disappeared during decode",
                            )?,
                            token_position,
                            )
                            .map_err(|error| -> Box<dyn Error> {
                                format!("pre-MTP generic device state audit failed: {error}").into()
                            })?;
                            generic_device_state_audit_wall_ns = generic_device_state_audit_wall_ns
                                .saturating_add(
                                    audit["audit_wall_ns"].as_u64().unwrap_or(u64::MAX),
                                );
                            generic_device_state_audit_heads.push(audit);
                            Some(snapshot)
                        } else {
                            None
                        };
                        let next_final_scope_started_ns =
                            phase_profile.then(|| timeline_elapsed_ns(request_started));
                        let next = run_id_only_final_head(
                            context,
                            module,
                            stream,
                            &final_weights,
                            &decode_hidden,
                            certified_argmax_production,
                            certified_argmax_audit,
                        )?;
                        let next_decision_elapsed_ns = timeline_elapsed_ns(request_started);
                        let (next_final_wall_ns, next_final_read_ns) =
                            final_head_profile_metrics(&next);
                        let mut next_profile = None;
                        if let Some(request_start_ns) = next_final_scope_started_ns {
                            let (profile, attributed_ns) = stationary_final_phase_profile_json(
                                request_start_ns,
                                next_decision_elapsed_ns,
                                0,
                                next_final_wall_ns,
                                next_final_read_ns,
                                qd4_enabled,
                            )?;
                            next_profile = Some(profile);
                            phase_profile_top_level_ns =
                                phase_profile_top_level_ns.saturating_add(attributed_ns);
                            one_token_decode_phase_profile_final_outer_ns =
                                one_token_decode_phase_profile_final_outer_ns
                                    .saturating_add(next_decision_elapsed_ns - request_start_ns);
                        }
                        if let Some(state) = stationary_mtp1_state.as_mut() {
                            generic_lifecycle
                                .as_ref()
                                .ok_or("stationary MTP lost lifecycle after worker drain")?
                                .assert_drained()?;
                            let alignment = mtp1_alignment(prompt_len, token_position)?;
                            let Mtp1Qd4Attempt { result, loan } = run_with_mtp1_qd4_loan(
                                qd4_lanes.as_deref_mut(),
                                worker_drained,
                                |borrowed_qd4_lane| {
                                    run_mtp1_draft_candidate(
                                        context,
                                        module,
                                        stream,
                                        root,
                                        &final_weights,
                                        borrowed_qd4_lane,
                                        None,
                                        &decode_hidden,
                                        input_id,
                                        alignment.kv_write_pos,
                                        state,
                                    )
                                },
                            );
                            match result {
                                Ok(draft) => {
                                    let mut record = serde_json::json!({
                                        "event": "mtp1_draft", "status": "DRAFT_ALIGNED",
                                        "input_pos": alignment.input_pos, "teacher_hidden_pos": alignment.teacher_hidden_pos,
                                        "candidate_pos": alignment.candidate_pos, "target_pos": alignment.teacher_target_pos,
                                        "input_id": input_id, "candidate": draft.candidate,
                                        "target_teacher_token": next.token, "scored": true,
                                        "prompt_forced": alignment.prompt_forced,
                                        "match": draft.candidate == i64::try_from(next.token)?,
                                        "source_bytes": draft.source_bytes, "source_read_ns": draft.source_read_ns,
                                        "h2d_ns": draft.h2d_ns, "h2d_bytes": draft.h2d_bytes,
                                        "kernel_ns": draft.kernel_ns, "head_read_ns": draft.head_read_ns,
                                        "head_h2d_ns": draft.head_h2d_ns, "head_kernel_ns": draft.head_kernel_ns,
                                        "head_d2h_ns": draft.head_d2h_ns, "head_wall_ns": draft.head_wall_ns,
                                        "total_ns": draft.total_ns, "mtp_kv_position": draft.kv_position,
                                        "mtp_kv_digest": format!("{:016x}", draft.kv_digest),
                                        "mtp_kv_persistent_bytes": draft.persistent_bytes,
                                        "transient_expert_peak_bytes": draft.transient_expert_peak_bytes,
                                        "device_peak_bytes": draft.device_peak_bytes,
                                        "borrowed_qd4_lane": draft.borrowed_qd4_lane,
                                    });
                                    append_mtp1_qd4_telemetry(
                                        &mut record,
                                        Some(&draft.routes.ids),
                                        Some(draft.routed_source_bytes),
                                        Some(draft.routed_h2d_bytes),
                                        loan,
                                    )?;
                                    stationary_mtp1_records.push(record);
                                }
                                Err(error) => {
                                    let reason = error.to_string();
                                    let mut record = serde_json::json!({
                                        "event": "mtp1_draft", "status": "DRAFT_FAIL",
                                        "candidate_pos": alignment.candidate_pos, "target_pos": alignment.teacher_target_pos,
                                        "scored": false, "reason": reason,
                                    });
                                    append_mtp1_qd4_telemetry(&mut record, None, None, None, loan)?;
                                    stationary_mtp1_records.push(record);
                                    stationary_mtp1_disabled = Some(error.to_string());
                                    stationary_mtp1_state = None;
                                }
                            }
                        }
                        if stationary_mtp1_draft {
                            if let Some(expected) = teacher_state_before_mtp.as_deref() {
                                generic_device_state_audit_wall_ns =
                                    generic_device_state_audit_wall_ns.saturating_add(
                                        check_generic_device_state_unchanged(
                                            stream,
                                            generic_states.as_ref().ok_or(
                                                "post-MTP teacher-state clobber: stores disappeared during decode",
                                            )?,
                                            expected,
                                            token_position,
                                        )?,
                                    );
                            }
                        }
                        input_id = i64::try_from(next.token)?;
                        decision = greedy_argmax_decision(
                            prompt_len,
                            token_position,
                            generated_ids.len(),
                            greedy_steps,
                            input_id,
                        );
                        short_decode_finals.push((
                            token_position,
                            generated_ids
                                .last()
                                .copied()
                                .ok_or("generic stationary lost emitted token")?,
                            next,
                            decode_hidden,
                            next_profile,
                            next_decision_elapsed_ns,
                        ));
                    }
                    Ok(())
                })();
                drop(request_tx.take());
                let cache = worker
                    .take()
                    .ok_or("generic stationary pinned worker handle disappeared")?
                    .join()
                    .map_err(|_| "generic stationary pinned worker panicked during request shutdown")?
                    .map_err(|message| -> Box<dyn Error> {
                        format!("generic stationary pinned worker failed during request shutdown: {message}").into()
                    })?;
                decode_result?;
                Ok(cache)
            },
        )?;
        if decode_overlay_uses_decode_target(
            short_decode_records.len(),
            manifests.len(),
            restored_pipeline_cache.decode_device_promotions.len(),
            restored_pipeline_cache.decode_host_promotions.len(),
        )? {
            restored_pipeline_cache.validate_decode_overlay_complete()?;
        } else {
            restored_pipeline_cache.validate_prefill_residency_complete()?;
        }
        let persistent_device_bytes = generic_states
            .as_ref()
            .ok_or("generic stationary stores disappeared at overlay ledger")?
            .iter()
            .map(|state| {
                state
                    .as_ref()
                    .map_or(0, DeviceLayerPersistentState::num_bytes)
            })
            .try_fold(0_usize, |total, bytes| total.checked_add(bytes))
            .ok_or("generic stationary overlay persistent byte overflow")?;
        let base_device_bytes = persistent_device_bytes
            .checked_add(
                generic_scratch
                    .as_ref()
                    .map_or(0, DeviceRequestScratch::num_bytes),
            )
            .and_then(|bytes| bytes.checked_add(usize::from(qd4_enabled) * V4_QD4_DEVICE_BYTES))
            .and_then(|bytes| bytes.checked_add(restored_pipeline_cache.device_resident_bytes))
            .and_then(|bytes| bytes.checked_add(PINNED_FIXED_PIPELINE_WORKER_SCRATCH_DEVICE_BYTES))
            .ok_or("generic stationary overlay base device ledger overflow")?;
        let overlay_device_bytes = base_device_bytes
            .checked_add(stationary_mtp1_peak_reserve_bytes)
            .ok_or("generic stationary overlay MTP device ledger overflow")?
            .max(
                base_device_bytes
                    .checked_add(PINNED_FIXED_PIPELINE_WORKER_PREPARED_DEVICE_BYTES)
                    .ok_or("generic stationary overlay prepared device ledger overflow")?,
            );
        let audit_mirror_host_bytes = if generic_device_state_audit {
            generic_stationary_audit_mirror_host_bytes(
                manifests,
                generic_plan.ok_or("generic stationary overlay audit has no request plan")?,
            )?
        } else {
            0
        };
        let overlay_host_bytes = restored_pipeline_cache
            .host_resident_bytes
            .checked_add(audit_mirror_host_bytes)
            .and_then(|bytes| {
                bytes.checked_add(
                    usize::from(stationary_mtp1_draft) * CERTIFIED_ARGMAX_HOST_TEMP_BYTES,
                )
            })
            .ok_or("generic stationary overlay host ledger overflow")?;
        let overlay_pinned_bytes = usize::from(qd4_enabled) * V4_QD4_PINNED_HOST_BYTES
            + PINNED_FIXED_PIPELINE_WORKER_PINNED_HOST_BYTES
            + mtp1_stationary_pinned_host_reserve_bytes(qd4_enabled);
        generic_decode_overlay_peak = Some((
            overlay_device_bytes,
            overlay_host_bytes,
            overlay_pinned_bytes,
        ));
        generic_lifecycle
            .as_mut()
            .ok_or("generic stationary lifecycle disappeared at overlay ledger")?
            .observe_peaks(
                overlay_device_bytes,
                overlay_host_bytes,
                overlay_pinned_bytes,
            );
        stationary_raw_retention = Some(restored_pipeline_cache);
        generic_lifecycle
            .as_ref()
            .ok_or("generic stationary lifecycle disappeared at final head")?
            .assert_drained()?;
        if qd4_enabled {
            let prefill_route_executions = qd4_expected_routed_loads(manifests.len(), prompt_len)?;
            let decode_source_loads = short_decode_records
                .len()
                .checked_mul(v4_router::V4_TOPK)
                .ok_or("generic stationary QD4 decode load count overflow")?;
            let expected_source_loads =
                if let Some(telemetry) = qd4_immutable_cache_telemetry.as_ref() {
                    let expected_d2d_bytes = prefill_route_executions
                        .checked_mul(v4_expert::V4_ROUTED_EXPERT_BYTES)
                        .ok_or("generic stationary QD4 immutable cache D2D byte count overflow")?;
                    if telemetry.route_executions != prefill_route_executions
                    || telemetry.source_loads != telemetry.expert_union_loads
                    || telemetry.source_bytes
                        != telemetry
                            .source_loads
                            .checked_mul(v4_expert::V4_ROUTED_EXPERT_BYTES)
                            .ok_or(
                                "generic stationary QD4 immutable cache source byte count overflow",
                            )?
                    || telemetry.h2d_loads != telemetry.source_loads
                    || telemetry.h2d_bytes != telemetry.source_bytes
                    || telemetry.hits
                        != prefill_route_executions.saturating_sub(telemetry.source_loads)
                    || telemetry.d2d_copies != prefill_route_executions
                    || telemetry.d2d_bytes != expected_d2d_bytes
                    || telemetry.max_occupancy > V4_QD4_IMMUTABLE_EXPERT_CACHE_SLOTS
                    || telemetry.completed_layers != manifests.len()
                    || telemetry.worker_errors != 0
                    || telemetry.fallbacks != 0
                {
                    return Err(
                        "generic stationary QD4 immutable cache prefill telemetry did not close"
                            .into(),
                    );
                }
                    telemetry
                        .source_loads
                        .checked_add(decode_source_loads)
                        .ok_or("generic stationary QD4 source load count overflow")?
                } else {
                    prefill_route_executions
                        .checked_add(decode_source_loads)
                        .ok_or("generic stationary QD4 source load count overflow")?
                };
            let expected_bytes = expected_source_loads
                .checked_mul(v4_expert::V4_ROUTED_EXPERT_BYTES)
                .ok_or("generic stationary QD4 routed byte count overflow")?;
            if routed_expert_loads != expected_source_loads
                || routed_source_bytes != expected_bytes
                || routed_h2d_bytes != expected_bytes
            {
                return Err(
                    "generic stationary QD4 routed load/byte accounting diverged from its six-route contract"
                        .into(),
                );
            }
        }
    }
    if device_short_decode {
        let states = device_states
            .as_mut()
            .ok_or("short device decode lost its persistent state vector")?;
        if states.len() != v4_state::V4_LAYER_COUNT {
            return Err("short device decode state vector is not full61".into());
        }
        let rope_lut = device_rope_lut
            .as_ref()
            .ok_or("short device decode lost its request RoPE LUT")?;
        let mut pipeline_cache = pinned_fixed_pipeline
            .then(|| {
                stationary_raw_retention
                    .take()
                    .ok_or("pinned fixed pipeline lost its request-local cache")
            })
            .transpose()?;
        let pipeline_transfer = pinned_fixed_pipeline
            .then(|| context.new_stream())
            .transpose()?;
        let mut pipeline_slots = Vec::new();
        if pinned_fixed_pipeline {
            let first_scratch = pipeline_cache
                .as_mut()
                .and_then(|cache| cache.raw_scratch.take())
                .ok_or("pinned fixed pipeline lost its retained decode scratch")?;
            let second_scratch = DeviceBuffer::<u16>::zeroed(
                stream,
                RAW_FIXED_SCRATCH_BYTES / std::mem::size_of::<u16>(),
            )?;
            pipeline_slots.push(PinnedFixedSlot::new(context, first_scratch)?);
            pipeline_slots.push(PinnedFixedSlot::new(context, second_scratch)?);
        }
        let restored_pipeline_cache = std::thread::scope(
            |scope| -> Result<Option<FixedLayerCache>, Box<dyn Error>> {
                let mut request_tx = None;
                let mut response_rx = None;
                let mut worker = if pinned_fixed_pipeline {
                    let (tx, rx) = std::sync::mpsc::sync_channel::<PinnedFixedRequest>(2);
                    let (reply_tx, reply_rx) = std::sync::mpsc::sync_channel::<
                        Result<Option<PinnedFixedPrepared>, String>,
                    >(2);
                    request_tx = Some(tx);
                    response_rx = Some(reply_rx);
                    let cache = pipeline_cache
                        .take()
                        .ok_or("pinned fixed worker lost its cache")?;
                    let transfer = pipeline_transfer
                        .as_ref()
                        .ok_or("pinned fixed worker lost its transfer stream")?;
                    let main = std::sync::Arc::clone(stream);
                    Some(scope.spawn(move || {
                        run_pinned_fixed_request_worker(
                            cache,
                            pipeline_slots,
                            root,
                            module,
                            &main,
                            transfer,
                            rx,
                            reply_tx,
                            bounded_readahead,
                        )
                    }))
                } else {
                    None
                };
                let mut decode_input_id = generated_id;
                for token_position in prompt_len..short_decode_rows {
                    if let Some(tx) = request_tx.as_ref() {
                        let first = manifests
                            .first()
                            .ok_or("short device decode has no layers")?;
                        let second = manifests
                            .get(1)
                            .ok_or("short device decode has one layer")?;
                        tx.send((None, Some((first.layer, decode_input_id, None))))?;
                        tx.send((
                            None,
                            Some((
                                second.layer,
                                decode_input_id,
                                manifests.get(2).map(|next| next.layer),
                            )),
                        ))?;
                    }
                    let embedding = v4_state::load(root, &[decode_input_id])?.embedding;
                    let mut decode_hidden = Vec::with_capacity(v4_state::HC * v4_state::HIDDEN);
                    for _ in 0..v4_state::HC {
                        decode_hidden.extend_from_slice(&embedding);
                    }
                    for (layer_position, manifest) in manifests.iter().enumerate() {
                        let layer_scope_started_ns =
                            phase_profile.then(|| timeline_elapsed_ns(request_started));
                        let layer_prepare_started = phase_profile.then(std::time::Instant::now);
                        let mut pipeline_prepared = None;
                        let mut runtime_lease = None;
                        let fixed = if let Some(rx) = response_rx.as_ref() {
                            let reply = match rx.recv() {
                                Ok(reply) => reply,
                                Err(_) => {
                                    drop(request_tx.take());
                                    return match worker
                                            .take()
                                            .ok_or("pinned fixed worker handle disappeared")?
                                            .join()
                                        {
                                            Ok(Err(message)) => Err(format!(
                                                "pinned fixed worker failed before layer {} preparation: {message}",
                                                manifest.layer
                                            )
                                            .into()),
                                            Ok(Ok(_)) => Err(format!(
                                                "pinned fixed worker response channel closed before layer {} preparation after a successful worker exit",
                                                manifest.layer
                                            )
                                            .into()),
                                            Err(_) => Err(format!(
                                                "pinned fixed worker panicked before layer {} preparation",
                                                manifest.layer
                                            )
                                            .into()),
                                        };
                                }
                            };
                            let mut prepared = reply
                                .map_err(|message| -> Box<dyn Error> { message.into() })?
                                .ok_or(
                                    "pinned fixed worker returned an early drain acknowledgement",
                                )?;
                            if prepared.lease.layer != manifest.layer {
                                return Err(format!(
                                    "pinned fixed worker returned layer {}, expected {}",
                                    prepared.lease.layer, manifest.layer
                                )
                                .into());
                            }
                            prepared.wait_ready(stream)?;
                            let fixed = std::mem::take(&mut prepared.telemetry);
                            pipeline_prepared = Some(prepared);
                            fixed
                        } else {
                            let (lease, fixed) = stationary_raw_retention
                                .as_mut()
                                .ok_or(
                                    "short device decode lost its request-local fixed retention",
                                )?
                                .access_stationary_raw(
                                    root,
                                    manifest.layer,
                                    &[decode_input_id],
                                    module,
                                    stream,
                                )?;
                            runtime_lease = Some(lease);
                            fixed
                        };
                        let runtime = if let Some(prepared) = pipeline_prepared.as_ref() {
                            prepared.runtime()
                        } else {
                            runtime_lease
                                .as_ref()
                                .ok_or("short device decode lost its fixed runtime lease")?
                        };
                        let hash_ids =
                            load_layer_hash_ids(root, manifest.layer, &[decode_input_id])?;
                        let layer_prepare_ns = layer_prepare_started
                            .map(|started| {
                                u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX)
                            })
                            .unwrap_or(0);
                        let mut oracle = None;
                        if device_attention_audit {
                            persistent
                                .as_mut()
                                .ok_or("short device decode audit lost its host persistent state")?
                                .transaction(manifest.layer, |host_state| {
                                    let run = run_real_layer_rows(
                                        context,
                                        module,
                                        stream,
                                        None,
                                        &runtime,
                                        qd4_lanes.as_deref_mut(),
                                        None,
                                        None,
                                        if runtime.weights.hash_router {
                                            &hash_ids
                                        } else {
                                            &[]
                                        },
                                        &decode_hidden,
                                        &[decode_input_id],
                                        token_position,
                                        Some(host_state),
                                        None,
                                        None,
                                        false,
                                        None,
                                        true,
                                        None,
                                        strict_scratch.as_mut(),
                                    None,
                                    )?;
                                    oracle = Some((run, host_persistent_state_audit(host_state)));
                                    Ok(())
                                })?;
                        }
                        let state = states
                            .get_mut(manifest.layer)
                            .and_then(Option::as_mut)
                            .ok_or(
                                "short device decode layer state was not retained during prefill",
                            )?;
                        if state.layer != manifest.layer || state.committed_rows != token_position {
                            return Err(
                                "short device decode persistent-state continuity mismatch".into()
                            );
                        }
                        let kv_rows_before = state.committed_rows;
                        let mut run = run_real_layer_rows(
                            context,
                            module,
                            stream,
                            None,
                            &runtime,
                            qd4_lanes.as_deref_mut(),
                            None,
                            None,
                            if runtime.weights.hash_router {
                                &hash_ids
                            } else {
                                &[]
                            },
                            &decode_hidden,
                            &[decode_input_id],
                            token_position,
                            None,
                            Some(state),
                            Some(rope_lut),
                            device_attention_audit,
                            None,
                            device_attention_audit,
                            None,
                            None,
                        None,
                        )?;
                        if phase_profile {
                            run.stage.layer_prepare_ns = layer_prepare_ns;
                            run.stage.layer_prepare_count = 1;
                        }
                        let observed_state = device_attention_audit
                            .then(|| device_persistent_state_audit(stream, state))
                            .transpose()?;
                        let kv_rows_after = state.committed_rows;
                        if kv_rows_after != kv_rows_before + 1 {
                            return Err(
                                "short device decode committed a noncontiguous KV row".into()
                            );
                        }
                        if run.routes.ids.len() != v4_router::V4_TOPK {
                            return Err("short device decode produced incomplete routes".into());
                        }
                        if let Some((expected_run, expected_state)) = oracle.as_ref() {
                            let actual_state = observed_state.as_ref().ok_or(
                                "short device decode audit omitted device persistent state",
                            )?;
                            check_short_decode_full_layer_audit(
                                manifest.layer,
                                token_position,
                                expected_run,
                                expected_state,
                                &run,
                                actual_state,
                            )?;
                        }
                        short_decode_fixed_source_bytes = short_decode_fixed_source_bytes
                            .checked_add(fixed.fixed_source_bytes)
                            .ok_or("short device decode fixed source byte overflow")?;
                        short_decode_fixed_h2d_bytes = short_decode_fixed_h2d_bytes
                            .checked_add(fixed.fixed_h2d_bytes)
                            .ok_or("short device decode fixed H2D byte overflow")?;
                        short_decode_routed_source_bytes = short_decode_routed_source_bytes
                            .checked_add(run.stage.routed_source_bytes)
                            .ok_or("short device decode routed source byte overflow")?;
                        short_decode_routed_h2d_bytes = short_decode_routed_h2d_bytes
                            .checked_add(run.stage.routed_h2d_bytes)
                            .ok_or("short device decode routed H2D byte overflow")?;
                        let topk_observed = run.attention_topk.is_some();
                        let mut record = serde_json::json!({
                            "event": "layer",
                            "token_position": token_position,
                            "token_id": decode_input_id,
                            "layer": manifest.layer,
                            "compression_ratio": manifest.compression_ratio.get(),
                            "hidden_bf16_digest": format!("{:016x}", fnv1a_u16(&run.output)),
                            "kv_cache_bf16_digest": observed_state.as_ref().map(|value| format!("{:016x}", fnv1a_u16(&value.kv))),
                            "main_compressed_kv_bf16_digest": observed_state.as_ref().map(|value| format!("{:016x}", fnv1a_u16(&value.compressed_kv))),
                            "main_compressor_kv_state_f32_digest": observed_state.as_ref().map(|value| format!("{:016x}", fnv1a_f32(&value.main_kv))),
                            "main_compressor_score_state_f32_digest": observed_state.as_ref().map(|value| format!("{:016x}", fnv1a_f32(&value.main_score))),
                            "indexer_compressed_kv_bf16_digest": observed_state.as_ref().map(|value| format!("{:016x}", fnv1a_u16(&value.indexer_kv))),
                            "indexer_compressor_kv_state_f32_digest": observed_state.as_ref().and_then(|value| value.indexer_main_kv.as_ref().map(|digest| format!("{:016x}", fnv1a_f32(digest)))),
                            "indexer_compressor_score_state_f32_digest": observed_state.as_ref().and_then(|value| value.indexer_main_score.as_ref().map(|digest| format!("{:016x}", fnv1a_f32(digest)))),
                            "route_ids": run.routes.ids,
                            "route_weights_f32": run.routes.weights,
                            "topk_observed": topk_observed,
                            "attention_topk_digest": topk_observed.then(|| format!("{:016x}", run.attention_topk_digest)),
                            "attention_topk_per_row": topk_observed.then_some(run.attention_topk_per_row),
                            "kv_rows_before": kv_rows_before,
                            "kv_rows_after": kv_rows_after,
                            "fixed_source_bytes": fixed.fixed_source_bytes,
                            "fixed_h2d_bytes": fixed.fixed_h2d_bytes,
                            "fixed_cache_status": fixed.cache_status,
                            "bounded_readahead": bounded_readahead_json(
                                fixed.bounded_readahead.as_ref(),
                                fixed.fixed_read_alloc_ns,
                            ),
                            "routed_source_bytes": run.stage.routed_source_bytes,
                            "routed_h2d_bytes": run.stage.routed_h2d_bytes,
                            "qd4_immutable_cache": false,
                        });
                        if let Some(request_start_ns) = layer_scope_started_ns {
                            let request_end_ns = timeline_elapsed_ns(request_started);
                            let (profile, attributed_ns, outer_ns) =
                                stationary_layer_phase_profile_json(
                                    request_start_ns,
                                    request_end_ns,
                                    Some(&fixed),
                                    &run.stage,
                                    run.indexer_topk_gpu_ns,
                                    qd4_enabled,
                                )?;
                            record
                                .as_object_mut()
                                .ok_or("short decode layer record is not an object")?
                                .insert("phase_profile".into(), profile);
                            phase_profile_top_level_ns =
                                phase_profile_top_level_ns.saturating_add(attributed_ns);
                            one_token_decode_phase_profile_layer_outer_ns =
                                one_token_decode_phase_profile_layer_outer_ns
                                    .saturating_add(outer_ns);
                        }
                        short_decode_records.push(record);
                        if let Some(tx) = request_tx.as_ref() {
                            let next = manifests.get(layer_position + 2).map(|next| {
                                (
                                    next.layer,
                                    decode_input_id,
                                    manifests.get(layer_position + 3).map(|next| next.layer),
                                )
                            });
                            tx.send((
                                Some(
                                    pipeline_prepared
                                        .take()
                                        .ok_or("pinned fixed layer lost its prepared slot")?,
                                ),
                                next,
                            ))?;
                        } else {
                            stationary_raw_retention
                                .as_mut()
                                .ok_or(
                                    "short device decode lost its request-local fixed retention",
                                )?
                                .restore_stationary_raw(
                                    runtime_lease
                                        .take()
                                        .ok_or("short device decode lost its runtime lease")?,
                                )?;
                        }
                        decode_hidden = run.output;
                    }
                    if let Some(rx) = response_rx.as_ref() {
                        let reply = match rx.recv() {
                            Ok(reply) => reply,
                            Err(_) => {
                                drop(request_tx.take());
                                return match worker
                                    .take()
                                    .ok_or("pinned fixed worker handle disappeared")?
                                    .join()
                                {
                                    Ok(Err(message)) => Err(format!(
                                        "pinned fixed worker failed before token position {token_position} drain: {message}"
                                    )
                                    .into()),
                                    Ok(Ok(_)) => Err(format!(
                                        "pinned fixed worker response channel closed before token position {token_position} drain after a successful worker exit"
                                    )
                                    .into()),
                                    Err(_) => Err(format!(
                                        "pinned fixed worker panicked before token position {token_position} drain"
                                    )
                                    .into()),
                                };
                            }
                        };
                        if reply
                            .map_err(|message| -> Box<dyn Error> { message.into() })?
                            .is_some()
                        {
                            return Err(
                                "pinned fixed worker returned a layer instead of drain".into()
                            );
                        }
                    }
                    let decode_final_scope_started_ns =
                        phase_profile.then(|| timeline_elapsed_ns(request_started));
                    let decode_decision = run_id_only_final_head(
                        context,
                        module,
                        stream,
                        &final_weights,
                        &decode_hidden,
                        certified_argmax_production,
                        certified_argmax_audit,
                    )?;
                    let decode_decision_elapsed_ns = timeline_elapsed_ns(request_started);
                    let (decode_final_wall_ns, decode_final_read_ns) =
                        final_head_profile_metrics(&decode_decision);
                    let mut decode_profile = None;
                    if let Some(request_start_ns) = decode_final_scope_started_ns {
                        let request_end_ns = timeline_elapsed_ns(request_started);
                        let (profile, attributed_ns) = stationary_final_phase_profile_json(
                            request_start_ns,
                            request_end_ns,
                            0,
                            decode_final_wall_ns,
                            decode_final_read_ns,
                            qd4_enabled,
                        )?;
                        decode_profile = Some(profile);
                        phase_profile_top_level_ns =
                            phase_profile_top_level_ns.saturating_add(attributed_ns);
                        one_token_decode_phase_profile_final_outer_ns =
                            one_token_decode_phase_profile_final_outer_ns
                                .saturating_add(request_end_ns - request_start_ns);
                    }
                    let next_id = i64::try_from(decode_decision.token)?;
                    generated_ids.push(next_id);
                    short_decode_finals.push((
                        token_position,
                        decode_input_id,
                        decode_decision,
                        decode_hidden,
                        decode_profile,
                        decode_decision_elapsed_ns,
                    ));
                    decode_input_id = next_id;
                }
                drop(request_tx.take());
                let cache = if let Some(worker) = worker {
                    Some(
                        worker
                            .join()
                            .map_err(|_| "pinned fixed worker panicked during request shutdown")?
                            .map_err(|message| -> Box<dyn Error> {
                                format!(
                                    "pinned fixed worker failed during request shutdown: {message}"
                                )
                                .into()
                            })?,
                    )
                } else {
                    None
                };
                Ok(cache)
            },
        )?;
        if let Some(cache) = restored_pipeline_cache {
            stationary_raw_retention = Some(cache);
        }
    }
    if generic_device_state_audit {
        let expected_positions = generic_plan
            .ok_or("generic device state audit request plan disappeared")?
            .actual_decision_positions(generated_ids.len())?
            .collect::<Vec<_>>();
        let actual_positions = generic_device_state_audit_heads
            .iter()
            .map(|head| -> Result<usize, Box<dyn Error>> {
                Ok(usize::try_from(head["position"].as_u64().ok_or(
                    "generic device state audit head lost its position",
                )?)?)
            })
            .collect::<Result<Vec<_>, _>>()?;
        if actual_positions != expected_positions {
            return Err(
                "generic device state audit did not close its actual decision-position prefix"
                    .into(),
            );
        }
    }
    let request_e2e_ns = timeline_elapsed_ns(request_started)
        .checked_sub(generic_device_state_audit_wall_ns)
        .ok_or("generic device state audit wall time exceeds request wall time")?;
    let (phase_profiled_outer_ns, phase_profile_request_gap_ns, one_token_decode_outer) =
        if phase_profile {
            if device_short_decode || generic_device_stationary {
                let (prefill_outer_ns, decode_outer_ns, profiled_outer_ns, request_gap_ns) =
                    stationary_one_token_phase_profile_outer_accounting(
                        phase_profile_layer_outer_ns,
                        phase_profile_final_outer_ns,
                        one_token_decode_phase_profile_layer_outer_ns,
                        one_token_decode_phase_profile_final_outer_ns,
                        request_e2e_ns,
                    )?;
                (
                    profiled_outer_ns,
                    request_gap_ns,
                    Some((prefill_outer_ns, decode_outer_ns)),
                )
            } else {
                let (profiled_outer_ns, request_gap_ns) =
                    stationary_phase_profile_outer_accounting(
                        phase_profile_layer_outer_ns,
                        phase_profile_final_outer_ns,
                        request_e2e_ns,
                    )?;
                (profiled_outer_ns, request_gap_ns, None)
            }
        } else {
            (0, 0, None)
        };
    if phase_profile && phase_profile_top_level_ns > request_e2e_ns {
        return Err("stationary phase profile top-level scopes exceed request E2E".into());
    }
    let mut certified_argmax_records = Vec::new();
    let mut certificate_rows = Vec::with_capacity(1 + short_decode_finals.len());
    certificate_rows.push((prompt_len - 1, &final_decision));
    certificate_rows.extend(
        short_decode_finals
            .iter()
            .map(|(position, _, decision, _, _, _)| (*position, decision)),
    );
    if certified_argmax_audit
        && certificate_rows
            .iter()
            .map(|(position, _)| *position)
            .collect::<Vec<_>>()
            != [4, 5, 6]
    {
        return Err(
            "certified argmax audit did not retain original D-exact p4/p5/p6 final decisions"
                .into(),
        );
    }
    for (position, decision) in certificate_rows {
        if certified_argmax_audit {
            validate_certified_argmax_audit(position, decision)?;
        }
        if let Some(record) = certified_argmax_record(position, decision) {
            certified_argmax_records.push(record);
        }
    }
    let cublas_provenance = if let Some(scratch) = strict_scratch.as_ref() {
        scratch.hca_cublas.provenance().clone()
    } else {
        cublas_exact::CublasExact::load()?.provenance().clone()
    };
    let result_path = result.path().to_owned();
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
    let sparse_attention_arm_label = match sparse_attention_arm {
        SparseAttentionArm::Reference => "reference:sparse_attn_v4_bf16",
        SparseAttentionArm::ScoreReuse => "score_reuse:sparse_attn_v4_scores+probabilities+values",
        SparseAttentionArm::Sm89Crossover => "sm89_crossover:K<4=reference,K>=4=score_reuse",
    };
    let explicit_decode_promotion_plan = std::env::var_os(DECODE_PROMOTION_PLAN_ENV).is_some();
    let normalized_promotion_targets = stationary_raw_retention
        .as_ref()
        .map(|cache| {
            let (prefill_device, prefill_host) = stationary_raw_prefill_selection(
                &cache.plan,
                cache.device_budget_bytes,
                cache.host_budget_bytes,
            );
            normalized_decode_promotion_targets(
                &cache.plan,
                &prefill_device,
                &prefill_host,
                &cache.decode_device_selected,
                &cache.decode_host_selected,
                explicit_decode_promotion_plan,
            )
        })
        .transpose()?;
    let relevant_config_fingerprint = normalized_config_fingerprint(serde_json::json!({
        "device_attention": {
            "production": device_attention,
            "generic_stationary": generic_device_stationary,
            "generic_state_audit": generic_device_state_audit,
            "audit_independent_official_host_oracle": device_attention_audit,
            "fallback_policy": "fail_closed",
            "request_rope_lut_rows": short_decode_rows,
            "short_device_decode": device_short_decode,
        },
        "execution_layout": "causal_layer_stationary",
        "fixed_prefetch": fixed_prefetch,
        "fixed_prefetch_wo_a_decode_arm": fixed_prefetch_wo_a_arm,
        "fixed_cache": {
            "device_budget_bytes": parse_gib_budget("COLIBRI_V4_DEVICE_FIXED_CACHE_GIB")?,
            "host_budget_bytes": parse_gib_budget("COLIBRI_V4_HOST_FIXED_CACHE_GIB")?,
            "raw_mode": optional_one_flag("COLIBRI_V4_RAW_FIXED_CACHE")?,
        },
        "stationary_raw_retention": {
            "layout": STATIONARY_RAW_LAYOUT,
            "device_budget_bytes": parse_gib_budget("COLIBRI_V4_STATIONARY_RAW_DEVICE_RETENTION_GIB")?,
            "host_budget_bytes": parse_gib_budget("COLIBRI_V4_STATIONARY_RAW_HOST_RETENTION_GIB")?,
            "decode_retention_overlay": decode_retention_overlay,
            "normalized_promotion_targets": normalized_promotion_targets,
        },
        "stationary_mtp1_draft": stationary_mtp1_draft,
        "l0_pos0_cutpoint": l0_cutpoint,
        "layer_count": manifests.len(),
        "model_revision": v4_state::REVISION,
        "phase_profile": phase_profile,
        "certified_argmax_audit": certified_argmax_audit,
        "certified_argmax_production": certified_argmax_production,
        "prompt_ids": token_ids,
        "prompt_len": prompt_len,
        "routed": {
            "expert_slot_reuse": expert_slot_reuse,
            "qd4": qd4_enabled,
            "qd4_lanes": qd4_lane_count(qd4_enabled),
            "qd4_immutable_expert_cache": qd4_immutable_expert_cache,
            "qd4_immutable_expert_cache_slots": qd4_immutable_expert_cache
                .then_some(V4_QD4_IMMUTABLE_EXPERT_CACHE_SLOTS),
        },
        "sparse_attention_arm": sparse_attention_arm_label,
    }))?;
    let request_rope_lut_h2d_count = usize::from(device_attention);
    let request_rope_lut_h2d_bytes = device_rope_lut
        .as_ref()
        .map_or(0, DeviceRequestRopeLut::num_bytes);
    let hidden_h2d_count = device_attention_h2d_count
        .checked_sub(request_rope_lut_h2d_count)
        .ok_or("device attention hidden H2D count underflow")?;
    let hidden_h2d_bytes = device_attention_h2d_bytes
        .checked_sub(request_rope_lut_h2d_bytes)
        .ok_or("device attention hidden H2D bytes underflow")?;
    let immutable_cache_telemetry = qd4_immutable_cache
        .as_ref()
        .map(|cache| &cache.telemetry)
        .or(qd4_immutable_cache_telemetry.as_ref());
    let mut run_header = serde_json::json!({
        "event": "run",
        "schema": "colibri-v4-layer-stationary-prefill-v1",
        "implementation": "rust-cuda",
        "execution_layout": "causal_layer_stationary",
        "device_attention_production": device_attention,
        "device_attention_audit": device_attention_audit,
        "generic_device_state_audit": {
            "enabled": generic_device_state_audit,
            "head_positions": generic_device_state_audit_heads.iter().filter_map(|head| head["position"].as_u64()).collect::<Vec<_>>(),
            "heads": generic_device_state_audit_heads,
            "wall_ns_excluded_from_request_e2e": generic_device_state_audit_wall_ns,
        },
        "fixed_prefetch_wo_a_decode_arm": fixed_prefetch_wo_a_arm,
        "relevant_config_fingerprint": relevant_config_fingerprint,
        "device_attention_capacity": device_attention_capacity.as_ref().map(|plan| serde_json::json!({
            "checked_sum_bytes": plan.required_bytes,
            "qd4_device_bytes": plan.qd4_device_bytes,
            "qd4_immutable_expert_cache_device_bytes": plan.qd4_immutable_expert_cache_device_bytes,
            "max_fixed_layer": plan.fixed_layer,
            "max_fixed_device_bytes": plan.max_fixed_device_bytes,
            "max_device_token_state_bytes": plan.max_device_token_state_bytes,
            "device_token_state_count": plan.device_token_state_count,
            "device_token_state_total_bytes": plan.device_token_state_total_bytes,
            "rope_lut_bytes": plan.rope_lut_bytes,
            "rope_lut_status": "device_request_global",
            "tid2eid_transient_max_bytes": plan.tid2eid_transient_max_bytes,
            "certified_argmax_device_temp_bytes": plan.certified_argmax_device_temp_bytes,
            "certified_argmax_host_temp_bytes": plan.certified_argmax_host_temp_bytes,
            "headroom_bytes": plan.headroom_bytes,
        })),
        "routed_expert_slot_reuse": expert_slot_reuse,
        "routed_qd4_immutable_expert_cache": qd4_immutable_expert_cache,
        "provenance": provenance,
        "model_revision": v4_state::REVISION,
        "prompt_ids": token_ids,
        "token_ids": token_ids,
        "emitted_token_ids": token_ids.iter().copied().chain(generated_ids.iter().copied()).collect::<Vec<_>>(),
        "greedy_steps": greedy_steps,
        "generated_ids": generated_ids,
        "generation_mode": if generic_device_stationary { "id_only_greedy_argmax_generic_device_stationary" } else if device_short_decode { "id_only_greedy_argmax_short_device_decode" } else { "id_only_greedy_argmax" },
        "start_pos": 0,
        "layers": v4_state::V4_LAYER_COUNT,
        "executed_layers": v4_state::V4_LAYER_COUNT,
        "model_dir": root,
        "request_e2e_ns": request_e2e_ns,
        "request_e2e_definition": if certified_argmax_production { "prompt input through layer-stationary causal execution and certified GPU final-head argmax; CPU full final is audit/fallback only" } else if generic_device_stationary { "prompt input through generic 61-store layer-stationary prefill and token-major EOS-aware greedy continuation; final head is reused per emitted row" } else if device_short_decode { "prompt input through layer-stationary causal prefill, then full61 short device decode rows and canonical CPU-streamed final logits/argmax after every row" } else { "prompt input through layer-stationary causal prefill and canonical CPU-streamed final logits/argmax" },
        "final_head": {"mode": if certified_argmax_production { "bf16_streamed_gpu_certificate" } else { "canonical_cpu_streamed_bf16" }, "production_enabled": certified_argmax_production, "audit_cpu": certified_argmax_audit, "invocation": if generic_device_stationary { "after_last_prompt_position_and_each_generic_stationary_decode_row" } else if device_short_decode { "after_last_prompt_position_and_each_short_device_decode_row" } else { "after_last_prompt_position" }, "weight_shard": "model-00063-of-00064.safetensors"},
        "physical": {
            "fixed_runtime_loads": manifests.len(),
            "fixed_runtime_loads_per_layer": 1,
            "fixed_source_bytes": fixed_source_bytes,
            "fixed_h2d_bytes": fixed_h2d_bytes,
            "routed_source_bytes": routed_source_bytes,
            "routed_h2d_bytes": routed_h2d_bytes,
            "routed_expert_loads": routed_expert_loads,
            "strict_audit_arena_bytes": strict_audit_arena_bytes,
            "short_device_decode": device_short_decode.then(|| serde_json::json!({
                "positions": (prompt_len..short_decode_rows).collect::<Vec<_>>(),
                "fixed_runtime_loads": (short_decode_rows - prompt_len) * manifests.len(),
                "fixed_source_bytes": short_decode_fixed_source_bytes,
                "fixed_h2d_bytes": short_decode_fixed_h2d_bytes,
                "routed_source_bytes": short_decode_routed_source_bytes,
                "routed_h2d_bytes": short_decode_routed_h2d_bytes,
                "qd4_immutable_cache": false,
                "semantic_oracle": device_attention_audit.then_some("independent_host_full_layer_oracle").unwrap_or("not_requested"),
            })),
            "stationary_raw_retention": stationary_raw_retention.as_ref().map(|cache| serde_json::json!({
                "scope": "request_local",
                "model_dir": cache.stationary_model_dir,
                "model_revision": v4_state::REVISION,
                "layout": cache.stationary_layout,
                "device_budget_bytes": cache.device_budget_bytes,
                "host_budget_bytes": cache.host_budget_bytes,
                "decode_device_budget_bytes": cache.decode_device_budget_bytes,
                "decode_host_budget_bytes": cache.decode_host_budget_bytes,
                "device_resident_bytes": cache.device_resident_bytes,
                "host_resident_bytes": cache.host_resident_bytes,
                "device_entries": cache.raw_device.iter().filter(|entry| entry.is_some()).count(),
                "host_entries": cache.raw_host.iter().filter(|entry| entry.is_some()).count(),
                "capture_source_bytes": 0,
                "capture_h2d_bytes": 0,
                "capture_d2d_bytes": 0,
                "device_hits": cache.device_hits,
                "host_hits": cache.host_hits,
                "misses": cache.misses,
                "source_bytes_saved": cache.stationary_source_bytes_saved,
                "h2d_bytes_saved": cache.stationary_h2d_bytes_saved,
                "device_hit_temporary_allocation_count": 0,
                "decode_overlay": {
                    "device_promotions": cache.decode_device_promotions,
                    "host_promotions": cache.decode_host_promotions,
                    "ownership": "move_after_runtime_done_and_transfer_wait",
                    "activation": "after_first_decode_row_before_second_decode_row",
                },
                "selection": if explicit_decode_promotion_plan { "explicit_monotone_promotion_plan" } else { "static_budgeted_saved_work_per_byte" },
                "eviction": "none_request_local",
            })),
            "stationary_peak_reservation": generic_device_stationary.then(|| serde_json::json!({
                "generic_persistent_device_bytes": generic_stationary_reserve_plan.map(|plan| generic_stationary_persistent_reserve_bytes(manifests, plan)).transpose().expect("preflight already checked generic persistent reserve"),
                "generic_shared_scratch_device_bytes": generic_stationary_reserve_plan.map(generic_stationary_scratch_reserve_bytes).transpose().expect("preflight already checked generic scratch reserve"),
                "qd4_device_bytes": usize::from(qd4_enabled) * V4_QD4_DEVICE_BYTES,
                "qd4_pinned_host_bytes": usize::from(qd4_enabled) * V4_QD4_PINNED_HOST_BYTES,
                "qd4_immutable_expert_cache_device_bytes": usize::from(qd4_immutable_expert_cache) * V4_QD4_IMMUTABLE_EXPERT_CACHE_DEVICE_BYTES,
                "qd4_immutable_expert_cache_pinned_host_bytes": usize::from(qd4_immutable_expert_cache) * V4_QD4_IMMUTABLE_EXPERT_CACHE_HOST_BYTES,
                "raw_cache_device_bytes": stationary_raw_retention.as_ref().map_or(0, |cache| cache.device_resident_bytes),
                "raw_cache_host_bytes": stationary_raw_retention.as_ref().map_or(0, |cache| cache.host_resident_bytes),
                "worker_scratch_device_bytes": PINNED_FIXED_PIPELINE_WORKER_SCRATCH_DEVICE_BYTES,
                "worker_prepared_runtime_device_bytes": PINNED_FIXED_PIPELINE_WORKER_PREPARED_DEVICE_BYTES,
                "worker_pinned_host_bytes": PINNED_FIXED_PIPELINE_WORKER_PINNED_HOST_BYTES,
                "mtp_device_peak_bytes": stationary_mtp1_peak_reserve_bytes,
                "mtp_candidate_local_device_bytes": usize::from(stationary_mtp1_draft) * MTP1_DRAFT_CANDIDATE_LOCAL_BYTES,
                "mtp_head_host_staging_bytes": usize::from(stationary_mtp1_draft) * CERTIFIED_ARGMAX_HOST_TEMP_BYTES,
                "mtp_extra_pinned_host_bytes": mtp1_stationary_pinned_host_reserve_bytes(qd4_enabled),
                "mtp_borrows_qd4_lane": qd4_enabled,
                "head_streams_per_mtp_candidate": 2,
                "prefill_immutable_peak": generic_prefill_peak.map(|(device, host, pinned)| serde_json::json!({
                    "device_bytes": device,
                    "host_bytes": host,
                    "pinned_host_bytes": pinned,
                })),
                "decode_overlay_peak": generic_decode_overlay_peak.map(|(device, host, pinned)| serde_json::json!({
                    "device_bytes": device,
                    "host_bytes": host,
                    "pinned_host_bytes": pinned,
                })),
            })),
            "routed_qd4_immutable_expert_cache": immutable_cache_telemetry.map(|telemetry| serde_json::json!({
                "route_executions": telemetry.route_executions,
                "expert_union_loads": telemetry.expert_union_loads,
                "source_loads": telemetry.source_loads,
                "source_bytes": telemetry.source_bytes,
                "h2d_loads": telemetry.h2d_loads,
                "h2d_bytes": telemetry.h2d_bytes,
                "hits": telemetry.hits,
                "d2d_copies": telemetry.d2d_copies,
                "d2d_bytes": telemetry.d2d_bytes,
                "max_occupancy": telemetry.max_occupancy,
                "completed_layers": telemetry.completed_layers,
                "worker_errors": telemetry.worker_errors,
                "fallbacks": telemetry.fallbacks,
                "slots": V4_QD4_IMMUTABLE_EXPERT_CACHE_SLOTS,
                "eviction": "none",
            })),
            "device_attention_dispatches": device_attention_dispatches,
            "generic_stationary": generic_lifecycle.as_ref().map(|lifecycle| serde_json::json!({
                "persistent_layer_stores": generic_states.as_ref().map_or(0, |states| states.iter().filter(|state| state.is_some()).count()),
                "layer_commits": lifecycle.telemetry.layer_commits,
                "device_peak_bytes": lifecycle.telemetry.device_peak_bytes,
                "host_peak_bytes": lifecycle.telemetry.host_peak_bytes,
                "pinned_peak_bytes": lifecycle.telemetry.pinned_peak_bytes,
                "live_lease_peak": lifecycle.telemetry.live_lease_peak,
                "live_leases_at_head": lifecycle.live_leases,
                "mtp": {
                    "enabled": stationary_mtp1_draft,
                    "peak_reserve_bytes": stationary_mtp1_peak_reserve_bytes,
                    "state": if stationary_mtp1_state.is_some() { "active" } else if stationary_mtp1_disabled.is_some() { "disabled" } else { "off" },
                    "disabled_reason": stationary_mtp1_disabled,
                    "candidate_target_alignment": "MTP input/teacher hidden/KV write at t; candidate predicts teacher token t+1",
                    "records": stationary_mtp1_records,
                },
            })),
            "host_attention_fallbacks": device_attention_host_fallbacks,
            "device_attention": {
                "dispatches_actual": device_attention_dispatches,
                "audit_oracle_dispatches_actual": if device_attention_audit {
                    device_attention_dispatches
                } else {
                    0
                },
                "host_fallbacks_actual": device_attention_host_fallbacks,
                "fallback_policy": "fail_closed",
                "request_rope_lut_setup_ns": device_rope_lut_setup_ns,
                "transfers": {
                    "hidden_h2d": {
                        "count": hidden_h2d_count,
                        "bytes": hidden_h2d_bytes,
                    },
                    "request_rope_lut_h2d": {
                        "count": request_rope_lut_h2d_count,
                        "bytes": request_rope_lut_h2d_bytes,
                    },
                    "request_rope_lut_d2d": {
                        "scope": "only request-global RoPE LUT row copies; internal attention/cache D2D is not claimed",
                        "count": device_attention_rope_lut_d2d_count,
                        "bytes": device_attention_rope_lut_d2d_bytes,
                    },
                    "terminal_after_attention_d2h": {
                        "count": device_attention_terminal_d2h_count,
                        "bytes": device_attention_terminal_d2h_bytes,
                    },
                    "intermediate_or_audit_d2h": {
                        "count": device_attention_audit_d2h_count,
                        "bytes": device_attention_audit_d2h_bytes,
                    },
                },
                "steady_device_temporary_alloc_count": 0,
                "topk_observed": device_attention_audit,
            },
        },
        "semantic_projection": [
            "token_position", "token_id", "layer", "compression_ratio", "hidden_bf16_digest",
            "persistent_state_observed",
            "kv_cache_bf16_digest", "main_compressed_kv_bf16_digest",
            "main_compressor_kv_state_f32_digest", "main_compressor_score_state_f32_digest",
            "indexer_compressed_kv_bf16_digest", "indexer_compressor_kv_state_f32_digest",
            "indexer_compressor_score_state_f32_digest", "route_ids", "route_weights_f32",
            "topk_observed", "attention_topk_digest", "attention_topk_per_row", "manifest_contract_range_bytes",
            "kv_rows_before", "kv_rows_after"
        ],
        "sparse_attention_arm": sparse_attention_arm_label,
        "stop_reason": stop_reason,
        "eos_token_id": V4_EOS_TOKEN_ID,
        "max_new_tokens": greedy_steps,
        "actual_generated_count": generated_ids.len(),
        "routed_qd4": {
            "enabled": qd4_enabled,
            "lanes": qd4_lane_count(qd4_enabled),
            "startup_setup_ns": qd4_startup_setup_ns,
            "pinned_host_reserve_bytes": usize::from(qd4_enabled) * V4_QD4_PINNED_HOST_BYTES,
            "device_reserve_bytes": usize::from(qd4_enabled) * V4_QD4_DEVICE_BYTES,
            "immutable_expert_cache": {
                "enabled": qd4_immutable_expert_cache,
                "slots": qd4_immutable_expert_cache.then_some(V4_QD4_IMMUTABLE_EXPERT_CACHE_SLOTS),
                "pinned_host_reserve_bytes": usize::from(qd4_immutable_expert_cache)
                    * V4_QD4_IMMUTABLE_EXPERT_CACHE_HOST_BYTES,
                "device_reserve_bytes": usize::from(qd4_immutable_expert_cache)
                    * V4_QD4_IMMUTABLE_EXPERT_CACHE_DEVICE_BYTES,
                "eviction": "none",
            },
            "expected_routed_expert_loads": qd4_enabled
                .then(|| {
                    qd4_expected_routed_loads(
                        manifests.len(),
                        if generic_device_stationary {
                            greedy_forward_tokens(prompt_len, greedy_steps)?
                        } else {
                            prompt_len
                        },
                    )
                })
                .transpose()?,
        },
    });
    if phase_profile {
        run_header
            .as_object_mut()
            .ok_or("stationary run header is not an object")?
            .insert(
                "phase_profile".into(),
                serde_json::json!({
                    "schema": "colibri-v4-stationary-phase-profile-v4",
                    "enabled": true,
                    "top_level_scope_model": "layer_prepare (including nested fixed-load diagnostics), attention (including position-0 device-state setup), router, shared, and routed are exclusive host-wall scopes; nested CUDA event and QD4 lane work sums are not additive",
                    "layer_scope_count": layer_records.iter().map(Vec::len).sum::<usize>().saturating_add(short_decode_records.len()),
                    "layer_prepare_scope_count": manifests.len().saturating_add(short_decode_records.len()),
                    "final_scope_count": 1 + short_decode_finals.len(),
                    "top_level_attributed_ns": phase_profile_top_level_ns,
                    "layer_outer_ns": phase_profile_layer_outer_ns.saturating_add(one_token_decode_phase_profile_layer_outer_ns),
                    "final_outer_ns": phase_profile_final_outer_ns.saturating_add(one_token_decode_phase_profile_final_outer_ns),
                    "profiled_outer_ns": phase_profiled_outer_ns,
                    "request_unprofiled_gap_ns": phase_profile_request_gap_ns,
                    "request_e2e_ns": request_e2e_ns,
                    "closure": "profiled_outer_ns + request_unprofiled_gap_ns == request_e2e_ns",
                    "short_device_decode_outer": if device_short_decode { one_token_decode_outer.map(|(prefill_outer_ns, decode_outer_ns)| serde_json::json!({
                        "prefill_outer_ns": prefill_outer_ns,
                        "decode_outer_ns": decode_outer_ns,
                        "decode_layer_outer_ns": one_token_decode_phase_profile_layer_outer_ns,
                        "decode_final_outer_ns": one_token_decode_phase_profile_final_outer_ns,
                        "closure": "prefill_outer_ns + decode_outer_ns + request_unprofiled_gap_ns == request_e2e_ns",
                        "exact": prefill_outer_ns
                            .saturating_add(decode_outer_ns)
                            .saturating_add(phase_profile_request_gap_ns)
                            == request_e2e_ns,
                    })) } else { None },
                    "generic_decode_outer": if generic_device_stationary { one_token_decode_outer.map(|(prefill_outer_ns, decode_outer_ns)| serde_json::json!({
                        "prefill_outer_ns": prefill_outer_ns,
                        "decode_outer_ns": decode_outer_ns,
                        "decode_layer_outer_ns": one_token_decode_phase_profile_layer_outer_ns,
                        "decode_final_outer_ns": one_token_decode_phase_profile_final_outer_ns,
                        "closure": "prefill_outer_ns + decode_outer_ns + request_unprofiled_gap_ns == request_e2e_ns",
                        "exact": prefill_outer_ns
                            .saturating_add(decode_outer_ns)
                            .saturating_add(phase_profile_request_gap_ns)
                            == request_e2e_ns,
                    })) } else { None },
                    "request_accounting_valid": phase_profile_top_level_ns <= request_e2e_ns
                        && phase_profiled_outer_ns <= request_e2e_ns
                        && phase_profiled_outer_ns.saturating_add(phase_profile_request_gap_ns)
                            == request_e2e_ns,
                }),
            );
    }
    if fixed_prefetch {
        let ready_count = fixed_prefetch_records
            .iter()
            .filter(|record| record["ready_before_consume"] == true)
            .count();
        let consume_wait_ns = fixed_prefetch_records
            .iter()
            .try_fold(0_u64, |total, record| {
                total
                    .checked_add(
                        record["consume_wait_ns"]
                            .as_u64()
                            .ok_or("fixed prefetch consume_wait_ns is invalid")?,
                    )
                    .ok_or("fixed prefetch consume wait total overflow")
            })?;
        run_header
            .as_object_mut()
            .ok_or("stationary run header is not an object")?
            .insert(
                "fixed_prefetch".into(),
                serde_json::json!({
                    "mode": "bounded_window_2_cpu_read_decode_then_ordered_boundary_sync_h2d",
                    "window": {
                        "depth": FIXED_PREFETCH_WINDOW_DEPTH,
                        "max_future_host_bytes": fixed_prefetch_max_future_host_bytes,
                        "reserved_future_host_bytes": fixed_prefetch_max_future_host_bytes
                            .and_then(|bytes| bytes.checked_mul(FIXED_PREFETCH_WINDOW_DEPTH)),
                        "host_headroom_bytes": FIXED_PREFETCH_HOST_HEADROOM_BYTES,
                    },
                    "summary": {
                        "transition_count": fixed_prefetch_records.len(),
                        "ready_count": ready_count,
                        "consume_wait_ns": consume_wait_ns,
                    },
                    "transitions": fixed_prefetch_records,
                    "top_level_phase_layer_prepare_scope": "main-thread consume wait plus H2D are in layer_prepare; worker read/decode remains asynchronous and is not additive",
                }),
            );
    }
    if let Some(tokenizer) = tokenizer {
        let header = run_header
            .as_object_mut()
            .ok_or("layer-stationary run header is not an object")?;
        header.insert("encoded_prompt_ids".into(), serde_json::json!(token_ids));
        header.insert(
            "generated_only_decoded_text".into(),
            serde_json::json!(tokenizer.decode(&generated_ids)?),
        );
        let cumulative = token_ids
            .iter()
            .copied()
            .chain(generated_ids.iter().copied())
            .filter(|&id| id != V4_EOS_TOKEN_ID)
            .collect::<Vec<_>>();
        header.insert(
            "cumulative_decoded_text".into(),
            serde_json::json!(tokenizer.decode(&cumulative)?),
        );
    }
    if let Some(user) = chat_user {
        run_header
            .as_object_mut()
            .ok_or("layer-stationary run header is not an object")?
            .extend(serde_json::Map::from_iter([
                ("chat_user".into(), serde_json::json!(user)),
                (
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
                ),
            ]));
    } else if let Some(text) = prompt_text {
        run_header
            .as_object_mut()
            .ok_or("layer-stationary run header is not an object")?
            .extend(serde_json::Map::from_iter([
                ("prompt_text".into(), serde_json::json!(text)),
                (
                    "tokenizer".into(),
                    serde_json::json!({
                        "mode": "raw",
                        "chat_template": false,
                        "add_special_tokens": false,
                    }),
                ),
            ]));
    }
    writeln!(result.file_mut(), "{run_header}")?;
    for records in &layer_records {
        for record in records {
            writeln!(result.file_mut(), "{record}")?;
        }
    }
    for record in &certified_argmax_records {
        writeln!(result.file_mut(), "{record}")?;
    }
    for record in &short_decode_records {
        writeln!(result.file_mut(), "{record}")?;
    }
    let mut final_record = id_only_final_record(
        prompt_len - 1,
        token_ids[prompt_len - 1],
        "prompt",
        if device_attention {
            "DeviceTokenState"
        } else if generic_device_stationary {
            "DeviceLayerPersistentState"
        } else {
            "V4SerialPersistentState"
        },
        &final_decision,
        None,
        final_decision_elapsed_ns,
    )?;
    if let Some(profile) = final_phase_profile {
        final_record
            .as_object_mut()
            .ok_or("stationary final record is not an object")?
            .insert("phase_profile".into(), profile);
    }
    writeln!(result.file_mut(), "{final_record}")?;
    for (
        position,
        input_id,
        decode_decision,
        decode_hidden,
        decode_profile,
        decode_decision_elapsed_ns,
    ) in &short_decode_finals
    {
        let mut decode_final_record = id_only_final_record(
            *position,
            *input_id,
            "generated",
            if generic_device_stationary {
                "DeviceLayerPersistentState"
            } else {
                "DeviceTokenState"
            },
            decode_decision,
            Some(decode_hidden),
            *decode_decision_elapsed_ns,
        )?;
        if let Some(profile) = decode_profile {
            decode_final_record
                .as_object_mut()
                .ok_or("short decode final record is not an object")?
                .insert("phase_profile".into(), profile.clone());
        }
        writeln!(result.file_mut(), "{decode_final_record}")?;
    }
    result.commit()?;
    println!(
        "{{\"status\":\"PASS\",\"mode\":{},\"token_ids\":{},\"generated_ids\":{},\"layers\":{},\"argmax\":{},\"result_jsonl\":{}}}",
        serde_json::to_string(if generic_device_stationary {
            "v4-generic-device-stationary"
        } else if device_short_decode {
            "v4-layer-stationary-short-device-decode"
        } else {
            "v4-layer-stationary-prefill"
        })?,
        serde_json::to_string(token_ids)?,
        serde_json::to_string(&generated_ids)?,
        v4_state::V4_LAYER_COUNT,
        argmax,
        serde_json::to_string(&result_path)?,
    );
    Ok(())
}
