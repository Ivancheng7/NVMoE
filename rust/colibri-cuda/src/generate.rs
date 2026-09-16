use super::*;

pub(crate) fn process_peak_rss_bytes() -> Result<usize, Box<dyn Error>> {
    let kib = std::fs::read_to_string("/proc/self/status")?
        .lines()
        .find_map(|line| line.strip_prefix("VmHWM:"))
        .and_then(|value| value.split_whitespace().next())
        .ok_or("VmHWM is missing from /proc/self/status")?
        .parse::<usize>()?;
    Ok(kib.saturating_mul(1024))
}

pub(crate) fn device_used_bytes() -> Result<usize, Box<dyn Error>> {
    let (mut free, mut total) = (0_usize, 0_usize);
    let result = unsafe { sys::cuMemGetInfo_v2(&mut free, &mut total) };
    if result != 0 {
        return Err(format!("cuMemGetInfo_v2 failed with {result:?}").into());
    }
    Ok(total.saturating_sub(free))
}

pub(crate) fn run_b2_serial_request(
    context: &std::sync::Arc<CudaContext>,
    module: &kernels::LoadedModule,
    stream: &std::sync::Arc<CudaStream>,
    model_dir: &std::path::Path,
    token_id: i64,
    layers: &[usize],
    initial_hidden: &[u16],
    lanes: &mut [ExpertSlot<'_>],
    scratch: &mut DeviceBuffer<u16>,
    physical: &mut B2Physical,
    peak_device_bytes: &mut usize,
) -> Result<B2RequestResult, Box<dyn Error>> {
    let mut hidden = initial_hidden.to_vec();
    let mut persistent = v4_state::V4SerialPersistentState::new(1)?;
    let mut digests = Vec::with_capacity(layers.len());
    for &layer in layers {
        let mut loaded = load_b2_raw_runtime(model_dir, layer, stream)?;
        let hash_ids = load_layer_hash_ids(model_dir, layer, &[token_id])?;
        let row_hash_ids =
            b2_router_metadata_row(loaded.raw.runtime.weights.hash_router, &hash_ids, 0, 1)?;
        physical.fixed_source_bytes += loaded.fixed_source_bytes;
        physical.fixed_h2d_bytes += loaded.fixed_h2d_bytes;
        physical
            .record_request_router_metadata(loaded.raw.runtime.weights.hash_router, row_hash_ids)?;
        *peak_device_bytes = (*peak_device_bytes).max(device_used_bytes()?);
        launch_e4m3fn_to_bf16(module, stream, &loaded.raw.wo_a_raw, scratch)?;
        stream.synchronize()?;
        std::mem::swap(&mut loaded.raw.runtime.weights.wo_a, scratch);
        let run = run_b2_layer(
            context,
            module,
            stream,
            None,
            &loaded.raw.runtime,
            lanes,
            row_hash_ids,
            &hidden,
            token_id,
            0,
            &mut persistent,
        None,
        );
        std::mem::swap(&mut loaded.raw.runtime.weights.wo_a, scratch);
        let (next, digest, stage) = run?;
        physical.routed_source_bytes += stage.routed_source_bytes;
        physical.routed_h2d_bytes += stage.routed_h2d_bytes;
        hidden = next;
        digests.push(digest);
    }
    Ok(B2RequestResult {
        layers: digests,
        output_digest: fnv1a_u16(&hidden),
    })
}

pub(crate) struct B2ArmRun<const REQUESTS: usize> {
    pub(crate) results: [B2RequestResult; REQUESTS],
    pub(crate) physical: B2Physical,
    pub(crate) total_ns: u64,
    pub(crate) completion_ns: [u64; REQUESTS],
    pub(crate) service_latency_ns: [u64; REQUESTS],
}

pub(crate) fn run_b2_serial_requests<const REQUESTS: usize>(
    context: &std::sync::Arc<CudaContext>,
    module: &kernels::LoadedModule,
    stream: &std::sync::Arc<CudaStream>,
    model_dir: &std::path::Path,
    token_ids: [i64; REQUESTS],
    initial_hidden: &[Vec<u16>; REQUESTS],
    layers: &[usize],
    lanes: &mut [ExpertSlot<'_>],
    scratch: &mut DeviceBuffer<u16>,
    peak_device_bytes: &mut usize,
) -> Result<B2ArmRun<REQUESTS>, Box<dyn Error>> {
    let mut physical = B2Physical::default();
    let started = std::time::Instant::now();
    let mut results = Vec::with_capacity(REQUESTS);
    let mut completion_ns = [0_u64; REQUESTS];
    let mut service_latency_ns = [0_u64; REQUESTS];
    for request in 0..REQUESTS {
        let service_started = std::time::Instant::now();
        results.push(run_b2_serial_request(
            context,
            module,
            stream,
            model_dir,
            token_ids[request],
            layers,
            &initial_hidden[request],
            lanes,
            scratch,
            &mut physical,
            peak_device_bytes,
        )?);
        completion_ns[request] = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        service_latency_ns[request] =
            u64::try_from(service_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
    }
    let total_ns = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
    Ok(B2ArmRun {
        results: results
            .try_into()
            .map_err(|_| "layer-stationary serial request count changed")?,
        physical,
        total_ns,
        completion_ns,
        service_latency_ns,
    })
}

pub(crate) fn run_b2_serial_pair(
    context: &std::sync::Arc<CudaContext>,
    module: &kernels::LoadedModule,
    stream: &std::sync::Arc<CudaStream>,
    model_dir: &std::path::Path,
    token_ids: [i64; 2],
    initial_hidden: &[Vec<u16>; 2],
    lanes: &mut [ExpertSlot<'_>],
    scratch: &mut DeviceBuffer<u16>,
    peak_device_bytes: &mut usize,
) -> Result<B2ArmRun<2>, Box<dyn Error>> {
    run_b2_serial_requests(
        context,
        module,
        stream,
        model_dir,
        token_ids,
        initial_hidden,
        &[3, 4],
        lanes,
        scratch,
        peak_device_bytes,
    )
}

pub(crate) fn run_b2_layer_stationary_requests<const REQUESTS: usize>(
    context: &std::sync::Arc<CudaContext>,
    module: &kernels::LoadedModule,
    stream: &std::sync::Arc<CudaStream>,
    model_dir: &std::path::Path,
    token_ids: [i64; REQUESTS],
    initial_hidden: &[Vec<u16>; REQUESTS],
    layers: &[usize],
    lanes: &mut [ExpertSlot<'_>],
    scratch: &mut DeviceBuffer<u16>,
    peak_device_bytes: &mut usize,
) -> Result<B2ArmRun<REQUESTS>, Box<dyn Error>> {
    let mut hidden: [Vec<u16>; REQUESTS] =
        std::array::from_fn(|request| initial_hidden[request].clone());
    let mut state = (0..REQUESTS)
        .map(|_| v4_state::V4SerialPersistentState::new(1))
        .collect::<Result<Vec<_>, _>>()?;
    let mut digests = (0..REQUESTS)
        .map(|_| Vec::with_capacity(layers.len()))
        .collect::<Vec<Vec<B2LayerDigest>>>();
    let mut physical = B2Physical::default();
    let mut completion_ns = [0_u64; REQUESTS];
    let mut service_started_ns = [None; REQUESTS];
    let mut service_latency_ns = [0_u64; REQUESTS];
    let started = std::time::Instant::now();
    let first_layer = *layers
        .first()
        .ok_or("layer-stationary layers must be non-empty")?;
    let last_layer = *layers
        .last()
        .ok_or("layer-stationary layers must be non-empty")?;
    if layers.windows(2).any(|pair| pair[1] != pair[0] + 1) {
        return Err("layer-stationary layers must be contiguous".into());
    }
    for &layer in layers {
        let mut loaded = load_b2_raw_runtime(model_dir, layer, stream)?;
        let hash_ids = load_layer_hash_ids(model_dir, layer, &token_ids)?;
        physical.fixed_source_bytes += loaded.fixed_source_bytes;
        physical.fixed_h2d_bytes += loaded.fixed_h2d_bytes;
        for request in 0..REQUESTS {
            let row = b2_router_metadata_row(
                loaded.raw.runtime.weights.hash_router,
                &hash_ids,
                request,
                REQUESTS,
            )?;
            physical.record_request_router_metadata(loaded.raw.runtime.weights.hash_router, row)?;
        }
        *peak_device_bytes = (*peak_device_bytes).max(device_used_bytes()?);
        launch_e4m3fn_to_bf16(module, stream, &loaded.raw.wo_a_raw, scratch)?;
        stream.synchronize()?;
        std::mem::swap(&mut loaded.raw.runtime.weights.wo_a, scratch);
        let mut layer_error = None;
        for request in 0..REQUESTS {
            if layer == first_layer {
                service_started_ns[request] =
                    Some(u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX));
            }
            match run_b2_layer(
                context,
                module,
                stream,
                None,
                &loaded.raw.runtime,
                lanes,
                b2_router_metadata_row(
                    loaded.raw.runtime.weights.hash_router,
                    &hash_ids,
                    request,
                    REQUESTS,
                )?,
                &hidden[request],
                token_ids[request],
                0,
                &mut state[request],
            None,
            ) {
                Ok((next, digest, stage)) => {
                    physical.routed_source_bytes += stage.routed_source_bytes;
                    physical.routed_h2d_bytes += stage.routed_h2d_bytes;
                    hidden[request] = next;
                    digests[request].push(digest);
                    if layer == last_layer {
                        completion_ns[request] =
                            u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
                        service_latency_ns[request] = completion_ns[request].saturating_sub(
                            service_started_ns[request].ok_or(
                                "layer-stationary request completed before service started",
                            )?,
                        );
                    }
                }
                Err(error) => {
                    layer_error = Some(error);
                    break;
                }
            }
        }
        std::mem::swap(&mut loaded.raw.runtime.weights.wo_a, scratch);
        if let Some(error) = layer_error {
            return Err(error);
        }
    }
    let total_ns = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
    Ok(B2ArmRun {
        results: (0..REQUESTS)
            .map(|request| B2RequestResult {
                layers: std::mem::take(&mut digests[request]),
                output_digest: fnv1a_u16(&hidden[request]),
            })
            .collect::<Vec<_>>()
            .try_into()
            .map_err(|_| "layer-stationary batch request count changed")?,
        physical,
        total_ns,
        completion_ns,
        service_latency_ns,
    })
}

pub(crate) fn run_b2_batch_pair(
    context: &std::sync::Arc<CudaContext>,
    module: &kernels::LoadedModule,
    stream: &std::sync::Arc<CudaStream>,
    model_dir: &std::path::Path,
    token_ids: [i64; 2],
    initial_hidden: &[Vec<u16>; 2],
    lanes: &mut [ExpertSlot<'_>],
    scratch: &mut DeviceBuffer<u16>,
    peak_device_bytes: &mut usize,
) -> Result<B2ArmRun<2>, Box<dyn Error>> {
    run_b2_layer_stationary_requests(
        context,
        module,
        stream,
        model_dir,
        token_ids,
        initial_hidden,
        &[3, 4],
        lanes,
        scratch,
        peak_device_bytes,
    )
}

/// Uploads one fixed layer out of a host payload the caller owns.
///
/// When the payload is retained (the cache still holds a reference) nothing is
/// freed here and the release costs nothing; when it is the last reference the
/// release is timed, because that free used to land on the main thread inside
/// the fixed window.
fn load_flash_b1_layer_from_host(
    spec: &'static v4_spec::V4Spec,
    model_dir: &std::path::Path,
    layer: usize,
    host: std::sync::Arc<fixed_host_cache::ResidentLayerHost>,
    telemetry: FixedLoadTelemetry,
    stream: &CudaStream,
    request_started: std::time::Instant,
) -> Result<B2RawRuntime, Box<dyn Error>> {
    let mut loaded = load_b2_raw_runtime_from_host_ref_for(
        spec,
        model_dir,
        layer,
        &host.host,
        &host.wo_a_raw,
        telemetry,
        stream,
    )?;
    let released_bytes = if std::sync::Arc::strong_count(&host) == 1 {
        host.bytes()
    } else {
        0
    };
    let release_started_ns = timeline_elapsed_ns(request_started);
    drop(host);
    loaded.fixed.window_release_ns =
        timeline_elapsed_ns(request_started).saturating_sub(release_started_ns);
    loaded.fixed.window_release_bytes = released_bytes;
    Ok(loaded)
}

pub(crate) fn run_v4_b2_generate_arm<const REQUESTS: usize>(
    spec: &'static v4_spec::V4Spec,
    context: &std::sync::Arc<CudaContext>,
    module: &kernels::LoadedModule,
    stream: &std::sync::Arc<CudaStream>,
    flash_cublas: Option<&FlashCublasSession>,
    model_dir: &std::path::Path,
    final_weights: Option<&v4_state::FinalHeadWeights>,
    flash_final_head_gpu: Option<&FlashDsparkFinalHeadGpuWorkspace>,
    lanes: &mut [ExpertSlot<'_>],
    mut scratch: &mut DeviceBuffer<u16>,
    prompts: [Vec<i64>; REQUESTS],
    greedy_steps: usize,
    forward_tokens: [usize; REQUESTS],
    record_final_head_loader: bool,
    capture_layer_trace: bool,
    parity_stop_after_layer: Option<&FlashParityStopAfterLayer>,
    on_generated: &mut dyn FnMut(
        usize,
        i64,
        Option<GreedyStopReason>,
    ) -> Result<(), Box<dyn Error>>,
) -> Result<B2GenerateArm<REQUESTS>, Box<dyn Error>> {
    let wo_a_elements = spec
        .q_heads
        .checked_mul(spec.q_head_dim)
        .and_then(|value| value.checked_mul(spec.o_lora_rank))
        .ok_or("row-set wo_a scratch size overflow")?;
    if scratch.len() != wo_a_elements {
        return Err(format!(
            "{} row-set wo_a scratch has {} elements, expected {wo_a_elements}",
            spec.name,
            scratch.len()
        )
        .into());
    }
    let mut requests = prompts
        .into_iter()
        .enumerate()
        .map(|(request, prompt_ids)| {
            Ok(B2GenerateRequest {
                input_ids: prompt_ids.clone(),
                prompt_ids,
                position: 0,
                persistent: v4_state::V4SerialPersistentState::new_for_spec(
                    spec,
                    forward_tokens[request],
                    1,
                )?,
                hidden: Vec::new(),
                generated_ids: Vec::with_capacity(greedy_steps),
                stop_reason: None,
                tokens: Vec::with_capacity(forward_tokens[request]),
                finals: Vec::with_capacity(greedy_steps),
                physical: B2RequestPhysical::default(),
                flash_b1_phase: FlashB1PhaseTiming::default(),
            })
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    let mut active = [true; REQUESTS];
    let mut physical = B2Physical::default();
    // The final-head fixed tensors are shared by this row-set run. Attribute their
    // one real loader read to the first request so request ledgers sum exactly
    // to the global physical ledger; streamed head rows stay per decision.
    if record_final_head_loader {
        let final_weights = final_weights.ok_or("row-set generation needs final head weights")?;
        requests[0]
            .physical
            .record_final_head_source_bytes(final_weights.loader_source_bytes)?;
        physical.record_final_head_source_bytes(final_weights.loader_source_bytes)?;
    }
    let mut peak_device_bytes = device_used_bytes()?;
    let request_started = std::time::Instant::now();
    let flash_b1_profile = spec == &v4_spec::FLASH_0731 && REQUESTS == 1;
    let final_head_arm = flash_final_head_arm(spec)?;
    let flash_b1_prefetch = flash_b1_prefetch_enabled(spec, REQUESTS)?;
    let layer_count = parity_stop_after_layer
        .map(|stop| {
            stop.layer
                .checked_add(1)
                .ok_or("Flash parity stop layer overflow")
        })
        .transpose()?
        .unwrap_or(spec.layers);
    if layer_count == 0 || layer_count > spec.layers {
        return Err("Flash parity stop layer is outside the model depth".into());
    }
    // One cache for the whole run: it has to outlive the token loop, otherwise
    // every token misses and the retained bytes are rebuilt and thrown away 8
    // times over instead of once.
    let host_cache = std::sync::Arc::new(std::sync::Mutex::new(
        fixed_host_cache::FixedHostCache::from_env(spec.layers)?,
    ));
    // Same scope argument for the routed expert slot: page-locking its staging
    // costs 17 ms, so it lives as long as the run instead of one layer.
    let mut routed_slot_scratch = RoutedSlotScratch::new();

    while !b2_generate_active_indices::<REQUESTS>(&active).is_empty() {
        let active_requests = b2_generate_active_indices::<REQUESTS>(&active);
        let token_ids = active_requests
            .iter()
            .map(|&request| {
                requests[request]
                    .input_ids
                    .get(requests[request].position)
                    .copied()
                    .ok_or("B2 request has no input token at its active position")
            })
            .collect::<Result<Vec<_>, _>>()?;
        for (&request, &token_id) in active_requests.iter().zip(&token_ids) {
            let (hidden, initial_state_source_bytes) =
                prepare_b2_initial_hidden_with_source_bytes_for(spec, model_dir, token_id)?;
            requests[request]
                .physical
                .record_initial_state_source_bytes(initial_state_source_bytes)?;
            physical.record_initial_state_source_bytes(initial_state_source_bytes)?;
            requests[request].hidden = hidden;
        }
        let mut layers = (0..REQUESTS)
            .map(|_| Vec::with_capacity(spec.layers))
            .collect::<Vec<Vec<B2LayerDigest>>>();
        let token_started = std::time::Instant::now();
        std::thread::scope(|scope| -> Result<(), Box<dyn Error>> {
            let mut prefetched_next: Option<fixed_host_cache::ResidentPrefetch<'_>> = None;
            for layer in 0..layer_count {
                let layer_started = std::time::Instant::now();
                let mut from_prefetch = false;
                let mut loaded = if let Some(task) = prefetched_next.take() {
                    from_prefetch = true;
                    let scheduled_layer = task.layer;
                    let join_entered_ns = timeline_elapsed_ns(request_started);
                    let prefetched = task
                        .worker
                        .join()
                        .map_err(|_| "Flash B1 fixed prefetch worker panicked")?;
                    let window_join_ns =
                        timeline_elapsed_ns(request_started).saturating_sub(join_entered_ns);
                    let prefetched =
                        flash_b1_prefetch_worker_result(layer, scheduled_layer, prefetched)?;
                    if prefetched.layer != layer {
                        return Err(
                            "Flash B1 fixed prefetch worker returned the wrong layer".into()
                        );
                    }
                    let window_worker_ns = prefetched
                        .worker_end_ns
                        .saturating_sub(prefetched.worker_start_ns);
                    let window_worker_ready = prefetched.worker_end_ns <= join_entered_ns;
                    let mut loaded = load_flash_b1_layer_from_host(
                        spec,
                        model_dir,
                        layer,
                        prefetched.host,
                        prefetched.telemetry,
                        stream,
                        request_started,
                    )?;
                    if loaded.fixed.cache_status != "host_resident_hit" {
                        loaded.fixed.cache_status = "prefetched_host";
                    }
                    loaded.fixed.window_join_ns = window_join_ns;
                    loaded.fixed.window_worker_ns = window_worker_ns;
                    loaded.fixed.window_worker_ready = window_worker_ready;
                    loaded
                } else {
                    let (host, telemetry) = fixed_host_cache::resolve_host_layer(
                        spec,
                        model_dir,
                        layer,
                        host_cache.as_ref(),
                    )?;
                    load_flash_b1_layer_from_host(
                        spec,
                        model_dir,
                        layer,
                        host,
                        telemetry,
                        stream,
                        request_started,
                    )?
                };
                if flash_b1_prefetch {
                    if let Some(next_layer) = flash_b1_prefetch_next_layer(layer, layer_count)? {
                        let worker_root = model_dir.to_owned();
                        let worker_cache = host_cache.clone();
                        let worker_started = timeline_elapsed_ns(request_started);
                        let spawn_started_ns = timeline_elapsed_ns(request_started);
                        let worker = scope.spawn(move || {
                            let (host, telemetry) = fixed_host_cache::resolve_host_layer(
                                spec,
                                &worker_root,
                                next_layer,
                                worker_cache.as_ref(),
                            )
                            .map_err(|error| error.to_string())?;
                            let worker_ended = timeline_elapsed_ns(request_started);
                            Ok(fixed_host_cache::ResidentPrefetched {
                                layer: next_layer,
                                host,
                                telemetry,
                                worker_start_ns: worker_started,
                                worker_end_ns: worker_ended,
                            })
                        });
                        prefetched_next = Some(fixed_host_cache::ResidentPrefetch {
                            layer: next_layer,
                            worker,
                        });
                        loaded.fixed.window_spawn_ns = timeline_elapsed_ns(request_started)
                            .saturating_sub(spawn_started_ns);
                    }
                }
                let fixed_prepare_ns =
                    u64::try_from(layer_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
                if from_prefetch {
                    loaded.fixed.window_residual_ns = fixed_prepare_ns
                        .saturating_sub(loaded.fixed.window_join_ns)
                        .saturating_sub(loaded.fixed.fixed_h2d_ns)
                        .saturating_sub(loaded.fixed.window_spawn_ns);
                }
                let hash_ids = load_layer_hash_ids_for(spec, model_dir, layer, &token_ids)?;
                let router_rows = active_requests
                    .iter()
                    .enumerate()
                    .map(|(row, _)| {
                        b2_router_metadata_row(
                            loaded.raw.runtime.weights.hash_router,
                            &hash_ids,
                            row,
                            active_requests.len(),
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                physical.fixed_source_bytes += loaded.fixed_source_bytes;
                physical.fixed_h2d_bytes += loaded.fixed_h2d_bytes;
                for row in &router_rows {
                    physical.record_request_router_metadata(
                        loaded.raw.runtime.weights.hash_router,
                        row,
                    )?;
                }
                peak_device_bytes = peak_device_bytes.max(device_used_bytes()?);
                let wo_a_decode_started = std::time::Instant::now();
                launch_e4m3fn_to_bf16(module, stream, &loaded.raw.wo_a_raw, &mut scratch)?;
                stream.synchronize()?;
                let wo_a_decode_sync_ns =
                    u64::try_from(wo_a_decode_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
                std::mem::swap(&mut loaded.raw.runtime.weights.wo_a, &mut scratch);
                let mut flash_stage = None;
                let layer_run = (|| -> Result<(), Box<dyn Error>> {
                    for ((row, &request), &token_id) in
                        active_requests.iter().enumerate().zip(&token_ids)
                    {
                        let state = &mut requests[request];
                        let (next, digest, stage) = run_b2_layer(
                            context,
                            module,
                            stream,
                            flash_cublas,
                            &loaded.raw.runtime,
                            lanes,
                            router_rows[row],
                            &state.hidden,
                            token_id,
                            state.position,
                            &mut state.persistent,
                            Some(&mut routed_slot_scratch),
                        )?;
                        physical.routed_source_bytes += stage.routed_source_bytes;
                        physical.routed_h2d_bytes += stage.routed_h2d_bytes;
                        state.hidden = next;
                        layers[request].push(digest);
                        if flash_b1_profile {
                            flash_stage = Some(stage);
                        }
                    }
                    Ok(())
                })();
                std::mem::swap(&mut loaded.raw.runtime.weights.wo_a, &mut scratch);
                layer_run?;
                if let Some(stage) = flash_stage {
                    let layer_outer_ns =
                        u64::try_from(layer_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
                    let profile = flash_b1_layer_phase_profile_json(
                        layer,
                        layer_outer_ns,
                        fixed_prepare_ns,
                        &loaded.fixed,
                        wo_a_decode_sync_ns,
                        &stage,
                    )?;
                    let timing = &mut requests[0].flash_b1_phase;
                    timing.layer_outer_ns = timing
                        .layer_outer_ns
                        .checked_add(layer_outer_ns)
                        .ok_or("Flash B1 layer outer timing overflow")?;
                    timing.layers.push(profile);
                }
            }
            if prefetched_next.is_some() {
                return Err("Flash B1 fixed prefetch left an unconsumed host slot".into());
            }
            Ok(())
        })?;
        if let Some(stop) = parity_stop_after_layer {
            if REQUESTS != 1 || active_requests.len() != 1 {
                return Err("Flash parity stop requires one active B1 request".into());
            }
            let elapsed_ns = u64::try_from(token_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
            let phase_profile = flash_b1_profile
                .then(|| {
                    flash_b1_request_phase_profile_json(&requests[0].flash_b1_phase, elapsed_ns)
                })
                .transpose()?;
            let path = write_flash_parity_stop(stop, spec, &layers[0], phase_profile.as_ref())?;
            return Err(Box::new(FlashParityStopped { path }));
        }
        let elapsed_ns = u64::try_from(token_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        for (&request, &token_id) in active_requests.iter().zip(&token_ids) {
            let state = &mut requests[request];
            if layers[request].len() != spec.layers {
                return Err(format!(
                    "{} row-set request committed fewer than {} layers",
                    spec.name, spec.layers
                )
                .into());
            }
            let input_kind = if state.position < state.prompt_ids.len() {
                "prompt"
            } else {
                "generated"
            };
            let ordered_layer_digest_sha256 =
                B2LayerDigest::ordered_chain_sha256(&layers[request])?;
            state.tokens.push(serde_json::json!({
                "token_position": state.position,
                "input_id": token_id,
                "input_kind": input_kind,
                "e2e_ns": elapsed_ns,
                "hidden_bf16_digest": format!("{:016x}", fnv1a_u16(&state.hidden)),
                "ordered_layer_digest_sha256": ordered_layer_digest_sha256,
                "layers": if capture_layer_trace { layers[request].iter().map(B2LayerDigest::json).collect::<Vec<_>>() } else { Vec::new() },
            }));
            if b1_needs_final_head(state.prompt_ids.len(), state.position) {
                let final_head_started = std::time::Instant::now();
                let final_weights =
                    final_weights.ok_or("row-set generation needs final head weights")?;
                let final_head_dump = final_head_dump_dir()?;
                if let Some(directory) = final_head_dump.as_deref() {
                    write_final_head_input(directory, &state.hidden)?;
                }
                let mut gpu_exact_telemetry = None;
                let cpu = match final_head_arm {
                    FlashFinalHeadArm::Cpu => {
                        v4_state::final_checkpoint_head_cpu_timed(final_weights, &state.hidden, 1)?
                    }
                    FlashFinalHeadArm::GpuExact => {
                        let (gpu, artifact_dir) = if let Some(workspace) = flash_final_head_gpu {
                            (workspace.run_runtime(&state.hidden)?, None)
                        } else {
                            let output_dir = flash_final_head_gpu_production_dir()?;
                            // Production execution must not require an unrelated
                            // capture directory. Set the official directory when
                            // an independent parity comparison is available.
                            let verification_manifest = std::env::var_os(
                                "COLIBRI_V4_FLASH_FINAL_HEAD_OFFICIAL_DIR",
                            )
                            .map(|_| serde_json::json!({"config": final_head_capture_config()}));
                            (
                                run_flash_final_head_gpu_exact(
                                    &output_dir,
                                    verification_manifest.as_ref(),
                                    &state.hidden,
                                    final_weights,
                                )?,
                                Some(output_dir),
                            )
                        };
                        let wall_ns = (gpu.timing["total"].as_f64().unwrap_or(f64::INFINITY)
                            * 1_000_000.0) as u64;
                        gpu_exact_telemetry = Some(serde_json::json!({
                            "timing_ms":gpu.timing, "memory_bytes":gpu.memory,
                            "artifact_dir":artifact_dir,
                        }));
                        v4_state::FinalHeadCpuRun {
                            output: gpu.output,
                            read_ns: 0,
                            streamed_source_bytes: gpu.source_bytes,
                            wall_ns,
                        }
                    }
                };
                if let Some(directory) = final_head_dump.as_deref() {
                    write_final_head_output(directory, &state.hidden, &cpu, "base_capture")?;
                }
                let final_head_wall_ns =
                    u64::try_from(final_head_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
                if cpu.wall_ns > final_head_wall_ns {
                    return Err("Flash B1 final-head timing exceeds its outer scope".into());
                }
                if flash_b1_profile {
                    state.flash_b1_phase.final_head_outer_ns = state
                        .flash_b1_phase
                        .final_head_outer_ns
                        .checked_add(final_head_wall_ns)
                        .ok_or("Flash B1 final-head timing overflow")?;
                    state.flash_b1_phase.final_heads.push(serde_json::json!({
                        "token_position": state.position,
                        "outer": {"wall_ns": final_head_wall_ns},
                        "top_level": {
                            "cpu_streamed_final_head": {
                                "wall_ns": cpu.wall_ns,
                                "streamed_weight_read_ns": cpu.read_ns,
                                "source_bytes": cpu.streamed_source_bytes,
                            },
                        },
                        "unattributed_gap_ns": final_head_wall_ns.saturating_sub(cpu.wall_ns),
                        "closure": {
                            "equation": "cpu_streamed_final_head.wall_ns + unattributed_gap_ns == outer.wall_ns",
                            "exact": true,
                        },
                    }));
                }
                state
                    .physical
                    .record_final_head_source_bytes(cpu.streamed_source_bytes)?;
                physical.record_final_head_source_bytes(cpu.streamed_source_bytes)?;
                let next_id = i64::try_from(cpu.output.argmax[0])?;
                let decision = greedy_argmax_decision(
                    state.prompt_ids.len(),
                    state.position,
                    state.generated_ids.len(),
                    greedy_steps,
                    next_id,
                );
                let decision_name = match decision {
                    GreedyDecision::IgnorePrompt => "ignore_prompt",
                    GreedyDecision::EmitAndContinue => "emit_and_continue",
                    GreedyDecision::EmitAndStop(GreedyStopReason::Eos) => "emit_and_stop_eos",
                    GreedyDecision::EmitAndStop(GreedyStopReason::MaxNewTokens) => {
                        "emit_and_stop_max_new_tokens"
                    }
                };
                state.finals.push(serde_json::json!({
                    "token_position": state.position,
                    "input_id": token_id,
                    "input_kind": input_kind,
                    "head": match final_head_arm { FlashFinalHeadArm::Cpu => "canonical_cpu_streamed_bf16", FlashFinalHeadArm::GpuExact => "gpu_exact" },
                    "collapsed_bf16_digest": format!("{:016x}", fnv1a_u16(&cpu.output.collapsed_bf16)),
                    "normalized_bf16_digest": format!("{:016x}", fnv1a_u16(&cpu.output.normalized_bf16)),
                    "logits_f32_digest": format!("{:016x}", fnv1a_f32(&cpu.output.logits_f32)),
                    "argmax": cpu.output.argmax[0],
                    "gpu_exact": gpu_exact_telemetry,
                    "decision": decision_name,
                    "status": "PASS",
                }));
                match decision {
                    GreedyDecision::IgnorePrompt => {}
                    GreedyDecision::EmitAndContinue => on_generated(request, next_id, None)?,
                    GreedyDecision::EmitAndStop(reason) => {
                        on_generated(request, next_id, Some(reason))?
                    }
                }
                match decision {
                    GreedyDecision::IgnorePrompt => {}
                    GreedyDecision::EmitAndContinue => {
                        state.generated_ids.push(next_id);
                        state.input_ids.push(next_id);
                    }
                    GreedyDecision::EmitAndStop(reason) => {
                        state.generated_ids.push(next_id);
                        state.input_ids.push(next_id);
                        state.stop_reason = Some(reason);
                        active[request] = false;
                    }
                }
            }
            state.position = state
                .position
                .checked_add(1)
                .ok_or("B2 token position overflow")?;
        }
    }
    if requests.iter().any(|request| request.stop_reason.is_none()) {
        return Err("B2 greedy generation ended without a stop reason".into());
    }
    let duration_ns = u64::try_from(request_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
    Ok(B2GenerateArm {
        requests: requests
            .try_into()
            .map_err(|_| "row-set request count changed")?,
        physical,
        duration_ns,
        peak_device_bytes,
    })
}

pub(crate) fn run_real_v4_b2_generate<const REQUESTS: usize>(
    context: &std::sync::Arc<CudaContext>,
    module: &kernels::LoadedModule,
    stream: &std::sync::Arc<CudaStream>,
    token_ids_env: &str,
    mode: &str,
    schema: &str,
) -> Result<(), Box<dyn Error>> {
    let (prompts, greedy_steps, forward_tokens) =
        b2_generate_inputs_from_env::<REQUESTS>(token_ids_env)?;
    let result_path = std::env::var_os("COLIBRI_V4_RESULT_JSONL")
        .map(std::path::PathBuf::from)
        .ok_or("set COLIBRI_V4_RESULT_JSONL")?;
    let mut result = PendingResultFile::reserve(result_path)?;
    let model_dir = v4_model_dir()?;
    let final_weights = v4_state::load_final_head(&model_dir)?;
    let mut lanes = (0..V4_QD4_LANES)
        .map(|_| {
            let lane_stream = context.new_stream()?;
            ExpertSlot::new_empty(context, module, &lane_stream)
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    let mut scratch =
        DeviceBuffer::<u16>::zeroed(stream, (128 * 1024 * 1024) / std::mem::size_of::<u16>())?;
    let arm = run_v4_b2_generate_arm(
        &v4_spec::PRO,
        context,
        module,
        stream,
        None,
        &model_dir,
        Some(&final_weights),
        None,
        &mut lanes,
        &mut scratch,
        prompts,
        greedy_steps,
        forward_tokens,
        true,
        true,
        None,
        &mut |_, _, _| Ok(()),
    )?;
    writeln!(
        result.file_mut(),
        "{}",
        serde_json::json!({
            "event": "run", "schema": schema, "implementation": "rust-cuda",
            "generation_mode": "id_only_greedy_argmax", "model_revision": v4_state::REVISION,
            "model_dir": model_dir, "layers": v4_state::V4_LAYER_COUNT, "greedy_steps": greedy_steps,
            "eos_token_id": V4_EOS_TOKEN_ID, "final_head": "canonical_cpu_streamed_bf16",
            "qd4_lanes": V4_QD4_LANES, "raw_scratch_bytes": scratch.num_bytes(),
            "request_e2e_ns": arm.duration_ns, "physical": arm.physical.json(),
            "requests": arm.requests.iter().enumerate().map(|(request, state)| b2_generate_request_json(request, state)).collect::<Vec<_>>(),
            "peak_device_used_bytes": arm.peak_device_bytes,
        })
    )?;
    result.commit()?;
    println!(
        "{{\"status\":\"PASS\",\"mode\":{},\"requests\":{REQUESTS},\"result_jsonl\":{}}}",
        serde_json::to_string(mode)?,
        serde_json::to_string(result.path())?
    );
    Ok(())
}

pub(crate) fn run_real_v4_flash_base(
    context: &std::sync::Arc<CudaContext>,
    module: &kernels::LoadedModule,
    stream: &std::sync::Arc<CudaStream>,
) -> Result<(), Box<dyn Error>> {
    let spec = &v4_spec::FLASH_0731;
    let mhc_linear_arm = flash_mhc_linear_arm(spec)?;
    let router_linear_arm = flash_router_linear_arm(spec)?;
    let router_score_arm = flash_router_score_arm(spec)?;
    let fp4_accum_arm = flash_fp4_accum_arm(spec)?;
    let final_head_arm = flash_final_head_arm(spec)?;
    let flash_cublas = (mhc_linear_arm == FlashMhcLinearArm::Cublas
        || router_linear_arm == FlashMhcLinearArm::Cublas)
        .then(|| FlashCublasSession::new(stream))
        .transpose()?;
    let cublas = flash_cublas.as_ref().map(|session| &session.cublas);
    let mhc_cublas = (mhc_linear_arm == FlashMhcLinearArm::Cublas)
        .then(|| flash_mhc_cublas_provenance(cublas.ok_or("Flash mHC provenance session missing")?))
        .transpose()?;
    let router_cublas = (router_linear_arm == FlashMhcLinearArm::Cublas)
        .then(|| {
            flash_router_cublas_provenance(cublas.ok_or("Flash router provenance session missing")?)
        })
        .transpose()?;
    let token_ids_env = "COLIBRI_V4_FLASH_TOKEN_IDS";
    let prompts = parse_b2_generate_token_ids_for::<1>(
        spec,
        &std::env::var(token_ids_env)
            .map_err(|_| format!("set {token_ids_env}='<comma-separated ids>'"))?,
        token_ids_env,
    )?;
    let greedy_steps = std::env::var("COLIBRI_V4_GREEDY_STEPS")
        .map_err(|_| "set COLIBRI_V4_GREEDY_STEPS for Flash-0731 generation")?
        .parse::<usize>()?;
    let forward_tokens = validate_b2_generate_inputs::<1>(&prompts, greedy_steps)?;
    let parity_stop = flash_parity_stop_after_layer(spec)?;
    let model_dir = v4_model_dir_for(spec)?;
    let final_weights = parity_stop
        .is_none()
        .then(|| v4_state::load_final_head_for(spec, &model_dir))
        .transpose()?;
    // Reuse one GPU final-head workspace across all greedy steps.  The
    // production path must not recreate trace artifacts for every token;
    // multi-token generation is a real serving path, not a one-shot probe.
    let flash_final_head_gpu = match final_head_arm {
        FlashFinalHeadArm::Cpu => None,
        FlashFinalHeadArm::GpuExact => Some(FlashDsparkFinalHeadGpuWorkspace::new(
            final_weights
                .as_ref()
                .ok_or("Flash GPU exact final head requires final-head weights")?,
        )?),
    };
    let mut lanes = Vec::<ExpertSlot<'_>>::new();
    let wo_a_elements = spec.q_heads * spec.q_head_dim * spec.o_lora_rank;
    let mut scratch = DeviceBuffer::<u16>::zeroed(stream, wo_a_elements)?;
    let arm = match run_v4_b2_generate_arm(
        spec,
        context,
        module,
        stream,
        flash_cublas.as_ref(),
        &model_dir,
        final_weights.as_ref(),
        flash_final_head_gpu.as_ref(),
        &mut lanes,
        &mut scratch,
        prompts,
        greedy_steps,
        forward_tokens,
        !parity_stop.is_some(),
        true,
        parity_stop.as_ref(),
        &mut |_, _, _| Ok(()),
    ) {
        Ok(arm) => arm,
        Err(error) if error.downcast_ref::<FlashParityStopped>().is_some() => return Ok(()),
        Err(error) => return Err(error),
    };
    let result_path = std::env::var_os("COLIBRI_V4_RESULT_JSONL")
        .map(std::path::PathBuf::from)
        .ok_or("set COLIBRI_V4_RESULT_JSONL")?;
    let mut result = PendingResultFile::reserve(result_path)?;
    let phase_profile =
        flash_b1_request_phase_profile_json(&arm.requests[0].flash_b1_phase, arm.duration_ns)?;
    writeln!(
        result.file_mut(),
        "{}",
        serde_json::json!({
            "event": "run",
            "schema": "colibri-deepseek-v4-flash-0731-base-v1",
            "implementation": "rust-cuda",
            "profile": spec.name,
            "model_revision": spec.revision,
            "model_dir": model_dir,
            "layers": spec.layers,
            "hidden_size": spec.hidden,
            "attention_heads": spec.q_heads,
            "routed_experts": spec.routed_experts,
            "experts_per_token": spec.experts_per_token,
            "generation_mode": "id_only_greedy_argmax",
            "execution": "base_only_no_speculation",
            "base_only_no_speculation": spec.base_only_no_speculation,
            "dspark": "disabled",
            "mhc_linear": {
                "arm": mhc_linear_arm.name(),
                "cublas": mhc_cublas,
            },
            "router_linear": {
                "arm": router_linear_arm.name(),
                "cublas": router_cublas,
            },
            "router_score": flash_router_score_provenance(router_score_arm),
            "fp4_accum": fp4_accum_arm.name(),
            "greedy_steps": greedy_steps,
            "eos_token_id": V4_EOS_TOKEN_ID,
            "final_head": match final_head_arm { FlashFinalHeadArm::Cpu => "canonical_cpu_streamed_bf16", FlashFinalHeadArm::GpuExact => "gpu_exact" },
            "qd4_lanes": 0,
            "raw_scratch_bytes": scratch.num_bytes(),
            "request_e2e_ns": arm.duration_ns,
            "phase_profile": phase_profile,
            "physical": arm.physical.json(),
            "requests": arm.requests.iter().enumerate().map(|(request, state)| b2_generate_request_json(request, state)).collect::<Vec<_>>(),
            "peak_device_used_bytes": arm.peak_device_bytes,
        })
    )?;
    result.commit()?;
    println!(
        "{{\"status\":\"PASS\",\"mode\":\"v4-flash-0731-base\",\"base_only_no_speculation\":true,\"dspark\":\"disabled\",\"result_jsonl\":{}}}",
        serde_json::to_string(result.path())?
    );
    Ok(())
}

#[derive(Debug)]
pub(crate) struct FlashServeCancelled;

impl std::fmt::Display for FlashServeCancelled {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("CANCELLED")
    }
}

impl Error for FlashServeCancelled {}

pub(crate) fn run_real_v4_flash_serve(
    context: &std::sync::Arc<CudaContext>,
    module: &kernels::LoadedModule,
    stream: &std::sync::Arc<CudaStream>,
) -> Result<(), Box<dyn Error>> {
    use std::collections::VecDeque;
    use std::io::{BufReader, BufWriter};
    use std::sync::mpsc;

    let spec = &v4_spec::FLASH_0731;
    let model_dir = v4_model_dir_for(spec)?;
    let tokenizer = v4_tokenizer::V4Tokenizer::open(&model_dir)?;
    let final_weights = v4_state::load_final_head_for(spec, &model_dir)?;
    let flash_final_head_gpu = match flash_final_head_arm(spec)? {
        FlashFinalHeadArm::Cpu => None,
        FlashFinalHeadArm::GpuExact => Some(FlashDsparkFinalHeadGpuWorkspace::new(&final_weights)?),
    };
    let wo_a_elements = spec.q_heads * spec.q_head_dim * spec.o_lora_rank;
    let mut scratch = DeviceBuffer::<u16>::zeroed(stream, wo_a_elements)?;
    let mut lanes = Vec::<ExpertSlot<'_>>::new();
    let flash_cublas = (flash_mhc_linear_arm(spec)? == FlashMhcLinearArm::Cublas
        || flash_router_linear_arm(spec)? == FlashMhcLinearArm::Cublas)
        .then(|| FlashCublasSession::new(stream))
        .transpose()?;

    let (command_tx, command_rx) = mpsc::channel::<Result<v4_serve::Command, String>>();
    std::thread::spawn(move || {
        let stdin = std::io::stdin();
        let mut input = BufReader::new(stdin.lock());
        loop {
            match v4_serve::read_command(&mut input) {
                Ok(Some(command)) => {
                    if command_tx.send(Ok(command)).is_err() {
                        return;
                    }
                }
                Ok(None) => return,
                Err(error) => {
                    let _ = command_tx.send(Err(error.to_string()));
                    return;
                }
            }
        }
    });

    let stdout = std::io::stdout();
    let mut output = BufWriter::new(stdout.lock());
    v4_serve::write_ready(&mut output)?;
    let mut pending = VecDeque::new();
    loop {
        let command = if let Some(command) = pending.pop_front() {
            command
        } else {
            match command_rx.recv() {
                Ok(command) => command,
                Err(_) => return Ok(()),
            }
        };
        let command = match command {
            Ok(command) => command,
            Err(message) => {
                v4_serve::write_error(&mut output, "0", &format!("PROTOCOL {message}"))?;
                return Ok(());
            }
        };
        let v4_serve::Command::Submit(submit) = command else {
            continue;
        };
        let request_id = submit.request_id.clone();
        let formatted = match v4_chat::format_messages_json_0731(
            &submit.payload,
            v4_chat::ThinkingMode::Chat,
        ) {
            Ok(formatted) => formatted,
            Err(message) => {
                v4_serve::write_error(&mut output, &request_id, &format!("PROMPT {message}"))?;
                continue;
            }
        };
        let prompt_ids = match tokenizer.encode_raw(&formatted) {
            Ok(ids) => ids,
            Err(error) => {
                v4_serve::write_error(&mut output, &request_id, &format!("TOKENIZER {error}"))?;
                continue;
            }
        };
        if let Err(message) = v4_serve::validate_submit(&submit, prompt_ids.len()) {
            v4_serve::write_error(&mut output, &request_id, &message)?;
            continue;
        }
        let forward_tokens = match greedy_forward_tokens(prompt_ids.len(), submit.max_tokens) {
            Ok(tokens) if tokens <= v4_state::KV_WINDOW => tokens,
            Ok(_) => {
                v4_serve::write_error(&mut output, &request_id, "CONTEXT_WINDOW_EXCEEDED")?;
                continue;
            }
            Err(error) => {
                v4_serve::write_error(&mut output, &request_id, &error.to_string())?;
                continue;
            }
        };

        let started = std::time::Instant::now();
        let mut boundaries = Vec::with_capacity(submit.max_tokens);
        let mut generated = Vec::with_capacity(submit.max_tokens);
        let mut decoded = String::new();
        let mut on_generated = |_: usize,
                                token_id: i64,
                                stop: Option<GreedyStopReason>|
         -> Result<(), Box<dyn Error>> {
            while let Ok(command) = command_rx.try_recv() {
                match command {
                    Ok(v4_serve::Command::Cancel(cancel_id)) if cancel_id == request_id => {
                        return Err(Box::new(FlashServeCancelled));
                    }
                    other => pending.push_back(other),
                }
            }
            boundaries.push(std::time::Instant::now());
            generated.push(token_id);
            if stop != Some(GreedyStopReason::Eos) {
                let next = tokenizer.decode(&generated)?;
                let delta = next.strip_prefix(&decoded).unwrap_or(&next);
                v4_serve::write_data(&mut output, &request_id, delta.as_bytes())?;
                decoded = next;
            }
            Ok(())
        };
        let result = run_v4_b2_generate_arm(
            spec,
            context,
            module,
            stream,
            flash_cublas.as_ref(),
            &model_dir,
            Some(&final_weights),
            flash_final_head_gpu.as_ref(),
            &mut lanes,
            &mut scratch,
            [prompt_ids.clone()],
            submit.max_tokens,
            [forward_tokens],
            false,
            false,
            None,
            &mut on_generated,
        );
        drop(on_generated);
        let arm = match result {
            Ok(arm) => arm,
            Err(error) if error.downcast_ref::<FlashServeCancelled>().is_some() => {
                v4_serve::write_error(&mut output, &request_id, "CANCELLED")?;
                continue;
            }
            Err(error) => {
                v4_serve::write_error(&mut output, &request_id, &format!("ENGINE {error}"))?;
                continue;
            }
        };
        let request = &arm.requests[0];
        let elapsed = started.elapsed().as_secs_f64();
        let completion_tokens = request.generated_ids.len();
        let tokens_per_second = completion_tokens as f64 / elapsed.max(f64::EPSILON);
        let ttft_ms = boundaries.first().map_or(elapsed * 1_000.0, |time| {
            time.duration_since(started).as_secs_f64() * 1_000.0
        });
        let itl_ms = if boundaries.len() < 2 {
            0.0
        } else {
            boundaries
                .last()
                .unwrap()
                .duration_since(boundaries[0])
                .as_secs_f64()
                * 1_000.0
                / (boundaries.len() - 1) as f64
        };
        let stop = request
            .stop_reason
            .ok_or("Flash serve request lacked stop reason")?;
        v4_serve::write_done(
            &mut output,
            &request_id,
            completion_tokens,
            tokens_per_second,
            prompt_ids.len(),
            stop == GreedyStopReason::MaxNewTokens,
            ttft_ms,
            itl_ms,
            b2_generate_stop_reason(stop),
            &request.generated_ids,
        )?;
    }
}

pub(crate) fn validate_flash_dspark_greedy_mode(temperature: f32) -> Result<(), Box<dyn Error>> {
    if !temperature.is_finite() || temperature != 0.0 {
        return Err("Flash-0731 DSpark exact mode requires temperature=0".into());
    }
    Ok(())
}

pub(crate) fn flash_dspark_state_capacity(
    prompt_rows: usize,
    greedy_steps: usize,
) -> Result<(usize, usize), Box<dyn Error>> {
    if prompt_rows == 0 || greedy_steps == 0 {
        return Err("Flash-0731 DSpark state capacity requires nonzero rows".into());
    }
    let max_new_tokens = greedy_steps
        .checked_add(v4_state::DSPARK_B5_VALIDATED_INPUTS)
        .ok_or("Flash-0731 DSpark generation capacity overflow")?;
    let context_rows = prompt_rows
        .checked_add(max_new_tokens - 1)
        .ok_or("Flash-0731 DSpark context capacity overflow")?;
    Ok((max_new_tokens, context_rows))
}

pub(crate) const FLASH_DSPARK_RSS_PROBE_CALLS: usize = 3;
pub(crate) const FLASH_DSPARK_RSS_PROBE_MAX_STEP_BYTES: usize = 512 * 1024 * 1024;

pub(crate) fn flash_dspark_rss_probe_enabled() -> Result<bool, Box<dyn Error>> {
    optional_one_flag("COLIBRI_V4_FLASH_DSPARK_RSS_PROBE")
}

pub(crate) fn flash_dspark_rss_probe_sample(
    position: usize,
    token_id: i64,
    argmax: u32,
) -> Result<serde_json::Value, Box<dyn Error>> {
    Ok(serde_json::json!({
        "position":position,
        "token_id":token_id,
        "argmax":argmax,
        "rss_bytes":process_rss_bytes()?,
        "mem_available_bytes":host_available_bytes()?,
        "head_runtime":"shared_compute_only",
    }))
}

pub(crate) fn flash_dspark_rss_probe_gate(samples: &[serde_json::Value]) -> Result<usize, Box<dyn Error>> {
    if samples.len() != FLASH_DSPARK_RSS_PROBE_CALLS {
        return Err("Flash DSpark RSS probe did not collect three base calls".into());
    }
    let mut max_step = 0_usize;
    for pair in samples.windows(2) {
        let before = pair[0]["rss_bytes"]
            .as_u64()
            .ok_or("RSS probe first sample lost VmRSS")?;
        let after = pair[1]["rss_bytes"]
            .as_u64()
            .ok_or("RSS probe second sample lost VmRSS")?;
        max_step = max_step.max(usize::try_from(after.saturating_sub(before))?);
    }
    if max_step > FLASH_DSPARK_RSS_PROBE_MAX_STEP_BYTES {
        return Err(format!(
            "Flash DSpark RSS probe grew {max_step} bytes between base calls; exceeds {} byte no-linear-growth gate",
            FLASH_DSPARK_RSS_PROBE_MAX_STEP_BYTES
        ).into());
    }
    Ok(max_step)
}

pub(crate) fn flash_dspark_markov_greedy(
    base_logits: &[f32],
    markov_w1: &[u16],
    markov_w2: &[u16],
    vocab: usize,
    rank: usize,
    anchor: u32,
) -> Result<([u32; 5], Vec<u16>), Box<dyn Error>> {
    let rows = v4_state::DSPARK_B5_DRAFT_TOKENS;
    if vocab == 0
        || rank == 0
        || base_logits.len() != rows * vocab
        || markov_w1.len() != vocab * rank
        || markov_w2.len() != vocab * rank
        || usize::try_from(anchor)? >= vocab
    {
        return Err("Flash-0731 sequential Markov contract mismatch".into());
    }
    let mut proposals = [0_u32; 5];
    let mut embeds = Vec::with_capacity(rows * rank);
    let mut previous = usize::try_from(anchor)?;
    for row in 0..rows {
        let markov = &markov_w1[previous * rank..(previous + 1) * rank];
        embeds.extend_from_slice(markov);
        let mut best_token = 0_usize;
        let mut best_logit = f32::NEG_INFINITY;
        for token in 0..vocab {
            let bias = markov
                .iter()
                .zip(&markov_w2[token * rank..(token + 1) * rank])
                .map(|(&left, &right)| v4_router::bf16_to_f32(left) * v4_router::bf16_to_f32(right))
                .sum::<f32>();
            let value = base_logits[row * vocab + token] + bias;
            if token == 0 || value > best_logit {
                best_token = token;
                best_logit = value;
            }
        }
        proposals[row] = u32::try_from(best_token)?;
        previous = best_token;
    }
    Ok((proposals, embeds))
}

pub(crate) fn flash_hc_mean(hidden: &[u16]) -> Result<Vec<u16>, Box<dyn Error>> {
    let spec = &v4_spec::FLASH_0731;
    if hidden.len() != v4_state::HC * spec.hidden {
        return Err("Flash-0731 auxiliary hidden row contract mismatch".into());
    }
    Ok((0..spec.hidden)
        .map(|column| {
            let sum = (0..v4_state::HC)
                .map(|copy| v4_router::bf16_to_f32(hidden[copy * spec.hidden + column]))
                .sum::<f32>();
            v4_router::f32_to_bf16_rne(sum / v4_state::HC as f32)
        })
        .collect())
}

pub(crate) fn run_flash_base_input(
    context: &std::sync::Arc<CudaContext>,
    module: &kernels::LoadedModule,
    stream: &std::sync::Arc<CudaStream>,
    flash_cublas: Option<&FlashCublasSession>,
    model_dir: &std::path::Path,
    gpu_exact: &FlashDsparkFinalHeadGpuWorkspace,
    scratch: &mut DeviceBuffer<u16>,
    persistent: &mut v4_state::V4SerialPersistentState,
    token_id: i64,
    position: usize,
) -> Result<(Vec<u16>, u32), Box<dyn Error>> {
    let spec = &v4_spec::FLASH_0731;
    let mut hidden = prepare_b2_initial_hidden_with_source_bytes_for(spec, model_dir, token_id)?.0;
    let mut captures = Vec::with_capacity(3 * spec.hidden);
    let mut lanes = Vec::<ExpertSlot<'_>>::new();
    // One routed slot for the whole forward pass: the page-lock inside it is
    // 17 ms, so it belongs to the request, not to each of the 43 layers.
    let mut routed_slot_scratch = RoutedSlotScratch::new();
    for layer in 0..spec.layers {
        let mut loaded = load_b2_raw_runtime_for(spec, model_dir, layer, stream)?;
        let hash_ids = load_layer_hash_ids_for(spec, model_dir, layer, &[token_id])?;
        launch_e4m3fn_to_bf16(module, stream, &loaded.raw.wo_a_raw, scratch)?;
        stream.synchronize()?;
        std::mem::swap(&mut loaded.raw.runtime.weights.wo_a, scratch);
        let result = run_b2_layer(
            context,
            module,
            stream,
            flash_cublas,
            &loaded.raw.runtime,
            &mut lanes,
            &hash_ids,
            &hidden,
            token_id,
            position,
            persistent,
            Some(&mut routed_slot_scratch),
        );
        std::mem::swap(&mut loaded.raw.runtime.weights.wo_a, scratch);
        hidden = result?.0;
        if matches!(layer, 40 | 41 | 42) {
            captures.extend(flash_hc_mean(&hidden)?);
        }
    }
    if captures.len() != 3 * spec.hidden {
        return Err("Flash-0731 did not capture layers 40/41/42".into());
    }
    let target = gpu_exact.run_runtime(&hidden)?.output.argmax[0];
    Ok((captures, u32::try_from(target)?))
}

pub(crate) fn flash_dspark_head(
    final_weights: &v4_state::FinalHeadWeights,
    stage2: &LoadedDsparkStage,
    hidden: &[u16],
    anchor: u32,
) -> Result<([u32; 5], v4_state::DsparkB5Observation), Box<dyn Error>> {
    let spec = &v4_spec::FLASH_0731;
    let rows = v4_state::DSPARK_B5_DRAFT_TOKENS;
    let head = stage2
        .head
        .as_ref()
        .ok_or("Flash-0731 stage 2 head is missing")?;
    let pre = v4_state::final_hc_gate_cpu(
        hidden,
        rows,
        v4_state::HC,
        spec.hidden,
        &head.hc_head_fn,
        head.hc_head_scale[0],
        &head.hc_head_base,
    )?;
    let collapsed = v4_state::final_hc_head_cpu(
        hidden,
        rows,
        v4_state::HC,
        spec.hidden,
        &pre,
        &head
            .norm
            .iter()
            .map(|&bits| v4_router::bf16_to_f32(bits))
            .collect::<Vec<_>>(),
        &vec![0.0; spec.hidden],
        1,
    )?;
    let logits = v4_state::final_checkpoint_logits_from_normalized_cpu(
        final_weights,
        &collapsed.normalized_bf16,
        rows,
    )?
    .logits_f32;
    let (proposals, markov_embeds) = flash_dspark_markov_greedy(
        &logits,
        &head.markov_w1,
        &head.markov_w2,
        spec.vocab,
        256,
        anchor,
    )?;
    let mut confidence = [0.0_f32; 5];
    for row in 0..rows {
        let markov = &markov_embeds[row * 256..(row + 1) * 256];
        let hidden_row = &collapsed.collapsed_bf16[row * spec.hidden..(row + 1) * spec.hidden];
        confidence[row] = hidden_row
            .iter()
            .chain(markov)
            .zip(&head.confidence)
            .map(|(&value, &weight)| v4_router::bf16_to_f32(value) * v4_router::bf16_to_f32(weight))
            .sum();
    }
    Ok((proposals, v4_state::DsparkB5Observation { confidence }))
}

pub(crate) fn flash_dspark_state_digests(
    state: &v4_state::DsparkB5PersistentState<
        v4_state::V4SerialPersistentState,
        v4_state::V4LayerPersistentState,
    >,
) -> [u64; 4] {
    [
        state.base.semantic_digest(),
        state.mtp_swa[0].semantic_digest(),
        state.mtp_swa[1].semantic_digest(),
        state.mtp_swa[2].semantic_digest(),
    ]
}

pub(crate) fn run_real_v4_flash_dspark(
    context: &std::sync::Arc<CudaContext>,
    module: &kernels::LoadedModule,
    stream: &std::sync::Arc<CudaStream>,
) -> Result<(), Box<dyn Error>> {
    let temperature = std::env::var("COLIBRI_V4_TEMPERATURE")
        .unwrap_or_else(|_| "0".into())
        .parse::<f32>()?;
    validate_flash_dspark_greedy_mode(temperature)?;
    let spec = &v4_spec::FLASH_0731;
    let prompts = parse_b2_generate_token_ids_for::<1>(
        spec,
        &std::env::var("COLIBRI_V4_FLASH_TOKEN_IDS")
            .map_err(|_| "set COLIBRI_V4_FLASH_TOKEN_IDS='<comma-separated ids>'")?,
        "COLIBRI_V4_FLASH_TOKEN_IDS",
    )?;
    let prompt = &prompts[0];
    if prompt.len() < 2 {
        return Err("Flash-0731 DSpark needs at least two prompt tokens".into());
    }
    let greedy_steps = std::env::var("COLIBRI_V4_GREEDY_STEPS")
        .map_err(|_| "set COLIBRI_V4_GREEDY_STEPS for Flash-0731 DSpark")?
        .parse::<usize>()?;
    if greedy_steps == 0 {
        return Err("Flash-0731 DSpark requires at least one greedy step".into());
    }
    let result_path = std::env::var_os("COLIBRI_V4_RESULT_JSONL")
        .map(std::path::PathBuf::from)
        .ok_or("set COLIBRI_V4_RESULT_JSONL")?;
    let mut result = PendingResultFile::reserve(result_path)?;
    if flash_final_head_arm(spec)? != FlashFinalHeadArm::GpuExact {
        return Err("Flash-0731 DSpark requires COLIBRI_V4_FLASH_FINAL_HEAD_ARM=gpu_exact".into());
    }
    let model_dir = v4_model_dir_for(spec)?;
    v4_state::validate_flash_0731_dspark_model_dir(&model_dir)?;
    let final_weights = v4_state::load_final_head_for(spec, &model_dir)?;
    let gpu_exact = FlashDsparkFinalHeadGpuWorkspace::new(&final_weights)?;
    let flash_cublas = (flash_mhc_linear_arm(spec)? == FlashMhcLinearArm::Cublas
        || flash_router_linear_arm(spec)? == FlashMhcLinearArm::Cublas)
        .then(|| FlashCublasSession::new(stream))
        .transpose()?;
    let rss_probe = flash_dspark_rss_probe_enabled()?;
    if rss_probe && prompt.len() != 2 {
        return Err("Flash DSpark RSS probe requires exactly two prompt inputs".into());
    }
    let rss_probe_before = rss_probe
        .then(|| {
            Ok::<_, Box<dyn Error>>(serde_json::json!({
                "rss_bytes":process_rss_bytes()?,
                "mem_available_bytes":host_available_bytes()?,
            }))
        })
        .transpose()?;
    let mut rss_probe_samples = Vec::with_capacity(FLASH_DSPARK_RSS_PROBE_CALLS);
    let stages: [LoadedDsparkStage; v4_state::DSPARK_STAGES] = [0, 1, 2]
        .map(|stage| load_flash_dspark_stage(module, stream, &model_dir, stage))
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?
        .try_into()
        .map_err(|_| "Flash-0731 DSpark stage count changed")?;
    let (state_max_new_tokens, context_rows) =
        flash_dspark_state_capacity(prompt.len(), greedy_steps)?;
    let mut state = v4_state::DsparkB5PersistentState {
        base: v4_state::V4SerialPersistentState::new_for_spec(
            spec,
            prompt.len(),
            state_max_new_tokens,
        )?,
        mtp_swa: [0, 1, 2]
            .map(|stage| v4_state::V4LayerPersistentState::new_flash_dspark(stage, context_rows))
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?
            .try_into()
            .map_err(|_| "Flash-0731 DSpark state count changed")?,
    };
    let mut scratch =
        DeviceBuffer::<u16>::zeroed(stream, spec.q_heads * spec.q_head_dim * spec.o_lora_rank)?;
    let mut pending_anchor = 0_u32;
    for (position, &token) in prompt.iter().enumerate() {
        let (main_hidden, target) = run_flash_base_input(
            context,
            module,
            stream,
            flash_cublas.as_ref(),
            &model_dir,
            &gpu_exact,
            &mut scratch,
            &mut state.base,
            token,
            position,
        )?;
        if rss_probe {
            rss_probe_samples.push(flash_dspark_rss_probe_sample(position, token, target)?);
        }
        let main_x = flash_dspark_main_x(module, stream, &stages[0], &main_hidden)?;
        advance_flash_dspark_context(
            module,
            stream,
            &stages,
            &mut state.mtp_swa,
            &main_x,
            position,
        )?;
        pending_anchor = target;
    }
    let mut ledger = v4_state::DsparkB5CacheLedger::new(prompt.len());
    let mut generated = Vec::<u32>::with_capacity(greedy_steps);
    let mut rounds = Vec::new();
    while generated.len() < greedy_steps && pending_anchor != u32::try_from(V4_EOS_TOKEN_ID)? {
        let position = ledger.next_input_position();
        let base_before = state.clone();
        let mut prefix0 = state.clone();
        let (anchor_main_hidden, target0) = run_flash_base_input(
            context,
            module,
            stream,
            flash_cublas.as_ref(),
            &model_dir,
            &gpu_exact,
            &mut scratch,
            &mut prefix0.base,
            i64::from(pending_anchor),
            position,
        )?;
        if rss_probe {
            rss_probe_samples.push(flash_dspark_rss_probe_sample(
                position,
                i64::from(pending_anchor),
                target0,
            )?);
            let max_step_bytes = flash_dspark_rss_probe_gate(&rss_probe_samples)?;
            writeln!(
                result.file_mut(),
                "{}",
                serde_json::json!({
                    "event":"rss_probe",
                    "schema":"colibri-deepseek-v4-flash-0731-dspark-rss-probe-v1",
                    "full_dspark":false,
                    "binary":"same_workspace_binary",
                    "workspace":"one immutable FP32 final head shared across base calls",
                    "before":rss_probe_before,
                    "samples":rss_probe_samples,
                    "max_step_rss_bytes":max_step_bytes,
                    "max_allowed_step_rss_bytes":FLASH_DSPARK_RSS_PROBE_MAX_STEP_BYTES,
                    "conclusion":"PASS: excludes the prior approximately 2.6GB-per-base-call growth",
                })
            )?;
            result.commit()?;
            println!(
                "{{\"status\":\"PASS\",\"mode\":\"v4-flash-0731-dspark-rss-probe\",\"base_calls\":3,\"max_step_rss_bytes\":{max_step_bytes}}}"
            );
            return Ok(());
        }
        let main_x = flash_dspark_main_x(module, stream, &stages[0], &anchor_main_hidden)?;
        advance_flash_dspark_context(
            module,
            stream,
            &stages,
            &mut prefix0.mtp_swa,
            &main_x,
            position,
        )?;
        let draft_ids = [
            i64::from(pending_anchor),
            128_799,
            128_799,
            128_799,
            128_799,
        ];
        let embeddings = v4_state::load_for(spec, &model_dir, &draft_ids)?.embedding;
        let mut draft_hidden =
            Vec::with_capacity(v4_state::DSPARK_B5_DRAFT_TOKENS * v4_state::HC * spec.hidden);
        for embedding in embeddings.chunks_exact(spec.hidden) {
            for _ in 0..v4_state::HC {
                draft_hidden.extend_from_slice(embedding);
            }
        }
        let mut draft_states = base_before.mtp_swa.clone();
        for (stage, stage_state) in stages.iter().zip(&mut draft_states) {
            draft_hidden = run_flash_dspark_stage(
                context,
                module,
                stream,
                stage,
                stage_state,
                &draft_hidden,
                &main_x,
                position,
            )?;
        }
        if draft_states
            .iter()
            .zip(&prefix0.mtp_swa)
            .any(|(draft, target)| draft.semantic_digest() != target.semantic_digest())
        {
            return Err("Flash-0731 DSpark draft/main context commit diverged".into());
        }
        let (proposed, observation) =
            flash_dspark_head(&final_weights, &stages[2], &draft_hidden, pending_anchor)?;
        let mut targets = [0_u32; v4_state::DSPARK_B5_VALIDATED_INPUTS];
        targets[0] = target0;
        let mut prefixes = Vec::with_capacity(v4_state::DSPARK_B5_VALIDATED_INPUTS);
        prefixes.push(prefix0.clone());
        let mut working = prefix0;
        for (offset, &token) in proposed.iter().enumerate() {
            let input_position = position + offset + 1;
            let (main_hidden, target) = run_flash_base_input(
                context,
                module,
                stream,
                flash_cublas.as_ref(),
                &model_dir,
                &gpu_exact,
                &mut scratch,
                &mut working.base,
                i64::from(token),
                input_position,
            )?;
            let next_main_x = flash_dspark_main_x(module, stream, &stages[0], &main_hidden)?;
            advance_flash_dspark_context(
                module,
                stream,
                &stages,
                &mut working.mtp_swa,
                &next_main_x,
                input_position,
            )?;
            targets[offset + 1] = target;
            prefixes.push(working.clone());
        }
        let accepted_unbounded = proposed
            .iter()
            .zip(targets)
            .take_while(|(draft, target)| *draft == target)
            .count();
        let mut max_inputs =
            (greedy_steps - generated.len()).min(v4_state::DSPARK_B5_VALIDATED_INPUTS);
        let mut accepted_tokens =
            std::iter::once(pending_anchor).chain(proposed[..accepted_unbounded].iter().copied());
        if let Some(eos) = accepted_tokens.position(|token| token == V4_EOS_TOKEN_ID as u32) {
            max_inputs = max_inputs.min(eos + 1);
        }
        let prepared = v4_state::DsparkB5Prepared::from_prefixes(
            &base_before,
            ledger,
            prefixes
                .try_into()
                .map_err(|_| "Flash-0731 target prefix count changed")?,
        );
        let validated =
            prepared.validate_greedy_target_bounded(proposed, targets, observation, max_inputs)?;
        let decision = validated.decision();
        let emitted = std::iter::once(pending_anchor)
            .chain(proposed[..decision.accepted_drafts].iter().copied())
            .collect::<Vec<_>>();
        validated.commit_prefix(&mut state, &mut ledger, |left, right| {
            flash_dspark_state_digests(left) == flash_dspark_state_digests(right)
        })?;
        generated.extend(&emitted);
        pending_anchor = decision.pending_anchor;
        let digests = flash_dspark_state_digests(&state);
        rounds.push(serde_json::json!({
            "full_dspark": true,
            "start_position": position,
            "proposed": proposed,
            "target": targets,
            "accepted": decision.accepted_drafts,
            "committed_inputs": decision.committed_inputs,
            "bonus": (decision.accepted_drafts == 5).then_some(targets[5]),
            "pending": pending_anchor,
            "confidence": validated.observation().confidence,
            "state_digests": {
                "base": format!("{:016x}", digests[0]),
                "mtp0": format!("{:016x}", digests[1]),
                "mtp1": format!("{:016x}", digests[2]),
                "mtp2": format!("{:016x}", digests[3]),
            }
        }));
        if emitted.contains(&(V4_EOS_TOKEN_ID as u32)) {
            break;
        }
    }
    writeln!(
        result.file_mut(),
        "{}",
        serde_json::json!({
            "event": "run",
            "schema": "colibri-deepseek-v4-flash-0731-dspark-v1",
            "implementation": "rust-cuda",
            "profile": spec.name,
            "model_revision": spec.revision,
            "generation_mode": "greedy_exact",
            "temperature": 0,
            "full_dspark": true,
            "block_size": 5,
            "prompt_ids": prompt,
            "generated_ids": generated,
            "pending": pending_anchor,
            "rounds": rounds,
            "state_digests": flash_dspark_state_digests(&state).map(|value| format!("{value:016x}")),
        })
    )?;
    result.commit()?;
    println!(
        "{{\"status\":\"PASS\",\"mode\":\"v4-flash-0731-dspark\",\"full_dspark\":true,\"generated\":{},\"result_jsonl\":{}}}",
        generated.len(),
        serde_json::to_string(result.path())?
    );
    Ok(())
}

pub(crate) fn run_real_v4_b4_greedy_abba_bench(
    context: &std::sync::Arc<CudaContext>,
    module: &kernels::LoadedModule,
    stream: &std::sync::Arc<CudaStream>,
) -> Result<(), Box<dyn Error>> {
    let (prompts, greedy_steps, forward_tokens) =
        b2_generate_inputs_from_env::<4>("COLIBRI_V4_B4_TOKEN_IDS")?;
    if greedy_steps != 1 {
        return Err("COLIBRI_V4_B4_GREEDY_ABBA_BENCH requires COLIBRI_V4_GREEDY_STEPS=1".into());
    }
    let arm = std::env::var("COLIBRI_V4_B4_GREEDY_ABBA_ARM")?;
    if arm != "A" && arm != "B" {
        return Err("COLIBRI_V4_B4_GREEDY_ABBA_ARM must be A or B".into());
    }
    let result_path = std::env::var_os("COLIBRI_V4_B4_GREEDY_ABBA_RESULT_JSONL")
        .map(std::path::PathBuf::from)
        .ok_or("set COLIBRI_V4_B4_GREEDY_ABBA_RESULT_JSONL")?;
    let mut output = PendingResultFile::reserve(result_path)?;
    let model_dir = v4_model_dir()?;
    let final_weights = v4_state::load_final_head(&model_dir)?;
    let mut lanes = (0..V4_QD4_LANES)
        .map(|_| {
            let lane_stream = context.new_stream()?;
            ExpertSlot::new_empty(context, module, &lane_stream)
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    let mut scratch =
        DeviceBuffer::<u16>::zeroed(stream, (128 * 1024 * 1024) / std::mem::size_of::<u16>())?;
    writeln!(
        output.file_mut(),
        "{}",
        serde_json::json!({
            "event": "run", "schema": "colibri-v4-b4-greedy-abba-v1",
            "provenance": benchmark_provenance()?, "model_revision": v4_state::REVISION,
            "model_dir": model_dir, "greedy_steps": greedy_steps, "layers": v4_state::V4_LAYER_COUNT,
            "trace": "final_and_ledger_only", "arm": arm,
            "A": "two sequential B2 greedy row-set calls: 0;1 then 2;3",
            "B": "one B4 greedy row-set call: 0;1;2;3",
            "shared_final_head_loader_source_bytes": final_weights.loader_source_bytes,
            "qd4_lanes": V4_QD4_LANES, "raw_scratch_bytes": scratch.num_bytes(),
        })
    )?;
    let summary = if arm == "A" {
        let first = run_v4_b2_generate_arm(
            &v4_spec::PRO,
            context,
            module,
            stream,
            None,
            &model_dir,
            Some(&final_weights),
            None,
            &mut lanes,
            &mut scratch,
            [prompts[0].clone(), prompts[1].clone()],
            greedy_steps,
            [forward_tokens[0], forward_tokens[1]],
            false,
            false,
            None,
            &mut |_, _, _| Ok(()),
        )?
        .into_final_summary();
        let second = run_v4_b2_generate_arm(
            &v4_spec::PRO,
            context,
            module,
            stream,
            None,
            &model_dir,
            Some(&final_weights),
            None,
            &mut lanes,
            &mut scratch,
            [prompts[2].clone(), prompts[3].clone()],
            greedy_steps,
            [forward_tokens[2], forward_tokens[3]],
            false,
            false,
            None,
            &mut |_, _, _| Ok(()),
        )?
        .into_final_summary();
        let duration_ns = b4_greedy_a_duration_ns(first.duration_ns, second.duration_ns)?;
        let [first0, first1] = first.requests;
        let [second0, second1] = second.requests;
        let mut physical = first.physical;
        physical.checked_add_assign(second.physical)?;
        B2GenerateSummary {
            requests: [first0, first1, second0, second1],
            physical,
            duration_ns,
            peak_device_bytes: first.peak_device_bytes.max(second.peak_device_bytes),
        }
    } else {
        run_v4_b2_generate_arm(
            &v4_spec::PRO,
            context,
            module,
            stream,
            None,
            &model_dir,
            Some(&final_weights),
            None,
            &mut lanes,
            &mut scratch,
            prompts,
            greedy_steps,
            forward_tokens,
            false,
            false,
            None,
            &mut |_, _, _| Ok(()),
        )?
        .into_final_summary()
    };
    writeln!(
        output.file_mut(),
        "{}",
        serde_json::json!({
            "event": "arm", "arm": arm, "decode_generate_ns": summary.duration_ns,
            "final_requests": summary.requests, "physical": summary.physical.json(),
            "peak_device_used_bytes": summary.peak_device_bytes,
        })
    )?;
    output.commit()?;
    println!(
        "{{\"status\":\"PASS\",\"mode\":\"v4-b4-greedy-abba-{arm}\",\"result_jsonl\":{}}}",
        serde_json::to_string(output.path())?
    );
    Ok(())
}

pub(crate) fn b2_serial_first(repeat: usize) -> bool {
    matches!(repeat % 4, 0 | 3)
}

pub(crate) fn expert_union_json<const REQUESTS: usize>(
    results: &[B2RequestResult; REQUESTS],
    layers: &[usize],
) -> serde_json::Value {
    serde_json::Value::Array(
        layers
            .iter()
            .copied()
            .enumerate()
            .map(|(index, layer)| {
                let mut union = results
                    .iter()
                    .flat_map(|request| request.layers[index].route_ids.iter().copied())
                    .collect::<Vec<_>>();
                union.sort_unstable();
                union.dedup();
                let union_experts = union.len();
                serde_json::json!({
                    "layer": layer,
                    "expert_ids": union,
                    "union_experts": union_experts,
                    "logical_union_source_bytes": union_experts * v4_expert::V4_ROUTED_EXPERT_BYTES,
                })
            })
            .collect(),
    )
}

pub(crate) fn run_real_v4_b2_layer_stationary(
    context: &std::sync::Arc<CudaContext>,
    module: &kernels::LoadedModule,
    stream: &std::sync::Arc<CudaStream>,
) -> Result<(), Box<dyn Error>> {
    let repeats = std::env::var("COLIBRI_V4_B2_REPEATS")
        .unwrap_or_else(|_| "20".into())
        .parse::<usize>()?;
    if repeats < 20 {
        return Err("COLIBRI_V4_B2_REPEATS must be at least 20".into());
    }
    let result_path = std::env::var_os("COLIBRI_V4_B2_RESULT_JSONL")
        .map(std::path::PathBuf::from)
        .ok_or("set COLIBRI_V4_B2_RESULT_JSONL")?;
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
    let initial_hidden = [
        prepare_b2_prefix(
            context,
            module,
            stream,
            &model_dir,
            token_ids[0],
            &mut lanes,
        )?,
        prepare_b2_prefix(
            context,
            module,
            stream,
            &model_dir,
            token_ids[1],
            &mut lanes,
        )?,
    ];
    let mut scratch =
        DeviceBuffer::<u16>::zeroed(stream, (128 * 1024 * 1024) / std::mem::size_of::<u16>())?;
    let mut output = std::fs::File::create(&result_path)?;
    writeln!(
        output,
        "{}",
        serde_json::json!({
            "event": "run",
            "schema": "colibri-v4-b2-layer-stationary-v2",
            "provenance": benchmark_provenance()?,
            "token_e2e_ns": serde_json::Value::Null,
            "model_revision": v4_state::REVISION,
            "layers": [3, 4],
            "requests": [
                {"token_id": token_ids[0], "position": 0},
                {"token_id": token_ids[1], "position": 0},
            ],
            "repeats": repeats,
            "tseq_definition": "serial_total for both independent requests",
            "speedup_definition": "serial_total_ns / batch_total_ns",
            "arm_order_schedule": "ABBA, A=serial_then_batch, B=batch_then_serial",
            "raw_scratch_bytes": scratch.num_bytes(),
            "qd4_lanes": V4_QD4_LANES,
        })
    )?;
    let mut oracle: Option<[B2RequestResult; 2]> = None;
    let mut serial_samples = Vec::with_capacity(repeats);
    let mut batch_samples = Vec::with_capacity(repeats);
    let mut serial_first_samples = Vec::with_capacity(repeats);
    let mut peak_device_bytes = device_used_bytes()?;
    for repeat in 0..repeats {
        let serial_first = b2_serial_first(repeat);
        let (serial, batch) = if serial_first {
            let serial = run_b2_serial_pair(
                context,
                module,
                stream,
                &model_dir,
                token_ids,
                &initial_hidden,
                &mut lanes,
                &mut scratch,
                &mut peak_device_bytes,
            )?;
            let batch = run_b2_batch_pair(
                context,
                module,
                stream,
                &model_dir,
                token_ids,
                &initial_hidden,
                &mut lanes,
                &mut scratch,
                &mut peak_device_bytes,
            )?;
            (serial, batch)
        } else {
            let batch = run_b2_batch_pair(
                context,
                module,
                stream,
                &model_dir,
                token_ids,
                &initial_hidden,
                &mut lanes,
                &mut scratch,
                &mut peak_device_bytes,
            )?;
            let serial = run_b2_serial_pair(
                context,
                module,
                stream,
                &model_dir,
                token_ids,
                &initial_hidden,
                &mut lanes,
                &mut scratch,
                &mut peak_device_bytes,
            )?;
            (serial, batch)
        };
        if batch.results != serial.results {
            return Err(format!("B2 repeat {repeat} changed per-request canonical state").into());
        }
        if let Some(expected) = &oracle {
            if expected != &serial.results {
                return Err(format!("B2 repeat {repeat} changed the fixed request oracle").into());
            }
        } else {
            oracle = Some(serial.results.clone());
        }
        serial_samples.push(serial.total_ns);
        batch_samples.push(batch.total_ns);
        serial_first_samples.push(serial_first);
        writeln!(
            output,
            "{}",
            serde_json::json!({
                "event": "repeat",
                "repeat": repeat,
                "arm_order": if serial_first { "serial_then_batch" } else { "batch_then_serial" },
                "serial_total_ns": serial.total_ns,
                "serial_completion_ns": serial.completion_ns,
                "batch_total_ns": batch.total_ns,
                "batch_service_latency_ns": batch.total_ns,
                "batch_completion_ns": batch.completion_ns,
                "throughput_speedup": serial.total_ns as f64 / batch.total_ns as f64,
                "serial_physical": serial.physical.json(),
                "batch_physical": batch.physical.json(),
                "expert_union": expert_union_json(&batch.results, &[3, 4]),
                "requests": batch.results.iter().enumerate().map(|(request, result)| serde_json::json!({
                    "request": request,
                    "token_id": token_ids[request],
                    "position": 0,
                    "output_digest": format!("{:016x}", result.output_digest),
                    "layers": result.layers.iter().map(B2LayerDigest::json).collect::<Vec<_>>(),
                })).collect::<Vec<_>>(),
                "peak_host_rss_bytes": process_peak_rss_bytes()?,
                "peak_device_used_bytes": peak_device_bytes,
            })
        )?;
    }
    let speedups = serial_samples
        .iter()
        .zip(&batch_samples)
        .map(|(&serial, &batch)| serial as f64 / batch as f64)
        .collect::<Vec<_>>();
    let serial_median_ns = median_u64(&serial_samples);
    let batch_median_ns = median_u64(&batch_samples);
    let speedup = serial_median_ns as f64 / batch_median_ns as f64;
    let status = if speedup >= 1.2 { "PASS" } else { "KILL" };
    let stratum = |serial_first: bool| {
        let indexes = serial_first_samples
            .iter()
            .enumerate()
            .filter_map(|(index, &value)| (value == serial_first).then_some(index))
            .collect::<Vec<_>>();
        let serial = indexes
            .iter()
            .map(|&index| serial_samples[index])
            .collect::<Vec<_>>();
        let batch = indexes
            .iter()
            .map(|&index| batch_samples[index])
            .collect::<Vec<_>>();
        let serial_median = median_u64(&serial);
        let batch_median = median_u64(&batch);
        serde_json::json!({
            "repeats": indexes.len(),
            "serial_total_median_ns": serial_median,
            "batch_total_median_ns": batch_median,
            "throughput_speedup": serial_median as f64 / batch_median as f64,
        })
    };
    writeln!(
        output,
        "{}",
        serde_json::json!({
            "event": "summary",
            "status": status,
            "serial_total_median_ns": serial_median_ns,
            "batch_total_median_ns": batch_median_ns,
            "throughput_speedup": speedup,
            "repeat_speedup_median": median_f64(&speedups),
            "order_strata": {
                "serial_then_batch": stratum(true),
                "batch_then_serial": stratum(false),
            },
            "repeats": repeats,
            "canonical_exact": true,
            "peak_host_rss_bytes": process_peak_rss_bytes()?,
            "peak_device_used_bytes": peak_device_bytes,
        })
    )?;
    println!(
        "{{\"status\":\"{status}\",\"mode\":\"v4-b2-layer-stationary\",\"speedup\":{speedup:.6},\"result_jsonl\":\"{}\"}}",
        result_path.display()
    );
    Ok(())
}

pub(crate) fn run_real_v4_b4_layer_stationary(
    context: &std::sync::Arc<CudaContext>,
    module: &kernels::LoadedModule,
    stream: &std::sync::Arc<CudaStream>,
    full61: bool,
) -> Result<(), Box<dyn Error>> {
    let repeats_key = if full61 {
        "COLIBRI_V4_B4_FULL61_REPEATS"
    } else {
        "COLIBRI_V4_B4_REPEATS"
    };
    let repeats = std::env::var(repeats_key)
        .unwrap_or_else(|_| if full61 { "4" } else { "12" }.into())
        .parse::<usize>()?;
    let required_repeats = if full61 { 4 } else { 12 };
    if repeats != required_repeats {
        return Err(
            format!("{repeats_key} must be exactly {required_repeats} for this B4 gate").into(),
        );
    }
    let result_key = if full61 {
        "COLIBRI_V4_B4_FULL61_RESULT_JSONL"
    } else {
        "COLIBRI_V4_B4_RESULT_JSONL"
    };
    let result_path = std::env::var_os(result_key)
        .map(std::path::PathBuf::from)
        .ok_or_else(|| format!("set {result_key}"))?;
    if let Some(parent) = result_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let model_dir = v4_model_dir()?;
    let token_ids = [0_i64, 1, 2, 3];
    let layers = if full61 {
        (0..61).collect::<Vec<_>>()
    } else {
        vec![3, 4]
    };
    let mut lanes = (0..V4_QD4_LANES)
        .map(|_| {
            let lane_stream = context.new_stream()?;
            ExpertSlot::new_empty(context, module, &lane_stream)
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    let initial_hidden = if full61 {
        [
            prepare_b2_initial_hidden(&model_dir, token_ids[0])?,
            prepare_b2_initial_hidden(&model_dir, token_ids[1])?,
            prepare_b2_initial_hidden(&model_dir, token_ids[2])?,
            prepare_b2_initial_hidden(&model_dir, token_ids[3])?,
        ]
    } else {
        [
            prepare_b2_prefix(
                context,
                module,
                stream,
                &model_dir,
                token_ids[0],
                &mut lanes,
            )?,
            prepare_b2_prefix(
                context,
                module,
                stream,
                &model_dir,
                token_ids[1],
                &mut lanes,
            )?,
            prepare_b2_prefix(
                context,
                module,
                stream,
                &model_dir,
                token_ids[2],
                &mut lanes,
            )?,
            prepare_b2_prefix(
                context,
                module,
                stream,
                &model_dir,
                token_ids[3],
                &mut lanes,
            )?,
        ]
    };
    let mut scratch =
        DeviceBuffer::<u16>::zeroed(stream, (128 * 1024 * 1024) / std::mem::size_of::<u16>())?;
    let mut output = std::fs::File::options()
        .write(true)
        .create_new(true)
        .open(&result_path)
        .map_err(|error| {
            format!(
                "refusing to overwrite B4 result artifact {}: {error}",
                result_path.display()
            )
        })?;
    let arm_order_realized = (0..repeats)
        .map(|repeat| if b2_serial_first(repeat) { "A" } else { "B" })
        .collect::<Vec<_>>();
    writeln!(
        output,
        "{}",
        serde_json::json!({
            "event": "run",
            "schema": if full61 { "colibri-v4-b4-full61-layer-stationary-v2" } else { "colibri-v4-b4-layer-stationary-v1" },
            "provenance": benchmark_provenance()?,
            "model_revision": v4_state::REVISION,
            "layers": layers,
            "requests": token_ids.map(|token_id| serde_json::json!({"token_id": token_id, "position": 0})),
            "warmup_pairs": 1,
            "repeats": repeats,
            "tseq_definition": "serial_total for four independent requests",
            "speedup_definition": "serial4_total_ns / batch4_total_ns",
            "arm_order_schedule": "ABBA, A=serial4_then_batch4, B=batch4_then_serial4",
            "arm_order_realized": arm_order_realized,
            "gate": {
                "exact_pairs": required_repeats,
                "median_speedup_min": if full61 { 2.0 } else { 1.80 },
                "positive_pairs_min": if full61 { 4 } else { 10 },
                "batch_makespan_p95_not_above_serial": !full61,
            },
            "raw_scratch_bytes": scratch.num_bytes(),
            "qd4_lanes": V4_QD4_LANES,
        })
    )?;
    let mut peak_device_bytes = device_used_bytes()?;
    let warm_layers = if full61 { vec![3, 4] } else { layers.clone() };
    let warm_hidden = if full61 {
        [
            prepare_b2_prefix(
                context,
                module,
                stream,
                &model_dir,
                token_ids[0],
                &mut lanes,
            )?,
            prepare_b2_prefix(
                context,
                module,
                stream,
                &model_dir,
                token_ids[1],
                &mut lanes,
            )?,
            prepare_b2_prefix(
                context,
                module,
                stream,
                &model_dir,
                token_ids[2],
                &mut lanes,
            )?,
            prepare_b2_prefix(
                context,
                module,
                stream,
                &model_dir,
                token_ids[3],
                &mut lanes,
            )?,
        ]
    } else {
        initial_hidden.clone()
    };
    let warm_serial = run_b2_serial_requests(
        context,
        module,
        stream,
        &model_dir,
        token_ids,
        &warm_hidden,
        &warm_layers,
        &mut lanes,
        &mut scratch,
        &mut peak_device_bytes,
    )?;
    let warm_batch = run_b2_layer_stationary_requests(
        context,
        module,
        stream,
        &model_dir,
        token_ids,
        &warm_hidden,
        &warm_layers,
        &mut lanes,
        &mut scratch,
        &mut peak_device_bytes,
    )?;
    if warm_batch.results != warm_serial.results {
        return Err("B4 warmup changed per-request canonical state".into());
    }
    writeln!(
        output,
        "{}",
        serde_json::json!({
            "event": "warmup",
            "serial_total_ns": warm_serial.total_ns,
            "batch_total_ns": warm_batch.total_ns,
            "canonical_exact": true,
            "serial_physical": warm_serial.physical.json(),
            "batch_physical": warm_batch.physical.json(),
            "requests": warm_batch.results.iter().enumerate().map(|(request, result)| serde_json::json!({
                "request": request,
                "token_id": token_ids[request],
                "position": 0,
                "output_digest": format!("{:016x}", result.output_digest),
                "layers": result.layers.iter().map(B2LayerDigest::json).collect::<Vec<_>>(),
            })).collect::<Vec<_>>(),
            "peak_host_rss_bytes": process_peak_rss_bytes()?,
            "peak_device_used_bytes": peak_device_bytes,
        })
    )?;
    let mut oracle: Option<[B2RequestResult; 4]> = None;
    let mut serial_samples = Vec::with_capacity(repeats);
    let mut batch_samples = Vec::with_capacity(repeats);
    let mut serial_first_samples = Vec::with_capacity(repeats);
    let mut positive_pairs = 0_usize;
    for repeat in 0..repeats {
        let serial_first = b2_serial_first(repeat);
        let (serial, batch) = if serial_first {
            let serial = run_b2_serial_requests(
                context,
                module,
                stream,
                &model_dir,
                token_ids,
                &initial_hidden,
                &layers,
                &mut lanes,
                &mut scratch,
                &mut peak_device_bytes,
            )?;
            let batch = run_b2_layer_stationary_requests(
                context,
                module,
                stream,
                &model_dir,
                token_ids,
                &initial_hidden,
                &layers,
                &mut lanes,
                &mut scratch,
                &mut peak_device_bytes,
            )?;
            (serial, batch)
        } else {
            let batch = run_b2_layer_stationary_requests(
                context,
                module,
                stream,
                &model_dir,
                token_ids,
                &initial_hidden,
                &layers,
                &mut lanes,
                &mut scratch,
                &mut peak_device_bytes,
            )?;
            let serial = run_b2_serial_requests(
                context,
                module,
                stream,
                &model_dir,
                token_ids,
                &initial_hidden,
                &layers,
                &mut lanes,
                &mut scratch,
                &mut peak_device_bytes,
            )?;
            (serial, batch)
        };
        if batch.results != serial.results {
            return Err(format!(
                "B4{} repeat {repeat} changed per-request canonical state",
                if full61 { " full61" } else { "" }
            )
            .into());
        }
        if let Some(expected) = &oracle {
            if expected != &serial.results {
                return Err(format!(
                    "B4{} repeat {repeat} changed the fixed request oracle",
                    if full61 { " full61" } else { "" }
                )
                .into());
            }
        } else {
            oracle = Some(serial.results.clone());
        }
        let speedup = serial.total_ns as f64 / batch.total_ns as f64;
        if speedup > 1.0 {
            positive_pairs += 1;
        }
        serial_samples.push(serial.total_ns);
        batch_samples.push(batch.total_ns);
        serial_first_samples.push(serial_first);
        writeln!(
            output,
            "{}",
            serde_json::json!({
                "event": "repeat",
                "repeat": repeat,
                "arm_order": if serial_first { "serial4_then_batch4" } else { "batch4_then_serial4" },
                "serial_total_ns": serial.total_ns,
                "serial_completion_ns": serial.completion_ns,
                "serial_service_latency_ns": serial.service_latency_ns,
                "batch_total_ns": batch.total_ns,
                "batch_completion_ns": batch.completion_ns,
                "batch_service_latency_ns": batch.service_latency_ns,
                "throughput_speedup": speedup,
                "serial_physical": serial.physical.json(),
                "batch_physical": batch.physical.json(),
                "expert_union": expert_union_json(&batch.results, &layers),
                "requests": batch.results.iter().enumerate().map(|(request, result)| serde_json::json!({
                    "request": request,
                    "token_id": token_ids[request],
                    "position": 0,
                    "output_digest": format!("{:016x}", result.output_digest),
                    "layers": result.layers.iter().map(B2LayerDigest::json).collect::<Vec<_>>(),
                })).collect::<Vec<_>>(),
                "peak_host_rss_bytes": process_peak_rss_bytes()?,
                "peak_device_used_bytes": peak_device_bytes,
            })
        )?;
    }
    let p95 = |samples: &[u64]| {
        let mut sorted = samples.to_vec();
        sorted.sort_unstable();
        sorted[(sorted.len() * 95).div_ceil(100) - 1]
    };
    let serial_median_ns = median_u64(&serial_samples);
    let batch_median_ns = median_u64(&batch_samples);
    let speedup = serial_median_ns as f64 / batch_median_ns as f64;
    let serial_p95_ns = p95(&serial_samples);
    let batch_p95_ns = p95(&batch_samples);
    let status = if speedup >= if full61 { 2.0 } else { 1.80 }
        && positive_pairs >= if full61 { 4 } else { 10 }
        && (full61 || batch_p95_ns <= serial_p95_ns)
    {
        "GO"
    } else {
        "KILL"
    };
    let stratum = |serial_first: bool| {
        let indexes = serial_first_samples
            .iter()
            .enumerate()
            .filter_map(|(index, &value)| (value == serial_first).then_some(index))
            .collect::<Vec<_>>();
        let serial = indexes
            .iter()
            .map(|&index| serial_samples[index])
            .collect::<Vec<_>>();
        let batch = indexes
            .iter()
            .map(|&index| batch_samples[index])
            .collect::<Vec<_>>();
        let serial_median = median_u64(&serial);
        let batch_median = median_u64(&batch);
        serde_json::json!({
            "repeats": indexes.len(),
            "serial4_total_median_ns": serial_median,
            "batch4_total_median_ns": batch_median,
            "throughput_speedup": serial_median as f64 / batch_median as f64,
        })
    };
    writeln!(
        output,
        "{}",
        serde_json::json!({
            "event": "summary",
            "status": status,
            "serial4_total_median_ns": serial_median_ns,
            "serial4_total_p95_ns": serial_p95_ns,
            "batch4_total_median_ns": batch_median_ns,
            "batch4_total_p95_ns": batch_p95_ns,
            "throughput_speedup": speedup,
            "repeat_speedup_median": median_f64(&serial_samples.iter().zip(&batch_samples).map(|(&serial, &batch)| serial as f64 / batch as f64).collect::<Vec<_>>()),
            "positive_pairs": positive_pairs,
            "canonical_exact_pairs": repeats,
            "canonical_exact": true,
            "order_strata": {
                "serial4_then_batch4": stratum(true),
                "batch4_then_serial4": stratum(false),
            },
            "arm_order_schedule": "ABBA, A=serial4_then_batch4, B=batch4_then_serial4",
            "arm_order_realized": arm_order_realized,
            "repeats": repeats,
            "peak_host_rss_bytes": process_peak_rss_bytes()?,
            "peak_device_used_bytes": peak_device_bytes,
        })
    )?;
    println!(
        "{{\"status\":\"{status}\",\"mode\":\"{}\",\"speedup\":{speedup:.6},\"result_jsonl\":\"{}\"}}",
        if full61 {
            "v4-b4-full61-layer-stationary"
        } else {
            "v4-b4-layer-stationary"
        },
        result_path.display()
    );
    Ok(())
}

