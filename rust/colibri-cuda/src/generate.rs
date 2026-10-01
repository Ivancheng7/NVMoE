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
        let run = step_b2_layer(
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
            None,
            None, // attention device chain (not this path)
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
            match step_b2_layer(
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
                None,
                None, // attention device chain (not this path)
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
    slot: Option<&mut fixed_layer_slot::FixedLayerSlot>,
) -> Result<B2RawRuntime, Box<dyn Error>> {
    // The slot restage copies the whole fixed tensor set into the reused device
    // buffers, but it runs *before* the loader starts its own H2D timer, so that
    // timer only ever sees the small wo_a upload. Span both here: `fixed_h2d_ns`
    // is meant to be time-to-device for the layer's fixed weights, and without
    // the restage in it the restage time hides in the window residual while
    // `h2d_ns` reads several times faster than the PCIe link can carry the
    // bytes. `take_restaged` is a no-op (and this span then equals the loader's
    // own value) whenever the slot has nothing to hand back.
    let h2d_started = std::time::Instant::now();
    let restaged = match slot {
        Some(slot) => slot.take_restaged(stream, &host.host)?,
        None => None,
    };
    let mut loaded = load_b2_raw_runtime_from_host_ref_with_weights(
        spec,
        model_dir,
        layer,
        &host.host,
        &host.wo_a_raw,
        telemetry,
        stream,
        restaged,
    )?;
    loaded.fixed.fixed_h2d_ns =
        u64::try_from(h2d_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
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

/// Positions one layer pass executes for this request.
///
/// With the opt-in prompt chunk a request that is still inside its prompt hands
/// its whole remaining prompt to a single layer pass, so one layer load serves
/// every one of those positions. Without it — and for every generated token,
/// which cannot start before its predecessor's argmax exists — this is 1 and
/// the pass is exactly the pre-existing single-position one.
fn b2_prompt_chunk_span(chunk_enabled: bool, request: &B2GenerateRequest) -> usize {
    if chunk_enabled && request.position < request.prompt_ids.len() {
        request.prompt_ids.len() - request.position
    } else {
        1
    }
}

pub(crate) fn run_v4_b2_generate_arm<'module, const REQUESTS: usize>(
    spec: &'static v4_spec::V4Spec,
    context: &std::sync::Arc<CudaContext>,
    module: &'module kernels::LoadedModule,
    stream: &std::sync::Arc<CudaStream>,
    flash_cublas: Option<&FlashCublasSession>,
    model_dir: &std::path::Path,
    final_weights: Option<&v4_state::FinalHeadWeights>,
    flash_final_head_gpu: Option<&FlashDsparkFinalHeadGpuWorkspace>,
    lanes: &mut [ExpertSlot<'_>],
    mut scratch: &mut DeviceBuffer<u16>,
    resident: &mut Vec<Option<B2RawRuntime>>,
    routed_slot_scratch: &mut RoutedSlotScratch<'module>,
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
    // Whole-model device residency for the per-layer fixed tensors, on by default
    // on the two delivery Flash surfaces. They are ~156 MB per layer and never
    // change during a request, so a request that has read them once serves every
    // later token from VRAM instead of re-uploading 6.71 GB/token over PCIe: the
    // 2026-09-20 A/B measured 3.183 -> 2.305 s/token steady state for +6.35 GB of
    // device memory, bit-identical outputs. `COLIBRI_V4_FIXED_RESIDENT_GIB=0`
    // streams as before; the budget is preflighted against free device memory.
    let resident_budget_bytes = flash_b1_resident_budget_bytes(spec, REQUESTS)?;
    // The pool comes from the caller's scope. The serve face hands the same one
    // back for every request, so only its first request pays the fill and the
    // later ones start every layer from VRAM; a fresh caller passes an empty
    // pool and the fill happens exactly as before. The capture budget below
    // still counts only bytes captured by this call — the carried entries are
    // already allocated and their cost was paid by whichever call filled them.
    if resident.len() != spec.layers {
        *resident = (0..spec.layers).map(|_| None).collect();
    }
    let resident_enabled =
        resident_budget_bytes > 0 || resident.iter().any(Option::is_some);
    let mut resident_bytes = 0usize;
    let mut resident_hits = 0usize;
    // A resident layer has nothing to preload, and its captured buffer *is* the
    // fixed-layer slot, so residency displaces the slot. Prefetch stays on: the
    // fill pass still has to cold-read all 43 host payloads, and overlapping
    // them with compute is exactly what keeps that first token from paying for
    // the whole 6.7 GB on the critical path. The spawn below simply skips layers
    // that are already resident, so steady state starts no worker at all.
    let flash_b1_prefetch = flash_b1_prefetch_enabled(spec, REQUESTS)?;
    let flash_b1_prompt_chunk = flash_b1_prompt_chunk_enabled(spec, REQUESTS)?;
    let flash_b1_fixed_slot = !resident_enabled && flash_b1_fixed_slot_enabled(spec, REQUESTS)?;
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
    // costs 17 ms, so it lives as long as the run instead of one layer. The
    // serve face hands the same scratch back for every request, so the fan-out
    // lanes and their page-locked staging are built once per engine process
    // instead of once per request.
    // The fixed-layer slot gets the same lifetime as the host cache: one device
    // slot per layer type, restaged every layer instead of rebuilt.
    let mut fixed_layer_slot = fixed_layer_slot::FixedLayerSlot::new();
    // Same lifetime again for the strict compressor's pools: three
    // geometry-sized sets of host and device buffers that the compressor
    // restages into instead of allocating (2x29 MB of host expansion per call
    // for Csa4) on every layer. The width is the spec's: built with the Pro
    // constant, the pool never engaged on Flash, because the compressor handed
    // it a 4096-wide input for a 7168-wide pool.
    let mut compressor_scratch = flash_b1_compressor_arena_enabled(spec, REQUESTS)?
        .then(|| StrictCompressorScratch::new(stream, spec.hidden))
        .transpose()?;
    // The attention device chain's preallocated intermediates get the same
    // lifetime as the compressor arena: one set of buffers for the whole run,
    // shared serially by every layer and request. `=0` keeps the host path.
    let mut attention_device_scratch = attention_device_chain_enabled()?
        .then(|| AttentionDeviceScratch::new(stream, spec))
        .transpose()?;

    while !b2_generate_active_indices::<REQUESTS>(&active).is_empty() {
        let active_requests = b2_generate_active_indices::<REQUESTS>(&active);
        let rows = active_requests.len();
        // Positions this pass executes per request. Without the opt-in chunk
        // this is 1 everywhere and the loop is byte-for-byte the old one; with
        // it, a request still inside its prompt hands its whole remaining
        // prompt to a single layer pass, so one layer load serves every one of
        // those positions instead of one load per position.
        let spans = active_requests
            .iter()
            .map(|&request| b2_prompt_chunk_span(flash_b1_prompt_chunk, &requests[request]))
            .collect::<Vec<_>>();
        let chunk = *spans
            .first()
            .ok_or("B2 generation pass has no active request")?;
        if spans.iter().any(|&span| span != chunk) {
            return Err("B2 prompt chunk spans differ across active requests".into());
        }
        if parity_stop_after_layer.is_some() && chunk != 1 {
            return Err("Flash parity stop requires one position per layer pass".into());
        }
        let token_ids = (0..rows)
            .map(|row| {
                let state = &requests[active_requests[row]];
                (0..chunk)
                    .map(|offset| {
                        let absolute = state
                            .position
                            .checked_add(offset)
                            .ok_or("B2 token position overflow")?;
                        state
                            .input_ids
                            .get(absolute)
                            .copied()
                            .ok_or_else(|| {
                                "B2 request has no input token at its active position".into()
                            })
                    })
                    .collect::<Result<Vec<i64>, Box<dyn Error>>>()
            })
            .collect::<Result<Vec<Vec<i64>>, Box<dyn Error>>>()?;
        let mut hidden_rows: Vec<Vec<Vec<u16>>> = Vec::with_capacity(rows);
        for row in 0..rows {
            let request = active_requests[row];
            let mut per_offset = Vec::with_capacity(chunk);
            for offset in 0..chunk {
                let (hidden, initial_state_source_bytes) =
                    prepare_b2_initial_hidden_with_source_bytes_for(
                        spec,
                        model_dir,
                        token_ids[row][offset],
                    )?;
                requests[request]
                    .physical
                    .record_initial_state_source_bytes(initial_state_source_bytes)?;
                physical.record_initial_state_source_bytes(initial_state_source_bytes)?;
                per_offset.push(hidden);
            }
            hidden_rows.push(per_offset);
        }
        // Flat position-major input order keeps one index space for the hash
        // router metadata rows and the digests: row = offset * rows + row.
        let mut flat_ids = Vec::with_capacity(rows * chunk);
        for offset in 0..chunk {
            for row in 0..rows {
                flat_ids.push(token_ids[row][offset]);
            }
        }
        let mut digests: Vec<Vec<Vec<B2LayerDigest>>> = (0..rows)
            .map(|_| {
                (0..chunk)
                    .map(|_| Vec::with_capacity(spec.layers))
                    .collect::<Vec<Vec<B2LayerDigest>>>()
            })
            .collect();
        let token_started = std::time::Instant::now();
        std::thread::scope(|scope| -> Result<(), Box<dyn Error>> {
            let mut prefetched_next: Option<fixed_host_cache::ResidentPrefetch<'_>> = None;
            for layer in 0..layer_count {
                let layer_started = std::time::Instant::now();
                let mut from_prefetch = false;
                let mut from_resident = false;
                let mut loaded = if resident_enabled && resident[layer].is_some() {
                    from_resident = true;
                    resident_hits += 1;
                    let mut entry = resident[layer]
                        .take()
                        .ok_or("resident fixed layer disappeared")?;
                    // Nothing was read or copied on this visit; the fill pass
                    // already charged those bytes to the ledger.
                    entry.fixed_source_bytes = 0;
                    entry.fixed_h2d_bytes = 0;
                    entry.fixed = FixedLoadTelemetry {
                        cache_status: "resident_hit",
                        ..FixedLoadTelemetry::default()
                    };
                    entry
                } else if let Some(task) = prefetched_next.take() {
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
                        if flash_b1_fixed_slot {
                            Some(&mut fixed_layer_slot)
                        } else {
                            None
                        },
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
                        if flash_b1_fixed_slot {
                            Some(&mut fixed_layer_slot)
                        } else {
                            None
                        },
                    )?
                };
                if flash_b1_prefetch {
                    if let Some(next_layer) = flash_b1_prefetch_next_layer(layer, layer_count)? {
                        // Residency already holds the next layer, so there is
                        // nothing to read and no host slot to hand back; a
                        // worker here would never be joined and would trip the
                        // unconsumed-slot check at the end of the pass.
                        let next_is_resident = resident_enabled && resident[next_layer].is_some();
                        if !next_is_resident {
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
                }
                let fixed_prepare_ns =
                    u64::try_from(layer_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
                if from_prefetch {
                    loaded.fixed.window_residual_ns = fixed_prepare_ns
                        .saturating_sub(loaded.fixed.window_join_ns)
                        .saturating_sub(loaded.fixed.fixed_h2d_ns)
                        .saturating_sub(loaded.fixed.window_spawn_ns);
                }
                let metadata_rows = rows * chunk;
                let hash_ids = load_layer_hash_ids_for(spec, model_dir, layer, &flat_ids)?;
                let router_rows = (0..metadata_rows)
                    .map(|row| {
                        b2_router_metadata_row(
                            loaded.raw.runtime.weights.hash_router,
                            &hash_ids,
                            row,
                            metadata_rows,
                        )
                    })
                    .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
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
                    // Layer-outer, position-inner. Each layer commits its own
                    // rows in strictly increasing order, so position p sees
                    // exactly the rows the single-position loop would have
                    // committed before it.
                    for offset in 0..chunk {
                        for row in 0..rows {
                            let request = active_requests[row];
                            let state = &mut requests[request];
                            let absolute = state
                                .position
                                .checked_add(offset)
                                .ok_or("B2 token position overflow")?;
                            let (next, digest, stage) = step_b2_layer(
                                context,
                                module,
                                stream,
                                flash_cublas,
                                &loaded.raw.runtime,
                                lanes,
                                router_rows[offset * rows + row],
                                &hidden_rows[row][offset],
                                token_ids[row][offset],
                                absolute,
                                &mut state.persistent,
                                compressor_scratch.as_mut(),
                                Some(&mut *routed_slot_scratch),
                                attention_device_scratch.as_mut(),
                            )?;
                            physical.routed_source_bytes += stage.routed_source_bytes;
                            physical.routed_h2d_bytes += stage.routed_h2d_bytes;
                            hidden_rows[row][offset] = next;
                            digests[row][offset].push(digest);
                            if flash_b1_profile {
                                match flash_stage.as_mut() {
                                    Some(accumulated) => *accumulated += stage,
                                    None => flash_stage = Some(stage),
                                }
                            }
                        }
                    }
                    Ok(())
                })();
                std::mem::swap(&mut loaded.raw.runtime.weights.wo_a, &mut scratch);
                layer_run?;
                if let Some(stage) = flash_stage.as_ref() {
                    let layer_outer_ns =
                        u64::try_from(layer_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
                    let profile = flash_b1_layer_phase_profile_json(
                        layer,
                        layer_outer_ns,
                        fixed_prepare_ns,
                        &loaded.fixed,
                        wo_a_decode_sync_ns,
                        stage,
                        chunk,
                    )?;
                    let timing = &mut requests[0].flash_b1_phase;
                    timing.layer_outer_ns = timing
                        .layer_outer_ns
                        .checked_add(layer_outer_ns)
                        .ok_or("Flash B1 layer outer timing overflow")?;
                    timing.layers.push(profile);
                }
                if resident_enabled {
                    // Put the layer back after serving it, whether it came from
                    // residency or was just captured — otherwise the first hit
                    // would empty the slot while its bytes stayed counted.
                    if from_resident {
                        resident[layer] = Some(loaded);
                    } else {
                        let bytes = loaded.raw.resident_bytes;
                        if resident_bytes.saturating_add(bytes) <= resident_budget_bytes {
                            resident_bytes += bytes;
                            resident[layer] = Some(loaded);
                        }
                    }
                } else if flash_b1_fixed_slot {
                    fixed_layer_slot.release(loaded.raw.runtime.weights);
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
            let path = write_flash_parity_stop(stop, spec, &digests[0][0], phase_profile.as_ref())?;
            return Err(Box::new(FlashParityStopped { path }));
        }
        let elapsed_ns = u64::try_from(token_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        for row in 0..rows {
            let request = active_requests[row];
            for offset in 0..chunk {
                let state = &mut requests[request];
                let token_id = token_ids[row][offset];
                if digests[row][offset].len() != spec.layers {
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
                    B2LayerDigest::ordered_chain_sha256(&digests[row][offset])?;
                state.hidden = std::mem::take(&mut hidden_rows[row][offset]);
                state.tokens.push(serde_json::json!({
                    "token_position": state.position,
                    "input_id": token_id,
                    "input_kind": input_kind,
                    "e2e_ns": elapsed_ns,
                    "hidden_bf16_digest": format!("{:016x}", fnv1a_u16(&state.hidden)),
                    "ordered_layer_digest_sha256": ordered_layer_digest_sha256,
                    "layers": if capture_layer_trace { digests[row][offset].iter().map(B2LayerDigest::json).collect::<Vec<_>>() } else { Vec::new() },
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
                                (workspace.step_runtime(&state.hidden)?, None)
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
                                    step_flash_final_head_gpu_exact(
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
    }
    if requests.iter().any(|request| request.stop_reason.is_none()) {
        return Err("B2 greedy generation ended without a stop reason".into());
    }
    let duration_ns = u64::try_from(request_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
    if resident_enabled {
        let layers_resident = resident.iter().filter(|slot| slot.is_some()).count();
        let pool_bytes: usize = resident
            .iter()
            .flatten()
            .map(|entry| entry.raw.resident_bytes)
            .sum();
        eprintln!(
            "fixed residency: {layers_resident} layers resident ({pool_bytes} bytes), \
             {resident_hits} layer-visits served from VRAM"
        );
    }
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
    let mut resident = Vec::new();
    let mut routed_slot_scratch = RoutedSlotScratch::new();
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
        &mut resident,
        &mut routed_slot_scratch,
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
    let mut resident = Vec::new();
    let mut routed_slot_scratch = RoutedSlotScratch::new();
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
        &mut resident,
        &mut routed_slot_scratch,
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

/// Opt-in per-request phase account for the Flash serve surface.
///
/// The serve path already collects the same per-layer `phase_profile` as the CLI
/// (`flash_b1_profile` holds for FLASH_0731 with a single request), but the
/// gateway deliberately clears `COLIBRI_V4_RESULT_JSONL` before it spawns the
/// engine, so there was never a file for that profile to land in. Point
/// `COLIBRI_V4_FLASH_SERVE_PHASE_JSONL` at a path and the engine appends one JSON
/// record per completed request. Unset (the default) leaves the serve surface
/// exactly as it was: no extra file, no protocol change. The file is opened for
/// append and never truncated, so a batch script picks a fresh tag per run the
/// same way the CLI result files do.
struct ServePhaseSink {
    file: Option<std::fs::File>,
}

impl ServePhaseSink {
    fn from_env() -> Result<Self, Box<dyn Error>> {
        let Some(path) = std::env::var_os("COLIBRI_V4_FLASH_SERVE_PHASE_JSONL") else {
            return Ok(Self { file: None });
        };
        let path = std::path::PathBuf::from(path);
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)?;
        Ok(Self { file: Some(file) })
    }

    fn record(&mut self, value: &serde_json::Value) -> Result<(), Box<dyn Error>> {
        let Some(file) = self.file.as_mut() else {
            return Ok(());
        };
        serde_json::to_writer(&mut *file, value)?;
        std::io::Write::write_all(file, b"\n")?;
        std::io::Write::flush(file)?;
        Ok(())
    }

    fn is_enabled(&self) -> bool {
        self.file.is_some()
    }
}

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
    // Both pools outlive the request loop: the fixed-weight residency would
    // otherwise be rebuilt (6.25 GiB re-read and re-uploaded) and the fan-out
    // lanes re-page-locked by every request, all for buffers whose contents
    // never depend on the request. The KV and per-request state stay inside
    // the arm as before.
    let mut resident = Vec::new();
    let mut routed_slot_scratch = RoutedSlotScratch::new();
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
    let mut phase_sink = ServePhaseSink::from_env()?;
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
            &mut resident,
            &mut routed_slot_scratch,
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
        // One record per completed request, only when the operator pointed the
        // sink at a file. Same evidence shape as the CLI result record so the
        // existing phase readers work on it unchanged.
        if phase_sink.is_enabled() {
            let phase_profile = flash_b1_request_phase_profile_json(
                &arm.requests[0].flash_b1_phase,
                arm.duration_ns,
            )?;
            phase_sink.record(&serde_json::json!({
                "event": "serve_request",
                "schema": "colibri-deepseek-v4-flash-0731-base-v1",
                "implementation": "rust-cuda",
                "profile": spec.name,
                "model_revision": spec.revision,
                "model_dir": model_dir,
                "request_id": &request_id,
                "prompt_tokens": prompt_ids.len(),
                "completion_tokens": completion_tokens,
                "ttft_ms": ttft_ms,
                "itl_ms": itl_ms,
                "request_e2e_ns": arm.duration_ns,
                "peak_device_used_bytes": arm.peak_device_bytes,
                "phase_profile": phase_profile,
                "physical": arm.physical.json(),
                "requests": arm.requests.iter().enumerate().map(|(request, state)| b2_generate_request_json(request, state)).collect::<Vec<_>>(),
            }))?;
        }
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
    let mut resident = Vec::new();
    let mut routed_slot_scratch = RoutedSlotScratch::new();
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
            &mut resident,
            &mut routed_slot_scratch,
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
            &mut resident,
            &mut routed_slot_scratch,
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
            &mut resident,
            &mut routed_slot_scratch,
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

