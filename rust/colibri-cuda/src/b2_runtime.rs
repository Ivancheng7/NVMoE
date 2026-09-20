use super::*;

pub(crate) struct B2RawRuntime {
    pub(crate) raw: RawLoadedLayerRuntime,
    pub(crate) fixed_source_bytes: usize,
    pub(crate) fixed_h2d_bytes: usize,
    pub(crate) fixed: FixedLoadTelemetry,
}

pub(crate) fn flash_b1_prefetch_enabled(
    spec: &'static v4_spec::V4Spec,
    requests: usize,
) -> Result<bool, Box<dyn Error>> {
    // Confined to the two delivery Flash surfaces, same as the prompt chunk:
    // DSpark, Pro, QD4 and every multi-row execution keep their existing
    // loading path.
    if spec != &v4_spec::FLASH_0731 || requests != 1 {
        return Ok(false);
    }
    let base =
        std::env::var_os("COLIBRI_V4_FLASH_BASE").as_deref() == Some(std::ffi::OsStr::new("1"));
    let serve =
        std::env::var_os("COLIBRI_V4_FLASH_SERVE").as_deref() == Some(std::ffi::OsStr::new("1"));
    if !base && !serve {
        return Ok(false);
    }
    // On by default since the serve gate was widened: the worker is what keeps
    // the per-layer fixed load and the parallel routed read off the critical
    // path, and serve used to lose it because the gate only knew the base
    // runner (`c/openai_server.py` never sets COLIBRI_V4_FLASH_PREFETCH).
    // `=0` puts the serial loader back so runs recorded against it stay
    // reproducible.
    match std::env::var_os("COLIBRI_V4_FLASH_PREFETCH").as_deref() {
        None => Ok(true),
        Some(value) if value == "1" => Ok(true),
        Some(value) if value == "0" => Ok(false),
        Some(_) => Err("COLIBRI_V4_FLASH_PREFETCH must be 1 or 0 when set".into()),
    }
}

pub(crate) fn flash_b1_prompt_chunk_enabled(
    spec: &'static v4_spec::V4Spec,
    requests: usize,
) -> Result<bool, Box<dyn Error>> {
    // Confined to the two delivery Flash surfaces so the restructured loop is
    // exercised where it is measured. Pro, QD4, DSpark and every multi-row row
    // set keep one position per layer pass.
    if spec != &v4_spec::FLASH_0731 || requests != 1 {
        return Ok(false);
    }
    let base =
        std::env::var_os("COLIBRI_V4_FLASH_BASE").as_deref() == Some(std::ffi::OsStr::new("1"));
    let serve =
        std::env::var_os("COLIBRI_V4_FLASH_SERVE").as_deref() == Some(std::ffi::OsStr::new("1"));
    if !base && !serve {
        return Ok(false);
    }
    // On by default since the merged-tree verification: same digests as the
    // per-position sweep, fewer fixed-weight sweeps for the prompt. `=0` puts
    // the pre-chunk loop back so runs recorded against it stay reproducible.
    match std::env::var_os("COLIBRI_V4_FLASH_PROMPT_CHUNK").as_deref() {
        None => Ok(true),
        Some(value) if value == "1" => Ok(true),
        Some(value) if value == "0" => Ok(false),
        Some(_) => {
            Err("COLIBRI_V4_FLASH_PROMPT_CHUNK must be 1 or 0 when set".into())
        }
    }
}

pub(crate) fn flash_b1_fixed_slot_enabled(
    spec: &'static v4_spec::V4Spec,
    requests: usize,
) -> Result<bool, Box<dyn Error>> {
    // Confined to the two delivery Flash surfaces, same as the prompt chunk and
    // the prefetch worker: the slot is only measured there.
    if spec != &v4_spec::FLASH_0731 || requests != 1 {
        return Ok(false);
    }
    let base =
        std::env::var_os("COLIBRI_V4_FLASH_BASE").as_deref() == Some(std::ffi::OsStr::new("1"));
    let serve =
        std::env::var_os("COLIBRI_V4_FLASH_SERVE").as_deref() == Some(std::ffi::OsStr::new("1"));
    if !base && !serve {
        return Ok(false);
    }
    // On by default: one device slot per layer type means the per-layer
    // `LayerDeviceWeights` is restaged instead of rebuilt, which took `fixed`
    // from 1.02 to 0.71 s/token for +216 MB of device memory with bit-identical
    // outputs. `=0` rebuilds every layer so runs recorded against the old shape
    // stay reproducible.
    match std::env::var_os("COLIBRI_V4_FLASH_FIXED_SLOT").as_deref() {
        None => Ok(true),
        Some(value) if value == "1" => Ok(true),
        Some(value) if value == "0" => Ok(false),
        Some(_) => Err("COLIBRI_V4_FLASH_FIXED_SLOT must be 1 or 0 when set".into()),
    }
}

pub(crate) fn flash_b1_compressor_arena_enabled(
    spec: &'static v4_spec::V4Spec,
    requests: usize,
) -> Result<bool, Box<dyn Error>> {
    // Confined to the two delivery Flash surfaces, same as the prompt chunk, the
    // prefetch worker and the fixed-layer slot.
    if spec != &v4_spec::FLASH_0731 || requests != 1 {
        return Ok(false);
    }
    let base =
        std::env::var_os("COLIBRI_V4_FLASH_BASE").as_deref() == Some(std::ffi::OsStr::new("1"));
    let serve =
        std::env::var_os("COLIBRI_V4_FLASH_SERVE").as_deref() == Some(std::ffi::OsStr::new("1"));
    if !base && !serve {
        return Ok(false);
    }
    // On by default: the compressor used to re-allocate every host expansion
    // (2x29 MB for Csa4) and every device buffer on each call, which the probe
    // instrumented at +15 ms per call in host expansion alone. The arena keeps
    // them and the static check proves reuse does not reallocate.
    // `=0` restores the per-call allocation so runs recorded against it stay
    // reproducible.
    match std::env::var_os("COLIBRI_V4_FLASH_COMPRESSOR_ARENA").as_deref() {
        None => Ok(true),
        Some(value) if value == "1" => Ok(true),
        Some(value) if value == "0" => Ok(false),
        Some(_) => Err("COLIBRI_V4_FLASH_COMPRESSOR_ARENA must be 1 or 0 when set".into()),
    }
}

/// Budget in bytes for holding the whole per-layer fixed tensor set resident in
/// device memory on the two delivery Flash surfaces.
///
/// On by default there: the fixed tensors never change during a request, so a
/// request that has read a layer once can serve every later token from VRAM
/// instead of re-uploading 6.71 GB/token over PCIe. Measured 2026-09-20 at
/// `3.183 -> 2.305 s/token` steady state for `+6.35 GB` of device memory, with
/// bit-identical outputs. `=0` puts the streaming path back so runs recorded
/// against it stay reproducible; any other whole number of GiB becomes the
/// budget. The result is clamped to what the card actually has free (minus
/// headroom) by [`flash_b1_resident_budget_bytes`], so a small card keeps a
/// prefix resident instead of failing to allocate.
pub(crate) fn flash_b1_resident_enabled(
    spec: &'static v4_spec::V4Spec,
    requests: usize,
) -> Result<bool, Box<dyn Error>> {
    if spec != &v4_spec::FLASH_0731 || requests != 1 {
        return Ok(false);
    }
    let base =
        std::env::var_os("COLIBRI_V4_FLASH_BASE").as_deref() == Some(std::ffi::OsStr::new("1"));
    let serve =
        std::env::var_os("COLIBRI_V4_FLASH_SERVE").as_deref() == Some(std::ffi::OsStr::new("1"));
    if !base && !serve {
        return Ok(false);
    }
    // The whole fixed model is 6.71 GB; 8 GiB holds every layer with room to
    // spare, and the preflight below cuts it down on a card that is smaller.
    match std::env::var_os("COLIBRI_V4_FIXED_RESIDENT_GIB").as_deref() {
        None => Ok(true),
        Some(value) if value == "0" => Ok(false),
        Some(_) => Ok(true),
    }
}

/// The residency budget in bytes for this run, or zero when residency is off.
///
/// The requested budget is clamped to the device's free memory minus the same
/// headroom the fixed-layer cache preflight uses, so the resident set can never
/// turn into an allocation failure on a smaller card. When the clamp bites, the
/// capture loop stops once the budget is spent and the remaining layers keep
/// streaming; the `CACHE_PREFLIGHT_FALLBACK` line says what it was cut to.
pub(crate) fn flash_b1_resident_budget_bytes(
    spec: &'static v4_spec::V4Spec,
    requests: usize,
) -> Result<usize, Box<dyn Error>> {
    if !flash_b1_resident_enabled(spec, requests)? {
        return Ok(0);
    }
    let requested = match std::env::var_os("COLIBRI_V4_FIXED_RESIDENT_GIB") {
        None => 10 * 1024 * 1024 * 1024,
        Some(value) => {
            let value = value.to_string_lossy();
            let gib = value
                .trim()
                .parse::<f64>()
                .map_err(|_| "COLIBRI_V4_FIXED_RESIDENT_GIB must be a GiB value")?;
            if !gib.is_finite() || gib < 0.0 {
                return Err("COLIBRI_V4_FIXED_RESIDENT_GIB must be finite and non-negative".into());
            }
            (gib * 1024.0 * 1024.0 * 1024.0) as usize
        }
    };
    Ok(preflight_cache_budget(
        requested,
        device_available_bytes()?,
        V4_DEVICE_CAPACITY_HEADROOM_BYTES,
        "fixed_resident",
    ))
}

/// Pure diagnostic, off by default: print the inside of the `shared` window.
///
/// The phase wall around `shared` covers both the shared MLP and the routed
/// read it overlaps, so it cannot say which of the two bounds it. A run with
/// this set times the MLP's own return and each read worker's own window, and
/// prints one `SHAREDPROBE` line per layer. Numbers do not move; only stderr
/// grows.
pub(crate) fn flash_shared_probe_enabled() -> bool {
    std::env::var_os("COLIBRI_V4_FLASH_SHARED_PROBE").as_deref()
        == Some(std::ffi::OsStr::new("1"))
}

/// Pure diagnostic, off by default: print the inside of one Compressor call.
///
/// The phase table prices the compressor as one number per call and the bench
/// block as one sync wall per step, neither of which says which segment carries
/// the 10 ms / 19 ms tier. This splits the reusable-arena path into its nine
/// segments. Numbers do not move; only stderr grows.
pub(crate) fn flash_compressor_probe_enabled() -> bool {
    std::env::var_os("COLIBRI_V4_FLASH_COMPRESSOR_PROBE").as_deref()
        == Some(std::ffi::OsStr::new("1"))
}

pub(crate) fn flash_b1_prefetch_next_layer(
    current_layer: usize,
    layer_count: usize,
) -> Result<Option<usize>, Box<dyn Error>> {
    if layer_count == 0 || current_layer >= layer_count {
        return Err("Flash B1 fixed prefetch layer is outside the model".into());
    }
    Ok((current_layer + 1 < layer_count).then_some(current_layer + 1))
}

pub(crate) fn flash_b1_prefetch_worker_result<T>(
    expected_layer: usize,
    scheduled_layer: usize,
    result: Result<T, String>,
) -> Result<T, Box<dyn Error>> {
    if scheduled_layer != expected_layer {
        return Err("Flash B1 fixed prefetch lost layer order".into());
    }
    result.map_err(|error| format!("Flash B1 fixed prefetch worker failed: {error}").into())
}

pub(crate) fn b2_raw_runtime_ledger(
    fixed_source_bytes: usize,
    fixed_h2d_without_wo_a_bytes: usize,
    raw_wo_a_h2d_bytes: usize,
) -> Result<(usize, usize), Box<dyn Error>> {
    Ok((
        fixed_source_bytes,
        fixed_h2d_without_wo_a_bytes
            .checked_add(raw_wo_a_h2d_bytes)
            .ok_or("B2 raw fixed H2D byte count overflow")?,
    ))
}

pub(crate) fn load_b2_raw_runtime(
    model_dir: &std::path::Path,
    layer: usize,
    stream: &CudaStream,
) -> Result<B2RawRuntime, Box<dyn Error>> {
    load_b2_raw_runtime_for(&v4_spec::PRO, model_dir, layer, stream)
}

pub(crate) fn load_b2_raw_runtime_for(
    spec: &'static v4_spec::V4Spec,
    model_dir: &std::path::Path,
    layer: usize,
    stream: &CudaStream,
) -> Result<B2RawRuntime, Box<dyn Error>> {
    // Request-specific router metadata belongs to the request ledger, not the
    // immutable layer runtime. `load_layer_state_weights` accepts this empty
    // row set only to build the hash-router mode without reading a request row.
    let (mut host, telemetry) = load_layer_host_inner_for(spec, model_dir, layer, &[], true)?;
    let raw_wo_a = host
        .tail
        .wo_a_raw
        .take()
        .ok_or("B2 raw layer host omitted wo_a")?;
    load_b2_raw_runtime_from_host_ref_for(
        spec,
        model_dir,
        layer,
        &host,
        &raw_wo_a,
        telemetry,
        stream,
    )
}

/// Uploads a layer from a host payload the caller keeps.
///
/// The Flash B1 decode loop borrows the payload out of its retention cache here.
/// Because the caller owns it, releasing it is the caller's step, and a retained
/// payload outlives the layer instead of being rebuilt next token. Raw `wo_a`
/// travels beside the host, never inside it: `new_layer` rejects a host that
/// still carries it.
pub(crate) fn load_b2_raw_runtime_from_host_ref_for(
    spec: &'static v4_spec::V4Spec,
    model_dir: &std::path::Path,
    layer: usize,
    host: &DecodedLayerHost,
    raw_wo_a_host: &[u8],
    telemetry: FixedLoadTelemetry,
    stream: &CudaStream,
) -> Result<B2RawRuntime, Box<dyn Error>> {
    load_b2_raw_runtime_from_host_ref_with_weights(
        spec,
        model_dir,
        layer,
        host,
        raw_wo_a_host,
        telemetry,
        stream,
        None,
    )
}

/// Same as [`load_b2_raw_runtime_from_host_ref_for`], but the caller may hand in
/// an existing `LayerDeviceWeights` to restage into instead of building one.
///
/// This is the fixed-layer slot: the weights already carry the right buffers for
/// this layer type, so the ~20 `cuMemAlloc`/`cuMemFree` pairs per layer go away.
/// The byte ledger stays what it was — a restage copies the whole tensor set, and
/// `num_bytes()` measures exactly that. The restage itself runs in the caller
/// (`FixedLayerSlot::take_restaged`), which starts its own timer before handing
/// the slot in and overwrites `fixed_h2d_ns` with the whole span, so a restaged
/// layer's time-to-device covers the restage and not just the small wo_a upload.
/// The honest witness for this change is still the `fixed` wall in the phase
/// profile plus the output digests, not the byte account.
pub(crate) fn load_b2_raw_runtime_from_host_ref_with_weights(
    spec: &'static v4_spec::V4Spec,
    model_dir: &std::path::Path,
    layer: usize,
    host: &DecodedLayerHost,
    raw_wo_a_host: &[u8],
    telemetry: FixedLoadTelemetry,
    stream: &CudaStream,
    weights: Option<LayerDeviceWeights>,
) -> Result<B2RawRuntime, Box<dyn Error>> {
    if host.tail.wo_a_raw.is_some() {
        return Err("B2 raw layer host still owns its raw wo_a".into());
    }
    if raw_wo_a_host.is_empty() {
        return Err("B2 raw layer host omitted wo_a".into());
    }
    if !host.tail.wo_a_bf16.is_empty() {
        return Err("B2 raw layer host retained decoded wo_a".into());
    }
    let h2d_started = std::time::Instant::now();
    let runtime = match weights {
        Some(weights) => LoadedLayerRuntime {
            weights,
            // The slot restages the weights but not the expert shard handle,
            // which each layer pass opens for the routed read.
            expert_shard: safetensors::ShardFile::open(model_dir.join(spec.shard_name(layer + 2)?))?,
        },
        None => upload_layer_runtime_for(spec, model_dir, layer, host, stream)?,
    };
    let fixed_h2d_without_wo_a_bytes = runtime.weights.num_bytes();
    let wo_a_raw = DeviceBuffer::from_host(stream, raw_wo_a_host)?;
    stream.synchronize()?;
    let raw_h2d_bytes = wo_a_raw.num_bytes();
    let fixed_h2d_ns = u64::try_from(h2d_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
    let (fixed_source_bytes, fixed_h2d_bytes) = b2_raw_runtime_ledger(
        telemetry.fixed_source_bytes,
        fixed_h2d_without_wo_a_bytes,
        raw_h2d_bytes,
    )?;
    let resident_bytes = runtime
        .weights
        .num_bytes()
        .checked_add(wo_a_raw.num_bytes())
        .ok_or("B2 raw device resident byte overflow")?;
    Ok(B2RawRuntime {
        fixed_source_bytes,
        fixed_h2d_bytes,
        fixed: FixedLoadTelemetry {
            fixed_read_alloc_ns: telemetry.fixed_read_alloc_ns,
            fixed_decode_ns: telemetry.fixed_decode_ns,
            fixed_h2d_ns,
            fixed_h2d_bytes,
            fixed_source_bytes,
            cache_status: telemetry.cache_status,
            ..FixedLoadTelemetry::default()
        },
        raw: RawLoadedLayerRuntime {
            runtime,
            wo_a_raw,
            resident_bytes,
        },
    })
}

pub(crate) fn run_b2_layer<'module>(
    context: &std::sync::Arc<CudaContext>,
    module: &'module kernels::LoadedModule,
    stream: &std::sync::Arc<CudaStream>,
    flash_cublas: Option<&FlashCublasSession>,
    runtime: &LoadedLayerRuntime,
    lanes: &mut [ExpertSlot<'_>],
    hash_ids: &[i64],
    hidden: &[u16],
    token_id: i64,
    token_position: usize,
    persistent: &mut v4_state::V4SerialPersistentState,
    strict_scratch: Option<&mut StrictCompressorScratch>,
    routed_slot_scratch: Option<&mut RoutedSlotScratch<'module>>,
    attention_scratch: Option<&mut AttentionDeviceScratch>,
) -> Result<(Vec<u16>, B2LayerDigest, LayerStageTelemetry), Box<dyn Error>> {
    let layer = runtime.weights.layer;
    let spec = runtime.weights.spec;
    let mut result = None;
    persistent.transaction(layer, |state| {
        let run = run_real_layer_rows(
            context,
            module,
            stream,
            flash_cublas,
            runtime,
            (spec == &v4_spec::PRO).then_some(lanes),
            None,
            None,
            hash_ids,
            hidden,
            &[token_id],
            token_position,
            Some(state),
            None,
            None,
            false,
            None,
            false,
            None,
            strict_scratch,
            routed_slot_scratch,
            attention_scratch,
        )?;
        let main = state.compressor();
        let indexer = state.indexer();
        let digest = B2LayerDigest {
            layer,
            route_ids: run.routes.ids.clone(),
            route_weight_bits: run
                .routes
                .weights
                .iter()
                .map(|value| value.to_bits())
                .collect(),
            attention_topk_digest: run.attention_topk_digest,
            hidden_digest: fnv1a_u16(&run.output),
            kv_digest: fnv1a_u16(state.active_kv_bf16()),
            main_kv_state_digest: fnv1a_f32(main.kv()),
            main_score_state_digest: fnv1a_f32(main.score()),
            indexer_kv_state_digest: indexer.map(|value| fnv1a_f32(value.kv())),
            indexer_score_state_digest: indexer.map(|value| fnv1a_f32(value.score())),
        };
        result = Some((run.output, digest, run.stage));
        Ok(())
    })?;
    result.ok_or_else(|| "B2 layer transaction produced no result".into())
}

pub(crate) fn prepare_b2_prefix(
    context: &std::sync::Arc<CudaContext>,
    module: &kernels::LoadedModule,
    stream: &std::sync::Arc<CudaStream>,
    model_dir: &std::path::Path,
    token_id: i64,
    lanes: &mut [ExpertSlot<'_>],
) -> Result<Vec<u16>, Box<dyn Error>> {
    let mut hidden = prepare_b2_initial_hidden(model_dir, token_id)?;
    let mut persistent = v4_state::V4SerialPersistentState::new(1)?;
    for layer in 0..3 {
        let (runtime, _) = load_layer_runtime(model_dir, layer, &[token_id], stream)?;
        let hash_ids = load_layer_hash_ids(model_dir, layer, &[token_id])?;
        let mut next = None;
        persistent.transaction(layer, |state| {
            next = Some(
                run_real_layer_rows(
                    context,
                    module,
                    stream,
                    None,
                    &runtime,
                    Some(lanes),
                    None,
                    None,
                    &hash_ids,
                    &hidden,
                    &[token_id],
                    0,
                    Some(state),
                    None,
                    None,
                    false,
                    None,
                    false,
                    None,
                    None,
                None,
                None, // attention device chain (not this path)
                )?
                .output,
            );
            Ok(())
        })?;
        hidden = next.ok_or("B2 prefix transaction produced no output")?;
    }
    Ok(hidden)
}

pub(crate) fn prepare_b2_initial_hidden(
    model_dir: &std::path::Path,
    token_id: i64,
) -> Result<Vec<u16>, Box<dyn Error>> {
    Ok(prepare_b2_initial_hidden_with_source_bytes(model_dir, token_id)?.0)
}

pub(crate) fn prepare_b2_initial_hidden_with_source_bytes(
    model_dir: &std::path::Path,
    token_id: i64,
) -> Result<(Vec<u16>, usize), Box<dyn Error>> {
    prepare_b2_initial_hidden_with_source_bytes_for(&v4_spec::PRO, model_dir, token_id)
}

pub(crate) fn prepare_b2_initial_hidden_with_source_bytes_for(
    spec: &'static v4_spec::V4Spec,
    model_dir: &std::path::Path,
    token_id: i64,
) -> Result<(Vec<u16>, usize), Box<dyn Error>> {
    let state = v4_state::load_for(spec, model_dir, &[token_id])?;
    let initial_state_source_bytes = state.range_bytes;
    let embedding = state.embedding;
    let mut hidden = Vec::with_capacity(v4_state::HC * spec.hidden);
    for _ in 0..v4_state::HC {
        hidden.extend_from_slice(&embedding);
    }
    Ok((hidden, initial_state_source_bytes))
}

pub(crate) fn prepare_two_token_prefix(
    context: &std::sync::Arc<CudaContext>,
    module: &kernels::LoadedModule,
    stream: &std::sync::Arc<CudaStream>,
    model_dir: &std::path::Path,
    token_ids: [i64; 2],
    lanes: &mut [ExpertSlot<'_>],
) -> Result<[Vec<u16>; 2], Box<dyn Error>> {
    let mut hidden = token_ids
        .map(|token_id| v4_state::load(model_dir, &[token_id]).map(|state| state.embedding))
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .map(|embedding| {
            let mut row = Vec::with_capacity(v4_state::HC * v4_state::HIDDEN);
            for _ in 0..v4_state::HC {
                row.extend_from_slice(&embedding);
            }
            row
        })
        .collect::<Vec<_>>();
    let mut persistent = v4_state::V4SerialPersistentState::new(2)?;
    for layer in 0..3 {
        let (runtime, _) = load_layer_runtime(model_dir, layer, &token_ids, stream)?;
        let hash_ids = load_layer_hash_ids(model_dir, layer, &token_ids)?;
        for position in 0..2 {
            let row_hash_ids = if runtime.weights.hash_router {
                &hash_ids[position * v4_router::V4_TOPK..(position + 1) * v4_router::V4_TOPK]
            } else {
                &[]
            };
            let mut next = None;
            persistent.transaction(layer, |state| {
                next = Some(
                    run_real_layer_rows(
                        context,
                        module,
                        stream,
                        None,
                        &runtime,
                        Some(lanes),
                        None,
                        None,
                        row_hash_ids,
                        &hidden[position],
                        &[token_ids[position]],
                        position,
                        Some(state),
                        None,
                        None,
                        false,
                        None,
                        false,
                        None,
                        None,
                    None,
                    None, // attention device chain (not this path)
                    )?
                    .output,
                );
                Ok(())
            })?;
            hidden[position] = next.ok_or("two-token prefix produced no output")?;
        }
    }
    hidden
        .try_into()
        .map_err(|_| "two-token prefix changed row count".into())
}
