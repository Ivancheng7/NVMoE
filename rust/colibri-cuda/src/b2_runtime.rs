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
    // Keep the experimental worker confined to the standalone Flash B1 base
    // runner. Serve, DSpark, Pro, QD4, and every multi-row execution retain
    // their existing loading path.
    if spec != &v4_spec::FLASH_0731 || requests != 1 || !optional_one_flag("COLIBRI_V4_FLASH_BASE")?
    {
        return Ok(false);
    }
    optional_one_flag("COLIBRI_V4_FLASH_PREFETCH")
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
    let runtime = upload_layer_runtime_for(spec, model_dir, layer, host, stream)?;
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
    routed_slot_scratch: Option<&mut RoutedSlotScratch<'module>>,
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
            None,
            routed_slot_scratch,
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
