use super::*;

pub(crate) struct FixedPrefetchedHost {
    pub(crate) layer: usize,
    pub(crate) host: DecodedLayerHost,
    pub(crate) telemetry: FixedLoadTelemetry,
    pub(crate) worker_start_ns: u64,
    pub(crate) worker_end_ns: u64,
    pub(crate) host_bytes: usize,
}

pub(crate) struct FixedPrefetchTask<'scope> {
    pub(crate) layer: usize,
    pub(crate) transition: Option<FixedPrefetchTransition>,
    pub(crate) worker: std::thread::ScopedJoinHandle<'scope, Result<FixedPrefetchedHost, String>>,
}

pub(crate) struct FixedPrefetchTransition {
    pub(crate) from_layer: usize,
    pub(crate) compute_start_ns: u64,
    pub(crate) compute_end_ns: u64,
}

pub(crate) fn fixed_prefetch_transition_json(
    current_layer: usize,
    next_layer: usize,
    telemetry: &FixedLoadTelemetry,
    worker_start_ns: u64,
    worker_end_ns: u64,
    host_bytes: usize,
    compute_start_ns: u64,
    compute_end_ns: u64,
    consume_start_ns: u64,
    window_depth: usize,
) -> Result<serde_json::Value, Box<dyn Error>> {
    if next_layer != current_layer + 1
        || worker_end_ns < worker_start_ns
        || compute_end_ns < compute_start_ns
    {
        return Err("invalid one-layer-ahead fixed prefetch schedule".into());
    }
    let overlap_start_ns = worker_start_ns.max(compute_start_ns);
    let overlap_end_ns = worker_end_ns.min(compute_end_ns);
    Ok(serde_json::json!({
        "from_layer": current_layer,
        "to_layer": next_layer,
        "window_depth": window_depth,
        "worker_start_ns": worker_start_ns,
        "worker_end_ns": worker_end_ns,
        "host_prefetched_bytes": host_bytes,
        "prefetched_read_alloc_ns": telemetry.fixed_read_alloc_ns,
        "prefetched_decode_ns": telemetry.fixed_decode_ns,
        "ready_before_consume": worker_end_ns <= consume_start_ns,
        "consume_wait_ns": worker_end_ns.saturating_sub(consume_start_ns),
        "overlap_ns": overlap_end_ns.saturating_sub(overlap_start_ns),
    }))
}

impl DecodedLayerHost {
    pub(crate) fn source_bytes(&self) -> usize {
        fn matrix(matrix: &v4_state::Fp8Matrix) -> usize {
            matrix.weight.len() + matrix.scale.len()
        }
        fn compressor(weights: &v4_state::LayerCompressorWeights) -> usize {
            (weights.wkv_bf16.len() + weights.wgate_bf16.len() + weights.norm_bf16.len()) * 2
                + weights.ape_f32.len() * 4
        }
        let state = &self.state;
        let state_bytes = (state.hc_fn.len() + state.hc_base.len() + state.hc_scale.len()) * 4
            + state.attn_norm.len() * 2
            + compressor(&state.compressor)
            + state.indexer.as_ref().map_or(0, |indexer| {
                matrix(&indexer.wq_b)
                    + indexer.weights_proj_bf16.len() * 2
                    + compressor(&indexer.compressor)
            })
            + (state.ffn_norm.len() + state.gate.len()) * 2
            + match &state.router {
                v4_state::LayerRouterWeights::Hash { ids } => ids.len() * 8,
                v4_state::LayerRouterWeights::Learned { bias } => bias.len() * 4,
            };
        // The state and tail loaders currently read ffn_norm and gate twice, but
        // source_range_bytes is the unique fixed source-tensor footprint.
        let duplicated_tail_bytes =
            (self.tail.ffn_norm.len() + self.tail.gate.len()) * std::mem::size_of::<u16>();
        state_bytes + self.projections.range_bytes + self.tail.source_range_bytes
            - duplicated_tail_bytes
    }

    pub(crate) fn num_bytes(&self) -> usize {
        fn matrix(matrix: &v4_state::Fp8Matrix) -> usize {
            matrix.weight.len() + matrix.scale.len()
        }
        fn compressor(weights: &v4_state::LayerCompressorWeights) -> usize {
            (weights.wkv_bf16.len() + weights.wgate_bf16.len() + weights.norm_bf16.len()) * 2
                + weights.ape_f32.len() * 4
        }
        let state = &self.state;
        let state_bytes = (state.hc_fn.len() + state.hc_base.len() + state.hc_scale.len()) * 4
            + state.attn_norm.len() * 2
            + compressor(&state.compressor)
            + state.indexer.as_ref().map_or(0, |indexer| {
                matrix(&indexer.wq_b)
                    + indexer.weights_proj_bf16.len() * 2
                    + compressor(&indexer.compressor)
            })
            + (state.ffn_norm.len() + state.gate.len()) * 2
            + match &state.router {
                v4_state::LayerRouterWeights::Hash { ids } => ids.len() * 8,
                v4_state::LayerRouterWeights::Learned { bias } => bias.len() * 4,
            };
        let projections = &self.projections;
        let projection_bytes = matrix(&projections.wq_a.matrix)
            + projections.wq_a.norm.len() * 2
            + matrix(&projections.wq_b)
            + matrix(&projections.wkv.matrix)
            + projections.wkv.norm.len() * 2;
        let tail = &self.tail;
        let tail_bytes = tail.attn_sink.len() * 4
            + tail.wo_a_bf16.len() * 2
            + tail.wo_a_raw.as_ref().map_or(0, Vec::len)
            + matrix(&tail.wo_b)
            + (tail.hc_ffn_fn.len() + tail.hc_ffn_base.len() + tail.hc_ffn_scale.len()) * 4
            + (tail.ffn_norm.len() + tail.gate.len()) * 2
            + matrix(&tail.shared.w1)
            + matrix(&tail.shared.w2)
            + matrix(&tail.shared.w3);
        state_bytes + projection_bytes + tail_bytes
    }
}

#[derive(Clone)]
pub(crate) struct FixedTensorPlan {
    pub(crate) name: String,
    pub(crate) dtype: String,
    pub(crate) shape: Vec<u64>,
    pub(crate) source_bytes: usize,
    pub(crate) decoded_bytes: usize,
}

#[derive(Clone)]
pub(crate) struct FixedLayerPlan {
    pub(crate) layer: usize,
    pub(crate) tensors: Vec<FixedTensorPlan>,
    pub(crate) device_bytes: usize,
    pub(crate) raw_device_bytes: usize,
    pub(crate) host_bytes: usize,
    pub(crate) raw_host_bytes: usize,
    pub(crate) device_saved_ns: u64,
    pub(crate) host_saved_ns: u64,
}

#[derive(Default)]
pub(crate) struct FixedLoadTelemetry {
    pub(crate) fixed_read_alloc_ns: u64,
    pub(crate) fixed_decode_ns: u64,
    pub(crate) fixed_h2d_ns: u64,
    pub(crate) fixed_h2d_bytes: usize,
    pub(crate) fixed_source_bytes: usize,
    pub(crate) raw_read_ns: u64,
    pub(crate) raw_source_bytes: usize,
    pub(crate) raw_h2d_ns: u64,
    pub(crate) raw_h2d_bytes: usize,
    pub(crate) cache_status: &'static str,
    pub(crate) window_join_ns: u64,
    pub(crate) window_spawn_ns: u64,
    pub(crate) window_residual_ns: u64,
    pub(crate) window_worker_ns: u64,
    pub(crate) window_worker_ready: bool,
    pub(crate) bounded_readahead: Option<BoundedReadaheadTelemetry>,
}

pub(crate) struct BoundedReadaheadTelemetry {
    pub(crate) layer: usize,
    pub(crate) issued: bool,
    pub(crate) skipped_reason: Option<&'static str>,
    pub(crate) bytes: u64,
    pub(crate) ranges: Vec<safetensors::FileRange>,
    pub(crate) advice_us: u64,
}

pub(crate) const STATIONARY_RAW_LAYOUT: &str = "deepseek-v4-pro-layer-stationary-raw-v1";
pub(crate) const DECODE_PROMOTION_PLAN_ENV: &str = "COLIBRI_V4_DECODE_PROMOTION_PLAN";
pub(crate) const RAW_FIXED_SCRATCH_BYTES: usize = 128 * 1024 * 1024;

pub(crate) fn restore_leased_slot<T>(slot: &mut Option<T>, value: T) {
    assert!(
        slot.is_none(),
        "exclusive fixed-layer cache lease observed an ABA replacement"
    );
    *slot = Some(value);
}

pub(crate) struct StationaryRawLease {
    pub(crate) layer: usize,
    pub(crate) device_generation: Option<u64>,
    pub(crate) host_generation: Option<u64>,
    pub(crate) scratch_generation: Option<u64>,
    pub(crate) runtime: Option<LoadedLayerRuntime>,
    pub(crate) raw_device: Option<DeviceBuffer<u8>>,
    pub(crate) raw_host: Option<RawDecodedLayerHost>,
    pub(crate) scratch: Option<DeviceBuffer<u16>>,
    pub(crate) resident_bytes: usize,
    pub(crate) decoded_wo_a_in_scratch: bool,
}

pub(crate) struct PinnedFixedSlot {
    pub(crate) generation: u64,
    pub(crate) host: PinnedHostBuffer<u8>,
    pub(crate) scratch: Option<DeviceBuffer<u16>>,
    pub(crate) h2d_start: cuda_core::CudaEvent,
    pub(crate) h2d_end: cuda_core::CudaEvent,
    pub(crate) ready: cuda_core::CudaEvent,
    pub(crate) runtime_done: cuda_core::CudaEvent,
}

pub(crate) struct PinnedFixedPrepared {
    pub(crate) slot: PinnedFixedSlot,
    pub(crate) generation: u64,
    pub(crate) runtime_done_recorded: bool,
    pub(crate) lease: StationaryRawLease,
    pub(crate) telemetry: FixedLoadTelemetry,
}

impl PinnedFixedSlot {
    pub(crate) fn new(
        context: &std::sync::Arc<CudaContext>,
        scratch: DeviceBuffer<u16>,
    ) -> Result<Self, Box<dyn Error>> {
        let event = || context.new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT));
        Ok(Self {
            generation: 0,
            host: PinnedHostBuffer::zeroed(context, PHASE1_CSA_PAYLOAD_BYTES)?,
            scratch: Some(scratch),
            h2d_start: event()?,
            h2d_end: event()?,
            ready: event()?,
            runtime_done: event()?,
        })
    }
}

impl PinnedFixedPrepared {
    pub(crate) fn wait_ready(&mut self, main: &CudaStream) -> Result<(), Box<dyn Error>> {
        self.slot.ready.synchronize()?;
        main.wait(&self.slot.ready)?;
        self.telemetry.fixed_h2d_ns =
            (self.slot.h2d_start.elapsed_ms(&self.slot.h2d_end)? * 1_000_000.0) as u64;
        self.telemetry.fixed_decode_ns =
            (self.slot.h2d_end.elapsed_ms(&self.slot.ready)? * 1_000_000.0) as u64;
        Ok(())
    }

    pub(crate) fn runtime(&self) -> &LoadedLayerRuntime {
        &self.lease
    }

    pub(crate) fn record_runtime_done(&mut self, main: &CudaStream) -> Result<(), Box<dyn Error>> {
        self.slot.runtime_done.record(main)?;
        self.runtime_done_recorded = true;
        Ok(())
    }

    pub(crate) fn finish(
        mut self,
        cache: &mut FixedLayerCache,
        main: &CudaStream,
        transfer: &CudaStream,
    ) -> Result<PinnedFixedSlot, Box<dyn Error>> {
        if self.generation != self.slot.generation {
            return Err("pinned fixed slot rejected an ABA generation".into());
        }
        if !self.runtime_done_recorded {
            self.slot.runtime_done.record(main)?;
        }
        transfer.wait(&self.slot.runtime_done)?;
        std::mem::swap(
            &mut self
                .lease
                .runtime
                .as_mut()
                .ok_or("pinned fixed prepared lost its runtime")?
                .weights
                .wo_a,
            self.slot
                .scratch
                .as_mut()
                .ok_or("pinned fixed prepared lost its slot scratch")?,
        );
        cache.restore_stationary_raw(self.lease)?;
        Ok(self.slot)
    }
}

pub(crate) type PinnedFixedRequest = (
    Option<PinnedFixedPrepared>,
    Option<(usize, i64, Option<usize>)>,
);
pub(crate) type PinnedFixedReply = Result<Option<PinnedFixedPrepared>, String>;

pub(crate) fn run_pinned_fixed_request_worker(
    mut cache: FixedLayerCache,
    mut slots: Vec<PinnedFixedSlot>,
    root: &std::path::Path,
    module: &kernels::LoadedModule,
    main: &std::sync::Arc<CudaStream>,
    transfer: &std::sync::Arc<CudaStream>,
    rx: std::sync::mpsc::Receiver<PinnedFixedRequest>,
    reply_tx: std::sync::mpsc::SyncSender<PinnedFixedReply>,
    bounded_readahead: bool,
) -> Result<FixedLayerCache, String> {
    let readahead_page_cap = bounded_readahead
        .then(|| bounded_readahead_metadata_page_cap(root))
        .and_then(Result::ok);
    let mut pending_readahead = None;
    while let Ok((completed, next)) = rx.recv() {
        if let Some(completed) = completed {
            let layer = completed.lease.layer;
            match completed.finish(&mut cache, main, transfer) {
                Ok(slot) => slots.push(slot),
                Err(error) => {
                    let message = format!("layer {layer} phase runtime_done_restore: {error}");
                    let _ = reply_tx.send(Err(message.clone()));
                    return Err(message);
                }
            }
        }
        if let Some((layer, token_id, readahead_layer)) = next {
            let slot = match slots.pop() {
                Some(slot) => slot,
                None => {
                    let message = format!(
                        "layer {layer} phase acquire_slot: pinned fixed worker exhausted its two slots"
                    );
                    let _ = reply_tx.send(Err(message.clone()));
                    return Err(message);
                }
            };
            let mut prepared = match cache
                .prepare_stationary_raw_pinned(root, layer, token_id, module, transfer, slot)
            {
                Ok(prepared) => prepared,
                Err(error) => {
                    let message = format!("layer {layer} phase prepare: {error}");
                    let _ = reply_tx.send(Err(message.clone()));
                    return Err(message);
                }
            };
            if pending_readahead
                .as_ref()
                .is_some_and(|telemetry: &BoundedReadaheadTelemetry| telemetry.layer == layer)
            {
                let mut telemetry = pending_readahead
                    .take()
                    .expect("matching bounded readahead telemetry disappeared");
                telemetry.layer = layer;
                prepared.telemetry.bounded_readahead = Some(telemetry);
            }
            reply_tx.send(Ok(Some(prepared))).map_err(|_| {
                format!("layer {layer} phase reply_prepared: pinned fixed response receiver closed")
            })?;
            if bounded_readahead {
                if let Some(readahead_layer) = readahead_layer {
                    pending_readahead = Some(production_bounded_readahead(
                        root,
                        readahead_layer,
                        readahead_page_cap,
                    ));
                }
            }
        } else if slots.len() == 2 {
            if let Err(error) = cache.validate_decode_overlay_complete() {
                let message = format!("request phase overlay_drain: {error}");
                let _ = reply_tx.send(Err(message.clone()));
                return Err(message);
            }
            reply_tx.send(Ok(None)).map_err(|_| {
                "request phase reply_drained: pinned fixed response receiver closed".to_string()
            })?;
        }
    }
    if slots.len() != 2 {
        return Err(format!(
            "request phase shutdown: pinned fixed worker stopped with {} of 2 slots",
            slots.len()
        ));
    }
    let mut retained = slots
        .pop()
        .ok_or("request phase retain_scratch: pinned fixed worker lost its retained scratch")?;
    cache.raw_scratch = retained.scratch.take();
    transfer
        .synchronize()
        .map_err(|error| format!("request phase transfer_sync: {error}"))?;
    Ok(cache)
}

impl std::ops::Deref for StationaryRawLease {
    type Target = LoadedLayerRuntime;

    fn deref(&self) -> &Self::Target {
        self.runtime
            .as_ref()
            .expect("live stationary raw lease must own a runtime")
    }
}

pub(crate) fn decode_device_promotion_residency(
    device_resident_bytes: usize,
    host_resident_bytes: usize,
    promoted_device_bytes: usize,
    released_host_bytes: Option<usize>,
    device_budget_bytes: usize,
) -> Result<(usize, usize), Box<dyn Error>> {
    let device_resident_bytes = device_resident_bytes
        .checked_add(promoted_device_bytes)
        .filter(|&bytes| bytes <= device_budget_bytes)
        .ok_or("decode overlay device promotion violated its target budget")?;
    let host_resident_bytes = match released_host_bytes {
        Some(bytes) => host_resident_bytes
            .checked_sub(bytes)
            .ok_or("decode overlay host resident byte underflow")?,
        None => host_resident_bytes,
    };
    Ok((device_resident_bytes, host_resident_bytes))
}

pub(crate) fn decode_host_promotion_residency(
    host_resident_bytes: usize,
    promoted_host_bytes: usize,
    host_budget_bytes: usize,
) -> Result<usize, Box<dyn Error>> {
    host_resident_bytes
        .checked_add(promoted_host_bytes)
        .filter(|&bytes| bytes <= host_budget_bytes)
        .ok_or_else(|| "decode overlay host promotion violated its target budget".into())
}

pub(crate) fn decode_overlay_uses_decode_target(
    record_count: usize,
    layer_count: usize,
    device_promotion_count: usize,
    host_promotion_count: usize,
) -> Result<bool, Box<dyn Error>> {
    if layer_count == 0 || record_count % layer_count != 0 {
        return Err("decode overlay layer records did not close complete decode rows".into());
    }
    if record_count == 0 && (device_promotion_count != 0 || host_promotion_count != 0) {
        return Err("stationary raw prefill residency unexpectedly contains promotions".into());
    }
    Ok(record_count != 0)
}
