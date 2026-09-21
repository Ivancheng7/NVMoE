use super::*;

pub(crate) struct ExpertSlot<'a> {
    pub(crate) spec: &'static v4_spec::V4Spec,
    pub(crate) fp4_accum_arm: FlashFp4AccumArm,
    pub(crate) tilelang_ref: Option<v4_tilelang_ref::TilelangFp4Reference>,
    pub(crate) module: &'a kernels::LoadedModule,
    pub(crate) stream: std::sync::Arc<CudaStream>,
    pub(crate) w1_host: PinnedHostBuffer<u8>,
    pub(crate) w1_scale_host: PinnedHostBuffer<u8>,
    pub(crate) w2_host: PinnedHostBuffer<u8>,
    pub(crate) w2_scale_host: PinnedHostBuffer<u8>,
    pub(crate) w3_host: PinnedHostBuffer<u8>,
    pub(crate) w3_scale_host: PinnedHostBuffer<u8>,
    pub(crate) input_host: PinnedHostBuffer<u16>,
    pub(crate) output_host: PinnedHostBuffer<f32>,
    pub(crate) input_device: DeviceBuffer<u16>,
    pub(crate) w1_device: DeviceBuffer<u8>,
    pub(crate) w1_scale_device: DeviceBuffer<u8>,
    pub(crate) w2_device: DeviceBuffer<u8>,
    pub(crate) w2_scale_device: DeviceBuffer<u8>,
    pub(crate) w3_device: DeviceBuffer<u8>,
    pub(crate) w3_scale_device: DeviceBuffer<u8>,
    pub(crate) input_quantized: DeviceBuffer<f32>,
    pub(crate) input_fp8: Option<DeviceBuffer<u8>>,
    pub(crate) input_fp8_scales: Option<DeviceBuffer<u8>>,
    pub(crate) gate: DeviceBuffer<u16>,
    pub(crate) up: DeviceBuffer<u16>,
    pub(crate) routed: DeviceBuffer<u16>,
    pub(crate) routed_quantized: DeviceBuffer<f32>,
    pub(crate) routed_fp8: Option<DeviceBuffer<u8>>,
    pub(crate) routed_fp8_scales: Option<DeviceBuffer<u8>>,
    pub(crate) output_bf16: DeviceBuffer<u16>,
    pub(crate) output: DeviceBuffer<f32>,
    pub(crate) weight_h2d_ms: f64,
    pub(crate) weight_start: cuda_core::CudaEvent,
    pub(crate) weight_end: cuda_core::CudaEvent,
    pub(crate) kernel_start: cuda_core::CudaEvent,
    pub(crate) kernel_end: cuda_core::CudaEvent,
    pub(crate) complete: cuda_core::CudaEvent,
}

/// One routed expert slot held for a whole request instead of one layer.
///
/// Building a slot page-locks 12.75 MiB of staging through `cuMemHostAlloc`
/// (measured at 17.4 ms per call, plus 3.9 ms for the slot's device buffers),
/// which is 21 ms of every layer while the slot dies at the end of that layer.
/// Whoever owns this scratch keeps it in the scope that spans the layer loop.
pub(crate) struct RoutedSlotScratch<'a> {
    slot: Option<ExpertSlot<'a>>,
    pool: Option<RoutedReadPool>,
    cache: Option<RoutedExpertCache>,
}

impl<'a> RoutedSlotScratch<'a> {
    pub(crate) fn new() -> Self {
        Self {
            slot: None,
            pool: None,
            cache: None,
        }
    }

    /// Hands the slot to the layer being run, leaving the scratch empty.
    pub(crate) fn take_slot(&mut self) -> Option<ExpertSlot<'a>> {
        self.slot.take()
    }

    /// Takes the layer's last slot back so the next layer can restage into it.
    pub(crate) fn put_slot(&mut self, slot: ExpertSlot<'a>) {
        self.slot = Some(slot);
    }

    /// Hands the pinned read pool to the layer's prefetch, leaving it empty.
    pub(crate) fn take_pool(&mut self) -> Option<RoutedReadPool> {
        self.pool.take()
    }

    /// Takes the pool back after the layer's routed block drained it.
    pub(crate) fn put_pool(&mut self, pool: RoutedReadPool) {
        self.pool = Some(pool);
    }

    /// Builds the identity cache on first use, so page-locking its lanes is a
    /// once-per-request cost like the pool's.
    pub(crate) fn ensure_cache(
        &mut self,
        context: &std::sync::Arc<CudaContext>,
        spec: &'static v4_spec::V4Spec,
        capacity: usize,
    ) -> Result<(), Box<dyn Error>> {
        if self.cache.as_ref().map(|cache| cache.capacity) != Some(capacity)
            || !self
                .cache
                .as_ref()
                .map(|cache| cache.shapes_match(spec))
                .unwrap_or(false)
        {
            self.cache = Some(RoutedExpertCache::new(context, spec, capacity)?);
        }
        Ok(())
    }

    /// One lane per route slot from the identity cache, with a `true` flag where
    /// the lane already holds the expert (skip the read).
    pub(crate) fn take_cache_lanes(
        &mut self,
        layer: usize,
        route_ids: &[i32],
    ) -> Result<(Vec<RoutedReadLane>, Vec<bool>), Box<dyn Error>> {
        self.cache
            .as_mut()
            .ok_or("routed expert cache is not built")?
            .take_lanes(layer, route_ids)
    }

    /// Returns this layer's lanes to the identity cache so a later token can hit.
    pub(crate) fn give_cache_lanes(
        &mut self,
        layer: usize,
        route_ids: &[i32],
        lanes: Vec<RoutedReadLane>,
    ) {
        if let Some(cache) = self.cache.as_mut() {
            cache.give_back(layer, route_ids, lanes);
        }
    }
}

/// The page-locked staging sizes one routed expert occupies, in the same order
/// the range plan reads them. Mirrors the arithmetic `new_empty_for` uses for
/// the device side; a divergence fails loudly on the range plan's length check
/// rather than quietly reading less than an expert.
fn routed_lane_bytes(spec: &v4_spec::V4Spec) -> [usize; 6] {
    let w1 = spec.moe_intermediate * spec.hidden / 2;
    let w1_scale = spec.moe_intermediate * spec.hidden / 32;
    let w2 = spec.hidden * spec.moe_intermediate / 2;
    let w2_scale = spec.hidden * spec.moe_intermediate / 32;
    [w1, w1_scale, w2, w2_scale, w1, w1_scale]
}

/// One routed slot's worth of page-locked staging, filled by a prefetch worker
/// straight from disk and drained by the weight H2D.
///
/// This is the buffer `restage_weights` used to copy into. Reading into it
/// directly removes the 12.75 MiB host-side `copy_from_slice` per expert and
/// the fresh zeroed `Vec` the streamed reader allocated per tensor. The
/// 2026-09-17 replay measured pinned staging reading at the reuse-buffer rate,
/// not the fresh-buffer rate, so neither removal should cost read bandwidth.
pub(crate) struct RoutedReadLane {
    spec: &'static v4_spec::V4Spec,
    w1_host: PinnedHostBuffer<u8>,
    w1_scale_host: PinnedHostBuffer<u8>,
    w2_host: PinnedHostBuffer<u8>,
    w2_scale_host: PinnedHostBuffer<u8>,
    w3_host: PinnedHostBuffer<u8>,
    w3_scale_host: PinnedHostBuffer<u8>,
    read_ns: u64,
}

impl RoutedReadLane {
    fn new(
        context: &std::sync::Arc<CudaContext>,
        spec: &'static v4_spec::V4Spec,
    ) -> Result<Self, Box<dyn Error>> {
        let [w1, w1_scale, w2, w2_scale, w3, w3_scale] = routed_lane_bytes(spec);
        Ok(Self {
            spec,
            w1_host: PinnedHostBuffer::zeroed(context, w1)?,
            w1_scale_host: PinnedHostBuffer::zeroed(context, w1_scale)?,
            w2_host: PinnedHostBuffer::zeroed(context, w2)?,
            w2_scale_host: PinnedHostBuffer::zeroed(context, w2_scale)?,
            w3_host: PinnedHostBuffer::zeroed(context, w3)?,
            w3_scale_host: PinnedHostBuffer::zeroed(context, w3_scale)?,
            read_ns: 0,
        })
    }

    fn shapes_match(&self, spec: &v4_spec::V4Spec) -> bool {
        let [w1, w1_scale, w2, w2_scale, w3, w3_scale] = routed_lane_bytes(spec);
        self.w1_host.len() == w1
            && self.w1_scale_host.len() == w1_scale
            && self.w2_host.len() == w2
            && self.w2_scale_host.len() == w2_scale
            && self.w3_host.len() == w3
            && self.w3_scale_host.len() == w3_scale
    }

    pub(crate) fn h2d_bytes(&self) -> usize {
        self.w1_host.num_bytes()
            + self.w1_scale_host.num_bytes()
            + self.w2_host.num_bytes()
            + self.w2_scale_host.num_bytes()
            + self.w3_host.num_bytes()
            + self.w3_scale_host.num_bytes()
    }

    pub(crate) fn read_ns(&self) -> u64 {
        self.read_ns
    }

    /// Reads the layer's expert straight into this lane's page-locked staging.
    /// The range plan enforces the same six-tensor shape and E8M0 scale check
    /// the `RoutedExpert` load path performed, so the bytes are validated
    /// identically before they reach the device.
    pub(crate) fn read_into(
        &mut self,
        shard: &safetensors::ShardFile,
        layer: usize,
        expert_id: u16,
    ) -> Result<(), Box<dyn Error>> {
        let plan = v4_expert::RoutedExpertRangePlan::new_for(self.spec, shard, layer, expert_id)?;
        let started = std::time::Instant::now();
        plan.read_into(
            shard,
            [
                self.w1_host.as_mut_slice(),
                self.w1_scale_host.as_mut_slice(),
                self.w2_host.as_mut_slice(),
                self.w2_scale_host.as_mut_slice(),
                self.w3_host.as_mut_slice(),
                self.w3_scale_host.as_mut_slice(),
            ],
        )?;
        self.read_ns = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        Ok(())
    }
}

/// The six page-locked read lanes a request-scoped scratch keeps, so the routed
/// prefetch reads disk straight into the staging the H2D drains instead of into
/// a `Vec` that is then copied across.
///
/// Six lanes because the six routed experts are read on six streams. The device
/// side still owns exactly one execution slot, so this trades pinned host
/// memory (12.75 MiB per lane) for the per-expert staging copy and per-read
/// allocation, not for six sets of device buffers.
pub(crate) struct RoutedReadPool {
    lanes: Vec<RoutedReadLane>,
}

impl RoutedReadPool {
    pub(crate) fn new(
        context: &std::sync::Arc<CudaContext>,
        spec: &'static v4_spec::V4Spec,
    ) -> Result<Self, Box<dyn Error>> {
        let lanes = (0..spec.experts_per_token)
            .map(|_| RoutedReadLane::new(context, spec))
            .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
        Ok(Self { lanes })
    }

    pub(crate) fn shapes_match(&self, spec: &v4_spec::V4Spec) -> bool {
        !self.lanes.is_empty() && self.lanes.iter().all(|lane| lane.shapes_match(spec))
    }

    /// Hands the lanes out by value so the read workers own them and the drain
    /// can carry them straight to the H2D (and, when the identity cache is on,
    /// back into it). Callers return them with [`Self::give_lanes`].
    pub(crate) fn take_lanes(&mut self) -> Vec<RoutedReadLane> {
        std::mem::take(&mut self.lanes)
    }

    pub(crate) fn give_lanes(&mut self, lanes: Vec<RoutedReadLane>) {
        self.lanes = lanes;
    }
}

/// Request-scoped cache of routed experts keyed by identity `(layer, expert)`,
/// not by route position.
///
/// The bytes a lane holds after `read_into` are exactly what the range plan
/// reads from disk, so serving a hit from a retained lane produces the same
/// device weights a fresh read would: the numeric path is untouched, only
/// *whether* a read happens changes. The cache owns every lane — `entries` for
/// the cached experts and `spares` for the checked-out/evicted ones — so a hit
/// hands its lane straight to the H2D with no copy at all.
///
/// Reuse distance is a full layer sweep (a token reads 43 layers before
/// revisiting one), measured at ~63% consecutive-token overlap on a real
/// prompt, so the payoff only appears once the capacity holds roughly a token's
/// working set (~258 experts); below that it thrashes toward 0%.
pub(crate) struct RoutedExpertCache {
    entries: std::collections::HashMap<(usize, u16), (RoutedReadLane, u64)>,
    spares: Vec<RoutedReadLane>,
    context: std::sync::Arc<CudaContext>,
    spec: &'static v4_spec::V4Spec,
    clock: u64,
    capacity: usize,
    max_lanes: usize,
    pub(crate) hits: usize,
    pub(crate) misses: usize,
}

impl RoutedExpertCache {
    pub(crate) fn new(
        context: &std::sync::Arc<CudaContext>,
        spec: &'static v4_spec::V4Spec,
        capacity: usize,
    ) -> Result<Self, Box<dyn Error>> {
        // Lanes are page-locked lazily as misses arrive, so a short request never
        // pays for the whole capacity. `experts_per_token` lanes are always kept
        // around so a layer where every route misses can still draw one each.
        Ok(Self {
            entries: std::collections::HashMap::new(),
            spares: Vec::new(),
            context: context.clone(),
            spec,
            clock: 0,
            capacity,
            max_lanes: capacity.saturating_add(spec.experts_per_token),
            hits: 0,
            misses: 0,
        })
    }

    pub(crate) fn shapes_match(&self, spec: &v4_spec::V4Spec) -> bool {
        self.entries
            .values()
            .all(|(lane, _)| lane.shapes_match(spec))
            && self.spares.iter().all(|lane| lane.shapes_match(spec))
    }

    /// A free lane, allocating one while under the cap and otherwise evicting the
    /// least-recently-used cached expert.
    fn acquire_lane(&mut self) -> Result<RoutedReadLane, Box<dyn Error>> {
        if let Some(lane) = self.spares.pop() {
            return Ok(lane);
        }
        if self.entries.len() + self.spares.len() < self.max_lanes {
            return RoutedReadLane::new(&self.context, self.spec);
        }
        let victim = self
            .entries
            .iter()
            .min_by_key(|(key, (_, tick))| (*tick, **key))
            .map(|(key, _)| *key)
            .ok_or("routed expert cache has no victim to evict")?;
        let (lane, _) = self
            .entries
            .remove(&victim)
            .ok_or("routed expert cache victim vanished")?;
        Ok(lane)
    }

    /// Hands out one lane per route slot: a cached lane for a hit (bytes already
    /// present), a free lane for a miss (to be read). The `true` flag tells the
    /// read worker to skip the disk read for that slot.
    pub(crate) fn take_lanes(
        &mut self,
        layer: usize,
        route_ids: &[i32],
    ) -> Result<(Vec<RoutedReadLane>, Vec<bool>), Box<dyn Error>> {
        let mut lanes = Vec::with_capacity(route_ids.len());
        let mut hits = Vec::with_capacity(route_ids.len());
        for &raw in route_ids {
            let expert = u16::try_from(raw).map_err(|error| error.to_string())?;
            self.clock += 1;
            match self.entries.remove(&(layer, expert)) {
                Some((lane, _)) => {
                    self.hits += 1;
                    lanes.push(lane);
                    hits.push(true);
                }
                None => {
                    self.misses += 1;
                    lanes.push(self.acquire_lane()?);
                    hits.push(false);
                }
            }
        }
        Ok((lanes, hits))
    }

    /// Takes this layer's lanes back: each becomes the cache entry for its
    /// `(layer, expert)` with a fresh use tick, and the least-recently-used
    /// experts spill into the spare pool while the capacity is exceeded.
    pub(crate) fn give_back(
        &mut self,
        layer: usize,
        route_ids: &[i32],
        lanes: Vec<RoutedReadLane>,
    ) {
        for (lane, &raw) in lanes.into_iter().zip(route_ids) {
            let Ok(expert) = u16::try_from(raw) else {
                self.spares.push(lane);
                continue;
            };
            if self.capacity == 0 {
                self.spares.push(lane);
                continue;
            }
            self.clock += 1;
            self.entries.insert((layer, expert), (lane, self.clock));
            while self.entries.len() > self.capacity {
                let Some(victim) = self
                    .entries
                    .iter()
                    .min_by_key(|(key, (_, tick))| (*tick, **key))
                    .map(|(key, _)| *key)
                else {
                    break;
                };
                if let Some((evicted, _)) = self.entries.remove(&victim) {
                    self.spares.push(evicted);
                }
            }
        }
    }
}

impl<'a> ExpertSlot<'a> {
    pub(crate) fn new(
        context: &std::sync::Arc<CudaContext>,
        module: &'a kernels::LoadedModule,
        stream: &std::sync::Arc<CudaStream>,
        weights: &v4_expert::RoutedExpert,
    ) -> Result<Self, Box<dyn Error>> {
        // ponytail: one L0 expert at a time; add slot management only after
        // measured multi-expert traffic requires it.
        let w1_host = PinnedHostBuffer::from_slice(context, &weights.w1)?;
        let w1_scale_host = PinnedHostBuffer::from_slice(context, &weights.w1_scale)?;
        let w2_host = PinnedHostBuffer::from_slice(context, &weights.w2)?;
        let w2_scale_host = PinnedHostBuffer::from_slice(context, &weights.w2_scale)?;
        let w3_host = PinnedHostBuffer::from_slice(context, &weights.w3)?;
        let w3_scale_host = PinnedHostBuffer::from_slice(context, &weights.w3_scale)?;
        let spec = weights.spec;
        let input_host = PinnedHostBuffer::zeroed(context, spec.hidden)?;
        let output_host = PinnedHostBuffer::zeroed(context, spec.hidden)?;
        let input_device = unsafe { DeviceBuffer::uninitialized_async(stream, spec.hidden)? };
        let mut w1_device = unsafe { DeviceBuffer::uninitialized_async(stream, weights.w1.len())? };
        let mut w1_scale_device =
            unsafe { DeviceBuffer::uninitialized_async(stream, weights.w1_scale.len())? };
        let mut w2_device = unsafe { DeviceBuffer::uninitialized_async(stream, weights.w2.len())? };
        let mut w2_scale_device =
            unsafe { DeviceBuffer::uninitialized_async(stream, weights.w2_scale.len())? };
        let mut w3_device = unsafe { DeviceBuffer::uninitialized_async(stream, weights.w3.len())? };
        let mut w3_scale_device =
            unsafe { DeviceBuffer::uninitialized_async(stream, weights.w3_scale.len())? };
        let weight_h2d_start = context.new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
        let weight_h2d_end = context.new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
        weight_h2d_start.record(stream)?;
        // SAFETY: pinned weight sources live through the synchronized completion
        // event below; every destination has the exact matching allocation.
        unsafe {
            w1_device.copy_from_pinned_host_async(stream, &w1_host)?;
            w1_scale_device.copy_from_pinned_host_async(stream, &w1_scale_host)?;
            w2_device.copy_from_pinned_host_async(stream, &w2_host)?;
            w2_scale_device.copy_from_pinned_host_async(stream, &w2_scale_host)?;
            w3_device.copy_from_pinned_host_async(stream, &w3_host)?;
            w3_scale_device.copy_from_pinned_host_async(stream, &w3_scale_host)?;
        }
        weight_h2d_end.record(stream)?;
        weight_h2d_end.synchronize()?;
        let weight_h2d_ms = weight_h2d_start.elapsed_ms(&weight_h2d_end)? as f64;
        let fp4_accum_arm = flash_fp4_accum_arm(spec)?;
        let tilelang_ref = if fp4_accum_arm == FlashFp4AccumArm::TilelangRef {
            let model_dir = v4_model_dir_for(spec)?;
            let reference_dir = v4_tilelang_ref::reference_dir_from_env()?;
            Some(v4_tilelang_ref::load_for_flash(
                context,
                spec,
                &model_dir,
                &reference_dir,
            )?)
        } else {
            None
        };
        let input_quantized = unsafe { DeviceBuffer::uninitialized_async(stream, spec.hidden)? };
        let (input_fp8, input_fp8_scales, routed_fp8, routed_fp8_scales) =
            if fp4_accum_arm.uses_raw_fp8() {
                (
                    Some(DeviceBuffer::<u8>::zeroed(stream, spec.hidden)?),
                    Some(DeviceBuffer::<u8>::zeroed(
                        stream,
                        spec.hidden / v4_state::FP8_BLOCK,
                    )?),
                    Some(DeviceBuffer::<u8>::zeroed(stream, spec.moe_intermediate)?),
                    Some(DeviceBuffer::<u8>::zeroed(
                        stream,
                        spec.moe_intermediate / v4_state::FP8_BLOCK,
                    )?),
                )
            } else {
                (None, None, None, None)
            };
        let gate = unsafe { DeviceBuffer::uninitialized_async(stream, spec.moe_intermediate)? };
        let up = unsafe { DeviceBuffer::uninitialized_async(stream, spec.moe_intermediate)? };
        let routed = unsafe { DeviceBuffer::uninitialized_async(stream, spec.moe_intermediate)? };
        let routed_quantized =
            unsafe { DeviceBuffer::uninitialized_async(stream, spec.moe_intermediate)? };
        let output_bf16 = unsafe { DeviceBuffer::uninitialized_async(stream, spec.hidden)? };
        let output = unsafe { DeviceBuffer::uninitialized_async(stream, spec.hidden)? };
        let kernel_start = context.new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
        let kernel_end = context.new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
        let complete = context.new_event(None)?;
        complete.record(stream)?;
        complete.synchronize()?;
        Ok(Self {
            spec,
            fp4_accum_arm,
            tilelang_ref,
            module,
            stream: stream.clone(),
            w1_host,
            w1_scale_host,
            w2_host,
            w2_scale_host,
            w3_host,
            w3_scale_host,
            input_host,
            output_host,
            input_device,
            w1_device,
            w1_scale_device,
            w2_device,
            w2_scale_device,
            w3_device,
            w3_scale_device,
            input_quantized,
            input_fp8,
            input_fp8_scales,
            gate,
            up,
            routed,
            routed_quantized,
            routed_fp8,
            routed_fp8_scales,
            output_bf16,
            output,
            weight_h2d_ms,
            weight_start: weight_h2d_start,
            weight_end: weight_h2d_end,
            kernel_start,
            kernel_end,
            complete,
        })
    }

    /// Builds a slot for `weights`, restaging `recycled`'s pinned staging
    /// buffers when the shapes match and falling back to [`Self::new`] when they
    /// do not.
    ///
    /// Page-locking those six buffers is the whole cost of building a slot
    /// (measured: 12.96 ms of the 14.40 ms each expert costs in a 1-step run),
    /// and the routed experts a layer visits all have the same shape, so a layer
    /// loop can keep one slot and restage it instead of rebuilding. The caller
    /// must pass the same context/module/stream the recycled slot was built
    /// with; the restaged buffers are the source of the weight H2D copies either
    /// way, so the lifetime rule from `new` still holds.
    pub(crate) fn new_or_restage(
        recycled: Option<Self>,
        context: &std::sync::Arc<CudaContext>,
        module: &'a kernels::LoadedModule,
        stream: &std::sync::Arc<CudaStream>,
        weights: &v4_expert::RoutedExpert,
    ) -> Result<Self, Box<dyn Error>> {
        match recycled {
            Some(mut slot) if slot.staging_shapes_match(weights) => {
                slot.restage_weights(weights)?;
                Ok(slot)
            }
            _ => Self::new(context, module, stream, weights),
        }
    }

    fn staging_shapes_match(&self, weights: &v4_expert::RoutedExpert) -> bool {
        self.w1_host.len() == weights.w1.len()
            && self.w1_scale_host.len() == weights.w1_scale.len()
            && self.w2_host.len() == weights.w2.len()
            && self.w2_scale_host.len() == weights.w2_scale.len()
            && self.w3_host.len() == weights.w3.len()
            && self.w3_scale_host.len() == weights.w3_scale.len()
    }

    /// Copies `weights` into this slot's existing pinned buffers and re-uploads
    /// them to the device, replacing both the device-side weights and
    /// `weight_h2d_ms`. Mirrors what `load_range_execute_into` does for the QD4
    /// lane path, which never rebuilds a slot per expert.
    pub(crate) fn restage_weights(
        &mut self,
        weights: &v4_expert::RoutedExpert,
    ) -> Result<(), Box<dyn Error>> {
        self.w1_host.as_mut_slice().copy_from_slice(&weights.w1);
        self.w1_scale_host
            .as_mut_slice()
            .copy_from_slice(&weights.w1_scale);
        self.w2_host.as_mut_slice().copy_from_slice(&weights.w2);
        self.w2_scale_host
            .as_mut_slice()
            .copy_from_slice(&weights.w2_scale);
        self.w3_host.as_mut_slice().copy_from_slice(&weights.w3);
        self.w3_scale_host
            .as_mut_slice()
            .copy_from_slice(&weights.w3_scale);
        self.weight_start.record(&self.stream)?;
        // SAFETY: the pinned sources outlive the synchronized completion event
        // below, same as in `new`. Overwriting the device weights is ordered
        // after the previous expert's kernels because `execute_into`
        // synchronizes its own completion event before returning.
        unsafe {
            self.w1_device
                .copy_from_pinned_host_async(&self.stream, &self.w1_host)?;
            self.w1_scale_device
                .copy_from_pinned_host_async(&self.stream, &self.w1_scale_host)?;
            self.w2_device
                .copy_from_pinned_host_async(&self.stream, &self.w2_host)?;
            self.w2_scale_device
                .copy_from_pinned_host_async(&self.stream, &self.w2_scale_host)?;
            self.w3_device
                .copy_from_pinned_host_async(&self.stream, &self.w3_host)?;
            self.w3_scale_device
                .copy_from_pinned_host_async(&self.stream, &self.w3_scale_host)?;
        }
        self.weight_end.record(&self.stream)?;
        self.weight_end.synchronize()?;
        self.weight_h2d_ms = self.weight_start.elapsed_ms(&self.weight_end)? as f64;
        Ok(())
    }

    /// Uploads one already-read lane's pinned staging to this slot's device
    /// weights and records the H2D time, without touching the slot's own pinned
    /// staging at all.
    ///
    /// This is the read-straight-into-pinned path's counterpart to
    /// [`Self::restage_weights`]: the bytes are already in the lane's
    /// page-locked buffers, so the per-expert host-side copy disappears. The
    /// slot's own `w*_host` staging is left idle on this path; it costs its
    /// one-time page-lock either way, and the first expert of the first layer
    /// builds the slot through [`Self::new_empty_for`] instead of [`Self::new`].
    pub(crate) fn upload_lane(&mut self, lane: &RoutedReadLane) -> Result<(), Box<dyn Error>> {
        if lane.w1_host.len() != self.w1_device.len()
            || lane.w1_scale_host.len() != self.w1_scale_device.len()
            || lane.w2_host.len() != self.w2_device.len()
            || lane.w2_scale_host.len() != self.w2_scale_device.len()
            || lane.w3_host.len() != self.w3_device.len()
            || lane.w3_scale_host.len() != self.w3_scale_device.len()
        {
            return Err("pinned read lane does not match the execution slot geometry".into());
        }
        self.weight_start.record(&self.stream)?;
        // SAFETY: the lane's pinned sources outlive the synchronized
        // completion event below, and the device destinations have the exact
        // matching allocation (checked above). Overwriting the device weights
        // is ordered after the previous expert's kernels because `execute_into`
        // synchronizes its own completion event before returning.
        unsafe {
            self.w1_device
                .copy_from_pinned_host_async(&self.stream, &lane.w1_host)?;
            self.w1_scale_device
                .copy_from_pinned_host_async(&self.stream, &lane.w1_scale_host)?;
            self.w2_device
                .copy_from_pinned_host_async(&self.stream, &lane.w2_host)?;
            self.w2_scale_device
                .copy_from_pinned_host_async(&self.stream, &lane.w2_scale_host)?;
            self.w3_device
                .copy_from_pinned_host_async(&self.stream, &lane.w3_host)?;
            self.w3_scale_device
                .copy_from_pinned_host_async(&self.stream, &lane.w3_scale_host)?;
        }
        self.weight_end.record(&self.stream)?;
        self.weight_end.synchronize()?;
        self.weight_h2d_ms = self.weight_start.elapsed_ms(&self.weight_end)? as f64;
        Ok(())
    }

    pub(crate) fn new_empty(
        context: &std::sync::Arc<CudaContext>,
        module: &'a kernels::LoadedModule,
        stream: &std::sync::Arc<CudaStream>,
    ) -> Result<Self, Box<dyn Error>> {
        Self::new_empty_for(&v4_spec::PRO, context, module, stream)
    }

    pub(crate) fn new_empty_for(
        spec: &'static v4_spec::V4Spec,
        context: &std::sync::Arc<CudaContext>,
        module: &'a kernels::LoadedModule,
        stream: &std::sync::Arc<CudaStream>,
    ) -> Result<Self, Box<dyn Error>> {
        let w1_bytes = spec.moe_intermediate * spec.hidden / 2;
        let w1_scale_bytes = spec.moe_intermediate * spec.hidden / 32;
        let w2_bytes = spec.hidden * spec.moe_intermediate / 2;
        let w2_scale_bytes = spec.hidden * spec.moe_intermediate / 32;
        let w1_host = PinnedHostBuffer::zeroed(context, w1_bytes)?;
        let w1_scale_host = PinnedHostBuffer::zeroed(context, w1_scale_bytes)?;
        let w2_host = PinnedHostBuffer::zeroed(context, w2_bytes)?;
        let w2_scale_host = PinnedHostBuffer::zeroed(context, w2_scale_bytes)?;
        let w3_host = PinnedHostBuffer::zeroed(context, w1_bytes)?;
        let w3_scale_host = PinnedHostBuffer::zeroed(context, w1_scale_bytes)?;
        let input_host = PinnedHostBuffer::zeroed(context, spec.hidden)?;
        let output_host = PinnedHostBuffer::zeroed(context, spec.hidden)?;
        let input_device = unsafe { DeviceBuffer::uninitialized_async(stream, spec.hidden)? };
        let w1_device = unsafe { DeviceBuffer::uninitialized_async(stream, w1_bytes)? };
        let w1_scale_device = unsafe { DeviceBuffer::uninitialized_async(stream, w1_scale_bytes)? };
        let w2_device = unsafe { DeviceBuffer::uninitialized_async(stream, w2_bytes)? };
        let w2_scale_device = unsafe { DeviceBuffer::uninitialized_async(stream, w2_scale_bytes)? };
        let w3_device = unsafe { DeviceBuffer::uninitialized_async(stream, w1_bytes)? };
        let w3_scale_device = unsafe { DeviceBuffer::uninitialized_async(stream, w1_scale_bytes)? };
        let fp4_accum_arm = flash_fp4_accum_arm(spec)?;
        let tilelang_ref = if fp4_accum_arm == FlashFp4AccumArm::TilelangRef {
            let model_dir = v4_model_dir_for(spec)?;
            let reference_dir = v4_tilelang_ref::reference_dir_from_env()?;
            Some(v4_tilelang_ref::load_for_flash(
                context,
                spec,
                &model_dir,
                &reference_dir,
            )?)
        } else {
            None
        };
        let input_quantized = unsafe { DeviceBuffer::uninitialized_async(stream, spec.hidden)? };
        let (input_fp8, input_fp8_scales, routed_fp8, routed_fp8_scales) =
            if fp4_accum_arm.uses_raw_fp8() {
                (
                    Some(DeviceBuffer::<u8>::zeroed(stream, spec.hidden)?),
                    Some(DeviceBuffer::<u8>::zeroed(
                        stream,
                        spec.hidden / v4_state::FP8_BLOCK,
                    )?),
                    Some(DeviceBuffer::<u8>::zeroed(stream, spec.moe_intermediate)?),
                    Some(DeviceBuffer::<u8>::zeroed(
                        stream,
                        spec.moe_intermediate / v4_state::FP8_BLOCK,
                    )?),
                )
            } else {
                (None, None, None, None)
            };
        let gate = unsafe { DeviceBuffer::uninitialized_async(stream, spec.moe_intermediate)? };
        let up = unsafe { DeviceBuffer::uninitialized_async(stream, spec.moe_intermediate)? };
        let routed = unsafe { DeviceBuffer::uninitialized_async(stream, spec.moe_intermediate)? };
        let routed_quantized =
            unsafe { DeviceBuffer::uninitialized_async(stream, spec.moe_intermediate)? };
        let output_bf16 = unsafe { DeviceBuffer::uninitialized_async(stream, spec.hidden)? };
        let output = unsafe { DeviceBuffer::uninitialized_async(stream, spec.hidden)? };
        let kernel_start = context.new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
        let kernel_end = context.new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
        let weight_start = context.new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
        let weight_end = context.new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
        let complete = context.new_event(None)?;
        complete.record(stream)?;
        complete.synchronize()?;
        Ok(Self {
            spec,
            fp4_accum_arm,
            tilelang_ref,
            module,
            stream: stream.clone(),
            w1_host,
            w1_scale_host,
            w2_host,
            w2_scale_host,
            w3_host,
            w3_scale_host,
            input_host,
            output_host,
            input_device,
            w1_device,
            w1_scale_device,
            w2_device,
            w2_scale_device,
            w3_device,
            w3_scale_device,
            input_quantized,
            input_fp8,
            input_fp8_scales,
            gate,
            up,
            routed,
            routed_quantized,
            routed_fp8,
            routed_fp8_scales,
            output_bf16,
            output,
            weight_h2d_ms: 0.0,
            weight_start,
            weight_end,
            kernel_start,
            kernel_end,
            complete,
        })
    }

    pub(crate) fn load_range_execute_into(
        &mut self,
        shard: &safetensors::ShardFile,
        plan: &v4_expert::RoutedExpertRangePlan,
        input: &[u16],
        route_weight: f32,
        output: &mut [f32],
        profile_origin: Option<std::time::Instant>,
        profile_submit_ns: Option<u64>,
    ) -> Result<RoutedLaneExecution, Box<dyn Error>> {
        let mut timeline = profile_origin.map(|origin| RoutedLaneTimeline {
            submit_ns: profile_submit_ns.unwrap_or_else(|| timeline_elapsed_ns(origin)),
            read_start_ns: timeline_elapsed_ns(origin),
            read_end_ns: 0,
            h2d_enqueued_ns: 0,
            kernel_enqueued_ns: 0,
            d2h_enqueued_ns: 0,
            complete_ns: 0,
            h2d_event_ns: 0,
            kernel_event_ns: 0,
        });
        let read_started = std::time::Instant::now();
        plan.read_into(
            shard,
            [
                self.w1_host.as_mut_slice(),
                self.w1_scale_host.as_mut_slice(),
                self.w2_host.as_mut_slice(),
                self.w2_scale_host.as_mut_slice(),
                self.w3_host.as_mut_slice(),
                self.w3_scale_host.as_mut_slice(),
            ],
        )?;
        let read_ns = u64::try_from(read_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        if let Some(timeline) = timeline.as_mut() {
            timeline.read_end_ns = timeline_elapsed_ns(profile_origin.expect("timeline origin"));
        }
        self.weight_start.record(&self.stream)?;
        unsafe {
            self.w1_device
                .copy_from_pinned_host_async(&self.stream, &self.w1_host)?;
            self.w1_scale_device
                .copy_from_pinned_host_async(&self.stream, &self.w1_scale_host)?;
            self.w2_device
                .copy_from_pinned_host_async(&self.stream, &self.w2_host)?;
            self.w2_scale_device
                .copy_from_pinned_host_async(&self.stream, &self.w2_scale_host)?;
            self.w3_device
                .copy_from_pinned_host_async(&self.stream, &self.w3_host)?;
            self.w3_scale_device
                .copy_from_pinned_host_async(&self.stream, &self.w3_scale_host)?;
        }
        if let Some(timeline) = timeline.as_mut() {
            timeline.h2d_enqueued_ns =
                timeline_elapsed_ns(profile_origin.expect("timeline origin"));
        }
        self.weight_end.record(&self.stream)?;
        let kernel_ns = self.execute_into(
            input,
            route_weight,
            output,
            timeline.as_mut(),
            profile_origin,
        )?;
        self.weight_end.synchronize()?;
        let h2d_ns = (self.weight_start.elapsed_ms(&self.weight_end)? as f64 * 1_000_000.0) as u64;
        if let Some(timeline) = timeline.as_mut() {
            timeline.h2d_event_ns = h2d_ns;
            timeline.kernel_event_ns = kernel_ns;
        }
        Ok(RoutedLaneExecution {
            read_ns,
            h2d_ns,
            kernel_ns,
            timeline,
        })
    }

    pub(crate) fn execute(
        &mut self,
        input: &[u16],
        route_weight: f32,
    ) -> Result<(Vec<f32>, u64), Box<dyn Error>> {
        let mut output = vec![0.0_f32; self.spec.hidden];
        let gpu_ns = self.execute_into(input, route_weight, &mut output, None, None)?;
        Ok((output, gpu_ns))
    }

    pub(crate) fn execute_with_flash_trace(
        &mut self,
        input: &[u16],
        route_weight: f32,
    ) -> Result<(Vec<f32>, u64, FlashExpertTrace), Box<dyn Error>> {
        let (output, gpu_ns) = self.execute(input, route_weight)?;
        let ((input_fp8, input_scale), (w2_input_fp8, w2_input_scale)) = if self
            .fp4_accum_arm
            .uses_raw_fp8()
        {
            let input_fp8 = self
                .input_fp8
                .as_ref()
                .ok_or("Flash raw-FP8 arm lost its input FP8 buffer")?;
            let input_scale = self
                .input_fp8_scales
                .as_ref()
                .ok_or("Flash raw-FP8 arm lost its input FP8 scales")?;
            let w2_input_fp8 = self
                .routed_fp8
                .as_ref()
                .ok_or("Flash raw-FP8 arm lost its routed FP8 buffer")?;
            let w2_input_scale = self
                .routed_fp8_scales
                .as_ref()
                .ok_or("Flash raw-FP8 arm lost its routed FP8 scales")?;
            (
                (
                    input_fp8.to_host_vec(&self.stream)?,
                    input_scale.to_host_vec(&self.stream)?,
                ),
                (
                    w2_input_fp8.to_host_vec(&self.stream)?,
                    w2_input_scale.to_host_vec(&self.stream)?,
                ),
            )
        } else {
            (
                flash_parity_capture_activation_fp8(self.module, &self.stream, &self.input_device)?,
                flash_parity_capture_activation_fp8(self.module, &self.stream, &self.routed)?,
            )
        };
        Ok((
            output,
            gpu_ns,
            FlashExpertTrace {
                input_fp8,
                input_scale,
                w1: self.gate.to_host_vec(&self.stream)?,
                w3: self.up.to_host_vec(&self.stream)?,
                swiglu: self.routed.to_host_vec(&self.stream)?,
                w2_input_fp8,
                w2_input_scale,
                output: self.output_bf16.to_host_vec(&self.stream)?,
            },
        ))
    }

    pub(crate) fn execute_into(
        &mut self,
        input: &[u16],
        route_weight: f32,
        output: &mut [f32],
        mut timeline: Option<&mut RoutedLaneTimeline>,
        profile_origin: Option<std::time::Instant>,
    ) -> Result<u64, Box<dyn Error>> {
        if v4_expert::validate_input_for(self.spec, input, route_weight)? != 1 {
            return Err("resident expert slot accepts exactly one row".into());
        }
        if output.len() != self.spec.hidden {
            return Err("resident expert output requires one complete hidden row".into());
        }
        self.stream.context().bind_to_thread()?;
        self.input_host.as_mut_slice().copy_from_slice(input);
        // SAFETY: the input stager and all device buffers remain owned by this
        // slot until the reused completion event synchronizes below.
        unsafe {
            self.input_device
                .copy_from_pinned_host_async(&self.stream, &self.input_host)?;
        }
        if let (Some(timeline), Some(origin)) = (timeline.as_deref_mut(), profile_origin) {
            timeline.kernel_enqueued_ns = timeline_elapsed_ns(origin);
        }
        self.kernel_start.record(&self.stream)?;
        if self.fp4_accum_arm.uses_raw_fp8() {
            {
                let input_fp8 = self
                    .input_fp8
                    .as_mut()
                    .ok_or("Flash block32 arm lost its input FP8 buffer")?;
                let input_scales = self
                    .input_fp8_scales
                    .as_mut()
                    .ok_or("Flash block32 arm lost its input FP8 scales")?;
                launch_v4_activation_quantize_raw(
                    self.module,
                    &self.stream,
                    &self.input_device,
                    input_fp8,
                    input_scales,
                )?;
            }
            let input_fp8 = self
                .input_fp8
                .as_ref()
                .ok_or("Flash block32 arm lost its input FP8 buffer")?;
            let input_scales = self
                .input_fp8_scales
                .as_ref()
                .ok_or("Flash block32 arm lost its input FP8 scales")?;
            match self.fp4_accum_arm {
                FlashFp4AccumArm::Block32 => {
                    launch_v4_fp4_matvec_block32(
                        self.module,
                        &self.stream,
                        input_fp8,
                        input_scales,
                        &self.w1_device,
                        &self.w1_scale_device,
                        &mut self.gate,
                        1,
                        self.spec.moe_intermediate,
                        self.spec.hidden,
                    )?;
                    launch_v4_fp4_matvec_block32(
                        self.module,
                        &self.stream,
                        input_fp8,
                        input_scales,
                        &self.w3_device,
                        &self.w3_scale_device,
                        &mut self.up,
                        1,
                        self.spec.moe_intermediate,
                        self.spec.hidden,
                    )?;
                }
                FlashFp4AccumArm::TilelangRef => {
                    let reference = self
                        .tilelang_ref
                        .as_ref()
                        .ok_or("tilelang_ref arm lost its validated reference")?;
                    launch_v4_fp4_tilelang_ref(
                        reference,
                        &self.stream,
                        input_fp8,
                        input_scales,
                        &self.w1_device,
                        &self.w1_scale_device,
                        &mut self.gate,
                        1,
                        self.spec.moe_intermediate,
                        self.spec.hidden,
                    )?;
                    launch_v4_fp4_tilelang_ref(
                        reference,
                        &self.stream,
                        input_fp8,
                        input_scales,
                        &self.w3_device,
                        &self.w3_scale_device,
                        &mut self.up,
                        1,
                        self.spec.moe_intermediate,
                        self.spec.hidden,
                    )?;
                }
                FlashFp4AccumArm::Fp8Mma => {
                    launch_v4_fp4_matvec_mma(
                        self.module,
                        &self.stream,
                        input_fp8,
                        input_scales,
                        &self.w1_device,
                        &self.w1_scale_device,
                        &mut self.gate,
                        1,
                        self.spec.moe_intermediate,
                        self.spec.hidden,
                    )?;
                    launch_v4_fp4_matvec_mma(
                        self.module,
                        &self.stream,
                        input_fp8,
                        input_scales,
                        &self.w3_device,
                        &self.w3_scale_device,
                        &mut self.up,
                        1,
                        self.spec.moe_intermediate,
                        self.spec.hidden,
                    )?;
                }
                FlashFp4AccumArm::Current => {
                    return Err("current FP4 arm entered raw-FP8 branch".into());
                }
            }
        } else {
            launch_v4_activation_quantize(
                self.module,
                &self.stream,
                &self.input_device,
                &mut self.input_quantized,
            )?;
            launch_v4_fp4_matvec(
                self.module,
                &self.stream,
                &self.input_quantized,
                &self.w1_device,
                &self.w1_scale_device,
                &mut self.gate,
                1,
                self.spec.moe_intermediate,
                self.spec.hidden,
            )?;
            launch_v4_fp4_matvec(
                self.module,
                &self.stream,
                &self.input_quantized,
                &self.w3_device,
                &self.w3_scale_device,
                &mut self.up,
                1,
                self.spec.moe_intermediate,
                self.spec.hidden,
            )?;
        }
        launch_v4_swiglu_route(
            self.module,
            &self.stream,
            &self.gate,
            &self.up,
            &mut self.routed,
            route_weight,
        )?;
        if self.fp4_accum_arm.uses_raw_fp8() {
            {
                let routed_fp8 = self
                    .routed_fp8
                    .as_mut()
                    .ok_or("Flash block32 arm lost its routed FP8 buffer")?;
                let routed_scales = self
                    .routed_fp8_scales
                    .as_mut()
                    .ok_or("Flash block32 arm lost its routed FP8 scales")?;
                launch_v4_activation_quantize_raw(
                    self.module,
                    &self.stream,
                    &self.routed,
                    routed_fp8,
                    routed_scales,
                )?;
            }
            let routed_fp8 = self
                .routed_fp8
                .as_ref()
                .ok_or("Flash block32 arm lost its routed FP8 buffer")?;
            let routed_scales = self
                .routed_fp8_scales
                .as_ref()
                .ok_or("Flash block32 arm lost its routed FP8 scales")?;
            match self.fp4_accum_arm {
                FlashFp4AccumArm::Block32 => launch_v4_fp4_matvec_block32(
                    self.module,
                    &self.stream,
                    routed_fp8,
                    routed_scales,
                    &self.w2_device,
                    &self.w2_scale_device,
                    &mut self.output_bf16,
                    1,
                    self.spec.hidden,
                    self.spec.moe_intermediate,
                )?,
                FlashFp4AccumArm::TilelangRef => launch_v4_fp4_tilelang_ref(
                    self.tilelang_ref
                        .as_ref()
                        .ok_or("tilelang_ref arm lost its validated reference")?,
                    &self.stream,
                    routed_fp8,
                    routed_scales,
                    &self.w2_device,
                    &self.w2_scale_device,
                    &mut self.output_bf16,
                    1,
                    self.spec.hidden,
                    self.spec.moe_intermediate,
                )?,
                FlashFp4AccumArm::Fp8Mma => launch_v4_fp4_matvec_mma(
                    self.module,
                    &self.stream,
                    routed_fp8,
                    routed_scales,
                    &self.w2_device,
                    &self.w2_scale_device,
                    &mut self.output_bf16,
                    1,
                    self.spec.hidden,
                    self.spec.moe_intermediate,
                )?,
                FlashFp4AccumArm::Current => {
                    return Err("current FP4 arm entered raw-FP8 branch".into());
                }
            }
        } else {
            launch_v4_activation_quantize(
                self.module,
                &self.stream,
                &self.routed,
                &mut self.routed_quantized,
            )?;
            launch_v4_fp4_matvec(
                self.module,
                &self.stream,
                &self.routed_quantized,
                &self.w2_device,
                &self.w2_scale_device,
                &mut self.output_bf16,
                1,
                self.spec.hidden,
                self.spec.moe_intermediate,
            )?;
        }
        launch_v4_bf16_to_f32(
            self.module,
            &self.stream,
            &self.output_bf16,
            &mut self.output,
        )?;
        self.kernel_end.record(&self.stream)?;
        // SAFETY: output_host stays exclusively borrowed until `complete`
        // proves the D2H copy has finished.
        unsafe {
            self.output
                .copy_to_pinned_host_async(&self.stream, &mut self.output_host)?;
        }
        if let (Some(timeline), Some(origin)) = (timeline.as_deref_mut(), profile_origin) {
            timeline.d2h_enqueued_ns = timeline_elapsed_ns(origin);
        }
        self.complete.record(&self.stream)?;
        self.complete.synchronize()?;
        if let (Some(timeline), Some(origin)) = (timeline.as_deref_mut(), profile_origin) {
            timeline.complete_ns = timeline_elapsed_ns(origin);
        }
        let gpu_ns = (self.kernel_start.elapsed_ms(&self.kernel_end)? as f64 * 1_000_000.0) as u64;
        output.copy_from_slice(self.output_host.as_slice());
        Ok(gpu_ns)
    }

    pub(crate) fn copy_weight_tensors_from(
        &mut self,
        source: &Self,
        stream: &CudaStream,
    ) -> Result<(), Box<dyn Error>> {
        self.w1_device
            .copy_from_device_async(&source.w1_device, stream)?;
        self.w1_scale_device
            .copy_from_device_async(&source.w1_scale_device, stream)?;
        self.w2_device
            .copy_from_device_async(&source.w2_device, stream)?;
        self.w2_scale_device
            .copy_from_device_async(&source.w2_scale_device, stream)?;
        self.w3_device
            .copy_from_device_async(&source.w3_device, stream)?;
        self.w3_scale_device
            .copy_from_device_async(&source.w3_scale_device, stream)?;
        Ok(())
    }

    pub(crate) fn finish_qd4_loan(&self) -> Result<bool, Box<dyn Error>> {
        self.complete.record(&self.stream)?;
        self.complete.synchronize()?;
        Ok(true)
    }
}
