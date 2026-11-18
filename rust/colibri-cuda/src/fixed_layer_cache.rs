use super::*;

pub(crate) struct FixedLayerCache {
    pub(crate) plan: Vec<FixedLayerPlan>,
    pub(crate) device_selected: Vec<bool>,
    pub(crate) host_selected: Vec<bool>,
    pub(crate) decode_device_selected: Vec<bool>,
    pub(crate) decode_host_selected: Vec<bool>,
    pub(crate) device: Vec<Option<LoadedLayerRuntime>>,
    pub(crate) raw_device: Vec<Option<RawLoadedLayerRuntime>>,
    pub(crate) raw_host: Vec<Option<RawDecodedLayerHost>>,
    pub(crate) raw_scratch: Option<DeviceBuffer<u16>>,
    pub(crate) raw_device_generation: Vec<u64>,
    pub(crate) raw_host_generation: Vec<u64>,
    pub(crate) raw_scratch_generation: u64,
    pub(crate) raw_mode: bool,
    pub(crate) host: Vec<Option<DecodedLayerHost>>,
    pub(crate) device_budget_bytes: usize,
    pub(crate) host_budget_bytes: usize,
    pub(crate) decode_device_budget_bytes: usize,
    pub(crate) decode_host_budget_bytes: usize,
    pub(crate) device_resident_bytes: usize,
    pub(crate) host_resident_bytes: usize,
    pub(crate) decode_device_promotions: Vec<usize>,
    pub(crate) decode_host_promotions: Vec<usize>,
    pub(crate) device_hits: usize,
    pub(crate) host_hits: usize,
    pub(crate) misses: usize,
    pub(crate) stationary_model_dir: Option<std::path::PathBuf>,
    pub(crate) stationary_layout: Option<&'static str>,
    pub(crate) stationary_source_bytes: Vec<usize>,
    pub(crate) stationary_h2d_bytes: Vec<usize>,
    pub(crate) stationary_source_bytes_saved: usize,
    pub(crate) stationary_h2d_bytes_saved: usize,
}

impl FixedLayerCache {
    pub(crate) fn validate_stationary_raw_residency(
        &self,
        expected_device: &[bool],
        expected_host: &[bool],
        device_budget_bytes: usize,
        host_budget_bytes: usize,
    ) -> Result<(), Box<dyn Error>> {
        if expected_device.len() != self.plan.len()
            || expected_host.len() != self.plan.len()
            || self.device_selected.len() != self.plan.len()
            || self.host_selected.len() != self.plan.len()
            || self
                .plan
                .iter()
                .map(|plan| plan.layer)
                .collect::<BTreeSet<_>>()
                .len()
                != self.plan.len()
            || expected_device
                .iter()
                .zip(expected_host)
                .any(|(&device, &host)| device && host)
            || self.device_selected != expected_device
            || self.host_selected != expected_host
        {
            return Err("stationary raw residency selection did not close".into());
        }
        let device_bytes = self
            .raw_device
            .iter()
            .flatten()
            .try_fold(0_usize, |total, raw| total.checked_add(raw.resident_bytes))
            .ok_or("decode overlay device resident byte overflow")?;
        let host_bytes = self
            .raw_host
            .iter()
            .flatten()
            .try_fold(0_usize, |total, raw| total.checked_add(raw.resident_bytes))
            .ok_or("stationary raw host resident byte overflow")?;
        if self.plan.iter().enumerate().any(|(index, plan)| {
            self.raw_device
                .get(plan.layer)
                .is_none_or(|raw| raw.is_some() != expected_device[index])
                || self
                    .raw_host
                    .get(plan.layer)
                    .is_none_or(|raw| raw.is_some() != expected_host[index])
        }) || device_bytes != self.device_resident_bytes
            || host_bytes != self.host_resident_bytes
            || device_bytes > device_budget_bytes
            || host_bytes > host_budget_bytes
        {
            return Err("stationary raw residency ownership/accounting did not close".into());
        }
        Ok(())
    }

    pub(crate) fn validate_prefill_residency_complete(&self) -> Result<(), Box<dyn Error>> {
        if !self.decode_device_promotions.is_empty() || !self.decode_host_promotions.is_empty() {
            return Err("stationary raw prefill residency unexpectedly contains promotions".into());
        }
        let (device_selected, host_selected) = stationary_raw_prefill_selection(
            &self.plan,
            self.device_budget_bytes,
            self.host_budget_bytes,
        );
        self.validate_stationary_raw_residency(
            &device_selected,
            &host_selected,
            self.device_budget_bytes,
            self.host_budget_bytes,
        )
    }

    pub(crate) fn validate_decode_overlay_complete(&self) -> Result<(), Box<dyn Error>> {
        if self
            .decode_device_promotions
            .iter()
            .copied()
            .collect::<BTreeSet<_>>()
            .len()
            != self.decode_device_promotions.len()
            || self
                .decode_host_promotions
                .iter()
                .copied()
                .collect::<BTreeSet<_>>()
                .len()
                != self.decode_host_promotions.len()
        {
            return Err("stationary raw decode overlay promotions did not close".into());
        }
        self.validate_stationary_raw_residency(
            &self.decode_device_selected,
            &self.decode_host_selected,
            self.decode_device_budget_bytes,
            self.decode_host_budget_bytes,
        )
    }

    pub(crate) fn restore_stationary_raw(
        &mut self,
        mut lease: StationaryRawLease,
    ) -> Result<(), Box<dyn Error>> {
        if lease.device_generation.is_some_and(|generation| {
            self.raw_device[lease.layer].is_some()
                || self.raw_device_generation[lease.layer] != generation
        }) || lease.host_generation.is_some_and(|generation| {
            self.raw_host[lease.layer].is_some()
                || self.raw_host_generation[lease.layer] != generation
        }) || lease.scratch_generation.is_some_and(|generation| {
            self.raw_scratch.is_some() || self.raw_scratch_generation != generation
        }) {
            return Err("stationary raw lease rejected an ABA replacement".into());
        }
        let index = self
            .plan
            .iter()
            .position(|plan| plan.layer == lease.layer)
            .ok_or("stationary raw restore layer is outside the retention plan")?;
        let promote_device = self.decode_device_selected[index] && !self.device_selected[index];
        let promote_host = self.decode_host_selected[index] && !self.host_selected[index];
        if promote_device && promote_host {
            return Err("decode overlay selected one layer for two tiers".into());
        }
        if lease.decoded_wo_a_in_scratch {
            std::mem::swap(
                &mut lease
                    .runtime
                    .as_mut()
                    .expect("stationary raw lease lost its runtime")
                    .weights
                    .wo_a,
                lease
                    .scratch
                    .as_mut()
                    .expect("stationary raw lease lost its decode scratch"),
            );
        }
        if lease.scratch_generation.is_some() {
            let _ = self.raw_scratch.insert(
                lease
                    .scratch
                    .take()
                    .expect("stationary raw lease lost its return scratch"),
            );
        }
        if promote_device {
            if lease.device_generation.is_some()
                || self.raw_device[lease.layer].is_some()
                || self.raw_host[lease.layer].is_some()
            {
                return Err("decode overlay device promotion ownership drift".into());
            }
            let old_host = match (
                self.host_selected[index],
                lease.host_generation,
                lease.raw_host.take(),
            ) {
                (true, Some(_), Some(host)) => Some(host),
                (false, None, None) => None,
                _ => return Err("decode overlay device promotion ownership drift".into()),
            };
            let runtime = lease
                .runtime
                .take()
                .ok_or("decode overlay device promotion lost its runtime")?;
            let wo_a_raw = lease
                .raw_device
                .take()
                .ok_or("decode overlay device promotion lost its existing raw wo_a")?;
            let resident_bytes = runtime
                .weights
                .num_bytes()
                .checked_add(wo_a_raw.num_bytes())
                .ok_or("decode overlay device resident byte overflow")?;
            if runtime.weights.wo_a.len() != 0
                || resident_bytes != self.plan[index].raw_device_bytes
            {
                return Err("decode overlay device promotion violated its target budget".into());
            }
            (self.device_resident_bytes, self.host_resident_bytes) =
                decode_device_promotion_residency(
                    self.device_resident_bytes,
                    self.host_resident_bytes,
                    resident_bytes,
                    old_host.as_ref().map(|host| host.resident_bytes),
                    self.decode_device_budget_bytes,
                )?;
            self.device_selected[index] = true;
            self.host_selected[index] = false;
            self.raw_device[lease.layer] = Some(RawLoadedLayerRuntime {
                runtime,
                wo_a_raw,
                resident_bytes,
            });
            self.decode_device_promotions.push(lease.layer);
            return Ok(());
        }
        if promote_host {
            if lease.device_generation.is_some()
                || lease.host_generation.is_some()
                || lease.raw_host.is_none()
                || self.raw_device[lease.layer].is_some()
                || self.raw_host[lease.layer].is_some()
            {
                return Err("decode overlay miss-to-host ownership drift".into());
            }
            let raw = lease
                .raw_host
                .take()
                .ok_or("decode overlay host promotion lost its existing decoded payload")?;
            if raw.resident_bytes != self.plan[index].raw_host_bytes {
                return Err("decode overlay host promotion violated its target budget".into());
            }
            drop(lease.runtime.take());
            drop(lease.raw_device.take());
            self.host_resident_bytes = decode_host_promotion_residency(
                self.host_resident_bytes,
                raw.resident_bytes,
                self.decode_host_budget_bytes,
            )?;
            self.host_selected[index] = true;
            self.raw_host[lease.layer] = Some(raw);
            self.decode_host_promotions.push(lease.layer);
            return Ok(());
        }
        if lease.device_generation.is_some() {
            let _ = self.raw_device[lease.layer].insert(RawLoadedLayerRuntime {
                runtime: lease
                    .runtime
                    .take()
                    .expect("stationary raw device lease lost its runtime"),
                wo_a_raw: lease
                    .raw_device
                    .take()
                    .expect("stationary raw device lease lost its raw payload"),
                resident_bytes: lease.resident_bytes,
            });
        } else if lease.host_generation.is_some() {
            drop(lease.runtime.take());
            let _ = self.raw_host[lease.layer].insert(
                lease
                    .raw_host
                    .take()
                    .expect("stationary raw host lease lost its raw payload"),
            );
        }
        Ok(())
    }
}

impl FixedLayerCache {
    pub(crate) fn stationary_raw_from_env(
        model_dir: &std::path::Path,
        layers: &[usize],
        device_reserve_bytes: usize,
        host_reserve_bytes: usize,
        decode_device_release_bytes: usize,
        decode_host_release_bytes: usize,
        stream: &CudaStream,
    ) -> Result<Self, Box<dyn Error>> {
        let requested_device = parse_gib_budget("COLIBRI_V4_STATIONARY_RAW_DEVICE_RETENTION_GIB")?;
        let requested_host = parse_gib_budget("COLIBRI_V4_STATIONARY_RAW_HOST_RETENTION_GIB")?;
        let pinned_pipeline = optional_one_flag("COLIBRI_V4_PINNED_FIXED_PIPELINE")?;
        let effective_requested_device = if pinned_pipeline {
            requested_device
                .checked_sub(PINNED_FIXED_PIPELINE_WORKER_DEVICE_BYTES)
                .ok_or("pinned fixed pipeline device budget cannot hold its next layer and second decode scratch")?
        } else {
            requested_device
        };
        let device_budget_bytes = preflight_cache_budget(
            effective_requested_device,
            device_available_bytes()?,
            device_reserve_bytes.saturating_add(if pinned_pipeline {
                PINNED_FIXED_PIPELINE_WORKER_DEVICE_BYTES
            } else {
                RAW_FIXED_SCRATCH_BYTES
            }),
            "stationary_raw_device",
        );
        let host_budget_bytes = preflight_cache_budget(
            requested_host,
            host_available_bytes()?,
            FIXED_PREFETCH_HOST_HEADROOM_BYTES.saturating_add(host_reserve_bytes),
            "stationary_raw_host",
        );
        let plan = if device_budget_bytes == 0 && host_budget_bytes == 0 {
            Vec::new()
        } else {
            stationary_raw_layer_plans(model_dir, layers)?
        };
        let none_selected = vec![false; plan.len()];
        let (device_selected, host_selected) =
            stationary_raw_prefill_selection(&plan, device_budget_bytes, host_budget_bytes);
        let decode_device_budget_bytes = device_budget_bytes
            .checked_add(decode_device_release_bytes)
            .ok_or("stationary raw decode overlay device budget overflow")?;
        let decode_host_budget_bytes = host_budget_bytes
            .checked_add(decode_host_release_bytes)
            .ok_or("stationary raw decode overlay host budget overflow")?;
        let explicit_decode_plan = std::env::var_os(DECODE_PROMOTION_PLAN_ENV);
        let (decode_device_selected, decode_host_selected) = match parse_decode_promotion_plan(
            explicit_decode_plan.as_deref(),
            &plan,
            &device_selected,
            &host_selected,
            device_budget_bytes,
            host_budget_bytes,
            decode_device_budget_bytes,
            decode_host_budget_bytes,
        )? {
            Some(targets) => targets,
            None => {
                let device = select_fixed_cache_layers(
                    &plan,
                    decode_device_budget_bytes,
                    &none_selected,
                    true,
                    true,
                );
                let host = select_fixed_cache_layers(
                    &plan,
                    decode_host_budget_bytes,
                    &device,
                    false,
                    true,
                );
                (device, host)
            }
        };
        if device_selected
            .iter()
            .zip(&decode_device_selected)
            .any(|(&prefill, &decode)| prefill && !decode)
            || host_selected
                .iter()
                .zip(&decode_device_selected)
                .zip(&decode_host_selected)
                .any(|((&prefill, &device), &host)| prefill && !device && !host)
            || decode_host_selected
                .iter()
                .zip(&host_selected)
                .zip(&device_selected)
                .any(|((&decode, &prefill_host), &prefill_device)| {
                    decode && !prefill_host && prefill_device
                })
        {
            return Err("stationary raw decode overlay requires monotone tier promotions".into());
        }
        let raw_scratch = (device_selected.iter().any(|&selected| selected)
            || host_selected.iter().any(|&selected| selected))
        .then(|| {
            DeviceBuffer::<u16>::zeroed(
                stream,
                RAW_FIXED_SCRATCH_BYTES / std::mem::size_of::<u16>(),
            )
        })
        .transpose()?;
        Ok(Self {
            plan,
            device_selected,
            host_selected,
            decode_device_selected,
            decode_host_selected,
            device: (0..v4_state::V4_LAYER_COUNT).map(|_| None).collect(),
            raw_device: (0..v4_state::V4_LAYER_COUNT).map(|_| None).collect(),
            raw_host: (0..v4_state::V4_LAYER_COUNT).map(|_| None).collect(),
            raw_scratch,
            raw_device_generation: vec![0; v4_state::V4_LAYER_COUNT],
            raw_host_generation: vec![0; v4_state::V4_LAYER_COUNT],
            raw_scratch_generation: 0,
            raw_mode: true,
            host: (0..v4_state::V4_LAYER_COUNT).map(|_| None).collect(),
            device_budget_bytes,
            host_budget_bytes,
            decode_device_budget_bytes,
            decode_host_budget_bytes,
            device_resident_bytes: 0,
            host_resident_bytes: 0,
            decode_device_promotions: Vec::new(),
            decode_host_promotions: Vec::new(),
            device_hits: 0,
            host_hits: 0,
            misses: 0,
            stationary_model_dir: Some(model_dir.to_owned()),
            stationary_layout: Some(STATIONARY_RAW_LAYOUT),
            stationary_source_bytes: vec![0; v4_state::V4_LAYER_COUNT],
            stationary_h2d_bytes: vec![0; v4_state::V4_LAYER_COUNT],
            stationary_source_bytes_saved: 0,
            stationary_h2d_bytes_saved: 0,
        })
    }

    pub(crate) fn validate_stationary_key(
        &self,
        model_dir: &std::path::Path,
    ) -> Result<(), Box<dyn Error>> {
        if self.stationary_model_dir.as_deref() != Some(model_dir)
            || self.stationary_layout != Some(STATIONARY_RAW_LAYOUT)
        {
            return Err("stationary raw cache model/layout key mismatch".into());
        }
        Ok(())
    }

    pub(crate) fn from_env(
        model_dir: &std::path::Path,
        executed_layers: usize,
        device_reserve_bytes: usize,
        host_reserve_bytes: usize,
        stream: &CudaStream,
    ) -> Result<Self, Box<dyn Error>> {
        const HOST_HEADROOM: usize = 4 * 1024 * 1024 * 1024;
        let requested_device = parse_gib_budget("COLIBRI_V4_DEVICE_FIXED_CACHE_GIB")?;
        let requested_host = parse_gib_budget("COLIBRI_V4_HOST_FIXED_CACHE_GIB")?;
        let raw_mode = match std::env::var_os("COLIBRI_V4_RAW_FIXED_CACHE") {
            None => false,
            Some(value) if value == "1" => true,
            Some(_) => return Err("COLIBRI_V4_RAW_FIXED_CACHE must be exactly 1 when set".into()),
        };
        let device_budget_bytes = preflight_cache_budget(
            requested_device,
            device_available_bytes()?,
            V4_DEVICE_CAPACITY_HEADROOM_BYTES
                .saturating_add(device_reserve_bytes)
                .saturating_add(usize::from(raw_mode) * RAW_FIXED_SCRATCH_BYTES),
            "device",
        );
        let host_budget_bytes = preflight_cache_budget(
            requested_host,
            host_available_bytes()?,
            HOST_HEADROOM.saturating_add(host_reserve_bytes),
            "host",
        );
        let baseline_path =
            std::env::var_os("COLIBRI_V4_FIXED_CACHE_BASELINE_JSONL").map(std::path::PathBuf::from);
        if baseline_path.is_none() && device_budget_bytes == 0 && host_budget_bytes == 0 {
            return Ok(Self {
                device: (0..v4_state::V4_LAYER_COUNT).map(|_| None).collect(),
                raw_device: (0..v4_state::V4_LAYER_COUNT).map(|_| None).collect(),
                raw_host: (0..v4_state::V4_LAYER_COUNT).map(|_| None).collect(),
                raw_scratch: None,
                raw_device_generation: vec![0; v4_state::V4_LAYER_COUNT],
                raw_host_generation: vec![0; v4_state::V4_LAYER_COUNT],
                raw_scratch_generation: 0,
                raw_mode,
                host: (0..v4_state::V4_LAYER_COUNT).map(|_| None).collect(),
                plan: Vec::new(),
                device_selected: Vec::new(),
                host_selected: Vec::new(),
                decode_device_selected: Vec::new(),
                decode_host_selected: Vec::new(),
                device_budget_bytes,
                host_budget_bytes,
                decode_device_budget_bytes: device_budget_bytes,
                decode_host_budget_bytes: host_budget_bytes,
                device_resident_bytes: 0,
                host_resident_bytes: 0,
                decode_device_promotions: Vec::new(),
                decode_host_promotions: Vec::new(),
                device_hits: 0,
                host_hits: 0,
                misses: 0,
                stationary_model_dir: None,
                stationary_layout: None,
                stationary_source_bytes: vec![0; v4_state::V4_LAYER_COUNT],
                stationary_h2d_bytes: vec![0; v4_state::V4_LAYER_COUNT],
                stationary_source_bytes_saved: 0,
                stationary_h2d_bytes_saved: 0,
            });
        }
        let candidate_layers = match std::env::var("COLIBRI_V4_FIXED_CACHE_CANDIDATE_LAYERS") {
            Ok(value) => value
                .split(',')
                .map(str::trim)
                .map(str::parse::<usize>)
                .collect::<Result<Vec<_>, _>>()?,
            Err(std::env::VarError::NotPresent) => (0..executed_layers).collect(),
            Err(error) => return Err(error.into()),
        };
        if candidate_layers.is_empty()
            || candidate_layers
                .iter()
                .any(|&layer| layer >= executed_layers)
        {
            return Err("fixed cache candidate layers must be non-empty executed layers".into());
        }
        let baseline_path = baseline_path
            .ok_or("set COLIBRI_V4_FIXED_CACHE_BASELINE_JSONL for fixed cache planning")?;
        let plan = fixed_layer_plans(model_dir, &candidate_layers, &baseline_path)?;
        let none_selected = vec![false; plan.len()];
        let device_selected =
            select_fixed_cache_layers(&plan, device_budget_bytes, &none_selected, true, raw_mode);
        let host_selected =
            select_fixed_cache_layers(&plan, host_budget_bytes, &device_selected, false, raw_mode);
        let raw_scratch = if raw_mode
            && (device_selected.iter().any(|&selected| selected)
                || host_selected.iter().any(|&selected| selected))
        {
            Some(DeviceBuffer::<u16>::zeroed(
                stream,
                RAW_FIXED_SCRATCH_BYTES / std::mem::size_of::<u16>(),
            )?)
        } else {
            None
        };
        Ok(Self {
            device: (0..v4_state::V4_LAYER_COUNT).map(|_| None).collect(),
            raw_device: (0..v4_state::V4_LAYER_COUNT).map(|_| None).collect(),
            raw_host: (0..v4_state::V4_LAYER_COUNT).map(|_| None).collect(),
            raw_scratch,
            raw_device_generation: vec![0; v4_state::V4_LAYER_COUNT],
            raw_host_generation: vec![0; v4_state::V4_LAYER_COUNT],
            raw_scratch_generation: 0,
            raw_mode,
            host: (0..v4_state::V4_LAYER_COUNT).map(|_| None).collect(),
            plan,
            decode_device_selected: device_selected.clone(),
            decode_host_selected: host_selected.clone(),
            device_selected,
            host_selected,
            device_budget_bytes,
            host_budget_bytes,
            decode_device_budget_bytes: device_budget_bytes,
            decode_host_budget_bytes: host_budget_bytes,
            device_resident_bytes: 0,
            host_resident_bytes: 0,
            decode_device_promotions: Vec::new(),
            decode_host_promotions: Vec::new(),
            device_hits: 0,
            host_hits: 0,
            misses: 0,
            stationary_model_dir: None,
            stationary_layout: None,
            stationary_source_bytes: vec![0; v4_state::V4_LAYER_COUNT],
            stationary_h2d_bytes: vec![0; v4_state::V4_LAYER_COUNT],
            stationary_source_bytes_saved: 0,
            stationary_h2d_bytes_saved: 0,
        })
    }

    pub(crate) fn selected(&self, layer: usize) -> (bool, bool, &FixedLayerPlan) {
        let index = self
            .plan
            .iter()
            .position(|candidate| candidate.layer == layer)
            .expect("cache access layer must be in the startup plan");
        (
            self.device_selected[index],
            self.host_selected[index],
            &self.plan[index],
        )
    }

    pub(crate) fn record_stationary_saved_bytes(
        &mut self,
        layer: usize,
        target_device: bool,
        target_host: bool,
    ) -> Result<(), Box<dyn Error>> {
        let (source_bytes, h2d_bytes) = fixed_saved_bytes_for_target(
            self.stationary_source_bytes[layer],
            self.stationary_h2d_bytes[layer],
            target_device,
            target_host,
        )?;
        let source_bytes_saved = self
            .stationary_source_bytes_saved
            .checked_add(source_bytes)
            .ok_or("stationary source-saved byte overflow")?;
        let h2d_bytes_saved = self
            .stationary_h2d_bytes_saved
            .checked_add(h2d_bytes)
            .ok_or("stationary H2D-saved byte overflow")?;
        self.stationary_source_bytes_saved = source_bytes_saved;
        self.stationary_h2d_bytes_saved = h2d_bytes_saved;
        Ok(())
    }

    pub(crate) fn capture_stationary_raw(
        &mut self,
        model_dir: &std::path::Path,
        layer: usize,
        prompt_rows: usize,
        mut host: DecodedLayerHost,
        mut runtime: LoadedLayerRuntime,
        wo_a_raw: DeviceBuffer<u8>,
        telemetry: &FixedLoadTelemetry,
        stream: &CudaStream,
    ) -> Result<(), Box<dyn Error>> {
        self.validate_stationary_key(model_dir)?;
        let Some(index) = self
            .plan
            .iter()
            .position(|candidate| candidate.layer == layer)
        else {
            return Ok(());
        };
        let device_selected = self.device_selected[index];
        let host_selected = self.host_selected[index];
        let raw_device_bytes = self.plan[index].raw_device_bytes;
        let raw_host_bytes = self.plan[index].raw_host_bytes;
        let host_raw_len = host
            .tail
            .wo_a_raw
            .as_ref()
            .ok_or("stationary capture lost its prefetched raw host payload")?
            .len();
        if runtime.weights.wo_a.len() != host_raw_len || wo_a_raw.len() != host_raw_len {
            return Err(format!("layer {layer} stationary raw wo_a shape mismatch").into());
        }
        self.stationary_source_bytes[layer] = telemetry.fixed_source_bytes;
        self.stationary_h2d_bytes[layer] = telemetry.fixed_h2d_bytes;
        if device_selected {
            let decoded_wo_a =
                std::mem::replace(&mut runtime.weights.wo_a, DeviceBuffer::zeroed(stream, 0)?);
            drop(decoded_wo_a);
            let resident_bytes = runtime
                .weights
                .num_bytes()
                .checked_add(wo_a_raw.num_bytes())
                .ok_or("stationary raw device resident byte overflow")?;
            if resident_bytes != raw_device_bytes
                || self
                    .device_resident_bytes
                    .checked_add(resident_bytes)
                    .is_none_or(|bytes| bytes > self.device_budget_bytes)
                || self.raw_device[layer].is_some()
            {
                return Err(format!(
                    "layer {layer} stationary raw device capture violates its startup plan"
                )
                .into());
            }
            self.device_resident_bytes += resident_bytes;
            self.raw_device[layer] = Some(RawLoadedLayerRuntime {
                runtime,
                wo_a_raw,
                resident_bytes,
            });
        } else if host_selected {
            drop_request_hash_ids_before_fixed_retention(&mut host.state.router, prompt_rows)?;
            let raw = host
                .tail
                .wo_a_raw
                .take()
                .ok_or("stationary raw host capture lost wo_a")?;
            let resident_bytes = host
                .num_bytes()
                .checked_add(raw.len())
                .ok_or("stationary raw host resident byte overflow")?;
            if resident_bytes != raw_host_bytes
                || self
                    .host_resident_bytes
                    .checked_add(resident_bytes)
                    .is_none_or(|bytes| bytes > self.host_budget_bytes)
                || self.raw_host[layer].is_some()
            {
                return Err(format!(
                    "layer {layer} stationary raw host capture violates its startup plan"
                )
                .into());
            }
            self.host_resident_bytes += resident_bytes;
            self.raw_host[layer] = Some(RawDecodedLayerHost {
                host,
                wo_a_raw: raw,
                resident_bytes,
            });
        }
        Ok(())
    }

    pub(crate) fn prepare_stationary_raw_pinned(
        &mut self,
        model_dir: &std::path::Path,
        layer: usize,
        token_id: i64,
        module: &kernels::LoadedModule,
        transfer: &std::sync::Arc<CudaStream>,
        mut slot: PinnedFixedSlot,
    ) -> Result<PinnedFixedPrepared, Box<dyn Error>> {
        self.validate_stationary_key(model_dir)?;
        slot.generation = slot
            .generation
            .checked_add(1)
            .ok_or("pinned fixed slot generation overflow")?;
        let generation = slot.generation;
        let candidate = self.plan.iter().position(|plan| plan.layer == layer);
        let (
            device_selected,
            host_selected,
            decode_device_selected,
            decode_host_selected,
            expected_bytes,
        ) = candidate
            .map(|index| {
                (
                    self.device_selected[index],
                    self.host_selected[index],
                    self.decode_device_selected[index],
                    self.decode_host_selected[index],
                    self.plan[index].raw_device_bytes,
                )
            })
            .ok_or("pinned fixed layer is outside the retention plan")?;
        if device_selected && self.raw_device[layer].is_some() {
            self.device_hits += 1;
            self.record_stationary_saved_bytes(layer, true, false)?;
            let cached = self.raw_device[layer]
                .take()
                .ok_or("pinned fixed device cache entry disappeared")?;
            self.raw_device_generation[layer] = self.raw_device_generation[layer]
                .checked_add(1)
                .ok_or("pinned fixed device generation overflow")?;
            let RawLoadedLayerRuntime {
                mut runtime,
                wo_a_raw,
                resident_bytes,
            } = cached;
            slot.h2d_start.record(transfer)?;
            slot.h2d_end.record(transfer)?;
            let mut wo_a = slot
                .scratch
                .replace(unsafe { DeviceBuffer::uninitialized_async(transfer, 0)? })
                .ok_or("pinned fixed device hit lost its scratch")?;
            launch_e4m3fn_to_bf16(module, transfer, &wo_a_raw, &mut wo_a)?;
            std::mem::swap(&mut runtime.weights.wo_a, &mut wo_a);
            slot.ready.record(transfer)?;
            return Ok(PinnedFixedPrepared {
                slot,
                generation,
                runtime_done_recorded: false,
                lease: StationaryRawLease {
                    layer,
                    device_generation: Some(self.raw_device_generation[layer]),
                    host_generation: None,
                    scratch_generation: None,
                    runtime: Some(runtime),
                    raw_device: Some(wo_a_raw),
                    raw_host: None,
                    scratch: None,
                    resident_bytes,
                    decoded_wo_a_in_scratch: false,
                },
                telemetry: FixedLoadTelemetry {
                    cache_status: "stationary_pinned_device_hit",
                    ..FixedLoadTelemetry::default()
                },
            });
        }
        let mut telemetry;
        let runtime;
        let raw_wo_a;
        let copied;
        let mut overlay_raw_host = None;
        let mut overlay_host_generation = None;
        if host_selected && self.raw_host[layer].is_some() {
            self.host_hits += 1;
            self.record_stationary_saved_bytes(layer, false, true)?;
            let mut cached = self.raw_host[layer]
                .take()
                .ok_or("pinned fixed host cache entry disappeared")?;
            self.raw_host_generation[layer] = self.raw_host_generation[layer]
                .checked_add(1)
                .ok_or("pinned fixed host generation overflow")?;
            let host_generation = self.raw_host_generation[layer];
            if cached.host.tail.wo_a_raw.is_some() || cached.wo_a_raw.is_empty() {
                return Err("pinned fixed host cache raw payload ownership drift".into());
            }
            cached.host.tail.wo_a_raw = Some(std::mem::take(&mut cached.wo_a_raw));
            let uploaded = upload_pinned_layer_runtime_raw(
                model_dir,
                layer,
                &cached.host,
                module,
                transfer,
                &mut slot,
            );
            cached.wo_a_raw = cached
                .host
                .tail
                .wo_a_raw
                .take()
                .ok_or("pinned fixed host cache lost its raw payload")?;
            (runtime, raw_wo_a, copied) = uploaded?;
            if self.raw_host[layer].is_some() || self.raw_host_generation[layer] != host_generation
            {
                return Err("pinned fixed host restore rejected an ABA replacement".into());
            }
            if decode_device_selected && !device_selected {
                overlay_host_generation = Some(host_generation);
                overlay_raw_host = Some(cached);
            } else {
                let _ = self.raw_host[layer].insert(cached);
            }
            telemetry = FixedLoadTelemetry {
                cache_status: "stationary_pinned_host_hit",
                ..FixedLoadTelemetry::default()
            };
        } else {
            self.misses += 1;
            let (mut host, loaded) = load_layer_host_raw_wo_a(model_dir, layer, &[token_id])?;
            telemetry = loaded;
            (runtime, raw_wo_a, copied) = upload_pinned_layer_runtime_raw(
                model_dir, layer, &host, module, transfer, &mut slot,
            )?;
            if decode_host_selected && !host_selected {
                drop_request_hash_ids_before_fixed_retention(&mut host.state.router, 1)?;
                let raw = host
                    .tail
                    .wo_a_raw
                    .take()
                    .ok_or("decode overlay host promotion lost its existing raw wo_a")?;
                let resident_bytes = host
                    .num_bytes()
                    .checked_add(raw.len())
                    .ok_or("decode overlay host resident byte overflow")?;
                let expected =
                    self.plan[candidate.expect("retention candidate disappeared")].raw_host_bytes;
                if resident_bytes != expected {
                    return Err(format!(
                        "layer {layer} decode overlay host bytes {resident_bytes} != plan {expected}"
                    )
                    .into());
                }
                overlay_raw_host = Some(RawDecodedLayerHost {
                    host,
                    wo_a_raw: raw,
                    resident_bytes,
                });
            }
            telemetry.cache_status = "stationary_pinned_miss";
        }
        if copied != expected_bytes || copied > PHASE1_CSA_PAYLOAD_BYTES {
            return Err(format!(
                "pinned fixed layer {layer} copied {copied} bytes, expected {expected_bytes}"
            )
            .into());
        }
        telemetry.fixed_h2d_bytes = copied;
        telemetry.raw_h2d_bytes = raw_wo_a.num_bytes();
        Ok(PinnedFixedPrepared {
            slot,
            generation,
            runtime_done_recorded: false,
            lease: StationaryRawLease {
                layer,
                device_generation: None,
                host_generation: overlay_host_generation,
                scratch_generation: None,
                runtime: Some(runtime),
                // Keep the async upload source alive until runtime_done closes
                // the transfer-stream happens-before chain.
                raw_device: Some(raw_wo_a),
                raw_host: overlay_raw_host,
                scratch: None,
                resident_bytes: 0,
                decoded_wo_a_in_scratch: false,
            },
            telemetry,
        })
    }

    pub(crate) fn access_stationary_raw(
        &mut self,
        model_dir: &std::path::Path,
        layer: usize,
        token_ids: &[i64],
        module: &kernels::LoadedModule,
        stream: &CudaStream,
    ) -> Result<(StationaryRawLease, FixedLoadTelemetry), Box<dyn Error>> {
        self.validate_stationary_key(model_dir)?;
        let candidate_index = self
            .plan
            .iter()
            .position(|candidate| candidate.layer == layer);
        let (device_selected, host_selected) = candidate_index
            .map(|index| (self.device_selected[index], self.host_selected[index]))
            .unwrap_or((false, false));
        if device_selected && self.raw_device[layer].is_some() {
            self.device_hits += 1;
            self.record_stationary_saved_bytes(layer, true, false)?;
            let cached = self.raw_device[layer]
                .take()
                .ok_or("stationary raw device cache entry disappeared")?;
            self.raw_device_generation[layer] = self.raw_device_generation[layer]
                .checked_add(1)
                .ok_or("stationary raw device generation overflow")?;
            let device_generation = self.raw_device_generation[layer];
            let mut scratch = self
                .raw_scratch
                .take()
                .ok_or("stationary raw device scratch disappeared")?;
            self.raw_scratch_generation = self
                .raw_scratch_generation
                .checked_add(1)
                .ok_or("stationary raw scratch generation overflow")?;
            let RawLoadedLayerRuntime {
                mut runtime,
                wo_a_raw,
                resident_bytes,
            } = cached;
            let decode_started = std::time::Instant::now();
            launch_e4m3fn_to_bf16(module, stream, &wo_a_raw, &mut scratch)?;
            stream.synchronize()?;
            let fixed_decode_ns =
                u64::try_from(decode_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
            std::mem::swap(&mut runtime.weights.wo_a, &mut scratch);
            return Ok((
                StationaryRawLease {
                    layer,
                    device_generation: Some(device_generation),
                    host_generation: None,
                    scratch_generation: Some(self.raw_scratch_generation),
                    runtime: Some(runtime),
                    raw_device: Some(wo_a_raw),
                    raw_host: None,
                    scratch: Some(scratch),
                    resident_bytes,
                    decoded_wo_a_in_scratch: true,
                },
                FixedLoadTelemetry {
                    fixed_decode_ns,
                    cache_status: "stationary_raw_device_hit",
                    ..FixedLoadTelemetry::default()
                },
            ));
        }
        if host_selected && self.raw_host[layer].is_some() {
            self.host_hits += 1;
            self.record_stationary_saved_bytes(layer, false, true)?;
            let cached = self.raw_host[layer]
                .take()
                .ok_or("stationary raw host cache entry disappeared")?;
            self.raw_host_generation[layer] = self.raw_host_generation[layer]
                .checked_add(1)
                .ok_or("stationary raw host generation overflow")?;
            let host_generation = self.raw_host_generation[layer];
            let mut scratch = self
                .raw_scratch
                .take()
                .ok_or("stationary raw host scratch disappeared")?;
            self.raw_scratch_generation = self
                .raw_scratch_generation
                .checked_add(1)
                .ok_or("stationary raw scratch generation overflow")?;
            let h2d_started = std::time::Instant::now();
            let mut runtime = upload_layer_runtime(model_dir, layer, &cached.host, stream)?;
            stream.synchronize()?;
            let fixed_without_raw_bytes = runtime.weights.num_bytes();
            let raw_h2d_started = std::time::Instant::now();
            let wo_a_raw = DeviceBuffer::from_host(stream, &cached.wo_a_raw)?;
            stream.synchronize()?;
            let raw_h2d_ns =
                u64::try_from(raw_h2d_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
            let raw_h2d_bytes = wo_a_raw.num_bytes();
            let fixed_h2d_ns = u64::try_from(h2d_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
            let decode_started = std::time::Instant::now();
            launch_e4m3fn_to_bf16(module, stream, &wo_a_raw, &mut scratch)?;
            stream.synchronize()?;
            let fixed_decode_ns =
                u64::try_from(decode_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
            std::mem::swap(&mut runtime.weights.wo_a, &mut scratch);
            let fixed_h2d_bytes = fixed_without_raw_bytes
                .checked_add(raw_h2d_bytes)
                .ok_or("stationary raw host H2D byte overflow")?;
            return Ok((
                StationaryRawLease {
                    layer,
                    device_generation: None,
                    host_generation: Some(host_generation),
                    scratch_generation: Some(self.raw_scratch_generation),
                    runtime: Some(runtime),
                    raw_device: None,
                    raw_host: Some(cached),
                    scratch: Some(scratch),
                    resident_bytes: 0,
                    decoded_wo_a_in_scratch: true,
                },
                FixedLoadTelemetry {
                    fixed_decode_ns,
                    fixed_h2d_ns,
                    fixed_h2d_bytes,
                    raw_h2d_ns,
                    raw_h2d_bytes,
                    cache_status: "stationary_raw_host_hit",
                    ..FixedLoadTelemetry::default()
                },
            ));
        }
        self.misses += 1;
        let (host, mut telemetry) = load_layer_host_raw_wo_a(model_dir, layer, token_ids)?;
        let (runtime, upload, wo_a_raw) =
            upload_layer_runtime_raw_wo_a(model_dir, layer, &host, module, stream)?;
        telemetry.fixed_decode_ns = upload.fixed_decode_ns;
        telemetry.fixed_h2d_ns = upload.fixed_h2d_ns;
        telemetry.fixed_h2d_bytes = upload.fixed_h2d_bytes;
        telemetry.raw_h2d_ns = upload.raw_h2d_ns;
        telemetry.raw_h2d_bytes = upload.raw_h2d_bytes;
        telemetry.cache_status = "stationary_raw_miss";
        drop(wo_a_raw);
        Ok((
            StationaryRawLease {
                layer,
                device_generation: None,
                host_generation: None,
                scratch_generation: None,
                runtime: Some(runtime),
                raw_device: None,
                raw_host: None,
                scratch: None,
                resident_bytes: 0,
                decoded_wo_a_in_scratch: false,
            },
            telemetry,
        ))
    }

    pub(crate) fn access<R>(
        &mut self,
        model_dir: &std::path::Path,
        layer: usize,
        token_ids: &[i64],
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        run: impl FnOnce(&LoadedLayerRuntime) -> Result<R, Box<dyn Error>>,
    ) -> Result<(R, FixedLoadTelemetry), Box<dyn Error>> {
        let candidate = self.plan.iter().any(|candidate| candidate.layer == layer);
        if !candidate {
            let (runtime, telemetry) = load_layer_runtime(model_dir, layer, token_ids, stream)?;
            self.misses += 1;
            return Ok((run(&runtime)?, telemetry));
        }
        let (device_selected, host_selected, plan) = self.selected(layer);
        let decoded_device_bytes = plan.device_bytes;
        let planned_device_bytes = if self.raw_mode {
            plan.raw_device_bytes
        } else {
            decoded_device_bytes
        };
        let decoded_host_bytes = plan.host_bytes;
        let planned_host_bytes = if self.raw_mode {
            plan.raw_host_bytes
        } else {
            decoded_host_bytes
        };
        if self.raw_mode && device_selected && self.raw_device[layer].is_some() {
            self.device_hits += 1;
            let mut cached = self.raw_device[layer]
                .take()
                .ok_or("raw device cache entry disappeared")?;
            let mut scratch = self
                .raw_scratch
                .take()
                .ok_or("raw fixed cache scratch disappeared")?;
            let decode_started = std::time::Instant::now();
            launch_e4m3fn_to_bf16(module, stream, &cached.wo_a_raw, &mut scratch)?;
            stream.synchronize()?;
            let fixed_decode_ns =
                u64::try_from(decode_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
            std::mem::swap(&mut cached.runtime.weights.wo_a, &mut scratch);
            let result = run(&cached.runtime);
            std::mem::swap(&mut cached.runtime.weights.wo_a, &mut scratch);
            self.raw_scratch = Some(scratch);
            self.raw_device[layer] = Some(cached);
            return Ok((
                result?,
                FixedLoadTelemetry {
                    fixed_decode_ns,
                    cache_status: "raw_device_hit",
                    ..FixedLoadTelemetry::default()
                },
            ));
        }
        if !self.raw_mode && device_selected && self.device[layer].is_some() {
            self.device_hits += 1;
            return Ok((
                run(self.device[layer].as_ref().unwrap())?,
                FixedLoadTelemetry {
                    cache_status: "device_hit",
                    ..FixedLoadTelemetry::default()
                },
            ));
        }
        if self.raw_mode && host_selected && self.raw_host[layer].is_some() {
            self.host_hits += 1;
            let cached = self.raw_host[layer]
                .take()
                .ok_or("raw host cache entry disappeared")?;
            let mut scratch = self
                .raw_scratch
                .take()
                .ok_or("raw fixed cache scratch disappeared")?;
            let h2d_started = std::time::Instant::now();
            let mut runtime = upload_layer_runtime(model_dir, layer, &cached.host, stream)?;
            stream.synchronize()?;
            let fixed_h2d_ns = u64::try_from(h2d_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
            let fixed_h2d_bytes = runtime.weights.num_bytes();
            let raw_h2d_started = std::time::Instant::now();
            let wo_a_raw = DeviceBuffer::from_host(stream, &cached.wo_a_raw)?;
            stream.synchronize()?;
            let raw_h2d_ns =
                u64::try_from(raw_h2d_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
            let raw_h2d_bytes = wo_a_raw.num_bytes();
            let decode_started = std::time::Instant::now();
            launch_e4m3fn_to_bf16(module, stream, &wo_a_raw, &mut scratch)?;
            stream.synchronize()?;
            let fixed_decode_ns =
                u64::try_from(decode_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
            std::mem::swap(&mut runtime.weights.wo_a, &mut scratch);
            let result = run(&runtime);
            std::mem::swap(&mut runtime.weights.wo_a, &mut scratch);
            self.raw_scratch = Some(scratch);
            self.raw_host[layer] = Some(cached);
            return Ok((
                result?,
                FixedLoadTelemetry {
                    fixed_decode_ns,
                    fixed_h2d_ns,
                    fixed_h2d_bytes,
                    raw_h2d_ns,
                    raw_h2d_bytes,
                    cache_status: "raw_host_hit",
                    ..FixedLoadTelemetry::default()
                },
            ));
        }
        if !self.raw_mode && host_selected && self.host[layer].is_some() {
            self.host_hits += 1;
            let h2d_started = std::time::Instant::now();
            let runtime =
                upload_layer_runtime(model_dir, layer, self.host[layer].as_ref().unwrap(), stream)?;
            if bench_telemetry_enabled() {
                stream.synchronize()?;
            }
            let fixed_h2d_ns = u64::try_from(h2d_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
            return Ok((
                run(&runtime)?,
                FixedLoadTelemetry {
                    fixed_h2d_ns,
                    fixed_h2d_bytes: runtime.weights.num_bytes(),
                    cache_status: "host_hit",
                    ..FixedLoadTelemetry::default()
                },
            ));
        }
        let (mut host, mut telemetry) = load_layer_host(model_dir, layer, token_ids)?;
        self.misses += 1;
        drop_request_hash_ids_before_fixed_retention(&mut host.state.router, token_ids.len())?;
        if host.num_bytes() != decoded_host_bytes {
            return Err(format!(
                "layer {layer} decoded host bytes {} != plan {decoded_host_bytes}",
                host.num_bytes()
            )
            .into());
        }
        let h2d_started = std::time::Instant::now();
        let runtime = upload_layer_runtime(model_dir, layer, &host, stream)?;
        if bench_telemetry_enabled() {
            stream.synchronize()?;
        }
        telemetry.fixed_h2d_ns =
            u64::try_from(h2d_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        telemetry.fixed_h2d_bytes = runtime.weights.num_bytes();
        let expected_fill_bytes = if self.raw_mode {
            decoded_device_bytes
        } else {
            planned_device_bytes
        };
        if telemetry.fixed_h2d_bytes != expected_fill_bytes {
            return Err(format!(
                "layer {layer} device bytes {} != fill plan {expected_fill_bytes}",
                telemetry.fixed_h2d_bytes
            )
            .into());
        }
        if self.raw_mode && device_selected {
            let result = run(&runtime);
            let (raw, raw_read_ns, raw_source_bytes, raw_h2d_ns, raw_h2d_bytes) =
                into_raw_layer_runtime(model_dir, layer, runtime, stream)?;
            telemetry.raw_read_ns = raw_read_ns;
            telemetry.raw_source_bytes = raw_source_bytes;
            telemetry.raw_h2d_ns = raw_h2d_ns;
            telemetry.raw_h2d_bytes = raw_h2d_bytes;
            if raw.resident_bytes != planned_device_bytes {
                return Err(format!(
                    "layer {layer} raw device bytes {} != plan {planned_device_bytes}",
                    raw.resident_bytes
                )
                .into());
            }
            telemetry.cache_status = "raw_device_fill";
            self.device_resident_bytes += raw.resident_bytes;
            self.raw_device[layer] = Some(raw);
            Ok((result?, telemetry))
        } else if device_selected {
            telemetry.cache_status = "device_fill";
            self.device_resident_bytes += telemetry.fixed_h2d_bytes;
            self.device[layer] = Some(runtime);
            Ok((run(self.device[layer].as_ref().unwrap())?, telemetry))
        } else if self.raw_mode && host_selected {
            let result = run(&runtime);
            let (raw, raw_read_ns, raw_source_bytes) = into_raw_layer_host(model_dir, layer, host)?;
            telemetry.raw_read_ns = raw_read_ns;
            telemetry.raw_source_bytes = raw_source_bytes;
            if raw.resident_bytes != planned_host_bytes {
                return Err(format!(
                    "layer {layer} raw host bytes {} != plan {planned_host_bytes}",
                    raw.resident_bytes
                )
                .into());
            }
            telemetry.cache_status = "raw_host_fill";
            self.host_resident_bytes += raw.resident_bytes;
            self.raw_host[layer] = Some(raw);
            Ok((result?, telemetry))
        } else if host_selected {
            telemetry.cache_status = "host_fill";
            self.host_resident_bytes += host.num_bytes();
            self.host[layer] = Some(host);
            Ok((run(&runtime)?, telemetry))
        } else {
            telemetry.cache_status = "miss";
            Ok((run(&runtime)?, telemetry))
        }
    }
}
