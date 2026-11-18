use super::*;

pub(crate) fn parse_gib_budget(name: &str) -> Result<usize, Box<dyn Error>> {
    let value = match std::env::var(name) {
        Ok(value) => value.parse::<f64>()?,
        Err(std::env::VarError::NotPresent) => 0.0,
        Err(error) => return Err(error.into()),
    };
    if !value.is_finite() || value < 0.0 {
        return Err(format!("{name} must be a finite non-negative GiB value").into());
    }
    Ok((value * 1024.0 * 1024.0 * 1024.0) as usize)
}

pub(crate) fn fixed_resident_bytes(
    plans: &[FixedLayerPlan],
    device_selected: &[bool],
    host_selected: &[bool],
) -> Result<(usize, usize), Box<dyn Error>> {
    if device_selected.len() != plans.len()
        || host_selected.len() != plans.len()
        || device_selected
            .iter()
            .zip(host_selected)
            .any(|(&device, &host)| device && host)
    {
        return Err("decode promotion plan selected one layer for two tiers".into());
    }
    plans
        .iter()
        .zip(device_selected)
        .zip(host_selected)
        .try_fold(
            (0_usize, 0_usize),
            |(device_bytes, host_bytes), ((plan, &device), &host)| {
                Ok((
                    if device {
                        device_bytes
                            .checked_add(plan.raw_device_bytes)
                            .ok_or("decode promotion device byte overflow")?
                    } else {
                        device_bytes
                    },
                    if host {
                        host_bytes
                            .checked_add(plan.raw_host_bytes)
                            .ok_or("decode promotion host byte overflow")?
                    } else {
                        host_bytes
                    },
                ))
            },
        )
}

pub(crate) fn normalized_decode_promotion_targets(
    plans: &[FixedLayerPlan],
    prefill_device: &[bool],
    prefill_host: &[bool],
    decode_device: &[bool],
    decode_host: &[bool],
    explicit: bool,
) -> Result<serde_json::Value, Box<dyn Error>> {
    if prefill_device.len() != plans.len()
        || prefill_host.len() != plans.len()
        || decode_device.len() != plans.len()
        || decode_host.len() != plans.len()
        || plans
            .iter()
            .map(|plan| plan.layer)
            .collect::<BTreeSet<_>>()
            .len()
            != plans.len()
    {
        return Err("decode promotion targets require unique aligned candidate layers".into());
    }
    let mut device_layers = Vec::new();
    let mut host_layers = Vec::new();
    for index in 0..plans.len() {
        match (
            prefill_device[index],
            prefill_host[index],
            decode_device[index],
            decode_host[index],
        ) {
            (false, false, false, false)
            | (true, false, true, false)
            | (false, true, false, true) => {}
            (false, false, true, false) | (false, true, true, false) => {
                device_layers.push(plans[index].layer);
            }
            (false, false, false, true) => host_layers.push(plans[index].layer),
            _ => return Err("decode promotion targets are not monotone tier promotions".into()),
        }
    }
    device_layers.sort_unstable();
    host_layers.sort_unstable();
    Ok(serde_json::json!({
        "mode": if explicit { "explicit" } else { "default_budgeted" },
        "device_layers": device_layers,
        "host_layers": host_layers,
    }))
}

pub(crate) fn validate_decode_promotion_prefixes(
    plans: &[FixedLayerPlan],
    prefill_device: &[bool],
    prefill_host: &[bool],
    decode_device: &[bool],
    decode_host: &[bool],
    decode_device_budget_bytes: usize,
    decode_host_budget_bytes: usize,
) -> Result<(usize, usize), Box<dyn Error>> {
    normalized_decode_promotion_targets(
        plans,
        prefill_device,
        prefill_host,
        decode_device,
        decode_host,
        true,
    )?;
    let (mut device_bytes, mut host_bytes) =
        fixed_resident_bytes(plans, prefill_device, prefill_host)?;
    if device_bytes > decode_device_budget_bytes || host_bytes > decode_host_budget_bytes {
        return Err(
            "decode promotion initial placement exceeds its decode retention budget".into(),
        );
    }
    for index in 0..plans.len() {
        match (
            prefill_device[index],
            prefill_host[index],
            decode_device[index],
            decode_host[index],
        ) {
            (false, false, true, false) => {
                (device_bytes, host_bytes) = decode_device_promotion_residency(
                    device_bytes,
                    host_bytes,
                    plans[index].raw_device_bytes,
                    None,
                    decode_device_budget_bytes,
                )?;
            }
            (false, true, true, false) => {
                (device_bytes, host_bytes) = decode_device_promotion_residency(
                    device_bytes,
                    host_bytes,
                    plans[index].raw_device_bytes,
                    Some(plans[index].raw_host_bytes),
                    decode_device_budget_bytes,
                )?;
            }
            (false, false, false, true) => {
                host_bytes = decode_host_promotion_residency(
                    host_bytes,
                    plans[index].raw_host_bytes,
                    decode_host_budget_bytes,
                )?;
            }
            _ => {}
        }
        if device_bytes > decode_device_budget_bytes || host_bytes > decode_host_budget_bytes {
            return Err("decode promotion prefix exceeds its decode retention budget".into());
        }
    }
    if (device_bytes, host_bytes) != fixed_resident_bytes(plans, decode_device, decode_host)? {
        return Err("decode promotion prefix accounting did not reach its target".into());
    }
    Ok((device_bytes, host_bytes))
}

pub(crate) fn parse_decode_promotion_plan(
    value: Option<&std::ffi::OsStr>,
    plans: &[FixedLayerPlan],
    device_selected: &[bool],
    host_selected: &[bool],
    device_budget_bytes: usize,
    host_budget_bytes: usize,
    decode_device_budget_bytes: usize,
    decode_host_budget_bytes: usize,
) -> Result<Option<(Vec<bool>, Vec<bool>)>, Box<dyn Error>> {
    let Some(value) = value else {
        return Ok(None);
    };
    let value = value
        .to_str()
        .ok_or_else(|| format!("{DECODE_PROMOTION_PLAN_ENV} must be valid UTF-8"))?;
    if value.is_empty()
        || plans.len() != device_selected.len()
        || plans.len() != host_selected.len()
        || plans
            .iter()
            .map(|plan| plan.layer)
            .collect::<BTreeSet<_>>()
            .len()
            != plans.len()
    {
        return Err(format!(
            "{DECODE_PROMOTION_PLAN_ENV} requires a non-empty plan over unique candidate layers"
        )
        .into());
    }
    let (prefill_device_bytes, prefill_host_bytes) =
        fixed_resident_bytes(plans, device_selected, host_selected)?;
    if prefill_device_bytes > device_budget_bytes || prefill_host_bytes > host_budget_bytes {
        return Err("decode promotion prefill placement exceeds its retention budget".into());
    }

    let mut decode_device_selected = device_selected.to_vec();
    let mut decode_host_selected = host_selected.to_vec();
    let mut promoted = BTreeSet::new();
    for promotion in value.split(',') {
        let Some((layer, target)) = promotion.split_once(':') else {
            return Err(format!(
                "{DECODE_PROMOTION_PLAN_ENV} must be comma-separated layer:D or layer:H promotions"
            )
            .into());
        };
        if layer.is_empty() || !layer.bytes().all(|byte| byte.is_ascii_digit()) {
            return Err(format!(
                "{DECODE_PROMOTION_PLAN_ENV} layers must be base-10 unsigned integers"
            )
            .into());
        }
        let layer = layer
            .parse::<usize>()
            .map_err(|_| format!("{DECODE_PROMOTION_PLAN_ENV} layer is outside usize"))?;
        if !promoted.insert(layer) {
            return Err(
                format!("{DECODE_PROMOTION_PLAN_ENV} contains duplicate layer {layer}").into(),
            );
        }
        let index = plans
            .iter()
            .position(|plan| plan.layer == layer)
            .ok_or_else(|| {
                format!("{DECODE_PROMOTION_PLAN_ENV} layer {layer} is not a retention candidate")
            })?;
        match (device_selected[index], host_selected[index], target) {
            (false, false, "D") => decode_device_selected[index] = true,
            (false, false, "H") => decode_host_selected[index] = true,
            (false, true, "D") => {
                decode_device_selected[index] = true;
                decode_host_selected[index] = false;
            }
            _ => {
                return Err(format!(
                    "{DECODE_PROMOTION_PLAN_ENV} layer {layer}:{target} is not S->H, S->D, or H->D"
                )
                .into());
            }
        }
    }
    validate_decode_promotion_prefixes(
        plans,
        device_selected,
        host_selected,
        &decode_device_selected,
        &decode_host_selected,
        decode_device_budget_bytes,
        decode_host_budget_bytes,
    )?;
    Ok(Some((decode_device_selected, decode_host_selected)))
}

pub(crate) fn fixed_saved_bytes_for_target(
    baseline_source_bytes: usize,
    baseline_h2d_bytes: usize,
    target_device: bool,
    target_host: bool,
) -> Result<(usize, usize), Box<dyn Error>> {
    match (target_device, target_host) {
        (true, false) => Ok((baseline_source_bytes, baseline_h2d_bytes)),
        (false, true) => Ok((baseline_source_bytes, 0)),
        (false, false) => Ok((0, 0)),
        (true, true) => Err("fixed saved-byte target selected two tiers".into()),
    }
}

pub(crate) fn select_fixed_cache_layers(
    plans: &[FixedLayerPlan],
    budget: usize,
    already_selected: &[bool],
    device: bool,
    raw_device: bool,
) -> Vec<bool> {
    let mut order = (0..plans.len())
        .filter(|&index| !already_selected[index])
        .collect::<Vec<_>>();
    order.sort_by(|&left, &right| {
        let left_saved = if device {
            plans[left].device_saved_ns
        } else {
            plans[left].host_saved_ns
        };
        let right_saved = if device {
            plans[right].device_saved_ns
        } else {
            plans[right].host_saved_ns
        };
        let left_bytes = if device {
            if raw_device {
                plans[left].raw_device_bytes
            } else {
                plans[left].device_bytes
            }
        } else {
            if raw_device {
                plans[left].raw_host_bytes
            } else {
                plans[left].host_bytes
            }
        };
        let right_bytes = if device {
            if raw_device {
                plans[right].raw_device_bytes
            } else {
                plans[right].device_bytes
            }
        } else {
            if raw_device {
                plans[right].raw_host_bytes
            } else {
                plans[right].host_bytes
            }
        };
        (right_saved as u128 * left_bytes as u128)
            .cmp(&(left_saved as u128 * right_bytes as u128))
            .then_with(|| plans[left].layer.cmp(&plans[right].layer))
    });
    let mut selected = vec![false; plans.len()];
    let mut used = 0_usize;
    for index in order {
        let bytes = if device {
            if raw_device {
                plans[index].raw_device_bytes
            } else {
                plans[index].device_bytes
            }
        } else {
            if raw_device {
                plans[index].raw_host_bytes
            } else {
                plans[index].host_bytes
            }
        };
        if used.checked_add(bytes).is_some_and(|next| next <= budget) {
            selected[index] = true;
            used += bytes;
        }
    }
    selected
}

pub(crate) fn stationary_raw_prefill_selection(
    plans: &[FixedLayerPlan],
    device_budget_bytes: usize,
    host_budget_bytes: usize,
) -> (Vec<bool>, Vec<bool>) {
    let none_selected = vec![false; plans.len()];
    let device_selected =
        select_fixed_cache_layers(plans, device_budget_bytes, &none_selected, true, true);
    let host_selected =
        select_fixed_cache_layers(plans, host_budget_bytes, &device_selected, false, true);
    (device_selected, host_selected)
}

pub(crate) fn preflight_cache_budget(
    requested: usize,
    available: usize,
    headroom: usize,
    name: &str,
) -> usize {
    let effective = requested.min(available.saturating_sub(headroom));
    if effective < requested {
        eprintln!(
            "CACHE_PREFLIGHT_FALLBACK {name} requested_bytes={requested} effective_bytes={effective} available_bytes={available} headroom_bytes={headroom}"
        );
    }
    effective
}

pub(crate) fn host_available_bytes() -> Result<usize, Box<dyn Error>> {
    let meminfo = std::fs::read_to_string("/proc/meminfo")?;
    let kib = meminfo
        .lines()
        .find_map(|line| line.strip_prefix("MemAvailable:"))
        .and_then(|value| value.split_whitespace().next())
        .ok_or("MemAvailable is missing from /proc/meminfo")?
        .parse::<usize>()?;
    kib.checked_mul(1024)
        .ok_or_else(|| "MemAvailable byte count overflow".into())
}

pub(crate) fn device_available_bytes() -> Result<usize, Box<dyn Error>> {
    let (mut free, mut total) = (0_usize, 0_usize);
    let result = unsafe { sys::cuMemGetInfo_v2(&mut free, &mut total) };
    if result != 0 {
        return Err(format!("cuMemGetInfo_v2 failed with {result:?}").into());
    }
    Ok(free)
}

pub(crate) const PHASE1_HCA_PAYLOAD_BYTES: usize = 391_959_576;
pub(crate) const PHASE1_CSA_PAYLOAD_BYTES: usize = 427_239_448;
pub(crate) const PINNED_FIXED_PIPELINE_WORKER_SLOTS: usize = 2;
// A prepared worker slot owns both its reusable wo_a decode scratch and the
// decoded fixed runtime held by PinnedFixedPrepared::lease until finish()
// restores it.  Two prepare replies may be outstanding simultaneously.
pub(crate) const PINNED_FIXED_PIPELINE_WORKER_PREPARED_DEVICE_BYTES: usize =
    PINNED_FIXED_PIPELINE_WORKER_SLOTS * PHASE1_CSA_PAYLOAD_BYTES;
pub(crate) const PINNED_FIXED_PIPELINE_WORKER_SCRATCH_DEVICE_BYTES: usize =
    PINNED_FIXED_PIPELINE_WORKER_SLOTS * RAW_FIXED_SCRATCH_BYTES;
pub(crate) const PINNED_FIXED_PIPELINE_WORKER_DEVICE_BYTES: usize =
    PINNED_FIXED_PIPELINE_WORKER_PREPARED_DEVICE_BYTES
        + PINNED_FIXED_PIPELINE_WORKER_SCRATCH_DEVICE_BYTES;
pub(crate) const PINNED_FIXED_PIPELINE_WORKER_PINNED_HOST_BYTES: usize =
    PINNED_FIXED_PIPELINE_WORKER_SLOTS * PHASE1_CSA_PAYLOAD_BYTES;
pub(crate) const PHASE1_P6_HCA_TRANSFERS: u64 = 23;
pub(crate) const PHASE1_P6_CSA_TRANSFERS: u64 = 29;
pub(crate) const PHASE1_P6_FIXED_PREPARE_NS: u64 = 11_466_446_202;
pub(crate) const PHASE1_P6_WITH_FINAL_NS: u64 = 17_162_779_092;

#[derive(Clone, Copy)]
pub(crate) struct ProcessFaults {
    pub(crate) minor: u64,
    pub(crate) major: u64,
}

pub(crate) fn process_faults() -> Result<ProcessFaults, Box<dyn Error>> {
    let stat = std::fs::read_to_string("/proc/self/stat")?;
    let fields = stat
        .rsplit_once(')')
        .ok_or("/proc/self/stat has no closing command delimiter")?
        .1
        .split_whitespace()
        .collect::<Vec<_>>();
    Ok(ProcessFaults {
        minor: fields
            .get(7)
            .ok_or("/proc/self/stat is missing minflt")?
            .parse()?,
        major: fields
            .get(9)
            .ok_or("/proc/self/stat is missing majflt")?
            .parse()?,
    })
}

pub(crate) fn fault_delta(before: ProcessFaults, after: ProcessFaults) -> serde_json::Value {
    serde_json::json!({
        "minor": after.minor.saturating_sub(before.minor),
        "major": after.major.saturating_sub(before.major),
    })
}

pub(crate) fn phase1_predicted_p6_ns(
    hca_pack_ns: u64,
    hca_h2d_ns: u64,
    csa_pack_ns: u64,
    csa_h2d_ns: u64,
) -> u64 {
    let flow_shop_ns = PHASE1_P6_HCA_TRANSFERS
        .saturating_mul(hca_pack_ns.max(hca_h2d_ns))
        .saturating_add(PHASE1_P6_CSA_TRANSFERS.saturating_mul(csa_pack_ns.max(csa_h2d_ns)));
    PHASE1_P6_WITH_FINAL_NS
        .saturating_sub(PHASE1_P6_FIXED_PREPARE_NS)
        .saturating_add(flow_shop_ns)
}

pub(crate) fn pinned_h2d_service_gate(
    context: &std::sync::Arc<CudaContext>,
) -> Result<(), Box<dyn Error>> {
    let pinned_capacity_bytes = PHASE1_CSA_PAYLOAD_BYTES
        .checked_mul(2)
        .ok_or("Phase1 pinned capacity overflow")?;
    let host_required_bytes = FIXED_PREFETCH_HOST_HEADROOM_BYTES
        .checked_add(pinned_capacity_bytes)
        .and_then(|bytes| bytes.checked_add(2 * PHASE1_CSA_PAYLOAD_BYTES))
        .ok_or("Phase1 host requirement overflow")?;
    let host_available = host_available_bytes()?;
    let device_available = device_available_bytes()?;
    let device_required = V4_DEVICE_CAPACITY_HEADROOM_BYTES
        .checked_add(PHASE1_CSA_PAYLOAD_BYTES)
        .ok_or("Phase1 device requirement overflow")?;
    if host_available < host_required_bytes || device_available < device_required {
        println!(
            "{}",
            serde_json::json!({
                "status": "KILL",
                "mode": "v4-pinned-h2d-service-gate-phase1",
                "reason": "WSL preflight/headroom",
                "host_available_bytes": host_available,
                "host_required_bytes": host_required_bytes,
                "device_available_bytes": device_available,
                "device_required_bytes": device_required,
            })
        );
        return Ok(());
    }

    let mut hca_slot = PinnedHostBuffer::<u8>::zeroed(context, PHASE1_CSA_PAYLOAD_BYTES)?;
    let mut csa_slot = PinnedHostBuffer::<u8>::zeroed(context, PHASE1_CSA_PAYLOAD_BYTES)?;
    let transfer_stream = context.new_stream()?;
    let mut measurements = Vec::with_capacity(2);

    for (name, bytes, pinned) in [
        ("hca", PHASE1_HCA_PAYLOAD_BYTES, &mut hca_slot),
        ("csa", PHASE1_CSA_PAYLOAD_BYTES, &mut csa_slot),
    ] {
        let mut payload = vec![0_u8; bytes];
        for (index, byte) in payload.iter_mut().enumerate() {
            *byte = (index as u64)
                .wrapping_mul(131)
                .wrapping_add(bytes as u64)
                .to_le_bytes()[0];
        }
        let expected_hash = fnv1a_bytes(&payload);

        let pageable_faults_before = process_faults()?;
        let pageable_start = context.new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
        let pageable_end = context.new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
        pageable_start.record(&transfer_stream)?;
        let pageable_wall_started = std::time::Instant::now();
        let pageable_device = DeviceBuffer::from_host(&transfer_stream, &payload)?;
        let pageable_submission_ns =
            u64::try_from(pageable_wall_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        pageable_end.record(&transfer_stream)?;
        pageable_end.synchronize()?;
        let pageable_wall_ns =
            u64::try_from(pageable_wall_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        let pageable_event_ns =
            (pageable_start.elapsed_ms(&pageable_end)? as f64 * 1_000_000.0) as u64;
        let pageable_faults_after = process_faults()?;
        let pageable_download = pageable_device.to_host_vec(&transfer_stream)?;
        let pageable_hash = fnv1a_bytes(&pageable_download);
        drop(pageable_download);
        drop(pageable_device);

        let pack_faults_before = process_faults()?;
        let pack_started = std::time::Instant::now();
        pinned.as_mut_slice()[..bytes].copy_from_slice(&payload);
        let pack_ns = u64::try_from(pack_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        let pack_faults_after = process_faults()?;

        let pinned_faults_before = process_faults()?;
        let pinned_start = context.new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
        let pinned_end = context.new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
        let pinned_wall_started = std::time::Instant::now();
        let pinned_device =
            unsafe { DeviceBuffer::<u8>::uninitialized_async(&transfer_stream, bytes)? };
        pinned_start.record(&transfer_stream)?;
        let copy_result = unsafe {
            sys::cuMemcpyHtoDAsync_v2(
                pinned_device.cu_deviceptr(),
                pinned.as_ptr().cast(),
                bytes,
                transfer_stream.cu_stream(),
            )
        };
        if copy_result != 0 {
            return Err(format!("Phase1 pinned H2D enqueue failed with {copy_result:?}").into());
        }
        pinned_end.record(&transfer_stream)?;
        let pinned_submission_ns =
            u64::try_from(pinned_wall_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        pinned_end.synchronize()?;
        let pinned_wall_ns =
            u64::try_from(pinned_wall_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        let pinned_event_ns = (pinned_start.elapsed_ms(&pinned_end)? as f64 * 1_000_000.0) as u64;
        let pinned_faults_after = process_faults()?;
        let pinned_download = pinned_device.to_host_vec(&transfer_stream)?;
        let pinned_hash = fnv1a_bytes(&pinned_download);
        drop(pinned_download);
        drop(pinned_device);
        transfer_stream.synchronize()?;

        measurements.push(serde_json::json!({
            "payload": name,
            "bytes": bytes,
            "expected_hash": format!("{expected_hash:016x}"),
            "pageable": {
                "allocation_success": true,
                "cuda_event_service_ns": pageable_event_ns,
                "submission_ns": pageable_submission_ns,
                "wall_ns": pageable_wall_ns,
                "faults": fault_delta(pageable_faults_before, pageable_faults_after),
                "d2h_hash": format!("{pageable_hash:016x}"),
                "hash_equal": pageable_hash == expected_hash,
            },
            "pinned": {
                "allocation_success": true,
                "pack_wall_ns": pack_ns,
                "pack_faults": fault_delta(pack_faults_before, pack_faults_after),
                "cuda_event_h2d_service_ns": pinned_event_ns,
                "submission_ns": pinned_submission_ns,
                "wall_ns": pinned_wall_ns,
                "transfer_faults": fault_delta(pinned_faults_before, pinned_faults_after),
                "d2h_hash": format!("{pinned_hash:016x}"),
                "hash_equal": pinned_hash == expected_hash,
            },
        }));
    }

    let value = |payload: usize, arm: &str, field: &str| -> Result<u64, Box<dyn Error>> {
        measurements[payload][arm][field]
            .as_u64()
            .ok_or_else(|| format!("Phase1 measurement {payload}.{arm}.{field} is missing").into())
    };
    let predicted_p6_ns = phase1_predicted_p6_ns(
        value(0, "pinned", "pack_wall_ns")?,
        value(0, "pinned", "cuda_event_h2d_service_ns")?,
        value(1, "pinned", "pack_wall_ns")?,
        value(1, "pinned", "cuda_event_h2d_service_ns")?,
    );
    let hashes_pass = measurements.iter().all(|measurement| {
        measurement["pageable"]["hash_equal"] == true && measurement["pinned"]["hash_equal"] == true
    });
    let threshold_ns = PHASE1_P6_WITH_FINAL_NS.saturating_mul(80) / 100;
    let status = if hashes_pass && predicted_p6_ns <= threshold_ns {
        "PASS"
    } else {
        "KILL"
    };
    println!(
        "{}",
        serde_json::json!({
            "status": status,
            "mode": "v4-pinned-h2d-service-gate-phase1",
            "payload_measurements": measurements,
            "allocation": {
                "pinned_slot_count": 2,
                "pinned_capacity_bytes": pinned_capacity_bytes,
                "host_available_bytes": host_available,
                "host_required_bytes": host_required_bytes,
                "device_available_bytes": device_available,
                "device_required_bytes": device_required,
            },
            "flow_shop": {
                "model": "p6_nonfixed + 23*max(hca_pack,hca_h2d) + 29*max(csa_pack,csa_h2d)",
                "p6_with_final_baseline_ns": PHASE1_P6_WITH_FINAL_NS,
                "p6_fixed_prepare_baseline_ns": PHASE1_P6_FIXED_PREPARE_NS,
                "predicted_p6_with_final_ns": predicted_p6_ns,
                "required_max_ns": threshold_ns,
                "required_reduction_percent": 20,
            },
            "hashes_pass": hashes_pass,
            "phase2_authorized": false,
        })
    );
    Ok(())
}

pub(crate) fn fixed_layer_tensor_plan(
    model_dir: &std::path::Path,
    layer: usize,
) -> Result<(Vec<FixedTensorPlan>, usize, usize), Box<dyn Error>> {
    let shard = safetensors::ShardFile::open(
        model_dir.join(format!("model-{:05}-of-00064.safetensors", layer + 2)),
    )?;
    let prefix = format!("layers.{layer}.");
    let mut tensors = shard
        .tensors_with_prefix(&prefix)
        .into_iter()
        .filter(|(name, _)| {
            !name.contains(".ffn.experts.")
                && !name.ends_with(".ffn.gate.tid2eid")
                && !name.ends_with(".attn.wo_a.scale")
        })
        .map(|(name, info)| {
            let source_bytes = usize::try_from(info.byte_len)?;
            let decoded_wo_a = name.ends_with(".attn.wo_a.weight");
            Ok::<_, Box<dyn Error>>(FixedTensorPlan {
                name,
                dtype: if decoded_wo_a {
                    format!("{:?}->Bf16", info.dtype)
                } else {
                    format!("{:?}", info.dtype)
                },
                shape: info.shape,
                source_bytes,
                decoded_bytes: if decoded_wo_a {
                    source_bytes.checked_mul(2).ok_or("decoded wo_a overflow")?
                } else {
                    source_bytes
                },
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    if layer < 3 {
        tensors.push(FixedTensorPlan {
            name: format!("layers.{layer}.ffn.gate.synthetic_zero_bias"),
            dtype: "F32".into(),
            shape: vec![u64::try_from(v4_router::V4_ROUTED_EXPERTS)?],
            source_bytes: 0,
            decoded_bytes: v4_router::V4_ROUTED_EXPERTS * 4,
        });
    }
    let device_bytes = tensors.iter().try_fold(0_usize, |sum, tensor| {
        sum.checked_add(tensor.decoded_bytes)
            .ok_or("fixed tensor byte sum overflow")
    })?;
    let wo_a_source_bytes = tensors
        .iter()
        .find(|tensor| tensor.name.ends_with(".attn.wo_a.weight"))
        .ok_or("fixed tensor plan is missing wo_a.weight")?
        .source_bytes;
    let wo_a_scale = shard.tensor(&format!("layers.{layer}.attn.wo_a.scale"))?;
    if wo_a_scale.dtype != safetensors::TensorDtype::F8E8M0 || wo_a_scale.shape != [128, 32] {
        return Err(format!("layer {layer} wo_a.scale metadata contract mismatch").into());
    }
    Ok((tensors, device_bytes, wo_a_source_bytes))
}

pub(crate) const BOUNDED_READAHEAD_MIN_AVAILABLE_BYTES: usize = 4 * 1024 * 1024 * 1024;

pub(crate) struct BoundedReadaheadSourceSpan {
    pub(crate) range: safetensors::FileRange,
}

pub(crate) struct BoundedReadaheadLayer {
    pub(crate) shard: safetensors::ShardFile,
    pub(crate) spans: Vec<BoundedReadaheadSourceSpan>,
    pub(crate) source_bytes: u64,
}

pub(crate) fn fixed_layer_bounded_readahead_source(
    model_dir: &std::path::Path,
    layer: usize,
) -> Result<BoundedReadaheadLayer, Box<dyn Error>> {
    let (fixed, _, _) = fixed_layer_tensor_plan(model_dir, layer)?;
    let shard = safetensors::ShardFile::open(
        model_dir.join(format!("model-{:05}-of-00064.safetensors", layer + 2)),
    )?;
    let mut spans = fixed
        .into_iter()
        .filter(|tensor| tensor.source_bytes != 0)
        .map(|tensor| {
            let len = u64::try_from(tensor.source_bytes)?;
            Ok::<_, Box<dyn Error>>(BoundedReadaheadSourceSpan {
                range: shard.tensor_file_range(&tensor.name, 0, len)?,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let prefix = format!("layers.{layer}");
    // fixed_layer_tensor_plan deliberately omits these because they are not
    // retained device payloads. They are nevertheless real loader reads.
    let wo_a_scale = format!("{prefix}.attn.wo_a.scale");
    let wo_a_scale_len = shard.tensor(&wo_a_scale)?.byte_len;
    spans.push(BoundedReadaheadSourceSpan {
        range: shard.tensor_file_range(&wo_a_scale, 0, wo_a_scale_len)?,
    });
    if layer < 3 {
        let tid2eid = format!("{prefix}.ffn.gate.tid2eid");
        let len = u64::try_from(v4_router::V4_TOPK * std::mem::size_of::<i64>())?;
        spans.push(BoundedReadaheadSourceSpan {
            range: shard.tensor_file_range(&tid2eid, 0, len)?,
        });
    }
    let source_bytes = spans.iter().try_fold(0_u64, |total, span| {
        total
            .checked_add(span.range.len)
            .ok_or("bounded readahead source byte count overflow")
    })?;
    Ok(BoundedReadaheadLayer {
        shard,
        spans,
        source_bytes,
    })
}

pub(crate) fn bounded_readahead_window_bytes(
    ranges: &[safetensors::FileRange],
) -> Result<u64, Box<dyn Error>> {
    ranges.iter().try_fold(0_u64, |total, range| {
        total
            .checked_add(range.len)
            .ok_or_else(|| "bounded readahead issued byte count overflow".into())
    })
}

pub(crate) fn bounded_readahead_page_cap(
    source_bytes: u64,
    page_bytes: u64,
) -> Result<u64, Box<dyn Error>> {
    if page_bytes == 0 {
        return Err("bounded readahead OS page size is zero".into());
    }
    source_bytes
        .checked_add(page_bytes - 1)
        .map(|bytes| bytes / page_bytes * page_bytes)
        .ok_or_else(|| "bounded readahead page-aligned cap overflow".into())
}

pub(crate) fn os_page_bytes() -> Result<u64, Box<dyn Error>> {
    unsafe extern "C" {
        fn getpagesize() -> i32;
    }
    // SAFETY: getpagesize has no arguments or side effects.
    u64::try_from(unsafe { getpagesize() }).map_err(Into::into)
}

pub(crate) fn bounded_readahead_metadata_page_cap(
    model_dir: &std::path::Path,
) -> Result<u64, Box<dyn Error>> {
    let max_source_bytes = (0..v4_state::V4_LAYER_COUNT)
        .map(|layer| {
            fixed_layer_bounded_readahead_source(model_dir, layer).map(|plan| plan.source_bytes)
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .max()
        .ok_or("bounded readahead metadata plan is empty")?;
    bounded_readahead_page_cap(max_source_bytes, os_page_bytes()?)
}

pub(crate) fn production_bounded_readahead(
    model_dir: &std::path::Path,
    layer: usize,
    page_cap: Option<u64>,
) -> BoundedReadaheadTelemetry {
    let skipped = |reason| BoundedReadaheadTelemetry {
        layer,
        issued: false,
        skipped_reason: Some(reason),
        bytes: 0,
        ranges: Vec::new(),
        advice_us: 0,
    };
    if !cfg!(target_os = "linux") {
        return skipped("not_linux");
    }
    let Some(page_cap) = page_cap else {
        return skipped("metadata_cap_unavailable");
    };
    match host_available_bytes() {
        Ok(available) if available >= BOUNDED_READAHEAD_MIN_AVAILABLE_BYTES => {}
        Ok(_) => return skipped("mem_available_below_4gib"),
        Err(_) => return skipped("mem_available_unavailable"),
    }
    let plan = match fixed_layer_bounded_readahead_source(model_dir, layer) {
        Ok(plan) => plan,
        Err(_) => return skipped("invalid_span"),
    };
    let ranges = plan.spans.iter().map(|span| span.range).collect::<Vec<_>>();
    let ranges = match safetensors::merge_adjacent_ranges(&ranges) {
        Ok(ranges) => ranges,
        Err(_) => return skipped("invalid_span"),
    };
    let bytes = match bounded_readahead_window_bytes(&ranges) {
        Ok(bytes) if plan.source_bytes <= page_cap && bytes <= page_cap => bytes,
        Ok(_) => return skipped("metadata_page_cap"),
        Err(_) => return skipped("invalid_span"),
    };
    let started = std::time::Instant::now();
    if plan.shard.advise_willneed_ranges(&ranges).is_err() {
        return BoundedReadaheadTelemetry {
            layer,
            issued: false,
            skipped_reason: Some("syscall_error"),
            bytes,
            ranges,
            advice_us: u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX),
        };
    }
    BoundedReadaheadTelemetry {
        layer,
        issued: true,
        skipped_reason: None,
        bytes,
        ranges,
        advice_us: u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX),
    }
}

pub(crate) fn bounded_readahead_json(
    telemetry: Option<&BoundedReadaheadTelemetry>,
    following_prepare_read_service_ns: u64,
) -> Option<serde_json::Value> {
    telemetry.map(|telemetry| {
        serde_json::json!({
            "issued": telemetry.issued,
            "skipped_reason": telemetry.skipped_reason,
            "bytes": telemetry.bytes,
            "ranges": telemetry.ranges.iter().map(|range| serde_json::json!({"offset": range.offset, "len": range.len})).collect::<Vec<_>>(),
            "advice_us": telemetry.advice_us,
            "following_prepare_read_service_ns": following_prepare_read_service_ns,
        })
    })
}

pub(crate) fn max_fixed_layer_device_bytes(
    model_dir: &std::path::Path,
) -> Result<(usize, usize), Box<dyn Error>> {
    (0..v4_state::V4_LAYER_COUNT)
        .map(|layer| {
            let (_, device_bytes, _) = fixed_layer_tensor_plan(model_dir, layer)?;
            Ok((layer, device_bytes))
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?
        .into_iter()
        .max_by_key(|&(_, bytes)| bytes)
        .ok_or_else(|| "fixed layer metadata plan is empty".into())
}

pub(crate) struct DeviceAttentionCapacityPlan {
    pub(crate) qd4_device_bytes: usize,
    pub(crate) qd4_immutable_expert_cache_device_bytes: usize,
    pub(crate) fixed_layer: usize,
    pub(crate) max_fixed_device_bytes: usize,
    pub(crate) max_device_token_state_bytes: usize,
    pub(crate) hca_device_token_state_bytes: usize,
    pub(crate) device_token_state_count: usize,
    pub(crate) device_token_state_total_bytes: usize,
    pub(crate) rope_lut_bytes: usize,
    pub(crate) tid2eid_transient_max_bytes: usize,
    pub(crate) certified_argmax_device_temp_bytes: usize,
    pub(crate) certified_argmax_host_temp_bytes: usize,
    pub(crate) headroom_bytes: usize,
    pub(crate) required_bytes: usize,
}

pub(crate) fn device_attention_capacity_plan(
    model_dir: &std::path::Path,
    qd4_enabled: bool,
    qd4_immutable_expert_cache: bool,
    request_rope_rows: usize,
    certified_argmax_production: bool,
) -> Result<DeviceAttentionCapacityPlan, Box<dyn Error>> {
    let (fixed_layer, max_fixed_device_bytes) = max_fixed_layer_device_bytes(model_dir)?;
    let qd4_device_bytes = usize::from(qd4_enabled) * V4_QD4_DEVICE_BYTES;
    let qd4_immutable_expert_cache_device_bytes =
        usize::from(qd4_immutable_expert_cache) * V4_QD4_IMMUTABLE_EXPERT_CACHE_DEVICE_BYTES;
    if !(V4_REQUEST_ROPE_LUT_ROWS..=V4_SHORT_DEVICE_DECODE_MAX_ROWS).contains(&request_rope_rows) {
        return Err("device-attention capacity rows are outside the short-decode ceiling".into());
    }
    let short_device_decode = request_rope_rows > V4_REQUEST_ROPE_LUT_ROWS;
    let rope_lut_bytes = 2 * request_rope_rows * v4_state::Q_ROPE_DIM * std::mem::size_of::<f32>();
    let tid2eid_transient_max_bytes = V4_TID2EID_TRANSIENT_MAX_BYTES;
    let certified_argmax_device_temp_bytes =
        usize::from(certified_argmax_production) * CERTIFIED_ARGMAX_DEVICE_TEMP_BYTES;
    let certified_argmax_host_temp_bytes =
        usize::from(certified_argmax_production) * CERTIFIED_ARGMAX_HOST_TEMP_BYTES;
    let manifests = v4_state::serial_layer_manifests(0)?;
    let (hca_states, csa_states) =
        manifests
            .iter()
            .fold((0_usize, 0_usize), |(hca, csa), manifest| {
                match manifest.compression_ratio {
                    v4_state::CompressionRatio::WindowOnly => (hca, csa),
                    v4_state::CompressionRatio::Hca128 => (hca + 1, csa),
                    v4_state::CompressionRatio::Csa4 => (hca, csa + 1),
                }
            });
    let device_token_state_count = if short_device_decode {
        manifests.len()
    } else {
        2
    };
    let device_token_state_total_bytes = if short_device_decode {
        hca_states
            .checked_mul(V4_DEVICE_TOKEN_STATE_HCA_BYTES)
            .and_then(|bytes| {
                csa_states
                    .checked_mul(V4_DEVICE_TOKEN_STATE_MAX_BYTES)
                    .and_then(|csa_bytes| bytes.checked_add(csa_bytes))
            })
            .ok_or("device attention persistent-state byte overflow")?
    } else {
        V4_DEVICE_TOKEN_STATE_MAX_BYTES + V4_DEVICE_TOKEN_STATE_HCA_BYTES
    };
    let required_bytes = [
        qd4_device_bytes,
        qd4_immutable_expert_cache_device_bytes,
        max_fixed_device_bytes,
        device_token_state_total_bytes,
        rope_lut_bytes,
        tid2eid_transient_max_bytes,
        certified_argmax_device_temp_bytes,
        V4_DEVICE_CAPACITY_HEADROOM_BYTES,
    ]
    .into_iter()
    .try_fold(0_usize, |total, bytes| {
        total
            .checked_add(bytes)
            .ok_or("device-attention capacity ledger overflow")
    })?;
    Ok(DeviceAttentionCapacityPlan {
        qd4_device_bytes,
        qd4_immutable_expert_cache_device_bytes,
        fixed_layer,
        max_fixed_device_bytes,
        max_device_token_state_bytes: V4_DEVICE_TOKEN_STATE_MAX_BYTES,
        hca_device_token_state_bytes: V4_DEVICE_TOKEN_STATE_HCA_BYTES,
        device_token_state_count,
        device_token_state_total_bytes,
        rope_lut_bytes,
        tid2eid_transient_max_bytes,
        certified_argmax_device_temp_bytes,
        certified_argmax_host_temp_bytes,
        headroom_bytes: V4_DEVICE_CAPACITY_HEADROOM_BYTES,
        required_bytes,
    })
}

pub(crate) fn validate_device_attention_capacity(
    model_dir: &std::path::Path,
    qd4_enabled: bool,
    qd4_immutable_expert_cache: bool,
    request_rope_rows: usize,
    certified_argmax_production: bool,
    available_bytes: usize,
) -> Result<DeviceAttentionCapacityPlan, Box<dyn Error>> {
    let plan = device_attention_capacity_plan(
        model_dir,
        qd4_enabled,
        qd4_immutable_expert_cache,
        request_rope_rows,
        certified_argmax_production,
    )?;
    if available_bytes < plan.required_bytes {
        return Err(format!(
            "device attention capacity requires {} bytes (QD4={} immutable_cache={} max_fixed={} layer={} states={} state_bytes={} max_state={} hca_state={} rope_lut={} tid2eid_transient={} certified_argmax_temp={} headroom={}), but cuMemGetInfo reports {} free",
            plan.required_bytes,
            plan.qd4_device_bytes,
            plan.qd4_immutable_expert_cache_device_bytes,
            plan.max_fixed_device_bytes,
            plan.fixed_layer,
            plan.device_token_state_count,
            plan.device_token_state_total_bytes,
            plan.max_device_token_state_bytes,
            plan.hca_device_token_state_bytes,
            plan.rope_lut_bytes,
            plan.tid2eid_transient_max_bytes,
            plan.certified_argmax_device_temp_bytes,
            plan.headroom_bytes,
            available_bytes,
        )
        .into());
    }
    Ok(plan)
}

pub(crate) fn fixed_layer_plans(
    model_dir: &std::path::Path,
    layers: &[usize],
    baseline_path: &std::path::Path,
) -> Result<Vec<FixedLayerPlan>, Box<dyn Error>> {
    let mut baseline = vec![None; v4_state::V4_LAYER_COUNT];
    for line in std::fs::read_to_string(baseline_path)?.lines() {
        let value: serde_json::Value = serde_json::from_str(line)?;
        if value.get("event").and_then(|value| value.as_str()) != Some("layer") {
            continue;
        }
        let layer = usize::try_from(
            value
                .get("layer")
                .and_then(|value| value.as_u64())
                .ok_or("baseline layer record is missing layer")?,
        )?;
        let bench = value
            .get("bench")
            .and_then(|value| value.as_object())
            .ok_or("baseline layer record is missing bench telemetry")?;
        baseline[layer] = Some((
            bench["fixed_read_alloc_ns"]
                .as_u64()
                .ok_or("baseline fixed_read_alloc_ns is invalid")?,
            bench["fixed_decode_ns"]
                .as_u64()
                .ok_or("baseline fixed_decode_ns is invalid")?,
            bench["fixed_h2d_ns"]
                .as_u64()
                .ok_or("baseline fixed_h2d_ns is invalid")?,
            usize::try_from(
                bench["fixed_h2d_bytes"]
                    .as_u64()
                    .ok_or("baseline fixed_h2d_bytes is invalid")?,
            )?,
        ));
    }
    layers
        .iter()
        .map(|&layer| {
            let (read_alloc_ns, decode_ns, h2d_ns, expected_device_bytes) =
                baseline[layer].ok_or("baseline does not cover every cache candidate layer")?;
            let (tensors, device_bytes, wo_a_source_bytes) =
                fixed_layer_tensor_plan(model_dir, layer)?;
            let raw_device_bytes = device_bytes
                .checked_sub(wo_a_source_bytes)
                .ok_or("raw fixed device byte sum overflow")?;
            if device_bytes != expected_device_bytes {
                return Err(format!(
                    "layer {layer} fixed manifest bytes {device_bytes} != baseline {expected_device_bytes}"
                )
                .into());
            }
            let host_bytes = fixed_layer_decoded_host_bytes(layer, device_bytes)?;
            let raw_host_bytes = host_bytes
                .checked_sub(wo_a_source_bytes)
                .ok_or("raw fixed host byte sum overflow")?;
            Ok(FixedLayerPlan {
                layer,
                tensors,
                device_bytes,
                raw_device_bytes,
                host_bytes,
                raw_host_bytes,
                device_saved_ns: read_alloc_ns
                    .saturating_add(decode_ns)
                    .saturating_add(h2d_ns),
                host_saved_ns: read_alloc_ns.saturating_add(decode_ns),
            })
        })
        .collect()
}

pub(crate) fn stationary_raw_layer_plans(
    model_dir: &std::path::Path,
    layers: &[usize],
) -> Result<Vec<FixedLayerPlan>, Box<dyn Error>> {
    layers
        .iter()
        .map(|&layer| {
            let (tensors, device_bytes, wo_a_source_bytes) =
                fixed_layer_tensor_plan(model_dir, layer)?;
            let raw_device_bytes = device_bytes
                .checked_sub(wo_a_source_bytes)
                .ok_or("stationary raw device byte sum overflow")?;
            let host_bytes = fixed_layer_decoded_host_bytes(layer, device_bytes)?;
            let raw_host_bytes = host_bytes
                .checked_sub(wo_a_source_bytes)
                .ok_or("stationary raw host byte sum overflow")?;
            // ponytail: prefill capture has no timing baseline yet. Equal benefit
            // makes the existing selector retain the most layers per byte.
            Ok(FixedLayerPlan {
                layer,
                tensors,
                device_bytes,
                raw_device_bytes,
                host_bytes,
                raw_host_bytes,
                device_saved_ns: 1,
                host_saved_ns: 1,
            })
        })
        .collect()
}
