use super::*;

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct AttentionAudit {
    pub(crate) outputs: [Vec<u16>; 2],
    pub(crate) stage_digests: [Vec<(&'static str, u64)>; 2],
    pub(crate) topk_digests: [u64; 2],
    pub(crate) kv: Vec<u16>,
    pub(crate) main_kv: Vec<f32>,
    pub(crate) main_score: Vec<f32>,
    pub(crate) indexer_kv: Option<Vec<f32>>,
    pub(crate) indexer_score: Option<Vec<f32>>,
}

#[derive(Clone, Debug)]
pub(crate) struct PersistentStateAudit {
    pub(crate) kv: Vec<u16>,
    pub(crate) compressed_kv: Vec<u16>,
    pub(crate) indexer_kv: Vec<u16>,
    pub(crate) main_kv: Vec<f32>,
    pub(crate) main_score: Vec<f32>,
    pub(crate) indexer_main_kv: Option<Vec<f32>>,
    pub(crate) indexer_main_score: Option<Vec<f32>>,
}

pub(crate) fn host_persistent_state_audit(state: &v4_state::V4LayerPersistentState) -> PersistentStateAudit {
    let main = state.compressor();
    let indexer = state.indexer();
    PersistentStateAudit {
        kv: state.kv_bf16().to_vec(),
        compressed_kv: state.compressed_kv_bf16().to_vec(),
        indexer_kv: state.indexer_kv_bf16().to_vec(),
        main_kv: main.kv().to_vec(),
        main_score: main.score().to_vec(),
        indexer_main_kv: indexer.map(|value| value.kv().to_vec()),
        indexer_main_score: indexer.map(|value| value.score().to_vec()),
    }
}

pub(crate) fn device_persistent_state_audit(
    stream: &CudaStream,
    state: &DeviceTokenState,
) -> Result<PersistentStateAudit, Box<dyn Error>> {
    let indexer = state.indexer_compressor.as_ref();
    Ok(PersistentStateAudit {
        kv: state.kv_cache.to_host_vec(stream)?[..state.committed_rows * v4_state::Q_HEAD_DIM]
            .to_vec(),
        compressed_kv: state.compressed_cache.to_host_vec(stream)?
            [..state.compressed_rows * v4_state::Q_HEAD_DIM]
            .to_vec(),
        indexer_kv: state.indexer_cache.to_host_vec(stream)?[..state.compressed_rows * 128]
            .to_vec(),
        main_kv: state.compressor.persistent_kv.to_host_vec(stream)?,
        main_score: state.compressor.persistent_score.to_host_vec(stream)?,
        indexer_main_kv: indexer
            .map(|value| value.persistent_kv.to_host_vec(stream))
            .transpose()?,
        indexer_main_score: indexer
            .map(|value| value.persistent_score.to_host_vec(stream))
            .transpose()?,
    })
}

/// The audit reads each listed device allocation in full before trimming the
/// logical cache prefix, so this is the physical D2H transfer ledger.
pub(crate) fn device_persistent_state_audit_d2h_transfer(state: &DeviceTokenState) -> (usize, usize) {
    let mut count = 5;
    let mut bytes = state.kv_cache.num_bytes()
        + state.compressed_cache.num_bytes()
        + state.indexer_cache.num_bytes()
        + state.compressor.persistent_kv.num_bytes()
        + state.compressor.persistent_score.num_bytes();
    if let Some(indexer) = state.indexer_compressor.as_ref() {
        count += 2;
        bytes += indexer.persistent_kv.num_bytes() + indexer.persistent_score.num_bytes();
    }
    (count, bytes)
}

/// Audit-only D2H snapshot of the generic layout.  It mirrors the host
/// persistent representation exactly: physical raw ring, compressor tails,
/// compressed cache, and optional CSA indexer state.
pub(crate) fn generic_device_persistent_state_audit(
    stream: &CudaStream,
    state: &DeviceLayerPersistentState,
) -> Result<PersistentStateAudit, Box<dyn Error>> {
    Ok(PersistentStateAudit {
        kv: state.raw_kv_ring.to_host_vec(stream)?,
        compressed_kv: state.compressed_kv.to_host_vec(stream)?,
        indexer_kv: state
            .indexer_kv
            .as_ref()
            .map(|value| value.to_host_vec(stream))
            .transpose()?
            .unwrap_or_default(),
        main_kv: state.compressor_tail_kv.to_host_vec(stream)?,
        main_score: state.compressor_tail_score.to_host_vec(stream)?,
        indexer_main_kv: state
            .indexer_tail_kv
            .as_ref()
            .map(|value| value.to_host_vec(stream))
            .transpose()?,
        indexer_main_score: state
            .indexer_tail_score
            .as_ref()
            .map(|value| value.to_host_vec(stream))
            .transpose()?,
    })
}

pub(crate) fn generic_device_persistent_state_audit_d2h_transfer(
    state: &DeviceLayerPersistentState,
) -> (usize, usize) {
    let mut count = 4;
    let mut bytes = state.raw_kv_ring.num_bytes()
        + state.compressor_tail_kv.num_bytes()
        + state.compressor_tail_score.num_bytes()
        + state.compressed_kv.num_bytes();
    if let (Some(indexer_kv), Some(indexer_tail_kv), Some(indexer_tail_score)) = (
        state.indexer_kv.as_ref(),
        state.indexer_tail_kv.as_ref(),
        state.indexer_tail_score.as_ref(),
    ) {
        count += 3;
        bytes +=
            indexer_kv.num_bytes() + indexer_tail_kv.num_bytes() + indexer_tail_score.num_bytes();
    }
    (count, bytes)
}

/// Runs the host recurrence used only to observe generic device state.  The
/// authoritative row is immutable here, and this helper receives no device
/// state, request scratch, expert slots, or router state.
pub(crate) fn run_generic_device_state_audit_mirror(
    enabled: bool,
    persistent: Option<&mut v4_state::V4SerialPersistentState>,
    layer: usize,
    hc_hidden: &[u16],
    run: impl FnOnce(&mut v4_state::V4LayerPersistentState, &[u16]) -> Result<(), Box<dyn Error>>,
) -> Result<(), Box<dyn Error>> {
    if enabled {
        persistent
            .ok_or("generic device state audit host mirror disappeared")?
            .transaction(layer, |host_state| run(host_state, hc_hidden))?;
    }
    Ok(())
}

/// Runs only after a generic head boundary has drained its worker leases. The
/// host mirror is an independent recurrence; audit copies never enter the
/// production transfer ledger or per-layer telemetry.
pub(crate) fn generic_device_state_audit_head(
    stream: &CudaStream,
    states: &[Option<DeviceLayerPersistentState>],
    host: &v4_state::V4SerialPersistentState,
    lifecycle: &GenericStationaryLifecycle,
    scratch: &DeviceRequestScratch,
    position: usize,
) -> Result<(serde_json::Value, Vec<(usize, PersistentStateAudit)>), Box<dyn Error>> {
    lifecycle.assert_drained()?;
    if states.len() != v4_state::V4_LAYER_COUNT {
        return Err("generic device state audit lost a layer store".into());
    }
    let started = std::time::Instant::now();
    let mut d2h_count = 0_usize;
    let mut d2h_bytes = 0_usize;
    let mut layers = Vec::with_capacity(v4_state::V4_LAYER_COUNT);
    let mut snapshots = Vec::with_capacity(v4_state::V4_LAYER_COUNT);
    for (layer, state) in states.iter().enumerate() {
        let state = state
            .as_ref()
            .ok_or("generic device state audit found an empty layer store")?;
        if state.layer != layer || state.committed_rows != position + 1 {
            return Err("generic device state audit found a noncontiguous store".into());
        }
        let expected = host_persistent_state_audit(
            host.layer(layer)
                .ok_or("generic device state audit host mirror lost a layer")?,
        );
        let actual = generic_device_persistent_state_audit(stream, state)?;
        check_persistent_state_audit(layer, position, &expected, &actual)?;
        let (count, bytes) = generic_device_persistent_state_audit_d2h_transfer(state);
        d2h_count = d2h_count.saturating_add(count);
        d2h_bytes = d2h_bytes.saturating_add(bytes);
        layers.push(serde_json::json!({
            "layer": layer,
            "committed_rows": state.committed_rows,
            "exact": true,
            "host": persistent_state_audit_json(&expected),
            "device": persistent_state_audit_json(&actual),
        }));
        snapshots.push((state.committed_rows, actual));
    }
    Ok((
        serde_json::json!({
            "position": position,
            "layers": layers,
            "exact": true,
            "d2h_count": d2h_count,
            "d2h_bytes": d2h_bytes,
            "strict_compressor_conversion_audit": scratch.exact_compressor.as_ref().map(GenericExactCompressorScratch::audit_telemetry),
            "audit_wall_ns": u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            "excluded_from_production_timing": true,
        }),
        snapshots,
    ))
}

pub(crate) fn check_post_mtp_persistent_state(
    layer: usize,
    position: usize,
    expected_committed_rows: usize,
    expected: &PersistentStateAudit,
    actual_committed_rows: usize,
    actual: &PersistentStateAudit,
) -> Result<(), Box<dyn Error>> {
    if actual_committed_rows != expected_committed_rows {
        return Err(format!(
            "post-MTP teacher-state clobber: layer {layer} position {position} committed_rows expected {expected_committed_rows} actual {actual_committed_rows}"
        )
        .into());
    }
    check_persistent_state_audit(layer, position, expected, actual)
        .map_err(|error| format!("post-MTP teacher-state clobber: {error}").into())
}

pub(crate) fn check_generic_device_state_unchanged(
    stream: &CudaStream,
    states: &[Option<DeviceLayerPersistentState>],
    expected: &[(usize, PersistentStateAudit)],
    position: usize,
) -> Result<u64, Box<dyn Error>> {
    let started = std::time::Instant::now();
    if states.len() != v4_state::V4_LAYER_COUNT || expected.len() != v4_state::V4_LAYER_COUNT {
        return Err("post-MTP teacher-state clobber: lost a layer store".into());
    }
    for (layer, (state, (expected_committed_rows, expected))) in
        states.iter().zip(expected).enumerate()
    {
        let state = state
            .as_ref()
            .ok_or("post-MTP teacher-state clobber: found an empty layer store")?;
        if state.layer != layer {
            return Err("post-MTP teacher-state clobber: found a misordered layer store".into());
        }
        let actual = generic_device_persistent_state_audit(stream, state)?;
        check_post_mtp_persistent_state(
            layer,
            position,
            *expected_committed_rows,
            expected,
            state.committed_rows,
            &actual,
        )?;
    }
    Ok(u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX))
}

pub(crate) fn first_f32_bits_mismatch(left: &[f32], right: &[f32]) -> Option<usize> {
    left.iter()
        .zip(right)
        .position(|(left, right)| left.to_bits() != right.to_bits())
        .or_else(|| (left.len() != right.len()).then_some(left.len().min(right.len())))
}

/// Diagnostic only: persistent Compressor state remains a strict FP32-bit
/// gate.  When it fails, record enough numerics to distinguish a slot/layout
/// fault from a bounded reduction-order difference before changing policy.
pub(crate) fn f32_state_mismatch_diagnostic(expected: &[f32], actual: &[f32], index: usize) -> String {
    let expected_value = expected.get(index).copied();
    let actual_value = actual.get(index).copied();
    let (mut max_abs, mut max_rel, mut max_ulp) = (0.0_f32, 0.0_f32, 0_u32);
    let mut bf16_first = None;
    for (element, (&expected, &actual)) in expected.iter().zip(actual).enumerate() {
        if expected.is_finite() && actual.is_finite() {
            let abs = (expected - actual).abs();
            max_abs = max_abs.max(abs);
            max_rel = max_rel.max(abs / expected.abs().max(f32::MIN_POSITIVE));
            max_ulp = max_ulp.max(f32_ulp_distance(expected, actual));
        } else if expected.to_bits() != actual.to_bits() {
            max_abs = f32::INFINITY;
            max_rel = f32::INFINITY;
            max_ulp = u32::MAX;
        }
        if bf16_first.is_none()
            && v4_router::f32_to_bf16_rne(expected) != v4_router::f32_to_bf16_rne(actual)
        {
            bf16_first = Some(element);
        }
    }
    let render = |value: Option<f32>| match value {
        Some(value) => format!("value={value:e} bits=0x{:08x}", value.to_bits()),
        None => "missing".to_owned(),
    };
    let first_abs = match (expected_value, actual_value) {
        (Some(expected), Some(actual)) => (expected - actual).abs(),
        _ => f32::INFINITY,
    };
    let first_rel = match (expected_value, actual_value) {
        (Some(expected), Some(_actual)) => first_abs / expected.abs().max(f32::MIN_POSITIVE),
        _ => f32::INFINITY,
    };
    let first_ulp = match (expected_value, actual_value) {
        (Some(expected), Some(actual)) if expected.is_finite() && actual.is_finite() => {
            f32_ulp_distance(expected, actual)
        }
        (Some(expected), Some(actual)) if expected.to_bits() == actual.to_bits() => 0,
        _ => u32::MAX,
    };
    format!(
        "expected_{} actual_{} first_abs={first_abs:e} first_rel={first_rel:e} first_ulp={first_ulp} max_abs={max_abs:e} max_rel={max_rel:e} max_ulp={max_ulp} bf16_rne_all_equal={} bf16_rne_first_mismatch={:?} expected_len={} actual_len={}",
        render(expected_value),
        render(actual_value),
        bf16_first.is_none() && expected.len() == actual.len(),
        bf16_first,
        expected.len(),
        actual.len(),
    )
}

pub(crate) fn check_persistent_state_audit(
    layer: usize,
    position: usize,
    expected: &PersistentStateAudit,
    actual: &PersistentStateAudit,
) -> Result<(), Box<dyn Error>> {
    check_persistent_state_audit_with_compressor_state(layer, position, expected, actual, true)
}

pub(crate) fn check_short_decode_full_layer_audit(
    layer: usize,
    position: usize,
    expected: &RealLayerRun,
    expected_state: &PersistentStateAudit,
    actual: &RealLayerRun,
    actual_state: &PersistentStateAudit,
) -> Result<(), Box<dyn Error>> {
    if let Some(index) = first_mismatch(&expected.attention_output, &actual.attention_output) {
        return Err(format!(
            "short decode layer {layer} position {position} attention post mismatch at {index}"
        )
        .into());
    }
    if expected.attention_topk != actual.attention_topk {
        return Err(format!(
            "short decode layer {layer} position {position} attention top-k mismatch"
        )
        .into());
    }
    check_persistent_state_audit(layer, position, expected_state, actual_state)?;
    if expected.routes.ids != actual.routes.ids
        || first_f32_bits_mismatch(&expected.routes.weights, &actual.routes.weights).is_some()
    {
        return Err(
            format!("short decode layer {layer} position {position} route mismatch").into(),
        );
    }
    let expected_routed = expected
        .routed_audit
        .as_ref()
        .ok_or("short decode host oracle omitted routed contribution audit")?;
    let actual_routed = actual
        .routed_audit
        .as_ref()
        .ok_or("short decode device candidate omitted routed contribution audit")?;
    if expected_routed.after_attention_digest != actual_routed.after_attention_digest
        || expected_routed.ffn_input_digest != actual_routed.ffn_input_digest
        || expected_routed.shared_digest != actual_routed.shared_digest
        || expected_routed.moe_digest != actual_routed.moe_digest
    {
        return Err(format!(
            "short decode layer {layer} position {position} routed stage mismatch"
        )
        .into());
    }
    if expected_routed.contributions.len() != actual_routed.contributions.len()
        || expected_routed
            .contributions
            .iter()
            .zip(&actual_routed.contributions)
            .any(|(left, right)| first_f32_bits_mismatch(left, right).is_some())
    {
        return Err(format!(
            "short decode layer {layer} position {position} expert output mismatch"
        )
        .into());
    }
    if let Some(index) = first_mismatch(&expected.output, &actual.output) {
        return Err(format!(
            "short decode layer {layer} position {position} canonical output mismatch at {index}"
        )
        .into());
    }
    Ok(())
}

/// HCA never reads Compressor's rolling FP32 slots after committing them.  On
/// a reused HCA allocation their unwritten tail is intentionally unspecified;
/// audit the observable KV/cache state and complete attention instead.  CSA
/// retains the full FP32 check because its 3->4 boundary reads the overlap.
pub(crate) fn check_persistent_state_audit_with_compressor_state(
    layer: usize,
    position: usize,
    expected: &PersistentStateAudit,
    actual: &PersistentStateAudit,
    require_compressor_state: bool,
) -> Result<(), Box<dyn Error>> {
    for (name, mismatch) in [
        ("kv", first_mismatch(&expected.kv, &actual.kv)),
        (
            "compressed_kv",
            first_mismatch(&expected.compressed_kv, &actual.compressed_kv),
        ),
        (
            "indexer_kv",
            first_mismatch(&expected.indexer_kv, &actual.indexer_kv),
        ),
    ] {
        if let Some(index) = mismatch {
            return Err(format!(
                "device token-state layer {layer} position {position} {name} first mismatch at {index}"
            )
            .into());
        }
    }
    if !require_compressor_state {
        return Ok(());
    }
    for (name, expected, actual) in [
        ("main_compressor_kv", &expected.main_kv, &actual.main_kv),
        (
            "main_compressor_score",
            &expected.main_score,
            &actual.main_score,
        ),
    ] {
        if let Some(index) = first_f32_bits_mismatch(expected, actual) {
            return Err(format!(
                "device token-state layer {layer} position {position} {name} first mismatch at {index} ({})",
                f32_state_mismatch_diagnostic(expected, actual, index),
            )
            .into());
        }
    }
    for (name, expected, actual) in [
        (
            "indexer_compressor_kv",
            &expected.indexer_main_kv,
            &actual.indexer_main_kv,
        ),
        (
            "indexer_compressor_score",
            &expected.indexer_main_score,
            &actual.indexer_main_score,
        ),
    ] {
        match (expected, actual) {
            (Some(expected), Some(actual)) => {
                if let Some(index) = first_f32_bits_mismatch(expected, actual) {
                    return Err(format!(
                        "device token-state layer {layer} position {position} {name} first mismatch at {index} ({})",
                        f32_state_mismatch_diagnostic(expected, actual, index),
                    )
                    .into());
                }
            }
            (None, None) => {}
            _ => {
                return Err(format!(
                    "device token-state layer {layer} position {position} {name} presence mismatch"
                )
                .into());
            }
        }
    }
    Ok(())
}

pub(crate) fn persistent_state_audit_json(audit: &PersistentStateAudit) -> serde_json::Value {
    serde_json::json!({
        "kv_bf16_digest": format!("{:016x}", fnv1a_u16(&audit.kv)),
        "compressed_kv_bf16_digest": format!("{:016x}", fnv1a_u16(&audit.compressed_kv)),
        "indexer_kv_bf16_digest": format!("{:016x}", fnv1a_u16(&audit.indexer_kv)),
        "main_compressor_kv_f32_digest": format!("{:016x}", fnv1a_f32(&audit.main_kv)),
        "main_compressor_score_f32_digest": format!("{:016x}", fnv1a_f32(&audit.main_score)),
        "indexer_compressor_kv_f32_digest": audit.indexer_main_kv.as_ref().map(|value| format!("{:016x}", fnv1a_f32(value))),
        "indexer_compressor_score_f32_digest": audit.indexer_main_score.as_ref().map(|value| format!("{:016x}", fnv1a_f32(value))),
    })
}

pub(crate) fn run_attention_arm_a(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    runtime: &LoadedLayerRuntime,
    inputs: &[Vec<u16>; 2],
) -> Result<(AttentionAudit, u64), Box<dyn Error>> {
    let mut persistent = v4_state::V4SerialPersistentState::new(2)?;
    let mut outputs = Vec::with_capacity(2);
    let mut topk_digests = Vec::with_capacity(2);
    let mut stage_digests = Vec::with_capacity(2);
    let started = std::time::Instant::now();
    for position in 0..2 {
        persistent.transaction(runtime.weights.layer, |state| {
            let run = step_real_layer_attention_rows(
                module,
                stream,
                None,
                runtime,
                &inputs[position],
                position,
                state,
                None,
                None,
            )?;
            stage_digests.push(run.stage_digests);
            outputs.push(run.output);
            topk_digests.push(run.attention_topk_digest);
            Ok(())
        })?;
    }
    let elapsed_ns = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
    let state = persistent
        .layer(runtime.weights.layer)
        .ok_or("canonical attention state disappeared")?;
    let main = state.compressor();
    let indexer = state.indexer();
    Ok((
        AttentionAudit {
            outputs: outputs
                .try_into()
                .map_err(|_| "canonical attention changed output row count")?,
            stage_digests: stage_digests
                .try_into()
                .map_err(|_| "canonical attention changed stage row count")?,
            topk_digests: topk_digests
                .try_into()
                .map_err(|_| "canonical attention changed top-k row count")?,
            kv: state.kv_bf16().to_vec(),
            main_kv: main.kv().to_vec(),
            main_score: main.score().to_vec(),
            indexer_kv: indexer.map(|value| value.kv().to_vec()),
            indexer_score: indexer.map(|value| value.score().to_vec()),
        },
        elapsed_ns,
    ))
}

pub(crate) fn device_attention_stage_digests(
    stream: &CudaStream,
    state: &DeviceTokenState,
) -> Result<Vec<(&'static str, u64)>, Box<dyn Error>> {
    let sparse_scratch = if state.committed_rows == 1 {
        &state.sparse_pos0
    } else {
        &state.sparse_pos1
    };
    Ok(vec![
        (
            "mhc_collapsed",
            fnv1a_u16(&state.collapsed.to_host_vec(stream)?),
        ),
        ("mhc_post", fnv1a_f32(&state.post.to_host_vec(stream)?)),
        ("mhc_comb", fnv1a_f32(&state.comb.to_host_vec(stream)?)),
        (
            "attn_input",
            fnv1a_u16(&state.attn_input.to_host_vec(stream)?),
        ),
        (
            "q_lora_raw",
            fnv1a_u16(&state.q_lora_raw.to_host_vec(stream)?),
        ),
        ("q_lora", fnv1a_u16(&state.q_lora.to_host_vec(stream)?)),
        ("q_raw", fnv1a_u16(&state.q_raw.to_host_vec(stream)?)),
        ("q_norm", fnv1a_u16(&state.q_norm.to_host_vec(stream)?)),
        ("q_rope", fnv1a_u16(&state.q_rope.to_host_vec(stream)?)),
        ("kv_raw", fnv1a_u16(&state.kv_raw.to_host_vec(stream)?)),
        ("kv_norm", fnv1a_u16(&state.kv_norm.to_host_vec(stream)?)),
        ("kv_rope", fnv1a_u16(&state.kv_rope.to_host_vec(stream)?)),
        ("kv_ready", fnv1a_u16(&state.kv_ready.to_host_vec(stream)?)),
        (
            "compressor_kv",
            fnv1a_f32(&state.compressor.kv.to_host_vec(stream)?),
        ),
        (
            "compressor_score",
            fnv1a_f32(&state.compressor.score.to_host_vec(stream)?),
        ),
        (
            "compressor_score_plus_ape",
            fnv1a_f32(&state.compressor.score_plus_ape.to_host_vec(stream)?),
        ),
        (
            "sparse_raw_scores",
            fnv1a_f32(&sparse_scratch.scores.to_host_vec(stream)?),
        ),
        (
            "sparse_probabilities",
            fnv1a_f32(&sparse_scratch.probabilities.to_host_vec(stream)?),
        ),
        (
            "sparse_denominators",
            fnv1a_f32(&sparse_scratch.denominators.to_host_vec(stream)?),
        ),
        (
            "sparse_valid_counts",
            fnv1a_i32(
                &sparse_scratch
                    .valid_counts
                    .to_host_vec(stream)?
                    .into_iter()
                    .map(|value| value as i32)
                    .collect::<Vec<_>>(),
            ),
        ),
        ("sparse", fnv1a_u16(&state.sparse.to_host_vec(stream)?)),
        (
            "sparse_inverse",
            fnv1a_u16(&state.sparse_inverse.to_host_vec(stream)?),
        ),
        ("wo_a", fnv1a_u16(&state.wo_a.to_host_vec(stream)?)),
        (
            "attention",
            fnv1a_u16(&state.attention.to_host_vec(stream)?),
        ),
        (
            "attention_post",
            fnv1a_u16(&state.after_attention.to_host_vec(stream)?),
        ),
    ])
}

pub(crate) fn run_attention_arm_b(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    runtime: &LoadedLayerRuntime,
    inputs: &[Vec<u16>; 2],
    state: &mut DeviceTokenState,
    rope_lut: &DeviceRequestRopeLut,
    sparse_arm: SparseAttentionArm,
) -> Result<(AttentionAudit, u64, [DeviceAttentionRun; 2]), Box<dyn Error>> {
    let run0 = run_wavefront_l3_l4_attention_row(
        module, stream, runtime, state, rope_lut, &inputs[0], 0, sparse_arm,
    )?;
    let stages0 = device_attention_stage_digests(stream, state)?;
    let run1 = run_wavefront_l3_l4_attention_row(
        module, stream, runtime, state, rope_lut, &inputs[1], 1, sparse_arm,
    )?;
    let stages1 = device_attention_stage_digests(stream, state)?;
    let elapsed_ns = run0.elapsed_ns.saturating_add(run1.elapsed_ns);
    // These downloads are a post-timing correctness audit. The measured path
    // above has only the terminal hidden D2H recorded in each row result.
    let kv = state.kv_cache.to_host_vec(stream)?;
    let main_kv = state.compressor.persistent_kv.to_host_vec(stream)?;
    let main_score = state.compressor.persistent_score.to_host_vec(stream)?;
    let indexer_kv = state
        .indexer_compressor
        .as_ref()
        .map(|value| value.persistent_kv.to_host_vec(stream))
        .transpose()?;
    let indexer_score = state
        .indexer_compressor
        .as_ref()
        .map(|value| value.persistent_score.to_host_vec(stream))
        .transpose()?;
    let topk_digests = [
        fnv1a_i32(&v4_state::window_topk_cpu(1, 0, v4_state::KV_WINDOW)?.0),
        fnv1a_i32(&v4_state::window_topk_cpu(1, 1, v4_state::KV_WINDOW)?.0),
    ];
    Ok((
        AttentionAudit {
            outputs: [run0.output.clone(), run1.output.clone()],
            stage_digests: [stages0, stages1],
            topk_digests,
            kv: kv[..state.committed_rows * v4_state::Q_HEAD_DIM].to_vec(),
            main_kv,
            main_score,
            indexer_kv,
            indexer_score,
        },
        elapsed_ns,
        [run0, run1],
    ))
}

pub(crate) fn first_mismatch<T: PartialEq>(left: &[T], right: &[T]) -> Option<usize> {
    left.iter()
        .zip(right)
        .position(|(left, right)| left != right)
        .or_else(|| (left.len() != right.len()).then_some(left.len().min(right.len())))
}

pub(crate) fn check_compressed_boundary_bf16(
    stage: &str,
    layer: usize,
    position: usize,
    first_row: usize,
    row_width: usize,
    expected: &[u16],
    actual: &[u16],
) -> Result<(), Box<dyn Error>> {
    if row_width == 0 {
        return Err("compressed boundary audit row width is zero".into());
    }
    if let Some(flat_index) = first_mismatch(expected, actual) {
        let row = first_row + flat_index / row_width;
        let index = flat_index % row_width;
        let render = |value: Option<&u16>| {
            value.map_or_else(|| "missing".to_owned(), |value| format!("0x{value:04x}"))
        };
        return Err(format!(
            "stage={stage} layer={layer} position={position} row={row} index={index} old={} new={}",
            render(expected.get(flat_index)),
            render(actual.get(flat_index)),
        )
        .into());
    }
    Ok(())
}

pub(crate) fn targeted_stage_u16(name: &str, expected: &[u16], actual: &[u16]) -> serde_json::Value {
    let mismatch = first_mismatch(expected, actual);
    serde_json::json!({
        "stage": name,
        "exact": mismatch.is_none(),
        "elements": {"reference": expected.len(), "candidate": actual.len()},
        "digest": {"reference": format!("{:016x}", fnv1a_u16(expected)), "candidate": format!("{:016x}", fnv1a_u16(actual))},
        "first_mismatch": mismatch.map(|index| serde_json::json!({
            "index": index,
            "reference_bits": expected.get(index).map(|value| format!("{value:04x}")),
            "candidate_bits": actual.get(index).map(|value| format!("{value:04x}")),
            "reference_bf16": expected.get(index).map(|&value| format!("{:.9e}", v4_router::bf16_to_f32(value))),
            "candidate_bf16": actual.get(index).map(|&value| format!("{:.9e}", v4_router::bf16_to_f32(value))),
        })),
    })
}

pub(crate) fn targeted_stage_f32(name: &str, expected: &[f32], actual: &[f32]) -> serde_json::Value {
    let mismatch = expected
        .iter()
        .zip(actual)
        .position(|(&expected, &actual)| expected.to_bits() != actual.to_bits())
        .or_else(|| (expected.len() != actual.len()).then_some(expected.len().min(actual.len())));
    serde_json::json!({
        "stage": name,
        "exact": mismatch.is_none(),
        "elements": {"reference": expected.len(), "candidate": actual.len()},
        "digest": {"reference": format!("{:016x}", fnv1a_f32(expected)), "candidate": format!("{:016x}", fnv1a_f32(actual))},
        "first_mismatch": mismatch.map(|index| serde_json::json!({
            "index": index,
            "reference_bits": expected.get(index).map(|value| format!("{:08x}", value.to_bits())),
            "candidate_bits": actual.get(index).map(|value| format!("{:08x}", value.to_bits())),
            "reference_f32": expected.get(index).map(|value| format!("{value:.9e}")),
            "candidate_f32": actual.get(index).map(|value| format!("{value:.9e}")),
        })),
    })
}

pub(crate) fn targeted_stage_i32(name: &str, expected: &[i32], actual: &[i32]) -> serde_json::Value {
    let mismatch = first_mismatch(expected, actual);
    serde_json::json!({
        "stage": name,
        "exact": mismatch.is_none(),
        "elements": {"reference": expected.len(), "candidate": actual.len()},
        "digest": {"reference": format!("{:016x}", fnv1a_i32(expected)), "candidate": format!("{:016x}", fnv1a_i32(actual))},
        "first_mismatch": mismatch.map(|index| serde_json::json!({
            "index": index,
            "reference": expected.get(index),
            "candidate": actual.get(index),
        })),
    })
}

pub(crate) fn targeted_stage_u32(name: &str, expected: &[u32], actual: &[u32]) -> serde_json::Value {
    let expected_i32 = expected
        .iter()
        .map(|&value| value as i32)
        .collect::<Vec<_>>();
    let actual_i32 = actual.iter().map(|&value| value as i32).collect::<Vec<_>>();
    let mismatch = first_mismatch(expected, actual);
    serde_json::json!({
        "stage": name,
        "exact": mismatch.is_none(),
        "elements": {"reference": expected.len(), "candidate": actual.len()},
        "digest": {"reference": format!("{:016x}", fnv1a_i32(&expected_i32)), "candidate": format!("{:016x}", fnv1a_i32(&actual_i32))},
        "first_mismatch": mismatch.map(|index| serde_json::json!({
            "index": index,
            "reference": expected.get(index),
            "candidate": actual.get(index),
        })),
    })
}

pub(crate) fn check_attention_audit(
    layer: usize,
    expected: &AttentionAudit,
    actual: &AttentionAudit,
) -> Result<(), Box<dyn Error>> {
    for position in 0..2 {
        let comparable = |stages: &[(&'static str, u64)]| {
            stages
                .iter()
                .copied()
                .filter(|(name, _)| {
                    !matches!(
                        *name,
                        "sparse_raw_scores"
                            | "sparse_probabilities"
                            | "sparse_denominators"
                            | "sparse_valid_counts"
                    )
                })
                .collect::<Vec<_>>()
        };
        let expected_stages = comparable(&expected.stage_digests[position]);
        let actual_stages = comparable(&actual.stage_digests[position]);
        if expected_stages.len() != actual_stages.len() {
            return Err(format!(
                "CDQW layer {layer} position {position} stage audit length mismatch"
            )
            .into());
        }
        for ((expected_name, expected_digest), (actual_name, actual_digest)) in
            expected_stages.iter().zip(&actual_stages)
        {
            if expected_name != actual_name || expected_digest != actual_digest {
                return Err(format!(
                    "CDQW layer {layer} position {position} first stage mismatch: \
                     A {expected_name}={expected_digest:016x}, \
                     B {actual_name}={actual_digest:016x}"
                )
                .into());
            }
        }
        if let Some(index) = first_mismatch(&expected.outputs[position], &actual.outputs[position])
        {
            return Err(format!(
                "CDQW layer {layer} position {position} attention_post first mismatch at {index}"
            )
            .into());
        }
        if expected.topk_digests[position] != actual.topk_digests[position] {
            return Err(format!(
                "CDQW layer {layer} position {position} sparse top-k digest mismatch"
            )
            .into());
        }
    }
    for (stage, mismatch) in [
        ("kv", first_mismatch(&expected.kv, &actual.kv)),
        (
            "main_compressor_kv",
            first_mismatch(&expected.main_kv, &actual.main_kv),
        ),
        (
            "main_compressor_score",
            first_mismatch(&expected.main_score, &actual.main_score),
        ),
    ] {
        if let Some(index) = mismatch {
            return Err(format!("CDQW layer {layer} {stage} first mismatch at {index}").into());
        }
    }
    match (&expected.indexer_kv, &actual.indexer_kv) {
        (Some(expected), Some(actual)) => {
            if let Some(index) = first_mismatch(expected, actual) {
                return Err(format!(
                    "CDQW layer {layer} indexer_compressor_kv mismatch at {index}"
                )
                .into());
            }
        }
        (None, None) => {}
        _ => return Err(format!("CDQW layer {layer} indexer KV presence mismatch").into()),
    }
    match (&expected.indexer_score, &actual.indexer_score) {
        (Some(expected), Some(actual)) => {
            if let Some(index) = first_mismatch(expected, actual) {
                return Err(format!(
                    "CDQW layer {layer} indexer_compressor_score mismatch at {index}"
                )
                .into());
            }
        }
        (None, None) => {}
        _ => return Err(format!("CDQW layer {layer} indexer score presence mismatch").into()),
    }
    Ok(())
}

pub(crate) fn run_attention_arm_a_timed(
    context: &std::sync::Arc<CudaContext>,
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    runtime: &LoadedLayerRuntime,
    inputs: &[Vec<u16>; 2],
) -> Result<(AttentionAudit, u64, u64), Box<dyn Error>> {
    let start = context.new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
    let end = context.new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
    start.record(stream)?;
    let (audit, wall_ns) = run_attention_arm_a(module, stream, runtime, inputs)?;
    end.record(stream)?;
    end.synchronize()?;
    Ok((
        audit,
        wall_ns,
        (start.elapsed_ms(&end)? as f64 * 1_000_000.0) as u64,
    ))
}
