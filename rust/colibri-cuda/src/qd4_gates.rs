use super::*;

pub(crate) fn qd4_lane_count(enabled: bool) -> usize {
    usize::from(enabled) * V4_QD4_LANES
}

pub(crate) fn validate_qd4_lane_reserve(
    host_available: usize,
    device_available: usize,
) -> Result<(), Box<dyn Error>> {
    if host_available < V4_QD4_PINNED_HOST_BYTES || device_available < V4_QD4_DEVICE_BYTES {
        return Err("persistent QD4 lanes lack the explicit pinned-host/device reserve".into());
    }
    Ok(())
}

pub(crate) fn qd4_expected_routed_loads(layers: usize, prompt_len: usize) -> Result<usize, Box<dyn Error>> {
    layers
        .checked_mul(prompt_len)
        .and_then(|rows| rows.checked_mul(v4_router::V4_TOPK))
        .ok_or_else(|| "QD4 routed load count overflow".into())
}

pub(crate) fn gpu_telemetry_snapshot() -> Result<serde_json::Value, Box<dyn Error>> {
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=temperature.gpu,power.draw,clocks.sm,clocks.mem",
            "--format=csv,noheader,nounits",
            "--id=0",
        ])
        .output()?;
    if !output.status.success() {
        return Err("nvidia-smi telemetry query failed".into());
    }
    let fields = std::str::from_utf8(&output.stdout)?
        .lines()
        .next()
        .ok_or("nvidia-smi telemetry is empty")?
        .split(',')
        .map(str::trim)
        .collect::<Vec<_>>();
    if fields.len() != 4 {
        return Err("nvidia-smi telemetry format changed".into());
    }
    Ok(serde_json::json!({
        "temperature_c": fields[0],
        "power_w": fields[1],
        "sm_clock_mhz": fields[2],
        "memory_clock_mhz": fields[3],
    }))
}

pub(crate) struct Qd4StationaryGatePosition {
    pub(crate) position: usize,
    pub(crate) route_ids: Vec<i32>,
    pub(crate) route_weight_bits: Vec<u32>,
    pub(crate) attention_topk_digest: u64,
    pub(crate) attention_topk_per_row: usize,
    pub(crate) after_attention_digest: u64,
    pub(crate) ffn_input_digest: u64,
    pub(crate) shared_digest: u64,
    pub(crate) contribution_digests: Vec<u64>,
    pub(crate) contributions: Vec<Vec<f32>>,
    pub(crate) moe_digest: u64,
    pub(crate) hidden_digest: u64,
    pub(crate) state: v4_state::V4LayerPersistentState,
}

pub(crate) struct Qd4StationaryGateArm {
    pub(crate) positions: Vec<Qd4StationaryGatePosition>,
    pub(crate) routed_wall_ns: u64,
    pub(crate) routed_source_bytes: usize,
    pub(crate) routed_h2d_bytes: usize,
    pub(crate) startup_setup_ns: u64,
    pub(crate) steady_slot_allocations: usize,
    pub(crate) device_used_after_startup_bytes: usize,
    pub(crate) worker_errors: usize,
    pub(crate) immutable_cache: Option<Qd4ImmutableExpertCacheTelemetry>,
}

pub(crate) fn qd4_gate_state_json(state: &v4_state::V4LayerPersistentState) -> serde_json::Value {
    let main = state.compressor();
    let indexer = state.indexer();
    serde_json::json!({
        "kv_bf16_digest": format!("{:016x}", fnv1a_u16(state.kv_bf16())),
        "compressed_kv_bf16_digest": format!("{:016x}", fnv1a_u16(state.compressed_kv_bf16())),
        "indexer_kv_bf16_digest": format!("{:016x}", fnv1a_u16(state.indexer_kv_bf16())),
        "main_compressor_kv_f32_digest": format!("{:016x}", fnv1a_f32(main.kv())),
        "main_compressor_score_f32_digest": format!("{:016x}", fnv1a_f32(main.score())),
        "indexer_compressor_kv_f32_digest": indexer.map(|value| format!("{:016x}", fnv1a_f32(value.kv()))),
        "indexer_compressor_score_f32_digest": indexer.map(|value| format!("{:016x}", fnv1a_f32(value.score()))),
        "committed_kv_rows": state.committed_kv_rows(),
    })
}

pub(crate) fn qd4_gate_position_json(
    layer: usize,
    pair: usize,
    position: &Qd4StationaryGatePosition,
) -> serde_json::Value {
    serde_json::json!({
        "event": "position",
        "layer": layer,
        "pair": pair,
        "token_position": position.position,
        "route_ids": position.route_ids,
        "route_weight_bits": position.route_weight_bits,
        "attention_topk_digest": format!("{:016x}", position.attention_topk_digest),
        "attention_topk_per_row": position.attention_topk_per_row,
        "after_attention_bf16_digest": format!("{:016x}", position.after_attention_digest),
        "ffn_input_bf16_digest": format!("{:016x}", position.ffn_input_digest),
        "shared_f32_digest": format!("{:016x}", position.shared_digest),
        "route_slot_contribution_f32_digests": position.contribution_digests.iter().map(|value| format!("{value:016x}")).collect::<Vec<_>>(),
        "canonical_moe_bf16_digest": format!("{:016x}", position.moe_digest),
        "post_hc_hidden_bf16_digest": format!("{:016x}", position.hidden_digest),
        "persistent_state": qd4_gate_state_json(&position.state),
        "route_slot_f32_values_checked_bitwise": true,
    })
}

pub(crate) fn qd4_gate_f32_bits_equal(left: &[f32], right: &[f32]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(left, right)| left.to_bits() == right.to_bits())
}

pub(crate) fn qd4_stationary_control_first(pair: usize) -> bool {
    matches!(pair, 0 | 3)
}

pub(crate) fn check_qd4_stationary_gate_pair(
    layer: usize,
    control: &Qd4StationaryGateArm,
    qd4: &Qd4StationaryGateArm,
) -> Result<(), Box<dyn Error>> {
    if control.positions.len() != 5 || qd4.positions.len() != 5 {
        return Err(format!("QD4 stationary layer {layer} did not produce five positions").into());
    }
    for (position, (control, qd4)) in control.positions.iter().zip(&qd4.positions).enumerate() {
        if control.position != position
            || qd4.position != position
            || control.route_ids != qd4.route_ids
            || control.route_weight_bits != qd4.route_weight_bits
            || control.attention_topk_digest != qd4.attention_topk_digest
            || control.attention_topk_per_row != qd4.attention_topk_per_row
            || control.after_attention_digest != qd4.after_attention_digest
            || control.ffn_input_digest != qd4.ffn_input_digest
            || control.shared_digest != qd4.shared_digest
            || control.moe_digest != qd4.moe_digest
            || control.hidden_digest != qd4.hidden_digest
            || !control.state.bits_equal(&qd4.state)
            || control.contribution_digests != qd4.contribution_digests
            || control.contributions.len() != v4_router::V4_TOPK
            || qd4.contributions.len() != v4_router::V4_TOPK
            || !control
                .contributions
                .iter()
                .zip(&qd4.contributions)
                .all(|(control, qd4)| qd4_gate_f32_bits_equal(control, qd4))
        {
            return Err(format!(
                "QD4 stationary layer {layer} position {position} changed canonical routed execution"
            )
            .into());
        }
    }
    Ok(())
}

pub(crate) fn check_qd4_immutable_cache_gate_physical(
    layer: usize,
    arm: &Qd4StationaryGateArm,
) -> Result<(), Box<dyn Error>> {
    let cache = arm
        .immutable_cache
        .as_ref()
        .ok_or("QD4 immutable cache gate candidate omitted cache telemetry")?;
    let expected_routes = 5 * v4_router::V4_TOPK;
    let expert_bytes = v4_expert::V4_ROUTED_EXPERT_BYTES;
    let source_bytes = cache
        .source_loads
        .checked_mul(expert_bytes)
        .ok_or("QD4 immutable cache gate source byte count overflow")?;
    let d2d_bytes = expected_routes
        .checked_mul(expert_bytes)
        .ok_or("QD4 immutable cache gate D2D byte count overflow")?;
    if cache.route_executions != expected_routes
        || cache.source_loads != cache.expert_union_loads
        || cache.source_bytes != source_bytes
        || cache.h2d_loads != cache.source_loads
        || cache.h2d_bytes != cache.source_bytes
        || cache.hits != expected_routes.saturating_sub(cache.source_loads)
        || cache.d2d_copies != expected_routes
        || cache.d2d_bytes != d2d_bytes
        || cache.max_occupancy > V4_QD4_IMMUTABLE_EXPERT_CACHE_SLOTS
        || cache.completed_layers != 1
        || cache.worker_errors != 0
        || cache.fallbacks != 0
        || arm.routed_source_bytes != cache.source_bytes
        || arm.routed_h2d_bytes != cache.h2d_bytes
        || arm.worker_errors != 0
    {
        return Err(format!(
            "QD4 immutable cache gate layer {layer} physical telemetry did not close"
        )
        .into());
    }
    Ok(())
}

pub(crate) fn qd4_gate_prompt_ids(model_dir: &std::path::Path) -> Result<Vec<i64>, Box<dyn Error>> {
    let tokenizer = v4_tokenizer::V4Tokenizer::open(model_dir)?;
    let prompt = v4_chat::format_single_user("hi", v4_chat::ThinkingMode::Chat);
    let ids = tokenizer.encode_raw(&prompt)?;
    if ids.as_slice() != [0, 128_803, 6_366, 128_804, 128_822] {
        return Err("QD4 stationary gate official hi prompt changed".into());
    }
    Ok(ids)
}

pub(crate) fn qd4_gate_initial_hidden(
    model_dir: &std::path::Path,
    token_ids: &[i64],
) -> Result<Vec<Vec<u16>>, Box<dyn Error>> {
    token_ids
        .iter()
        .map(|&token_id| {
            let embedding = v4_state::load(model_dir, &[token_id])?.embedding;
            let mut hidden = Vec::with_capacity(v4_state::HC * v4_state::HIDDEN);
            for _ in 0..v4_state::HC {
                hidden.extend_from_slice(&embedding);
            }
            Ok(hidden)
        })
        .collect()
}

pub(crate) fn prepare_qd4_stationary_gate_inputs(
    context: &std::sync::Arc<CudaContext>,
    module: &kernels::LoadedModule,
    stream: &std::sync::Arc<CudaStream>,
    model_dir: &std::path::Path,
    token_ids: &[i64],
    target_layer: usize,
) -> Result<BTreeMap<usize, Vec<Vec<u16>>>, Box<dyn Error>> {
    if !(3..=4).contains(&target_layer) {
        return Err("QD4 stationary prefix supports only L3 or L4 targets".into());
    }
    let mut hidden = qd4_gate_initial_hidden(model_dir, token_ids)?;
    let mut persistent = v4_state::V4SerialPersistentState::new(token_ids.len())?;
    let mut inputs = BTreeMap::new();
    for layer in 0..target_layer {
        inputs.insert(layer, hidden.clone());
        let (runtime, _) = load_layer_runtime(model_dir, layer, token_ids, stream)?;
        let hash_ids = load_layer_hash_ids(model_dir, layer, token_ids)?;
        // ponytail: this is one cold prefix only; it mirrors current per-layer
        // union retention rather than adding a second input-cache mechanism.
        let mut expert_slots = BTreeMap::new();
        for position in 0..token_ids.len() {
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
                        None,
                        None,
                        Some(&mut expert_slots),
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
            hidden[position] = next.ok_or("QD4 stationary prefix did not produce hidden")?;
        }
    }
    inputs.insert(target_layer, hidden);
    Ok(inputs)
}

pub(crate) fn qd4_stationary_gate_arm_json(arm: &Qd4StationaryGateArm) -> serde_json::Value {
    serde_json::json!({
        "routed_wall_ns": arm.routed_wall_ns,
        "routed_source_bytes": arm.routed_source_bytes,
        "routed_h2d_bytes": arm.routed_h2d_bytes,
        "startup_setup_ns": arm.startup_setup_ns,
        "steady_slot_allocations": arm.steady_slot_allocations,
        "device_used_after_startup_bytes": arm.device_used_after_startup_bytes,
        "worker_errors": arm.worker_errors,
        "immutable_cache": arm.immutable_cache.as_ref().map(|cache| serde_json::json!({
            "route_executions": cache.route_executions,
            "expert_union_loads": cache.expert_union_loads,
            "source_loads": cache.source_loads,
            "source_bytes": cache.source_bytes,
            "h2d_loads": cache.h2d_loads,
            "h2d_bytes": cache.h2d_bytes,
            "hits": cache.hits,
            "d2d_copies": cache.d2d_copies,
            "d2d_bytes": cache.d2d_bytes,
            "max_occupancy": cache.max_occupancy,
            "completed_layers": cache.completed_layers,
            "worker_errors": cache.worker_errors,
            "fallbacks": cache.fallbacks,
            "slots": V4_QD4_IMMUTABLE_EXPERT_CACHE_SLOTS,
        })),
    })
}
