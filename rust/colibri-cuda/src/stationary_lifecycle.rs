use super::*;

/// Request-scoped bookkeeping for the generic stationary device path.  It is
/// deliberately data-only: the worker/cache remain the existing
/// `FixedLayerCache` and `run_pinned_fixed_request_worker` implementation.
/// Keeping this boundary small makes the failure rule observable without
/// inventing a second scheduler.
#[derive(Debug)]
pub(crate) struct GenericStationaryLifecycle {
    pub(crate) committed_rows: Vec<usize>,
    pub(crate) poisoned: bool,
    pub(crate) live_leases: usize,
    pub(crate) telemetry: GenericStationaryTelemetry,
}

#[derive(Debug, Default)]
pub(crate) struct GenericStationaryTelemetry {
    pub(crate) layer_commits: usize,
    pub(crate) device_peak_bytes: usize,
    pub(crate) host_peak_bytes: usize,
    pub(crate) pinned_peak_bytes: usize,
    pub(crate) live_lease_peak: usize,
}

/// State-root observation is determined by which authoritative store exists,
/// not by whether a device-attention implementation happens to be enabled.
/// Generic stationary has device-owned state and deliberately no host store.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PersistentStateDigestSource {
    ObservedDevice,
    HostPersistent,
    Unobserved,
}

pub(crate) const fn persistent_state_digest_source(
    observed_device_state: bool,
    generic_device_stationary: bool,
    host_persistent_available: bool,
) -> PersistentStateDigestSource {
    if observed_device_state {
        PersistentStateDigestSource::ObservedDevice
    } else if generic_device_stationary {
        PersistentStateDigestSource::Unobserved
    } else if host_persistent_available {
        PersistentStateDigestSource::HostPersistent
    } else {
        PersistentStateDigestSource::Unobserved
    }
}

impl GenericStationaryLifecycle {
    pub(crate) fn new(layers: usize) -> Result<Self, Box<dyn Error>> {
        if layers != v4_state::V4_LAYER_COUNT {
            return Err("generic stationary lifecycle requires all 61 layer stores".into());
        }
        Ok(Self {
            committed_rows: vec![0; layers],
            poisoned: false,
            live_leases: 0,
            telemetry: GenericStationaryTelemetry::default(),
        })
    }

    /// Checks the causal boundary before a layer touches its persistent store.
    pub(crate) fn begin_layer(&self, layer: usize, position: usize) -> Result<(), Box<dyn Error>> {
        if self.poisoned || self.committed_rows.get(layer).copied() != Some(position) {
            return Err("generic stationary layer is not at its causal commit boundary".into());
        }
        Ok(())
    }

    /// The sole semantic commit point.  Call only after attention, FFN, and
    /// the terminal hidden row have all succeeded.
    pub(crate) fn commit_layer(&mut self, layer: usize, position: usize) -> Result<(), Box<dyn Error>> {
        self.begin_layer(layer, position)?;
        let next = position
            .checked_add(1)
            .ok_or("generic stationary committed-row overflow")?;
        *self
            .committed_rows
            .get_mut(layer)
            .ok_or("generic stationary layer is outside the 61 stores")? = next;
        self.telemetry.layer_commits = self.telemetry.layer_commits.saturating_add(1);
        Ok(())
    }

    /// A failed prepared row never changes any committed counter.  The request
    /// is fail-closed so no later row can observe a partially prepared cache.
    pub(crate) fn poison(&mut self) {
        self.poisoned = true;
    }

    pub(crate) fn acquire_lease(&mut self) {
        self.live_leases = self.live_leases.saturating_add(1);
        self.telemetry.live_lease_peak = self.telemetry.live_lease_peak.max(self.live_leases);
    }

    pub(crate) fn release_lease(&mut self) -> Result<(), Box<dyn Error>> {
        self.live_leases = self
            .live_leases
            .checked_sub(1)
            .ok_or("generic stationary released a non-live lease")?;
        Ok(())
    }

    pub(crate) fn observe_peaks(&mut self, device: usize, host: usize, pinned: usize) {
        self.telemetry.device_peak_bytes = self.telemetry.device_peak_bytes.max(device);
        self.telemetry.host_peak_bytes = self.telemetry.host_peak_bytes.max(host);
        self.telemetry.pinned_peak_bytes = self.telemetry.pinned_peak_bytes.max(pinned);
    }

    pub(crate) fn assert_drained(&self) -> Result<(), Box<dyn Error>> {
        if self.live_leases != 0 {
            return Err("generic stationary request ended with a live fixed lease".into());
        }
        Ok(())
    }
}

pub(crate) fn should_retain_layer_expert(retained: usize) -> bool {
    retained < V4_LAYER_EXPERT_RETAINED_SLOTS
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GreedyStopReason {
    Eos,
    MaxNewTokens,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GreedyDecision {
    IgnorePrompt,
    EmitAndContinue,
    EmitAndStop(GreedyStopReason),
}

pub(crate) fn greedy_argmax_decision(
    prompt_len: usize,
    token_position: usize,
    generated_count: usize,
    max_new_tokens: usize,
    argmax: i64,
) -> GreedyDecision {
    if token_position + 1 < prompt_len {
        GreedyDecision::IgnorePrompt
    } else if argmax == V4_EOS_TOKEN_ID {
        GreedyDecision::EmitAndStop(GreedyStopReason::Eos)
    } else if generated_count + 1 == max_new_tokens {
        GreedyDecision::EmitAndStop(GreedyStopReason::MaxNewTokens)
    } else {
        GreedyDecision::EmitAndContinue
    }
}

pub(crate) fn b1_needs_final_head(prompt_len: usize, token_position: usize) -> bool {
    token_position + 1 >= prompt_len
}

pub(crate) fn chat_input_conflict(
    chat_user_present: bool,
    raw_prompt_present: bool,
    id_input_present: bool,
) -> bool {
    chat_user_present && (raw_prompt_present || id_input_present)
}

pub(crate) fn greedy_forward_tokens(prompt_len: usize, greedy_steps: usize) -> Result<usize, Box<dyn Error>> {
    if prompt_len == 0 {
        return Err("greedy decoding requires at least one prompt token".into());
    }
    prompt_len
        .checked_add(greedy_steps.saturating_sub(1))
        .ok_or_else(|| "greedy decoding token count overflow".into())
}

pub(crate) fn validate_layer_stationary_prefill(
    prompt_len: usize,
    greedy_steps: usize,
    max_layers: usize,
    cache_is_off: bool,
    qd4_replay_bench: bool,
    short_device_decode: bool,
) -> Result<(), Box<dyn Error>> {
    let targeted_attention_audit = optional_one_flag("COLIBRI_V4_TARGETED_ATTENTION_AUDIT")?;
    if prompt_len == 0
        || if short_device_decode {
            !(2..=V4_SHORT_DEVICE_DECODE_MAX_NEW_TOKENS).contains(&greedy_steps)
        } else {
            greedy_steps != 1
        }
        || if targeted_attention_audit {
            max_layers != 3 || short_device_decode
        } else {
            max_layers != v4_state::V4_LAYER_COUNT
        }
        || !cache_is_off
        || qd4_replay_bench
    {
        return Err(
            "layer-stationary prefill requires a nonempty prompt, greedy_steps=1 (or 2..=5 for short device decode), full61 (or exactly 3 layers for the targeted attention audit), fixed cache off, and the QD4 replay benchmark off"
                .into(),
        );
    }
    Ok(())
}

pub(crate) fn short_device_decode_rows(
    prompt_len: usize,
    greedy_steps: usize,
    enabled: bool,
) -> Result<usize, Box<dyn Error>> {
    if !enabled {
        return Ok(prompt_len);
    }
    if !(2..=V4_SHORT_DEVICE_DECODE_MAX_NEW_TOKENS).contains(&greedy_steps) {
        return Err("short device decode requires COLIBRI_V4_GREEDY_STEPS=2..=5".into());
    }
    let rows = greedy_forward_tokens(prompt_len, greedy_steps)?;
    if rows > V4_SHORT_DEVICE_DECODE_MAX_ROWS {
        return Err("short device decode exceeds its nine-row request-state ceiling".into());
    }
    Ok(rows)
}

pub(crate) fn short_device_decode_static_check() -> Result<(), Box<dyn Error>> {
    for (topk_per_row, expected_score_reuse, scratch_topk_capacity) in [
        (1, false, 1),
        (SM89_SCORE_REUSE_MIN_TOPK - 1, false, 3),
        (SM89_SCORE_REUSE_MIN_TOPK, true, 4),
        (v4_state::KV_WINDOW, true, v4_state::KV_WINDOW),
        (v4_state::KV_WINDOW + 1, true, v4_state::KV_WINDOW + 2),
    ] {
        if SparseAttentionArm::Sm89Crossover.uses_score_reuse(topk_per_row) != expected_score_reuse
            || scratch_topk_capacity < topk_per_row
        {
            return Err("short device decode sm89 crossover/workspace boundary drift".into());
        }
    }
    let state_bytes = 31usize
        .checked_mul(V4_DEVICE_TOKEN_STATE_HCA_BYTES)
        .and_then(|bytes| {
            30usize
                .checked_mul(V4_DEVICE_TOKEN_STATE_MAX_BYTES)
                .and_then(|csa| bytes.checked_add(csa))
        })
        .ok_or("short device decode state ledger overflow")?;
    let rope_bytes = 2usize
        .checked_mul(V4_SHORT_DEVICE_DECODE_MAX_ROWS)
        .and_then(|bytes| bytes.checked_mul(v4_state::Q_ROPE_DIM))
        .and_then(|bytes| bytes.checked_mul(std::mem::size_of::<f32>()))
        .ok_or("short device decode RoPE ledger overflow")?;
    let total_bytes = [
        V4_QD4_DEVICE_BYTES,
        V4_QD4_IMMUTABLE_EXPERT_CACHE_DEVICE_BYTES,
        494_348_312,
        state_bytes,
        rope_bytes,
        V4_TID2EID_TRANSIENT_MAX_BYTES,
        V4_DEVICE_CAPACITY_HEADROOM_BYTES,
    ]
    .into_iter()
    .try_fold(0usize, |total, bytes| total.checked_add(bytes))
    .ok_or("short device decode capacity ledger overflow")?;
    if short_device_decode_rows(5, 2, true)? != 6
        || short_device_decode_rows(5, 5, true)? != 9
        || csa_compressed_rows_after(3) != 1
        || csa_compressed_rows_after(6) != 1
        || csa_compressed_rows_after(7) != 2
        || csa_compressed_rows_after(8) != 2
        || state_bytes != 3_763_828_054
        || rope_bytes != 4_608
        || total_bytes != 7_673_270_174
    {
        return Err("short device decode static contract mismatch".into());
    }
    strict_compressor_host_arena_static_check()?;
    let mut compressor = v4_state::V4CompressorState::new_decode(4, 1)?;
    let mut compressed_rows = 0;
    for position in 0..9 {
        let prepared = v4_state::CompressorPrepared {
            kv_f32: vec![position as f32 + 1.0; 2],
            score_f32: vec![position as f32 + 2.0; 2],
            score_plus_ape_f32: vec![position as f32 + 3.0; 2],
        };
        let (next, pooled) = compressor.prepare_decode_row_and_pool(&prepared, position)?;
        compressed_rows += usize::from(pooled.is_some());
        if (position == 3 || position == 7) != pooled.is_some() {
            return Err("short device decode CSA compressor boundary drift".into());
        }
        compressor = next;
    }
    if compressed_rows != 2 {
        return Err("short device decode CSA compressed row count drift at position 8".into());
    }
    stationary_raw_retention_static_check()?;
    println!(
        "{{\"status\":\"PASS\",\"mode\":\"v4-short-device-decode-static-check\",\"rows\":9,\"state_bytes\":{state_bytes},\"rope_bytes\":{rope_bytes},\"capacity_bytes\":{total_bytes},\"csa_compressed_rows_pos8\":{compressed_rows},\"sm89_crossover\":\"K<4=reference,K>=4=score_reuse\",\"sm89_workspace_cases\":[1,3,4,128,129],\"strict_arena_reuse_repeats\":100,\"stationary_raw_selection\":\"device+host+miss\",\"stationary_raw_lease\":\"exclusive_take_restore\",\"fixed_host_request_ids_cached_bytes\":0,\"prompt_hash_id_bytes_dropped\":240,\"decode_hash_id_bytes_external\":48}}"
    );
    Ok(())
}

pub(crate) fn stationary_raw_retention_static_check() -> Result<(), Box<dyn Error>> {
    const PROMPT_ROWS: usize = 5;
    const DECODE_ROWS: usize = 1;
    let mut router = v4_state::LayerRouterWeights::Hash {
        ids: vec![0; PROMPT_ROWS * v4_router::V4_TOPK],
    };
    let removed_bytes = drop_request_hash_ids_before_fixed_retention(&mut router, PROMPT_ROWS)?;
    let fixed_host_bytes = 1_000_usize;
    let prompt_host_bytes = fixed_host_bytes
        .checked_add(removed_bytes)
        .ok_or("stationary raw static host byte overflow")?;
    let decode_request_bytes = DECODE_ROWS
        .checked_mul(v4_router::V4_TOPK)
        .and_then(|values| values.checked_mul(std::mem::size_of::<i64>()))
        .ok_or("stationary raw static decode-ID byte overflow")?;
    if removed_bytes != PROMPT_ROWS * v4_router::V4_TOPK * std::mem::size_of::<i64>()
        || prompt_host_bytes - removed_bytes != fixed_host_bytes
        || decode_request_bytes != 48
        || !matches!(
            router,
            v4_state::LayerRouterWeights::Hash { ref ids } if ids.is_empty()
        )
    {
        return Err("stationary raw request IDs leaked into fixed host bytes".into());
    }
    let plans = [(0, 10, 12), (1, 11, 13), (2, 20, 20)]
        .into_iter()
        .map(|(layer, raw_device_bytes, raw_host_bytes)| FixedLayerPlan {
            layer,
            tensors: Vec::new(),
            device_bytes: raw_device_bytes + 1,
            raw_device_bytes,
            host_bytes: raw_host_bytes + 1,
            raw_host_bytes,
            device_saved_ns: 1,
            host_saved_ns: 1,
        })
        .collect::<Vec<_>>();
    let none_selected = vec![false; plans.len()];
    let device = select_fixed_cache_layers(&plans, 10, &none_selected, true, true);
    let host = select_fixed_cache_layers(&plans, 13, &device, false, true);
    if device != [true, false, false] || host != [false, true, false] {
        return Err("stationary raw budget selection/miss boundary drift".into());
    }
    let mut slot = Some((7_u64, 11_u64));
    let leased = slot
        .take()
        .ok_or("stationary raw ownership self-check could not take its slot")?;
    if slot.is_some() {
        return Err("stationary raw ownership slot remained visible during its lease".into());
    }
    restore_leased_slot(&mut slot, leased);
    if slot != Some((7, 11)) {
        return Err("stationary raw ownership lease failed to restore exactly once".into());
    }
    Ok(())
}

pub(crate) fn strict_compressor_host_arena_static_check() -> Result<(), Box<dyn Error>> {
    for (outputs, ratio) in [(512_usize, 128_usize), (1_024, 4)] {
        let weights = outputs
            .checked_mul(v4_state::HIDDEN)
            .ok_or("strict arena static weight size overflow")?;
        let mut wkv_bf16 = vec![0_u16; weights];
        let mut wgate_bf16 = vec![0_u16; weights];
        let mut ape_f32 = vec![0.0_f32; outputs * ratio];
        let mut input_f32 = vec![0.0_f32; v4_state::HIDDEN];
        let mut wkv_f32 = vec![0.0_f32; weights];
        let mut wgate_f32 = vec![0.0_f32; weights];
        let mut kv_f32 = vec![0.0_f32; outputs];
        let mut score_f32 = vec![0.0_f32; outputs];
        let mut score_plus_ape_f32 = vec![0.0_f32; outputs];
        let baseline = [
            (wkv_bf16.as_ptr() as usize, wkv_bf16.capacity()),
            (wgate_bf16.as_ptr() as usize, wgate_bf16.capacity()),
            (ape_f32.as_ptr() as usize, ape_f32.capacity()),
            (input_f32.as_ptr() as usize, input_f32.capacity()),
            (wkv_f32.as_ptr() as usize, wkv_f32.capacity()),
            (wgate_f32.as_ptr() as usize, wgate_f32.capacity()),
            (kv_f32.as_ptr() as usize, kv_f32.capacity()),
            (score_f32.as_ptr() as usize, score_f32.capacity()),
            (
                score_plus_ape_f32.as_ptr() as usize,
                score_plus_ape_f32.capacity(),
            ),
        ];
        for repeat in 0..100 {
            wkv_bf16.fill(repeat as u16);
            wgate_bf16.fill((repeat + 1) as u16);
            ape_f32.fill(repeat as f32);
            input_f32.fill(repeat as f32);
            wkv_f32.fill(repeat as f32);
            wgate_f32.fill(repeat as f32);
            kv_f32.fill(repeat as f32);
            score_f32.fill(repeat as f32);
            score_plus_ape_f32.fill(repeat as f32);
            let current = [
                (wkv_bf16.as_ptr() as usize, wkv_bf16.capacity()),
                (wgate_bf16.as_ptr() as usize, wgate_bf16.capacity()),
                (ape_f32.as_ptr() as usize, ape_f32.capacity()),
                (input_f32.as_ptr() as usize, input_f32.capacity()),
                (wkv_f32.as_ptr() as usize, wkv_f32.capacity()),
                (wgate_f32.as_ptr() as usize, wgate_f32.capacity()),
                (kv_f32.as_ptr() as usize, kv_f32.capacity()),
                (score_f32.as_ptr() as usize, score_f32.capacity()),
                (
                    score_plus_ape_f32.as_ptr() as usize,
                    score_plus_ape_f32.capacity(),
                ),
            ];
            if baseline != current {
                return Err("strict Compressor host arena reallocated during reuse check".into());
            }
        }
    }
    Ok(())
}
