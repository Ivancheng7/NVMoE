use super::*;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct B2LayerDigest {
    pub(crate) layer: usize,
    pub(crate) route_ids: Vec<i32>,
    pub(crate) route_weight_bits: Vec<u32>,
    pub(crate) attention_topk_digest: u64,
    pub(crate) hidden_digest: u64,
    pub(crate) kv_digest: u64,
    pub(crate) main_kv_state_digest: u64,
    pub(crate) main_score_state_digest: u64,
    pub(crate) indexer_kv_state_digest: Option<u64>,
    pub(crate) indexer_score_state_digest: Option<u64>,
}

impl B2LayerDigest {
    pub(crate) fn json(&self) -> serde_json::Value {
        serde_json::json!({
            "layer": self.layer,
            "route_ids": self.route_ids,
            "route_weight_bits": self.route_weight_bits,
            "attention_topk_digest": format!("{:016x}", self.attention_topk_digest),
            "hidden_digest": format!("{:016x}", self.hidden_digest),
            "kv_digest": format!("{:016x}", self.kv_digest),
            "main_kv_state_digest": format!("{:016x}", self.main_kv_state_digest),
            "main_score_state_digest": format!("{:016x}", self.main_score_state_digest),
            "indexer_kv_state_digest": self.indexer_kv_state_digest.map(|value| format!("{value:016x}")),
            "indexer_score_state_digest": self.indexer_score_state_digest.map(|value| format!("{value:016x}")),
        })
    }

    pub(crate) fn ordered_chain_sha256(layers: &[Self]) -> Result<String, Box<dyn Error>> {
        let canonical = serde_json::to_vec(&serde_json::json!({
            "schema": "colibri-v4-b2-ordered-layer-digest-v1",
            "layers": layers.iter().map(Self::json).collect::<Vec<_>>(),
        }))?;
        sha256_bytes(&canonical)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct B2RequestResult {
    pub(crate) layers: Vec<B2LayerDigest>,
    pub(crate) output_digest: u64,
}

pub(crate) fn parse_b2_generate_token_ids<const REQUESTS: usize>(
    value: &str,
    token_ids_env: &str,
) -> Result<[Vec<i64>; REQUESTS], Box<dyn Error>> {
    parse_b2_generate_token_ids_for(&v4_spec::PRO, value, token_ids_env)
}

pub(crate) fn parse_b2_generate_token_ids_for<const REQUESTS: usize>(
    spec: &'static v4_spec::V4Spec,
    value: &str,
    token_ids_env: &str,
) -> Result<[Vec<i64>; REQUESTS], Box<dyn Error>> {
    let prompts = value
        .split(';')
        .map(|prompt| {
            let prompt = prompt.trim();
            if prompt.is_empty() {
                return Err(format!(
                    "{token_ids_env} requires {REQUESTS} non-empty comma-separated token lists"
                )
                .into());
            }
            let ids = prompt
                .split(',')
                .map(str::trim)
                .map(str::parse::<i64>)
                .collect::<Result<Vec<_>, _>>()?;
            v4_state::validate_ids_for(spec, &ids)?;
            Ok(ids)
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    prompts.try_into().map_err(|_| {
        format!("{token_ids_env} requires exactly {REQUESTS} token lists separated by ';'").into()
    })
}

pub(crate) fn validate_b2_generate_inputs<const REQUESTS: usize>(
    prompts: &[Vec<i64>; REQUESTS],
    greedy_steps: usize,
) -> Result<[usize; REQUESTS], Box<dyn Error>> {
    if greedy_steps == 0 {
        return Err("COLIBRI_V4_GREEDY_STEPS must be at least 1 for row-set generation".into());
    }
    let forward_tokens = |request: usize| -> Result<usize, Box<dyn Error>> {
        let forward_tokens = greedy_forward_tokens(prompts[request].len(), greedy_steps)?;
        if forward_tokens > v4_state::KV_WINDOW {
            return Err(format!(
                "row-set request {request} prompt plus greedy decode exceeds the existing V4 KV window"
            )
            .into());
        }
        Ok(forward_tokens)
    };
    (0..REQUESTS)
        .map(forward_tokens)
        .collect::<Result<Vec<_>, _>>()?
        .try_into()
        .map_err(|_| "row-set request count changed".into())
}

pub(crate) fn b2_generate_inputs_from_env<const REQUESTS: usize>(
    token_ids_env: &str,
) -> Result<([Vec<i64>; REQUESTS], usize, [usize; REQUESTS]), Box<dyn Error>> {
    let prompts = parse_b2_generate_token_ids::<REQUESTS>(
        &std::env::var(token_ids_env).map_err(|_| format!("set {token_ids_env}='<ids>;...'"))?,
        token_ids_env,
    )?;
    let greedy_steps = std::env::var("COLIBRI_V4_GREEDY_STEPS")
        .map_err(|_| "set COLIBRI_V4_GREEDY_STEPS for B2 generation")?
        .parse::<usize>()?;
    let forward_tokens = validate_b2_generate_inputs::<REQUESTS>(&prompts, greedy_steps)?;
    Ok((prompts, greedy_steps, forward_tokens))
}

#[derive(Default)]
pub(crate) struct B2Physical {
    pub(crate) fixed_source_bytes: usize,
    pub(crate) fixed_h2d_bytes: usize,
    pub(crate) request_router_metadata_source_bytes: usize,
    pub(crate) request_router_metadata_h2d_bytes: usize,
    pub(crate) routed_source_bytes: usize,
    pub(crate) routed_h2d_bytes: usize,
    pub(crate) initial_state_source_bytes: usize,
    pub(crate) final_head_source_bytes: usize,
}

#[derive(Default)]
pub(crate) struct B2RequestPhysical {
    pub(crate) initial_state_source_bytes: usize,
    pub(crate) final_head_source_bytes: usize,
}

pub(crate) struct B2GenerateRequest {
    pub(crate) prompt_ids: Vec<i64>,
    pub(crate) input_ids: Vec<i64>,
    pub(crate) position: usize,
    pub(crate) persistent: v4_state::V4SerialPersistentState,
    pub(crate) hidden: Vec<u16>,
    pub(crate) generated_ids: Vec<i64>,
    pub(crate) stop_reason: Option<GreedyStopReason>,
    pub(crate) tokens: Vec<serde_json::Value>,
    pub(crate) finals: Vec<serde_json::Value>,
    pub(crate) physical: B2RequestPhysical,
    pub(crate) flash_b1_phase: FlashB1PhaseTiming,
}

#[derive(Default)]
pub(crate) struct FlashB1PhaseTiming {
    pub(crate) layers: Vec<serde_json::Value>,
    pub(crate) layer_outer_ns: u64,
    pub(crate) final_heads: Vec<serde_json::Value>,
    pub(crate) final_head_outer_ns: u64,
}

pub(crate) struct B2GenerateArm<const REQUESTS: usize> {
    pub(crate) requests: [B2GenerateRequest; REQUESTS],
    pub(crate) physical: B2Physical,
    pub(crate) duration_ns: u64,
    pub(crate) peak_device_bytes: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct FlashParityStopAfterLayer {
    pub(crate) layer: usize,
    pub(crate) dir: std::path::PathBuf,
}

#[derive(Debug)]
pub(crate) struct FlashParityStopped {
    pub(crate) path: std::path::PathBuf,
}

impl std::fmt::Display for FlashParityStopped {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "Flash parity intentionally stopped after the requested layer; partial artifact is {}",
            self.path.display()
        )
    }
}

impl std::error::Error for FlashParityStopped {}

pub(crate) fn flash_parity_stop_after_layer_for(
    spec: &v4_spec::V4Spec,
    flash_base: bool,
    dump_dir: Option<&std::ffi::OsStr>,
    value: Option<&std::ffi::OsStr>,
) -> Result<Option<FlashParityStopAfterLayer>, Box<dyn Error>> {
    let Some(value) = value else {
        return Ok(None);
    };
    if !flash_base || spec != &v4_spec::FLASH_0731 {
        return Err(
            "COLIBRI_V4_FLASH_PARITY_STOP_AFTER_LAYER is only valid for Flash-0731 base mode"
                .into(),
        );
    }
    let Some(dump_dir) = dump_dir else {
        return Err(
            "COLIBRI_V4_FLASH_PARITY_STOP_AFTER_LAYER requires COLIBRI_V4_FLASH_PARITY_DUMP_DIR"
                .into(),
        );
    };
    if dump_dir.is_empty() {
        return Err("COLIBRI_V4_FLASH_PARITY_DUMP_DIR must not be empty".into());
    }
    let layer = value
        .to_string_lossy()
        .parse::<usize>()
        .map_err(|_| "COLIBRI_V4_FLASH_PARITY_STOP_AFTER_LAYER must be a layer index")?;
    if layer >= spec.layers {
        return Err("COLIBRI_V4_FLASH_PARITY_STOP_AFTER_LAYER is outside Flash-0731 depth".into());
    }
    Ok(Some(FlashParityStopAfterLayer {
        layer,
        dir: std::path::PathBuf::from(dump_dir),
    }))
}

pub(crate) fn flash_parity_stop_after_layer(
    spec: &v4_spec::V4Spec,
) -> Result<Option<FlashParityStopAfterLayer>, Box<dyn Error>> {
    flash_parity_stop_after_layer_for(
        spec,
        true,
        std::env::var_os("COLIBRI_V4_FLASH_PARITY_DUMP_DIR").as_deref(),
        std::env::var_os("COLIBRI_V4_FLASH_PARITY_STOP_AFTER_LAYER").as_deref(),
    )
}

pub(crate) fn write_flash_parity_stop(
    stop: &FlashParityStopAfterLayer,
    spec: &v4_spec::V4Spec,
    layers: &[B2LayerDigest],
    phase_profile: Option<&serde_json::Value>,
) -> Result<std::path::PathBuf, Box<dyn Error>> {
    if layers.len() != stop.layer + 1
        || layers.last().map(|digest| digest.layer) != Some(stop.layer)
    {
        return Err("Flash parity stop did not retain exactly the requested layer digest".into());
    }
    std::fs::create_dir_all(&stop.dir)?;
    let path = stop.dir.join("parity-stop.json");
    let ordered_layer_digest_sha256 = B2LayerDigest::ordered_chain_sha256(layers)?;
    let mut output = PendingResultFile::reserve(path.clone())?;
    let mut payload = serde_json::json!({
        "schema": "colibri-flash-parity-stop-v1",
        "revision": spec.revision,
        "layer": stop.layer,
        "ordered_layer_digest_sha256": ordered_layer_digest_sha256,
        "partial": true,
    });
    if let Some(phase_profile) = phase_profile {
        payload
            .as_object_mut()
            .ok_or("Flash parity-stop payload must be a JSON object")?
            .insert("phase_profile".to_owned(), phase_profile.clone());
    }
    serde_json::to_writer_pretty(output.file_mut(), &payload)?;
    output.file_mut().write_all(b"\n")?;
    output.commit()?;
    Ok(path)
}

pub(crate) struct B2GenerateSummary<const REQUESTS: usize> {
    pub(crate) requests: [serde_json::Value; REQUESTS],
    pub(crate) physical: B2Physical,
    pub(crate) duration_ns: u64,
    pub(crate) peak_device_bytes: usize,
}

impl<const REQUESTS: usize> B2GenerateArm<REQUESTS> {
    pub(crate) fn into_final_summary(self) -> B2GenerateSummary<REQUESTS> {
        B2GenerateSummary {
            requests: self
                .requests
                .map(|request| b2_generate_semantic_json(&request)),
            physical: self.physical,
            duration_ns: self.duration_ns,
            peak_device_bytes: self.peak_device_bytes,
        }
    }
}

pub(crate) fn b4_greedy_a_duration_ns(first_ns: u64, second_ns: u64) -> Result<u64, Box<dyn Error>> {
    first_ns
        .checked_add(second_ns)
        .ok_or_else(|| "B4 greedy A duration overflow".into())
}

pub(crate) fn b2_generate_active_indices<const REQUESTS: usize>(active: &[bool; REQUESTS]) -> Vec<usize> {
    active
        .iter()
        .enumerate()
        .filter_map(|(request, &live)| live.then_some(request))
        .collect()
}

pub(crate) fn b2_generate_stop_reason(reason: GreedyStopReason) -> &'static str {
    match reason {
        GreedyStopReason::Eos => "eos",
        GreedyStopReason::MaxNewTokens => "max_new_tokens",
    }
}

pub(crate) fn b2_generate_request_json(request: usize, state: &B2GenerateRequest) -> serde_json::Value {
    serde_json::json!({
        "request": request,
        "prompt_ids": state.prompt_ids,
        "emitted_token_ids": state.input_ids,
        "generated_ids": state.generated_ids,
        "stop_reason": state.stop_reason.map(b2_generate_stop_reason),
        "tokens": state.tokens,
        "finals": state.finals,
        "physical": state.physical.json(),
    })
}

pub(crate) fn b2_generate_semantic_json(state: &B2GenerateRequest) -> serde_json::Value {
    serde_json::json!({
        "prompt_ids": state.prompt_ids,
        "emitted_token_ids": state.input_ids,
        "generated_ids": state.generated_ids,
        "stop_reason": state.stop_reason.map(b2_generate_stop_reason),
        "tokens": state.tokens.iter().map(|token| serde_json::json!({
            "token_position": token["token_position"], "input_id": token["input_id"],
            "input_kind": token["input_kind"], "hidden_bf16_digest": token["hidden_bf16_digest"],
            "ordered_layer_digest_sha256": token["ordered_layer_digest_sha256"],
        })).collect::<Vec<_>>(),
        "finals": state.finals,
    })
}

pub(crate) fn b2_router_metadata_row<'a>(
    hash_router: bool,
    ids: &'a [i64],
    request: usize,
    requests: usize,
) -> Result<&'a [i64], Box<dyn Error>> {
    if !hash_router {
        if ids.is_empty() {
            return Ok(&[]);
        }
        return Err("B2 learned router received request metadata".into());
    }
    let start = request
        .checked_mul(v4_router::V4_TOPK)
        .ok_or("B2 router metadata row offset overflow")?;
    let end = start
        .checked_add(v4_router::V4_TOPK)
        .ok_or("B2 router metadata row end overflow")?;
    if request >= requests || ids.len() != requests * v4_router::V4_TOPK {
        return Err("B2 hash router metadata rows do not match the row set".into());
    }
    Ok(&ids[start..end])
}

impl B2Physical {
    pub(crate) fn checked_add_assign(&mut self, other: B2Physical) -> Result<(), Box<dyn Error>> {
        self.fixed_source_bytes = self
            .fixed_source_bytes
            .checked_add(other.fixed_source_bytes)
            .ok_or("B2 fixed source byte overflow")?;
        self.fixed_h2d_bytes = self
            .fixed_h2d_bytes
            .checked_add(other.fixed_h2d_bytes)
            .ok_or("B2 fixed H2D byte overflow")?;
        self.request_router_metadata_source_bytes = self
            .request_router_metadata_source_bytes
            .checked_add(other.request_router_metadata_source_bytes)
            .ok_or("B2 router metadata source byte overflow")?;
        self.request_router_metadata_h2d_bytes = self
            .request_router_metadata_h2d_bytes
            .checked_add(other.request_router_metadata_h2d_bytes)
            .ok_or("B2 router metadata H2D byte overflow")?;
        self.routed_source_bytes = self
            .routed_source_bytes
            .checked_add(other.routed_source_bytes)
            .ok_or("B2 routed source byte overflow")?;
        self.routed_h2d_bytes = self
            .routed_h2d_bytes
            .checked_add(other.routed_h2d_bytes)
            .ok_or("B2 routed H2D byte overflow")?;
        self.initial_state_source_bytes = self
            .initial_state_source_bytes
            .checked_add(other.initial_state_source_bytes)
            .ok_or("B2 initial state source byte overflow")?;
        self.final_head_source_bytes = self
            .final_head_source_bytes
            .checked_add(other.final_head_source_bytes)
            .ok_or("B2 final head source byte overflow")?;
        Ok(())
    }

    pub(crate) fn record_initial_state_source_bytes(&mut self, bytes: usize) -> Result<(), Box<dyn Error>> {
        self.initial_state_source_bytes = self
            .initial_state_source_bytes
            .checked_add(bytes)
            .ok_or("B2 initial state source byte overflow")?;
        Ok(())
    }

    pub(crate) fn record_final_head_source_bytes(&mut self, bytes: usize) -> Result<(), Box<dyn Error>> {
        self.final_head_source_bytes = self
            .final_head_source_bytes
            .checked_add(bytes)
            .ok_or("B2 final head source byte overflow")?;
        Ok(())
    }

    pub(crate) fn record_request_router_metadata(
        &mut self,
        hash_router: bool,
        ids: &[i64],
    ) -> Result<(), Box<dyn Error>> {
        if hash_router {
            if ids.len() != v4_router::V4_TOPK {
                return Err("B2 hash router metadata row is incomplete".into());
            }
        } else if !ids.is_empty() {
            return Err("B2 learned router received request metadata".into());
        }
        let bytes = std::mem::size_of_val(ids);
        self.request_router_metadata_source_bytes = self
            .request_router_metadata_source_bytes
            .checked_add(bytes)
            .ok_or("B2 router metadata source byte overflow")?;
        self.request_router_metadata_h2d_bytes = self
            .request_router_metadata_h2d_bytes
            .checked_add(bytes)
            .ok_or("B2 router metadata H2D byte overflow")?;
        Ok(())
    }

    pub(crate) fn json(&self) -> serde_json::Value {
        serde_json::json!({
            "fixed_source_bytes": self.fixed_source_bytes,
            "fixed_h2d_bytes": self.fixed_h2d_bytes,
            "request_router_metadata_source_bytes": self.request_router_metadata_source_bytes,
            "request_router_metadata_h2d_bytes": self.request_router_metadata_h2d_bytes,
            "routed_source_bytes": self.routed_source_bytes,
            "routed_h2d_bytes": self.routed_h2d_bytes,
            "initial_state_source_bytes": self.initial_state_source_bytes,
            "final_head_source_bytes": self.final_head_source_bytes,
        })
    }
}

impl B2RequestPhysical {
    pub(crate) fn record_initial_state_source_bytes(&mut self, bytes: usize) -> Result<(), Box<dyn Error>> {
        self.initial_state_source_bytes = self
            .initial_state_source_bytes
            .checked_add(bytes)
            .ok_or("B2 request initial state source byte overflow")?;
        Ok(())
    }

    pub(crate) fn record_final_head_source_bytes(&mut self, bytes: usize) -> Result<(), Box<dyn Error>> {
        self.final_head_source_bytes = self
            .final_head_source_bytes
            .checked_add(bytes)
            .ok_or("B2 request final head source byte overflow")?;
        Ok(())
    }

    pub(crate) fn json(&self) -> serde_json::Value {
        serde_json::json!({
            "initial_state_source_bytes": self.initial_state_source_bytes,
            "final_head_source_bytes": self.final_head_source_bytes,
        })
    }
}

pub(crate) fn flash_b1_layer_phase_profile_json(
    layer: usize,
    outer_wall_ns: u64,
    fixed_prepare_ns: u64,
    fixed: &FixedLoadTelemetry,
    wo_a_decode_sync_ns: u64,
    stage: &LayerStageTelemetry,
) -> Result<serde_json::Value, Box<dyn Error>> {
    let fixed_service_ns = fixed
        .fixed_read_alloc_ns
        .checked_add(fixed.fixed_decode_ns)
        .and_then(|value| value.checked_add(fixed.fixed_h2d_ns))
        .ok_or("Flash B1 fixed service timing overflow")?;
    if (fixed.cache_status != "prefetched_host" && fixed_service_ns > fixed_prepare_ns)
        || stage.attention_ns > outer_wall_ns
        || stage.commit_ns > outer_wall_ns
        || stage.router_ns > outer_wall_ns
        || stage.shared_ns > outer_wall_ns
        || stage.routed_ns > outer_wall_ns
        || (!stage.routed_prefetched && stage.routed_read_ns > stage.routed_ns)
        || stage.routed_h2d_ns > stage.routed_ns
        || stage.routed_kernel_ns > stage.routed_ns
    {
        return Err("Flash B1 nested phase timing exceeds its parent scope".into());
    }
    let attention_wall_ns = stage
        .attention_ns
        .checked_add(stage.commit_ns)
        .ok_or("Flash B1 attention timing overflow")?;
    let attributed_ns = fixed_prepare_ns
        .checked_add(wo_a_decode_sync_ns)
        .and_then(|value| value.checked_add(attention_wall_ns))
        .and_then(|value| value.checked_add(stage.router_ns))
        .and_then(|value| value.checked_add(stage.shared_ns))
        .and_then(|value| value.checked_add(stage.routed_ns))
        .ok_or("Flash B1 layer phase timing overflow")?;
    if attributed_ns > outer_wall_ns {
        return Err("Flash B1 top-level phases exceed layer wall time".into());
    }
    let bench = bench_telemetry_enabled().then(|| serde_json::json!({
        "attention": {
            "device_state_setup_ns": stage.device_state_setup_ns,
            "device_state_setup_count": stage.device_state_setup_count,
            "device_state_reset_cuda_ns": stage.device_state_reset_cuda_ns,
            "device_state_reset_calls": stage.device_state_reset_calls,
            "audit_oracle_ns": stage.attention_audit_oracle_ns,
            "hidden_h2d": {"count": stage.attention_hidden_h2d_count, "bytes": stage.attention_hidden_h2d_bytes},
            "rope_lut_d2d": {"count": stage.attention_rope_lut_d2d_count, "bytes": stage.attention_rope_lut_d2d_bytes},
            "terminal_d2h": {"count": stage.attention_terminal_d2h_count, "bytes": stage.attention_terminal_d2h_bytes},
            "intermediate_or_audit_d2h": {"count": stage.attention_intermediate_or_audit_d2h_count, "bytes": stage.attention_intermediate_or_audit_d2h_bytes},
            "steady_device_temporary_alloc_count": stage.attention_steady_device_temporary_alloc_count,
            "certified_host_path": stage.attention_profile.as_ref().map(AttentionProfile::json),
        },
        "router": {
            "mhc": {
                "wall_ns": stage.router_mhc_ns,
                "input_alloc_h2d_sync_ns": stage.router_mhc_input_alloc_h2d_sync_ns,
                "workspace_alloc_sync_ns": stage.router_mhc_workspace_alloc_sync_ns,
                "workspace_init_gpu_ns": stage.router_mhc_workspace_init_gpu_ns,
                "mixing_linear_sync_ns": stage.router_mhc_mixing_linear_sync_ns,
                "mixing_linear_gpu_ns": stage.router_mhc_mixing_linear_gpu_ns,
                "sinkhorn_normalize_collapse_sync_ns": stage.router_mhc_sinkhorn_normalize_collapse_sync_ns,
                "sinkhorn_normalize_collapse_gpu_ns": stage.router_mhc_sinkhorn_normalize_collapse_gpu_ns,
                "output_d2h_sync_ns": stage.router_mhc_output_d2h_sync_ns,
                "other_ns": stage.router_mhc_other_ns,
            },
            "host_prep_ns": stage.router_host_prep_ns,
            "alloc_h2d_sync_ns": stage.router_alloc_h2d_sync_ns,
            "workspace_alloc_ns": stage.router_workspace_alloc_ns,
            "kernel_select_sync_ns": stage.router_kernel_select_sync_ns,
            "kernel_select_gpu_ns": stage.router_kernel_select_gpu_ns,
            "d2h_sync_ns": stage.router_d2h_sync_ns,
            "other_ns": stage.router_other_ns,
        },
    }));
    Ok(serde_json::json!({
        "layer": layer,
        "outer": {"wall_ns": outer_wall_ns},
        "top_level": {
            "fixed": {
                "wall_ns": fixed_prepare_ns,
                "host_prefetched": fixed.cache_status == "prefetched_host",
                "read_alloc_ns": fixed.fixed_read_alloc_ns,
                "decode_ns": fixed.fixed_decode_ns,
                "h2d_ns": fixed.fixed_h2d_ns,
                "source_bytes": fixed.fixed_source_bytes,
                "h2d_bytes": fixed.fixed_h2d_bytes,
                "window": {
                    "join_ns": fixed.window_join_ns,
                    "spawn_ns": fixed.window_spawn_ns,
                    "residual_ns": fixed.window_residual_ns,
                    "release_ns": fixed.window_release_ns,
                    "release_bytes": fixed.window_release_bytes,
                    "worker_ns": fixed.window_worker_ns,
                    "worker_ready_before_consume": fixed.window_worker_ready,
                },
            },
            "wo_a_decode_gpu_sync": {"wall_ns": wo_a_decode_sync_ns},
            "attention": {"wall_ns": attention_wall_ns, "execution_ns": stage.attention_ns, "commit_ns": stage.commit_ns},
            "router": {"wall_ns": stage.router_ns},
            "shared": {"wall_ns": stage.shared_ns},
            "routed": {
                "wall_ns": stage.routed_ns,
                "host_prefetched": stage.routed_prefetched,
                "diagnostics": {
                    "read_ns": stage.routed_read_ns,
                    "alloc_ns": stage.routed_alloc_ns,
                    "setup_ns": stage.routed_setup_ns,
                    "h2d_ns": stage.routed_h2d_ns,
                    "kernel_ns": stage.routed_kernel_ns,
                    "source_bytes": stage.routed_source_bytes,
                    "h2d_bytes": stage.routed_h2d_bytes,
                    "expert_loads": stage.routed_expert_loads,
                },
            },
        },
        "top_level_attributed_ns": attributed_ns,
        "unattributed_gap_ns": outer_wall_ns - attributed_ns,
        "bench": bench,
        "closure": {
            "equation": "top_level_attributed_ns + unattributed_gap_ns == outer.wall_ns",
            "exact": true,
        },
    }))
}

pub(crate) fn flash_b1_request_phase_profile_json(
    timing: &FlashB1PhaseTiming,
    request_wall_ns: u64,
) -> Result<serde_json::Value, Box<dyn Error>> {
    let layers_outer_ns = timing
        .layers
        .iter()
        .map(|layer| {
            layer["outer"]["wall_ns"]
                .as_u64()
                .ok_or("Flash B1 layer profile lacks wall time")
        })
        .try_fold(0_u64, |sum, value| {
            value.and_then(|value| {
                sum.checked_add(value)
                    .ok_or("Flash B1 layer wall sum overflow")
            })
        })?;
    if layers_outer_ns != timing.layer_outer_ns {
        return Err("Flash B1 layer wall ledger does not close".into());
    }
    let attributed_ns = layers_outer_ns
        .checked_add(timing.final_head_outer_ns)
        .ok_or("Flash B1 request phase timing overflow")?;
    if attributed_ns > request_wall_ns {
        return Err("Flash B1 request phases exceed request wall time".into());
    }
    Ok(serde_json::json!({
        "schema": "colibri-flash-0731-b1-phase-profile-v1",
        "clock": "monotonic_host_ns; existing stream synchronization boundaries only",
        "outer": {"wall_ns": request_wall_ns},
        "layers": &timing.layers,
        "final_heads": &timing.final_heads,
        "top_level_attributed_ns": attributed_ns,
        "unattributed_gap_ns": request_wall_ns - attributed_ns,
        "closure": {
            "equation": "top_level_attributed_ns + unattributed_gap_ns == outer.wall_ns",
            "exact": true,
        },
    }))
}
