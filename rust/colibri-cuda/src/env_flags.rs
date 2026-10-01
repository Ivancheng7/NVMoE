use super::*;

pub(crate) fn parse_optional_one_flag(
    name: &str,
    value: Option<&std::ffi::OsStr>,
) -> Result<bool, Box<dyn Error>> {
    match value {
        None => Ok(false),
        Some(value) if value == "1" => Ok(true),
        Some(_) => Err(format!("{name} must be exactly 1 when set").into()),
    }
}

pub(crate) fn optional_one_flag(name: &str) -> Result<bool, Box<dyn Error>> {
    let value = std::env::var_os(name);
    parse_optional_one_flag(name, value.as_deref())
}

pub(crate) fn validate_flash_parity_stop_mode() -> Result<(), Box<dyn Error>> {
    if std::env::var_os("COLIBRI_V4_FLASH_PARITY_STOP_AFTER_LAYER").is_none() {
        return Ok(());
    }
    let base = optional_one_flag("COLIBRI_V4_FLASH_BASE")?;
    let serve = optional_one_flag("COLIBRI_V4_FLASH_SERVE")?;
    if !base || serve {
        return Err(
            "COLIBRI_V4_FLASH_PARITY_STOP_AFTER_LAYER is only valid when Flash base mode is the selected execution"
                .into(),
        );
    }
    Ok(())
}

pub(crate) fn mtp1_draft_max_scored_transitions() -> Result<usize, Box<dyn Error>> {
    match std::env::var("COLIBRI_V4_MTP1_MAX_SCORED_TRANSITIONS") {
        Ok(value) => {
            let limit = value.parse::<usize>()?;
            if limit > 8 {
                return Err("COLIBRI_V4_MTP1_MAX_SCORED_TRANSITIONS must be at most 8".into());
            }
            Ok(limit)
        }
        Err(std::env::VarError::NotPresent) => Ok(MTP1_DRAFT_DEFAULT_MAX_SCORED_TRANSITIONS),
        Err(error) => Err(error.into()),
    }
}

/// Fan-out GPU-phase window: how many lanes may be inside their upload +
/// kernel section at the same time. Unset or 0 means unlimited, which is the
/// shipped shape; 1-6 bounds the six lanes. Reads are never gated by this.
pub(crate) fn fanout_gpu_window() -> Result<usize, Box<dyn Error>> {
    match std::env::var("COLIBRI_V4_FLASH_FANOUT_GPU_WINDOW") {
        Ok(value) => {
            let window = value.parse::<usize>()?;
            if window > 6 {
                return Err("COLIBRI_V4_FLASH_FANOUT_GPU_WINDOW must be at most 6 when set".into());
            }
            Ok(window)
        }
        Err(std::env::VarError::NotPresent) => Ok(0),
        Err(error) => Err(error.into()),
    }
}

pub(crate) fn parse_default_true_zero_one_flag(
    name: &str,
    value: Option<&std::ffi::OsStr>,
) -> Result<bool, Box<dyn Error>> {
    match value {
        None => Ok(true),
        Some(value) if value == "1" => Ok(true),
        Some(value) if value == "0" => Ok(false),
        Some(_) => Err(format!("{name} must be exactly 0 or 1 when set").into()),
    }
}

pub(crate) fn raw_wo_a_gpu_decode_enabled() -> Result<bool, Box<dyn Error>> {
    let value = std::env::var_os("COLIBRI_V4_RAW_WO_A_GPU_DECODE");
    parse_default_true_zero_one_flag("COLIBRI_V4_RAW_WO_A_GPU_DECODE", value.as_deref())
}

pub(crate) fn certified_argmax_production_enabled(
    device_attention: bool,
    greedy_steps: usize,
) -> Result<bool, Box<dyn Error>> {
    let value = std::env::var_os("COLIBRI_V4_CERTIFIED_ARGMAX_PRODUCTION");
    Ok(device_attention
        && greedy_steps != 0
        && parse_default_true_zero_one_flag(
            "COLIBRI_V4_CERTIFIED_ARGMAX_PRODUCTION",
            value.as_deref(),
        )?)
}

pub(crate) fn fixed_prefetch_wo_a_decode_arm(
    fixed_prefetch: bool,
    device_attention: bool,
    raw_wo_a_gpu_decode: bool,
) -> &'static str {
    if !fixed_prefetch || !device_attention {
        "inactive"
    } else if raw_wo_a_gpu_decode {
        "raw_fp8_gpu_decode"
    } else {
        "host_bf16_decode"
    }
}

pub(crate) fn validate_phase_profile_mode(
    full61_requested: bool,
    layer_stationary_prefill: bool,
) -> Result<(), Box<dyn Error>> {
    if !full61_requested || !layer_stationary_prefill {
        return Err("COLIBRI_V4_PHASE_PROFILE requires full61 layer-stationary prefill".into());
    }
    Ok(())
}

pub(crate) fn validate_fixed_prefetch_mode(
    full61_requested: bool,
    layer_stationary_prefill: bool,
    cache_is_off: bool,
) -> Result<(), Box<dyn Error>> {
    validate_fixed_prefetch_requirements(
        full61_requested,
        layer_stationary_prefill,
        cache_is_off,
        host_available_bytes()?,
    )
}

pub(crate) fn validate_fixed_prefetch_requirements(
    full61_requested: bool,
    layer_stationary_prefill: bool,
    cache_is_off: bool,
    host_available_bytes: usize,
) -> Result<(), Box<dyn Error>> {
    if !full61_requested || !layer_stationary_prefill || !cache_is_off {
        return Err(
            "COLIBRI_V4_LAYER_STATIONARY_FIXED_PREFETCH requires full61 layer-stationary prefill with fixed cache off"
                .into(),
        );
    }
    pub(crate) const PREFETCH_HOST_HEADROOM: usize = 4 * 1024 * 1024 * 1024;
    if host_available_bytes < PREFETCH_HOST_HEADROOM {
        return Err(
            "layer-stationary fixed prefetch requires at least 4 GiB WSL MemAvailable".into(),
        );
    }
    Ok(())
}

pub(crate) const FIXED_PREFETCH_WINDOW_DEPTH: usize = 2;
pub(crate) const FIXED_PREFETCH_HOST_HEADROOM_BYTES: usize = 4 * 1024 * 1024 * 1024;

pub(crate) fn fixed_layer_decoded_host_bytes(
    layer: usize,
    device_bytes: usize,
) -> Result<usize, Box<dyn Error>> {
    let duplicated_tail_bytes = (v4_state::HIDDEN
        + v4_router::V4_ROUTED_EXPERTS * v4_state::HIDDEN)
        .checked_mul(std::mem::size_of::<u16>())
        .ok_or("fixed decoded host duplicate-tail size overflow")?;
    if layer < 3 {
        device_bytes
            .checked_sub(v4_router::V4_ROUTED_EXPERTS * std::mem::size_of::<f32>())
            .and_then(|bytes| bytes.checked_add(duplicated_tail_bytes))
            .ok_or_else(|| "fixed decoded host size overflow".into())
    } else {
        device_bytes
            .checked_add(duplicated_tail_bytes)
            .ok_or_else(|| "fixed decoded host size overflow".into())
    }
}

pub(crate) fn validate_fixed_prefetch_window(
    model_dir: &std::path::Path,
    manifests: &[v4_state::LayerExecutionManifest],
    qd4_enabled: bool,
    qd4_immutable_expert_cache: bool,
    strict_audit_arena_bytes: usize,
) -> Result<usize, Box<dyn Error>> {
    let max_future_host_bytes = manifests
        .iter()
        .map(|manifest| {
            let (_, device_bytes, _) = fixed_layer_tensor_plan(model_dir, manifest.layer)?;
            fixed_layer_decoded_host_bytes(manifest.layer, device_bytes)
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?
        .into_iter()
        .max()
        .ok_or("fixed prefetch window has no layer plans")?;
    let future_host_bytes = max_future_host_bytes
        .checked_mul(FIXED_PREFETCH_WINDOW_DEPTH)
        .ok_or("fixed prefetch window host reserve overflow")?;
    let required_bytes = FIXED_PREFETCH_HOST_HEADROOM_BYTES
        .checked_add(future_host_bytes)
        .and_then(|bytes| bytes.checked_add(usize::from(qd4_enabled) * V4_QD4_PINNED_HOST_BYTES))
        .and_then(|bytes| {
            bytes.checked_add(
                usize::from(qd4_immutable_expert_cache) * V4_QD4_IMMUTABLE_EXPERT_CACHE_HOST_BYTES,
            )
        })
        .and_then(|bytes| bytes.checked_add(strict_audit_arena_bytes))
        .ok_or("fixed prefetch window host requirement overflow")?;
    let available_bytes = host_available_bytes()?;
    if available_bytes < required_bytes {
        return Err(format!(
            "bounded fixed prefetch requires MemAvailable >= {required_bytes} (headroom={} + {} future decoded hosts x {} + QD4 reserve={} + immutable cache={} + strict audit arena={}), got {available_bytes}",
            FIXED_PREFETCH_HOST_HEADROOM_BYTES,
            FIXED_PREFETCH_WINDOW_DEPTH,
            max_future_host_bytes,
            usize::from(qd4_enabled) * V4_QD4_PINNED_HOST_BYTES,
            usize::from(qd4_immutable_expert_cache) * V4_QD4_IMMUTABLE_EXPERT_CACHE_HOST_BYTES,
            strict_audit_arena_bytes,
        )
        .into());
    }
    Ok(max_future_host_bytes)
}

pub(crate) struct Full61Preflight {
    pub(crate) layer_stationary_prefill: bool,
    pub(crate) layer_stationary_expert_reuse: bool,
    pub(crate) phase_profile: bool,
    pub(crate) fixed_prefetch: bool,
    pub(crate) qd4_enabled: bool,
    pub(crate) qd4_immutable_expert_cache: bool,
    pub(crate) qd4_bench: bool,
    pub(crate) certified_argmax_audit: bool,
    pub(crate) result: Option<PendingResultFile>,
}

impl Full61Preflight {
    pub(crate) fn from_env() -> Result<Self, Box<dyn Error>> {
        let full61_requested = optional_one_flag("COLIBRI_V4_FULL61_B1")?;
        let phase_profile = optional_one_flag("COLIBRI_V4_PHASE_PROFILE")?;
        let fixed_prefetch = optional_one_flag("COLIBRI_V4_LAYER_STATIONARY_FIXED_PREFETCH")?;
        if !full61_requested {
            return Err("set COLIBRI_V4_FULL61_B1=1".into());
        }
        let layer_stationary_prefill = optional_one_flag("COLIBRI_V4_LAYER_STATIONARY_PREFILL")?;
        if phase_profile {
            validate_phase_profile_mode(full61_requested, layer_stationary_prefill)?;
        }
        let layer_stationary_expert_reuse =
            optional_one_flag("COLIBRI_V4_LAYER_STATIONARY_EXPERT_REUSE")?;
        let qd4_enabled = optional_one_flag("COLIBRI_V4_ROUTED_QD4")?;
        let qd4_immutable_expert_cache =
            optional_one_flag("COLIBRI_V4_QD4_IMMUTABLE_EXPERT_CACHE")?;
        let qd4_bench = optional_one_flag("COLIBRI_V4_ROUTED_QD4_BENCH")?;
        let certified_argmax_audit = optional_one_flag("COLIBRI_V4_CERTIFIED_ARGMAX_AUDIT")?;
        let raw_cache = optional_one_flag("COLIBRI_V4_RAW_FIXED_CACHE")?;
        let cache_is_off = parse_gib_budget("COLIBRI_V4_DEVICE_FIXED_CACHE_GIB")? == 0
            && parse_gib_budget("COLIBRI_V4_HOST_FIXED_CACHE_GIB")? == 0
            && std::env::var_os("COLIBRI_V4_FIXED_CACHE_BASELINE_JSONL").is_none()
            && !raw_cache;
        if qd4_enabled && qd4_bench {
            return Err("QD4 full inference requires the replay benchmark off".into());
        }
        if layer_stationary_expert_reuse && !layer_stationary_prefill {
            return Err(
                "COLIBRI_V4_LAYER_STATIONARY_EXPERT_REUSE requires layer-stationary prefill".into(),
            );
        }
        if layer_stationary_expert_reuse && qd4_enabled {
            return Err(
                "layer-stationary expert reuse and persistent QD4 lanes are mutually exclusive"
                    .into(),
            );
        }
        if qd4_immutable_expert_cache && (!qd4_enabled || !layer_stationary_prefill) {
            return Err(
                "COLIBRI_V4_QD4_IMMUTABLE_EXPERT_CACHE=1 requires layer-stationary prefill and COLIBRI_V4_ROUTED_QD4=1"
                    .into(),
            );
        }
        if layer_stationary_prefill && (!cache_is_off || qd4_bench) {
            return Err(
                "layer-stationary prefill requires fixed cache and the QD4 replay benchmark off"
                    .into(),
            );
        }
        if fixed_prefetch {
            validate_fixed_prefetch_mode(full61_requested, layer_stationary_prefill, cache_is_off)?;
        }
        if certified_argmax_audit && !layer_stationary_prefill {
            return Err(
                "COLIBRI_V4_CERTIFIED_ARGMAX_AUDIT requires full61 layer-stationary prefill".into(),
            );
        }
        let result_path = std::env::var_os("COLIBRI_V4_RESULT_JSONL").map(std::path::PathBuf::from);
        if layer_stationary_prefill && result_path.is_none() {
            return Err("layer-stationary prefill requires COLIBRI_V4_RESULT_JSONL".into());
        }
        Ok(Self {
            layer_stationary_prefill,
            layer_stationary_expert_reuse,
            phase_profile,
            fixed_prefetch,
            qd4_enabled,
            qd4_immutable_expert_cache,
            qd4_bench,
            certified_argmax_audit,
            result: result_path.map(PendingResultFile::reserve).transpose()?,
        })
    }
}

pub(crate) fn run_frozen_alpaca_tokenizer_preflight() -> Result<bool, Box<dyn Error>> {
    let Some(path) = std::env::var_os("COLIBRI_V4_TOKENIZER_ONLY_ALPACA_JSON") else {
        return Ok(false);
    };
    let rows: serde_json::Value = serde_json::from_slice(&std::fs::read(&path)?)?;
    let rows = rows
        .as_array()
        .ok_or("tokenizer-only Alpaca input must be a JSON array")?;
    if rows.len() < 16 {
        return Err("tokenizer-only Alpaca input needs the frozen first 16 rows".into());
    }
    let tokenizer = v4_tokenizer::V4Tokenizer::open(&v4_model_dir()?)?;
    let empty = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n";
    let with_input = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n";
    let mut records = Vec::with_capacity(16);
    let mut lengths = Vec::with_capacity(16);
    for (index, row) in rows.iter().take(16).enumerate() {
        let instruction = row["instruction"]
            .as_str()
            .ok_or("frozen Alpaca row lacks instruction")?;
        let input = row["input"]
            .as_str()
            .ok_or("frozen Alpaca row lacks input")?;
        let prompt = if input.is_empty() {
            empty.replace("{instruction}", instruction)
        } else {
            with_input
                .replace("{instruction}", instruction)
                .replace("{input}", input)
        };
        let ids = tokenizer.encode_raw(&prompt)?;
        lengths.push(ids.len());
        records.push(serde_json::json!({
            "frozen_index": index,
            "input_nonempty": !input.is_empty(),
            "prompt_chars": prompt.len(),
            "raw_token_count": ids.len(),
            "prompt_digest": format!("{:016x}", fnv1a_bytes(prompt.as_bytes())),
        }));
    }
    let uniformly_truncate_to_120 = lengths.iter().any(|&length| length > 128);
    for (record, &raw_token_count) in records.iter_mut().zip(&lengths) {
        record["runner_token_count"] = serde_json::json!(if uniformly_truncate_to_120 {
            raw_token_count.min(120)
        } else {
            raw_token_count
        });
    }
    println!(
        "{}",
        serde_json::json!({
            "event": "frozen_alpaca_tokenizer_preflight",
            "rows": records,
            "raw_prompt_tokens_total": lengths.iter().sum::<usize>(),
            "raw_prompt_tokens_min": lengths.iter().copied().min().unwrap_or(0),
            "raw_prompt_tokens_max": lengths.iter().copied().max().unwrap_or(0),
            "uniform_120_token_fallback": uniformly_truncate_to_120,
            "runner_prompt_tokens_total": lengths.iter().map(|&length| if uniformly_truncate_to_120 { length.min(120) } else { length }).sum::<usize>(),
            "runner_input": if uniformly_truncate_to_120 { "COLIBRI_V4_TOKEN_IDS" } else { "COLIBRI_V4_PROMPT_TEXT" },
            "model_execution": false,
        })
    );
    Ok(true)
}
