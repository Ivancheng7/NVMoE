use super::*;

pub(crate) struct StreamedFinalHeadRun {
    pub(crate) output: v4_state::FinalHeadOutput,
    pub(crate) sumabs_f32: Vec<f32>,
    pub(crate) status: Vec<u32>,
    pub(crate) read_ns: u64,
    pub(crate) h2d_ns: u64,
    pub(crate) kernel_ns: u64,
    pub(crate) d2h_ns: u64,
    pub(crate) cpu_preprocess_ns: u64,
    pub(crate) wall_ns: u64,
}

pub(crate) struct StreamedLmHeadNormalizedRun {
    pub(crate) candidate: usize,
    pub(crate) logits_f32: Vec<f32>,
    pub(crate) sumabs_f32: Vec<f32>,
    pub(crate) status: Vec<u32>,
    pub(crate) read_ns: u64,
    pub(crate) h2d_ns: u64,
    pub(crate) kernel_ns: u64,
    pub(crate) d2h_ns: u64,
    pub(crate) cpu_preprocess_ns: u64,
    pub(crate) wall_ns: u64,
}

pub(crate) fn f32_ulp_distance(left: f32, right: f32) -> u32 {
    let ordered = |value: f32| {
        let bits = value.to_bits();
        if bits & 0x8000_0000 == 0 {
            bits | 0x8000_0000
        } else {
            !bits
        }
    };
    ordered(left).abs_diff(ordered(right))
}

pub(crate) fn final_head_comparison(actual: &[f32], expected: &[f32]) -> (usize, f32, f32, u32) {
    let mut mismatches = 0;
    let mut max_abs = 0.0_f32;
    let mut max_rel = 0.0_f32;
    let mut max_ulp = 0_u32;
    for (&actual, &expected) in actual.iter().zip(expected) {
        if !actual.is_finite() || !expected.is_finite() {
            mismatches += 1;
            max_abs = f32::INFINITY;
            max_rel = f32::INFINITY;
            max_ulp = u32::MAX;
            continue;
        }
        let difference = (actual - expected).abs();
        max_abs = max_abs.max(difference);
        max_rel = max_rel.max(difference / expected.abs().max(f32::MIN_POSITIVE));
        max_ulp = max_ulp.max(f32_ulp_distance(actual, expected));
        mismatches += usize::from(difference > 2e-4 * expected.abs().max(1.0));
    }
    (mismatches, max_abs, max_rel, max_ulp)
}

pub(crate) fn run_real_v4_final_head_bench(
    context: &std::sync::Arc<CudaContext>,
    module: &kernels::LoadedModule,
    stream: &std::sync::Arc<CudaStream>,
) -> Result<(), Box<dyn Error>> {
    let repeats = std::env::var("COLIBRI_V4_FINAL_HEAD_REPEATS")
        .unwrap_or_else(|_| "3".into())
        .parse::<usize>()?;
    if repeats < 3 {
        return Err("COLIBRI_V4_FINAL_HEAD_REPEATS must be at least 3".into());
    }
    let layer_baseline_ns = std::env::var("COLIBRI_V4_FINAL_HEAD_BASELINE_LAYER_NS")
        .map_err(|_| "set COLIBRI_V4_FINAL_HEAD_BASELINE_LAYER_NS for the request projection")?
        .parse::<u64>()?;
    let result_path = std::env::var_os("COLIBRI_V4_FINAL_HEAD_RESULT_JSONL")
        .map(std::path::PathBuf::from)
        .ok_or("set COLIBRI_V4_FINAL_HEAD_RESULT_JSONL")?;
    let root = v4_model_dir()?;
    let embedding = v4_state::load(&root, &[0])?.embedding;
    let mut hc_hidden = Vec::with_capacity(v4_state::HC * v4_state::HIDDEN);
    for _ in 0..v4_state::HC {
        hc_hidden.extend_from_slice(&embedding);
    }
    let weights = v4_state::load_final_head(&root)?;
    let source_bytes = v4_state::VOCAB * v4_state::HIDDEN * 2;
    let mut result = std::fs::File::create(&result_path)?;
    writeln!(
        result,
        "{}",
        serde_json::json!({
            "event": "run",
            "schema": "colibri-v4-final-head-stage1-v1",
            "provenance": benchmark_provenance()?,
            "token_e2e_ns": serde_json::Value::Null,
            "request_e2e_ns": serde_json::Value::Null,
            "model_revision": v4_state::REVISION,
            "input": "official token-0 embedding repeated across HC=4",
            "head_dtype": "BF16",
            "head_shape": [v4_state::VOCAB, v4_state::HIDDEN],
            "source_bytes": source_bytes,
            "gpu_chunk_rows": 1024,
            "repeats": repeats,
            "cache_protocol": "repeat 0: POSIX_FADV_DONTNEED before each CPU/GPU arm; later repeats: warm",
            "baseline_layer_ns": layer_baseline_ns,
        })
    )?;
    let mut cpu_wall = Vec::with_capacity(repeats);
    let mut gpu_wall = Vec::with_capacity(repeats);
    let mut all_canonical = true;
    for repeat in 0..repeats {
        let cache = if repeat == 0 { "fadvise_cold" } else { "warm" };
        if repeat == 0 {
            weights.advise_head_dontneed()?;
        }
        let cpu = v4_state::final_checkpoint_head_cpu_timed(&weights, &hc_hidden, 1)?;
        if repeat == 0 {
            weights.advise_head_dontneed()?;
        }
        let gpu = run_streamed_final_head_gpu(context, module, stream, &weights, &hc_hidden)?;
        let (mismatches, max_abs, max_rel, max_ulp) =
            final_head_comparison(&gpu.output.logits_f32, &cpu.output.logits_f32);
        let argmax_exact = gpu.output.argmax == cpu.output.argmax;
        all_canonical &= mismatches == 0 && argmax_exact;
        cpu_wall.push(cpu.wall_ns);
        gpu_wall.push(gpu.wall_ns);
        writeln!(
            result,
            "{}",
            serde_json::json!({
                "event": "repeat",
                "repeat": repeat,
                "cache": cache,
                "cpu": {
                    "wall_ns": cpu.wall_ns,
                    "read_ns": cpu.read_ns,
                    "source_bytes": source_bytes,
                },
                "gpu": {
                    "wall_ns": gpu.wall_ns,
                    "read_ns": gpu.read_ns,
                    "source_bytes": source_bytes,
                    "h2d_ns": gpu.h2d_ns,
                    "h2d_bytes": source_bytes + v4_state::HIDDEN * 2,
                    "kernel_ns": gpu.kernel_ns,
                    "d2h_ns": gpu.d2h_ns,
                    "d2h_bytes": v4_state::VOCAB * (std::mem::size_of::<f32>() * 2 + std::mem::size_of::<u32>()),
                    "cpu_preprocess_ns": gpu.cpu_preprocess_ns,
                },
                "canonical": {
                    "logit_relative_tolerance": 2e-4,
                    "logit_mismatches": mismatches,
                    "max_abs": max_abs,
                    "max_rel": max_rel,
                    "max_ulp": max_ulp,
                    "argmax_exact": argmax_exact,
                    "cpu_argmax": cpu.output.argmax[0],
                    "gpu_argmax": gpu.output.argmax[0],
                },
            })
        )?;
    }
    let warm_cpu = median_u64(&cpu_wall[1..]);
    let warm_gpu = median_u64(&gpu_wall[1..]);
    let final_improvement = 1.0 - warm_gpu as f64 / warm_cpu as f64;
    let baseline_request_ns = layer_baseline_ns.saturating_add(warm_cpu);
    let projected_request_ns = layer_baseline_ns.saturating_add(warm_gpu);
    let request_improvement = 1.0 - projected_request_ns as f64 / baseline_request_ns as f64;
    let pass = all_canonical && final_improvement >= 0.50 && request_improvement >= 0.20;
    writeln!(
        result,
        "{}",
        serde_json::json!({
            "event": "summary",
            "status": if pass { "PASS" } else { "KILL" },
            "canonical": all_canonical,
            "warm_cpu_wall_median_ns": warm_cpu,
            "warm_gpu_wall_median_ns": warm_gpu,
            "final_head_improvement": final_improvement,
            "required_final_head_improvement": 0.50,
            "baseline_request_ns": baseline_request_ns,
            "projected_request_ns": projected_request_ns,
            "projected_request_improvement": request_improvement,
            "required_request_improvement": 0.20,
        })
    )?;
    result.sync_all()?;
    println!(
        "{{\"status\":\"{}\",\"mode\":\"v4-final-head-stage1\",\"final_head_improvement\":{final_improvement:.6},\"projected_request_improvement\":{request_improvement:.6},\"result_jsonl\":{}}}",
        if pass { "PASS" } else { "KILL" },
        serde_json::to_string(&result_path)?,
    );
    Ok(())
}

pub(crate) fn run_streamed_final_head_gpu(
    context: &std::sync::Arc<CudaContext>,
    module: &kernels::LoadedModule,
    stream: &std::sync::Arc<CudaStream>,
    weights: &v4_state::FinalHeadWeights,
    hc_hidden_bf16: &[u16],
) -> Result<StreamedFinalHeadRun, Box<dyn Error>> {
    let wall_started = std::time::Instant::now();
    let cpu_started = std::time::Instant::now();
    let pre = v4_state::final_hc_gate_cpu(
        hc_hidden_bf16,
        1,
        v4_state::HC,
        v4_state::HIDDEN,
        &weights.hc_fn_f32,
        weights.hc_scale_f32,
        &weights.hc_base_f32,
    )?;
    let mut output = v4_state::final_hc_head_cpu(
        hc_hidden_bf16,
        1,
        v4_state::HC,
        v4_state::HIDDEN,
        &pre,
        &weights
            .norm_bf16
            .iter()
            .map(|&bits| v4_router::bf16_to_f32(bits))
            .collect::<Vec<_>>(),
        &vec![0.0; v4_state::HIDDEN],
        1,
    )?;
    let cpu_preprocess_ns = u64::try_from(cpu_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
    let streamed = run_streamed_lm_head_gpu_normalized(
        context,
        module,
        stream,
        weights,
        &output.normalized_bf16,
    )?;
    output.logits_f32 = streamed.logits_f32;
    output.argmax[0] = streamed.candidate;
    Ok(StreamedFinalHeadRun {
        output,
        sumabs_f32: streamed.sumabs_f32,
        status: streamed.status,
        read_ns: streamed.read_ns,
        h2d_ns: streamed.h2d_ns,
        kernel_ns: streamed.kernel_ns,
        d2h_ns: streamed.d2h_ns,
        cpu_preprocess_ns: cpu_preprocess_ns.saturating_add(streamed.cpu_preprocess_ns),
        wall_ns: u64::try_from(wall_started.elapsed().as_nanos()).unwrap_or(u64::MAX),
    })
}

/// The shared streamed LM-head execution after a caller has produced one
/// normalized BF16 row.  Teacher certification and the MTP draft differ only
/// in how that row is formed; this keeps the chunk upload/dot/argmax mechanism
/// single-sourced.
pub(crate) fn run_streamed_lm_head_gpu_normalized(
    context: &std::sync::Arc<CudaContext>,
    module: &kernels::LoadedModule,
    stream: &std::sync::Arc<CudaStream>,
    weights: &v4_state::FinalHeadWeights,
    normalized_bf16: &[u16],
) -> Result<StreamedLmHeadNormalizedRun, Box<dyn Error>> {
    if normalized_bf16.len() != v4_state::HIDDEN {
        return Err("streamed LM head requires one normalized hidden row".into());
    }
    let wall_started = std::time::Instant::now();
    let input_upload_started = std::time::Instant::now();
    let device_input = DeviceBuffer::from_host(stream, normalized_bf16)?;
    let mut h2d_ns = u64::try_from(input_upload_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
    let device_weight =
        DeviceBuffer::<u16>::zeroed(stream, CERTIFIED_ARGMAX_CHUNK_ROWS * v4_state::HIDDEN)?;
    let mut device_output = DeviceBuffer::<f32>::zeroed(stream, CERTIFIED_ARGMAX_CHUNK_ROWS)?;
    let mut device_sumabs = DeviceBuffer::<f32>::zeroed(stream, CERTIFIED_ARGMAX_CHUNK_ROWS)?;
    let mut device_status = DeviceBuffer::<u32>::zeroed(stream, CERTIFIED_ARGMAX_CHUNK_ROWS)?;
    let start = context.new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
    let end = context.new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
    let mut bytes = vec![0_u8; CERTIFIED_ARGMAX_CHUNK_ROWS * v4_state::HIDDEN * 2];
    let mut words = vec![0_u16; CERTIFIED_ARGMAX_CHUNK_ROWS * v4_state::HIDDEN];
    let mut read_ns = 0_u64;
    let mut kernel_ns = 0_u64;
    let mut d2h_ns = 0_u64;
    let mut cpu_preprocess_ns = 0_u64;
    let mut logits_f32 = vec![0.0_f32; v4_state::VOCAB];
    let mut sumabs_f32 = vec![0.0_f32; v4_state::VOCAB];
    let mut status = vec![0_u32; v4_state::VOCAB];
    for token_start in (0..v4_state::VOCAB).step_by(CERTIFIED_ARGMAX_CHUNK_ROWS) {
        let chunk_rows = (v4_state::VOCAB - token_start).min(CERTIFIED_ARGMAX_CHUNK_ROWS);
        let chunk_elements = chunk_rows * v4_state::HIDDEN;
        let chunk_bytes = chunk_elements * 2;
        let read_started = std::time::Instant::now();
        weights.read_head_weight_bytes(token_start, chunk_rows, &mut bytes[..chunk_bytes])?;
        read_ns = read_ns
            .saturating_add(u64::try_from(read_started.elapsed().as_nanos()).unwrap_or(u64::MAX));
        let decode_started = std::time::Instant::now();
        for (word, bytes) in words[..chunk_elements]
            .iter_mut()
            .zip(bytes[..chunk_bytes].chunks_exact(2))
        {
            *word = u16::from_le_bytes([bytes[0], bytes[1]]);
        }
        cpu_preprocess_ns = cpu_preprocess_ns
            .saturating_add(u64::try_from(decode_started.elapsed().as_nanos()).unwrap_or(u64::MAX));
        let upload_started = std::time::Instant::now();
        // SAFETY: the preallocated device buffer is at least `chunk_bytes`
        // long, and the host words remain borrowed until synchronization.
        unsafe {
            cuda_core::memory::memcpy_htod_async(
                device_weight.cu_deviceptr(),
                words.as_ptr(),
                chunk_bytes,
                stream.cu_stream(),
            )?;
        }
        stream.synchronize()?;
        h2d_ns = h2d_ns
            .saturating_add(u64::try_from(upload_started.elapsed().as_nanos()).unwrap_or(u64::MAX));
        start.record(stream)?;
        // SAFETY: one complete warp owns each output row and all buffer
        // extents were checked above.
        unsafe {
            module.matvec_bf16_streamed(
                stream,
                LaunchConfig::for_num_elems(u32::try_from(chunk_rows * 32)?),
                &device_input,
                &device_weight,
                &mut device_output,
                &mut device_sumabs,
                &mut device_status,
                u32::try_from(v4_state::HIDDEN)?,
                u32::try_from(chunk_rows)?,
            )?;
        }
        end.record(stream)?;
        end.synchronize()?;
        kernel_ns = kernel_ns.saturating_add((start.elapsed_ms(&end)? as f64 * 1_000_000.0) as u64);
        let download_started = std::time::Instant::now();
        let chunk = device_output.to_host_vec(stream)?;
        let chunk_sumabs = device_sumabs.to_host_vec(stream)?;
        let chunk_status = device_status.to_host_vec(stream)?;
        d2h_ns = d2h_ns.saturating_add(
            u64::try_from(download_started.elapsed().as_nanos()).unwrap_or(u64::MAX),
        );
        logits_f32[token_start..token_start + chunk_rows].copy_from_slice(&chunk[..chunk_rows]);
        sumabs_f32[token_start..token_start + chunk_rows]
            .copy_from_slice(&chunk_sumabs[..chunk_rows]);
        status[token_start..token_start + chunk_rows].copy_from_slice(&chunk_status[..chunk_rows]);
    }
    let mut candidate = 0_usize;
    for token in 1..v4_state::VOCAB {
        if logits_f32[token] > logits_f32[candidate] {
            candidate = token;
        }
    }
    Ok(StreamedLmHeadNormalizedRun {
        candidate,
        logits_f32,
        sumabs_f32,
        status,
        read_ns,
        h2d_ns,
        kernel_ns,
        d2h_ns,
        cpu_preprocess_ns,
        wall_ns: u64::try_from(wall_started.elapsed().as_nanos()).unwrap_or(u64::MAX),
    })
}

pub(crate) struct CertifiedArgmaxRun {
    pub(crate) token: usize,
    pub(crate) candidate_count: usize,
    pub(crate) status: u32,
    pub(crate) fallback: bool,
    pub(crate) certified: bool,
    pub(crate) coeff: f64,
    pub(crate) margin: Option<f64>,
    pub(crate) normalized_bf16_digest: u64,
    pub(crate) mechanism_ns: u64,
    pub(crate) read_ns: u64,
    pub(crate) h2d_ns: u64,
    pub(crate) kernel_ns: u64,
    pub(crate) d2h_ns: u64,
    pub(crate) cpu_preprocess_ns: u64,
    pub(crate) repair_ns: u64,
    pub(crate) cpu_fallback: Option<v4_state::FinalHeadCpuRun>,
}

pub(crate) struct IdOnlyFinalHeadDecision {
    pub(crate) token: usize,
    pub(crate) certificate: Option<CertifiedArgmaxRun>,
    pub(crate) cpu: Option<v4_state::FinalHeadCpuRun>,
    pub(crate) certificate_error: Option<String>,
}

pub(crate) fn certified_argmax_coeff() -> f64 {
    let unit_roundoff = 1.0_f64 / (1_u64 << 24) as f64;
    let gamma = |operations: u32| {
        let error = operations as f64 * unit_roundoff;
        error / (1.0 - error)
    };
    let gamma_c = gamma(7_167);
    let gamma_lane = gamma(223);
    let gamma_tree = gamma(5);
    let beta = (1.0 + gamma_lane) * (1.0 + gamma_tree) - 1.0;
    let coeff = (gamma_c + beta) / (1.0 - beta);
    debug_assert!(coeff.is_finite() && coeff > 0.0);
    coeff
}

pub(crate) fn run_certified_argmax_p0(
    context: &std::sync::Arc<CudaContext>,
    module: &kernels::LoadedModule,
    stream: &std::sync::Arc<CudaStream>,
    weights: &v4_state::FinalHeadWeights,
    hidden_bf16: &[u16],
) -> Result<CertifiedArgmaxRun, Box<dyn Error>> {
    let started = std::time::Instant::now();
    let streamed = run_streamed_final_head_gpu(context, module, stream, weights, hidden_bf16)?;
    let coeff = certified_argmax_coeff();
    let mut status = streamed
        .status
        .iter()
        .copied()
        .fold(0_u32, |all, row| all | row);
    let mut lower = Vec::with_capacity(v4_state::VOCAB);
    let mut upper = Vec::with_capacity(v4_state::VOCAB);
    if streamed.output.logits_f32.len() != v4_state::VOCAB
        || streamed.sumabs_f32.len() != v4_state::VOCAB
        || streamed.status.len() != v4_state::VOCAB
    {
        status |= CERTIFIED_HEAD_INTERVAL_INVALID;
    } else {
        for (&logit, &sumabs) in streamed.output.logits_f32.iter().zip(&streamed.sumabs_f32) {
            if !logit.is_finite() || !sumabs.is_finite() || sumabs < 0.0 {
                status |= CERTIFIED_HEAD_INTERVAL_INVALID;
                break;
            }
            let radius = (coeff * sumabs as f64).next_up();
            let lo = (logit as f64 - radius).next_down();
            let hi = (logit as f64 + radius).next_up();
            if !radius.is_finite() || !lo.is_finite() || !hi.is_finite() || lo > hi {
                status |= CERTIFIED_HEAD_INTERVAL_INVALID;
                break;
            }
            lower.push(lo);
            upper.push(hi);
        }
    }
    let mut candidates = Vec::new();
    if status == 0 {
        let max_lower = lower.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        candidates.extend(
            upper
                .iter()
                .enumerate()
                .filter_map(|(token, &hi)| (hi >= max_lower).then_some(token)),
        );
        if candidates.is_empty() {
            status |= CERTIFIED_HEAD_INTERVAL_INVALID;
        } else if candidates.len() == v4_state::VOCAB {
            status |= CERTIFIED_HEAD_ALL_CANDIDATES;
        }
    }
    let fallback = status != 0;
    let repair_started = std::time::Instant::now();
    let mut cpu_fallback = if fallback {
        Some(v4_state::final_checkpoint_head_cpu_timed(
            weights,
            hidden_bf16,
            1,
        )?)
    } else {
        None
    };
    let token = if let Some(cpu) = cpu_fallback.as_ref() {
        cpu.output.argmax[0]
    } else {
        let mut bytes = vec![0_u8; v4_state::HIDDEN * 2];
        let mut row_bf16 = vec![0_u16; v4_state::HIDDEN];
        let mut best = candidates[0];
        let mut best_logit = f32::NEG_INFINITY;
        for token in candidates.iter().copied() {
            weights.read_head_weight_bytes(token, 1, &mut bytes)?;
            for (weight, pair) in row_bf16.iter_mut().zip(bytes.chunks_exact(2)) {
                *weight = u16::from_le_bytes([pair[0], pair[1]]);
            }
            let logit =
                v4_state::canonical_final_head_row_dot(&streamed.output.normalized_bf16, &row_bf16);
            if token == candidates[0] || logit > best_logit {
                best = token;
                best_logit = logit;
            }
        }
        best
    };
    let repair_ns = u64::try_from(repair_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
    let margin = (!fallback).then(|| {
        let best_other = upper
            .iter()
            .enumerate()
            .filter(|(other, _)| *other != token)
            .map(|(_, &value)| value)
            .fold(f64::NEG_INFINITY, f64::max);
        (lower[token] - best_other).next_down()
    });
    Ok(CertifiedArgmaxRun {
        token,
        candidate_count: candidates.len(),
        status,
        fallback,
        certified: !fallback,
        coeff,
        margin,
        normalized_bf16_digest: fnv1a_u16(&streamed.output.normalized_bf16),
        mechanism_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
        read_ns: streamed.read_ns,
        h2d_ns: streamed.h2d_ns,
        kernel_ns: streamed.kernel_ns,
        d2h_ns: streamed.d2h_ns,
        cpu_preprocess_ns: streamed.cpu_preprocess_ns,
        repair_ns,
        cpu_fallback: cpu_fallback.take(),
    })
}

pub(crate) fn step_id_only_final_head(
    context: &std::sync::Arc<CudaContext>,
    module: &kernels::LoadedModule,
    stream: &std::sync::Arc<CudaStream>,
    weights: &v4_state::FinalHeadWeights,
    hidden_bf16: &[u16],
    certified_production: bool,
    audit_cpu: bool,
) -> Result<IdOnlyFinalHeadDecision, Box<dyn Error>> {
    if !certified_production {
        let cpu = v4_state::final_checkpoint_head_cpu_timed(weights, hidden_bf16, 1)?;
        return Ok(IdOnlyFinalHeadDecision {
            token: cpu.output.argmax[0],
            certificate: None,
            cpu: Some(cpu),
            certificate_error: None,
        });
    }
    let started = std::time::Instant::now();
    let mut certificate =
        match run_certified_argmax_p0(context, module, stream, weights, hidden_bf16) {
            Ok(certificate) => certificate,
            Err(error) => {
                let repair_started = std::time::Instant::now();
                let cpu = v4_state::final_checkpoint_head_cpu_timed(weights, hidden_bf16, 1)?;
                let repair_ns =
                    u64::try_from(repair_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
                return Ok(IdOnlyFinalHeadDecision {
                    token: cpu.output.argmax[0],
                    certificate: Some(CertifiedArgmaxRun {
                        token: cpu.output.argmax[0],
                        candidate_count: v4_state::VOCAB,
                        status: CERTIFIED_HEAD_RUNTIME_ERROR,
                        fallback: true,
                        certified: false,
                        coeff: certified_argmax_coeff(),
                        margin: None,
                        normalized_bf16_digest: fnv1a_u16(&cpu.output.normalized_bf16),
                        mechanism_ns: u64::try_from(started.elapsed().as_nanos())
                            .unwrap_or(u64::MAX),
                        read_ns: 0,
                        h2d_ns: 0,
                        kernel_ns: 0,
                        d2h_ns: 0,
                        cpu_preprocess_ns: 0,
                        repair_ns,
                        cpu_fallback: None,
                    }),
                    cpu: Some(cpu),
                    certificate_error: Some(error.to_string()),
                });
            }
        };
    let token = certificate.token;
    let cpu = if certificate.fallback {
        certificate.cpu_fallback.take()
    } else if audit_cpu {
        Some(v4_state::final_checkpoint_head_cpu_timed(
            weights,
            hidden_bf16,
            1,
        )?)
    } else {
        None
    };
    Ok(IdOnlyFinalHeadDecision {
        token,
        certificate: Some(certificate),
        cpu,
        certificate_error: None,
    })
}

pub(crate) fn final_head_profile_metrics(decision: &IdOnlyFinalHeadDecision) -> (u64, u64) {
    if let Some(cpu) = decision.cpu.as_ref() {
        (cpu.wall_ns, cpu.read_ns)
    } else if let Some(certificate) = decision.certificate.as_ref() {
        (certificate.mechanism_ns, certificate.read_ns)
    } else {
        (0, 0)
    }
}

pub(crate) fn certified_argmax_record(
    position: usize,
    decision: &IdOnlyFinalHeadDecision,
) -> Option<serde_json::Value> {
    let certificate = decision.certificate.as_ref()?;
    let cpu = decision.cpu.as_ref();
    Some(serde_json::json!({
        "event": "certified_argmax",
        "position": position,
        "head": "bf16_streamed_gpu_certificate",
        "cert_token": certificate.token,
        "candidate_count": certificate.candidate_count,
        "status_flags": certificate.status,
        "fallback": certificate.fallback,
        "certified": certificate.certified,
        "coeff": certificate.coeff,
        "margin": certificate.margin,
        "certificate_mechanism_ns": certificate.mechanism_ns,
        "repair_ns": certificate.repair_ns,
        "certificate_error": decision.certificate_error,
        "gpu": {
            "read_ns": certificate.read_ns,
            "h2d_ns": certificate.h2d_ns,
            "kernel_ns": certificate.kernel_ns,
            "d2h_ns": certificate.d2h_ns,
            "cpu_preprocess_ns": certificate.cpu_preprocess_ns,
        },
        "cpu_audit": cpu.map(|cpu| serde_json::json!({
            "argmax": cpu.output.argmax[0],
            "normalized_bf16_digest": format!("{:016x}", fnv1a_u16(&cpu.output.normalized_bf16)),
            "logits_f32_digest": format!("{:016x}", fnv1a_f32(&cpu.output.logits_f32)),
            "wall_ns": cpu.wall_ns,
            "read_ns": cpu.read_ns,
        })),
        "argmax_exact": cpu.map(|cpu| certificate.token == cpu.output.argmax[0]),
        "normalized_digest_exact": cpu.map(|cpu| {
            certificate.normalized_bf16_digest == fnv1a_u16(&cpu.output.normalized_bf16)
        }),
    }))
}

pub(crate) fn validate_certified_argmax_audit(
    position: usize,
    decision: &IdOnlyFinalHeadDecision,
) -> Result<(), Box<dyn Error>> {
    let certificate = decision
        .certificate
        .as_ref()
        .ok_or("certified argmax audit did not run the certificate")?;
    let cpu = decision
        .cpu
        .as_ref()
        .ok_or("certified argmax audit did not retain its CPU final")?;
    if certificate.token != cpu.output.argmax[0]
        || certificate.normalized_bf16_digest != fnv1a_u16(&cpu.output.normalized_bf16)
    {
        return Err(format!(
            "certified argmax or normalized digest diverged from canonical CPU audit at position {position}"
        )
        .into());
    }
    Ok(())
}

pub(crate) fn id_only_final_record(
    position: usize,
    input_id: i64,
    input_kind: &str,
    state: &str,
    decision: &IdOnlyFinalHeadDecision,
    input_hidden_bf16: Option<&[u16]>,
    request_elapsed_to_final_ns: u64,
) -> Result<serde_json::Value, Box<dyn Error>> {
    let mut record = serde_json::json!({
        "event": "final",
        "position": position,
        "input_id": input_id,
        "input_kind": input_kind,
        "state": state,
        "head": if decision.certificate.is_some() { "bf16_streamed_gpu_certificate" } else { "canonical_cpu_streamed_bf16" },
        "argmax": decision.token,
        "status": "PASS",
        "request_elapsed_to_final_ns": request_elapsed_to_final_ns,
    });
    let object = record
        .as_object_mut()
        .ok_or("id-only final record is not an object")?;
    if let Some(cpu) = decision.cpu.as_ref() {
        object.insert(
            "collapsed_bf16_digest".into(),
            serde_json::json!(format!("{:016x}", fnv1a_u16(&cpu.output.collapsed_bf16))),
        );
        object.insert(
            "normalized_bf16_digest".into(),
            serde_json::json!(format!("{:016x}", fnv1a_u16(&cpu.output.normalized_bf16))),
        );
        object.insert(
            "logits_f32_digest".into(),
            serde_json::json!(format!("{:016x}", fnv1a_f32(&cpu.output.logits_f32))),
        );
    } else if let Some(certificate) = decision.certificate.as_ref() {
        object.insert(
            "normalized_bf16_digest".into(),
            serde_json::json!(format!("{:016x}", certificate.normalized_bf16_digest)),
        );
        object.insert(
            "certificate".into(),
            serde_json::json!({
                "candidate_count": certificate.candidate_count,
                "status_flags": certificate.status,
                "fallback": certificate.fallback,
                "certified": certificate.certified,
            }),
        );
    }
    if let Some(hidden) = input_hidden_bf16 {
        object.insert(
            "input_hidden_bf16_digest".into(),
            serde_json::json!(format!("{:016x}", fnv1a_u16(hidden))),
        );
    }
    Ok(record)
}

pub(crate) const V4_EOS_TOKEN_ID: i64 = 1;
pub(crate) const V4_LAYER_EXPERT_RETAINED_SLOTS: usize = 32;
