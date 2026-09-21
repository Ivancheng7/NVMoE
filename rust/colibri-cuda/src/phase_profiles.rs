use super::*;

pub(crate) fn fixed_prepare_dual_accounting(
    fixed: &FixedLoadTelemetry,
    main_exposed_prepare_ns: u64,
) -> Result<serde_json::Value, Box<dyn Error>> {
    let service_work_sum_ns = fixed
        .fixed_read_alloc_ns
        .checked_add(fixed.fixed_h2d_ns)
        .and_then(|value| value.checked_add(fixed.fixed_decode_ns))
        .ok_or("fixed prepare service-work sum overflow")?;
    if fixed.raw_source_bytes > fixed.fixed_source_bytes
        || fixed.raw_h2d_bytes > fixed.fixed_h2d_bytes
        || fixed.raw_h2d_ns > fixed.fixed_h2d_ns
    {
        return Err(format!(
            "fixed prepare nested byte/service accounting is invalid: raw_source_bytes={} source_bytes={} raw_h2d_bytes={} h2d_bytes={} raw_h2d_ns={} h2d_ns={}",
            fixed.raw_source_bytes,
            fixed.fixed_source_bytes,
            fixed.raw_h2d_bytes,
            fixed.fixed_h2d_bytes,
            fixed.raw_h2d_ns,
            fixed.fixed_h2d_ns,
        )
        .into());
    }
    let (origin, device_hit_count, host_hit_count, source_load_count) = match fixed.cache_status {
        "stationary_pinned_device_hit" | "stationary_raw_device_hit" => {
            ("device_hit", 1_usize, 0_usize, 0_usize)
        }
        "stationary_pinned_host_hit" | "stationary_raw_host_hit" => ("host_hit", 0, 1, 0),
        "stationary_pinned_miss" | "stationary_raw_miss" | "prefetched_blocking_h2d" | "miss" => {
            ("source_load", 0, 0, 1)
        }
        status => {
            return Err(format!("fixed prepare has unknown origin status {status}").into());
        }
    };
    let origin_count = device_hit_count + host_hit_count + source_load_count;
    let origin_bytes_valid = match origin {
        "device_hit" => fixed.fixed_source_bytes == 0 && fixed.fixed_h2d_bytes == 0,
        "host_hit" => fixed.fixed_source_bytes == 0 && fixed.fixed_h2d_bytes > 0,
        "source_load" => fixed.fixed_source_bytes > 0 && fixed.fixed_h2d_bytes > 0,
        _ => unreachable!(),
    };
    if origin_count != 1 || !origin_bytes_valid {
        return Err(format!(
            "fixed prepare origin/byte accounting is invalid: status={} device_hit_count={} host_hit_count={} source_load_count={} source_bytes={} h2d_bytes={}",
            fixed.cache_status,
            device_hit_count,
            host_hit_count,
            source_load_count,
            fixed.fixed_source_bytes,
            fixed.fixed_h2d_bytes,
        )
        .into());
    }
    let exposed_service_bound_ns = service_work_sum_ns.min(main_exposed_prepare_ns);
    let derived_overlap_credit_ns = service_work_sum_ns - exposed_service_bound_ns;
    let queue_or_overhead_ns = main_exposed_prepare_ns - exposed_service_bound_ns;
    if service_work_sum_ns
        != exposed_service_bound_ns
            .checked_add(derived_overlap_credit_ns)
            .ok_or("fixed prepare service dual-ledger overflow")?
        || main_exposed_prepare_ns
            != exposed_service_bound_ns
                .checked_add(queue_or_overhead_ns)
                .ok_or("fixed prepare exposed dual-ledger overflow")?
    {
        return Err(format!(
            "fixed prepare dual-ledger closure is invalid: service_work_sum_ns={} main_exposed_prepare_ns={} exposed_service_bound_ns={} derived_overlap_credit_ns={} queue_or_overhead_ns={}",
            service_work_sum_ns,
            main_exposed_prepare_ns,
            exposed_service_bound_ns,
            derived_overlap_credit_ns,
            queue_or_overhead_ns,
        )
        .into());
    }
    Ok(serde_json::json!({
        "cache_status": fixed.cache_status,
        "origin": {
            "class": origin,
            "device_hit_count": device_hit_count,
            "host_hit_count": host_hit_count,
            "source_load_count": source_load_count,
            "count_closure_exact": origin_count == 1,
        },
        "service": {
            "work_sum_ns": service_work_sum_ns,
            "read_alloc_ns": fixed.fixed_read_alloc_ns,
            "h2d_ns": fixed.fixed_h2d_ns,
            "decode_ns": fixed.fixed_decode_ns,
            "internal_closure": "work_sum_ns == read_alloc_ns + h2d_ns + decode_ns",
        },
        "critical": {
            "main_exposed_prepare_ns": main_exposed_prepare_ns,
            "exposed_service_bound_ns": exposed_service_bound_ns,
            "derived_overlap_credit_ns": derived_overlap_credit_ns,
            "queue_or_overhead_ns": queue_or_overhead_ns,
            "service_closure": "work_sum_ns == exposed_service_bound_ns + derived_overlap_credit_ns",
            "main_closure": "main_exposed_prepare_ns == exposed_service_bound_ns + queue_or_overhead_ns",
            "interpretation": "dual-ledger algebra; derived_overlap_credit_ns is not a directly measured causal overlap",
        },
        "bytes": {
            "source_bytes": fixed.fixed_source_bytes,
            "h2d_bytes": fixed.fixed_h2d_bytes,
            "raw_source_bytes": fixed.raw_source_bytes,
            "raw_h2d_bytes": fixed.raw_h2d_bytes,
        },
        "nested": {
            "raw_read_ns": fixed.raw_read_ns,
            "raw_h2d_ns": fixed.raw_h2d_ns,
        },
    }))
}

pub(crate) fn stationary_layer_phase_profile_json(
    request_start_ns: u64,
    request_end_ns: u64,
    fixed: Option<&FixedLoadTelemetry>,
    stage: &LayerStageTelemetry,
    indexer_topk_gpu_ns: u64,
    routed_parallel: bool,
) -> Result<(serde_json::Value, u64, u64), Box<dyn Error>> {
    if request_end_ns < request_start_ns {
        return Err("stationary phase profile layer span is not monotonic".into());
    }
    let expected_prepare_count = usize::from(fixed.is_some());
    if stage.layer_prepare_count != expected_prepare_count
        || (stage.layer_prepare_count == 0 && stage.layer_prepare_ns != 0)
    {
        return Err(format!(
            "stationary layer-prepare count accounting is invalid: count={} expected_count={} main_exposed_prepare_ns={}",
            stage.layer_prepare_count, expected_prepare_count, stage.layer_prepare_ns
        )
        .into());
    }
    let attention_wall_ns = stage
        .device_state_setup_ns
        .saturating_add(stage.attention_ns)
        .saturating_add(stage.commit_ns)
        .saturating_add(stage.attention_audit_oracle_ns);
    let top_level_attributed_ns = stage
        .layer_prepare_ns
        .saturating_add(attention_wall_ns)
        .saturating_add(stage.router_ns)
        .saturating_add(stage.shared_ns)
        .saturating_add(stage.routed_ns);
    let outer_wall_ns = request_end_ns - request_start_ns;
    if top_level_attributed_ns > outer_wall_ns {
        return Err("stationary phase profile top-level scopes exceed layer outer span".into());
    }
    let routed_wall_diagnostics = [
        stage.routed_read_ns,
        stage.routed_alloc_ns,
        stage.routed_setup_ns,
    ];
    if routed_wall_diagnostics
        .iter()
        .any(|&diagnostic| diagnostic > stage.routed_ns)
    {
        return Err("stationary routed diagnostic exceeds its opaque parent scope".into());
    }
    if !routed_parallel
        && (stage.routed_h2d_ns > stage.routed_ns || stage.routed_kernel_ns > stage.routed_ns)
    {
        return Err("stationary routed diagnostic exceeds its opaque parent scope".into());
    }
    if stage.router_mhc_ns > stage.router_ns
        || stage.router_host_prep_ns > stage.router_ns
        || stage.router_alloc_h2d_sync_ns > stage.router_ns
        || stage.router_workspace_alloc_ns > stage.router_ns
        || stage.router_kernel_select_sync_ns > stage.router_ns
        || stage.router_d2h_sync_ns > stage.router_ns
        || stage.router_other_ns > stage.router_ns
        || indexer_topk_gpu_ns > attention_wall_ns
    {
        return Err("stationary nested diagnostic exceeds its parent scope".into());
    }
    let fixed_load = fixed
        .map(|fixed| fixed_prepare_dual_accounting(fixed, stage.layer_prepare_ns))
        .transpose()?;
    let layer_prepare = (stage.layer_prepare_count == 1).then(|| {
        serde_json::json!({
            "wall_ns": stage.layer_prepare_ns,
            "count": stage.layer_prepare_count,
            "scope": "main-thread current-layer consume/load through fixed device runtime and hash IDs ready; prefetch worker work is not additive, while any consume wait is included",
            "fixed_load": fixed_load,
        })
    });
    let routed = if routed_parallel {
        let batch_wall_ns = stage
            .routed_parallel_batch_wall_ns
            .ok_or("parallel routed profile is missing batch wall time")?;
        let read_completion_horizon_ns = stage
            .routed_parallel_read_horizon_ns
            .ok_or("parallel routed profile is missing read completion horizon")?;
        if stage.routed_read_ns != read_completion_horizon_ns
            || read_completion_horizon_ns > batch_wall_ns
            || batch_wall_ns > stage.routed_ns
        {
            return Err("parallel routed wall accounting is invalid".into());
        }
        serde_json::json!({
            "wall_ns": stage.routed_ns,
            "parallel_lanes": {
                "lane_count": V4_QD4_LANES,
                "route_job_count": v4_router::V4_TOPK,
                "batch_wall_ns": batch_wall_ns,
                "read_completion_horizon_ns": read_completion_horizon_ns,
                "h2d_event_work_sum_ns": stage.routed_h2d_ns,
                "kernel_event_work_sum_ns": stage.routed_kernel_ns,
            },
            "diagnostics": {
                "alloc_ns": stage.routed_alloc_ns,
                "setup_ns": stage.routed_setup_ns,
                "source_bytes": stage.routed_source_bytes,
                "h2d_bytes": stage.routed_h2d_bytes,
                "cache_hits": stage.routed_cache_hits,
            },
        })
    } else {
        serde_json::json!({
            "wall_ns": stage.routed_ns,
            "diagnostics": {
                "read_ns": stage.routed_read_ns,
                "alloc_ns": stage.routed_alloc_ns,
                "setup_ns": stage.routed_setup_ns,
                "h2d_ns": stage.routed_h2d_ns,
                "kernel_ns": stage.routed_kernel_ns,
                "source_bytes": stage.routed_source_bytes,
                "h2d_bytes": stage.routed_h2d_bytes,
                "cache_hits": stage.routed_cache_hits,
            },
        })
    };
    Ok((
        serde_json::json!({
            "schema": "colibri-v4-stationary-phase-profile-v5",
            "clock": "monotonic_host_ns; nested CUDA event durations are diagnostics only",
            "outer": {
                "request_start_ns": request_start_ns,
                "request_end_ns": request_end_ns,
                "wall_ns": outer_wall_ns,
            },
            "top_level": {
                "layer_prepare": layer_prepare,
                "attention": {
                    "wall_ns": attention_wall_ns,
                    "execution_ns": stage.attention_ns,
                    "state_commit_ns": stage.commit_ns,
                    "audit_oracle_ns": stage.attention_audit_oracle_ns,
                    "device_state_setup": {
                        "wall_ns": stage.device_state_setup_ns,
                        "count": stage.device_state_setup_count,
                        "state_reset": {
                            "cuda_event_ns": stage.device_state_reset_cuda_ns,
                            "calls": stage.device_state_reset_calls,
                        },
                    },
                    "transfers": {
                        "hidden_h2d": {
                            "count": stage.attention_hidden_h2d_count,
                            "bytes": stage.attention_hidden_h2d_bytes,
                        },
                        "request_rope_lut_d2d": {
                            "scope": "only request-global RoPE LUT row copies; internal attention/cache D2D is not claimed",
                            "count": stage.attention_rope_lut_d2d_count,
                            "bytes": stage.attention_rope_lut_d2d_bytes,
                        },
                        "terminal_after_attention_d2h": {
                            "count": stage.attention_terminal_d2h_count,
                            "bytes": stage.attention_terminal_d2h_bytes,
                        },
                        "intermediate_or_audit_d2h": {
                            "count": stage.attention_intermediate_or_audit_d2h_count,
                            "bytes": stage.attention_intermediate_or_audit_d2h_bytes,
                        },
                    },
                    "temporary": {
                        "steady_device_alloc_count": stage.attention_steady_device_temporary_alloc_count,
                    },
                    "diagnostics": {"indexer_topk_gpu_ns": indexer_topk_gpu_ns},
                },
                "router": {
                    "wall_ns": stage.router_ns,
                    "diagnostics": {
                        "mhc_wall_ns": stage.router_mhc_ns,
                        "mhc_input_alloc_h2d_sync_ns": stage.router_mhc_input_alloc_h2d_sync_ns,
                        "mhc_workspace_alloc_sync_ns": stage.router_mhc_workspace_alloc_sync_ns,
                        "mhc_workspace_init_gpu_ns": stage.router_mhc_workspace_init_gpu_ns,
                        "mhc_mixing_linear_sync_ns": stage.router_mhc_mixing_linear_sync_ns,
                        "mhc_mixing_linear_gpu_ns": stage.router_mhc_mixing_linear_gpu_ns,
                        "mhc_sinkhorn_normalize_collapse_sync_ns": stage.router_mhc_sinkhorn_normalize_collapse_sync_ns,
                        "mhc_sinkhorn_normalize_collapse_gpu_ns": stage.router_mhc_sinkhorn_normalize_collapse_gpu_ns,
                        "mhc_output_d2h_sync_ns": stage.router_mhc_output_d2h_sync_ns,
                        "mhc_other_ns": stage.router_mhc_other_ns,
                        "host_prep_ns": stage.router_host_prep_ns,
                        "alloc_h2d_sync_ns": stage.router_alloc_h2d_sync_ns,
                        "workspace_alloc_ns": stage.router_workspace_alloc_ns,
                        "kernel_select_sync_ns": stage.router_kernel_select_sync_ns,
                        "kernel_select_gpu_ns": stage.router_kernel_select_gpu_ns,
                        "d2h_sync_ns": stage.router_d2h_sync_ns,
                        "other_ns": stage.router_other_ns,
                    },
                },
                "shared": {
                    "wall_ns": stage.shared_ns,
                    // `main_ns` is the shared MLP on the main thread; the worker
                    // numbers are the routed lanes read alongside it. The window
                    // ends when the last reader joins, so join_tail_ns is the
                    // wait after the MLP returned — zero under the routed
                    // fan-out, which closes this window the moment the MLP
                    // returns and moves the lane tails into `routed`.
                    "diagnostics": {
                        "main_ns": stage.shared_main_ns,
                        "read_workers": stage.shared_read_workers,
                        "read_worker_sum_ns": stage.shared_read_worker_sum_ns,
                        "read_worker_max_ns": stage.shared_read_worker_max_ns,
                        "join_tail_ns": stage.shared_ns.saturating_sub(stage.shared_main_ns),
                    },
                },
                "routed": routed,
            },
            "top_level_attributed_ns": top_level_attributed_ns,
            "unattributed_gap_ns": outer_wall_ns - top_level_attributed_ns,
            "closure": {
                "equation": "top_level_attributed_ns + unattributed_gap_ns == outer.wall_ns",
                "exact": true,
            },
        }),
        top_level_attributed_ns,
        outer_wall_ns,
    ))
}

pub(crate) fn stationary_phase_profile_outer_accounting(
    layer_outer_ns: u64,
    final_outer_ns: u64,
    request_e2e_ns: u64,
) -> Result<(u64, u64), Box<dyn Error>> {
    let profiled_outer_ns = layer_outer_ns.saturating_add(final_outer_ns);
    if profiled_outer_ns > request_e2e_ns {
        return Err("stationary phase profile outer scopes exceed request E2E".into());
    }
    Ok((profiled_outer_ns, request_e2e_ns - profiled_outer_ns))
}

pub(crate) fn stationary_one_token_phase_profile_outer_accounting(
    prefill_layer_outer_ns: u64,
    prefill_final_outer_ns: u64,
    decode_layer_outer_ns: u64,
    decode_final_outer_ns: u64,
    request_e2e_ns: u64,
) -> Result<(u64, u64, u64, u64), Box<dyn Error>> {
    let prefill_outer_ns = prefill_layer_outer_ns
        .checked_add(prefill_final_outer_ns)
        .ok_or("one-token phase profile prefill outer overflow")?;
    let decode_outer_ns = decode_layer_outer_ns
        .checked_add(decode_final_outer_ns)
        .ok_or("one-token phase profile decode outer overflow")?;
    let profiled_outer_ns = prefill_outer_ns
        .checked_add(decode_outer_ns)
        .ok_or("one-token phase profile request outer overflow")?;
    if profiled_outer_ns > request_e2e_ns {
        return Err("one-token phase profile outer scopes exceed request E2E".into());
    }
    Ok((
        prefill_outer_ns,
        decode_outer_ns,
        profiled_outer_ns,
        request_e2e_ns - profiled_outer_ns,
    ))
}

pub(crate) fn stationary_final_phase_profile_json(
    request_start_ns: u64,
    request_end_ns: u64,
    fixed_read_ns: u64,
    cpu_wall_ns: u64,
    cpu_streamed_weight_read_ns: u64,
    routed_parallel: bool,
) -> Result<(serde_json::Value, u64), Box<dyn Error>> {
    if request_end_ns < request_start_ns {
        return Err("stationary phase profile final span is not monotonic".into());
    }
    let outer_wall_ns = request_end_ns - request_start_ns;
    let top_level_attributed_ns = fixed_read_ns.saturating_add(cpu_wall_ns);
    if top_level_attributed_ns > outer_wall_ns || cpu_streamed_weight_read_ns > cpu_wall_ns {
        return Err("stationary phase profile final accounting is invalid".into());
    }
    Ok((
        serde_json::json!({
            "schema": "colibri-v4-stationary-phase-profile-v3",
            "clock": "monotonic_host_ns; streamed weight read is nested in cpu wall",
            "routed_parallel": routed_parallel,
            "outer": {
                "request_start_ns": request_start_ns,
                "request_end_ns": request_end_ns,
                "wall_ns": outer_wall_ns,
            },
            "top_level": {
                "fixed": {"wall_ns": fixed_read_ns},
                "cpu_final": {
                    "wall_ns": cpu_wall_ns,
                    "diagnostics": {"streamed_weight_read_ns": cpu_streamed_weight_read_ns},
                },
            },
            "top_level_attributed_ns": top_level_attributed_ns,
            "unattributed_gap_ns": outer_wall_ns - top_level_attributed_ns,
            "closure": {
                "equation": "top_level_attributed_ns + unattributed_gap_ns == outer.wall_ns",
                "exact": true,
            },
        }),
        top_level_attributed_ns,
    ))
}
