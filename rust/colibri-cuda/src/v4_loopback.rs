use super::*;

pub(crate) fn run_v4_two_worker_loopback(
    context: &std::sync::Arc<CudaContext>,
    module: &kernels::LoadedModule,
    shard: &safetensors::ShardFile,
    shard_path: &std::path::Path,
    bench: Option<&DualBenchConfig>,
) -> Result<(), Box<dyn Error>> {
    const MEASURED_PAIRS: usize = 100;
    const EXPECTED_IDS: [u16; v4_router::V4_TOPK] = [254, 222, 371, 378, 245, 279];
    const SOCKET_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

    let input = deterministic_v4_bf16(1, 97);
    let norm = read_v4_bf16(shard, "layers.0.ffn_norm.weight", &[7_168])?;
    let gate = read_v4_bf16(shard, "layers.0.ffn.gate.weight", &[384, 7_168])?;
    let routed_input = v4_router::rmsnorm_bf16(&input, &norm)?;
    let logits = v4_router::project_gate_logits(&routed_input, &gate)?;
    let hash_ids = read_v4_hash_rows(shard, "layers.0.ffn.gate.tid2eid", &[0])?;
    let routes = v4_router::select_hash(&logits, &hash_ids)?;
    let route_ids: [u16; v4_router::V4_TOPK] = routes
        .ids
        .iter()
        .map(|&id| u16::try_from(id))
        .collect::<Result<Vec<_>, _>>()?
        .try_into()
        .map_err(|_| "V4 route ID count changed")?;
    let route_weights: [f32; v4_router::V4_TOPK] = routes
        .weights
        .clone()
        .try_into()
        .map_err(|_| "V4 route weight count changed")?;
    assert_eq!(route_ids, EXPECTED_IDS, "frozen token-0 route IDs");

    let oracle_start = std::time::Instant::now();
    let mut oracle_contributions = Vec::with_capacity(v4_router::V4_TOPK);
    for (&expert_id, &weight) in route_ids.iter().zip(&route_weights) {
        let weights = v4_expert::RoutedExpert::load(shard, expert_id)?;
        oracle_contributions.push(v4_expert::forward_cpu(&weights, &routed_input, weight)?);
    }
    let oracle_bf16 = canonical_audit_bits(&oracle_contributions);
    let oracle_ns = oracle_start
        .elapsed()
        .as_nanos()
        .try_into()
        .unwrap_or(u64::MAX);
    drop(oracle_contributions);

    let owner_ids = [
        [route_ids[0], route_ids[1], route_ids[2]],
        [route_ids[3], route_ids[4], route_ids[5]],
    ];
    let owner_weights = [
        [route_weights[0], route_weights[1], route_weights[2]],
        [route_weights[3], route_weights[4], route_weights[5]],
    ];
    let listeners = [
        std::net::TcpListener::bind(("127.0.0.1", 0))?,
        std::net::TcpListener::bind(("127.0.0.1", 0))?,
    ];
    let addresses = [listeners[0].local_addr()?, listeners[1].local_addr()?];
    let worker_streams = [context.new_stream()?, context.new_stream()?];
    let (ready_tx, ready_rx) = std::sync::mpsc::channel();
    let audit_first = bench
        .map(|config| config.audit_first)
        .unwrap_or([true, false, true, false]);
    let mut expected_modes = Vec::with_capacity(2 + 2 * MEASURED_PAIRS);
    expected_modes.extend([
        v4_federated::OwnerTaskMode::Audit,
        v4_federated::OwnerTaskMode::Certified,
    ]);
    for pair in 0..MEASURED_PAIRS {
        let first_is_audit = if bench.is_some() {
            audit_first[pair / NETEM_PAIRS_PER_BLOCK]
        } else {
            pair.is_multiple_of(2)
        };
        expected_modes.extend(if first_is_audit {
            [
                v4_federated::OwnerTaskMode::Audit,
                v4_federated::OwnerTaskMode::Certified,
            ]
        } else {
            [
                v4_federated::OwnerTaskMode::Certified,
                v4_federated::OwnerTaskMode::Audit,
            ]
        });
    }

    let (mut cold, audit_samples, certified_samples, audit_blocks, certified_blocks) =
        std::thread::scope(|scope| -> Result<_, String> {
            let mut workers = Vec::with_capacity(2);
            for (owner, (listener, worker_stream)) in
                listeners.into_iter().zip(worker_streams).enumerate()
            {
                let worker_context = context.clone();
                let worker_input = routed_input.clone();
                let worker_path = shard_path.to_owned();
                let worker_ready = ready_tx.clone();
                let ids = owner_ids[owner];
                let weights = owner_weights[owner];
                let worker_modes = expected_modes.clone();
                workers.push(scope.spawn(move || {
                    run_v4_owner_worker(
                        listener,
                        worker_context,
                        module,
                        worker_stream,
                        &worker_path,
                        owner as u8,
                        ids,
                        weights,
                        &worker_input,
                        worker_modes,
                        worker_ready,
                    )
                }));
            }
            drop(ready_tx);

            let mut sockets = [
                std::net::TcpStream::connect(addresses[0]).map_err(|error| error.to_string())?,
                std::net::TcpStream::connect(addresses[1]).map_err(|error| error.to_string())?,
            ];
            for socket in &sockets {
                socket
                    .set_nodelay(true)
                    .and_then(|_| socket.set_read_timeout(Some(SOCKET_TIMEOUT)))
                    .and_then(|_| socket.set_write_timeout(Some(SOCKET_TIMEOUT)))
                    .map_err(|error| error.to_string())?;
            }

            let mut cold = Vec::with_capacity(2);
            for _ in 0..2 {
                cold.push(
                    ready_rx
                        .recv_timeout(std::time::Duration::from_secs(120))
                        .map_err(|error| format!("worker cold-start readiness failed: {error}"))?,
                );
            }

            let _ = run_v4_audit_client(
                &mut sockets,
                1,
                owner_ids,
                owner_weights,
                &routed_input,
                &oracle_bf16,
            )?;
            let warm_certified = run_v4_certified_client(
                &mut sockets,
                2,
                owner_ids,
                owner_weights,
                &routed_input,
                &oracle_bf16,
            )?;
            if warm_certified.repair_lanes == 0 {
                return Err("frozen certified warmup unexpectedly needed no repair".into());
            }

            let mut audit_samples = Vec::with_capacity(MEASURED_PAIRS);
            let mut certified_samples = Vec::with_capacity(MEASURED_PAIRS);
            let mut audit_blocks: [Vec<LoopbackSample>; NETEM_BLOCKS] =
                std::array::from_fn(|_| Vec::with_capacity(NETEM_PAIRS_PER_BLOCK));
            let mut certified_blocks: [Vec<LoopbackSample>; NETEM_BLOCKS] =
                std::array::from_fn(|_| Vec::with_capacity(NETEM_PAIRS_PER_BLOCK));
            let mut request_id = 3_u64;
            for pair in 0..MEASURED_PAIRS {
                let block = pair / NETEM_PAIRS_PER_BLOCK;
                if expected_modes[request_id as usize - 1] == v4_federated::OwnerTaskMode::Audit {
                    let audit = run_v4_audit_client(
                        &mut sockets,
                        request_id,
                        owner_ids,
                        owner_weights,
                        &routed_input,
                        &oracle_bf16,
                    )?;
                    audit_blocks[block].push(audit.clone());
                    audit_samples.push(audit);
                    request_id += 1;
                    let certified = run_v4_certified_client(
                        &mut sockets,
                        request_id,
                        owner_ids,
                        owner_weights,
                        &routed_input,
                        &oracle_bf16,
                    )?;
                    certified_blocks[block].push(certified.clone());
                    certified_samples.push(certified);
                } else {
                    let certified = run_v4_certified_client(
                        &mut sockets,
                        request_id,
                        owner_ids,
                        owner_weights,
                        &routed_input,
                        &oracle_bf16,
                    )?;
                    certified_blocks[block].push(certified.clone());
                    certified_samples.push(certified);
                    request_id += 1;
                    let audit = run_v4_audit_client(
                        &mut sockets,
                        request_id,
                        owner_ids,
                        owner_weights,
                        &routed_input,
                        &oracle_bf16,
                    )?;
                    audit_blocks[block].push(audit.clone());
                    audit_samples.push(audit);
                }
                request_id += 1;
            }
            drop(sockets);
            for worker in workers {
                worker
                    .join()
                    .map_err(|_| "owner worker panicked".to_string())??;
            }
            Ok((
                cold,
                audit_samples,
                certified_samples,
                audit_blocks,
                certified_blocks,
            ))
        })
        .map_err(|error| -> Box<dyn Error> { error.into() })?;

    cold.sort_by_key(|worker| worker.owner);
    println!(
        "DUAL_TCP_FIXED token=0 seed=97 ids={route_ids:?} weights={route_weights:?} placement=A:[0,1,2],B:[3,4,5] block=256 oracle_cpu={:.3}ms",
        oracle_ns as f64 / 1_000_000.0,
    );
    for worker in &cold {
        println!(
            "DUAL_TCP_COLD owner={} resident_slots=3 range_load+alloc+one_weight_H2D={:.3}ms weight_H2D={}B",
            worker.owner,
            worker.elapsed_ns as f64 / 1_000_000.0,
            worker.weight_bytes,
        );
    }

    let audit_bytes = audit_samples[0].bytes;
    let certified_bytes = certified_samples[0].bytes;
    assert!(
        audit_samples
            .iter()
            .all(|sample| sample.bytes == audit_bytes && sample.repair_lanes == 0)
    );
    let repair_lanes = certified_samples[0].repair_lanes;
    assert!(
        certified_samples.iter().all(|sample| {
            sample.bytes == certified_bytes && sample.repair_lanes == repair_lanes
        })
    );
    print_loopback_timings("AUDIT", &audit_samples, false);
    print_loopback_timings("CERTIFIED", &certified_samples, true);
    println!(
        "DUAL_TCP_BYTES audit initial_request_double={} audit_responses={} total_bidirectional={} return_side={}; certified initial_request_double={} certified_initial_responses={} repair_requests={} repair_responses={} total_bidirectional={} return_side={}",
        audit_bytes.initial_requests,
        audit_bytes.audit_responses,
        audit_bytes.total(),
        audit_bytes.returned(),
        certified_bytes.initial_requests,
        certified_bytes.certified_responses,
        certified_bytes.repair_requests,
        certified_bytes.repair_responses,
        certified_bytes.total(),
        certified_bytes.returned(),
    );
    println!(
        "DUAL_TCP_BYTE_RATIOS certified/audit total_bidirectional={:.6} return_side={:.6}; offline 2.187x is not extrapolated to total network bytes",
        certified_bytes.total() as f64 / audit_bytes.total() as f64,
        certified_bytes.returned() as f64 / audit_bytes.returned() as f64,
    );
    if let Some(config) = bench {
        let block_order = config
            .audit_first
            .iter()
            .map(|&first| if first { "AB" } else { "BA" })
            .collect::<Vec<_>>()
            .join("|");
        println!(
            "NETEM_PREREG scenario={} seed={} blocks={} pairs_per_block={} order={} unit=same_GPU_same_resident6_one_protocol_measurement primary=certified_minus_audit_E2E_median",
            config.scenario, config.seed, NETEM_BLOCKS, NETEM_PAIRS_PER_BLOCK, block_order,
        );
        print_netem_result(
            &config.scenario,
            config.seed,
            "audit",
            &block_order,
            &audit_samples,
            &audit_blocks,
        );
        print_netem_result(
            &config.scenario,
            config.seed,
            "certified",
            &block_order,
            &certified_samples,
            &certified_blocks,
        );
    }
    let network_scope = if bench.is_some() {
        "container-loopback netem condition is externally measured"
    } else {
        "loopback speed is observational and bandwidth-limited netem remains future validation"
    };
    println!(
        "DUAL_TCP_RESULT audit_n={} certified_n={} repair_lanes={} BF16_mismatch=0 identity=fail_closed order=alternating_AUDIT/CERTIFIED; one GPU with two CUDA streams is protocol/timing simulation, not dual-GPU parallelism; {network_scope}",
        audit_samples.len(),
        certified_samples.len(),
        repair_lanes,
    );
    Ok(())
}

pub(crate) fn run_v4_replica_recovery(
    context: &std::sync::Arc<CudaContext>,
    module: &kernels::LoadedModule,
    shard: &safetensors::ShardFile,
    shard_path: &std::path::Path,
) -> Result<(), Box<dyn Error>> {
    const EXPECTED_IDS: [u16; v4_router::V4_TOPK] = [254, 222, 371, 378, 245, 279];
    const CASES: [RecoveryCase; 5] = [
        RecoveryCase::AuditBaseline,
        RecoveryCase::CertifiedBaseline,
        RecoveryCase::AuditF1,
        RecoveryCase::CertifiedF1,
        RecoveryCase::CertifiedF2,
    ];

    let input = deterministic_v4_bf16(1, 97);
    let norm = read_v4_bf16(shard, "layers.0.ffn_norm.weight", &[7_168])?;
    let gate = read_v4_bf16(shard, "layers.0.ffn.gate.weight", &[384, 7_168])?;
    let routed_input = v4_router::rmsnorm_bf16(&input, &norm)?;
    let logits = v4_router::project_gate_logits(&routed_input, &gate)?;
    let hash_ids = read_v4_hash_rows(shard, "layers.0.ffn.gate.tid2eid", &[0])?;
    let routes = v4_router::select_hash(&logits, &hash_ids)?;
    let route_ids: [u16; v4_router::V4_TOPK] = routes
        .ids
        .iter()
        .map(|&id| u16::try_from(id))
        .collect::<Result<Vec<_>, _>>()?
        .try_into()
        .map_err(|_| "V4 recovery route ID count changed")?;
    let route_weights: [f32; v4_router::V4_TOPK] = routes
        .weights
        .clone()
        .try_into()
        .map_err(|_| "V4 recovery route weight count changed")?;
    assert_eq!(route_ids, EXPECTED_IDS, "frozen recovery route IDs");

    let mut oracle_contributions = Vec::with_capacity(v4_router::V4_TOPK);
    for (&expert_id, &weight) in route_ids.iter().zip(&route_weights) {
        let weights = v4_expert::RoutedExpert::load(shard, expert_id)?;
        oracle_contributions.push(v4_expert::forward_cpu(&weights, &routed_input, weight)?);
    }
    let oracle_bf16 = canonical_audit_bits(&oracle_contributions);
    drop(oracle_contributions);
    let owner_ids = [
        [route_ids[0], route_ids[1], route_ids[2]],
        [route_ids[3], route_ids[4], route_ids[5]],
    ];
    let owner_weights = [
        [route_weights[0], route_weights[1], route_weights[2]],
        [route_weights[3], route_weights[4], route_weights[5]],
    ];
    let endpoints = [
        RecoveryEndpoint::PrimaryA,
        RecoveryEndpoint::PrimaryB,
        RecoveryEndpoint::ReplicaA,
    ];
    let listeners = [
        std::net::TcpListener::bind(("127.0.0.1", 0))?,
        std::net::TcpListener::bind(("127.0.0.1", 0))?,
        std::net::TcpListener::bind(("127.0.0.1", 0))?,
    ];
    let addresses = [
        listeners[0].local_addr()?,
        listeners[1].local_addr()?,
        listeners[2].local_addr()?,
    ];
    let streams = [
        context.new_stream()?,
        context.new_stream()?,
        context.new_stream()?,
    ];
    let (ready_tx, ready_rx) = std::sync::mpsc::channel();

    let (mut cold, results) = std::thread::scope(|scope| -> Result<_, String> {
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let mut workers = Vec::with_capacity(3);
        for ((endpoint, listener), stream) in endpoints.into_iter().zip(listeners).zip(streams) {
            let worker_context = context.clone();
            let worker_input = routed_input.clone();
            let worker_path = shard_path.to_owned();
            let worker_ready = ready_tx.clone();
            let worker_stop = stop.clone();
            let owner = endpoint.owner() as usize;
            let accept_count = match endpoint {
                RecoveryEndpoint::PrimaryA | RecoveryEndpoint::PrimaryB => 102,
                RecoveryEndpoint::ReplicaA => 62,
            };
            workers.push(scope.spawn(move || {
                run_v4_recovery_endpoint(
                    listener,
                    worker_context,
                    module,
                    stream,
                    &worker_path,
                    endpoint,
                    owner_ids[owner],
                    owner_weights[owner],
                    &worker_input,
                    accept_count,
                    worker_ready,
                    worker_stop,
                )
            }));
        }
        drop(ready_tx);
        let client_result = (|| -> Result<_, String> {
            let mut cold = Vec::with_capacity(3);
            for _ in 0..3 {
                cold.push(
                    ready_rx
                        .recv_timeout(std::time::Duration::from_secs(120))
                        .map_err(|error| format!("recovery endpoint readiness failed: {error}"))?,
                );
            }

            let mut results = Vec::with_capacity(CASES.len());
            for case in CASES {
                let mut samples = Vec::with_capacity(RECOVERY_SAMPLES as usize);
                for offset in 0..RECOVERY_SAMPLES {
                    samples.push(run_v4_recovery_case(
                        addresses,
                        case,
                        case.start_id() + offset,
                        owner_ids,
                        owner_weights,
                        &routed_input,
                        &oracle_bf16,
                    )?);
                }
                results.push((case, samples));
            }

            for case in [
                RecoveryCase::WrongReplicaIdentity,
                RecoveryCase::SecondFailure,
            ] {
                match run_v4_recovery_case(
                    addresses,
                    case,
                    case.start_id(),
                    owner_ids,
                    owner_weights,
                    &routed_input,
                    &oracle_bf16,
                ) {
                    Ok(_) => return Err(format!("{} unexpectedly committed", case.label())),
                    Err(error) => println!(
                        "RECOVERY_NEGATIVE case={} result=error commit=false detail={error}",
                        case.label()
                    ),
                }
            }
            Ok((cold, results))
        })();
        stop.store(true, std::sync::atomic::Ordering::Release);
        for address in addresses {
            let _ = std::net::TcpStream::connect(address);
        }
        let mut worker_error = None;
        for worker in workers {
            match worker.join() {
                Ok(Ok(())) => {}
                Ok(Err(error)) => {
                    worker_error.get_or_insert(error);
                }
                Err(_) => {
                    worker_error.get_or_insert("recovery endpoint panicked".into());
                }
            };
        }
        match (client_result, worker_error) {
            (Err(error), _) => Err(error),
            (Ok(_), Some(error)) => Err(error),
            (Ok(result), None) => Ok(result),
        }
    })
    .map_err(|error| -> Box<dyn Error> { error.into() })?;

    cold.sort_by_key(|worker| match worker.endpoint {
        RecoveryEndpoint::PrimaryA => 0,
        RecoveryEndpoint::PrimaryB => 1,
        RecoveryEndpoint::ReplicaA => 2,
    });
    println!(
        "RECOVERY_FIXED token=0 seed=97 activation_hash={:016x} primary=A:[0,1,2],B:[3,4,5] replica=A':[0,1,2] one_failover_max=true",
        v4_activation_hash(&routed_input),
    );
    for worker in cold {
        println!(
            "RECOVERY_COLD endpoint={} resident_slots=3 elapsed={:.3}ms weight_H2D={}B",
            worker.endpoint.label(),
            worker.elapsed_ns as f64 / 1_000_000.0,
            worker.weight_bytes,
        );
    }
    for (case, samples) in &results {
        print_recovery_case(*case, samples);
    }
    println!(
        "RECOVERY_RESULT cases=5 n_per_case={} BF16_mismatch=0 partial_commit=0 negatives=2_fail_closed single_GPU_three_stream_protocol_simulation=true",
        RECOVERY_SAMPLES,
    );
    Ok(())
}

pub(crate) fn run_v4_recovery_endpoint(
    listener: std::net::TcpListener,
    context: std::sync::Arc<CudaContext>,
    module: &kernels::LoadedModule,
    stream: std::sync::Arc<CudaStream>,
    shard_path: &std::path::Path,
    endpoint: RecoveryEndpoint,
    expert_ids: [u16; v4_federated::OWNER_EXPERTS],
    route_weights: [f32; v4_federated::OWNER_EXPERTS],
    expected_input: &[u16],
    accept_count: usize,
    ready: std::sync::mpsc::Sender<RecoveryCold>,
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
) -> Result<(), String> {
    let cold_start = std::time::Instant::now();
    let shard = safetensors::ShardFile::open(shard_path).map_err(|error| error.to_string())?;
    let mut weight_bytes = 0;
    let mut slots = Vec::with_capacity(v4_federated::OWNER_EXPERTS);
    for &expert_id in &expert_ids {
        let weights =
            v4_expert::RoutedExpert::load(&shard, expert_id).map_err(|error| error.to_string())?;
        weight_bytes += weights.h2d_bytes();
        slots.push(
            ExpertSlot::new(&context, module, &stream, &weights)
                .map_err(|error| error.to_string())?,
        );
    }
    let cold_ns = cold_start
        .elapsed()
        .as_nanos()
        .try_into()
        .unwrap_or(u64::MAX);
    // Warm all three resident slots before any measured connection.
    let _ = execute_v4_owner_slots(&mut slots, expected_input, route_weights)?;
    ready
        .send(RecoveryCold {
            endpoint,
            elapsed_ns: cold_ns,
            weight_bytes,
        })
        .map_err(|error| error.to_string())?;

    let mut seen = HashSet::with_capacity(accept_count);
    let mut accepted = 0;
    while accepted < accept_count {
        let (mut socket, peer) = listener.accept().map_err(|error| error.to_string())?;
        if stop.load(std::sync::atomic::Ordering::Acquire) {
            return Ok(());
        }
        accepted += 1;
        if !peer.ip().is_loopback() {
            return Err(format!(
                "recovery endpoint {} accepted non-loopback peer",
                endpoint.label()
            ));
        }
        let timeout = std::time::Duration::from_secs(30);
        socket
            .set_nodelay(true)
            .and_then(|_| socket.set_read_timeout(Some(timeout)))
            .and_then(|_| socket.set_write_timeout(Some(timeout)))
            .map_err(|error| error.to_string())?;
        let frame = v4_federated::receive_frame(&mut socket).map_err(|error| error.to_string())?;
        let task = v4_federated::decode_owner_task(&frame).map_err(|error| error.to_string())?;
        let request_id = task.identity.request_id;
        let case = recovery_case_for_id(request_id)
            .ok_or_else(|| format!("unknown recovery request {request_id}"))?;
        if !seen.insert(request_id) {
            return Err(format!(
                "recovery endpoint {} rejected duplicate request {request_id}",
                endpoint.label()
            ));
        }
        validate_v4_recovery_task(
            &task,
            endpoint,
            case,
            expert_ids,
            route_weights,
            expected_input,
        )?;
        let fault = recovery_fault(endpoint, case);
        if fault == RecoveryFault::F1 {
            // Injected only after the full initial task has passed validation.
            continue;
        }

        let compute_start = std::time::Instant::now();
        let (contributions, gpu_ns) =
            execute_v4_owner_slots(&mut slots, &task.hidden_bf16, task.route_weights)?;
        match task.mode {
            v4_federated::OwnerTaskMode::Audit => {
                let mut identity = task.identity;
                if fault == RecoveryFault::WrongIdentity {
                    identity.request_id += 1;
                }
                let response = v4_federated::OwnerAuditResponse {
                    identity,
                    worker_ns: compute_start
                        .elapsed()
                        .as_nanos()
                        .try_into()
                        .unwrap_or(u64::MAX),
                    gpu_ns,
                    contributions: contributions.into_iter().flatten().collect(),
                };
                let frame = v4_federated::encode_owner_audit_response(&response)
                    .map_err(|error| error.to_string())?;
                v4_federated::send_frame(&mut socket, &frame).map_err(|error| error.to_string())?;
            }
            v4_federated::OwnerTaskMode::Certified => {
                let refs: Vec<_> = contributions.iter().map(Vec::as_slice).collect();
                let reduced = v4_certified::reduce_owner(&refs, v4_federated::FROZEN_BLOCK_SIZE)
                    .map_err(|error| error.to_string())?;
                let response = v4_federated::OwnerCertifiedResponse {
                    identity: task.identity.clone(),
                    worker_ns: compute_start
                        .elapsed()
                        .as_nanos()
                        .try_into()
                        .unwrap_or(u64::MAX),
                    gpu_ns,
                    grouped_f32: reduced.grouped_f32,
                    bounds: reduced.bounds,
                };
                let frame = v4_federated::encode_owner_certified_response(&response)
                    .map_err(|error| error.to_string())?;
                v4_federated::send_frame(&mut socket, &frame).map_err(|error| error.to_string())?;

                let repair_frame =
                    v4_federated::receive_frame(&mut socket).map_err(|error| error.to_string())?;
                let repair_request = v4_federated::decode_owner_repair_request(&repair_frame)
                    .map_err(|error| error.to_string())?;
                v4_federated::validate_owner_repair_request(&task, &repair_request)
                    .map_err(|error| error.to_string())?;
                if fault == RecoveryFault::F2 {
                    // The complete initial response is now unusable by contract.
                    continue;
                }
                let repair_start = std::time::Instant::now();
                let repair_lanes = v4_certified::mask_count(&repair_request.mask);
                let mut values = Vec::with_capacity(v4_federated::OWNER_EXPERTS * repair_lanes);
                for contribution in &contributions {
                    for (lane, &value) in contribution.iter().enumerate() {
                        if v4_certified::mask_contains(&repair_request.mask, lane) {
                            values.push(value);
                        }
                    }
                }
                let response = v4_federated::OwnerRepairResponse {
                    identity: task.identity,
                    worker_ns: repair_start
                        .elapsed()
                        .as_nanos()
                        .try_into()
                        .unwrap_or(u64::MAX),
                    lane_count: u16::try_from(repair_lanes)
                        .map_err(|_| "recovery repair count exceeds u16".to_string())?,
                    values,
                };
                let frame = v4_federated::encode_owner_repair_response(&response)
                    .map_err(|error| error.to_string())?;
                v4_federated::send_frame(&mut socket, &frame).map_err(|error| error.to_string())?;
            }
        }
    }
    Ok(())
}

pub(crate) fn run_v4_recovery_certified(
    mut primary_a: std::net::TcpStream,
    mut primary_b: std::net::TcpStream,
    replica_address: std::net::SocketAddr,
    case: RecoveryCase,
    start: std::time::Instant,
    tasks: &[v4_federated::OwnerTask; 2],
    task_frames: &[Vec<u8>; 2],
    oracle_bf16: &[u16],
    mut bytes: LoopbackWireBytes,
) -> Result<RecoverySample, String> {
    let primary_result = receive_recovery_certified(&mut primary_a, &tasks[0]);
    let (b_frame, b_response) = receive_recovery_certified(&mut primary_b, &tasks[1])?;
    bytes.certified_responses += wire_frame_bytes(&b_frame);
    let expected_f1 = case == RecoveryCase::CertifiedF1;
    let mut failovers = 0;
    let mut failover_started = None;
    let mut active_a = primary_a;
    let mut a_response = match primary_result {
        Ok((frame, response)) => {
            if expected_f1 {
                return Err("certified_F1 did not inject F1".into());
            }
            bytes.certified_responses += wire_frame_bytes(&frame);
            response
        }
        Err(_) => {
            failovers = 1;
            failover_started = Some(std::time::Instant::now());
            active_a = connect_recovery_endpoint(replica_address)?;
            v4_federated::send_frame(&mut active_a, &task_frames[0])
                .map_err(|error| error.to_string())?;
            bytes.initial_requests += wire_frame_bytes(&task_frames[0]);
            let (frame, response) = receive_recovery_certified(&mut active_a, &tasks[0])?;
            bytes.certified_responses += wire_frame_bytes(&frame);
            response
        }
    };

    if case == RecoveryCase::CertifiedF2 {
        let preliminary = certify_recovery_pair(&a_response, &b_response)?;
        let trigger = v4_federated::OwnerRepairRequest {
            identity: tasks[0].identity.clone(),
            mask: preliminary.repair_mask,
        };
        let frame = v4_federated::encode_owner_repair_request(&trigger)
            .map_err(|error| error.to_string())?;
        v4_federated::send_frame(&mut active_a, &frame).map_err(|error| error.to_string())?;
        bytes.repair_requests += wire_frame_bytes(&frame);
        if receive_recovery_repair(&mut active_a, &tasks[0], &trigger).is_ok() {
            return Err("certified_F2 primary A returned repair instead of failing".into());
        }
        failovers = 1;
        failover_started = Some(std::time::Instant::now());

        // Discard the primary A group and certificate completely.
        active_a = connect_recovery_endpoint(replica_address)?;
        v4_federated::send_frame(&mut active_a, &task_frames[0])
            .map_err(|error| error.to_string())?;
        bytes.initial_requests += wire_frame_bytes(&task_frames[0]);
        let (frame, replica_response) = receive_recovery_certified(&mut active_a, &tasks[0])?;
        bytes.certified_responses += wire_frame_bytes(&frame);
        a_response = replica_response;
    }

    let certification = certify_recovery_pair(&a_response, &b_response)?;
    let repair_lanes = v4_certified::mask_count(&certification.repair_mask);
    if repair_lanes == 0 {
        return Err(format!("{} unexpectedly required no repair", case.label()));
    }
    let requests = [
        v4_federated::OwnerRepairRequest {
            identity: tasks[0].identity.clone(),
            mask: certification.repair_mask.clone(),
        },
        v4_federated::OwnerRepairRequest {
            identity: tasks[1].identity.clone(),
            mask: certification.repair_mask.clone(),
        },
    ];
    let frames = [
        v4_federated::encode_owner_repair_request(&requests[0])
            .map_err(|error| error.to_string())?,
        v4_federated::encode_owner_repair_request(&requests[1])
            .map_err(|error| error.to_string())?,
    ];
    v4_federated::send_frame(&mut active_a, &frames[0]).map_err(|error| error.to_string())?;
    v4_federated::send_frame(&mut primary_b, &frames[1]).map_err(|error| error.to_string())?;
    bytes.repair_requests += frames.iter().map(wire_frame_bytes).sum::<usize>();
    let (a_frame, a_repair) = receive_recovery_repair(&mut active_a, &tasks[0], &requests[0])?;
    let (b_frame, b_repair) = receive_recovery_repair(&mut primary_b, &tasks[1], &requests[1])?;
    bytes.repair_responses += wire_frame_bytes(&a_frame) + wire_frame_bytes(&b_frame);

    let mut actual = certification.output_bf16;
    apply_owner_repairs(
        &mut actual,
        &certification.repair_mask,
        &[a_repair, b_repair],
    )?;
    if actual != oracle_bf16 {
        return Err(format!(
            "{} BF16 mismatch count {} before commit",
            case.label(),
            bf16_mismatch_count(&actual, oracle_bf16)
        ));
    }
    let expected_failovers = usize::from(matches!(
        case,
        RecoveryCase::CertifiedF1 | RecoveryCase::CertifiedF2
    ));
    if failovers != expected_failovers {
        return Err(format!(
            "{} failover count {failovers}, expected {expected_failovers}",
            case.label()
        ));
    }
    Ok(RecoverySample {
        e2e_ns: start.elapsed().as_nanos().try_into().unwrap_or(u64::MAX),
        failover_ns: failover_started
            .map(|started| started.elapsed().as_nanos().try_into().unwrap_or(u64::MAX))
            .unwrap_or(0),
        failovers,
        bytes,
    })
}

pub(crate) fn run_v4_owner_worker(
    listener: std::net::TcpListener,
    context: std::sync::Arc<CudaContext>,
    module: &kernels::LoadedModule,
    stream: std::sync::Arc<CudaStream>,
    shard_path: &std::path::Path,
    owner: u8,
    expert_ids: [u16; v4_federated::OWNER_EXPERTS],
    route_weights: [f32; v4_federated::OWNER_EXPERTS],
    expected_input: &[u16],
    expected_modes: Vec<v4_federated::OwnerTaskMode>,
    ready: std::sync::mpsc::Sender<OwnerCold>,
) -> Result<(), String> {
    let cold_start = std::time::Instant::now();
    let shard = safetensors::ShardFile::open(shard_path).map_err(|error| error.to_string())?;
    let mut weight_bytes = 0;
    let mut slots = Vec::with_capacity(v4_federated::OWNER_EXPERTS);
    for &expert_id in &expert_ids {
        let weights =
            v4_expert::RoutedExpert::load(&shard, expert_id).map_err(|error| error.to_string())?;
        weight_bytes += weights.h2d_bytes();
        slots.push(
            ExpertSlot::new(&context, module, &stream, &weights)
                .map_err(|error| error.to_string())?,
        );
    }
    let elapsed_ns = cold_start
        .elapsed()
        .as_nanos()
        .try_into()
        .unwrap_or(u64::MAX);
    ready
        .send(OwnerCold {
            owner,
            elapsed_ns,
            weight_bytes,
        })
        .map_err(|error| error.to_string())?;

    let (mut socket, peer) = listener.accept().map_err(|error| error.to_string())?;
    if !peer.ip().is_loopback() {
        return Err("owner worker accepted a non-loopback peer".into());
    }
    let timeout = std::time::Duration::from_secs(30);
    socket
        .set_nodelay(true)
        .and_then(|_| socket.set_read_timeout(Some(timeout)))
        .and_then(|_| socket.set_write_timeout(Some(timeout)))
        .map_err(|error| error.to_string())?;

    for (operation, expected_mode) in expected_modes.into_iter().enumerate() {
        let request_id = operation as u64 + 1;
        let frame = v4_federated::receive_frame(&mut socket).map_err(|error| error.to_string())?;
        let task = v4_federated::decode_owner_task(&frame).map_err(|error| error.to_string())?;
        validate_v4_owner_task(
            &task,
            owner,
            request_id,
            expert_ids,
            route_weights,
            expected_input,
            expected_mode,
        )?;

        let compute_start = std::time::Instant::now();
        let (contributions, gpu_ns) =
            execute_v4_owner_slots(&mut slots, &task.hidden_bf16, task.route_weights)?;
        match task.mode {
            v4_federated::OwnerTaskMode::Audit => {
                let worker_ns = compute_start
                    .elapsed()
                    .as_nanos()
                    .try_into()
                    .unwrap_or(u64::MAX);
                let response = v4_federated::OwnerAuditResponse {
                    identity: task.identity,
                    worker_ns,
                    gpu_ns,
                    contributions: contributions.into_iter().flatten().collect(),
                };
                let frame = v4_federated::encode_owner_audit_response(&response)
                    .map_err(|error| error.to_string())?;
                v4_federated::send_frame(&mut socket, &frame).map_err(|error| error.to_string())?;
            }
            v4_federated::OwnerTaskMode::Certified => {
                let refs: Vec<_> = contributions.iter().map(Vec::as_slice).collect();
                let reduced = v4_certified::reduce_owner(&refs, v4_federated::FROZEN_BLOCK_SIZE)
                    .map_err(|error| error.to_string())?;
                let worker_ns = compute_start
                    .elapsed()
                    .as_nanos()
                    .try_into()
                    .unwrap_or(u64::MAX);
                let response = v4_federated::OwnerCertifiedResponse {
                    identity: task.identity.clone(),
                    worker_ns,
                    gpu_ns,
                    grouped_f32: reduced.grouped_f32,
                    bounds: reduced.bounds,
                };
                let frame = v4_federated::encode_owner_certified_response(&response)
                    .map_err(|error| error.to_string())?;
                v4_federated::send_frame(&mut socket, &frame).map_err(|error| error.to_string())?;

                let repair_frame =
                    v4_federated::receive_frame(&mut socket).map_err(|error| error.to_string())?;
                let repair_request = v4_federated::decode_owner_repair_request(&repair_frame)
                    .map_err(|error| error.to_string())?;
                v4_federated::validate_owner_repair_request(&task, &repair_request)
                    .map_err(|error| error.to_string())?;
                let repair_start = std::time::Instant::now();
                let repair_lanes = v4_certified::mask_count(&repair_request.mask);
                // Repair selects only from this request-local cache; it never
                // invokes an ExpertSlot again.
                let mut values = Vec::with_capacity(v4_federated::OWNER_EXPERTS * repair_lanes);
                for contribution in &contributions {
                    for (lane, &value) in contribution.iter().enumerate() {
                        if v4_certified::mask_contains(&repair_request.mask, lane) {
                            values.push(value);
                        }
                    }
                }
                let repair_response = v4_federated::OwnerRepairResponse {
                    identity: task.identity,
                    worker_ns: repair_start
                        .elapsed()
                        .as_nanos()
                        .try_into()
                        .unwrap_or(u64::MAX),
                    lane_count: u16::try_from(repair_lanes)
                        .map_err(|_| "repair lane count exceeds u16".to_string())?,
                    values,
                };
                let frame = v4_federated::encode_owner_repair_response(&repair_response)
                    .map_err(|error| error.to_string())?;
                v4_federated::send_frame(&mut socket, &frame).map_err(|error| error.to_string())?;
            }
        }
    }
    Ok(())
}

pub(crate) fn run_v4_certified_client(
    sockets: &mut [std::net::TcpStream; 2],
    request_id: u64,
    owner_ids: [[u16; v4_federated::OWNER_EXPERTS]; 2],
    owner_weights: [[f32; v4_federated::OWNER_EXPERTS]; 2],
    input: &[u16],
    oracle_bf16: &[u16],
) -> Result<LoopbackSample, String> {
    let start = std::time::Instant::now();
    let tasks = owner_tasks(
        v4_federated::OwnerTaskMode::Certified,
        request_id,
        owner_ids,
        owner_weights,
        input,
    );
    let frames = [
        v4_federated::encode_owner_task(&tasks[0]).map_err(|error| error.to_string())?,
        v4_federated::encode_owner_task(&tasks[1]).map_err(|error| error.to_string())?,
    ];
    for owner in 0..2 {
        v4_federated::send_frame(&mut sockets[owner], &frames[owner])
            .map_err(|error| error.to_string())?;
    }
    let certified_frames = [
        v4_federated::receive_frame(&mut sockets[0]).map_err(|error| error.to_string())?,
        v4_federated::receive_frame(&mut sockets[1]).map_err(|error| error.to_string())?,
    ];
    let certified_responses = [
        v4_federated::decode_owner_certified_response(&certified_frames[0])
            .map_err(|error| error.to_string())?,
        v4_federated::decode_owner_certified_response(&certified_frames[1])
            .map_err(|error| error.to_string())?,
    ];
    for owner in 0..2 {
        v4_federated::validate_owner_certified_response(&tasks[owner], &certified_responses[owner])
            .map_err(|error| error.to_string())?;
    }
    let owner_reductions = certified_responses
        .iter()
        .map(|response| v4_certified::OwnerReduction {
            grouped_f32: response.grouped_f32.clone(),
            bounds: response.bounds.clone(),
        })
        .collect::<Vec<_>>();
    let certification =
        v4_certified::certify_owner_groups(&owner_reductions, v4_federated::FROZEN_BLOCK_SIZE)
            .map_err(|error| error.to_string())?;
    let repair_lanes = v4_certified::mask_count(&certification.repair_mask);
    if repair_lanes == 0 {
        return Err(format!(
            "certified request {request_id} had no repair phase"
        ));
    }

    let repair_start = std::time::Instant::now();
    let repair_requests = [
        v4_federated::OwnerRepairRequest {
            identity: tasks[0].identity.clone(),
            mask: certification.repair_mask.clone(),
        },
        v4_federated::OwnerRepairRequest {
            identity: tasks[1].identity.clone(),
            mask: certification.repair_mask.clone(),
        },
    ];
    let repair_request_frames = [
        v4_federated::encode_owner_repair_request(&repair_requests[0])
            .map_err(|error| error.to_string())?,
        v4_federated::encode_owner_repair_request(&repair_requests[1])
            .map_err(|error| error.to_string())?,
    ];
    for owner in 0..2 {
        v4_federated::send_frame(&mut sockets[owner], &repair_request_frames[owner])
            .map_err(|error| error.to_string())?;
    }
    let repair_response_frames = [
        v4_federated::receive_frame(&mut sockets[0]).map_err(|error| error.to_string())?,
        v4_federated::receive_frame(&mut sockets[1]).map_err(|error| error.to_string())?,
    ];
    let repair_responses = [
        v4_federated::decode_owner_repair_response(&repair_response_frames[0])
            .map_err(|error| error.to_string())?,
        v4_federated::decode_owner_repair_response(&repair_response_frames[1])
            .map_err(|error| error.to_string())?,
    ];
    for owner in 0..2 {
        v4_federated::validate_owner_repair_response(
            &tasks[owner],
            &repair_requests[owner],
            &repair_responses[owner],
        )
        .map_err(|error| error.to_string())?;
    }
    let repair_ns = repair_start
        .elapsed()
        .as_nanos()
        .try_into()
        .unwrap_or(u64::MAX);
    // No lane is committed until both repair identities and lengths validate.
    let mut actual = certification.output_bf16;
    apply_owner_repairs(&mut actual, &certification.repair_mask, &repair_responses)?;
    if actual != oracle_bf16 {
        return Err(format!(
            "certified request {request_id} BF16 mismatch count {}",
            bf16_mismatch_count(&actual, oracle_bf16)
        ));
    }
    let e2e_ns = start.elapsed().as_nanos().try_into().unwrap_or(u64::MAX);
    let worker_ns = certified_responses
        .iter()
        .map(|response| response.worker_ns)
        .max()
        .unwrap()
        .saturating_add(
            repair_responses
                .iter()
                .map(|response| response.worker_ns)
                .max()
                .unwrap(),
        );
    Ok(LoopbackSample {
        e2e_ns,
        worker_ns,
        codec_tcp_ns: e2e_ns.saturating_sub(worker_ns),
        repair_ns,
        gpu_ns: certified_responses
            .iter()
            .map(|response| response.gpu_ns)
            .sum(),
        repair_lanes,
        bytes: LoopbackWireBytes {
            initial_requests: frames.iter().map(wire_frame_bytes).sum(),
            audit_responses: 0,
            certified_responses: certified_frames.iter().map(wire_frame_bytes).sum(),
            repair_requests: repair_request_frames.iter().map(wire_frame_bytes).sum(),
            repair_responses: repair_response_frames.iter().map(wire_frame_bytes).sum(),
        },
    })
}
