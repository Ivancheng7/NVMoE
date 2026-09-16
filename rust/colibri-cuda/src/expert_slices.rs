use super::*;

pub(crate) fn run_real_v4_expert0_slice_if_available(
    context: &std::sync::Arc<CudaContext>,
    module: &kernels::LoadedModule,
    stream: &std::sync::Arc<CudaStream>,
) -> Result<(), Box<dyn Error>> {
    const WARM_REQUESTS: usize = 100;
    let path = v4_expert_shard()?;

    let range_load_start = std::time::Instant::now();
    let shard = safetensors::ShardFile::open(&path)?;
    let weights = v4_expert::RoutedExpert::load_layer(&shard, 0, 0)?;
    let range_load_elapsed = range_load_start.elapsed();
    let weight_bytes = weights.h2d_bytes();
    let route_weight = 0.625;
    let input = deterministic_v4_bf16(1, 97);
    v4_expert::validate_input(&input, route_weight)?;
    let cpu_start = std::time::Instant::now();
    let expected = v4_expert::forward_cpu(&weights, &input, route_weight)?;
    let cpu_elapsed = cpu_start.elapsed();

    let slot_start = std::time::Instant::now();
    let mut slot = ExpertSlot::new(context, module, stream, &weights)?;
    let slot_elapsed = slot_start.elapsed();
    drop(weights);
    let (warmup, _) = slot.execute(&input, route_weight)?;
    assert_v4_expert_close(&warmup, &expected);

    let mut direct_ns = Vec::with_capacity(WARM_REQUESTS);
    let mut direct_gpu_ns = Vec::with_capacity(WARM_REQUESTS);
    let mut direct_error = 0.0_f32;
    for _ in 0..WARM_REQUESTS {
        let start = std::time::Instant::now();
        let (output, gpu_ns) = slot.execute(&input, route_weight)?;
        direct_ns.push(start.elapsed().as_nanos().try_into().unwrap_or(u64::MAX));
        direct_gpu_ns.push(gpu_ns);
        direct_error = direct_error.max(v4_expert_maximum_error(&output, &expected));
    }

    let listener = std::net::TcpListener::bind(("127.0.0.1", 0))?;
    let address = listener.local_addr()?;
    let (
        round_trip_ns,
        worker_ns,
        gpu_ns,
        tcp_codec_ns,
        network_error,
        request_bytes,
        response_bytes,
    ) = std::thread::scope(|scope| -> Result<_, String> {
        let worker = scope.spawn(move || -> Result<(), String> {
            let (mut socket, peer) = listener.accept().map_err(|error| error.to_string())?;
            if !peer.ip().is_loopback() {
                return Err("loopback smoke accepted a non-loopback peer".into());
            }
            socket
                .set_nodelay(true)
                .map_err(|error| error.to_string())?;
            for index in 0..WARM_REQUESTS {
                let frame =
                    v4_federated::receive_frame(&mut socket).map_err(|error| error.to_string())?;
                let request =
                    v4_federated::decode_request(&frame).map_err(|error| error.to_string())?;
                if (request.layer, request.expert, request.rows) != (0, 0, 1)
                    || (
                        request.session_epoch,
                        request.request_id,
                        request.deadline_ns,
                    ) != (1, index as u64 + 1, 0)
                {
                    return Err("worker rejected the fixed warm-request identity".into());
                }
                let worker_start = std::time::Instant::now();
                let (output, gpu_ns) = slot
                    .execute(&request.hidden_bf16, request.route_weight)
                    .map_err(|error| {
                        eprintln!("V4 resident slot request {index} failed: {error}");
                        error.to_string()
                    })?;
                let worker_ns = worker_start
                    .elapsed()
                    .as_nanos()
                    .try_into()
                    .unwrap_or(u64::MAX);
                let response =
                    v4_federated::ExecuteResponse::for_request(&request, worker_ns, gpu_ns, output);
                let frame =
                    v4_federated::encode_response(&response).map_err(|error| error.to_string())?;
                v4_federated::send_frame(&mut socket, &frame).map_err(|error| error.to_string())?;
            }
            Ok(())
        });

        let mut socket =
            std::net::TcpStream::connect(address).map_err(|error| error.to_string())?;
        socket
            .set_nodelay(true)
            .map_err(|error| error.to_string())?;
        let mut request = v4_federated::ExecuteRequest::live(input.clone(), route_weight);
        let mut round_trip_ns = Vec::with_capacity(WARM_REQUESTS);
        let mut worker_ns = Vec::with_capacity(WARM_REQUESTS);
        let mut gpu_ns = Vec::with_capacity(WARM_REQUESTS);
        let mut tcp_codec_ns = Vec::with_capacity(WARM_REQUESTS);
        let mut network_error = 0.0_f32;
        let mut request_bytes = 0;
        let mut response_bytes = 0;
        for index in 0..WARM_REQUESTS {
            request.request_id = index as u64 + 1;
            let start = std::time::Instant::now();
            let request_frame =
                v4_federated::encode_request(&request).map_err(|error| error.to_string())?;
            v4_federated::send_frame(&mut socket, &request_frame)
                .map_err(|error| error.to_string())?;
            let response_frame =
                v4_federated::receive_frame(&mut socket).map_err(|error| error.to_string())?;
            let response = v4_federated::decode_response(&response_frame)
                .map_err(|error| error.to_string())?;
            v4_federated::validate_response(&request, &response)
                .map_err(|error| error.to_string())?;
            let elapsed_ns = start.elapsed().as_nanos().try_into().unwrap_or(u64::MAX);
            round_trip_ns.push(elapsed_ns);
            worker_ns.push(response.worker_ns);
            gpu_ns.push(response.gpu_ns);
            tcp_codec_ns.push(elapsed_ns.saturating_sub(response.worker_ns));
            network_error =
                network_error.max(v4_expert_maximum_error(&response.output_f32, &expected));
            request_bytes = request_frame.len() + 4;
            response_bytes = response_frame.len() + 4;
        }
        worker
            .join()
            .map_err(|_| "loopback worker panicked".to_string())??;
        Ok((
            round_trip_ns,
            worker_ns,
            gpu_ns,
            tcp_codec_ns,
            network_error,
            request_bytes,
            response_bytes,
        ))
    })
    .map_err(|error| -> Box<dyn Error> { error.into() })?;

    let (direct_median, direct_p95) = median_p95_ms(&direct_ns);
    let (direct_gpu_median, direct_gpu_p95) = median_p95_ms(&direct_gpu_ns);
    let (round_trip_median, round_trip_p95) = median_p95_ms(&round_trip_ns);
    let (worker_median, worker_p95) = median_p95_ms(&worker_ns);
    let (gpu_median, gpu_p95) = median_p95_ms(&gpu_ns);
    let (tcp_codec_median, tcp_codec_p95) = median_p95_ms(&tcp_codec_ns);
    println!(
        "COLD V4 expert-0: range load={:.3} ms, pinned staging+alloc+weight H2D={:.3} ms, weight H2D={} B, CPU oracle={:.3} ms",
        range_load_elapsed.as_secs_f64() * 1_000.0,
        slot_elapsed.as_secs_f64() * 1_000.0,
        weight_bytes,
        cpu_elapsed.as_secs_f64() * 1_000.0,
    );
    println!(
        "WARM direct-local n={WARM_REQUESTS}: service median/p95={direct_median:.3}/{direct_p95:.3} ms, CUDA median/p95={direct_gpu_median:.3}/{direct_gpu_p95:.3} ms, max error={direct_error:.6e}",
    );
    println!(
        "WARM loopback n={WARM_REQUESTS}: RTT median/p95={round_trip_median:.3}/{round_trip_p95:.3} ms, worker median/p95={worker_median:.3}/{worker_p95:.3} ms, CUDA median/p95={gpu_median:.3}/{gpu_p95:.3} ms, TCP+codec median/p95={tcp_codec_median:.3}/{tcp_codec_p95:.3} ms, max error={network_error:.6e}",
    );
    println!(
        "WARM loopback bytes/request: input H2D={} B, weight H2D=0 B, TCP request={request_bytes} B, TCP response={response_bytes} B",
        input.len() * std::mem::size_of::<u16>(),
    );
    run_v4_block_certificate_experiment(context, module, stream, &shard)?;
    run_v4_two_worker_loopback(context, module, &shard, &path, None)?;
    run_v4_replica_recovery(context, module, &shard, &path)?;
    Ok(())
}

pub(crate) fn run_v4_block_certificate_experiment(
    context: &std::sync::Arc<CudaContext>,
    module: &kernels::LoadedModule,
    stream: &std::sync::Arc<CudaStream>,
    shard: &safetensors::ShardFile,
) -> Result<(), Box<dyn Error>> {
    const BLOCK_SIZES: [usize; 7] = [1, 8, 16, 32, 64, 128, 256];
    const PLACEMENTS: [(&str, [usize; v4_router::V4_TOPK]); 2] = [
        ("contiguous", [0, 0, 0, 1, 1, 1]),
        ("round-robin", [0, 1, 0, 1, 0, 1]),
    ];

    let norm = read_v4_bf16(shard, "layers.0.ffn_norm.weight", &[7_168])?;
    let gate = read_v4_bf16(shard, "layers.0.ffn.gate.weight", &[384, 7_168])?;
    println!(
        "BLOCK_CERT_PREREG train_tokens=0..3 holdout_tokens=4..7 seeds=97+token candidates={BLOCK_SIZES:?} placements=contiguous,round-robin selection=min_train_median_wire_bytes,tie=smaller_block holdout=selected_block_once primary=pooled_holdout_median<=50%_audit_and_zero_mismatch"
    );
    println!(
        "BLOCK_CERT_WIRE audit={}B=6*(16B_vector_header+7168*4B); candidate=owner_grouped_vectors+8B_certificate_header/owner+1B_bound/owner/block+16B_repair_request_header/owner+896B_bitmask/owner+16B_repair_response_header/owner+4B_expert_id_count_header+4B/raw_FP32_repair",
        v4_certified::AUDIT_WIRE_BYTES
    );
    println!(
        "split,token,seed,placement,block,owners,fp32_mismatch,bf16_mismatch,certified,repair,contained,repaired_mismatch,certificate_bytes,wire_bytes,audit_bytes,wire_over_audit,max_difference,max_expert_error"
    );

    let mut train_wire: [Vec<usize>; BLOCK_SIZES.len()] = std::array::from_fn(|_| Vec::new());
    let mut holdout_wire: [Vec<usize>; PLACEMENTS.len()] = std::array::from_fn(|_| Vec::new());
    let mut selected_block = None;
    for token_id in 0..8_i64 {
        if token_id == 4 {
            let mut best_index = 0;
            let mut best_score = median_twice_usize(&train_wire[0]);
            for (index, samples) in train_wire.iter().enumerate() {
                let score = median_twice_usize(samples);
                println!(
                    "BLOCK_CERT_TRAIN block={} n={} median_wire={:.1}B",
                    BLOCK_SIZES[index],
                    samples.len(),
                    score as f64 / 2.0
                );
                if score < best_score {
                    best_index = index;
                    best_score = score;
                }
            }
            selected_block = Some(BLOCK_SIZES[best_index]);
            println!(
                "BLOCK_CERT_SELECTED block={} train_median_wire={:.1}B",
                BLOCK_SIZES[best_index],
                best_score as f64 / 2.0
            );
        }

        let seed = 97 + usize::try_from(token_id)?;
        let input = deterministic_v4_bf16(1, seed);
        let routed_input = v4_router::rmsnorm_bf16(&input, &norm)?;
        let logits = v4_router::project_gate_logits(&routed_input, &gate)?;
        let hash_ids = read_v4_hash_rows(shard, "layers.0.ffn.gate.tid2eid", &[token_id])?;
        let routes = v4_router::select_hash(&logits, &hash_ids)?;
        println!(
            "BLOCK_CERT_TOKEN split={} token={} seed={} expert_ids={:?} gate_weights={:?}",
            if token_id < 4 { "train" } else { "holdout" },
            token_id,
            seed,
            routes.ids,
            routes.weights
        );

        let mut contributions = Vec::with_capacity(v4_router::V4_TOPK);
        let mut maximum_expert_error = 0.0_f32;
        for (&expert_id, &route_weight) in routes.ids.iter().zip(&routes.weights) {
            let expert_id = u16::try_from(expert_id)?;
            let weights = v4_expert::RoutedExpert::load(shard, expert_id)?;
            let expected = v4_expert::forward_cpu(&weights, &routed_input, route_weight)?;
            let mut slot = ExpertSlot::new(context, module, stream, &weights)?;
            let (actual, _) = slot.execute(&routed_input, route_weight)?;
            maximum_expert_error =
                maximum_expert_error.max(v4_expert_maximum_error(&actual, &expected));
            contributions.push(actual);
        }

        let mut audit_bits = None;
        for (placement_index, (placement, owners)) in PLACEMENTS.iter().enumerate() {
            for (block_index, &block_size) in BLOCK_SIZES.iter().enumerate() {
                if token_id >= 4 && Some(block_size) != selected_block {
                    continue;
                }
                let report = v4_certified::evaluate(&contributions, owners, block_size)?;
                assert_eq!(
                    report.bound_contained_lanes,
                    v4_expert::V4_HIDDEN,
                    "token={token_id} {placement} block={block_size}: containment"
                );
                assert_eq!(
                    report.repaired_bf16, report.audit_bf16,
                    "token={token_id} {placement} block={block_size}: repaired bits"
                );
                if let Some(expected) = &audit_bits {
                    assert_eq!(
                        &report.audit_bf16, expected,
                        "token={token_id} {placement} block={block_size}: audit bits"
                    );
                } else {
                    audit_bits = Some(report.audit_bf16.clone());
                }
                let repaired_mismatch = report
                    .repaired_bf16
                    .iter()
                    .zip(&report.audit_bf16)
                    .filter(|(actual, expected)| actual != expected)
                    .count();
                let wire_bytes = v4_certified::wire_bytes(&report);
                if token_id < 4 {
                    train_wire[block_index].push(wire_bytes);
                } else {
                    holdout_wire[placement_index].push(wire_bytes);
                }
                println!(
                    "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{:.6},{:.6e},{:.6e}",
                    if token_id < 4 { "train" } else { "holdout" },
                    token_id,
                    seed,
                    placement,
                    report.block_size,
                    report.owner_count,
                    report.grouped_f32_mismatches,
                    report.grouped_bf16_mismatches,
                    report.certified_lanes,
                    report.repair_lanes,
                    report.bound_contained_lanes,
                    repaired_mismatch,
                    report.certificate_bytes,
                    wire_bytes,
                    v4_certified::AUDIT_WIRE_BYTES,
                    wire_bytes as f64 / v4_certified::AUDIT_WIRE_BYTES as f64,
                    report.maximum_difference,
                    maximum_expert_error,
                );
            }
        }
    }

    let pooled: Vec<_> = holdout_wire.iter().flatten().copied().collect();
    let holdout_score = median_twice_usize(&pooled);
    for (index, (placement, _)) in PLACEMENTS.iter().enumerate() {
        println!(
            "BLOCK_CERT_HOLDOUT placement={placement} block={} n={} median_wire={:.1}B ratio={:.6}",
            selected_block.unwrap(),
            holdout_wire[index].len(),
            median_twice_usize(&holdout_wire[index]) as f64 / 2.0,
            median_twice_usize(&holdout_wire[index]) as f64
                / (2.0 * v4_certified::AUDIT_WIRE_BYTES as f64),
        );
    }
    let passes_two_x = holdout_score <= v4_certified::AUDIT_WIRE_BYTES;
    println!(
        "BLOCK_CERT_PRIMARY block={} n={} pooled_holdout_median={:.1}B audit={}B ratio={:.6} repaired_mismatch=0 result={}",
        selected_block.unwrap(),
        pooled.len(),
        holdout_score as f64 / 2.0,
        v4_certified::AUDIT_WIRE_BYTES,
        holdout_score as f64 / (2.0 * v4_certified::AUDIT_WIRE_BYTES as f64),
        if passes_two_x {
            "PASS_2X"
        } else {
            "STOP_LT_2X_NO_RETUNE"
        },
    );
    println!(
        "BLOCK_CERT_EXTERNAL_VALIDITY official DeepSeek-V4-Pro L0 router/hash and L0 FP4 experts; token_ids=0..7 deterministic BF16 rows; one Ada GPU; byte accounting is protocol-model arithmetic, not measured TCP/QUIC latency or multi-worker throughput"
    );
    Ok(())
}
