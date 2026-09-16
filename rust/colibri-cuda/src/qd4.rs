use super::*;

impl<'a> Qd4ImmutableExpertCache<'a> {
    pub(crate) fn new(
        context: &std::sync::Arc<CudaContext>,
        module: &'a kernels::LoadedModule,
    ) -> Result<Self, Box<dyn Error>> {
        let cache_stream = context.new_stream()?;
        let free = (0..V4_QD4_IMMUTABLE_EXPERT_CACHE_SLOTS)
            .map(|_| ExpertSlot::new_empty(context, module, &cache_stream))
            .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
        Ok(Self {
            free,
            layer: None,
            resident: BTreeMap::new(),
            telemetry: Qd4ImmutableExpertCacheTelemetry::default(),
        })
    }

    pub(crate) fn begin_layer(&mut self, layer: usize) -> Result<(), Box<dyn Error>> {
        if self.layer.is_some()
            || !self.resident.is_empty()
            || self.free.len() != V4_QD4_IMMUTABLE_EXPERT_CACHE_SLOTS
        {
            return Err(
                "QD4 immutable expert cache crossed a layer boundary before completion".into(),
            );
        }
        self.layer = Some(layer);
        Ok(())
    }

    pub(crate) fn lease_routes(
        &mut self,
        layer: usize,
        routes: &v4_router::RouterSelection,
    ) -> Result<[Qd4ImmutableCacheLease<'a>; v4_router::V4_TOPK], Box<dyn Error>> {
        if self.layer != Some(layer) {
            return Err(
                "QD4 immutable expert cache route layer does not match its active layer".into(),
            );
        }
        let mut seen = BTreeSet::new();
        let leases = routes
            .ids
            .iter()
            .map(|&id| {
                let expert = u16::try_from(id)?;
                if !seen.insert(expert) {
                    return Err("QD4 immutable expert cache route IDs must be unique".into());
                }
                if let Some(slot) = self.resident.remove(&expert) {
                    self.telemetry.hits = self.telemetry.hits.saturating_add(1);
                    Ok(Qd4ImmutableCacheLease {
                        expert,
                        slot,
                        miss: false,
                    })
                } else {
                    let slot = self
                        .free
                        .pop()
                        .ok_or("QD4 immutable expert cache free pool exhausted")?;
                    self.telemetry.expert_union_loads =
                        self.telemetry.expert_union_loads.saturating_add(1);
                    Ok(Qd4ImmutableCacheLease {
                        expert,
                        slot,
                        miss: true,
                    })
                }
            })
            .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
        leases
            .try_into()
            .map_err(|_| "QD4 immutable expert cache route count changed".into())
    }

    pub(crate) fn return_lease(
        &mut self,
        layer: usize,
        lease: Qd4ImmutableCacheLease<'a>,
    ) -> Result<(), Box<dyn Error>> {
        if self.layer != Some(layer) {
            return Err(
                "QD4 immutable expert cache return layer does not match active layer".into(),
            );
        }
        if self.resident.insert(lease.expert, lease.slot).is_some() {
            return Err("QD4 immutable expert cache returned duplicate expert identity".into());
        }
        self.telemetry.d2d_copies = self.telemetry.d2d_copies.saturating_add(1);
        self.telemetry.d2d_bytes = self
            .telemetry
            .d2d_bytes
            .saturating_add(v4_expert::V4_ROUTED_EXPERT_BYTES);
        Ok(())
    }

    pub(crate) fn record_batch(&mut self, misses: usize) {
        self.telemetry.route_executions = self
            .telemetry
            .route_executions
            .saturating_add(v4_router::V4_TOPK);
        self.telemetry.source_loads = self.telemetry.source_loads.saturating_add(misses);
        self.telemetry.source_bytes = self
            .telemetry
            .source_bytes
            .saturating_add(misses.saturating_mul(v4_expert::V4_ROUTED_EXPERT_BYTES));
        self.telemetry.h2d_loads = self.telemetry.h2d_loads.saturating_add(misses);
        self.telemetry.h2d_bytes = self
            .telemetry
            .h2d_bytes
            .saturating_add(misses.saturating_mul(v4_expert::V4_ROUTED_EXPERT_BYTES));
    }

    pub(crate) fn finish_layer(&mut self, layer: usize) -> Result<(), Box<dyn Error>> {
        if self.layer != Some(layer)
            || self.free.len() + self.resident.len() != V4_QD4_IMMUTABLE_EXPERT_CACHE_SLOTS
        {
            return Err("QD4 immutable expert cache layer did not return all slot leases".into());
        }
        if self.resident.len() > V4_QD4_IMMUTABLE_EXPERT_CACHE_SLOTS {
            return Err("QD4 immutable expert cache layer exceeded 32-slot occupancy".into());
        }
        self.telemetry.max_occupancy = self.telemetry.max_occupancy.max(self.resident.len());
        self.telemetry.completed_layers = self.telemetry.completed_layers.saturating_add(1);
        self.free
            .extend(std::mem::take(&mut self.resident).into_values());
        self.layer = None;
        Ok(())
    }
}

pub(crate) fn run_supervised_qd4_batches<W, T, R, J>(
    workers: Vec<W>,
    batches: usize,
    timeout: std::time::Duration,
    profile_origin: Option<std::time::Instant>,
    mut make_job: impl FnMut(usize) -> Result<J, String>,
    work: impl Fn(&mut W, usize, RoutedLaneJob<J>) -> Result<T, String> + Sync,
    mut finish: impl FnMut(usize, SupervisedQd4Batch<T>) -> Result<R, String>,
) -> Result<Vec<R>, String>
where
    W: Send,
    T: Send,
    J: Send,
{
    if workers.is_empty() || workers.len() > v4_router::V4_TOPK || batches == 0 {
        return Err("QD4 supervisor received an invalid lane/batch count".into());
    }
    std::thread::scope(|scope| {
        let (done_tx, done_rx) =
            std::sync::mpsc::channel::<Result<SupervisedLaneDone<T>, String>>();
        let mut senders = Vec::with_capacity(workers.len());
        let mut handles = Vec::with_capacity(workers.len());
        for (lane, mut worker) in workers.into_iter().enumerate() {
            let (job_tx, job_rx) = std::sync::mpsc::sync_channel::<RoutedLaneJob<J>>(1);
            senders.push(job_tx);
            let worker_done = done_tx.clone();
            let worker_work = &work;
            handles.push(scope.spawn(move || -> Result<(), String> {
                while let Ok(job) = job_rx.recv() {
                    match worker_work(&mut worker, lane, job) {
                        Ok(value) => worker_done
                            .send(Ok(SupervisedLaneDone { lane, value }))
                            .map_err(|error| error.to_string())?,
                        Err(error) => {
                            let _ = worker_done.send(Err(error.clone()));
                            return Err(error);
                        }
                    }
                }
                Ok(())
            }));
        }
        drop(done_tx);

        let scheduled = (|| {
            let mut results = Vec::with_capacity(batches);
            for batch in 0..batches {
                let iteration_started = std::time::Instant::now();
                let mut submit_ns = [0_u64; v4_router::V4_TOPK];
                let mut next_route = 0_usize;
                for sender in &senders {
                    submit_ns[next_route] =
                        u64::try_from(iteration_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
                    sender
                        .send(RoutedLaneJob {
                            route_slot: next_route,
                            iteration_started,
                            profile_submit_ns: profile_origin.map(timeline_elapsed_ns),
                            payload: make_job(next_route)?,
                        })
                        .map_err(|error| error.to_string())?;
                    next_route += 1;
                }
                let mut completions = Vec::with_capacity(v4_router::V4_TOPK);
                for _ in 0..v4_router::V4_TOPK {
                    let done = match done_rx.recv_timeout(timeout) {
                        Ok(result) => result?,
                        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                            return Err(format!(
                                "QD4 batch {batch} timed out waiting for a lane result"
                            ));
                        }
                        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                            return Err(format!("QD4 batch {batch} lost a lane result channel"));
                        }
                    };
                    if next_route < v4_router::V4_TOPK {
                        submit_ns[next_route] =
                            u64::try_from(iteration_started.elapsed().as_nanos())
                                .unwrap_or(u64::MAX);
                        senders[done.lane]
                            .send(RoutedLaneJob {
                                route_slot: next_route,
                                iteration_started,
                                profile_submit_ns: profile_origin.map(timeline_elapsed_ns),
                                payload: make_job(next_route)?,
                            })
                            .map_err(|error| error.to_string())?;
                        next_route += 1;
                    }
                    completions.push(done.value);
                }
                results.push(finish(
                    batch,
                    SupervisedQd4Batch {
                        iteration_started,
                        submit_ns,
                        completions,
                    },
                )?);
            }
            Ok(results)
        })();

        drop(senders);
        let mut worker_failure = None;
        for handle in handles {
            match handle.join() {
                Ok(Ok(())) => {}
                Ok(Err(error)) => {
                    worker_failure.get_or_insert(error);
                }
                Err(_) => {
                    worker_failure.get_or_insert("QD4 lane worker panicked".into());
                }
            };
        }
        match (scheduled, worker_failure) {
            (Ok(results), None) => Ok(results),
            (Ok(_), Some(error)) => Err(error),
            (Err(error), None) => Err(error),
            (Err(error), Some(worker)) => Err(format!("{error}; {worker}")),
        }
    })
}

pub(crate) fn run_qd4_routed_replays(
    context: &std::sync::Arc<CudaContext>,
    module: &kernels::LoadedModule,
    shard: &safetensors::ShardFile,
    input: &[u16],
    routes: &v4_router::RouterSelection,
    shared: &[f32],
    plans: &[v4_expert::RoutedExpertRangePlan; v4_router::V4_TOPK],
    expected: &[u16],
    trials: usize,
) -> Result<(u64, Vec<RoutedReplaySample>), Box<dyn Error>> {
    const LANES: usize = 4;
    let setup_started = std::time::Instant::now();
    let lanes = (0..LANES)
        .map(|_| {
            let stream = context.new_stream()?;
            ExpertSlot::new_empty(context, module, &stream)
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    let startup_setup_ns = u64::try_from(setup_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
    let outputs = (0..v4_router::V4_TOPK)
        .map(|_| std::sync::Mutex::new(vec![0.0_f32; v4_expert::V4_HIDDEN]))
        .collect::<Vec<_>>();
    let samples = run_supervised_qd4_batches(
        lanes,
        trials + 1,
        std::time::Duration::from_secs(30),
        None,
        |_| Ok(()),
        |lane, _lane_index, job| {
            let job_started = std::time::Instant::now();
            let read_started_ns =
                u64::try_from(job.iteration_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
            let mut output = outputs[job.route_slot]
                .lock()
                .map_err(|_| "QD4 output mutex poisoned".to_owned())?;
            lane.load_range_execute_into(
                shard,
                &plans[job.route_slot],
                input,
                routes.weights[job.route_slot],
                &mut output,
                None,
                None,
            )
            .map_err(|error| error.to_string())
            .map(|execution| RoutedLaneDone {
                route_slot: job.route_slot,
                read_ns: execution.read_ns,
                read_ready_ns: read_started_ns.saturating_add(execution.read_ns),
                h2d_ns: execution.h2d_ns,
                kernel_ns: execution.kernel_ns,
                ready_ns: u64::try_from(job.iteration_started.elapsed().as_nanos())
                    .unwrap_or(u64::MAX),
                job_ns: u64::try_from(job_started.elapsed().as_nanos()).unwrap_or(u64::MAX),
                timeline: execution.timeline,
                cache_lease: job.payload,
            })
        },
        |trial, batch| {
            let mut submit_ns = [0_u64; v4_router::V4_TOPK];
            let mut ready_ns = [0_u64; v4_router::V4_TOPK];
            let mut read_ns = 0_u64;
            let mut read_wall_ns = 0_u64;
            let mut h2d_ns = 0_u64;
            let mut kernel_ns = 0_u64;
            let mut lane_job_ns = 0_u64;
            submit_ns.copy_from_slice(&batch.submit_ns);
            for done in batch.completions {
                ready_ns[done.route_slot] = done.ready_ns;
                read_ns = read_ns.saturating_add(done.read_ns);
                read_wall_ns = read_wall_ns.max(done.read_ready_ns);
                h2d_ns = h2d_ns.saturating_add(done.h2d_ns);
                kernel_ns = kernel_ns.saturating_add(done.kernel_ns);
                lane_job_ns = lane_job_ns.saturating_add(done.job_ns);
            }
            let guards: [std::sync::MutexGuard<'_, Vec<f32>>; v4_router::V4_TOPK] =
                std::array::from_fn(|index| outputs[index].lock().unwrap());
            let routed: [(u16, &[f32]); v4_router::V4_TOPK] = std::array::from_fn(|index| {
                (
                    u16::try_from(routes.ids[index]).unwrap(),
                    guards[index].as_slice(),
                )
            });
            let actual = v4_expert::reduce_moe_canonical(&routed, shared)
                .map_err(|error| error.to_string())?;
            if actual != expected {
                return Err(format!("QD4 trial {trial} changed canonical routed output"));
            }
            let inclusive_ns =
                u64::try_from(batch.iteration_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
            Ok((trial > 0).then_some(RoutedReplaySample {
                inclusive_ns,
                setup_ns: 0,
                read_ns,
                read_wall_ns,
                h2d_ns,
                kernel_ns,
                lane_occupancy: lane_job_ns as f64 / (inclusive_ns as f64 * LANES as f64),
                submit_ns,
                ready_ns,
                timeline: None,
            }))
        },
    )?
    .into_iter()
    .flatten()
    .collect();
    Ok((startup_setup_ns, samples))
}

pub(crate) fn run_qd4_routed_once<'module>(
    lanes: &mut [ExpertSlot<'module>],
    mut immutable_cache: Option<&mut Qd4ImmutableExpertCache<'module>>,
    shard: &safetensors::ShardFile,
    layer: usize,
    input: &[u16],
    routes: &v4_router::RouterSelection,
    shared: &[f32],
    plans: &[v4_expert::RoutedExpertRangePlan; v4_router::V4_TOPK],
    profile_origin: Option<std::time::Instant>,
    capture_contributions: bool,
) -> Result<Qd4RoutedRun<'module>, Box<dyn Error>> {
    if lanes.len() != 4 {
        return Err("QD4 execution requires exactly four persistent lanes".into());
    }
    let mut cache_leases = immutable_cache
        .as_deref_mut()
        .map(|cache| {
            cache
                .lease_routes(layer, routes)
                .map(|leases| leases.map(Some))
        })
        .transpose()?;
    let outputs = (0..v4_router::V4_TOPK)
        .map(|_| std::sync::Mutex::new(vec![0.0_f32; v4_expert::V4_HIDDEN]))
        .collect::<Vec<_>>();
    let lane_refs = lanes.iter_mut().collect::<Vec<_>>();
    let mut batches = run_supervised_qd4_batches(
        lane_refs,
        1,
        std::time::Duration::from_secs(30),
        profile_origin,
        |route_slot| match cache_leases.as_mut() {
            Some(leases) => leases[route_slot].take().map(Some).ok_or_else(|| {
                "QD4 immutable expert cache lease disappeared before dispatch".into()
            }),
            None => Ok(None),
        },
        |lane, _lane_index, job| {
            let route_slot = job.route_slot;
            let job_started = std::time::Instant::now();
            let read_started_ns =
                u64::try_from(job.iteration_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
            let mut output = outputs[route_slot]
                .lock()
                .map_err(|_| "QD4 output mutex poisoned".to_owned())?;
            let (execution, cache_lease) = match job.payload {
                Some(lease) if !lease.miss => {
                    let lane_stream = lane.stream.clone();
                    lane.copy_weight_tensors_from(&lease.slot, &lane_stream)
                        .map_err(|error| error.to_string())?;
                    let kernel_ns = lane
                        .execute_into(input, routes.weights[route_slot], &mut output, None, None)
                        .map_err(|error| error.to_string())?;
                    (
                        RoutedLaneExecution {
                            read_ns: 0,
                            h2d_ns: 0,
                            kernel_ns,
                            timeline: None,
                        },
                        Some(lease),
                    )
                }
                Some(mut lease) => {
                    let execution = lane
                        .load_range_execute_into(
                            shard,
                            &plans[route_slot],
                            input,
                            routes.weights[route_slot],
                            &mut output,
                            profile_origin,
                            job.profile_submit_ns,
                        )
                        .map_err(|error| error.to_string())?;
                    let lane_stream = lane.stream.clone();
                    lease
                        .slot
                        .copy_weight_tensors_from(lane, &lane_stream)
                        .map_err(|error| error.to_string())?;
                    lane_stream
                        .synchronize()
                        .map_err(|error| error.to_string())?;
                    (execution, Some(lease))
                }
                None => (
                    lane.load_range_execute_into(
                        shard,
                        &plans[route_slot],
                        input,
                        routes.weights[route_slot],
                        &mut output,
                        profile_origin,
                        job.profile_submit_ns,
                    )
                    .map_err(|error| error.to_string())?,
                    None,
                ),
            };
            Ok::<_, String>(execution).map(|execution| RoutedLaneDone {
                route_slot,
                read_ns: execution.read_ns,
                read_ready_ns: read_started_ns.saturating_add(execution.read_ns),
                h2d_ns: execution.h2d_ns,
                kernel_ns: execution.kernel_ns,
                ready_ns: u64::try_from(job.iteration_started.elapsed().as_nanos())
                    .unwrap_or(u64::MAX),
                job_ns: u64::try_from(job_started.elapsed().as_nanos()).unwrap_or(u64::MAX),
                timeline: execution.timeline,
                cache_lease,
            })
        },
        |_, batch| {
            let mut ready_ns = [0_u64; v4_router::V4_TOPK];
            let mut lane_timeline: [Option<RoutedLaneTimeline>; v4_router::V4_TOPK] =
                std::array::from_fn(|_| None);
            let mut read_ns = 0_u64;
            let mut read_wall_ns = 0_u64;
            let mut h2d_ns = 0_u64;
            let mut kernel_ns = 0_u64;
            let mut lane_job_ns = 0_u64;
            let mut cache_miss_count = 0_usize;
            let mut returned_leases = Vec::new();
            for done in batch.completions {
                ready_ns[done.route_slot] = done.ready_ns;
                read_ns = read_ns.saturating_add(done.read_ns);
                read_wall_ns = read_wall_ns.max(done.read_ready_ns);
                h2d_ns = h2d_ns.saturating_add(done.h2d_ns);
                kernel_ns = kernel_ns.saturating_add(done.kernel_ns);
                lane_job_ns = lane_job_ns.saturating_add(done.job_ns);
                lane_timeline[done.route_slot] = done.timeline;
                if let Some(lease) = done.cache_lease {
                    cache_miss_count = cache_miss_count.saturating_add(usize::from(lease.miss));
                    returned_leases.push(lease);
                }
            }
            let guards: [std::sync::MutexGuard<'_, Vec<f32>>; v4_router::V4_TOPK] =
                std::array::from_fn(|index| outputs[index].lock().unwrap());
            let routed: [(u16, &[f32]); v4_router::V4_TOPK] = std::array::from_fn(|index| {
                (
                    u16::try_from(routes.ids[index]).unwrap(),
                    guards[index].as_slice(),
                )
            });
            let canonical_reduction_start_ns = profile_origin.map(timeline_elapsed_ns);
            let output = v4_expert::reduce_moe_canonical(&routed, shared)
                .map_err(|error| error.to_string())?;
            let canonical_reduction_end_ns = profile_origin.map(timeline_elapsed_ns);
            let inclusive_ns =
                u64::try_from(batch.iteration_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
            let contributions = capture_contributions.then(|| {
                guards
                    .iter()
                    .map(|contribution| contribution.to_vec())
                    .collect()
            });
            Ok(Qd4RoutedRun {
                output,
                contributions,
                sample: RoutedReplaySample {
                    inclusive_ns,
                    setup_ns: 0,
                    read_ns,
                    read_wall_ns,
                    h2d_ns,
                    kernel_ns,
                    lane_occupancy: lane_job_ns as f64 / (inclusive_ns as f64 * 4.0),
                    submit_ns: batch.submit_ns,
                    ready_ns,
                    timeline: canonical_reduction_start_ns
                        .zip(canonical_reduction_end_ns)
                        .map(
                            |(canonical_reduction_start_ns, canonical_reduction_end_ns)| {
                                RoutedQd4Timeline {
                                    lanes: lane_timeline,
                                    canonical_reduction_start_ns,
                                    canonical_reduction_end_ns,
                                }
                            },
                        ),
                },
                cache_miss_count,
                cache_leases: returned_leases,
            })
        },
    )?;
    let mut run = batches.pop().ok_or("QD4 supervisor returned no batch")?;
    if let Some(cache) = immutable_cache {
        for lease in std::mem::take(&mut run.cache_leases) {
            cache.return_lease(layer, lease)?;
        }
        cache.record_batch(run.cache_miss_count);
    }
    Ok(run)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn maybe_run_routed_qd4_benchmark(
    context: &std::sync::Arc<CudaContext>,
    module: &kernels::LoadedModule,
    stream: &std::sync::Arc<CudaStream>,
    shard: &safetensors::ShardFile,
    layer: usize,
    input: &[u16],
    routes: &v4_router::RouterSelection,
    shared: &[f32],
) -> Result<(), Box<dyn Error>> {
    if std::env::var_os("COLIBRI_V4_ROUTED_QD4_BENCH").as_deref() != Some(std::ffi::OsStr::new("1"))
        || !matches!(layer, 3 | 4)
    {
        return Ok(());
    }
    let trials = std::env::var("COLIBRI_V4_ROUTED_QD4_TRIALS")
        .unwrap_or_else(|_| "30".into())
        .parse::<usize>()?;
    if trials < 30 || input.len() != v4_state::HIDDEN || routes.ids.len() != v4_router::V4_TOPK {
        return Err("QD4 benchmark requires >=30 trials and one complete real route row".into());
    }
    let plans: [v4_expert::RoutedExpertRangePlan; v4_router::V4_TOPK] = routes
        .ids
        .iter()
        .map(|&expert| v4_expert::RoutedExpertRangePlan::new(shard, layer, u16::try_from(expert)?))
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?
        .try_into()
        .map_err(|_| "QD4 route plan count changed")?;
    let (expected, cold_serial) =
        run_serial_routed_replay(context, module, stream, shard, layer, input, routes, shared)?;
    let mut serial = Vec::with_capacity(trials);
    for _ in 0..trials {
        let (actual, sample) =
            run_serial_routed_replay(context, module, stream, shard, layer, input, routes, shared)?;
        if actual != expected {
            return Err("serial fixed-route replay changed canonical output".into());
        }
        serial.push(sample);
    }
    let (startup_setup_ns, qd4) = run_qd4_routed_replays(
        context, module, shard, input, routes, shared, &plans, &expected, trials,
    )?;
    let (serial_inclusive_p50_ms, serial_inclusive_p95_ms) =
        routed_metric(&serial, |sample| sample.inclusive_ns);
    let (qd4_inclusive_p50_ms, qd4_inclusive_p95_ms) =
        routed_metric(&qd4, |sample| sample.inclusive_ns);
    let (serial_setup_p50_ms, serial_setup_p95_ms) =
        routed_metric(&serial, |sample| sample.setup_ns);
    let (qd4_setup_p50_ms, qd4_setup_p95_ms) = routed_metric(&qd4, |sample| sample.setup_ns);
    let (qd4_read_p50_ms, qd4_read_p95_ms) = routed_metric(&qd4, |sample| sample.read_wall_ns);
    let (qd4_h2d_p50_ms, qd4_h2d_p95_ms) = routed_metric(&qd4, |sample| sample.h2d_ns);
    let (qd4_kernel_p50_ms, qd4_kernel_p95_ms) = routed_metric(&qd4, |sample| sample.kernel_ns);
    let read_gbps = v4_expert::V4_ROUTED_EXPERT_BYTES as f64 * v4_router::V4_TOPK as f64
        / (qd4_read_p50_ms / 1_000.0)
        / 1e9;
    let inclusive_improvement = 1.0 - qd4_inclusive_p50_ms / serial_inclusive_p50_ms;
    let setup_improvement = 1.0 - qd4_setup_p50_ms / serial_setup_p50_ms;
    let pass = inclusive_improvement >= 0.20
        && setup_improvement >= 0.80
        && read_gbps >= 4.9
        && qd4_inclusive_p95_ms <= serial_inclusive_p95_ms;
    let path = std::env::var_os("COLIBRI_V4_ROUTED_QD4_RESULT_JSONL")
        .map(std::path::PathBuf::from)
        .ok_or("set COLIBRI_V4_ROUTED_QD4_RESULT_JSONL")?;
    let mut result = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    writeln!(
        result,
        "{}",
        serde_json::json!({
            "event": "routed_qd4_benchmark",
            "layer": layer,
            "compression_ratio": v4_state::compression_ratio_for_layer(layer)?.get(),
            "trials": trials,
            "route_ids": routes.ids,
            "route_weights_f32": routes.weights,
            "range_expert_ids": plans.iter().map(|plan| plan.expert_id).collect::<Vec<_>>(),
            "range_bytes_per_expert": v4_expert::V4_ROUTED_EXPERT_BYTES,
            "range_bytes_per_replay": v4_expert::V4_ROUTED_EXPERT_BYTES * v4_router::V4_TOPK,
            "output_bf16_digest": format!("{:016x}", fnv1a_u16(&expected)),
            "cold_serial": routed_replay_json(&cold_serial),
            "startup_lane_setup_ns": startup_setup_ns,
            "pinned_host_bytes": v4_expert::V4_ROUTED_EXPERT_BYTES * 4,
            "device_weight_bytes": v4_expert::V4_ROUTED_EXPERT_BYTES * 4,
            "steady_lane_alloc_count": 0,
            "serial": serial.iter().map(routed_replay_json).collect::<Vec<_>>(),
            "qd4": qd4.iter().map(routed_replay_json).collect::<Vec<_>>(),
            "summary": {
                "serial_inclusive_p50_ms": serial_inclusive_p50_ms,
                "serial_inclusive_p95_ms": serial_inclusive_p95_ms,
                "qd4_inclusive_p50_ms": qd4_inclusive_p50_ms,
                "qd4_inclusive_p95_ms": qd4_inclusive_p95_ms,
                "inclusive_improvement": inclusive_improvement,
                "serial_setup_p50_ms": serial_setup_p50_ms,
                "serial_setup_p95_ms": serial_setup_p95_ms,
                "qd4_setup_p50_ms": qd4_setup_p50_ms,
                "qd4_setup_p95_ms": qd4_setup_p95_ms,
                "setup_improvement": setup_improvement,
                "qd4_read_p50_ms": qd4_read_p50_ms,
                "qd4_read_p95_ms": qd4_read_p95_ms,
                "read_gbps": read_gbps,
                "qd4_h2d_p50_ms": qd4_h2d_p50_ms,
                "qd4_h2d_p95_ms": qd4_h2d_p95_ms,
                "qd4_kernel_p50_ms": qd4_kernel_p50_ms,
                "qd4_kernel_p95_ms": qd4_kernel_p95_ms,
                "pass": pass,
            },
        })
    )?;
    result.sync_all()?;
    Ok(())
}

pub(crate) fn run_qd4_stationary_gate_arm(
    context: &std::sync::Arc<CudaContext>,
    module: &kernels::LoadedModule,
    stream: &std::sync::Arc<CudaStream>,
    runtime: &LoadedLayerRuntime,
    token_ids: &[i64],
    hash_ids: &[i64],
    inputs: &[Vec<u16>],
    use_qd4: bool,
    use_immutable_cache: bool,
) -> Result<Qd4StationaryGateArm, Box<dyn Error>> {
    if token_ids.len() != 5 || inputs.len() != token_ids.len() {
        return Err("QD4 stationary gate requires five real prompt rows".into());
    }
    if use_immutable_cache && !use_qd4 {
        return Err("QD4 immutable cache gate requires persistent QD4 lanes".into());
    }
    let startup_started = std::time::Instant::now();
    let mut lanes = use_qd4
        .then(|| {
            (0..V4_QD4_LANES)
                .map(|_| {
                    let lane_stream = context.new_stream()?;
                    ExpertSlot::new_empty(context, module, &lane_stream)
                })
                .collect::<Result<Vec<_>, Box<dyn Error>>>()
        })
        .transpose()?;
    let mut immutable_cache = use_immutable_cache
        .then(|| Qd4ImmutableExpertCache::new(context, module))
        .transpose()?;
    let startup_setup_ns = use_qd4
        .then(|| u64::try_from(startup_started.elapsed().as_nanos()).unwrap_or(u64::MAX))
        .unwrap_or(0);
    let device_used_after_startup_bytes = device_used_bytes()?;
    let mut persistent = v4_state::V4SerialPersistentState::new(token_ids.len())?;
    let mut expert_slots = (!use_qd4).then(BTreeMap::new);
    let mut positions = Vec::with_capacity(token_ids.len());
    let mut routed_wall_ns = 0_u64;
    let mut routed_source_bytes = 0_usize;
    let mut routed_h2d_bytes = 0_usize;
    if let Some(cache) = immutable_cache.as_mut() {
        cache.begin_layer(runtime.weights.layer)?;
    }
    for position in 0..token_ids.len() {
        let row_hash_ids = if runtime.weights.hash_router {
            &hash_ids[position * v4_router::V4_TOPK..(position + 1) * v4_router::V4_TOPK]
        } else {
            &[]
        };
        let mut completed = None;
        persistent.transaction(runtime.weights.layer, |state| {
            let run = run_real_layer_rows(
                context,
                module,
                stream,
                None,
                runtime,
                lanes.as_deref_mut(),
                immutable_cache.as_mut(),
                expert_slots.as_mut(),
                row_hash_ids,
                &inputs[position],
                &[token_ids[position]],
                position,
                Some(state),
                None,
                None,
                false,
                None,
                true,
                None,
                None,
            None,
            )?;
            let audit = run
                .routed_audit
                .ok_or("QD4 stationary gate omitted routed contribution audit")?;
            let position_result = Qd4StationaryGatePosition {
                position,
                route_ids: run.routes.ids,
                route_weight_bits: run
                    .routes
                    .weights
                    .iter()
                    .map(|value| value.to_bits())
                    .collect(),
                attention_topk_digest: run.attention_topk_digest,
                attention_topk_per_row: run.attention_topk_per_row,
                after_attention_digest: audit.after_attention_digest,
                ffn_input_digest: audit.ffn_input_digest,
                shared_digest: audit.shared_digest,
                contribution_digests: audit
                    .contributions
                    .iter()
                    .map(|value| fnv1a_f32(value))
                    .collect(),
                contributions: audit.contributions,
                moe_digest: audit.moe_digest,
                hidden_digest: fnv1a_u16(&run.output),
                state: state.clone(),
            };
            completed = Some((
                position_result,
                run.stage.routed_ns,
                run.stage.routed_source_bytes,
                run.stage.routed_h2d_bytes,
            ));
            Ok(())
        })?;
        let (result, wall_ns, source_bytes, h2d_bytes) =
            completed.ok_or("QD4 stationary arm did not commit a row")?;
        routed_wall_ns = routed_wall_ns.saturating_add(wall_ns);
        routed_source_bytes = routed_source_bytes.saturating_add(source_bytes);
        routed_h2d_bytes = routed_h2d_bytes.saturating_add(h2d_bytes);
        positions.push(result);
    }
    if let Some(cache) = immutable_cache.as_mut() {
        cache.finish_layer(runtime.weights.layer)?;
    }
    Ok(Qd4StationaryGateArm {
        positions,
        routed_wall_ns,
        routed_source_bytes,
        routed_h2d_bytes,
        startup_setup_ns,
        steady_slot_allocations: expert_slots.as_ref().map_or(0, BTreeMap::len),
        device_used_after_startup_bytes,
        worker_errors: 0,
        immutable_cache: immutable_cache.map(|cache| cache.telemetry),
    })
}

pub(crate) fn run_qd4_immutable_cache_targeted_gate(
    context: &std::sync::Arc<CudaContext>,
    module: &kernels::LoadedModule,
    stream: &std::sync::Arc<CudaStream>,
) -> Result<(), Box<dyn Error>> {
    const LAYER: usize = 3;
    let result_path = std::env::var_os("COLIBRI_V4_QD4_IMMUTABLE_CACHE_TARGETED_GATE_RESULT_JSONL")
        .map(std::path::PathBuf::from)
        .ok_or("set COLIBRI_V4_QD4_IMMUTABLE_CACHE_TARGETED_GATE_RESULT_JSONL")?;
    if std::env::var_os("COLIBRI_V4_FULL61_B1").is_some()
        || std::env::var_os("COLIBRI_V4_LAYER_STATIONARY_PREFILL").is_some()
        || std::env::var_os("COLIBRI_V4_ROUTED_QD4").is_some()
        || std::env::var_os("COLIBRI_V4_ROUTED_QD4_BENCH").is_some()
        || std::env::var_os("COLIBRI_V4_QD4_STATIONARY_GATE").is_some()
        || std::env::var_os("COLIBRI_V4_QD4_IMMUTABLE_EXPERT_CACHE").is_some()
        || parse_gib_budget("COLIBRI_V4_DEVICE_FIXED_CACHE_GIB")? != 0
        || parse_gib_budget("COLIBRI_V4_HOST_FIXED_CACHE_GIB")? != 0
    {
        return Err(
            "QD4 immutable-cache targeted gate is isolated from full61, stationary, QD4, and fixed-cache modes"
                .into(),
        );
    }
    let required_host = V4_QD4_PINNED_HOST_BYTES + V4_QD4_IMMUTABLE_EXPERT_CACHE_HOST_BYTES;
    let required_device = V4_QD4_DEVICE_BYTES + V4_QD4_IMMUTABLE_EXPERT_CACHE_DEVICE_BYTES;
    let host_available = host_available_bytes()?;
    let device_available = device_available_bytes()?;
    if host_available < required_host || device_available < required_device {
        return Err(
            "QD4 immutable-cache targeted gate lacks the explicit lane/cache reserve".into(),
        );
    }

    let mut result = PendingResultFile::reserve(result_path)?;
    let model_dir = v4_model_dir()?;
    let token_ids = qd4_gate_prompt_ids(&model_dir)?;
    let inputs =
        prepare_qd4_stationary_gate_inputs(context, module, stream, &model_dir, &token_ids, LAYER)?;
    let input = inputs
        .get(&LAYER)
        .and_then(|rows| rows.first())
        .ok_or("QD4 targeted prefix omitted L3 position zero input")?;
    let (runtime, _) = load_layer_runtime(&model_dir, LAYER, &token_ids, stream)?;
    let hash_ids = load_layer_hash_ids(&model_dir, LAYER, &token_ids)?;
    let mut persistent = v4_state::V4SerialPersistentState::new(token_ids.len())?;
    let mut prepared = None;
    persistent.transaction(LAYER, |state| {
        let attention = run_real_layer_attention_rows(
            module, stream, None, &runtime, input, 0, state, None, None,
        )?;
        let ffn_hc = run_l0_mhc_pre(
            module,
            stream,
            &attention.output,
            &runtime.weights.hc_ffn_fn,
            &runtime.weights.hc_ffn_scale,
            &runtime.weights.hc_ffn_base,
        )?;
        let row_hash_ids = if runtime.weights.hash_router {
            &hash_ids[..v4_router::V4_TOPK]
        } else {
            &[]
        };
        let router = run_layer_router(
            module,
            stream,
            None,
            &ffn_hc.collapsed,
            &runtime.weights,
            row_hash_ids,
            false,
        )?;
        let shared = run_layer_shared_expert(module, stream, &router.input, &runtime.weights)?;
        prepared = Some((router.input, router.routes, shared));
        Ok(())
    })?;
    let (ffn_input, routes, shared) =
        prepared.ok_or("QD4 targeted router preparation disappeared")?;
    if routes.ids.len() != v4_router::V4_TOPK
        || routes.weights.len() != v4_router::V4_TOPK
        || routes.ids.iter().collect::<BTreeSet<_>>().len() != v4_router::V4_TOPK
    {
        return Err("QD4 targeted L3 route batch must contain six unique experts".into());
    }
    let state_before_routed = persistent
        .layer(LAYER)
        .ok_or("QD4 targeted L3 persistent state disappeared")?
        .clone();
    let plans: [v4_expert::RoutedExpertRangePlan; v4_router::V4_TOPK] = routes
        .ids
        .iter()
        .map(|&expert| {
            v4_expert::RoutedExpertRangePlan::new(
                &runtime.expert_shard,
                LAYER,
                u16::try_from(expert)?,
            )
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?
        .try_into()
        .map_err(|_| "QD4 targeted route plan count changed")?;
    let mut lanes = (0..V4_QD4_LANES)
        .map(|_| {
            let lane_stream = context.new_stream()?;
            ExpertSlot::new_empty(context, module, &lane_stream)
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    let mut cache = Qd4ImmutableExpertCache::new(context, module)?;
    cache.begin_layer(LAYER)?;
    let first = run_qd4_routed_once(
        &mut lanes,
        Some(&mut cache),
        &runtime.expert_shard,
        LAYER,
        &ffn_input,
        &routes,
        &shared,
        &plans,
        None,
        true,
    )?;
    let second = run_qd4_routed_once(
        &mut lanes,
        Some(&mut cache),
        &runtime.expert_shard,
        LAYER,
        &ffn_input,
        &routes,
        &shared,
        &plans,
        None,
        true,
    )?;
    let first_contributions = first
        .contributions
        .as_ref()
        .ok_or("QD4 targeted first batch omitted F32 route contributions")?;
    let second_contributions = second
        .contributions
        .as_ref()
        .ok_or("QD4 targeted second batch omitted F32 route contributions")?;
    if first.cache_miss_count != v4_router::V4_TOPK
        || second.cache_miss_count != 0
        || first.output != second.output
        || first_contributions.len() != v4_router::V4_TOPK
        || second_contributions.len() != v4_router::V4_TOPK
        || !first_contributions
            .iter()
            .zip(second_contributions)
            .all(|(first, second)| qd4_gate_f32_bits_equal(first, second))
        || !state_before_routed.bits_equal(
            persistent
                .layer(LAYER)
                .ok_or("QD4 targeted L3 persistent state disappeared after routing")?,
        )
    {
        return Err(
            "QD4 targeted L3 cache batches changed exact routed or persistent state".into(),
        );
    }
    cache.finish_layer(LAYER)?;
    let telemetry = cache.telemetry;
    let expert_bytes = v4_expert::V4_ROUTED_EXPERT_BYTES;
    let expected_source_bytes = v4_router::V4_TOPK
        .checked_mul(expert_bytes)
        .ok_or("QD4 targeted source byte count overflow")?;
    let expected_d2d_bytes = 2_usize
        .checked_mul(v4_router::V4_TOPK)
        .and_then(|count| count.checked_mul(expert_bytes))
        .ok_or("QD4 targeted D2D byte count overflow")?;
    if telemetry.route_executions != 2 * v4_router::V4_TOPK
        || telemetry.expert_union_loads != v4_router::V4_TOPK
        || telemetry.source_loads != v4_router::V4_TOPK
        || telemetry.source_bytes != expected_source_bytes
        || telemetry.h2d_loads != v4_router::V4_TOPK
        || telemetry.h2d_bytes != expected_source_bytes
        || telemetry.hits != v4_router::V4_TOPK
        || telemetry.d2d_copies != 2 * v4_router::V4_TOPK
        || telemetry.d2d_bytes != expected_d2d_bytes
        || telemetry.max_occupancy != v4_router::V4_TOPK
        || telemetry.completed_layers != 1
        || telemetry.worker_errors != 0
        || telemetry.fallbacks != 0
        || cache.free.len() != V4_QD4_IMMUTABLE_EXPERT_CACHE_SLOTS
        || !cache.resident.is_empty()
    {
        return Err("QD4 targeted L3 immutable-cache physical accounting did not close".into());
    }
    writeln!(
        result.file_mut(),
        "{}",
        serde_json::json!({
            "event": "run",
            "schema": "colibri-v4-qd4-immutable-cache-targeted-l3-v1",
            "provenance": benchmark_provenance()?,
            "model_revision": v4_state::REVISION,
            "prompt": "official single-user chat hi",
            "prompt_ids": token_ids,
            "target": "one real L3 position-zero six-route batch twice: compulsory misses then immutable-cache hits",
            "route_ids": routes.ids,
            "route_weight_bits": routes.weights.iter().map(|weight| weight.to_bits()).collect::<Vec<_>>(),
            "exact": {
                "route_slot_f32_bitwise": true,
                "canonical_moe_bf16": true,
                "persistent_state_unchanged_across_routed_replay": true,
                "first_miss_count": first.cache_miss_count,
                "second_miss_count": second.cache_miss_count,
                "canonical_moe_bf16_digest": format!("{:016x}", fnv1a_u16(&first.output)),
                "route_slot_f32_digests": first_contributions.iter().map(|values| format!("{:016x}", fnv1a_f32(values))).collect::<Vec<_>>(),
                "persistent_state": qd4_gate_state_json(&state_before_routed),
            },
            "physical": {
                "lanes": V4_QD4_LANES,
                "slots": V4_QD4_IMMUTABLE_EXPERT_CACHE_SLOTS,
                "route_executions": telemetry.route_executions,
                "expert_union_loads": telemetry.expert_union_loads,
                "source_loads": telemetry.source_loads,
                "source_bytes": telemetry.source_bytes,
                "h2d_loads": telemetry.h2d_loads,
                "h2d_bytes": telemetry.h2d_bytes,
                "hits": telemetry.hits,
                "d2d_copies": telemetry.d2d_copies,
                "d2d_bytes": telemetry.d2d_bytes,
                "max_occupancy": telemetry.max_occupancy,
                "completed_layers": telemetry.completed_layers,
                "free_slots_after_finish": cache.free.len(),
                "resident_slots_after_finish": cache.resident.len(),
                "host_available_before_bytes": host_available,
                "device_available_before_bytes": device_available,
            },
        })
    )?;
    result.commit()?;
    println!(
        "{{\"status\":\"GO\",\"mode\":\"v4-qd4-immutable-cache-targeted-l3\",\"result_jsonl\":{}}}",
        serde_json::to_string(result.path())?
    );
    Ok(())
}

pub(crate) fn run_qd4_stationary_gate(
    context: &std::sync::Arc<CudaContext>,
    module: &kernels::LoadedModule,
    stream: &std::sync::Arc<CudaStream>,
) -> Result<(), Box<dyn Error>> {
    const PAIRS: usize = 4;
    const REQUIRED_SPEEDUP: f64 = 1.10;
    let immutable_cache_gate = optional_one_flag("COLIBRI_V4_QD4_IMMUTABLE_EXPERT_CACHE")?;
    let result_path = std::env::var_os("COLIBRI_V4_QD4_STATIONARY_GATE_RESULT_JSONL")
        .map(std::path::PathBuf::from)
        .ok_or("set COLIBRI_V4_QD4_STATIONARY_GATE_RESULT_JSONL")?;
    if std::env::var_os("COLIBRI_V4_FULL61_B1").is_some()
        || std::env::var_os("COLIBRI_V4_LAYER_STATIONARY_PREFILL").is_some()
        || std::env::var_os("COLIBRI_V4_ROUTED_QD4").is_some()
        || std::env::var_os("COLIBRI_V4_ROUTED_QD4_BENCH").is_some()
        || parse_gib_budget("COLIBRI_V4_DEVICE_FIXED_CACHE_GIB")? != 0
        || parse_gib_budget("COLIBRI_V4_HOST_FIXED_CACHE_GIB")? != 0
    {
        return Err(
            "QD4 stationary gate is isolated: clear full61, stationary, QD4, and fixed-cache modes"
                .into(),
        );
    }
    let required_host = V4_QD4_PINNED_HOST_BYTES
        + usize::from(immutable_cache_gate) * V4_QD4_IMMUTABLE_EXPERT_CACHE_HOST_BYTES;
    let required_device = V4_QD4_DEVICE_BYTES
        + usize::from(immutable_cache_gate) * V4_QD4_IMMUTABLE_EXPERT_CACHE_DEVICE_BYTES;
    if host_available_bytes()? < required_host || device_available_bytes()? < required_device {
        return Err("QD4 stationary gate lacks the explicit pinned/device lane reserve".into());
    }
    if let Some(parent) = result_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let model_dir = v4_model_dir()?;
    let token_ids = qd4_gate_prompt_ids(&model_dir)?;
    let inputs =
        prepare_qd4_stationary_gate_inputs(context, module, stream, &model_dir, &token_ids, 4)?;
    let telemetry_before = gpu_telemetry_snapshot()?;
    let mut output = std::fs::File::create(&result_path)?;
    writeln!(
        output,
        "{}",
        serde_json::json!({
            "event": "run",
            "schema": "colibri-v4-qd4-stationary-gate-v2",
            "provenance": benchmark_provenance()?,
            "model_revision": v4_state::REVISION,
            "prompt": "official single-user chat hi",
            "prompt_ids": token_ids,
            "layers": [3, 4],
            "positions": [0, 1, 2, 3, 4],
            "pairs_per_layer": PAIRS,
            "arm_order_schedule": if immutable_cache_gate { "ABBA/BAAB; A=four persistent QD4 lanes cache off, B=same four persistent QD4 lanes with request-scoped 32-slot immutable cache" } else { "ABBABAAB; A=current per-layer 32-slot union reuse, B=four persistent QD4 lanes" },
            "timing": "sum of five routed inclusive host-wall scopes; fixed runtimes and prefix preparation are outside timing",
            "fixed_prefetch": "not exercised or claimed",
            "reserve": {
                "qd4_pinned_host_bytes": V4_QD4_PINNED_HOST_BYTES,
                "qd4_device_bytes": V4_QD4_DEVICE_BYTES,
                "immutable_cache_enabled": immutable_cache_gate,
                "immutable_cache_pinned_host_bytes": usize::from(immutable_cache_gate) * V4_QD4_IMMUTABLE_EXPERT_CACHE_HOST_BYTES,
                "immutable_cache_device_bytes": usize::from(immutable_cache_gate) * V4_QD4_IMMUTABLE_EXPERT_CACHE_DEVICE_BYTES,
                "host_available_before_bytes": host_available_bytes()?,
                "device_available_before_bytes": device_available_bytes()?,
            },
            "gate": {
                "exact": "all five positions: routes/weight bits, attention/FFN/shared digests, six route-slot F32 contribution values bitwise, canonical expert-ID-order MoE BF16, post-HC hidden, and KV/compressor/indexer state",
                "timing": if immutable_cache_gate { "development-only; no speed threshold" } else { "speed gate retained for legacy QD4 comparison" },
                "ratio_of_medians_at_least": (!immutable_cache_gate).then_some(REQUIRED_SPEEDUP),
                "paired_positive_required": (!immutable_cache_gate).then_some(PAIRS),
                "p95_nonregression": !immutable_cache_gate,
                "immutable_cache_physical": immutable_cache_gate.then_some("route_exec=30/layer; source/H2D=union; occupancy<=32; D2D=30*expert_bytes; worker/fallback/capacity errors=0"),
                "zero_oom_or_worker_errors": true,
            },
            "gpu_before": telemetry_before,
        })
    )?;
    let mut all_go = true;
    for layer in [3_usize, 4] {
        let (runtime, _) = load_layer_runtime(&model_dir, layer, &token_ids, stream)?;
        let hash_ids = load_layer_hash_ids(&model_dir, layer, &token_ids)?;
        let layer_inputs = inputs
            .get(&layer)
            .ok_or("QD4 stationary prefix omitted target inputs")?;
        let mut control_samples = Vec::with_capacity(PAIRS);
        let mut qd4_samples = Vec::with_capacity(PAIRS);
        let mut positive_pairs = 0_usize;
        let mut exact_pairs = 0_usize;
        let mut worker_errors = 0_usize;
        for pair in 0..PAIRS {
            let control_first = qd4_stationary_control_first(pair);
            let (control, qd4) = if control_first {
                let control = run_qd4_stationary_gate_arm(
                    context,
                    module,
                    stream,
                    &runtime,
                    &token_ids,
                    &hash_ids,
                    layer_inputs,
                    immutable_cache_gate,
                    false,
                )?;
                let qd4 = run_qd4_stationary_gate_arm(
                    context,
                    module,
                    stream,
                    &runtime,
                    &token_ids,
                    &hash_ids,
                    layer_inputs,
                    true,
                    immutable_cache_gate,
                )?;
                (control, qd4)
            } else {
                let qd4 = run_qd4_stationary_gate_arm(
                    context,
                    module,
                    stream,
                    &runtime,
                    &token_ids,
                    &hash_ids,
                    layer_inputs,
                    true,
                    immutable_cache_gate,
                )?;
                let control = run_qd4_stationary_gate_arm(
                    context,
                    module,
                    stream,
                    &runtime,
                    &token_ids,
                    &hash_ids,
                    layer_inputs,
                    immutable_cache_gate,
                    false,
                )?;
                (control, qd4)
            };
            check_qd4_stationary_gate_pair(layer, &control, &qd4)?;
            if immutable_cache_gate {
                check_qd4_immutable_cache_gate_physical(layer, &qd4)?;
            }
            exact_pairs += 1;
            worker_errors = worker_errors
                .saturating_add(control.worker_errors)
                .saturating_add(qd4.worker_errors);
            positive_pairs += usize::from(control.routed_wall_ns > qd4.routed_wall_ns);
            control_samples.push(control.routed_wall_ns);
            qd4_samples.push(qd4.routed_wall_ns);
            writeln!(
                output,
                "{}",
                serde_json::json!({
                    "event": "pair",
                    "layer": layer,
                    "pair": pair,
                    "arm_order": if control_first { "AB" } else { "BA" },
                    "exact": true,
                    "control": qd4_stationary_gate_arm_json(&control),
                    "qd4": qd4_stationary_gate_arm_json(&qd4),
                    "routed_wall_speedup": control.routed_wall_ns as f64 / qd4.routed_wall_ns as f64,
                    "positive": control.routed_wall_ns > qd4.routed_wall_ns,
                })
            )?;
            for position in &qd4.positions {
                writeln!(output, "{}", qd4_gate_position_json(layer, pair, position))?;
            }
        }
        let control_p50 = median_u64(&control_samples);
        let qd4_p50 = median_u64(&qd4_samples);
        let (_, control_p95_ms) = median_p95_ms(&control_samples);
        let (_, qd4_p95_ms) = median_p95_ms(&qd4_samples);
        let ratio_of_medians = control_p50 as f64 / qd4_p50 as f64;
        let exact = exact_pairs == PAIRS;
        let speed_go = immutable_cache_gate || ratio_of_medians >= REQUIRED_SPEEDUP;
        let positive_go = immutable_cache_gate || positive_pairs == PAIRS;
        let p95_go = immutable_cache_gate || qd4_p95_ms <= control_p95_ms;
        let error_go = worker_errors == 0;
        let status = if exact && speed_go && positive_go && p95_go && error_go {
            "GO"
        } else {
            "KILL"
        };
        all_go &= status == "GO";
        writeln!(
            output,
            "{}",
            serde_json::json!({
                "event": "layer_summary",
                "layer": layer,
                "status": status,
                "exact_pairs": exact_pairs,
                "control_routed_wall_p50_ns": control_p50,
                "qd4_routed_wall_p50_ns": qd4_p50,
                "control_routed_wall_p95_ms": control_p95_ms,
                "qd4_routed_wall_p95_ms": qd4_p95_ms,
                "ratio_of_medians": ratio_of_medians,
                "positive_pairs": positive_pairs,
                "worker_errors": worker_errors,
                "speed_gate": speed_go,
                "positive_gate": positive_go,
                "p95_gate": p95_go,
                "error_gate": error_go,
                "required_ratio_of_medians": (!immutable_cache_gate).then_some(REQUIRED_SPEEDUP),
                "immutable_cache_gate": immutable_cache_gate,
            })
        )?;
    }
    let status = if all_go { "GO" } else { "KILL" };
    writeln!(
        output,
        "{}",
        serde_json::json!({
            "event": "summary",
            "status": status,
            "full61_promotion": false,
            "production_guard_change": false,
            "gpu_after": gpu_telemetry_snapshot()?,
        })
    )?;
    output.sync_all()?;
    println!(
        "{{\"status\":\"{status}\",\"mode\":\"v4-qd4-stationary-gate\",\"result_jsonl\":{}}}",
        serde_json::to_string(&result_path)?
    );
    Ok(())
}
