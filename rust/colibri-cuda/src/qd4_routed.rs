use super::*;

pub(crate) struct FlashExpertTrace {
    pub(crate) input_fp8: Vec<u8>,
    pub(crate) input_scale: Vec<u8>,
    pub(crate) w1: Vec<u16>,
    pub(crate) w3: Vec<u16>,
    pub(crate) swiglu: Vec<u16>,
    pub(crate) w2_input_fp8: Vec<u8>,
    pub(crate) w2_input_scale: Vec<u8>,
    pub(crate) output: Vec<u16>,
}

pub(crate) struct Qd4ImmutableCacheLease<'a> {
    pub(crate) expert: u16,
    pub(crate) slot: ExpertSlot<'a>,
    pub(crate) miss: bool,
}

#[derive(Clone, Copy, Default)]
pub(crate) struct Qd4ImmutableExpertCacheTelemetry {
    pub(crate) route_executions: usize,
    pub(crate) expert_union_loads: usize,
    pub(crate) source_loads: usize,
    pub(crate) source_bytes: usize,
    pub(crate) h2d_loads: usize,
    pub(crate) h2d_bytes: usize,
    pub(crate) hits: usize,
    pub(crate) d2d_copies: usize,
    pub(crate) d2d_bytes: usize,
    pub(crate) max_occupancy: usize,
    pub(crate) completed_layers: usize,
    pub(crate) worker_errors: usize,
    pub(crate) fallbacks: usize,
}

pub(crate) struct Qd4ImmutableExpertCache<'a> {
    pub(crate) free: Vec<ExpertSlot<'a>>,
    pub(crate) layer: Option<usize>,
    pub(crate) resident: BTreeMap<u16, ExpertSlot<'a>>,
    pub(crate) telemetry: Qd4ImmutableExpertCacheTelemetry,
}

#[derive(Clone)]
pub(crate) struct RoutedReplaySample {
    pub(crate) inclusive_ns: u64,
    pub(crate) setup_ns: u64,
    pub(crate) read_ns: u64,
    pub(crate) read_wall_ns: u64,
    pub(crate) h2d_ns: u64,
    pub(crate) kernel_ns: u64,
    pub(crate) lane_occupancy: f64,
    pub(crate) submit_ns: [u64; v4_router::V4_TOPK],
    pub(crate) ready_ns: [u64; v4_router::V4_TOPK],
    pub(crate) timeline: Option<RoutedQd4Timeline>,
}

#[derive(Clone)]
pub(crate) struct RoutedLaneTimeline {
    pub(crate) submit_ns: u64,
    pub(crate) read_start_ns: u64,
    pub(crate) read_end_ns: u64,
    pub(crate) h2d_enqueued_ns: u64,
    pub(crate) kernel_enqueued_ns: u64,
    pub(crate) d2h_enqueued_ns: u64,
    pub(crate) complete_ns: u64,
    pub(crate) h2d_event_ns: u64,
    pub(crate) kernel_event_ns: u64,
}

#[derive(Clone)]
pub(crate) struct RoutedQd4Timeline {
    pub(crate) lanes: [Option<RoutedLaneTimeline>; v4_router::V4_TOPK],
    pub(crate) canonical_reduction_start_ns: u64,
    pub(crate) canonical_reduction_end_ns: u64,
}

impl RoutedQd4Timeline {
    pub(crate) fn as_json(&self) -> serde_json::Value {
        let lanes = self
            .lanes
            .iter()
            .map(|lane| {
                lane.as_ref().map(|lane| {
                    serde_json::json!({
                        "submit_ns": lane.submit_ns,
                        "read": {"start_ns": lane.read_start_ns, "end_ns": lane.read_end_ns},
                        "h2d_enqueued_ns": lane.h2d_enqueued_ns,
                        "kernel_enqueued_ns": lane.kernel_enqueued_ns,
                        "d2h_enqueued_ns": lane.d2h_enqueued_ns,
                        "complete_ns": lane.complete_ns,
                        "h2d_event_ns": lane.h2d_event_ns,
                        "kernel_event_ns": lane.kernel_event_ns,
                    })
                })
            })
            .collect::<Vec<_>>();
        serde_json::json!({
            "lanes": lanes,
            "canonical_reduction": {
                "start_ns": self.canonical_reduction_start_ns,
                "end_ns": self.canonical_reduction_end_ns,
            },
        })
    }
}

pub(crate) struct RoutedLaneExecution {
    pub(crate) read_ns: u64,
    pub(crate) h2d_ns: u64,
    pub(crate) kernel_ns: u64,
    pub(crate) timeline: Option<RoutedLaneTimeline>,
}

pub(crate) struct Qd4RoutedRun<'a> {
    pub(crate) output: Vec<u16>,
    pub(crate) contributions: Option<Vec<Vec<f32>>>,
    pub(crate) sample: RoutedReplaySample,
    pub(crate) cache_miss_count: usize,
    pub(crate) cache_leases: Vec<Qd4ImmutableCacheLease<'a>>,
}

pub(crate) fn run_serial_routed_replay(
    context: &std::sync::Arc<CudaContext>,
    module: &kernels::LoadedModule,
    stream: &std::sync::Arc<CudaStream>,
    shard: &safetensors::ShardFile,
    layer: usize,
    input: &[u16],
    routes: &v4_router::RouterSelection,
    shared: &[f32],
) -> Result<(Vec<u16>, RoutedReplaySample), Box<dyn Error>> {
    let started = std::time::Instant::now();
    let mut contributions = Vec::with_capacity(v4_router::V4_TOPK);
    let mut setup_ns = 0_u64;
    let mut read_ns = 0_u64;
    let mut h2d_ns = 0_u64;
    let mut kernel_ns = 0_u64;
    for route_slot in 0..v4_router::V4_TOPK {
        let expert_id = u16::try_from(routes.ids[route_slot])?;
        let expert = v4_expert::RoutedExpert::load_layer(shard, layer, expert_id)?;
        read_ns = read_ns.saturating_add((expert.read_ms * 1_000_000.0) as u64);
        let setup_started = std::time::Instant::now();
        let mut slot = ExpertSlot::new(context, module, stream, &expert)?;
        setup_ns = setup_ns
            .saturating_add(u64::try_from(setup_started.elapsed().as_nanos()).unwrap_or(u64::MAX));
        h2d_ns = h2d_ns.saturating_add((slot.weight_h2d_ms * 1_000_000.0) as u64);
        let (output, expert_kernel_ns) = slot.execute(input, routes.weights[route_slot])?;
        kernel_ns = kernel_ns.saturating_add(expert_kernel_ns);
        contributions.push((expert_id, output));
    }
    let routed = contributions
        .iter()
        .map(|(expert, output)| (*expert, output.as_slice()))
        .collect::<Vec<_>>();
    let output = v4_expert::reduce_moe_canonical(&routed, shared)?;
    let inclusive_ns = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
    Ok((
        output,
        RoutedReplaySample {
            inclusive_ns,
            setup_ns,
            read_ns,
            read_wall_ns: read_ns,
            h2d_ns,
            kernel_ns,
            lane_occupancy: 0.25,
            submit_ns: [0; v4_router::V4_TOPK],
            ready_ns: [0; v4_router::V4_TOPK],
            timeline: None,
        },
    ))
}

pub(crate) struct RoutedLaneJob<J> {
    pub(crate) route_slot: usize,
    pub(crate) iteration_started: std::time::Instant,
    pub(crate) profile_submit_ns: Option<u64>,
    pub(crate) payload: J,
}

pub(crate) struct RoutedLaneDone<J> {
    pub(crate) route_slot: usize,
    pub(crate) read_ns: u64,
    pub(crate) read_ready_ns: u64,
    pub(crate) h2d_ns: u64,
    pub(crate) kernel_ns: u64,
    pub(crate) ready_ns: u64,
    pub(crate) job_ns: u64,
    pub(crate) timeline: Option<RoutedLaneTimeline>,
    pub(crate) cache_lease: J,
}

pub(crate) struct SupervisedLaneDone<T> {
    pub(crate) lane: usize,
    pub(crate) value: T,
}

pub(crate) struct SupervisedQd4Batch<T> {
    pub(crate) iteration_started: std::time::Instant,
    pub(crate) submit_ns: [u64; v4_router::V4_TOPK],
    pub(crate) completions: Vec<T>,
}

pub(crate) fn routed_replay_json(sample: &RoutedReplaySample) -> serde_json::Value {
    serde_json::json!({
        "inclusive_ns": sample.inclusive_ns,
        "setup_ns": sample.setup_ns,
        "read_ns": sample.read_ns,
        "read_wall_ns": sample.read_wall_ns,
        "h2d_ns": sample.h2d_ns,
        "kernel_ns": sample.kernel_ns,
        "lane_occupancy": sample.lane_occupancy,
        "submit_ns": sample.submit_ns,
        "ready_ns": sample.ready_ns,
    })
}

pub(crate) fn routed_metric(
    samples: &[RoutedReplaySample],
    get: impl Fn(&RoutedReplaySample) -> u64,
) -> (f64, f64) {
    let values = samples.iter().map(get).collect::<Vec<_>>();
    median_p95_ms(&values)
}
