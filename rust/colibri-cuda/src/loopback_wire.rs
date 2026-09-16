use super::*;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct LoopbackWireBytes {
    pub(crate) initial_requests: usize,
    pub(crate) audit_responses: usize,
    pub(crate) certified_responses: usize,
    pub(crate) repair_requests: usize,
    pub(crate) repair_responses: usize,
}

impl LoopbackWireBytes {
    pub(crate) fn total(self) -> usize {
        self.initial_requests
            + self.audit_responses
            + self.certified_responses
            + self.repair_requests
            + self.repair_responses
    }

    pub(crate) fn returned(self) -> usize {
        self.audit_responses + self.certified_responses + self.repair_responses
    }
}

#[derive(Clone)]
pub(crate) struct LoopbackSample {
    pub(crate) e2e_ns: u64,
    pub(crate) worker_ns: u64,
    pub(crate) codec_tcp_ns: u64,
    pub(crate) repair_ns: u64,
    pub(crate) gpu_ns: u64,
    pub(crate) repair_lanes: usize,
    pub(crate) bytes: LoopbackWireBytes,
}

pub(crate) const NETEM_BLOCKS: usize = 4;
pub(crate) const NETEM_PAIRS_PER_BLOCK: usize = 25;

pub(crate) struct DualBenchConfig {
    pub(crate) scenario: String,
    pub(crate) seed: u64,
    pub(crate) audit_first: [bool; NETEM_BLOCKS],
}

impl DualBenchConfig {
    pub(crate) fn from_env() -> Result<Self, Box<dyn Error>> {
        let scenario =
            std::env::var("COLIBRI_NETEM_SCENARIO").unwrap_or_else(|_| "baseline".into());
        if !scenario
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
        {
            return Err("COLIBRI_NETEM_SCENARIO must be an ASCII identifier".into());
        }
        let seed = std::env::var("COLIBRI_NETEM_SEED")
            .unwrap_or_else(|_| "20260724".into())
            .parse::<u64>()?;
        let mut audit_first = [true, true, false, false];
        let mut state = seed
            ^ scenario.bytes().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
                (hash ^ u64::from(byte)).wrapping_mul(0x100_0000_01b3)
            });
        for upper in (1..audit_first.len()).rev() {
            state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut mixed = state;
            mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            mixed ^= mixed >> 31;
            audit_first.swap(upper, mixed as usize % (upper + 1));
        }
        Ok(Self {
            scenario,
            seed,
            audit_first,
        })
    }
}

pub(crate) struct OwnerCold {
    pub(crate) owner: u8,
    pub(crate) elapsed_ns: u64,
    pub(crate) weight_bytes: usize,
}

pub(crate) const RECOVERY_SAMPLES: u64 = 20;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RecoveryCase {
    AuditBaseline,
    CertifiedBaseline,
    AuditF1,
    CertifiedF1,
    CertifiedF2,
    WrongReplicaIdentity,
    SecondFailure,
}

impl RecoveryCase {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::AuditBaseline => "audit_baseline",
            Self::CertifiedBaseline => "certified_baseline",
            Self::AuditF1 => "audit_F1",
            Self::CertifiedF1 => "certified_F1",
            Self::CertifiedF2 => "certified_F2",
            Self::WrongReplicaIdentity => "negative_wrong_replica_identity",
            Self::SecondFailure => "negative_second_failure",
        }
    }

    pub(crate) fn mode(self) -> v4_federated::OwnerTaskMode {
        match self {
            Self::AuditBaseline
            | Self::AuditF1
            | Self::WrongReplicaIdentity
            | Self::SecondFailure => v4_federated::OwnerTaskMode::Audit,
            Self::CertifiedBaseline | Self::CertifiedF1 | Self::CertifiedF2 => {
                v4_federated::OwnerTaskMode::Certified
            }
        }
    }

    pub(crate) fn start_id(self) -> u64 {
        match self {
            Self::AuditBaseline => 10_000,
            Self::CertifiedBaseline => 10_020,
            Self::AuditF1 => 10_040,
            Self::CertifiedF1 => 10_060,
            Self::CertifiedF2 => 10_080,
            Self::WrongReplicaIdentity => 10_100,
            Self::SecondFailure => 10_101,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RecoveryEndpoint {
    PrimaryA,
    PrimaryB,
    ReplicaA,
}

impl RecoveryEndpoint {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::PrimaryA => "A",
            Self::PrimaryB => "B",
            Self::ReplicaA => "A'",
        }
    }

    pub(crate) fn owner(self) -> u8 {
        match self {
            Self::PrimaryB => 1,
            Self::PrimaryA | Self::ReplicaA => 0,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RecoveryFault {
    None,
    F1,
    F2,
    WrongIdentity,
}

pub(crate) struct RecoverySample {
    pub(crate) e2e_ns: u64,
    pub(crate) failover_ns: u64,
    pub(crate) failovers: usize,
    pub(crate) bytes: LoopbackWireBytes,
}

pub(crate) struct RecoveryCold {
    pub(crate) endpoint: RecoveryEndpoint,
    pub(crate) elapsed_ns: u64,
    pub(crate) weight_bytes: usize,
}

pub(crate) fn print_netem_result(
    scenario: &str,
    seed: u64,
    mode: &str,
    block_order: &str,
    samples: &[LoopbackSample],
    blocks: &[Vec<LoopbackSample>; NETEM_BLOCKS],
) {
    let e2e: Vec<_> = samples.iter().map(|sample| sample.e2e_ns).collect();
    let worker: Vec<_> = samples.iter().map(|sample| sample.worker_ns).collect();
    let residual: Vec<_> = samples.iter().map(|sample| sample.codec_tcp_ns).collect();
    let repair: Vec<_> = samples.iter().map(|sample| sample.repair_ns).collect();
    let cuda: Vec<_> = samples.iter().map(|sample| sample.gpu_ns).collect();
    let (e2e_median, e2e_p95) = median_p95_ms(&e2e);
    let (worker_median, worker_p95) = median_p95_ms(&worker);
    let (residual_median, residual_p95) = median_p95_ms(&residual);
    let (repair_median, repair_p95) = median_p95_ms(&repair);
    let (cuda_median, cuda_p95) = median_p95_ms(&cuda);
    let block_medians = blocks
        .iter()
        .map(|block| {
            let values: Vec<_> = block.iter().map(|sample| sample.e2e_ns).collect();
            format!("{:.6}", median_p95_ms(&values).0)
        })
        .collect::<Vec<_>>()
        .join("|");
    let bytes = samples[0].bytes;
    println!(
        "NETEM_RESULT,{scenario},{seed},{mode},{},{e2e_median:.6},{e2e_p95:.6},{worker_median:.6},{worker_p95:.6},{residual_median:.6},{residual_p95:.6},{repair_median:.6},{repair_p95:.6},{cuda_median:.6},{cuda_p95:.6},{},{},{},0,{block_order},{block_medians}",
        samples.len(),
        bytes.total(),
        bytes.returned(),
        samples[0].repair_lanes,
    );
}

pub(crate) fn recovery_case_for_id(request_id: u64) -> Option<RecoveryCase> {
    match request_id {
        10_000..=10_019 => Some(RecoveryCase::AuditBaseline),
        10_020..=10_039 => Some(RecoveryCase::CertifiedBaseline),
        10_040..=10_059 => Some(RecoveryCase::AuditF1),
        10_060..=10_079 => Some(RecoveryCase::CertifiedF1),
        10_080..=10_099 => Some(RecoveryCase::CertifiedF2),
        10_100 => Some(RecoveryCase::WrongReplicaIdentity),
        10_101 => Some(RecoveryCase::SecondFailure),
        _ => None,
    }
}

pub(crate) fn recovery_fault(endpoint: RecoveryEndpoint, case: RecoveryCase) -> RecoveryFault {
    match (endpoint, case) {
        (
            RecoveryEndpoint::PrimaryA,
            RecoveryCase::AuditF1
            | RecoveryCase::CertifiedF1
            | RecoveryCase::WrongReplicaIdentity
            | RecoveryCase::SecondFailure,
        ) => RecoveryFault::F1,
        (RecoveryEndpoint::PrimaryA, RecoveryCase::CertifiedF2) => RecoveryFault::F2,
        (RecoveryEndpoint::ReplicaA, RecoveryCase::WrongReplicaIdentity) => {
            RecoveryFault::WrongIdentity
        }
        (RecoveryEndpoint::ReplicaA, RecoveryCase::SecondFailure) => RecoveryFault::F1,
        _ => RecoveryFault::None,
    }
}

pub(crate) fn validate_v4_recovery_task(
    task: &v4_federated::OwnerTask,
    endpoint: RecoveryEndpoint,
    case: RecoveryCase,
    expert_ids: [u16; v4_federated::OWNER_EXPERTS],
    route_weights: [f32; v4_federated::OWNER_EXPERTS],
    expected_input: &[u16],
) -> Result<(), String> {
    let expected_identity =
        v4_federated::OwnerIdentity::live(endpoint.owner(), task.identity.request_id, expert_ids);
    if task.identity != expected_identity
        || task.mode != case.mode()
        || task
            .route_weights
            .iter()
            .zip(route_weights)
            .any(|(actual, expected)| actual.to_bits() != expected.to_bits())
        || v4_activation_hash(&task.hidden_bf16) != v4_activation_hash(expected_input)
        || task.hidden_bf16 != expected_input
    {
        return Err(format!(
            "recovery endpoint {} rejected task identity {}",
            endpoint.label(),
            task.identity.request_id
        ));
    }
    Ok(())
}

pub(crate) fn v4_activation_hash(values: &[u16]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for value in values {
        for byte in value.to_le_bytes() {
            hash = (hash ^ byte as u64).wrapping_mul(0x100_0000_01b3);
        }
    }
    hash
}

pub(crate) fn run_v4_recovery_case(
    addresses: [std::net::SocketAddr; 3],
    case: RecoveryCase,
    request_id: u64,
    owner_ids: [[u16; v4_federated::OWNER_EXPERTS]; 2],
    owner_weights: [[f32; v4_federated::OWNER_EXPERTS]; 2],
    input: &[u16],
    oracle_bf16: &[u16],
) -> Result<RecoverySample, String> {
    let start = std::time::Instant::now();
    let tasks = owner_tasks(case.mode(), request_id, owner_ids, owner_weights, input);
    let task_frames = [
        v4_federated::encode_owner_task(&tasks[0]).map_err(|error| error.to_string())?,
        v4_federated::encode_owner_task(&tasks[1]).map_err(|error| error.to_string())?,
    ];
    let mut bytes = LoopbackWireBytes::default();
    let mut primary_a = connect_recovery_endpoint(addresses[0])?;
    let mut primary_b = connect_recovery_endpoint(addresses[1])?;
    v4_federated::send_frame(&mut primary_a, &task_frames[0]).map_err(|error| error.to_string())?;
    v4_federated::send_frame(&mut primary_b, &task_frames[1]).map_err(|error| error.to_string())?;
    bytes.initial_requests += wire_frame_bytes(&task_frames[0]) + wire_frame_bytes(&task_frames[1]);

    match case.mode() {
        v4_federated::OwnerTaskMode::Audit => run_v4_recovery_audit(
            primary_a,
            primary_b,
            addresses[2],
            case,
            start,
            &tasks,
            &task_frames,
            oracle_bf16,
            bytes,
        ),
        v4_federated::OwnerTaskMode::Certified => run_v4_recovery_certified(
            primary_a,
            primary_b,
            addresses[2],
            case,
            start,
            &tasks,
            &task_frames,
            oracle_bf16,
            bytes,
        ),
    }
}

pub(crate) fn run_v4_recovery_audit(
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
    let primary_result = receive_recovery_audit(&mut primary_a, &tasks[0]);
    let (b_frame, b_response) = receive_recovery_audit(&mut primary_b, &tasks[1])?;
    bytes.audit_responses += wire_frame_bytes(&b_frame);
    let expected_f1 = matches!(
        case,
        RecoveryCase::AuditF1 | RecoveryCase::WrongReplicaIdentity | RecoveryCase::SecondFailure
    );
    let mut failover_started = None;
    let mut failovers = 0;
    let a_response = match primary_result {
        Ok((frame, response)) => {
            if expected_f1 {
                return Err(format!("{} did not inject F1", case.label()));
            }
            bytes.audit_responses += wire_frame_bytes(&frame);
            response
        }
        Err(_) => {
            failovers = 1;
            failover_started = Some(std::time::Instant::now());
            let mut replica = connect_recovery_endpoint(replica_address)?;
            v4_federated::send_frame(&mut replica, &task_frames[0])
                .map_err(|error| error.to_string())?;
            bytes.initial_requests += wire_frame_bytes(&task_frames[0]);
            let (frame, response) = receive_recovery_audit(&mut replica, &tasks[0])?;
            bytes.audit_responses += wire_frame_bytes(&frame);
            response
        }
    };
    let actual = audit_bits_from_owner_responses(&[a_response, b_response])?;
    if actual != oracle_bf16 {
        return Err(format!(
            "{} BF16 mismatch count {} before commit",
            case.label(),
            bf16_mismatch_count(&actual, oracle_bf16)
        ));
    }
    let expected_failovers = usize::from(expected_f1);
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

pub(crate) fn connect_recovery_endpoint(address: std::net::SocketAddr) -> Result<std::net::TcpStream, String> {
    let socket = std::net::TcpStream::connect(address).map_err(|error| error.to_string())?;
    let timeout = std::time::Duration::from_secs(30);
    socket
        .set_nodelay(true)
        .and_then(|_| socket.set_read_timeout(Some(timeout)))
        .and_then(|_| socket.set_write_timeout(Some(timeout)))
        .map_err(|error| error.to_string())?;
    Ok(socket)
}

pub(crate) fn receive_recovery_audit(
    socket: &mut std::net::TcpStream,
    task: &v4_federated::OwnerTask,
) -> Result<(Vec<u8>, v4_federated::OwnerAuditResponse), String> {
    let frame = v4_federated::receive_frame(socket).map_err(|error| error.to_string())?;
    let response =
        v4_federated::decode_owner_audit_response(&frame).map_err(|error| error.to_string())?;
    v4_federated::validate_owner_audit_response(task, &response)
        .map_err(|error| error.to_string())?;
    Ok((frame, response))
}

pub(crate) fn receive_recovery_certified(
    socket: &mut std::net::TcpStream,
    task: &v4_federated::OwnerTask,
) -> Result<(Vec<u8>, v4_federated::OwnerCertifiedResponse), String> {
    let frame = v4_federated::receive_frame(socket).map_err(|error| error.to_string())?;
    let response =
        v4_federated::decode_owner_certified_response(&frame).map_err(|error| error.to_string())?;
    v4_federated::validate_owner_certified_response(task, &response)
        .map_err(|error| error.to_string())?;
    Ok((frame, response))
}

pub(crate) fn receive_recovery_repair(
    socket: &mut std::net::TcpStream,
    task: &v4_federated::OwnerTask,
    request: &v4_federated::OwnerRepairRequest,
) -> Result<(Vec<u8>, v4_federated::OwnerRepairResponse), String> {
    let frame = v4_federated::receive_frame(socket).map_err(|error| error.to_string())?;
    let response =
        v4_federated::decode_owner_repair_response(&frame).map_err(|error| error.to_string())?;
    v4_federated::validate_owner_repair_response(task, request, &response)
        .map_err(|error| error.to_string())?;
    Ok((frame, response))
}

pub(crate) fn certify_recovery_pair(
    a: &v4_federated::OwnerCertifiedResponse,
    b: &v4_federated::OwnerCertifiedResponse,
) -> Result<v4_certified::Certification, String> {
    v4_certified::certify_owner_groups(
        &[
            v4_certified::OwnerReduction {
                grouped_f32: a.grouped_f32.clone(),
                bounds: a.bounds.clone(),
            },
            v4_certified::OwnerReduction {
                grouped_f32: b.grouped_f32.clone(),
                bounds: b.bounds.clone(),
            },
        ],
        v4_federated::FROZEN_BLOCK_SIZE,
    )
    .map_err(|error| error.to_string())
}

pub(crate) fn validate_v4_owner_task(
    task: &v4_federated::OwnerTask,
    owner: u8,
    request_id: u64,
    expert_ids: [u16; v4_federated::OWNER_EXPERTS],
    route_weights: [f32; v4_federated::OWNER_EXPERTS],
    expected_input: &[u16],
    expected_mode: v4_federated::OwnerTaskMode,
) -> Result<(), String> {
    let expected_identity = v4_federated::OwnerIdentity::live(owner, request_id, expert_ids);
    if task.identity != expected_identity
        || task.mode != expected_mode
        || task
            .route_weights
            .iter()
            .zip(route_weights)
            .any(|(actual, expected)| actual.to_bits() != expected.to_bits())
        || task.hidden_bf16 != expected_input
    {
        return Err(format!(
            "owner {owner} rejected duplicate, missing, or wrong task identity {request_id}"
        ));
    }
    Ok(())
}

pub(crate) fn execute_v4_owner_slots(
    slots: &mut [ExpertSlot<'_>],
    input: &[u16],
    route_weights: [f32; v4_federated::OWNER_EXPERTS],
) -> Result<(Vec<Vec<f32>>, u64), String> {
    let mut contributions = Vec::with_capacity(v4_federated::OWNER_EXPERTS);
    let mut gpu_ns = 0_u64;
    for (slot, weight) in slots.iter_mut().zip(route_weights) {
        let (values, elapsed) = slot
            .execute(input, weight)
            .map_err(|error| error.to_string())?;
        contributions.push(values);
        gpu_ns = gpu_ns.saturating_add(elapsed);
    }
    Ok((contributions, gpu_ns))
}

pub(crate) fn owner_tasks(
    mode: v4_federated::OwnerTaskMode,
    request_id: u64,
    owner_ids: [[u16; v4_federated::OWNER_EXPERTS]; 2],
    owner_weights: [[f32; v4_federated::OWNER_EXPERTS]; 2],
    input: &[u16],
) -> [v4_federated::OwnerTask; 2] {
    std::array::from_fn(|owner| v4_federated::OwnerTask {
        identity: v4_federated::OwnerIdentity::live(owner as u8, request_id, owner_ids[owner]),
        mode,
        route_weights: owner_weights[owner],
        hidden_bf16: input.to_vec(),
    })
}

pub(crate) fn run_v4_audit_client(
    sockets: &mut [std::net::TcpStream; 2],
    request_id: u64,
    owner_ids: [[u16; v4_federated::OWNER_EXPERTS]; 2],
    owner_weights: [[f32; v4_federated::OWNER_EXPERTS]; 2],
    input: &[u16],
    oracle_bf16: &[u16],
) -> Result<LoopbackSample, String> {
    let start = std::time::Instant::now();
    let tasks = owner_tasks(
        v4_federated::OwnerTaskMode::Audit,
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
    let response_frames = [
        v4_federated::receive_frame(&mut sockets[0]).map_err(|error| error.to_string())?,
        v4_federated::receive_frame(&mut sockets[1]).map_err(|error| error.to_string())?,
    ];
    let responses = [
        v4_federated::decode_owner_audit_response(&response_frames[0])
            .map_err(|error| error.to_string())?,
        v4_federated::decode_owner_audit_response(&response_frames[1])
            .map_err(|error| error.to_string())?,
    ];
    for owner in 0..2 {
        v4_federated::validate_owner_audit_response(&tasks[owner], &responses[owner])
            .map_err(|error| error.to_string())?;
    }
    // Commit only after both complete responses pass identity validation.
    let actual = audit_bits_from_owner_responses(&responses)?;
    if actual != oracle_bf16 {
        return Err(format!(
            "audit request {request_id} BF16 mismatch count {}",
            bf16_mismatch_count(&actual, oracle_bf16)
        ));
    }
    let e2e_ns = start.elapsed().as_nanos().try_into().unwrap_or(u64::MAX);
    let worker_ns = responses
        .iter()
        .map(|response| response.worker_ns)
        .max()
        .unwrap();
    Ok(LoopbackSample {
        e2e_ns,
        worker_ns,
        codec_tcp_ns: e2e_ns.saturating_sub(worker_ns),
        repair_ns: 0,
        gpu_ns: responses.iter().map(|response| response.gpu_ns).sum(),
        repair_lanes: 0,
        bytes: LoopbackWireBytes {
            initial_requests: frames.iter().map(wire_frame_bytes).sum(),
            audit_responses: response_frames.iter().map(wire_frame_bytes).sum(),
            certified_responses: 0,
            repair_requests: 0,
            repair_responses: 0,
        },
    })
}

pub(crate) fn audit_bits_from_owner_responses(
    responses: &[v4_federated::OwnerAuditResponse; 2],
) -> Result<Vec<u16>, String> {
    let mut routes: [Option<&[f32]>; v4_router::V4_TOPK] = [None; v4_router::V4_TOPK];
    for response in responses {
        for (local, values) in response
            .contributions
            .chunks_exact(v4_expert::V4_HIDDEN)
            .enumerate()
        {
            let position = response.identity.route_positions[local] as usize;
            if routes[position].replace(values).is_some() {
                return Err("duplicate audit route position".into());
            }
        }
    }
    if routes.iter().any(Option::is_none) {
        return Err("missing audit route position".into());
    }
    let mut output = Vec::with_capacity(v4_expert::V4_HIDDEN);
    for lane in 0..v4_expert::V4_HIDDEN {
        let mut sum = 0.0_f32;
        for values in &routes {
            sum += values.unwrap()[lane];
        }
        output.push(v4_router::f32_to_bf16_rne(sum));
    }
    Ok(output)
}

pub(crate) fn apply_owner_repairs(
    output: &mut [u16],
    mask: &[u8],
    responses: &[v4_federated::OwnerRepairResponse; 2],
) -> Result<(), String> {
    if output.len() != v4_expert::V4_HIDDEN {
        return Err("certified output length changed".into());
    }
    let repair_lanes = v4_certified::mask_count(mask);
    let mut routes: [Option<&[f32]>; v4_router::V4_TOPK] = [None; v4_router::V4_TOPK];
    for response in responses {
        for (local, values) in response.values.chunks_exact(repair_lanes).enumerate() {
            let position = response.identity.route_positions[local] as usize;
            if routes[position].replace(values).is_some() {
                return Err("duplicate repair route position".into());
            }
        }
    }
    if routes.iter().any(Option::is_none) {
        return Err("missing repair route position".into());
    }
    let mut repair_index = 0;
    for (lane, output) in output.iter_mut().enumerate() {
        if v4_certified::mask_contains(mask, lane) {
            let mut sum = 0.0_f32;
            for values in &routes {
                sum += values.unwrap()[repair_index];
            }
            *output = v4_router::f32_to_bf16_rne(sum);
            repair_index += 1;
        }
    }
    if repair_index != repair_lanes {
        return Err("repair mask iteration count changed".into());
    }
    Ok(())
}

pub(crate) fn canonical_audit_bits(contributions: &[Vec<f32>]) -> Vec<u16> {
    (0..v4_expert::V4_HIDDEN)
        .map(|lane| {
            let mut sum = 0.0_f32;
            for contribution in contributions {
                sum += contribution[lane];
            }
            v4_router::f32_to_bf16_rne(sum)
        })
        .collect()
}

pub(crate) fn bf16_mismatch_count(actual: &[u16], expected: &[u16]) -> usize {
    actual
        .iter()
        .zip(expected)
        .filter(|(actual, expected)| actual != expected)
        .count()
}

pub(crate) fn wire_frame_bytes(frame: &Vec<u8>) -> usize {
    frame.len() + size_of::<u32>()
}

pub(crate) fn print_loopback_timings(name: &str, samples: &[LoopbackSample], include_repair: bool) {
    let e2e: Vec<_> = samples.iter().map(|sample| sample.e2e_ns).collect();
    let worker: Vec<_> = samples.iter().map(|sample| sample.worker_ns).collect();
    let codec_tcp: Vec<_> = samples.iter().map(|sample| sample.codec_tcp_ns).collect();
    let gpu: Vec<_> = samples.iter().map(|sample| sample.gpu_ns).collect();
    let (e2e_median, e2e_p95) = median_p95_ms(&e2e);
    let (worker_median, worker_p95) = median_p95_ms(&worker);
    let (codec_median, codec_p95) = median_p95_ms(&codec_tcp);
    let (gpu_median, gpu_p95) = median_p95_ms(&gpu);
    if include_repair {
        let repair: Vec<_> = samples.iter().map(|sample| sample.repair_ns).collect();
        let (repair_median, repair_p95) = median_p95_ms(&repair);
        println!(
            "DUAL_TCP_{name} n={} E2E_median/p95={e2e_median:.3}/{e2e_p95:.3}ms worker_critical_median/p95={worker_median:.3}/{worker_p95:.3}ms TCP+codec+client_reduce_residual_median/p95={codec_median:.3}/{codec_p95:.3}ms repair_extra_RTT_median/p95={repair_median:.3}/{repair_p95:.3}ms CUDA_total_median/p95={gpu_median:.3}/{gpu_p95:.3}ms",
            samples.len(),
        );
    } else {
        println!(
            "DUAL_TCP_{name} n={} E2E_median/p95={e2e_median:.3}/{e2e_p95:.3}ms worker_critical_median/p95={worker_median:.3}/{worker_p95:.3}ms TCP+codec+client_reduce_residual_median/p95={codec_median:.3}/{codec_p95:.3}ms CUDA_total_median/p95={gpu_median:.3}/{gpu_p95:.3}ms",
            samples.len(),
        );
    }
}

pub(crate) fn print_recovery_case(case: RecoveryCase, samples: &[RecoverySample]) {
    let e2e: Vec<_> = samples.iter().map(|sample| sample.e2e_ns).collect();
    let failover: Vec<_> = samples.iter().map(|sample| sample.failover_ns).collect();
    let (e2e_median, e2e_p95) = median_p95_ms(&e2e);
    let (failover_median, failover_p95) = median_p95_ms(&failover);
    let bytes = samples[0].bytes;
    assert!(
        samples
            .iter()
            .all(|sample| sample.bytes == bytes && sample.failovers == samples[0].failovers)
    );
    println!(
        "RECOVERY_CASE case={} n={} E2E_median/p95={e2e_median:.3}/{e2e_p95:.3}ms failover_count={} failover_latency_median/p95={failover_median:.3}/{failover_p95:.3}ms initial_requests={} audit_responses={} certified_responses={} repair_requests={} repair_responses={} total_bidirectional={} return_side={} BF16_mismatch=0 partial_commit=0",
        case.label(),
        samples.len(),
        samples[0].failovers,
        bytes.initial_requests,
        bytes.audit_responses,
        bytes.certified_responses,
        bytes.repair_requests,
        bytes.repair_responses,
        bytes.total(),
        bytes.returned(),
    );
}

pub(crate) fn median_twice_usize(samples: &[usize]) -> usize {
    assert!(!samples.is_empty(), "median needs at least one sample");
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    if sorted.len() % 2 == 0 {
        sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]
    } else {
        sorted[sorted.len() / 2] * 2
    }
}

pub(crate) fn median_p95_ms(samples: &[u64]) -> (f64, f64) {
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let middle = sorted.len() / 2;
    let median = (sorted[middle - 1] as f64 + sorted[middle] as f64) / 2.0;
    let p95 = sorted[(sorted.len() * 95).div_ceil(100) - 1] as f64;
    (median / 1_000_000.0, p95 / 1_000_000.0)
}

pub(crate) fn median_u64(samples: &[u64]) -> u64 {
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let middle = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        sorted[middle - 1].saturating_add(sorted[middle]) / 2
    } else {
        sorted[middle]
    }
}

pub(crate) fn median_f64(samples: &[f64]) -> f64 {
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let middle = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        (sorted[middle - 1] + sorted[middle]) / 2.0
    } else {
        sorted[middle]
    }
}
