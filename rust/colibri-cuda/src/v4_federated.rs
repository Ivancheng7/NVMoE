//! Minimal fixed wire contracts for the live V4-Pro loopback smokes.

use std::io::{Read, Write};
use std::net::TcpStream;

use crate::v4_expert::{V4_HIDDEN, V4_MAX_EXPERT_ROWS};
use crate::v4_router::V4_ROUTED_EXPERTS;

pub const PINNED_REVISION: [u8; 40] = *b"b5968e9190ef611bbf34a7229255be88a0e937c1";
const MAGIC: &[u8; 4] = b"CV4F";
const VERSION: u16 = 4;
const REQUEST: u8 = 1;
const RESPONSE: u8 = 2;
const OWNER_TASK: u8 = 3;
const OWNER_AUDIT_RESPONSE: u8 = 4;
const OWNER_CERTIFIED_RESPONSE: u8 = 5;
const OWNER_REPAIR_REQUEST: u8 = 6;
const OWNER_REPAIR_RESPONSE: u8 = 7;
const HEADER_BYTES: usize = 76;
const OWNER_IDENTITY_BYTES: usize = HEADER_BYTES + 10;
const MAX_FRAME_BYTES: usize = HEADER_BYTES + 64 + 6 * V4_HIDDEN * 4;
pub const OWNER_EXPERTS: usize = 3;
pub const FROZEN_BLOCK_SIZE: usize = 256;
pub const FROZEN_BLOCKS: usize = V4_HIDDEN / FROZEN_BLOCK_SIZE;
pub const REPAIR_MASK_BYTES: usize = V4_HIDDEN.div_ceil(8);

#[derive(Clone, Debug, PartialEq)]
pub struct ExecuteRequest {
    pub revision: [u8; 40],
    pub session_epoch: u64,
    pub request_id: u64,
    pub deadline_ns: u64,
    pub layer: u16,
    pub expert: u16,
    pub rows: u8,
    pub route_weight: f32,
    pub hidden_bf16: Vec<u16>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExecuteResponse {
    pub revision: [u8; 40],
    pub session_epoch: u64,
    pub request_id: u64,
    pub deadline_ns: u64,
    pub layer: u16,
    pub expert: u16,
    pub rows: u8,
    pub worker_ns: u64,
    pub gpu_ns: u64,
    pub output_f32: Vec<f32>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OwnerTaskMode {
    Audit,
    Certified,
}

impl OwnerTaskMode {
    fn wire(self) -> u8 {
        match self {
            Self::Audit => 0,
            Self::Certified => 1,
        }
    }

    fn from_wire(value: u8) -> Result<Self, ProtocolError> {
        match value {
            0 => Ok(Self::Audit),
            1 => Ok(Self::Certified),
            _ => Err(ProtocolError::InvalidMode(value)),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OwnerIdentity {
    pub revision: [u8; 40],
    pub session_epoch: u64,
    pub request_id: u64,
    pub deadline_ns: u64,
    pub layer: u16,
    pub rows: u8,
    pub owner: u8,
    pub route_positions: [u8; OWNER_EXPERTS],
    pub expert_ids: [u16; OWNER_EXPERTS],
}

impl OwnerIdentity {
    pub fn live(owner: u8, request_id: u64, expert_ids: [u16; OWNER_EXPERTS]) -> Self {
        let route_positions = match owner {
            0 => [0, 1, 2],
            1 => [3, 4, 5],
            _ => [u8::MAX; OWNER_EXPERTS],
        };
        Self {
            revision: PINNED_REVISION,
            session_epoch: 2,
            request_id,
            deadline_ns: 0,
            layer: 0,
            rows: 1,
            owner,
            route_positions,
            expert_ids,
        }
    }

    fn validate(&self) -> Result<(), ProtocolError> {
        validate_header(&self.revision, self.layer, self.expert_ids[0], self.rows)?;
        let expected_positions = match self.owner {
            0 => [0, 1, 2],
            1 => [3, 4, 5],
            owner => return Err(ProtocolError::InvalidOwner(owner)),
        };
        if self.route_positions != expected_positions {
            return Err(ProtocolError::InvalidRoutePosition);
        }
        for (index, &expert) in self.expert_ids.iter().enumerate() {
            if expert as usize >= V4_ROUTED_EXPERTS {
                return Err(ProtocolError::InvalidExpert(expert));
            }
            if self.expert_ids[..index].contains(&expert) {
                return Err(ProtocolError::IdentityMismatch);
            }
        }
        if self.layer != 0 || self.rows != 1 {
            return Err(ProtocolError::IdentityMismatch);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct OwnerTask {
    pub identity: OwnerIdentity,
    pub mode: OwnerTaskMode,
    pub route_weights: [f32; OWNER_EXPERTS],
    pub hidden_bf16: Vec<u16>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct OwnerAuditResponse {
    pub identity: OwnerIdentity,
    pub worker_ns: u64,
    pub gpu_ns: u64,
    pub contributions: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct OwnerCertifiedResponse {
    pub identity: OwnerIdentity,
    pub worker_ns: u64,
    pub gpu_ns: u64,
    pub grouped_f32: Vec<f32>,
    pub bounds: Vec<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OwnerRepairRequest {
    pub identity: OwnerIdentity,
    pub mask: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct OwnerRepairResponse {
    pub identity: OwnerIdentity,
    pub worker_ns: u64,
    pub lane_count: u16,
    pub values: Vec<f32>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ProtocolError {
    Io(std::io::ErrorKind),
    FrameTooLarge,
    Truncated,
    BadMagic,
    WrongVersion(u16),
    WrongKind,
    WrongRevision,
    InvalidLayer(u16),
    InvalidExpert(u16),
    InvalidRows(u8),
    InvalidLength,
    InvalidFloat,
    InvalidMode(u8),
    InvalidOwner(u8),
    InvalidRoutePosition,
    InvalidStatus(u8),
    IdentityMismatch,
}

impl std::fmt::Display for ProtocolError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for ProtocolError {}

impl From<std::io::Error> for ProtocolError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error.kind())
    }
}

impl ExecuteRequest {
    pub fn live(input: Vec<u16>, route_weight: f32) -> Self {
        Self {
            revision: PINNED_REVISION,
            session_epoch: 1,
            request_id: 1,
            deadline_ns: 0,
            layer: 0,
            expert: 0,
            rows: 1,
            route_weight,
            hidden_bf16: input,
        }
    }

    fn validate(&self) -> Result<(), ProtocolError> {
        validate_header(&self.revision, self.layer, self.expert, self.rows)?;
        if self.hidden_bf16.len() != self.rows as usize * V4_HIDDEN {
            return Err(ProtocolError::InvalidLength);
        }
        if !self.route_weight.is_finite() {
            return Err(ProtocolError::InvalidFloat);
        }
        Ok(())
    }
}

impl ExecuteResponse {
    pub fn for_request(
        request: &ExecuteRequest,
        worker_ns: u64,
        gpu_ns: u64,
        output_f32: Vec<f32>,
    ) -> Self {
        Self {
            revision: request.revision,
            session_epoch: request.session_epoch,
            request_id: request.request_id,
            deadline_ns: request.deadline_ns,
            layer: request.layer,
            expert: request.expert,
            rows: request.rows,
            worker_ns,
            gpu_ns,
            output_f32,
        }
    }

    fn validate(&self) -> Result<(), ProtocolError> {
        validate_header(&self.revision, self.layer, self.expert, self.rows)?;
        if self.output_f32.len() != self.rows as usize * V4_HIDDEN {
            return Err(ProtocolError::InvalidLength);
        }
        if self.output_f32.iter().any(|value| !value.is_finite()) {
            return Err(ProtocolError::InvalidFloat);
        }
        Ok(())
    }
}

pub fn encode_request(request: &ExecuteRequest) -> Result<Vec<u8>, ProtocolError> {
    request.validate()?;
    let mut frame = Vec::with_capacity(HEADER_BYTES + 4 + request.hidden_bf16.len() * 2);
    encode_header(
        &mut frame,
        REQUEST,
        &request.revision,
        request.session_epoch,
        request.request_id,
        request.deadline_ns,
        request.layer,
        request.expert,
        request.rows,
    );
    put_u32(&mut frame, request.route_weight.to_bits());
    for value in &request.hidden_bf16 {
        put_u16(&mut frame, *value);
    }
    Ok(frame)
}

pub fn decode_request(frame: &[u8]) -> Result<ExecuteRequest, ProtocolError> {
    let (header, mut reader) = decode_header(frame, REQUEST)?;
    if frame.len() != HEADER_BYTES + 4 + header.rows as usize * V4_HIDDEN * 2 {
        return Err(ProtocolError::InvalidLength);
    }
    let route_weight = f32::from_bits(reader.u32()?);
    let mut hidden_bf16 = Vec::with_capacity(header.rows as usize * V4_HIDDEN);
    for _ in 0..header.rows as usize * V4_HIDDEN {
        hidden_bf16.push(reader.u16()?);
    }
    let request = ExecuteRequest {
        revision: header.revision,
        session_epoch: header.session_epoch,
        request_id: header.request_id,
        deadline_ns: header.deadline_ns,
        layer: header.layer,
        expert: header.expert,
        rows: header.rows,
        route_weight,
        hidden_bf16,
    };
    request.validate()?;
    Ok(request)
}

pub fn encode_response(response: &ExecuteResponse) -> Result<Vec<u8>, ProtocolError> {
    response.validate()?;
    let mut frame = Vec::with_capacity(HEADER_BYTES + 17 + response.output_f32.len() * 4);
    encode_header(
        &mut frame,
        RESPONSE,
        &response.revision,
        response.session_epoch,
        response.request_id,
        response.deadline_ns,
        response.layer,
        response.expert,
        response.rows,
    );
    frame.push(0);
    put_u64(&mut frame, response.worker_ns);
    put_u64(&mut frame, response.gpu_ns);
    for value in &response.output_f32 {
        put_u32(&mut frame, value.to_bits());
    }
    Ok(frame)
}

pub fn decode_response(frame: &[u8]) -> Result<ExecuteResponse, ProtocolError> {
    let (header, mut reader) = decode_header(frame, RESPONSE)?;
    if frame.len() != HEADER_BYTES + 17 + header.rows as usize * V4_HIDDEN * 4 {
        return Err(ProtocolError::InvalidLength);
    }
    let status = reader.u8()?;
    if status != 0 {
        return Err(ProtocolError::InvalidStatus(status));
    }
    let worker_ns = reader.u64()?;
    let gpu_ns = reader.u64()?;
    let mut output_f32 = Vec::with_capacity(header.rows as usize * V4_HIDDEN);
    for _ in 0..header.rows as usize * V4_HIDDEN {
        output_f32.push(f32::from_bits(reader.u32()?));
    }
    let response = ExecuteResponse {
        revision: header.revision,
        session_epoch: header.session_epoch,
        request_id: header.request_id,
        deadline_ns: header.deadline_ns,
        layer: header.layer,
        expert: header.expert,
        rows: header.rows,
        worker_ns,
        gpu_ns,
        output_f32,
    };
    response.validate()?;
    Ok(response)
}

pub fn validate_response(
    request: &ExecuteRequest,
    response: &ExecuteResponse,
) -> Result<(), ProtocolError> {
    response.validate()?;
    if response.revision != request.revision
        || response.session_epoch != request.session_epoch
        || response.request_id != request.request_id
        || response.deadline_ns != request.deadline_ns
        || response.layer != request.layer
        || response.expert != request.expert
        || response.rows != request.rows
    {
        return Err(ProtocolError::IdentityMismatch);
    }
    Ok(())
}

pub fn encode_owner_task(task: &OwnerTask) -> Result<Vec<u8>, ProtocolError> {
    validate_owner_task(task)?;
    let mut frame = Vec::with_capacity(OWNER_IDENTITY_BYTES + 13 + task.hidden_bf16.len() * 2);
    encode_owner_identity(&mut frame, OWNER_TASK, &task.identity)?;
    frame.push(task.mode.wire());
    for weight in task.route_weights {
        put_u32(&mut frame, weight.to_bits());
    }
    for value in &task.hidden_bf16 {
        put_u16(&mut frame, *value);
    }
    Ok(frame)
}

pub fn decode_owner_task(frame: &[u8]) -> Result<OwnerTask, ProtocolError> {
    let (identity, mut reader) = decode_owner_identity(frame, OWNER_TASK)?;
    if frame.len() != OWNER_IDENTITY_BYTES + 13 + identity.rows as usize * V4_HIDDEN * 2 {
        return Err(ProtocolError::InvalidLength);
    }
    let mode = OwnerTaskMode::from_wire(reader.u8()?)?;
    let mut route_weights = [0.0; OWNER_EXPERTS];
    for weight in &mut route_weights {
        *weight = f32::from_bits(reader.u32()?);
    }
    let mut hidden_bf16 = Vec::with_capacity(identity.rows as usize * V4_HIDDEN);
    for _ in 0..identity.rows as usize * V4_HIDDEN {
        hidden_bf16.push(reader.u16()?);
    }
    let task = OwnerTask {
        identity,
        mode,
        route_weights,
        hidden_bf16,
    };
    validate_owner_task(&task)?;
    Ok(task)
}

pub fn encode_owner_audit_response(
    response: &OwnerAuditResponse,
) -> Result<Vec<u8>, ProtocolError> {
    validate_owner_audit_payload(response)?;
    let mut frame =
        Vec::with_capacity(OWNER_IDENTITY_BYTES + 16 + response.contributions.len() * 4);
    encode_owner_identity(&mut frame, OWNER_AUDIT_RESPONSE, &response.identity)?;
    put_u64(&mut frame, response.worker_ns);
    put_u64(&mut frame, response.gpu_ns);
    put_f32s(&mut frame, &response.contributions);
    Ok(frame)
}

pub fn decode_owner_audit_response(frame: &[u8]) -> Result<OwnerAuditResponse, ProtocolError> {
    let (identity, mut reader) = decode_owner_identity(frame, OWNER_AUDIT_RESPONSE)?;
    if frame.len() != OWNER_IDENTITY_BYTES + 16 + OWNER_EXPERTS * V4_HIDDEN * 4 {
        return Err(ProtocolError::InvalidLength);
    }
    let worker_ns = reader.u64()?;
    let gpu_ns = reader.u64()?;
    let contributions = read_f32s(&mut reader, OWNER_EXPERTS * V4_HIDDEN)?;
    let response = OwnerAuditResponse {
        identity,
        worker_ns,
        gpu_ns,
        contributions,
    };
    validate_owner_audit_payload(&response)?;
    Ok(response)
}

pub fn validate_owner_audit_response(
    task: &OwnerTask,
    response: &OwnerAuditResponse,
) -> Result<(), ProtocolError> {
    validate_owner_task(task)?;
    validate_owner_audit_payload(response)?;
    if task.mode != OwnerTaskMode::Audit || response.identity != task.identity {
        return Err(ProtocolError::IdentityMismatch);
    }
    Ok(())
}

pub fn encode_owner_certified_response(
    response: &OwnerCertifiedResponse,
) -> Result<Vec<u8>, ProtocolError> {
    validate_owner_certified_payload(response)?;
    let mut frame = Vec::with_capacity(
        OWNER_IDENTITY_BYTES + 16 + response.grouped_f32.len() * 4 + response.bounds.len(),
    );
    encode_owner_identity(&mut frame, OWNER_CERTIFIED_RESPONSE, &response.identity)?;
    put_u64(&mut frame, response.worker_ns);
    put_u64(&mut frame, response.gpu_ns);
    put_f32s(&mut frame, &response.grouped_f32);
    frame.extend_from_slice(&response.bounds);
    Ok(frame)
}

pub fn decode_owner_certified_response(
    frame: &[u8],
) -> Result<OwnerCertifiedResponse, ProtocolError> {
    let (identity, mut reader) = decode_owner_identity(frame, OWNER_CERTIFIED_RESPONSE)?;
    if frame.len() != OWNER_IDENTITY_BYTES + 16 + V4_HIDDEN * 4 + FROZEN_BLOCKS {
        return Err(ProtocolError::InvalidLength);
    }
    let worker_ns = reader.u64()?;
    let gpu_ns = reader.u64()?;
    let grouped_f32 = read_f32s(&mut reader, V4_HIDDEN)?;
    let bounds = reader.take(FROZEN_BLOCKS)?.to_vec();
    let response = OwnerCertifiedResponse {
        identity,
        worker_ns,
        gpu_ns,
        grouped_f32,
        bounds,
    };
    validate_owner_certified_payload(&response)?;
    Ok(response)
}

pub fn validate_owner_certified_response(
    task: &OwnerTask,
    response: &OwnerCertifiedResponse,
) -> Result<(), ProtocolError> {
    validate_owner_task(task)?;
    validate_owner_certified_payload(response)?;
    if task.mode != OwnerTaskMode::Certified || response.identity != task.identity {
        return Err(ProtocolError::IdentityMismatch);
    }
    Ok(())
}

pub fn encode_owner_repair_request(request: &OwnerRepairRequest) -> Result<Vec<u8>, ProtocolError> {
    validate_owner_repair_request_payload(request)?;
    let mut frame = Vec::with_capacity(OWNER_IDENTITY_BYTES + request.mask.len());
    encode_owner_identity(&mut frame, OWNER_REPAIR_REQUEST, &request.identity)?;
    frame.extend_from_slice(&request.mask);
    Ok(frame)
}

pub fn decode_owner_repair_request(frame: &[u8]) -> Result<OwnerRepairRequest, ProtocolError> {
    let (identity, mut reader) = decode_owner_identity(frame, OWNER_REPAIR_REQUEST)?;
    if frame.len() != OWNER_IDENTITY_BYTES + REPAIR_MASK_BYTES {
        return Err(ProtocolError::InvalidLength);
    }
    let request = OwnerRepairRequest {
        identity,
        mask: reader.take(REPAIR_MASK_BYTES)?.to_vec(),
    };
    validate_owner_repair_request_payload(&request)?;
    Ok(request)
}

pub fn validate_owner_repair_request(
    task: &OwnerTask,
    request: &OwnerRepairRequest,
) -> Result<(), ProtocolError> {
    validate_owner_task(task)?;
    validate_owner_repair_request_payload(request)?;
    if task.mode != OwnerTaskMode::Certified || request.identity != task.identity {
        return Err(ProtocolError::IdentityMismatch);
    }
    Ok(())
}

pub fn encode_owner_repair_response(
    response: &OwnerRepairResponse,
) -> Result<Vec<u8>, ProtocolError> {
    validate_owner_repair_payload(response)?;
    let mut frame = Vec::with_capacity(OWNER_IDENTITY_BYTES + 10 + response.values.len() * 4);
    encode_owner_identity(&mut frame, OWNER_REPAIR_RESPONSE, &response.identity)?;
    put_u64(&mut frame, response.worker_ns);
    put_u16(&mut frame, response.lane_count);
    put_f32s(&mut frame, &response.values);
    Ok(frame)
}

pub fn decode_owner_repair_response(frame: &[u8]) -> Result<OwnerRepairResponse, ProtocolError> {
    let (identity, mut reader) = decode_owner_identity(frame, OWNER_REPAIR_RESPONSE)?;
    let expected_without_values = OWNER_IDENTITY_BYTES + 10;
    if frame.len() < expected_without_values
        || !(frame.len() - expected_without_values).is_multiple_of(4)
    {
        return Err(ProtocolError::InvalidLength);
    }
    let worker_ns = reader.u64()?;
    let lane_count = reader.u16()?;
    let values = read_f32s(&mut reader, (frame.len() - expected_without_values) / 4)?;
    let response = OwnerRepairResponse {
        identity,
        worker_ns,
        lane_count,
        values,
    };
    validate_owner_repair_payload(&response)?;
    Ok(response)
}

pub fn validate_owner_repair_response(
    task: &OwnerTask,
    request: &OwnerRepairRequest,
    response: &OwnerRepairResponse,
) -> Result<(), ProtocolError> {
    validate_owner_repair_request(task, request)?;
    validate_owner_repair_payload(response)?;
    if response.identity != task.identity
        || response.lane_count as usize != mask_count(&request.mask)
    {
        return Err(ProtocolError::IdentityMismatch);
    }
    Ok(())
}

fn validate_owner_task(task: &OwnerTask) -> Result<(), ProtocolError> {
    task.identity.validate()?;
    if task.hidden_bf16.len() != V4_HIDDEN {
        return Err(ProtocolError::InvalidLength);
    }
    if task.route_weights.iter().any(|weight| !weight.is_finite()) {
        return Err(ProtocolError::InvalidFloat);
    }
    Ok(())
}

fn validate_owner_audit_payload(response: &OwnerAuditResponse) -> Result<(), ProtocolError> {
    response.identity.validate()?;
    validate_f32_payload(&response.contributions, OWNER_EXPERTS * V4_HIDDEN)
}

fn validate_owner_certified_payload(
    response: &OwnerCertifiedResponse,
) -> Result<(), ProtocolError> {
    response.identity.validate()?;
    validate_f32_payload(&response.grouped_f32, V4_HIDDEN)?;
    if response.bounds.len() != FROZEN_BLOCKS {
        return Err(ProtocolError::InvalidLength);
    }
    Ok(())
}

fn validate_owner_repair_request_payload(
    request: &OwnerRepairRequest,
) -> Result<(), ProtocolError> {
    request.identity.validate()?;
    if request.mask.len() != REPAIR_MASK_BYTES {
        return Err(ProtocolError::InvalidLength);
    }
    Ok(())
}

fn validate_owner_repair_payload(response: &OwnerRepairResponse) -> Result<(), ProtocolError> {
    response.identity.validate()?;
    if response.lane_count as usize > V4_HIDDEN {
        return Err(ProtocolError::InvalidLength);
    }
    validate_f32_payload(
        &response.values,
        OWNER_EXPERTS * response.lane_count as usize,
    )
}

fn validate_f32_payload(values: &[f32], expected: usize) -> Result<(), ProtocolError> {
    if values.len() != expected {
        return Err(ProtocolError::InvalidLength);
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(ProtocolError::InvalidFloat);
    }
    Ok(())
}

fn encode_owner_identity(
    frame: &mut Vec<u8>,
    kind: u8,
    identity: &OwnerIdentity,
) -> Result<(), ProtocolError> {
    identity.validate()?;
    encode_header(
        frame,
        kind,
        &identity.revision,
        identity.session_epoch,
        identity.request_id,
        identity.deadline_ns,
        identity.layer,
        identity.expert_ids[0],
        identity.rows,
    );
    frame.push(identity.owner);
    for index in 0..OWNER_EXPERTS {
        frame.push(identity.route_positions[index]);
        put_u16(frame, identity.expert_ids[index]);
    }
    Ok(())
}

fn decode_owner_identity<'a>(
    frame: &'a [u8],
    kind: u8,
) -> Result<(OwnerIdentity, Reader<'a>), ProtocolError> {
    let (header, mut reader) = decode_header(frame, kind)?;
    let owner = reader.u8()?;
    let mut route_positions = [0; OWNER_EXPERTS];
    let mut expert_ids = [0; OWNER_EXPERTS];
    for index in 0..OWNER_EXPERTS {
        route_positions[index] = reader.u8()?;
        expert_ids[index] = reader.u16()?;
    }
    let identity = OwnerIdentity {
        revision: header.revision,
        session_epoch: header.session_epoch,
        request_id: header.request_id,
        deadline_ns: header.deadline_ns,
        layer: header.layer,
        rows: header.rows,
        owner,
        route_positions,
        expert_ids,
    };
    identity.validate()?;
    if header.expert != identity.expert_ids[0] {
        return Err(ProtocolError::IdentityMismatch);
    }
    Ok((identity, reader))
}

fn put_f32s(frame: &mut Vec<u8>, values: &[f32]) {
    for value in values {
        put_u32(frame, value.to_bits());
    }
}

fn read_f32s(reader: &mut Reader<'_>, count: usize) -> Result<Vec<f32>, ProtocolError> {
    let mut values = Vec::with_capacity(count);
    for _ in 0..count {
        values.push(f32::from_bits(reader.u32()?));
    }
    Ok(values)
}

fn mask_count(mask: &[u8]) -> usize {
    mask.iter().map(|byte| byte.count_ones() as usize).sum()
}

pub fn send_frame(stream: &mut TcpStream, frame: &[u8]) -> Result<(), ProtocolError> {
    if frame.len() > MAX_FRAME_BYTES {
        return Err(ProtocolError::FrameTooLarge);
    }
    stream.write_all(&(frame.len() as u32).to_le_bytes())?;
    stream.write_all(frame)?;
    Ok(())
}

pub fn receive_frame(stream: &mut TcpStream) -> Result<Vec<u8>, ProtocolError> {
    let mut length = [0; 4];
    stream.read_exact(&mut length)?;
    let length = u32::from_le_bytes(length) as usize;
    if length > MAX_FRAME_BYTES {
        return Err(ProtocolError::FrameTooLarge);
    }
    if length < HEADER_BYTES {
        return Err(ProtocolError::Truncated);
    }
    let mut frame = vec![0; length];
    stream.read_exact(&mut frame)?;
    Ok(frame)
}

fn validate_header(
    revision: &[u8; 40],
    layer: u16,
    expert: u16,
    rows: u8,
) -> Result<(), ProtocolError> {
    if revision != &PINNED_REVISION {
        return Err(ProtocolError::WrongRevision);
    }
    if layer >= 61 {
        return Err(ProtocolError::InvalidLayer(layer));
    }
    if expert as usize >= V4_ROUTED_EXPERTS {
        return Err(ProtocolError::InvalidExpert(expert));
    }
    if rows == 0 || rows as usize > V4_MAX_EXPERT_ROWS {
        return Err(ProtocolError::InvalidRows(rows));
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct Header {
    revision: [u8; 40],
    session_epoch: u64,
    request_id: u64,
    deadline_ns: u64,
    layer: u16,
    expert: u16,
    rows: u8,
}

fn encode_header(
    frame: &mut Vec<u8>,
    kind: u8,
    revision: &[u8; 40],
    session_epoch: u64,
    request_id: u64,
    deadline_ns: u64,
    layer: u16,
    expert: u16,
    rows: u8,
) {
    frame.extend_from_slice(MAGIC);
    put_u16(frame, VERSION);
    frame.push(kind);
    frame.extend_from_slice(revision);
    put_u64(frame, session_epoch);
    put_u64(frame, request_id);
    put_u64(frame, deadline_ns);
    put_u16(frame, layer);
    put_u16(frame, expert);
    frame.push(rows);
}

fn decode_header<'a>(
    frame: &'a [u8],
    expected_kind: u8,
) -> Result<(Header, Reader<'a>), ProtocolError> {
    if frame.len() > MAX_FRAME_BYTES {
        return Err(ProtocolError::FrameTooLarge);
    }
    let mut reader = Reader { frame, position: 0 };
    if reader.take(4)? != MAGIC {
        return Err(ProtocolError::BadMagic);
    }
    let version = reader.u16()?;
    if version != VERSION {
        return Err(ProtocolError::WrongVersion(version));
    }
    if reader.u8()? != expected_kind {
        return Err(ProtocolError::WrongKind);
    }
    let mut revision = [0; 40];
    revision.copy_from_slice(reader.take(40)?);
    let header = Header {
        revision,
        session_epoch: reader.u64()?,
        request_id: reader.u64()?,
        deadline_ns: reader.u64()?,
        layer: reader.u16()?,
        expert: reader.u16()?,
        rows: reader.u8()?,
    };
    validate_header(&header.revision, header.layer, header.expert, header.rows)?;
    Ok((header, reader))
}

fn put_u16(frame: &mut Vec<u8>, value: u16) {
    frame.extend_from_slice(&value.to_le_bytes());
}

fn put_u32(frame: &mut Vec<u8>, value: u32) {
    frame.extend_from_slice(&value.to_le_bytes());
}

fn put_u64(frame: &mut Vec<u8>, value: u64) {
    frame.extend_from_slice(&value.to_le_bytes());
}

struct Reader<'a> {
    frame: &'a [u8],
    position: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, count: usize) -> Result<&'a [u8], ProtocolError> {
        let end = self
            .position
            .checked_add(count)
            .ok_or(ProtocolError::Truncated)?;
        let value = self
            .frame
            .get(self.position..end)
            .ok_or(ProtocolError::Truncated)?;
        self.position = end;
        Ok(value)
    }

    fn u8(&mut self) -> Result<u8, ProtocolError> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, ProtocolError> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }

    fn u32(&mut self) -> Result<u32, ProtocolError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64, ProtocolError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_codec_rejects_malformed_trust_boundary() {
        let request = ExecuteRequest::live(vec![0x3f80; V4_HIDDEN], 0.625);
        let frame = encode_request(&request).unwrap();
        assert_eq!(decode_request(&frame).unwrap(), request);
        assert_eq!(
            decode_request(&frame[..frame.len() - 1]),
            Err(ProtocolError::InvalidLength)
        );

        let mut bad_revision = frame.clone();
        bad_revision[7] = b'0';
        assert_eq!(
            decode_request(&bad_revision),
            Err(ProtocolError::WrongRevision)
        );

        let mut bad_layer = frame.clone();
        bad_layer[71..73].copy_from_slice(&61_u16.to_le_bytes());
        assert_eq!(
            decode_request(&bad_layer),
            Err(ProtocolError::InvalidLayer(61))
        );

        let mut response = ExecuteResponse::for_request(&request, 10, 5, vec![1.0; V4_HIDDEN]);
        assert_eq!(
            decode_response(&encode_response(&response).unwrap()).unwrap(),
            response
        );
        response.request_id += 1;
        assert_eq!(
            validate_response(&request, &response),
            Err(ProtocolError::IdentityMismatch)
        );
    }

    #[test]
    fn owner_codec_round_trips_and_rejects_identity_before_commit() {
        let identity = OwnerIdentity::live(0, 7, [254, 222, 371]);
        let audit_task = OwnerTask {
            identity: identity.clone(),
            mode: OwnerTaskMode::Audit,
            route_weights: [0.42, 0.40, 0.49],
            hidden_bf16: vec![0x3f80; V4_HIDDEN],
        };
        let task_frame = encode_owner_task(&audit_task).unwrap();
        assert_eq!(decode_owner_task(&task_frame).unwrap(), audit_task);

        let audit = OwnerAuditResponse {
            identity: identity.clone(),
            worker_ns: 20,
            gpu_ns: 10,
            contributions: vec![1.0; OWNER_EXPERTS * V4_HIDDEN],
        };
        let audit_frame = encode_owner_audit_response(&audit).unwrap();
        let decoded_audit = decode_owner_audit_response(&audit_frame).unwrap();
        validate_owner_audit_response(&audit_task, &decoded_audit).unwrap();
        assert_eq!(
            decode_owner_certified_response(&audit_frame),
            Err(ProtocolError::WrongKind)
        );

        let certified_task = OwnerTask {
            mode: OwnerTaskMode::Certified,
            ..audit_task.clone()
        };
        let certified = OwnerCertifiedResponse {
            identity: identity.clone(),
            worker_ns: 21,
            gpu_ns: 11,
            grouped_f32: vec![3.0; V4_HIDDEN],
            bounds: vec![129; FROZEN_BLOCKS],
        };
        let certified =
            decode_owner_certified_response(&encode_owner_certified_response(&certified).unwrap())
                .unwrap();
        validate_owner_certified_response(&certified_task, &certified).unwrap();

        let mut mask = vec![0; REPAIR_MASK_BYTES];
        mask[0] = 0b0000_0101;
        let repair_request = OwnerRepairRequest {
            identity: identity.clone(),
            mask,
        };
        let repair_request =
            decode_owner_repair_request(&encode_owner_repair_request(&repair_request).unwrap())
                .unwrap();
        validate_owner_repair_request(&certified_task, &repair_request).unwrap();
        let repair = OwnerRepairResponse {
            identity,
            worker_ns: 2,
            lane_count: 2,
            values: vec![1.0; OWNER_EXPERTS * 2],
        };
        let repair =
            decode_owner_repair_response(&encode_owner_repair_response(&repair).unwrap()).unwrap();
        validate_owner_repair_response(&certified_task, &repair_request, &repair).unwrap();

        let mut duplicate = repair;
        duplicate.identity.request_id -= 1;
        assert_eq!(
            validate_owner_repair_response(&certified_task, &repair_request, &duplicate),
            Err(ProtocolError::IdentityMismatch)
        );
        let mut wrong_expert = certified.clone();
        wrong_expert.identity.expert_ids[0] += 1;
        assert_eq!(
            validate_owner_certified_response(&certified_task, &wrong_expert),
            Err(ProtocolError::IdentityMismatch)
        );
        let bad_owner = OwnerIdentity::live(2, 7, [254, 222, 371]);
        assert_eq!(
            encode_owner_task(&OwnerTask {
                identity: bad_owner,
                ..audit_task
            }),
            Err(ProtocolError::InvalidOwner(2))
        );
    }
}
