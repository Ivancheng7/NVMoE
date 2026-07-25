//! Local audit, owner-grouped, and block-certified-with-repair V4 reductions.

use crate::v4_expert::V4_HIDDEN;
use crate::v4_router::{V4_TOPK, f32_to_bf16_rne};

const VECTOR_HEADER_BYTES: usize = 16;
const CERTIFICATE_HEADER_BYTES: usize = 8;
const REPAIR_REQUEST_HEADER_BYTES: usize = 16;
const REPAIR_RESPONSE_HEADER_BYTES: usize = 16;
const REPAIR_EXPERT_HEADER_BYTES: usize = 4;
const REPAIR_MASK_BYTES: usize = V4_HIDDEN.div_ceil(8);
pub const AUDIT_WIRE_BYTES: usize = V4_TOPK * (VECTOR_HEADER_BYTES + V4_HIDDEN * size_of::<f32>());

#[derive(Debug, Eq, PartialEq)]
pub enum ReductionError {
    ContributionCount,
    ContributionLength,
    InvalidOwners,
    InvalidBlockSize,
    InvalidBounds,
    NonFiniteInput,
    NonFiniteSum,
}

impl std::fmt::Display for ReductionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for ReductionError {}

pub struct ReductionReport {
    pub owner_count: usize,
    pub block_size: usize,
    pub certificate_bytes: usize,
    pub audit_bf16: Vec<u16>,
    pub repaired_bf16: Vec<u16>,
    pub grouped_f32_mismatches: usize,
    pub grouped_bf16_mismatches: usize,
    pub certified_lanes: usize,
    pub repair_lanes: usize,
    pub bound_contained_lanes: usize,
    pub maximum_difference: f64,
}

#[derive(Clone)]
pub struct OwnerReduction {
    pub grouped_f32: Vec<f32>,
    pub bounds: Vec<u8>,
}

pub struct Certification {
    pub grouped_f32: Vec<f32>,
    pub output_bf16: Vec<u16>,
    pub repair_mask: Vec<u8>,
    pub certified_lanes: usize,
    error_bounds: Vec<f64>,
}

pub fn reduce_owner(
    contributions: &[&[f32]],
    block_size: usize,
) -> Result<OwnerReduction, ReductionError> {
    if contributions.is_empty() || contributions.len() > V4_TOPK {
        return Err(ReductionError::ContributionCount);
    }
    if contributions.iter().any(|values| values.len() != V4_HIDDEN) {
        return Err(ReductionError::ContributionLength);
    }
    if contributions
        .iter()
        .flat_map(|values| values.iter())
        .any(|value| !value.is_finite())
    {
        return Err(ReductionError::NonFiniteInput);
    }
    validate_block_size(block_size)?;

    let mut grouped_f32 = vec![0.0_f32; V4_HIDDEN];
    let mut local_abs = vec![0.0_f64; V4_HIDDEN];
    for values in contributions {
        for lane in 0..V4_HIDDEN {
            grouped_f32[lane] += values[lane];
            local_abs[lane] = next_up_f64(local_abs[lane] + values[lane].abs() as f64);
        }
    }
    if grouped_f32.iter().any(|value| !value.is_finite()) {
        return Err(ReductionError::NonFiniteSum);
    }

    let mut bounds = Vec::with_capacity(V4_HIDDEN / block_size);
    for lanes in local_abs.chunks_exact(block_size) {
        let maximum = lanes.iter().copied().fold(0.0_f64, f64::max);
        let code = encode_power_bound(maximum);
        let decoded = decode_power_bound(code);
        assert!(
            lanes.iter().all(|local| decoded >= *local),
            "encoded block bound rounded down"
        );
        bounds.push(code);
    }
    Ok(OwnerReduction {
        grouped_f32,
        bounds,
    })
}

pub fn certify_owner_groups(
    owners: &[OwnerReduction],
    block_size: usize,
) -> Result<Certification, ReductionError> {
    validate_block_size(block_size)?;
    if owners.is_empty() || owners.len() > V4_TOPK {
        return Err(ReductionError::InvalidOwners);
    }
    let block_count = V4_HIDDEN / block_size;
    if owners
        .iter()
        .any(|owner| owner.grouped_f32.len() != V4_HIDDEN || owner.bounds.len() != block_count)
    {
        return Err(ReductionError::InvalidBounds);
    }
    if owners
        .iter()
        .flat_map(|owner| owner.grouped_f32.iter())
        .any(|value| !value.is_finite())
    {
        return Err(ReductionError::NonFiniteInput);
    }

    let mut grouped_f32 = Vec::with_capacity(V4_HIDDEN);
    let mut output_bf16 = Vec::with_capacity(V4_HIDDEN);
    let mut repair_mask = vec![0_u8; V4_HIDDEN.div_ceil(8)];
    let mut error_bounds = Vec::with_capacity(V4_HIDDEN);
    let mut certified_lanes = 0;
    for lane in 0..V4_HIDDEN {
        let mut grouped = 0.0_f32;
        let mut sum_abs_bound = 0.0_f64;
        for owner in owners {
            grouped += owner.grouped_f32[lane];
            sum_abs_bound =
                next_up_f64(sum_abs_bound + decode_power_bound(owner.bounds[lane / block_size]));
        }
        if !grouped.is_finite() {
            return Err(ReductionError::NonFiniteSum);
        }
        let grouped_bits = f32_to_bf16_rne(grouped);
        grouped_f32.push(grouped);
        output_bf16.push(grouped_bits);
        let delta = error_bound(sum_abs_bound);
        error_bounds.push(delta);
        let certified = if delta.is_infinite() {
            false
        } else {
            let lower = outward_f32(next_down_f64(grouped as f64 - delta), false)?;
            let upper = outward_f32(next_up_f64(grouped as f64 + delta), true)?;
            f32_to_bf16_rne(lower) == f32_to_bf16_rne(upper)
        };
        if certified {
            certified_lanes += 1;
        } else {
            repair_mask[lane / 8] |= 1 << (lane % 8);
        }
    }
    Ok(Certification {
        grouped_f32,
        output_bf16,
        repair_mask,
        certified_lanes,
        error_bounds,
    })
}

pub fn mask_contains(mask: &[u8], lane: usize) -> bool {
    mask.get(lane / 8)
        .is_some_and(|byte| byte & (1 << (lane % 8)) != 0)
}

pub fn mask_count(mask: &[u8]) -> usize {
    mask.iter().map(|byte| byte.count_ones() as usize).sum()
}

pub fn evaluate(
    contributions: &[Vec<f32>],
    owners: &[usize; V4_TOPK],
    block_size: usize,
) -> Result<ReductionReport, ReductionError> {
    if contributions.len() != V4_TOPK {
        return Err(ReductionError::ContributionCount);
    }
    if contributions.iter().any(|values| values.len() != V4_HIDDEN) {
        return Err(ReductionError::ContributionLength);
    }
    if contributions
        .iter()
        .flatten()
        .any(|value| !value.is_finite())
    {
        return Err(ReductionError::NonFiniteInput);
    }
    validate_block_size(block_size)?;
    let owner_count = owners.iter().copied().max().unwrap() + 1;
    if owner_count > V4_TOPK || (0..owner_count).any(|owner| !owners.contains(&owner)) {
        return Err(ReductionError::InvalidOwners);
    }

    let mut owner_reductions = Vec::with_capacity(owner_count);
    for owner in 0..owner_count {
        let local: Vec<_> = contributions
            .iter()
            .zip(owners)
            .filter_map(|(values, &assigned)| (assigned == owner).then_some(values.as_slice()))
            .collect();
        owner_reductions.push(reduce_owner(&local, block_size)?);
    }
    let certification = certify_owner_groups(&owner_reductions, block_size)?;

    let mut audit_bf16 = Vec::with_capacity(V4_HIDDEN);
    let mut repaired_bf16 = certification.output_bf16.clone();
    let mut grouped_f32_mismatches = 0;
    let mut grouped_bf16_mismatches = 0;
    let mut bound_contained_lanes = 0;
    let mut maximum_difference = 0.0_f64;
    for lane in 0..V4_HIDDEN {
        let mut audit = 0.0_f32;
        for slot in 0..V4_TOPK {
            audit += contributions[slot][lane];
        }
        let grouped = certification.grouped_f32[lane];
        if !audit.is_finite() || !grouped.is_finite() {
            return Err(ReductionError::NonFiniteSum);
        }
        let audit_bits = f32_to_bf16_rne(audit);
        let grouped_bits = f32_to_bf16_rne(grouped);
        audit_bf16.push(audit_bits);
        grouped_f32_mismatches += usize::from(audit.to_bits() != grouped.to_bits());
        grouped_bf16_mismatches += usize::from(audit_bits != grouped_bits);

        let difference = (audit as f64 - grouped as f64).abs();
        maximum_difference = maximum_difference.max(difference);
        if difference <= certification.error_bounds[lane] {
            bound_contained_lanes += 1;
        }
        if mask_contains(&certification.repair_mask, lane) {
            // Repair uses the raw route-ordered audit result, not a grouped partial.
            repaired_bf16[lane] = audit_bits;
        }
    }
    Ok(ReductionReport {
        owner_count,
        block_size,
        certificate_bytes: owner_count * V4_HIDDEN / block_size,
        audit_bf16,
        repaired_bf16,
        grouped_f32_mismatches,
        grouped_bf16_mismatches,
        certified_lanes: certification.certified_lanes,
        repair_lanes: V4_HIDDEN - certification.certified_lanes,
        bound_contained_lanes,
        maximum_difference,
    })
}

fn validate_block_size(block_size: usize) -> Result<(), ReductionError> {
    if block_size == 0 || !V4_HIDDEN.is_multiple_of(block_size) {
        Err(ReductionError::InvalidBlockSize)
    } else {
        Ok(())
    }
}

/// Response plus repair traffic. Each owner sends one grouped FP32 vector and
/// its u8 block bounds, receives the full repair bitmask, then returns one raw
/// FP32 value per owned expert and ambiguous lane. Fixed headers include IDs.
pub fn wire_bytes(report: &ReductionReport) -> usize {
    let grouped = report.owner_count
        * (VECTOR_HEADER_BYTES + V4_HIDDEN * size_of::<f32>() + CERTIFICATE_HEADER_BYTES)
        + report.certificate_bytes;
    let requests = report.owner_count * (REPAIR_REQUEST_HEADER_BYTES + REPAIR_MASK_BYTES);
    let responses = report.owner_count * REPAIR_RESPONSE_HEADER_BYTES
        + V4_TOPK * REPAIR_EXPERT_HEADER_BYTES
        + V4_TOPK * report.repair_lanes * size_of::<f32>();
    grouped + requests + responses
}

/// Code 0 is exactly zero, 1..=254 decode to 2^(code-128), and 255 is +infinity.
/// Integer exponent/mantissa inspection provides ceil(log2(value)) without a
/// transcendental or a rounding-mode assumption.
fn encode_power_bound(value: f64) -> u8 {
    if value == 0.0 {
        return 0;
    }
    if !value.is_finite() || value < 0.0 {
        return u8::MAX;
    }
    let minimum = f64::from_bits(((1023 - 127) as u64) << 52);
    if value <= minimum {
        return 1;
    }
    let bits = value.to_bits();
    let exponent = ((bits >> 52) & 0x7ff) as i32 - 1023;
    let ceil_exponent = exponent + i32::from(bits & ((1_u64 << 52) - 1) != 0);
    if ceil_exponent > 126 {
        u8::MAX
    } else {
        (ceil_exponent + 128) as u8
    }
}

fn decode_power_bound(code: u8) -> f64 {
    match code {
        0 => 0.0,
        u8::MAX => f64::INFINITY,
        _ => f64::from_bits(((code as i32 - 128 + 1023) as u64) << 52),
    }
}

fn error_bound(sum_abs: f64) -> f64 {
    const UNIT_ROUNDOFF: f64 = 1.0 / (1_u64 << 24) as f64;
    let gamma_5 = next_up_f64(5.0 * UNIT_ROUNDOFF / (1.0 - 5.0 * UNIT_ROUNDOFF));
    let relative = next_up_f64(2.0 * gamma_5 * sum_abs);
    // Covers the absolute error term if a real lane enters gradual underflow.
    next_up_f64(relative + 12.0 * f32::from_bits(1) as f64)
}

fn next_up_f64(value: f64) -> f64 {
    if value == f64::INFINITY {
        return value;
    }
    if value == 0.0 {
        return f64::from_bits(1);
    }
    f64::from_bits(if value > 0.0 {
        value.to_bits() + 1
    } else {
        value.to_bits() - 1
    })
}

fn next_down_f64(value: f64) -> f64 {
    if value == f64::NEG_INFINITY {
        return value;
    }
    if value == 0.0 {
        return f64::from_bits(0x8000_0000_0000_0001);
    }
    f64::from_bits(if value > 0.0 {
        value.to_bits() - 1
    } else {
        value.to_bits() + 1
    })
}

fn outward_f32(value: f64, upward: bool) -> Result<f32, ReductionError> {
    if !value.is_finite() || value.abs() > f32::MAX as f64 {
        return Err(ReductionError::NonFiniteSum);
    }
    let rounded = value as f32;
    if upward && (rounded as f64) < value {
        Ok(next_f32(rounded, true))
    } else if !upward && (rounded as f64) > value {
        Ok(next_f32(rounded, false))
    } else {
        Ok(rounded)
    }
}

fn next_f32(value: f32, upward: bool) -> f32 {
    if value == 0.0 {
        return f32::from_bits(if upward { 1 } else { 0x8000_0001 });
    }
    let increment = (value > 0.0) == upward;
    f32::from_bits(if increment {
        value.to_bits() + 1
    } else {
        value.to_bits() - 1
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encoded_bound_never_rounds_down() {
        let values = [
            0.0,
            f32::from_bits(1) as f64,
            f32::MIN_POSITIVE as f64,
            0.5,
            1.0,
            1.000_000_1,
            f32::MAX as f64,
            f64::INFINITY,
        ];
        for value in values {
            assert!(decode_power_bound(encode_power_bound(value)) >= value);
        }
        for code in 1..u8::MAX {
            let exact = decode_power_bound(code);
            assert_eq!(decode_power_bound(encode_power_bound(exact)), exact);
        }
    }

    #[test]
    fn every_block_lane_local_sum_is_below_its_decoded_bound() {
        let owners = [0, 1, 0, 1, 0, 1];
        let block_size = 32;
        for owner in 0..2 {
            for block in 0..V4_HIDDEN / block_size {
                let start = block * block_size;
                let mut locals = [0.0_f64; 32];
                for (offset, local) in locals.iter_mut().enumerate() {
                    let lane = start + offset;
                    for slot in 0..V4_TOPK {
                        if owners[slot] == owner {
                            let value = match lane % 4 {
                                0 => 0.0,
                                1 => f32::from_bits(1),
                                2 => f32::MIN_POSITIVE,
                                _ => (slot + 1) as f32 / 8.0,
                            };
                            *local = next_up_f64(*local + value.abs() as f64);
                        }
                    }
                }
                let maximum = locals.iter().copied().fold(0.0_f64, f64::max);
                let decoded = decode_power_bound(encode_power_bound(maximum));
                assert!(locals.into_iter().all(|local| decoded >= local));
            }
        }
    }

    #[test]
    fn compressed_groupings_contain_and_repair_to_audit_bits() {
        let mut contributions = vec![vec![0.0; V4_HIDDEN]; V4_TOPK];
        for lane in 0..V4_HIDDEN {
            let values = if (lane / 32) % 2 == 0 {
                [0.125, 0.25, 0.375, 0.5, 0.625, 0.75]
            } else {
                [1.0e8, 1.0, -1.0e8, 0.5, 0.25, 0.125]
            };
            for slot in 0..V4_TOPK {
                contributions[slot][lane] = values[slot];
            }
        }
        let placements = [[0, 0, 0, 1, 1, 1], [0, 1, 0, 1, 0, 1]];
        for owners in placements {
            let report = evaluate(&contributions, &owners, 32).unwrap();
            assert_eq!(report.bound_contained_lanes, V4_HIDDEN);
            assert_eq!(report.repaired_bf16, report.audit_bf16);
            assert!(report.certified_lanes > 0);
            assert!(report.repair_lanes > 0);
        }
    }
}
