//! DeepSeek-V4 router oracle. The projection is deliberately scalar: it
//! validates pinned semantics until a measured GEMM path exists.

use crate::v4_spec::{PRO, V4Spec};

pub const V4_HIDDEN: usize = 7_168;
pub const V4_ROUTED_EXPERTS: usize = 384;
pub const V4_TOPK: usize = 6;
pub const V4_RMS_EPS: f32 = 1e-6;
pub const V4_ROUTE_SCALE: f32 = 2.5;

#[derive(Debug, PartialEq, Eq)]
pub enum RouterError {
    InputLength {
        expected_multiple: usize,
        actual: usize,
    },
    NormLength {
        expected: usize,
        actual: usize,
    },
    GateLength {
        expected: usize,
        actual: usize,
    },
    LogitLength {
        expected_multiple: usize,
        actual: usize,
    },
    BiasLength {
        expected: usize,
        actual: usize,
    },
    HashLength {
        expected_multiple: usize,
        actual: usize,
    },
    NonFiniteInput {
        index: usize,
    },
    NonFiniteNorm {
        index: usize,
    },
    NonFiniteGate {
        index: usize,
    },
    NonFiniteLogit {
        index: usize,
    },
    NonFiniteBias {
        index: usize,
    },
    ExpertIdOutOfRange {
        index: usize,
        value: i64,
    },
}

impl std::fmt::Display for RouterError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for RouterError {}

#[derive(Debug, PartialEq)]
pub struct RouterSelection {
    pub ids: Vec<i32>,
    pub weights: Vec<f32>,
}

pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

pub fn f32_to_bf16_rne(value: f32) -> u16 {
    let bits = value.to_bits();
    ((bits.wrapping_add(0x7fff + ((bits >> 16) & 1))) >> 16) as u16
}

pub fn rmsnorm_bf16(input: &[u16], norm: &[u16]) -> Result<Vec<u16>, RouterError> {
    rmsnorm_bf16_width(input, norm, V4_HIDDEN)
}

pub fn rmsnorm_bf16_width(
    input: &[u16],
    norm: &[u16],
    width: usize,
) -> Result<Vec<u16>, RouterError> {
    if width == 0 || input.len() % width != 0 {
        return Err(RouterError::InputLength {
            expected_multiple: width,
            actual: input.len(),
        });
    }
    if norm.len() != width {
        return Err(RouterError::NormLength {
            expected: width,
            actual: norm.len(),
        });
    }
    for (index, &value) in norm.iter().enumerate() {
        if !bf16_to_f32(value).is_finite() {
            return Err(RouterError::NonFiniteNorm { index });
        }
    }
    let mut output = vec![0; input.len()];
    for (row, (input, output)) in input
        .chunks_exact(width)
        .zip(output.chunks_exact_mut(width))
        .enumerate()
    {
        let mut squares = Vec::with_capacity(width);
        for (column, &value) in input.iter().enumerate() {
            let value = bf16_to_f32(value);
            if !value.is_finite() {
                return Err(RouterError::NonFiniteInput {
                    index: row * width + column,
                });
            }
            squares.push(value * value);
        }
        let square_sum = pytorch_contiguous_sum_f32(&squares);
        let inv_rms = (square_sum / width as f32 + V4_RMS_EPS).sqrt().recip();
        for ((output, &input), &norm) in output.iter_mut().zip(input).zip(norm) {
            let normalized = bf16_to_f32(input) * inv_rms;
            *output = f32_to_bf16_rne(normalized * bf16_to_f32(norm));
        }
    }
    Ok(output)
}

/// Match the pinned PyTorch CPU float reduction: AVX2-width vectors, four
/// interleaved partials, and the four-level cascade used by SumKernel.cpp.
///
/// ponytail: this is the one reduction shape required by the official V4
/// oracle. Add another backend/order only when a pinned fixture requires it.
pub(crate) fn pytorch_contiguous_sum_f32(values: &[f32]) -> f32 {
    const LANES: usize = 8;
    const ILP: usize = 4;
    const LEVELS: usize = 4;

    let vector_count = values.len() / LANES;
    let groups = vector_count / ILP;
    if groups == 0 {
        return values.iter().copied().sum();
    }
    let ceil_log2 = usize::BITS as usize - (groups - 1).leading_zeros() as usize;
    let level_power = 4.max(ceil_log2 / LEVELS);
    let level_step = 1 << level_power;
    let level_mask = level_step - 1;
    let mut partials = [[[0.0_f32; LANES]; ILP]; LEVELS];
    let mut group = 0;

    while group + level_step <= groups {
        for _ in 0..level_step {
            for interleave in 0..ILP {
                let start = (group * ILP + interleave) * LANES;
                for lane in 0..LANES {
                    partials[0][interleave][lane] += values[start + lane];
                }
            }
            group += 1;
        }
        for level in 1..LEVELS {
            for interleave in 0..ILP {
                for lane in 0..LANES {
                    partials[level][interleave][lane] += partials[level - 1][interleave][lane];
                    partials[level - 1][interleave][lane] = 0.0;
                }
            }
            if group & (level_mask << (level * level_power)) != 0 {
                break;
            }
        }
    }
    while group < groups {
        for interleave in 0..ILP {
            let start = (group * ILP + interleave) * LANES;
            for lane in 0..LANES {
                partials[0][interleave][lane] += values[start + lane];
            }
        }
        group += 1;
    }
    for level in 1..LEVELS {
        for interleave in 0..ILP {
            for lane in 0..LANES {
                partials[0][interleave][lane] += partials[level][interleave][lane];
            }
        }
    }
    for interleave in 1..ILP {
        for lane in 0..LANES {
            partials[0][0][lane] += partials[0][interleave][lane];
        }
    }

    let mut sum = 0.0_f32;
    for &value in &values[vector_count * LANES..] {
        sum += value;
    }
    for lane in 0..LANES {
        sum += partials[0][0][lane];
    }
    sum
}

pub fn project_gate_logits(input: &[u16], gate: &[u16]) -> Result<Vec<f32>, RouterError> {
    project_gate_logits_for(&PRO, input, gate)
}

pub fn project_gate_logits_for(
    spec: &V4Spec,
    input: &[u16],
    gate: &[u16],
) -> Result<Vec<f32>, RouterError> {
    if input.len() % spec.hidden != 0 {
        return Err(RouterError::InputLength {
            expected_multiple: spec.hidden,
            actual: input.len(),
        });
    }
    let expected = spec.routed_experts * spec.hidden;
    if gate.len() != expected {
        return Err(RouterError::GateLength {
            expected,
            actual: gate.len(),
        });
    }
    for (index, &value) in gate.iter().enumerate() {
        if !bf16_to_f32(value).is_finite() {
            return Err(RouterError::NonFiniteGate { index });
        }
    }
    // ponytail: scalar oracle GEMM; replace only after an SM89 projection
    // benchmark and selected-ID parity prove a tensor-core path.
    let mut logits = vec![0.0; input.len() / spec.hidden * spec.routed_experts];
    for (row, input) in input.chunks_exact(spec.hidden).enumerate() {
        for expert in 0..spec.routed_experts {
            let mut dot = 0.0;
            for column in 0..spec.hidden {
                dot +=
                    bf16_to_f32(input[column]) * bf16_to_f32(gate[expert * spec.hidden + column]);
            }
            logits[row * spec.routed_experts + expert] = dot;
        }
    }
    Ok(logits)
}

pub fn select_hash(logits: &[f32], hash_ids: &[i64]) -> Result<RouterSelection, RouterError> {
    select_hash_for(&PRO, logits, hash_ids)
}

pub fn select_hash_for(
    spec: &V4Spec,
    logits: &[f32],
    hash_ids: &[i64],
) -> Result<RouterSelection, RouterError> {
    let rows = checked_logit_rows(spec, logits)?;
    if hash_ids.len() != rows * spec.experts_per_token {
        return Err(RouterError::HashLength {
            expected_multiple: rows * spec.experts_per_token,
            actual: hash_ids.len(),
        });
    }
    let mut ids = Vec::with_capacity(hash_ids.len());
    let mut weights = Vec::with_capacity(hash_ids.len());
    for row in 0..rows {
        let selected = &hash_ids[row * spec.experts_per_token..][..spec.experts_per_token];
        let mut raw = vec![0.0; spec.experts_per_token];
        for (slot, &id) in selected.iter().enumerate() {
            let id = checked_expert_id(spec, row * spec.experts_per_token + slot, id)?;
            ids.push(id as i32);
            raw[slot] = sqrtsoftplus(logits[row * spec.routed_experts + id]);
        }
        append_weights(spec, &raw, &mut weights);
    }
    Ok(RouterSelection { ids, weights })
}

pub fn select_learned(logits: &[f32], bias: &[f32]) -> Result<RouterSelection, RouterError> {
    select_learned_for(&PRO, logits, bias)
}

pub fn select_learned_for(
    spec: &V4Spec,
    logits: &[f32],
    bias: &[f32],
) -> Result<RouterSelection, RouterError> {
    let rows = checked_logit_rows(spec, logits)?;
    if bias.len() != spec.routed_experts {
        return Err(RouterError::BiasLength {
            expected: spec.routed_experts,
            actual: bias.len(),
        });
    }
    for (index, &value) in bias.iter().enumerate() {
        if !value.is_finite() {
            return Err(RouterError::NonFiniteBias { index });
        }
    }
    let mut ids = Vec::with_capacity(rows * spec.experts_per_token);
    let mut weights = Vec::with_capacity(rows * spec.experts_per_token);
    for row in 0..rows {
        let mut selected_ids = vec![0; spec.experts_per_token];
        let mut selected_scores = vec![f32::NEG_INFINITY; spec.experts_per_token];
        let mut selected_raw = vec![0.0; spec.experts_per_token];
        for expert in 0..spec.routed_experts {
            let raw = sqrtsoftplus(logits[row * spec.routed_experts + expert]);
            let score = raw + bias[expert];
            // PyTorch does not promise a tie order. The pinned smoke has no
            // ties; this stable fallback exists only so CPU and GPU oracles
            // take the same path for malformed/adversarial inputs.
            let mut slot = 0;
            while slot < spec.experts_per_token && score <= selected_scores[slot] {
                slot += 1;
            }
            if slot < spec.experts_per_token {
                let mut tail = spec.experts_per_token - 1;
                while tail > slot {
                    selected_ids[tail] = selected_ids[tail - 1];
                    selected_scores[tail] = selected_scores[tail - 1];
                    selected_raw[tail] = selected_raw[tail - 1];
                    tail -= 1;
                }
                selected_ids[slot] = expert;
                selected_scores[slot] = score;
                selected_raw[slot] = raw;
            }
        }
        ids.extend(selected_ids.into_iter().map(|id| id as i32));
        append_weights(spec, &selected_raw, &mut weights);
    }
    Ok(RouterSelection { ids, weights })
}

fn checked_logit_rows(spec: &V4Spec, logits: &[f32]) -> Result<usize, RouterError> {
    if logits.len() % spec.routed_experts != 0 {
        return Err(RouterError::LogitLength {
            expected_multiple: spec.routed_experts,
            actual: logits.len(),
        });
    }
    for (index, &value) in logits.iter().enumerate() {
        if !value.is_finite() {
            return Err(RouterError::NonFiniteLogit { index });
        }
    }
    Ok(logits.len() / spec.routed_experts)
}

fn checked_expert_id(spec: &V4Spec, index: usize, value: i64) -> Result<usize, RouterError> {
    let id = usize::try_from(value).ok();
    match id.filter(|id| *id < spec.routed_experts) {
        Some(id) => Ok(id),
        None => Err(RouterError::ExpertIdOutOfRange { index, value }),
    }
}

fn sqrtsoftplus(value: f32) -> f32 {
    let softplus = if value > 20.0 {
        value
    } else if value > 0.0 {
        value + (-value).exp().ln_1p()
    } else {
        value.exp().ln_1p()
    };
    softplus.sqrt()
}

fn append_weights(spec: &V4Spec, raw: &[f32], weights: &mut Vec<f32>) {
    let sum = raw.iter().sum::<f32>();
    weights.extend(raw.iter().map(|raw| raw / sum * spec.route_scale()));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v4_spec::FLASH_0731;

    #[test]
    fn flash_0731_router_uses_4096_by_256_and_scale_1_5() {
        let input = vec![f32_to_bf16_rne(1.0); FLASH_0731.hidden];
        let gate = vec![f32_to_bf16_rne(0.0); FLASH_0731.hidden * FLASH_0731.routed_experts];
        let logits = project_gate_logits_for(&FLASH_0731, &input, &gate).unwrap();
        let routes = select_hash_for(&FLASH_0731, &logits, &[0, 1, 2, 3, 4, 5]).unwrap();
        assert_eq!(routes.ids, [0, 1, 2, 3, 4, 5]);
        assert!((routes.weights.iter().sum::<f32>() - 1.5).abs() < 1e-6);
        assert!(select_hash_for(&FLASH_0731, &logits, &[256, 1, 2, 3, 4, 5]).is_err());
    }

    #[test]
    fn hash_uses_table_order_and_learned_uses_biased_selection() {
        let mut logits = vec![-10.0; V4_ROUTED_EXPERTS];
        logits[1] = 1.0;
        logits[2] = 2.0;
        logits[3] = 3.0;
        logits[4] = 4.0;
        logits[5] = 5.0;
        logits[6] = 6.0;
        logits[7] = 7.0;
        let hash = select_hash(&logits, &[6, 1, 5, 2, 4, 3]).unwrap();
        assert_eq!(hash.ids, [6, 1, 5, 2, 4, 3]);
        assert!((hash.weights.iter().sum::<f32>() - 2.5).abs() < 1e-6);

        let mut bias = vec![0.0; V4_ROUTED_EXPERTS];
        bias[1] = 100.0;
        let learned = select_learned(&logits, &bias).unwrap();
        assert_eq!(learned.ids[0], 1, "bias changes only selected IDs");
        let raw = sqrtsoftplus(logits[1]);
        let expected = raw
            / learned
                .ids
                .iter()
                .map(|&id| sqrtsoftplus(logits[id as usize]))
                .sum::<f32>()
            * V4_ROUTE_SCALE;
        assert!((learned.weights[0] - expected).abs() < 1e-6);
    }

    #[test]
    fn rmsnorm_rounds_to_bf16_and_rejects_malformed_router_inputs() {
        let input = vec![f32_to_bf16_rne(1.0); V4_HIDDEN];
        let norm = vec![f32_to_bf16_rne(1.0); V4_HIDDEN];
        let output = rmsnorm_bf16(&input, &norm).unwrap();
        assert_eq!(output, input);
        assert!(matches!(
            select_hash(&vec![0.0; V4_ROUTED_EXPERTS], &[0, 1, 2, 3, 4, 384]),
            Err(RouterError::ExpertIdOutOfRange { .. })
        ));
        assert!(matches!(
            select_learned(&vec![0.0; V4_ROUTED_EXPERTS], &[0.0; 383]),
            Err(RouterError::BiasLength { .. })
        ));
        assert!(matches!(
            select_learned(
                &vec![f32::NAN; V4_ROUTED_EXPERTS],
                &[0.0; V4_ROUTED_EXPERTS]
            ),
            Err(RouterError::NonFiniteLogit { .. })
        ));
        let mut bad_norm = norm;
        bad_norm[0] = f32_to_bf16_rne(f32::NAN);
        assert!(matches!(
            rmsnorm_bf16(&input, &bad_norm),
            Err(RouterError::NonFiniteNorm { .. })
        ));
    }
}
