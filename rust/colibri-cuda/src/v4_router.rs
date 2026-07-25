//! DeepSeek-V4 router oracle. The projection is deliberately scalar: it
//! validates pinned semantics until a measured GEMM path exists.

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
    if input.len() % V4_HIDDEN != 0 {
        return Err(RouterError::InputLength {
            expected_multiple: V4_HIDDEN,
            actual: input.len(),
        });
    }
    let expected = V4_ROUTED_EXPERTS * V4_HIDDEN;
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
    let mut logits = vec![0.0; input.len() / V4_HIDDEN * V4_ROUTED_EXPERTS];
    for (row, input) in input.chunks_exact(V4_HIDDEN).enumerate() {
        for expert in 0..V4_ROUTED_EXPERTS {
            let mut dot = 0.0;
            for column in 0..V4_HIDDEN {
                dot += bf16_to_f32(input[column]) * bf16_to_f32(gate[expert * V4_HIDDEN + column]);
            }
            logits[row * V4_ROUTED_EXPERTS + expert] = dot;
        }
    }
    Ok(logits)
}

pub fn select_hash(logits: &[f32], hash_ids: &[i64]) -> Result<RouterSelection, RouterError> {
    let rows = checked_logit_rows(logits)?;
    if hash_ids.len() != rows * V4_TOPK {
        return Err(RouterError::HashLength {
            expected_multiple: rows * V4_TOPK,
            actual: hash_ids.len(),
        });
    }
    let mut ids = Vec::with_capacity(hash_ids.len());
    let mut weights = Vec::with_capacity(hash_ids.len());
    for row in 0..rows {
        let selected = &hash_ids[row * V4_TOPK..][..V4_TOPK];
        let mut raw = [0.0; V4_TOPK];
        for (slot, &id) in selected.iter().enumerate() {
            let id = checked_expert_id(row * V4_TOPK + slot, id)?;
            ids.push(id as i32);
            raw[slot] = sqrtsoftplus(logits[row * V4_ROUTED_EXPERTS + id]);
        }
        append_weights(&raw, &mut weights);
    }
    Ok(RouterSelection { ids, weights })
}

pub fn select_learned(logits: &[f32], bias: &[f32]) -> Result<RouterSelection, RouterError> {
    let rows = checked_logit_rows(logits)?;
    if bias.len() != V4_ROUTED_EXPERTS {
        return Err(RouterError::BiasLength {
            expected: V4_ROUTED_EXPERTS,
            actual: bias.len(),
        });
    }
    for (index, &value) in bias.iter().enumerate() {
        if !value.is_finite() {
            return Err(RouterError::NonFiniteBias { index });
        }
    }
    let mut ids = Vec::with_capacity(rows * V4_TOPK);
    let mut weights = Vec::with_capacity(rows * V4_TOPK);
    for row in 0..rows {
        let mut selected_ids = [0usize; V4_TOPK];
        let mut selected_scores = [f32::NEG_INFINITY; V4_TOPK];
        let mut selected_raw = [0.0; V4_TOPK];
        for expert in 0..V4_ROUTED_EXPERTS {
            let raw = sqrtsoftplus(logits[row * V4_ROUTED_EXPERTS + expert]);
            let score = raw + bias[expert];
            // PyTorch does not promise a tie order. The pinned smoke has no
            // ties; this stable fallback exists only so CPU and GPU oracles
            // take the same path for malformed/adversarial inputs.
            let mut slot = 0;
            while slot < V4_TOPK && score <= selected_scores[slot] {
                slot += 1;
            }
            if slot < V4_TOPK {
                let mut tail = V4_TOPK - 1;
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
        ids.extend(selected_ids.map(|id| id as i32));
        append_weights(&selected_raw, &mut weights);
    }
    Ok(RouterSelection { ids, weights })
}

fn checked_logit_rows(logits: &[f32]) -> Result<usize, RouterError> {
    if logits.len() % V4_ROUTED_EXPERTS != 0 {
        return Err(RouterError::LogitLength {
            expected_multiple: V4_ROUTED_EXPERTS,
            actual: logits.len(),
        });
    }
    for (index, &value) in logits.iter().enumerate() {
        if !value.is_finite() {
            return Err(RouterError::NonFiniteLogit { index });
        }
    }
    Ok(logits.len() / V4_ROUTED_EXPERTS)
}

fn checked_expert_id(index: usize, value: i64) -> Result<usize, RouterError> {
    let id = usize::try_from(value).ok();
    match id.filter(|id| *id < V4_ROUTED_EXPERTS) {
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

fn append_weights(raw: &[f32; V4_TOPK], weights: &mut Vec<f32>) {
    let sum = raw.iter().sum::<f32>();
    weights.extend(raw.iter().map(|raw| raw / sum * V4_ROUTE_SCALE));
}

#[cfg(test)]
mod tests {
    use super::*;

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
