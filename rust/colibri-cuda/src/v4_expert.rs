//! One bounded, range-backed DeepSeek-V4-Pro routed-expert oracle.

use std::error::Error;

use crate::safetensors::{ShardFile, TensorDtype};
use crate::v4_fp4::{PackedFp4E2M1x2, decode_e8m0, decode_fp4};
use crate::v4_router::{bf16_to_f32, f32_to_bf16_rne};
use crate::v4_spec::{FLASH_0731, PRO, V4Spec};

pub const V4_HIDDEN: usize = 7_168;
pub const V4_EXPERT_INTERMEDIATE: usize = 3_072;
pub const V4_MAX_EXPERT_ROWS: usize = 4;
pub const V4_W1_BYTES: usize = V4_EXPERT_INTERMEDIATE * V4_HIDDEN / 2;
pub const V4_W1_SCALE_BYTES: usize = V4_EXPERT_INTERMEDIATE * V4_HIDDEN / 32;
pub const V4_W2_BYTES: usize = V4_HIDDEN * V4_EXPERT_INTERMEDIATE / 2;
pub const V4_W2_SCALE_BYTES: usize = V4_HIDDEN * V4_EXPERT_INTERMEDIATE / 32;
pub const V4_ROUTED_EXPERT_BYTES: usize =
    V4_W1_BYTES * 2 + V4_W2_BYTES + V4_W1_SCALE_BYTES * 2 + V4_W2_SCALE_BYTES;
const ACT_BLOCK: usize = 128;
const WEIGHT_BLOCK: usize = 32;

#[derive(Debug)]
pub enum ExpertError {
    Rows { actual: usize },
    InputLength { expected: usize, actual: usize },
    ContributionCount { actual: usize },
    ContributionLength { actual: usize },
    DuplicateExpert(u16),
    RouteWeight(f32),
    NonFiniteInput { index: usize },
    Tensor(String),
}

impl std::fmt::Display for ExpertError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Rows { actual } => write!(
                formatter,
                "rows must be 1..={V4_MAX_EXPERT_ROWS}, got {actual}"
            ),
            Self::InputLength { expected, actual } => {
                write!(
                    formatter,
                    "input must be a multiple of {expected} BF16 values, got {actual}"
                )
            }
            Self::ContributionCount { actual } => write!(
                formatter,
                "canonical MoE reduction requires {} routed contributions, got {actual}",
                crate::v4_router::V4_TOPK
            ),
            Self::ContributionLength { actual } => write!(
                formatter,
                "MoE contributions must be 1..={V4_MAX_EXPERT_ROWS} complete hidden rows, got {actual} values"
            ),
            Self::DuplicateExpert(expert) => {
                write!(formatter, "duplicate routed expert {expert}")
            }
            Self::RouteWeight(weight) => write!(formatter, "route weight is not finite: {weight}"),
            Self::NonFiniteInput { index } => write!(formatter, "input[{index}] is not finite"),
            Self::Tensor(message) => formatter.write_str(message),
        }
    }
}

impl Error for ExpertError {}

/// One compressed official L0 expert. No decoded expert matrix is retained.
pub struct RoutedExpert {
    pub spec: &'static V4Spec,
    pub w1: Vec<u8>,
    pub w1_scale: Vec<u8>,
    pub w2: Vec<u8>,
    pub w2_scale: Vec<u8>,
    pub w3: Vec<u8>,
    pub w3_scale: Vec<u8>,
    pub alloc_ms: f64,
    pub read_ms: f64,
}

pub struct RoutedExpertRangePlan {
    pub expert_id: u16,
    names: [String; 6],
    bytes: [usize; 6],
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RoutedRangeReadTelemetry {
    pub logical_bytes: usize,
    pub physical_bytes: usize,
}

impl RoutedExpertRangePlan {
    pub fn new(shard: &ShardFile, layer: usize, expert_id: u16) -> Result<Self, Box<dyn Error>> {
        Self::new_for(&PRO, shard, layer, expert_id)
    }

    pub fn new_for(
        spec: &V4Spec,
        shard: &ShardFile,
        layer: usize,
        expert_id: u16,
    ) -> Result<Self, Box<dyn Error>> {
        let prefix = routed_expert_prefix_for(spec, layer, expert_id)?;
        Self::new_with_prefix(spec, shard, expert_id, &prefix)
    }

    /// MTP is the one official routed-expert namespace outside `layers.*`.
    /// Keeping it explicit prevents the ordinary layer range from widening.
    pub fn new_mtp(shard: &ShardFile, expert_id: u16) -> Result<Self, Box<dyn Error>> {
        let prefix = mtp_routed_expert_prefix(expert_id)?;
        Self::new_with_prefix(&PRO, shard, expert_id, &prefix)
    }

    pub fn new_dspark(
        shard: &ShardFile,
        stage: usize,
        expert_id: u16,
    ) -> Result<Self, Box<dyn Error>> {
        let prefix = dspark_routed_expert_prefix(stage, expert_id)?;
        Self::new_with_prefix(&FLASH_0731, shard, expert_id, &prefix)
    }

    fn new_with_prefix(
        spec: &V4Spec,
        shard: &ShardFile,
        expert_id: u16,
        prefix: &str,
    ) -> Result<Self, Box<dyn Error>> {
        let names = [
            format!("{prefix}.w1.weight"),
            format!("{prefix}.w1.scale"),
            format!("{prefix}.w2.weight"),
            format!("{prefix}.w2.scale"),
            format!("{prefix}.w3.weight"),
            format!("{prefix}.w3.scale"),
        ];
        let bytes = [
            spec.moe_intermediate * spec.hidden / 2,
            spec.moe_intermediate * spec.hidden / WEIGHT_BLOCK,
            spec.hidden * spec.moe_intermediate / 2,
            spec.hidden * spec.moe_intermediate / WEIGHT_BLOCK,
            spec.moe_intermediate * spec.hidden / 2,
            spec.moe_intermediate * spec.hidden / WEIGHT_BLOCK,
        ];
        let expected = [
            (
                TensorDtype::Fp4E2M1x2,
                [spec.moe_intermediate as u64, (spec.hidden / 2) as u64],
            ),
            (
                TensorDtype::F8E8M0,
                [
                    spec.moe_intermediate as u64,
                    (spec.hidden / WEIGHT_BLOCK) as u64,
                ],
            ),
            (
                TensorDtype::Fp4E2M1x2,
                [spec.hidden as u64, (spec.moe_intermediate / 2) as u64],
            ),
            (
                TensorDtype::F8E8M0,
                [
                    spec.hidden as u64,
                    (spec.moe_intermediate / WEIGHT_BLOCK) as u64,
                ],
            ),
            (
                TensorDtype::Fp4E2M1x2,
                [spec.moe_intermediate as u64, (spec.hidden / 2) as u64],
            ),
            (
                TensorDtype::F8E8M0,
                [
                    spec.moe_intermediate as u64,
                    (spec.hidden / WEIGHT_BLOCK) as u64,
                ],
            ),
        ];
        for (((name, bytes), (dtype, shape)), index) in
            names.iter().zip(bytes).zip(expected).zip(0_usize..)
        {
            let info = shard.tensor(name)?;
            if info.dtype != dtype
                || info.shape != shape
                || usize::try_from(info.byte_len)? != bytes
            {
                return Err(ExpertError::Tensor(format!(
                    "{name} range {index} disagrees with the routed lane contract"
                ))
                .into());
            }
        }
        Ok(Self {
            expert_id,
            names,
            bytes,
        })
    }

    pub fn read_into(
        &self,
        shard: &ShardFile,
        buffers: [&mut [u8]; 6],
    ) -> Result<(), Box<dyn Error>> {
        let mode = crate::safetensors::RangeReadMode::from_stream_env()?;
        self.read_into_mode(shard, buffers, mode)?;
        Ok(())
    }

    pub fn read_into_mode(
        &self,
        shard: &ShardFile,
        buffers: [&mut [u8]; 6],
        mode: crate::safetensors::RangeReadMode,
    ) -> Result<RoutedRangeReadTelemetry, Box<dyn Error>> {
        let mut telemetry = RoutedRangeReadTelemetry::default();
        for (index, ((name, &expected), buffer)) in
            self.names.iter().zip(&self.bytes).zip(buffers).enumerate()
        {
            if buffer.len() != expected {
                return Err(ExpertError::Tensor(format!(
                    "{name} lane buffer has {} bytes, expected {expected}",
                    buffer.len()
                ))
                .into());
            }
            let read = shard.read_at_mode(name, 0, buffer, mode)?;
            telemetry.logical_bytes = telemetry
                .logical_bytes
                .checked_add(read.logical_bytes)
                .ok_or("routed logical byte count overflow")?;
            telemetry.physical_bytes = telemetry
                .physical_bytes
                .checked_add(read.physical_bytes)
                .ok_or("routed physical byte count overflow")?;
            if index % 2 == 1 {
                if let Some((offset, scale)) = buffer
                    .iter()
                    .copied()
                    .enumerate()
                    .find(|(_, scale)| *scale == 0 || *scale == u8::MAX)
                {
                    return Err(ExpertError::Tensor(format!(
                        "{name}[{offset}] has invalid E8M0 byte {scale}"
                    ))
                    .into());
                }
            }
        }
        Ok(telemetry)
    }

    pub fn advise_dontneed(&self, shard: &ShardFile) -> Result<(), Box<dyn Error>> {
        for (name, &bytes) in self.names.iter().zip(&self.bytes) {
            shard.advise_range_dontneed(name, 0, bytes)?;
        }
        Ok(())
    }
}

impl RoutedExpert {
    pub fn load(shard: &ShardFile, expert_id: u16) -> Result<Self, Box<dyn Error>> {
        Self::load_layer(shard, 0, expert_id)
    }

    pub fn load_layer(
        shard: &ShardFile,
        layer: usize,
        expert_id: u16,
    ) -> Result<Self, Box<dyn Error>> {
        Self::load_layer_for(&PRO, shard, layer, expert_id)
    }

    pub fn load_layer_for(
        spec: &'static V4Spec,
        shard: &ShardFile,
        layer: usize,
        expert_id: u16,
    ) -> Result<Self, Box<dyn Error>> {
        let prefix = routed_expert_prefix_for(spec, layer, expert_id)?;
        Self::load_with_prefix(spec, shard, &prefix)
    }

    /// Loads one official MTP expert without extending the production layer ABI.
    pub fn load_mtp(shard: &ShardFile, expert_id: u16) -> Result<Self, Box<dyn Error>> {
        let prefix = mtp_routed_expert_prefix(expert_id)?;
        Self::load_with_prefix(&PRO, shard, &prefix)
    }

    pub fn load_dspark(
        shard: &ShardFile,
        stage: usize,
        expert_id: u16,
    ) -> Result<Self, Box<dyn Error>> {
        let prefix = dspark_routed_expert_prefix(stage, expert_id)?;
        Self::load_with_prefix(&FLASH_0731, shard, &prefix)
    }

    fn load_with_prefix(
        spec: &'static V4Spec,
        shard: &ShardFile,
        prefix: &str,
    ) -> Result<Self, Box<dyn Error>> {
        let w1 = read_fp4(
            shard,
            &format!("{prefix}.w1"),
            spec.moe_intermediate,
            spec.hidden,
        )?;
        let w1_scale = read_scale(
            shard,
            &format!("{prefix}.w1"),
            spec.moe_intermediate,
            spec.hidden,
        )?;
        let w2 = read_fp4(
            shard,
            &format!("{prefix}.w2"),
            spec.hidden,
            spec.moe_intermediate,
        )?;
        let w2_scale = read_scale(
            shard,
            &format!("{prefix}.w2"),
            spec.hidden,
            spec.moe_intermediate,
        )?;
        let w3 = read_fp4(
            shard,
            &format!("{prefix}.w3"),
            spec.moe_intermediate,
            spec.hidden,
        )?;
        let w3_scale = read_scale(
            shard,
            &format!("{prefix}.w3"),
            spec.moe_intermediate,
            spec.hidden,
        )?;
        Ok(Self {
            spec,
            w1: w1.bytes,
            w1_scale: w1_scale.bytes,
            w2: w2.bytes,
            w2_scale: w2_scale.bytes,
            w3: w3.bytes,
            w3_scale: w3_scale.bytes,
            alloc_ms: w1.alloc_ms
                + w1_scale.alloc_ms
                + w2.alloc_ms
                + w2_scale.alloc_ms
                + w3.alloc_ms
                + w3_scale.alloc_ms,
            read_ms: w1.read_ms
                + w1_scale.read_ms
                + w2.read_ms
                + w2_scale.read_ms
                + w3.read_ms
                + w3_scale.read_ms,
        })
    }

    pub fn h2d_bytes(&self) -> usize {
        self.w1.len()
            + self.w1_scale.len()
            + self.w2.len()
            + self.w2_scale.len()
            + self.w3.len()
            + self.w3_scale.len()
    }
}

fn routed_expert_prefix(layer: usize, expert_id: u16) -> Result<String, ExpertError> {
    routed_expert_prefix_for(&PRO, layer, expert_id)
}

fn routed_expert_prefix_for(
    spec: &V4Spec,
    layer: usize,
    expert_id: u16,
) -> Result<String, ExpertError> {
    if layer >= spec.layers {
        return Err(ExpertError::Tensor(format!(
            "invalid {} layer {layer}; expected 0..{}",
            spec.name, spec.layers
        )));
    }
    if expert_id as usize >= spec.routed_experts {
        return Err(ExpertError::Tensor(format!(
            "invalid routed expert id {expert_id}"
        )));
    }
    Ok(format!("layers.{layer}.ffn.experts.{expert_id}"))
}

fn mtp_routed_expert_prefix(expert_id: u16) -> Result<String, ExpertError> {
    if expert_id as usize >= crate::v4_router::V4_ROUTED_EXPERTS {
        return Err(ExpertError::Tensor(format!(
            "invalid routed expert id {expert_id}"
        )));
    }
    Ok(format!("mtp.0.ffn.experts.{expert_id}"))
}

fn dspark_routed_expert_prefix(stage: usize, expert_id: u16) -> Result<String, ExpertError> {
    if stage >= 3 {
        return Err(ExpertError::Tensor(format!(
            "invalid Flash-0731 DSpark stage {stage}"
        )));
    }
    if expert_id as usize >= FLASH_0731.routed_experts {
        return Err(ExpertError::Tensor(format!(
            "invalid routed expert id {expert_id}"
        )));
    }
    Ok(format!("mtp.{stage}.ffn.experts.{expert_id}"))
}

struct ReadBuffer {
    bytes: Vec<u8>,
    alloc_ms: f64,
    read_ms: f64,
}

fn read_fp4(
    shard: &ShardFile,
    prefix: &str,
    out_features: usize,
    in_features: usize,
) -> Result<ReadBuffer, Box<dyn Error>> {
    let name = format!("{prefix}.weight");
    let info = shard.tensor(&name)?;
    let expected_shape = [out_features as u64, (in_features / 2) as u64];
    if info.dtype != TensorDtype::Fp4E2M1x2 || info.shape != expected_shape {
        return Err(ExpertError::Tensor(format!("{name} must be FP4 {expected_shape:?}")).into());
    }
    let alloc_started = std::time::Instant::now();
    let mut bytes = vec![0; out_features * in_features / 2];
    let alloc_ms = alloc_started.elapsed().as_secs_f64() * 1_000.0;
    let read_started = std::time::Instant::now();
    shard.read_at(&name, 0, &mut bytes)?;
    Ok(ReadBuffer {
        bytes,
        alloc_ms,
        read_ms: read_started.elapsed().as_secs_f64() * 1_000.0,
    })
}

fn read_scale(
    shard: &ShardFile,
    prefix: &str,
    out_features: usize,
    in_features: usize,
) -> Result<ReadBuffer, Box<dyn Error>> {
    let name = format!("{prefix}.scale");
    let info = shard.tensor(&name)?;
    let expected_shape = [out_features as u64, (in_features / WEIGHT_BLOCK) as u64];
    if info.dtype != TensorDtype::F8E8M0 || info.shape != expected_shape {
        return Err(ExpertError::Tensor(format!("{name} must be E8M0 {expected_shape:?}")).into());
    }
    let alloc_started = std::time::Instant::now();
    let mut bytes = vec![0; out_features * in_features / WEIGHT_BLOCK];
    let alloc_ms = alloc_started.elapsed().as_secs_f64() * 1_000.0;
    let read_started = std::time::Instant::now();
    shard.read_at(&name, 0, &mut bytes)?;
    let read_ms = read_started.elapsed().as_secs_f64() * 1_000.0;
    if let Some((index, scale)) = bytes
        .iter()
        .copied()
        .enumerate()
        .find(|(_, scale)| *scale == 0 || *scale == u8::MAX)
    {
        return Err(
            ExpertError::Tensor(format!("{name}[{index}] has invalid E8M0 byte {scale}")).into(),
        );
    }
    Ok(ReadBuffer {
        bytes,
        alloc_ms,
        read_ms,
    })
}

pub fn validate_input(input: &[u16], route_weight: f32) -> Result<usize, ExpertError> {
    validate_input_for(&PRO, input, route_weight)
}

pub fn validate_input_for(
    spec: &V4Spec,
    input: &[u16],
    route_weight: f32,
) -> Result<usize, ExpertError> {
    if !route_weight.is_finite() {
        return Err(ExpertError::RouteWeight(route_weight));
    }
    if input.len() % spec.hidden != 0 {
        return Err(ExpertError::InputLength {
            expected: spec.hidden,
            actual: input.len(),
        });
    }
    let rows = input.len() / spec.hidden;
    if rows == 0 || rows > V4_MAX_EXPERT_ROWS {
        return Err(ExpertError::Rows { actual: rows });
    }
    if let Some((index, _)) = input
        .iter()
        .enumerate()
        .find(|(_, bits)| !bf16_to_f32(**bits).is_finite())
    {
        return Err(ExpertError::NonFiniteInput { index });
    }
    Ok(rows)
}

/// Matches the pinned model's `act_quant(..., 128, "ue8m0")` value boundary:
/// BF16 input -> E8M0 power-of-two scale -> E4M3FN -> dequantized FP32.
pub fn quantize_activation(input: &[u16]) -> Vec<f32> {
    let mut output = vec![0.0; input.len()];
    for (source, destination) in input
        .chunks_exact(ACT_BLOCK)
        .zip(output.chunks_exact_mut(ACT_BLOCK))
    {
        let scale = source
            .iter()
            .map(|bits| bf16_to_f32(*bits).abs())
            .fold(0.0_f32, f32::max)
            .max(1e-4);
        let scale = pow2_ceil(scale / 448.0);
        for (source, destination) in source.iter().zip(destination) {
            *destination = quantize_e4m3fn(bf16_to_f32(*source) / scale) * scale;
        }
    }
    output
}

pub fn forward_cpu(
    weights: &RoutedExpert,
    input: &[u16],
    route_weight: f32,
) -> Result<Vec<f32>, ExpertError> {
    let rows = validate_input_for(weights.spec, input, route_weight)?;
    let input_quantized = quantize_activation(input);
    let mut gate = matvec_fp4(
        &weights.w1,
        &weights.w1_scale,
        &input_quantized,
        rows,
        weights.spec.moe_intermediate,
        weights.spec.hidden,
    );
    let up = matvec_fp4(
        &weights.w3,
        &weights.w3_scale,
        &input_quantized,
        rows,
        weights.spec.moe_intermediate,
        weights.spec.hidden,
    );
    for (gate, up) in gate.iter_mut().zip(up) {
        let gate_value = bf16_to_f32(*gate);
        *gate = f32_to_bf16_rne(
            (gate_value / (1.0 + (-gate_value).exp())) * bf16_to_f32(up) * route_weight,
        );
    }
    let gate_quantized = quantize_activation(&gate);
    let output = matvec_fp4(
        &weights.w2,
        &weights.w2_scale,
        &gate_quantized,
        rows,
        weights.spec.hidden,
        weights.spec.moe_intermediate,
    );
    Ok(output.into_iter().map(bf16_to_f32).collect())
}

/// Official V4 MoE accumulation: routed expert ID ascending, then shared.
/// Expert execution may finish in any order; only this reduction order is semantic.
pub fn reduce_moe_canonical(
    routed: &[(u16, &[f32])],
    shared: &[f32],
) -> Result<Vec<u16>, ExpertError> {
    reduce_moe_canonical_for(&PRO, routed, shared)
}

pub fn reduce_moe_canonical_for(
    spec: &V4Spec,
    routed: &[(u16, &[f32])],
    shared: &[f32],
) -> Result<Vec<u16>, ExpertError> {
    reduce_moe_canonical_inner(spec, routed, shared, None)
}

/// Same canonical reducer as production, with the routed-only FP32 state
/// retained for the Flash L0 parity harness before the shared expert is added.
pub fn reduce_moe_canonical_with_routed_sum_for(
    spec: &V4Spec,
    routed: &[(u16, &[f32])],
    shared: &[f32],
) -> Result<(Vec<u16>, Vec<f32>), ExpertError> {
    let mut routed_sum = Vec::with_capacity(shared.len());
    let output = reduce_moe_canonical_inner(spec, routed, shared, Some(&mut routed_sum))?;
    Ok((output, routed_sum))
}

fn reduce_moe_canonical_inner(
    spec: &V4Spec,
    routed: &[(u16, &[f32])],
    shared: &[f32],
    mut routed_sum: Option<&mut Vec<f32>>,
) -> Result<Vec<u16>, ExpertError> {
    if routed.len() != spec.experts_per_token {
        return Err(ExpertError::ContributionCount {
            actual: routed.len(),
        });
    }
    if shared.len() % spec.hidden != 0 {
        return Err(ExpertError::ContributionLength {
            actual: shared.len(),
        });
    }
    let rows = shared.len() / spec.hidden;
    if rows == 0
        || rows > V4_MAX_EXPERT_ROWS
        || routed
            .iter()
            .any(|(_, values)| values.len() != shared.len())
    {
        return Err(ExpertError::ContributionLength {
            actual: shared.len(),
        });
    }
    if let Some((index, _)) = routed
        .iter()
        .flat_map(|(_, values)| values.iter())
        .chain(shared)
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(ExpertError::NonFiniteInput { index });
    }

    let mut canonical = routed.to_vec();
    canonical.sort_unstable_by_key(|(expert, _)| *expert);
    for pair in canonical.windows(2) {
        if pair[0].0 == pair[1].0 {
            return Err(ExpertError::DuplicateExpert(pair[0].0));
        }
    }
    if let Some(&(expert, _)) = canonical
        .iter()
        .find(|(expert, _)| *expert as usize >= spec.routed_experts)
    {
        return Err(ExpertError::Tensor(format!(
            "invalid routed expert id {expert}"
        )));
    }

    let mut output = Vec::with_capacity(shared.len());
    for lane in 0..shared.len() {
        let mut sum = 0.0_f32;
        for (_, values) in &canonical {
            sum += values[lane];
        }
        if let Some(routed_sum) = routed_sum.as_deref_mut() {
            routed_sum.push(sum);
        }
        sum += shared[lane];
        if !sum.is_finite() {
            return Err(ExpertError::NonFiniteInput { index: lane });
        }
        output.push(f32_to_bf16_rne(sum));
    }
    Ok(output)
}

fn matvec_fp4(
    packed: &[u8],
    scales: &[u8],
    input: &[f32],
    rows: usize,
    out_features: usize,
    in_features: usize,
) -> Vec<u16> {
    let mut output = vec![0; rows * out_features];
    for row in 0..rows {
        for output_row in 0..out_features {
            let mut acc = 0.0;
            for block in 0..in_features / WEIGHT_BLOCK {
                let scale = decode_e8m0(scales[output_row * (in_features / WEIGHT_BLOCK) + block]);
                let packed_start = output_row * (in_features / 2) + block * (WEIGHT_BLOCK / 2);
                for pair in 0..WEIGHT_BLOCK / 2 {
                    let byte = PackedFp4E2M1x2(packed[packed_start + pair]);
                    let input_start = row * in_features + block * WEIGHT_BLOCK + pair * 2;
                    acc += input[input_start] * decode_fp4(byte, 0) * scale;
                    acc += input[input_start + 1] * decode_fp4(byte, 1) * scale;
                }
            }
            output[row * out_features + output_row] = f32_to_bf16_rne(acc);
        }
    }
    output
}

fn pow2_ceil(value: f32) -> f32 {
    let bits = value.to_bits();
    let exponent = ((bits >> 23) & 0xff) as i32 - 127;
    let has_fraction = bits & 0x007f_ffff != 0;
    f32::from_bits(((exponent + has_fraction as i32 + 127) as u32) << 23)
}

/// Values are not usually ties; the explicit even-bit tie rule is retained so
/// the CPU and device scalar reference have the same E4M3FN cast boundary.
pub fn quantize_e4m3fn(value: f32) -> f32 {
    let sign = if value.is_sign_negative() { -1.0 } else { 1.0 };
    let value = value.abs().min(448.0);
    let mut best = 0.0;
    let mut best_bits = 0_u8;
    let mut best_distance = f32::INFINITY;
    for bits in 0_u8..=126 {
        let candidate = decode_e4m3fn(bits);
        let distance = (value - candidate).abs();
        if distance < best_distance
            || (distance == best_distance && bits & 1 == 0 && best_bits & 1 != 0)
        {
            best = candidate;
            best_bits = bits;
            best_distance = distance;
        }
    }
    sign * best
}

fn decode_e4m3fn(bits: u8) -> f32 {
    let exponent = bits >> 3;
    let mantissa = bits & 7;
    if exponent == 0 {
        mantissa as f32 * 0.001953125
    } else {
        (1.0 + mantissa as f32 / 8.0) * f32::from_bits(((exponent as i32 - 7 + 127) as u32) << 23)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PYTORCH_FIXTURE_JSON: &str = include_str!("../fixtures/v4_expert0_pytorch_oracle.json");
    const PYTORCH_FIXTURE_BIN: &[u8] = include_bytes!("../fixtures/v4_expert0_pytorch_oracle.bin");

    #[test]
    fn routed_expert_names_are_layer_parameterized_and_bounded() {
        assert_eq!(
            routed_expert_prefix(0, 0).unwrap(),
            "layers.0.ffn.experts.0"
        );
        assert_eq!(
            routed_expert_prefix(60, 383).unwrap(),
            "layers.60.ffn.experts.383"
        );
        assert!(routed_expert_prefix(61, 0).is_err());
        assert!(routed_expert_prefix(0, 384).is_err());
    }

    #[test]
    fn e4m3fn_quantization_keeps_representable_values_and_validates_rows() {
        assert_eq!(quantize_e4m3fn(1.0), 1.0);
        assert_eq!(quantize_e4m3fn(448.0), 448.0);
        assert_eq!(quantize_e4m3fn(-448.0), -448.0);
        assert_eq!(pow2_ceil(1.0 / 448.0), 0.00390625);
        assert!(matches!(
            validate_input(&[], 1.0),
            Err(ExpertError::Rows { actual: 0 })
        ));
        let bad = vec![f32_to_bf16_rne(f32::NAN); V4_HIDDEN];
        assert!(matches!(
            validate_input(&bad, 1.0),
            Err(ExpertError::NonFiniteInput { .. })
        ));
    }

    #[test]
    fn canonical_moe_reduction_is_order_independent_and_row_invariant() {
        let execution_order = [6_u16, 3, 1, 5, 2, 4];
        let mut contributions = Vec::new();
        for &expert in &execution_order {
            let mut values = vec![0.0_f32; 2 * V4_HIDDEN];
            let row0 = match expert {
                1 => 1.0e8,
                2 => -1.0e8,
                3 => 16.0,
                _ => 0.0,
            };
            values[..V4_HIDDEN].fill(row0);
            values[V4_HIDDEN..].fill(expert as f32 / 8.0);
            contributions.push((expert, values));
        }
        let shared = [vec![1.0_f32; V4_HIDDEN], vec![2.0_f32; V4_HIDDEN]].concat();
        let routed: Vec<_> = contributions
            .iter()
            .map(|(expert, values)| (*expert, values.as_slice()))
            .collect();
        let batched = reduce_moe_canonical(&routed, &shared).unwrap();
        let (traced, routed_sum) =
            reduce_moe_canonical_with_routed_sum_for(&PRO, &routed, &shared).unwrap();
        assert_eq!(
            traced, batched,
            "parity trace changed canonical BF16 output"
        );
        assert_eq!(routed_sum.len(), shared.len());
        assert_eq!(
            routed_sum[0], 16.0,
            "trace is expert-id ordered and shared-free"
        );
        assert_eq!(routed_sum[V4_HIDDEN], 2.625);

        let mut sequential = Vec::new();
        for row in 0..2 {
            let start = row * V4_HIDDEN;
            let routed_row: Vec<_> = contributions
                .iter()
                .map(|(expert, values)| (*expert, &values[start..start + V4_HIDDEN]))
                .collect();
            sequential.extend(
                reduce_moe_canonical(&routed_row, &shared[start..start + V4_HIDDEN]).unwrap(),
            );
        }
        assert_eq!(batched, sequential, "K2 rows changed canonical BF16 bits");
        assert!(
            batched[..V4_HIDDEN]
                .iter()
                .all(|&bits| bits == f32_to_bf16_rne(17.0))
        );
        assert!(
            batched[V4_HIDDEN..]
                .iter()
                .all(|&bits| bits == f32_to_bf16_rne(4.625))
        );

        let mut shared_first = 1.0_f32;
        for expert in 1..=6 {
            shared_first += match expert {
                1 => 1.0e8,
                2 => -1.0e8,
                3 => 16.0,
                _ => 0.0,
            };
        }
        assert_eq!(shared_first, 16.0);
        assert_ne!(
            batched[0],
            f32_to_bf16_rne(shared_first),
            "test vector must distinguish shared-last from shared-first"
        );

        let duplicate = [
            (1, contributions[0].1.as_slice()),
            (1, contributions[1].1.as_slice()),
            (2, contributions[2].1.as_slice()),
            (3, contributions[3].1.as_slice()),
            (4, contributions[4].1.as_slice()),
            (5, contributions[5].1.as_slice()),
        ];
        assert!(matches!(
            reduce_moe_canonical(&duplicate, &shared),
            Err(ExpertError::DuplicateExpert(1))
        ));
    }

    #[test]
    #[ignore = "requires COLIBRI_DSV4_EXPERT_SHARD pointing to the pinned model-00002 shard"]
    fn real_expert0_matches_independent_pytorch_fixture() {
        let metadata: serde_json::Value =
            serde_json::from_str(PYTORCH_FIXTURE_JSON).expect("parse committed PyTorch metadata");
        assert_eq!(
            metadata["schema"].as_str(),
            Some("colibri-v4-expert0-pytorch-oracle-v1")
        );
        assert_eq!(
            metadata["model_revision"].as_str(),
            Some("b5968e9190ef611bbf34a7229255be88a0e937c1")
        );
        assert_eq!(
            metadata["shard"].as_str(),
            Some("model-00002-of-00064.safetensors")
        );
        assert_eq!(metadata["case"]["layer"].as_u64(), Some(0));
        assert_eq!(metadata["case"]["expert"].as_u64(), Some(0));
        assert_eq!(metadata["case"]["rows"].as_u64(), Some(1));
        assert_eq!(metadata["case"]["hidden"].as_u64(), Some(V4_HIDDEN as u64));
        assert_eq!(
            metadata["case"]["intermediate"].as_u64(),
            Some(V4_EXPERT_INTERMEDIATE as u64)
        );
        assert_eq!(metadata["case"]["input_seed"].as_u64(), Some(97));
        assert_eq!(metadata["case"]["route_weight"].as_f64(), Some(0.625));
        assert_eq!(
            metadata["stages"]["output_f32"]["sha256"].as_str(),
            Some("2e36cb58d4aadbe21344b59f46d0de979da4e67d120b258bf4f79b8e4eef8225")
        );
        assert_eq!(
            metadata["binary"]["sha256"].as_str(),
            Some("1da5119527f2e4b1644afc8d8d7d2c3561e7abcdcde6b36b8e6eef5bdea3e950")
        );
        assert_eq!(
            metadata["binary"]["input"]["dtype"].as_str(),
            Some("bfloat16-le")
        );
        assert_eq!(
            metadata["binary"]["input"]["shape"],
            serde_json::json!([1, V4_HIDDEN])
        );
        assert_eq!(
            metadata["binary"]["output"]["dtype"].as_str(),
            Some("float32-le")
        );
        assert_eq!(
            metadata["binary"]["output"]["shape"],
            serde_json::json!([1, V4_HIDDEN])
        );

        assert_eq!(&PYTORCH_FIXTURE_BIN[..8], b"CV4E0P1\n");
        let input_offset = metadata["binary"]["input"]["offset"].as_u64().unwrap() as usize;
        let input_bytes = metadata["binary"]["input"]["bytes"].as_u64().unwrap() as usize;
        let output_offset = metadata["binary"]["output"]["offset"].as_u64().unwrap() as usize;
        let output_bytes = metadata["binary"]["output"]["bytes"].as_u64().unwrap() as usize;
        assert_eq!((input_offset, input_bytes), (8, V4_HIDDEN * 2));
        assert_eq!(
            (output_offset, output_bytes),
            (input_offset + input_bytes, V4_HIDDEN * 4)
        );
        assert_eq!(output_offset + output_bytes, PYTORCH_FIXTURE_BIN.len());

        let input: Vec<u16> = PYTORCH_FIXTURE_BIN[input_offset..output_offset]
            .chunks_exact(2)
            .map(|bytes| u16::from_le_bytes(bytes.try_into().unwrap()))
            .collect();
        let expected: Vec<f32> = PYTORCH_FIXTURE_BIN[output_offset..]
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect();
        assert_eq!(input.len(), V4_HIDDEN);
        assert_eq!(expected.len(), V4_HIDDEN);
        for (index, &bits) in input.iter().enumerate() {
            let value = ((index * 17 + 97) % 113) as f32;
            assert_eq!(bits, f32_to_bf16_rne((value - 56.0) / 64.0));
        }

        let path = std::env::var_os("COLIBRI_DSV4_EXPERT_SHARD")
            .expect("set COLIBRI_DSV4_EXPERT_SHARD to pinned model-00002-of-00064.safetensors");
        let path = std::path::PathBuf::from(path);
        assert_eq!(
            path.file_name().and_then(|name| name.to_str()),
            metadata["shard"].as_str(),
            "real fixture requires the pinned official shard name"
        );
        let shard = ShardFile::open(path).expect("open pinned V4 expert shard");
        let weights = RoutedExpert::load(&shard, 0).expect("load official L0 expert 0 ranges");
        let cpu = forward_cpu(&weights, &input, 0.625).expect("run Rust CPU expert oracle");
        assert_exact_fixture("CPU", &cpu, &expected);

        // ponytail: referencing the GPU module loader makes ordinary `cargo test`
        // require an Oxide artifact anchor. Add direct GPU fixture parity when
        // cuda-oxide can gate that reference to its test target.
    }

    fn assert_exact_fixture(name: &str, actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        let mismatch_count = actual
            .iter()
            .zip(expected)
            .filter(|(actual, expected)| actual.to_bits() != expected.to_bits())
            .count();
        let maximum_error = actual
            .iter()
            .zip(expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0_f32, f32::max);
        eprintln!(
            "PyTorch fixture/Rust {name}: bit mismatches={mismatch_count}, max abs error={maximum_error:.6e}"
        );
        assert_eq!(mismatch_count, 0, "{name} output is not bit-exact");
    }
}
