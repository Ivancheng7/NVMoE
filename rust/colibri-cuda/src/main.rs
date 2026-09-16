//! First Rust-CUDA pipeline primitives corresponding to `pipe_add_n`,
//! `silu_mul`, and `pipe_rows_add` in `c/backend_cuda.cu`.

#![recursion_limit = "256"]
#![feature(core_float_math)]

#[allow(dead_code)] // wired into model dispatch after the current CUDA smoke phase
mod cublas_exact;
mod safetensors;
mod v4_certified;
mod v4_chat;
mod v4_expert;
mod v4_federated;
mod v4_fp4;
mod v4_router;
mod v4_serve;
mod v4_spec;
mod v4_state;
mod v4_tilelang_ref;
mod v4_tokenizer;

use std::collections::{BTreeMap, BTreeSet, HashSet, VecDeque};
use std::error::Error;
use std::io::Write;

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig, PinnedHostBuffer, sys};
use cuda_device::{DisjointSlice, SharedArray, cuda_module, device, kernel, thread, warp};

const CERTIFIED_HEAD_NONFINITE: u32 = 1;
const CERTIFIED_HEAD_SUBNORMAL: u32 = 2;
const CERTIFIED_HEAD_OVERFLOW: u32 = 4;
const CERTIFIED_HEAD_INTERVAL_INVALID: u32 = 8;
const CERTIFIED_HEAD_ALL_CANDIDATES: u32 = 16;
const CERTIFIED_HEAD_RUNTIME_ERROR: u32 = 32;

const CERTIFIED_ARGMAX_CHUNK_ROWS: usize = 1_024;
const CERTIFIED_ARGMAX_DEVICE_TEMP_BYTES: usize = v4_state::HIDDEN * 2
    + CERTIFIED_ARGMAX_CHUNK_ROWS * v4_state::HIDDEN * 2
    + CERTIFIED_ARGMAX_CHUNK_ROWS * (std::mem::size_of::<f32>() * 2 + std::mem::size_of::<u32>());
const CERTIFIED_ARGMAX_HOST_TEMP_BYTES: usize = CERTIFIED_ARGMAX_CHUNK_ROWS * v4_state::HIDDEN * 4
    + v4_state::VOCAB * (std::mem::size_of::<f32>() * 3)
    + v4_state::HIDDEN * (std::mem::size_of::<f32>() * 2 + std::mem::size_of::<u16>() * 2);

// TileLang's four-lane xor reduction is (0 + 2) + (1 + 3), not a left fold.
macro_rules! tilelang_sum4 {
    ($a:expr, $b:expr, $c:expr, $d:expr) => {{
        let ac = $a + $c;
        let bd = $b + $d;
        let ac = unsafe { core::ptr::read_volatile(&ac) };
        let bd = unsafe { core::ptr::read_volatile(&bd) };
        ac + bd
    }};
}

// Business modules extracted from this file. Keep them after the
// tilelang_sum4 macro so its textual scope still covers them.
mod artifacts;
mod attention_gates;
mod device_attention;
mod env_flags;
mod expert_slot;
mod final_head;
mod fixed_layer_cache;
mod full61_b1;
mod generate;
mod layer_ops;
mod layer_rows;
mod layer_stationary_prefill;
mod mtp1_l0;
mod qd4;
mod slice_checks;
mod v4_loopback;

// Keep the existing call sites in this file working after the moves.
use artifacts::*;
use attention_gates::*;
use device_attention::*;
use env_flags::*;
use expert_slot::*;
use final_head::*;
use fixed_layer_cache::*;
use full61_b1::*;
use generate::*;
use layer_ops::*;
use layer_rows::*;
use layer_stationary_prefill::*;
use mtp1_l0::*;
use qd4::*;
use slice_checks::*;
use v4_loopback::*;

// DeepSeek-V4-Flash E2M1 nibbles converted exactly to raw E4M3 bytes.
// Packed immediates keep this LUT in registers instead of thread-local memory.
const V4_FP4_E2M1_TO_E4M3_POSITIVE: u64 = 0x4c48_4440_3c38_3000;
const V4_FP4_E2M1_TO_E4M3_NEGATIVE: u64 = 0xccc8_c4c0_bcb8_b080;

#[inline(always)]
const fn v4_fp4_nibble_to_e4m3(nibble: u8) -> u8 {
    let lut = if nibble & 8 == 0 {
        V4_FP4_E2M1_TO_E4M3_POSITIVE
    } else {
        V4_FP4_E2M1_TO_E4M3_NEGATIVE
    };
    (lut >> ((nibble as u64 & 7) * 8)) as u8
}

#[inline(always)]
const fn v4_fp4_pair_to_e4m3x2(packed: u8) -> u16 {
    v4_fp4_nibble_to_e4m3(packed & 0x0f) as u16 | ((v4_fp4_nibble_to_e4m3(packed >> 4) as u16) << 8)
}

#[cuda_module]
mod kernels {
    use super::*;

    // Direct libdevice FFI: candidate-only parity probe, never a model output.
    #[device]
    unsafe extern "C" {
        fn __nv_expf(value: f32) -> f32;
        fn __nv_log1pf(value: f32) -> f32;
        fn __nv_sqrtf(value: f32) -> f32;
        fn __nv_rsqrtf(value: f32) -> f32;
    }

    #[inline(always)]
    fn mhc_committed_rsqrt(value: f32) -> f32 {
        // SAFETY: libdevice `__nv_rsqrtf` is defined for every finite input;
        // the caller supplies RMSNorm's positive epsilon-shifted mean square.
        unsafe { __nv_rsqrtf(value) }
    }

    #[inline(always)]
    fn router_softplus(logit: f32, score_arm: u32) -> f32 {
        if score_arm != 0 {
            return if logit > 20.0 {
                logit
            } else {
                // SAFETY: finite router logits are valid libdevice inputs.
                unsafe { __nv_log1pf(__nv_expf(logit)) }
            };
        }
        if logit > 20.0 {
            logit
        } else if logit > 0.0 {
            logit + libm::log1pf(libm::expf(-logit))
        } else {
            libm::log1pf(libm::expf(logit))
        }
    }

    #[inline(always)]
    fn router_sqrtsoftplus(logit: f32, score_arm: u32) -> f32 {
        let softplus = router_softplus(logit, score_arm);
        if score_arm != 0 {
            router_accurate_sqrt(softplus)
        } else {
            sqrt_newton(softplus)
        }
    }

    /// Mirrors the positive-FP32 sqrt refinement emitted by the pinned
    /// official Flash SASS: rsqrt, two FTZ multiplies, then two FTZ FMAs.
    /// Softplus is mathematically positive; preserving zero/non-finite input
    /// avoids creating a spurious `sqrt.approx` fallback in this score path.
    #[inline(always)]
    fn router_accurate_sqrt(softplus: f32) -> f32 {
        if !(softplus > 0.0) || !softplus.is_finite() {
            return softplus;
        }
        // SAFETY: the guard proves a finite positive libdevice rsqrt input.
        let reciprocal = unsafe { __nv_rsqrtf(softplus) };
        let y = cuda_device::float::mul_rn_ftz_f32(softplus, reciprocal);
        let half = cuda_device::float::mul_rn_ftz_f32(reciprocal, 0.5);
        let error = cuda_device::float::fma_rn_ftz_f32(-y, y, softplus);
        cuda_device::float::fma_rn_ftz_f32(error, half, y)
    }

    fn bf16_to_f32(bits: u16) -> f32 {
        f32::from_bits((bits as u32) << 16)
    }

    fn f32_to_bf16_rne(value: f32) -> u16 {
        let bits = value.to_bits();
        ((bits.wrapping_add(0x7fff + ((bits >> 16) & 1))) >> 16) as u16
    }

    fn decode_fp4(byte: u8, pair: usize) -> f32 {
        let nibble = if pair == 0 { byte & 0x0f } else { byte >> 4 };
        let magnitude = match nibble & 0x07 {
            0 => 0.0,
            1 => 0.5,
            2 => 1.0,
            3 => 1.5,
            4 => 2.0,
            5 => 3.0,
            6 => 4.0,
            _ => 6.0,
        };
        if magnitude == 0.0 || nibble & 0x08 == 0 {
            magnitude
        } else {
            -magnitude
        }
    }

    fn decode_e4m3fn(bits: u8) -> f32 {
        let negative = bits & 0x80 != 0;
        let magnitude = bits & 0x7f;
        if magnitude == 0x7f {
            return f32::NAN;
        }
        let exponent = magnitude >> 3;
        let mantissa = magnitude & 7;
        let value = if exponent == 0 {
            mantissa as f32 * 0.001953125
        } else {
            (1.0 + mantissa as f32 / 8.0)
                * f32::from_bits(((exponent as i32 - 7 + 127) as u32) << 23)
        };
        if negative { -value } else { value }
    }

    fn quantize_e4m3fn_bits(value: f32) -> u8 {
        let sign = if value.to_bits() >> 31 != 0 { 0x80 } else { 0 };
        let value = if value.abs() > 448.0 {
            448.0
        } else {
            value.abs()
        };
        let mut best_bits = 0_u8;
        let mut best_distance = f32::INFINITY;
        let mut bits = 0_u8;
        while bits <= 126 {
            let candidate = decode_e4m3fn(bits);
            let distance = (value - candidate).abs();
            if distance < best_distance
                || (distance == best_distance && bits & 1 == 0 && best_bits & 1 != 0)
            {
                best_bits = bits;
                best_distance = distance;
            }
            bits += 1;
        }
        best_bits | sign
    }

    fn quantize_e4m3fn(value: f32) -> f32 {
        decode_e4m3fn(quantize_e4m3fn_bits(value))
    }

    #[kernel]
    pub fn e4m3fn_to_bf16(input: &[u8], mut output: DisjointSlice<u16>) {
        let index = thread::index_1d().get() as usize;
        if index < input.len() && index < output.len() {
            unsafe {
                *output.as_mut_ptr().add(index) = f32_to_bf16_rne(decode_e4m3fn(input[index]));
            }
        }
    }

    fn decode_e8m0fnu(bits: u8) -> f32 {
        match bits {
            0 => f32::from_bits(1 << 22),
            255 => f32::NAN,
            exponent => f32::from_bits((exponent as u32) << 23),
        }
    }

    fn encode_e8m0_power_of_two(value: f32) -> u8 {
        if value.to_bits() == 1 << 22 {
            0
        } else {
            ((value.to_bits() >> 23) & 0xff) as u8
        }
    }

    fn pow2_ceil(value: f32) -> f32 {
        let bits = value.to_bits();
        let exponent = ((bits >> 23) & 0xff) as i32 - 127;
        let has_fraction = bits & 0x007f_ffff != 0;
        f32::from_bits(((exponent + has_fraction as i32 + 127) as u32) << 23)
    }

    fn e2m1_even(value: f32) -> bool {
        matches!(
            value.abs().to_bits(),
            0 | 0x3f80_0000 | 0x4000_0000 | 0x4080_0000
        )
    }

    fn quantize_e2m1(value: f32) -> f32 {
        const LEVELS: [f32; 15] = [
            -6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        ];
        let mut best = 0.0_f32;
        let mut distance = f32::INFINITY;
        for candidate in LEVELS {
            let candidate_distance = (value - candidate).abs();
            if candidate_distance < distance
                || (candidate_distance == distance && e2m1_even(candidate) && !e2m1_even(best))
            {
                best = candidate;
                distance = candidate_distance;
            }
        }
        best
    }

    fn sqrt_newton(value: f32) -> f32 {
        if value <= 0.0 {
            return 0.0;
        }
        let mut estimate = if value < 1.0 { 1.0 } else { value };
        for _ in 0..16 {
            estimate = 0.5 * (estimate + value / estimate);
        }
        estimate
    }

    fn pytorch_square_sum_bf16(input: &[u16], start: usize, width: usize) -> f32 {
        const LANES: usize = 8;
        const ILP: usize = 4;
        const LEVELS: usize = 4;
        const LEVEL_STEP: usize = 16;
        let groups = width / (LANES * ILP);
        let mut partials = [0.0_f32; LEVELS * ILP * LANES];
        let mut group = 0;

        while group + LEVEL_STEP <= groups {
            for _ in 0..LEVEL_STEP {
                for interleave in 0..ILP {
                    let source = start + (group * ILP + interleave) * LANES;
                    for lane in 0..LANES {
                        let value = bf16_to_f32(input[source + lane]);
                        partials[interleave * LANES + lane] += value * value;
                    }
                }
                group += 1;
            }
            for level in 1..LEVELS {
                for interleave in 0..ILP {
                    for lane in 0..LANES {
                        let destination = (level * ILP + interleave) * LANES + lane;
                        let source = ((level - 1) * ILP + interleave) * LANES + lane;
                        partials[destination] += partials[source];
                        partials[source] = 0.0;
                    }
                }
                if group & (15 << (level * 4)) != 0 {
                    break;
                }
            }
        }
        while group < groups {
            for interleave in 0..ILP {
                let source = start + (group * ILP + interleave) * LANES;
                for lane in 0..LANES {
                    let value = bf16_to_f32(input[source + lane]);
                    partials[interleave * LANES + lane] += value * value;
                }
            }
            group += 1;
        }
        for level in 1..LEVELS {
            for interleave in 0..ILP {
                for lane in 0..LANES {
                    partials[interleave * LANES + lane] +=
                        partials[(level * ILP + interleave) * LANES + lane];
                }
            }
        }
        for interleave in 1..ILP {
            for lane in 0..LANES {
                partials[lane] += partials[interleave * LANES + lane];
            }
        }
        let mut sum = 0.0_f32;
        for column in groups * LANES * ILP..width {
            let value = bf16_to_f32(input[start + column]);
            sum += value * value;
        }
        for lane in 0..LANES {
            sum += partials[lane];
        }
        sum
    }

    /// PyTorch 2.10+cu126's contiguous [1, 4096] RMSNorm square pass.  Keep
    /// this materialized: the official sm86 cubin rounds the BF16->FP32 square
    /// before its separate reduction kernel.
    #[kernel]
    pub fn rmsnorm_flash_square_bf16(input: &[u16], mut squares: DisjointSlice<f32>) {
        let index = thread::index_1d();
        let index_raw = index.get() as usize;
        if index_raw >= input.len() {
            return;
        }
        let Some(square) = squares.get_mut(index) else {
            return;
        };
        let value = bf16_to_f32(input[index_raw]);
        *square = cuda_device::float::mul_rn_f32(value, value);
    }

    /// Fixed Flash mHC reduction for one contiguous [4, 4096] row.  The
    /// square pass above is intentionally separate, matching the pinned
    /// PyTorch 2.10 reduction kernel's globally materialized FP32 input.
    #[kernel]
    pub fn mhc_flash_reduce_rsqrt_16384(
        squares: &[f32],
        mut rsqrt: DisjointSlice<f32>,
        mut debug: DisjointSlice<u32>,
    ) {
        const WIDTH: usize = 16_384;
        const FIELDS: usize = 5;
        static mut REDUCE: SharedArray<f32, 512> = SharedArray::UNINIT;

        let row = thread::blockIdx_x() as usize;
        let tid = thread::threadIdx_x() as usize;
        let start = row * WIDTH;
        if start + WIDTH > squares.len()
            || row >= rsqrt.len()
            || !(debug.is_empty() || debug.len() == rsqrt.len() * FIELDS)
            || tid >= 512
        {
            return;
        }

        let base = start + tid * 4;
        let mut a0 = squares[base];
        let mut a1 = squares[base + 1];
        let mut a2 = squares[base + 2];
        let mut a3 = squares[base + 3];
        for k in 1..8 {
            let offset = 2_048 * k;
            a0 = cuda_device::float::add_rn_f32(a0, squares[base + offset]);
            a1 = cuda_device::float::add_rn_f32(a1, squares[base + offset + 1]);
            a2 = cuda_device::float::add_rn_f32(a2, squares[base + offset + 2]);
            a3 = cuda_device::float::add_rn_f32(a3, squares[base + offset + 3]);
        }
        let a01 = cuda_device::float::add_rn_f32(a0, a1);
        let a012 = cuda_device::float::add_rn_f32(a01, a2);
        unsafe { REDUCE[tid] = cuda_device::float::add_rn_f32(a012, a3) };
        thread::sync_threads();
        for offset in [256_usize, 128, 64, 32] {
            if tid < offset {
                unsafe {
                    REDUCE[tid] = cuda_device::float::add_rn_f32(REDUCE[tid], REDUCE[tid + offset]);
                }
            }
            thread::sync_threads();
        }

        if tid < 32 {
            let mut sum = unsafe { REDUCE[tid] };
            sum = cuda_device::float::add_rn_f32(sum, warp::shuffle_down_f32(sum, 16));
            sum = cuda_device::float::add_rn_f32(sum, warp::shuffle_down_f32(sum, 8));
            sum = cuda_device::float::add_rn_f32(sum, warp::shuffle_down_f32(sum, 4));
            sum = cuda_device::float::add_rn_f32(sum, warp::shuffle_down_f32(sum, 2));
            sum = cuda_device::float::add_rn_f32(sum, warp::shuffle_down_f32(sum, 1));
            if tid == 0 {
                let mean = cuda_device::float::mul_rn_f32(sum, 1.0 / WIDTH as f32);
                let mean_eps = cuda_device::float::add_rn_f32(mean, 1e-6);
                let current_newton = 1.0 / sqrt_newton(mean_eps);
                let libdevice_sqrt = 1.0 / core::f32::math::sqrt(mean_eps);
                let inv = mhc_committed_rsqrt(mean_eps);
                unsafe {
                    *rsqrt.as_mut_ptr().add(row) = inv;
                    if !debug.is_empty() {
                        let base = row * FIELDS;
                        *debug.as_mut_ptr().add(base) = sum.to_bits();
                        *debug.as_mut_ptr().add(base + 1) = mean_eps.to_bits();
                        *debug.as_mut_ptr().add(base + 2) = current_newton.to_bits();
                        *debug.as_mut_ptr().add(base + 3) = libdevice_sqrt.to_bits();
                        *debug.as_mut_ptr().add(base + 4) = inv.to_bits();
                    }
                }
            }
        }
    }

    /// Replay-only final HC head: the gate's four already-normalized raw dots
    /// are transformed with the official affine/sigmoid+epsilon expression,
    /// then collapsed in source order with explicit RN operations.  It is not
    /// wired into Flash production dispatch.
    #[kernel]
    pub fn flash_final_head_sigmoid_collapse4(
        input: &[u16],
        raw_dot: &[f32],
        rsqrt: &[f32],
        hc_scale: &[f32],
        hc_base: &[f32],
        mut pre: DisjointSlice<f32>,
        mut collapsed: DisjointSlice<u16>,
    ) {
        const HC: usize = 4;
        const HIDDEN: usize = 4_096;
        let column = thread::index_1d().get() as usize;
        if column >= HIDDEN
            || input.len() != HC * HIDDEN
            || raw_dot.len() != HC
            || rsqrt.len() != 1
            || hc_scale.len() != 1
            || hc_base.len() != HC
            || pre.len() != HC
            || collapsed.len() != HIDDEN
        {
            return;
        }
        let mut gate = [0.0_f32; HC];
        for copy in 0..HC {
            let scaled = cuda_device::float::mul_rn_f32(raw_dot[copy], rsqrt[0]);
            let affine = cuda_device::float::add_rn_f32(
                cuda_device::float::mul_rn_f32(scaled, hc_scale[0]),
                hc_base[copy],
            );
            // SAFETY: finite replay inputs and locked weights make expf's
            // domain valid; this is the pinned official sigmoid expression.
            gate[copy] = 1.0 / (1.0 + unsafe { __nv_expf(-affine) }) + 1e-6;
            if column < HC {
                unsafe { *pre.as_mut_ptr().add(column) = gate[column] };
            }
        }
        let product = |copy: usize| {
            cuda_device::float::mul_rn_f32(gate[copy], bf16_to_f32(input[copy * HIDDEN + column]))
        };
        let p0 = product(0);
        let p1 = product(1);
        let p2 = product(2);
        let p3 = product(3);
        let p01 = cuda_device::float::add_rn_f32(p0, p1);
        let p012 = cuda_device::float::add_rn_f32(p01, p2);
        let value = cuda_device::float::add_rn_f32(p012, p3);
        unsafe { *collapsed.as_mut_ptr().add(column) = f32_to_bf16_rne(value) };
    }

    /// Fixed Flash learned-router reduction recovered from the pinned PyTorch
    /// 2.10+cu126 sm86 cubin.  One 512-thread block owns one contiguous row.
    #[kernel]
    pub fn rmsnorm_flash_learned_4096_bf16(
        input: &[u16],
        norm: &[u16],
        squares: &[f32],
        mut output: DisjointSlice<u16>,
        mut trace: DisjointSlice<f32>,
    ) {
        const WIDTH: usize = 4_096;
        static mut REDUCE: SharedArray<f32, 512> = SharedArray::UNINIT;

        let row = thread::blockIdx_x() as usize;
        let tid = thread::threadIdx_x() as usize;
        let start = row * WIDTH;
        if norm.len() != WIDTH
            || start + WIDTH > input.len()
            || start + WIDTH > squares.len()
            || start + WIDTH > output.len()
            || tid >= 512
        {
            return;
        }

        let base = start + tid * 4;
        let upper = base + WIDTH / 2;
        let a0 = cuda_device::float::add_rn_f32(squares[base], squares[upper]);
        let a1 = cuda_device::float::add_rn_f32(squares[base + 1], squares[upper + 1]);
        let a2 = cuda_device::float::add_rn_f32(squares[base + 2], squares[upper + 2]);
        let a3 = cuda_device::float::add_rn_f32(squares[base + 3], squares[upper + 3]);
        let p01 = cuda_device::float::add_rn_f32(a0, a1);
        let p012 = cuda_device::float::add_rn_f32(p01, a2);
        let partial = cuda_device::float::add_rn_f32(p012, a3);
        unsafe { REDUCE[tid] = partial };
        thread::sync_threads();
        for offset in [256_usize, 128, 64, 32] {
            if tid < offset {
                unsafe {
                    REDUCE[tid] = cuda_device::float::add_rn_f32(REDUCE[tid], REDUCE[tid + offset]);
                }
            }
            thread::sync_threads();
        }

        if tid < 32 {
            let mut sum = unsafe { REDUCE[tid] };
            sum = cuda_device::float::add_rn_f32(sum, warp::shuffle_down_f32(sum, 16));
            sum = cuda_device::float::add_rn_f32(sum, warp::shuffle_down_f32(sum, 8));
            sum = cuda_device::float::add_rn_f32(sum, warp::shuffle_down_f32(sum, 4));
            sum = cuda_device::float::add_rn_f32(sum, warp::shuffle_down_f32(sum, 2));
            sum = cuda_device::float::add_rn_f32(sum, warp::shuffle_down_f32(sum, 1));
            if tid == 0 {
                let mean = cuda_device::float::mul_rn_f32(sum, 1.0 / WIDTH as f32);
                let inv = mhc_committed_rsqrt(cuda_device::float::add_rn_f32(mean, 1e-6));
                unsafe { REDUCE[0] = inv };
                if trace.len().is_multiple_of(3) && row < trace.len() / 3 {
                    let rows = trace.len() / 3;
                    unsafe {
                        *trace.as_mut_ptr().add(row) = sum;
                        *trace.as_mut_ptr().add(rows + row) = mean;
                        *trace.as_mut_ptr().add(rows * 2 + row) = inv;
                    }
                }
            }
        }
        thread::sync_threads();
        let inv = unsafe { REDUCE[0] };
        for column in (tid..WIDTH).step_by(512) {
            let normalized = f32::from_bits((input[start + column] as u32) << 16) * inv;
            let value = normalized * f32::from_bits((norm[column] as u32) << 16);
            let bits = value.to_bits();
            unsafe {
                *output.as_mut_ptr().add(start + column) =
                    ((bits.wrapping_add(0x7fff + ((bits >> 16) & 1))) >> 16) as u16;
            }
        }
    }

    #[kernel]
    pub fn quantize_v4_activation(input: &[u16], mut output: DisjointSlice<f32>) {
        const BLOCK: usize = 128;
        let index = thread::index_1d();
        let output_index = index.get() as usize;
        let Some(destination) = output.get_mut(index) else {
            return;
        };
        if output_index >= input.len() {
            return;
        }
        let start = output_index / BLOCK * BLOCK;
        let mut amax = 0.0_f32;
        for offset in 0..BLOCK {
            let value = bf16_to_f32(input[start + offset]).abs();
            if value > amax {
                amax = value;
            }
        }
        let scale = pow2_ceil(if amax > 1e-4 { amax } else { 1e-4 } / 448.0);
        *destination = quantize_e4m3fn(bf16_to_f32(input[output_index]) / scale) * scale;
    }

    #[kernel]
    pub fn quantize_v4_activation_raw(
        input: &[u16],
        mut output: DisjointSlice<u8>,
        mut scales: DisjointSlice<u8>,
    ) {
        const BLOCK: usize = 128;
        let group = thread::index_1d().get() as usize;
        if input.is_empty()
            || input.len() % BLOCK != 0
            || output.len() != input.len()
            || scales.len() != input.len() / BLOCK
            || group >= scales.len()
        {
            return;
        }
        let start = group * BLOCK;
        let mut amax = 0.0_f32;
        for offset in 0..BLOCK {
            amax = amax.max(bf16_to_f32(input[start + offset]).abs());
        }
        let scale = pow2_ceil(amax.max(1e-4) / 448.0);
        // SAFETY: one thread owns a complete 128-value activation group.
        unsafe {
            *scales.as_mut_ptr().add(group) = encode_e8m0_power_of_two(scale);
            for offset in 0..BLOCK {
                *output.as_mut_ptr().add(start + offset) =
                    quantize_e4m3fn_bits(bf16_to_f32(input[start + offset]) / scale);
            }
        }
    }

    #[kernel]
    pub fn quantize_v4_fp8_bf16(
        input: &[u16],
        mut output: DisjointSlice<u8>,
        mut scales: DisjointSlice<u8>,
        mut dequantized: DisjointSlice<u16>,
        row_width: u32,
        quantized_width: u32,
        block_size: u32,
    ) {
        let row_width = row_width as usize;
        let quantized_width = quantized_width as usize;
        let block_size = block_size as usize;
        let group = thread::index_1d().get() as usize;
        if row_width == 0
            || quantized_width == 0
            || block_size == 0
            || quantized_width > row_width
            || quantized_width % block_size != 0
            || input.len() % row_width != 0
            || group >= scales.len()
        {
            return;
        }
        let rows = input.len() / row_width;
        let groups_per_row = quantized_width / block_size;
        if scales.len() != rows * groups_per_row
            || (!output.is_empty() && output.len() != rows * quantized_width)
            || (!dequantized.is_empty() && dequantized.len() != input.len())
        {
            return;
        }
        let row = group / groups_per_row;
        let group_in_row = group % groups_per_row;
        let input_start = row * row_width + group_in_row * block_size;
        let output_start = group * block_size;
        let mut amax = 0.0_f32;
        for column in 0..block_size {
            amax = amax.max(bf16_to_f32(input[input_start + column]).abs());
        }
        let scale = pow2_ceil(amax.max(1e-4) / 448.0);
        // SAFETY: one thread owns one complete disjoint quantization group.
        unsafe {
            *scales.as_mut_ptr().add(group) = encode_e8m0_power_of_two(scale);
            for column in 0..block_size {
                let bits = quantize_e4m3fn_bits(bf16_to_f32(input[input_start + column]) / scale);
                if !output.is_empty() {
                    *output.as_mut_ptr().add(output_start + column) = bits;
                }
                if !dequantized.is_empty() {
                    *dequantized.as_mut_ptr().add(input_start + column) =
                        f32_to_bf16_rne(decode_e4m3fn(bits) * scale);
                }
            }
        }
    }

    /// Correctness-first FP8 GEMV: one warp owns one output element and
    /// accumulates official 128-K scaled partials in K order.
    #[kernel]
    pub fn matvec_v4_fp8(
        input: &[u8],
        input_scales: &[u8],
        weight: &[u8],
        weight_scales: &[u8],
        mut output_f32: DisjointSlice<f32>,
        mut output_bf16: DisjointSlice<u16>,
        in_features: u32,
        out_features: u32,
        strict_k_order: bool,
    ) {
        const BLOCK: usize = 128;
        let thread_index = thread::index_1d().get() as usize;
        let output_index = thread_index / 32;
        let lane = warp::lane_id() as usize;
        let in_features = in_features as usize;
        let out_features = out_features as usize;
        if in_features == 0 || in_features % BLOCK != 0 || out_features == 0 {
            return;
        }
        let blocks = in_features / BLOCK;
        let rows = input.len() / in_features;
        if input.len() != rows * in_features
            || input_scales.len() != rows * blocks
            || weight.len() != out_features * in_features
            || weight_scales.len() != ((out_features + BLOCK - 1) / BLOCK) * blocks
            || output_f32.len() != rows * out_features
            || output_bf16.len() != rows * out_features
            || output_index >= output_f32.len()
        {
            return;
        }
        let row = output_index / out_features;
        let output_row = output_index % out_features;
        let mut accumulated = 0.0_f32;
        if strict_k_order && lane == 0 {
            // ponytail: one lane preserves the official K order for exact BF16;
            // restore a parallel reduction only when it also passes this gate.
            for block in 0..blocks {
                let mut partial = 0.0_f32;
                let input_start = row * in_features + block * BLOCK;
                let weight_start = output_row * in_features + block * BLOCK;
                for column in 0..BLOCK {
                    partial += decode_e4m3fn(input[input_start + column])
                        * decode_e4m3fn(weight[weight_start + column]);
                }
                let scale = decode_e8m0fnu(input_scales[row * blocks + block])
                    * decode_e8m0fnu(weight_scales[(output_row / BLOCK) * blocks + block]);
                accumulated += partial * scale;
            }
        } else if !strict_k_order {
            for block in 0..blocks {
                let mut partial = 0.0_f32;
                let input_start = row * in_features + block * BLOCK;
                let weight_start = output_row * in_features + block * BLOCK;
                let mut column = lane;
                while column < BLOCK {
                    partial += decode_e4m3fn(input[input_start + column])
                        * decode_e4m3fn(weight[weight_start + column]);
                    column += 32;
                }
                partial += warp::shuffle_xor_f32(partial, 16);
                partial += warp::shuffle_xor_f32(partial, 8);
                partial += warp::shuffle_xor_f32(partial, 4);
                partial += warp::shuffle_xor_f32(partial, 2);
                partial += warp::shuffle_xor_f32(partial, 1);
                if lane == 0 {
                    let scale = decode_e8m0fnu(input_scales[row * blocks + block])
                        * decode_e8m0fnu(weight_scales[(output_row / BLOCK) * blocks + block]);
                    accumulated += partial * scale;
                }
            }
        }
        if lane == 0 {
            // SAFETY: lane zero uniquely owns this checked output element.
            unsafe {
                *output_f32.as_mut_ptr().add(output_index) = accumulated;
                *output_bf16.as_mut_ptr().add(output_index) = f32_to_bf16_rne(accumulated);
            }
        }
    }

    /// Streamed BF16 output-head GEMV: one warp owns one vocabulary row.
    #[kernel]
    pub fn matvec_bf16_streamed(
        input: &[u16],
        weight: &[u16],
        mut output: DisjointSlice<f32>,
        mut sumabs: DisjointSlice<f32>,
        mut status: DisjointSlice<u32>,
        in_features: u32,
        out_features: u32,
    ) {
        let lane = warp::lane_id() as usize;
        let output_row = thread::index_1d().get() as usize / 32;
        let in_features = in_features as usize;
        let out_features = out_features as usize;
        if input.len() != in_features
            || weight.len() < out_features * in_features
            || output.len() < out_features
            || sumabs.len() < out_features
            || status.len() < out_features
            || output_row >= out_features
        {
            return;
        }
        let mut sum = 0.0_f32;
        let mut absolute_sum = 0.0_f32;
        let mut row_status = 0_i32;
        let mut column = lane;
        while column < in_features {
            let input_bits = input[column];
            let weight_bits = weight[output_row * in_features + column];
            let input_exponent = (input_bits >> 7) & 0xff;
            let weight_exponent = (weight_bits >> 7) & 0xff;
            if input_exponent == 0xff || weight_exponent == 0xff {
                row_status |= CERTIFIED_HEAD_NONFINITE as i32;
            }
            if (input_exponent == 0 && input_bits & 0x7f != 0)
                || (weight_exponent == 0 && weight_bits & 0x7f != 0)
            {
                row_status |= CERTIFIED_HEAD_SUBNORMAL as i32;
            }
            let product = bf16_to_f32(input_bits) * bf16_to_f32(weight_bits);
            sum += product;
            absolute_sum += product.abs();
            let product_bits = product.to_bits();
            let sum_bits = sum.to_bits();
            if product_bits & 0x7f80_0000 == 0x7f80_0000 || sum_bits & 0x7f80_0000 == 0x7f80_0000 {
                row_status |= CERTIFIED_HEAD_NONFINITE as i32;
                row_status |= CERTIFIED_HEAD_OVERFLOW as i32;
            }
            if (product_bits & 0x7f80_0000 == 0 && product_bits & 0x007f_ffff != 0)
                || (sum_bits & 0x7f80_0000 == 0 && sum_bits & 0x007f_ffff != 0)
            {
                row_status |= CERTIFIED_HEAD_SUBNORMAL as i32;
            }
            column += 32;
        }
        sum += warp::shuffle_xor_f32(sum, 16);
        sum += warp::shuffle_xor_f32(sum, 8);
        sum += warp::shuffle_xor_f32(sum, 4);
        sum += warp::shuffle_xor_f32(sum, 2);
        sum += warp::shuffle_xor_f32(sum, 1);
        absolute_sum += warp::shuffle_xor_f32(absolute_sum, 16);
        absolute_sum += warp::shuffle_xor_f32(absolute_sum, 8);
        absolute_sum += warp::shuffle_xor_f32(absolute_sum, 4);
        absolute_sum += warp::shuffle_xor_f32(absolute_sum, 2);
        absolute_sum += warp::shuffle_xor_f32(absolute_sum, 1);
        row_status |= warp::shuffle_xor(row_status as u32, 16) as i32;
        row_status |= warp::shuffle_xor(row_status as u32, 8) as i32;
        row_status |= warp::shuffle_xor(row_status as u32, 4) as i32;
        row_status |= warp::shuffle_xor(row_status as u32, 2) as i32;
        row_status |= warp::shuffle_xor(row_status as u32, 1) as i32;
        if lane == 0 {
            let sum_bits = sum.to_bits();
            let sumabs_bits = absolute_sum.to_bits();
            if sum_bits & 0x7f80_0000 == 0x7f80_0000 || sumabs_bits & 0x7f80_0000 == 0x7f80_0000 {
                row_status |= CERTIFIED_HEAD_NONFINITE as i32;
                row_status |= CERTIFIED_HEAD_OVERFLOW as i32;
            }
            if (sum_bits & 0x7f80_0000 == 0 && sum_bits & 0x007f_ffff != 0)
                || (sumabs_bits & 0x7f80_0000 == 0 && sumabs_bits & 0x007f_ffff != 0)
            {
                row_status |= CERTIFIED_HEAD_SUBNORMAL as i32;
            }
            // SAFETY: lane zero uniquely owns this checked output row.
            unsafe {
                *output.as_mut_ptr().add(output_row) = sum;
                *sumabs.as_mut_ptr().add(output_row) = absolute_sum;
                *status.as_mut_ptr().add(output_row) = row_status as u32;
            }
        }
    }

    /// Official V4 FP8 GEMM arithmetic for row-major A and transposed row-major B.
    ///
    /// One warp computes a 16x8 tensor-core tile with all 16 logical A rows
    /// duplicated from one input row; lanes 0..3 own that row's eight outputs.
    #[kernel]
    pub fn matvec_v4_fp8_mma(
        input: &[u8],
        input_scales: &[u8],
        weight: &[u8],
        weight_scales: &[u8],
        mut output_f32: DisjointSlice<f32>,
        mut output_bf16: DisjointSlice<u16>,
        in_features: u32,
        out_features: u32,
    ) {
        const FP8_BLOCK: usize = 128;
        const MMA_K: usize = 32;
        const MMA_N: usize = 8;
        let thread_index = thread::index_1d().get() as usize;
        let warp_index = thread_index / 32;
        let lane = warp::lane_id() as usize;
        let group = lane / 4;
        let thread_in_group = lane % 4;
        let in_features = in_features as usize;
        let out_features = out_features as usize;
        if in_features == 0
            || in_features % FP8_BLOCK != 0
            || out_features == 0
            || out_features % MMA_N != 0
        {
            return;
        }
        let rows = input.len() / in_features;
        let output_tiles = out_features / MMA_N;
        if input.len() != rows * in_features
            || output_f32.len() != rows * out_features
            || output_bf16.len() != rows * out_features
            || warp_index >= rows * output_tiles
        {
            return;
        }
        let blocks = in_features / FP8_BLOCK;
        if input_scales.len() != rows * blocks
            || weight.len() != out_features * in_features
            || weight_scales.len() != ((out_features + FP8_BLOCK - 1) / FP8_BLOCK) * blocks
        {
            return;
        }
        let row = warp_index / output_tiles;
        let output_start = (warp_index % output_tiles) * MMA_N;
        let weight_row = output_start + group;
        let mut total = [0.0_f32; 4];
        for block in 0..blocks {
            let mut fragment = [0.0_f32; 4];
            for mma_k in 0..(FP8_BLOCK / MMA_K) {
                let k_start = block * FP8_BLOCK + mma_k * MMA_K;
                let a0_start = row * in_features + k_start + thread_in_group * 4;
                let a1_start = a0_start + 16;
                let a0 = (input[a0_start] as u32)
                    | ((input[a0_start + 1] as u32) << 8)
                    | ((input[a0_start + 2] as u32) << 16)
                    | ((input[a0_start + 3] as u32) << 24);
                let a1 = (input[a1_start] as u32)
                    | ((input[a1_start + 1] as u32) << 8)
                    | ((input[a1_start + 2] as u32) << 16)
                    | ((input[a1_start + 3] as u32) << 24);
                let b0_start = weight_row * in_features + k_start + thread_in_group * 4;
                let b1_start = b0_start + 16;
                let b0 = (weight[b0_start] as u32)
                    | ((weight[b0_start + 1] as u32) << 8)
                    | ((weight[b0_start + 2] as u32) << 16)
                    | ((weight[b0_start + 3] as u32) << 24);
                let b1 = (weight[b1_start] as u32)
                    | ((weight[b1_start + 1] as u32) << 8)
                    | ((weight[b1_start + 2] as u32) << 16)
                    | ((weight[b1_start + 3] as u32) << 24);
                // SAFETY: every active lane in this full warp executes the same
                // register-only m16n8k32 instruction with PTX-documented fragments.
                fragment = unsafe {
                    cuda_device::wmma::mma_m16n8k32_fp8_f32_e4m3_e4m3(
                        fragment,
                        [a0, a0, a1, a1],
                        [b0, b1],
                    )
                };
            }
            let scale = decode_e8m0fnu(input_scales[row * blocks + block])
                * decode_e8m0fnu(weight_scales[(output_start / FP8_BLOCK) * blocks + block]);
            for index in 0..4 {
                total[index] = cuda_device::float::fma_rn_f32(fragment[index], scale, total[index]);
            }
        }
        if group == 0 {
            let output_index = row * out_features + output_start + thread_in_group * 2;
            // SAFETY: lanes 0..3 uniquely own two outputs in this warp's 8-wide tile.
            unsafe {
                *output_f32.as_mut_ptr().add(output_index) = total[0];
                *output_f32.as_mut_ptr().add(output_index + 1) = total[1];
                *output_bf16.as_mut_ptr().add(output_index) = f32_to_bf16_rne(total[0]);
                *output_bf16.as_mut_ptr().add(output_index + 1) = f32_to_bf16_rne(total[1]);
            }
        }
    }

    /// DeepSeek-V4-Flash FP8 x FP4 expert matvec on Ada FP8 tensor cores.
    ///
    /// One warp computes eight outputs. FP4 E2M1 nibbles are expanded in
    /// registers to raw E4M3, then each K32 partial is scaled and accumulated
    /// in the same strict order as the official 0731 kernel.
    #[kernel]
    pub fn matvec_v4_fp4_mma(
        input: &[u8],
        input_scales: &[u8],
        packed: &[u8],
        scales: &[u8],
        mut output: DisjointSlice<u16>,
        in_features: u32,
        out_features: u32,
    ) {
        const ACTIVATION_BLOCK: usize = 128;
        const WEIGHT_BLOCK: usize = 32;
        const MMA_N: usize = 8;
        let thread_index = thread::index_1d().get() as usize;
        let warp_index = thread_index / 32;
        let lane = warp::lane_id() as usize;
        let group = lane / 4;
        let thread_in_group = lane % 4;
        let in_features = in_features as usize;
        let out_features = out_features as usize;
        if in_features == 0
            || in_features % ACTIVATION_BLOCK != 0
            || out_features == 0
            || out_features % MMA_N != 0
        {
            return;
        }
        let rows = input.len() / in_features;
        let output_tiles = out_features / MMA_N;
        let activation_blocks = in_features / ACTIVATION_BLOCK;
        let weight_blocks = in_features / WEIGHT_BLOCK;
        if input.len() != rows * in_features
            || input_scales.len() != rows * activation_blocks
            || packed.len() != out_features * (in_features / 2)
            || scales.len() != out_features * weight_blocks
            || output.len() != rows * out_features
            || warp_index >= rows * output_tiles
        {
            return;
        }
        let row = warp_index / output_tiles;
        let output_start = (warp_index % output_tiles) * MMA_N;
        let weight_row = output_start + group;
        let packed_row = weight_row * (in_features / 2);
        let mut total = [0.0_f32; 4];
        for block in 0..weight_blocks {
            let k_start = block * WEIGHT_BLOCK;
            let a0_start = row * in_features + k_start + thread_in_group * 4;
            let a1_start = a0_start + 16;
            let a0 = (input[a0_start] as u32)
                | ((input[a0_start + 1] as u32) << 8)
                | ((input[a0_start + 2] as u32) << 16)
                | ((input[a0_start + 3] as u32) << 24);
            let a1 = (input[a1_start] as u32)
                | ((input[a1_start + 1] as u32) << 8)
                | ((input[a1_start + 2] as u32) << 16)
                | ((input[a1_start + 3] as u32) << 24);
            let b0_start = packed_row + k_start / 2 + thread_in_group * 2;
            let b1_start = b0_start + 8;
            let b0 = v4_fp4_pair_to_e4m3x2(packed[b0_start]) as u32
                | ((v4_fp4_pair_to_e4m3x2(packed[b0_start + 1]) as u32) << 16);
            let b1 = v4_fp4_pair_to_e4m3x2(packed[b1_start]) as u32
                | ((v4_fp4_pair_to_e4m3x2(packed[b1_start + 1]) as u32) << 16);
            // SAFETY: every active lane executes the same register-only
            // m16n8k32 instruction with PTX-documented fragment layouts.
            let fragment = unsafe {
                cuda_device::wmma::mma_m16n8k32_fp8_f32_e4m3_e4m3(
                    [0.0_f32; 4],
                    [a0, a0, a1, a1],
                    [b0, b1],
                )
            };
            let activation_scale = decode_e8m0fnu(
                input_scales[row * activation_blocks + block / (ACTIVATION_BLOCK / WEIGHT_BLOCK)],
            );
            let output_col = output_start + thread_in_group * 2;
            let weight_scale0 = decode_e8m0fnu(scales[output_col * weight_blocks + block]);
            let weight_scale1 = decode_e8m0fnu(scales[(output_col + 1) * weight_blocks + block]);
            let scaled0 = fragment[0] * activation_scale;
            let scaled1 = fragment[1] * activation_scale;
            let scaled2 = fragment[2] * activation_scale;
            let scaled3 = fragment[3] * activation_scale;
            total[0] = cuda_device::float::fma_rn_f32(scaled0, weight_scale0, total[0]);
            total[1] = cuda_device::float::fma_rn_f32(scaled1, weight_scale1, total[1]);
            total[2] = cuda_device::float::fma_rn_f32(scaled2, weight_scale0, total[2]);
            total[3] = cuda_device::float::fma_rn_f32(scaled3, weight_scale1, total[3]);
        }
        if group == 0 {
            let output_index = row * out_features + output_start + thread_in_group * 2;
            // SAFETY: lanes 0..3 uniquely own two outputs in this warp's tile.
            unsafe {
                *output.as_mut_ptr().add(output_index) = f32_to_bf16_rne(total[0]);
                *output.as_mut_ptr().add(output_index + 1) = f32_to_bf16_rne(total[1]);
            }
        }
    }

    /// V4 Compressor projection for HCA, CSA, and the CSA indexer.
    #[kernel]
    pub fn compressor_v4_bf16_f32(
        input: &[u16],
        wkv: &[u16],
        wgate: &[u16],
        ape: &[f32],
        mut kv: DisjointSlice<f32>,
        mut score: DisjointSlice<f32>,
        mut score_plus_ape: DisjointSlice<f32>,
        start_pos: u32,
        hidden: u32,
        outputs: u32,
        compress_ratio: u32,
    ) {
        const K_BLOCK: usize = 192;
        let index = thread::index_1d();
        let output_index = index.get() as usize;
        let hidden = hidden as usize;
        let outputs = outputs as usize;
        let compress_ratio = compress_ratio as usize;
        if hidden == 0 || outputs == 0 || !matches!(compress_ratio, 4 | 128) {
            return;
        }
        let rows = input.len() / hidden;
        if input.len() % hidden != 0
            || rows == 0
            || wkv.len() != outputs * hidden
            || wgate.len() != outputs * hidden
            || ape.len() != compress_ratio * outputs
            || kv.len() != rows * outputs
            || score.len() != rows * outputs
            || score_plus_ape.len() != rows * outputs
            || output_index >= rows * outputs
        {
            return;
        }
        let row = output_index / outputs;
        let output = output_index % outputs;
        let ape_row = (start_pos as usize + row) % compress_ratio;
        let ape_index = ape_row * outputs + output;
        let mut kv_accumulated = 0.0_f32;
        let mut score_accumulated = 0.0_f32;
        let mut block = 0;
        while block < hidden {
            let mut kv_block = 0.0_f32;
            let mut score_block = 0.0_f32;
            let end = (block + K_BLOCK).min(hidden);
            let mut column = block;
            while column < end {
                let value = bf16_to_f32(input[row * hidden + column]);
                kv_block = cuda_device::float::fma_rn_f32(
                    value,
                    bf16_to_f32(wkv[output * hidden + column]),
                    kv_block,
                );
                score_block = cuda_device::float::fma_rn_f32(
                    value,
                    bf16_to_f32(wgate[output * hidden + column]),
                    score_block,
                );
                column += 1;
            }
            kv_accumulated += kv_block;
            score_accumulated += score_block;
            block += K_BLOCK;
        }
        // SAFETY: this thread uniquely owns the checked row-major output.
        unsafe {
            *kv.as_mut_ptr().add(output_index) = kv_accumulated;
            *score.as_mut_ptr().add(output_index) = score_accumulated;
            *score_plus_ape.as_mut_ptr().add(output_index) = score_accumulated + ape[ape_index];
        }
    }

    /// Ratio-4 decode boundary pooling.  The first four slots are the prior
    /// overlapping window and the second four slots hold the current rows.
    #[kernel]
    pub fn compressor_pool_ratio4_v4_f32(
        persistent_kv: &[f32],
        persistent_score: &[f32],
        mut output: DisjointSlice<f32>,
        head_dim: u32,
    ) {
        let dimension = thread::index_1d().get() as usize;
        let head_dim = head_dim as usize;
        let value_width = 2 * head_dim;
        if head_dim == 0
            || output.len() != head_dim
            || persistent_kv.len() != 8 * value_width
            || persistent_score.len() != 8 * value_width
            || dimension >= head_dim
        {
            return;
        }
        let mut maximum = f32::NEG_INFINITY;
        for slot in 0..4 {
            maximum = maximum.max(persistent_score[slot * value_width + dimension]);
        }
        for slot in 0..4 {
            maximum =
                maximum.max(persistent_score[(4 + slot) * value_width + head_dim + dimension]);
        }
        let mut denominator = 0.0_f32;
        let mut numerator = 0.0_f32;
        for slot in 0..4 {
            let index = slot * value_width + dimension;
            let weight = v4_state::pytorch_sleef_expf(persistent_score[index] - maximum);
            denominator += weight;
            numerator = libm::fmaf(persistent_kv[index], weight, numerator);
        }
        for slot in 0..4 {
            let index = (4 + slot) * value_width + head_dim + dimension;
            let weight = v4_state::pytorch_sleef_expf(persistent_score[index] - maximum);
            denominator += weight;
            numerator = libm::fmaf(persistent_kv[index], weight, numerator);
        }
        unsafe { *output.as_mut_ptr().add(dimension) = numerator / denominator }
    }

    /// Ratio-128 HCA boundary pooling.  One thread owns one output channel and
    /// scans slots in the same 0..128 order as `V4CompressorState`, avoiding a
    /// nondeterministic parallel reduction at the long-context boundary.
    #[kernel]
    pub fn compressor_pool_ratio128_v4_f32(
        persistent_kv: &[f32],
        persistent_score: &[f32],
        mut output: DisjointSlice<f32>,
        head_dim: u32,
    ) {
        let dimension = thread::index_1d().get() as usize;
        let head_dim = head_dim as usize;
        const RATIO: usize = 128;
        if head_dim == 0
            || output.len() != head_dim
            || persistent_kv.len() != RATIO * head_dim
            || persistent_score.len() != RATIO * head_dim
            || dimension >= head_dim
        {
            return;
        }
        let mut maximum = f32::NEG_INFINITY;
        for slot in 0..RATIO {
            maximum = maximum.max(persistent_score[slot * head_dim + dimension]);
        }
        let mut denominator = 0.0_f32;
        let mut numerator = 0.0_f32;
        for slot in 0..RATIO {
            let index = slot * head_dim + dimension;
            let weight = v4_state::pytorch_sleef_expf(persistent_score[index] - maximum);
            denominator += weight;
            numerator = libm::fmaf(persistent_kv[index], weight, numerator);
        }
        unsafe { *output.as_mut_ptr().add(dimension) = numerator / denominator }
    }

    /// The exact host CSA indexer uses a 128-wide Walsh-Hadamard transform,
    /// BF16 rounding, then blockwise E2M1 simulation.  A single thread owns a
    /// complete vector so this preserves that observable operation order.
    #[kernel]
    pub fn indexer_fp4_sim_v4_bf16(input: &[u16], mut output: DisjointSlice<u16>) {
        const WIDTH: usize = 128;
        let vector = thread::index_1d().get() as usize;
        if input.len() % WIDTH != 0 || output.len() != input.len() || vector >= input.len() / WIDTH
        {
            return;
        }
        let mut values = [0.0_f32; WIDTH];
        for column in 0..WIDTH {
            values[column] = bf16_to_f32(input[vector * WIDTH + column]);
        }
        let mut stride = 1;
        while stride < WIDTH {
            let mut block = 0;
            while block < WIDTH {
                for offset in 0..stride {
                    let left = values[block + offset];
                    let right = values[block + stride + offset];
                    values[block + offset] = left + right;
                    values[block + stride + offset] = left - right;
                }
                block += stride * 2;
            }
            stride *= 2;
        }
        let scale = 1.0_f32 / sqrt_newton(WIDTH as f32);
        for value in &mut values {
            *value = bf16_to_f32(f32_to_bf16_rne(*value * scale));
        }
        for block in (0..WIDTH).step_by(32) {
            let mut amax = 0.0_f32;
            for offset in 0..32 {
                amax = amax.max(values[block + offset].abs());
            }
            let quant_scale = pow2_ceil((amax / 6.0).max(f32::from_bits(1 << 22)));
            for offset in 0..32 {
                values[block + offset] =
                    quantize_e2m1(values[block + offset] / quant_scale) * quant_scale;
            }
        }
        for column in 0..WIDTH {
            unsafe {
                *output.as_mut_ptr().add(vector * WIDTH + column) = f32_to_bf16_rne(values[column]);
            }
        }
    }

    #[kernel]
    pub fn f32_to_bf16_rne_v4(input: &[f32], mut output: DisjointSlice<u16>) {
        let index = thread::index_1d().get() as usize;
        if input.len() != output.len() || index >= input.len() {
            return;
        }
        unsafe { *output.as_mut_ptr().add(index) = f32_to_bf16_rne(input[index]) }
    }

    #[kernel]
    pub fn scale_f32_in_place_v4(mut values: DisjointSlice<f32>, scale: f32) {
        let index = thread::index_1d();
        if let Some(value) = values.get_mut(index) {
            *value *= scale;
        }
    }

    /// CSA indexer scoring after its q/kv projections, rotation, FP4
    /// simulation, and per-head weight projection.
    #[kernel]
    pub fn indexer_scores_v4_f32(
        q: &[f32],
        kv: &[f32],
        weights: &[f32],
        mut scores: DisjointSlice<f32>,
        rows: u32,
        heads: u32,
        head_dim: u32,
        keys: u32,
    ) {
        let index = thread::index_1d();
        let output = index.get() as usize;
        let rows = rows as usize;
        let heads = heads as usize;
        let head_dim = head_dim as usize;
        let keys = keys as usize;
        if rows == 0
            || heads == 0
            || head_dim == 0
            || keys == 0
            || q.len() != rows * heads * head_dim
            || kv.len() < keys * head_dim
            || weights.len() != rows * heads
            || scores.len() < rows * keys
            || output >= rows * keys
        {
            return;
        }
        let row = output / keys;
        let key = output % keys;
        let mut score = 0.0_f32;
        for head in 0..heads {
            let mut dot = 0.0_f32;
            for dimension in 0..head_dim {
                dot +=
                    q[(row * heads + head) * head_dim + dimension] * kv[key * head_dim + dimension];
            }
            score += dot.max(0.0) * weights[row * heads + head];
        }
        if let Some(destination) = scores.get_mut(index) {
            *destination = score;
        }
    }

    #[kernel]
    pub fn linear_v4_bf16_f32(
        input: &[u16],
        weight: &[u16],
        mut output: DisjointSlice<f32>,
        in_features: u32,
        out_features: u32,
    ) {
        let index = thread::index_1d();
        let output_index = index.get() as usize;
        let in_features = in_features as usize;
        let out_features = out_features as usize;
        if in_features == 0
            || out_features == 0
            || input.len() % in_features != 0
            || weight.len() != in_features * out_features
            || output.len() != input.len() / in_features * out_features
            || output_index >= output.len()
        {
            return;
        }
        let row = output_index / out_features;
        let column = output_index % out_features;
        let mut value = 0.0_f32;
        for feature in 0..in_features {
            value += bf16_to_f32(input[row * in_features + feature])
                * bf16_to_f32(weight[column * in_features + feature]);
        }
        if let Some(destination) = output.get_mut(index) {
            *destination = value;
        }
    }

    /// Deterministic CSA selection ordered by score descending, then key ascending.
    /// This correctness-first kernel intentionally uses one independent scan per rank.
    #[kernel]
    pub fn indexer_topk_v4_f32(
        scores: &[f32],
        valid_keys: &[u32],
        mut output: DisjointSlice<i32>,
        rows: u32,
        keys: u32,
        topk: u32,
        offset: u32,
    ) {
        let rows = rows as usize;
        let keys = keys as usize;
        let topk = topk as usize;
        let index = thread::index_1d().get() as usize;
        if rows == 0
            || keys == 0
            || topk == 0
            || topk > 1_024
            || topk > keys
            || scores.len() < rows * keys
            || valid_keys.len() < rows
            || output.len() < rows * topk
            || index >= rows * topk
        {
            return;
        }
        let row = index / topk;
        let rank = index % topk;
        let valid = valid_keys[row] as usize;
        if valid > keys || rank >= valid {
            unsafe { *output.as_mut_ptr().add(index) = -1 }
            return;
        }
        let mut previous_score = f32::INFINITY;
        let mut previous_key = 0_usize;
        let mut final_selected_key = 0_usize;
        for selection in 0..=rank {
            let mut selected_score = f32::NEG_INFINITY;
            let mut selected_key = 0_usize;
            for key in 0..valid {
                let score = scores[row * keys + key];
                if !score.is_finite() {
                    unsafe { *output.as_mut_ptr().add(index) = -1 }
                    return;
                }
                let follows_previous = selection == 0
                    || score < previous_score
                    || (score == previous_score && key > previous_key);
                if follows_previous
                    && (score > selected_score || (score == selected_score && key < selected_key))
                {
                    selected_score = score;
                    selected_key = key;
                }
            }
            previous_score = selected_score;
            previous_key = selected_key;
            final_selected_key = selected_key;
        }
        unsafe { *output.as_mut_ptr().add(index) = (offset as usize + final_selected_key) as i32 }
    }

    /// Decode-mode FP4 matvec: one warp owns one output row. Its lanes walk
    /// strided 32-K E8M0 tiles and reduce in registers; FP4 remains packed.
    #[kernel]
    pub fn matvec_v4_fp4(
        input: &[f32],
        packed: &[u8],
        scales: &[u8],
        mut output: DisjointSlice<u16>,
        in_features: u32,
        out_features: u32,
    ) {
        const WEIGHT_BLOCK: usize = 32;
        let thread_index = thread::index_1d();
        let thread_index = thread_index.get() as usize;
        let output_index = thread_index / 32;
        let lane = warp::lane_id() as usize;
        let output_len = output.len();
        let in_features = in_features as usize;
        let out_features = out_features as usize;
        if output_index >= output_len
            || in_features == 0
            || in_features % WEIGHT_BLOCK != 0
            || output_index >= input.len() / in_features * out_features
        {
            return;
        }
        let row = output_index / out_features;
        let output_row = output_index % out_features;
        let packed_row = output_row * (in_features / 2);
        let scale_row = output_row * (in_features / WEIGHT_BLOCK);
        let input_row = row * in_features;
        let mut acc = 0.0_f32;
        let mut block = lane;
        while block < in_features / WEIGHT_BLOCK {
            let scale = f32::from_bits((scales[scale_row + block] as u32) << 23);
            let packed_start = packed_row + block * (WEIGHT_BLOCK / 2);
            let input_start = input_row + block * WEIGHT_BLOCK;
            for pair in 0..WEIGHT_BLOCK / 2 {
                let byte = packed[packed_start + pair];
                acc += input[input_start + pair * 2] * decode_fp4(byte, 0) * scale;
                acc += input[input_start + pair * 2 + 1] * decode_fp4(byte, 1) * scale;
            }
            block += 32;
        }
        acc += warp::shuffle_xor_f32(acc, 16);
        acc += warp::shuffle_xor_f32(acc, 8);
        acc += warp::shuffle_xor_f32(acc, 4);
        acc += warp::shuffle_xor_f32(acc, 2);
        acc += warp::shuffle_xor_f32(acc, 1);
        if lane == 0 {
            // SAFETY: one lane per warp writes a unique output_index; bounds
            // were checked above and the host launches exactly 32 lanes/output.
            unsafe { *output.as_mut_ptr().add(output_index) = f32_to_bf16_rne(acc) };
        }
    }

    /// Exact-diagnostic Flash arm: each lane owns one K value in a 32-wide
    /// block, reduces unscaled FP8×FP4 first, then lane 0 applies both scales.
    #[kernel]
    pub fn matvec_v4_fp4_block32(
        input: &[u8],
        input_scales: &[u8],
        packed: &[u8],
        scales: &[u8],
        mut output: DisjointSlice<u16>,
        in_features: u32,
        out_features: u32,
    ) {
        const WEIGHT_BLOCK: usize = 32;
        const ACTIVATION_BLOCK: usize = 128;
        let thread_index = thread::index_1d().get() as usize;
        let output_index = thread_index / 32;
        let lane = warp::lane_id() as usize;
        let in_features = in_features as usize;
        let out_features = out_features as usize;
        if output_index >= output.len()
            || in_features == 0
            || in_features % ACTIVATION_BLOCK != 0
            || output_index >= input.len() / in_features * out_features
            || input_scales.len() != input.len() / ACTIVATION_BLOCK
            || packed.len() != out_features * in_features / 2
            || scales.len() != out_features * in_features / WEIGHT_BLOCK
        {
            return;
        }
        let row = output_index / out_features;
        let output_row = output_index % out_features;
        let input_row = row * in_features;
        let packed_row = output_row * (in_features / 2);
        let scale_row = output_row * (in_features / WEIGHT_BLOCK);
        let mut acc = 0.0_f32;
        for block in 0..in_features / WEIGHT_BLOCK {
            let input_index = input_row + block * WEIGHT_BLOCK + lane;
            let packed_byte = packed[packed_row + block * (WEIGHT_BLOCK / 2) + lane / 2];
            let mut partial = decode_e4m3fn(input[input_index]) * decode_fp4(packed_byte, lane % 2);
            partial += warp::shuffle_xor_f32(partial, 16);
            partial += warp::shuffle_xor_f32(partial, 8);
            partial += warp::shuffle_xor_f32(partial, 4);
            partial += warp::shuffle_xor_f32(partial, 2);
            partial += warp::shuffle_xor_f32(partial, 1);
            if lane == 0 {
                let activation_scale = decode_e8m0fnu(
                    input_scales[row * (in_features / ACTIVATION_BLOCK) + block / 4],
                );
                let weight_scale = decode_e8m0fnu(scales[scale_row + block]);
                acc += partial * activation_scale * weight_scale;
            }
        }
        if lane == 0 {
            // ponytail: scalar warp arm is exact diagnosis only; assess MMA
            // performance only after this L0 boundary passes parity.
            unsafe { *output.as_mut_ptr().add(output_index) = f32_to_bf16_rne(acc) };
        }
    }

    #[kernel]
    pub fn swiglu_route_bf16(
        gate: &[u16],
        up: &[u16],
        mut output: DisjointSlice<u16>,
        route_weight: f32,
    ) {
        let index = thread::index_1d();
        let index_raw = index.get();
        let Some(value) = output.get_mut(index) else {
            return;
        };
        if index_raw >= gate.len() || index_raw >= up.len() {
            return;
        }
        let gate = bf16_to_f32(gate[index_raw]).min(10.0);
        let up = bf16_to_f32(up[index_raw]).max(-10.0).min(10.0);
        *value = f32_to_bf16_rne(gate / (1.0 + libm::expf(-gate)) * up * route_weight);
    }

    #[kernel]
    pub fn bf16_to_f32_rows(input: &[u16], mut output: DisjointSlice<f32>) {
        let index = thread::index_1d();
        let index_raw = index.get();
        let Some(value) = output.get_mut(index) else {
            return;
        };
        if index_raw < input.len() {
            *value = bf16_to_f32(input[index_raw]);
        }
    }

    #[kernel]
    pub fn add_in_place(mut x: DisjointSlice<f32>, t: &[f32]) {
        let index = thread::index_1d();
        let index_raw = index.get();
        if let Some(value) = x.get_mut(index) {
            *value += t[index_raw];
        }
    }

    #[kernel]
    pub fn fill_f32(mut output: DisjointSlice<f32>, value: f32) {
        let index = thread::index_1d().get() as usize;
        if index < output.len() {
            unsafe { *output.as_mut_ptr().add(index) = value };
        }
    }

    #[kernel]
    pub fn fill_u32(mut output: DisjointSlice<u32>, value: u32) {
        let index = thread::index_1d().get() as usize;
        if index < output.len() {
            unsafe { *output.as_mut_ptr().add(index) = value };
        }
    }

    #[kernel]
    pub fn silu_mul_in_place(mut gate: DisjointSlice<f32>, up: &[f32]) {
        let index = thread::index_1d();
        let index_raw = index.get();
        if let Some(value) = gate.get_mut(index) {
            let input = *value;
            *value = (input / (1.0 + libm::expf(-input))) * up[index_raw];
        }
    }

    #[kernel]
    pub fn rows_add_in_place(mut x: DisjointSlice<f32>, partial: &[f32], rows: &[i32], d: u32) {
        let index = thread::index_1d();
        let index_raw = index.get();
        if index_raw >= partial.len() {
            return;
        }
        let d = d as usize;
        let partial_row = index_raw / d;
        let x_index = rows[partial_row] as usize * d + index_raw % d;
        // SAFETY: launch_rows_add proves d, lengths, and row targets before enqueueing.
        unsafe { *x.as_mut_ptr().add(x_index) += partial[index_raw] }
    }

    #[kernel]
    pub fn decode_v4_fp4_blockwise(
        packed: &[u8],
        scales: &[u8],
        mut output: DisjointSlice<f32>,
        logical_cols: u32,
    ) {
        let index = thread::index_1d();
        let index_raw = index.get();
        let Some(value) = output.get_mut(index) else {
            return;
        };
        let logical_cols = logical_cols as usize;
        let row = index_raw / logical_cols;
        let column = index_raw % logical_cols;
        let byte = packed[row * (logical_cols / 2) + column / 2];
        let nibble = if column % 2 == 0 {
            byte & 0x0f
        } else {
            byte >> 4
        };
        let magnitude = match nibble & 0x07 {
            0 => 0.0,
            1 => 0.5,
            2 => 1.0,
            3 => 1.5,
            4 => 2.0,
            5 => 3.0,
            6 => 4.0,
            _ => 6.0,
        };
        let fp4 = if magnitude == 0.0 || nibble & 0x08 == 0 {
            magnitude
        } else {
            -magnitude
        };
        let scale = scales[row * (logical_cols / 32) + column / 32];
        // E8M0's stored exponent matches the exponent field in an f32 power of two.
        *value = fp4 * f32::from_bits((scale as u32) << 23);
    }

    #[kernel]
    pub fn rmsnorm_v4_bf16(
        input: &[u16],
        norm: &[u16],
        mut output: DisjointSlice<u16>,
        hidden: u32,
    ) {
        let row = thread::index_1d().get() as usize;
        let hidden = hidden as usize;
        let start = row * hidden;
        let unit_rms = norm.is_empty();
        if start + hidden > input.len()
            || (!unit_rms && norm.len() != hidden)
            || start + hidden > output.len()
        {
            return;
        }
        let mut square_sum = 0.0;
        if unit_rms {
            for column in 0..hidden {
                let value = f32::from_bits((input[start + column] as u32) << 16);
                square_sum += bf16_to_f32(f32_to_bf16_rne(value * value));
            }
        } else {
            square_sum = pytorch_square_sum_bf16(input, start, hidden);
        }
        let inv_rms = if unit_rms {
            let mean = bf16_to_f32(f32_to_bf16_rne(square_sum / hidden as f32));
            let mean_eps = bf16_to_f32(f32_to_bf16_rne(mean + 1e-6));
            bf16_to_f32(f32_to_bf16_rne(1.0 / sqrt_newton(mean_eps)))
        } else {
            // The pinned RMSNorm evaluates rsqrt directly in FP32.  The
            // Newton reciprocal-sqrt path is one ULP high on L1's exact FFN
            // input and crosses three BF16 output boundaries.
            mhc_committed_rsqrt(square_sum / hidden as f32 + 1e-6)
        };
        for column in 0..hidden {
            let normalized = f32::from_bits((input[start + column] as u32) << 16) * inv_rms;
            let value = if unit_rms {
                normalized
            } else {
                normalized * f32::from_bits((norm[column] as u32) << 16)
            };
            let bits = value.to_bits();
            // SAFETY: the row bounds above prove this output index is valid.
            unsafe {
                *output.as_mut_ptr().add(start + column) =
                    ((bits.wrapping_add(0x7fff + ((bits >> 16) & 1))) >> 16) as u16;
            }
        }
    }

    #[kernel]
    pub fn rope_v4_bf16(
        input: &[u16],
        freqs_cis_f32_pairs: &[f32],
        mut output: DisjointSlice<u16>,
        heads: u32,
        head_dim: u32,
    ) {
        const ROPE_DIM: usize = 64;
        let heads = heads as usize;
        let head_dim = head_dim as usize;
        let index = thread::index_1d();
        let offset = index.get() as usize;
        let row_width = heads * head_dim;
        if heads == 0
            || head_dim < ROPE_DIM
            || input.len() % row_width != 0
            || input.len() / row_width == 0
            || input.len() / row_width > 4
            || output.len() != input.len()
            || freqs_cis_f32_pairs.len() != input.len() / row_width * ROPE_DIM
        {
            return;
        }
        let Some(destination) = output.get_mut(index) else {
            return;
        };
        let dimension = offset % head_dim;
        if dimension < head_dim - ROPE_DIM {
            *destination = input[offset];
            return;
        }
        let position = offset / row_width;
        let pair_dimension = (dimension - (head_dim - ROPE_DIM)) / 2;
        let pair_start = offset - (dimension & 1);
        let frequency = position * ROPE_DIM + pair_dimension * 2;
        let real = bf16_to_f32(input[pair_start]);
        let imaginary = bf16_to_f32(input[pair_start + 1]);
        let cosine = freqs_cis_f32_pairs[frequency];
        let sine = freqs_cis_f32_pairs[frequency + 1];
        let product_0 = if dimension & 1 == 0 {
            real * cosine
        } else {
            real * sine
        };
        let product_1 = if dimension & 1 == 0 {
            imaginary * sine
        } else {
            imaginary * cosine
        };
        // SAFETY: volatile reads of initialized locals preserve the official
        // two-rounded-products order instead of contracting this into FMA.
        let product_0 = unsafe { core::ptr::read_volatile(&product_0) };
        let product_1 = unsafe { core::ptr::read_volatile(&product_1) };
        let rotated = if dimension & 1 == 0 {
            product_0 - product_1
        } else {
            product_0 + product_1
        };
        *destination = f32_to_bf16_rne(rotated);
    }

    #[kernel]
    pub fn causal_topk_v4_i32(mut output: DisjointSlice<i32>) {
        let index = thread::index_1d();
        let offset = index.get() as usize;
        if output.len() != 16 {
            return;
        }
        let Some(destination) = output.get_mut(index) else {
            return;
        };
        let row = offset / 4;
        let column = offset % 4;
        *destination = if column <= row { column as i32 } else { -1 };
    }

    #[kernel]
    pub fn sparse_attn_v4_bf16(
        q: &[u16],
        kv: &[u16],
        attn_sink: &[f32],
        topk: &[i32],
        mut output: DisjointSlice<u16>,
        topk_per_row: u32,
        heads: u32,
        dim: u32,
    ) {
        let heads = heads as usize;
        let dim = dim as usize;
        const SCALE: f32 = 0.044_194_173;
        let index = thread::index_1d().get() as usize;
        let topk_per_row = topk_per_row as usize;
        if heads == 0
            || dim == 0
            || q.len() != output.len()
            || q.len() % (heads * dim) != 0
            || kv.len() % dim != 0
            || attn_sink.len() != heads
            || topk_per_row == 0
            || topk.len() < q.len() / (heads * dim) * topk_per_row
            || index >= output.len()
        {
            return;
        }
        let row = index / (heads * dim);
        let head = index / dim % heads;
        let dimension = index % dim;
        let q_start = (row * heads + head) * dim;
        let mut score_max = f32::NEG_INFINITY;
        let mut valid = 0;
        for slot in 0..topk_per_row {
            let key = topk[row * topk_per_row + slot];
            if key < 0 || key as usize >= kv.len() / dim {
                continue;
            }
            let mut score = 0.0_f32;
            for column in 0..dim {
                score +=
                    bf16_to_f32(q[q_start + column]) * bf16_to_f32(kv[key as usize * dim + column]);
            }
            score *= SCALE;
            score_max = score_max.max(score);
            valid += 1;
        }
        if valid == 0 {
            unsafe { *output.as_mut_ptr().add(index) = 0 }
            return;
        }
        let mut numerator = 0.0_f32;
        let mut denominator = 0.0_f32;
        for slot in 0..topk_per_row {
            let key = topk[row * topk_per_row + slot];
            if key < 0 || key as usize >= kv.len() / dim {
                continue;
            }
            let mut score = 0.0_f32;
            for column in 0..dim {
                score +=
                    bf16_to_f32(q[q_start + column]) * bf16_to_f32(kv[key as usize * dim + column]);
            }
            let probability = libm::expf(score * SCALE - score_max);
            denominator += probability;
            numerator += probability * bf16_to_f32(kv[key as usize * dim + dimension]);
        }
        denominator += libm::expf(attn_sink[head] - score_max);
        unsafe {
            *output.as_mut_ptr().add(index) = f32_to_bf16_rne(numerator / denominator);
        }
    }

    // Candidate arm: retain the scalar kernel above as the reference.  These
    // three phases only materialize values that the reference recomputes for
    // every output dimension; every floating-point chain keeps its order.
    #[kernel]
    pub fn sparse_attn_v4_reference_state(
        q: &[u16],
        kv: &[u16],
        attn_sink: &[f32],
        topk: &[i32],
        mut scores: DisjointSlice<f32>,
        mut probabilities: DisjointSlice<f32>,
        mut denominators: DisjointSlice<f32>,
        mut valid_counts: DisjointSlice<u32>,
        topk_per_row: u32,
        heads: u32,
        dim: u32,
    ) {
        let heads = heads as usize;
        let dim = dim as usize;
        const SCALE: f32 = 0.044_194_173;
        let index = thread::index_1d().get() as usize;
        let topk_per_row = topk_per_row as usize;
        let rows = denominators.len() / heads.max(1);
        if heads == 0
            || dim == 0
            || rows == 0
            || q.len() != rows * heads * dim
            || kv.len() % dim != 0
            || attn_sink.len() != heads
            || topk_per_row == 0
            || topk.len() < rows * topk_per_row
            || scores.len() < rows * heads * topk_per_row
            || probabilities.len() < rows * heads * topk_per_row
            || valid_counts.len() != rows * heads
            || index >= denominators.len()
        {
            return;
        }
        let row = index / heads;
        let head = index % heads;
        let q_start = (row * heads + head) * dim;
        let mut score_max = f32::NEG_INFINITY;
        let mut valid = 0_u32;
        for slot in 0..topk_per_row {
            let key = topk[row * topk_per_row + slot];
            if key < 0 || key as usize >= kv.len() / dim {
                continue;
            }
            let mut score = 0.0_f32;
            for column in 0..dim {
                score +=
                    bf16_to_f32(q[q_start + column]) * bf16_to_f32(kv[key as usize * dim + column]);
            }
            unsafe { *scores.as_mut_ptr().add(index * topk_per_row + slot) = score }
            score *= SCALE;
            score_max = score_max.max(score);
            valid += 1;
        }
        if valid == 0 {
            unsafe {
                *denominators.as_mut_ptr().add(index) = 0.0;
                *valid_counts.as_mut_ptr().add(index) = 0;
            }
            return;
        }
        let mut denominator = 0.0_f32;
        for slot in 0..topk_per_row {
            let key = topk[row * topk_per_row + slot];
            if key < 0 || key as usize >= kv.len() / dim {
                continue;
            }
            let mut score = 0.0_f32;
            for column in 0..dim {
                score +=
                    bf16_to_f32(q[q_start + column]) * bf16_to_f32(kv[key as usize * dim + column]);
            }
            let probability = libm::expf(score * SCALE - score_max);
            unsafe {
                *probabilities.as_mut_ptr().add(index * topk_per_row + slot) = probability;
            }
            denominator += probability;
        }
        denominator += libm::expf(attn_sink[head] - score_max);
        unsafe {
            *denominators.as_mut_ptr().add(index) = denominator;
            *valid_counts.as_mut_ptr().add(index) = valid;
        }
    }

    #[kernel]
    pub fn sparse_attn_v4_scores(
        q: &[u16],
        kv: &[u16],
        topk: &[i32],
        mut scores: DisjointSlice<f32>,
        topk_per_row: u32,
        heads: u32,
        dim: u32,
    ) {
        let heads = heads as usize;
        let dim = dim as usize;
        let index = thread::index_1d().get() as usize;
        let topk_per_row = topk_per_row as usize;
        if heads == 0 || dim == 0 {
            return;
        }
        let score_width = heads * topk_per_row;
        let score_len = q.len() / (heads * dim) * score_width;
        if q.len() % (heads * dim) != 0
            || kv.len() % dim != 0
            || topk_per_row == 0
            || topk.len() < q.len() / (heads * dim) * topk_per_row
            || scores.len() < score_len
            || index >= score_len
        {
            return;
        }
        let row = index / score_width;
        let head = index / topk_per_row % heads;
        let slot = index % topk_per_row;
        let key = topk[row * topk_per_row + slot];
        let score = if key < 0 || key as usize >= kv.len() / dim {
            0.0_f32
        } else {
            let q_start = (row * heads + head) * dim;
            let mut score = 0.0_f32;
            for column in 0..dim {
                score +=
                    bf16_to_f32(q[q_start + column]) * bf16_to_f32(kv[key as usize * dim + column]);
            }
            score
        };
        unsafe { *scores.as_mut_ptr().add(index) = score }
    }

    #[kernel]
    pub fn sparse_attn_v4_probabilities(
        scores: &[f32],
        kv: &[u16],
        attn_sink: &[f32],
        topk: &[i32],
        mut probabilities: DisjointSlice<f32>,
        mut denominators: DisjointSlice<f32>,
        mut valid_counts: DisjointSlice<u32>,
        topk_per_row: u32,
        heads: u32,
        dim: u32,
    ) {
        let heads = heads as usize;
        let dim = dim as usize;
        const SCALE: f32 = 0.044_194_173;
        let index = thread::index_1d().get() as usize;
        let topk_per_row = topk_per_row as usize;
        let rows = denominators.len() / heads.max(1);
        if heads == 0
            || dim == 0
            || rows == 0
            || denominators.len() != rows * heads
            || valid_counts.len() != rows * heads
            || attn_sink.len() != heads
            || kv.len() % dim != 0
            || topk_per_row == 0
            || topk.len() < rows * topk_per_row
            || scores.len() < rows * heads * topk_per_row
            || probabilities.len() < rows * heads * topk_per_row
            || index >= denominators.len()
        {
            return;
        }
        let row = index / heads;
        let mut score_max = f32::NEG_INFINITY;
        let mut valid = 0_u32;
        for slot in 0..topk_per_row {
            let key = topk[row * topk_per_row + slot];
            if key < 0 || key as usize >= kv.len() / dim {
                continue;
            }
            let score = scores[index * topk_per_row + slot] * SCALE;
            score_max = score_max.max(score);
            valid += 1;
        }
        if valid == 0 {
            unsafe {
                *denominators.as_mut_ptr().add(index) = 0.0;
                *valid_counts.as_mut_ptr().add(index) = 0;
            }
            return;
        }
        let mut denominator = 0.0_f32;
        for slot in 0..topk_per_row {
            let key = topk[row * topk_per_row + slot];
            if key < 0 || key as usize >= kv.len() / dim {
                continue;
            }
            let probability = libm::expf(scores[index * topk_per_row + slot] * SCALE - score_max);
            unsafe { *probabilities.as_mut_ptr().add(index * topk_per_row + slot) = probability }
            denominator += probability;
        }
        denominator += libm::expf(attn_sink[index % heads] - score_max);
        unsafe {
            *denominators.as_mut_ptr().add(index) = denominator;
            *valid_counts.as_mut_ptr().add(index) = valid;
        }
    }

    #[kernel]
    pub fn sparse_attn_v4_values(
        kv: &[u16],
        topk: &[i32],
        probabilities: &[f32],
        denominators: &[f32],
        valid_counts: &[u32],
        mut output: DisjointSlice<u16>,
        topk_per_row: u32,
        heads: u32,
        dim: u32,
    ) {
        let heads = heads as usize;
        let dim = dim as usize;
        let index = thread::index_1d().get() as usize;
        let topk_per_row = topk_per_row as usize;
        if heads == 0 || dim == 0 {
            return;
        }
        let rows = output.len() / (heads * dim);
        if output.len() != rows * heads * dim
            || kv.len() % dim != 0
            || topk_per_row == 0
            || topk.len() < rows * topk_per_row
            || probabilities.len() < rows * heads * topk_per_row
            || denominators.len() != rows * heads
            || valid_counts.len() != rows * heads
            || index >= output.len()
        {
            return;
        }
        let row = index / (heads * dim);
        let head = index / dim % heads;
        let dimension = index % dim;
        let head_index = row * heads + head;
        if valid_counts[head_index] == 0 {
            unsafe { *output.as_mut_ptr().add(index) = 0 }
            return;
        }
        let mut numerator = 0.0_f32;
        for slot in 0..topk_per_row {
            let key = topk[row * topk_per_row + slot];
            if key < 0 || key as usize >= kv.len() / dim {
                continue;
            }
            numerator += probabilities[head_index * topk_per_row + slot]
                * bf16_to_f32(kv[key as usize * dim + dimension]);
        }
        unsafe {
            *output.as_mut_ptr().add(index) = f32_to_bf16_rne(numerator / denominators[head_index]);
        }
    }

    #[kernel]
    pub fn wo_a_v4_bf16(
        input: &[u16],
        weight: &[u16],
        mut output: DisjointSlice<u16>,
        heads: u32,
        head_dim: u32,
        groups: u32,
        group_output: u32,
    ) {
        let heads = heads as usize;
        let head_dim = head_dim as usize;
        let groups = groups as usize;
        let group_output = group_output as usize;
        if heads == 0 || head_dim == 0 || groups == 0 || heads % groups != 0 {
            return;
        }
        let group_input = heads * head_dim / groups;
        let output_width = groups * group_output;
        let index = thread::index_1d().get() as usize;
        if input.len() % (heads * head_dim) != 0
            || weight.len() != output_width * group_input
            || output.len() != input.len() / (heads * head_dim) * output_width
            || index >= output.len()
        {
            return;
        }
        let row = index / output_width;
        let output_row = index % output_width;
        let group = output_row / group_output;
        let input_start = row * heads * head_dim + group * group_input;
        let weight_start = output_row * group_input;
        let mut accumulated = 0.0_f32;
        for column in 0..group_input {
            accumulated = cuda_device::float::fma_rn_f32(
                bf16_to_f32(input[input_start + column]),
                bf16_to_f32(weight[weight_start + column]),
                accumulated,
            );
        }
        unsafe { *output.as_mut_ptr().add(index) = f32_to_bf16_rne(accumulated) }
    }

    #[kernel]
    pub fn gate_logits_v4_bf16(
        input: &[u16],
        gate: &[u16],
        mut logits: DisjointSlice<f32>,
        hidden: u32,
        experts: u32,
    ) {
        let hidden = hidden as usize;
        let experts = experts as usize;
        const K_BLOCK: usize = 192;
        let index = thread::index_1d().get() as usize;
        if hidden == 0
            || experts == 0
            || input.len() % hidden != 0
            || gate.len() != experts * hidden
            || logits.len() != input.len() / hidden * experts
            || index >= logits.len()
        {
            return;
        }
        let row = index / experts;
        let expert = index % experts;
        let mut accumulated = 0.0_f32;
        let mut block = 0;
        while block < hidden {
            let mut partial = 0.0_f32;
            let end = (block + K_BLOCK).min(hidden);
            for column in block..end {
                partial = cuda_device::float::fma_rn_f32(
                    bf16_to_f32(input[row * hidden + column]),
                    bf16_to_f32(gate[expert * hidden + column]),
                    partial,
                );
            }
            accumulated += partial;
            block += K_BLOCK;
        }
        unsafe { *logits.as_mut_ptr().add(index) = accumulated }
    }

    #[kernel]
    pub fn hc_post_v4_bf16(
        input: &[u16],
        residual: &[u16],
        post: &[f32],
        comb: &[f32],
        mut output: DisjointSlice<u16>,
    ) {
        const HC: usize = 4;
        let index = thread::index_1d().get() as usize;
        if post.len() % HC != 0 {
            return;
        }
        let rows = post.len() / HC;
        if rows == 0 || input.len() % rows != 0 {
            return;
        }
        let hidden = input.len() / rows;
        if residual.len() != input.len() * HC
            || comb.len() != rows * HC * HC
            || output.len() != residual.len()
            || index >= output.len()
        {
            return;
        }
        let row = index / (HC * hidden);
        let copy = index / hidden % HC;
        let column = index % hidden;
        let residual_start = row * HC * hidden;
        let mut products = [0.0_f32; HC];
        for source in 0..HC {
            let product = comb[(row * HC + source) * HC + copy]
                * bf16_to_f32(residual[residual_start + source * hidden + column]);
            products[source] = unsafe { core::ptr::read_volatile(&product) };
        }
        // The pinned four-lane `torch.sum(..., dim=source_lane)` kernel folds
        // source lanes in order.  Each FP32 boundary is observable at BF16
        // RNE, so spell out the three rounded adds.
        let residual_sum01 = cuda_device::float::add_rn_f32(products[0], products[1]);
        let residual_sum012 = cuda_device::float::add_rn_f32(residual_sum01, products[2]);
        let residual_sum = cuda_device::float::add_rn_f32(residual_sum012, products[3]);
        let branch = post[row * HC + copy] * bf16_to_f32(input[row * hidden + column]);
        let branch = unsafe { core::ptr::read_volatile(&branch) };
        unsafe {
            *output.as_mut_ptr().add(index) = f32_to_bf16_rne(branch + residual_sum);
        }
    }

    /// Parity-only raw-word probe for one HC-post output.  It never feeds the
    /// model result and is launched only by the Flash L0 dump path.
    #[kernel]
    pub fn hc_post_v4_bf16_trace(
        input: &[u16],
        residual: &[u16],
        post: &[f32],
        comb: &[f32],
        audit_index: u32,
        mut audit: DisjointSlice<u32>,
    ) {
        const HC: usize = 4;
        const WORDS: usize = 21;
        if thread::index_1d().get() != 0 || audit.len() != WORDS || post.len() % HC != 0 {
            return;
        }
        let rows = post.len() / HC;
        if rows == 0
            || input.len() % rows != 0
            || residual.len() != input.len() * HC
            || comb.len() != rows * HC * HC
            || audit_index as usize >= residual.len()
        {
            return;
        }
        let hidden = input.len() / rows;
        let index = audit_index as usize;
        let row = index / (HC * hidden);
        let copy = index / hidden % HC;
        let column = index % hidden;
        let residual_start = row * HC * hidden;
        let branch_input = input[row * hidden + column];
        let branch = post[row * HC + copy] * bf16_to_f32(branch_input);
        let branch = unsafe { core::ptr::read_volatile(&branch) };
        let mut products = [0.0_f32; HC];
        unsafe {
            *audit.as_mut_ptr().add(0) = audit_index;
            *audit.as_mut_ptr().add(1) = branch_input as u32;
            *audit.as_mut_ptr().add(2) = post[row * HC + copy].to_bits();
            *audit.as_mut_ptr().add(3) = branch.to_bits();
            for source in 0..HC {
                let residual_bits = residual[residual_start + source * hidden + column];
                let coefficient = comb[(row * HC + source) * HC + copy];
                let product = coefficient * bf16_to_f32(residual_bits);
                products[source] = core::ptr::read_volatile(&product);
                *audit.as_mut_ptr().add(4 + source) = residual_bits as u32;
                *audit.as_mut_ptr().add(8 + source) = coefficient.to_bits();
                *audit.as_mut_ptr().add(12 + source) = products[source].to_bits();
            }
            let sum01 = cuda_device::float::add_rn_f32(products[0], products[1]);
            let sum012 = cuda_device::float::add_rn_f32(sum01, products[2]);
            let residual_sum = cuda_device::float::add_rn_f32(sum012, products[3]);
            *audit.as_mut_ptr().add(16) = sum01.to_bits();
            *audit.as_mut_ptr().add(17) = sum012.to_bits();
            *audit.as_mut_ptr().add(18) = residual_sum.to_bits();
            *audit.as_mut_ptr().add(19) = (branch + residual_sum).to_bits();
            *audit.as_mut_ptr().add(20) = f32_to_bf16_rne(branch + residual_sum) as u32;
        }
    }

    #[kernel]
    pub fn mhc_pre_v4_bf16(
        input: &[u16],
        hc_fn: &[f32],
        hc_scale: &[f32],
        hc_base: &[f32],
        mut collapsed: DisjointSlice<u16>,
        mut post: DisjointSlice<f32>,
        mut comb: DisjointSlice<f32>,
    ) {
        const HC: usize = 4;
        const MIX: usize = 24;
        const EPS: f32 = 1e-6;
        const K_BLOCK: usize = 192;
        let row = thread::index_1d().get() as usize;
        if hc_fn.len() % (MIX * HC) != 0 {
            return;
        }
        let h = hc_fn.len() / (MIX * HC);
        if h == 0
            || input.len() % (HC * h) != 0
            || row >= input.len() / (HC * h)
            || hc_scale.len() != 3
            || hc_base.len() != MIX
            || collapsed.len() != input.len() / HC
            || post.len() != input.len() / h
            || comb.len() != input.len() / (HC * h) * HC * HC
        {
            return;
        }
        let start = row * HC * h;
        let square = pytorch_square_sum_bf16(input, start, HC * h);
        let inv = 1.0 / sqrt_newton(square / (HC * h) as f32 + EPS);
        let mut mixes = [0.0; MIX];
        for out in 0..MIX {
            let mut dot = 0.0;
            let mut block = 0;
            while block < HC * h {
                let mut partial = 0.0;
                let end = (block + K_BLOCK).min(HC * h);
                for i in block..end {
                    partial = cuda_device::float::fma_rn_f32(
                        bf16_to_f32(input[start + i]),
                        hc_fn[out * HC * h + i],
                        partial,
                    );
                }
                dot += partial;
                block += K_BLOCK;
            }
            mixes[out] = dot * inv;
        }
        let mut pre = [0.0; HC];
        for i in 0..HC {
            pre[i] = 1.0 / (1.0 + libm::expf(-(mixes[i] * hc_scale[0] + hc_base[i]))) + EPS;
            unsafe {
                *post.as_mut_ptr().add(row * HC + i) =
                    2.0 / (1.0 + libm::expf(-(mixes[HC + i] * hc_scale[1] + hc_base[HC + i])));
            }
        }
        let base = row * HC * HC;
        for i in 0..HC {
            let mut max = f32::NEG_INFINITY;
            for j in 0..HC {
                let x = mixes[2 * HC + i * HC + j] * hc_scale[2] + hc_base[2 * HC + i * HC + j];
                unsafe {
                    *comb.as_mut_ptr().add(base + i * HC + j) = x;
                }
                max = max.max(x);
            }
            let mut sum = 0.0;
            for j in 0..HC {
                let x = libm::expf(unsafe { *comb.as_mut_ptr().add(base + i * HC + j) } - max);
                unsafe {
                    *comb.as_mut_ptr().add(base + i * HC + j) = x;
                }
                sum += x;
            }
            for j in 0..HC {
                unsafe {
                    let p = comb.as_mut_ptr().add(base + i * HC + j);
                    *p = *p / sum + EPS;
                }
            }
        }
        for step in 0..20 {
            for j in 0..HC {
                let mut sum = 0.0;
                for i in 0..HC {
                    unsafe { sum += *comb.as_mut_ptr().add(base + i * HC + j) }
                }
                for i in 0..HC {
                    unsafe {
                        let p = comb.as_mut_ptr().add(base + i * HC + j);
                        *p = *p / (sum + EPS)
                    }
                }
            }
            if step != 19 {
                for i in 0..HC {
                    let mut sum = 0.0;
                    for j in 0..HC {
                        unsafe { sum += *comb.as_mut_ptr().add(base + i * HC + j) }
                    }
                    for j in 0..HC {
                        unsafe {
                            let p = comb.as_mut_ptr().add(base + i * HC + j);
                            *p = *p / (sum + EPS)
                        }
                    }
                }
            }
        }
        for col in 0..h {
            let mut value = 0.0;
            for copy in 0..HC {
                value += pre[copy] * bf16_to_f32(input[start + copy * h + col]);
            }
            unsafe {
                *collapsed.as_mut_ptr().add(row * h + col) = f32_to_bf16_rne(value);
            }
        }
    }

    /// mHC's normalized linear stage only: [rows, 4, H] -> [rows, 24].
    #[kernel]
    pub fn mhc_linear_v4_bf16_scalar(input: &[u16], hc_fn: &[f32], mut mixes: DisjointSlice<f32>) {
        const HC: usize = 4;
        const MIX: usize = 24;
        const EPS: f32 = 1e-6;
        const K_BLOCK: usize = 192;
        let row = thread::index_1d().get() as usize;
        if hc_fn.len() % (MIX * HC) != 0 {
            return;
        }
        let h = hc_fn.len() / (MIX * HC);
        if h == 0
            || input.len() % (HC * h) != 0
            || row >= input.len() / (HC * h)
            || mixes.len() != input.len() / (HC * h) * MIX
        {
            return;
        }
        let start = row * HC * h;
        let square = pytorch_square_sum_bf16(input, start, HC * h);
        let inv = 1.0 / sqrt_newton(square / (HC * h) as f32 + EPS);
        for out in 0..MIX {
            let mut dot = 0.0;
            let mut block = 0;
            while block < HC * h {
                let mut partial = 0.0;
                let end = (block + K_BLOCK).min(HC * h);
                for i in block..end {
                    partial = cuda_device::float::fma_rn_f32(
                        bf16_to_f32(input[start + i]),
                        hc_fn[out * HC * h + i],
                        partial,
                    );
                }
                dot += partial;
                block += K_BLOCK;
            }
            // SAFETY: the layout check above proves this output row exists.
            unsafe { *mixes.as_mut_ptr().add(row * MIX + out) = dot * inv }
        }
    }

    /// SM89 candidate: one warp owns one row, with lane 0 preserving the
    /// scalar RMS reduction and each of lanes 0..24 preserving one output's
    /// original 192-wide FMA/block accumulation order.
    #[kernel]
    pub fn mhc_linear_v4_bf16(input: &[u16], hc_fn: &[f32], mut mixes: DisjointSlice<f32>) {
        const HC: usize = 4;
        const MIX: usize = 24;
        const EPS: f32 = 1e-6;
        const K_BLOCK: usize = 192;
        let thread = thread::index_1d().get() as usize;
        let row = thread / 32;
        let lane = thread % 32;
        let h = hc_fn.len() / (MIX * HC);
        let layout_valid = h != 0
            && hc_fn.len() % (MIX * HC) == 0
            && input.len() % (HC * h) == 0
            && mixes.len() == input.len() / (HC * h) * MIX;
        let row_valid = layout_valid && row < input.len() / (HC * h);
        let lane_zero_inv = if row_valid && lane == 0 {
            let start = row * HC * h;
            let square = pytorch_square_sum_bf16(input, start, HC * h);
            1.0 / sqrt_newton(square / (HC * h) as f32 + EPS)
        } else {
            0.0
        };
        let inv = warp::shuffle_f32(lane_zero_inv, 0);
        if !row_valid || lane >= MIX {
            return;
        }
        let start = row * HC * h;
        let mut dot = 0.0;
        let mut block = 0;
        while block < HC * h {
            let mut partial = 0.0;
            let end = (block + K_BLOCK).min(HC * h);
            for i in block..end {
                partial = cuda_device::float::fma_rn_f32(
                    bf16_to_f32(input[start + i]),
                    hc_fn[lane * HC * h + i],
                    partial,
                );
            }
            dot += partial;
            block += K_BLOCK;
        }
        // SAFETY: the layout check above proves this output row exists.
        unsafe { *mixes.as_mut_ptr().add(row * MIX + lane) = dot * inv }
    }

    /// Flash cuBLAS arm post-GEMM scale.  The fixed 16,384-wide reduction
    /// above owns the reciprocal square root; this kernel only applies it to
    /// the 24 already-computed raw dots.
    #[kernel]
    pub fn mhc_scale_raw_dot_by_rsqrt_v4(
        raw_dot: &[f32],
        rsqrt: &[f32],
        mut mixes: DisjointSlice<f32>,
    ) {
        const MIX: usize = 24;
        let row = thread::index_1d().get() as usize;
        if raw_dot.len() % MIX != 0 {
            return;
        }
        let rows = raw_dot.len() / MIX;
        if rows == 0 || rsqrt.len() != rows || mixes.len() != raw_dot.len() || row >= rows {
            return;
        }
        let inv = rsqrt[row];
        unsafe {
            for lane in 0..MIX {
                *mixes.as_mut_ptr().add(row * MIX + lane) =
                    cuda_device::float::mul_rn_f32(raw_dot[row * MIX + lane], inv);
            }
        }
    }

    /// Parity-only mHC normalization probe.  This runs after the committed
    /// scale kernel and writes no tensor consumed by inference.
    #[kernel]
    pub fn mhc_rsqrt_debug_v4(input: &[u16], mut debug: DisjointSlice<u32>) {
        const HC: usize = 4;
        const EPS: f32 = 1e-6;
        const FIELDS: usize = 5;
        let row = thread::index_1d().get() as usize;
        if input.len() % HC != 0 || debug.len() % FIELDS != 0 {
            return;
        }
        let rows = debug.len() / FIELDS;
        if rows == 0 || input.len() % (rows * HC) != 0 || row >= rows {
            return;
        }
        let width = input.len() / rows;
        let square = pytorch_square_sum_bf16(input, row * width, width);
        let mean_eps = square / width as f32 + EPS;
        let current_newton = 1.0 / sqrt_newton(mean_eps);
        // cuda-oxide lowers `core::f32::math::sqrt` to libdevice
        // `__nv_sqrtf`; the inherent `f32::sqrt` resolves through `std` here.
        let libdevice_sqrt_value = core::f32::math::sqrt(mean_eps);
        let libdevice_sqrt = 1.0 / libdevice_sqrt_value;
        let libdevice_rsqrt = mhc_committed_rsqrt(mean_eps);
        unsafe {
            let base = row * FIELDS;
            *debug.as_mut_ptr().add(base) = square.to_bits();
            *debug.as_mut_ptr().add(base + 1) = mean_eps.to_bits();
            *debug.as_mut_ptr().add(base + 2) = current_newton.to_bits();
            *debug.as_mut_ptr().add(base + 3) = libdevice_sqrt.to_bits();
            *debug.as_mut_ptr().add(base + 4) = libdevice_rsqrt.to_bits();
        }
    }

    /// mHC's Sinkhorn/collapse stage. Trace order is logits, softmax, then
    /// every column and (except the final iteration) row normalization.
    #[kernel]
    pub fn mhc_sinkhorn_v4_bf16(
        input: &[u16],
        mixes: &[f32],
        hc_scale: &[f32],
        hc_base: &[f32],
        mut pre: DisjointSlice<f32>,
        mut post: DisjointSlice<f32>,
        mut comb: DisjointSlice<f32>,
        mut collapsed: DisjointSlice<u16>,
        mut trace: DisjointSlice<f32>,
    ) {
        const HC: usize = 4;
        const MIX: usize = 24;
        const EPS: f32 = 1e-6;
        let row = thread::index_1d().get() as usize;
        if mixes.len() % MIX != 0 {
            return;
        }
        let rows = mixes.len() / MIX;
        if rows == 0 || input.len() % (rows * HC) != 0 {
            return;
        }
        let h = input.len() / (rows * HC);
        if row >= rows
            || hc_scale.len() != 3
            || hc_base.len() != MIX
            || pre.len() != input.len() / h
            || post.len() != input.len() / h
            || comb.len() != rows * HC * HC
            || collapsed.len() != input.len() / HC
            || trace.len() != rows * 41 * HC * HC
        {
            return;
        }
        let start = row * HC * h;
        let mix = row * MIX;
        let base = row * HC * HC;
        let trace_base = row * 41 * HC * HC;
        let mut local_pre = [0.0; HC];
        for i in 0..HC {
            local_pre[i] =
                1.0 / (1.0 + libm::expf(-(mixes[mix + i] * hc_scale[0] + hc_base[i]))) + EPS;
            // SAFETY: checked row-major output layouts above.
            unsafe {
                *pre.as_mut_ptr().add(row * HC + i) = local_pre[i];
                *post.as_mut_ptr().add(row * HC + i) = 2.0
                    / (1.0 + libm::expf(-(mixes[mix + HC + i] * hc_scale[1] + hc_base[HC + i])));
            }
        }
        for i in 0..HC {
            let mut max = f32::NEG_INFINITY;
            for j in 0..HC {
                let value =
                    mixes[mix + 2 * HC + i * HC + j] * hc_scale[2] + hc_base[2 * HC + i * HC + j];
                unsafe { *comb.as_mut_ptr().add(base + i * HC + j) = value }
                max = max.max(value);
            }
            for j in 0..HC {
                unsafe {
                    *trace.as_mut_ptr().add(trace_base + i * HC + j) =
                        *comb.as_mut_ptr().add(base + i * HC + j)
                }
            }
            let mut values = [0.0; HC];
            for j in 0..HC {
                let value = libm::expf(unsafe { *comb.as_mut_ptr().add(base + i * HC + j) } - max);
                unsafe { *comb.as_mut_ptr().add(base + i * HC + j) = value }
                values[j] = value;
            }
            let sum = tilelang_sum4!(values[0], values[1], values[2], values[3]);
            for j in 0..HC {
                unsafe {
                    let value = comb.as_mut_ptr().add(base + i * HC + j);
                    *value = *value / sum + EPS;
                    *trace.as_mut_ptr().add(trace_base + HC * HC + i * HC + j) = *value;
                }
            }
        }
        for step in 0..20 {
            for j in 0..HC {
                let mut values = [0.0; HC];
                for i in 0..HC {
                    values[i] = unsafe { *comb.as_mut_ptr().add(base + i * HC + j) };
                }
                let sum = tilelang_sum4!(values[0], values[1], values[2], values[3]);
                for i in 0..HC {
                    unsafe {
                        let value = comb.as_mut_ptr().add(base + i * HC + j);
                        *value = *value / (sum + EPS);
                    }
                }
            }
            for i in 0..HC {
                for j in 0..HC {
                    unsafe {
                        *trace
                            .as_mut_ptr()
                            .add(trace_base + (2 + step * 2) * HC * HC + i * HC + j) =
                            *comb.as_mut_ptr().add(base + i * HC + j)
                    }
                }
            }
            if step != 19 {
                for i in 0..HC {
                    let mut values = [0.0; HC];
                    for j in 0..HC {
                        values[j] = unsafe { *comb.as_mut_ptr().add(base + i * HC + j) };
                    }
                    let sum = tilelang_sum4!(values[0], values[1], values[2], values[3]);
                    for j in 0..HC {
                        unsafe {
                            let value = comb.as_mut_ptr().add(base + i * HC + j);
                            *value = *value / (sum + EPS);
                        }
                    }
                }
                for i in 0..HC {
                    for j in 0..HC {
                        unsafe {
                            *trace
                                .as_mut_ptr()
                                .add(trace_base + (3 + step * 2) * HC * HC + i * HC + j) =
                                *comb.as_mut_ptr().add(base + i * HC + j)
                        }
                    }
                }
            }
        }
        for col in 0..h {
            // PyTorch's `torch.sum(..., dim=2)` at the pinned four-lane
            // boundary is a source-lane serial reduction.  Keep each product
            // and add explicitly rounded so cuda-oxide cannot contract the
            // cross-lane TileLang tree into FMAs.
            let p0 = cuda_device::float::mul_rn_f32(local_pre[0], bf16_to_f32(input[start + col]));
            let p1 =
                cuda_device::float::mul_rn_f32(local_pre[1], bf16_to_f32(input[start + h + col]));
            let p2 = cuda_device::float::mul_rn_f32(
                local_pre[2],
                bf16_to_f32(input[start + 2 * h + col]),
            );
            let p3 = cuda_device::float::mul_rn_f32(
                local_pre[3],
                bf16_to_f32(input[start + 3 * h + col]),
            );
            let p01 = cuda_device::float::add_rn_f32(p0, p1);
            let p012 = cuda_device::float::add_rn_f32(p01, p2);
            let value = cuda_device::float::add_rn_f32(p012, p3);
            unsafe { *collapsed.as_mut_ptr().add(row * h + col) = f32_to_bf16_rne(value) }
        }
    }

    #[kernel]
    pub fn select_v4_router(
        logits: &[f32],
        bias: &[f32],
        hash_ids: &[i64],
        mut output_ids: DisjointSlice<i32>,
        mut output_weights: DisjointSlice<f32>,
        mut trace_softplus: DisjointSlice<f32>,
        mut trace_post_scores: DisjointSlice<f32>,
        mut trace_selected_scores: DisjointSlice<f32>,
        mut trace_denom_steps: DisjointSlice<f32>,
        hash_mode: u32,
        experts: u32,
        route_scale: f32,
        score_arm: u32,
    ) {
        const TOPK: usize = 6;
        let experts = experts as usize;
        let row = thread::index_1d().get() as usize;
        let start = row * experts;
        let output_start = row * TOPK;
        // All trace buffers are either all zero-length (normal execution) or
        // complete row-major buffers.  The trace is written from this primary
        // selection kernel so it cannot be confused with a parity recompute.
        let trace = trace_softplus.len() == logits.len()
            && trace_post_scores.len() == logits.len()
            && trace_selected_scores.len() == output_weights.len()
            && trace_denom_steps.len() == output_weights.len();
        if experts == 0
            || start + experts > logits.len()
            || output_start + TOPK > output_ids.len()
            || output_start + TOPK > output_weights.len()
            || bias.len() != experts
            || (hash_mode != 0 && output_start + TOPK > hash_ids.len())
        {
            return;
        }

        let mut ids = [0_i32; TOPK];
        let mut raw = [0.0; TOPK];
        if hash_mode != 0 {
            for slot in 0..TOPK {
                let id = hash_ids[output_start + slot];
                if id < 0 || id >= experts as i64 {
                    return;
                }
                ids[slot] = id as i32;
                let softplus = router_softplus(logits[start + id as usize], score_arm);
                raw[slot] = if score_arm != 0 {
                    router_accurate_sqrt(softplus)
                } else {
                    sqrt_newton(softplus)
                };
                if trace {
                    unsafe {
                        *trace_softplus.as_mut_ptr().add(start + id as usize) = softplus;
                        *trace_post_scores.as_mut_ptr().add(start + id as usize) = raw[slot];
                    }
                }
            }
        } else {
            let mut selected = [f32::NEG_INFINITY; TOPK];
            for expert in 0..experts {
                let softplus = router_softplus(logits[start + expert], score_arm);
                let candidate_raw = if score_arm != 0 {
                    router_accurate_sqrt(softplus)
                } else {
                    sqrt_newton(softplus)
                };
                if trace {
                    unsafe {
                        *trace_softplus.as_mut_ptr().add(start + expert) = softplus;
                        *trace_post_scores.as_mut_ptr().add(start + expert) = candidate_raw;
                    }
                }
                let candidate = candidate_raw + bias[expert];
                let mut slot = 0;
                while slot < TOPK && candidate <= selected[slot] {
                    slot += 1;
                }
                if slot < TOPK {
                    let mut tail = TOPK - 1;
                    while tail > slot {
                        selected[tail] = selected[tail - 1];
                        ids[tail] = ids[tail - 1];
                        raw[tail] = raw[tail - 1];
                        tail -= 1;
                    }
                    selected[slot] = candidate;
                    ids[slot] = expert as i32;
                    raw[slot] = candidate_raw;
                }
            }
        }
        // Pinned official route denominator tree, not a generic left fold.
        let left_04 = raw[0] + raw[4];
        let left = left_04 + raw[2];
        let right_15 = raw[1] + raw[5];
        let right = right_15 + raw[3];
        let sum = left + right;
        if trace {
            let tree = [left_04, left, right_15, right, sum, sum];
            for slot in 0..TOPK {
                unsafe {
                    *trace_selected_scores.as_mut_ptr().add(output_start + slot) = raw[slot];
                    *trace_denom_steps.as_mut_ptr().add(output_start + slot) = tree[slot];
                }
            }
        }
        for slot in 0..TOPK {
            // SAFETY: the output bounds above prove both row-local indices.
            unsafe {
                *output_ids.as_mut_ptr().add(output_start + slot) = ids[slot];
                *output_weights.as_mut_ptr().add(output_start + slot) =
                    raw[slot] / sum * route_scale;
            }
        }
    }

    /// Parity-only router score dump.  This never feeds routing or model state.
    #[kernel]
    pub fn router_post_score_v4(logits: &[f32], mut scores: DisjointSlice<f32>, score_arm: u32) {
        let index = thread::index_1d().get() as usize;
        if index >= logits.len() || scores.len() != logits.len() {
            return;
        }
        unsafe { *scores.as_mut_ptr().add(index) = router_sqrtsoftplus(logits[index], score_arm) }
    }

    /// Parity-only selected-score trace in the already committed route order.
    #[kernel]
    pub fn router_selected_score_denom_v4(
        scores: &[f32],
        ids: &[i32],
        mut selected_scores: DisjointSlice<f32>,
        mut denoms: DisjointSlice<f32>,
        experts: u32,
    ) {
        const TOPK: usize = 6;
        let experts = experts as usize;
        let row = thread::index_1d().get() as usize;
        let start = row * experts;
        let output_start = row * TOPK;
        if experts == 0
            || start + experts > scores.len()
            || output_start + TOPK > ids.len()
            || output_start + TOPK > selected_scores.len()
            || row >= denoms.len()
        {
            return;
        }
        let mut denom = 0.0_f32;
        for slot in 0..TOPK {
            let id = ids[output_start + slot];
            if id < 0 || id as usize >= experts {
                return;
            }
            let score = scores[start + id as usize];
            denom += score;
            unsafe { *selected_scores.as_mut_ptr().add(output_start + slot) = score }
        }
        unsafe { *denoms.as_mut_ptr().add(row) = denom }
    }
}

fn main() {
    if run_frozen_alpaca_tokenizer_preflight().expect("run frozen Alpaca tokenizer-only preflight")
    {
        return;
    }
    if run_stationary_mtp1_scratch_constructor_preflight()
        .expect("run stationary MTP1 scratch constructor preflight")
    {
        return;
    }
    if run_stationary_mtp1_capacity_preflight().expect("run stationary MTP1 capacity preflight") {
        return;
    }
    let benchmark_requested = std::env::var_os("COLIBRI_V4_B2_LAYER_STATIONARY_BENCH").is_some()
        || std::env::var_os("COLIBRI_V4_B4_LAYER_STATIONARY_BENCH").is_some()
        || std::env::var_os("COLIBRI_V4_B4_FULL61_PROMOTION").is_some()
        || std::env::var_os("COLIBRI_V4_B4_GREEDY_ABBA_BENCH").is_some()
        || std::env::var_os("COLIBRI_V4_FULL61_B1").is_some()
        || std::env::var_os("COLIBRI_V4_PHASE_PROFILE").is_some()
        || std::env::var_os("COLIBRI_V4_LAYER_STATIONARY_FIXED_PREFETCH").is_some()
        || std::env::var_os("COLIBRI_V4_FINAL_HEAD_BENCH").is_some()
        || std::env::var_os("COLIBRI_V4_WAVEFRONT_ATTENTION_BENCH").is_some()
        || std::env::var_os("COLIBRI_V4_DEVICE_ATTENTION_PREFLIGHT_GATE").is_some()
        || std::env::var_os("COLIBRI_V4_DEVICE_TOKEN_STATE_GATE").is_some()
        || std::env::var_os("COLIBRI_V4_SPARSE_TOPK_SWEEP").is_some()
        || std::env::var_os("COLIBRI_V4_CACHE_OFF_BENCH").is_some()
        || std::env::var_os("COLIBRI_V4_QD4_STATIONARY_GATE").is_some()
        || std::env::var_os("COLIBRI_V4_QD4_IMMUTABLE_CACHE_TARGETED_GATE").is_some()
        || std::env::var_os("COLIBRI_V4_SPARSE_K128_MICROGATE").is_some()
        || std::env::var_os("COLIBRI_V4_TARGETED_ATTENTION_AUDIT").is_some()
        || std::env::var_os("COLIBRI_V4_PINNED_H2D_SERVICE_GATE").is_some()
        || std::env::var_os("COLIBRI_V4_PINNED_FIXED_WORKER_BOUNDARY_GATE").is_some()
        || std::env::var_os("COLIBRI_V4_CERTIFIED_ARGMAX_AUDIT").is_some()
        || std::env::var_os("COLIBRI_V4_DUAL_BENCH_ONLY").is_some();
    let mut full61_preflight = (std::env::var_os("COLIBRI_V4_FULL61_B1").is_some()
        || std::env::var_os("COLIBRI_V4_PHASE_PROFILE").is_some()
        || std::env::var_os("COLIBRI_V4_LAYER_STATIONARY_FIXED_PREFETCH").is_some()
        || std::env::var_os("COLIBRI_V4_CERTIFIED_ARGMAX_AUDIT").is_some())
    .then(Full61Preflight::from_env)
    .transpose()
    .expect("validate V4 full61 configuration");
    if optional_one_flag("COLIBRI_V4_SHORT_DECODE_STATIC_CHECK")
        .expect("validate short decode static-check flag")
    {
        short_device_decode_static_check().expect("validate short decode static contract");
        return;
    }
    if benchmark_requested {
        benchmark_provenance().expect("validate benchmark source identity");
    }
    if optional_one_flag("COLIBRI_V4_SPARSE_K128_MICROGATE")
        .expect("validate sparse K128 microgate flag")
    {
        sparse_attention_score_reuse_k128_attention_post_microgate();
        return;
    }
    validate_flash_parity_stop_mode().expect("validate Flash parity L0 stop mode");
    if optional_one_flag("COLIBRI_V4_FLASH_FINAL_HEAD_REPLAY")
        .expect("validate Flash final-head replay flag")
    {
        if std::env::var_os("COLIBRI_V4_FLASH_BASE").is_some()
            || std::env::var_os("COLIBRI_V4_FLASH_DSPARK").is_some()
            || std::env::var_os("COLIBRI_V4_FLASH_SERVE").is_some()
        {
            panic!("Flash final-head replay cannot be combined with a runtime mode");
        }
        run_flash_final_head_replay().expect("run Flash final-head replay");
        return;
    }
    // ponytail: this smoke uses synchronous safe copies for a terminal parity
    // check; switch to event-owned async buffers when production dispatch lands.
    let context = CudaContext::new(0).expect("create CUDA context");
    if optional_one_flag("COLIBRI_V4_PINNED_H2D_SERVICE_GATE")
        .expect("validate pinned H2D service-gate flag")
    {
        pinned_h2d_service_gate(&context).expect("run pinned H2D service gate");
        return;
    }
    let stream = context.default_stream();
    let module = kernels::load(&context).expect("load embedded Rust PTX");

    if optional_one_flag("COLIBRI_V4_FLASH_SERVE").expect("validate Flash-0731 serve flag") {
        run_real_v4_flash_serve(&context, &module, &stream)
            .expect("run DeepSeek V4 Flash-0731 local service");
        return;
    }

    if optional_one_flag("COLIBRI_V4_FLASH_DSPARK").expect("validate Flash-0731 DSpark flag") {
        run_real_v4_flash_dspark(&context, &module, &stream)
            .expect("run DeepSeek V4 Flash-0731 full DSpark generation");
        return;
    }

    if optional_one_flag("COLIBRI_V4_FLASH_BASE").expect("validate Flash-0731 base flag") {
        run_real_v4_flash_base(&context, &module, &stream)
            .expect("run DeepSeek V4 Flash-0731 base-only generation");
        return;
    }

    if optional_one_flag("COLIBRI_V4_PINNED_FIXED_WORKER_BOUNDARY_GATE")
        .expect("validate pinned fixed worker boundary-gate flag")
    {
        run_pinned_fixed_worker_boundary_gate(&context, &module, &stream)
            .expect("run pinned fixed worker boundary gate");
        return;
    }

    if std::env::var_os("COLIBRI_V4_FINAL_HEAD_BENCH").as_deref() == Some(std::ffi::OsStr::new("1"))
    {
        run_real_v4_final_head_bench(&context, &module, &stream)
            .expect("run real V4 final-head Stage1 benchmark");
        return;
    }

    if std::env::var_os("COLIBRI_V4_WAVEFRONT_ATTENTION_BENCH").as_deref()
        == Some(std::ffi::OsStr::new("1"))
    {
        run_real_v4_wavefront_attention_bench(&context, &module, &stream)
            .expect("run real V4 CDQW attention benchmark");
        return;
    }

    if std::env::var_os("COLIBRI_V4_DEVICE_ATTENTION_PREFLIGHT_GATE").as_deref()
        == Some(std::ffi::OsStr::new("1"))
    {
        run_device_attention_preflight_gate(&context, &module, &stream)
            .expect("run device-attention capacity preflight gate");
        return;
    }

    if std::env::var_os("COLIBRI_V4_DEVICE_TOKEN_STATE_GATE").as_deref()
        == Some(std::ffi::OsStr::new("1"))
    {
        run_real_v4_device_token_state_gate(&context, &module, &stream)
            .expect("run real V4 device token-state pos3 gate");
        return;
    }

    if std::env::var_os("COLIBRI_V4_QD4_STATIONARY_GATE").as_deref()
        == Some(std::ffi::OsStr::new("1"))
    {
        run_qd4_stationary_gate(&context, &module, &stream).expect("run V4 QD4 stationary gate");
        return;
    }

    if std::env::var_os("COLIBRI_V4_QD4_IMMUTABLE_CACHE_TARGETED_GATE").as_deref()
        == Some(std::ffi::OsStr::new("1"))
    {
        run_qd4_immutable_cache_targeted_gate(&context, &module, &stream)
            .expect("run V4 QD4 immutable-cache targeted gate");
        return;
    }

    if std::env::var_os("COLIBRI_V4_SPARSE_TOPK_SWEEP").as_deref()
        == Some(std::ffi::OsStr::new("1"))
    {
        run_sparse_attention_topk_sweep(&context, &module, &stream)
            .expect("run sparse-attention top-k crossover sweep");
        return;
    }

    if std::env::var_os("COLIBRI_V4_CACHE_OFF_BENCH").as_deref() == Some(std::ffi::OsStr::new("1"))
    {
        run_real_v4_cache_off_bench().expect("run real V4 cache-off Stage0 benchmark");
        return;
    }

    if std::env::var_os("COLIBRI_V4_B2_LAYER_STATIONARY_BENCH").as_deref()
        == Some(std::ffi::OsStr::new("1"))
    {
        run_real_v4_b2_layer_stationary(&context, &module, &stream)
            .expect("run real V4 B2 layer-stationary benchmark");
        return;
    }

    if optional_one_flag("COLIBRI_V4_B2_GENERATE").expect("validate V4 B2 generate flag") {
        run_real_v4_b2_generate::<2>(
            &context,
            &module,
            &stream,
            "COLIBRI_V4_B2_TOKEN_IDS",
            "v4-b2-greedy-rowset",
            "colibri-v4-b2-greedy-rowset-v1",
        )
        .expect("run real V4 B2 greedy generation");
        return;
    }

    if optional_one_flag("COLIBRI_V4_B4_GREEDY_ABBA_BENCH")
        .expect("validate V4 B4 greedy ABBA flag")
    {
        run_real_v4_b4_greedy_abba_bench(&context, &module, &stream)
            .expect("run real V4 B4 greedy ABBA benchmark");
        return;
    }

    if optional_one_flag("COLIBRI_V4_B4_GENERATE").expect("validate V4 B4 generate flag") {
        run_real_v4_b2_generate::<4>(
            &context,
            &module,
            &stream,
            "COLIBRI_V4_B4_TOKEN_IDS",
            "v4-b4-greedy-rowset",
            "colibri-v4-b4-greedy-rowset-v1",
        )
        .expect("run real V4 B4 greedy generation");
        return;
    }

    if std::env::var_os("COLIBRI_V4_B4_LAYER_STATIONARY_BENCH").as_deref()
        == Some(std::ffi::OsStr::new("1"))
    {
        run_real_v4_b4_layer_stationary(&context, &module, &stream, false)
            .expect("run real V4 B4 layer-stationary benchmark");
        return;
    }

    if std::env::var_os("COLIBRI_V4_B4_FULL61_PROMOTION").as_deref()
        == Some(std::ffi::OsStr::new("1"))
    {
        run_real_v4_b4_layer_stationary(&context, &module, &stream, true)
            .expect("run real V4 B4 full61 layer-stationary promotion benchmark");
        return;
    }

    if std::env::var_os("COLIBRI_V4_FULL61_B1").is_some() {
        run_real_v4_full61_b1(
            &context,
            &module,
            &stream,
            full61_preflight
                .take()
                .expect("V4 full61 preflight disappeared"),
        )
        .expect("run real V4 full61 B=1 inference");
        return;
    }

    if std::env::var_os("COLIBRI_V4_DUAL_BENCH_ONLY").is_some() {
        let config = DualBenchConfig::from_env().expect("validate netem benchmark configuration");
        let path = v4_expert_shard().expect("validate pinned V4 expert shard");
        let shard = safetensors::ShardFile::open(&path).expect("open V4 expert shard");
        run_v4_two_worker_loopback(&context, &module, &shard, &path, Some(&config))
            .expect("run V4 dual-worker benchmark");
        println!("PASS: V4 dual-worker benchmark-only entry");
        return;
    }

    for &length in &[0usize, 1, 17, 4_097] {
        let input: Vec<f32> = (0..length).map(|i| i as f32 * 0.25 - 1.0).collect();
        let rhs: Vec<f32> = (0..length)
            .map(|i| ((i % 7) as i32 - 3) as f32 * 0.5)
            .collect();
        let expected: Vec<f32> = input.iter().zip(&rhs).map(|(x, y)| x + y).collect();
        if length == 0 {
            assert_close("add_in_place", &input, &expected);
            continue;
        }

        let mut device_input = DeviceBuffer::from_host(&stream, &input).expect("upload add input");
        let device_rhs = DeviceBuffer::from_host(&stream, &rhs).expect("upload add rhs");
        // SAFETY: the 1-D launch covers length elements and the kernel
        // bounds-checks before accessing either equally sized device buffer.
        unsafe {
            module.add_in_place(
                &stream,
                LaunchConfig::for_num_elems(length as u32),
                &mut device_input,
                &device_rhs,
            )
        }
        .expect("launch add_in_place");
        let actual = device_input
            .to_host_vec(&stream)
            .expect("download add result");
        assert_close("add_in_place", &actual, &expected);
    }

    for &length in &[0usize, 1, 17, 4_097] {
        let gate: Vec<f32> = (0..length).map(|i| i as f32 * 0.25 - 2.0).collect();
        let up: Vec<f32> = (0..length)
            .map(|i| ((i % 5) as i32 - 2) as f32 * 0.75)
            .collect();
        let expected: Vec<f32> = gate
            .iter()
            .zip(&up)
            .map(|(gate, up)| (gate / (1.0 + (-gate).exp())) * up)
            .collect();
        if length == 0 {
            assert_close("silu_mul_in_place", &gate, &expected);
            continue;
        }

        let mut device_gate = DeviceBuffer::from_host(&stream, &gate).expect("upload silu gate");
        let device_up = DeviceBuffer::from_host(&stream, &up).expect("upload silu up");
        // SAFETY: the 1-D launch covers length elements and the kernel
        // bounds-checks before accessing either equally sized device buffer.
        unsafe {
            module.silu_mul_in_place(
                &stream,
                LaunchConfig::for_num_elems(length as u32),
                &mut device_gate,
                &device_up,
            )
        }
        .expect("launch silu_mul_in_place");
        let actual = device_gate
            .to_host_vec(&stream)
            .expect("download silu result");
        assert_close("silu_mul_in_place", &actual, &expected);
    }

    let (row_count, d) = (8usize, 257usize);
    let rows = [6i32, 1, 4];
    let input: Vec<f32> = (0..row_count * d).map(|i| i as f32 * 0.125 - 3.0).collect();
    let partial: Vec<f32> = (0..rows.len() * d)
        .map(|i| ((i % 11) as i32 - 5) as f32 * 0.25)
        .collect();
    let mut expected = input.clone();
    for (partial_row, &row) in rows.iter().enumerate() {
        for column in 0..d {
            expected[row as usize * d + column] += partial[partial_row * d + column];
        }
    }
    let mut device_input = DeviceBuffer::from_host(&stream, &input).expect("upload rows add input");
    let device_partial =
        DeviceBuffer::from_host(&stream, &partial).expect("upload rows add partial");
    let device_rows = DeviceBuffer::from_host(&stream, &rows).expect("upload rows add indices");
    launch_rows_add(
        &module,
        &stream,
        &mut device_input,
        &device_partial,
        &device_rows,
        &rows,
        d,
    )
    .expect("launch rows_add_in_place");
    let actual = device_input
        .to_host_vec(&stream)
        .expect("download rows add result");
    assert_close("rows_add_in_place", &actual, &expected);

    let packed: Vec<u8> = (0..64)
        .map(|index| [0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe][index % 8])
        .collect();
    let scales = vec![126u8, 128, 127, 129];
    let layout =
        v4_fp4::Fp4E2M1x2Layout::new(&packed, &scales, 2, 64).expect("validate V4 FP4 fixture");
    let mut expected = vec![0.0; layout.element_len()];
    for row in 0..2 {
        for block in 0..2 {
            layout
                .decode_block(row, block, &mut expected[row * 64 + block * 32..][..32])
                .expect("decode validated V4 FP4 fixture block");
        }
    }
    let device_packed = DeviceBuffer::from_host(&stream, &packed).expect("upload V4 FP4 weights");
    let device_scales = DeviceBuffer::from_host(&stream, &scales).expect("upload V4 FP4 scales");
    let mut device_output = DeviceBuffer::from_host(&stream, &vec![0.0f32; expected.len()])
        .expect("allocate V4 FP4 output");
    launch_v4_fp4_decode(
        &module,
        &stream,
        &device_packed,
        &device_scales,
        &mut device_output,
        layout,
    )
    .expect("launch V4 FP4 format oracle");
    let actual = device_output
        .to_host_vec(&stream)
        .expect("download V4 FP4 output");
    assert_close("decode_v4_fp4_blockwise", &actual, &expected);
    if synthetic_only().expect("validate COLIBRI_SYNTHETIC_ONLY") {
        println!("SKIP: explicit COLIBRI_SYNTHETIC_ONLY=1 disables all real V4 checks");
    } else {
        run_real_v4_fp4_slice_if_available(&module, &stream).expect("real V4 FP4 range check");
        run_real_v4_router_slice_if_available(&module, &stream)
            .expect("real V4 router range check");
        run_real_v4_state_slice_if_available(&module, &stream).expect("real V4 state range check");
        run_real_v4_fp8_projection_slice_if_available(&context, &module, &stream)
            .expect("real V4 FP8 projection check");
        run_real_v4_compressor_slice_if_available(&module, &stream)
            .expect("real V4 Compressor check");
        run_real_v4_expert0_slice_if_available(&context, &module, &stream)
            .expect("real V4 routed expert parity check");
    }

    println!("PASS: Rust CUDA primitives, V4 FP4/router, and routed expert-0 parity");
}

struct StreamedFinalHeadRun {
    output: v4_state::FinalHeadOutput,
    sumabs_f32: Vec<f32>,
    status: Vec<u32>,
    read_ns: u64,
    h2d_ns: u64,
    kernel_ns: u64,
    d2h_ns: u64,
    cpu_preprocess_ns: u64,
    wall_ns: u64,
}

struct StreamedLmHeadNormalizedRun {
    candidate: usize,
    logits_f32: Vec<f32>,
    sumabs_f32: Vec<f32>,
    status: Vec<u32>,
    read_ns: u64,
    h2d_ns: u64,
    kernel_ns: u64,
    d2h_ns: u64,
    cpu_preprocess_ns: u64,
    wall_ns: u64,
}

fn f32_ulp_distance(left: f32, right: f32) -> u32 {
    let ordered = |value: f32| {
        let bits = value.to_bits();
        if bits & 0x8000_0000 == 0 {
            bits | 0x8000_0000
        } else {
            !bits
        }
    };
    ordered(left).abs_diff(ordered(right))
}

fn final_head_comparison(actual: &[f32], expected: &[f32]) -> (usize, f32, f32, u32) {
    let mut mismatches = 0;
    let mut max_abs = 0.0_f32;
    let mut max_rel = 0.0_f32;
    let mut max_ulp = 0_u32;
    for (&actual, &expected) in actual.iter().zip(expected) {
        if !actual.is_finite() || !expected.is_finite() {
            mismatches += 1;
            max_abs = f32::INFINITY;
            max_rel = f32::INFINITY;
            max_ulp = u32::MAX;
            continue;
        }
        let difference = (actual - expected).abs();
        max_abs = max_abs.max(difference);
        max_rel = max_rel.max(difference / expected.abs().max(f32::MIN_POSITIVE));
        max_ulp = max_ulp.max(f32_ulp_distance(actual, expected));
        mismatches += usize::from(difference > 2e-4 * expected.abs().max(1.0));
    }
    (mismatches, max_abs, max_rel, max_ulp)
}

fn run_real_v4_final_head_bench(
    context: &std::sync::Arc<CudaContext>,
    module: &kernels::LoadedModule,
    stream: &std::sync::Arc<CudaStream>,
) -> Result<(), Box<dyn Error>> {
    let repeats = std::env::var("COLIBRI_V4_FINAL_HEAD_REPEATS")
        .unwrap_or_else(|_| "3".into())
        .parse::<usize>()?;
    if repeats < 3 {
        return Err("COLIBRI_V4_FINAL_HEAD_REPEATS must be at least 3".into());
    }
    let layer_baseline_ns = std::env::var("COLIBRI_V4_FINAL_HEAD_BASELINE_LAYER_NS")
        .map_err(|_| "set COLIBRI_V4_FINAL_HEAD_BASELINE_LAYER_NS for the request projection")?
        .parse::<u64>()?;
    let result_path = std::env::var_os("COLIBRI_V4_FINAL_HEAD_RESULT_JSONL")
        .map(std::path::PathBuf::from)
        .ok_or("set COLIBRI_V4_FINAL_HEAD_RESULT_JSONL")?;
    let root = v4_model_dir()?;
    let embedding = v4_state::load(&root, &[0])?.embedding;
    let mut hc_hidden = Vec::with_capacity(v4_state::HC * v4_state::HIDDEN);
    for _ in 0..v4_state::HC {
        hc_hidden.extend_from_slice(&embedding);
    }
    let weights = v4_state::load_final_head(&root)?;
    let source_bytes = v4_state::VOCAB * v4_state::HIDDEN * 2;
    let mut result = std::fs::File::create(&result_path)?;
    writeln!(
        result,
        "{}",
        serde_json::json!({
            "event": "run",
            "schema": "colibri-v4-final-head-stage1-v1",
            "provenance": benchmark_provenance()?,
            "token_e2e_ns": serde_json::Value::Null,
            "request_e2e_ns": serde_json::Value::Null,
            "model_revision": v4_state::REVISION,
            "input": "official token-0 embedding repeated across HC=4",
            "head_dtype": "BF16",
            "head_shape": [v4_state::VOCAB, v4_state::HIDDEN],
            "source_bytes": source_bytes,
            "gpu_chunk_rows": 1024,
            "repeats": repeats,
            "cache_protocol": "repeat 0: POSIX_FADV_DONTNEED before each CPU/GPU arm; later repeats: warm",
            "baseline_layer_ns": layer_baseline_ns,
        })
    )?;
    let mut cpu_wall = Vec::with_capacity(repeats);
    let mut gpu_wall = Vec::with_capacity(repeats);
    let mut all_canonical = true;
    for repeat in 0..repeats {
        let cache = if repeat == 0 { "fadvise_cold" } else { "warm" };
        if repeat == 0 {
            weights.advise_head_dontneed()?;
        }
        let cpu = v4_state::final_checkpoint_head_cpu_timed(&weights, &hc_hidden, 1)?;
        if repeat == 0 {
            weights.advise_head_dontneed()?;
        }
        let gpu = run_streamed_final_head_gpu(context, module, stream, &weights, &hc_hidden)?;
        let (mismatches, max_abs, max_rel, max_ulp) =
            final_head_comparison(&gpu.output.logits_f32, &cpu.output.logits_f32);
        let argmax_exact = gpu.output.argmax == cpu.output.argmax;
        all_canonical &= mismatches == 0 && argmax_exact;
        cpu_wall.push(cpu.wall_ns);
        gpu_wall.push(gpu.wall_ns);
        writeln!(
            result,
            "{}",
            serde_json::json!({
                "event": "repeat",
                "repeat": repeat,
                "cache": cache,
                "cpu": {
                    "wall_ns": cpu.wall_ns,
                    "read_ns": cpu.read_ns,
                    "source_bytes": source_bytes,
                },
                "gpu": {
                    "wall_ns": gpu.wall_ns,
                    "read_ns": gpu.read_ns,
                    "source_bytes": source_bytes,
                    "h2d_ns": gpu.h2d_ns,
                    "h2d_bytes": source_bytes + v4_state::HIDDEN * 2,
                    "kernel_ns": gpu.kernel_ns,
                    "d2h_ns": gpu.d2h_ns,
                    "d2h_bytes": v4_state::VOCAB * (std::mem::size_of::<f32>() * 2 + std::mem::size_of::<u32>()),
                    "cpu_preprocess_ns": gpu.cpu_preprocess_ns,
                },
                "canonical": {
                    "logit_relative_tolerance": 2e-4,
                    "logit_mismatches": mismatches,
                    "max_abs": max_abs,
                    "max_rel": max_rel,
                    "max_ulp": max_ulp,
                    "argmax_exact": argmax_exact,
                    "cpu_argmax": cpu.output.argmax[0],
                    "gpu_argmax": gpu.output.argmax[0],
                },
            })
        )?;
    }
    let warm_cpu = median_u64(&cpu_wall[1..]);
    let warm_gpu = median_u64(&gpu_wall[1..]);
    let final_improvement = 1.0 - warm_gpu as f64 / warm_cpu as f64;
    let baseline_request_ns = layer_baseline_ns.saturating_add(warm_cpu);
    let projected_request_ns = layer_baseline_ns.saturating_add(warm_gpu);
    let request_improvement = 1.0 - projected_request_ns as f64 / baseline_request_ns as f64;
    let pass = all_canonical && final_improvement >= 0.50 && request_improvement >= 0.20;
    writeln!(
        result,
        "{}",
        serde_json::json!({
            "event": "summary",
            "status": if pass { "PASS" } else { "KILL" },
            "canonical": all_canonical,
            "warm_cpu_wall_median_ns": warm_cpu,
            "warm_gpu_wall_median_ns": warm_gpu,
            "final_head_improvement": final_improvement,
            "required_final_head_improvement": 0.50,
            "baseline_request_ns": baseline_request_ns,
            "projected_request_ns": projected_request_ns,
            "projected_request_improvement": request_improvement,
            "required_request_improvement": 0.20,
        })
    )?;
    result.sync_all()?;
    println!(
        "{{\"status\":\"{}\",\"mode\":\"v4-final-head-stage1\",\"final_head_improvement\":{final_improvement:.6},\"projected_request_improvement\":{request_improvement:.6},\"result_jsonl\":{}}}",
        if pass { "PASS" } else { "KILL" },
        serde_json::to_string(&result_path)?,
    );
    Ok(())
}

fn run_streamed_final_head_gpu(
    context: &std::sync::Arc<CudaContext>,
    module: &kernels::LoadedModule,
    stream: &std::sync::Arc<CudaStream>,
    weights: &v4_state::FinalHeadWeights,
    hc_hidden_bf16: &[u16],
) -> Result<StreamedFinalHeadRun, Box<dyn Error>> {
    let wall_started = std::time::Instant::now();
    let cpu_started = std::time::Instant::now();
    let pre = v4_state::final_hc_gate_cpu(
        hc_hidden_bf16,
        1,
        v4_state::HC,
        v4_state::HIDDEN,
        &weights.hc_fn_f32,
        weights.hc_scale_f32,
        &weights.hc_base_f32,
    )?;
    let mut output = v4_state::final_hc_head_cpu(
        hc_hidden_bf16,
        1,
        v4_state::HC,
        v4_state::HIDDEN,
        &pre,
        &weights
            .norm_bf16
            .iter()
            .map(|&bits| v4_router::bf16_to_f32(bits))
            .collect::<Vec<_>>(),
        &vec![0.0; v4_state::HIDDEN],
        1,
    )?;
    let cpu_preprocess_ns = u64::try_from(cpu_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
    let streamed = run_streamed_lm_head_gpu_normalized(
        context,
        module,
        stream,
        weights,
        &output.normalized_bf16,
    )?;
    output.logits_f32 = streamed.logits_f32;
    output.argmax[0] = streamed.candidate;
    Ok(StreamedFinalHeadRun {
        output,
        sumabs_f32: streamed.sumabs_f32,
        status: streamed.status,
        read_ns: streamed.read_ns,
        h2d_ns: streamed.h2d_ns,
        kernel_ns: streamed.kernel_ns,
        d2h_ns: streamed.d2h_ns,
        cpu_preprocess_ns: cpu_preprocess_ns.saturating_add(streamed.cpu_preprocess_ns),
        wall_ns: u64::try_from(wall_started.elapsed().as_nanos()).unwrap_or(u64::MAX),
    })
}

/// The shared streamed LM-head execution after a caller has produced one
/// normalized BF16 row.  Teacher certification and the MTP draft differ only
/// in how that row is formed; this keeps the chunk upload/dot/argmax mechanism
/// single-sourced.
fn run_streamed_lm_head_gpu_normalized(
    context: &std::sync::Arc<CudaContext>,
    module: &kernels::LoadedModule,
    stream: &std::sync::Arc<CudaStream>,
    weights: &v4_state::FinalHeadWeights,
    normalized_bf16: &[u16],
) -> Result<StreamedLmHeadNormalizedRun, Box<dyn Error>> {
    if normalized_bf16.len() != v4_state::HIDDEN {
        return Err("streamed LM head requires one normalized hidden row".into());
    }
    let wall_started = std::time::Instant::now();
    let input_upload_started = std::time::Instant::now();
    let device_input = DeviceBuffer::from_host(stream, normalized_bf16)?;
    let mut h2d_ns = u64::try_from(input_upload_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
    let device_weight =
        DeviceBuffer::<u16>::zeroed(stream, CERTIFIED_ARGMAX_CHUNK_ROWS * v4_state::HIDDEN)?;
    let mut device_output = DeviceBuffer::<f32>::zeroed(stream, CERTIFIED_ARGMAX_CHUNK_ROWS)?;
    let mut device_sumabs = DeviceBuffer::<f32>::zeroed(stream, CERTIFIED_ARGMAX_CHUNK_ROWS)?;
    let mut device_status = DeviceBuffer::<u32>::zeroed(stream, CERTIFIED_ARGMAX_CHUNK_ROWS)?;
    let start = context.new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
    let end = context.new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
    let mut bytes = vec![0_u8; CERTIFIED_ARGMAX_CHUNK_ROWS * v4_state::HIDDEN * 2];
    let mut words = vec![0_u16; CERTIFIED_ARGMAX_CHUNK_ROWS * v4_state::HIDDEN];
    let mut read_ns = 0_u64;
    let mut kernel_ns = 0_u64;
    let mut d2h_ns = 0_u64;
    let mut cpu_preprocess_ns = 0_u64;
    let mut logits_f32 = vec![0.0_f32; v4_state::VOCAB];
    let mut sumabs_f32 = vec![0.0_f32; v4_state::VOCAB];
    let mut status = vec![0_u32; v4_state::VOCAB];
    for token_start in (0..v4_state::VOCAB).step_by(CERTIFIED_ARGMAX_CHUNK_ROWS) {
        let chunk_rows = (v4_state::VOCAB - token_start).min(CERTIFIED_ARGMAX_CHUNK_ROWS);
        let chunk_elements = chunk_rows * v4_state::HIDDEN;
        let chunk_bytes = chunk_elements * 2;
        let read_started = std::time::Instant::now();
        weights.read_head_weight_bytes(token_start, chunk_rows, &mut bytes[..chunk_bytes])?;
        read_ns = read_ns
            .saturating_add(u64::try_from(read_started.elapsed().as_nanos()).unwrap_or(u64::MAX));
        let decode_started = std::time::Instant::now();
        for (word, bytes) in words[..chunk_elements]
            .iter_mut()
            .zip(bytes[..chunk_bytes].chunks_exact(2))
        {
            *word = u16::from_le_bytes([bytes[0], bytes[1]]);
        }
        cpu_preprocess_ns = cpu_preprocess_ns
            .saturating_add(u64::try_from(decode_started.elapsed().as_nanos()).unwrap_or(u64::MAX));
        let upload_started = std::time::Instant::now();
        // SAFETY: the preallocated device buffer is at least `chunk_bytes`
        // long, and the host words remain borrowed until synchronization.
        unsafe {
            cuda_core::memory::memcpy_htod_async(
                device_weight.cu_deviceptr(),
                words.as_ptr(),
                chunk_bytes,
                stream.cu_stream(),
            )?;
        }
        stream.synchronize()?;
        h2d_ns = h2d_ns
            .saturating_add(u64::try_from(upload_started.elapsed().as_nanos()).unwrap_or(u64::MAX));
        start.record(stream)?;
        // SAFETY: one complete warp owns each output row and all buffer
        // extents were checked above.
        unsafe {
            module.matvec_bf16_streamed(
                stream,
                LaunchConfig::for_num_elems(u32::try_from(chunk_rows * 32)?),
                &device_input,
                &device_weight,
                &mut device_output,
                &mut device_sumabs,
                &mut device_status,
                u32::try_from(v4_state::HIDDEN)?,
                u32::try_from(chunk_rows)?,
            )?;
        }
        end.record(stream)?;
        end.synchronize()?;
        kernel_ns = kernel_ns.saturating_add((start.elapsed_ms(&end)? as f64 * 1_000_000.0) as u64);
        let download_started = std::time::Instant::now();
        let chunk = device_output.to_host_vec(stream)?;
        let chunk_sumabs = device_sumabs.to_host_vec(stream)?;
        let chunk_status = device_status.to_host_vec(stream)?;
        d2h_ns = d2h_ns.saturating_add(
            u64::try_from(download_started.elapsed().as_nanos()).unwrap_or(u64::MAX),
        );
        logits_f32[token_start..token_start + chunk_rows].copy_from_slice(&chunk[..chunk_rows]);
        sumabs_f32[token_start..token_start + chunk_rows]
            .copy_from_slice(&chunk_sumabs[..chunk_rows]);
        status[token_start..token_start + chunk_rows].copy_from_slice(&chunk_status[..chunk_rows]);
    }
    let mut candidate = 0_usize;
    for token in 1..v4_state::VOCAB {
        if logits_f32[token] > logits_f32[candidate] {
            candidate = token;
        }
    }
    Ok(StreamedLmHeadNormalizedRun {
        candidate,
        logits_f32,
        sumabs_f32,
        status,
        read_ns,
        h2d_ns,
        kernel_ns,
        d2h_ns,
        cpu_preprocess_ns,
        wall_ns: u64::try_from(wall_started.elapsed().as_nanos()).unwrap_or(u64::MAX),
    })
}

struct CertifiedArgmaxRun {
    token: usize,
    candidate_count: usize,
    status: u32,
    fallback: bool,
    certified: bool,
    coeff: f64,
    margin: Option<f64>,
    normalized_bf16_digest: u64,
    mechanism_ns: u64,
    read_ns: u64,
    h2d_ns: u64,
    kernel_ns: u64,
    d2h_ns: u64,
    cpu_preprocess_ns: u64,
    repair_ns: u64,
    cpu_fallback: Option<v4_state::FinalHeadCpuRun>,
}

struct IdOnlyFinalHeadDecision {
    token: usize,
    certificate: Option<CertifiedArgmaxRun>,
    cpu: Option<v4_state::FinalHeadCpuRun>,
    certificate_error: Option<String>,
}

fn certified_argmax_coeff() -> f64 {
    let unit_roundoff = 1.0_f64 / (1_u64 << 24) as f64;
    let gamma = |operations: u32| {
        let error = operations as f64 * unit_roundoff;
        error / (1.0 - error)
    };
    let gamma_c = gamma(7_167);
    let gamma_lane = gamma(223);
    let gamma_tree = gamma(5);
    let beta = (1.0 + gamma_lane) * (1.0 + gamma_tree) - 1.0;
    let coeff = (gamma_c + beta) / (1.0 - beta);
    debug_assert!(coeff.is_finite() && coeff > 0.0);
    coeff
}

fn run_certified_argmax_p0(
    context: &std::sync::Arc<CudaContext>,
    module: &kernels::LoadedModule,
    stream: &std::sync::Arc<CudaStream>,
    weights: &v4_state::FinalHeadWeights,
    hidden_bf16: &[u16],
) -> Result<CertifiedArgmaxRun, Box<dyn Error>> {
    let started = std::time::Instant::now();
    let streamed = run_streamed_final_head_gpu(context, module, stream, weights, hidden_bf16)?;
    let coeff = certified_argmax_coeff();
    let mut status = streamed
        .status
        .iter()
        .copied()
        .fold(0_u32, |all, row| all | row);
    let mut lower = Vec::with_capacity(v4_state::VOCAB);
    let mut upper = Vec::with_capacity(v4_state::VOCAB);
    if streamed.output.logits_f32.len() != v4_state::VOCAB
        || streamed.sumabs_f32.len() != v4_state::VOCAB
        || streamed.status.len() != v4_state::VOCAB
    {
        status |= CERTIFIED_HEAD_INTERVAL_INVALID;
    } else {
        for (&logit, &sumabs) in streamed.output.logits_f32.iter().zip(&streamed.sumabs_f32) {
            if !logit.is_finite() || !sumabs.is_finite() || sumabs < 0.0 {
                status |= CERTIFIED_HEAD_INTERVAL_INVALID;
                break;
            }
            let radius = (coeff * sumabs as f64).next_up();
            let lo = (logit as f64 - radius).next_down();
            let hi = (logit as f64 + radius).next_up();
            if !radius.is_finite() || !lo.is_finite() || !hi.is_finite() || lo > hi {
                status |= CERTIFIED_HEAD_INTERVAL_INVALID;
                break;
            }
            lower.push(lo);
            upper.push(hi);
        }
    }
    let mut candidates = Vec::new();
    if status == 0 {
        let max_lower = lower.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        candidates.extend(
            upper
                .iter()
                .enumerate()
                .filter_map(|(token, &hi)| (hi >= max_lower).then_some(token)),
        );
        if candidates.is_empty() {
            status |= CERTIFIED_HEAD_INTERVAL_INVALID;
        } else if candidates.len() == v4_state::VOCAB {
            status |= CERTIFIED_HEAD_ALL_CANDIDATES;
        }
    }
    let fallback = status != 0;
    let repair_started = std::time::Instant::now();
    let mut cpu_fallback = if fallback {
        Some(v4_state::final_checkpoint_head_cpu_timed(
            weights,
            hidden_bf16,
            1,
        )?)
    } else {
        None
    };
    let token = if let Some(cpu) = cpu_fallback.as_ref() {
        cpu.output.argmax[0]
    } else {
        let mut bytes = vec![0_u8; v4_state::HIDDEN * 2];
        let mut row_bf16 = vec![0_u16; v4_state::HIDDEN];
        let mut best = candidates[0];
        let mut best_logit = f32::NEG_INFINITY;
        for token in candidates.iter().copied() {
            weights.read_head_weight_bytes(token, 1, &mut bytes)?;
            for (weight, pair) in row_bf16.iter_mut().zip(bytes.chunks_exact(2)) {
                *weight = u16::from_le_bytes([pair[0], pair[1]]);
            }
            let logit =
                v4_state::canonical_final_head_row_dot(&streamed.output.normalized_bf16, &row_bf16);
            if token == candidates[0] || logit > best_logit {
                best = token;
                best_logit = logit;
            }
        }
        best
    };
    let repair_ns = u64::try_from(repair_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
    let margin = (!fallback).then(|| {
        let best_other = upper
            .iter()
            .enumerate()
            .filter(|(other, _)| *other != token)
            .map(|(_, &value)| value)
            .fold(f64::NEG_INFINITY, f64::max);
        (lower[token] - best_other).next_down()
    });
    Ok(CertifiedArgmaxRun {
        token,
        candidate_count: candidates.len(),
        status,
        fallback,
        certified: !fallback,
        coeff,
        margin,
        normalized_bf16_digest: fnv1a_u16(&streamed.output.normalized_bf16),
        mechanism_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
        read_ns: streamed.read_ns,
        h2d_ns: streamed.h2d_ns,
        kernel_ns: streamed.kernel_ns,
        d2h_ns: streamed.d2h_ns,
        cpu_preprocess_ns: streamed.cpu_preprocess_ns,
        repair_ns,
        cpu_fallback: cpu_fallback.take(),
    })
}

fn run_id_only_final_head(
    context: &std::sync::Arc<CudaContext>,
    module: &kernels::LoadedModule,
    stream: &std::sync::Arc<CudaStream>,
    weights: &v4_state::FinalHeadWeights,
    hidden_bf16: &[u16],
    certified_production: bool,
    audit_cpu: bool,
) -> Result<IdOnlyFinalHeadDecision, Box<dyn Error>> {
    if !certified_production {
        let cpu = v4_state::final_checkpoint_head_cpu_timed(weights, hidden_bf16, 1)?;
        return Ok(IdOnlyFinalHeadDecision {
            token: cpu.output.argmax[0],
            certificate: None,
            cpu: Some(cpu),
            certificate_error: None,
        });
    }
    let started = std::time::Instant::now();
    let mut certificate =
        match run_certified_argmax_p0(context, module, stream, weights, hidden_bf16) {
            Ok(certificate) => certificate,
            Err(error) => {
                let repair_started = std::time::Instant::now();
                let cpu = v4_state::final_checkpoint_head_cpu_timed(weights, hidden_bf16, 1)?;
                let repair_ns =
                    u64::try_from(repair_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
                return Ok(IdOnlyFinalHeadDecision {
                    token: cpu.output.argmax[0],
                    certificate: Some(CertifiedArgmaxRun {
                        token: cpu.output.argmax[0],
                        candidate_count: v4_state::VOCAB,
                        status: CERTIFIED_HEAD_RUNTIME_ERROR,
                        fallback: true,
                        certified: false,
                        coeff: certified_argmax_coeff(),
                        margin: None,
                        normalized_bf16_digest: fnv1a_u16(&cpu.output.normalized_bf16),
                        mechanism_ns: u64::try_from(started.elapsed().as_nanos())
                            .unwrap_or(u64::MAX),
                        read_ns: 0,
                        h2d_ns: 0,
                        kernel_ns: 0,
                        d2h_ns: 0,
                        cpu_preprocess_ns: 0,
                        repair_ns,
                        cpu_fallback: None,
                    }),
                    cpu: Some(cpu),
                    certificate_error: Some(error.to_string()),
                });
            }
        };
    let token = certificate.token;
    let cpu = if certificate.fallback {
        certificate.cpu_fallback.take()
    } else if audit_cpu {
        Some(v4_state::final_checkpoint_head_cpu_timed(
            weights,
            hidden_bf16,
            1,
        )?)
    } else {
        None
    };
    Ok(IdOnlyFinalHeadDecision {
        token,
        certificate: Some(certificate),
        cpu,
        certificate_error: None,
    })
}

fn final_head_profile_metrics(decision: &IdOnlyFinalHeadDecision) -> (u64, u64) {
    if let Some(cpu) = decision.cpu.as_ref() {
        (cpu.wall_ns, cpu.read_ns)
    } else if let Some(certificate) = decision.certificate.as_ref() {
        (certificate.mechanism_ns, certificate.read_ns)
    } else {
        (0, 0)
    }
}

fn certified_argmax_record(
    position: usize,
    decision: &IdOnlyFinalHeadDecision,
) -> Option<serde_json::Value> {
    let certificate = decision.certificate.as_ref()?;
    let cpu = decision.cpu.as_ref();
    Some(serde_json::json!({
        "event": "certified_argmax",
        "position": position,
        "head": "bf16_streamed_gpu_certificate",
        "cert_token": certificate.token,
        "candidate_count": certificate.candidate_count,
        "status_flags": certificate.status,
        "fallback": certificate.fallback,
        "certified": certificate.certified,
        "coeff": certificate.coeff,
        "margin": certificate.margin,
        "certificate_mechanism_ns": certificate.mechanism_ns,
        "repair_ns": certificate.repair_ns,
        "certificate_error": decision.certificate_error,
        "gpu": {
            "read_ns": certificate.read_ns,
            "h2d_ns": certificate.h2d_ns,
            "kernel_ns": certificate.kernel_ns,
            "d2h_ns": certificate.d2h_ns,
            "cpu_preprocess_ns": certificate.cpu_preprocess_ns,
        },
        "cpu_audit": cpu.map(|cpu| serde_json::json!({
            "argmax": cpu.output.argmax[0],
            "normalized_bf16_digest": format!("{:016x}", fnv1a_u16(&cpu.output.normalized_bf16)),
            "logits_f32_digest": format!("{:016x}", fnv1a_f32(&cpu.output.logits_f32)),
            "wall_ns": cpu.wall_ns,
            "read_ns": cpu.read_ns,
        })),
        "argmax_exact": cpu.map(|cpu| certificate.token == cpu.output.argmax[0]),
        "normalized_digest_exact": cpu.map(|cpu| {
            certificate.normalized_bf16_digest == fnv1a_u16(&cpu.output.normalized_bf16)
        }),
    }))
}

fn validate_certified_argmax_audit(
    position: usize,
    decision: &IdOnlyFinalHeadDecision,
) -> Result<(), Box<dyn Error>> {
    let certificate = decision
        .certificate
        .as_ref()
        .ok_or("certified argmax audit did not run the certificate")?;
    let cpu = decision
        .cpu
        .as_ref()
        .ok_or("certified argmax audit did not retain its CPU final")?;
    if certificate.token != cpu.output.argmax[0]
        || certificate.normalized_bf16_digest != fnv1a_u16(&cpu.output.normalized_bf16)
    {
        return Err(format!(
            "certified argmax or normalized digest diverged from canonical CPU audit at position {position}"
        )
        .into());
    }
    Ok(())
}

fn id_only_final_record(
    position: usize,
    input_id: i64,
    input_kind: &str,
    state: &str,
    decision: &IdOnlyFinalHeadDecision,
    input_hidden_bf16: Option<&[u16]>,
    request_elapsed_to_final_ns: u64,
) -> Result<serde_json::Value, Box<dyn Error>> {
    let mut record = serde_json::json!({
        "event": "final",
        "position": position,
        "input_id": input_id,
        "input_kind": input_kind,
        "state": state,
        "head": if decision.certificate.is_some() { "bf16_streamed_gpu_certificate" } else { "canonical_cpu_streamed_bf16" },
        "argmax": decision.token,
        "status": "PASS",
        "request_elapsed_to_final_ns": request_elapsed_to_final_ns,
    });
    let object = record
        .as_object_mut()
        .ok_or("id-only final record is not an object")?;
    if let Some(cpu) = decision.cpu.as_ref() {
        object.insert(
            "collapsed_bf16_digest".into(),
            serde_json::json!(format!("{:016x}", fnv1a_u16(&cpu.output.collapsed_bf16))),
        );
        object.insert(
            "normalized_bf16_digest".into(),
            serde_json::json!(format!("{:016x}", fnv1a_u16(&cpu.output.normalized_bf16))),
        );
        object.insert(
            "logits_f32_digest".into(),
            serde_json::json!(format!("{:016x}", fnv1a_f32(&cpu.output.logits_f32))),
        );
    } else if let Some(certificate) = decision.certificate.as_ref() {
        object.insert(
            "normalized_bf16_digest".into(),
            serde_json::json!(format!("{:016x}", certificate.normalized_bf16_digest)),
        );
        object.insert(
            "certificate".into(),
            serde_json::json!({
                "candidate_count": certificate.candidate_count,
                "status_flags": certificate.status,
                "fallback": certificate.fallback,
                "certified": certificate.certified,
            }),
        );
    }
    if let Some(hidden) = input_hidden_bf16 {
        object.insert(
            "input_hidden_bf16_digest".into(),
            serde_json::json!(format!("{:016x}", fnv1a_u16(hidden))),
        );
    }
    Ok(record)
}

const V4_EOS_TOKEN_ID: i64 = 1;
const V4_LAYER_EXPERT_RETAINED_SLOTS: usize = 32;

/// Request-scoped bookkeeping for the generic stationary device path.  It is
/// deliberately data-only: the worker/cache remain the existing
/// `FixedLayerCache` and `run_pinned_fixed_request_worker` implementation.
/// Keeping this boundary small makes the failure rule observable without
/// inventing a second scheduler.
#[derive(Debug)]
struct GenericStationaryLifecycle {
    committed_rows: Vec<usize>,
    poisoned: bool,
    live_leases: usize,
    telemetry: GenericStationaryTelemetry,
}

#[derive(Debug, Default)]
struct GenericStationaryTelemetry {
    layer_commits: usize,
    device_peak_bytes: usize,
    host_peak_bytes: usize,
    pinned_peak_bytes: usize,
    live_lease_peak: usize,
}

/// State-root observation is determined by which authoritative store exists,
/// not by whether a device-attention implementation happens to be enabled.
/// Generic stationary has device-owned state and deliberately no host store.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PersistentStateDigestSource {
    ObservedDevice,
    HostPersistent,
    Unobserved,
}

const fn persistent_state_digest_source(
    observed_device_state: bool,
    generic_device_stationary: bool,
    host_persistent_available: bool,
) -> PersistentStateDigestSource {
    if observed_device_state {
        PersistentStateDigestSource::ObservedDevice
    } else if generic_device_stationary {
        PersistentStateDigestSource::Unobserved
    } else if host_persistent_available {
        PersistentStateDigestSource::HostPersistent
    } else {
        PersistentStateDigestSource::Unobserved
    }
}

impl GenericStationaryLifecycle {
    fn new(layers: usize) -> Result<Self, Box<dyn Error>> {
        if layers != v4_state::V4_LAYER_COUNT {
            return Err("generic stationary lifecycle requires all 61 layer stores".into());
        }
        Ok(Self {
            committed_rows: vec![0; layers],
            poisoned: false,
            live_leases: 0,
            telemetry: GenericStationaryTelemetry::default(),
        })
    }

    /// Checks the causal boundary before a layer touches its persistent store.
    fn begin_layer(&self, layer: usize, position: usize) -> Result<(), Box<dyn Error>> {
        if self.poisoned || self.committed_rows.get(layer).copied() != Some(position) {
            return Err("generic stationary layer is not at its causal commit boundary".into());
        }
        Ok(())
    }

    /// The sole semantic commit point.  Call only after attention, FFN, and
    /// the terminal hidden row have all succeeded.
    fn commit_layer(&mut self, layer: usize, position: usize) -> Result<(), Box<dyn Error>> {
        self.begin_layer(layer, position)?;
        let next = position
            .checked_add(1)
            .ok_or("generic stationary committed-row overflow")?;
        *self
            .committed_rows
            .get_mut(layer)
            .ok_or("generic stationary layer is outside the 61 stores")? = next;
        self.telemetry.layer_commits = self.telemetry.layer_commits.saturating_add(1);
        Ok(())
    }

    /// A failed prepared row never changes any committed counter.  The request
    /// is fail-closed so no later row can observe a partially prepared cache.
    fn poison(&mut self) {
        self.poisoned = true;
    }

    fn acquire_lease(&mut self) {
        self.live_leases = self.live_leases.saturating_add(1);
        self.telemetry.live_lease_peak = self.telemetry.live_lease_peak.max(self.live_leases);
    }

    fn release_lease(&mut self) -> Result<(), Box<dyn Error>> {
        self.live_leases = self
            .live_leases
            .checked_sub(1)
            .ok_or("generic stationary released a non-live lease")?;
        Ok(())
    }

    fn observe_peaks(&mut self, device: usize, host: usize, pinned: usize) {
        self.telemetry.device_peak_bytes = self.telemetry.device_peak_bytes.max(device);
        self.telemetry.host_peak_bytes = self.telemetry.host_peak_bytes.max(host);
        self.telemetry.pinned_peak_bytes = self.telemetry.pinned_peak_bytes.max(pinned);
    }

    fn assert_drained(&self) -> Result<(), Box<dyn Error>> {
        if self.live_leases != 0 {
            return Err("generic stationary request ended with a live fixed lease".into());
        }
        Ok(())
    }
}

fn should_retain_layer_expert(retained: usize) -> bool {
    retained < V4_LAYER_EXPERT_RETAINED_SLOTS
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GreedyStopReason {
    Eos,
    MaxNewTokens,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GreedyDecision {
    IgnorePrompt,
    EmitAndContinue,
    EmitAndStop(GreedyStopReason),
}

fn greedy_argmax_decision(
    prompt_len: usize,
    token_position: usize,
    generated_count: usize,
    max_new_tokens: usize,
    argmax: i64,
) -> GreedyDecision {
    if token_position + 1 < prompt_len {
        GreedyDecision::IgnorePrompt
    } else if argmax == V4_EOS_TOKEN_ID {
        GreedyDecision::EmitAndStop(GreedyStopReason::Eos)
    } else if generated_count + 1 == max_new_tokens {
        GreedyDecision::EmitAndStop(GreedyStopReason::MaxNewTokens)
    } else {
        GreedyDecision::EmitAndContinue
    }
}

fn b1_needs_final_head(prompt_len: usize, token_position: usize) -> bool {
    token_position + 1 >= prompt_len
}

fn chat_input_conflict(
    chat_user_present: bool,
    raw_prompt_present: bool,
    id_input_present: bool,
) -> bool {
    chat_user_present && (raw_prompt_present || id_input_present)
}

fn greedy_forward_tokens(prompt_len: usize, greedy_steps: usize) -> Result<usize, Box<dyn Error>> {
    if prompt_len == 0 {
        return Err("greedy decoding requires at least one prompt token".into());
    }
    prompt_len
        .checked_add(greedy_steps.saturating_sub(1))
        .ok_or_else(|| "greedy decoding token count overflow".into())
}

fn validate_layer_stationary_prefill(
    prompt_len: usize,
    greedy_steps: usize,
    max_layers: usize,
    cache_is_off: bool,
    qd4_replay_bench: bool,
    short_device_decode: bool,
) -> Result<(), Box<dyn Error>> {
    let targeted_attention_audit = optional_one_flag("COLIBRI_V4_TARGETED_ATTENTION_AUDIT")?;
    if prompt_len == 0
        || if short_device_decode {
            !(2..=V4_SHORT_DEVICE_DECODE_MAX_NEW_TOKENS).contains(&greedy_steps)
        } else {
            greedy_steps != 1
        }
        || if targeted_attention_audit {
            max_layers != 3 || short_device_decode
        } else {
            max_layers != v4_state::V4_LAYER_COUNT
        }
        || !cache_is_off
        || qd4_replay_bench
    {
        return Err(
            "layer-stationary prefill requires a nonempty prompt, greedy_steps=1 (or 2..=5 for short device decode), full61 (or exactly 3 layers for the targeted attention audit), fixed cache off, and the QD4 replay benchmark off"
                .into(),
        );
    }
    Ok(())
}

fn short_device_decode_rows(
    prompt_len: usize,
    greedy_steps: usize,
    enabled: bool,
) -> Result<usize, Box<dyn Error>> {
    if !enabled {
        return Ok(prompt_len);
    }
    if !(2..=V4_SHORT_DEVICE_DECODE_MAX_NEW_TOKENS).contains(&greedy_steps) {
        return Err("short device decode requires COLIBRI_V4_GREEDY_STEPS=2..=5".into());
    }
    let rows = greedy_forward_tokens(prompt_len, greedy_steps)?;
    if rows > V4_SHORT_DEVICE_DECODE_MAX_ROWS {
        return Err("short device decode exceeds its nine-row request-state ceiling".into());
    }
    Ok(rows)
}

fn short_device_decode_static_check() -> Result<(), Box<dyn Error>> {
    for (topk_per_row, expected_score_reuse, scratch_topk_capacity) in [
        (1, false, 1),
        (SM89_SCORE_REUSE_MIN_TOPK - 1, false, 3),
        (SM89_SCORE_REUSE_MIN_TOPK, true, 4),
        (v4_state::KV_WINDOW, true, v4_state::KV_WINDOW),
        (v4_state::KV_WINDOW + 1, true, v4_state::KV_WINDOW + 2),
    ] {
        if SparseAttentionArm::Sm89Crossover.uses_score_reuse(topk_per_row) != expected_score_reuse
            || scratch_topk_capacity < topk_per_row
        {
            return Err("short device decode sm89 crossover/workspace boundary drift".into());
        }
    }
    let state_bytes = 31usize
        .checked_mul(V4_DEVICE_TOKEN_STATE_HCA_BYTES)
        .and_then(|bytes| {
            30usize
                .checked_mul(V4_DEVICE_TOKEN_STATE_MAX_BYTES)
                .and_then(|csa| bytes.checked_add(csa))
        })
        .ok_or("short device decode state ledger overflow")?;
    let rope_bytes = 2usize
        .checked_mul(V4_SHORT_DEVICE_DECODE_MAX_ROWS)
        .and_then(|bytes| bytes.checked_mul(v4_state::Q_ROPE_DIM))
        .and_then(|bytes| bytes.checked_mul(std::mem::size_of::<f32>()))
        .ok_or("short device decode RoPE ledger overflow")?;
    let total_bytes = [
        V4_QD4_DEVICE_BYTES,
        V4_QD4_IMMUTABLE_EXPERT_CACHE_DEVICE_BYTES,
        494_348_312,
        state_bytes,
        rope_bytes,
        V4_TID2EID_TRANSIENT_MAX_BYTES,
        V4_DEVICE_CAPACITY_HEADROOM_BYTES,
    ]
    .into_iter()
    .try_fold(0usize, |total, bytes| total.checked_add(bytes))
    .ok_or("short device decode capacity ledger overflow")?;
    if short_device_decode_rows(5, 2, true)? != 6
        || short_device_decode_rows(5, 5, true)? != 9
        || csa_compressed_rows_after(3) != 1
        || csa_compressed_rows_after(6) != 1
        || csa_compressed_rows_after(7) != 2
        || csa_compressed_rows_after(8) != 2
        || state_bytes != 3_763_828_054
        || rope_bytes != 4_608
        || total_bytes != 7_673_270_174
    {
        return Err("short device decode static contract mismatch".into());
    }
    strict_compressor_host_arena_static_check()?;
    let mut compressor = v4_state::V4CompressorState::new_decode(4, 1)?;
    let mut compressed_rows = 0;
    for position in 0..9 {
        let prepared = v4_state::CompressorPrepared {
            kv_f32: vec![position as f32 + 1.0; 2],
            score_f32: vec![position as f32 + 2.0; 2],
            score_plus_ape_f32: vec![position as f32 + 3.0; 2],
        };
        let (next, pooled) = compressor.prepare_decode_row_and_pool(&prepared, position)?;
        compressed_rows += usize::from(pooled.is_some());
        if (position == 3 || position == 7) != pooled.is_some() {
            return Err("short device decode CSA compressor boundary drift".into());
        }
        compressor = next;
    }
    if compressed_rows != 2 {
        return Err("short device decode CSA compressed row count drift at position 8".into());
    }
    stationary_raw_retention_static_check()?;
    println!(
        "{{\"status\":\"PASS\",\"mode\":\"v4-short-device-decode-static-check\",\"rows\":9,\"state_bytes\":{state_bytes},\"rope_bytes\":{rope_bytes},\"capacity_bytes\":{total_bytes},\"csa_compressed_rows_pos8\":{compressed_rows},\"sm89_crossover\":\"K<4=reference,K>=4=score_reuse\",\"sm89_workspace_cases\":[1,3,4,128,129],\"strict_arena_reuse_repeats\":100,\"stationary_raw_selection\":\"device+host+miss\",\"stationary_raw_lease\":\"exclusive_take_restore\",\"fixed_host_request_ids_cached_bytes\":0,\"prompt_hash_id_bytes_dropped\":240,\"decode_hash_id_bytes_external\":48}}"
    );
    Ok(())
}

fn stationary_raw_retention_static_check() -> Result<(), Box<dyn Error>> {
    const PROMPT_ROWS: usize = 5;
    const DECODE_ROWS: usize = 1;
    let mut router = v4_state::LayerRouterWeights::Hash {
        ids: vec![0; PROMPT_ROWS * v4_router::V4_TOPK],
    };
    let removed_bytes = drop_request_hash_ids_before_fixed_retention(&mut router, PROMPT_ROWS)?;
    let fixed_host_bytes = 1_000_usize;
    let prompt_host_bytes = fixed_host_bytes
        .checked_add(removed_bytes)
        .ok_or("stationary raw static host byte overflow")?;
    let decode_request_bytes = DECODE_ROWS
        .checked_mul(v4_router::V4_TOPK)
        .and_then(|values| values.checked_mul(std::mem::size_of::<i64>()))
        .ok_or("stationary raw static decode-ID byte overflow")?;
    if removed_bytes != PROMPT_ROWS * v4_router::V4_TOPK * std::mem::size_of::<i64>()
        || prompt_host_bytes - removed_bytes != fixed_host_bytes
        || decode_request_bytes != 48
        || !matches!(
            router,
            v4_state::LayerRouterWeights::Hash { ref ids } if ids.is_empty()
        )
    {
        return Err("stationary raw request IDs leaked into fixed host bytes".into());
    }
    let plans = [(0, 10, 12), (1, 11, 13), (2, 20, 20)]
        .into_iter()
        .map(|(layer, raw_device_bytes, raw_host_bytes)| FixedLayerPlan {
            layer,
            tensors: Vec::new(),
            device_bytes: raw_device_bytes + 1,
            raw_device_bytes,
            host_bytes: raw_host_bytes + 1,
            raw_host_bytes,
            device_saved_ns: 1,
            host_saved_ns: 1,
        })
        .collect::<Vec<_>>();
    let none_selected = vec![false; plans.len()];
    let device = select_fixed_cache_layers(&plans, 10, &none_selected, true, true);
    let host = select_fixed_cache_layers(&plans, 13, &device, false, true);
    if device != [true, false, false] || host != [false, true, false] {
        return Err("stationary raw budget selection/miss boundary drift".into());
    }
    let mut slot = Some((7_u64, 11_u64));
    let leased = slot
        .take()
        .ok_or("stationary raw ownership self-check could not take its slot")?;
    if slot.is_some() {
        return Err("stationary raw ownership slot remained visible during its lease".into());
    }
    restore_leased_slot(&mut slot, leased);
    if slot != Some((7, 11)) {
        return Err("stationary raw ownership lease failed to restore exactly once".into());
    }
    Ok(())
}

fn strict_compressor_host_arena_static_check() -> Result<(), Box<dyn Error>> {
    for (outputs, ratio) in [(512_usize, 128_usize), (1_024, 4)] {
        let weights = outputs
            .checked_mul(v4_state::HIDDEN)
            .ok_or("strict arena static weight size overflow")?;
        let mut wkv_bf16 = vec![0_u16; weights];
        let mut wgate_bf16 = vec![0_u16; weights];
        let mut ape_f32 = vec![0.0_f32; outputs * ratio];
        let mut input_f32 = vec![0.0_f32; v4_state::HIDDEN];
        let mut wkv_f32 = vec![0.0_f32; weights];
        let mut wgate_f32 = vec![0.0_f32; weights];
        let mut kv_f32 = vec![0.0_f32; outputs];
        let mut score_f32 = vec![0.0_f32; outputs];
        let mut score_plus_ape_f32 = vec![0.0_f32; outputs];
        let baseline = [
            (wkv_bf16.as_ptr() as usize, wkv_bf16.capacity()),
            (wgate_bf16.as_ptr() as usize, wgate_bf16.capacity()),
            (ape_f32.as_ptr() as usize, ape_f32.capacity()),
            (input_f32.as_ptr() as usize, input_f32.capacity()),
            (wkv_f32.as_ptr() as usize, wkv_f32.capacity()),
            (wgate_f32.as_ptr() as usize, wgate_f32.capacity()),
            (kv_f32.as_ptr() as usize, kv_f32.capacity()),
            (score_f32.as_ptr() as usize, score_f32.capacity()),
            (
                score_plus_ape_f32.as_ptr() as usize,
                score_plus_ape_f32.capacity(),
            ),
        ];
        for repeat in 0..100 {
            wkv_bf16.fill(repeat as u16);
            wgate_bf16.fill((repeat + 1) as u16);
            ape_f32.fill(repeat as f32);
            input_f32.fill(repeat as f32);
            wkv_f32.fill(repeat as f32);
            wgate_f32.fill(repeat as f32);
            kv_f32.fill(repeat as f32);
            score_f32.fill(repeat as f32);
            score_plus_ape_f32.fill(repeat as f32);
            let current = [
                (wkv_bf16.as_ptr() as usize, wkv_bf16.capacity()),
                (wgate_bf16.as_ptr() as usize, wgate_bf16.capacity()),
                (ape_f32.as_ptr() as usize, ape_f32.capacity()),
                (input_f32.as_ptr() as usize, input_f32.capacity()),
                (wkv_f32.as_ptr() as usize, wkv_f32.capacity()),
                (wgate_f32.as_ptr() as usize, wgate_f32.capacity()),
                (kv_f32.as_ptr() as usize, kv_f32.capacity()),
                (score_f32.as_ptr() as usize, score_f32.capacity()),
                (
                    score_plus_ape_f32.as_ptr() as usize,
                    score_plus_ape_f32.capacity(),
                ),
            ];
            if baseline != current {
                return Err("strict Compressor host arena reallocated during reuse check".into());
            }
        }
    }
    Ok(())
}

fn qd4_lane_count(enabled: bool) -> usize {
    usize::from(enabled) * V4_QD4_LANES
}

fn validate_qd4_lane_reserve(
    host_available: usize,
    device_available: usize,
) -> Result<(), Box<dyn Error>> {
    if host_available < V4_QD4_PINNED_HOST_BYTES || device_available < V4_QD4_DEVICE_BYTES {
        return Err("persistent QD4 lanes lack the explicit pinned-host/device reserve".into());
    }
    Ok(())
}

fn qd4_expected_routed_loads(layers: usize, prompt_len: usize) -> Result<usize, Box<dyn Error>> {
    layers
        .checked_mul(prompt_len)
        .and_then(|rows| rows.checked_mul(v4_router::V4_TOPK))
        .ok_or_else(|| "QD4 routed load count overflow".into())
}

fn launch_v4_activation_quantize(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    input: &DeviceBuffer<u16>,
    output: &mut DeviceBuffer<f32>,
) -> Result<(), Box<dyn Error>> {
    const BLOCK: usize = 128;
    if input.len() == 0 || input.len() % BLOCK != 0 || output.len() != input.len() {
        return Err("V4 activation quantization requires complete 128-value groups".into());
    }
    let elements = u32::try_from(input.len())?;
    // SAFETY: the validated launch assigns one output element to each thread;
    // each thread scans its complete 128-value source group before writing.
    unsafe {
        module.quantize_v4_activation(stream, LaunchConfig::for_num_elems(elements), input, output)
    }?;
    Ok(())
}

fn launch_v4_activation_quantize_raw(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    input: &DeviceBuffer<u16>,
    output: &mut DeviceBuffer<u8>,
    scales: &mut DeviceBuffer<u8>,
) -> Result<(), Box<dyn Error>> {
    const BLOCK: usize = v4_state::FP8_BLOCK;
    if input.is_empty()
        || input.len() % BLOCK != 0
        || output.len() != input.len()
        || scales.len() != input.len() / BLOCK
    {
        return Err("Flash FP4 raw activation buffers require complete 128-value groups".into());
    }
    // SAFETY: one thread owns one complete 128-value raw-FP8 group.
    unsafe {
        module.quantize_v4_activation_raw(
            stream,
            LaunchConfig::for_num_elems(u32::try_from(scales.len())?),
            input,
            output,
            scales,
        )
    }?;
    Ok(())
}

/// Materializes the raw FP8/E8M0 view of a live BF16 device buffer for the
/// Flash L0 parity harness.  This is deliberately debug-only: production
/// expert GEMMs retain their existing dequantized-F32 input path.
fn flash_parity_capture_activation_fp8(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    input: &DeviceBuffer<u16>,
) -> Result<(Vec<u8>, Vec<u8>), Box<dyn Error>> {
    const BLOCK: usize = v4_state::FP8_BLOCK;
    if input.is_empty() || input.len() % BLOCK != 0 {
        return Err("Flash parity FP8 trace requires complete 128-value groups".into());
    }
    let mut fp8 = DeviceBuffer::<u8>::zeroed(stream, input.len())?;
    let mut scales = DeviceBuffer::<u8>::zeroed(stream, input.len() / BLOCK)?;
    let mut no_dequantized = DeviceBuffer::<u16>::zeroed(stream, 0)?;
    // SAFETY: one launch thread owns each complete 128-value group; this only
    // snapshots the live source buffer and never feeds a model kernel.
    unsafe {
        module.quantize_v4_fp8_bf16(
            stream,
            LaunchConfig::for_num_elems(u32::try_from(scales.len())?),
            input,
            &mut fp8,
            &mut scales,
            &mut no_dequantized,
            u32::try_from(input.len())?,
            u32::try_from(input.len())?,
            u32::try_from(BLOCK)?,
        )?;
    }
    Ok((fp8.to_host_vec(stream)?, scales.to_host_vec(stream)?))
}

fn launch_v4_fp4_matvec(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    input: &DeviceBuffer<f32>,
    packed: &DeviceBuffer<u8>,
    scales: &DeviceBuffer<u8>,
    output: &mut DeviceBuffer<u16>,
    rows: usize,
    out_features: usize,
    in_features: usize,
) -> Result<(), Box<dyn Error>> {
    if rows == 0
        || rows > v4_expert::V4_MAX_EXPERT_ROWS
        || in_features % 32 != 0
        || input.len() != rows * in_features
        || packed.len() != out_features * in_features / 2
        || scales.len() != out_features * in_features / 32
        || output.len() != rows * out_features
    {
        return Err("V4 FP4 matvec buffers disagree with the fixed expert layout".into());
    }
    let threads = u32::try_from(
        output
            .len()
            .checked_mul(32)
            .ok_or("V4 matvec launch overflow")?,
    )?;
    // SAFETY: each complete warp owns one bounded output element. Host validation
    // proves the packed-two-value and one-E8M0-per-32-K indexing used by the kernel.
    unsafe {
        module.matvec_v4_fp4(
            stream,
            LaunchConfig::for_num_elems(threads),
            input,
            packed,
            scales,
            output,
            u32::try_from(in_features)?,
            u32::try_from(out_features)?,
        )
    }?;
    Ok(())
}

fn launch_v4_fp4_matvec_block32(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    input: &DeviceBuffer<u8>,
    input_scales: &DeviceBuffer<u8>,
    packed: &DeviceBuffer<u8>,
    scales: &DeviceBuffer<u8>,
    output: &mut DeviceBuffer<u16>,
    rows: usize,
    out_features: usize,
    in_features: usize,
) -> Result<(), Box<dyn Error>> {
    const WEIGHT_BLOCK: usize = 32;
    const ACTIVATION_BLOCK: usize = v4_state::FP8_BLOCK;
    if rows == 0
        || rows > v4_expert::V4_MAX_EXPERT_ROWS
        || in_features % ACTIVATION_BLOCK != 0
        || input.len() != rows * in_features
        || input_scales.len() != rows * in_features / ACTIVATION_BLOCK
        || packed.len() != out_features * in_features / 2
        || scales.len() != out_features * in_features / WEIGHT_BLOCK
        || output.len() != rows * out_features
    {
        return Err("Flash FP4 block32 buffers disagree with the expert layout".into());
    }
    let threads = u32::try_from(
        output
            .len()
            .checked_mul(32)
            .ok_or("Flash FP4 block32 launch overflow")?,
    )?;
    // SAFETY: one complete warp owns each output; each 32-K block is reduced
    // before lane 0 applies the corresponding activation and weight scales.
    unsafe {
        module.matvec_v4_fp4_block32(
            stream,
            LaunchConfig::for_num_elems(threads),
            input,
            input_scales,
            packed,
            scales,
            output,
            u32::try_from(in_features)?,
            u32::try_from(out_features)?,
        )
    }?;
    Ok(())
}

fn launch_v4_fp4_matvec_mma(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    input: &DeviceBuffer<u8>,
    input_scales: &DeviceBuffer<u8>,
    packed: &DeviceBuffer<u8>,
    scales: &DeviceBuffer<u8>,
    output: &mut DeviceBuffer<u16>,
    rows: usize,
    out_features: usize,
    in_features: usize,
) -> Result<(), Box<dyn Error>> {
    if rows == 0
        || rows > v4_expert::V4_MAX_EXPERT_ROWS
        || in_features % v4_state::FP8_BLOCK != 0
        || out_features % 8 != 0
        || input.len() != rows * in_features
        || input_scales.len() != rows * in_features / v4_state::FP8_BLOCK
        || packed.len() != out_features * in_features / 2
        || scales.len() != out_features * in_features / 32
        || output.len() != rows * out_features
    {
        return Err("Flash FP8-MMA buffers disagree with the expert layout".into());
    }
    let threads = rows
        .checked_mul(out_features / 8)
        .and_then(|tiles| tiles.checked_mul(32))
        .ok_or("Flash FP8-MMA launch overflow")?;
    // SAFETY: the locked kernel maps one full warp to an eight-wide output tile;
    // host validation proves every FP8/FP4 packed span and E8M0 scale span.
    unsafe {
        module.matvec_v4_fp4_mma(
            stream,
            LaunchConfig::for_num_elems(u32::try_from(threads)?),
            input,
            input_scales,
            packed,
            scales,
            output,
            u32::try_from(in_features)?,
            u32::try_from(out_features)?,
        )
    }?;
    Ok(())
}

fn launch_v4_fp4_tilelang_ref(
    reference: &v4_tilelang_ref::TilelangFp4Reference,
    stream: &CudaStream,
    input: &DeviceBuffer<u8>,
    input_scales: &DeviceBuffer<u8>,
    packed: &DeviceBuffer<u8>,
    scales: &DeviceBuffer<u8>,
    output: &mut DeviceBuffer<u16>,
    rows: usize,
    out_features: usize,
    in_features: usize,
) -> Result<(), Box<dyn Error>> {
    if rows == 0
        || rows > v4_expert::V4_MAX_EXPERT_ROWS
        || input.len() != rows * in_features
        || input_scales.len() != rows * in_features / v4_state::FP8_BLOCK
        || packed.len() != out_features * in_features / 2
        || scales.len() != out_features * in_features / 32
        || output.len() != rows * out_features
    {
        return Err("TileLang reference buffers disagree with the locked expert layout".into());
    }
    let kernel = reference.for_dimensions(out_features, in_features)?;
    let (block_x, block_y, block_z) = kernel.block();
    let mut input_ptr = input.cu_deviceptr();
    let mut packed_ptr = packed.cu_deviceptr();
    let mut output_ptr = output.cu_deviceptr();
    let mut input_scales_ptr = input_scales.cu_deviceptr();
    let mut scales_ptr = scales.cu_deviceptr();
    let mut rows_u32 = u32::try_from(rows)?;
    let mut arguments = [
        (&mut input_ptr as *mut _) as *mut std::ffi::c_void,
        (&mut packed_ptr as *mut _) as *mut std::ffi::c_void,
        (&mut output_ptr as *mut _) as *mut std::ffi::c_void,
        (&mut input_scales_ptr as *mut _) as *mut std::ffi::c_void,
        (&mut scales_ptr as *mut _) as *mut std::ffi::c_void,
        (&mut rows_u32 as *mut _) as *mut std::ffi::c_void,
    ];
    // SAFETY: `v4_tilelang_ref` has already validated the exact raw ABI and
    // PTX identity. These six local values remain alive through submission.
    unsafe {
        cuda_core::launch_kernel_on_stream(
            kernel.function(),
            (kernel.grid_x(), kernel.grid_y_for_rows(rows)?, 1),
            (block_x, block_y, block_z),
            kernel.dynamic_shared_bytes(),
            stream,
            &mut arguments,
        )
    }?;
    Ok(())
}

fn launch_v4_swiglu_route(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    gate: &DeviceBuffer<u16>,
    up: &DeviceBuffer<u16>,
    output: &mut DeviceBuffer<u16>,
    route_weight: f32,
) -> Result<(), Box<dyn Error>> {
    if !route_weight.is_finite()
        || gate.len() == 0
        || up.len() != gate.len()
        || output.len() != gate.len()
    {
        return Err("V4 SwiGLU route buffers are invalid".into());
    }
    // SAFETY: all three equally sized buffers are validated; every thread
    // writes one output. This is the official FP32 SwiGLU/route then BF16 cast boundary.
    unsafe {
        module.swiglu_route_bf16(
            stream,
            LaunchConfig::for_num_elems(u32::try_from(gate.len())?),
            gate,
            up,
            output,
            route_weight,
        )
    }?;
    Ok(())
}

fn launch_v4_bf16_to_f32(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    input: &DeviceBuffer<u16>,
    output: &mut DeviceBuffer<f32>,
) -> Result<(), Box<dyn Error>> {
    if input.len() == 0 || output.len() != input.len() {
        return Err("V4 routed contribution output buffers disagree".into());
    }
    // SAFETY: validation proves one output slot per BF16 input element.
    unsafe {
        module.bf16_to_f32_rows(
            stream,
            LaunchConfig::for_num_elems(u32::try_from(input.len())?),
            input,
            output,
        )
    }?;
    Ok(())
}

fn launch_e4m3fn_to_bf16(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    input: &DeviceBuffer<u8>,
    output: &mut DeviceBuffer<u16>,
) -> Result<(), Box<dyn Error>> {
    if output.len() < input.len() {
        return Err("E4M3FN decode scratch is too small".into());
    }
    unsafe {
        module.e4m3fn_to_bf16(
            stream,
            LaunchConfig::for_num_elems(u32::try_from(input.len())?),
            input,
            output,
        )?;
    }
    Ok(())
}

fn assert_v4_expert_close(actual: &[f32], expected: &[f32]) {
    let maximum_error = v4_expert_maximum_error(actual, expected);
    println!("V4 expert-0 maximum absolute CPU/GPU error: {maximum_error:.6e}");
}

fn v4_expert_maximum_error(actual: &[f32], expected: &[f32]) -> f32 {
    assert_eq!(actual.len(), expected.len(), "V4 expert output length");
    let mut maximum_error = 0.0_f32;
    for (actual, expected) in actual.iter().zip(expected) {
        maximum_error = maximum_error.max((actual - expected).abs());
        let tolerance = 3e-3 * expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= tolerance,
            "V4 expert output differs: expected {expected}, got {actual}, tolerance {tolerance}"
        );
    }
    maximum_error
}

fn run_real_v4_expert0_slice_if_available(
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

fn run_v4_block_certificate_experiment(
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

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct LoopbackWireBytes {
    initial_requests: usize,
    audit_responses: usize,
    certified_responses: usize,
    repair_requests: usize,
    repair_responses: usize,
}

impl LoopbackWireBytes {
    fn total(self) -> usize {
        self.initial_requests
            + self.audit_responses
            + self.certified_responses
            + self.repair_requests
            + self.repair_responses
    }

    fn returned(self) -> usize {
        self.audit_responses + self.certified_responses + self.repair_responses
    }
}

#[derive(Clone)]
struct LoopbackSample {
    e2e_ns: u64,
    worker_ns: u64,
    codec_tcp_ns: u64,
    repair_ns: u64,
    gpu_ns: u64,
    repair_lanes: usize,
    bytes: LoopbackWireBytes,
}

const NETEM_BLOCKS: usize = 4;
const NETEM_PAIRS_PER_BLOCK: usize = 25;

struct DualBenchConfig {
    scenario: String,
    seed: u64,
    audit_first: [bool; NETEM_BLOCKS],
}

impl DualBenchConfig {
    fn from_env() -> Result<Self, Box<dyn Error>> {
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

struct OwnerCold {
    owner: u8,
    elapsed_ns: u64,
    weight_bytes: usize,
}

const RECOVERY_SAMPLES: u64 = 20;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RecoveryCase {
    AuditBaseline,
    CertifiedBaseline,
    AuditF1,
    CertifiedF1,
    CertifiedF2,
    WrongReplicaIdentity,
    SecondFailure,
}

impl RecoveryCase {
    fn label(self) -> &'static str {
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

    fn mode(self) -> v4_federated::OwnerTaskMode {
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

    fn start_id(self) -> u64 {
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
enum RecoveryEndpoint {
    PrimaryA,
    PrimaryB,
    ReplicaA,
}

impl RecoveryEndpoint {
    fn label(self) -> &'static str {
        match self {
            Self::PrimaryA => "A",
            Self::PrimaryB => "B",
            Self::ReplicaA => "A'",
        }
    }

    fn owner(self) -> u8 {
        match self {
            Self::PrimaryB => 1,
            Self::PrimaryA | Self::ReplicaA => 0,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RecoveryFault {
    None,
    F1,
    F2,
    WrongIdentity,
}

struct RecoverySample {
    e2e_ns: u64,
    failover_ns: u64,
    failovers: usize,
    bytes: LoopbackWireBytes,
}

struct RecoveryCold {
    endpoint: RecoveryEndpoint,
    elapsed_ns: u64,
    weight_bytes: usize,
}

fn print_netem_result(
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

fn recovery_case_for_id(request_id: u64) -> Option<RecoveryCase> {
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

fn recovery_fault(endpoint: RecoveryEndpoint, case: RecoveryCase) -> RecoveryFault {
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

fn validate_v4_recovery_task(
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

fn v4_activation_hash(values: &[u16]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for value in values {
        for byte in value.to_le_bytes() {
            hash = (hash ^ byte as u64).wrapping_mul(0x100_0000_01b3);
        }
    }
    hash
}

fn run_v4_recovery_case(
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

fn run_v4_recovery_audit(
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

fn connect_recovery_endpoint(address: std::net::SocketAddr) -> Result<std::net::TcpStream, String> {
    let socket = std::net::TcpStream::connect(address).map_err(|error| error.to_string())?;
    let timeout = std::time::Duration::from_secs(30);
    socket
        .set_nodelay(true)
        .and_then(|_| socket.set_read_timeout(Some(timeout)))
        .and_then(|_| socket.set_write_timeout(Some(timeout)))
        .map_err(|error| error.to_string())?;
    Ok(socket)
}

fn receive_recovery_audit(
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

fn receive_recovery_certified(
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

fn receive_recovery_repair(
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

fn certify_recovery_pair(
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

fn validate_v4_owner_task(
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

fn execute_v4_owner_slots(
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

fn owner_tasks(
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

fn run_v4_audit_client(
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

fn audit_bits_from_owner_responses(
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

fn apply_owner_repairs(
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

fn canonical_audit_bits(contributions: &[Vec<f32>]) -> Vec<u16> {
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

fn bf16_mismatch_count(actual: &[u16], expected: &[u16]) -> usize {
    actual
        .iter()
        .zip(expected)
        .filter(|(actual, expected)| actual != expected)
        .count()
}

fn wire_frame_bytes(frame: &Vec<u8>) -> usize {
    frame.len() + size_of::<u32>()
}

fn print_loopback_timings(name: &str, samples: &[LoopbackSample], include_repair: bool) {
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

fn print_recovery_case(case: RecoveryCase, samples: &[RecoverySample]) {
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

fn median_twice_usize(samples: &[usize]) -> usize {
    assert!(!samples.is_empty(), "median needs at least one sample");
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    if sorted.len() % 2 == 0 {
        sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]
    } else {
        sorted[sorted.len() / 2] * 2
    }
}

fn median_p95_ms(samples: &[u64]) -> (f64, f64) {
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let middle = sorted.len() / 2;
    let median = (sorted[middle - 1] as f64 + sorted[middle] as f64) / 2.0;
    let p95 = sorted[(sorted.len() * 95).div_ceil(100) - 1] as f64;
    (median / 1_000_000.0, p95 / 1_000_000.0)
}

fn median_u64(samples: &[u64]) -> u64 {
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let middle = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        sorted[middle - 1].saturating_add(sorted[middle]) / 2
    } else {
        sorted[middle]
    }
}

fn median_f64(samples: &[f64]) -> f64 {
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let middle = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        (sorted[middle - 1] + sorted[middle]) / 2.0
    } else {
        sorted[middle]
    }
}

const V4_QD4_LANES: usize = 4;
const V4_DEVICE_CAPACITY_HEADROOM_BYTES: usize = 2 * 1024 * 1024 * 1024;
// MTP P0 owns only its sliding KV persistently.  The routed FFN uses one
// ordinary ExpertSlot at a time and never enters the teacher cache.
const MTP1_DRAFT_PERSISTENT_BYTES: usize = 131_072;
// These are the candidate-owned allocations which outlive the fixed-weight
// upload through the streamed head.  They are deliberately separate from the
// fixed tensor ledger: the candidate keeps its sliding-KV input, raw-window
// attention cache, Q, sparse output, and score-reuse scratch live while the
// fixed block is resident and while the final head is streamed.
const MTP1_DRAFT_ENORM_HNORM_BYTES: usize = 2 * v4_state::HIDDEN * std::mem::size_of::<u16>();
const MTP1_DRAFT_KV_READY_BYTES: usize = v4_state::Q_HEAD_DIM * std::mem::size_of::<u16>();
const MTP1_DRAFT_Q_BYTES: usize =
    v4_state::Q_HEADS * v4_state::Q_HEAD_DIM * std::mem::size_of::<u16>();
const MTP1_DRAFT_CACHE_BYTES: usize =
    v4_state::KV_WINDOW * v4_state::Q_HEAD_DIM * std::mem::size_of::<u16>();
const MTP1_DRAFT_TOPK_BYTES: usize = v4_state::KV_WINDOW * std::mem::size_of::<i32>();
const MTP1_DRAFT_SPARSE_BYTES: usize =
    v4_state::Q_HEADS * v4_state::Q_HEAD_DIM * std::mem::size_of::<u16>();
const MTP1_DRAFT_SCORE_REUSE_BYTES: usize =
    2 * v4_state::Q_HEADS * v4_state::KV_WINDOW * std::mem::size_of::<f32>()
        + v4_state::Q_HEADS * (std::mem::size_of::<f32>() + std::mem::size_of::<u32>());
const MTP1_DRAFT_CANDIDATE_LOCAL_BYTES: usize = MTP1_DRAFT_ENORM_HNORM_BYTES
    + MTP1_DRAFT_KV_READY_BYTES
    + MTP1_DRAFT_Q_BYTES
    + MTP1_DRAFT_CACHE_BYTES
    + MTP1_DRAFT_TOPK_BYTES
    + MTP1_DRAFT_SPARSE_BYTES
    + MTP1_DRAFT_SCORE_REUSE_BYTES;
const MTP1_DRAFT_TRANSIENT_EXPERT_SLOT_BYTES: usize = 35_094_528;
const MTP1_DRAFT_EXTRA_CAPACITY_BYTES: usize = 43_230_456;
const MTP1_DRAFT_DEFAULT_MAX_SCORED_TRANSITIONS: usize = 8;

fn mtp1_draft_peak_bytes(
    persistent_bytes: usize,
    fixed_bytes: usize,
    e_projection_bytes: usize,
    h_projection_bytes: usize,
    expert_slot_bytes: usize,
    lm_head_bytes: usize,
) -> Result<usize, Box<dyn Error>> {
    let add = |left: usize, right: usize| {
        left.checked_add(right)
            .ok_or("MTP1 draft peak ledger overflow")
    };
    let candidate_base = add(persistent_bytes, MTP1_DRAFT_CANDIDATE_LOCAL_BYTES)?;
    let projection_peak = add(candidate_base, e_projection_bytes.max(h_projection_bytes))?;
    let fixed_expert_peak = add(add(candidate_base, fixed_bytes)?, expert_slot_bytes)?;
    let lm_head_peak = add(candidate_base, lm_head_bytes)?;
    Ok(projection_peak.max(fixed_expert_peak).max(lm_head_peak))
}

fn mtp1_draft_fixed_device_bytes(
    host: &v4_state::MtpDraftWeights,
) -> Result<usize, Box<dyn Error>> {
    let fp8 = |matrix: &v4_state::Fp8Matrix| matrix.weight.len().checked_add(matrix.scale.len());
    let bytes = [
        host.hc_attn_fn
            .len()
            .checked_mul(std::mem::size_of::<f32>()),
        host.hc_attn_base
            .len()
            .checked_mul(std::mem::size_of::<f32>()),
        host.hc_attn_scale
            .len()
            .checked_mul(std::mem::size_of::<f32>()),
        host.attn_norm.len().checked_mul(std::mem::size_of::<u16>()),
        fp8(&host.wq_a.matrix),
        host.wq_a.norm.len().checked_mul(std::mem::size_of::<u16>()),
        fp8(&host.wq_b),
        fp8(&host.wkv.matrix),
        host.wkv.norm.len().checked_mul(std::mem::size_of::<u16>()),
        host.attn_sink.len().checked_mul(std::mem::size_of::<f32>()),
        host.wo_a
            .weight
            .len()
            .checked_mul(std::mem::size_of::<u16>()),
        // Upload converts FP8 wo_a while the raw input remains live.
        Some(host.wo_a.weight.len()),
        fp8(&host.wo_b),
        host.hc_ffn_fn.len().checked_mul(std::mem::size_of::<f32>()),
        host.hc_ffn_base
            .len()
            .checked_mul(std::mem::size_of::<f32>()),
        host.hc_ffn_scale
            .len()
            .checked_mul(std::mem::size_of::<f32>()),
        host.ffn_norm.len().checked_mul(std::mem::size_of::<u16>()),
        host.gate.len().checked_mul(std::mem::size_of::<u16>()),
        host.router_bias
            .len()
            .checked_mul(std::mem::size_of::<f32>()),
        fp8(&host.shared.w1),
        fp8(&host.shared.w2),
        fp8(&host.shared.w3),
    ];
    bytes.into_iter().try_fold(0usize, |total, bytes| {
        total
            .checked_add(bytes.ok_or("MTP1 fixed device ledger overflow")?)
            .ok_or("MTP1 fixed device ledger overflow".into())
    })
}

fn mtp1_draft_fixed_device_bytes_from_geometry() -> Result<usize, Box<dyn Error>> {
    let fp8 = |input: usize, output: usize| {
        input
            .checked_mul(output)
            .and_then(|weight| {
                output
                    .div_ceil(v4_state::FP8_BLOCK)
                    .checked_mul(input / v4_state::FP8_BLOCK)
                    .and_then(|scale| weight.checked_add(scale))
            })
            .ok_or("MTP1 geometry fixed ledger overflow")
    };
    [
        v4_state::MIX * v4_state::HC * v4_state::HIDDEN * 4,
        v4_state::MIX * 4,
        3 * 4,
        v4_state::HIDDEN * 2,
        fp8(v4_state::HIDDEN, V4_Q_LORA_RANK)?,
        V4_Q_LORA_RANK * 2,
        fp8(V4_Q_LORA_RANK, v4_state::Q_HEADS * v4_state::Q_HEAD_DIM)?,
        fp8(v4_state::HIDDEN, v4_state::Q_HEAD_DIM)?,
        v4_state::Q_HEAD_DIM * 2,
        v4_state::Q_HEADS * 4,
        (v4_state::Q_HEADS * v4_state::Q_HEAD_DIM / v4_state::O_GROUPS)
            * (v4_state::O_GROUPS * v4_state::O_LORA_RANK)
            * 3,
        fp8(v4_state::O_GROUPS * v4_state::O_LORA_RANK, v4_state::HIDDEN)?,
        v4_state::MIX * v4_state::HC * v4_state::HIDDEN * 4,
        v4_state::MIX * 4,
        3 * 4,
        v4_state::HIDDEN * 2,
        v4_router::V4_ROUTED_EXPERTS * v4_state::HIDDEN * 2,
        v4_router::V4_ROUTED_EXPERTS * 4,
        fp8(v4_state::HIDDEN, v4_expert::V4_EXPERT_INTERMEDIATE)?,
        fp8(v4_expert::V4_EXPERT_INTERMEDIATE, v4_state::HIDDEN)?,
        fp8(v4_state::HIDDEN, v4_expert::V4_EXPERT_INTERMEDIATE)?,
    ]
    .into_iter()
    .try_fold(0usize, |total, bytes| {
        total
            .checked_add(bytes)
            .ok_or("MTP1 geometry fixed ledger overflow".into())
    })
}

fn mtp1_stationary_peak_reserve_bytes(
    model_dir: &std::path::Path,
    borrows_qd4_lane: bool,
) -> Result<usize, Box<dyn Error>> {
    let host = v4_state::load_mtp_draft_weights(model_dir)?;
    mtp1_draft_peak_bytes(
        MTP1_DRAFT_PERSISTENT_BYTES,
        mtp1_draft_fixed_device_bytes(&host)?,
        host.e_proj.source_bytes(),
        host.h_proj.source_bytes(),
        usize::from(!borrows_qd4_lane) * MTP1_DRAFT_TRANSIENT_EXPERT_SLOT_BYTES,
        CERTIFIED_ARGMAX_DEVICE_TEMP_BYTES,
    )
}

/// One extra routed slot is needed only when stationary MTP cannot borrow a
/// drained QD4 lane.  Keep its pinned staging reservation beside the raw-cache
/// host selection, rather than discovering it after that cache has filled RAM.
fn mtp1_stationary_pinned_host_reserve_bytes(borrows_qd4_lane: bool) -> usize {
    if borrows_qd4_lane {
        0
    } else {
        V4_QD4_PINNED_HOST_BYTES / V4_QD4_LANES
    }
}

fn generic_stationary_persistent_reserve_bytes(
    manifests: &[v4_state::LayerExecutionManifest],
    plan: v4_state::LayerStationaryRequestPlan,
) -> Result<usize, Box<dyn Error>> {
    let bytes = |elements: usize, element_bytes: usize| {
        elements
            .checked_mul(element_bytes)
            .ok_or("generic stationary persistent reserve overflow")
    };
    let state_bytes = |path: GenericStationaryPath| -> Result<usize, Box<dyn Error>> {
        let raw = bytes(v4_state::KV_WINDOW * v4_state::Q_HEAD_DIM, 2)?;
        let tails = bytes(path.tail_elements(), 4)?
            .checked_mul(2)
            .ok_or("generic stationary persistent tail reserve overflow")?;
        let compressed = bytes(path.compressed_rows(plan) * v4_state::Q_HEAD_DIM, 2)?;
        let indexer = if path.is_csa() {
            let indexer_kv = bytes(plan.csa_compressed_rows * 128, 2)?;
            bytes(8 * 256, 4)?
                .checked_mul(2)
                .and_then(|value| value.checked_add(indexer_kv))
                .ok_or("generic stationary indexer reserve overflow")?
        } else {
            0
        };
        raw.checked_add(tails)
            .and_then(|value| value.checked_add(compressed))
            .and_then(|value| value.checked_add(indexer))
            .ok_or("generic stationary persistent reserve overflow".into())
    };
    manifests.iter().try_fold(0usize, |total, manifest| {
        total
            .checked_add(state_bytes(GenericStationaryPath::from_ratio(
                manifest.compression_ratio,
            ))?)
            .ok_or("generic stationary persistent reserve overflow".into())
    })
}

fn generic_stationary_scratch_reserve_bytes(
    plan: v4_state::LayerStationaryRequestPlan,
) -> Result<usize, Box<dyn Error>> {
    generic_stationary_scratch_reserve_bytes_with_strict(plan, strict_compressor_enabled())
}

fn generic_stationary_scratch_reserve_bytes_with_strict(
    plan: v4_state::LayerStationaryRequestPlan,
    strict: bool,
) -> Result<usize, Box<dyn Error>> {
    DeviceRequestScratchGeometry::new(plan, strict)?.num_bytes()
}

fn stationary_mtp1_target_strict_scratch_contract(
    plan: v4_state::LayerStationaryRequestPlan,
    target_strict: bool,
    pinned_cublas: Option<&std::path::Path>,
) -> Result<usize, Box<dyn Error>> {
    if target_strict && pinned_cublas.is_none() {
        return Err(
            "strict stationary MTP capacity requires pinned COLIBRI_CUBLAS_EXACT_LIB".into(),
        );
    }
    generic_stationary_scratch_reserve_bytes_with_strict(plan, target_strict)
}

#[derive(Default)]
struct StationaryMtpCapacityProbe {
    device_u8: Vec<DeviceBuffer<u8>>,
    device_u16: Vec<DeviceBuffer<u16>>,
    device_f32: Vec<DeviceBuffer<f32>>,
    device_i32: Vec<DeviceBuffer<i32>>,
    device_u32: Vec<DeviceBuffer<u32>>,
    pinned_u8: Vec<PinnedHostBuffer<u8>>,
    pinned_u16: Vec<PinnedHostBuffer<u16>>,
    pinned_f32: Vec<PinnedHostBuffer<f32>>,
    host: Vec<Vec<u8>>,
    raw_host: Vec<Vec<u8>>,
    generic_scratch: Option<DeviceRequestScratch>,
}

const STATIONARY_MTP1_CAPACITY_PREFLIGHT_SCHEMA: &str = "colibri-v4-stationary-mtp1-capacity-v2";
const STATIONARY_MTP1_CAPACITY_PREFLIGHT_TARGET_STRICT: bool = true;

impl StationaryMtpCapacityProbe {
    fn device_bytes(&self) -> usize {
        self.device_u8
            .iter()
            .map(DeviceBuffer::num_bytes)
            .sum::<usize>()
            + self
                .device_u16
                .iter()
                .map(DeviceBuffer::num_bytes)
                .sum::<usize>()
            + self
                .device_f32
                .iter()
                .map(DeviceBuffer::num_bytes)
                .sum::<usize>()
            + self
                .device_i32
                .iter()
                .map(DeviceBuffer::num_bytes)
                .sum::<usize>()
            + self
                .device_u32
                .iter()
                .map(DeviceBuffer::num_bytes)
                .sum::<usize>()
            + self
                .generic_scratch
                .as_ref()
                .map_or(0, DeviceRequestScratch::num_bytes)
    }

    fn pinned_bytes(&self) -> usize {
        self.pinned_u8
            .iter()
            .map(PinnedHostBuffer::num_bytes)
            .sum::<usize>()
            + self
                .pinned_u16
                .iter()
                .map(PinnedHostBuffer::num_bytes)
                .sum::<usize>()
            + self
                .pinned_f32
                .iter()
                .map(PinnedHostBuffer::num_bytes)
                .sum::<usize>()
    }

    fn raw_host_bytes(&self) -> usize {
        self.raw_host.iter().map(Vec::len).sum()
    }

    fn host_bytes(&self) -> usize {
        self.host.iter().map(Vec::len).sum()
    }

    fn u8(&mut self, stream: &CudaStream, elements: usize) -> Result<(), Box<dyn Error>> {
        self.device_u8.push(DeviceBuffer::zeroed(stream, elements)?);
        Ok(())
    }

    fn u16(&mut self, stream: &CudaStream, elements: usize) -> Result<(), Box<dyn Error>> {
        self.device_u16
            .push(DeviceBuffer::zeroed(stream, elements)?);
        Ok(())
    }

    fn f32(&mut self, stream: &CudaStream, elements: usize) -> Result<(), Box<dyn Error>> {
        self.device_f32
            .push(DeviceBuffer::zeroed(stream, elements)?);
        Ok(())
    }

    fn i32(&mut self, stream: &CudaStream, elements: usize) -> Result<(), Box<dyn Error>> {
        self.device_i32
            .push(DeviceBuffer::zeroed(stream, elements)?);
        Ok(())
    }

    fn u32(&mut self, stream: &CudaStream, elements: usize) -> Result<(), Box<dyn Error>> {
        self.device_u32
            .push(DeviceBuffer::zeroed(stream, elements)?);
        Ok(())
    }
}

fn generic_stationary_audit_mirror_host_bytes(
    manifests: &[v4_state::LayerExecutionManifest],
    plan: v4_state::LayerStationaryRequestPlan,
) -> Result<usize, Box<dyn Error>> {
    let compressor_bytes = |ratio: usize, width: usize| {
        let coefficient = 1 + usize::from(ratio == 4);
        coefficient
            .checked_mul(ratio)
            .and_then(|slots| slots.checked_mul(coefficient))
            .and_then(|elements| elements.checked_mul(width))
            .and_then(|elements| elements.checked_mul(2 * std::mem::size_of::<f32>()))
            .ok_or("generic stationary audit mirror compressor ledger overflow")
    };
    let mut total = plan.token_scratch_bytes;
    for manifest in manifests {
        let ratio = manifest.compression_ratio.get();
        let compressed_rows = if ratio == 4 {
            plan.csa_compressed_rows
        } else {
            plan.hca_compressed_rows
        };
        let layer = plan
            .raw_kv_rows
            .checked_mul(v4_state::Q_HEAD_DIM)
            .and_then(|elements| elements.checked_mul(std::mem::size_of::<u16>()))
            .and_then(|bytes| {
                bytes.checked_add(
                    compressed_rows
                        .checked_mul(v4_state::Q_HEAD_DIM)
                        .and_then(|elements| elements.checked_mul(std::mem::size_of::<u16>()))?,
                )
            })
            .and_then(|bytes| {
                (ratio == 4)
                    .then(|| {
                        compressed_rows
                            .checked_mul(128)
                            .and_then(|elements| elements.checked_mul(std::mem::size_of::<u16>()))
                    })
                    .flatten()
                    .map_or(Some(bytes), |indexer| bytes.checked_add(indexer))
            })
            .and_then(|bytes| {
                bytes.checked_add(compressor_bytes(ratio, v4_state::Q_HEAD_DIM).ok()?)
            })
            .and_then(|bytes| {
                if ratio == 4 {
                    bytes.checked_add(compressor_bytes(4, 128).ok()?)
                } else {
                    Some(bytes)
                }
            })
            .ok_or("generic stationary audit mirror layer ledger overflow")?;
        total = total
            .checked_add(layer)
            .ok_or("generic stationary audit mirror ledger overflow")?;
    }
    Ok(total)
}

fn probe_mtp1_candidate_locals(
    probe: &mut StationaryMtpCapacityProbe,
    stream: &CudaStream,
) -> Result<(), Box<dyn Error>> {
    // Keep this list in the same ownership shape as run_mtp1_draft_candidate.
    // `q`, cache, sparse, and score scratch remain live until after the head.
    probe.u16(stream, v4_state::HIDDEN)?; // enorm
    probe.u16(stream, v4_state::HIDDEN)?; // hnorm
    probe.u16(stream, v4_state::Q_HEAD_DIM)?; // kv_ready
    probe.u16(stream, v4_state::Q_HEADS * v4_state::Q_HEAD_DIM)?; // q
    probe.u16(stream, v4_state::KV_WINDOW * v4_state::Q_HEAD_DIM)?; // cache
    probe.i32(stream, v4_state::KV_WINDOW)?; // topk
    probe.u16(stream, v4_state::Q_HEADS * v4_state::Q_HEAD_DIM)?; // sparse
    probe.f32(stream, 2 * v4_state::Q_HEADS * v4_state::KV_WINDOW)?; // scores/probabilities
    probe.f32(stream, v4_state::Q_HEADS)?; // denominators
    probe.u32(stream, v4_state::Q_HEADS)?; // valid_counts
    let bytes = MTP1_DRAFT_ENORM_HNORM_BYTES
        + MTP1_DRAFT_KV_READY_BYTES
        + MTP1_DRAFT_Q_BYTES
        + MTP1_DRAFT_CACHE_BYTES
        + MTP1_DRAFT_TOPK_BYTES
        + MTP1_DRAFT_SPARSE_BYTES
        + MTP1_DRAFT_SCORE_REUSE_BYTES;
    if bytes != MTP1_DRAFT_CANDIDATE_LOCAL_BYTES {
        return Err("MTP1 candidate-local allocation ledger drifted".into());
    }
    Ok(())
}

fn stationary_mtp1_capacity_preflight_rejects_execution_flags() -> Result<(), Box<dyn Error>> {
    for name in [
        "COLIBRI_V4_FULL61_B1",
        "COLIBRI_V4_LAYER_STATIONARY_PREFILL",
        "COLIBRI_V4_GENERIC_DEVICE_STATIONARY",
        "COLIBRI_V4_STATIONARY_MTP1_DRAFT",
        "COLIBRI_V4_MTP1_DRAFT_P0",
        "COLIBRI_V4_ROUTED_QD4",
        "COLIBRI_V4_PHASE_PROFILE",
        "COLIBRI_V4_RESULT_JSONL",
        "COLIBRI_V4_PROMPT_TEXT",
        "COLIBRI_V4_CHAT_USER",
        "COLIBRI_V4_TOKEN_IDS",
        "COLIBRI_V4_TOKEN_ID",
        "COLIBRI_V4_GREEDY_STEPS",
    ] {
        if std::env::var_os(name).is_some() {
            return Err(format!(
                "COLIBRI_V4_STATIONARY_MTP1_CAPACITY_PREFLIGHT=1 rejects execution flag {name}"
            )
            .into());
        }
    }
    Ok(())
}

fn stationary_mtp1_preflight_plan(
    prompt_rows: Option<&std::ffi::OsStr>,
    max_new_tokens: Option<&std::ffi::OsStr>,
) -> Result<v4_state::LayerStationaryRequestPlan, Box<dyn Error>> {
    let (prompt_rows, max_new_tokens) = match (prompt_rows, max_new_tokens) {
        (None, None) => (5, 3),
        (Some(prompt_rows), Some(max_new_tokens)) => {
            let parse = |name: &str, value: &std::ffi::OsStr| -> Result<usize, Box<dyn Error>> {
                value
                    .to_str()
                    .ok_or_else(|| format!("{name} must be valid UTF-8"))?
                    .parse::<usize>()
                    .map_err(|_| format!("{name} must be a base-10 usize").into())
            };
            (
                parse("COLIBRI_V4_STATIONARY_PREFLIGHT_PROMPT_ROWS", prompt_rows)?,
                parse(
                    "COLIBRI_V4_STATIONARY_PREFLIGHT_MAX_NEW_TOKENS",
                    max_new_tokens,
                )?,
            )
        }
        _ => {
            return Err(
                "COLIBRI_V4_STATIONARY_PREFLIGHT_PROMPT_ROWS and COLIBRI_V4_STATIONARY_PREFLIGHT_MAX_NEW_TOKENS must be set together"
                    .into(),
            );
        }
    };
    if prompt_rows == 0 {
        return Err("stationary preflight prompt rows must be nonzero".into());
    }
    v4_state::LayerStationaryRequestPlan::new(prompt_rows, max_new_tokens)
}

fn probe_generic_stationary_resources(
    probe: &mut StationaryMtpCapacityProbe,
    stream: &CudaStream,
    manifests: &[v4_state::LayerExecutionManifest],
    plan: v4_state::LayerStationaryRequestPlan,
    strict: bool,
) -> Result<(), Box<dyn Error>> {
    for manifest in manifests {
        let path = GenericStationaryPath::from_ratio(manifest.compression_ratio);
        probe.u16(stream, v4_state::KV_WINDOW * v4_state::Q_HEAD_DIM)?;
        probe.f32(stream, path.tail_elements())?;
        probe.f32(stream, path.tail_elements())?;
        probe.u16(stream, path.compressed_rows(plan) * v4_state::Q_HEAD_DIM)?;
        if path.is_csa() {
            probe.f32(stream, 8 * 256)?;
            probe.f32(stream, 8 * 256)?;
            probe.u16(stream, plan.csa_compressed_rows * 128)?;
        }
    }
    probe.generic_scratch = Some(DeviceRequestScratch::new_with_strict(stream, plan, strict)?);
    Ok(())
}

fn probe_qd4_lanes(
    probe: &mut StationaryMtpCapacityProbe,
    context: &std::sync::Arc<CudaContext>,
    stream: &CudaStream,
) -> Result<(), Box<dyn Error>> {
    for _ in 0..V4_QD4_LANES {
        for bytes in [
            v4_expert::V4_W1_BYTES,
            v4_expert::V4_W1_SCALE_BYTES,
            v4_expert::V4_W2_BYTES,
            v4_expert::V4_W2_SCALE_BYTES,
            v4_expert::V4_W1_BYTES,
            v4_expert::V4_W1_SCALE_BYTES,
        ] {
            probe
                .pinned_u8
                .push(PinnedHostBuffer::zeroed(context, bytes)?);
            probe.u8(stream, bytes)?;
        }
        probe
            .pinned_u16
            .push(PinnedHostBuffer::zeroed(context, v4_expert::V4_HIDDEN)?);
        probe
            .pinned_f32
            .push(PinnedHostBuffer::zeroed(context, v4_expert::V4_HIDDEN)?);
        probe.u16(stream, v4_expert::V4_HIDDEN)?;
        probe.f32(stream, v4_expert::V4_HIDDEN)?;
        for elements in [v4_expert::V4_EXPERT_INTERMEDIATE; 3] {
            probe.u16(stream, elements)?;
        }
        probe.f32(stream, v4_expert::V4_EXPERT_INTERMEDIATE)?;
        probe.u16(stream, v4_expert::V4_HIDDEN)?;
        probe.f32(stream, v4_expert::V4_HIDDEN)?;
    }
    Ok(())
}

/// Builds the same request-shared scratch constructor used by generic Full61
/// execution, but stops before any layer row or kernel launch.  Strict target
/// mode is explicit because this isolated process intentionally has no Full61
/// execution flags set.
fn run_stationary_mtp1_scratch_constructor_preflight() -> Result<bool, Box<dyn Error>> {
    if !optional_one_flag("COLIBRI_V4_STATIONARY_MTP1_SCRATCH_CONSTRUCTOR_PREFLIGHT")? {
        return Ok(false);
    }
    stationary_mtp1_capacity_preflight_rejects_execution_flags()?;
    let plan = stationary_mtp1_preflight_plan(
        std::env::var_os("COLIBRI_V4_STATIONARY_PREFLIGHT_PROMPT_ROWS").as_deref(),
        std::env::var_os("COLIBRI_V4_STATIONARY_PREFLIGHT_MAX_NEW_TOKENS").as_deref(),
    )?;
    let target_strict_compressor = true;
    let runtime_strict_compressor = strict_compressor_enabled();
    let pinned_cublas = std::env::var_os("COLIBRI_CUBLAS_EXACT_LIB").map(std::path::PathBuf::from);
    let expected_bytes = stationary_mtp1_target_strict_scratch_contract(
        plan,
        target_strict_compressor,
        pinned_cublas.as_deref(),
    )?;
    drop(cublas_exact::CublasExact::load()?);
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let scratch = DeviceRequestScratch::new_with_strict(&stream, plan, target_strict_compressor)?;
    let actual_bytes = scratch.num_bytes();
    if actual_bytes != expected_bytes {
        return Err("constructor preflight reached an inconsistent scratch total".into());
    }
    println!(
        "{}",
        serde_json::json!({
            "event": "stationary_mtp1_scratch_constructor_preflight",
            "model_execution": false,
            "layer_execution": false,
            "kernel_launches": 0,
            "metadata_shards_opened": 0,
            "tensor_payload_bytes_read": 0,
            "target": {"prompt_rows": plan.prompt_rows, "max_new_tokens": plan.max_new_tokens, "context_rows": plan.context_rows, "csa_compressed_rows": plan.csa_compressed_rows, "hca_compressed_rows": plan.hca_compressed_rows, "generic_device_stationary": true, "routed_qd4": true, "generic_state_audit": true, "stationary_mtp1_draft": true},
            "target_strict_compressor": target_strict_compressor,
            "runtime_strict_compressor": runtime_strict_compressor,
            "expected_bytes": expected_bytes,
            "actual_bytes": actual_bytes,
            "expected_breakdown": DeviceRequestScratchGeometry::new(plan, target_strict_compressor)?.byte_breakdown()?.json(),
            "actual_breakdown": scratch.byte_breakdown().json(),
        })
    );
    drop(scratch);
    stream.synchronize()?;
    Ok(true)
}
const V4_Q_LORA_RANK: usize = 1_536;
const V4_DEVICE_TOKEN_STATE_CSA_WITHOUT_Q_LORA_F32_BYTES: usize =
    83_863_054 - V4_Q_LORA_RANK * std::mem::size_of::<f32>();
const V4_DEVICE_TOKEN_STATE_HCA_WITHOUT_Q_LORA_F32_BYTES: usize = 40_249_870;
const fn device_token_state_ledger_bytes(csa: bool, q_lora_rank: usize) -> usize {
    (if csa {
        V4_DEVICE_TOKEN_STATE_CSA_WITHOUT_Q_LORA_F32_BYTES
    } else {
        V4_DEVICE_TOKEN_STATE_HCA_WITHOUT_Q_LORA_F32_BYTES
    }) + q_lora_rank * std::mem::size_of::<f32>()
}
const V4_DEVICE_TOKEN_STATE_MAX_BYTES: usize =
    device_token_state_ledger_bytes(true, V4_Q_LORA_RANK);
const V4_DEVICE_TOKEN_STATE_HCA_BYTES: usize =
    device_token_state_ledger_bytes(false, V4_Q_LORA_RANK);
const V4_REQUEST_ROPE_LUT_ROWS: usize = 5;
const V4_SHORT_DEVICE_DECODE_MAX_NEW_TOKENS: usize = 5;
// ponytail: this is a short exact gate, not paged serving; lift only when a
// semantic gate covers a paged/rolling DeviceTokenState beyond nine rows.
const V4_SHORT_DEVICE_DECODE_MAX_ROWS: usize = 9;

fn csa_compressed_rows_after(position: usize) -> usize {
    (position + 1) / 4
}
const V4_REQUEST_ROPE_LUT_BYTES: usize =
    2 * V4_REQUEST_ROPE_LUT_ROWS * v4_state::Q_ROPE_DIM * std::mem::size_of::<f32>();
const V4_TID2EID_TRANSIENT_MAX_BYTES: usize = v4_router::V4_TOPK * std::mem::size_of::<i64>();

fn device_attention_expected_h2d_bytes(
    rope_lut_bytes: usize,
    dispatches: usize,
) -> Result<usize, Box<dyn Error>> {
    rope_lut_bytes
        .checked_add(
            dispatches
                .checked_mul(v4_state::HC * v4_state::HIDDEN * std::mem::size_of::<u16>())
                .ok_or("device attention H2D byte count overflow")?,
        )
        .ok_or("device attention H2D byte count overflow".into())
}
const V4_QD4_PINNED_HOST_BYTES: usize = V4_QD4_LANES
    * (v4_expert::V4_ROUTED_EXPERT_BYTES
        + v4_expert::V4_HIDDEN * std::mem::size_of::<u16>()
        + v4_expert::V4_HIDDEN * std::mem::size_of::<f32>());
const V4_QD4_DEVICE_BYTES: usize = V4_QD4_LANES
    * (v4_expert::V4_ROUTED_EXPERT_BYTES
        + v4_expert::V4_HIDDEN * std::mem::size_of::<u16>()
        + v4_expert::V4_HIDDEN * std::mem::size_of::<f32>()
        + 3 * v4_expert::V4_EXPERT_INTERMEDIATE * std::mem::size_of::<u16>()
        + v4_expert::V4_EXPERT_INTERMEDIATE * std::mem::size_of::<f32>()
        + v4_expert::V4_HIDDEN * std::mem::size_of::<u16>()
        + v4_expert::V4_HIDDEN * std::mem::size_of::<f32>());
const V4_QD4_IMMUTABLE_EXPERT_CACHE_SLOTS: usize = 32;
const V4_QD4_IMMUTABLE_EXPERT_CACHE_HOST_BYTES: usize = V4_QD4_IMMUTABLE_EXPERT_CACHE_SLOTS
    * (v4_expert::V4_ROUTED_EXPERT_BYTES
        + v4_expert::V4_HIDDEN * std::mem::size_of::<u16>()
        + v4_expert::V4_HIDDEN * std::mem::size_of::<f32>());
const V4_QD4_IMMUTABLE_EXPERT_CACHE_DEVICE_BYTES: usize = V4_QD4_IMMUTABLE_EXPERT_CACHE_SLOTS
    * (v4_expert::V4_ROUTED_EXPERT_BYTES
        + v4_expert::V4_HIDDEN * std::mem::size_of::<u16>()
        + v4_expert::V4_HIDDEN * std::mem::size_of::<f32>()
        + 3 * v4_expert::V4_EXPERT_INTERMEDIATE * std::mem::size_of::<u16>()
        + v4_expert::V4_EXPERT_INTERMEDIATE * std::mem::size_of::<f32>()
        + v4_expert::V4_HIDDEN * std::mem::size_of::<u16>()
        + v4_expert::V4_HIDDEN * std::mem::size_of::<f32>());

struct FlashExpertTrace {
    input_fp8: Vec<u8>,
    input_scale: Vec<u8>,
    w1: Vec<u16>,
    w3: Vec<u16>,
    swiglu: Vec<u16>,
    w2_input_fp8: Vec<u8>,
    w2_input_scale: Vec<u8>,
    output: Vec<u16>,
}

struct Qd4ImmutableCacheLease<'a> {
    expert: u16,
    slot: ExpertSlot<'a>,
    miss: bool,
}

#[derive(Clone, Copy, Default)]
struct Qd4ImmutableExpertCacheTelemetry {
    route_executions: usize,
    expert_union_loads: usize,
    source_loads: usize,
    source_bytes: usize,
    h2d_loads: usize,
    h2d_bytes: usize,
    hits: usize,
    d2d_copies: usize,
    d2d_bytes: usize,
    max_occupancy: usize,
    completed_layers: usize,
    worker_errors: usize,
    fallbacks: usize,
}

struct Qd4ImmutableExpertCache<'a> {
    free: Vec<ExpertSlot<'a>>,
    layer: Option<usize>,
    resident: BTreeMap<u16, ExpertSlot<'a>>,
    telemetry: Qd4ImmutableExpertCacheTelemetry,
}

#[derive(Clone)]
struct RoutedReplaySample {
    inclusive_ns: u64,
    setup_ns: u64,
    read_ns: u64,
    read_wall_ns: u64,
    h2d_ns: u64,
    kernel_ns: u64,
    lane_occupancy: f64,
    submit_ns: [u64; v4_router::V4_TOPK],
    ready_ns: [u64; v4_router::V4_TOPK],
    timeline: Option<RoutedQd4Timeline>,
}

#[derive(Clone)]
struct RoutedLaneTimeline {
    submit_ns: u64,
    read_start_ns: u64,
    read_end_ns: u64,
    h2d_enqueued_ns: u64,
    kernel_enqueued_ns: u64,
    d2h_enqueued_ns: u64,
    complete_ns: u64,
    h2d_event_ns: u64,
    kernel_event_ns: u64,
}

#[derive(Clone)]
struct RoutedQd4Timeline {
    lanes: [Option<RoutedLaneTimeline>; v4_router::V4_TOPK],
    canonical_reduction_start_ns: u64,
    canonical_reduction_end_ns: u64,
}

impl RoutedQd4Timeline {
    fn as_json(&self) -> serde_json::Value {
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

struct RoutedLaneExecution {
    read_ns: u64,
    h2d_ns: u64,
    kernel_ns: u64,
    timeline: Option<RoutedLaneTimeline>,
}

struct Qd4RoutedRun<'a> {
    output: Vec<u16>,
    contributions: Option<Vec<Vec<f32>>>,
    sample: RoutedReplaySample,
    cache_miss_count: usize,
    cache_leases: Vec<Qd4ImmutableCacheLease<'a>>,
}

fn run_serial_routed_replay(
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

struct RoutedLaneJob<J> {
    route_slot: usize,
    iteration_started: std::time::Instant,
    profile_submit_ns: Option<u64>,
    payload: J,
}

struct RoutedLaneDone<J> {
    route_slot: usize,
    read_ns: u64,
    read_ready_ns: u64,
    h2d_ns: u64,
    kernel_ns: u64,
    ready_ns: u64,
    job_ns: u64,
    timeline: Option<RoutedLaneTimeline>,
    cache_lease: J,
}

struct SupervisedLaneDone<T> {
    lane: usize,
    value: T,
}

struct SupervisedQd4Batch<T> {
    iteration_started: std::time::Instant,
    submit_ns: [u64; v4_router::V4_TOPK],
    completions: Vec<T>,
}

fn routed_replay_json(sample: &RoutedReplaySample) -> serde_json::Value {
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

fn routed_metric(
    samples: &[RoutedReplaySample],
    get: impl Fn(&RoutedReplaySample) -> u64,
) -> (f64, f64) {
    let values = samples.iter().map(get).collect::<Vec<_>>();
    median_p95_ms(&values)
}

struct DeviceFp8Matrix {
    weight: DeviceBuffer<u8>,
    scale: DeviceBuffer<u8>,
    in_features: usize,
    out_features: usize,
}

impl DeviceFp8Matrix {
    fn new(stream: &CudaStream, matrix: &v4_state::Fp8Matrix) -> Result<Self, Box<dyn Error>> {
        Ok(Self {
            weight: DeviceBuffer::from_host(stream, &matrix.weight)?,
            scale: DeviceBuffer::from_host(stream, &matrix.scale)?,
            in_features: matrix.in_features,
            out_features: matrix.out_features,
        })
    }

    fn num_bytes(&self) -> usize {
        self.weight.num_bytes() + self.scale.num_bytes()
    }
}

struct DeviceLayerIndexer {
    wq_b: DeviceFp8Matrix,
    weights_proj: DeviceBuffer<u16>,
    compressor_wkv: DeviceBuffer<u16>,
    compressor_wgate: DeviceBuffer<u16>,
    compressor_ape: DeviceBuffer<f32>,
    compressor_norm: DeviceBuffer<u16>,
}

struct LayerDeviceWeights {
    spec: &'static v4_spec::V4Spec,
    layer: usize,
    compression_ratio: v4_state::CompressionRatio,
    hc_attn_fn: DeviceBuffer<f32>,
    hc_attn_base: DeviceBuffer<f32>,
    hc_attn_scale: DeviceBuffer<f32>,
    attn_norm: DeviceBuffer<u16>,
    wq_a: DeviceFp8Matrix,
    q_norm: DeviceBuffer<u16>,
    wq_b: DeviceFp8Matrix,
    wkv: DeviceFp8Matrix,
    kv_norm: DeviceBuffer<u16>,
    compressor_wkv: DeviceBuffer<u16>,
    compressor_wgate: DeviceBuffer<u16>,
    compressor_ape: DeviceBuffer<f32>,
    compressor_norm: DeviceBuffer<u16>,
    indexer: Option<DeviceLayerIndexer>,
    attn_sink: DeviceBuffer<f32>,
    wo_a: DeviceBuffer<u16>,
    wo_b: DeviceFp8Matrix,
    hc_ffn_fn: DeviceBuffer<f32>,
    hc_ffn_base: DeviceBuffer<f32>,
    hc_ffn_scale: DeviceBuffer<f32>,
    ffn_norm: DeviceBuffer<u16>,
    gate: DeviceBuffer<u16>,
    router_bias: DeviceBuffer<f32>,
    hash_router: bool,
    shared_w1: DeviceFp8Matrix,
    shared_w2: DeviceFp8Matrix,
    shared_w3: DeviceFp8Matrix,
}

impl LayerDeviceWeights {
    fn new_l0(
        stream: &CudaStream,
        state: &v4_state::L0State,
        projections: &v4_state::L0Fp8Projections,
        compressor: &v4_state::CompressorWeights,
        tail: &v4_state::L0TailWeights,
    ) -> Result<Self, Box<dyn Error>> {
        Ok(Self {
            spec: &v4_spec::PRO,
            layer: 0,
            compression_ratio: v4_state::CompressionRatio::Hca128,
            hc_attn_fn: DeviceBuffer::from_host(stream, &state.hc_fn)?,
            hc_attn_base: DeviceBuffer::from_host(stream, &state.hc_base)?,
            hc_attn_scale: DeviceBuffer::from_host(stream, &state.hc_scale)?,
            attn_norm: DeviceBuffer::from_host(stream, &state.attn_norm)?,
            wq_a: DeviceFp8Matrix::new(stream, &projections.wq_a.matrix)?,
            q_norm: DeviceBuffer::from_host(stream, &projections.wq_a.norm)?,
            wq_b: DeviceFp8Matrix::new(stream, &projections.wq_b)?,
            wkv: DeviceFp8Matrix::new(stream, &projections.wkv.matrix)?,
            kv_norm: DeviceBuffer::from_host(stream, &projections.wkv.norm)?,
            compressor_wkv: DeviceBuffer::from_host(stream, &compressor.wkv_bf16)?,
            compressor_wgate: DeviceBuffer::from_host(stream, &compressor.wgate_bf16)?,
            compressor_ape: DeviceBuffer::from_host(stream, &compressor.ape_f32)?,
            compressor_norm: DeviceBuffer::from_host(
                stream,
                &vec![v4_router::f32_to_bf16_rne(1.0); v4_state::Q_HEAD_DIM],
            )?,
            indexer: None,
            attn_sink: DeviceBuffer::from_host(stream, &tail.attn_sink)?,
            wo_a: DeviceBuffer::from_host(stream, &tail.wo_a_bf16)?,
            wo_b: DeviceFp8Matrix::new(stream, &tail.wo_b)?,
            hc_ffn_fn: DeviceBuffer::from_host(stream, &tail.hc_ffn_fn)?,
            hc_ffn_base: DeviceBuffer::from_host(stream, &tail.hc_ffn_base)?,
            hc_ffn_scale: DeviceBuffer::from_host(stream, &tail.hc_ffn_scale)?,
            ffn_norm: DeviceBuffer::from_host(stream, &tail.ffn_norm)?,
            gate: DeviceBuffer::from_host(stream, &tail.gate)?,
            router_bias: DeviceBuffer::from_host(
                stream,
                &vec![0.0_f32; v4_router::V4_ROUTED_EXPERTS],
            )?,
            hash_router: true,
            shared_w1: DeviceFp8Matrix::new(stream, &tail.shared.w1)?,
            shared_w2: DeviceFp8Matrix::new(stream, &tail.shared.w2)?,
            shared_w3: DeviceFp8Matrix::new(stream, &tail.shared.w3)?,
        })
    }

    fn new_layer(
        stream: &CudaStream,
        state: &v4_state::LayerStateWeights,
        projections: &v4_state::L0Fp8Projections,
        tail: &v4_state::L0TailWeights,
    ) -> Result<Self, Box<dyn Error>> {
        if tail.wo_a_raw.is_some() {
            return Err("raw wo_a requires the production GPU decode upload path".into());
        }
        Self::new_layer_with_wo_a(stream, state, projections, tail, None)
    }

    fn new_layer_with_wo_a(
        stream: &CudaStream,
        state: &v4_state::LayerStateWeights,
        projections: &v4_state::L0Fp8Projections,
        tail: &v4_state::L0TailWeights,
        wo_a: Option<DeviceBuffer<u16>>,
    ) -> Result<Self, Box<dyn Error>> {
        let (router_bias, hash_router) = match &state.router {
            v4_state::LayerRouterWeights::Hash { .. } => {
                (vec![0.0_f32; state.spec.routed_experts], true)
            }
            v4_state::LayerRouterWeights::Learned { bias } => (bias.clone(), false),
        };
        let indexer = state
            .indexer
            .as_ref()
            .map(|indexer| {
                Ok::<_, Box<dyn Error>>(DeviceLayerIndexer {
                    wq_b: DeviceFp8Matrix::new(stream, &indexer.wq_b)?,
                    weights_proj: DeviceBuffer::from_host(stream, &indexer.weights_proj_bf16)?,
                    compressor_wkv: DeviceBuffer::from_host(stream, &indexer.compressor.wkv_bf16)?,
                    compressor_wgate: DeviceBuffer::from_host(
                        stream,
                        &indexer.compressor.wgate_bf16,
                    )?,
                    compressor_ape: DeviceBuffer::from_host(stream, &indexer.compressor.ape_f32)?,
                    compressor_norm: DeviceBuffer::from_host(
                        stream,
                        &indexer.compressor.norm_bf16,
                    )?,
                })
            })
            .transpose()?;
        Ok(Self {
            spec: state.spec,
            layer: state.layer,
            compression_ratio: state.compression_ratio,
            hc_attn_fn: DeviceBuffer::from_host(stream, &state.hc_fn)?,
            hc_attn_base: DeviceBuffer::from_host(stream, &state.hc_base)?,
            hc_attn_scale: DeviceBuffer::from_host(stream, &state.hc_scale)?,
            attn_norm: DeviceBuffer::from_host(stream, &state.attn_norm)?,
            wq_a: DeviceFp8Matrix::new(stream, &projections.wq_a.matrix)?,
            q_norm: DeviceBuffer::from_host(stream, &projections.wq_a.norm)?,
            wq_b: DeviceFp8Matrix::new(stream, &projections.wq_b)?,
            wkv: DeviceFp8Matrix::new(stream, &projections.wkv.matrix)?,
            kv_norm: DeviceBuffer::from_host(stream, &projections.wkv.norm)?,
            compressor_wkv: DeviceBuffer::from_host(stream, &state.compressor.wkv_bf16)?,
            compressor_wgate: DeviceBuffer::from_host(stream, &state.compressor.wgate_bf16)?,
            compressor_ape: DeviceBuffer::from_host(stream, &state.compressor.ape_f32)?,
            compressor_norm: DeviceBuffer::from_host(stream, &state.compressor.norm_bf16)?,
            indexer,
            attn_sink: DeviceBuffer::from_host(stream, &tail.attn_sink)?,
            wo_a: match wo_a {
                Some(wo_a) => wo_a,
                None => DeviceBuffer::from_host(stream, &tail.wo_a_bf16)?,
            },
            wo_b: DeviceFp8Matrix::new(stream, &tail.wo_b)?,
            hc_ffn_fn: DeviceBuffer::from_host(stream, &tail.hc_ffn_fn)?,
            hc_ffn_base: DeviceBuffer::from_host(stream, &tail.hc_ffn_base)?,
            hc_ffn_scale: DeviceBuffer::from_host(stream, &tail.hc_ffn_scale)?,
            ffn_norm: DeviceBuffer::from_host(stream, &state.ffn_norm)?,
            gate: DeviceBuffer::from_host(stream, &state.gate)?,
            router_bias: DeviceBuffer::from_host(stream, &router_bias)?,
            hash_router,
            shared_w1: DeviceFp8Matrix::new(stream, &tail.shared.w1)?,
            shared_w2: DeviceFp8Matrix::new(stream, &tail.shared.w2)?,
            shared_w3: DeviceFp8Matrix::new(stream, &tail.shared.w3)?,
        })
    }

    fn num_bytes(&self) -> usize {
        self.hc_attn_fn.num_bytes()
            + self.hc_attn_base.num_bytes()
            + self.hc_attn_scale.num_bytes()
            + self.attn_norm.num_bytes()
            + self.wq_a.num_bytes()
            + self.q_norm.num_bytes()
            + self.wq_b.num_bytes()
            + self.wkv.num_bytes()
            + self.kv_norm.num_bytes()
            + self.compressor_wkv.num_bytes()
            + self.compressor_wgate.num_bytes()
            + self.compressor_ape.num_bytes()
            + self.compressor_norm.num_bytes()
            + self.indexer.as_ref().map_or(0, |indexer| {
                indexer.wq_b.num_bytes()
                    + indexer.weights_proj.num_bytes()
                    + indexer.compressor_wkv.num_bytes()
                    + indexer.compressor_wgate.num_bytes()
                    + indexer.compressor_ape.num_bytes()
                    + indexer.compressor_norm.num_bytes()
            })
            + self.attn_sink.num_bytes()
            + self.wo_a.num_bytes()
            + self.wo_b.num_bytes()
            + self.hc_ffn_fn.num_bytes()
            + self.hc_ffn_base.num_bytes()
            + self.hc_ffn_scale.num_bytes()
            + self.ffn_norm.num_bytes()
            + self.gate.num_bytes()
            + self.router_bias.num_bytes()
            + self.shared_w1.num_bytes()
            + self.shared_w2.num_bytes()
            + self.shared_w3.num_bytes()
    }

    fn num_bytes_without_wo_a(&self) -> usize {
        self.num_bytes() - self.wo_a.num_bytes()
    }

    fn shared_num_bytes(&self) -> usize {
        self.shared_w1.num_bytes() + self.shared_w2.num_bytes() + self.shared_w3.num_bytes()
    }

    fn dense_num_bytes(&self) -> usize {
        self.num_bytes() - self.shared_num_bytes()
    }
}

/// MTP reuses the ordinary device weight layout only for the kernels that
/// consume the identical fixed tensors.  Its ratio-0 attention state is
/// `MtpDraftState`; the zero-length compressor fields are unreachable guards,
/// not a second persistent state.
fn upload_mtp_draft_weights(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    host: &v4_state::MtpDraftWeights,
) -> Result<LayerDeviceWeights, Box<dyn Error>> {
    let raw_wo_a = DeviceBuffer::from_host(stream, &host.wo_a.weight)?;
    let mut wo_a = DeviceBuffer::zeroed(stream, raw_wo_a.len())?;
    launch_e4m3fn_to_bf16(module, stream, &raw_wo_a, &mut wo_a)?;
    drop(raw_wo_a);
    Ok(LayerDeviceWeights {
        spec: &v4_spec::PRO,
        layer: v4_state::V4_LAYER_COUNT,
        compression_ratio: v4_state::CompressionRatio::Hca128,
        hc_attn_fn: DeviceBuffer::from_host(stream, &host.hc_attn_fn)?,
        hc_attn_base: DeviceBuffer::from_host(stream, &host.hc_attn_base)?,
        hc_attn_scale: DeviceBuffer::from_host(stream, &host.hc_attn_scale)?,
        attn_norm: DeviceBuffer::from_host(stream, &host.attn_norm)?,
        wq_a: DeviceFp8Matrix::new(stream, &host.wq_a.matrix)?,
        q_norm: DeviceBuffer::from_host(stream, &host.wq_a.norm)?,
        wq_b: DeviceFp8Matrix::new(stream, &host.wq_b)?,
        wkv: DeviceFp8Matrix::new(stream, &host.wkv.matrix)?,
        kv_norm: DeviceBuffer::from_host(stream, &host.wkv.norm)?,
        compressor_wkv: DeviceBuffer::zeroed(stream, 0)?,
        compressor_wgate: DeviceBuffer::zeroed(stream, 0)?,
        compressor_ape: DeviceBuffer::zeroed(stream, 0)?,
        compressor_norm: DeviceBuffer::zeroed(stream, 0)?,
        indexer: None,
        attn_sink: DeviceBuffer::from_host(stream, &host.attn_sink)?,
        wo_a,
        wo_b: DeviceFp8Matrix::new(stream, &host.wo_b)?,
        hc_ffn_fn: DeviceBuffer::from_host(stream, &host.hc_ffn_fn)?,
        hc_ffn_base: DeviceBuffer::from_host(stream, &host.hc_ffn_base)?,
        hc_ffn_scale: DeviceBuffer::from_host(stream, &host.hc_ffn_scale)?,
        ffn_norm: DeviceBuffer::from_host(stream, &host.ffn_norm)?,
        gate: DeviceBuffer::from_host(stream, &host.gate)?,
        router_bias: DeviceBuffer::from_host(stream, &host.router_bias)?,
        hash_router: false,
        shared_w1: DeviceFp8Matrix::new(stream, &host.shared.w1)?,
        shared_w2: DeviceFp8Matrix::new(stream, &host.shared.w2)?,
        shared_w3: DeviceFp8Matrix::new(stream, &host.shared.w3)?,
    })
}

struct MtpDraftPrefixDeviceWeights {
    hc_attn_fn: DeviceBuffer<f32>,
    hc_attn_base: DeviceBuffer<f32>,
    hc_attn_scale: DeviceBuffer<f32>,
    attn_norm: DeviceBuffer<u16>,
    wkv: DeviceFp8Matrix,
    kv_norm: DeviceBuffer<u16>,
}

fn upload_mtp_draft_prefix_weights(
    stream: &CudaStream,
    host: &v4_state::MtpDraftPrefixWeights,
) -> Result<MtpDraftPrefixDeviceWeights, Box<dyn Error>> {
    Ok(MtpDraftPrefixDeviceWeights {
        hc_attn_fn: DeviceBuffer::from_host(stream, &host.hc_attn_fn)?,
        hc_attn_base: DeviceBuffer::from_host(stream, &host.hc_attn_base)?,
        hc_attn_scale: DeviceBuffer::from_host(stream, &host.hc_attn_scale)?,
        attn_norm: DeviceBuffer::from_host(stream, &host.attn_norm)?,
        wkv: DeviceFp8Matrix::new(stream, &host.wkv.matrix)?,
        kv_norm: DeviceBuffer::from_host(stream, &host.wkv.norm)?,
    })
}

struct LoadedLayerRuntime {
    weights: LayerDeviceWeights,
    expert_shard: safetensors::ShardFile,
}

struct LoadedDsparkStage {
    stage: usize,
    weights: LayerDeviceWeights,
    expert_shard: safetensors::ShardFile,
    main_proj: Option<DeviceFp8Matrix>,
    main_norm: Option<DeviceBuffer<u16>>,
    head: Option<v4_state::DsparkHeadWeights>,
}

fn load_flash_dspark_stage(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    model_dir: &std::path::Path,
    stage: usize,
) -> Result<LoadedDsparkStage, Box<dyn Error>> {
    let host = v4_state::load_flash_dspark_stage(model_dir, stage)?;
    let weights =
        LayerDeviceWeights::new_layer(stream, &host.state, &host.projections, &host.tail)?;
    let main_proj = host
        .main_proj
        .as_ref()
        .map(|matrix| DeviceFp8Matrix::new(stream, matrix))
        .transpose()?;
    let main_norm = host
        .main_norm
        .as_ref()
        .map(|norm| DeviceBuffer::from_host(stream, norm))
        .transpose()?;
    let _ = module;
    Ok(LoadedDsparkStage {
        stage,
        weights,
        expert_shard: host.shard,
        main_proj,
        main_norm,
        head: host.head,
    })
}

struct RawLoadedLayerRuntime {
    runtime: LoadedLayerRuntime,
    wo_a_raw: DeviceBuffer<u8>,
    resident_bytes: usize,
}

struct RawDecodedLayerHost {
    host: DecodedLayerHost,
    wo_a_raw: Vec<u8>,
    resident_bytes: usize,
}

struct DecodedLayerHost {
    state: v4_state::LayerStateWeights,
    projections: v4_state::L0Fp8Projections,
    tail: v4_state::L0TailWeights,
}

fn drop_request_hash_ids_before_fixed_retention(
    router: &mut v4_state::LayerRouterWeights,
    request_rows: usize,
) -> Result<usize, Box<dyn Error>> {
    let v4_state::LayerRouterWeights::Hash { ids } = router else {
        return Ok(0);
    };
    let expected = request_rows
        .checked_mul(v4_router::V4_TOPK)
        .ok_or("fixed-retention hash-row count overflow")?;
    if ids.len() != expected {
        return Err(format!(
            "fixed-retention hash IDs have {} values, expected {expected}",
            ids.len()
        )
        .into());
    }
    let removed_bytes = ids
        .len()
        .checked_mul(std::mem::size_of::<i64>())
        .ok_or("fixed-retention hash-ID byte count overflow")?;
    ids.clear();
    Ok(removed_bytes)
}

struct FixedPrefetchedHost {
    layer: usize,
    host: DecodedLayerHost,
    telemetry: FixedLoadTelemetry,
    worker_start_ns: u64,
    worker_end_ns: u64,
    host_bytes: usize,
}

struct FixedPrefetchTask<'scope> {
    layer: usize,
    transition: Option<FixedPrefetchTransition>,
    worker: std::thread::ScopedJoinHandle<'scope, Result<FixedPrefetchedHost, String>>,
}

struct FixedPrefetchTransition {
    from_layer: usize,
    compute_start_ns: u64,
    compute_end_ns: u64,
}

fn fixed_prefetch_transition_json(
    current_layer: usize,
    next_layer: usize,
    telemetry: &FixedLoadTelemetry,
    worker_start_ns: u64,
    worker_end_ns: u64,
    host_bytes: usize,
    compute_start_ns: u64,
    compute_end_ns: u64,
    consume_start_ns: u64,
    window_depth: usize,
) -> Result<serde_json::Value, Box<dyn Error>> {
    if next_layer != current_layer + 1
        || worker_end_ns < worker_start_ns
        || compute_end_ns < compute_start_ns
    {
        return Err("invalid one-layer-ahead fixed prefetch schedule".into());
    }
    let overlap_start_ns = worker_start_ns.max(compute_start_ns);
    let overlap_end_ns = worker_end_ns.min(compute_end_ns);
    Ok(serde_json::json!({
        "from_layer": current_layer,
        "to_layer": next_layer,
        "window_depth": window_depth,
        "worker_start_ns": worker_start_ns,
        "worker_end_ns": worker_end_ns,
        "host_prefetched_bytes": host_bytes,
        "prefetched_read_alloc_ns": telemetry.fixed_read_alloc_ns,
        "prefetched_decode_ns": telemetry.fixed_decode_ns,
        "ready_before_consume": worker_end_ns <= consume_start_ns,
        "consume_wait_ns": worker_end_ns.saturating_sub(consume_start_ns),
        "overlap_ns": overlap_end_ns.saturating_sub(overlap_start_ns),
    }))
}

impl DecodedLayerHost {
    fn source_bytes(&self) -> usize {
        fn matrix(matrix: &v4_state::Fp8Matrix) -> usize {
            matrix.weight.len() + matrix.scale.len()
        }
        fn compressor(weights: &v4_state::LayerCompressorWeights) -> usize {
            (weights.wkv_bf16.len() + weights.wgate_bf16.len() + weights.norm_bf16.len()) * 2
                + weights.ape_f32.len() * 4
        }
        let state = &self.state;
        let state_bytes = (state.hc_fn.len() + state.hc_base.len() + state.hc_scale.len()) * 4
            + state.attn_norm.len() * 2
            + compressor(&state.compressor)
            + state.indexer.as_ref().map_or(0, |indexer| {
                matrix(&indexer.wq_b)
                    + indexer.weights_proj_bf16.len() * 2
                    + compressor(&indexer.compressor)
            })
            + (state.ffn_norm.len() + state.gate.len()) * 2
            + match &state.router {
                v4_state::LayerRouterWeights::Hash { ids } => ids.len() * 8,
                v4_state::LayerRouterWeights::Learned { bias } => bias.len() * 4,
            };
        // The state and tail loaders currently read ffn_norm and gate twice, but
        // source_range_bytes is the unique fixed source-tensor footprint.
        let duplicated_tail_bytes =
            (self.tail.ffn_norm.len() + self.tail.gate.len()) * std::mem::size_of::<u16>();
        state_bytes + self.projections.range_bytes + self.tail.source_range_bytes
            - duplicated_tail_bytes
    }

    fn num_bytes(&self) -> usize {
        fn matrix(matrix: &v4_state::Fp8Matrix) -> usize {
            matrix.weight.len() + matrix.scale.len()
        }
        fn compressor(weights: &v4_state::LayerCompressorWeights) -> usize {
            (weights.wkv_bf16.len() + weights.wgate_bf16.len() + weights.norm_bf16.len()) * 2
                + weights.ape_f32.len() * 4
        }
        let state = &self.state;
        let state_bytes = (state.hc_fn.len() + state.hc_base.len() + state.hc_scale.len()) * 4
            + state.attn_norm.len() * 2
            + compressor(&state.compressor)
            + state.indexer.as_ref().map_or(0, |indexer| {
                matrix(&indexer.wq_b)
                    + indexer.weights_proj_bf16.len() * 2
                    + compressor(&indexer.compressor)
            })
            + (state.ffn_norm.len() + state.gate.len()) * 2
            + match &state.router {
                v4_state::LayerRouterWeights::Hash { ids } => ids.len() * 8,
                v4_state::LayerRouterWeights::Learned { bias } => bias.len() * 4,
            };
        let projections = &self.projections;
        let projection_bytes = matrix(&projections.wq_a.matrix)
            + projections.wq_a.norm.len() * 2
            + matrix(&projections.wq_b)
            + matrix(&projections.wkv.matrix)
            + projections.wkv.norm.len() * 2;
        let tail = &self.tail;
        let tail_bytes = tail.attn_sink.len() * 4
            + tail.wo_a_bf16.len() * 2
            + tail.wo_a_raw.as_ref().map_or(0, Vec::len)
            + matrix(&tail.wo_b)
            + (tail.hc_ffn_fn.len() + tail.hc_ffn_base.len() + tail.hc_ffn_scale.len()) * 4
            + (tail.ffn_norm.len() + tail.gate.len()) * 2
            + matrix(&tail.shared.w1)
            + matrix(&tail.shared.w2)
            + matrix(&tail.shared.w3);
        state_bytes + projection_bytes + tail_bytes
    }
}

#[derive(Clone)]
struct FixedTensorPlan {
    name: String,
    dtype: String,
    shape: Vec<u64>,
    source_bytes: usize,
    decoded_bytes: usize,
}

#[derive(Clone)]
struct FixedLayerPlan {
    layer: usize,
    tensors: Vec<FixedTensorPlan>,
    device_bytes: usize,
    raw_device_bytes: usize,
    host_bytes: usize,
    raw_host_bytes: usize,
    device_saved_ns: u64,
    host_saved_ns: u64,
}

#[derive(Default)]
struct FixedLoadTelemetry {
    fixed_read_alloc_ns: u64,
    fixed_decode_ns: u64,
    fixed_h2d_ns: u64,
    fixed_h2d_bytes: usize,
    fixed_source_bytes: usize,
    raw_read_ns: u64,
    raw_source_bytes: usize,
    raw_h2d_ns: u64,
    raw_h2d_bytes: usize,
    cache_status: &'static str,
    bounded_readahead: Option<BoundedReadaheadTelemetry>,
}

struct BoundedReadaheadTelemetry {
    layer: usize,
    issued: bool,
    skipped_reason: Option<&'static str>,
    bytes: u64,
    ranges: Vec<safetensors::FileRange>,
    advice_us: u64,
}

const STATIONARY_RAW_LAYOUT: &str = "deepseek-v4-pro-layer-stationary-raw-v1";
const DECODE_PROMOTION_PLAN_ENV: &str = "COLIBRI_V4_DECODE_PROMOTION_PLAN";
const RAW_FIXED_SCRATCH_BYTES: usize = 128 * 1024 * 1024;

fn restore_leased_slot<T>(slot: &mut Option<T>, value: T) {
    assert!(
        slot.is_none(),
        "exclusive fixed-layer cache lease observed an ABA replacement"
    );
    *slot = Some(value);
}

struct StationaryRawLease {
    layer: usize,
    device_generation: Option<u64>,
    host_generation: Option<u64>,
    scratch_generation: Option<u64>,
    runtime: Option<LoadedLayerRuntime>,
    raw_device: Option<DeviceBuffer<u8>>,
    raw_host: Option<RawDecodedLayerHost>,
    scratch: Option<DeviceBuffer<u16>>,
    resident_bytes: usize,
    decoded_wo_a_in_scratch: bool,
}

struct PinnedFixedSlot {
    generation: u64,
    host: PinnedHostBuffer<u8>,
    scratch: Option<DeviceBuffer<u16>>,
    h2d_start: cuda_core::CudaEvent,
    h2d_end: cuda_core::CudaEvent,
    ready: cuda_core::CudaEvent,
    runtime_done: cuda_core::CudaEvent,
}

struct PinnedFixedPrepared {
    slot: PinnedFixedSlot,
    generation: u64,
    runtime_done_recorded: bool,
    lease: StationaryRawLease,
    telemetry: FixedLoadTelemetry,
}

impl PinnedFixedSlot {
    fn new(
        context: &std::sync::Arc<CudaContext>,
        scratch: DeviceBuffer<u16>,
    ) -> Result<Self, Box<dyn Error>> {
        let event = || context.new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT));
        Ok(Self {
            generation: 0,
            host: PinnedHostBuffer::zeroed(context, PHASE1_CSA_PAYLOAD_BYTES)?,
            scratch: Some(scratch),
            h2d_start: event()?,
            h2d_end: event()?,
            ready: event()?,
            runtime_done: event()?,
        })
    }
}

impl PinnedFixedPrepared {
    fn wait_ready(&mut self, main: &CudaStream) -> Result<(), Box<dyn Error>> {
        self.slot.ready.synchronize()?;
        main.wait(&self.slot.ready)?;
        self.telemetry.fixed_h2d_ns =
            (self.slot.h2d_start.elapsed_ms(&self.slot.h2d_end)? * 1_000_000.0) as u64;
        self.telemetry.fixed_decode_ns =
            (self.slot.h2d_end.elapsed_ms(&self.slot.ready)? * 1_000_000.0) as u64;
        Ok(())
    }

    fn runtime(&self) -> &LoadedLayerRuntime {
        &self.lease
    }

    fn record_runtime_done(&mut self, main: &CudaStream) -> Result<(), Box<dyn Error>> {
        self.slot.runtime_done.record(main)?;
        self.runtime_done_recorded = true;
        Ok(())
    }

    fn finish(
        mut self,
        cache: &mut FixedLayerCache,
        main: &CudaStream,
        transfer: &CudaStream,
    ) -> Result<PinnedFixedSlot, Box<dyn Error>> {
        if self.generation != self.slot.generation {
            return Err("pinned fixed slot rejected an ABA generation".into());
        }
        if !self.runtime_done_recorded {
            self.slot.runtime_done.record(main)?;
        }
        transfer.wait(&self.slot.runtime_done)?;
        std::mem::swap(
            &mut self
                .lease
                .runtime
                .as_mut()
                .ok_or("pinned fixed prepared lost its runtime")?
                .weights
                .wo_a,
            self.slot
                .scratch
                .as_mut()
                .ok_or("pinned fixed prepared lost its slot scratch")?,
        );
        cache.restore_stationary_raw(self.lease)?;
        Ok(self.slot)
    }
}

type PinnedFixedRequest = (
    Option<PinnedFixedPrepared>,
    Option<(usize, i64, Option<usize>)>,
);
type PinnedFixedReply = Result<Option<PinnedFixedPrepared>, String>;

fn run_pinned_fixed_request_worker(
    mut cache: FixedLayerCache,
    mut slots: Vec<PinnedFixedSlot>,
    root: &std::path::Path,
    module: &kernels::LoadedModule,
    main: &std::sync::Arc<CudaStream>,
    transfer: &std::sync::Arc<CudaStream>,
    rx: std::sync::mpsc::Receiver<PinnedFixedRequest>,
    reply_tx: std::sync::mpsc::SyncSender<PinnedFixedReply>,
    bounded_readahead: bool,
) -> Result<FixedLayerCache, String> {
    let readahead_page_cap = bounded_readahead
        .then(|| bounded_readahead_metadata_page_cap(root))
        .and_then(Result::ok);
    let mut pending_readahead = None;
    while let Ok((completed, next)) = rx.recv() {
        if let Some(completed) = completed {
            let layer = completed.lease.layer;
            match completed.finish(&mut cache, main, transfer) {
                Ok(slot) => slots.push(slot),
                Err(error) => {
                    let message = format!("layer {layer} phase runtime_done_restore: {error}");
                    let _ = reply_tx.send(Err(message.clone()));
                    return Err(message);
                }
            }
        }
        if let Some((layer, token_id, readahead_layer)) = next {
            let slot = match slots.pop() {
                Some(slot) => slot,
                None => {
                    let message = format!(
                        "layer {layer} phase acquire_slot: pinned fixed worker exhausted its two slots"
                    );
                    let _ = reply_tx.send(Err(message.clone()));
                    return Err(message);
                }
            };
            let mut prepared = match cache
                .prepare_stationary_raw_pinned(root, layer, token_id, module, transfer, slot)
            {
                Ok(prepared) => prepared,
                Err(error) => {
                    let message = format!("layer {layer} phase prepare: {error}");
                    let _ = reply_tx.send(Err(message.clone()));
                    return Err(message);
                }
            };
            if pending_readahead
                .as_ref()
                .is_some_and(|telemetry: &BoundedReadaheadTelemetry| telemetry.layer == layer)
            {
                let mut telemetry = pending_readahead
                    .take()
                    .expect("matching bounded readahead telemetry disappeared");
                telemetry.layer = layer;
                prepared.telemetry.bounded_readahead = Some(telemetry);
            }
            reply_tx.send(Ok(Some(prepared))).map_err(|_| {
                format!("layer {layer} phase reply_prepared: pinned fixed response receiver closed")
            })?;
            if bounded_readahead {
                if let Some(readahead_layer) = readahead_layer {
                    pending_readahead = Some(production_bounded_readahead(
                        root,
                        readahead_layer,
                        readahead_page_cap,
                    ));
                }
            }
        } else if slots.len() == 2 {
            if let Err(error) = cache.validate_decode_overlay_complete() {
                let message = format!("request phase overlay_drain: {error}");
                let _ = reply_tx.send(Err(message.clone()));
                return Err(message);
            }
            reply_tx.send(Ok(None)).map_err(|_| {
                "request phase reply_drained: pinned fixed response receiver closed".to_string()
            })?;
        }
    }
    if slots.len() != 2 {
        return Err(format!(
            "request phase shutdown: pinned fixed worker stopped with {} of 2 slots",
            slots.len()
        ));
    }
    let mut retained = slots
        .pop()
        .ok_or("request phase retain_scratch: pinned fixed worker lost its retained scratch")?;
    cache.raw_scratch = retained.scratch.take();
    transfer
        .synchronize()
        .map_err(|error| format!("request phase transfer_sync: {error}"))?;
    Ok(cache)
}

impl std::ops::Deref for StationaryRawLease {
    type Target = LoadedLayerRuntime;

    fn deref(&self) -> &Self::Target {
        self.runtime
            .as_ref()
            .expect("live stationary raw lease must own a runtime")
    }
}

fn decode_device_promotion_residency(
    device_resident_bytes: usize,
    host_resident_bytes: usize,
    promoted_device_bytes: usize,
    released_host_bytes: Option<usize>,
    device_budget_bytes: usize,
) -> Result<(usize, usize), Box<dyn Error>> {
    let device_resident_bytes = device_resident_bytes
        .checked_add(promoted_device_bytes)
        .filter(|&bytes| bytes <= device_budget_bytes)
        .ok_or("decode overlay device promotion violated its target budget")?;
    let host_resident_bytes = match released_host_bytes {
        Some(bytes) => host_resident_bytes
            .checked_sub(bytes)
            .ok_or("decode overlay host resident byte underflow")?,
        None => host_resident_bytes,
    };
    Ok((device_resident_bytes, host_resident_bytes))
}

fn decode_host_promotion_residency(
    host_resident_bytes: usize,
    promoted_host_bytes: usize,
    host_budget_bytes: usize,
) -> Result<usize, Box<dyn Error>> {
    host_resident_bytes
        .checked_add(promoted_host_bytes)
        .filter(|&bytes| bytes <= host_budget_bytes)
        .ok_or_else(|| "decode overlay host promotion violated its target budget".into())
}

fn decode_overlay_uses_decode_target(
    record_count: usize,
    layer_count: usize,
    device_promotion_count: usize,
    host_promotion_count: usize,
) -> Result<bool, Box<dyn Error>> {
    if layer_count == 0 || record_count % layer_count != 0 {
        return Err("decode overlay layer records did not close complete decode rows".into());
    }
    if record_count == 0 && (device_promotion_count != 0 || host_promotion_count != 0) {
        return Err("stationary raw prefill residency unexpectedly contains promotions".into());
    }
    Ok(record_count != 0)
}

fn parse_gib_budget(name: &str) -> Result<usize, Box<dyn Error>> {
    let value = match std::env::var(name) {
        Ok(value) => value.parse::<f64>()?,
        Err(std::env::VarError::NotPresent) => 0.0,
        Err(error) => return Err(error.into()),
    };
    if !value.is_finite() || value < 0.0 {
        return Err(format!("{name} must be a finite non-negative GiB value").into());
    }
    Ok((value * 1024.0 * 1024.0 * 1024.0) as usize)
}

fn fixed_resident_bytes(
    plans: &[FixedLayerPlan],
    device_selected: &[bool],
    host_selected: &[bool],
) -> Result<(usize, usize), Box<dyn Error>> {
    if device_selected.len() != plans.len()
        || host_selected.len() != plans.len()
        || device_selected
            .iter()
            .zip(host_selected)
            .any(|(&device, &host)| device && host)
    {
        return Err("decode promotion plan selected one layer for two tiers".into());
    }
    plans
        .iter()
        .zip(device_selected)
        .zip(host_selected)
        .try_fold(
            (0_usize, 0_usize),
            |(device_bytes, host_bytes), ((plan, &device), &host)| {
                Ok((
                    if device {
                        device_bytes
                            .checked_add(plan.raw_device_bytes)
                            .ok_or("decode promotion device byte overflow")?
                    } else {
                        device_bytes
                    },
                    if host {
                        host_bytes
                            .checked_add(plan.raw_host_bytes)
                            .ok_or("decode promotion host byte overflow")?
                    } else {
                        host_bytes
                    },
                ))
            },
        )
}

fn normalized_decode_promotion_targets(
    plans: &[FixedLayerPlan],
    prefill_device: &[bool],
    prefill_host: &[bool],
    decode_device: &[bool],
    decode_host: &[bool],
    explicit: bool,
) -> Result<serde_json::Value, Box<dyn Error>> {
    if prefill_device.len() != plans.len()
        || prefill_host.len() != plans.len()
        || decode_device.len() != plans.len()
        || decode_host.len() != plans.len()
        || plans
            .iter()
            .map(|plan| plan.layer)
            .collect::<BTreeSet<_>>()
            .len()
            != plans.len()
    {
        return Err("decode promotion targets require unique aligned candidate layers".into());
    }
    let mut device_layers = Vec::new();
    let mut host_layers = Vec::new();
    for index in 0..plans.len() {
        match (
            prefill_device[index],
            prefill_host[index],
            decode_device[index],
            decode_host[index],
        ) {
            (false, false, false, false)
            | (true, false, true, false)
            | (false, true, false, true) => {}
            (false, false, true, false) | (false, true, true, false) => {
                device_layers.push(plans[index].layer);
            }
            (false, false, false, true) => host_layers.push(plans[index].layer),
            _ => return Err("decode promotion targets are not monotone tier promotions".into()),
        }
    }
    device_layers.sort_unstable();
    host_layers.sort_unstable();
    Ok(serde_json::json!({
        "mode": if explicit { "explicit" } else { "default_budgeted" },
        "device_layers": device_layers,
        "host_layers": host_layers,
    }))
}

fn validate_decode_promotion_prefixes(
    plans: &[FixedLayerPlan],
    prefill_device: &[bool],
    prefill_host: &[bool],
    decode_device: &[bool],
    decode_host: &[bool],
    decode_device_budget_bytes: usize,
    decode_host_budget_bytes: usize,
) -> Result<(usize, usize), Box<dyn Error>> {
    normalized_decode_promotion_targets(
        plans,
        prefill_device,
        prefill_host,
        decode_device,
        decode_host,
        true,
    )?;
    let (mut device_bytes, mut host_bytes) =
        fixed_resident_bytes(plans, prefill_device, prefill_host)?;
    if device_bytes > decode_device_budget_bytes || host_bytes > decode_host_budget_bytes {
        return Err(
            "decode promotion initial placement exceeds its decode retention budget".into(),
        );
    }
    for index in 0..plans.len() {
        match (
            prefill_device[index],
            prefill_host[index],
            decode_device[index],
            decode_host[index],
        ) {
            (false, false, true, false) => {
                (device_bytes, host_bytes) = decode_device_promotion_residency(
                    device_bytes,
                    host_bytes,
                    plans[index].raw_device_bytes,
                    None,
                    decode_device_budget_bytes,
                )?;
            }
            (false, true, true, false) => {
                (device_bytes, host_bytes) = decode_device_promotion_residency(
                    device_bytes,
                    host_bytes,
                    plans[index].raw_device_bytes,
                    Some(plans[index].raw_host_bytes),
                    decode_device_budget_bytes,
                )?;
            }
            (false, false, false, true) => {
                host_bytes = decode_host_promotion_residency(
                    host_bytes,
                    plans[index].raw_host_bytes,
                    decode_host_budget_bytes,
                )?;
            }
            _ => {}
        }
        if device_bytes > decode_device_budget_bytes || host_bytes > decode_host_budget_bytes {
            return Err("decode promotion prefix exceeds its decode retention budget".into());
        }
    }
    if (device_bytes, host_bytes) != fixed_resident_bytes(plans, decode_device, decode_host)? {
        return Err("decode promotion prefix accounting did not reach its target".into());
    }
    Ok((device_bytes, host_bytes))
}

fn parse_decode_promotion_plan(
    value: Option<&std::ffi::OsStr>,
    plans: &[FixedLayerPlan],
    device_selected: &[bool],
    host_selected: &[bool],
    device_budget_bytes: usize,
    host_budget_bytes: usize,
    decode_device_budget_bytes: usize,
    decode_host_budget_bytes: usize,
) -> Result<Option<(Vec<bool>, Vec<bool>)>, Box<dyn Error>> {
    let Some(value) = value else {
        return Ok(None);
    };
    let value = value
        .to_str()
        .ok_or_else(|| format!("{DECODE_PROMOTION_PLAN_ENV} must be valid UTF-8"))?;
    if value.is_empty()
        || plans.len() != device_selected.len()
        || plans.len() != host_selected.len()
        || plans
            .iter()
            .map(|plan| plan.layer)
            .collect::<BTreeSet<_>>()
            .len()
            != plans.len()
    {
        return Err(format!(
            "{DECODE_PROMOTION_PLAN_ENV} requires a non-empty plan over unique candidate layers"
        )
        .into());
    }
    let (prefill_device_bytes, prefill_host_bytes) =
        fixed_resident_bytes(plans, device_selected, host_selected)?;
    if prefill_device_bytes > device_budget_bytes || prefill_host_bytes > host_budget_bytes {
        return Err("decode promotion prefill placement exceeds its retention budget".into());
    }

    let mut decode_device_selected = device_selected.to_vec();
    let mut decode_host_selected = host_selected.to_vec();
    let mut promoted = BTreeSet::new();
    for promotion in value.split(',') {
        let Some((layer, target)) = promotion.split_once(':') else {
            return Err(format!(
                "{DECODE_PROMOTION_PLAN_ENV} must be comma-separated layer:D or layer:H promotions"
            )
            .into());
        };
        if layer.is_empty() || !layer.bytes().all(|byte| byte.is_ascii_digit()) {
            return Err(format!(
                "{DECODE_PROMOTION_PLAN_ENV} layers must be base-10 unsigned integers"
            )
            .into());
        }
        let layer = layer
            .parse::<usize>()
            .map_err(|_| format!("{DECODE_PROMOTION_PLAN_ENV} layer is outside usize"))?;
        if !promoted.insert(layer) {
            return Err(
                format!("{DECODE_PROMOTION_PLAN_ENV} contains duplicate layer {layer}").into(),
            );
        }
        let index = plans
            .iter()
            .position(|plan| plan.layer == layer)
            .ok_or_else(|| {
                format!("{DECODE_PROMOTION_PLAN_ENV} layer {layer} is not a retention candidate")
            })?;
        match (device_selected[index], host_selected[index], target) {
            (false, false, "D") => decode_device_selected[index] = true,
            (false, false, "H") => decode_host_selected[index] = true,
            (false, true, "D") => {
                decode_device_selected[index] = true;
                decode_host_selected[index] = false;
            }
            _ => {
                return Err(format!(
                    "{DECODE_PROMOTION_PLAN_ENV} layer {layer}:{target} is not S->H, S->D, or H->D"
                )
                .into());
            }
        }
    }
    validate_decode_promotion_prefixes(
        plans,
        device_selected,
        host_selected,
        &decode_device_selected,
        &decode_host_selected,
        decode_device_budget_bytes,
        decode_host_budget_bytes,
    )?;
    Ok(Some((decode_device_selected, decode_host_selected)))
}

fn fixed_saved_bytes_for_target(
    baseline_source_bytes: usize,
    baseline_h2d_bytes: usize,
    target_device: bool,
    target_host: bool,
) -> Result<(usize, usize), Box<dyn Error>> {
    match (target_device, target_host) {
        (true, false) => Ok((baseline_source_bytes, baseline_h2d_bytes)),
        (false, true) => Ok((baseline_source_bytes, 0)),
        (false, false) => Ok((0, 0)),
        (true, true) => Err("fixed saved-byte target selected two tiers".into()),
    }
}

fn select_fixed_cache_layers(
    plans: &[FixedLayerPlan],
    budget: usize,
    already_selected: &[bool],
    device: bool,
    raw_device: bool,
) -> Vec<bool> {
    let mut order = (0..plans.len())
        .filter(|&index| !already_selected[index])
        .collect::<Vec<_>>();
    order.sort_by(|&left, &right| {
        let left_saved = if device {
            plans[left].device_saved_ns
        } else {
            plans[left].host_saved_ns
        };
        let right_saved = if device {
            plans[right].device_saved_ns
        } else {
            plans[right].host_saved_ns
        };
        let left_bytes = if device {
            if raw_device {
                plans[left].raw_device_bytes
            } else {
                plans[left].device_bytes
            }
        } else {
            if raw_device {
                plans[left].raw_host_bytes
            } else {
                plans[left].host_bytes
            }
        };
        let right_bytes = if device {
            if raw_device {
                plans[right].raw_device_bytes
            } else {
                plans[right].device_bytes
            }
        } else {
            if raw_device {
                plans[right].raw_host_bytes
            } else {
                plans[right].host_bytes
            }
        };
        (right_saved as u128 * left_bytes as u128)
            .cmp(&(left_saved as u128 * right_bytes as u128))
            .then_with(|| plans[left].layer.cmp(&plans[right].layer))
    });
    let mut selected = vec![false; plans.len()];
    let mut used = 0_usize;
    for index in order {
        let bytes = if device {
            if raw_device {
                plans[index].raw_device_bytes
            } else {
                plans[index].device_bytes
            }
        } else {
            if raw_device {
                plans[index].raw_host_bytes
            } else {
                plans[index].host_bytes
            }
        };
        if used.checked_add(bytes).is_some_and(|next| next <= budget) {
            selected[index] = true;
            used += bytes;
        }
    }
    selected
}

fn stationary_raw_prefill_selection(
    plans: &[FixedLayerPlan],
    device_budget_bytes: usize,
    host_budget_bytes: usize,
) -> (Vec<bool>, Vec<bool>) {
    let none_selected = vec![false; plans.len()];
    let device_selected =
        select_fixed_cache_layers(plans, device_budget_bytes, &none_selected, true, true);
    let host_selected =
        select_fixed_cache_layers(plans, host_budget_bytes, &device_selected, false, true);
    (device_selected, host_selected)
}

fn preflight_cache_budget(
    requested: usize,
    available: usize,
    headroom: usize,
    name: &str,
) -> usize {
    let effective = requested.min(available.saturating_sub(headroom));
    if effective < requested {
        eprintln!(
            "CACHE_PREFLIGHT_FALLBACK {name} requested_bytes={requested} effective_bytes={effective} available_bytes={available} headroom_bytes={headroom}"
        );
    }
    effective
}

fn host_available_bytes() -> Result<usize, Box<dyn Error>> {
    let meminfo = std::fs::read_to_string("/proc/meminfo")?;
    let kib = meminfo
        .lines()
        .find_map(|line| line.strip_prefix("MemAvailable:"))
        .and_then(|value| value.split_whitespace().next())
        .ok_or("MemAvailable is missing from /proc/meminfo")?
        .parse::<usize>()?;
    kib.checked_mul(1024)
        .ok_or_else(|| "MemAvailable byte count overflow".into())
}

fn process_rss_bytes() -> Result<usize, Box<dyn Error>> {
    let status = std::fs::read_to_string("/proc/self/status")?;
    let kib = status
        .lines()
        .find_map(|line| line.strip_prefix("VmRSS:"))
        .and_then(|value| value.split_whitespace().next())
        .ok_or("VmRSS is missing from /proc/self/status")?
        .parse::<usize>()?;
    kib.checked_mul(1024)
        .ok_or_else(|| "VmRSS byte count overflow".into())
}

fn device_available_bytes() -> Result<usize, Box<dyn Error>> {
    let (mut free, mut total) = (0_usize, 0_usize);
    let result = unsafe { sys::cuMemGetInfo_v2(&mut free, &mut total) };
    if result != 0 {
        return Err(format!("cuMemGetInfo_v2 failed with {result:?}").into());
    }
    Ok(free)
}

const PHASE1_HCA_PAYLOAD_BYTES: usize = 391_959_576;
const PHASE1_CSA_PAYLOAD_BYTES: usize = 427_239_448;
const PINNED_FIXED_PIPELINE_WORKER_SLOTS: usize = 2;
// A prepared worker slot owns both its reusable wo_a decode scratch and the
// decoded fixed runtime held by PinnedFixedPrepared::lease until finish()
// restores it.  Two prepare replies may be outstanding simultaneously.
const PINNED_FIXED_PIPELINE_WORKER_PREPARED_DEVICE_BYTES: usize =
    PINNED_FIXED_PIPELINE_WORKER_SLOTS * PHASE1_CSA_PAYLOAD_BYTES;
const PINNED_FIXED_PIPELINE_WORKER_SCRATCH_DEVICE_BYTES: usize =
    PINNED_FIXED_PIPELINE_WORKER_SLOTS * RAW_FIXED_SCRATCH_BYTES;
const PINNED_FIXED_PIPELINE_WORKER_DEVICE_BYTES: usize =
    PINNED_FIXED_PIPELINE_WORKER_PREPARED_DEVICE_BYTES
        + PINNED_FIXED_PIPELINE_WORKER_SCRATCH_DEVICE_BYTES;
const PINNED_FIXED_PIPELINE_WORKER_PINNED_HOST_BYTES: usize =
    PINNED_FIXED_PIPELINE_WORKER_SLOTS * PHASE1_CSA_PAYLOAD_BYTES;
const PHASE1_P6_HCA_TRANSFERS: u64 = 23;
const PHASE1_P6_CSA_TRANSFERS: u64 = 29;
const PHASE1_P6_FIXED_PREPARE_NS: u64 = 11_466_446_202;
const PHASE1_P6_WITH_FINAL_NS: u64 = 17_162_779_092;

#[derive(Clone, Copy)]
struct ProcessFaults {
    minor: u64,
    major: u64,
}

fn process_faults() -> Result<ProcessFaults, Box<dyn Error>> {
    let stat = std::fs::read_to_string("/proc/self/stat")?;
    let fields = stat
        .rsplit_once(')')
        .ok_or("/proc/self/stat has no closing command delimiter")?
        .1
        .split_whitespace()
        .collect::<Vec<_>>();
    Ok(ProcessFaults {
        minor: fields
            .get(7)
            .ok_or("/proc/self/stat is missing minflt")?
            .parse()?,
        major: fields
            .get(9)
            .ok_or("/proc/self/stat is missing majflt")?
            .parse()?,
    })
}

fn fault_delta(before: ProcessFaults, after: ProcessFaults) -> serde_json::Value {
    serde_json::json!({
        "minor": after.minor.saturating_sub(before.minor),
        "major": after.major.saturating_sub(before.major),
    })
}

fn phase1_predicted_p6_ns(
    hca_pack_ns: u64,
    hca_h2d_ns: u64,
    csa_pack_ns: u64,
    csa_h2d_ns: u64,
) -> u64 {
    let flow_shop_ns = PHASE1_P6_HCA_TRANSFERS
        .saturating_mul(hca_pack_ns.max(hca_h2d_ns))
        .saturating_add(PHASE1_P6_CSA_TRANSFERS.saturating_mul(csa_pack_ns.max(csa_h2d_ns)));
    PHASE1_P6_WITH_FINAL_NS
        .saturating_sub(PHASE1_P6_FIXED_PREPARE_NS)
        .saturating_add(flow_shop_ns)
}

fn pinned_h2d_service_gate(context: &std::sync::Arc<CudaContext>) -> Result<(), Box<dyn Error>> {
    let pinned_capacity_bytes = PHASE1_CSA_PAYLOAD_BYTES
        .checked_mul(2)
        .ok_or("Phase1 pinned capacity overflow")?;
    let host_required_bytes = FIXED_PREFETCH_HOST_HEADROOM_BYTES
        .checked_add(pinned_capacity_bytes)
        .and_then(|bytes| bytes.checked_add(2 * PHASE1_CSA_PAYLOAD_BYTES))
        .ok_or("Phase1 host requirement overflow")?;
    let host_available = host_available_bytes()?;
    let device_available = device_available_bytes()?;
    let device_required = V4_DEVICE_CAPACITY_HEADROOM_BYTES
        .checked_add(PHASE1_CSA_PAYLOAD_BYTES)
        .ok_or("Phase1 device requirement overflow")?;
    if host_available < host_required_bytes || device_available < device_required {
        println!(
            "{}",
            serde_json::json!({
                "status": "KILL",
                "mode": "v4-pinned-h2d-service-gate-phase1",
                "reason": "WSL preflight/headroom",
                "host_available_bytes": host_available,
                "host_required_bytes": host_required_bytes,
                "device_available_bytes": device_available,
                "device_required_bytes": device_required,
            })
        );
        return Ok(());
    }

    let mut hca_slot = PinnedHostBuffer::<u8>::zeroed(context, PHASE1_CSA_PAYLOAD_BYTES)?;
    let mut csa_slot = PinnedHostBuffer::<u8>::zeroed(context, PHASE1_CSA_PAYLOAD_BYTES)?;
    let transfer_stream = context.new_stream()?;
    let mut measurements = Vec::with_capacity(2);

    for (name, bytes, pinned) in [
        ("hca", PHASE1_HCA_PAYLOAD_BYTES, &mut hca_slot),
        ("csa", PHASE1_CSA_PAYLOAD_BYTES, &mut csa_slot),
    ] {
        let mut payload = vec![0_u8; bytes];
        for (index, byte) in payload.iter_mut().enumerate() {
            *byte = (index as u64)
                .wrapping_mul(131)
                .wrapping_add(bytes as u64)
                .to_le_bytes()[0];
        }
        let expected_hash = fnv1a_bytes(&payload);

        let pageable_faults_before = process_faults()?;
        let pageable_start = context.new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
        let pageable_end = context.new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
        pageable_start.record(&transfer_stream)?;
        let pageable_wall_started = std::time::Instant::now();
        let pageable_device = DeviceBuffer::from_host(&transfer_stream, &payload)?;
        let pageable_submission_ns =
            u64::try_from(pageable_wall_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        pageable_end.record(&transfer_stream)?;
        pageable_end.synchronize()?;
        let pageable_wall_ns =
            u64::try_from(pageable_wall_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        let pageable_event_ns =
            (pageable_start.elapsed_ms(&pageable_end)? as f64 * 1_000_000.0) as u64;
        let pageable_faults_after = process_faults()?;
        let pageable_download = pageable_device.to_host_vec(&transfer_stream)?;
        let pageable_hash = fnv1a_bytes(&pageable_download);
        drop(pageable_download);
        drop(pageable_device);

        let pack_faults_before = process_faults()?;
        let pack_started = std::time::Instant::now();
        pinned.as_mut_slice()[..bytes].copy_from_slice(&payload);
        let pack_ns = u64::try_from(pack_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        let pack_faults_after = process_faults()?;

        let pinned_faults_before = process_faults()?;
        let pinned_start = context.new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
        let pinned_end = context.new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
        let pinned_wall_started = std::time::Instant::now();
        let pinned_device =
            unsafe { DeviceBuffer::<u8>::uninitialized_async(&transfer_stream, bytes)? };
        pinned_start.record(&transfer_stream)?;
        let copy_result = unsafe {
            sys::cuMemcpyHtoDAsync_v2(
                pinned_device.cu_deviceptr(),
                pinned.as_ptr().cast(),
                bytes,
                transfer_stream.cu_stream(),
            )
        };
        if copy_result != 0 {
            return Err(format!("Phase1 pinned H2D enqueue failed with {copy_result:?}").into());
        }
        pinned_end.record(&transfer_stream)?;
        let pinned_submission_ns =
            u64::try_from(pinned_wall_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        pinned_end.synchronize()?;
        let pinned_wall_ns =
            u64::try_from(pinned_wall_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        let pinned_event_ns = (pinned_start.elapsed_ms(&pinned_end)? as f64 * 1_000_000.0) as u64;
        let pinned_faults_after = process_faults()?;
        let pinned_download = pinned_device.to_host_vec(&transfer_stream)?;
        let pinned_hash = fnv1a_bytes(&pinned_download);
        drop(pinned_download);
        drop(pinned_device);
        transfer_stream.synchronize()?;

        measurements.push(serde_json::json!({
            "payload": name,
            "bytes": bytes,
            "expected_hash": format!("{expected_hash:016x}"),
            "pageable": {
                "allocation_success": true,
                "cuda_event_service_ns": pageable_event_ns,
                "submission_ns": pageable_submission_ns,
                "wall_ns": pageable_wall_ns,
                "faults": fault_delta(pageable_faults_before, pageable_faults_after),
                "d2h_hash": format!("{pageable_hash:016x}"),
                "hash_equal": pageable_hash == expected_hash,
            },
            "pinned": {
                "allocation_success": true,
                "pack_wall_ns": pack_ns,
                "pack_faults": fault_delta(pack_faults_before, pack_faults_after),
                "cuda_event_h2d_service_ns": pinned_event_ns,
                "submission_ns": pinned_submission_ns,
                "wall_ns": pinned_wall_ns,
                "transfer_faults": fault_delta(pinned_faults_before, pinned_faults_after),
                "d2h_hash": format!("{pinned_hash:016x}"),
                "hash_equal": pinned_hash == expected_hash,
            },
        }));
    }

    let value = |payload: usize, arm: &str, field: &str| -> Result<u64, Box<dyn Error>> {
        measurements[payload][arm][field]
            .as_u64()
            .ok_or_else(|| format!("Phase1 measurement {payload}.{arm}.{field} is missing").into())
    };
    let predicted_p6_ns = phase1_predicted_p6_ns(
        value(0, "pinned", "pack_wall_ns")?,
        value(0, "pinned", "cuda_event_h2d_service_ns")?,
        value(1, "pinned", "pack_wall_ns")?,
        value(1, "pinned", "cuda_event_h2d_service_ns")?,
    );
    let hashes_pass = measurements.iter().all(|measurement| {
        measurement["pageable"]["hash_equal"] == true && measurement["pinned"]["hash_equal"] == true
    });
    let threshold_ns = PHASE1_P6_WITH_FINAL_NS.saturating_mul(80) / 100;
    let status = if hashes_pass && predicted_p6_ns <= threshold_ns {
        "PASS"
    } else {
        "KILL"
    };
    println!(
        "{}",
        serde_json::json!({
            "status": status,
            "mode": "v4-pinned-h2d-service-gate-phase1",
            "payload_measurements": measurements,
            "allocation": {
                "pinned_slot_count": 2,
                "pinned_capacity_bytes": pinned_capacity_bytes,
                "host_available_bytes": host_available,
                "host_required_bytes": host_required_bytes,
                "device_available_bytes": device_available,
                "device_required_bytes": device_required,
            },
            "flow_shop": {
                "model": "p6_nonfixed + 23*max(hca_pack,hca_h2d) + 29*max(csa_pack,csa_h2d)",
                "p6_with_final_baseline_ns": PHASE1_P6_WITH_FINAL_NS,
                "p6_fixed_prepare_baseline_ns": PHASE1_P6_FIXED_PREPARE_NS,
                "predicted_p6_with_final_ns": predicted_p6_ns,
                "required_max_ns": threshold_ns,
                "required_reduction_percent": 20,
            },
            "hashes_pass": hashes_pass,
            "phase2_authorized": false,
        })
    );
    Ok(())
}

fn fixed_layer_tensor_plan(
    model_dir: &std::path::Path,
    layer: usize,
) -> Result<(Vec<FixedTensorPlan>, usize, usize), Box<dyn Error>> {
    let shard = safetensors::ShardFile::open(
        model_dir.join(format!("model-{:05}-of-00064.safetensors", layer + 2)),
    )?;
    let prefix = format!("layers.{layer}.");
    let mut tensors = shard
        .tensors_with_prefix(&prefix)
        .into_iter()
        .filter(|(name, _)| {
            !name.contains(".ffn.experts.")
                && !name.ends_with(".ffn.gate.tid2eid")
                && !name.ends_with(".attn.wo_a.scale")
        })
        .map(|(name, info)| {
            let source_bytes = usize::try_from(info.byte_len)?;
            let decoded_wo_a = name.ends_with(".attn.wo_a.weight");
            Ok::<_, Box<dyn Error>>(FixedTensorPlan {
                name,
                dtype: if decoded_wo_a {
                    format!("{:?}->Bf16", info.dtype)
                } else {
                    format!("{:?}", info.dtype)
                },
                shape: info.shape,
                source_bytes,
                decoded_bytes: if decoded_wo_a {
                    source_bytes.checked_mul(2).ok_or("decoded wo_a overflow")?
                } else {
                    source_bytes
                },
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    if layer < 3 {
        tensors.push(FixedTensorPlan {
            name: format!("layers.{layer}.ffn.gate.synthetic_zero_bias"),
            dtype: "F32".into(),
            shape: vec![u64::try_from(v4_router::V4_ROUTED_EXPERTS)?],
            source_bytes: 0,
            decoded_bytes: v4_router::V4_ROUTED_EXPERTS * 4,
        });
    }
    let device_bytes = tensors.iter().try_fold(0_usize, |sum, tensor| {
        sum.checked_add(tensor.decoded_bytes)
            .ok_or("fixed tensor byte sum overflow")
    })?;
    let wo_a_source_bytes = tensors
        .iter()
        .find(|tensor| tensor.name.ends_with(".attn.wo_a.weight"))
        .ok_or("fixed tensor plan is missing wo_a.weight")?
        .source_bytes;
    let wo_a_scale = shard.tensor(&format!("layers.{layer}.attn.wo_a.scale"))?;
    if wo_a_scale.dtype != safetensors::TensorDtype::F8E8M0 || wo_a_scale.shape != [128, 32] {
        return Err(format!("layer {layer} wo_a.scale metadata contract mismatch").into());
    }
    Ok((tensors, device_bytes, wo_a_source_bytes))
}

const BOUNDED_READAHEAD_MIN_AVAILABLE_BYTES: usize = 4 * 1024 * 1024 * 1024;

struct BoundedReadaheadSourceSpan {
    range: safetensors::FileRange,
}

struct BoundedReadaheadLayer {
    shard: safetensors::ShardFile,
    spans: Vec<BoundedReadaheadSourceSpan>,
    source_bytes: u64,
}

fn fixed_layer_bounded_readahead_source(
    model_dir: &std::path::Path,
    layer: usize,
) -> Result<BoundedReadaheadLayer, Box<dyn Error>> {
    let (fixed, _, _) = fixed_layer_tensor_plan(model_dir, layer)?;
    let shard = safetensors::ShardFile::open(
        model_dir.join(format!("model-{:05}-of-00064.safetensors", layer + 2)),
    )?;
    let mut spans = fixed
        .into_iter()
        .filter(|tensor| tensor.source_bytes != 0)
        .map(|tensor| {
            let len = u64::try_from(tensor.source_bytes)?;
            Ok::<_, Box<dyn Error>>(BoundedReadaheadSourceSpan {
                range: shard.tensor_file_range(&tensor.name, 0, len)?,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let prefix = format!("layers.{layer}");
    // fixed_layer_tensor_plan deliberately omits these because they are not
    // retained device payloads. They are nevertheless real loader reads.
    let wo_a_scale = format!("{prefix}.attn.wo_a.scale");
    let wo_a_scale_len = shard.tensor(&wo_a_scale)?.byte_len;
    spans.push(BoundedReadaheadSourceSpan {
        range: shard.tensor_file_range(&wo_a_scale, 0, wo_a_scale_len)?,
    });
    if layer < 3 {
        let tid2eid = format!("{prefix}.ffn.gate.tid2eid");
        let len = u64::try_from(v4_router::V4_TOPK * std::mem::size_of::<i64>())?;
        spans.push(BoundedReadaheadSourceSpan {
            range: shard.tensor_file_range(&tid2eid, 0, len)?,
        });
    }
    let source_bytes = spans.iter().try_fold(0_u64, |total, span| {
        total
            .checked_add(span.range.len)
            .ok_or("bounded readahead source byte count overflow")
    })?;
    Ok(BoundedReadaheadLayer {
        shard,
        spans,
        source_bytes,
    })
}

fn bounded_readahead_window_bytes(
    ranges: &[safetensors::FileRange],
) -> Result<u64, Box<dyn Error>> {
    ranges.iter().try_fold(0_u64, |total, range| {
        total
            .checked_add(range.len)
            .ok_or_else(|| "bounded readahead issued byte count overflow".into())
    })
}

fn bounded_readahead_page_cap(source_bytes: u64, page_bytes: u64) -> Result<u64, Box<dyn Error>> {
    if page_bytes == 0 {
        return Err("bounded readahead OS page size is zero".into());
    }
    source_bytes
        .checked_add(page_bytes - 1)
        .map(|bytes| bytes / page_bytes * page_bytes)
        .ok_or_else(|| "bounded readahead page-aligned cap overflow".into())
}

fn os_page_bytes() -> Result<u64, Box<dyn Error>> {
    unsafe extern "C" {
        fn getpagesize() -> i32;
    }
    // SAFETY: getpagesize has no arguments or side effects.
    u64::try_from(unsafe { getpagesize() }).map_err(Into::into)
}

fn bounded_readahead_metadata_page_cap(model_dir: &std::path::Path) -> Result<u64, Box<dyn Error>> {
    let max_source_bytes = (0..v4_state::V4_LAYER_COUNT)
        .map(|layer| {
            fixed_layer_bounded_readahead_source(model_dir, layer).map(|plan| plan.source_bytes)
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .max()
        .ok_or("bounded readahead metadata plan is empty")?;
    bounded_readahead_page_cap(max_source_bytes, os_page_bytes()?)
}

fn production_bounded_readahead(
    model_dir: &std::path::Path,
    layer: usize,
    page_cap: Option<u64>,
) -> BoundedReadaheadTelemetry {
    let skipped = |reason| BoundedReadaheadTelemetry {
        layer,
        issued: false,
        skipped_reason: Some(reason),
        bytes: 0,
        ranges: Vec::new(),
        advice_us: 0,
    };
    if !cfg!(target_os = "linux") {
        return skipped("not_linux");
    }
    let Some(page_cap) = page_cap else {
        return skipped("metadata_cap_unavailable");
    };
    match host_available_bytes() {
        Ok(available) if available >= BOUNDED_READAHEAD_MIN_AVAILABLE_BYTES => {}
        Ok(_) => return skipped("mem_available_below_4gib"),
        Err(_) => return skipped("mem_available_unavailable"),
    }
    let plan = match fixed_layer_bounded_readahead_source(model_dir, layer) {
        Ok(plan) => plan,
        Err(_) => return skipped("invalid_span"),
    };
    let ranges = plan.spans.iter().map(|span| span.range).collect::<Vec<_>>();
    let ranges = match safetensors::merge_adjacent_ranges(&ranges) {
        Ok(ranges) => ranges,
        Err(_) => return skipped("invalid_span"),
    };
    let bytes = match bounded_readahead_window_bytes(&ranges) {
        Ok(bytes) if plan.source_bytes <= page_cap && bytes <= page_cap => bytes,
        Ok(_) => return skipped("metadata_page_cap"),
        Err(_) => return skipped("invalid_span"),
    };
    let started = std::time::Instant::now();
    if plan.shard.advise_willneed_ranges(&ranges).is_err() {
        return BoundedReadaheadTelemetry {
            layer,
            issued: false,
            skipped_reason: Some("syscall_error"),
            bytes,
            ranges,
            advice_us: u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX),
        };
    }
    BoundedReadaheadTelemetry {
        layer,
        issued: true,
        skipped_reason: None,
        bytes,
        ranges,
        advice_us: u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX),
    }
}

fn bounded_readahead_json(
    telemetry: Option<&BoundedReadaheadTelemetry>,
    following_prepare_read_service_ns: u64,
) -> Option<serde_json::Value> {
    telemetry.map(|telemetry| {
        serde_json::json!({
            "issued": telemetry.issued,
            "skipped_reason": telemetry.skipped_reason,
            "bytes": telemetry.bytes,
            "ranges": telemetry.ranges.iter().map(|range| serde_json::json!({"offset": range.offset, "len": range.len})).collect::<Vec<_>>(),
            "advice_us": telemetry.advice_us,
            "following_prepare_read_service_ns": following_prepare_read_service_ns,
        })
    })
}

fn max_fixed_layer_device_bytes(
    model_dir: &std::path::Path,
) -> Result<(usize, usize), Box<dyn Error>> {
    (0..v4_state::V4_LAYER_COUNT)
        .map(|layer| {
            let (_, device_bytes, _) = fixed_layer_tensor_plan(model_dir, layer)?;
            Ok((layer, device_bytes))
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?
        .into_iter()
        .max_by_key(|&(_, bytes)| bytes)
        .ok_or_else(|| "fixed layer metadata plan is empty".into())
}

struct DeviceAttentionCapacityPlan {
    qd4_device_bytes: usize,
    qd4_immutable_expert_cache_device_bytes: usize,
    fixed_layer: usize,
    max_fixed_device_bytes: usize,
    max_device_token_state_bytes: usize,
    hca_device_token_state_bytes: usize,
    device_token_state_count: usize,
    device_token_state_total_bytes: usize,
    rope_lut_bytes: usize,
    tid2eid_transient_max_bytes: usize,
    certified_argmax_device_temp_bytes: usize,
    certified_argmax_host_temp_bytes: usize,
    headroom_bytes: usize,
    required_bytes: usize,
}

fn device_attention_capacity_plan(
    model_dir: &std::path::Path,
    qd4_enabled: bool,
    qd4_immutable_expert_cache: bool,
    request_rope_rows: usize,
    certified_argmax_production: bool,
) -> Result<DeviceAttentionCapacityPlan, Box<dyn Error>> {
    let (fixed_layer, max_fixed_device_bytes) = max_fixed_layer_device_bytes(model_dir)?;
    let qd4_device_bytes = usize::from(qd4_enabled) * V4_QD4_DEVICE_BYTES;
    let qd4_immutable_expert_cache_device_bytes =
        usize::from(qd4_immutable_expert_cache) * V4_QD4_IMMUTABLE_EXPERT_CACHE_DEVICE_BYTES;
    if !(V4_REQUEST_ROPE_LUT_ROWS..=V4_SHORT_DEVICE_DECODE_MAX_ROWS).contains(&request_rope_rows) {
        return Err("device-attention capacity rows are outside the short-decode ceiling".into());
    }
    let short_device_decode = request_rope_rows > V4_REQUEST_ROPE_LUT_ROWS;
    let rope_lut_bytes = 2 * request_rope_rows * v4_state::Q_ROPE_DIM * std::mem::size_of::<f32>();
    let tid2eid_transient_max_bytes = V4_TID2EID_TRANSIENT_MAX_BYTES;
    let certified_argmax_device_temp_bytes =
        usize::from(certified_argmax_production) * CERTIFIED_ARGMAX_DEVICE_TEMP_BYTES;
    let certified_argmax_host_temp_bytes =
        usize::from(certified_argmax_production) * CERTIFIED_ARGMAX_HOST_TEMP_BYTES;
    let manifests = v4_state::serial_layer_manifests(0)?;
    let (hca_states, csa_states) =
        manifests
            .iter()
            .fold((0_usize, 0_usize), |(hca, csa), manifest| {
                match manifest.compression_ratio {
                    v4_state::CompressionRatio::WindowOnly => (hca, csa),
                    v4_state::CompressionRatio::Hca128 => (hca + 1, csa),
                    v4_state::CompressionRatio::Csa4 => (hca, csa + 1),
                }
            });
    let device_token_state_count = if short_device_decode {
        manifests.len()
    } else {
        2
    };
    let device_token_state_total_bytes = if short_device_decode {
        hca_states
            .checked_mul(V4_DEVICE_TOKEN_STATE_HCA_BYTES)
            .and_then(|bytes| {
                csa_states
                    .checked_mul(V4_DEVICE_TOKEN_STATE_MAX_BYTES)
                    .and_then(|csa_bytes| bytes.checked_add(csa_bytes))
            })
            .ok_or("device attention persistent-state byte overflow")?
    } else {
        V4_DEVICE_TOKEN_STATE_MAX_BYTES + V4_DEVICE_TOKEN_STATE_HCA_BYTES
    };
    let required_bytes = [
        qd4_device_bytes,
        qd4_immutable_expert_cache_device_bytes,
        max_fixed_device_bytes,
        device_token_state_total_bytes,
        rope_lut_bytes,
        tid2eid_transient_max_bytes,
        certified_argmax_device_temp_bytes,
        V4_DEVICE_CAPACITY_HEADROOM_BYTES,
    ]
    .into_iter()
    .try_fold(0_usize, |total, bytes| {
        total
            .checked_add(bytes)
            .ok_or("device-attention capacity ledger overflow")
    })?;
    Ok(DeviceAttentionCapacityPlan {
        qd4_device_bytes,
        qd4_immutable_expert_cache_device_bytes,
        fixed_layer,
        max_fixed_device_bytes,
        max_device_token_state_bytes: V4_DEVICE_TOKEN_STATE_MAX_BYTES,
        hca_device_token_state_bytes: V4_DEVICE_TOKEN_STATE_HCA_BYTES,
        device_token_state_count,
        device_token_state_total_bytes,
        rope_lut_bytes,
        tid2eid_transient_max_bytes,
        certified_argmax_device_temp_bytes,
        certified_argmax_host_temp_bytes,
        headroom_bytes: V4_DEVICE_CAPACITY_HEADROOM_BYTES,
        required_bytes,
    })
}

fn validate_device_attention_capacity(
    model_dir: &std::path::Path,
    qd4_enabled: bool,
    qd4_immutable_expert_cache: bool,
    request_rope_rows: usize,
    certified_argmax_production: bool,
    available_bytes: usize,
) -> Result<DeviceAttentionCapacityPlan, Box<dyn Error>> {
    let plan = device_attention_capacity_plan(
        model_dir,
        qd4_enabled,
        qd4_immutable_expert_cache,
        request_rope_rows,
        certified_argmax_production,
    )?;
    if available_bytes < plan.required_bytes {
        return Err(format!(
            "device attention capacity requires {} bytes (QD4={} immutable_cache={} max_fixed={} layer={} states={} state_bytes={} max_state={} hca_state={} rope_lut={} tid2eid_transient={} certified_argmax_temp={} headroom={}), but cuMemGetInfo reports {} free",
            plan.required_bytes,
            plan.qd4_device_bytes,
            plan.qd4_immutable_expert_cache_device_bytes,
            plan.max_fixed_device_bytes,
            plan.fixed_layer,
            plan.device_token_state_count,
            plan.device_token_state_total_bytes,
            plan.max_device_token_state_bytes,
            plan.hca_device_token_state_bytes,
            plan.rope_lut_bytes,
            plan.tid2eid_transient_max_bytes,
            plan.certified_argmax_device_temp_bytes,
            plan.headroom_bytes,
            available_bytes,
        )
        .into());
    }
    Ok(plan)
}

fn fixed_layer_plans(
    model_dir: &std::path::Path,
    layers: &[usize],
    baseline_path: &std::path::Path,
) -> Result<Vec<FixedLayerPlan>, Box<dyn Error>> {
    let mut baseline = vec![None; v4_state::V4_LAYER_COUNT];
    for line in std::fs::read_to_string(baseline_path)?.lines() {
        let value: serde_json::Value = serde_json::from_str(line)?;
        if value.get("event").and_then(|value| value.as_str()) != Some("layer") {
            continue;
        }
        let layer = usize::try_from(
            value
                .get("layer")
                .and_then(|value| value.as_u64())
                .ok_or("baseline layer record is missing layer")?,
        )?;
        let bench = value
            .get("bench")
            .and_then(|value| value.as_object())
            .ok_or("baseline layer record is missing bench telemetry")?;
        baseline[layer] = Some((
            bench["fixed_read_alloc_ns"]
                .as_u64()
                .ok_or("baseline fixed_read_alloc_ns is invalid")?,
            bench["fixed_decode_ns"]
                .as_u64()
                .ok_or("baseline fixed_decode_ns is invalid")?,
            bench["fixed_h2d_ns"]
                .as_u64()
                .ok_or("baseline fixed_h2d_ns is invalid")?,
            usize::try_from(
                bench["fixed_h2d_bytes"]
                    .as_u64()
                    .ok_or("baseline fixed_h2d_bytes is invalid")?,
            )?,
        ));
    }
    layers
        .iter()
        .map(|&layer| {
            let (read_alloc_ns, decode_ns, h2d_ns, expected_device_bytes) =
                baseline[layer].ok_or("baseline does not cover every cache candidate layer")?;
            let (tensors, device_bytes, wo_a_source_bytes) =
                fixed_layer_tensor_plan(model_dir, layer)?;
            let raw_device_bytes = device_bytes
                .checked_sub(wo_a_source_bytes)
                .ok_or("raw fixed device byte sum overflow")?;
            if device_bytes != expected_device_bytes {
                return Err(format!(
                    "layer {layer} fixed manifest bytes {device_bytes} != baseline {expected_device_bytes}"
                )
                .into());
            }
            let host_bytes = fixed_layer_decoded_host_bytes(layer, device_bytes)?;
            let raw_host_bytes = host_bytes
                .checked_sub(wo_a_source_bytes)
                .ok_or("raw fixed host byte sum overflow")?;
            Ok(FixedLayerPlan {
                layer,
                tensors,
                device_bytes,
                raw_device_bytes,
                host_bytes,
                raw_host_bytes,
                device_saved_ns: read_alloc_ns
                    .saturating_add(decode_ns)
                    .saturating_add(h2d_ns),
                host_saved_ns: read_alloc_ns.saturating_add(decode_ns),
            })
        })
        .collect()
}

fn stationary_raw_layer_plans(
    model_dir: &std::path::Path,
    layers: &[usize],
) -> Result<Vec<FixedLayerPlan>, Box<dyn Error>> {
    layers
        .iter()
        .map(|&layer| {
            let (tensors, device_bytes, wo_a_source_bytes) =
                fixed_layer_tensor_plan(model_dir, layer)?;
            let raw_device_bytes = device_bytes
                .checked_sub(wo_a_source_bytes)
                .ok_or("stationary raw device byte sum overflow")?;
            let host_bytes = fixed_layer_decoded_host_bytes(layer, device_bytes)?;
            let raw_host_bytes = host_bytes
                .checked_sub(wo_a_source_bytes)
                .ok_or("stationary raw host byte sum overflow")?;
            // ponytail: prefill capture has no timing baseline yet. Equal benefit
            // makes the existing selector retain the most layers per byte.
            Ok(FixedLayerPlan {
                layer,
                tensors,
                device_bytes,
                raw_device_bytes,
                host_bytes,
                raw_host_bytes,
                device_saved_ns: 1,
                host_saved_ns: 1,
            })
        })
        .collect()
}

fn bench_telemetry_enabled() -> bool {
    std::env::var_os("COLIBRI_V4_BENCH_TELEMETRY").as_deref() == Some(std::ffi::OsStr::new("1"))
        || std::env::var_os("COLIBRI_V4_PHASE_PROFILE").as_deref()
            == Some(std::ffi::OsStr::new("1"))
}

fn load_layer_host(
    model_dir: &std::path::Path,
    layer: usize,
    token_ids: &[i64],
) -> Result<(DecodedLayerHost, FixedLoadTelemetry), Box<dyn Error>> {
    load_layer_host_inner(model_dir, layer, token_ids, false)
}

fn load_layer_host_raw_wo_a(
    model_dir: &std::path::Path,
    layer: usize,
    token_ids: &[i64],
) -> Result<(DecodedLayerHost, FixedLoadTelemetry), Box<dyn Error>> {
    load_layer_host_inner(model_dir, layer, token_ids, true)
}

fn load_layer_host_inner(
    model_dir: &std::path::Path,
    layer: usize,
    token_ids: &[i64],
    raw_wo_a: bool,
) -> Result<(DecodedLayerHost, FixedLoadTelemetry), Box<dyn Error>> {
    load_layer_host_inner_for(&v4_spec::PRO, model_dir, layer, token_ids, raw_wo_a)
}

fn load_layer_host_inner_for(
    spec: &'static v4_spec::V4Spec,
    model_dir: &std::path::Path,
    layer: usize,
    token_ids: &[i64],
    raw_wo_a: bool,
) -> Result<(DecodedLayerHost, FixedLoadTelemetry), Box<dyn Error>> {
    let read_started = std::time::Instant::now();
    let state = v4_state::load_layer_state_weights_for(spec, model_dir, layer, token_ids)?;
    let projections = v4_state::load_layer_fp8_projections_for(spec, model_dir, layer)?;
    let tail = v4_state::load_layer_tail_for(spec, model_dir, layer, raw_wo_a)?;
    let fixed_decode_ns = (tail.wo_a_decode_ms * 1_000_000.0) as u64;
    let fixed_read_alloc_ns = u64::try_from(read_started.elapsed().as_nanos())
        .unwrap_or(u64::MAX)
        .saturating_sub(fixed_decode_ns);
    let host = DecodedLayerHost {
        state,
        projections,
        tail,
    };
    let fixed_source_bytes = host.source_bytes();
    Ok((
        host,
        FixedLoadTelemetry {
            fixed_read_alloc_ns,
            fixed_decode_ns,
            fixed_source_bytes,
            cache_status: "miss",
            ..FixedLoadTelemetry::default()
        },
    ))
}

fn upload_layer_runtime(
    model_dir: &std::path::Path,
    layer: usize,
    host: &DecodedLayerHost,
    stream: &CudaStream,
) -> Result<LoadedLayerRuntime, Box<dyn Error>> {
    upload_layer_runtime_for(&v4_spec::PRO, model_dir, layer, host, stream)
}

fn upload_layer_runtime_for(
    spec: &'static v4_spec::V4Spec,
    model_dir: &std::path::Path,
    layer: usize,
    host: &DecodedLayerHost,
    stream: &CudaStream,
) -> Result<LoadedLayerRuntime, Box<dyn Error>> {
    let expert_shard = safetensors::ShardFile::open(model_dir.join(spec.shard_name(layer + 2)?))?;
    let weights =
        LayerDeviceWeights::new_layer(stream, &host.state, &host.projections, &host.tail)?;
    Ok(LoadedLayerRuntime {
        weights,
        expert_shard,
    })
}

fn upload_layer_runtime_raw_wo_a(
    model_dir: &std::path::Path,
    layer: usize,
    host: &DecodedLayerHost,
    module: &kernels::LoadedModule,
    stream: &CudaStream,
) -> Result<(LoadedLayerRuntime, FixedLoadTelemetry, DeviceBuffer<u8>), Box<dyn Error>> {
    let raw = host
        .tail
        .wo_a_raw
        .as_deref()
        .ok_or("raw wo_a upload received a decoded host payload")?;
    let raw_h2d_started = std::time::Instant::now();
    let raw_wo_a = DeviceBuffer::from_host(stream, raw)?;
    let raw_h2d_ns = u64::try_from(raw_h2d_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
    let raw_h2d_bytes = raw_wo_a.num_bytes();
    let decode_started = std::time::Instant::now();
    let mut wo_a = DeviceBuffer::zeroed(stream, raw_wo_a.len())?;
    launch_e4m3fn_to_bf16(module, stream, &raw_wo_a, &mut wo_a)?;
    stream.synchronize()?;
    let fixed_decode_ns = u64::try_from(decode_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
    let fixed_h2d_started = std::time::Instant::now();
    let expert_shard = safetensors::ShardFile::open(
        model_dir.join(format!("model-{:05}-of-00064.safetensors", layer + 2)),
    )?;
    let weights = LayerDeviceWeights::new_layer_with_wo_a(
        stream,
        &host.state,
        &host.projections,
        &host.tail,
        Some(wo_a),
    )?;
    let fixed_h2d_ns = raw_h2d_ns
        .saturating_add(u64::try_from(fixed_h2d_started.elapsed().as_nanos()).unwrap_or(u64::MAX));
    let fixed_h2d_bytes = raw_h2d_bytes
        .checked_add(weights.num_bytes_without_wo_a())
        .ok_or("raw wo_a fixed H2D byte count overflow")?;
    Ok((
        LoadedLayerRuntime {
            weights,
            expert_shard,
        },
        FixedLoadTelemetry {
            fixed_decode_ns,
            fixed_h2d_ns,
            fixed_h2d_bytes,
            raw_h2d_ns,
            raw_h2d_bytes,
            cache_status: "prefetched_blocking_h2d",
            ..FixedLoadTelemetry::default()
        },
        raw_wo_a,
    ))
}

fn upload_pinned_layer_runtime_raw(
    model_dir: &std::path::Path,
    layer: usize,
    host: &DecodedLayerHost,
    module: &kernels::LoadedModule,
    stream: &std::sync::Arc<CudaStream>,
    slot: &mut PinnedFixedSlot,
) -> Result<(LoadedLayerRuntime, DeviceBuffer<u8>, usize), Box<dyn Error>> {
    let mut cursor = 0_usize;
    macro_rules! copy {
        ($source:expr) => {{
            let source = $source;
            let bytes = std::mem::size_of_val(source);
            let end = cursor
                .checked_add(bytes)
                .ok_or("pinned fixed slot cursor overflow")?;
            if end > slot.host.len() {
                return Err("pinned fixed layer exceeds its CSA-max slot".into());
            }
            unsafe {
                std::ptr::copy_nonoverlapping(
                    source.as_ptr().cast::<u8>(),
                    slot.host.as_mut_ptr().add(cursor),
                    bytes,
                );
            }
            let device = unsafe { DeviceBuffer::uninitialized_async(stream, source.len())? };
            let result = unsafe {
                sys::cuMemcpyHtoDAsync_v2(
                    device.cu_deviceptr(),
                    slot.host.as_ptr().add(cursor).cast(),
                    bytes,
                    stream.cu_stream(),
                )
            };
            if result != 0 {
                return Err(format!("pinned fixed H2D enqueue failed with {result:?}").into());
            }
            cursor = end;
            device
        }};
    }
    macro_rules! fp8 {
        ($matrix:expr) => {{
            let matrix = $matrix;
            DeviceFp8Matrix {
                weight: copy!(matrix.weight.as_slice()),
                scale: copy!(matrix.scale.as_slice()),
                in_features: matrix.in_features,
                out_features: matrix.out_features,
            }
        }};
    }
    let state = &host.state;
    let projections = &host.projections;
    let tail = &host.tail;
    let router_bias = match &state.router {
        v4_state::LayerRouterWeights::Hash { .. } => {
            vec![0.0_f32; v4_router::V4_ROUTED_EXPERTS]
        }
        v4_state::LayerRouterWeights::Learned { bias } => bias.clone(),
    };
    slot.h2d_start.record(stream)?;
    let hc_attn_fn = copy!(state.hc_fn.as_slice());
    let hc_attn_base = copy!(state.hc_base.as_slice());
    let hc_attn_scale = copy!(state.hc_scale.as_slice());
    let attn_norm = copy!(state.attn_norm.as_slice());
    let wq_a = fp8!(&projections.wq_a.matrix);
    let q_norm = copy!(projections.wq_a.norm.as_slice());
    let wq_b = fp8!(&projections.wq_b);
    let wkv = fp8!(&projections.wkv.matrix);
    let kv_norm = copy!(projections.wkv.norm.as_slice());
    let compressor_wkv = copy!(state.compressor.wkv_bf16.as_slice());
    let compressor_wgate = copy!(state.compressor.wgate_bf16.as_slice());
    let compressor_ape = copy!(state.compressor.ape_f32.as_slice());
    let compressor_norm = copy!(state.compressor.norm_bf16.as_slice());
    let indexer = state
        .indexer
        .as_ref()
        .map(|indexer| {
            Ok::<_, Box<dyn Error>>(DeviceLayerIndexer {
                wq_b: fp8!(&indexer.wq_b),
                weights_proj: copy!(indexer.weights_proj_bf16.as_slice()),
                compressor_wkv: copy!(indexer.compressor.wkv_bf16.as_slice()),
                compressor_wgate: copy!(indexer.compressor.wgate_bf16.as_slice()),
                compressor_ape: copy!(indexer.compressor.ape_f32.as_slice()),
                compressor_norm: copy!(indexer.compressor.norm_bf16.as_slice()),
            })
        })
        .transpose()?;
    let attn_sink = copy!(tail.attn_sink.as_slice());
    let raw_wo_a = copy!(
        tail.wo_a_raw
            .as_deref()
            .ok_or("pinned fixed upload requires raw wo_a")?
    );
    let wo_b = fp8!(&tail.wo_b);
    let hc_ffn_fn = copy!(tail.hc_ffn_fn.as_slice());
    let hc_ffn_base = copy!(tail.hc_ffn_base.as_slice());
    let hc_ffn_scale = copy!(tail.hc_ffn_scale.as_slice());
    let ffn_norm = copy!(state.ffn_norm.as_slice());
    let gate = copy!(state.gate.as_slice());
    let router_bias = copy!(router_bias.as_slice());
    let shared_w1 = fp8!(&tail.shared.w1);
    let shared_w2 = fp8!(&tail.shared.w2);
    let shared_w3 = fp8!(&tail.shared.w3);
    slot.h2d_end.record(stream)?;
    let mut wo_a = slot
        .scratch
        .replace(unsafe { DeviceBuffer::uninitialized_async(stream, 0)? })
        .ok_or("pinned fixed slot lost its decode scratch")?;
    launch_e4m3fn_to_bf16(module, stream, &raw_wo_a, &mut wo_a)?;
    slot.ready.record(stream)?;
    let expert_shard = safetensors::ShardFile::open(
        model_dir.join(format!("model-{:05}-of-00064.safetensors", layer + 2)),
    )?;
    Ok((
        LoadedLayerRuntime {
            weights: LayerDeviceWeights {
                spec: state.spec,
                layer,
                compression_ratio: state.compression_ratio,
                hc_attn_fn,
                hc_attn_base,
                hc_attn_scale,
                attn_norm,
                wq_a,
                q_norm,
                wq_b,
                wkv,
                kv_norm,
                compressor_wkv,
                compressor_wgate,
                compressor_ape,
                compressor_norm,
                indexer,
                attn_sink,
                wo_a,
                wo_b,
                hc_ffn_fn,
                hc_ffn_base,
                hc_ffn_scale,
                ffn_norm,
                gate,
                router_bias,
                hash_router: matches!(state.router, v4_state::LayerRouterWeights::Hash { .. }),
                shared_w1,
                shared_w2,
                shared_w3,
            },
            expert_shard,
        },
        raw_wo_a,
        cursor,
    ))
}

fn read_raw_wo_a(
    model_dir: &std::path::Path,
    layer: usize,
) -> Result<(Vec<u8>, u64, usize), Box<dyn Error>> {
    let shard = safetensors::ShardFile::open(
        model_dir.join(format!("model-{:05}-of-00064.safetensors", layer + 2)),
    )?;
    let weight_name = format!("layers.{layer}.attn.wo_a.weight");
    let scale_name = format!("layers.{layer}.attn.wo_a.scale");
    let weight_info = shard.tensor(&weight_name)?;
    let scale_info = shard.tensor(&scale_name)?;
    if weight_info.dtype != safetensors::TensorDtype::F8E4M3
        || scale_info.dtype != safetensors::TensorDtype::F8E8M0
        || weight_info.shape.len() != 2
        || scale_info.shape
            != [
                weight_info.shape[0] / u64::try_from(v4_state::FP8_BLOCK)?,
                weight_info.shape[1] / u64::try_from(v4_state::FP8_BLOCK)?,
            ]
    {
        return Err(format!("layer {layer} raw wo_a dtype contract mismatch").into());
    }
    let mut weight = vec![0_u8; usize::try_from(weight_info.byte_len)?];
    let read_started = std::time::Instant::now();
    shard.read_at(&weight_name, 0, &mut weight)?;
    let read_ns = u64::try_from(read_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
    let source_bytes = weight.len();
    Ok((weight, read_ns, source_bytes))
}

fn into_raw_layer_runtime(
    model_dir: &std::path::Path,
    layer: usize,
    mut runtime: LoadedLayerRuntime,
    stream: &CudaStream,
) -> Result<(RawLoadedLayerRuntime, u64, usize, u64, usize), Box<dyn Error>> {
    let (weight, raw_read_ns, raw_source_bytes) = read_raw_wo_a(model_dir, layer)?;
    if runtime.weights.wo_a.len() != weight.len() {
        return Err(format!("layer {layer} raw wo_a shape contract mismatch").into());
    }
    let raw_h2d_started = std::time::Instant::now();
    let wo_a_raw = DeviceBuffer::from_host(stream, &weight)?;
    stream.synchronize()?;
    let raw_h2d_ns = u64::try_from(raw_h2d_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
    let raw_h2d_bytes = wo_a_raw.num_bytes();
    let decoded_wo_a =
        std::mem::replace(&mut runtime.weights.wo_a, DeviceBuffer::zeroed(stream, 0)?);
    drop(decoded_wo_a);
    let resident_bytes = runtime.weights.num_bytes() + wo_a_raw.num_bytes();
    Ok((
        RawLoadedLayerRuntime {
            runtime,
            wo_a_raw,
            resident_bytes,
        },
        raw_read_ns,
        raw_source_bytes,
        raw_h2d_ns,
        raw_h2d_bytes,
    ))
}

fn into_raw_layer_host(
    model_dir: &std::path::Path,
    layer: usize,
    mut host: DecodedLayerHost,
) -> Result<(RawDecodedLayerHost, u64, usize), Box<dyn Error>> {
    let (wo_a_raw, raw_read_ns, raw_source_bytes) = read_raw_wo_a(model_dir, layer)?;
    if host.tail.wo_a_bf16.len() != wo_a_raw.len() {
        return Err(format!("layer {layer} raw host wo_a shape contract mismatch").into());
    }
    drop(std::mem::take(&mut host.tail.wo_a_bf16));
    let resident_bytes = host.num_bytes() + wo_a_raw.len();
    Ok((
        RawDecodedLayerHost {
            host,
            wo_a_raw,
            resident_bytes,
        },
        raw_read_ns,
        raw_source_bytes,
    ))
}

fn load_layer_runtime(
    model_dir: &std::path::Path,
    layer: usize,
    token_ids: &[i64],
    stream: &CudaStream,
) -> Result<(LoadedLayerRuntime, FixedLoadTelemetry), Box<dyn Error>> {
    load_layer_runtime_for(&v4_spec::PRO, model_dir, layer, token_ids, stream)
}

fn load_layer_runtime_for(
    spec: &'static v4_spec::V4Spec,
    model_dir: &std::path::Path,
    layer: usize,
    token_ids: &[i64],
    stream: &CudaStream,
) -> Result<(LoadedLayerRuntime, FixedLoadTelemetry), Box<dyn Error>> {
    let (host, mut telemetry) =
        load_layer_host_inner_for(spec, model_dir, layer, token_ids, false)?;
    let h2d_started = std::time::Instant::now();
    let runtime = upload_layer_runtime_for(spec, model_dir, layer, &host, stream)?;
    if bench_telemetry_enabled() {
        stream.synchronize()?;
    }
    telemetry.fixed_h2d_ns = u64::try_from(h2d_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
    telemetry.fixed_h2d_bytes = runtime.weights.num_bytes();
    Ok((runtime, telemetry))
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct B2LayerDigest {
    layer: usize,
    route_ids: Vec<i32>,
    route_weight_bits: Vec<u32>,
    attention_topk_digest: u64,
    hidden_digest: u64,
    kv_digest: u64,
    main_kv_state_digest: u64,
    main_score_state_digest: u64,
    indexer_kv_state_digest: Option<u64>,
    indexer_score_state_digest: Option<u64>,
}

impl B2LayerDigest {
    fn json(&self) -> serde_json::Value {
        serde_json::json!({
            "layer": self.layer,
            "route_ids": self.route_ids,
            "route_weight_bits": self.route_weight_bits,
            "attention_topk_digest": format!("{:016x}", self.attention_topk_digest),
            "hidden_digest": format!("{:016x}", self.hidden_digest),
            "kv_digest": format!("{:016x}", self.kv_digest),
            "main_kv_state_digest": format!("{:016x}", self.main_kv_state_digest),
            "main_score_state_digest": format!("{:016x}", self.main_score_state_digest),
            "indexer_kv_state_digest": self.indexer_kv_state_digest.map(|value| format!("{value:016x}")),
            "indexer_score_state_digest": self.indexer_score_state_digest.map(|value| format!("{value:016x}")),
        })
    }

    fn ordered_chain_sha256(layers: &[Self]) -> Result<String, Box<dyn Error>> {
        let canonical = serde_json::to_vec(&serde_json::json!({
            "schema": "colibri-v4-b2-ordered-layer-digest-v1",
            "layers": layers.iter().map(Self::json).collect::<Vec<_>>(),
        }))?;
        sha256_bytes(&canonical)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct B2RequestResult {
    layers: Vec<B2LayerDigest>,
    output_digest: u64,
}

fn parse_b2_generate_token_ids<const REQUESTS: usize>(
    value: &str,
    token_ids_env: &str,
) -> Result<[Vec<i64>; REQUESTS], Box<dyn Error>> {
    parse_b2_generate_token_ids_for(&v4_spec::PRO, value, token_ids_env)
}

fn parse_b2_generate_token_ids_for<const REQUESTS: usize>(
    spec: &'static v4_spec::V4Spec,
    value: &str,
    token_ids_env: &str,
) -> Result<[Vec<i64>; REQUESTS], Box<dyn Error>> {
    let prompts = value
        .split(';')
        .map(|prompt| {
            let prompt = prompt.trim();
            if prompt.is_empty() {
                return Err(format!(
                    "{token_ids_env} requires {REQUESTS} non-empty comma-separated token lists"
                )
                .into());
            }
            let ids = prompt
                .split(',')
                .map(str::trim)
                .map(str::parse::<i64>)
                .collect::<Result<Vec<_>, _>>()?;
            v4_state::validate_ids_for(spec, &ids)?;
            Ok(ids)
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    prompts.try_into().map_err(|_| {
        format!("{token_ids_env} requires exactly {REQUESTS} token lists separated by ';'").into()
    })
}

fn validate_b2_generate_inputs<const REQUESTS: usize>(
    prompts: &[Vec<i64>; REQUESTS],
    greedy_steps: usize,
) -> Result<[usize; REQUESTS], Box<dyn Error>> {
    if greedy_steps == 0 {
        return Err("COLIBRI_V4_GREEDY_STEPS must be at least 1 for row-set generation".into());
    }
    let forward_tokens = |request: usize| -> Result<usize, Box<dyn Error>> {
        let forward_tokens = greedy_forward_tokens(prompts[request].len(), greedy_steps)?;
        if forward_tokens > v4_state::KV_WINDOW {
            return Err(format!(
                "row-set request {request} prompt plus greedy decode exceeds the existing V4 KV window"
            )
            .into());
        }
        Ok(forward_tokens)
    };
    (0..REQUESTS)
        .map(forward_tokens)
        .collect::<Result<Vec<_>, _>>()?
        .try_into()
        .map_err(|_| "row-set request count changed".into())
}

fn b2_generate_inputs_from_env<const REQUESTS: usize>(
    token_ids_env: &str,
) -> Result<([Vec<i64>; REQUESTS], usize, [usize; REQUESTS]), Box<dyn Error>> {
    let prompts = parse_b2_generate_token_ids::<REQUESTS>(
        &std::env::var(token_ids_env).map_err(|_| format!("set {token_ids_env}='<ids>;...'"))?,
        token_ids_env,
    )?;
    let greedy_steps = std::env::var("COLIBRI_V4_GREEDY_STEPS")
        .map_err(|_| "set COLIBRI_V4_GREEDY_STEPS for B2 generation")?
        .parse::<usize>()?;
    let forward_tokens = validate_b2_generate_inputs::<REQUESTS>(&prompts, greedy_steps)?;
    Ok((prompts, greedy_steps, forward_tokens))
}

#[derive(Default)]
struct B2Physical {
    fixed_source_bytes: usize,
    fixed_h2d_bytes: usize,
    request_router_metadata_source_bytes: usize,
    request_router_metadata_h2d_bytes: usize,
    routed_source_bytes: usize,
    routed_h2d_bytes: usize,
    initial_state_source_bytes: usize,
    final_head_source_bytes: usize,
}

#[derive(Default)]
struct B2RequestPhysical {
    initial_state_source_bytes: usize,
    final_head_source_bytes: usize,
}

struct B2GenerateRequest {
    prompt_ids: Vec<i64>,
    input_ids: Vec<i64>,
    position: usize,
    persistent: v4_state::V4SerialPersistentState,
    hidden: Vec<u16>,
    generated_ids: Vec<i64>,
    stop_reason: Option<GreedyStopReason>,
    tokens: Vec<serde_json::Value>,
    finals: Vec<serde_json::Value>,
    physical: B2RequestPhysical,
    flash_b1_phase: FlashB1PhaseTiming,
}

#[derive(Default)]
struct FlashB1PhaseTiming {
    layers: Vec<serde_json::Value>,
    layer_outer_ns: u64,
    final_heads: Vec<serde_json::Value>,
    final_head_outer_ns: u64,
}

struct B2GenerateArm<const REQUESTS: usize> {
    requests: [B2GenerateRequest; REQUESTS],
    physical: B2Physical,
    duration_ns: u64,
    peak_device_bytes: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct FlashParityStopAfterLayer {
    layer: usize,
    dir: std::path::PathBuf,
}

#[derive(Debug)]
struct FlashParityStopped {
    path: std::path::PathBuf,
}

impl std::fmt::Display for FlashParityStopped {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "Flash parity intentionally stopped after the requested layer; partial artifact is {}",
            self.path.display()
        )
    }
}

impl std::error::Error for FlashParityStopped {}

fn flash_parity_stop_after_layer_for(
    spec: &v4_spec::V4Spec,
    flash_base: bool,
    dump_dir: Option<&std::ffi::OsStr>,
    value: Option<&std::ffi::OsStr>,
) -> Result<Option<FlashParityStopAfterLayer>, Box<dyn Error>> {
    let Some(value) = value else {
        return Ok(None);
    };
    if !flash_base || spec != &v4_spec::FLASH_0731 {
        return Err(
            "COLIBRI_V4_FLASH_PARITY_STOP_AFTER_LAYER is only valid for Flash-0731 base mode"
                .into(),
        );
    }
    let Some(dump_dir) = dump_dir else {
        return Err(
            "COLIBRI_V4_FLASH_PARITY_STOP_AFTER_LAYER requires COLIBRI_V4_FLASH_PARITY_DUMP_DIR"
                .into(),
        );
    };
    if dump_dir.is_empty() {
        return Err("COLIBRI_V4_FLASH_PARITY_DUMP_DIR must not be empty".into());
    }
    let layer = value
        .to_string_lossy()
        .parse::<usize>()
        .map_err(|_| "COLIBRI_V4_FLASH_PARITY_STOP_AFTER_LAYER must be a layer index")?;
    if layer >= spec.layers {
        return Err("COLIBRI_V4_FLASH_PARITY_STOP_AFTER_LAYER is outside Flash-0731 depth".into());
    }
    Ok(Some(FlashParityStopAfterLayer {
        layer,
        dir: std::path::PathBuf::from(dump_dir),
    }))
}

fn flash_parity_stop_after_layer(
    spec: &v4_spec::V4Spec,
) -> Result<Option<FlashParityStopAfterLayer>, Box<dyn Error>> {
    flash_parity_stop_after_layer_for(
        spec,
        true,
        std::env::var_os("COLIBRI_V4_FLASH_PARITY_DUMP_DIR").as_deref(),
        std::env::var_os("COLIBRI_V4_FLASH_PARITY_STOP_AFTER_LAYER").as_deref(),
    )
}

fn write_flash_parity_stop(
    stop: &FlashParityStopAfterLayer,
    spec: &v4_spec::V4Spec,
    layers: &[B2LayerDigest],
    phase_profile: Option<&serde_json::Value>,
) -> Result<std::path::PathBuf, Box<dyn Error>> {
    if layers.len() != stop.layer + 1
        || layers.last().map(|digest| digest.layer) != Some(stop.layer)
    {
        return Err("Flash parity stop did not retain exactly the requested layer digest".into());
    }
    std::fs::create_dir_all(&stop.dir)?;
    let path = stop.dir.join("parity-stop.json");
    let ordered_layer_digest_sha256 = B2LayerDigest::ordered_chain_sha256(layers)?;
    let mut output = PendingResultFile::reserve(path.clone())?;
    let mut payload = serde_json::json!({
        "schema": "colibri-flash-parity-stop-v1",
        "revision": spec.revision,
        "layer": stop.layer,
        "ordered_layer_digest_sha256": ordered_layer_digest_sha256,
        "partial": true,
    });
    if let Some(phase_profile) = phase_profile {
        payload
            .as_object_mut()
            .ok_or("Flash parity-stop payload must be a JSON object")?
            .insert("phase_profile".to_owned(), phase_profile.clone());
    }
    serde_json::to_writer_pretty(output.file_mut(), &payload)?;
    output.file_mut().write_all(b"\n")?;
    output.commit()?;
    Ok(path)
}

struct B2GenerateSummary<const REQUESTS: usize> {
    requests: [serde_json::Value; REQUESTS],
    physical: B2Physical,
    duration_ns: u64,
    peak_device_bytes: usize,
}

impl<const REQUESTS: usize> B2GenerateArm<REQUESTS> {
    fn into_final_summary(self) -> B2GenerateSummary<REQUESTS> {
        B2GenerateSummary {
            requests: self
                .requests
                .map(|request| b2_generate_semantic_json(&request)),
            physical: self.physical,
            duration_ns: self.duration_ns,
            peak_device_bytes: self.peak_device_bytes,
        }
    }
}

fn b4_greedy_a_duration_ns(first_ns: u64, second_ns: u64) -> Result<u64, Box<dyn Error>> {
    first_ns
        .checked_add(second_ns)
        .ok_or_else(|| "B4 greedy A duration overflow".into())
}

fn b2_generate_active_indices<const REQUESTS: usize>(active: &[bool; REQUESTS]) -> Vec<usize> {
    active
        .iter()
        .enumerate()
        .filter_map(|(request, &live)| live.then_some(request))
        .collect()
}

fn b2_generate_stop_reason(reason: GreedyStopReason) -> &'static str {
    match reason {
        GreedyStopReason::Eos => "eos",
        GreedyStopReason::MaxNewTokens => "max_new_tokens",
    }
}

fn b2_generate_request_json(request: usize, state: &B2GenerateRequest) -> serde_json::Value {
    serde_json::json!({
        "request": request,
        "prompt_ids": state.prompt_ids,
        "emitted_token_ids": state.input_ids,
        "generated_ids": state.generated_ids,
        "stop_reason": state.stop_reason.map(b2_generate_stop_reason),
        "tokens": state.tokens,
        "finals": state.finals,
        "physical": state.physical.json(),
    })
}

fn b2_generate_semantic_json(state: &B2GenerateRequest) -> serde_json::Value {
    serde_json::json!({
        "prompt_ids": state.prompt_ids,
        "emitted_token_ids": state.input_ids,
        "generated_ids": state.generated_ids,
        "stop_reason": state.stop_reason.map(b2_generate_stop_reason),
        "tokens": state.tokens.iter().map(|token| serde_json::json!({
            "token_position": token["token_position"], "input_id": token["input_id"],
            "input_kind": token["input_kind"], "hidden_bf16_digest": token["hidden_bf16_digest"],
            "ordered_layer_digest_sha256": token["ordered_layer_digest_sha256"],
        })).collect::<Vec<_>>(),
        "finals": state.finals,
    })
}

fn b2_router_metadata_row<'a>(
    hash_router: bool,
    ids: &'a [i64],
    request: usize,
    requests: usize,
) -> Result<&'a [i64], Box<dyn Error>> {
    if !hash_router {
        if ids.is_empty() {
            return Ok(&[]);
        }
        return Err("B2 learned router received request metadata".into());
    }
    let start = request
        .checked_mul(v4_router::V4_TOPK)
        .ok_or("B2 router metadata row offset overflow")?;
    let end = start
        .checked_add(v4_router::V4_TOPK)
        .ok_or("B2 router metadata row end overflow")?;
    if request >= requests || ids.len() != requests * v4_router::V4_TOPK {
        return Err("B2 hash router metadata rows do not match the row set".into());
    }
    Ok(&ids[start..end])
}

impl B2Physical {
    fn checked_add_assign(&mut self, other: B2Physical) -> Result<(), Box<dyn Error>> {
        self.fixed_source_bytes = self
            .fixed_source_bytes
            .checked_add(other.fixed_source_bytes)
            .ok_or("B2 fixed source byte overflow")?;
        self.fixed_h2d_bytes = self
            .fixed_h2d_bytes
            .checked_add(other.fixed_h2d_bytes)
            .ok_or("B2 fixed H2D byte overflow")?;
        self.request_router_metadata_source_bytes = self
            .request_router_metadata_source_bytes
            .checked_add(other.request_router_metadata_source_bytes)
            .ok_or("B2 router metadata source byte overflow")?;
        self.request_router_metadata_h2d_bytes = self
            .request_router_metadata_h2d_bytes
            .checked_add(other.request_router_metadata_h2d_bytes)
            .ok_or("B2 router metadata H2D byte overflow")?;
        self.routed_source_bytes = self
            .routed_source_bytes
            .checked_add(other.routed_source_bytes)
            .ok_or("B2 routed source byte overflow")?;
        self.routed_h2d_bytes = self
            .routed_h2d_bytes
            .checked_add(other.routed_h2d_bytes)
            .ok_or("B2 routed H2D byte overflow")?;
        self.initial_state_source_bytes = self
            .initial_state_source_bytes
            .checked_add(other.initial_state_source_bytes)
            .ok_or("B2 initial state source byte overflow")?;
        self.final_head_source_bytes = self
            .final_head_source_bytes
            .checked_add(other.final_head_source_bytes)
            .ok_or("B2 final head source byte overflow")?;
        Ok(())
    }

    fn record_initial_state_source_bytes(&mut self, bytes: usize) -> Result<(), Box<dyn Error>> {
        self.initial_state_source_bytes = self
            .initial_state_source_bytes
            .checked_add(bytes)
            .ok_or("B2 initial state source byte overflow")?;
        Ok(())
    }

    fn record_final_head_source_bytes(&mut self, bytes: usize) -> Result<(), Box<dyn Error>> {
        self.final_head_source_bytes = self
            .final_head_source_bytes
            .checked_add(bytes)
            .ok_or("B2 final head source byte overflow")?;
        Ok(())
    }

    fn record_request_router_metadata(
        &mut self,
        hash_router: bool,
        ids: &[i64],
    ) -> Result<(), Box<dyn Error>> {
        if hash_router {
            if ids.len() != v4_router::V4_TOPK {
                return Err("B2 hash router metadata row is incomplete".into());
            }
        } else if !ids.is_empty() {
            return Err("B2 learned router received request metadata".into());
        }
        let bytes = std::mem::size_of_val(ids);
        self.request_router_metadata_source_bytes = self
            .request_router_metadata_source_bytes
            .checked_add(bytes)
            .ok_or("B2 router metadata source byte overflow")?;
        self.request_router_metadata_h2d_bytes = self
            .request_router_metadata_h2d_bytes
            .checked_add(bytes)
            .ok_or("B2 router metadata H2D byte overflow")?;
        Ok(())
    }

    fn json(&self) -> serde_json::Value {
        serde_json::json!({
            "fixed_source_bytes": self.fixed_source_bytes,
            "fixed_h2d_bytes": self.fixed_h2d_bytes,
            "request_router_metadata_source_bytes": self.request_router_metadata_source_bytes,
            "request_router_metadata_h2d_bytes": self.request_router_metadata_h2d_bytes,
            "routed_source_bytes": self.routed_source_bytes,
            "routed_h2d_bytes": self.routed_h2d_bytes,
            "initial_state_source_bytes": self.initial_state_source_bytes,
            "final_head_source_bytes": self.final_head_source_bytes,
        })
    }
}

impl B2RequestPhysical {
    fn record_initial_state_source_bytes(&mut self, bytes: usize) -> Result<(), Box<dyn Error>> {
        self.initial_state_source_bytes = self
            .initial_state_source_bytes
            .checked_add(bytes)
            .ok_or("B2 request initial state source byte overflow")?;
        Ok(())
    }

    fn record_final_head_source_bytes(&mut self, bytes: usize) -> Result<(), Box<dyn Error>> {
        self.final_head_source_bytes = self
            .final_head_source_bytes
            .checked_add(bytes)
            .ok_or("B2 request final head source byte overflow")?;
        Ok(())
    }

    fn json(&self) -> serde_json::Value {
        serde_json::json!({
            "initial_state_source_bytes": self.initial_state_source_bytes,
            "final_head_source_bytes": self.final_head_source_bytes,
        })
    }
}

fn flash_b1_layer_phase_profile_json(
    layer: usize,
    outer_wall_ns: u64,
    fixed_prepare_ns: u64,
    fixed: &FixedLoadTelemetry,
    wo_a_decode_sync_ns: u64,
    stage: &LayerStageTelemetry,
) -> Result<serde_json::Value, Box<dyn Error>> {
    let fixed_service_ns = fixed
        .fixed_read_alloc_ns
        .checked_add(fixed.fixed_decode_ns)
        .and_then(|value| value.checked_add(fixed.fixed_h2d_ns))
        .ok_or("Flash B1 fixed service timing overflow")?;
    if (fixed.cache_status != "prefetched_host" && fixed_service_ns > fixed_prepare_ns)
        || stage.attention_ns > outer_wall_ns
        || stage.commit_ns > outer_wall_ns
        || stage.router_ns > outer_wall_ns
        || stage.shared_ns > outer_wall_ns
        || stage.routed_ns > outer_wall_ns
        || (!stage.routed_prefetched && stage.routed_read_ns > stage.routed_ns)
        || stage.routed_h2d_ns > stage.routed_ns
        || stage.routed_kernel_ns > stage.routed_ns
    {
        return Err("Flash B1 nested phase timing exceeds its parent scope".into());
    }
    let attention_wall_ns = stage
        .attention_ns
        .checked_add(stage.commit_ns)
        .ok_or("Flash B1 attention timing overflow")?;
    let attributed_ns = fixed_prepare_ns
        .checked_add(wo_a_decode_sync_ns)
        .and_then(|value| value.checked_add(attention_wall_ns))
        .and_then(|value| value.checked_add(stage.router_ns))
        .and_then(|value| value.checked_add(stage.shared_ns))
        .and_then(|value| value.checked_add(stage.routed_ns))
        .ok_or("Flash B1 layer phase timing overflow")?;
    if attributed_ns > outer_wall_ns {
        return Err("Flash B1 top-level phases exceed layer wall time".into());
    }
    let bench = bench_telemetry_enabled().then(|| serde_json::json!({
        "attention": {
            "device_state_setup_ns": stage.device_state_setup_ns,
            "device_state_setup_count": stage.device_state_setup_count,
            "device_state_reset_cuda_ns": stage.device_state_reset_cuda_ns,
            "device_state_reset_calls": stage.device_state_reset_calls,
            "audit_oracle_ns": stage.attention_audit_oracle_ns,
            "hidden_h2d": {"count": stage.attention_hidden_h2d_count, "bytes": stage.attention_hidden_h2d_bytes},
            "rope_lut_d2d": {"count": stage.attention_rope_lut_d2d_count, "bytes": stage.attention_rope_lut_d2d_bytes},
            "terminal_d2h": {"count": stage.attention_terminal_d2h_count, "bytes": stage.attention_terminal_d2h_bytes},
            "intermediate_or_audit_d2h": {"count": stage.attention_intermediate_or_audit_d2h_count, "bytes": stage.attention_intermediate_or_audit_d2h_bytes},
            "steady_device_temporary_alloc_count": stage.attention_steady_device_temporary_alloc_count,
            "certified_host_path": stage.attention_profile.as_ref().map(AttentionProfile::json),
        },
        "router": {
            "mhc": {
                "wall_ns": stage.router_mhc_ns,
                "input_alloc_h2d_sync_ns": stage.router_mhc_input_alloc_h2d_sync_ns,
                "workspace_alloc_sync_ns": stage.router_mhc_workspace_alloc_sync_ns,
                "workspace_init_gpu_ns": stage.router_mhc_workspace_init_gpu_ns,
                "mixing_linear_sync_ns": stage.router_mhc_mixing_linear_sync_ns,
                "mixing_linear_gpu_ns": stage.router_mhc_mixing_linear_gpu_ns,
                "sinkhorn_normalize_collapse_sync_ns": stage.router_mhc_sinkhorn_normalize_collapse_sync_ns,
                "sinkhorn_normalize_collapse_gpu_ns": stage.router_mhc_sinkhorn_normalize_collapse_gpu_ns,
                "output_d2h_sync_ns": stage.router_mhc_output_d2h_sync_ns,
                "other_ns": stage.router_mhc_other_ns,
            },
            "host_prep_ns": stage.router_host_prep_ns,
            "alloc_h2d_sync_ns": stage.router_alloc_h2d_sync_ns,
            "workspace_alloc_ns": stage.router_workspace_alloc_ns,
            "kernel_select_sync_ns": stage.router_kernel_select_sync_ns,
            "kernel_select_gpu_ns": stage.router_kernel_select_gpu_ns,
            "d2h_sync_ns": stage.router_d2h_sync_ns,
            "other_ns": stage.router_other_ns,
        },
    }));
    Ok(serde_json::json!({
        "layer": layer,
        "outer": {"wall_ns": outer_wall_ns},
        "top_level": {
            "fixed": {
                "wall_ns": fixed_prepare_ns,
                "host_prefetched": fixed.cache_status == "prefetched_host",
                "read_alloc_ns": fixed.fixed_read_alloc_ns,
                "decode_ns": fixed.fixed_decode_ns,
                "h2d_ns": fixed.fixed_h2d_ns,
                "source_bytes": fixed.fixed_source_bytes,
                "h2d_bytes": fixed.fixed_h2d_bytes,
            },
            "wo_a_decode_gpu_sync": {"wall_ns": wo_a_decode_sync_ns},
            "attention": {"wall_ns": attention_wall_ns, "execution_ns": stage.attention_ns, "commit_ns": stage.commit_ns},
            "router": {"wall_ns": stage.router_ns},
            "shared": {"wall_ns": stage.shared_ns},
            "routed": {
                "wall_ns": stage.routed_ns,
                "host_prefetched": stage.routed_prefetched,
                "diagnostics": {
                    "read_ns": stage.routed_read_ns,
                    "alloc_ns": stage.routed_alloc_ns,
                    "setup_ns": stage.routed_setup_ns,
                    "h2d_ns": stage.routed_h2d_ns,
                    "kernel_ns": stage.routed_kernel_ns,
                    "source_bytes": stage.routed_source_bytes,
                    "h2d_bytes": stage.routed_h2d_bytes,
                    "expert_loads": stage.routed_expert_loads,
                },
            },
        },
        "top_level_attributed_ns": attributed_ns,
        "unattributed_gap_ns": outer_wall_ns - attributed_ns,
        "bench": bench,
        "closure": {
            "equation": "top_level_attributed_ns + unattributed_gap_ns == outer.wall_ns",
            "exact": true,
        },
    }))
}

fn flash_b1_request_phase_profile_json(
    timing: &FlashB1PhaseTiming,
    request_wall_ns: u64,
) -> Result<serde_json::Value, Box<dyn Error>> {
    let layers_outer_ns = timing
        .layers
        .iter()
        .map(|layer| {
            layer["outer"]["wall_ns"]
                .as_u64()
                .ok_or("Flash B1 layer profile lacks wall time")
        })
        .try_fold(0_u64, |sum, value| {
            value.and_then(|value| {
                sum.checked_add(value)
                    .ok_or("Flash B1 layer wall sum overflow")
            })
        })?;
    if layers_outer_ns != timing.layer_outer_ns {
        return Err("Flash B1 layer wall ledger does not close".into());
    }
    let attributed_ns = layers_outer_ns
        .checked_add(timing.final_head_outer_ns)
        .ok_or("Flash B1 request phase timing overflow")?;
    if attributed_ns > request_wall_ns {
        return Err("Flash B1 request phases exceed request wall time".into());
    }
    Ok(serde_json::json!({
        "schema": "colibri-flash-0731-b1-phase-profile-v1",
        "clock": "monotonic_host_ns; existing stream synchronization boundaries only",
        "outer": {"wall_ns": request_wall_ns},
        "layers": &timing.layers,
        "final_heads": &timing.final_heads,
        "top_level_attributed_ns": attributed_ns,
        "unattributed_gap_ns": request_wall_ns - attributed_ns,
        "closure": {
            "equation": "top_level_attributed_ns + unattributed_gap_ns == outer.wall_ns",
            "exact": true,
        },
    }))
}

struct B2RawRuntime {
    raw: RawLoadedLayerRuntime,
    fixed_source_bytes: usize,
    fixed_h2d_bytes: usize,
    fixed: FixedLoadTelemetry,
}

fn flash_b1_prefetch_enabled(
    spec: &'static v4_spec::V4Spec,
    requests: usize,
) -> Result<bool, Box<dyn Error>> {
    // Keep the experimental worker confined to the standalone Flash B1 base
    // runner. Serve, DSpark, Pro, QD4, and every multi-row execution retain
    // their existing loading path.
    if spec != &v4_spec::FLASH_0731 || requests != 1 || !optional_one_flag("COLIBRI_V4_FLASH_BASE")?
    {
        return Ok(false);
    }
    optional_one_flag("COLIBRI_V4_FLASH_PREFETCH")
}

fn flash_b1_prefetch_next_layer(
    current_layer: usize,
    layer_count: usize,
) -> Result<Option<usize>, Box<dyn Error>> {
    if layer_count == 0 || current_layer >= layer_count {
        return Err("Flash B1 fixed prefetch layer is outside the model".into());
    }
    Ok((current_layer + 1 < layer_count).then_some(current_layer + 1))
}

fn flash_b1_prefetch_worker_result<T>(
    expected_layer: usize,
    scheduled_layer: usize,
    result: Result<T, String>,
) -> Result<T, Box<dyn Error>> {
    if scheduled_layer != expected_layer {
        return Err("Flash B1 fixed prefetch lost layer order".into());
    }
    result.map_err(|error| format!("Flash B1 fixed prefetch worker failed: {error}").into())
}

fn b2_raw_runtime_ledger(
    fixed_source_bytes: usize,
    fixed_h2d_without_wo_a_bytes: usize,
    raw_wo_a_h2d_bytes: usize,
) -> Result<(usize, usize), Box<dyn Error>> {
    Ok((
        fixed_source_bytes,
        fixed_h2d_without_wo_a_bytes
            .checked_add(raw_wo_a_h2d_bytes)
            .ok_or("B2 raw fixed H2D byte count overflow")?,
    ))
}

fn load_b2_raw_runtime(
    model_dir: &std::path::Path,
    layer: usize,
    stream: &CudaStream,
) -> Result<B2RawRuntime, Box<dyn Error>> {
    load_b2_raw_runtime_for(&v4_spec::PRO, model_dir, layer, stream)
}

fn load_b2_raw_runtime_for(
    spec: &'static v4_spec::V4Spec,
    model_dir: &std::path::Path,
    layer: usize,
    stream: &CudaStream,
) -> Result<B2RawRuntime, Box<dyn Error>> {
    // Request-specific router metadata belongs to the request ledger, not the
    // immutable layer runtime. `load_layer_state_weights` accepts this empty
    // row set only to build the hash-router mode without reading a request row.
    let (host, telemetry) = load_layer_host_inner_for(spec, model_dir, layer, &[], true)?;
    load_b2_raw_runtime_from_host_for(spec, model_dir, layer, host, telemetry, stream)
}

fn load_b2_raw_runtime_from_host_for(
    spec: &'static v4_spec::V4Spec,
    model_dir: &std::path::Path,
    layer: usize,
    mut host: DecodedLayerHost,
    telemetry: FixedLoadTelemetry,
    stream: &CudaStream,
) -> Result<B2RawRuntime, Box<dyn Error>> {
    let raw_wo_a_host = host
        .tail
        .wo_a_raw
        .take()
        .ok_or("B2 raw layer host omitted wo_a")?;
    if !host.tail.wo_a_bf16.is_empty() {
        return Err("B2 raw layer host retained decoded wo_a".into());
    }
    let h2d_started = std::time::Instant::now();
    let runtime = upload_layer_runtime_for(spec, model_dir, layer, &host, stream)?;
    let fixed_h2d_without_wo_a_bytes = runtime.weights.num_bytes();
    let wo_a_raw = DeviceBuffer::from_host(stream, &raw_wo_a_host)?;
    stream.synchronize()?;
    let raw_h2d_bytes = wo_a_raw.num_bytes();
    let fixed_h2d_ns = u64::try_from(h2d_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
    let (fixed_source_bytes, fixed_h2d_bytes) = b2_raw_runtime_ledger(
        telemetry.fixed_source_bytes,
        fixed_h2d_without_wo_a_bytes,
        raw_h2d_bytes,
    )?;
    let resident_bytes = runtime
        .weights
        .num_bytes()
        .checked_add(wo_a_raw.num_bytes())
        .ok_or("B2 raw device resident byte overflow")?;
    Ok(B2RawRuntime {
        fixed_source_bytes,
        fixed_h2d_bytes,
        fixed: FixedLoadTelemetry {
            fixed_read_alloc_ns: telemetry.fixed_read_alloc_ns,
            fixed_decode_ns: telemetry.fixed_decode_ns,
            fixed_h2d_ns,
            fixed_h2d_bytes,
            fixed_source_bytes,
            cache_status: "miss",
            ..FixedLoadTelemetry::default()
        },
        raw: RawLoadedLayerRuntime {
            runtime,
            wo_a_raw,
            resident_bytes,
        },
    })
}

fn run_b2_layer(
    context: &std::sync::Arc<CudaContext>,
    module: &kernels::LoadedModule,
    stream: &std::sync::Arc<CudaStream>,
    flash_cublas: Option<&FlashCublasSession>,
    runtime: &LoadedLayerRuntime,
    lanes: &mut [ExpertSlot<'_>],
    hash_ids: &[i64],
    hidden: &[u16],
    token_id: i64,
    token_position: usize,
    persistent: &mut v4_state::V4SerialPersistentState,
) -> Result<(Vec<u16>, B2LayerDigest, LayerStageTelemetry), Box<dyn Error>> {
    let layer = runtime.weights.layer;
    let spec = runtime.weights.spec;
    let mut result = None;
    persistent.transaction(layer, |state| {
        let run = run_real_layer_rows(
            context,
            module,
            stream,
            flash_cublas,
            runtime,
            (spec == &v4_spec::PRO).then_some(lanes),
            None,
            None,
            hash_ids,
            hidden,
            &[token_id],
            token_position,
            Some(state),
            None,
            None,
            false,
            None,
            false,
            None,
            None,
        )?;
        let main = state.compressor();
        let indexer = state.indexer();
        let digest = B2LayerDigest {
            layer,
            route_ids: run.routes.ids.clone(),
            route_weight_bits: run
                .routes
                .weights
                .iter()
                .map(|value| value.to_bits())
                .collect(),
            attention_topk_digest: run.attention_topk_digest,
            hidden_digest: fnv1a_u16(&run.output),
            kv_digest: fnv1a_u16(state.active_kv_bf16()),
            main_kv_state_digest: fnv1a_f32(main.kv()),
            main_score_state_digest: fnv1a_f32(main.score()),
            indexer_kv_state_digest: indexer.map(|value| fnv1a_f32(value.kv())),
            indexer_score_state_digest: indexer.map(|value| fnv1a_f32(value.score())),
        };
        result = Some((run.output, digest, run.stage));
        Ok(())
    })?;
    result.ok_or_else(|| "B2 layer transaction produced no result".into())
}

fn prepare_b2_prefix(
    context: &std::sync::Arc<CudaContext>,
    module: &kernels::LoadedModule,
    stream: &std::sync::Arc<CudaStream>,
    model_dir: &std::path::Path,
    token_id: i64,
    lanes: &mut [ExpertSlot<'_>],
) -> Result<Vec<u16>, Box<dyn Error>> {
    let mut hidden = prepare_b2_initial_hidden(model_dir, token_id)?;
    let mut persistent = v4_state::V4SerialPersistentState::new(1)?;
    for layer in 0..3 {
        let (runtime, _) = load_layer_runtime(model_dir, layer, &[token_id], stream)?;
        let hash_ids = load_layer_hash_ids(model_dir, layer, &[token_id])?;
        let mut next = None;
        persistent.transaction(layer, |state| {
            next = Some(
                run_real_layer_rows(
                    context,
                    module,
                    stream,
                    None,
                    &runtime,
                    Some(lanes),
                    None,
                    None,
                    &hash_ids,
                    &hidden,
                    &[token_id],
                    0,
                    Some(state),
                    None,
                    None,
                    false,
                    None,
                    false,
                    None,
                    None,
                )?
                .output,
            );
            Ok(())
        })?;
        hidden = next.ok_or("B2 prefix transaction produced no output")?;
    }
    Ok(hidden)
}

fn prepare_b2_initial_hidden(
    model_dir: &std::path::Path,
    token_id: i64,
) -> Result<Vec<u16>, Box<dyn Error>> {
    Ok(prepare_b2_initial_hidden_with_source_bytes(model_dir, token_id)?.0)
}

fn prepare_b2_initial_hidden_with_source_bytes(
    model_dir: &std::path::Path,
    token_id: i64,
) -> Result<(Vec<u16>, usize), Box<dyn Error>> {
    prepare_b2_initial_hidden_with_source_bytes_for(&v4_spec::PRO, model_dir, token_id)
}

fn prepare_b2_initial_hidden_with_source_bytes_for(
    spec: &'static v4_spec::V4Spec,
    model_dir: &std::path::Path,
    token_id: i64,
) -> Result<(Vec<u16>, usize), Box<dyn Error>> {
    let state = v4_state::load_for(spec, model_dir, &[token_id])?;
    let initial_state_source_bytes = state.range_bytes;
    let embedding = state.embedding;
    let mut hidden = Vec::with_capacity(v4_state::HC * spec.hidden);
    for _ in 0..v4_state::HC {
        hidden.extend_from_slice(&embedding);
    }
    Ok((hidden, initial_state_source_bytes))
}

fn prepare_two_token_prefix(
    context: &std::sync::Arc<CudaContext>,
    module: &kernels::LoadedModule,
    stream: &std::sync::Arc<CudaStream>,
    model_dir: &std::path::Path,
    token_ids: [i64; 2],
    lanes: &mut [ExpertSlot<'_>],
) -> Result<[Vec<u16>; 2], Box<dyn Error>> {
    let mut hidden = token_ids
        .map(|token_id| v4_state::load(model_dir, &[token_id]).map(|state| state.embedding))
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .map(|embedding| {
            let mut row = Vec::with_capacity(v4_state::HC * v4_state::HIDDEN);
            for _ in 0..v4_state::HC {
                row.extend_from_slice(&embedding);
            }
            row
        })
        .collect::<Vec<_>>();
    let mut persistent = v4_state::V4SerialPersistentState::new(2)?;
    for layer in 0..3 {
        let (runtime, _) = load_layer_runtime(model_dir, layer, &token_ids, stream)?;
        let hash_ids = load_layer_hash_ids(model_dir, layer, &token_ids)?;
        for position in 0..2 {
            let row_hash_ids = if runtime.weights.hash_router {
                &hash_ids[position * v4_router::V4_TOPK..(position + 1) * v4_router::V4_TOPK]
            } else {
                &[]
            };
            let mut next = None;
            persistent.transaction(layer, |state| {
                next = Some(
                    run_real_layer_rows(
                        context,
                        module,
                        stream,
                        None,
                        &runtime,
                        Some(lanes),
                        None,
                        None,
                        row_hash_ids,
                        &hidden[position],
                        &[token_ids[position]],
                        position,
                        Some(state),
                        None,
                        None,
                        false,
                        None,
                        false,
                        None,
                        None,
                    )?
                    .output,
                );
                Ok(())
            })?;
            hidden[position] = next.ok_or("two-token prefix produced no output")?;
        }
    }
    hidden
        .try_into()
        .map_err(|_| "two-token prefix changed row count".into())
}

struct MhcGpuOutput {
    collapsed: Vec<u16>,
    post: Vec<f32>,
    comb: Vec<f32>,
    debug: Option<MhcDebugOutput>,
    telemetry: MhcTelemetry,
}

struct MhcDebugOutput {
    mixes: Vec<f32>,
    pre: Vec<f32>,
    sinkhorn_trace: Vec<f32>,
    raw_dot: Option<Vec<f32>>,
    rsqrt: Option<Vec<f32>>,
    rsqrt_debug: Option<Vec<u32>>,
}

const FLASH_MHC_RSQRT_DEBUG_FIELDS: usize = 5;
const FLASH_MHC_RSQRT_COMMITTED_CANDIDATE: usize = 4;
const FLASH_MHC_RSQRT_COMMITTED_FIELD: &str = "libdevice_rsqrt_f32_bits";

fn flash_mhc_rsqrt_debug_fields() -> [&'static str; FLASH_MHC_RSQRT_DEBUG_FIELDS] {
    [
        "square_sum_f32_bits",
        "mean_square_plus_eps_f32_bits",
        "newton_candidate_f32_bits",
        "libdevice_sqrt_f32_bits",
        "libdevice_rsqrt_f32_bits",
    ]
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FlashMhcLinearArm {
    Canonical,
    Cublas,
}

impl FlashMhcLinearArm {
    const fn name(self) -> &'static str {
        match self {
            Self::Canonical => "canonical",
            Self::Cublas => "cublas",
        }
    }
}

fn flash_mhc_linear_arm_for(
    spec: &v4_spec::V4Spec,
    value: Option<&std::ffi::OsStr>,
) -> Result<FlashMhcLinearArm, Box<dyn Error>> {
    if spec != &v4_spec::FLASH_0731 {
        return Ok(FlashMhcLinearArm::Canonical);
    }
    match value {
        None => Ok(FlashMhcLinearArm::Canonical),
        Some(value) if value == "canonical" => Ok(FlashMhcLinearArm::Canonical),
        Some(value) if value == "cublas" => Ok(FlashMhcLinearArm::Cublas),
        Some(_) => {
            Err("COLIBRI_V4_FLASH_MHC_LINEAR_ARM must be canonical or cublas for Flash-0731".into())
        }
    }
}

fn flash_mhc_linear_arm(spec: &v4_spec::V4Spec) -> Result<FlashMhcLinearArm, Box<dyn Error>> {
    let value = std::env::var_os("COLIBRI_V4_FLASH_MHC_LINEAR_ARM");
    flash_mhc_linear_arm_for(spec, value.as_deref())
}

fn flash_router_linear_arm_for(
    spec: &v4_spec::V4Spec,
    value: Option<&std::ffi::OsStr>,
) -> Result<FlashMhcLinearArm, Box<dyn Error>> {
    if spec != &v4_spec::FLASH_0731 {
        return Ok(FlashMhcLinearArm::Canonical);
    }
    match value {
        None => Ok(FlashMhcLinearArm::Canonical),
        Some(value) if value == "canonical" => Ok(FlashMhcLinearArm::Canonical),
        Some(value) if value == "cublas" => Ok(FlashMhcLinearArm::Cublas),
        Some(_) => Err(
            "COLIBRI_V4_FLASH_ROUTER_LINEAR_ARM must be canonical or cublas for Flash-0731".into(),
        ),
    }
}

fn flash_router_linear_arm(spec: &v4_spec::V4Spec) -> Result<FlashMhcLinearArm, Box<dyn Error>> {
    let value = std::env::var_os("COLIBRI_V4_FLASH_ROUTER_LINEAR_ARM");
    flash_router_linear_arm_for(spec, value.as_deref())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FlashRouterScoreArm {
    Canonical,
    Libdevice,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FlashFp4AccumArm {
    Current,
    Block32,
    TilelangRef,
    Fp8Mma,
}

impl FlashFp4AccumArm {
    const fn name(self) -> &'static str {
        match self {
            Self::Current => "current",
            Self::Block32 => "block32",
            Self::TilelangRef => "tilelang_ref",
            Self::Fp8Mma => "fp8_mma",
        }
    }

    const fn uses_raw_fp8(self) -> bool {
        matches!(self, Self::Block32 | Self::TilelangRef | Self::Fp8Mma)
    }
}

fn flash_fp4_accum_arm_for(
    spec: &v4_spec::V4Spec,
    value: Option<&std::ffi::OsStr>,
) -> Result<FlashFp4AccumArm, Box<dyn Error>> {
    if spec != &v4_spec::FLASH_0731 {
        return Ok(FlashFp4AccumArm::Current);
    }
    match value {
        // sm_120 business baseline: keep the raw-FP8/block32 path as the
        // reproducible default.  The WMMA and cached TileLang paths remain
        // explicit experiments until they have stable target-GPU evidence.
        None => Ok(FlashFp4AccumArm::Block32),
        Some(value) if value == "current" => Ok(FlashFp4AccumArm::Current),
        Some(value) if value == "block32" => Ok(FlashFp4AccumArm::Block32),
        Some(value) if value == "tilelang_ref" => Ok(FlashFp4AccumArm::TilelangRef),
        Some(value) if value == "fp8_mma" => Ok(FlashFp4AccumArm::Fp8Mma),
        Some(_) => {
            Err("COLIBRI_V4_FLASH_FP4_ACCUM_ARM must be current, block32, tilelang_ref, or fp8_mma for Flash-0731".into())
        }
    }
}

fn flash_fp4_accum_arm(spec: &v4_spec::V4Spec) -> Result<FlashFp4AccumArm, Box<dyn Error>> {
    let value = std::env::var_os("COLIBRI_V4_FLASH_FP4_ACCUM_ARM");
    flash_fp4_accum_arm_for(spec, value.as_deref())
}

impl FlashRouterScoreArm {
    const fn name(self) -> &'static str {
        match self {
            Self::Canonical => "canonical",
            Self::Libdevice => "libdevice",
        }
    }

    const fn kernel_flag(self) -> u32 {
        match self {
            Self::Canonical => 0,
            Self::Libdevice => 1,
        }
    }
}

fn flash_router_score_arm_for(
    spec: &v4_spec::V4Spec,
    value: Option<&std::ffi::OsStr>,
) -> Result<FlashRouterScoreArm, Box<dyn Error>> {
    if spec != &v4_spec::FLASH_0731 {
        return Ok(FlashRouterScoreArm::Canonical);
    }
    match value {
        None => Ok(FlashRouterScoreArm::Canonical),
        Some(value) if value == "canonical" => Ok(FlashRouterScoreArm::Canonical),
        Some(value) if value == "libdevice" => Ok(FlashRouterScoreArm::Libdevice),
        Some(_) => Err(
            "COLIBRI_V4_FLASH_ROUTER_SCORE_ARM must be canonical or libdevice for Flash-0731"
                .into(),
        ),
    }
}

fn flash_router_score_arm(spec: &v4_spec::V4Spec) -> Result<FlashRouterScoreArm, Box<dyn Error>> {
    let value = std::env::var_os("COLIBRI_V4_FLASH_ROUTER_SCORE_ARM");
    flash_router_score_arm_for(spec, value.as_deref())
}

fn flash_router_score_provenance(arm: FlashRouterScoreArm) -> serde_json::Value {
    match arm {
        FlashRouterScoreArm::Canonical => serde_json::json!({
            "arm": arm.name(),
        }),
        FlashRouterScoreArm::Libdevice => serde_json::json!({
            "arm": arm.name(),
            "softplus": "x > 20 ? x : __nv_log1pf(__nv_expf(x))",
            "sqrt": "__nv_sqrtf",
            "device_symbol_execution": "requires GPU parity gate",
        }),
    }
}

#[cfg(test)]
fn flash_router_libdevice_score_host_reference(logit: f32) -> f32 {
    let softplus = if logit > 20.0 {
        logit
    } else {
        libm::log1pf(libm::expf(logit))
    };
    libm::sqrtf(softplus)
}

struct FlashCublasSession {
    cublas: cublas_exact::CublasExact,
    workspace: DeviceBuffer<u8>,
}

impl FlashCublasSession {
    fn new(stream: &CudaStream) -> Result<Self, Box<dyn Error>> {
        Ok(Self {
            // Keep the verified object mapped for the full request.  Reopening it
            // cannot improve numerical behavior and used to repeat a 105 MiB SHA.
            cublas: cublas_exact::CublasExact::load()?,
            workspace: DeviceBuffer::zeroed(stream, cublas_exact::WORKSPACE_BYTES)?,
        })
    }
}

fn flash_mhc_cublas_provenance(
    cublas: &cublas_exact::CublasExact,
) -> Result<serde_json::Value, Box<dyn Error>> {
    let mut provenance = flash_cublas_sgemm_provenance(
        cublas,
        v4_state::MIX,
        v4_state::HC * v4_spec::FLASH_0731.hidden,
    )?;
    provenance
        .as_object_mut()
        .ok_or("Flash cuBLAS provenance must be an object")?
        .insert(
            "rsqrt".into(),
            serde_json::json!({
                "actual": "direct __nv_rsqrtf(mean_square_plus_eps)",
                "actual_instruction": "shared device helper calling libdevice __nv_rsqrtf; this is debug candidate 4",
                "committed_debug_candidate": FLASH_MHC_RSQRT_COMMITTED_FIELD,
                "debug_candidates": flash_mhc_rsqrt_debug_fields(),
                "debug_candidate_notes": {
                    "libdevice_sqrt_f32_bits": "1.0 / core::f32::math::sqrt(x), lowered by cuda-oxide to libdevice __nv_sqrtf",
                    "libdevice_rsqrt_f32_bits": "direct #[device] extern C call to libdevice __nv_rsqrtf(x), also the committed Flash cuBLAS-arm normalization",
                },
            }),
        );
    Ok(provenance)
}

fn flash_router_cublas_provenance(
    cublas: &cublas_exact::CublasExact,
) -> Result<serde_json::Value, Box<dyn Error>> {
    flash_cublas_sgemm_provenance(
        cublas,
        v4_spec::FLASH_0731.routed_experts,
        v4_spec::FLASH_0731.hidden,
    )
}

fn flash_cublas_sgemm_provenance(
    cublas: &cublas_exact::CublasExact,
    m: usize,
    k: usize,
) -> Result<serde_json::Value, Box<dyn Error>> {
    let provenance = cublas.provenance();
    Ok(serde_json::json!({
        "abi": "cublasSgemm_v2",
        "resolved_path": provenance.resolved_path,
        "dladdr_path": provenance.dladdr_path,
        "maps_path": provenance.maps_path,
        "sha256": provenance.actual_sha256,
        "version": provenance.version,
        "math_mode": "CUBLAS_DEFAULT_MATH",
        "tf32": "off",
        "trans_a": "T",
        "trans_b": "N",
        "m": m,
        "n": 1,
        "k": k,
        "alpha": 1.0_f32,
        "beta": 0.0_f32,
        "workspace_bytes": cublas_exact::WORKSPACE_BYTES,
    }))
}

struct MhcCublasBuffers {
    _input_f32: DeviceBuffer<f32>,
    _squares: DeviceBuffer<f32>,
    raw_dot: DeviceBuffer<f32>,
    rsqrt: DeviceBuffer<f32>,
    rsqrt_debug: Option<DeviceBuffer<u32>>,
}

#[allow(clippy::too_many_arguments)]
fn run_flash_mhc_cublas_linear_device_into(
    session: &FlashCublasSession,
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    input: &DeviceBuffer<u16>,
    hc_fn: &DeviceBuffer<f32>,
    mixes: &mut DeviceBuffer<f32>,
    rows: usize,
    hidden: usize,
    capture_rsqrt_debug: bool,
) -> Result<MhcCublasBuffers, Box<dyn Error>> {
    let k = v4_state::HC
        .checked_mul(hidden)
        .ok_or("Flash mHC cuBLAS K overflow")?;
    if hidden != v4_spec::FLASH_0731.hidden
        || k != 16_384
        || rows == 0
        || input.len() != rows * k
        || hc_fn.len() != v4_state::MIX * k
        || mixes.len() != rows * v4_state::MIX
    {
        return Err("Flash mHC cuBLAS SGEMM geometry must be rows x 24 x 16384".into());
    }
    let mut input_f32 = DeviceBuffer::<f32>::zeroed(stream, input.len())?;
    let mut squares = DeviceBuffer::<f32>::zeroed(stream, input.len())?;
    let raw_dot = DeviceBuffer::<f32>::zeroed(stream, mixes.len())?;
    let mut rsqrt = DeviceBuffer::<f32>::zeroed(stream, rows)?;
    let mut rsqrt_debug = capture_rsqrt_debug
        .then(|| DeviceBuffer::<u32>::zeroed(stream, rows * FLASH_MHC_RSQRT_DEBUG_FIELDS))
        .transpose()?;
    launch_v4_bf16_to_f32(module, stream, input, &mut input_f32)?;
    let device_const = |pointer: u64| pointer as usize as *const std::ffi::c_void;
    let device_mut = |pointer: u64| pointer as usize as *mut std::ffi::c_void;
    for row in 0..rows {
        let input_offset = row
            .checked_mul(k)
            .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
            .ok_or("Flash mHC cuBLAS input offset overflow")?;
        let output_offset = row
            .checked_mul(v4_state::MIX)
            .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
            .ok_or("Flash mHC cuBLAS output offset overflow")?;
        session.cublas.sgemm_mn1(
            stream.cu_stream().cast::<std::ffi::c_void>(),
            device_mut(session.workspace.cu_deviceptr()),
            session.workspace.num_bytes(),
            device_const(hc_fn.cu_deviceptr()),
            device_const(
                input_f32
                    .cu_deviceptr()
                    .checked_add(u64::try_from(input_offset)?)
                    .ok_or("Flash mHC cuBLAS input pointer overflow")?,
            ),
            device_mut(
                raw_dot
                    .cu_deviceptr()
                    .checked_add(u64::try_from(output_offset)?)
                    .ok_or("Flash mHC cuBLAS output pointer overflow")?,
            ),
            v4_state::MIX,
            k,
        )?;
    }
    let mut empty_debug = DeviceBuffer::<u32>::zeroed(stream, 0)?;
    let debug = rsqrt_debug.as_mut().unwrap_or(&mut empty_debug);
    // SAFETY: the square launch owns each input lane, the fixed reduction owns
    // one complete 16,384-wide row, and the final scale owns 24 disjoint dots.
    unsafe {
        module.rmsnorm_flash_square_bf16(
            stream,
            LaunchConfig::for_num_elems(u32::try_from(input.len())?),
            input,
            &mut squares,
        )?;
        module.mhc_flash_reduce_rsqrt_16384(
            stream,
            LaunchConfig {
                grid_dim: (u32::try_from(rows)?, 1, 1),
                block_dim: (512, 1, 1),
                shared_mem_bytes: 0,
            },
            &squares,
            &mut rsqrt,
            debug,
        )?;
        module.mhc_scale_raw_dot_by_rsqrt_v4(
            stream,
            LaunchConfig::for_num_elems(u32::try_from(rows)?),
            &raw_dot,
            &rsqrt,
            mixes,
        )?;
    }
    Ok(MhcCublasBuffers {
        _input_f32: input_f32,
        _squares: squares,
        raw_dot,
        rsqrt,
        rsqrt_debug,
    })
}

struct RouterCublasBuffers {
    _input_f32: DeviceBuffer<f32>,
    _gate_f32: DeviceBuffer<f32>,
}

fn run_flash_router_cublas_linear_device_into(
    session: &FlashCublasSession,
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    input: &DeviceBuffer<u16>,
    gate: &DeviceBuffer<u16>,
    logits: &mut DeviceBuffer<f32>,
    rows: usize,
    hidden: usize,
    experts: usize,
) -> Result<RouterCublasBuffers, Box<dyn Error>> {
    if hidden != v4_spec::FLASH_0731.hidden
        || experts != v4_spec::FLASH_0731.routed_experts
        || rows == 0
        || input.len() != rows * hidden
        || gate.len() != experts * hidden
        || logits.len() != rows * experts
    {
        return Err("Flash router cuBLAS SGEMM geometry must be rows x 256 x 4096".into());
    }
    let mut input_f32 = DeviceBuffer::<f32>::zeroed(stream, input.len())?;
    let mut gate_f32 = DeviceBuffer::<f32>::zeroed(stream, gate.len())?;
    // ponytail: converts the 256x4096 gate per router call; merge into an
    // existing fixed-weight cache only after this arm proves parity and a
    // profile attributes material time to conversion.
    launch_v4_bf16_to_f32(module, stream, input, &mut input_f32)?;
    launch_v4_bf16_to_f32(module, stream, gate, &mut gate_f32)?;
    let device_const = |pointer: u64| pointer as usize as *const std::ffi::c_void;
    let device_mut = |pointer: u64| pointer as usize as *mut std::ffi::c_void;
    for row in 0..rows {
        let input_offset = row
            .checked_mul(hidden)
            .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
            .ok_or("Flash router cuBLAS input offset overflow")?;
        let output_offset = row
            .checked_mul(experts)
            .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
            .ok_or("Flash router cuBLAS output offset overflow")?;
        session.cublas.sgemm_mn1(
            stream.cu_stream().cast::<std::ffi::c_void>(),
            device_mut(session.workspace.cu_deviceptr()),
            session.workspace.num_bytes(),
            device_const(gate_f32.cu_deviceptr()),
            device_const(
                input_f32
                    .cu_deviceptr()
                    .checked_add(u64::try_from(input_offset)?)
                    .ok_or("Flash router cuBLAS input pointer overflow")?,
            ),
            device_mut(
                logits
                    .cu_deviceptr()
                    .checked_add(u64::try_from(output_offset)?)
                    .ok_or("Flash router cuBLAS output pointer overflow")?,
            ),
            experts,
            hidden,
        )?;
    }
    Ok(RouterCublasBuffers {
        _input_f32: input_f32,
        _gate_f32: gate_f32,
    })
}

#[derive(Clone, Default)]
struct AttentionProfileStep {
    sync_wall_ns: u64,
    gpu_event_ns: u64,
}

#[derive(Clone, Default)]
struct AttentionProfile {
    mhc: AttentionProfileStep,
    attn_norm: AttentionProfileStep,
    q_a_projection: AttentionProfileStep,
    q_a_norm: AttentionProfileStep,
    q_b_projection: AttentionProfileStep,
    q_b_norm: AttentionProfileStep,
    q_rope: AttentionProfileStep,
    kv_projection: AttentionProfileStep,
    kv_norm: AttentionProfileStep,
    kv_rope_quantize: AttentionProfileStep,
    compressor: AttentionProfileStep,
    indexer: AttentionProfileStep,
    commit_topk: AttentionProfileStep,
    sparse_score: AttentionProfileStep,
    sparse_inverse_rope: AttentionProfileStep,
    wo_a: AttentionProfileStep,
    wo_b: AttentionProfileStep,
    hc_post: AttentionProfileStep,
}

impl AttentionProfile {
    fn step_json(step: &AttentionProfileStep) -> serde_json::Value {
        serde_json::json!({"sync_wall_ns": step.sync_wall_ns, "gpu_event_ns": step.gpu_event_ns})
    }

    fn json(&self) -> serde_json::Value {
        serde_json::json!({
            "mhc": Self::step_json(&self.mhc),
            "attn_norm": Self::step_json(&self.attn_norm),
            "q_a_projection": Self::step_json(&self.q_a_projection),
            "q_a_norm": Self::step_json(&self.q_a_norm),
            "q_b_projection": Self::step_json(&self.q_b_projection),
            "q_b_norm": Self::step_json(&self.q_b_norm),
            "q_rope": Self::step_json(&self.q_rope),
            "kv_projection": Self::step_json(&self.kv_projection),
            "kv_norm": Self::step_json(&self.kv_norm),
            "kv_rope_quantize": Self::step_json(&self.kv_rope_quantize),
            "compressor": Self::step_json(&self.compressor),
            "indexer": Self::step_json(&self.indexer),
            "commit_topk": Self::step_json(&self.commit_topk),
            "sparse_score": Self::step_json(&self.sparse_score),
            "sparse_inverse_rope": Self::step_json(&self.sparse_inverse_rope),
            "wo_a": Self::step_json(&self.wo_a),
            "wo_b": Self::step_json(&self.wo_b),
            "hc_post": Self::step_json(&self.hc_post),
        })
    }
}

#[derive(Default)]
struct LayerStageTelemetry {
    layer_prepare_ns: u64,
    layer_prepare_count: usize,
    attention_ns: u64,
    device_state_setup_ns: u64,
    device_state_setup_count: usize,
    device_state_reset_cuda_ns: u64,
    device_state_reset_calls: usize,
    attention_audit_oracle_ns: u64,
    attention_hidden_h2d_count: usize,
    attention_hidden_h2d_bytes: usize,
    attention_rope_lut_d2d_count: usize,
    attention_rope_lut_d2d_bytes: usize,
    attention_terminal_d2h_count: usize,
    attention_terminal_d2h_bytes: usize,
    attention_intermediate_or_audit_d2h_count: usize,
    attention_intermediate_or_audit_d2h_bytes: usize,
    attention_steady_device_temporary_alloc_count: usize,
    attention_profile: Option<AttentionProfile>,
    router_ns: u64,
    router_mhc_ns: u64,
    router_mhc_input_alloc_h2d_sync_ns: u64,
    router_mhc_workspace_alloc_sync_ns: u64,
    router_mhc_workspace_init_gpu_ns: u64,
    router_mhc_mixing_linear_sync_ns: u64,
    router_mhc_mixing_linear_gpu_ns: u64,
    router_mhc_sinkhorn_normalize_collapse_sync_ns: u64,
    router_mhc_sinkhorn_normalize_collapse_gpu_ns: u64,
    router_mhc_output_d2h_sync_ns: u64,
    router_mhc_other_ns: u64,
    router_host_prep_ns: u64,
    router_alloc_h2d_sync_ns: u64,
    router_workspace_alloc_ns: u64,
    router_kernel_select_sync_ns: u64,
    router_kernel_select_gpu_ns: u64,
    router_d2h_sync_ns: u64,
    router_other_ns: u64,
    shared_ns: u64,
    routed_ns: u64,
    routed_prefetched: bool,
    routed_read_ns: u64,
    routed_alloc_ns: u64,
    routed_setup_ns: u64,
    routed_h2d_ns: u64,
    routed_kernel_ns: u64,
    commit_ns: u64,
    routed_source_bytes: usize,
    routed_h2d_bytes: usize,
    routed_expert_loads: usize,
    routed_parallel_batch_wall_ns: Option<u64>,
    routed_parallel_read_horizon_ns: Option<u64>,
}

impl LayerStageTelemetry {
    fn router_accounted_ns(&self) -> u64 {
        self.router_mhc_ns
            .saturating_add(self.router_host_prep_ns)
            .saturating_add(self.router_alloc_h2d_sync_ns)
            .saturating_add(self.router_workspace_alloc_ns)
            .saturating_add(self.router_kernel_select_sync_ns)
            .saturating_add(self.router_d2h_sync_ns)
            .saturating_add(self.router_other_ns)
    }
}

fn fixed_prepare_dual_accounting(
    fixed: &FixedLoadTelemetry,
    main_exposed_prepare_ns: u64,
) -> Result<serde_json::Value, Box<dyn Error>> {
    let service_work_sum_ns = fixed
        .fixed_read_alloc_ns
        .checked_add(fixed.fixed_h2d_ns)
        .and_then(|value| value.checked_add(fixed.fixed_decode_ns))
        .ok_or("fixed prepare service-work sum overflow")?;
    if fixed.raw_source_bytes > fixed.fixed_source_bytes
        || fixed.raw_h2d_bytes > fixed.fixed_h2d_bytes
        || fixed.raw_h2d_ns > fixed.fixed_h2d_ns
    {
        return Err(format!(
            "fixed prepare nested byte/service accounting is invalid: raw_source_bytes={} source_bytes={} raw_h2d_bytes={} h2d_bytes={} raw_h2d_ns={} h2d_ns={}",
            fixed.raw_source_bytes,
            fixed.fixed_source_bytes,
            fixed.raw_h2d_bytes,
            fixed.fixed_h2d_bytes,
            fixed.raw_h2d_ns,
            fixed.fixed_h2d_ns,
        )
        .into());
    }
    let (origin, device_hit_count, host_hit_count, source_load_count) = match fixed.cache_status {
        "stationary_pinned_device_hit" | "stationary_raw_device_hit" => {
            ("device_hit", 1_usize, 0_usize, 0_usize)
        }
        "stationary_pinned_host_hit" | "stationary_raw_host_hit" => ("host_hit", 0, 1, 0),
        "stationary_pinned_miss" | "stationary_raw_miss" | "prefetched_blocking_h2d" | "miss" => {
            ("source_load", 0, 0, 1)
        }
        status => {
            return Err(format!("fixed prepare has unknown origin status {status}").into());
        }
    };
    let origin_count = device_hit_count + host_hit_count + source_load_count;
    let origin_bytes_valid = match origin {
        "device_hit" => fixed.fixed_source_bytes == 0 && fixed.fixed_h2d_bytes == 0,
        "host_hit" => fixed.fixed_source_bytes == 0 && fixed.fixed_h2d_bytes > 0,
        "source_load" => fixed.fixed_source_bytes > 0 && fixed.fixed_h2d_bytes > 0,
        _ => unreachable!(),
    };
    if origin_count != 1 || !origin_bytes_valid {
        return Err(format!(
            "fixed prepare origin/byte accounting is invalid: status={} device_hit_count={} host_hit_count={} source_load_count={} source_bytes={} h2d_bytes={}",
            fixed.cache_status,
            device_hit_count,
            host_hit_count,
            source_load_count,
            fixed.fixed_source_bytes,
            fixed.fixed_h2d_bytes,
        )
        .into());
    }
    let exposed_service_bound_ns = service_work_sum_ns.min(main_exposed_prepare_ns);
    let derived_overlap_credit_ns = service_work_sum_ns - exposed_service_bound_ns;
    let queue_or_overhead_ns = main_exposed_prepare_ns - exposed_service_bound_ns;
    if service_work_sum_ns
        != exposed_service_bound_ns
            .checked_add(derived_overlap_credit_ns)
            .ok_or("fixed prepare service dual-ledger overflow")?
        || main_exposed_prepare_ns
            != exposed_service_bound_ns
                .checked_add(queue_or_overhead_ns)
                .ok_or("fixed prepare exposed dual-ledger overflow")?
    {
        return Err(format!(
            "fixed prepare dual-ledger closure is invalid: service_work_sum_ns={} main_exposed_prepare_ns={} exposed_service_bound_ns={} derived_overlap_credit_ns={} queue_or_overhead_ns={}",
            service_work_sum_ns,
            main_exposed_prepare_ns,
            exposed_service_bound_ns,
            derived_overlap_credit_ns,
            queue_or_overhead_ns,
        )
        .into());
    }
    Ok(serde_json::json!({
        "cache_status": fixed.cache_status,
        "origin": {
            "class": origin,
            "device_hit_count": device_hit_count,
            "host_hit_count": host_hit_count,
            "source_load_count": source_load_count,
            "count_closure_exact": origin_count == 1,
        },
        "service": {
            "work_sum_ns": service_work_sum_ns,
            "read_alloc_ns": fixed.fixed_read_alloc_ns,
            "h2d_ns": fixed.fixed_h2d_ns,
            "decode_ns": fixed.fixed_decode_ns,
            "internal_closure": "work_sum_ns == read_alloc_ns + h2d_ns + decode_ns",
        },
        "critical": {
            "main_exposed_prepare_ns": main_exposed_prepare_ns,
            "exposed_service_bound_ns": exposed_service_bound_ns,
            "derived_overlap_credit_ns": derived_overlap_credit_ns,
            "queue_or_overhead_ns": queue_or_overhead_ns,
            "service_closure": "work_sum_ns == exposed_service_bound_ns + derived_overlap_credit_ns",
            "main_closure": "main_exposed_prepare_ns == exposed_service_bound_ns + queue_or_overhead_ns",
            "interpretation": "dual-ledger algebra; derived_overlap_credit_ns is not a directly measured causal overlap",
        },
        "bytes": {
            "source_bytes": fixed.fixed_source_bytes,
            "h2d_bytes": fixed.fixed_h2d_bytes,
            "raw_source_bytes": fixed.raw_source_bytes,
            "raw_h2d_bytes": fixed.raw_h2d_bytes,
        },
        "nested": {
            "raw_read_ns": fixed.raw_read_ns,
            "raw_h2d_ns": fixed.raw_h2d_ns,
        },
    }))
}

fn stationary_layer_phase_profile_json(
    request_start_ns: u64,
    request_end_ns: u64,
    fixed: Option<&FixedLoadTelemetry>,
    stage: &LayerStageTelemetry,
    indexer_topk_gpu_ns: u64,
    routed_parallel: bool,
) -> Result<(serde_json::Value, u64, u64), Box<dyn Error>> {
    if request_end_ns < request_start_ns {
        return Err("stationary phase profile layer span is not monotonic".into());
    }
    let expected_prepare_count = usize::from(fixed.is_some());
    if stage.layer_prepare_count != expected_prepare_count
        || (stage.layer_prepare_count == 0 && stage.layer_prepare_ns != 0)
    {
        return Err(format!(
            "stationary layer-prepare count accounting is invalid: count={} expected_count={} main_exposed_prepare_ns={}",
            stage.layer_prepare_count, expected_prepare_count, stage.layer_prepare_ns
        )
        .into());
    }
    let attention_wall_ns = stage
        .device_state_setup_ns
        .saturating_add(stage.attention_ns)
        .saturating_add(stage.commit_ns)
        .saturating_add(stage.attention_audit_oracle_ns);
    let top_level_attributed_ns = stage
        .layer_prepare_ns
        .saturating_add(attention_wall_ns)
        .saturating_add(stage.router_ns)
        .saturating_add(stage.shared_ns)
        .saturating_add(stage.routed_ns);
    let outer_wall_ns = request_end_ns - request_start_ns;
    if top_level_attributed_ns > outer_wall_ns {
        return Err("stationary phase profile top-level scopes exceed layer outer span".into());
    }
    let routed_wall_diagnostics = [
        stage.routed_read_ns,
        stage.routed_alloc_ns,
        stage.routed_setup_ns,
    ];
    if routed_wall_diagnostics
        .iter()
        .any(|&diagnostic| diagnostic > stage.routed_ns)
    {
        return Err("stationary routed diagnostic exceeds its opaque parent scope".into());
    }
    if !routed_parallel
        && (stage.routed_h2d_ns > stage.routed_ns || stage.routed_kernel_ns > stage.routed_ns)
    {
        return Err("stationary routed diagnostic exceeds its opaque parent scope".into());
    }
    if stage.router_mhc_ns > stage.router_ns
        || stage.router_host_prep_ns > stage.router_ns
        || stage.router_alloc_h2d_sync_ns > stage.router_ns
        || stage.router_workspace_alloc_ns > stage.router_ns
        || stage.router_kernel_select_sync_ns > stage.router_ns
        || stage.router_d2h_sync_ns > stage.router_ns
        || stage.router_other_ns > stage.router_ns
        || indexer_topk_gpu_ns > attention_wall_ns
    {
        return Err("stationary nested diagnostic exceeds its parent scope".into());
    }
    let fixed_load = fixed
        .map(|fixed| fixed_prepare_dual_accounting(fixed, stage.layer_prepare_ns))
        .transpose()?;
    let layer_prepare = (stage.layer_prepare_count == 1).then(|| {
        serde_json::json!({
            "wall_ns": stage.layer_prepare_ns,
            "count": stage.layer_prepare_count,
            "scope": "main-thread current-layer consume/load through fixed device runtime and hash IDs ready; prefetch worker work is not additive, while any consume wait is included",
            "fixed_load": fixed_load,
        })
    });
    let routed = if routed_parallel {
        let batch_wall_ns = stage
            .routed_parallel_batch_wall_ns
            .ok_or("parallel routed profile is missing batch wall time")?;
        let read_completion_horizon_ns = stage
            .routed_parallel_read_horizon_ns
            .ok_or("parallel routed profile is missing read completion horizon")?;
        if stage.routed_read_ns != read_completion_horizon_ns
            || read_completion_horizon_ns > batch_wall_ns
            || batch_wall_ns > stage.routed_ns
        {
            return Err("parallel routed wall accounting is invalid".into());
        }
        serde_json::json!({
            "wall_ns": stage.routed_ns,
            "parallel_lanes": {
                "lane_count": V4_QD4_LANES,
                "route_job_count": v4_router::V4_TOPK,
                "batch_wall_ns": batch_wall_ns,
                "read_completion_horizon_ns": read_completion_horizon_ns,
                "h2d_event_work_sum_ns": stage.routed_h2d_ns,
                "kernel_event_work_sum_ns": stage.routed_kernel_ns,
            },
            "diagnostics": {
                "alloc_ns": stage.routed_alloc_ns,
                "setup_ns": stage.routed_setup_ns,
                "source_bytes": stage.routed_source_bytes,
                "h2d_bytes": stage.routed_h2d_bytes,
            },
        })
    } else {
        serde_json::json!({
            "wall_ns": stage.routed_ns,
            "diagnostics": {
                "read_ns": stage.routed_read_ns,
                "alloc_ns": stage.routed_alloc_ns,
                "setup_ns": stage.routed_setup_ns,
                "h2d_ns": stage.routed_h2d_ns,
                "kernel_ns": stage.routed_kernel_ns,
                "source_bytes": stage.routed_source_bytes,
                "h2d_bytes": stage.routed_h2d_bytes,
            },
        })
    };
    Ok((
        serde_json::json!({
            "schema": "colibri-v4-stationary-phase-profile-v5",
            "clock": "monotonic_host_ns; nested CUDA event durations are diagnostics only",
            "outer": {
                "request_start_ns": request_start_ns,
                "request_end_ns": request_end_ns,
                "wall_ns": outer_wall_ns,
            },
            "top_level": {
                "layer_prepare": layer_prepare,
                "attention": {
                    "wall_ns": attention_wall_ns,
                    "execution_ns": stage.attention_ns,
                    "state_commit_ns": stage.commit_ns,
                    "audit_oracle_ns": stage.attention_audit_oracle_ns,
                    "device_state_setup": {
                        "wall_ns": stage.device_state_setup_ns,
                        "count": stage.device_state_setup_count,
                        "state_reset": {
                            "cuda_event_ns": stage.device_state_reset_cuda_ns,
                            "calls": stage.device_state_reset_calls,
                        },
                    },
                    "transfers": {
                        "hidden_h2d": {
                            "count": stage.attention_hidden_h2d_count,
                            "bytes": stage.attention_hidden_h2d_bytes,
                        },
                        "request_rope_lut_d2d": {
                            "scope": "only request-global RoPE LUT row copies; internal attention/cache D2D is not claimed",
                            "count": stage.attention_rope_lut_d2d_count,
                            "bytes": stage.attention_rope_lut_d2d_bytes,
                        },
                        "terminal_after_attention_d2h": {
                            "count": stage.attention_terminal_d2h_count,
                            "bytes": stage.attention_terminal_d2h_bytes,
                        },
                        "intermediate_or_audit_d2h": {
                            "count": stage.attention_intermediate_or_audit_d2h_count,
                            "bytes": stage.attention_intermediate_or_audit_d2h_bytes,
                        },
                    },
                    "temporary": {
                        "steady_device_alloc_count": stage.attention_steady_device_temporary_alloc_count,
                    },
                    "diagnostics": {"indexer_topk_gpu_ns": indexer_topk_gpu_ns},
                },
                "router": {
                    "wall_ns": stage.router_ns,
                    "diagnostics": {
                        "mhc_wall_ns": stage.router_mhc_ns,
                        "mhc_input_alloc_h2d_sync_ns": stage.router_mhc_input_alloc_h2d_sync_ns,
                        "mhc_workspace_alloc_sync_ns": stage.router_mhc_workspace_alloc_sync_ns,
                        "mhc_workspace_init_gpu_ns": stage.router_mhc_workspace_init_gpu_ns,
                        "mhc_mixing_linear_sync_ns": stage.router_mhc_mixing_linear_sync_ns,
                        "mhc_mixing_linear_gpu_ns": stage.router_mhc_mixing_linear_gpu_ns,
                        "mhc_sinkhorn_normalize_collapse_sync_ns": stage.router_mhc_sinkhorn_normalize_collapse_sync_ns,
                        "mhc_sinkhorn_normalize_collapse_gpu_ns": stage.router_mhc_sinkhorn_normalize_collapse_gpu_ns,
                        "mhc_output_d2h_sync_ns": stage.router_mhc_output_d2h_sync_ns,
                        "mhc_other_ns": stage.router_mhc_other_ns,
                        "host_prep_ns": stage.router_host_prep_ns,
                        "alloc_h2d_sync_ns": stage.router_alloc_h2d_sync_ns,
                        "workspace_alloc_ns": stage.router_workspace_alloc_ns,
                        "kernel_select_sync_ns": stage.router_kernel_select_sync_ns,
                        "kernel_select_gpu_ns": stage.router_kernel_select_gpu_ns,
                        "d2h_sync_ns": stage.router_d2h_sync_ns,
                        "other_ns": stage.router_other_ns,
                    },
                },
                "shared": {"wall_ns": stage.shared_ns},
                "routed": routed,
            },
            "top_level_attributed_ns": top_level_attributed_ns,
            "unattributed_gap_ns": outer_wall_ns - top_level_attributed_ns,
            "closure": {
                "equation": "top_level_attributed_ns + unattributed_gap_ns == outer.wall_ns",
                "exact": true,
            },
        }),
        top_level_attributed_ns,
        outer_wall_ns,
    ))
}

fn stationary_phase_profile_outer_accounting(
    layer_outer_ns: u64,
    final_outer_ns: u64,
    request_e2e_ns: u64,
) -> Result<(u64, u64), Box<dyn Error>> {
    let profiled_outer_ns = layer_outer_ns.saturating_add(final_outer_ns);
    if profiled_outer_ns > request_e2e_ns {
        return Err("stationary phase profile outer scopes exceed request E2E".into());
    }
    Ok((profiled_outer_ns, request_e2e_ns - profiled_outer_ns))
}

fn stationary_one_token_phase_profile_outer_accounting(
    prefill_layer_outer_ns: u64,
    prefill_final_outer_ns: u64,
    decode_layer_outer_ns: u64,
    decode_final_outer_ns: u64,
    request_e2e_ns: u64,
) -> Result<(u64, u64, u64, u64), Box<dyn Error>> {
    let prefill_outer_ns = prefill_layer_outer_ns
        .checked_add(prefill_final_outer_ns)
        .ok_or("one-token phase profile prefill outer overflow")?;
    let decode_outer_ns = decode_layer_outer_ns
        .checked_add(decode_final_outer_ns)
        .ok_or("one-token phase profile decode outer overflow")?;
    let profiled_outer_ns = prefill_outer_ns
        .checked_add(decode_outer_ns)
        .ok_or("one-token phase profile request outer overflow")?;
    if profiled_outer_ns > request_e2e_ns {
        return Err("one-token phase profile outer scopes exceed request E2E".into());
    }
    Ok((
        prefill_outer_ns,
        decode_outer_ns,
        profiled_outer_ns,
        request_e2e_ns - profiled_outer_ns,
    ))
}

fn stationary_final_phase_profile_json(
    request_start_ns: u64,
    request_end_ns: u64,
    fixed_read_ns: u64,
    cpu_wall_ns: u64,
    cpu_streamed_weight_read_ns: u64,
    routed_parallel: bool,
) -> Result<(serde_json::Value, u64), Box<dyn Error>> {
    if request_end_ns < request_start_ns {
        return Err("stationary phase profile final span is not monotonic".into());
    }
    let outer_wall_ns = request_end_ns - request_start_ns;
    let top_level_attributed_ns = fixed_read_ns.saturating_add(cpu_wall_ns);
    if top_level_attributed_ns > outer_wall_ns || cpu_streamed_weight_read_ns > cpu_wall_ns {
        return Err("stationary phase profile final accounting is invalid".into());
    }
    Ok((
        serde_json::json!({
            "schema": "colibri-v4-stationary-phase-profile-v3",
            "clock": "monotonic_host_ns; streamed weight read is nested in cpu wall",
            "routed_parallel": routed_parallel,
            "outer": {
                "request_start_ns": request_start_ns,
                "request_end_ns": request_end_ns,
                "wall_ns": outer_wall_ns,
            },
            "top_level": {
                "fixed": {"wall_ns": fixed_read_ns},
                "cpu_final": {
                    "wall_ns": cpu_wall_ns,
                    "diagnostics": {"streamed_weight_read_ns": cpu_streamed_weight_read_ns},
                },
            },
            "top_level_attributed_ns": top_level_attributed_ns,
            "unattributed_gap_ns": outer_wall_ns - top_level_attributed_ns,
            "closure": {
                "equation": "top_level_attributed_ns + unattributed_gap_ns == outer.wall_ns",
                "exact": true,
            },
        }),
        top_level_attributed_ns,
    ))
}

#[derive(Default)]
struct RouterTelemetry {
    alloc_h2d_sync_ns: u64,
    workspace_alloc_ns: u64,
    kernel_select_sync_ns: u64,
    kernel_select_gpu_ns: u64,
    d2h_sync_ns: u64,
}

struct LayerRouterRun {
    input: Vec<u16>,
    routes: v4_router::RouterSelection,
    debug: Option<RouterDebug>,
    telemetry: RouterTelemetry,
}

struct RouterDebug {
    rmsnorm: Vec<f32>,
    rmsnorm_squares: Vec<f32>,
    logits: Vec<f32>,
    softplus: Vec<f32>,
    post_scores: Vec<f32>,
    selected_scores: Vec<f32>,
    denom_steps: Vec<f32>,
}

struct RealLayerRun {
    output: Vec<u16>,
    attention_output: Vec<u16>,
    routes: v4_router::RouterSelection,
    attention_topk: Option<Vec<i32>>,
    attention_topk_digest: u64,
    attention_topk_per_row: usize,
    device_transfer: Option<DeviceAttentionTransfer>,
    indexer_topk_gpu_ns: u64,
    stage: LayerStageTelemetry,
    critical_path_timeline: Option<CriticalPathTimeline>,
    routed_audit: Option<RoutedExecutionAudit>,
}

/// Gate-only routed values. The production path leaves this absent so it does
/// not retain six expert outputs after canonical reduction.
struct RoutedExecutionAudit {
    after_attention_digest: u64,
    ffn_input_digest: u64,
    shared_digest: u64,
    moe_digest: u64,
    contributions: Vec<Vec<f32>>,
}

struct CriticalPathTimeline {
    origin: std::time::Instant,
    fixed_ready_ns: u64,
    attention_start_ns: u64,
    attention_end_ns: u64,
    router_start_ns: u64,
    router_end_ns: u64,
    shared_start_ns: u64,
    shared_end_ns: u64,
    routed_start_ns: u64,
    routed_end_ns: u64,
    hc_post_start_ns: u64,
    hc_post_end_ns: u64,
    output_ready_ns: u64,
    attention_commit_ns: u64,
    routed: Option<RoutedQd4Timeline>,
}

impl CriticalPathTimeline {
    fn new(origin: std::time::Instant) -> Self {
        let fixed_ready_ns = timeline_elapsed_ns(origin);
        Self {
            origin,
            fixed_ready_ns,
            attention_start_ns: fixed_ready_ns,
            attention_end_ns: fixed_ready_ns,
            router_start_ns: fixed_ready_ns,
            router_end_ns: fixed_ready_ns,
            shared_start_ns: fixed_ready_ns,
            shared_end_ns: fixed_ready_ns,
            routed_start_ns: fixed_ready_ns,
            routed_end_ns: fixed_ready_ns,
            hc_post_start_ns: fixed_ready_ns,
            hc_post_end_ns: fixed_ready_ns,
            output_ready_ns: fixed_ready_ns,
            attention_commit_ns: 0,
            routed: None,
        }
    }

    fn now(&self) -> u64 {
        timeline_elapsed_ns(self.origin)
    }

    fn as_json(&self) -> serde_json::Value {
        serde_json::json!({
            "clock": "monotonic_host_ns; CUDA event durations are nested diagnostics, not cross-stream timestamps",
            "fixed_ready_ns": self.fixed_ready_ns,
            "attention": {"start_ns": self.attention_start_ns, "end_ns": self.attention_end_ns, "state_commit_ns": self.attention_commit_ns},
            "router": {"start_ns": self.router_start_ns, "end_ns": self.router_end_ns},
            "shared": {"start_ns": self.shared_start_ns, "end_ns": self.shared_end_ns},
            "routed": self.routed.as_ref().map(RoutedQd4Timeline::as_json),
            "routed_scope": {"start_ns": self.routed_start_ns, "end_ns": self.routed_end_ns},
            "hc_post": {"start_ns": self.hc_post_start_ns, "end_ns": self.hc_post_end_ns},
            "output_ready_ns": self.output_ready_ns,
        })
    }
}

fn timeline_elapsed_ns(origin: std::time::Instant) -> u64 {
    u64::try_from(origin.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

#[derive(Clone)]
struct RealAttentionRun {
    output: Vec<u16>,
    topk: Option<Vec<i32>>,
    stage_digests: Vec<(&'static str, u64)>,
    attention_topk_digest: u64,
    attention_topk_per_row: usize,
    indexer_topk_gpu_ns: u64,
    attention_ns: u64,
    commit_ns: u64,
    device_transfer: Option<DeviceAttentionTransfer>,
    profile: Option<AttentionProfile>,
}

#[derive(Clone, Default)]
struct DeviceAttentionTransfer {
    h2d_count: usize,
    h2d_bytes: usize,
    rope_lut_d2d_count: usize,
    rope_lut_d2d_bytes: usize,
    terminal_d2h_count: usize,
    terminal_d2h_bytes: usize,
    intermediate_or_audit_d2h_count: usize,
    intermediate_or_audit_d2h_bytes: usize,
}

struct DeviceCompressorRows {
    kv: DeviceBuffer<f32>,
    score: DeviceBuffer<f32>,
    score_plus_ape: DeviceBuffer<f32>,
    persistent_kv: DeviceBuffer<f32>,
    persistent_score: DeviceBuffer<f32>,
}

/// Per-layer state for the generic stationary path.  It intentionally owns
/// only data that survives from row p to p+1; all projections and attention
/// intermediates live in the one `DeviceRequestScratch` below.
struct DeviceLayerPersistentState {
    layer: usize,
    committed_rows: usize,
    compressor_width: usize,
    raw_kv_ring: DeviceBuffer<u16>,
    compressor_tail_kv: DeviceBuffer<f32>,
    compressor_tail_score: DeviceBuffer<f32>,
    compressed_kv: DeviceBuffer<u16>,
    indexer_tail_kv: Option<DeviceBuffer<f32>>,
    indexer_tail_score: Option<DeviceBuffer<f32>>,
    indexer_kv: Option<DeviceBuffer<u16>>,
}

/// Exactly one request-local GPU workspace.  This is deliberately separate
/// from `DeviceTokenState`, which remains the frozen short/audit fixture.
struct DeviceRequestScratch {
    hc_hidden: DeviceBuffer<u16>,
    freqs: DeviceBuffer<f32>,
    inverse_freqs: DeviceBuffer<f32>,
    compressed_freqs: DeviceBuffer<f32>,
    mixes: DeviceBuffer<f32>,
    pre: DeviceBuffer<f32>,
    post: DeviceBuffer<f32>,
    comb: DeviceBuffer<f32>,
    trace: DeviceBuffer<f32>,
    collapsed: DeviceBuffer<u16>,
    attn_input: DeviceBuffer<u16>,
    fp8_7168: DeviceBuffer<u8>,
    fp8_7168_scales: DeviceBuffer<u8>,
    fp8_1536: DeviceBuffer<u8>,
    fp8_1536_scales: DeviceBuffer<u8>,
    fp8_16384: DeviceBuffer<u8>,
    fp8_16384_scales: DeviceBuffer<u8>,
    empty_u8: DeviceBuffer<u8>,
    empty_u16: DeviceBuffer<u16>,
    q_lora_f32: DeviceBuffer<f32>,
    q_lora_raw: DeviceBuffer<u16>,
    q_lora: DeviceBuffer<u16>,
    q_f32: DeviceBuffer<f32>,
    q_raw: DeviceBuffer<u16>,
    q_norm: DeviceBuffer<u16>,
    q_rope: DeviceBuffer<u16>,
    kv_f32: DeviceBuffer<f32>,
    kv_raw: DeviceBuffer<u16>,
    kv_norm: DeviceBuffer<u16>,
    kv_rope: DeviceBuffer<u16>,
    kv_ready: DeviceBuffer<u16>,
    kv_scales: DeviceBuffer<u8>,
    candidate_tail_kv: DeviceBuffer<f32>,
    candidate_tail_score: DeviceBuffer<f32>,
    candidate_hca_tail_kv: DeviceBuffer<f32>,
    candidate_hca_tail_score: DeviceBuffer<f32>,
    compressor_kv: DeviceBuffer<f32>,
    compressor_score: DeviceBuffer<f32>,
    compressor_score_plus_ape: DeviceBuffer<f32>,
    hca_compressor_kv: DeviceBuffer<f32>,
    hca_compressor_score: DeviceBuffer<f32>,
    hca_compressor_score_plus_ape: DeviceBuffer<f32>,
    exact_compressor: Option<GenericExactCompressorScratch>,
    indexer_compressor_kv: Option<DeviceBuffer<f32>>,
    indexer_compressor_score: Option<DeviceBuffer<f32>>,
    indexer_compressor_score_plus_ape: Option<DeviceBuffer<f32>>,
    candidate_indexer_tail_kv: Option<DeviceBuffer<f32>>,
    candidate_indexer_tail_score: Option<DeviceBuffer<f32>>,
    compressed_pool: DeviceBuffer<f32>,
    compressed_bf16: DeviceBuffer<u16>,
    compressed_norm: DeviceBuffer<u16>,
    compressed_rope: DeviceBuffer<u16>,
    compressed_ready: DeviceBuffer<u16>,
    compressed_scales: DeviceBuffer<u8>,
    indexer_pool: DeviceBuffer<f32>,
    indexer_bf16: DeviceBuffer<u16>,
    indexer_norm: DeviceBuffer<u16>,
    indexer_rope: DeviceBuffer<u16>,
    indexer_ready: DeviceBuffer<u16>,
    candidate_indexer_cache: DeviceBuffer<u16>,
    indexer_query_raw: DeviceBuffer<u16>,
    indexer_query_rope: DeviceBuffer<u16>,
    indexer_query_fp4: DeviceBuffer<u16>,
    indexer_query_f32: DeviceBuffer<f32>,
    indexer_weights_f32: DeviceBuffer<f32>,
    indexer_kv_f32: DeviceBuffer<f32>,
    indexer_scores: DeviceBuffer<f32>,
    indexer_valid: DeviceBuffer<u32>,
    indexer_selected: DeviceBuffer<i32>,
    topk: DeviceBuffer<i32>,
    attention_cache: DeviceBuffer<u16>,
    sparse: DeviceBuffer<u16>,
    sparse_scratch: SparseAttentionScoreReuseScratch,
    sparse_inverse: DeviceBuffer<u16>,
    wo_a: DeviceBuffer<u16>,
    attention_f32: DeviceBuffer<f32>,
    attention: DeviceBuffer<u16>,
    after_attention: DeviceBuffer<u16>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GenericStationaryPath {
    Hca128,
    Csa4,
}

impl GenericStationaryPath {
    const fn from_ratio(ratio: v4_state::CompressionRatio) -> Self {
        match ratio {
            v4_state::CompressionRatio::WindowOnly => {
                panic!("generic stationary path does not execute window-only layers")
            }
            v4_state::CompressionRatio::Hca128 => Self::Hca128,
            v4_state::CompressionRatio::Csa4 => Self::Csa4,
        }
    }

    const fn is_csa(self) -> bool {
        matches!(self, Self::Csa4)
    }

    const fn ratio(self) -> usize {
        match self {
            Self::Hca128 => 128,
            Self::Csa4 => 4,
        }
    }

    const fn compressor_width(self) -> usize {
        match self {
            Self::Hca128 => v4_state::Q_HEAD_DIM,
            Self::Csa4 => 1_024,
        }
    }

    const fn tail_rows(self) -> usize {
        match self {
            Self::Hca128 => v4_state::KV_WINDOW,
            Self::Csa4 => 8,
        }
    }

    const fn tail_elements(self) -> usize {
        self.tail_rows() * self.compressor_width()
    }

    const fn compressor_slot(self, position: usize) -> usize {
        match self {
            Self::Hca128 => position % v4_state::KV_WINDOW,
            Self::Csa4 => 4 + position % 4,
        }
    }

    const fn topk_rows(self, compressed_rows: usize) -> usize {
        v4_state::KV_WINDOW
            + match self {
                Self::Hca128 => compressed_rows,
                Self::Csa4 if compressed_rows < v4_state::INDEX_TOPK => compressed_rows,
                Self::Csa4 => v4_state::INDEX_TOPK,
            }
    }

    /// Scratch capacity, not the number of causal keys.  Before the raw ring
    /// fills, `window_topk_cpu` encodes uncommitted slots as `-1`.
    const fn composite_capacity_rows(self, position: usize) -> usize {
        self.topk_rows((position + 1) / self.ratio())
    }

    const fn max_topk_rows(self, plan: v4_state::LayerStationaryRequestPlan) -> usize {
        self.topk_rows(self.compressed_rows(plan))
    }

    const fn compressed_rows(self, plan: v4_state::LayerStationaryRequestPlan) -> usize {
        match self {
            Self::Hca128 => plan.hca_compressed_rows,
            Self::Csa4 => plan.csa_compressed_rows,
        }
    }
}

/// Element counts for the one request-shared generic stationary workspace.
/// The HCA and CSA rows/tails coexist, but every request path selects exactly
/// one row and one candidate tail from this table.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct GenericStationaryScratchGeometry {
    hca_main_elements: usize,
    csa_main_elements: usize,
    csa_indexer_elements: usize,
    hca_tail_elements: usize,
    csa_tail_elements: usize,
}

impl GenericStationaryScratchGeometry {
    const fn max() -> Self {
        Self {
            hca_main_elements: v4_state::Q_HEAD_DIM,
            csa_main_elements: 1_024,
            csa_indexer_elements: 256,
            hca_tail_elements: GenericStationaryPath::Hca128.tail_elements(),
            csa_tail_elements: GenericStationaryPath::Csa4.tail_elements(),
        }
    }

    const fn main_elements(self, path: GenericStationaryPath) -> usize {
        match path {
            GenericStationaryPath::Hca128 => self.hca_main_elements,
            GenericStationaryPath::Csa4 => self.csa_main_elements,
        }
    }

    const fn tail_elements(self, path: GenericStationaryPath) -> usize {
        match path {
            GenericStationaryPath::Hca128 => self.hca_tail_elements,
            GenericStationaryPath::Csa4 => self.csa_tail_elements,
        }
    }
}

/// The one source of truth for the request-shared generic scratch.  Both the
/// pre-cache reservation and `DeviceRequestScratch::new` consume this exact
/// geometry, so a Full61 strict run cannot silently diverge from its ledger.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct DeviceRequestScratchGeometry {
    generic: GenericStationaryScratchGeometry,
    csa_compressed_rows: usize,
    topk_rows: usize,
    strict: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct DeviceRequestScratchByteBreakdown {
    core_mhc: usize,
    projections: usize,
    compressor: usize,
    exact_compressor: usize,
    indexer: usize,
    topk_composite: usize,
    sparse: usize,
    score_reuse: usize,
    attention_output: usize,
}

impl DeviceRequestScratchByteBreakdown {
    fn total(self) -> Result<usize, Box<dyn Error>> {
        [
            self.core_mhc,
            self.projections,
            self.compressor,
            self.exact_compressor,
            self.indexer,
            self.topk_composite,
            self.sparse,
            self.score_reuse,
            self.attention_output,
        ]
        .into_iter()
        .try_fold(0usize, |total, value| {
            total
                .checked_add(value)
                .ok_or_else(|| -> Box<dyn Error> { "generic scratch breakdown overflow".into() })
        })
    }

    fn json(self) -> serde_json::Value {
        serde_json::json!({
            "core_mhc": self.core_mhc,
            "projections": self.projections,
            "compressor": self.compressor,
            "exact_compressor": self.exact_compressor,
            "indexer": self.indexer,
            "topk_composite": self.topk_composite,
            "sparse": self.sparse,
            "score_reuse": self.score_reuse,
            "attention_output": self.attention_output,
        })
    }
}

impl DeviceRequestScratchGeometry {
    fn new(
        plan: v4_state::LayerStationaryRequestPlan,
        strict: bool,
    ) -> Result<Self, Box<dyn Error>> {
        let topk_rows = GenericStationaryPath::Hca128
            .max_topk_rows(plan)
            .max(GenericStationaryPath::Csa4.max_topk_rows(plan));
        Ok(Self {
            generic: GenericStationaryScratchGeometry::max(),
            csa_compressed_rows: plan.csa_compressed_rows,
            topk_rows,
            strict,
        })
    }

    const fn attention_f32_elements(self) -> usize {
        v4_state::HIDDEN
    }

    fn byte_breakdown(self) -> Result<DeviceRequestScratchByteBreakdown, Box<dyn Error>> {
        let bytes = |elements: usize, element_bytes: usize| {
            elements
                .checked_mul(element_bytes)
                .ok_or("generic stationary scratch breakdown overflow")
        };
        let sum = |values: &[usize]| {
            values.iter().try_fold(0usize, |total, &value| {
                total
                    .checked_add(value)
                    .ok_or("generic stationary scratch breakdown overflow")
            })
        };
        Ok(DeviceRequestScratchByteBreakdown {
            core_mhc: sum(&[
                bytes(v4_state::HC * v4_state::HIDDEN, 2)?,
                bytes(2 * v4_state::HIDDEN, 2)?,
                bytes(3 * v4_state::Q_ROPE_DIM, 4)?,
                bytes(
                    v4_state::MIX + 2 * v4_state::HC + v4_state::HC * v4_state::HC,
                    4,
                )?,
                bytes(41 * v4_state::HC * v4_state::HC, 4)?,
            ])?,
            projections: sum(&[
                bytes(
                    v4_state::HIDDEN
                        + v4_state::HIDDEN / v4_state::FP8_BLOCK
                        + V4_Q_LORA_RANK
                        + V4_Q_LORA_RANK / v4_state::FP8_BLOCK
                        + v4_state::O_GROUPS * v4_state::O_LORA_RANK
                        + v4_state::O_GROUPS * v4_state::O_LORA_RANK / v4_state::FP8_BLOCK,
                    1,
                )?,
                bytes(
                    2 * V4_Q_LORA_RANK
                        + 3 * v4_state::Q_HEADS * v4_state::Q_HEAD_DIM
                        + 4 * v4_state::Q_HEAD_DIM,
                    2,
                )?,
                bytes(
                    V4_Q_LORA_RANK
                        + v4_state::Q_HEADS * v4_state::Q_HEAD_DIM
                        + v4_state::Q_HEAD_DIM,
                    4,
                )?,
            ])?,
            compressor: sum(&[
                bytes(4 * v4_state::Q_HEAD_DIM, 2)?,
                bytes(14, 1)?,
                bytes(
                    2 * self.generic.csa_tail_elements
                        + 2 * self.generic.hca_tail_elements
                        + 3 * self.generic.csa_main_elements
                        + 3 * self.generic.hca_main_elements
                        + v4_state::Q_HEAD_DIM,
                    4,
                )?,
            ])?,
            exact_compressor: if self.strict {
                sum(&[
                    cublas_exact::WORKSPACE_BYTES,
                    bytes(
                        v4_state::HIDDEN
                            + 2 * self.generic.hca_main_elements * v4_state::HIDDEN
                            + 2 * self.generic.csa_main_elements * v4_state::HIDDEN
                            + 2 * self.generic.csa_indexer_elements * v4_state::HIDDEN,
                        4,
                    )?,
                ])?
            } else {
                0
            },
            indexer: sum(&[
                bytes(4 * 128 + self.csa_compressed_rows * 128 + 3 * 64 * 128, 2)?,
                bytes(
                    3 * self.generic.csa_indexer_elements
                        + 16 * self.generic.csa_indexer_elements
                        + 128
                        + 64 * 128
                        + 64
                        + self.csa_compressed_rows * 128
                        + self.csa_compressed_rows,
                    4,
                )?,
                bytes(self.csa_compressed_rows.min(v4_state::INDEX_TOPK), 4)?,
                bytes(self.csa_compressed_rows, 4)?,
            ])?,
            topk_composite: sum(&[
                bytes(self.topk_rows * v4_state::Q_HEAD_DIM, 2)?,
                bytes(self.topk_rows, 4)?,
            ])?,
            sparse: bytes(2 * v4_state::Q_HEADS * v4_state::Q_HEAD_DIM, 2)?,
            score_reuse: sum(&[
                bytes(
                    2 * self.topk_rows * v4_state::Q_HEADS + v4_state::Q_HEADS,
                    4,
                )?,
                bytes(v4_state::Q_HEADS, 4)?,
            ])?,
            attention_output: sum(&[
                bytes(
                    v4_state::O_GROUPS * v4_state::O_LORA_RANK
                        + v4_state::HIDDEN
                        + v4_state::HC * v4_state::HIDDEN,
                    2,
                )?,
                bytes(self.attention_f32_elements(), 4)?,
            ])?,
        })
    }

    fn num_bytes(self) -> Result<usize, Box<dyn Error>> {
        let add = |total: usize, elements: usize, element_bytes: usize| {
            elements
                .checked_mul(element_bytes)
                .and_then(|value| total.checked_add(value))
                .ok_or("generic stationary scratch reserve overflow")
        };
        let mut total = 0usize;
        for elements in [
            v4_state::HC * v4_state::HIDDEN,
            v4_state::HIDDEN,
            v4_state::HIDDEN,
            V4_Q_LORA_RANK,
            V4_Q_LORA_RANK,
            3 * v4_state::Q_HEADS * v4_state::Q_HEAD_DIM,
            4 * v4_state::Q_HEAD_DIM,
            4 * v4_state::Q_HEAD_DIM,
            4 * 128,
            self.csa_compressed_rows * 128,
            3 * 64 * 128,
            self.topk_rows * v4_state::Q_HEAD_DIM,
            2 * v4_state::Q_HEADS * v4_state::Q_HEAD_DIM,
            v4_state::O_GROUPS * v4_state::O_LORA_RANK,
            v4_state::HIDDEN,
            v4_state::HC * v4_state::HIDDEN,
        ] {
            total = add(total, elements, 2)?;
        }
        for elements in [
            v4_state::HIDDEN,
            v4_state::HIDDEN / v4_state::FP8_BLOCK,
            V4_Q_LORA_RANK,
            V4_Q_LORA_RANK / v4_state::FP8_BLOCK,
            v4_state::O_GROUPS * v4_state::O_LORA_RANK,
            v4_state::O_GROUPS * v4_state::O_LORA_RANK / v4_state::FP8_BLOCK,
            14,
        ] {
            total = add(total, elements, 1)?;
        }
        for elements in [
            3 * v4_state::Q_ROPE_DIM,
            v4_state::MIX + 2 * v4_state::HC + v4_state::HC * v4_state::HC,
            41 * v4_state::HC * v4_state::HC,
            V4_Q_LORA_RANK,
            v4_state::Q_HEADS * v4_state::Q_HEAD_DIM,
            v4_state::Q_HEAD_DIM,
            2 * self.generic.csa_tail_elements + 2 * self.generic.hca_tail_elements,
            3 * self.generic.csa_main_elements + 3 * self.generic.hca_main_elements,
            3 * self.generic.csa_indexer_elements + 16 * self.generic.csa_indexer_elements,
            v4_state::Q_HEAD_DIM + 128 + 64 * 128 + 64,
            self.csa_compressed_rows * 128 + self.csa_compressed_rows,
            // Score/probability rows and denominator are f32 per head; the
            // attention projection retains its independent hidden-wide f32 row.
            2 * self.topk_rows * v4_state::Q_HEADS
                + v4_state::Q_HEADS
                + self.attention_f32_elements(),
        ] {
            total = add(total, elements, 4)?;
        }
        total = add(
            total,
            self.csa_compressed_rows.min(v4_state::INDEX_TOPK) + self.topk_rows,
            4,
        )?;
        // `SparseAttentionScoreReuseScratch::valid_counts` is one u32/head.
        total = add(total, self.csa_compressed_rows + v4_state::Q_HEADS, 4)?;
        if self.strict {
            total = add(total, cublas_exact::WORKSPACE_BYTES, 1)?;
            total = add(
                total,
                v4_state::HIDDEN
                    + 2 * self.generic.hca_main_elements * v4_state::HIDDEN
                    + 2 * self.generic.csa_main_elements * v4_state::HIDDEN
                    + 2 * self.generic.csa_indexer_elements * v4_state::HIDDEN,
                4,
            )?;
        }
        if total != self.byte_breakdown()?.total()? {
            return Err("generic stationary scratch total differs from its grouped ledger".into());
        }
        Ok(total)
    }
}

#[cfg(test)]
fn generic_stationary_range_fits(
    source_len: usize,
    source_offset: usize,
    destination_len: usize,
    destination_offset: usize,
    count: usize,
) -> bool {
    source_offset
        .checked_add(count)
        .is_some_and(|end| end <= source_len)
        && destination_offset
            .checked_add(count)
            .is_some_and(|end| end <= destination_len)
}

struct DeviceExactCompressorScratch {
    cublas: cublas_exact::CublasExact,
    workspace: DeviceBuffer<u8>,
    input: DeviceBuffer<f32>,
    wkv: DeviceBuffer<f32>,
    wgate: DeviceBuffer<f32>,
    indexer_wkv: Option<DeviceBuffer<f32>>,
    indexer_wgate: Option<DeviceBuffer<f32>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SparseAttentionArm {
    Reference,
    ScoreReuse,
    Sm89Crossover,
}

const SM89_SCORE_REUSE_MIN_TOPK: usize = 4;

impl SparseAttentionArm {
    fn uses_score_reuse(self, topk_per_row: usize) -> bool {
        matches!(self, Self::ScoreReuse)
            || matches!(self, Self::Sm89Crossover) && topk_per_row >= SM89_SCORE_REUSE_MIN_TOPK
    }
}

struct DeviceTokenState {
    layer: usize,
    committed_rows: usize,
    compressor_width: usize,
    hc_hidden: DeviceBuffer<u16>,
    freqs: DeviceBuffer<f32>,
    inverse_freqs: DeviceBuffer<f32>,
    compressed_freqs: DeviceBuffer<f32>,
    mixes: DeviceBuffer<f32>,
    pre: DeviceBuffer<f32>,
    post: DeviceBuffer<f32>,
    comb: DeviceBuffer<f32>,
    trace: DeviceBuffer<f32>,
    collapsed: DeviceBuffer<u16>,
    attn_input: DeviceBuffer<u16>,
    fp8_7168: DeviceBuffer<u8>,
    fp8_7168_scales: DeviceBuffer<u8>,
    fp8_1536: DeviceBuffer<u8>,
    fp8_1536_scales: DeviceBuffer<u8>,
    fp8_16384: DeviceBuffer<u8>,
    fp8_16384_scales: DeviceBuffer<u8>,
    empty_u8: DeviceBuffer<u8>,
    empty_u16: DeviceBuffer<u16>,
    q_lora_f32: DeviceBuffer<f32>,
    q_lora_raw: DeviceBuffer<u16>,
    q_lora: DeviceBuffer<u16>,
    q_f32: DeviceBuffer<f32>,
    q_raw: DeviceBuffer<u16>,
    q_norm: DeviceBuffer<u16>,
    q_rope: DeviceBuffer<u16>,
    kv_f32: DeviceBuffer<f32>,
    kv_raw: DeviceBuffer<u16>,
    kv_norm: DeviceBuffer<u16>,
    kv_rope: DeviceBuffer<u16>,
    kv_ready: DeviceBuffer<u16>,
    kv_scales: DeviceBuffer<u8>,
    kv_cache: DeviceBuffer<u16>,
    compressor: DeviceCompressorRows,
    exact_compressor: Option<DeviceExactCompressorScratch>,
    indexer_compressor: Option<DeviceCompressorRows>,
    compressed_pool: DeviceBuffer<f32>,
    compressed_bf16: DeviceBuffer<u16>,
    compressed_norm: DeviceBuffer<u16>,
    compressed_rope: DeviceBuffer<u16>,
    compressed_ready: DeviceBuffer<u16>,
    compressed_scales: DeviceBuffer<u8>,
    compressed_cache: DeviceBuffer<u16>,
    indexer_pool: DeviceBuffer<f32>,
    indexer_bf16: DeviceBuffer<u16>,
    indexer_norm: DeviceBuffer<u16>,
    indexer_rope: DeviceBuffer<u16>,
    indexer_ready: DeviceBuffer<u16>,
    indexer_cache: DeviceBuffer<u16>,
    indexer_query_raw: DeviceBuffer<u16>,
    indexer_query_rope: DeviceBuffer<u16>,
    indexer_query_fp4: DeviceBuffer<u16>,
    indexer_query_f32: DeviceBuffer<f32>,
    indexer_weights_f32: DeviceBuffer<f32>,
    indexer_kv_f32: DeviceBuffer<f32>,
    indexer_scores: DeviceBuffer<f32>,
    indexer_valid: DeviceBuffer<u32>,
    indexer_selected: DeviceBuffer<i32>,
    topk_csa: Vec<DeviceBuffer<i32>>,
    attention_cache: DeviceBuffer<u16>,
    compressed_rows: usize,
    topk_raw: Vec<DeviceBuffer<i32>>,
    sparse: DeviceBuffer<u16>,
    sparse_pos0: SparseAttentionScoreReuseScratch,
    sparse_pos1: SparseAttentionScoreReuseScratch,
    sparse_csa: SparseAttentionScoreReuseScratch,
    sparse_inverse: DeviceBuffer<u16>,
    wo_a: DeviceBuffer<u16>,
    attention_f32: DeviceBuffer<f32>,
    attention: DeviceBuffer<u16>,
    after_attention: DeviceBuffer<u16>,
}

/// Persistent state owned solely by the MTP draft.  Unlike `DeviceTokenState`,
/// this ratio-0 layout has exactly one BF16 sliding KV allocation and no
/// compressor/indexer state, so a rejected candidate cannot mutate the teacher.
struct MtpDraftState {
    position: usize,
    kv_cache: DeviceBuffer<u16>,
}

impl MtpDraftState {
    fn new(stream: &CudaStream) -> Result<Self, Box<dyn Error>> {
        let plan =
            v4_state::mtp1_swa::mtp1_swa_attention_state_plan(v4_state::mtp1_swa::MTP1_SWA_LAYER)?;
        let kv_cache = DeviceBuffer::zeroed(stream, plan.kv_elements())?;
        if kv_cache.num_bytes() != plan.capacity_contribution_bytes()
            || kv_cache.num_bytes() != 131_072
        {
            return Err("MTP draft BF16 sliding-KV allocation drifted from its plan".into());
        }
        Ok(Self {
            position: 0,
            kv_cache,
        })
    }

    fn persistent_bytes(&self) -> usize {
        self.kv_cache.num_bytes()
    }

    fn digest(&self, stream: &CudaStream) -> Result<u64, Box<dyn Error>> {
        Ok(fnv1a_u16(&self.kv_cache.to_host_vec(stream)?))
    }

    fn commit_kv_row(
        &mut self,
        stream: &CudaStream,
        kv_ready: &DeviceBuffer<u16>,
    ) -> Result<(), Box<dyn Error>> {
        if kv_ready.len() != v4_state::Q_HEAD_DIM
            || self.kv_cache.len() != v4_state::KV_WINDOW * v4_state::Q_HEAD_DIM
        {
            return Err("MTP draft sliding-KV commit contract mismatch".into());
        }
        let slot =
            v4_state::mtp1_swa::mtp1_swa_attention_state_plan(v4_state::mtp1_swa::MTP1_SWA_LAYER)?
                .slot_for_position(self.position)?;
        // SAFETY: the validated plan maps one complete 512-wide row into the
        // unique in-range ring slot owned by this MTP state.
        unsafe { copy_device_row(stream, &self.kv_cache, slot, kv_ready)? };
        self.position = self
            .position
            .checked_add(1)
            .ok_or("MTP draft KV position overflow")?;
        Ok(())
    }
}

struct DeviceRequestRopeLut {
    values: DeviceBuffer<f32>,
    rows: usize,
}

impl DeviceRequestRopeLut {
    fn new(stream: &CudaStream, rows: usize) -> Result<Self, Box<dyn Error>> {
        if !(V4_REQUEST_ROPE_LUT_ROWS..=V4_SHORT_DEVICE_DECODE_MAX_ROWS).contains(&rows) {
            return Err("request RoPE LUT rows are outside the supported decode ceiling".into());
        }
        let forward = (0..rows)
            .flat_map(v4_state::compressed_rope_freqs)
            .collect::<Vec<_>>();
        let mut inverse = forward.clone();
        for sine in inverse.iter_mut().skip(1).step_by(2) {
            *sine = -*sine;
        }
        let mut values = forward;
        values.extend(inverse);
        if values.len() * std::mem::size_of::<f32>()
            != 2 * rows * v4_state::Q_ROPE_DIM * std::mem::size_of::<f32>()
        {
            return Err("request RoPE LUT byte contract mismatch".into());
        }
        Ok(Self {
            values: DeviceBuffer::from_host(stream, &values)?,
            rows,
        })
    }

    fn num_bytes(&self) -> usize {
        self.values.num_bytes()
    }

    fn copy_forward_row(
        &self,
        stream: &CudaStream,
        position: usize,
        destination: &DeviceBuffer<f32>,
    ) -> Result<(), Box<dyn Error>> {
        self.copy_row(stream, position, destination)
    }

    fn copy_inverse_row(
        &self,
        stream: &CudaStream,
        position: usize,
        destination: &DeviceBuffer<f32>,
    ) -> Result<(), Box<dyn Error>> {
        self.copy_row(stream, self.rows + position, destination)
    }

    fn copy_row(
        &self,
        stream: &CudaStream,
        position: usize,
        destination: &DeviceBuffer<f32>,
    ) -> Result<(), Box<dyn Error>> {
        if position >= 2 * self.rows || destination.len() != v4_state::Q_ROPE_DIM {
            return Err("request RoPE LUT row copy contract mismatch".into());
        }
        unsafe {
            copy_device_range(
                stream,
                destination,
                0,
                &self.values,
                position * v4_state::Q_ROPE_DIM,
                v4_state::Q_ROPE_DIM,
            )?;
        }
        Ok(())
    }
}

impl DeviceCompressorRows {
    fn new(
        stream: &CudaStream,
        width: usize,
        capacity_rows: usize,
    ) -> Result<Self, Box<dyn Error>> {
        Ok(Self {
            kv: DeviceBuffer::zeroed(stream, width)?,
            score: DeviceBuffer::zeroed(stream, width)?,
            score_plus_ape: DeviceBuffer::zeroed(stream, width)?,
            persistent_kv: DeviceBuffer::zeroed(stream, capacity_rows * width)?,
            persistent_score: DeviceBuffer::from_host(
                stream,
                &vec![f32::NEG_INFINITY; capacity_rows * width],
            )?,
        })
    }

    fn num_bytes(&self) -> usize {
        self.kv.num_bytes()
            + self.score.num_bytes()
            + self.score_plus_ape.num_bytes()
            + self.persistent_kv.num_bytes()
            + self.persistent_score.num_bytes()
    }

    fn reset_persistent(
        &mut self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
    ) -> Result<usize, Box<dyn Error>> {
        zero_device_buffer(stream, &self.persistent_kv)?;
        // SAFETY: one checked thread writes each persistent score element.
        unsafe {
            module.fill_f32(
                stream,
                LaunchConfig::for_num_elems(u32::try_from(self.persistent_score.len())?),
                &mut self.persistent_score,
                f32::NEG_INFINITY,
            )?;
        }
        Ok(2)
    }
}

impl DeviceLayerPersistentState {
    fn new(
        stream: &CudaStream,
        weights: &LayerDeviceWeights,
        plan: v4_state::LayerStationaryRequestPlan,
    ) -> Result<Self, Box<dyn Error>> {
        let path = GenericStationaryPath::from_ratio(weights.compression_ratio);
        let csa = path.is_csa();
        let compressor_width = path.compressor_width();
        let compressed_rows = path.compressed_rows(plan);
        let tail_elements = path.tail_elements();
        let compressed_elements = compressed_rows
            .checked_mul(v4_state::Q_HEAD_DIM)
            .ok_or("generic stationary compressed KV capacity overflow")?;
        let indexer_elements = plan
            .csa_compressed_rows
            .checked_mul(128)
            .ok_or("generic stationary indexer KV capacity overflow")?;
        let state = Self {
            layer: weights.layer,
            committed_rows: 0,
            compressor_width,
            raw_kv_ring: DeviceBuffer::zeroed(stream, v4_state::KV_WINDOW * v4_state::Q_HEAD_DIM)?,
            compressor_tail_kv: DeviceBuffer::zeroed(stream, tail_elements)?,
            compressor_tail_score: DeviceBuffer::from_host(
                stream,
                &vec![f32::NEG_INFINITY; tail_elements],
            )?,
            compressed_kv: DeviceBuffer::zeroed(stream, compressed_elements)?,
            indexer_tail_kv: csa
                .then(|| DeviceBuffer::zeroed(stream, 8 * 256))
                .transpose()?,
            indexer_tail_score: csa
                .then(|| DeviceBuffer::from_host(stream, &vec![f32::NEG_INFINITY; 8 * 256]))
                .transpose()?,
            indexer_kv: csa
                .then(|| DeviceBuffer::zeroed(stream, indexer_elements))
                .transpose()?,
        };
        if state.raw_kv_ring.len() != v4_state::KV_WINDOW * v4_state::Q_HEAD_DIM
            || state.compressor_tail_kv.len() != tail_elements
            || state.compressor_tail_score.len() != tail_elements
            || state.compressed_kv.len() != compressed_elements
        {
            return Err("generic stationary persistent layout drifted from request plan".into());
        }
        Ok(state)
    }

    fn raw_slot(&self, position: usize) -> Result<usize, Box<dyn Error>> {
        if position != self.committed_rows {
            return Err("generic stationary raw-ring commit is not contiguous".into());
        }
        Ok(position % v4_state::KV_WINDOW)
    }

    fn compressed_rows(&self, ratio: v4_state::CompressionRatio) -> usize {
        self.committed_rows / ratio.get()
    }

    fn num_bytes(&self) -> usize {
        self.raw_kv_ring.num_bytes()
            + self.compressor_tail_kv.num_bytes()
            + self.compressor_tail_score.num_bytes()
            + self.compressed_kv.num_bytes()
            + self
                .indexer_tail_kv
                .as_ref()
                .map_or(0, DeviceBuffer::num_bytes)
            + self
                .indexer_tail_score
                .as_ref()
                .map_or(0, DeviceBuffer::num_bytes)
            + self.indexer_kv.as_ref().map_or(0, DeviceBuffer::num_bytes)
    }
}

#[derive(Clone, Debug, PartialEq)]
struct AttentionAudit {
    outputs: [Vec<u16>; 2],
    stage_digests: [Vec<(&'static str, u64)>; 2],
    topk_digests: [u64; 2],
    kv: Vec<u16>,
    main_kv: Vec<f32>,
    main_score: Vec<f32>,
    indexer_kv: Option<Vec<f32>>,
    indexer_score: Option<Vec<f32>>,
}

#[derive(Clone, Debug)]
struct PersistentStateAudit {
    kv: Vec<u16>,
    compressed_kv: Vec<u16>,
    indexer_kv: Vec<u16>,
    main_kv: Vec<f32>,
    main_score: Vec<f32>,
    indexer_main_kv: Option<Vec<f32>>,
    indexer_main_score: Option<Vec<f32>>,
}

fn host_persistent_state_audit(state: &v4_state::V4LayerPersistentState) -> PersistentStateAudit {
    let main = state.compressor();
    let indexer = state.indexer();
    PersistentStateAudit {
        kv: state.kv_bf16().to_vec(),
        compressed_kv: state.compressed_kv_bf16().to_vec(),
        indexer_kv: state.indexer_kv_bf16().to_vec(),
        main_kv: main.kv().to_vec(),
        main_score: main.score().to_vec(),
        indexer_main_kv: indexer.map(|value| value.kv().to_vec()),
        indexer_main_score: indexer.map(|value| value.score().to_vec()),
    }
}

fn device_persistent_state_audit(
    stream: &CudaStream,
    state: &DeviceTokenState,
) -> Result<PersistentStateAudit, Box<dyn Error>> {
    let indexer = state.indexer_compressor.as_ref();
    Ok(PersistentStateAudit {
        kv: state.kv_cache.to_host_vec(stream)?[..state.committed_rows * v4_state::Q_HEAD_DIM]
            .to_vec(),
        compressed_kv: state.compressed_cache.to_host_vec(stream)?
            [..state.compressed_rows * v4_state::Q_HEAD_DIM]
            .to_vec(),
        indexer_kv: state.indexer_cache.to_host_vec(stream)?[..state.compressed_rows * 128]
            .to_vec(),
        main_kv: state.compressor.persistent_kv.to_host_vec(stream)?,
        main_score: state.compressor.persistent_score.to_host_vec(stream)?,
        indexer_main_kv: indexer
            .map(|value| value.persistent_kv.to_host_vec(stream))
            .transpose()?,
        indexer_main_score: indexer
            .map(|value| value.persistent_score.to_host_vec(stream))
            .transpose()?,
    })
}

/// The audit reads each listed device allocation in full before trimming the
/// logical cache prefix, so this is the physical D2H transfer ledger.
fn device_persistent_state_audit_d2h_transfer(state: &DeviceTokenState) -> (usize, usize) {
    let mut count = 5;
    let mut bytes = state.kv_cache.num_bytes()
        + state.compressed_cache.num_bytes()
        + state.indexer_cache.num_bytes()
        + state.compressor.persistent_kv.num_bytes()
        + state.compressor.persistent_score.num_bytes();
    if let Some(indexer) = state.indexer_compressor.as_ref() {
        count += 2;
        bytes += indexer.persistent_kv.num_bytes() + indexer.persistent_score.num_bytes();
    }
    (count, bytes)
}

/// Audit-only D2H snapshot of the generic layout.  It mirrors the host
/// persistent representation exactly: physical raw ring, compressor tails,
/// compressed cache, and optional CSA indexer state.
fn generic_device_persistent_state_audit(
    stream: &CudaStream,
    state: &DeviceLayerPersistentState,
) -> Result<PersistentStateAudit, Box<dyn Error>> {
    Ok(PersistentStateAudit {
        kv: state.raw_kv_ring.to_host_vec(stream)?,
        compressed_kv: state.compressed_kv.to_host_vec(stream)?,
        indexer_kv: state
            .indexer_kv
            .as_ref()
            .map(|value| value.to_host_vec(stream))
            .transpose()?
            .unwrap_or_default(),
        main_kv: state.compressor_tail_kv.to_host_vec(stream)?,
        main_score: state.compressor_tail_score.to_host_vec(stream)?,
        indexer_main_kv: state
            .indexer_tail_kv
            .as_ref()
            .map(|value| value.to_host_vec(stream))
            .transpose()?,
        indexer_main_score: state
            .indexer_tail_score
            .as_ref()
            .map(|value| value.to_host_vec(stream))
            .transpose()?,
    })
}

fn generic_device_persistent_state_audit_d2h_transfer(
    state: &DeviceLayerPersistentState,
) -> (usize, usize) {
    let mut count = 4;
    let mut bytes = state.raw_kv_ring.num_bytes()
        + state.compressor_tail_kv.num_bytes()
        + state.compressor_tail_score.num_bytes()
        + state.compressed_kv.num_bytes();
    if let (Some(indexer_kv), Some(indexer_tail_kv), Some(indexer_tail_score)) = (
        state.indexer_kv.as_ref(),
        state.indexer_tail_kv.as_ref(),
        state.indexer_tail_score.as_ref(),
    ) {
        count += 3;
        bytes +=
            indexer_kv.num_bytes() + indexer_tail_kv.num_bytes() + indexer_tail_score.num_bytes();
    }
    (count, bytes)
}

/// Runs the host recurrence used only to observe generic device state.  The
/// authoritative row is immutable here, and this helper receives no device
/// state, request scratch, expert slots, or router state.
fn run_generic_device_state_audit_mirror(
    enabled: bool,
    persistent: Option<&mut v4_state::V4SerialPersistentState>,
    layer: usize,
    hc_hidden: &[u16],
    run: impl FnOnce(&mut v4_state::V4LayerPersistentState, &[u16]) -> Result<(), Box<dyn Error>>,
) -> Result<(), Box<dyn Error>> {
    if enabled {
        persistent
            .ok_or("generic device state audit host mirror disappeared")?
            .transaction(layer, |host_state| run(host_state, hc_hidden))?;
    }
    Ok(())
}

/// Runs only after a generic head boundary has drained its worker leases. The
/// host mirror is an independent recurrence; audit copies never enter the
/// production transfer ledger or per-layer telemetry.
fn generic_device_state_audit_head(
    stream: &CudaStream,
    states: &[Option<DeviceLayerPersistentState>],
    host: &v4_state::V4SerialPersistentState,
    lifecycle: &GenericStationaryLifecycle,
    scratch: &DeviceRequestScratch,
    position: usize,
) -> Result<(serde_json::Value, Vec<(usize, PersistentStateAudit)>), Box<dyn Error>> {
    lifecycle.assert_drained()?;
    if states.len() != v4_state::V4_LAYER_COUNT {
        return Err("generic device state audit lost a layer store".into());
    }
    let started = std::time::Instant::now();
    let mut d2h_count = 0_usize;
    let mut d2h_bytes = 0_usize;
    let mut layers = Vec::with_capacity(v4_state::V4_LAYER_COUNT);
    let mut snapshots = Vec::with_capacity(v4_state::V4_LAYER_COUNT);
    for (layer, state) in states.iter().enumerate() {
        let state = state
            .as_ref()
            .ok_or("generic device state audit found an empty layer store")?;
        if state.layer != layer || state.committed_rows != position + 1 {
            return Err("generic device state audit found a noncontiguous store".into());
        }
        let expected = host_persistent_state_audit(
            host.layer(layer)
                .ok_or("generic device state audit host mirror lost a layer")?,
        );
        let actual = generic_device_persistent_state_audit(stream, state)?;
        check_persistent_state_audit(layer, position, &expected, &actual)?;
        let (count, bytes) = generic_device_persistent_state_audit_d2h_transfer(state);
        d2h_count = d2h_count.saturating_add(count);
        d2h_bytes = d2h_bytes.saturating_add(bytes);
        layers.push(serde_json::json!({
            "layer": layer,
            "committed_rows": state.committed_rows,
            "exact": true,
            "host": persistent_state_audit_json(&expected),
            "device": persistent_state_audit_json(&actual),
        }));
        snapshots.push((state.committed_rows, actual));
    }
    Ok((
        serde_json::json!({
            "position": position,
            "layers": layers,
            "exact": true,
            "d2h_count": d2h_count,
            "d2h_bytes": d2h_bytes,
            "strict_compressor_conversion_audit": scratch.exact_compressor.as_ref().map(GenericExactCompressorScratch::audit_telemetry),
            "audit_wall_ns": u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            "excluded_from_production_timing": true,
        }),
        snapshots,
    ))
}

fn check_post_mtp_persistent_state(
    layer: usize,
    position: usize,
    expected_committed_rows: usize,
    expected: &PersistentStateAudit,
    actual_committed_rows: usize,
    actual: &PersistentStateAudit,
) -> Result<(), Box<dyn Error>> {
    if actual_committed_rows != expected_committed_rows {
        return Err(format!(
            "post-MTP teacher-state clobber: layer {layer} position {position} committed_rows expected {expected_committed_rows} actual {actual_committed_rows}"
        )
        .into());
    }
    check_persistent_state_audit(layer, position, expected, actual)
        .map_err(|error| format!("post-MTP teacher-state clobber: {error}").into())
}

fn check_generic_device_state_unchanged(
    stream: &CudaStream,
    states: &[Option<DeviceLayerPersistentState>],
    expected: &[(usize, PersistentStateAudit)],
    position: usize,
) -> Result<u64, Box<dyn Error>> {
    let started = std::time::Instant::now();
    if states.len() != v4_state::V4_LAYER_COUNT || expected.len() != v4_state::V4_LAYER_COUNT {
        return Err("post-MTP teacher-state clobber: lost a layer store".into());
    }
    for (layer, (state, (expected_committed_rows, expected))) in
        states.iter().zip(expected).enumerate()
    {
        let state = state
            .as_ref()
            .ok_or("post-MTP teacher-state clobber: found an empty layer store")?;
        if state.layer != layer {
            return Err("post-MTP teacher-state clobber: found a misordered layer store".into());
        }
        let actual = generic_device_persistent_state_audit(stream, state)?;
        check_post_mtp_persistent_state(
            layer,
            position,
            *expected_committed_rows,
            expected,
            state.committed_rows,
            &actual,
        )?;
    }
    Ok(u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX))
}

fn first_f32_bits_mismatch(left: &[f32], right: &[f32]) -> Option<usize> {
    left.iter()
        .zip(right)
        .position(|(left, right)| left.to_bits() != right.to_bits())
        .or_else(|| (left.len() != right.len()).then_some(left.len().min(right.len())))
}

/// Diagnostic only: persistent Compressor state remains a strict FP32-bit
/// gate.  When it fails, record enough numerics to distinguish a slot/layout
/// fault from a bounded reduction-order difference before changing policy.
fn f32_state_mismatch_diagnostic(expected: &[f32], actual: &[f32], index: usize) -> String {
    let expected_value = expected.get(index).copied();
    let actual_value = actual.get(index).copied();
    let (mut max_abs, mut max_rel, mut max_ulp) = (0.0_f32, 0.0_f32, 0_u32);
    let mut bf16_first = None;
    for (element, (&expected, &actual)) in expected.iter().zip(actual).enumerate() {
        if expected.is_finite() && actual.is_finite() {
            let abs = (expected - actual).abs();
            max_abs = max_abs.max(abs);
            max_rel = max_rel.max(abs / expected.abs().max(f32::MIN_POSITIVE));
            max_ulp = max_ulp.max(f32_ulp_distance(expected, actual));
        } else if expected.to_bits() != actual.to_bits() {
            max_abs = f32::INFINITY;
            max_rel = f32::INFINITY;
            max_ulp = u32::MAX;
        }
        if bf16_first.is_none()
            && v4_router::f32_to_bf16_rne(expected) != v4_router::f32_to_bf16_rne(actual)
        {
            bf16_first = Some(element);
        }
    }
    let render = |value: Option<f32>| match value {
        Some(value) => format!("value={value:e} bits=0x{:08x}", value.to_bits()),
        None => "missing".to_owned(),
    };
    let first_abs = match (expected_value, actual_value) {
        (Some(expected), Some(actual)) => (expected - actual).abs(),
        _ => f32::INFINITY,
    };
    let first_rel = match (expected_value, actual_value) {
        (Some(expected), Some(_actual)) => first_abs / expected.abs().max(f32::MIN_POSITIVE),
        _ => f32::INFINITY,
    };
    let first_ulp = match (expected_value, actual_value) {
        (Some(expected), Some(actual)) if expected.is_finite() && actual.is_finite() => {
            f32_ulp_distance(expected, actual)
        }
        (Some(expected), Some(actual)) if expected.to_bits() == actual.to_bits() => 0,
        _ => u32::MAX,
    };
    format!(
        "expected_{} actual_{} first_abs={first_abs:e} first_rel={first_rel:e} first_ulp={first_ulp} max_abs={max_abs:e} max_rel={max_rel:e} max_ulp={max_ulp} bf16_rne_all_equal={} bf16_rne_first_mismatch={:?} expected_len={} actual_len={}",
        render(expected_value),
        render(actual_value),
        bf16_first.is_none() && expected.len() == actual.len(),
        bf16_first,
        expected.len(),
        actual.len(),
    )
}

fn check_persistent_state_audit(
    layer: usize,
    position: usize,
    expected: &PersistentStateAudit,
    actual: &PersistentStateAudit,
) -> Result<(), Box<dyn Error>> {
    check_persistent_state_audit_with_compressor_state(layer, position, expected, actual, true)
}

fn check_short_decode_full_layer_audit(
    layer: usize,
    position: usize,
    expected: &RealLayerRun,
    expected_state: &PersistentStateAudit,
    actual: &RealLayerRun,
    actual_state: &PersistentStateAudit,
) -> Result<(), Box<dyn Error>> {
    if let Some(index) = first_mismatch(&expected.attention_output, &actual.attention_output) {
        return Err(format!(
            "short decode layer {layer} position {position} attention post mismatch at {index}"
        )
        .into());
    }
    if expected.attention_topk != actual.attention_topk {
        return Err(format!(
            "short decode layer {layer} position {position} attention top-k mismatch"
        )
        .into());
    }
    check_persistent_state_audit(layer, position, expected_state, actual_state)?;
    if expected.routes.ids != actual.routes.ids
        || first_f32_bits_mismatch(&expected.routes.weights, &actual.routes.weights).is_some()
    {
        return Err(
            format!("short decode layer {layer} position {position} route mismatch").into(),
        );
    }
    let expected_routed = expected
        .routed_audit
        .as_ref()
        .ok_or("short decode host oracle omitted routed contribution audit")?;
    let actual_routed = actual
        .routed_audit
        .as_ref()
        .ok_or("short decode device candidate omitted routed contribution audit")?;
    if expected_routed.after_attention_digest != actual_routed.after_attention_digest
        || expected_routed.ffn_input_digest != actual_routed.ffn_input_digest
        || expected_routed.shared_digest != actual_routed.shared_digest
        || expected_routed.moe_digest != actual_routed.moe_digest
    {
        return Err(format!(
            "short decode layer {layer} position {position} routed stage mismatch"
        )
        .into());
    }
    if expected_routed.contributions.len() != actual_routed.contributions.len()
        || expected_routed
            .contributions
            .iter()
            .zip(&actual_routed.contributions)
            .any(|(left, right)| first_f32_bits_mismatch(left, right).is_some())
    {
        return Err(format!(
            "short decode layer {layer} position {position} expert output mismatch"
        )
        .into());
    }
    if let Some(index) = first_mismatch(&expected.output, &actual.output) {
        return Err(format!(
            "short decode layer {layer} position {position} canonical output mismatch at {index}"
        )
        .into());
    }
    Ok(())
}

/// HCA never reads Compressor's rolling FP32 slots after committing them.  On
/// a reused HCA allocation their unwritten tail is intentionally unspecified;
/// audit the observable KV/cache state and complete attention instead.  CSA
/// retains the full FP32 check because its 3->4 boundary reads the overlap.
fn check_persistent_state_audit_with_compressor_state(
    layer: usize,
    position: usize,
    expected: &PersistentStateAudit,
    actual: &PersistentStateAudit,
    require_compressor_state: bool,
) -> Result<(), Box<dyn Error>> {
    for (name, mismatch) in [
        ("kv", first_mismatch(&expected.kv, &actual.kv)),
        (
            "compressed_kv",
            first_mismatch(&expected.compressed_kv, &actual.compressed_kv),
        ),
        (
            "indexer_kv",
            first_mismatch(&expected.indexer_kv, &actual.indexer_kv),
        ),
    ] {
        if let Some(index) = mismatch {
            return Err(format!(
                "device token-state layer {layer} position {position} {name} first mismatch at {index}"
            )
            .into());
        }
    }
    if !require_compressor_state {
        return Ok(());
    }
    for (name, expected, actual) in [
        ("main_compressor_kv", &expected.main_kv, &actual.main_kv),
        (
            "main_compressor_score",
            &expected.main_score,
            &actual.main_score,
        ),
    ] {
        if let Some(index) = first_f32_bits_mismatch(expected, actual) {
            return Err(format!(
                "device token-state layer {layer} position {position} {name} first mismatch at {index} ({})",
                f32_state_mismatch_diagnostic(expected, actual, index),
            )
            .into());
        }
    }
    for (name, expected, actual) in [
        (
            "indexer_compressor_kv",
            &expected.indexer_main_kv,
            &actual.indexer_main_kv,
        ),
        (
            "indexer_compressor_score",
            &expected.indexer_main_score,
            &actual.indexer_main_score,
        ),
    ] {
        match (expected, actual) {
            (Some(expected), Some(actual)) => {
                if let Some(index) = first_f32_bits_mismatch(expected, actual) {
                    return Err(format!(
                        "device token-state layer {layer} position {position} {name} first mismatch at {index} ({})",
                        f32_state_mismatch_diagnostic(expected, actual, index),
                    )
                    .into());
                }
            }
            (None, None) => {}
            _ => {
                return Err(format!(
                    "device token-state layer {layer} position {position} {name} presence mismatch"
                )
                .into());
            }
        }
    }
    Ok(())
}

fn persistent_state_audit_json(audit: &PersistentStateAudit) -> serde_json::Value {
    serde_json::json!({
        "kv_bf16_digest": format!("{:016x}", fnv1a_u16(&audit.kv)),
        "compressed_kv_bf16_digest": format!("{:016x}", fnv1a_u16(&audit.compressed_kv)),
        "indexer_kv_bf16_digest": format!("{:016x}", fnv1a_u16(&audit.indexer_kv)),
        "main_compressor_kv_f32_digest": format!("{:016x}", fnv1a_f32(&audit.main_kv)),
        "main_compressor_score_f32_digest": format!("{:016x}", fnv1a_f32(&audit.main_score)),
        "indexer_compressor_kv_f32_digest": audit.indexer_main_kv.as_ref().map(|value| format!("{:016x}", fnv1a_f32(value))),
        "indexer_compressor_score_f32_digest": audit.indexer_main_score.as_ref().map(|value| format!("{:016x}", fnv1a_f32(value))),
    })
}

fn run_attention_arm_a(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    runtime: &LoadedLayerRuntime,
    inputs: &[Vec<u16>; 2],
) -> Result<(AttentionAudit, u64), Box<dyn Error>> {
    let mut persistent = v4_state::V4SerialPersistentState::new(2)?;
    let mut outputs = Vec::with_capacity(2);
    let mut topk_digests = Vec::with_capacity(2);
    let mut stage_digests = Vec::with_capacity(2);
    let started = std::time::Instant::now();
    for position in 0..2 {
        persistent.transaction(runtime.weights.layer, |state| {
            let run = run_real_layer_attention_rows(
                module,
                stream,
                None,
                runtime,
                &inputs[position],
                position,
                state,
                None,
                None,
            )?;
            stage_digests.push(run.stage_digests);
            outputs.push(run.output);
            topk_digests.push(run.attention_topk_digest);
            Ok(())
        })?;
    }
    let elapsed_ns = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
    let state = persistent
        .layer(runtime.weights.layer)
        .ok_or("canonical attention state disappeared")?;
    let main = state.compressor();
    let indexer = state.indexer();
    Ok((
        AttentionAudit {
            outputs: outputs
                .try_into()
                .map_err(|_| "canonical attention changed output row count")?,
            stage_digests: stage_digests
                .try_into()
                .map_err(|_| "canonical attention changed stage row count")?,
            topk_digests: topk_digests
                .try_into()
                .map_err(|_| "canonical attention changed top-k row count")?,
            kv: state.kv_bf16().to_vec(),
            main_kv: main.kv().to_vec(),
            main_score: main.score().to_vec(),
            indexer_kv: indexer.map(|value| value.kv().to_vec()),
            indexer_score: indexer.map(|value| value.score().to_vec()),
        },
        elapsed_ns,
    ))
}

fn device_attention_stage_digests(
    stream: &CudaStream,
    state: &DeviceTokenState,
) -> Result<Vec<(&'static str, u64)>, Box<dyn Error>> {
    let sparse_scratch = if state.committed_rows == 1 {
        &state.sparse_pos0
    } else {
        &state.sparse_pos1
    };
    Ok(vec![
        (
            "mhc_collapsed",
            fnv1a_u16(&state.collapsed.to_host_vec(stream)?),
        ),
        ("mhc_post", fnv1a_f32(&state.post.to_host_vec(stream)?)),
        ("mhc_comb", fnv1a_f32(&state.comb.to_host_vec(stream)?)),
        (
            "attn_input",
            fnv1a_u16(&state.attn_input.to_host_vec(stream)?),
        ),
        (
            "q_lora_raw",
            fnv1a_u16(&state.q_lora_raw.to_host_vec(stream)?),
        ),
        ("q_lora", fnv1a_u16(&state.q_lora.to_host_vec(stream)?)),
        ("q_raw", fnv1a_u16(&state.q_raw.to_host_vec(stream)?)),
        ("q_norm", fnv1a_u16(&state.q_norm.to_host_vec(stream)?)),
        ("q_rope", fnv1a_u16(&state.q_rope.to_host_vec(stream)?)),
        ("kv_raw", fnv1a_u16(&state.kv_raw.to_host_vec(stream)?)),
        ("kv_norm", fnv1a_u16(&state.kv_norm.to_host_vec(stream)?)),
        ("kv_rope", fnv1a_u16(&state.kv_rope.to_host_vec(stream)?)),
        ("kv_ready", fnv1a_u16(&state.kv_ready.to_host_vec(stream)?)),
        (
            "compressor_kv",
            fnv1a_f32(&state.compressor.kv.to_host_vec(stream)?),
        ),
        (
            "compressor_score",
            fnv1a_f32(&state.compressor.score.to_host_vec(stream)?),
        ),
        (
            "compressor_score_plus_ape",
            fnv1a_f32(&state.compressor.score_plus_ape.to_host_vec(stream)?),
        ),
        (
            "sparse_raw_scores",
            fnv1a_f32(&sparse_scratch.scores.to_host_vec(stream)?),
        ),
        (
            "sparse_probabilities",
            fnv1a_f32(&sparse_scratch.probabilities.to_host_vec(stream)?),
        ),
        (
            "sparse_denominators",
            fnv1a_f32(&sparse_scratch.denominators.to_host_vec(stream)?),
        ),
        (
            "sparse_valid_counts",
            fnv1a_i32(
                &sparse_scratch
                    .valid_counts
                    .to_host_vec(stream)?
                    .into_iter()
                    .map(|value| value as i32)
                    .collect::<Vec<_>>(),
            ),
        ),
        ("sparse", fnv1a_u16(&state.sparse.to_host_vec(stream)?)),
        (
            "sparse_inverse",
            fnv1a_u16(&state.sparse_inverse.to_host_vec(stream)?),
        ),
        ("wo_a", fnv1a_u16(&state.wo_a.to_host_vec(stream)?)),
        (
            "attention",
            fnv1a_u16(&state.attention.to_host_vec(stream)?),
        ),
        (
            "attention_post",
            fnv1a_u16(&state.after_attention.to_host_vec(stream)?),
        ),
    ])
}

fn run_attention_arm_b(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    runtime: &LoadedLayerRuntime,
    inputs: &[Vec<u16>; 2],
    state: &mut DeviceTokenState,
    rope_lut: &DeviceRequestRopeLut,
    sparse_arm: SparseAttentionArm,
) -> Result<(AttentionAudit, u64, [DeviceAttentionRun; 2]), Box<dyn Error>> {
    let run0 = run_wavefront_l3_l4_attention_row(
        module, stream, runtime, state, rope_lut, &inputs[0], 0, sparse_arm,
    )?;
    let stages0 = device_attention_stage_digests(stream, state)?;
    let run1 = run_wavefront_l3_l4_attention_row(
        module, stream, runtime, state, rope_lut, &inputs[1], 1, sparse_arm,
    )?;
    let stages1 = device_attention_stage_digests(stream, state)?;
    let elapsed_ns = run0.elapsed_ns.saturating_add(run1.elapsed_ns);
    // These downloads are a post-timing correctness audit. The measured path
    // above has only the terminal hidden D2H recorded in each row result.
    let kv = state.kv_cache.to_host_vec(stream)?;
    let main_kv = state.compressor.persistent_kv.to_host_vec(stream)?;
    let main_score = state.compressor.persistent_score.to_host_vec(stream)?;
    let indexer_kv = state
        .indexer_compressor
        .as_ref()
        .map(|value| value.persistent_kv.to_host_vec(stream))
        .transpose()?;
    let indexer_score = state
        .indexer_compressor
        .as_ref()
        .map(|value| value.persistent_score.to_host_vec(stream))
        .transpose()?;
    let topk_digests = [
        fnv1a_i32(&v4_state::window_topk_cpu(1, 0, v4_state::KV_WINDOW)?.0),
        fnv1a_i32(&v4_state::window_topk_cpu(1, 1, v4_state::KV_WINDOW)?.0),
    ];
    Ok((
        AttentionAudit {
            outputs: [run0.output.clone(), run1.output.clone()],
            stage_digests: [stages0, stages1],
            topk_digests,
            kv: kv[..state.committed_rows * v4_state::Q_HEAD_DIM].to_vec(),
            main_kv,
            main_score,
            indexer_kv,
            indexer_score,
        },
        elapsed_ns,
        [run0, run1],
    ))
}

fn first_mismatch<T: PartialEq>(left: &[T], right: &[T]) -> Option<usize> {
    left.iter()
        .zip(right)
        .position(|(left, right)| left != right)
        .or_else(|| (left.len() != right.len()).then_some(left.len().min(right.len())))
}

fn check_compressed_boundary_bf16(
    stage: &str,
    layer: usize,
    position: usize,
    first_row: usize,
    row_width: usize,
    expected: &[u16],
    actual: &[u16],
) -> Result<(), Box<dyn Error>> {
    if row_width == 0 {
        return Err("compressed boundary audit row width is zero".into());
    }
    if let Some(flat_index) = first_mismatch(expected, actual) {
        let row = first_row + flat_index / row_width;
        let index = flat_index % row_width;
        let render = |value: Option<&u16>| {
            value.map_or_else(|| "missing".to_owned(), |value| format!("0x{value:04x}"))
        };
        return Err(format!(
            "stage={stage} layer={layer} position={position} row={row} index={index} old={} new={}",
            render(expected.get(flat_index)),
            render(actual.get(flat_index)),
        )
        .into());
    }
    Ok(())
}

fn targeted_stage_u16(name: &str, expected: &[u16], actual: &[u16]) -> serde_json::Value {
    let mismatch = first_mismatch(expected, actual);
    serde_json::json!({
        "stage": name,
        "exact": mismatch.is_none(),
        "elements": {"reference": expected.len(), "candidate": actual.len()},
        "digest": {"reference": format!("{:016x}", fnv1a_u16(expected)), "candidate": format!("{:016x}", fnv1a_u16(actual))},
        "first_mismatch": mismatch.map(|index| serde_json::json!({
            "index": index,
            "reference_bits": expected.get(index).map(|value| format!("{value:04x}")),
            "candidate_bits": actual.get(index).map(|value| format!("{value:04x}")),
            "reference_bf16": expected.get(index).map(|&value| format!("{:.9e}", v4_router::bf16_to_f32(value))),
            "candidate_bf16": actual.get(index).map(|&value| format!("{:.9e}", v4_router::bf16_to_f32(value))),
        })),
    })
}

fn targeted_stage_f32(name: &str, expected: &[f32], actual: &[f32]) -> serde_json::Value {
    let mismatch = expected
        .iter()
        .zip(actual)
        .position(|(&expected, &actual)| expected.to_bits() != actual.to_bits())
        .or_else(|| (expected.len() != actual.len()).then_some(expected.len().min(actual.len())));
    serde_json::json!({
        "stage": name,
        "exact": mismatch.is_none(),
        "elements": {"reference": expected.len(), "candidate": actual.len()},
        "digest": {"reference": format!("{:016x}", fnv1a_f32(expected)), "candidate": format!("{:016x}", fnv1a_f32(actual))},
        "first_mismatch": mismatch.map(|index| serde_json::json!({
            "index": index,
            "reference_bits": expected.get(index).map(|value| format!("{:08x}", value.to_bits())),
            "candidate_bits": actual.get(index).map(|value| format!("{:08x}", value.to_bits())),
            "reference_f32": expected.get(index).map(|value| format!("{value:.9e}")),
            "candidate_f32": actual.get(index).map(|value| format!("{value:.9e}")),
        })),
    })
}

fn targeted_stage_i32(name: &str, expected: &[i32], actual: &[i32]) -> serde_json::Value {
    let mismatch = first_mismatch(expected, actual);
    serde_json::json!({
        "stage": name,
        "exact": mismatch.is_none(),
        "elements": {"reference": expected.len(), "candidate": actual.len()},
        "digest": {"reference": format!("{:016x}", fnv1a_i32(expected)), "candidate": format!("{:016x}", fnv1a_i32(actual))},
        "first_mismatch": mismatch.map(|index| serde_json::json!({
            "index": index,
            "reference": expected.get(index),
            "candidate": actual.get(index),
        })),
    })
}

fn targeted_stage_u32(name: &str, expected: &[u32], actual: &[u32]) -> serde_json::Value {
    let expected_i32 = expected
        .iter()
        .map(|&value| value as i32)
        .collect::<Vec<_>>();
    let actual_i32 = actual.iter().map(|&value| value as i32).collect::<Vec<_>>();
    let mismatch = first_mismatch(expected, actual);
    serde_json::json!({
        "stage": name,
        "exact": mismatch.is_none(),
        "elements": {"reference": expected.len(), "candidate": actual.len()},
        "digest": {"reference": format!("{:016x}", fnv1a_i32(&expected_i32)), "candidate": format!("{:016x}", fnv1a_i32(&actual_i32))},
        "first_mismatch": mismatch.map(|index| serde_json::json!({
            "index": index,
            "reference": expected.get(index),
            "candidate": actual.get(index),
        })),
    })
}

fn check_attention_audit(
    layer: usize,
    expected: &AttentionAudit,
    actual: &AttentionAudit,
) -> Result<(), Box<dyn Error>> {
    for position in 0..2 {
        let comparable = |stages: &[(&'static str, u64)]| {
            stages
                .iter()
                .copied()
                .filter(|(name, _)| {
                    !matches!(
                        *name,
                        "sparse_raw_scores"
                            | "sparse_probabilities"
                            | "sparse_denominators"
                            | "sparse_valid_counts"
                    )
                })
                .collect::<Vec<_>>()
        };
        let expected_stages = comparable(&expected.stage_digests[position]);
        let actual_stages = comparable(&actual.stage_digests[position]);
        if expected_stages.len() != actual_stages.len() {
            return Err(format!(
                "CDQW layer {layer} position {position} stage audit length mismatch"
            )
            .into());
        }
        for ((expected_name, expected_digest), (actual_name, actual_digest)) in
            expected_stages.iter().zip(&actual_stages)
        {
            if expected_name != actual_name || expected_digest != actual_digest {
                return Err(format!(
                    "CDQW layer {layer} position {position} first stage mismatch: \
                     A {expected_name}={expected_digest:016x}, \
                     B {actual_name}={actual_digest:016x}"
                )
                .into());
            }
        }
        if let Some(index) = first_mismatch(&expected.outputs[position], &actual.outputs[position])
        {
            return Err(format!(
                "CDQW layer {layer} position {position} attention_post first mismatch at {index}"
            )
            .into());
        }
        if expected.topk_digests[position] != actual.topk_digests[position] {
            return Err(format!(
                "CDQW layer {layer} position {position} sparse top-k digest mismatch"
            )
            .into());
        }
    }
    for (stage, mismatch) in [
        ("kv", first_mismatch(&expected.kv, &actual.kv)),
        (
            "main_compressor_kv",
            first_mismatch(&expected.main_kv, &actual.main_kv),
        ),
        (
            "main_compressor_score",
            first_mismatch(&expected.main_score, &actual.main_score),
        ),
    ] {
        if let Some(index) = mismatch {
            return Err(format!("CDQW layer {layer} {stage} first mismatch at {index}").into());
        }
    }
    match (&expected.indexer_kv, &actual.indexer_kv) {
        (Some(expected), Some(actual)) => {
            if let Some(index) = first_mismatch(expected, actual) {
                return Err(format!(
                    "CDQW layer {layer} indexer_compressor_kv mismatch at {index}"
                )
                .into());
            }
        }
        (None, None) => {}
        _ => return Err(format!("CDQW layer {layer} indexer KV presence mismatch").into()),
    }
    match (&expected.indexer_score, &actual.indexer_score) {
        (Some(expected), Some(actual)) => {
            if let Some(index) = first_mismatch(expected, actual) {
                return Err(format!(
                    "CDQW layer {layer} indexer_compressor_score mismatch at {index}"
                )
                .into());
            }
        }
        (None, None) => {}
        _ => return Err(format!("CDQW layer {layer} indexer score presence mismatch").into()),
    }
    Ok(())
}

fn run_attention_arm_a_timed(
    context: &std::sync::Arc<CudaContext>,
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    runtime: &LoadedLayerRuntime,
    inputs: &[Vec<u16>; 2],
) -> Result<(AttentionAudit, u64, u64), Box<dyn Error>> {
    let start = context.new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
    let end = context.new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
    start.record(stream)?;
    let (audit, wall_ns) = run_attention_arm_a(module, stream, runtime, inputs)?;
    end.record(stream)?;
    end.synchronize()?;
    Ok((
        audit,
        wall_ns,
        (start.elapsed_ms(&end)? as f64 * 1_000_000.0) as u64,
    ))
}

fn gpu_telemetry_snapshot() -> Result<serde_json::Value, Box<dyn Error>> {
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=temperature.gpu,power.draw,clocks.sm,clocks.mem",
            "--format=csv,noheader,nounits",
            "--id=0",
        ])
        .output()?;
    if !output.status.success() {
        return Err("nvidia-smi telemetry query failed".into());
    }
    let fields = std::str::from_utf8(&output.stdout)?
        .lines()
        .next()
        .ok_or("nvidia-smi telemetry is empty")?
        .split(',')
        .map(str::trim)
        .collect::<Vec<_>>();
    if fields.len() != 4 {
        return Err("nvidia-smi telemetry format changed".into());
    }
    Ok(serde_json::json!({
        "temperature_c": fields[0],
        "power_w": fields[1],
        "sm_clock_mhz": fields[2],
        "memory_clock_mhz": fields[3],
    }))
}

struct Qd4StationaryGatePosition {
    position: usize,
    route_ids: Vec<i32>,
    route_weight_bits: Vec<u32>,
    attention_topk_digest: u64,
    attention_topk_per_row: usize,
    after_attention_digest: u64,
    ffn_input_digest: u64,
    shared_digest: u64,
    contribution_digests: Vec<u64>,
    contributions: Vec<Vec<f32>>,
    moe_digest: u64,
    hidden_digest: u64,
    state: v4_state::V4LayerPersistentState,
}

struct Qd4StationaryGateArm {
    positions: Vec<Qd4StationaryGatePosition>,
    routed_wall_ns: u64,
    routed_source_bytes: usize,
    routed_h2d_bytes: usize,
    startup_setup_ns: u64,
    steady_slot_allocations: usize,
    device_used_after_startup_bytes: usize,
    worker_errors: usize,
    immutable_cache: Option<Qd4ImmutableExpertCacheTelemetry>,
}

fn qd4_gate_state_json(state: &v4_state::V4LayerPersistentState) -> serde_json::Value {
    let main = state.compressor();
    let indexer = state.indexer();
    serde_json::json!({
        "kv_bf16_digest": format!("{:016x}", fnv1a_u16(state.kv_bf16())),
        "compressed_kv_bf16_digest": format!("{:016x}", fnv1a_u16(state.compressed_kv_bf16())),
        "indexer_kv_bf16_digest": format!("{:016x}", fnv1a_u16(state.indexer_kv_bf16())),
        "main_compressor_kv_f32_digest": format!("{:016x}", fnv1a_f32(main.kv())),
        "main_compressor_score_f32_digest": format!("{:016x}", fnv1a_f32(main.score())),
        "indexer_compressor_kv_f32_digest": indexer.map(|value| format!("{:016x}", fnv1a_f32(value.kv()))),
        "indexer_compressor_score_f32_digest": indexer.map(|value| format!("{:016x}", fnv1a_f32(value.score()))),
        "committed_kv_rows": state.committed_kv_rows(),
    })
}

fn qd4_gate_position_json(
    layer: usize,
    pair: usize,
    position: &Qd4StationaryGatePosition,
) -> serde_json::Value {
    serde_json::json!({
        "event": "position",
        "layer": layer,
        "pair": pair,
        "token_position": position.position,
        "route_ids": position.route_ids,
        "route_weight_bits": position.route_weight_bits,
        "attention_topk_digest": format!("{:016x}", position.attention_topk_digest),
        "attention_topk_per_row": position.attention_topk_per_row,
        "after_attention_bf16_digest": format!("{:016x}", position.after_attention_digest),
        "ffn_input_bf16_digest": format!("{:016x}", position.ffn_input_digest),
        "shared_f32_digest": format!("{:016x}", position.shared_digest),
        "route_slot_contribution_f32_digests": position.contribution_digests.iter().map(|value| format!("{value:016x}")).collect::<Vec<_>>(),
        "canonical_moe_bf16_digest": format!("{:016x}", position.moe_digest),
        "post_hc_hidden_bf16_digest": format!("{:016x}", position.hidden_digest),
        "persistent_state": qd4_gate_state_json(&position.state),
        "route_slot_f32_values_checked_bitwise": true,
    })
}

fn qd4_gate_f32_bits_equal(left: &[f32], right: &[f32]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(left, right)| left.to_bits() == right.to_bits())
}

fn qd4_stationary_control_first(pair: usize) -> bool {
    matches!(pair, 0 | 3)
}

fn check_qd4_stationary_gate_pair(
    layer: usize,
    control: &Qd4StationaryGateArm,
    qd4: &Qd4StationaryGateArm,
) -> Result<(), Box<dyn Error>> {
    if control.positions.len() != 5 || qd4.positions.len() != 5 {
        return Err(format!("QD4 stationary layer {layer} did not produce five positions").into());
    }
    for (position, (control, qd4)) in control.positions.iter().zip(&qd4.positions).enumerate() {
        if control.position != position
            || qd4.position != position
            || control.route_ids != qd4.route_ids
            || control.route_weight_bits != qd4.route_weight_bits
            || control.attention_topk_digest != qd4.attention_topk_digest
            || control.attention_topk_per_row != qd4.attention_topk_per_row
            || control.after_attention_digest != qd4.after_attention_digest
            || control.ffn_input_digest != qd4.ffn_input_digest
            || control.shared_digest != qd4.shared_digest
            || control.moe_digest != qd4.moe_digest
            || control.hidden_digest != qd4.hidden_digest
            || !control.state.bits_equal(&qd4.state)
            || control.contribution_digests != qd4.contribution_digests
            || control.contributions.len() != v4_router::V4_TOPK
            || qd4.contributions.len() != v4_router::V4_TOPK
            || !control
                .contributions
                .iter()
                .zip(&qd4.contributions)
                .all(|(control, qd4)| qd4_gate_f32_bits_equal(control, qd4))
        {
            return Err(format!(
                "QD4 stationary layer {layer} position {position} changed canonical routed execution"
            )
            .into());
        }
    }
    Ok(())
}

fn check_qd4_immutable_cache_gate_physical(
    layer: usize,
    arm: &Qd4StationaryGateArm,
) -> Result<(), Box<dyn Error>> {
    let cache = arm
        .immutable_cache
        .as_ref()
        .ok_or("QD4 immutable cache gate candidate omitted cache telemetry")?;
    let expected_routes = 5 * v4_router::V4_TOPK;
    let expert_bytes = v4_expert::V4_ROUTED_EXPERT_BYTES;
    let source_bytes = cache
        .source_loads
        .checked_mul(expert_bytes)
        .ok_or("QD4 immutable cache gate source byte count overflow")?;
    let d2d_bytes = expected_routes
        .checked_mul(expert_bytes)
        .ok_or("QD4 immutable cache gate D2D byte count overflow")?;
    if cache.route_executions != expected_routes
        || cache.source_loads != cache.expert_union_loads
        || cache.source_bytes != source_bytes
        || cache.h2d_loads != cache.source_loads
        || cache.h2d_bytes != cache.source_bytes
        || cache.hits != expected_routes.saturating_sub(cache.source_loads)
        || cache.d2d_copies != expected_routes
        || cache.d2d_bytes != d2d_bytes
        || cache.max_occupancy > V4_QD4_IMMUTABLE_EXPERT_CACHE_SLOTS
        || cache.completed_layers != 1
        || cache.worker_errors != 0
        || cache.fallbacks != 0
        || arm.routed_source_bytes != cache.source_bytes
        || arm.routed_h2d_bytes != cache.h2d_bytes
        || arm.worker_errors != 0
    {
        return Err(format!(
            "QD4 immutable cache gate layer {layer} physical telemetry did not close"
        )
        .into());
    }
    Ok(())
}

fn qd4_gate_prompt_ids(model_dir: &std::path::Path) -> Result<Vec<i64>, Box<dyn Error>> {
    let tokenizer = v4_tokenizer::V4Tokenizer::open(model_dir)?;
    let prompt = v4_chat::format_single_user("hi", v4_chat::ThinkingMode::Chat);
    let ids = tokenizer.encode_raw(&prompt)?;
    if ids.as_slice() != [0, 128_803, 6_366, 128_804, 128_822] {
        return Err("QD4 stationary gate official hi prompt changed".into());
    }
    Ok(ids)
}

fn qd4_gate_initial_hidden(
    model_dir: &std::path::Path,
    token_ids: &[i64],
) -> Result<Vec<Vec<u16>>, Box<dyn Error>> {
    token_ids
        .iter()
        .map(|&token_id| {
            let embedding = v4_state::load(model_dir, &[token_id])?.embedding;
            let mut hidden = Vec::with_capacity(v4_state::HC * v4_state::HIDDEN);
            for _ in 0..v4_state::HC {
                hidden.extend_from_slice(&embedding);
            }
            Ok(hidden)
        })
        .collect()
}

fn prepare_qd4_stationary_gate_inputs(
    context: &std::sync::Arc<CudaContext>,
    module: &kernels::LoadedModule,
    stream: &std::sync::Arc<CudaStream>,
    model_dir: &std::path::Path,
    token_ids: &[i64],
    target_layer: usize,
) -> Result<BTreeMap<usize, Vec<Vec<u16>>>, Box<dyn Error>> {
    if !(3..=4).contains(&target_layer) {
        return Err("QD4 stationary prefix supports only L3 or L4 targets".into());
    }
    let mut hidden = qd4_gate_initial_hidden(model_dir, token_ids)?;
    let mut persistent = v4_state::V4SerialPersistentState::new(token_ids.len())?;
    let mut inputs = BTreeMap::new();
    for layer in 0..target_layer {
        inputs.insert(layer, hidden.clone());
        let (runtime, _) = load_layer_runtime(model_dir, layer, token_ids, stream)?;
        let hash_ids = load_layer_hash_ids(model_dir, layer, token_ids)?;
        // ponytail: this is one cold prefix only; it mirrors current per-layer
        // union retention rather than adding a second input-cache mechanism.
        let mut expert_slots = BTreeMap::new();
        for position in 0..token_ids.len() {
            let row_hash_ids = if runtime.weights.hash_router {
                &hash_ids[position * v4_router::V4_TOPK..(position + 1) * v4_router::V4_TOPK]
            } else {
                &[]
            };
            let mut next = None;
            persistent.transaction(layer, |state| {
                next = Some(
                    run_real_layer_rows(
                        context,
                        module,
                        stream,
                        None,
                        &runtime,
                        None,
                        None,
                        Some(&mut expert_slots),
                        row_hash_ids,
                        &hidden[position],
                        &[token_ids[position]],
                        position,
                        Some(state),
                        None,
                        None,
                        false,
                        None,
                        false,
                        None,
                        None,
                    )?
                    .output,
                );
                Ok(())
            })?;
            hidden[position] = next.ok_or("QD4 stationary prefix did not produce hidden")?;
        }
    }
    inputs.insert(target_layer, hidden);
    Ok(inputs)
}

fn qd4_stationary_gate_arm_json(arm: &Qd4StationaryGateArm) -> serde_json::Value {
    serde_json::json!({
        "routed_wall_ns": arm.routed_wall_ns,
        "routed_source_bytes": arm.routed_source_bytes,
        "routed_h2d_bytes": arm.routed_h2d_bytes,
        "startup_setup_ns": arm.startup_setup_ns,
        "steady_slot_allocations": arm.steady_slot_allocations,
        "device_used_after_startup_bytes": arm.device_used_after_startup_bytes,
        "worker_errors": arm.worker_errors,
        "immutable_cache": arm.immutable_cache.as_ref().map(|cache| serde_json::json!({
            "route_executions": cache.route_executions,
            "expert_union_loads": cache.expert_union_loads,
            "source_loads": cache.source_loads,
            "source_bytes": cache.source_bytes,
            "h2d_loads": cache.h2d_loads,
            "h2d_bytes": cache.h2d_bytes,
            "hits": cache.hits,
            "d2d_copies": cache.d2d_copies,
            "d2d_bytes": cache.d2d_bytes,
            "max_occupancy": cache.max_occupancy,
            "completed_layers": cache.completed_layers,
            "worker_errors": cache.worker_errors,
            "fallbacks": cache.fallbacks,
            "slots": V4_QD4_IMMUTABLE_EXPERT_CACHE_SLOTS,
        })),
    })
}

#[derive(Clone, Copy)]
struct LinuxMemorySnapshot {
    available_bytes: u64,
    cached_bytes: u64,
}

fn linux_memory_snapshot() -> Result<LinuxMemorySnapshot, Box<dyn Error>> {
    let contents = std::fs::read_to_string("/proc/meminfo")?;
    let value = |name: &str| -> Result<u64, Box<dyn Error>> {
        let kib = contents
            .lines()
            .find_map(|line| line.strip_prefix(name))
            .and_then(|tail| tail.split_whitespace().next())
            .ok_or_else(|| format!("{name} missing from /proc/meminfo"))?
            .parse::<u64>()?;
        kib.checked_mul(1_024)
            .ok_or_else(|| format!("{name} byte count overflow").into())
    };
    Ok(LinuxMemorySnapshot {
        available_bytes: value("MemAvailable:")?,
        cached_bytes: value("Cached:")?,
    })
}

fn model_mount_provenance(
    model_dir: &std::path::Path,
) -> Result<serde_json::Value, Box<dyn Error>> {
    let canonical = model_dir.canonicalize()?;
    let mountinfo = std::fs::read_to_string("/proc/self/mountinfo")?;
    let (_, source, fstype) = mountinfo
        .lines()
        .filter_map(|line| {
            let (mount, filesystem) = line.split_once(" - ")?;
            let target = mount.split_whitespace().nth(4)?;
            let mut filesystem = filesystem.split_whitespace();
            let fstype = filesystem.next()?;
            let source = filesystem.next()?;
            canonical
                .starts_with(target)
                .then_some((target.len(), source, fstype))
        })
        .max_by_key(|(length, _, _)| *length)
        .ok_or("model mount provenance is unavailable")?;
    let source_canonical = std::fs::canonicalize(source)?;
    let uuid = std::fs::read_dir("/dev/disk/by-uuid")?
        .filter_map(Result::ok)
        .find_map(|entry| {
            std::fs::canonicalize(entry.path())
                .ok()
                .filter(|candidate| candidate == &source_canonical)
                .and_then(|_| entry.file_name().into_string().ok())
        })
        .ok_or("model block device UUID is unavailable")?;
    Ok(serde_json::json!({
        "source": source,
        "fstype": fstype,
        "uuid": uuid,
    }))
}

struct CacheOffSource {
    layer: usize,
    shard: safetensors::ShardFile,
    raw_name: String,
    raw: Vec<u8>,
    plan: v4_expert::RoutedExpertRangePlan,
    routed: [Vec<u8>; 6],
}

impl CacheOffSource {
    fn new(model_dir: &std::path::Path, layer: usize) -> Result<Self, Box<dyn Error>> {
        let shard = safetensors::ShardFile::open(
            model_dir.join(format!("model-{:05}-of-00064.safetensors", layer + 2)),
        )?;
        let raw_name = format!("layers.{layer}.attn.wo_a.weight");
        let raw = vec![0_u8; usize::try_from(shard.tensor(&raw_name)?.byte_len)?];
        let plan = v4_expert::RoutedExpertRangePlan::new(&shard, layer, 0)?;
        let routed = [
            vec![0_u8; v4_expert::V4_W1_BYTES],
            vec![0_u8; v4_expert::V4_W1_SCALE_BYTES],
            vec![0_u8; v4_expert::V4_W2_BYTES],
            vec![0_u8; v4_expert::V4_W2_SCALE_BYTES],
            vec![0_u8; v4_expert::V4_W1_BYTES],
            vec![0_u8; v4_expert::V4_W1_SCALE_BYTES],
        ];
        Ok(Self {
            layer,
            shard,
            raw_name,
            raw,
            plan,
            routed,
        })
    }

    fn evict(&self) -> Result<(), Box<dyn Error>> {
        self.shard
            .advise_range_dontneed(&self.raw_name, 0, self.raw.len())?;
        self.plan.advise_dontneed(&self.shard)
    }

    fn read(
        &mut self,
        mode: safetensors::RangeReadMode,
    ) -> Result<safetensors::RangeReadTelemetry, Box<dyn Error>> {
        let raw = self
            .shard
            .read_at_mode(&self.raw_name, 0, &mut self.raw, mode)?;
        let [w1, w1_scale, w2, w2_scale, w3, w3_scale] = &mut self.routed;
        let routed = self.plan.read_into_mode(
            &self.shard,
            [
                w1.as_mut_slice(),
                w1_scale.as_mut_slice(),
                w2.as_mut_slice(),
                w2_scale.as_mut_slice(),
                w3.as_mut_slice(),
                w3_scale.as_mut_slice(),
            ],
            mode,
        )?;
        Ok(safetensors::RangeReadTelemetry {
            logical_bytes: raw
                .logical_bytes
                .checked_add(routed.logical_bytes)
                .ok_or("cache-off logical byte count overflow")?,
            physical_bytes: raw
                .physical_bytes
                .checked_add(routed.physical_bytes)
                .ok_or("cache-off physical byte count overflow")?,
        })
    }

    fn update_digest(&self, mut hash: u64) -> u64 {
        for byte in self.raw.iter().chain(self.routed.iter().flatten()) {
            hash = (hash ^ u64::from(*byte)).wrapping_mul(0x100_0000_01b3);
        }
        hash
    }
}

#[derive(Clone, Copy)]
struct CacheOffSample {
    read_wall_ns: u64,
    logical_bytes: usize,
    physical_bytes: usize,
    digest: u64,
    available_delta_bytes: i64,
    cached_delta_bytes: i64,
}

fn run_cache_off_arm(
    sources: &mut [CacheOffSource],
    mode: safetensors::RangeReadMode,
) -> Result<CacheOffSample, Box<dyn Error>> {
    for source in sources.iter() {
        source.evict()?;
    }
    let before = linux_memory_snapshot()?;
    let started = std::time::Instant::now();
    let mut logical_bytes = 0_usize;
    let mut physical_bytes = 0_usize;
    for source in sources.iter_mut() {
        let read = source.read(mode)?;
        logical_bytes = logical_bytes
            .checked_add(read.logical_bytes)
            .ok_or("cache-off logical byte total overflow")?;
        physical_bytes = physical_bytes
            .checked_add(read.physical_bytes)
            .ok_or("cache-off physical byte total overflow")?;
    }
    let read_wall_ns = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
    let after = linux_memory_snapshot()?;
    let digest = sources
        .iter()
        .fold(0xcbf2_9ce4_8422_2325_u64, |hash, source| {
            source.update_digest(hash)
        });
    Ok(CacheOffSample {
        read_wall_ns,
        logical_bytes,
        physical_bytes,
        digest,
        available_delta_bytes: i64::try_from(after.available_bytes)?
            - i64::try_from(before.available_bytes)?,
        cached_delta_bytes: i64::try_from(after.cached_bytes)?
            - i64::try_from(before.cached_bytes)?,
    })
}

fn median_i64(samples: &[i64]) -> i64 {
    let mut samples = samples.to_vec();
    samples.sort_unstable();
    let middle = samples.len() / 2;
    if samples.len().is_multiple_of(2) {
        (samples[middle - 1] as i128 + samples[middle] as i128) as i64 / 2
    } else {
        samples[middle]
    }
}

fn run_device_attention_row(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    runtime: &LoadedLayerRuntime,
    state: &mut DeviceTokenState,
    rope_lut: &DeviceRequestRopeLut,
    hc_hidden: &[u16],
    start_pos: usize,
    observe_topk: bool,
) -> Result<RealAttentionRun, Box<dyn Error>> {
    let started = std::time::Instant::now();
    let rope_lut_d2d_count = run_device_token_state_commit_row(
        module, stream, runtime, state, rope_lut, hc_hidden, start_pos,
    )?;
    let commit_ns = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
    let (output, topk) = run_device_attention_after_commit(
        module,
        stream,
        runtime,
        state,
        rope_lut,
        start_pos,
        observe_topk,
    )?;
    let attention_topk_digest = topk.as_deref().map_or(0, fnv1a_i32);
    let attention_topk_per_row = topk.as_ref().map_or(0, Vec::len);
    let audit_topk_bytes = topk.as_ref().map_or(0, Vec::len) * std::mem::size_of::<i32>();
    let attention_ns = u64::try_from(started.elapsed().as_nanos())
        .unwrap_or(u64::MAX)
        .saturating_sub(commit_ns);
    Ok(RealAttentionRun {
        output,
        topk,
        stage_digests: Vec::new(),
        attention_topk_digest,
        attention_topk_per_row,
        indexer_topk_gpu_ns: 0,
        attention_ns,
        commit_ns,
        device_transfer: Some(DeviceAttentionTransfer {
            h2d_count: 1,
            h2d_bytes: state.hc_hidden.num_bytes(),
            rope_lut_d2d_count: rope_lut_d2d_count + 1,
            rope_lut_d2d_bytes: (rope_lut_d2d_count + 1) * state.freqs.num_bytes(),
            terminal_d2h_count: 1,
            terminal_d2h_bytes: state.after_attention.num_bytes(),
            intermediate_or_audit_d2h_count: usize::from(observe_topk),
            intermediate_or_audit_d2h_bytes: audit_topk_bytes,
        }),
        profile: None,
    })
}

fn flash_dspark_main_x(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    stage0: &LoadedDsparkStage,
    main_hidden: &[u16],
) -> Result<Vec<u16>, Box<dyn Error>> {
    let spec = &v4_spec::FLASH_0731;
    if stage0.stage != 0 || main_hidden.len() != 3 * spec.hidden {
        return Err("Flash-0731 DSpark main hidden contract mismatch".into());
    }
    let projection = stage0
        .main_proj
        .as_ref()
        .ok_or("Flash-0731 stage 0 main_proj is missing")?;
    let norm = stage0
        .main_norm
        .as_ref()
        .ok_or("Flash-0731 stage 0 main_norm is missing")?;
    let projected = run_l0_fp8_linear(module, stream, main_hidden, projection, false)?.1;
    run_l0_rmsnorm(module, stream, &projected, norm, spec.hidden)
}

fn flash_dspark_context_kv(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    stage: &LoadedDsparkStage,
    main_x: &[u16],
    position: usize,
) -> Result<Vec<u16>, Box<dyn Error>> {
    let spec = &v4_spec::FLASH_0731;
    if main_x.len() != spec.hidden {
        return Err("Flash-0731 DSpark main_x contract mismatch".into());
    }
    let kv = run_l0_fp8_linear(module, stream, main_x, &stage.weights.wkv, false)?.1;
    let kv = run_l0_rmsnorm(module, stream, &kv, &stage.weights.kv_norm, spec.q_head_dim)?;
    let kv = run_l0_rope(
        module,
        stream,
        &kv,
        &v4_state::attention_rope_freqs(spec, stage.weights.layer, position)?,
        1,
        false,
    )?;
    run_l0_kv_quantize(module, stream, &kv)
}

fn advance_flash_dspark_context(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    stages: &[LoadedDsparkStage; v4_state::DSPARK_STAGES],
    states: &mut [v4_state::V4LayerPersistentState; v4_state::DSPARK_STAGES],
    main_x: &[u16],
    position: usize,
) -> Result<(), Box<dyn Error>> {
    for (stage, state) in stages.iter().zip(states) {
        let kv = flash_dspark_context_kv(module, stream, stage, main_x, position)?;
        state.append_kv_row(position, &kv)?;
    }
    Ok(())
}

fn flash_dspark_moe(
    context: &std::sync::Arc<CudaContext>,
    module: &kernels::LoadedModule,
    stream: &std::sync::Arc<CudaStream>,
    stage: &LoadedDsparkStage,
    after_attention: &[u16],
) -> Result<Vec<u16>, Box<dyn Error>> {
    let spec = &v4_spec::FLASH_0731;
    let rows = after_attention.len() / (v4_state::HC * spec.hidden);
    if rows != v4_state::DSPARK_B5_DRAFT_TOKENS {
        return Err("Flash-0731 DSpark MoE requires exactly five rows".into());
    }
    let ffn_hc = run_l0_mhc_pre(
        module,
        stream,
        after_attention,
        &stage.weights.hc_ffn_fn,
        &stage.weights.hc_ffn_scale,
        &stage.weights.hc_ffn_base,
    )?;
    let router = run_layer_router(
        module,
        stream,
        None,
        &ffn_hc.collapsed,
        &stage.weights,
        &[],
        false,
    )?;
    let shared = run_layer_shared_expert(module, stream, &router.input, &stage.weights)?;
    let mut contributions = (0..rows)
        .map(|_| {
            (0..spec.experts_per_token)
                .map(|_| None)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<Vec<Option<Vec<f32>>>>>();
    let mut experts = router
        .routes
        .ids
        .iter()
        .map(|&expert| u16::try_from(expert))
        .collect::<Result<Vec<_>, _>>()?;
    experts.sort_unstable();
    experts.dedup();
    for expert_id in experts {
        let expert =
            v4_expert::RoutedExpert::load_dspark(&stage.expert_shard, stage.stage, expert_id)?;
        let mut slot = ExpertSlot::new(context, module, stream, &expert)?;
        for (row, row_contributions) in contributions.iter_mut().enumerate() {
            for (route_slot, contribution) in row_contributions.iter_mut().enumerate() {
                let route = row * spec.experts_per_token + route_slot;
                if router.routes.ids[route] == i32::from(expert_id) {
                    *contribution = Some(
                        slot.execute(
                            &router.input[row * spec.hidden..(row + 1) * spec.hidden],
                            router.routes.weights[route],
                        )?
                        .0,
                    );
                }
            }
        }
    }
    let mut moe = Vec::with_capacity(rows * spec.hidden);
    for (row, row_contributions) in contributions.iter().enumerate() {
        let routed = row_contributions
            .iter()
            .enumerate()
            .map(|(slot, output)| {
                Ok((
                    u16::try_from(router.routes.ids[row * spec.experts_per_token + slot])?,
                    output
                        .as_deref()
                        .ok_or("Flash-0731 DSpark routed contribution is missing")?,
                ))
            })
            .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
        moe.extend(v4_expert::reduce_moe_canonical_for(
            spec,
            &routed,
            &shared[row * spec.hidden..(row + 1) * spec.hidden],
        )?);
    }
    run_l0_hc_post(
        module,
        stream,
        &moe,
        after_attention,
        &ffn_hc.post,
        &ffn_hc.comb,
    )
}

fn run_flash_dspark_stage(
    context: &std::sync::Arc<CudaContext>,
    module: &kernels::LoadedModule,
    stream: &std::sync::Arc<CudaStream>,
    stage: &LoadedDsparkStage,
    state: &mut v4_state::V4LayerPersistentState,
    x: &[u16],
    main_x: &[u16],
    start_pos: usize,
) -> Result<Vec<u16>, Box<dyn Error>> {
    let spec = &v4_spec::FLASH_0731;
    let rows = v4_state::DSPARK_B5_DRAFT_TOKENS;
    if x.len() != rows * v4_state::HC * spec.hidden
        || state.layer() != spec.layers + stage.stage
        || state.committed_kv_rows() != start_pos
    {
        return Err("Flash-0731 DSpark stage state/input contract mismatch".into());
    }
    let attn_hc = run_l0_mhc_pre(
        module,
        stream,
        x,
        &stage.weights.hc_attn_fn,
        &stage.weights.hc_attn_scale,
        &stage.weights.hc_attn_base,
    )?;
    let attn_input = run_l0_rmsnorm(
        module,
        stream,
        &attn_hc.collapsed,
        &stage.weights.attn_norm,
        spec.hidden,
    )?;
    let q_lora = run_l0_fp8_linear(module, stream, &attn_input, &stage.weights.wq_a, false)?.1;
    let q_lora = run_l0_rmsnorm(
        module,
        stream,
        &q_lora,
        &stage.weights.q_norm,
        spec.q_lora_rank,
    )?;
    let q = run_l0_fp8_linear(module, stream, &q_lora, &stage.weights.wq_b, true)?.1;
    let unit_norm = DeviceBuffer::<u16>::zeroed(stream, 0)?;
    let q = run_l0_rmsnorm(module, stream, &q, &unit_norm, spec.q_head_dim)?;
    let mut query_freqs = Vec::with_capacity(rows * v4_state::Q_ROPE_DIM);
    for offset in 1..=rows {
        query_freqs.extend(v4_state::attention_rope_freqs(
            spec,
            stage.weights.layer,
            start_pos + offset,
        )?);
    }
    let q = run_l0_rope(module, stream, &q, &query_freqs, spec.q_heads, false)?;
    let query_kv = run_l0_fp8_linear(module, stream, &attn_input, &stage.weights.wkv, false)?.1;
    let query_kv = run_l0_rmsnorm(
        module,
        stream,
        &query_kv,
        &stage.weights.kv_norm,
        spec.q_head_dim,
    )?;
    let query_kv = run_l0_rope(module, stream, &query_kv, &query_freqs, 1, false)?;
    let query_kv = run_l0_kv_quantize(module, stream, &query_kv)?;
    let main_kv = flash_dspark_context_kv(module, stream, stage, main_x, start_pos)?;
    let mut cache = state.attention_cache_bf16();
    let slot = start_pos % v4_state::KV_WINDOW;
    cache[slot * spec.q_head_dim..(slot + 1) * spec.q_head_dim].copy_from_slice(&main_kv);
    cache.extend_from_slice(&query_kv);
    let topk = v4_state::flash_dspark_b5_topk(start_pos)?;
    let sparse = run_sparse_attention(module, stream, &q, &cache, &stage.weights.attn_sink, &topk)?;
    let sparse = run_l0_rope(module, stream, &sparse, &query_freqs, spec.q_heads, true)?;
    let wo_a = run_l0_wo_a_for(spec, module, stream, &sparse, &stage.weights.wo_a)?;
    let attention = run_l0_fp8_linear(module, stream, &wo_a, &stage.weights.wo_b, false)?.1;
    let after_attention =
        run_l0_hc_post(module, stream, &attention, x, &attn_hc.post, &attn_hc.comb)?;
    let output = flash_dspark_moe(context, module, stream, stage, &after_attention)?;
    state.append_kv_row(start_pos, &main_kv)?;
    Ok(output)
}

struct MtpDraftRun {
    candidate: i64,
    routes: v4_router::RouterSelection,
    source_bytes: usize,
    routed_source_bytes: usize,
    source_read_ns: u64,
    h2d_ns: u64,
    h2d_bytes: usize,
    routed_h2d_bytes: usize,
    kernel_ns: u64,
    head_read_ns: u64,
    head_h2d_ns: u64,
    head_kernel_ns: u64,
    head_d2h_ns: u64,
    head_wall_ns: u64,
    total_ns: u64,
    persistent_bytes: usize,
    transient_expert_peak_bytes: usize,
    device_peak_bytes: usize,
    borrowed_qd4_lane: bool,
    kv_position: usize,
    kv_digest: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Mtp1Qd4LoanTelemetry {
    normal_lanes_before: usize,
    normal_lanes_after: usize,
    loan_depth_before: usize,
    loan_depth_peak: usize,
    loan_depth_final: usize,
    event_drained: bool,
}

struct Mtp1Qd4Attempt<T> {
    result: Result<T, Box<dyn Error>>,
    loan: Mtp1Qd4LoanTelemetry,
}

fn mtp1_qd4_telemetry_fields(
    route_ids: Option<&[i32]>,
    routed_source_bytes: Option<usize>,
    routed_h2d_bytes: Option<usize>,
    loan: Mtp1Qd4LoanTelemetry,
) -> Result<serde_json::Map<String, serde_json::Value>, Box<dyn Error>> {
    let mut fields = serde_json::Map::new();
    fields.insert(
        "qd4_normal_lanes_before".into(),
        serde_json::json!(loan.normal_lanes_before),
    );
    fields.insert(
        "qd4_normal_lanes_after".into(),
        serde_json::json!(loan.normal_lanes_after),
    );
    fields.insert(
        "qd4_loan_depth_before".into(),
        serde_json::json!(loan.loan_depth_before),
    );
    fields.insert(
        "qd4_loan_depth_peak".into(),
        serde_json::json!(loan.loan_depth_peak),
    );
    fields.insert(
        "qd4_loan_depth_final".into(),
        serde_json::json!(loan.loan_depth_final),
    );
    fields.insert(
        "qd4_loan_event_drained".into(),
        serde_json::json!(loan.event_drained),
    );
    if let Some(route_ids) = route_ids {
        let source_bytes =
            routed_source_bytes.ok_or("MTP routed source byte telemetry disappeared")?;
        let h2d_bytes = routed_h2d_bytes.ok_or("MTP routed H2D byte telemetry disappeared")?;
        if source_bytes % v4_expert::V4_ROUTED_EXPERT_BYTES != 0 {
            return Err("MTP routed source bytes do not contain whole expert loads".into());
        }
        if source_bytes != h2d_bytes {
            return Err("MTP routed source and H2D bytes diverged".into());
        }
        fields.insert("route_ids".into(), serde_json::json!(route_ids));
        fields.insert(
            "routed_expert_loads".into(),
            serde_json::json!(source_bytes / v4_expert::V4_ROUTED_EXPERT_BYTES),
        );
        fields.insert(
            "routed_source_bytes".into(),
            serde_json::json!(source_bytes),
        );
        fields.insert("routed_h2d_bytes".into(), serde_json::json!(h2d_bytes));
    } else if routed_source_bytes.is_some() || routed_h2d_bytes.is_some() {
        return Err("MTP routed bytes require route IDs".into());
    }
    Ok(fields)
}

fn append_mtp1_qd4_telemetry(
    record: &mut serde_json::Value,
    route_ids: Option<&[i32]>,
    routed_source_bytes: Option<usize>,
    routed_h2d_bytes: Option<usize>,
    loan: Mtp1Qd4LoanTelemetry,
) -> Result<(), Box<dyn Error>> {
    record
        .as_object_mut()
        .ok_or("MTP telemetry record must be a JSON object")?
        .extend(mtp1_qd4_telemetry_fields(
            route_ids,
            routed_source_bytes,
            routed_h2d_bytes,
            loan,
        )?);
    Ok(())
}

struct MtpDraftPrefixRun {
    persistent_bytes: usize,
    kv_position: usize,
    kv_digest: u64,
}

enum MtpDraftStep {
    Prefix(Result<MtpDraftPrefixRun, Box<dyn Error>>),
    Candidate(Result<MtpDraftRun, Box<dyn Error>>),
}

fn mtp1_requires_full_candidate(prompt_len: usize, token_position: usize) -> bool {
    token_position + 1 >= prompt_len
}

/// One official MTP-1 row consumes the teacher's final hidden row and token at
/// `candidate_pos`, appends its KV at that same position, then predicts the
/// teacher's token at `candidate_pos + 1`.  Keep this separate from the P0
/// score cap: stationary acceptance never changes which rows are causal.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Mtp1Alignment {
    input_pos: usize,
    teacher_hidden_pos: usize,
    kv_write_pos: usize,
    candidate_pos: usize,
    teacher_target_pos: usize,
    prompt_forced: bool,
}

fn mtp1_alignment(
    prompt_len: usize,
    candidate_pos: usize,
) -> Result<Mtp1Alignment, Box<dyn Error>> {
    if prompt_len == 0 {
        return Err("MTP1 alignment needs a nonempty prompt".into());
    }
    let teacher_target_pos = candidate_pos
        .checked_add(1)
        .ok_or("MTP1 candidate position overflow")?;
    Ok(Mtp1Alignment {
        input_pos: candidate_pos,
        teacher_hidden_pos: candidate_pos,
        kv_write_pos: candidate_pos,
        candidate_pos,
        teacher_target_pos,
        prompt_forced: teacher_target_pos < prompt_len,
    })
}

fn stationary_mtp1_enabled(generic_stationary: bool, flag: bool) -> Result<bool, Box<dyn Error>> {
    if flag && !generic_stationary {
        return Err("COLIBRI_V4_STATIONARY_MTP1_DRAFT requires generic device stationary".into());
    }
    Ok(flag)
}

fn mtp1_can_borrow_qd4_lane(worker_drained: bool, qd4_lanes: usize) -> bool {
    worker_drained && qd4_lanes == V4_QD4_LANES
}

fn run_with_mtp1_qd4_loan<T>(
    mut qd4_lanes: Option<&mut [ExpertSlot<'_>]>,
    worker_drained: bool,
    run: impl FnOnce(Option<&mut ExpertSlot<'_>>) -> Result<T, Box<dyn Error>>,
) -> Mtp1Qd4Attempt<T> {
    let normal_lanes_before = qd4_lanes.as_deref().map_or(0, <[ExpertSlot]>::len);
    let lanes_present = qd4_lanes.is_some();
    let borrowed = lanes_present && mtp1_can_borrow_qd4_lane(worker_drained, normal_lanes_before);
    let mut event_drained = !lanes_present;
    let result = if lanes_present && !borrowed {
        Err("stationary MTP candidate requires a drained normal four-lane QD4 set".into())
    } else if let Some(lanes) = qd4_lanes.as_deref_mut() {
        let lane = &mut lanes[0];
        let result = run(Some(lane));
        let drain = lane.finish_qd4_loan();
        event_drained = drain.is_ok();
        match (result, drain) {
            (Ok(value), Ok(true)) => Ok(value),
            (Ok(_), Ok(false)) => Err("stationary MTP QD4 loan event did not drain".into()),
            (Ok(_), Err(error)) => Err(error),
            (Err(error), Ok(_)) => Err(error),
            (Err(error), Err(drain)) => {
                Err(format!("{error}; stationary MTP QD4 loan drain also failed: {drain}").into())
            }
        }
    } else {
        run(None)
    };
    let normal_lanes_after = qd4_lanes.as_deref().map_or(0, <[ExpertSlot]>::len);
    Mtp1Qd4Attempt {
        result,
        loan: Mtp1Qd4LoanTelemetry {
            normal_lanes_before,
            normal_lanes_after,
            loan_depth_before: 0,
            loan_depth_peak: usize::from(borrowed),
            loan_depth_final: 0,
            event_drained,
        },
    }
}

#[allow(clippy::too_many_arguments)]
fn advance_mtp1_draft_state(
    module: &kernels::LoadedModule,
    stream: &std::sync::Arc<CudaStream>,
    model_dir: &std::path::Path,
    teacher_hc_hidden: &[u16],
    input_id: i64,
    start_pos: usize,
    state: &mut MtpDraftState,
) -> Result<MtpDraftPrefixRun, Box<dyn Error>> {
    if state.position != start_pos || teacher_hc_hidden.len() != v4_state::HC * v4_state::HIDDEN {
        return Err("MTP1 draft state position or teacher hidden-row contract mismatch".into());
    }
    let host = v4_state::load_mtp_draft_prefix_weights(model_dir)?;
    let weights = upload_mtp_draft_prefix_weights(stream, &host)?;
    let embedding = load_mtp_embedding_row(model_dir, input_id)?;
    let enorm = DeviceBuffer::from_host(stream, &host.enorm)?;
    let hnorm = DeviceBuffer::from_host(stream, &host.hnorm)?;
    let e_input = run_l0_rmsnorm(module, stream, &embedding, &enorm, v4_state::HIDDEN)?;
    let e_projection = host.e_proj.load(&host.shard)?;
    let e_projection = DeviceFp8Matrix::new(stream, &e_projection)?;
    let e = run_l0_fp8_linear(module, stream, &e_input, &e_projection, false)?.1;
    let h_input = run_l0_rmsnorm(module, stream, teacher_hc_hidden, &hnorm, v4_state::HIDDEN)?;
    let h_projection = host.h_proj.load(&host.shard)?;
    let h_projection = DeviceFp8Matrix::new(stream, &h_projection)?;
    let h = run_l0_fp8_linear(module, stream, &h_input, &h_projection, false)?.1;
    let mut residual = Vec::with_capacity(teacher_hc_hidden.len());
    for h_row in h.chunks_exact(v4_state::HIDDEN) {
        residual.extend(h_row.iter().zip(&e).map(|(&h, &e)| {
            v4_router::f32_to_bf16_rne(v4_router::bf16_to_f32(h) + v4_router::bf16_to_f32(e))
        }));
    }
    let attn_hc = run_l0_mhc_pre(
        module,
        stream,
        &residual,
        &weights.hc_attn_fn,
        &weights.hc_attn_scale,
        &weights.hc_attn_base,
    )?;
    let attn_input = run_l0_rmsnorm(
        module,
        stream,
        &attn_hc.collapsed,
        &weights.attn_norm,
        v4_state::HIDDEN,
    )?;
    let (_, kv_raw) = run_l0_fp8_linear(module, stream, &attn_input, &weights.wkv, false)?;
    let kv_norm = run_l0_rmsnorm(
        module,
        stream,
        &kv_raw,
        &weights.kv_norm,
        v4_state::Q_HEAD_DIM,
    )?;
    let freqs = v4_state::attention_rope_freqs(&v4_spec::PRO, v4_state::V4_LAYER_COUNT, start_pos)?;
    let kv_rope = run_l0_rope(module, stream, &kv_norm, &freqs, 1, false)?;
    let kv = run_l0_kv_quantize(module, stream, &kv_rope)?;
    let kv_ready = DeviceBuffer::from_host(stream, &kv)?;
    state.commit_kv_row(stream, &kv_ready)?;
    Ok(MtpDraftPrefixRun {
        persistent_bytes: state.persistent_bytes(),
        kv_position: state.position,
        kv_digest: state.digest(stream)?,
    })
}

fn load_mtp_embedding_row(
    model_dir: &std::path::Path,
    input_id: i64,
) -> Result<Vec<u16>, Box<dyn Error>> {
    v4_state::validate_ids(&[input_id])?;
    let shard = safetensors::ShardFile::open(model_dir.join("model-00001-of-00064.safetensors"))?;
    let tensor = shard.tensor("embed.weight")?;
    if tensor.dtype != safetensors::TensorDtype::Bf16
        || tensor.shape != [v4_state::VOCAB as u64, v4_state::HIDDEN as u64]
    {
        return Err("MTP shared embedding tensor contract mismatch".into());
    }
    let mut embedding = vec![0_u16; v4_state::HIDDEN];
    let bytes = unsafe {
        std::slice::from_raw_parts_mut(embedding.as_mut_ptr().cast(), embedding.len() * 2)
    };
    shard.read_at(
        "embed.weight",
        u64::try_from(usize::try_from(input_id)? * v4_state::HIDDEN * 2)?,
        bytes,
    )?;
    Ok(embedding)
}

#[derive(Clone)]
struct L0Persistent {
    kv: v4_state::V4KvCache,
    compressor: v4_state::V4CompressorState,
}

impl L0Persistent {
    fn new() -> Result<Self, Box<dyn Error>> {
        Ok(Self {
            kv: v4_state::V4KvCache::new(),
            compressor: v4_state::V4CompressorState::new_decode(
                v4_state::COMPRESS_RATIO,
                v4_state::Q_HEAD_DIM,
            )?,
        })
    }

    fn bits_equal(&self, other: &Self) -> bool {
        self.kv.bits_equal(&other.kv) && self.compressor.bits_equal(&other.compressor)
    }
}

struct L0Run {
    output: Vec<u16>,
    routes: v4_router::RouterSelection,
    prefixes: Vec<L0Persistent>,
    persistent: L0Persistent,
    stage_hashes: Vec<(&'static str, u64)>,
    expert_union: Vec<u16>,
    expert_bytes: usize,
    expert_alloc_ms: f64,
    expert_read_ms: f64,
    expert_setup_ms: f64,
    expert_setup_samples_ms: Vec<f64>,
    expert_h2d_samples_ms: Vec<f64>,
    expert_kernel_ms: f64,
    dynamic_bytes: usize,
    elapsed_ms: f64,
}

fn fnv1a_u16(values: &[u16]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for value in values {
        for byte in value.to_le_bytes() {
            hash = (hash ^ byte as u64).wrapping_mul(0x100_0000_01b3);
        }
    }
    hash
}

fn fnv1a_bytes(values: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for &byte in values {
        hash = (hash ^ u64::from(byte)).wrapping_mul(0x100_0000_01b3);
    }
    hash
}

fn normalized_config_fingerprint(
    config: serde_json::Value,
) -> Result<serde_json::Value, Box<dyn Error>> {
    if !config.is_object() {
        return Err("normalized relevant configuration must be a JSON object".into());
    }
    let canonical_json = serde_json::to_string(&config)?;
    Ok(serde_json::json!({
        "algorithm": "fnv1a64",
        "digest": format!("{:016x}", fnv1a_bytes(canonical_json.as_bytes())),
        "canonical_json": canonical_json,
        "normalized": config,
    }))
}

fn fnv1a_i32(values: &[i32]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for value in values {
        for byte in value.to_le_bytes() {
            hash = (hash ^ byte as u64).wrapping_mul(0x100_0000_01b3);
        }
    }
    hash
}

fn fnv1a_f32(values: &[f32]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for value in values {
        for byte in value.to_bits().to_le_bytes() {
            hash = (hash ^ byte as u64).wrapping_mul(0x100_0000_01b3);
        }
    }
    hash
}

fn p95_f64(values: &[f64]) -> f64 {
    let mut values = values.to_vec();
    values.sort_by(f64::total_cmp);
    values[(values.len() * 95).div_ceil(100) - 1]
}

fn l0_persistent_digest(state: &L0Persistent) -> u64 {
    let mut hash = fnv1a_u16(state.kv.as_slice());
    for value in state.compressor.kv().iter().chain(state.compressor.score()) {
        for byte in value.to_bits().to_le_bytes() {
            hash = (hash ^ byte as u64).wrapping_mul(0x100_0000_01b3);
        }
    }
    hash
}

/// Actual host-to-device bytes issued by one host-boundary L0 P0 run.
/// CUDA memset/allocations and device-to-host trace copies are not H2D.
fn l0_dynamic_h2d_bytes(rows: usize) -> usize {
    let bf16 = std::mem::size_of::<u16>();
    let f32_bytes = std::mem::size_of::<f32>();
    let i32_bytes = std::mem::size_of::<i32>();
    let i64_bytes = std::mem::size_of::<i64>();
    let h = v4_state::HIDDEN;
    let hc_state = rows * v4_state::HC * h;
    let q = rows * v4_state::Q_HEADS * v4_state::Q_HEAD_DIM;
    let kv = rows * v4_state::Q_HEAD_DIM;
    let q_lora = rows * 1_536;
    let o_lora = rows * v4_state::O_GROUPS * v4_state::O_LORA_RANK;
    let expert_intermediate = rows * v4_expert::V4_EXPERT_INTERMEDIATE;
    let post = rows * v4_state::HC;
    let comb = rows * v4_state::HC * v4_state::HC;
    let freqs = rows * v4_state::Q_ROPE_DIM;
    let cache = v4_state::KV_CACHE_POSITIONS * v4_state::Q_HEAD_DIM;
    let topk = rows * 128;

    hc_state * bf16 // attention mHC
        + rows * h * bf16 // attention RMSNorm
        + rows * h * bf16 // wq_a
        + q_lora * bf16 // q_norm
        + q_lora * bf16 // wq_b
        + q * bf16 // per-head RMSNorm
        + q * bf16 + freqs * f32_bytes // forward q RoPE
        + rows * h * bf16 // wkv
        + kv * bf16 // kv_norm
        + kv * bf16 + freqs * f32_bytes // forward kv RoPE
        + kv * bf16 * 2 // fake-quant source and initialized destination
        + rows * h * bf16 // Compressor input
        + q * bf16 + cache * bf16 + topk * i32_bytes // sparse attention
        + q * bf16 + freqs * f32_bytes // inverse RoPE
        + q * bf16 // wo_a
        + o_lora * bf16 // wo_b
        + rows * h * bf16 + hc_state * bf16 + post * f32_bytes + comb * f32_bytes // attention HC post
        + hc_state * bf16 // FFN mHC
        + rows * h * bf16 // FFN RMSNorm
        + rows * h * bf16 + v4_router::V4_ROUTED_EXPERTS * f32_bytes
        + rows * v4_router::V4_TOPK * i64_bytes // gate projection/select
        + rows * h * bf16 * 2 // shared w1/w3
        + expert_intermediate * bf16 * 3 // shared gate/up uploads and w2 input
        + rows * v4_router::V4_TOPK * h * bf16 // routed expert inputs
        + rows * h * bf16 + hc_state * bf16 + post * f32_bytes + comb * f32_bytes
    // final HC post
}

#[allow(clippy::too_many_arguments)]
fn benchmark_v4_fp8_matvec(
    context: &std::sync::Arc<CudaContext>,
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    input: &DeviceBuffer<u8>,
    input_scales: &DeviceBuffer<u8>,
    weight: &DeviceBuffer<u8>,
    weight_scales: &DeviceBuffer<u8>,
    output_f32: &mut DeviceBuffer<f32>,
    output_bf16: &mut DeviceBuffer<u16>,
    in_features: usize,
    out_features: usize,
    strict_k_order: bool,
    requests: usize,
) -> Result<Vec<u64>, Box<dyn Error>> {
    let start = context.new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
    let end = context.new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
    for _ in 0..5 {
        // SAFETY: this helper preserves the fixed validated buffers.
        unsafe {
            launch_v4_fp8_matvec(
                module,
                stream,
                input,
                input_scales,
                weight,
                weight_scales,
                output_f32,
                output_bf16,
                in_features,
                out_features,
                strict_k_order,
            )
        }?;
    }
    end.record(stream)?;
    end.synchronize()?;
    let mut samples = Vec::with_capacity(requests);
    for _ in 0..requests {
        start.record(stream)?;
        // SAFETY: this helper preserves the fixed validated buffers.
        unsafe {
            launch_v4_fp8_matvec(
                module,
                stream,
                input,
                input_scales,
                weight,
                weight_scales,
                output_f32,
                output_bf16,
                in_features,
                out_features,
                strict_k_order,
            )
        }?;
        end.record(stream)?;
        end.synchronize()?;
        samples.push((start.elapsed_ms(&end)? as f64 * 1_000_000.0) as u64);
    }
    Ok(samples)
}

#[allow(clippy::too_many_arguments)]
fn benchmark_v4_kv_ready(
    context: &std::sync::Arc<CudaContext>,
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    input: &DeviceBuffer<u16>,
    freqs: &DeviceBuffer<f32>,
    rope: &mut DeviceBuffer<u16>,
    ready: &mut DeviceBuffer<u16>,
    empty_fp8: &mut DeviceBuffer<u8>,
    scales: &mut DeviceBuffer<u8>,
    topk: &mut DeviceBuffer<i32>,
    requests: usize,
) -> Result<Vec<u64>, Box<dyn Error>> {
    let start = context.new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
    let end = context.new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
    for _ in 0..5 {
        // SAFETY: benchmark reuses only the validated resident Phase A buffers.
        unsafe {
            launch_v4_kv_ready(
                module, stream, input, freqs, rope, ready, empty_fp8, scales, topk,
            )
        }?;
    }
    end.record(stream)?;
    end.synchronize()?;
    let mut samples = Vec::with_capacity(requests);
    for _ in 0..requests {
        start.record(stream)?;
        // SAFETY: benchmark reuses only the validated resident Phase A buffers.
        unsafe {
            launch_v4_kv_ready(
                module, stream, input, freqs, rope, ready, empty_fp8, scales, topk,
            )
        }?;
        end.record(stream)?;
        end.synchronize()?;
        samples.push((start.elapsed_ms(&end)? as f64 * 1_000_000.0) as u64);
    }
    Ok(samples)
}

fn benchmark_v4_kv_commit(
    context: &std::sync::Arc<CudaContext>,
    stream: &CudaStream,
    ready: &DeviceBuffer<u16>,
    cache: &DeviceBuffer<u16>,
    bytes: usize,
    requests: usize,
) -> Result<Vec<u64>, Box<dyn Error>> {
    let start = context.new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
    let end = context.new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
    for _ in 0..5 {
        // SAFETY: Phase B preflight already proved this fixed contiguous copy.
        unsafe {
            cuda_core::memory::memcpy_dtod_async(
                cache.cu_deviceptr(),
                ready.cu_deviceptr(),
                bytes,
                stream.cu_stream(),
            )?;
        }
    }
    end.record(stream)?;
    end.synchronize()?;
    let mut samples = Vec::with_capacity(requests);
    for _ in 0..requests {
        start.record(stream)?;
        // SAFETY: Phase B preflight already proved this fixed contiguous copy.
        unsafe {
            cuda_core::memory::memcpy_dtod_async(
                cache.cu_deviceptr(),
                ready.cu_deviceptr(),
                bytes,
                stream.cu_stream(),
            )?;
        }
        end.record(stream)?;
        end.synchronize()?;
        samples.push((start.elapsed_ms(&end)? as f64 * 1_000_000.0) as u64);
    }
    Ok(samples)
}

fn report_v4_state_u8(name: &str, actual: &[u8], expected: &[u8]) -> usize {
    assert_eq!(actual.len(), expected.len(), "{name}: length");
    let mismatches = actual
        .iter()
        .zip(expected)
        .filter(|(actual, expected)| actual != expected)
        .count();
    println!("{name}: bit_mismatches={mismatches}");
    mismatches
}

fn report_v4_state_i32(name: &str, actual: &[i32], expected: &[i32]) -> usize {
    assert_eq!(actual.len(), expected.len(), "{name}: length");
    let mismatches = actual
        .iter()
        .zip(expected)
        .filter(|(actual, expected)| actual != expected)
        .count();
    println!("{name}: mismatches={mismatches}");
    mismatches
}

fn run_real_v4_router_slice_if_available(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
) -> Result<(), Box<dyn Error>> {
    let root = v4_model_dir()?;
    let hash_path = root.join("model-00002-of-00064.safetensors");
    let learned_path = root.join("model-00005-of-00064.safetensors");
    if !hash_path.is_file() || !learned_path.is_file() {
        return Err(format!(
            "V4 L0/L3 router shards are unavailable under {}",
            root.display()
        )
        .into());
    }

    let rows = 4; // B=2, S=2 in the fixture's flattened reference order.
    let hash_input_ids = [0_i64, 1, 2, 129_279];
    let hash = safetensors::ShardFile::open(hash_path)?;
    let hash_input = deterministic_v4_bf16(rows, 7);
    let hash_norm = read_v4_bf16(&hash, "layers.0.ffn_norm.weight", &[7_168])?;
    let hash_gate = read_v4_bf16(&hash, "layers.0.ffn.gate.weight", &[384, 7_168])?;
    let hash_ids = read_v4_hash_rows(&hash, "layers.0.ffn.gate.tid2eid", &hash_input_ids)?;
    let hash_normalized = v4_router::rmsnorm_bf16(&hash_input, &hash_norm)?;
    let hash_logits = v4_router::project_gate_logits(&hash_normalized, &hash_gate)?;
    let hash_expected = v4_router::select_hash(&hash_logits, &hash_ids)?;
    let (hash_actual_normalized, hash_actual) = run_v4_router_gpu_oracle(
        module,
        stream,
        &hash_input,
        &hash_norm,
        &hash_logits,
        &vec![0.0; v4_router::V4_ROUTED_EXPERTS],
        &hash_ids,
        true,
    )?;
    assert_eq!(hash_actual_normalized, hash_normalized, "L0 BF16 RMSNorm");
    assert_router_selection_close("L0 V4 hash router", &hash_actual, &hash_expected);

    let learned = safetensors::ShardFile::open(learned_path)?;
    let learned_input = deterministic_v4_bf16(rows, 31);
    let learned_norm = read_v4_bf16(&learned, "layers.3.ffn_norm.weight", &[7_168])?;
    let learned_gate = read_v4_bf16(&learned, "layers.3.ffn.gate.weight", &[384, 7_168])?;
    let learned_bias = read_v4_f32(&learned, "layers.3.ffn.gate.bias", &[384])?;
    let learned_normalized = v4_router::rmsnorm_bf16(&learned_input, &learned_norm)?;
    let learned_logits = v4_router::project_gate_logits(&learned_normalized, &learned_gate)?;
    let learned_expected = v4_router::select_learned(&learned_logits, &learned_bias)?;
    let (learned_actual_normalized, learned_actual) = run_v4_router_gpu_oracle(
        module,
        stream,
        &learned_input,
        &learned_norm,
        &learned_logits,
        &learned_bias,
        &[],
        false,
    )?;
    assert_eq!(
        learned_actual_normalized, learned_normalized,
        "L3 BF16 RMSNorm"
    );
    assert_router_selection_close("L3 V4 learned router", &learned_actual, &learned_expected);
    println!(
        "PASS: real V4 L0 hash and L3 learned router ranges, B=2/S=2, match CPU and GPU selection"
    );
    Ok(())
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod tests_continued;
