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
mod argmax_records;
mod artifacts;
mod attention_gates;
mod b2_generate;
mod b2_runtime;
mod device_attention;
mod device_geometry;
mod device_weights;
mod dspark;
mod env_flags;
mod expert_slices;
mod expert_slot;
mod final_head;
mod fixed_layer_cache;
mod fixed_layer_planning;
mod fixed_prefetch;
mod flash_arms;
mod full61_b1;
mod generate;
mod host_provenance;
mod l0_bench;
mod launch_wrappers;
mod layer_loading;
mod layer_ops;
mod layer_rows;
mod layer_stationary_prefill;
mod loopback_wire;
mod mtp1_draft;
mod mtp1_l0;
mod mtp1_resources;
mod phase_profiles;
mod qd4;
mod qd4_gates;
mod qd4_routed;
mod run_records;
mod slice_checks;
mod state_audit;
mod stationary_lifecycle;
mod v4_loopback;

// Keep the existing call sites in this file working after the moves.
use argmax_records::*;
use artifacts::*;
use attention_gates::*;
use b2_generate::*;
use b2_runtime::*;
use device_attention::*;
use device_geometry::*;
use device_weights::*;
use dspark::*;
use env_flags::*;
use expert_slices::*;
use expert_slot::*;
use final_head::*;
use fixed_layer_cache::*;
use fixed_layer_planning::*;
use fixed_prefetch::*;
use flash_arms::*;
use full61_b1::*;
use generate::*;
use host_provenance::*;
use l0_bench::*;
use launch_wrappers::*;
use layer_loading::*;
use layer_ops::*;
use layer_rows::*;
use layer_stationary_prefill::*;
use loopback_wire::*;
use mtp1_draft::*;
use mtp1_l0::*;
use mtp1_resources::*;
use phase_profiles::*;
use qd4::*;
use qd4_gates::*;
use qd4_routed::*;
use run_records::*;
use slice_checks::*;
use state_audit::*;
use stationary_lifecycle::*;
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
