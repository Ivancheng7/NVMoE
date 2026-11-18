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
#[allow(dead_code)]
mod argmax_records;
mod artifacts;
mod attention_device_chain;
mod b2_generate;
mod b2_runtime;
mod device_attention;
mod device_geometry;
mod device_weights;
mod env_flags;
mod expert_slices;
mod expert_slot;
mod final_head;
mod fixed_host_cache;
#[allow(dead_code)]
mod fixed_layer_cache;
#[allow(dead_code)]
mod fixed_layer_planning;
mod fixed_layer_slot;
mod fixed_prefetch;
mod flash_arms;
mod generate;
#[allow(dead_code)]
mod host_provenance;
mod l0_bench;
mod launch_wrappers;
mod layer_loading;
mod layer_ops;
mod layer_rows;
mod loopback_wire;
#[allow(dead_code)]
mod mtp1_draft;
#[allow(dead_code)]
mod mtp1_l0;
#[allow(dead_code)]
mod mtp1_resources;
mod phase_profiles;
#[allow(dead_code)]
mod qd4;
#[allow(dead_code)]
mod qd4_gates;
#[allow(dead_code)]
mod qd4_routed;
mod run_records;
mod slice_checks;
#[allow(dead_code)]
mod state_audit;
#[allow(dead_code)]
mod stationary_lifecycle;
#[allow(dead_code)]
mod v4_loopback;

// Keep the existing call sites in this file working after the moves.
#[allow(unused_imports)]
use argmax_records::*;
use artifacts::*;
use attention_device_chain::*;
use b2_generate::*;
use b2_runtime::*;
use device_attention::*;
use device_geometry::*;
use device_weights::*;
use env_flags::*;
use expert_slices::*;
use expert_slot::*;
use final_head::*;
#[allow(unused_imports)]
use fixed_layer_cache::*;
#[allow(unused_imports)]
use fixed_layer_planning::*;
use fixed_prefetch::*;
use flash_arms::*;
use generate::*;
#[allow(unused_imports)]
use host_provenance::*;
use l0_bench::*;
use launch_wrappers::*;
use layer_loading::*;
use layer_ops::*;
use layer_rows::*;
use loopback_wire::*;
#[allow(unused_imports)]
use mtp1_draft::*;
#[allow(unused_imports)]
use mtp1_l0::*;
#[allow(unused_imports)]
use mtp1_resources::*;
use phase_profiles::*;
#[allow(unused_imports)]
use qd4::*;
#[allow(unused_imports)]
use qd4_gates::*;
#[allow(unused_imports)]
use qd4_routed::*;
use run_records::*;
use slice_checks::*;
#[allow(unused_imports)]
use state_audit::*;
#[allow(unused_imports)]
use stationary_lifecycle::*;
#[allow(unused_imports)]
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

// The device module is included as raw tokens: cuda-oxide's #[cuda_module]
// only accepts an inline module, so the body lives in its own file and is
// pasted here at compile time.
include!("kernels_module.rs");

fn main() {
    validate_flash_parity_stop_mode().expect("validate Flash parity L0 stop mode");
    // The DSpark speculative-decoding arm was removed on 2026-10-01; keep an explicit
    // hard error so stale scripts never silently fall through to the self-test smoke.
    // Evidence: docs/experiments/2026-09-18-sm120-dspark-acceptance.md
    if std::env::var_os("COLIBRI_V4_FLASH_DSPARK").is_some() {
        panic!(
            "COLIBRI_V4_FLASH_DSPARK was removed; use COLIBRI_V4_FLASH_BASE=1 or \
             COLIBRI_V4_FLASH_SERVE=1 instead"
        );
    }

    let context = CudaContext::new(0).expect("create CUDA context");
    let stream = context.default_stream();
    let module = kernels::load(&context).expect("load embedded Rust PTX");

    if optional_one_flag("COLIBRI_V4_FLASH_SERVE").expect("validate Flash-0731 serve flag") {
        run_real_v4_flash_serve(&context, &module, &stream)
            .expect("run DeepSeek V4 Flash-0731 local service");
        return;
    }

    if optional_one_flag("COLIBRI_V4_FLASH_BASE").expect("validate Flash-0731 base flag") {
        run_real_v4_flash_base(&context, &module, &stream)
            .expect("run DeepSeek V4 Flash-0731 base-only generation");
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

#[cfg(test)]
mod tests;

#[cfg(test)]
mod tests_continued;
