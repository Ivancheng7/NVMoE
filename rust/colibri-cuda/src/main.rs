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
mod fixed_host_cache;
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

// The device module is included as raw tokens: cuda-oxide's #[cuda_module]
// only accepts an inline module, so the body lives in its own file and is
// pasted here at compile time.
include!("kernels_module.rs");

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

#[cfg(test)]
mod tests;

#[cfg(test)]
mod tests_continued;
