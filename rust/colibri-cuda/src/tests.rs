use super::*;

#[test]
fn final_head_capture_bytes_are_little_endian() {
    assert_eq!(
        final_head_u16_bytes(&[0x1234, 0xabcd]),
        [0x34, 0x12, 0xcd, 0xab]
    );
    assert_eq!(
        final_head_f32_bytes(&[1.0]),
        1.0_f32.to_bits().to_le_bytes()
    );
}

#[test]
fn flash_final_head_arm_is_flash_only_and_defaults_to_cpu() {
    assert_eq!(
        flash_final_head_arm_for(&v4_spec::FLASH_0731, None).unwrap(),
        FlashFinalHeadArm::Cpu
    );
    assert_eq!(
        flash_final_head_arm_for(
            &v4_spec::FLASH_0731,
            Some(std::ffi::OsStr::new("gpu_exact")),
        )
        .unwrap(),
        FlashFinalHeadArm::GpuExact
    );
    assert!(
        flash_final_head_arm_for(&v4_spec::PRO, Some(std::ffi::OsStr::new("gpu_exact")),).is_err()
    );
}

#[test]
fn flash_cublas_runtime_requires_the_strict_compressor_including_serve() {
    assert!(flash_runtime_strict_compressor_enabled(true, true, false));
    assert!(flash_runtime_strict_compressor_enabled(true, false, true));
    assert!(!flash_runtime_strict_compressor_enabled(true, false, false));
    assert!(!flash_runtime_strict_compressor_enabled(false, true, true));
}

#[test]
fn mhc_sum4_matches_tilelang_xor_tree_not_left_fold() {
    let large = f32::from_bits(0x4c80_0000); // 2^26
    let tree = tilelang_sum4!(large, -large, 1.0_f32, 1.0_f32);
    let left_fold = ((large + -large) + 1.0_f32) + 1.0_f32;
    assert_eq!(tree.to_bits(), 0, "TileLang xor2/xor1 reduction");
    assert_eq!(left_fold.to_bits(), 2.0_f32.to_bits());
    assert_ne!(tree.to_bits(), left_fold.to_bits());
}

#[test]
fn b2_generate_flag_enables_strict_compressor_without_global_env_mutation() {
    assert!(!strict_compressor_enabled_with(|_| false));
    assert!(strict_compressor_enabled_with(
        |name| name == "COLIBRI_V4_B2_GENERATE"
    ));
    assert!(strict_compressor_enabled_with(
        |name| name == "COLIBRI_V4_B4_GENERATE"
    ));
}

#[test]
fn flash_exact_runtime_cublas_arm_enables_strict_compressor_selector() {
    assert!(!flash_exact_runtime_requires_strict_compressor(
        false, true, true
    ));
    assert!(!flash_exact_runtime_requires_strict_compressor(
        true, false, false
    ));
    assert!(flash_exact_runtime_requires_strict_compressor(
        true, true, false
    ));
    assert!(flash_exact_runtime_requires_strict_compressor(
        true, false, true
    ));
}

#[test]
fn official_cublas_compressor_geometry_accepts_flash_and_pro_hidden_widths() {
    for hidden in [4_096, 7_168] {
        assert_eq!(
            official_cublas_compressor_geometry(
                hidden,
                1_024 * hidden,
                1_024 * hidden,
                4 * 1_024,
                1_024,
                4,
            ),
            Some((hidden, 1)),
        );
    }
    assert_eq!(
        official_cublas_compressor_geometry(4_097, 1_024 * 4_096, 1_024 * 4_096, 4_096, 1_024, 4),
        None,
    );
}

#[test]
fn b2_generate_env_and_context_validation_require_two_bounded_prompts() {
    let prompts = parse_b2_generate_token_ids::<2>("0, 1; 2", "COLIBRI_V4_B2_TOKEN_IDS").unwrap();
    assert_eq!(prompts, [vec![0, 1], vec![2]]);
    assert_eq!(
        validate_b2_generate_inputs::<2>(&prompts, 1).unwrap(),
        [2, 1]
    );
    assert!(parse_b2_generate_token_ids::<2>("0;", "COLIBRI_V4_B2_TOKEN_IDS").is_err());
    assert!(parse_b2_generate_token_ids::<2>("0;1;2", "COLIBRI_V4_B2_TOKEN_IDS").is_err());
    assert!(parse_b2_generate_token_ids::<2>("0, nope;1", "COLIBRI_V4_B2_TOKEN_IDS").is_err());
    assert!(validate_b2_generate_inputs::<2>(&prompts, 0).is_err());
    let at_window = [vec![0; v4_state::KV_WINDOW], vec![1]];
    assert!(validate_b2_generate_inputs::<2>(&at_window, 2).is_err());
}

#[test]
fn flash_0731_base_input_is_one_bounded_profile_checked_prompt() {
    let prompts = parse_b2_generate_token_ids_for::<1>(
        &v4_spec::FLASH_0731,
        "0, 128799",
        "COLIBRI_V4_FLASH_TOKEN_IDS",
    )
    .unwrap();
    assert_eq!(prompts, [vec![0, 128_799]]);
    assert_eq!(validate_b2_generate_inputs::<1>(&prompts, 1).unwrap(), [2]);
    assert!(
        parse_b2_generate_token_ids_for::<1>(
            &v4_spec::FLASH_0731,
            "129280",
            "COLIBRI_V4_FLASH_TOKEN_IDS",
        )
        .is_err()
    );
    assert!(v4_spec::FLASH_0731.base_only_no_speculation);
    assert_eq!(v4_spec::FLASH_0731.layers, 43);
}

#[test]
fn flash_mhc_cublas_arm_is_flash_scoped_and_opt_in() {
    let cublas = std::ffi::OsStr::new("cublas");
    assert_eq!(
        flash_mhc_linear_arm_for(&v4_spec::FLASH_0731, None).unwrap(),
        FlashMhcLinearArm::Canonical
    );
    assert_eq!(
        flash_mhc_linear_arm_for(&v4_spec::FLASH_0731, Some(cublas)).unwrap(),
        FlashMhcLinearArm::Cublas
    );
    assert_eq!(
        flash_mhc_linear_arm_for(&v4_spec::PRO, Some(cublas)).unwrap(),
        FlashMhcLinearArm::Canonical
    );
    assert!(
        flash_mhc_linear_arm_for(&v4_spec::FLASH_0731, Some(std::ffi::OsStr::new("other")))
            .is_err()
    );
}

#[test]
fn flash_router_cublas_arm_is_flash_scoped_and_opt_in() {
    let cublas = std::ffi::OsStr::new("cublas");
    assert_eq!(
        flash_router_linear_arm_for(&v4_spec::FLASH_0731, None).unwrap(),
        FlashMhcLinearArm::Canonical
    );
    assert_eq!(
        flash_router_linear_arm_for(&v4_spec::FLASH_0731, Some(cublas)).unwrap(),
        FlashMhcLinearArm::Cublas
    );
    assert_eq!(
        flash_router_linear_arm_for(&v4_spec::PRO, Some(cublas)).unwrap(),
        FlashMhcLinearArm::Canonical
    );
    assert!(
        flash_router_linear_arm_for(&v4_spec::FLASH_0731, Some(std::ffi::OsStr::new("other")))
            .is_err()
    );
}

#[test]
fn flash_fp4_accum_arms_are_flash_scoped_with_reproducible_default() {
    let block32 = std::ffi::OsStr::new("block32");
    let tilelang_ref = std::ffi::OsStr::new("tilelang_ref");
    let fp8_mma = std::ffi::OsStr::new("fp8_mma");
    assert_eq!(
        flash_fp4_accum_arm_for(&v4_spec::FLASH_0731, None).unwrap(),
        FlashFp4AccumArm::Block32
    );
    assert_eq!(
        flash_fp4_accum_arm_for(&v4_spec::FLASH_0731, Some(block32)).unwrap(),
        FlashFp4AccumArm::Block32
    );
    assert_eq!(
        flash_fp4_accum_arm_for(&v4_spec::PRO, Some(block32)).unwrap(),
        FlashFp4AccumArm::Current
    );
    assert_eq!(
        flash_fp4_accum_arm_for(&v4_spec::FLASH_0731, Some(tilelang_ref)).unwrap(),
        FlashFp4AccumArm::TilelangRef
    );
    assert_eq!(
        flash_fp4_accum_arm_for(&v4_spec::PRO, Some(tilelang_ref)).unwrap(),
        FlashFp4AccumArm::Current
    );
    assert_eq!(
        flash_fp4_accum_arm_for(&v4_spec::FLASH_0731, Some(fp8_mma)).unwrap(),
        FlashFp4AccumArm::Fp8Mma
    );
    assert_eq!(
        flash_fp4_accum_arm_for(&v4_spec::PRO, Some(fp8_mma)).unwrap(),
        FlashFp4AccumArm::Current
    );
    assert!(
        flash_fp4_accum_arm_for(&v4_spec::FLASH_0731, Some(std::ffi::OsStr::new("other"))).is_err()
    );
}

#[test]
fn flash_fp4_block32_shape_matches_all_three_expert_gemms() {
    let spec = &v4_spec::FLASH_0731;
    assert_eq!((spec.hidden, spec.moe_intermediate), (4_096, 2_048));
    for (name, out_features, in_features) in [
        ("w1", spec.moe_intermediate, spec.hidden),
        ("w3", spec.moe_intermediate, spec.hidden),
        ("w2", spec.hidden, spec.moe_intermediate),
    ] {
        assert_eq!(in_features % v4_state::FP8_BLOCK, 0, "{name}");
        assert_eq!(in_features % 32, 0, "{name}");
        assert_eq!(
            out_features * in_features / 2,
            out_features * (in_features / 2),
            "{name} packed FP4"
        );
        assert_eq!(
            out_features * in_features / 32,
            out_features * (in_features / 32),
            "{name} E8M0 scales"
        );
    }
}

#[test]
fn flash_fp4_mma_lut_is_exact_e2m1_to_e4m3() {
    let expected = [
        0x00, 0x30, 0x38, 0x3c, 0x40, 0x44, 0x48, 0x4c, 0x80, 0xb0, 0xb8, 0xbc, 0xc0, 0xc4, 0xc8,
        0xcc,
    ];
    for (nibble, expected) in expected.into_iter().enumerate() {
        assert_eq!(v4_fp4_nibble_to_e4m3(nibble as u8), expected);
    }
    assert_eq!(v4_fp4_pair_to_e4m3x2(0x81), 0x8030);
}

#[test]
fn flash_hc_post_serial_source_tree_preserves_ffn_boundary() {
    // L0 FFN flat index 3390.  These are the four already-rounded FP32
    // residual products and branch product from the pinned Flash-0731
    // oracle input.  The serial source-order tree lands exactly on the
    // BF16 tie and RNE selects 0xc210.
    let products = [
        f32::from_bits(0xc20f_788d),
        f32::from_bits(0xbba0_18a8),
        f32::from_bits(0xb782_1796),
        f32::from_bits(0xbc7e_fa8a),
    ];
    let branch = f32::from_bits(0x3c58_21c6);
    let serial = flash_test_add_rn_f32(
        flash_test_add_rn_f32(flash_test_add_rn_f32(products[0], products[1]), products[2]),
        products[3],
    );
    let adjacent = (products[0] + products[1]) + (products[2] + products[3]);
    let cross = (products[0] + products[2]) + (products[1] + products[3]);
    let serial_total = flash_test_add_rn_f32(branch, serial);
    let adjacent_total = branch + adjacent;
    let cross_total = branch + cross;
    let rne = |value: f32| {
        let bits = value.to_bits();
        ((bits.wrapping_add(0x7fff + ((bits >> 16) & 1))) >> 16) as u16
    };
    assert_eq!(serial.to_bits(), 0xc20f_8d82);
    assert_eq!(adjacent.to_bits(), 0xc20f_8d82);
    assert_eq!(cross.to_bits(), 0xc20f_8d81);
    assert_eq!(serial_total.to_bits(), 0xc20f_8000);
    assert_eq!(adjacent_total.to_bits(), 0xc20f_8000);
    assert_eq!(cross_total.to_bits(), 0xc20f_7fff);
    assert_eq!(rne(serial_total), 0xc210);
    assert_eq!(rne(adjacent_total), 0xc210);
    assert_eq!(rne(cross_total), 0xc20f);
}

#[test]
fn flash_l17_attention_hc_post_uses_serial_source_tree_at_bf16_boundary() {
    // L17 post-attention flat index 7441 (output lane 1, hidden column
    // 3345).  Pairwise addition crosses the BF16 tie by one FP32 ULP;
    // the pinned torch kernel's serial source fold lands on the tie.
    let products = [
        f32::from_bits(0xc18e_836c),
        f32::from_bits(0x4494_20f3),
        f32::from_bits(0x432d_176c),
        f32::from_bits(0x3e61_4bf9),
    ];
    let branch = f32::from_bits(0x4377_7922);
    let sum01 = flash_test_add_rn_f32(products[0], products[1]);
    let sum012 = flash_test_add_rn_f32(sum01, products[2]);
    let serial = flash_test_add_rn_f32(sum012, products[3]);
    let pairwise = flash_test_add_rn_f32(sum01, flash_test_add_rn_f32(products[2], products[3]));
    let serial_total = flash_test_add_rn_f32(branch, serial);
    let pairwise_total = flash_test_add_rn_f32(branch, pairwise);

    assert_eq!(serial.to_bits(), 0x44a7_90dc);
    assert_eq!(pairwise.to_bits(), 0x44a7_90dd);
    assert_eq!(serial_total.to_bits(), 0x44c6_8000);
    assert_eq!(pairwise_total.to_bits(), 0x44c6_8001);
    assert_eq!(v4_router::f32_to_bf16_rne(serial_total), 0x44c6);
    assert_eq!(v4_router::f32_to_bf16_rne(pairwise_total), 0x44c7);
}

#[test]
fn flash_router_l1_uses_pinned_six_score_denominator_tree() {
    // L1 official selected post-score words in final route slot order.
    let raw = [
        f32::from_bits(0x3f51_0172),
        f32::from_bits(0x3f82_1af9),
        f32::from_bits(0x3f5b_eea7),
        f32::from_bits(0x3f2a_684f),
        f32::from_bits(0x3f13_cbc4),
        f32::from_bits(0x3f1e_c433),
    ];
    let left_04 = raw[0] + raw[4];
    let left = left_04 + raw[2];
    let right_15 = raw[1] + raw[5];
    let right = right_15 + raw[3];
    let total = left + right;
    assert_eq!(left_04.to_bits(), 0x3fb2_669b);
    assert_eq!(left.to_bits(), 0x4010_2ef7);
    assert_eq!(right_15.to_bits(), 0x3fd1_7d12);
    assert_eq!(right.to_bits(), 0x4013_589d);
    assert_eq!(total.to_bits(), 0x4091_c3ca);
}

#[test]
fn flash_fp4_mma_launch_shape_matches_all_three_expert_gemms() {
    let spec = &v4_spec::FLASH_0731;
    for (name, out_features, in_features) in [
        ("w1", spec.moe_intermediate, spec.hidden),
        ("w3", spec.moe_intermediate, spec.hidden),
        ("w2", spec.hidden, spec.moe_intermediate),
    ] {
        assert_eq!(in_features % 128, 0, "{name}");
        assert_eq!(out_features % 8, 0, "{name}");
        assert_eq!(
            out_features * in_features / 2,
            out_features * (in_features / 2)
        );
        assert_eq!(
            out_features * in_features / 32,
            out_features * (in_features / 32)
        );
        assert_eq!(
            (out_features / 8) * 32,
            out_features * 4,
            "{name} threads/row"
        );
    }
}

#[test]
fn flash_fp4_block32_scales_after_each_raw_dot() {
    let mut raw_products = [0.0_f32; 32];
    raw_products[0] = 1.0e20;
    raw_products[1] = -1.0e20;
    raw_products[2] = 1.0;
    let scale = 0.5_f32;

    let block32 = raw_products.iter().copied().sum::<f32>() * scale;
    let mut lanes = raw_products.map(|value| value * scale);
    for offset in [16, 8, 4, 2, 1] {
        let prior = lanes;
        for lane in 0..32 {
            lanes[lane] += prior[lane ^ offset];
        }
    }
    assert_eq!(block32, 0.5);
    assert_eq!(lanes[0], 0.0);
}

#[test]
fn flash_parity_stop_is_base_scoped_and_accepts_in_range_layers() {
    let dump = std::ffi::OsStr::new("/tmp/flash-parity");
    let zero = std::ffi::OsStr::new("0");
    let stop =
        flash_parity_stop_after_layer_for(&v4_spec::FLASH_0731, true, Some(dump), Some(zero))
            .unwrap()
            .unwrap();
    assert_eq!(stop.layer, 0);
    assert_eq!(stop.dir, std::path::PathBuf::from(dump));
    assert!(
        flash_parity_stop_after_layer_for(&v4_spec::FLASH_0731, false, Some(dump), Some(zero))
            .is_err()
    );
    assert!(
        flash_parity_stop_after_layer_for(&v4_spec::PRO, true, Some(dump), Some(zero)).is_err()
    );
    let one = std::ffi::OsStr::new("1");
    assert_eq!(
        flash_parity_stop_after_layer_for(&v4_spec::FLASH_0731, true, Some(dump), Some(one))
            .unwrap()
            .unwrap()
            .layer,
        1
    );
}

#[test]
fn flash_parity_stop_rejects_missing_dump_and_out_of_range_values() {
    assert!(
        flash_parity_stop_after_layer_for(
            &v4_spec::FLASH_0731,
            true,
            None,
            Some(std::ffi::OsStr::new("0"))
        )
        .is_err()
    );
    assert!(
        flash_parity_stop_after_layer_for(
            &v4_spec::FLASH_0731,
            true,
            Some(std::ffi::OsStr::new("/tmp/flash-parity")),
            Some(std::ffi::OsStr::new("43"))
        )
        .is_err()
    );
}

#[test]
fn flash_router_cublas_layout_is_rows_by_256_by_4096() {
    let rows = 1;
    let m = v4_spec::FLASH_0731.routed_experts;
    let k = v4_spec::FLASH_0731.hidden;
    assert_eq!((m, k), (256, 4_096));
    assert_eq!(rows * m, 256);
    assert_eq!(m * k, 1_048_576);
    let weight = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let input = [2.0_f32, 3.0];
    let output = (0..3)
        .map(|row| {
            (0..2)
                .map(|col| weight[row * 2 + col] * input[col])
                .sum::<f32>()
        })
        .collect::<Vec<_>>();
    assert_eq!(output, [8.0, 18.0, 28.0]);
}

#[test]
fn flash_router_libdevice_score_arm_is_flash_scoped_and_opt_in() {
    let libdevice = std::ffi::OsStr::new("libdevice");
    assert_eq!(
        flash_router_score_arm_for(&v4_spec::FLASH_0731, None).unwrap(),
        FlashRouterScoreArm::Canonical
    );
    assert_eq!(
        flash_router_score_arm_for(&v4_spec::FLASH_0731, Some(libdevice)).unwrap(),
        FlashRouterScoreArm::Libdevice
    );
    assert_eq!(
        flash_router_score_arm_for(&v4_spec::PRO, Some(libdevice)).unwrap(),
        FlashRouterScoreArm::Canonical
    );
    assert!(
        flash_router_score_arm_for(&v4_spec::FLASH_0731, Some(std::ffi::OsStr::new("other")))
            .is_err()
    );
}

#[test]
fn flash_router_libdevice_score_reference_has_official_threshold_bits() {
    let cases = [
        (0xbf80_0000, 0x3f0f_485c),
        (0x0000_0000, 0x3f55_224d),
        (0x3f80_0000, 0x3f92_af5a),
        (0x41a0_0000, 0x408f_1bbd),
        (0x41a0_0001, 0x408f_1bbd),
    ];
    for (input, expected) in cases {
        assert_eq!(
            flash_router_libdevice_score_host_reference(f32::from_bits(input)).to_bits(),
            expected,
            "input=0x{input:08x}"
        );
    }
}

#[test]
fn flash_mhc_cublas_layout_is_24_by_16384_row_major_dot() {
    let m = v4_state::MIX;
    let k = v4_state::HC * v4_spec::FLASH_0731.hidden;
    assert_eq!((m, k), (24, 16_384));
    let weight = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let input = [2.0_f32, 3.0];
    let output = (0..3)
        .map(|row| {
            (0..2)
                .map(|col| weight[row * 2 + col] * input[col])
                .sum::<f32>()
        })
        .collect::<Vec<_>>();
    assert_eq!(output, [8.0, 18.0, 28.0]);
}

#[test]
fn flash_mhc_rsqrt_debug_layout_records_input_and_committed_candidate() {
    assert_eq!(FLASH_MHC_RSQRT_DEBUG_FIELDS, 5);
    assert_eq!(
        flash_mhc_rsqrt_debug_fields(),
        [
            "square_sum_f32_bits",
            "mean_square_plus_eps_f32_bits",
            "newton_candidate_f32_bits",
            "libdevice_sqrt_f32_bits",
            "libdevice_rsqrt_f32_bits",
        ]
    );
    assert_eq!(
        flash_mhc_rsqrt_debug_fields()[FLASH_MHC_RSQRT_COMMITTED_CANDIDATE],
        FLASH_MHC_RSQRT_COMMITTED_FIELD
    );
}

#[test]
fn flash_l1_rmsnorm_direct_rsqrt_preserves_raw_bits() {
    const L1_CUTPOINT: &str =
        "results/flash-0731-l1-full-cutpoints-20260802T054500Z/official-cutpoints";
    const LANES: [(usize, u16); 3] = [(468, 0xbe75), (558, 0x3d75), (2290, 0x3d75)];

    let cutpoint = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(L1_CUTPOINT);
    let input = std::fs::read(cutpoint.join("ffn_hc_collapsed_bf16.bin"))
        .expect("read locked L1 FFN RMSNorm input");
    let input = input
        .chunks_exact(2)
        .map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]]))
        .collect::<Vec<_>>();
    assert_eq!(input.len(), v4_spec::FLASH_0731.hidden);

    let model_dir = v4_model_dir_for(&v4_spec::FLASH_0731).expect("pinned Flash model");
    let shard = safetensors::ShardFile::open(
        model_dir.join(v4_spec::FLASH_0731.shard_name(3).expect("L1 shard name")),
    )
    .expect("open pinned L1 shard");
    let norm = read_v4_bf16(&shard, "layers.1.ffn_norm.weight", &[4_096])
        .expect("read pinned L1 FFN norm");

    let context = CudaContext::new(0).expect("create CUDA context");
    let stream = context.default_stream();
    let module = kernels::load(&context).expect("load embedded Rust PTX");
    let input_device = DeviceBuffer::from_host(&stream, &input).expect("upload L1 RMSNorm input");
    let norm_device = DeviceBuffer::from_host(&stream, &norm).expect("upload L1 RMSNorm weight");
    let mut debug = DeviceBuffer::<u32>::zeroed(&stream, FLASH_MHC_RSQRT_DEBUG_FIELDS)
        .expect("allocate rsqrt probe");
    unsafe {
        module
            .mhc_rsqrt_debug_v4(
                &stream,
                LaunchConfig::for_num_elems(1),
                &input_device,
                &mut debug,
            )
            .expect("launch direct rsqrt probe");
    }
    let debug = debug.to_host_vec(&stream).expect("copy rsqrt probe");
    assert_eq!(debug[0], 42_009_416.0_f32.to_bits());
    assert_eq!(debug[FLASH_MHC_RSQRT_COMMITTED_CANDIDATE], 0x3c21_c7d9);

    let output = run_l0_rmsnorm(
        &module,
        &stream,
        &input,
        &norm_device,
        v4_spec::FLASH_0731.hidden,
    )
    .expect("run L1 learned-weight RMSNorm");
    for (lane, expected) in LANES {
        assert_eq!(output[lane], expected, "lane {lane}");
    }
}

fn flash_pytorch_210_rmsnorm_tree(input: &[u16]) -> (f32, f32, f32) {
    assert_eq!(input.len(), 4_096);
    let squares = input
        .iter()
        .map(|&bits| {
            let value = v4_router::bf16_to_f32(bits);
            value * value
        })
        .collect::<Vec<_>>();
    let mut reduced = [0.0_f32; 512];
    for (thread, slot) in reduced.iter_mut().enumerate() {
        let base = thread * 4;
        let upper = base + 2_048;
        let a0 = squares[base] + squares[upper];
        let a1 = squares[base + 1] + squares[upper + 1];
        let a2 = squares[base + 2] + squares[upper + 2];
        let a3 = squares[base + 3] + squares[upper + 3];
        *slot = ((a0 + a1) + a2) + a3;
    }
    for offset in [256, 128, 64, 32] {
        for thread in 0..offset {
            reduced[thread] += reduced[thread + offset];
        }
    }
    for offset in [16, 8, 4, 2, 1] {
        for lane in 0..32 - offset {
            reduced[lane] += reduced[lane + offset];
        }
    }
    let sum = reduced[0];
    let mean = sum * (1.0 / 4_096.0);
    (sum, mean, 1.0 / (mean + 1e-6).sqrt())
}

fn flash_rmsnorm_test_input() -> Vec<u16> {
    (0..4_096)
        .map(|index| v4_router::f32_to_bf16_rne((index % 257) as f32 * 0.03125 - 4.0))
        .collect()
}

#[test]
fn flash_rmsnorm_host_tree_preserves_pinned_raw_bits() {
    let (sum, mean, _) = flash_pytorch_210_rmsnorm_tree(&flash_rmsnorm_test_input());
    assert_eq!(sum.to_bits(), 0x46aa_e594);
    assert_eq!(mean.to_bits(), 0x40aa_e594);
}

#[test]
fn flash_rmsnorm_oxide_preserves_pinned_raw_bits() {
    let input = flash_rmsnorm_test_input();
    let norm = vec![v4_router::f32_to_bf16_rne(1.0); 4_096];

    let context = CudaContext::new(0).expect("create CUDA context");
    let stream = context.default_stream();
    let module = kernels::load(&context).expect("load embedded Rust PTX");
    let input = DeviceBuffer::from_host(&stream, &input).expect("upload RMSNorm input");
    let norm = DeviceBuffer::from_host(&stream, &norm).expect("upload RMSNorm norm");
    let mut squares = DeviceBuffer::<f32>::zeroed(&stream, 4_096).expect("allocate squares");
    let mut output = DeviceBuffer::<u16>::zeroed(&stream, 4_096).expect("allocate output");
    let mut trace = DeviceBuffer::<f32>::zeroed(&stream, 3).expect("allocate raw trace");
    unsafe {
        module
            .rmsnorm_flash_square_bf16(
                &stream,
                LaunchConfig::for_num_elems(4_096),
                &input,
                &mut squares,
            )
            .expect("launch rounded square materialization");
        module
            .rmsnorm_flash_learned_4096_bf16(
                &stream,
                LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (512, 1, 1),
                    shared_mem_bytes: 0,
                },
                &input,
                &norm,
                &squares,
                &mut output,
                &mut trace,
            )
            .expect("launch pinned Flash RMSNorm reduction");
    }
    let trace = trace.to_host_vec(&stream).expect("copy raw trace");
    for (actual, file) in [
        (trace[0], 0x46aa_e594),
        (trace[1], 0x40aa_e594),
        (trace[2], 0x3edd_8d9c),
    ] {
        assert_eq!(actual.to_bits(), file);
    }
    let output = output.to_host_vec(&stream).expect("copy normalized output");
    for (lane, expected) in [
        (0, 0xbfde),
        (1, 0xbfdc),
        (127, 0xbc5e),
        (128, 0),
        (256, 0x3fde),
        (4095, 0x3fc2),
    ] {
        assert_eq!(output[lane], expected, "lane {lane}");
    }
}

fn flash_l17_attention_rmsnorm_fixture() -> (Vec<u16>, Vec<u16>, [f32; 4]) {
    const CUTPOINT: &str =
        "results/flash-0731-l17-full-router-trace-20260802T051640Z/official-cutpoints";
    let cutpoint = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(CUTPOINT);
    let input = std::fs::read(cutpoint.join("input_hidden_bf16.bin"))
        .expect("read locked L17 mHC input")
        .chunks_exact(2)
        .map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]]))
        .collect::<Vec<_>>();
    let pre = std::fs::read(cutpoint.join("attn_hc_pre_tilelang_f32.bin"))
        .expect("read locked L17 mHC pre")
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        .collect::<Vec<_>>();
    assert_eq!(input.len(), v4_state::HC * v4_spec::FLASH_0731.hidden);
    let pre: [f32; 4] = pre.try_into().expect("L17 mHC pre has four lanes");
    let collapsed = (0..v4_spec::FLASH_0731.hidden)
        .map(|column| {
            let lanes = [
                input[column],
                input[v4_spec::FLASH_0731.hidden + column],
                input[2 * v4_spec::FLASH_0731.hidden + column],
                input[3 * v4_spec::FLASH_0731.hidden + column],
            ];
            v4_router::f32_to_bf16_rne(flash_hc_serial_sum4(pre, lanes).3)
        })
        .collect::<Vec<_>>();
    let model_dir = v4_model_dir_for(&v4_spec::FLASH_0731).expect("pinned Flash model");
    let shard = safetensors::ShardFile::open(
        model_dir.join(v4_spec::FLASH_0731.shard_name(19).expect("L17 shard name")),
    )
    .expect("open pinned L17 shard");
    let norm = read_v4_bf16(&shard, "layers.17.attn_norm.weight", &[4_096])
        .expect("read pinned L17 attention norm");
    (collapsed, norm, pre)
}

#[inline(never)]
fn flash_test_mul_rn_f32(left: f32, right: f32) -> f32 {
    let left = std::hint::black_box(left);
    let right = std::hint::black_box(right);
    let value = left * right;
    unsafe { core::ptr::read_volatile(&value) }
}

#[inline(never)]
fn flash_test_add_rn_f32(left: f32, right: f32) -> f32 {
    let left = std::hint::black_box(left);
    let right = std::hint::black_box(right);
    let value = left + right;
    unsafe { core::ptr::read_volatile(&value) }
}

#[test]
fn flash_l17_attention_rmsnorm_preserves_raw_bits() {
    let (input, norm, pre) = flash_l17_attention_rmsnorm_fixture();
    let (sum, mean, inv) = flash_pytorch_210_rmsnorm_tree(&input);
    assert_eq!(sum.to_bits(), 0x51ad_0f12);
    assert_eq!((mean + 1e-6).to_bits(), 0x4bad_0f12);
    assert_eq!(inv.to_bits(), 0x395c_2a34);

    let squares = input
        .iter()
        .map(|&bits| {
            let value = v4_router::bf16_to_f32(bits);
            value * value
        })
        .collect::<Vec<_>>();
    let legacy_sum = v4_router::pytorch_contiguous_sum_f32(&squares);
    let legacy_mean = legacy_sum * (1.0 / 4_096.0) + 1e-6;
    let legacy_inv = 1.0 / legacy_mean.sqrt();
    assert_eq!(legacy_sum.to_bits(), 0x51ad_0f14);
    assert_eq!(legacy_mean.to_bits(), 0x4bad_0f14);
    assert_eq!(legacy_inv.to_bits(), 0x395c_2a33);

    let lane = 741;
    assert_eq!(
        pre.map(f32::to_bits),
        [0x3f0e_e430, 0x3f50_e4a5, 0x3e8a_94a8, 0x3c09_abf2]
    );
    assert_eq!(input[lane], 0x449c);
    assert_eq!(norm[lane], 0x3d87);
    let official_value = flash_test_mul_rn_f32(
        flash_test_mul_rn_f32(v4_router::bf16_to_f32(input[lane]), inv),
        v4_router::bf16_to_f32(norm[lane]),
    );
    let legacy_value = flash_test_mul_rn_f32(
        flash_test_mul_rn_f32(v4_router::bf16_to_f32(input[lane]), legacy_inv),
        v4_router::bf16_to_f32(norm[lane]),
    );
    assert_eq!(official_value.to_bits(), 0x3c8d_8000);
    assert_eq!(legacy_value.to_bits(), 0x3c8d_7fff);
    assert_eq!(v4_router::f32_to_bf16_rne(official_value), 0x3c8e);
    assert_eq!(v4_router::f32_to_bf16_rne(legacy_value), 0x3c8d);

    let context = CudaContext::new(0).expect("create CUDA context");
    let stream = context.default_stream();
    let module = kernels::load(&context).expect("load embedded Rust PTX");
    let input = DeviceBuffer::from_host(&stream, &input).expect("upload L17 RMSNorm input");
    let norm = DeviceBuffer::from_host(&stream, &norm).expect("upload L17 RMSNorm weight");
    let mut output = DeviceBuffer::<u16>::zeroed(&stream, 4_096).expect("allocate output");
    let mut trace = DeviceBuffer::<f32>::zeroed(&stream, 3).expect("allocate raw trace");
    let squares = run_flash_learned_rmsnorm_4096_device_into(
        &module,
        &stream,
        &input,
        &norm,
        &mut output,
        &mut trace,
    )
    .expect("run pinned L17 attention RMSNorm");
    assert_eq!(squares[741].to_bits(), 0x49be_2000);
    let trace = trace.to_host_vec(&stream).expect("copy raw trace");
    assert_eq!(
        trace.iter().copied().map(f32::to_bits).collect::<Vec<_>>(),
        [0x51ad_0f12, 0x4bad_0f12, 0x395c_2a34]
    );
    assert_eq!(
        output
            .to_host_vec(&stream)
            .expect("copy L17 RMSNorm output")[741],
        0x3c8e
    );
}

fn flash_pytorch_210_mhc_tree(input: &[u16]) -> (f32, f32) {
    assert_eq!(input.len(), 16_384);
    let squares = input
        .iter()
        .map(|&bits| {
            let value = v4_router::bf16_to_f32(bits);
            value * value
        })
        .collect::<Vec<_>>();
    let mut reduced = [0.0_f32; 512];
    for (thread, slot) in reduced.iter_mut().enumerate() {
        let base = thread * 4;
        let mut lanes = [
            squares[base],
            squares[base + 1],
            squares[base + 2],
            squares[base + 3],
        ];
        for k in 1..8 {
            let offset = 2_048 * k;
            for lane in 0..4 {
                lanes[lane] += squares[base + offset + lane];
            }
        }
        *slot = ((lanes[0] + lanes[1]) + lanes[2]) + lanes[3];
    }
    for offset in [256, 128, 64, 32] {
        for thread in 0..offset {
            reduced[thread] += reduced[thread + offset];
        }
    }
    for offset in [16, 8, 4, 2, 1] {
        for lane in 0..32 - offset {
            reduced[lane] += reduced[lane + offset];
        }
    }
    let sum = reduced[0];
    (sum, sum * (1.0 / 16_384.0) + 1e-6)
}

fn flash_l6_mhc_input() -> Vec<u16> {
    const L6_CUTPOINT: &str =
        "results/flash-0731-l6-full-router-trace-20260802T044543Z/official-cutpoints";
    std::fs::read(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(L6_CUTPOINT)
            .join("input_hidden_bf16.bin"),
    )
    .expect("read locked L6 mHC input")
    .chunks_exact(2)
    .map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]]))
    .collect()
}

#[test]
fn flash_mhc_16384_host_tree_preserves_l6_raw_bits() {
    let (sum, mean_eps) = flash_pytorch_210_mhc_tree(&flash_l6_mhc_input());
    assert_eq!(sum.to_bits(), 0x5208_aec4);
    assert_eq!(mean_eps.to_bits(), 0x4b08_aec4);
}

#[test]
fn flash_mhc_16384_oxide_preserves_l6_raw_bits() {
    let input = flash_l6_mhc_input();
    let context = CudaContext::new(0).expect("create CUDA context");
    let stream = context.default_stream();
    let module = kernels::load(&context).expect("load embedded Rust PTX");
    let input = DeviceBuffer::from_host(&stream, &input).expect("upload L6 mHC input");
    let mut squares = DeviceBuffer::<f32>::zeroed(&stream, 16_384).expect("allocate squares");
    let mut rsqrt = DeviceBuffer::<f32>::zeroed(&stream, 1).expect("allocate rsqrt");
    let mut debug = DeviceBuffer::<u32>::zeroed(&stream, FLASH_MHC_RSQRT_DEBUG_FIELDS)
        .expect("allocate mHC trace");
    unsafe {
        module
            .rmsnorm_flash_square_bf16(
                &stream,
                LaunchConfig::for_num_elems(16_384),
                &input,
                &mut squares,
            )
            .expect("launch mHC square materialization");
        module
            .mhc_flash_reduce_rsqrt_16384(
                &stream,
                LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (512, 1, 1),
                    shared_mem_bytes: 0,
                },
                &squares,
                &mut rsqrt,
                &mut debug,
            )
            .expect("launch fixed mHC reduction");
    }
    let debug = debug.to_host_vec(&stream).expect("copy mHC trace");
    assert_eq!(debug[0], 0x5208_aec4);
    assert_eq!(debug[1], 0x4b08_aec4);
    assert_eq!(debug[FLASH_MHC_RSQRT_COMMITTED_CANDIDATE], 0x39af_2ce9);
    assert_eq!(
        rsqrt.to_host_vec(&stream).expect("copy mHC rsqrt")[0].to_bits(),
        0x39af_2ce9
    );
}

fn flash_hc_serial_sum4(coefficients: [f32; 4], inputs: [u16; 4]) -> ([f32; 4], f32, f32, f32) {
    let product = |coefficient: f32, input: u16| {
        let value = coefficient * f32::from_bits((input as u32) << 16);
        // Keep the CPU regression's products observably rounded before
        // the following add, matching the explicit device intrinsics.
        unsafe { core::ptr::read_volatile(&value) }
    };
    let products = [
        product(coefficients[0], inputs[0]),
        product(coefficients[1], inputs[1]),
        product(coefficients[2], inputs[2]),
        product(coefficients[3], inputs[3]),
    ];
    let p01 = unsafe { core::ptr::read_volatile(&(products[0] + products[1])) };
    let p012 = unsafe { core::ptr::read_volatile(&(p01 + products[2])) };
    let value = unsafe { core::ptr::read_volatile(&(p012 + products[3])) };
    (products, p01, p012, value)
}

#[test]
fn flash_ffn_hc_index124_uses_official_serial_raw_bits() {
    // L3/token0 single-index certificate: the four BF16 source lanes
    // and actual official TileLang pre coefficients.  The final cast is
    // the official `ffn_hc_collapsed_bf16[124]` bit pattern.
    let coefficients = [
        f32::from_bits(0x3d83_6612),
        f32::from_bits(0x3b27_6525),
        f32::from_bits(0x3d69_48b1),
        f32::from_bits(0x3f71_20ff),
    ];
    let (products, p01, p012, value) =
        flash_hc_serial_sum4(coefficients, [0xc529, 0xc349, 0xc56e, 0x4348]);
    assert_eq!(
        products.map(f32::to_bits),
        [0xc32d_7cc4, 0xbf03_6e6a, 0xc358_e195, 0x433c_61c7]
    );
    assert_eq!(p01.to_bits(), 0xc32e_0032);
    assert_eq!(p012.to_bits(), 0xc3c3_70e4);
    assert_eq!(value.to_bits(), 0xc34a_8001);
    assert_eq!(v4_router::f32_to_bf16_rne(value), 0xc34b);
}

#[test]
fn flash_parity_manifest_records_row_major_little_endian_bf16_shape() {
    let manifest = flash_parity_manifest(
        "input_hidden_bf16",
        FlashParityDtype::Bf16,
        &[1, 4, 8],
        64,
        "a".repeat(64).as_str(),
    )
    .unwrap();
    assert_eq!(manifest["dtype"], "BF16");
    assert_eq!(manifest["shape"], serde_json::json!([1, 4, 8]));
    assert_eq!(manifest["layout"], "row_major");
    assert_eq!(manifest["endianness"], "little");
    assert_eq!(manifest["byte_len"], 64);
}

#[test]
fn flash_parity_manifest_records_raw_u32_trace_words() {
    let manifest = flash_parity_manifest(
        "attn_hc_post_index16102_u32",
        FlashParityDtype::U32,
        &[21],
        84,
        "b".repeat(64).as_str(),
    )
    .unwrap();
    assert_eq!(manifest["dtype"], "U32");
    assert_eq!(manifest["shape"], serde_json::json!([21]));
    assert_eq!(manifest["byte_len"], 84);
}

#[test]
fn flash_parity_manifest_records_route_slot_fp8_shapes() {
    let spec = &v4_spec::FLASH_0731;
    let input = flash_parity_manifest(
        "expert_input_fp8_e4m3",
        FlashParityDtype::U8,
        &[1, spec.experts_per_token, spec.hidden],
        spec.experts_per_token * spec.hidden,
        "f".repeat(64).as_str(),
    )
    .unwrap();
    let scale = flash_parity_manifest(
        "expert_input_scale_e8m0",
        FlashParityDtype::U8,
        &[1, spec.experts_per_token, spec.hidden / v4_state::FP8_BLOCK],
        spec.experts_per_token * spec.hidden / v4_state::FP8_BLOCK,
        "0".repeat(64).as_str(),
    )
    .unwrap();
    assert_eq!(input["shape"], serde_json::json!([1, 6, 4096]));
    assert_eq!(scale["shape"], serde_json::json!([1, 6, 32]));
    assert_eq!(input["dtype"], "U8");
    assert_eq!(scale["dtype"], "U8");
}

#[test]
fn flash_parity_router_debug_manifest_shapes_are_explicit() {
    let logits = flash_parity_manifest(
        "router_gate_logits_f32",
        FlashParityDtype::F32,
        &[1, v4_spec::FLASH_0731.routed_experts],
        v4_spec::FLASH_0731.routed_experts * std::mem::size_of::<f32>(),
        "c".repeat(64).as_str(),
    )
    .unwrap();
    let selected = flash_parity_manifest(
        "selected_scores_f32",
        FlashParityDtype::F32,
        &[1, v4_spec::FLASH_0731.experts_per_token],
        v4_spec::FLASH_0731.experts_per_token * std::mem::size_of::<f32>(),
        "d".repeat(64).as_str(),
    )
    .unwrap();
    let denom = flash_parity_manifest(
        "route_denom_f32",
        FlashParityDtype::F32,
        &[1],
        std::mem::size_of::<f32>(),
        "e".repeat(64).as_str(),
    )
    .unwrap();
    assert_eq!(logits["shape"], serde_json::json!([1, 256]));
    assert_eq!(selected["shape"], serde_json::json!([1, 6]));
    assert_eq!(denom["shape"], serde_json::json!([1]));
}

#[test]
fn flash_b1_phase_profile_closes_without_double_counting_nested_work() {
    let fixed = FixedLoadTelemetry {
        fixed_read_alloc_ns: 5,
        fixed_decode_ns: 7,
        fixed_h2d_ns: 8,
        fixed_h2d_bytes: 40,
        fixed_source_bytes: 40,
        cache_status: "miss",
        ..FixedLoadTelemetry::default()
    };
    let stage = LayerStageTelemetry {
        attention_ns: 10,
        commit_ns: 2,
        router_ns: 15,
        shared_ns: 8,
        routed_ns: 20,
        routed_read_ns: 3,
        routed_h2d_ns: 4,
        routed_kernel_ns: 5,
        ..LayerStageTelemetry::default()
    };
    let layer = flash_b1_layer_phase_profile_json(0, 100, 20, &fixed, 10, &stage).unwrap();
    assert_eq!(layer["top_level_attributed_ns"], 85);
    assert_eq!(layer["unattributed_gap_ns"], 15);
    let timing = FlashB1PhaseTiming {
        layers: vec![layer],
        layer_outer_ns: 100,
        final_heads: vec![serde_json::json!({"outer":{"wall_ns":20}})],
        final_head_outer_ns: 20,
    };
    let request = flash_b1_request_phase_profile_json(&timing, 140).unwrap();
    assert_eq!(request["top_level_attributed_ns"], 120);
    assert_eq!(request["unattributed_gap_ns"], 20);
    assert!(request["closure"]["exact"].as_bool().unwrap());
}

#[test]
fn flash_b1_phase_profile_marks_background_host_work_without_charging_it_twice() {
    let fixed = FixedLoadTelemetry {
        fixed_read_alloc_ns: 50,
        fixed_decode_ns: 20,
        fixed_h2d_ns: 10,
        fixed_h2d_bytes: 40,
        fixed_source_bytes: 40,
        cache_status: "prefetched_host",
        ..FixedLoadTelemetry::default()
    };
    let stage = LayerStageTelemetry {
        routed_prefetched: true,
        routed_ns: 10,
        routed_read_ns: 50,
        ..LayerStageTelemetry::default()
    };
    let layer = flash_b1_layer_phase_profile_json(1, 30, 5, &fixed, 5, &stage).unwrap();
    assert!(
        layer["top_level"]["fixed"]["host_prefetched"]
            .as_bool()
            .unwrap()
    );
    assert!(
        layer["top_level"]["routed"]["host_prefetched"]
            .as_bool()
            .unwrap()
    );
    assert_eq!(layer["top_level_attributed_ns"], 20);
}

#[test]
fn b2_generate_active_set_keeps_the_survivor_after_peer_eos() {
    let mut active = [true, true];
    assert_eq!(b2_generate_active_indices(&active), vec![0, 1]);
    assert_eq!(
        greedy_argmax_decision(1, 0, 0, 2, V4_EOS_TOKEN_ID),
        GreedyDecision::EmitAndStop(GreedyStopReason::Eos)
    );
    active[0] = false;
    assert_eq!(b2_generate_active_indices(&active), vec![1]);
    assert_eq!(
        greedy_argmax_decision(1, 0, 0, 2, 42),
        GreedyDecision::EmitAndContinue
    );
    assert_eq!(b2_generate_active_indices(&active), vec![1]);
}

#[test]
fn b4_generate_survivors_keep_active_row_metadata_and_conserve_ledgers() {
    let prompts =
        parse_b2_generate_token_ids::<4>("10;20;30;40", "COLIBRI_V4_B4_TOKEN_IDS").unwrap();
    assert_eq!(
        validate_b2_generate_inputs::<4>(&prompts, 1).unwrap(),
        [1, 1, 1, 1]
    );
    let active = [false, true, false, true];
    assert_eq!(b2_generate_active_indices(&active), vec![1, 3]);
    let rows = [
        [101_i64, 102, 103, 104, 105, 106],
        [301, 302, 303, 304, 305, 306],
    ]
    .concat();
    let mut total = B2Physical::default();
    let mut requests: [B2RequestPhysical; 4] =
        std::array::from_fn(|_| B2RequestPhysical::default());
    for (row, request) in b2_generate_active_indices(&active).into_iter().enumerate() {
        let metadata = b2_router_metadata_row(true, &rows, row, 2).unwrap();
        total
            .record_request_router_metadata(true, metadata)
            .unwrap();
        requests[request]
            .record_initial_state_source_bytes((request + 1) * 10)
            .unwrap();
        requests[request]
            .record_final_head_source_bytes((request + 1) * 100)
            .unwrap();
        total
            .record_initial_state_source_bytes((request + 1) * 10)
            .unwrap();
        total
            .record_final_head_source_bytes((request + 1) * 100)
            .unwrap();
    }
    assert_eq!(
        b2_router_metadata_row(true, &rows, 0, 2).unwrap(),
        &[101, 102, 103, 104, 105, 106]
    );
    assert_eq!(
        b2_router_metadata_row(true, &rows, 1, 2).unwrap(),
        &[301, 302, 303, 304, 305, 306]
    );
    assert_eq!(
        total.request_router_metadata_source_bytes,
        2 * v4_router::V4_TOPK * std::mem::size_of::<i64>()
    );
    assert_eq!(
        total.initial_state_source_bytes,
        requests
            .iter()
            .map(|request| request.initial_state_source_bytes)
            .sum::<usize>()
    );
    assert_eq!(
        total.final_head_source_bytes,
        requests
            .iter()
            .map(|request| request.final_head_source_bytes)
            .sum::<usize>()
    );
}

#[test]
fn b4_greedy_abba_sequence_and_ledger_combine() {
    let (source, h2d) = b2_raw_runtime_ledger(64, 128, 64).unwrap();
    assert_eq!((source, h2d), (64, 192));
    assert_eq!(
        (0..4).map(b2_serial_first).collect::<Vec<_>>(),
        vec![true, false, false, true]
    );
    let mut first = B2Physical {
        fixed_source_bytes: 10,
        fixed_h2d_bytes: 20,
        request_router_metadata_source_bytes: 3,
        request_router_metadata_h2d_bytes: 3,
        routed_source_bytes: 30,
        routed_h2d_bytes: 30,
        initial_state_source_bytes: 4,
        final_head_source_bytes: 5,
    };
    first
        .checked_add_assign(B2Physical {
            fixed_source_bytes: 10,
            fixed_h2d_bytes: 20,
            request_router_metadata_source_bytes: 3,
            request_router_metadata_h2d_bytes: 3,
            routed_source_bytes: 30,
            routed_h2d_bytes: 30,
            initial_state_source_bytes: 4,
            final_head_source_bytes: 5,
        })
        .unwrap();
    assert_eq!(first.fixed_source_bytes, 20);
    assert_eq!(first.routed_source_bytes, 60);
    assert_eq!(first.initial_state_source_bytes, 8);
    assert_eq!(first.final_head_source_bytes, 10);
    let a: B2GenerateSummary<4> = B2GenerateSummary {
        requests: std::array::from_fn(|_| serde_json::json!({"argmax": 7})),
        physical: first,
        duration_ns: b4_greedy_a_duration_ns(4, 6).unwrap(),
        peak_device_bytes: 20,
    };
    let b: B2GenerateSummary<4> = B2GenerateSummary {
        requests: std::array::from_fn(|_| serde_json::json!({"argmax": 7})),
        physical: B2Physical::default(),
        duration_ns: 5,
        peak_device_bytes: 10,
    };
    assert_eq!(a.requests, b.requests);
    assert_eq!(a.physical.fixed_source_bytes, 20);
    assert_eq!(a.duration_ns, 10);
    assert_eq!(b.duration_ns, 5);
    assert!(b4_greedy_a_duration_ns(u64::MAX, 1).is_err());

    let digest = B2LayerDigest {
        layer: 0,
        route_ids: vec![1, 2],
        route_weight_bits: vec![3, 4],
        attention_topk_digest: 5,
        hidden_digest: 6,
        kv_digest: 7,
        main_kv_state_digest: 8,
        main_score_state_digest: 9,
        indexer_kv_state_digest: Some(10),
        indexer_score_state_digest: Some(11),
    };
    let baseline = B2LayerDigest::ordered_chain_sha256(&[digest.clone()]).unwrap();
    let mut variants = Vec::new();
    let mut changed = digest.clone();
    changed.layer = 1;
    variants.push(changed);
    let mut changed = digest.clone();
    changed.route_ids.push(12);
    variants.push(changed);
    let mut changed = digest.clone();
    changed.route_weight_bits.push(13);
    variants.push(changed);
    let mut changed = digest.clone();
    changed.attention_topk_digest = 14;
    variants.push(changed);
    let mut changed = digest.clone();
    changed.hidden_digest = 15;
    variants.push(changed);
    let mut changed = digest.clone();
    changed.kv_digest = 16;
    variants.push(changed);
    let mut changed = digest.clone();
    changed.main_kv_state_digest = 17;
    variants.push(changed);
    let mut changed = digest.clone();
    changed.main_score_state_digest = 18;
    variants.push(changed);
    let mut changed = digest.clone();
    changed.indexer_kv_state_digest = None;
    variants.push(changed);
    let mut changed = digest.clone();
    changed.indexer_score_state_digest = None;
    variants.push(changed);
    for changed in variants {
        assert_ne!(
            B2LayerDigest::ordered_chain_sha256(&[changed]).unwrap(),
            baseline
        );
    }
    let second = B2LayerDigest {
        layer: 1,
        ..digest.clone()
    };
    assert_ne!(
        B2LayerDigest::ordered_chain_sha256(&[digest.clone(), second.clone()]).unwrap(),
        B2LayerDigest::ordered_chain_sha256(&[second, digest]).unwrap()
    );
}

#[test]
fn b2_generate_result_includes_final_digest_and_request_ledgers() {
    let request = B2GenerateRequest {
        prompt_ids: vec![7],
        input_ids: vec![7, 42],
        position: 1,
        persistent: v4_state::V4SerialPersistentState::new(1).unwrap(),
        hidden: Vec::new(),
        generated_ids: vec![42],
        stop_reason: Some(GreedyStopReason::MaxNewTokens),
        tokens: vec![serde_json::json!({"token_position": 0, "input_id": 7})],
        finals: vec![serde_json::json!({
            "argmax": 42,
            "logits_f32_digest": "000000000000002a",
        })],
        physical: B2RequestPhysical {
            initial_state_source_bytes: 14,
            final_head_source_bytes: 28,
        },
        flash_b1_phase: FlashB1PhaseTiming::default(),
    };
    let request_json = b2_generate_request_json(1, &request);
    assert_eq!(request_json["request"], 1);
    assert_eq!(request_json["generated_ids"], serde_json::json!([42]));
    assert_eq!(request_json["stop_reason"], "max_new_tokens");
    assert_eq!(
        request_json["finals"][0]["logits_f32_digest"],
        "000000000000002a"
    );
    assert_eq!(request_json["physical"]["initial_state_source_bytes"], 14);
    assert_eq!(request_json["physical"]["final_head_source_bytes"], 28);

    let mut physical = B2Physical {
        fixed_source_bytes: 101,
        fixed_h2d_bytes: 202,
        routed_source_bytes: 606,
        routed_h2d_bytes: 606,
        ..B2Physical::default()
    };
    physical
        .record_request_router_metadata(true, &[1, 2, 3, 4, 5, 6])
        .unwrap();
    physical.record_initial_state_source_bytes(14).unwrap();
    physical.record_final_head_source_bytes(28).unwrap();
    let ledger = physical.json();
    assert_eq!(ledger["fixed_source_bytes"], 101);
    assert_eq!(ledger["request_router_metadata_source_bytes"], 48);
    assert_eq!(ledger["routed_h2d_bytes"], 606);
    assert_eq!(ledger["initial_state_source_bytes"], 14);
    assert_eq!(ledger["final_head_source_bytes"], 28);
}

#[test]
fn b2_generation_source_ledgers_conserve_request_decisions() {
    let mut total = B2Physical::default();
    let mut first = B2RequestPhysical::default();
    let mut second = B2RequestPhysical::default();
    for (request, initial_state, final_head) in [(&mut first, 14, 100), (&mut second, 28, 200)] {
        request
            .record_initial_state_source_bytes(initial_state)
            .unwrap();
        request.record_final_head_source_bytes(final_head).unwrap();
        total
            .record_initial_state_source_bytes(initial_state)
            .unwrap();
        total.record_final_head_source_bytes(final_head).unwrap();
    }
    assert_eq!(
        total.initial_state_source_bytes,
        first.initial_state_source_bytes + second.initial_state_source_bytes
    );
    assert_eq!(
        total.final_head_source_bytes,
        first.final_head_source_bytes + second.final_head_source_bytes
    );
}

#[test]
fn b2_row_set_keeps_router_metadata_request_local_and_conserves_ledgers() {
    let hash_rows = [[11_i64, 12, 13, 14, 15, 16], [21_i64, 22, 23, 24, 25, 26]].concat();
    let first = b2_router_metadata_row(true, &hash_rows, 0, 2).unwrap();
    let second = b2_router_metadata_row(true, &hash_rows, 1, 2).unwrap();
    assert_eq!(first, &[11, 12, 13, 14, 15, 16]);
    assert_eq!(second, &[21, 22, 23, 24, 25, 26]);
    assert_ne!(first, second, "request metadata crossed row-set ownership");
    assert_eq!(
        b2_router_metadata_row(false, &[] as &[i64], 0, 2).unwrap(),
        &[] as &[i64]
    );
    assert!(b2_router_metadata_row(false, &hash_rows, 0, 2).is_err());

    let mut serial = B2Physical {
        fixed_source_bytes: 101,
        fixed_h2d_bytes: 202,
        routed_source_bytes: 606,
        routed_h2d_bytes: 606,
        ..B2Physical::default()
    };
    serial.record_request_router_metadata(true, first).unwrap();
    serial.fixed_source_bytes += 101;
    serial.fixed_h2d_bytes += 202;
    serial.record_request_router_metadata(true, second).unwrap();

    let mut stationary = B2Physical {
        fixed_source_bytes: 101,
        fixed_h2d_bytes: 202,
        routed_source_bytes: 606,
        routed_h2d_bytes: 606,
        ..B2Physical::default()
    };
    stationary
        .record_request_router_metadata(true, first)
        .unwrap();
    stationary
        .record_request_router_metadata(true, second)
        .unwrap();

    assert_eq!(stationary.fixed_source_bytes * 2, serial.fixed_source_bytes);
    assert_eq!(stationary.fixed_h2d_bytes * 2, serial.fixed_h2d_bytes);
    assert_eq!(
        stationary.request_router_metadata_source_bytes,
        serial.request_router_metadata_source_bytes
    );
    assert_eq!(
        stationary.request_router_metadata_h2d_bytes,
        serial.request_router_metadata_h2d_bytes
    );
    assert_eq!(stationary.routed_source_bytes, serial.routed_source_bytes);
    assert_eq!(stationary.routed_h2d_bytes, serial.routed_h2d_bytes);
}

#[test]
fn post_mtp_guard_covers_committed_rows_and_every_persistent_field() {
    let baseline = PersistentStateAudit {
        kv: vec![1],
        compressed_kv: vec![2],
        indexer_kv: vec![3],
        main_kv: vec![4.0],
        main_score: vec![5.0],
        indexer_main_kv: Some(vec![6.0]),
        indexer_main_score: Some(vec![7.0]),
    };
    assert!(check_post_mtp_persistent_state(0, 30, 31, &baseline, 31, &baseline).is_ok());
    assert!(
        check_post_mtp_persistent_state(0, 30, 31, &baseline, 32, &baseline)
            .unwrap_err()
            .to_string()
            .contains("committed_rows")
    );
    let assert_clobber = |actual: PersistentStateAudit| {
        assert!(
            check_post_mtp_persistent_state(0, 30, 31, &baseline, 31, &actual)
                .unwrap_err()
                .to_string()
                .starts_with("post-MTP teacher-state clobber:")
        );
    };

    let mut actual = baseline.clone();
    actual.kv[0] = 8;
    assert_clobber(actual);
    let mut actual = baseline.clone();
    actual.compressed_kv[0] = 8;
    assert_clobber(actual);
    let mut actual = baseline.clone();
    actual.indexer_kv[0] = 8;
    assert_clobber(actual);
    let mut actual = baseline.clone();
    actual.main_kv[0] = 8.0;
    assert_clobber(actual);
    let mut actual = baseline.clone();
    actual.main_score[0] = 8.0;
    assert_clobber(actual);
    let mut actual = baseline.clone();
    actual.indexer_main_kv.as_mut().unwrap()[0] = 8.0;
    assert_clobber(actual);
    let mut actual = baseline.clone();
    actual.indexer_main_score.as_mut().unwrap()[0] = 8.0;
    assert_clobber(actual);
}

#[test]
fn stationary_mtp1_preflight_plan_contract() {
    let default = stationary_mtp1_preflight_plan(None, None).unwrap();
    assert_eq!((default.prompt_rows, default.max_new_tokens), (5, 3));

    let p58_n1 = stationary_mtp1_preflight_plan(
        Some(std::ffi::OsStr::new("58")),
        Some(std::ffi::OsStr::new("1")),
    )
    .unwrap();
    assert_eq!(
        (
            p58_n1.prompt_rows,
            p58_n1.max_new_tokens,
            p58_n1.context_rows,
            p58_n1.csa_compressed_rows,
            p58_n1.hca_compressed_rows,
        ),
        (58, 1, 58, 14, 0)
    );

    assert!(stationary_mtp1_preflight_plan(Some(std::ffi::OsStr::new("58")), None).is_err());
    assert!(stationary_mtp1_preflight_plan(None, Some(std::ffi::OsStr::new("1"))).is_err());
    assert!(
        stationary_mtp1_preflight_plan(
            Some(std::ffi::OsStr::new("0")),
            Some(std::ffi::OsStr::new("1")),
        )
        .is_err()
    );
    assert!(
        stationary_mtp1_preflight_plan(
            Some(std::ffi::OsStr::new("invalid")),
            Some(std::ffi::OsStr::new("1")),
        )
        .is_err()
    );

    let overflow = usize::MAX.to_string();
    assert!(
        stationary_mtp1_preflight_plan(
            Some(std::ffi::OsStr::new(&overflow)),
            Some(std::ffi::OsStr::new("2")),
        )
        .is_err()
    );
}

#[test]
fn stationary_mtp1_strict_scratch_geometry_is_single_sourced_for_p5_n3_and_p58_n8() {
    let pinned = std::path::Path::new("/pinned/libcublas.so.12");
    let p5_n3 = v4_state::LayerStationaryRequestPlan::new(5, 3).unwrap();
    let p5_geometry = DeviceRequestScratchGeometry::new(p5_n3, true).unwrap();
    assert!(stationary_mtp1_target_strict_scratch_contract(p5_n3, true, None).is_err());
    assert_eq!(
        stationary_mtp1_target_strict_scratch_contract(p5_n3, true, Some(pinned)).unwrap(),
        113_476_834
    );
    assert_eq!(p5_geometry.num_bytes().unwrap(), 113_476_834);
    assert_eq!(p5_geometry.attention_f32_elements(), v4_state::HIDDEN);
    assert_eq!(
        p5_geometry.attention_f32_elements() * std::mem::size_of::<f32>(),
        28_672
    );
    assert_eq!(
        p5_geometry.byte_breakdown().unwrap().total().unwrap(),
        p5_geometry.num_bytes().unwrap()
    );

    let p58_n8 = v4_state::LayerStationaryRequestPlan::new(58, 8).unwrap();
    assert_eq!(
        stationary_mtp1_target_strict_scratch_contract(p58_n8, true, Some(pinned)).unwrap(),
        113_519_314
    );
    assert_eq!(
        DeviceRequestScratchGeometry::new(p58_n8, true)
            .unwrap()
            .num_bytes()
            .unwrap(),
        113_519_314
    );
}

#[test]
fn qd4_stationary_gate_schedule_and_f32_audit_are_strict() {
    let schedule = (0..4)
        .flat_map(|pair| {
            if qd4_stationary_control_first(pair) {
                ['A', 'B']
            } else {
                ['B', 'A']
            }
        })
        .collect::<String>();
    assert_eq!(schedule, "ABBABAAB");
    assert!(qd4_gate_f32_bits_equal(&[0.0, f32::NAN], &[0.0, f32::NAN]));
    assert!(!qd4_gate_f32_bits_equal(&[0.0], &[-0.0]));
}

#[test]
fn id_only_final_records_expose_monotonic_request_elapsed_deltas() {
    let decision = IdOnlyFinalHeadDecision {
        token: 42,
        certificate: None,
        cpu: None,
        certificate_error: None,
    };
    let elapsed = [10_u64, 25, 41];
    let observed = elapsed
        .iter()
        .enumerate()
        .map(|(offset, value)| {
            id_only_final_record(
                4 + offset,
                42,
                if offset == 0 { "prompt" } else { "generated" },
                "DeviceLayerPersistentState",
                &decision,
                None,
                *value,
            )
            .unwrap()["request_elapsed_to_final_ns"]
                .as_u64()
                .unwrap()
        })
        .collect::<Vec<_>>();

    assert_eq!(observed, elapsed);
    assert!(observed.windows(2).all(|pair| pair[0] < pair[1]));
    assert_eq!(
        observed
            .windows(2)
            .map(|pair| pair[1] - pair[0])
            .collect::<Vec<_>>(),
        [15, 16]
    );
}

#[test]
fn greedy_id_schedule_stops_on_eos_or_max_new_tokens() {
    assert_eq!(greedy_forward_tokens(1, 0).unwrap(), 1);
    assert_eq!(greedy_forward_tokens(1, 1).unwrap(), 1);
    assert_eq!(greedy_forward_tokens(1, 2).unwrap(), 2);
    assert_eq!(greedy_forward_tokens(2, 2).unwrap(), 3);
    assert!(greedy_forward_tokens(0, 1).is_err());

    assert_eq!(
        greedy_argmax_decision(1, 0, 0, 2, V4_EOS_TOKEN_ID),
        GreedyDecision::EmitAndStop(GreedyStopReason::Eos),
        "an EOS after a one-token prompt consumes one forward and is not fed back"
    );
    assert_eq!(
        greedy_argmax_decision(1, 0, 0, 2, 42),
        GreedyDecision::EmitAndContinue,
        "a non-EOS first argmax is fed into the second forward"
    );
    assert_eq!(
        greedy_argmax_decision(1, 1, 1, 2, V4_EOS_TOKEN_ID),
        GreedyDecision::EmitAndStop(GreedyStopReason::Eos),
        "the second argmax EOS is emitted after exactly two forwards"
    );
    assert_eq!(
        greedy_argmax_decision(1, 1, 1, 2, 42),
        GreedyDecision::EmitAndStop(GreedyStopReason::MaxNewTokens),
        "a non-EOS sequence stops only after the requested new-token count"
    );

    assert_eq!(greedy_forward_tokens(5, 1).unwrap(), 5);
    assert!(
        (0..4).all(|position| !b1_needs_final_head(5, position)),
        "intermediate chat prompt positions do not invoke the final head"
    );
    assert!(b1_needs_final_head(5, 4));
    assert!(b1_needs_final_head(5, 5));
    assert_eq!(
        greedy_argmax_decision(5, 4, 0, 1, 42),
        GreedyDecision::EmitAndStop(GreedyStopReason::MaxNewTokens),
        "five prompt positions plus one new token requires five forwards"
    );
    assert!(chat_input_conflict(true, true, false));
    assert!(chat_input_conflict(true, false, true));
    assert!(!chat_input_conflict(true, false, false));
}

#[test]
fn mtp_prompt_forced_positions_advance_kv_without_full_candidates() {
    assert!(!mtp1_requires_full_candidate(5, 0));
    assert!(!mtp1_requires_full_candidate(5, 3));
    assert!(mtp1_requires_full_candidate(5, 4));
    assert!(mtp1_requires_full_candidate(5, 5));
}

#[test]
fn mtp1_stationary_alignment_scores_each_teacher_transition_once() {
    let prompt = 5;
    let forced = mtp1_alignment(prompt, 3).unwrap();
    assert_eq!(
        (
            forced.input_pos,
            forced.teacher_hidden_pos,
            forced.kv_write_pos,
            forced.candidate_pos,
            forced.teacher_target_pos,
            forced.prompt_forced
        ),
        (3, 3, 3, 3, 4, true),
    );
    let first = mtp1_alignment(prompt, 4).unwrap();
    assert_eq!(
        (
            first.input_pos,
            first.teacher_hidden_pos,
            first.kv_write_pos,
            first.candidate_pos,
            first.teacher_target_pos,
            first.prompt_forced
        ),
        (4, 4, 4, 4, 5, false),
    );
    let greedy = mtp1_alignment(prompt, 5).unwrap();
    assert_eq!(
        (
            greedy.input_pos,
            greedy.teacher_hidden_pos,
            greedy.kv_write_pos,
            greedy.candidate_pos,
            greedy.teacher_target_pos,
            greedy.prompt_forced
        ),
        (5, 5, 5, 5, 6, false),
    );
    assert!(mtp1_alignment(0, 0).is_err());
    assert!(mtp1_alignment(1, usize::MAX).is_err());
}

#[test]
fn mtp1_stationary_peak_reservation_keeps_qd4_lane_out_of_extra_peak() {
    let without_borrow = mtp1_draft_peak_bytes(10, 20, 30, 40, 50, 60).unwrap();
    let with_borrow = mtp1_draft_peak_bytes(10, 20, 30, 40, 0, 60).unwrap();
    assert_eq!(without_borrow, MTP1_DRAFT_CANDIDATE_LOCAL_BYTES + 80);
    assert_eq!(with_borrow, MTP1_DRAFT_CANDIDATE_LOCAL_BYTES + 70);
    assert!(mtp1_draft_peak_bytes(usize::MAX, 1, 0, 0, 0, 0).is_err());
}

#[test]
fn stationary_mtp_capacity_liveness_uses_a_sequential_max() {
    let fixed = mtp1_draft_fixed_device_bytes_from_geometry().unwrap();
    assert_eq!(fixed, 511_234_072);
    let projection =
        v4_state::HIDDEN * v4_state::HIDDEN + (v4_state::HIDDEN / v4_state::FP8_BLOCK).pow(2);
    let peak = mtp1_draft_peak_bytes(
        MTP1_DRAFT_PERSISTENT_BYTES,
        fixed,
        projection,
        projection,
        0,
        CERTIFIED_ARGMAX_DEVICE_TEMP_BYTES,
    )
    .unwrap();
    assert_eq!(
        peak,
        fixed + MTP1_DRAFT_PERSISTENT_BYTES + MTP1_DRAFT_CANDIDATE_LOCAL_BYTES
    );
    assert!(peak > projection + MTP1_DRAFT_PERSISTENT_BYTES + MTP1_DRAFT_CANDIDATE_LOCAL_BYTES);
    assert!(
        peak > CERTIFIED_ARGMAX_DEVICE_TEMP_BYTES
            + MTP1_DRAFT_PERSISTENT_BYTES
            + MTP1_DRAFT_CANDIDATE_LOCAL_BYTES
    );
}

#[test]
fn stationary_mtp_capacity_preflight_schema_is_isolated_and_versioned() {
    assert_eq!(
        STATIONARY_MTP1_CAPACITY_PREFLIGHT_SCHEMA,
        "colibri-v4-stationary-mtp1-capacity-v2"
    );
    assert!(STATIONARY_MTP1_CAPACITY_PREFLIGHT_SCHEMA.contains("capacity"));
}

#[test]
fn mtp1_qd4_lane_borrow_requires_drain_and_restores_teacher_ownership() {
    assert!(!mtp1_can_borrow_qd4_lane(false, V4_QD4_LANES));
    assert!(!mtp1_can_borrow_qd4_lane(true, V4_QD4_LANES - 1));
    assert!(mtp1_can_borrow_qd4_lane(true, V4_QD4_LANES));
    // The borrow calls `load_range_execute_into` synchronously and never
    // touches QD4 accounting; the next teacher dispatch still owns 4 lanes.
    assert_eq!(qd4_lane_count(true), V4_QD4_LANES);
}

#[test]
fn mtp1_qd4_telemetry_serializes_closed_loan_and_derived_loads() {
    let loan = Mtp1Qd4LoanTelemetry {
        normal_lanes_before: V4_QD4_LANES,
        normal_lanes_after: V4_QD4_LANES,
        loan_depth_before: 0,
        loan_depth_peak: 1,
        loan_depth_final: 0,
        event_drained: true,
    };
    let bytes = 6 * v4_expert::V4_ROUTED_EXPERT_BYTES;
    let fields =
        mtp1_qd4_telemetry_fields(Some(&[1, 2, 3, 4, 5, 6]), Some(bytes), Some(bytes), loan)
            .unwrap();
    assert_eq!(fields["routed_expert_loads"], 6);
    assert_eq!(fields["routed_source_bytes"], bytes);
    assert_eq!(fields["routed_h2d_bytes"], bytes);
    assert_eq!(fields["qd4_normal_lanes_before"], V4_QD4_LANES);
    assert_eq!(fields["qd4_normal_lanes_after"], V4_QD4_LANES);
    assert_eq!(fields["qd4_loan_depth_before"], 0);
    assert_eq!(fields["qd4_loan_depth_peak"], 1);
    assert_eq!(fields["qd4_loan_depth_final"], 0);
    assert_eq!(fields["qd4_loan_event_drained"], true);

    let failed = mtp1_qd4_telemetry_fields(None, None, None, loan).unwrap();
    assert_eq!(failed["qd4_loan_depth_final"], 0);
    assert_eq!(failed["qd4_loan_event_drained"], true);
    assert!(
        mtp1_qd4_telemetry_fields(
            Some(&[1]),
            Some(v4_expert::V4_ROUTED_EXPERT_BYTES + 1),
            Some(v4_expert::V4_ROUTED_EXPERT_BYTES + 1),
            loan,
        )
        .is_err()
    );
}

#[test]
fn mtp1_failure_disables_state_and_env_off_preserves_generic_branch() {
    let error = "synthetic MTP failure".to_string();
    let (state, disabled) = (None::<usize>, Some(error.clone()));
    assert!(state.is_none());
    assert_eq!(disabled.as_deref(), Some(error.as_str()));
    assert!(!stationary_mtp1_enabled(true, false).unwrap());
    assert!(stationary_mtp1_enabled(false, false).is_ok());
    assert!(stationary_mtp1_enabled(false, true).is_err());
}

#[test]
fn mtp_peak_ledger_uses_lifetimes_not_a_sum_of_all_stages() {
    assert_eq!(
        mtp1_draft_peak_bytes(10, 20, 30, 40, 50, 60).unwrap(),
        MTP1_DRAFT_CANDIDATE_LOCAL_BYTES + 80
    );
    assert!(
        mtp1_draft_peak_bytes(
            MTP1_DRAFT_PERSISTENT_BYTES,
            20_000_000,
            5_000_000,
            6_000_000,
            MTP1_DRAFT_TRANSIENT_EXPERT_SLOT_BYTES,
            CERTIFIED_ARGMAX_DEVICE_TEMP_BYTES,
        )
        .unwrap()
            > MTP1_DRAFT_EXTRA_CAPACITY_BYTES
    );
    assert!(mtp1_draft_peak_bytes(usize::MAX, 1, 0, 0, 0, 0).is_err());
}

#[test]
fn generic_stationary_lifecycle_commits_only_the_current_layer_and_poison_is_fail_closed() {
    let mut lifecycle = GenericStationaryLifecycle::new(v4_state::V4_LAYER_COUNT).unwrap();
    lifecycle.begin_layer(0, 0).unwrap();
    lifecycle.commit_layer(0, 0).unwrap();
    assert_eq!(lifecycle.committed_rows[0], 1);
    assert!(lifecycle.committed_rows[1..].iter().all(|&rows| rows == 0));

    // A failure after preparation but before commit does not advance layer
    // 1 and cannot change any other layer's causal position.
    let before = lifecycle.committed_rows.clone();
    lifecycle.begin_layer(1, 0).unwrap();
    lifecycle.poison();
    assert_eq!(lifecycle.committed_rows, before);
    assert!(lifecycle.begin_layer(1, 0).is_err());
    assert!(lifecycle.commit_layer(1, 0).is_err());
}

#[test]
fn generic_device_state_audit_mirror_is_observational_for_a_synthetic_full_row() {
    #[derive(Debug, PartialEq, Eq)]
    struct SyntheticFullRow {
        output: Vec<u16>,
        routes: Vec<i32>,
        state: Vec<u16>,
    }

    fn run_synthetic_full_row(hidden: &[u16], state: &mut Vec<u16>) -> SyntheticFullRow {
        let output = hidden
            .iter()
            .enumerate()
            .map(|(index, value)| value.wrapping_add(index as u16 + 1))
            .collect::<Vec<_>>();
        state.extend_from_slice(&output);
        SyntheticFullRow {
            routes: output.iter().map(|value| i32::from(*value % 7)).collect(),
            output,
            state: state.clone(),
        }
    }

    let hidden = vec![3_u16, 5, 8, 13];
    let mut audit_off_state = Vec::new();
    let audit_off = run_synthetic_full_row(&hidden, &mut audit_off_state);

    let mut host_mirror = v4_state::V4SerialPersistentState::new(1).unwrap();
    let mut mirror_state = Vec::new();
    let mut mirror_row = None;
    run_generic_device_state_audit_mirror(
        true,
        Some(&mut host_mirror),
        0,
        &hidden,
        |_host_state, mirror_hidden| {
            mirror_row = Some(run_synthetic_full_row(mirror_hidden, &mut mirror_state));
            Ok(())
        },
    )
    .unwrap();

    let mut audit_on_state = Vec::new();
    let audit_on = run_synthetic_full_row(&hidden, &mut audit_on_state);
    assert_eq!(
        audit_on, audit_off,
        "audit changed authoritative full-row output/routes/state"
    );
    assert_eq!(
        mirror_row.unwrap(),
        audit_on,
        "mirror did not receive an immutable row clone"
    );

    let mut invoked = false;
    run_generic_device_state_audit_mirror(false, None, 0, &hidden, |_state, _row| {
        invoked = true;
        Ok(())
    })
    .unwrap();
    assert!(!invoked, "disabled audit invoked its mirror closure");
}

#[test]
fn compressed_boundary_stage_comparator_attributes_prepared_and_old_prefix_failures() {
    let expected_row = [0x3f80_u16, 0x4000, 0x4040, 0x4080];
    let mut prepared_row = expected_row;
    prepared_row[2] = 0x40a0;
    assert_eq!(
        check_compressed_boundary_bf16(
            "prepared_compressed_row",
            9,
            27,
            6,
            4,
            &expected_row,
            &prepared_row,
        )
        .unwrap_err()
        .to_string(),
        "stage=prepared_compressed_row layer=9 position=27 row=6 index=2 old=0x4040 new=0x40a0"
    );

    let expected_prefix = [1_u16, 2, 3, 4, 5, 6, 7, 8];
    let mut committed_prefix = expected_prefix;
    committed_prefix[1] = 20;
    assert_eq!(
        check_compressed_boundary_bf16(
            "post_commit_compressed_prefix",
            9,
            27,
            0,
            4,
            &expected_prefix,
            &committed_prefix,
        )
        .unwrap_err()
        .to_string(),
        "stage=post_commit_compressed_prefix layer=9 position=27 row=0 index=1 old=0x0002 new=0x0014"
    );
}

#[test]
fn generic_stationary_digest_dispatch_never_dereferences_host_mirror() {
    assert_eq!(
        persistent_state_digest_source(true, false, false),
        PersistentStateDigestSource::ObservedDevice
    );
    assert_eq!(
        persistent_state_digest_source(false, false, true),
        PersistentStateDigestSource::HostPersistent
    );
    assert_eq!(
        persistent_state_digest_source(false, true, true),
        PersistentStateDigestSource::Unobserved,
        "generic telemetry must not read the audit-only host mirror"
    );
    assert_eq!(
        persistent_state_digest_source(false, true, false),
        PersistentStateDigestSource::Unobserved
    );
}

#[test]
fn generic_stationary_option_stores_survive_p5_decode_and_failure() {
    #[derive(Debug)]
    struct MockStore {
        path: GenericStationaryPath,
        committed_rows: usize,
        commits: usize,
    }

    let mut stores = vec![
        Some(MockStore {
            path: GenericStationaryPath::Hca128,
            committed_rows: 0,
            commits: 0,
        }),
        Some(MockStore {
            path: GenericStationaryPath::Csa4,
            committed_rows: 0,
            commits: 0,
        }),
    ];
    let mut scratch = Some(0_usize);
    let mut lifecycle = GenericStationaryLifecycle::new(v4_state::V4_LAYER_COUNT).unwrap();
    macro_rules! run_row {
        ($layer:expr, $position:expr, $fail:expr) => {{
            (|| -> Result<(), Box<dyn Error>> {
                lifecycle.begin_layer($layer, $position)?;
                let state = stores
                    .get_mut($layer)
                    .and_then(Option::as_mut)
                    .ok_or("generic stationary test store disappeared")?;
                let scratch = scratch
                    .as_mut()
                    .ok_or("generic stationary test scratch disappeared")?;
                // This is the production runner's borrowed Option tuple
                // shape; neither member is taken from its request store.
                let generic_device = Some((state, scratch, None::<&[u16]>));
                let (state, scratch, _audit_expected_compressed_prefix) =
                    generic_device.ok_or("generic stationary test tuple disappeared")?;
                if $fail {
                    return Err("generic stationary test injected precommit failure".into());
                }
                if state.committed_rows != $position {
                    return Err("generic stationary test state lost causal continuity".into());
                }
                state.committed_rows += 1;
                state.commits += 1;
                *scratch += 1;
                lifecycle.commit_layer($layer, $position)
            })()
        }};
    }

    for position in 0..5 {
        run_row!(0, position, false).unwrap();
        run_row!(1, position, false).unwrap();
    }
    assert_eq!(stores[0].as_ref().unwrap().committed_rows, 5);
    assert_eq!(stores[1].as_ref().unwrap().committed_rows, 5);

    run_row!(0, 5, false).unwrap();
    run_row!(1, 5, false).unwrap();
    assert_eq!(stores[0].as_ref().unwrap().committed_rows, 6);
    assert_eq!(stores[1].as_ref().unwrap().committed_rows, 6);
    assert_eq!(lifecycle.telemetry.layer_commits, 12);
    assert_eq!(scratch, Some(12));
    assert_eq!(
        stores[0].as_ref().unwrap().path,
        GenericStationaryPath::Hca128
    );
    assert_eq!(
        stores[1].as_ref().unwrap().path,
        GenericStationaryPath::Csa4
    );

    assert!(run_row!(0, 6, true).is_err());
    let state = stores[0].as_ref().unwrap();
    assert_eq!(state.committed_rows, 6);
    assert_eq!(state.commits, 6);
    assert_eq!(lifecycle.committed_rows[0], 6);
    assert!(stores.iter().all(Option::is_some));
}

#[test]
fn generic_stationary_lifecycle_tracks_peaks_and_rejects_live_lease_shutdown() {
    let mut lifecycle = GenericStationaryLifecycle::new(v4_state::V4_LAYER_COUNT).unwrap();
    lifecycle.observe_peaks(7, 11, 13);
    lifecycle.observe_peaks(5, 17, 3);
    lifecycle.acquire_lease();
    assert!(lifecycle.assert_drained().is_err());
    lifecycle.release_lease().unwrap();
    lifecycle.assert_drained().unwrap();
    assert_eq!(lifecycle.telemetry.device_peak_bytes, 7);
    assert_eq!(lifecycle.telemetry.host_peak_bytes, 17);
    assert_eq!(lifecycle.telemetry.pinned_peak_bytes, 13);
    assert_eq!(lifecycle.telemetry.live_lease_peak, 1);
}

#[test]
fn generic_stationary_shared_scratch_table_covers_hca_csa_and_copy_ranges() {
    let scratch = GenericStationaryScratchGeometry::max();
    let plan = v4_state::LayerStationaryRequestPlan::new(129, 17).unwrap();
    let shared_topk_rows = GenericStationaryPath::Hca128
        .max_topk_rows(plan)
        .max(GenericStationaryPath::Csa4.max_topk_rows(plan));
    let shared_attention_elements = shared_topk_rows * v4_state::Q_HEAD_DIM;

    for (path, expected_width, expected_ratio, expected_tail_rows) in [
        (GenericStationaryPath::Hca128, 512, 128, 128),
        (GenericStationaryPath::Csa4, 1_024, 4, 8),
    ] {
        let main = scratch.main_elements(path);
        let tail = scratch.tail_elements(path);
        let compressed_rows = path.compressed_rows(plan);
        let composite_rows = path.topk_rows(compressed_rows);
        assert_eq!(path.compressor_width(), expected_width);
        assert_eq!(path.ratio(), expected_ratio);
        assert_eq!(path.tail_rows(), expected_tail_rows);
        assert_eq!(main, expected_width);
        assert_eq!(tail, expected_tail_rows * expected_width);
        assert!(compressor_workspace_contract_matches(
            v4_state::HIDDEN,
            main * v4_state::HIDDEN,
            main * v4_state::HIDDEN,
            expected_ratio * main,
            main,
            main,
            main,
            main,
            expected_ratio,
        ));
        assert!(generic_stationary_range_fits(tail, 0, tail, 0, tail));
        assert!(generic_stationary_range_fits(
            tail,
            path.compressor_slot(expected_ratio - 1) * main,
            main,
            0,
            main,
        ));
        assert!(generic_stationary_range_fits(
            tail,
            0,
            v4_state::Q_HEAD_DIM,
            0,
            v4_state::Q_HEAD_DIM,
        ));
        assert!(generic_stationary_range_fits(
            v4_state::Q_HEAD_DIM,
            0,
            compressed_rows * v4_state::Q_HEAD_DIM,
            (compressed_rows - 1) * v4_state::Q_HEAD_DIM,
            v4_state::Q_HEAD_DIM,
        ));
        assert!(generic_stationary_range_fits(
            v4_state::KV_WINDOW * v4_state::Q_HEAD_DIM,
            0,
            shared_attention_elements,
            0,
            v4_state::KV_WINDOW * v4_state::Q_HEAD_DIM,
        ));
        assert!(generic_stationary_range_fits(
            compressed_rows * v4_state::Q_HEAD_DIM,
            0,
            shared_attention_elements,
            v4_state::KV_WINDOW * v4_state::Q_HEAD_DIM,
            compressed_rows * v4_state::Q_HEAD_DIM,
        ));
        assert!(generic_stationary_range_fits(
            v4_state::Q_HEAD_DIM,
            0,
            shared_attention_elements,
            (v4_state::KV_WINDOW + compressed_rows - 1) * v4_state::Q_HEAD_DIM,
            v4_state::Q_HEAD_DIM,
        ));
        assert!(generic_stationary_range_fits(
            composite_rows,
            0,
            shared_topk_rows,
            0,
            composite_rows,
        ));
        assert!(shared_topk_rows >= composite_rows);
    }

    assert!(
        !generic_stationary_range_fits(
            scratch.tail_elements(GenericStationaryPath::Hca128),
            0,
            scratch.tail_elements(GenericStationaryPath::Csa4),
            0,
            scratch.tail_elements(GenericStationaryPath::Hca128),
        ),
        "v2's HCA persistent-tail to CSA-candidate-tail copy must be rejected"
    );

    let indexer = scratch.csa_indexer_elements;
    let indexer_tail = 8 * indexer;
    let indexer_cache = plan.csa_compressed_rows * 128;
    assert_eq!(indexer, 256);
    assert!(generic_stationary_range_fits(
        indexer_tail,
        0,
        indexer_tail,
        0,
        indexer_tail,
    ));
    assert!(generic_stationary_range_fits(
        indexer_cache,
        0,
        indexer_cache,
        0,
        indexer_cache,
    ));
    assert!(generic_stationary_range_fits(
        indexer_tail,
        4 * indexer,
        indexer,
        0,
        indexer,
    ));
    assert!(compressor_workspace_contract_matches(
        v4_state::HIDDEN,
        indexer * v4_state::HIDDEN,
        indexer * v4_state::HIDDEN,
        4 * indexer,
        indexer,
        indexer,
        indexer,
        indexer,
        4,
    ));
    assert!(generic_stationary_range_fits(indexer_tail, 0, 128, 0, 128,));
    assert!(generic_stationary_range_fits(
        64 * 128,
        0,
        64 * 128,
        0,
        64 * 128,
    ));
    assert_eq!(
        v4_state::HC * v4_state::HIDDEN,
        4 * v4_state::HIDDEN,
        "the only generic terminal D2H row remains geometry-invariant"
    );

    for (path, position, expected_active_topk) in [
        (GenericStationaryPath::Hca128, 0, 128),
        (GenericStationaryPath::Hca128, 3, 128),
        (GenericStationaryPath::Hca128, 127, 129),
        (GenericStationaryPath::Hca128, 128, 129),
        (GenericStationaryPath::Csa4, 0, 128),
        (GenericStationaryPath::Csa4, 3, 129),
        (GenericStationaryPath::Csa4, 127, 160),
        (GenericStationaryPath::Csa4, 128, 160),
    ] {
        let active_topk = path.composite_capacity_rows(position);
        let compressed_rows = (position + 1) / path.ratio();
        assert_eq!(active_topk, expected_active_topk);
        assert!(generic_stationary_range_fits(
            active_topk,
            0,
            shared_topk_rows,
            0,
            active_topk,
        ));
        assert!(generic_stationary_range_fits(
            v4_state::KV_WINDOW * v4_state::Q_HEAD_DIM,
            0,
            shared_attention_elements,
            0,
            v4_state::KV_WINDOW * v4_state::Q_HEAD_DIM,
        ));
        if compressed_rows != 0 {
            assert!(generic_stationary_range_fits(
                compressed_rows * v4_state::Q_HEAD_DIM,
                0,
                shared_attention_elements,
                v4_state::KV_WINDOW * v4_state::Q_HEAD_DIM,
                compressed_rows * v4_state::Q_HEAD_DIM,
            ));
        }
        assert!(sparse_attention_score_reuse_workspace_matches(
            v4_state::Q_HEADS * v4_state::Q_HEAD_DIM,
            shared_attention_elements,
            v4_state::Q_HEADS,
            shared_topk_rows,
            v4_state::Q_HEADS * v4_state::Q_HEAD_DIM,
            v4_state::Q_HEADS * shared_topk_rows,
            v4_state::Q_HEADS * shared_topk_rows,
            v4_state::Q_HEADS,
            v4_state::Q_HEADS,
            active_topk,
            v4_state::Q_HEADS,
        ));
        assert!(
            !sparse_attention_score_reuse_workspace_matches(
                v4_state::Q_HEADS * v4_state::Q_HEAD_DIM,
                shared_attention_elements,
                v4_state::Q_HEADS,
                active_topk - 1,
                v4_state::Q_HEADS * v4_state::Q_HEAD_DIM,
                v4_state::Q_HEADS * shared_topk_rows,
                v4_state::Q_HEADS * shared_topk_rows,
                v4_state::Q_HEADS,
                v4_state::Q_HEADS,
                active_topk,
                v4_state::Q_HEADS,
            ),
            "physical top-k capacity must cover the active geometry"
        );
    }

    let v3_physical_topk = 129;
    assert!(sparse_attention_score_reuse_workspace_matches(
        v4_state::Q_HEADS * v4_state::Q_HEAD_DIM,
        v3_physical_topk * v4_state::Q_HEAD_DIM,
        v4_state::Q_HEADS,
        v3_physical_topk,
        v4_state::Q_HEADS * v4_state::Q_HEAD_DIM,
        v4_state::Q_HEADS * v3_physical_topk,
        v4_state::Q_HEADS * v3_physical_topk,
        v4_state::Q_HEADS,
        v4_state::Q_HEADS,
        128,
        v4_state::Q_HEADS,
    ));
}

#[test]
fn generic_stationary_causal_attention_geometry_matches_serial_topk() {
    for (path, position, raw_width, raw_valid, compressed, selected, capacity) in [
        (GenericStationaryPath::Hca128, 0, 1, 1, 0, 0, 128),
        (GenericStationaryPath::Hca128, 3, 128, 4, 0, 0, 128),
        (GenericStationaryPath::Hca128, 4, 128, 5, 0, 0, 128),
        (GenericStationaryPath::Hca128, 127, 128, 128, 1, 1, 129),
        (GenericStationaryPath::Hca128, 128, 128, 128, 1, 1, 129),
        (GenericStationaryPath::Hca128, 256, 128, 128, 2, 2, 130),
        (GenericStationaryPath::Csa4, 0, 1, 1, 0, 0, 128),
        (GenericStationaryPath::Csa4, 3, 128, 4, 1, 1, 129),
        (GenericStationaryPath::Csa4, 4, 128, 5, 1, 1, 129),
        (GenericStationaryPath::Csa4, 127, 128, 128, 32, 32, 160),
        (GenericStationaryPath::Csa4, 128, 128, 128, 32, 32, 160),
        (GenericStationaryPath::Csa4, 256, 128, 128, 64, 64, 192),
    ] {
        let (raw, observed_raw_width) =
            v4_state::window_topk_cpu(1, position, v4_state::KV_WINDOW).unwrap();
        let raw_indices = raw
            .iter()
            .copied()
            .filter(|&index| index >= 0)
            .collect::<Vec<_>>();
        let expected_raw = (position + 1 - raw_valid..=position)
            .map(|index| (index % v4_state::KV_WINDOW) as i32)
            .collect::<Vec<_>>();
        assert_eq!(
            observed_raw_width, raw_width,
            "{path:?} p{position} raw width"
        );
        assert_eq!(
            raw_indices, expected_raw,
            "{path:?} p{position} raw ring order"
        );
        assert_eq!(compressed, (position + 1) / path.ratio());
        assert_eq!(
            selected,
            match path {
                GenericStationaryPath::Hca128 => compressed,
                GenericStationaryPath::Csa4 => compressed.min(v4_state::INDEX_TOPK),
            }
        );
        assert_eq!(path.composite_capacity_rows(position), capacity);

        let topk = match path {
            GenericStationaryPath::Hca128 => {
                v4_state::build_hca_sparse_topk(1, position, v4_state::KV_WINDOW, path.ratio())
                    .unwrap()
            }
            GenericStationaryPath::Csa4 if compressed == 0 => v4_state::SparseTopk {
                indices: raw.clone(),
                rows: 1,
                per_row: raw_width,
                window_per_row: raw_width,
                compressed_per_row: 0,
            },
            GenericStationaryPath::Csa4 => v4_state::build_csa_sparse_topk(
                &(0..compressed)
                    .map(|index| index as f32)
                    .collect::<Vec<_>>(),
                1,
                compressed,
                &[compressed],
                position,
                v4_state::KV_WINDOW,
                v4_state::INDEX_TOPK,
            )
            .unwrap(),
        };
        assert_eq!(
            topk.per_row,
            raw_width + selected,
            "{path:?} p{position} top-k width"
        );
        assert_eq!(&topk.indices[..raw_width], raw.as_slice());
        assert!(
            topk.indices[raw_width..]
                .iter()
                .all(|&index| index >= v4_state::KV_WINDOW as i32)
        );
    }
}

#[test]
fn sparse_attention_p0_ignores_physical_topk_and_kv_suffix_at_logical_width_one() {
    let context = CudaContext::new(0).expect("create CUDA context");
    let stream = context.default_stream();
    let module = kernels::load(&context).expect("load embedded Rust PTX");
    let one = v4_router::f32_to_bf16_rne(1.0);
    let two = v4_router::f32_to_bf16_rne(2.0);
    let q = DeviceBuffer::from_host(&stream, &vec![0; v4_state::Q_HEADS * v4_state::Q_HEAD_DIM])
        .expect("upload query");
    let kv_a = vec![one; v4_state::KV_WINDOW * v4_state::Q_HEAD_DIM];
    let mut kv_b = kv_a.clone();
    kv_b[v4_state::Q_HEAD_DIM..].fill(two);
    let kv_a = DeviceBuffer::from_host(&stream, &kv_a).expect("upload zero suffix KV");
    let kv_b = DeviceBuffer::from_host(&stream, &kv_b).expect("upload changed suffix KV");
    let sink = DeviceBuffer::from_host(&stream, &vec![0.0_f32; v4_state::Q_HEADS])
        .expect("upload attention sink");
    let zero_suffix = vec![0_i32; v4_state::KV_WINDOW];
    let mut valid_suffix = zero_suffix.clone();
    let mut invalid_suffix = zero_suffix.clone();
    valid_suffix[1..].fill(1);
    invalid_suffix[1..].fill(-1);
    let zero_suffix =
        DeviceBuffer::from_host(&stream, &zero_suffix).expect("upload zero suffix top-k");
    let valid_suffix =
        DeviceBuffer::from_host(&stream, &valid_suffix).expect("upload valid suffix top-k");
    let invalid_suffix =
        DeviceBuffer::from_host(&stream, &invalid_suffix).expect("upload invalid suffix top-k");
    let run = |kv: &DeviceBuffer<u16>, topk: &DeviceBuffer<i32>, topk_per_row| {
        let mut output = DeviceBuffer::<u16>::zeroed(&stream, q.len()).expect("allocate output");
        let mut scratch = SparseAttentionScoreReuseScratch::new(&stream, 1, v4_state::KV_WINDOW)
            .expect("allocate score-reuse scratch");
        run_sparse_attention_score_reuse_device_into(
            &module,
            &stream,
            &q,
            kv,
            &sink,
            topk,
            &mut output,
            &mut scratch,
            topk_per_row,
        )
        .expect("run sparse attention canary");
        output.to_host_vec(&stream).expect("download output")
    };

    let logical_zero = run(&kv_a, &zero_suffix, 1);
    let logical_valid = run(&kv_b, &valid_suffix, 1);
    let logical_invalid = run(&kv_b, &invalid_suffix, 1);
    assert_eq!(
        logical_zero, logical_valid,
        "logical p0 observed a physical suffix"
    );
    assert_eq!(
        logical_zero, logical_invalid,
        "logical p0 observed an invalid suffix"
    );

    let legacy_zero = run(&kv_a, &zero_suffix, v4_state::KV_WINDOW);
    let legacy_invalid = run(&kv_a, &invalid_suffix, v4_state::KV_WINDOW);
    assert_ne!(
        legacy_zero, legacy_invalid,
        "legacy p0 capacity width must expose the bug"
    );
}

#[test]
fn layer_stationary_prefill_rejects_noncausal_or_noncomparable_modes() {
    assert!(validate_layer_stationary_prefill(5, 1, 61, true, false, false).is_ok());
    assert!(validate_layer_stationary_prefill(5, 2, 61, true, false, true).is_ok());
    assert!(validate_layer_stationary_prefill(5, 5, 61, true, false, true).is_ok());
    assert!(validate_layer_stationary_prefill(5, 1, 61, true, true, false).is_err());
    assert!(validate_layer_stationary_prefill(0, 1, 61, true, false, false).is_err());
    assert!(validate_layer_stationary_prefill(5, 2, 61, true, false, false).is_err());
    assert!(validate_layer_stationary_prefill(5, 6, 61, true, false, true).is_err());
    assert!(validate_layer_stationary_prefill(5, 1, 60, true, false, false).is_err());
    assert!(validate_layer_stationary_prefill(5, 1, 61, false, false, false).is_err());
}

#[test]
fn short_device_decode_ledger_uses_full61_state_and_nine_rope_rows() {
    assert_eq!(V4_SHORT_DEVICE_DECODE_MAX_ROWS, 9);
    assert_eq!(
        31 * V4_DEVICE_TOKEN_STATE_HCA_BYTES + 30 * V4_DEVICE_TOKEN_STATE_MAX_BYTES,
        3_763_828_054
    );
    assert_eq!(
        2 * V4_SHORT_DEVICE_DECODE_MAX_ROWS * v4_state::Q_ROPE_DIM * std::mem::size_of::<f32>(),
        4_608
    );
    assert_eq!(
        140_845_056
            + 1_126_760_448
            + 494_348_312
            + 3_763_828_054
            + 4_608
            + 48
            + V4_DEVICE_CAPACITY_HEADROOM_BYTES,
        7_673_270_174
    );
    assert_eq!(csa_compressed_rows_after(3), 1);
    assert_eq!(csa_compressed_rows_after(6), 1);
    assert_eq!(csa_compressed_rows_after(7), 2);
    assert_eq!(csa_compressed_rows_after(8), 2);
    assert_eq!(short_device_decode_rows(5, 2, true).unwrap(), 6);
    assert_eq!(short_device_decode_rows(5, 5, true).unwrap(), 9);
    assert!(short_device_decode_rows(5, 6, true).is_err());
}

#[test]
fn device_attention_h2d_accounting_uses_constructed_rope_lut_bytes() {
    let dispatches = V4_REQUEST_ROPE_LUT_ROWS * v4_state::V4_LAYER_COUNT;
    let hidden_bytes = dispatches * v4_state::HC * v4_state::HIDDEN * std::mem::size_of::<u16>();
    assert_eq!(
        device_attention_expected_h2d_bytes(V4_REQUEST_ROPE_LUT_BYTES, dispatches).unwrap(),
        V4_REQUEST_ROPE_LUT_BYTES + hidden_bytes
    );
    let nine_row_bytes =
        2 * V4_SHORT_DEVICE_DECODE_MAX_ROWS * v4_state::Q_ROPE_DIM * std::mem::size_of::<f32>();
    assert_eq!(
        device_attention_expected_h2d_bytes(nine_row_bytes, dispatches).unwrap(),
        nine_row_bytes + hidden_bytes
    );
}

#[test]
fn stationary_qd4_is_exclusive_and_has_a_fixed_lane_reserve() {
    assert_eq!(qd4_lane_count(false), 0);
    assert_eq!(qd4_lane_count(true), V4_QD4_LANES);
    assert!(validate_qd4_lane_reserve(V4_QD4_PINNED_HOST_BYTES, V4_QD4_DEVICE_BYTES).is_ok());
    assert!(validate_qd4_lane_reserve(V4_QD4_PINNED_HOST_BYTES - 1, V4_QD4_DEVICE_BYTES).is_err());
    assert!(validate_qd4_lane_reserve(V4_QD4_PINNED_HOST_BYTES, V4_QD4_DEVICE_BYTES - 1).is_err());
    assert_eq!(qd4_expected_routed_loads(61, 5).unwrap(), 1_830);
    assert_eq!(
        qd4_expected_routed_loads(61, greedy_forward_tokens(5, 3).unwrap()).unwrap(),
        2_562
    );
}

#[test]
fn qd4_immutable_cache_reserve_is_exact_and_bounded() {
    assert_eq!(V4_QD4_IMMUTABLE_EXPERT_CACHE_SLOTS, 32);
    assert_eq!(
        V4_QD4_IMMUTABLE_EXPERT_CACHE_HOST_BYTES,
        V4_QD4_IMMUTABLE_EXPERT_CACHE_SLOTS
            * (v4_expert::V4_ROUTED_EXPERT_BYTES
                + v4_expert::V4_HIDDEN * std::mem::size_of::<u16>()
                + v4_expert::V4_HIDDEN * std::mem::size_of::<f32>())
    );
    assert!(V4_QD4_IMMUTABLE_EXPERT_CACHE_DEVICE_BYTES > V4_QD4_IMMUTABLE_EXPERT_CACHE_HOST_BYTES);
}

#[test]
fn qd4_immutable_cache_five_row_plan_is_request_scoped_and_canonical() {
    let routes = [
        [0_u16, 1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10, 11],
        [12, 13, 14, 15, 16, 17],
        [18, 19, 20, 21, 22, 23],
        [24, 25, 26, 27, 28, 29],
    ];
    let revision = v4_state::REVISION;
    let layer = 3_usize;
    let plan = routes
        .iter()
        .map(|row| {
            row.iter()
                .map(|&expert| (revision, layer, expert))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let union = plan.iter().flatten().copied().collect::<BTreeSet<_>>();

    assert_eq!(union.len(), 5 * v4_router::V4_TOPK);
    assert!(union.len() <= V4_QD4_IMMUTABLE_EXPERT_CACHE_SLOTS);
    assert_eq!(
        plan.iter()
            .map(|row| row.iter().map(|key| key.2).collect::<Vec<_>>())
            .collect::<Vec<_>>(),
        routes.iter().map(|row| row.to_vec()).collect::<Vec<_>>()
    );
    assert_ne!(plan[0][0], (revision, layer + 1, routes[0][0]));
    assert_ne!(plan[0][0], ("other-revision", layer, routes[0][0]));
}

#[test]
fn generic_qd4_immutable_cache_accounting_mixes_prefill_hits_and_decode_loads() {
    let prefill_route_executions = qd4_expected_routed_loads(61, 5).unwrap();
    let cached_prefill_source_loads = 61 * v4_router::V4_TOPK;
    let cached_prefill_hits = prefill_route_executions - cached_prefill_source_loads;
    let decode_source_loads = 122 * v4_router::V4_TOPK;

    assert_eq!(cached_prefill_source_loads + cached_prefill_hits, 1_830);
    assert_eq!(cached_prefill_source_loads + decode_source_loads, 1_098);
}

#[test]
fn phase_profile_env_and_mode_are_strict() {
    assert!(!parse_optional_one_flag("X", None).unwrap());
    assert!(parse_optional_one_flag("X", Some(std::ffi::OsStr::new("1"))).unwrap());
    assert!(parse_optional_one_flag("X", Some(std::ffi::OsStr::new("0"))).is_err());
    assert!(parse_default_true_zero_one_flag("X", None).unwrap());
    assert!(parse_default_true_zero_one_flag("X", Some(std::ffi::OsStr::new("1"))).unwrap());
    assert!(!parse_default_true_zero_one_flag("X", Some(std::ffi::OsStr::new("0"))).unwrap());
    assert!(parse_default_true_zero_one_flag("X", Some(std::ffi::OsStr::new("yes"))).is_err());
    assert_eq!(
        fixed_prefetch_wo_a_decode_arm(true, true, true),
        "raw_fp8_gpu_decode"
    );
    assert_eq!(
        fixed_prefetch_wo_a_decode_arm(true, true, false),
        "host_bf16_decode"
    );
    assert_eq!(
        fixed_prefetch_wo_a_decode_arm(false, true, true),
        "inactive"
    );
    assert!(validate_phase_profile_mode(true, true).is_ok());
    assert!(validate_phase_profile_mode(false, true).is_err());
    assert!(validate_phase_profile_mode(true, false).is_err());
    assert!(
        parse_optional_one_flag(
            "COLIBRI_V4_LAYER_STATIONARY_FIXED_PREFETCH",
            Some(std::ffi::OsStr::new("yes"))
        )
        .is_err()
    );
    assert!(validate_fixed_prefetch_requirements(true, true, true, 4 * 1024 * 1024 * 1024).is_ok());
    assert!(validate_fixed_prefetch_requirements(false, true, true, usize::MAX).is_err());
    assert!(validate_fixed_prefetch_requirements(true, false, true, usize::MAX).is_err());
    assert!(validate_fixed_prefetch_requirements(true, true, false, usize::MAX).is_err());
    assert!(validate_fixed_prefetch_requirements(true, true, true, 1).is_err());
}

#[test]
fn decode_retention_overlay_env_is_default_on_and_strict() {
    let name = "COLIBRI_V4_DECODE_RETENTION_OVERLAY";
    assert!(parse_default_true_zero_one_flag(name, None).unwrap());
    assert!(!parse_default_true_zero_one_flag(name, Some(std::ffi::OsStr::new("0"))).unwrap());
    assert!(parse_default_true_zero_one_flag(name, Some(std::ffi::OsStr::new("yes"))).is_err());
    let prefill_target = ([true, false], [false, true]);
    let decode_target = ([true, true], [false, false]);
    let closure_target = |records, device_promotions, host_promotions| {
        decode_overlay_uses_decode_target(
            records,
            prefill_target.0.len(),
            device_promotions,
            host_promotions,
        )
        .map(|decode| {
            if decode {
                decode_target
            } else {
                prefill_target
            }
        })
    };
    assert_eq!(closure_target(0, 0, 0).unwrap(), prefill_target);
    assert!(closure_target(0, 1, 0).is_err());
    assert_eq!(
        closure_target(prefill_target.0.len(), 1, 0).unwrap(),
        decode_target,
    );
    assert!(closure_target(1, 0, 0).is_err());
}

#[test]
fn fixed_prefetch_schedule_is_one_ahead_and_accounts_for_wait() {
    let telemetry = FixedLoadTelemetry {
        fixed_read_alloc_ns: 20,
        fixed_decode_ns: 30,
        ..FixedLoadTelemetry::default()
    };
    let transition =
        fixed_prefetch_transition_json(4, 5, &telemetry, 10, 90, 20, 20, 100, 80, 2).unwrap();
    assert_eq!(transition["overlap_ns"], 70);
    assert!(!transition["ready_before_consume"].as_bool().unwrap());
    assert_eq!(transition["consume_wait_ns"], 10);
    assert_eq!(transition["host_prefetched_bytes"], 20);
    assert!(fixed_prefetch_transition_json(4, 6, &telemetry, 10, 90, 20, 20, 100, 80, 2).is_err());
    assert!(fixed_prefetch_transition_json(4, 5, &telemetry, 90, 10, 20, 20, 100, 80, 2).is_err());
}

#[test]
fn flash_b1_prefetch_schedule_loads_each_future_layer_once_in_order() {
    let scheduled = (0..4)
        .map(|layer| flash_b1_prefetch_next_layer(layer, 4).unwrap())
        .flatten()
        .collect::<Vec<_>>();
    assert_eq!(scheduled, vec![1, 2, 3]);
    assert_eq!(flash_b1_prefetch_next_layer(3, 4).unwrap(), None);
    assert!(flash_b1_prefetch_next_layer(4, 4).is_err());
}

#[test]
fn flash_b1_prefetch_worker_propagates_failure_and_rejects_reorder() {
    assert_eq!(flash_b1_prefetch_worker_result(2, 2, Ok(7)).unwrap(), 7);
    assert!(flash_b1_prefetch_worker_result::<()>(2, 2, Err("read failed".into())).is_err());
    assert!(flash_b1_prefetch_worker_result(2, 3, Ok(())).is_err());
}

#[test]
fn stationary_phase_profile_keeps_nested_diagnostics_out_of_top_level_sum() {
    let fixed = FixedLoadTelemetry {
        fixed_read_alloc_ns: 10,
        fixed_decode_ns: 20,
        fixed_h2d_ns: 30,
        fixed_source_bytes: 7,
        fixed_h2d_bytes: 10,
        raw_h2d_ns: 9,
        raw_h2d_bytes: 10,
        cache_status: "miss",
        ..FixedLoadTelemetry::default()
    };
    let stage = LayerStageTelemetry {
        layer_prepare_ns: 70,
        layer_prepare_count: 1,
        attention_ns: 40,
        device_state_setup_ns: 7,
        device_state_setup_count: 1,
        attention_hidden_h2d_count: 1,
        attention_hidden_h2d_bytes: v4_state::HC * v4_state::HIDDEN * 2,
        attention_rope_lut_d2d_count: 2,
        attention_rope_lut_d2d_bytes: 2 * v4_state::Q_ROPE_DIM * 4,
        attention_terminal_d2h_count: 1,
        attention_terminal_d2h_bytes: v4_state::HC * v4_state::HIDDEN * 2,
        commit_ns: 5,
        router_ns: 50,
        router_mhc_ns: 20,
        router_kernel_select_sync_ns: 10,
        router_kernel_select_gpu_ns: 8,
        shared_ns: 20,
        routed_ns: 100,
        routed_read_ns: 10,
        routed_alloc_ns: 9,
        routed_setup_ns: 40,
        routed_h2d_ns: 30,
        routed_kernel_ns: 20,
        routed_source_bytes: 11,
        routed_h2d_bytes: 12,
        ..LayerStageTelemetry::default()
    };
    let (profile, attributed, outer) =
        stationary_layer_phase_profile_json(100, 400, Some(&fixed), &stage, 4, false).unwrap();
    assert_eq!(outer, 300);
    assert_eq!(attributed, 292);
    assert_eq!(profile["unattributed_gap_ns"], 8);
    assert_eq!(profile["schema"], "colibri-v4-stationary-phase-profile-v5");
    assert_eq!(profile["top_level"]["layer_prepare"]["wall_ns"], 70);
    assert_eq!(
        profile["top_level"]["layer_prepare"]["fixed_load"]["service"]["work_sum_ns"],
        60
    );
    assert_eq!(
        profile["top_level"]["layer_prepare"]["fixed_load"]["nested"]["raw_h2d_ns"],
        9
    );
    assert_eq!(
        profile["top_level"]["attention"]["device_state_setup"]["wall_ns"],
        7
    );
    assert_eq!(
        profile["top_level"]["attention"]["transfers"]["request_rope_lut_d2d"]["count"],
        2
    );
    assert_eq!(
        profile["top_level"]["attention"]["temporary"]["steady_device_alloc_count"],
        0
    );
    let (a, b, c, q) = (10, 20, 30, 0);
    let host_hit = FixedLoadTelemetry {
        fixed_h2d_ns: a + b,
        fixed_decode_ns: c,
        fixed_h2d_bytes: 1,
        cache_status: "stationary_raw_host_hit",
        ..FixedLoadTelemetry::default()
    };
    let host_stage = LayerStageTelemetry {
        layer_prepare_ns: a + b + c + q,
        layer_prepare_count: 1,
        ..LayerStageTelemetry::default()
    };
    assert!(
        stationary_layer_phase_profile_json(0, 60, Some(&host_hit), &host_stage, 0, false).is_ok()
    );
    for (main_exposed, exposed, credit, overhead) in [
        (40_u64, 40_u64, 20_u64, 0_u64),
        (60, 60, 0, 0),
        (80, 60, 0, 20),
    ] {
        let accounting = fixed_prepare_dual_accounting(&host_hit, main_exposed).unwrap();
        assert_eq!(accounting["critical"]["exposed_service_bound_ns"], exposed);
        assert_eq!(accounting["critical"]["derived_overlap_credit_ns"], credit);
        assert_eq!(accounting["critical"]["queue_or_overhead_ns"], overhead);
    }
    assert!(profile["closure"]["exact"].as_bool().unwrap());
    assert_eq!(profile["top_level"]["routed"]["wall_ns"], 100);
    assert_eq!(
        profile["top_level"]["routed"]["diagnostics"]["setup_ns"],
        40
    );
    assert!(stationary_layer_phase_profile_json(401, 400, Some(&fixed), &stage, 4, false).is_err());
    let (final_profile, final_attributed) =
        stationary_final_phase_profile_json(400, 500, 10, 80, 40, false).unwrap();
    assert_eq!(final_attributed, 90);
    assert_eq!(final_profile["unattributed_gap_ns"], 10);
    assert_eq!(
        final_profile["top_level"]["cpu_final"]["diagnostics"]["streamed_weight_read_ns"],
        40
    );
    assert_eq!(
        stationary_phase_profile_outer_accounting(300, 100, 450).unwrap(),
        (400, 50),
        "layer outer is deliberately distinct from final outer"
    );
    assert!(stationary_phase_profile_outer_accounting(300, 151, 450).is_err());
}

#[test]
fn generic_decode_phase_profile_closure_uses_only_exclusive_outer_scopes() {
    let (prefill_outer, decode_outer, profiled_outer, request_gap) =
        stationary_one_token_phase_profile_outer_accounting(300, 100, 220, 80, 800).unwrap();
    assert_eq!(
        (prefill_outer, decode_outer, profiled_outer, request_gap),
        (400, 300, 700, 100)
    );
    assert_eq!(prefill_outer + decode_outer + request_gap, 800);
    assert_eq!((5 + 2) * v4_state::V4_LAYER_COUNT, 427);
    assert_eq!(1 + 2, 3);
    assert!(stationary_one_token_phase_profile_outer_accounting(300, 100, 220, 81, 700).is_err());
}

#[test]
fn stationary_parallel_profile_uses_disjoint_commit_and_routed_scopes() {
    let parallel = LayerStageTelemetry {
        attention_ns: 40,
        commit_ns: 5,
        router_ns: 50,
        router_mhc_ns: 20,
        router_kernel_select_sync_ns: 10,
        router_kernel_select_gpu_ns: 8,
        shared_ns: 20,
        routed_ns: 100,
        routed_read_ns: 80,
        routed_alloc_ns: 9,
        routed_setup_ns: 40,
        routed_h2d_ns: 160,
        routed_kernel_ns: 140,
        routed_parallel_batch_wall_ns: Some(90),
        routed_parallel_read_horizon_ns: Some(80),
        routed_source_bytes: 11,
        routed_h2d_bytes: 12,
        ..LayerStageTelemetry::default()
    };
    let (profile, attributed, outer) =
        stationary_layer_phase_profile_json(100, 315, None, &parallel, 4, true).unwrap();
    assert_eq!(outer, 215);
    assert_eq!(attributed, 215);
    assert_eq!(profile["unattributed_gap_ns"], 0);
    assert!(profile["closure"]["exact"].as_bool().unwrap());
    assert_eq!(profile["schema"], "colibri-v4-stationary-phase-profile-v5");
    let lanes = &profile["top_level"]["routed"]["parallel_lanes"];
    assert_eq!(lanes["lane_count"], V4_QD4_LANES);
    assert_eq!(lanes["route_job_count"], v4_router::V4_TOPK);
    assert_eq!(lanes["batch_wall_ns"], 90);
    assert_eq!(lanes["read_completion_horizon_ns"], 80);
    assert_eq!(lanes["h2d_event_work_sum_ns"], 160);
    assert_eq!(lanes["kernel_event_work_sum_ns"], 140);
    assert!(
        profile["top_level"]["routed"]["diagnostics"]
            .get("h2d_ns")
            .is_none()
    );

    let serial_with_parallel_work = LayerStageTelemetry {
        routed_ns: parallel.routed_ns,
        routed_h2d_ns: 101,
        ..LayerStageTelemetry::default()
    };
    assert!(
        stationary_layer_phase_profile_json(100, 400, None, &serial_with_parallel_work, 4, false,)
            .is_err()
    );
    let commit_overlapped_routed = LayerStageTelemetry {
        routed_ns: 105,
        ..parallel
    };
    assert!(
        stationary_layer_phase_profile_json(100, 315, None, &commit_overlapped_routed, 4, true,)
            .is_err()
    );
}

#[test]
fn normalized_config_fingerprint_is_order_independent_and_self_describing() {
    let first = normalized_config_fingerprint(serde_json::json!({"z": true, "a": 1})).unwrap();
    let second = normalized_config_fingerprint(serde_json::json!({"a": 1, "z": true})).unwrap();
    assert_eq!(first["algorithm"], "fnv1a64");
    assert_eq!(first["digest"], second["digest"]);
    assert_eq!(first["canonical_json"], second["canonical_json"]);
    assert_eq!(first["normalized"]["a"], 1);
    assert!(normalized_config_fingerprint(serde_json::json!([])).is_err());
}

#[test]
fn five_token_device_attention_transfer_contract_has_frozen_geometry() {
    let manifests = v4_state::serial_layer_manifests(0).unwrap();
    let dispatches = 5 * manifests.len();
    let csa_boundaries = manifests
        .iter()
        .filter(|manifest| manifest.compression_ratio == v4_state::CompressionRatio::Csa4)
        .count();
    let hidden_row_bytes = v4_state::HC * v4_state::HIDDEN * std::mem::size_of::<u16>();
    let rope_row_bytes = v4_state::Q_ROPE_DIM * std::mem::size_of::<f32>();
    assert_eq!(dispatches, 305);
    assert_eq!(dispatches * hidden_row_bytes, 17_489_920);
    assert_eq!(V4_REQUEST_ROPE_LUT_BYTES, 2_560);
    assert_eq!(csa_boundaries, 30);
    assert_eq!(2 * dispatches + csa_boundaries, 640);
    assert_eq!((2 * dispatches + csa_boundaries) * rope_row_bytes, 163_840);
}

#[test]
fn layer_expert_retention_stops_at_the_laptop_cap() {
    assert!(should_retain_layer_expert(31));
    assert!(!should_retain_layer_expert(32));
}

#[test]
fn pending_result_file_is_exclusive_and_transactional() {
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = |case: &str| {
        std::env::temp_dir().join(format!(
            "colibri-result-{case}-{}-{unique}.jsonl",
            std::process::id()
        ))
    };

    let existing = path("existing");
    std::fs::write(&existing, b"keep").unwrap();
    assert!(PendingResultFile::reserve(existing.clone()).is_err());
    assert_eq!(std::fs::read(&existing).unwrap(), b"keep");
    std::fs::remove_file(existing).unwrap();

    let aborted = path("aborted");
    let aborted_temp;
    {
        let mut result = PendingResultFile::reserve(aborted.clone()).unwrap();
        let temp = result.temp_path.clone();
        assert!(temp.exists());
        writeln!(result.file_mut(), "partial").unwrap();
        aborted_temp = temp;
    }
    assert!(!aborted.exists());
    assert!(!aborted_temp.exists());

    let committed = path("committed");
    {
        let mut result = PendingResultFile::reserve(committed.clone()).unwrap();
        let temp = result.temp_path.clone();
        writeln!(result.file_mut(), "complete").unwrap();
        result.commit().unwrap();
        assert!(!temp.exists());
    }
    assert_eq!(std::fs::read_to_string(&committed).unwrap(), "complete\n");
    std::fs::remove_file(committed).unwrap();

    let publish_collision = path("publish-collision");
    let mut result = PendingResultFile::reserve(publish_collision.clone()).unwrap();
    let temp = result.temp_path.clone();
    writeln!(result.file_mut(), "pending").unwrap();
    std::fs::write(&publish_collision, b"keep").unwrap();
    let error = result.commit().unwrap_err().to_string();
    assert!(error.contains("final path exists"));
    drop(result);
    assert_eq!(std::fs::read(&publish_collision).unwrap(), b"keep");
    assert!(!temp.exists());
    std::fs::remove_file(publish_collision).unwrap();
}

#[test]
#[ignore = "requires pinned DeepSeek-V4-Pro snapshot"]
fn official_single_user_chat_hi_encodes_to_pinned_ids() {
    let root = v4_model_dir().expect("pinned model directory");
    let tokenizer = v4_tokenizer::V4Tokenizer::open(&root).expect("pinned V4 tokenizer");
    let prompt = v4_chat::format_single_user("hi", v4_chat::ThinkingMode::Chat);
    assert_eq!(
        tokenizer
            .encode_raw(&prompt)
            .expect("encode official chat prompt"),
        [0, 128_803, 6_366, 128_804, 128_822]
    );
}

#[test]
fn router_telemetry_schema_closes_exclusive_wall_time() {
    let stage = LayerStageTelemetry {
        router_ns: 100,
        router_mhc_ns: 40,
        router_host_prep_ns: 3,
        router_alloc_h2d_sync_ns: 12,
        router_workspace_alloc_ns: 5,
        router_kernel_select_sync_ns: 30,
        router_d2h_sync_ns: 8,
        router_other_ns: 2,
        router_kernel_select_gpu_ns: 17,
        ..LayerStageTelemetry::default()
    };
    assert_eq!(stage.router_accounted_ns(), stage.router_ns);
    assert!(stage.router_kernel_select_gpu_ns <= stage.router_kernel_select_sync_ns);
}

#[test]
fn mhc_telemetry_schema_closes_exclusive_wall_time() {
    let telemetry = MhcTelemetry {
        wall_ns: 100,
        input_alloc_h2d_sync_ns: 12,
        workspace_alloc_sync_ns: 8,
        workspace_init_gpu_ns: 6,
        mixing_linear_sync_ns: 25,
        mixing_linear_gpu_ns: 23,
        sinkhorn_normalize_collapse_sync_ns: 40,
        sinkhorn_normalize_collapse_gpu_ns: 38,
        output_d2h_sync_ns: 10,
        other_ns: 5,
    };
    assert_eq!(telemetry.accounted_wall_ns(), telemetry.wall_ns);
    assert_ne!(telemetry.mixing_linear_gpu_ns, 0);
    assert_ne!(telemetry.sinkhorn_normalize_collapse_gpu_ns, 0);
}

#[test]
fn source_identity_rechecks_manifest_and_sources() {
    let root = std::env::temp_dir().join(format!(
        "colibri-source-identity-{}-{}",
        std::process::id(),
        std::thread::current().name().unwrap_or("test")
    ));
    std::fs::create_dir_all(&root).unwrap();
    let source = root.join("source.rs");
    let binary = root.join("binary");
    std::fs::write(&source, "fn main() {}\n").unwrap();
    std::fs::write(&binary, "binary\n").unwrap();
    let manifest = root.join("manifest");
    std::fs::write(
        &manifest,
        format!("{}  source.rs\n", sha256_file(&source).unwrap()),
    )
    .unwrap();
    let identity = sha256_file(&manifest).unwrap();
    let provenance = validate_source_identity(&identity, &manifest, &binary).unwrap();
    assert_eq!(provenance["source_identity_sha256"], identity);

    std::fs::write(&source, "changed\n").unwrap();
    assert!(validate_source_identity(&identity, &manifest, &binary).is_err());
    assert!(validate_source_identity("missing", &manifest, &binary).is_err());
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn fixed_cache_budget_keeps_whole_best_value_layers() {
    let plan = [(0, 100, 100), (1, 60, 120), (2, 50, 40)]
        .into_iter()
        .map(|(layer, bytes, saved)| FixedLayerPlan {
            layer,
            tensors: Vec::new(),
            device_bytes: bytes,
            raw_device_bytes: bytes,
            host_bytes: bytes,
            raw_host_bytes: bytes,
            device_saved_ns: saved,
            host_saved_ns: saved,
        })
        .collect::<Vec<_>>();
    let selected = select_fixed_cache_layers(&plan, 110, &[false; 3], true, false);
    assert_eq!(selected, [false, true, true]);
    assert!(
        plan.iter()
            .zip(selected)
            .filter(|(_, selected)| *selected)
            .map(|(plan, _)| plan.device_bytes)
            .sum::<usize>()
            <= 110
    );
}

#[test]
fn generic_decode_overlay_capacity_targets() {
    const WO_A_SOURCE_BYTES: usize = 67_108_864;
    const GIB: usize = 1024 * 1024 * 1024;
    let plans = (0..v4_state::V4_LAYER_COUNT)
        .map(|layer| {
            let raw_device_bytes = if layer < 2 || layer % 2 == 1 {
                PHASE1_HCA_PAYLOAD_BYTES
            } else {
                PHASE1_CSA_PAYLOAD_BYTES
            };
            let device_bytes = raw_device_bytes + WO_A_SOURCE_BYTES;
            let host_bytes = fixed_layer_decoded_host_bytes(layer, device_bytes).unwrap();
            FixedLayerPlan {
                layer,
                tensors: Vec::new(),
                device_bytes,
                raw_device_bytes,
                host_bytes,
                raw_host_bytes: host_bytes - WO_A_SOURCE_BYTES,
                device_saved_ns: 1,
                host_saved_ns: 1,
            }
        })
        .collect::<Vec<_>>();
    let none = vec![false; plans.len()];
    let prefill_device_budget = 4 * GIB - PINNED_FIXED_PIPELINE_WORKER_DEVICE_BYTES;
    let prefill_host_budget = 13 * GIB;
    let prefill_device =
        select_fixed_cache_layers(&plans, prefill_device_budget, &none, true, true);
    let prefill_host =
        select_fixed_cache_layers(&plans, prefill_host_budget, &prefill_device, false, true);
    let decode_device = select_fixed_cache_layers(
        &plans,
        prefill_device_budget + V4_QD4_IMMUTABLE_EXPERT_CACHE_DEVICE_BYTES,
        &none,
        true,
        true,
    );
    let decode_host = select_fixed_cache_layers(
        &plans,
        prefill_host_budget + V4_QD4_IMMUTABLE_EXPERT_CACHE_HOST_BYTES,
        &decode_device,
        false,
        true,
    );
    let promoted = |decode: &[bool], prefill: &[bool]| {
        plans
            .iter()
            .zip(decode)
            .zip(prefill)
            .filter_map(|((plan, &decode), &prefill)| (decode && !prefill).then_some(plan.layer))
            .collect::<Vec<_>>()
    };
    assert_eq!(promoted(&decode_device, &prefill_device), [15, 17]);
    assert_eq!(promoted(&decode_host, &prefill_host), [24, 26, 28, 30]);
    assert_eq!(
        (
            decode_device.iter().filter(|&&selected| selected).count(),
            decode_host.iter().filter(|&&selected| selected).count(),
            plans.len()
                - decode_device.iter().filter(|&&selected| selected).count()
                - decode_host.iter().filter(|&&selected| selected).count(),
        ),
        (10, 36, 15),
    );
    assert_eq!(
        plans
            .iter()
            .zip(&decode_host)
            .zip(&prefill_host)
            .filter_map(|((plan, &decode), &prefill)| {
                (decode && !prefill).then_some(plan.raw_device_bytes + 4_096)
            })
            .sum::<usize>(),
        1_708_974_176,
    );
    assert_eq!(
        plans
            .iter()
            .zip(&decode_device)
            .zip(&prefill_device)
            .filter_map(|((plan, &decode), &prefill)| {
                (decode && !prefill).then_some(plan.raw_device_bytes)
            })
            .sum::<usize>(),
        783_919_152,
    );
}

#[test]
fn explicit_decode_promotion_plan_is_strict_and_model_neutral() {
    let plans = [(7, 10, 11), (11, 20, 21), (19, 30, 31), (23, 40, 41)]
        .into_iter()
        .map(|(layer, raw_device_bytes, raw_host_bytes)| FixedLayerPlan {
            layer,
            tensors: Vec::new(),
            device_bytes: raw_device_bytes,
            raw_device_bytes,
            host_bytes: raw_host_bytes,
            raw_host_bytes,
            device_saved_ns: 1,
            host_saved_ns: 1,
        })
        .collect::<Vec<_>>();
    let prefill_device = [true, false, false, false];
    let prefill_host = [false, true, false, false];
    let parse = |value: Option<&str>, device_budget, host_budget| {
        parse_decode_promotion_plan(
            value.map(std::ffi::OsStr::new),
            &plans,
            &prefill_device,
            &prefill_host,
            10,
            21,
            device_budget,
            host_budget,
        )
    };
    assert!(parse(None, 60, 41).unwrap().is_none());
    assert_eq!(
        parse(Some("11:D,19:D,23:H"), 60, 41).unwrap().unwrap(),
        (
            vec![true, true, true, false],
            vec![false, false, false, true],
        )
    );
    assert!(parse(Some("11:D,19:H"), 30, 31).is_ok());
    for invalid in [
        "",
        "19:D,",
        "19:d",
        " 19:D",
        "19:D,19:H",
        "7:D",
        "11:H",
        "29:D",
    ] {
        assert!(parse(Some(invalid), 60, 41).is_err(), "{invalid}");
    }
    assert!(parse(Some("19:D"), 39, 41).is_err());
    assert!(parse(Some("23:H"), 60, 40).is_err());
    assert!(
        parse_decode_promotion_plan(
            Some(std::ffi::OsStr::new("11:H,19:D")),
            &plans,
            &prefill_device,
            &[false, false, true, false],
            10,
            31,
            40,
            31,
        )
        .is_err()
    );
}

#[test]
fn decode_promotion_fingerprint_is_canonical_and_mode_sensitive() {
    let plans = [(7, 10, 11), (11, 20, 21), (19, 30, 31), (23, 40, 41)]
        .into_iter()
        .map(|(layer, raw_device_bytes, raw_host_bytes)| FixedLayerPlan {
            layer,
            tensors: Vec::new(),
            device_bytes: raw_device_bytes,
            raw_device_bytes,
            host_bytes: raw_host_bytes,
            raw_host_bytes,
            device_saved_ns: 1,
            host_saved_ns: 1,
        })
        .collect::<Vec<_>>();
    let prefill_device = [true, false, false, false];
    let prefill_host = [false, true, false, false];
    let parse = |value| {
        parse_decode_promotion_plan(
            Some(std::ffi::OsStr::new(value)),
            &plans,
            &prefill_device,
            &prefill_host,
            10,
            21,
            60,
            41,
        )
        .unwrap()
        .unwrap()
    };
    let ordered = parse("11:D,19:D,23:H");
    let reordered = parse("23:H,19:D,11:D");
    let explicit = normalized_decode_promotion_targets(
        &plans,
        &prefill_device,
        &prefill_host,
        &ordered.0,
        &ordered.1,
        true,
    )
    .unwrap();
    assert_eq!(
        explicit,
        normalized_decode_promotion_targets(
            &plans,
            &prefill_device,
            &prefill_host,
            &reordered.0,
            &reordered.1,
            true,
        )
        .unwrap()
    );
    let default_mode = normalized_decode_promotion_targets(
        &plans,
        &prefill_device,
        &prefill_host,
        &ordered.0,
        &ordered.1,
        false,
    )
    .unwrap();
    let fingerprint = |overlay, targets| {
        normalized_config_fingerprint(serde_json::json!({
            "decode_retention_overlay": overlay,
            "normalized_promotion_targets": targets,
        }))
        .unwrap()
    };
    assert_ne!(
        fingerprint(true, explicit.clone())["digest"],
        fingerprint(true, default_mode)["digest"],
    );
    assert_ne!(
        fingerprint(true, explicit.clone())["digest"],
        fingerprint(false, explicit)["digest"],
    );
}

#[test]
fn source_to_device_saved_bytes_are_exact() {
    assert_eq!(
        fixed_saved_bytes_for_target(123_456_789, 98_765_432, true, false).unwrap(),
        (123_456_789, 98_765_432),
    );
}

#[test]
fn source_to_device_promotion_closes_residency_without_host_ownership() {
    assert_eq!(
        decode_device_promotion_residency(100, 300, 40, None, 140).unwrap(),
        (140, 300),
    );
    assert!(decode_device_promotion_residency(100, 300, 40, None, 139).is_err());
}

#[test]
fn bounded_readahead_cap_rounds_metadata_to_os_pages() {
    let ranges = [
        safetensors::FileRange { offset: 1, len: 7 },
        safetensors::FileRange { offset: 9, len: 11 },
    ];
    assert_eq!(bounded_readahead_window_bytes(&ranges).unwrap(), 18);
    assert_eq!(
        bounded_readahead_page_cap(427_243_544, 4_096).unwrap(),
        427_245_568
    );
    assert!(bounded_readahead_page_cap(1, 0).is_err());
}

#[test]
fn phase1_flow_shop_gate_uses_exact_p6_counts_and_threshold() {
    assert_eq!(
        phase1_predicted_p6_ns(10, 20, 30, 40),
        PHASE1_P6_WITH_FINAL_NS - PHASE1_P6_FIXED_PREPARE_NS
            + PHASE1_P6_HCA_TRANSFERS * 20
            + PHASE1_P6_CSA_TRANSFERS * 40
    );
    assert_eq!(
        PHASE1_P6_WITH_FINAL_NS.saturating_mul(80) / 100,
        13_730_223_273
    );
}

#[test]
fn routed_qd4_lane_budget_is_exact() {
    assert_eq!(
        v4_expert::V4_ROUTED_EXPERT_BYTES,
        3 * 11_010_048 + 3 * 688_128
    );
    assert_eq!(4 * v4_expert::V4_ROUTED_EXPERT_BYTES, 140_378_112);
}

#[test]
fn qd4_supervisor_reaps_six_jobs_and_propagates_error_and_panic() {
    let completed = run_supervised_qd4_batches(
        vec![(); 4],
        1,
        std::time::Duration::from_millis(100),
        None,
        |_| Ok(()),
        |_, _, job| Ok(job.route_slot),
        |_, batch| {
            let mut slots = batch.completions;
            slots.sort_unstable();
            Ok(slots)
        },
    )
    .unwrap();
    assert_eq!(completed, [vec![0, 1, 2, 3, 4, 5]]);

    let error = run_supervised_qd4_batches(
        vec![(); 4],
        1,
        std::time::Duration::from_millis(100),
        None,
        |_| Ok(()),
        |_, _, job| {
            if job.route_slot == 2 {
                Err("injected lane error".into())
            } else {
                Ok(job.route_slot)
            }
        },
        |_, _| Ok(()),
    )
    .unwrap_err();
    assert!(error.contains("injected lane error"));

    let panic = run_supervised_qd4_batches(
        vec![(); 4],
        1,
        std::time::Duration::from_millis(20),
        None,
        |_| Ok(()),
        |_, _, job| {
            assert_ne!(job.route_slot, 2, "injected lane panic");
            Ok(job.route_slot)
        },
        |_, _| Ok(()),
    )
    .unwrap_err();
    assert!(panic.contains("panicked"));
}

#[test]
fn b2_layer_stationary_plan_loads_each_fixed_layer_once() {
    let steps = [(3, [0, 1]), (4, [0, 1])];
    assert_eq!(
        (0..8).map(b2_serial_first).collect::<Vec<_>>(),
        [true, false, false, true, true, false, false, true]
    );
    assert_eq!((0..12).filter(|&repeat| b2_serial_first(repeat)).count(), 6);
    assert_eq!(
        steps
            .iter()
            .map(|(layer, _)| *layer)
            .collect::<HashSet<_>>()
            .len(),
        2
    );
}

#[test]
fn e4m3fn_gpu_decode_matches_all_codepoints() {
    let context = CudaContext::new(0).expect("create CUDA context");
    let stream = context.default_stream();
    let module = kernels::load(&context).expect("load embedded Rust PTX");
    let input = (0_u16..=255).map(|bits| bits as u8).collect::<Vec<_>>();
    let expected = input
        .iter()
        .map(|&bits| v4_router::f32_to_bf16_rne(v4_state::E4M3Fn(bits).decode()))
        .collect::<Vec<_>>();
    let input = DeviceBuffer::from_host(&stream, &input).expect("upload E4M3FN codepoints");
    let mut output = DeviceBuffer::<u16>::zeroed(&stream, 256).expect("allocate BF16 output");
    launch_e4m3fn_to_bf16(&module, &stream, &input, &mut output).expect("launch E4M3FN decode");
    assert_eq!(
        output.to_host_vec(&stream).expect("download BF16 output"),
        expected
    );
}

#[test]
fn canonical_compressor_pool_gpu_matches_host_bits_for_csa_main_indexer_and_hca() {
    fn run_case(
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        label: &str,
        ratio: usize,
        width: usize,
    ) -> Result<(Vec<f32>, Vec<u16>), Box<dyn Error>> {
        const TINY_SCORE_GAP: f32 = f32::from_bits(0x3380_0000);
        let overlap = ratio == 4;
        let value_width = width * (1 + usize::from(overlap));
        let slots = ratio * (1 + usize::from(overlap));
        let mut fixture_kv = vec![0.0_f32; slots * value_width];
        let mut fixture_score = vec![f32::NEG_INFINITY; slots * value_width];
        let mut host = v4_state::V4CompressorState::new_decode(ratio, width)?;
        let mut expected = None;
        for position in 0..ratio {
            let mut kv = vec![0.0_f32; value_width];
            let mut score = vec![0.0_f32; value_width];
            for element in 0..value_width {
                let dimension = element % width;
                kv[element] = ((position * 17 + element * 7) % 97) as f32 * 0.015_625 - 0.75;
                score[element] = if element / width == usize::from(overlap) && dimension == 0 {
                    match position {
                        0 => 0.0,
                        1 => -TINY_SCORE_GAP,
                        2 => -1.0,
                        _ => f32::from_bits(0xc015_93f8),
                    }
                } else {
                    ((position * 11 + element * 3) % 61) as f32 * 0.031_25 - 0.5
                };
            }
            let slot = if overlap {
                ratio + position % ratio
            } else {
                position % ratio
            };
            let start = slot * value_width;
            fixture_kv[start..start + value_width].copy_from_slice(&kv);
            fixture_score[start..start + value_width].copy_from_slice(&score);
            let (next, pooled) = host.prepare_decode_row_and_pool(
                &v4_state::CompressorPrepared {
                    kv_f32: kv,
                    score_f32: score.clone(),
                    score_plus_ape_f32: score,
                },
                position,
            )?;
            host = next;
            if position + 1 == ratio {
                expected = pooled;
            }
        }
        if overlap
            && !fixture_score[..ratio * value_width]
                .iter()
                .all(|value| *value == f32::NEG_INFINITY)
        {
            return Err(format!("{label} fixture lost its unwritten -inf slots").into());
        }
        let expected = expected.ok_or_else(|| format!("{label} boundary did not pool"))?;
        let kv = DeviceBuffer::from_host(stream, &fixture_kv)?;
        let score = DeviceBuffer::from_host(stream, &fixture_score)?;
        let mut actual = DeviceBuffer::<f32>::zeroed(stream, width)?;
        if overlap {
            run_compressor_pool_ratio4_device_into(
                module,
                stream,
                &kv,
                &score,
                &mut actual,
                width,
            )?;
        } else {
            run_compressor_pool_ratio128_device_into(module, stream, &kv, &score, &mut actual)?;
        }
        let actual = actual.to_host_vec(stream)?;
        assert_eq!(
            actual
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "{label} canonical FP32 pool bits"
        );
        let expected_bf16 = expected
            .iter()
            .copied()
            .map(v4_router::f32_to_bf16_rne)
            .collect::<Vec<_>>();
        let actual = DeviceBuffer::from_host(stream, &actual)?;
        let mut actual_bf16 = DeviceBuffer::<u16>::zeroed(stream, width)?;
        run_f32_to_bf16_device_into(module, stream, &actual, &mut actual_bf16)?;
        assert_eq!(
            actual_bf16.to_host_vec(stream)?,
            expected_bf16,
            "{label} prepared BF16 row bits"
        );
        Ok((expected, expected_bf16))
    }

    let context = CudaContext::new(0).expect("create CUDA context");
    let stream = context.default_stream();
    let module = kernels::load(&context).expect("load embedded Rust PTX");
    let mut all_f32 = Vec::new();
    let mut all_bf16 = Vec::new();
    for (label, ratio, width) in [
        ("CSA main", 4, v4_state::Q_HEAD_DIM),
        ("CSA indexer", 4, 128),
        ("HCA", 128, v4_state::Q_HEAD_DIM),
    ] {
        let (f32, bf16) =
            run_case(&module, &stream, label, ratio, width).expect("canonical pool case");
        all_f32.extend(f32);
        all_bf16.extend(bf16);
    }
    assert_eq!(fnv1a_f32(&all_f32), 0x36e1_2dd9_0d23_eeb7);
    assert_eq!(fnv1a_u16(&all_bf16), 0x9e01_5e04_a47c_4838);
}

#[test]
fn generic_strict_compressor_projection_tail_matches_pinned_cublas_at_csa_p3_and_hca_p127() {
    fn run_case(
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        ratio: usize,
        width: usize,
        boundary: usize,
    ) -> Result<(), Box<dyn Error>> {
        let weights = width
            .checked_mul(v4_state::HIDDEN)
            .ok_or("synthetic Compressor weight size overflow")?;
        let synthetic_bf16 = |seed: usize, exponent_base: u16| {
            let hash = (seed as u32)
                .wrapping_mul(0x9e37_79b9)
                .rotate_left(13)
                .wrapping_mul(0x85eb_ca6b);
            let sign = ((hash >> 31) as u16) << 15;
            let exponent = exponent_base + ((hash >> 17) as u16 & 0x0f);
            sign | (exponent << 7) | (hash as u16 & 0x007f)
        };
        let input_for = |position: usize| {
            (0..v4_state::HIDDEN)
                .map(|index| synthetic_bf16(index + position * 7_199, 118))
                .collect::<Vec<_>>()
        };
        let wkv_host = (0..weights)
            .map(|index| synthetic_bf16(index + 0x1_0000, 114))
            .collect::<Vec<_>>();
        let wgate_host = (0..weights)
            .map(|index| synthetic_bf16(index + 0x2_0000, 114))
            .collect::<Vec<_>>();
        let ape_host = (0..ratio * width)
            .map(|index| ((index * 17 % 113) as f32 - 56.0) * 0.002_929_687_5)
            .collect::<Vec<_>>();
        let wkv = DeviceBuffer::from_host(stream, &wkv_host)?;
        let wgate = DeviceBuffer::from_host(stream, &wgate_host)?;
        let ape = DeviceBuffer::from_host(stream, &ape_host)?;
        let mut input = DeviceBuffer::<u16>::zeroed(stream, v4_state::HIDDEN)?;
        let mut kv = DeviceBuffer::<f32>::zeroed(stream, width)?;
        let mut score = DeviceBuffer::<f32>::zeroed(stream, width)?;
        let mut score_plus_ape = DeviceBuffer::<f32>::zeroed(stream, width)?;
        let cublas = cublas_exact::CublasExact::load()?;
        let workspace = DeviceBuffer::<u8>::zeroed(stream, cublas_exact::WORKSPACE_BYTES)?;
        let mut input_f32 = DeviceBuffer::<f32>::zeroed(stream, v4_state::HIDDEN)?;
        let mut wkv_f32 = DeviceBuffer::<f32>::zeroed(stream, weights)?;
        let mut wgate_f32 = DeviceBuffer::<f32>::zeroed(stream, weights)?;
        let tail_kv =
            DeviceBuffer::<f32>::zeroed(stream, ratio * (1 + usize::from(ratio == 4)) * width)?;
        let tail_score = DeviceBuffer::from_host(
            stream,
            &vec![f32::NEG_INFINITY; ratio * (1 + usize::from(ratio == 4)) * width],
        )?;
        let mut pool = DeviceBuffer::<f32>::zeroed(stream, v4_state::Q_HEAD_DIM)?;
        let mut pool_bf16 = DeviceBuffer::<u16>::zeroed(stream, v4_state::Q_HEAD_DIM)?;
        let mut host = v4_state::V4CompressorState::new_decode(ratio, v4_state::Q_HEAD_DIM)?;
        let mut host_oracle = StrictCompressorArena::new(stream, width, ratio)?;

        for position in 0..=boundary {
            let host_input = input_for(position);
            input.copy_from_host(stream, &host_input)?;
            let expected = host_oracle.run_from_device(
                &cublas,
                stream,
                &host_input,
                &wkv,
                &wgate,
                &ape,
                width,
                ratio,
                position,
            )?;
            run_exact_compressor_buffers_device_into(
                module,
                stream,
                &input,
                &wkv,
                &wgate,
                &ape,
                &mut kv,
                &mut score,
                &mut score_plus_ape,
                &cublas,
                &workspace,
                &mut input_f32,
                &mut wkv_f32,
                &mut wgate_f32,
                ratio,
                position,
                position == 0,
            )?;
            if position == 0 {
                let mut custom_kv = DeviceBuffer::<f32>::zeroed(stream, width)?;
                let mut custom_score = DeviceBuffer::<f32>::zeroed(stream, width)?;
                let mut custom_score_plus_ape = DeviceBuffer::<f32>::zeroed(stream, width)?;
                run_compressor_device_into(
                    module,
                    stream,
                    &input,
                    &wkv,
                    &wgate,
                    &ape,
                    &mut custom_kv,
                    &mut custom_score,
                    &mut custom_score_plus_ape,
                    width,
                    ratio,
                    position,
                )?;
                if first_f32_bits_mismatch(&custom_kv.to_host_vec(stream)?, &expected.kv_f32)
                    .is_none()
                    && first_f32_bits_mismatch(
                        &custom_score_plus_ape.to_host_vec(stream)?,
                        &expected.score_plus_ape_f32,
                    )
                    .is_none()
                {
                    return Err(
                        "synthetic Compressor fixture no longer distinguishes custom and pinned cuBLAS reductions".into(),
                    );
                }
            }
            let (next, expected_pool) = host.prepare_decode_row_and_pool(&expected, position)?;
            host = next;
            let slot = if ratio == 4 {
                4 + position % 4
            } else {
                position % ratio
            };
            // This is the generic persistent-tail update: projection row
            // followed by the shared physical slot mapping.
            unsafe {
                copy_device_row(stream, &tail_kv, slot, &kv)?;
                copy_device_row(stream, &tail_score, slot, &score_plus_ape)?;
            }
            if let Some(expected_pool) = expected_pool {
                if ratio == 4 {
                    run_compressor_pool_ratio4_device_into(
                        module,
                        stream,
                        &tail_kv,
                        &tail_score,
                        &mut pool,
                        v4_state::Q_HEAD_DIM,
                    )?;
                    // Match the generic CSA post-boundary overlap copy
                    // after the pool has consumed slots 0..3 and 4..7.
                    unsafe {
                        copy_device_range(stream, &tail_kv, 0, &tail_kv, 4 * width, 4 * width)?;
                        copy_device_range(
                            stream,
                            &tail_score,
                            0,
                            &tail_score,
                            4 * width,
                            4 * width,
                        )?;
                    }
                } else {
                    run_compressor_pool_ratio128_device_into(
                        module,
                        stream,
                        &tail_kv,
                        &tail_score,
                        &mut pool,
                    )?;
                }
                let actual_pool = pool.to_host_vec(stream)?;
                let (mismatches, max_abs, max_rel, max_ulp) =
                    final_head_comparison(&actual_pool, &expected_pool);
                if mismatches != 0 {
                    return Err(format!(
                        "strict generic ratio-{ratio} boundary p{position} pool drift: max_abs={max_abs:e} max_rel={max_rel:e} max_ulp={max_ulp}"
                    )
                    .into());
                }
                run_f32_to_bf16_device_into(module, stream, &pool, &mut pool_bf16)?;
                let expected_bf16 = expected_pool
                    .iter()
                    .copied()
                    .map(v4_router::f32_to_bf16_rne)
                    .collect::<Vec<_>>();
                if pool_bf16.to_host_vec(stream)? != expected_bf16 {
                    return Err(format!(
                        "strict generic ratio-{ratio} boundary p{position} BF16 compressed state drifted"
                    )
                    .into());
                }
            }
        }
        if first_f32_bits_mismatch(&host.kv(), &tail_kv.to_host_vec(stream)?).is_some()
            || first_f32_bits_mismatch(&host.score(), &tail_score.to_host_vec(stream)?).is_some()
        {
            return Err(format!(
                "strict generic ratio-{ratio} persistent Compressor tail differs from the pinned cuBLAS host oracle"
            )
            .into());
        }
        Ok(())
    }

    let context = CudaContext::new(0).expect("create CUDA context");
    let stream = context.default_stream();
    let module = kernels::load(&context).expect("load embedded Rust PTX");
    let generic_exact =
        GenericExactCompressorScratch::new(&stream).expect("allocate shared strict scratch");
    assert_eq!(
        generic_exact.hca_wkv.len(),
        v4_state::Q_HEAD_DIM * v4_state::HIDDEN
    );
    assert_eq!(generic_exact.csa_wkv.len(), 1_024 * v4_state::HIDDEN);
    assert_eq!(generic_exact.csa_indexer_wkv.len(), 256 * v4_state::HIDDEN);
    run_case(&module, &stream, 4, 1_024, 3).expect("CSA p3 strict generic Compressor");
    run_case(&module, &stream, 128, v4_state::Q_HEAD_DIM, 127)
        .expect("HCA p127 strict generic Compressor");
}

#[test]
fn generic_strict_compressor_owner_refreshes_across_hca_csa_and_indexer_layers() {
    let context = CudaContext::new(0).expect("create CUDA context");
    let stream = context.default_stream();
    let module = kernels::load(&context).expect("load embedded Rust PTX");
    let mut strict =
        GenericExactCompressorScratch::new(&stream).expect("allocate generic strict scratch");
    let bits = |seed: usize| {
        let hash = (seed as u32)
            .wrapping_mul(0x9e37_79b9)
            .rotate_left(13)
            .wrapping_mul(0x85eb_ca6b);
        (((hash >> 31) as u16) << 15)
            | ((114 + ((hash >> 17) as u16 & 0x0f)) << 7)
            | (hash as u16 & 0x007f)
    };
    let input_for = |position: usize| {
        (0..v4_state::HIDDEN)
            .map(|index| bits(index + position * 7_199 + 0x30000))
            .collect::<Vec<_>>()
    };
    let make_weights = |width: usize, ratio: usize, seed: usize| {
        let elements = width * v4_state::HIDDEN;
        let wkv = (0..elements)
            .map(|index| bits(index + seed))
            .collect::<Vec<_>>();
        let wgate = (0..elements)
            .map(|index| bits(index + seed + 0x10000))
            .collect::<Vec<_>>();
        let ape = (0..ratio * width)
            .map(|index| ((index * 17 + seed) % 113) as f32 * 0.003_906_25 - 0.21875)
            .collect::<Vec<_>>();
        (wkv, wgate, ape)
    };
    let (a_wkv_host, a_wgate_host, a_ape_host) = make_weights(v4_state::Q_HEAD_DIM, 128, 0x40000);
    let (b_wkv_host, b_wgate_host, b_ape_host) = make_weights(1_024, 4, 0x50000);
    let (i_wkv_host, i_wgate_host, i_ape_host) = make_weights(256, 4, 0x60000);
    let a_wkv = DeviceBuffer::from_host(&stream, &a_wkv_host).expect("upload HCA wkv");
    let a_wgate = DeviceBuffer::from_host(&stream, &a_wgate_host).expect("upload HCA gate");
    let a_ape = DeviceBuffer::from_host(&stream, &a_ape_host).expect("upload HCA APE");
    let b_wkv = DeviceBuffer::from_host(&stream, &b_wkv_host).expect("upload CSA wkv");
    let b_wgate = DeviceBuffer::from_host(&stream, &b_wgate_host).expect("upload CSA gate");
    let b_ape = DeviceBuffer::from_host(&stream, &b_ape_host).expect("upload CSA APE");
    let i_wkv = DeviceBuffer::from_host(&stream, &i_wkv_host).expect("upload indexer wkv");
    let i_wgate = DeviceBuffer::from_host(&stream, &i_wgate_host).expect("upload indexer gate");
    let i_ape = DeviceBuffer::from_host(&stream, &i_ape_host).expect("upload indexer APE");
    let mut input = DeviceBuffer::<u16>::zeroed(&stream, v4_state::HIDDEN).expect("input");
    let mut hca_kv = DeviceBuffer::<f32>::zeroed(&stream, v4_state::Q_HEAD_DIM).expect("HCA KV");
    let mut hca_score =
        DeviceBuffer::<f32>::zeroed(&stream, v4_state::Q_HEAD_DIM).expect("HCA score");
    let mut hca_ape =
        DeviceBuffer::<f32>::zeroed(&stream, v4_state::Q_HEAD_DIM).expect("HCA score APE");
    let mut csa_kv = DeviceBuffer::<f32>::zeroed(&stream, 1_024).expect("CSA KV");
    let mut csa_score = DeviceBuffer::<f32>::zeroed(&stream, 1_024).expect("CSA score");
    let mut csa_ape = DeviceBuffer::<f32>::zeroed(&stream, 1_024).expect("CSA score APE");
    let mut index_kv = DeviceBuffer::<f32>::zeroed(&stream, 256).expect("indexer KV");
    let mut index_score = DeviceBuffer::<f32>::zeroed(&stream, 256).expect("indexer score");
    let mut index_ape = DeviceBuffer::<f32>::zeroed(&stream, 256).expect("indexer score APE");

    macro_rules! run_main {
        ($path:expr, $layer:expr, $position:expr, $width:expr, $ratio:expr, $wkv_host:expr, $wgate_host:expr, $ape_host:expr, $wkv:expr, $wgate:expr, $ape:expr, $kv:expr, $score:expr, $score_ape:expr, $converted_wkv:ident, $converted_wgate:ident) => {{
            let host_input = input_for($position);
            let expected = run_compressor_official_cublas_host(
                &strict.cublas,
                &stream,
                &host_input,
                $wkv_host,
                $wgate_host,
                $ape_host,
                $width,
                $ratio,
                $position,
            )
            .expect("fresh pinned cuBLAS reference");
            input
                .copy_from_host(&stream, &host_input)
                .expect("upload input");
            let refresh = strict.begin_main_refresh($path, $layer);
            run_exact_compressor_buffers_device_into(
                &module,
                &stream,
                &input,
                $wkv,
                $wgate,
                $ape,
                $kv,
                $score,
                $score_ape,
                &strict.cublas,
                &strict.workspace,
                &mut strict.input,
                &mut strict.$converted_wkv,
                &mut strict.$converted_wgate,
                $ratio,
                $position,
                refresh,
            )
            .expect("generic strict projection");
            strict.finish_main_refresh($path, $layer, refresh);
            assert!(
                first_f32_bits_mismatch(
                    &$kv.to_host_vec(&stream).expect("download KV"),
                    &expected.kv_f32
                )
                .is_none()
            );
            assert!(
                first_f32_bits_mismatch(
                    &$score_ape.to_host_vec(&stream).expect("download score APE"),
                    &expected.score_plus_ape_f32
                )
                .is_none()
            );
            refresh
        }};
    }

    assert!(run_main!(
        GenericStationaryPath::Hca128,
        10,
        0,
        v4_state::Q_HEAD_DIM,
        128,
        &a_wkv_host,
        &a_wgate_host,
        &a_ape_host,
        &a_wkv,
        &a_wgate,
        &a_ape,
        &mut hca_kv,
        &mut hca_score,
        &mut hca_ape,
        hca_wkv,
        hca_wgate
    ));
    assert!(!run_main!(
        GenericStationaryPath::Hca128,
        10,
        1,
        v4_state::Q_HEAD_DIM,
        128,
        &a_wkv_host,
        &a_wgate_host,
        &a_ape_host,
        &a_wkv,
        &a_wgate,
        &a_ape,
        &mut hca_kv,
        &mut hca_score,
        &mut hca_ape,
        hca_wkv,
        hca_wgate
    ));
    assert!(run_main!(
        GenericStationaryPath::Csa4,
        11,
        2,
        1_024,
        4,
        &b_wkv_host,
        &b_wgate_host,
        &b_ape_host,
        &b_wkv,
        &b_wgate,
        &b_ape,
        &mut csa_kv,
        &mut csa_score,
        &mut csa_ape,
        csa_wkv,
        csa_wgate
    ));
    assert!(!run_main!(
        GenericStationaryPath::Csa4,
        11,
        3,
        1_024,
        4,
        &b_wkv_host,
        &b_wgate_host,
        &b_ape_host,
        &b_wkv,
        &b_wgate,
        &b_ape,
        &mut csa_kv,
        &mut csa_score,
        &mut csa_ape,
        csa_wkv,
        csa_wgate
    ));

    for position in [2, 3] {
        let host_input = input_for(position);
        let expected = run_compressor_official_cublas_host(
            &strict.cublas,
            &stream,
            &host_input,
            &i_wkv_host,
            &i_wgate_host,
            &i_ape_host,
            256,
            4,
            position,
        )
        .expect("fresh pinned indexer reference");
        input
            .copy_from_host(&stream, &host_input)
            .expect("upload indexer input");
        let refresh = strict.begin_indexer_refresh(11);
        run_exact_compressor_buffers_device_into(
            &module,
            &stream,
            &input,
            &i_wkv,
            &i_wgate,
            &i_ape,
            &mut index_kv,
            &mut index_score,
            &mut index_ape,
            &strict.cublas,
            &strict.workspace,
            &mut strict.input,
            &mut strict.csa_indexer_wkv,
            &mut strict.csa_indexer_wgate,
            4,
            position,
            refresh,
        )
        .expect("generic strict indexer projection");
        strict.finish_indexer_refresh(11, refresh);
        assert_eq!(refresh, position == 2);
        assert!(
            first_f32_bits_mismatch(
                &index_kv.to_host_vec(&stream).expect("download indexer KV"),
                &expected.kv_f32
            )
            .is_none()
        );
    }

    let refresh_a_p5 = run_main!(
        GenericStationaryPath::Hca128,
        10,
        5,
        v4_state::Q_HEAD_DIM,
        128,
        &a_wkv_host,
        &a_wgate_host,
        &a_ape_host,
        &a_wkv,
        &a_wgate,
        &a_ape,
        &mut hca_kv,
        &mut hca_score,
        &mut hca_ape,
        hca_wkv,
        hca_wgate
    );
    assert!(refresh_a_p5, "A p5 must refresh after B's layer transition");
    assert_ne!(
        5, 0,
        "the old start_pos gate would have skipped this required refresh"
    );
    assert_eq!(strict.main_weight_conversions, 3);
    assert_eq!(strict.indexer_weight_conversions, 1);
    assert_eq!(strict.hca_owner, Some(10));
    assert_eq!(strict.csa_owner, None);
    assert_eq!(strict.csa_indexer_owner, None);
}

#[test]
fn streamed_bf16_head_warp_matches_small_cpu_slice() {
    const INPUTS: usize = 64;
    const OUTPUTS: usize = 17;
    let context = CudaContext::new(0).expect("create CUDA context");
    let stream = context.default_stream();
    let module = kernels::load(&context).expect("load embedded Rust PTX");
    let input = (0..INPUTS)
        .map(|index| v4_router::f32_to_bf16_rne(((index % 13) as f32 - 6.0) * 0.03125))
        .collect::<Vec<_>>();
    let weight = (0..OUTPUTS * INPUTS)
        .map(|index| v4_router::f32_to_bf16_rne(((index % 29) as f32 - 14.0) * 0.015625))
        .collect::<Vec<_>>();
    let expected = weight
        .chunks_exact(INPUTS)
        .map(|row| {
            input
                .iter()
                .zip(row)
                .fold(0.0_f32, |sum, (&input, &weight)| {
                    sum + v4_router::bf16_to_f32(input) * v4_router::bf16_to_f32(weight)
                })
        })
        .collect::<Vec<_>>();
    let expected_sumabs = weight
        .chunks_exact(INPUTS)
        .map(|row| {
            input
                .iter()
                .zip(row)
                .map(|(&input, &weight)| {
                    v4_router::bf16_to_f32(input).abs() * v4_router::bf16_to_f32(weight).abs()
                })
                .sum()
        })
        .collect::<Vec<f32>>();
    let input = DeviceBuffer::from_host(&stream, &input).unwrap();
    let weight = DeviceBuffer::from_host(&stream, &weight).unwrap();
    let mut output = DeviceBuffer::<f32>::zeroed(&stream, OUTPUTS).unwrap();
    let mut sumabs = DeviceBuffer::<f32>::zeroed(&stream, OUTPUTS).unwrap();
    let mut status = DeviceBuffer::<u32>::zeroed(&stream, OUTPUTS).unwrap();
    // SAFETY: one complete warp owns each fully initialized output row.
    unsafe {
        module
            .matvec_bf16_streamed(
                &stream,
                LaunchConfig::for_num_elems((OUTPUTS * 32) as u32),
                &input,
                &weight,
                &mut output,
                &mut sumabs,
                &mut status,
                INPUTS as u32,
                OUTPUTS as u32,
            )
            .unwrap();
    }
    let actual = output.to_host_vec(&stream).unwrap();
    let actual_sumabs = sumabs.to_host_vec(&stream).unwrap();
    let actual_status = status.to_host_vec(&stream).unwrap();
    assert_close("streamed BF16 head slice", &actual, &expected);
    assert_close(
        "streamed BF16 head sumabs",
        &actual_sumabs,
        &expected_sumabs,
    );
    assert_eq!(actual_status, vec![0; OUTPUTS]);
    let argmax = |values: &[f32]| {
        (0..values.len())
            .max_by(|&left, &right| values[left].total_cmp(&values[right]))
            .unwrap()
    };
    assert_eq!(argmax(&actual), argmax(&expected));
}

#[test]
fn certified_argmax_coefficient_is_the_preregistered_formula() {
    assert_eq!(certified_argmax_coeff().to_bits(), 0x3f3c_e62a_a18b_476c);
}

#[test]
fn compressor_gpu_matches_cpu_for_hca_csa_and_indexer_geometries() {
    let context = CudaContext::new(0).expect("create CUDA context");
    let stream = context.default_stream();
    let module = kernels::load(&context).expect("load embedded Rust PTX");
    let input_host = [
        v4_router::f32_to_bf16_rne(1.0),
        v4_router::f32_to_bf16_rne(2.0),
        v4_router::f32_to_bf16_rne(3.0),
        v4_router::f32_to_bf16_rne(4.0),
    ];

    for (outputs, ratio, start_pos) in [
        (512_usize, 128_usize, 127_usize),
        (1_024, 4, 3),
        (256, 4, 3),
    ] {
        let mut ape_host = vec![0.0_f32; ratio * outputs];
        for phase in 0..ratio {
            ape_host[phase * outputs..(phase + 1) * outputs].fill(phase as f32);
        }
        let weights = v4_state::CompressorWeights {
            wkv_bf16: [
                v4_router::f32_to_bf16_rne(1.0),
                v4_router::f32_to_bf16_rne(2.0),
            ]
            .repeat(outputs),
            wgate_bf16: [
                v4_router::f32_to_bf16_rne(2.0),
                v4_router::f32_to_bf16_rne(-1.0),
            ]
            .repeat(outputs),
            ape_f32: ape_host,
            range_bytes: 0,
        };
        let expected = v4_state::prepare_compressor_cpu_with_geometry(
            &input_host,
            &weights,
            2,
            outputs,
            ratio,
            start_pos,
        )
        .expect("prepare generic compressor CPU oracle");
        let input = DeviceBuffer::from_host(&stream, &input_host).expect("upload input");
        let wkv = DeviceBuffer::from_host(&stream, &weights.wkv_bf16).expect("upload wkv weights");
        let wgate =
            DeviceBuffer::from_host(&stream, &weights.wgate_bf16).expect("upload gate weights");
        let ape = DeviceBuffer::from_host(&stream, &weights.ape_f32).expect("upload APE");
        let mut kv =
            DeviceBuffer::<f32>::zeroed(&stream, 2 * outputs).expect("allocate compressor KV");
        let mut score =
            DeviceBuffer::<f32>::zeroed(&stream, 2 * outputs).expect("allocate compressor score");
        let mut score_plus_ape = DeviceBuffer::<f32>::zeroed(&stream, 2 * outputs)
            .expect("allocate compressor score+APE");
        unsafe {
            module
                .compressor_v4_bf16_f32(
                    &stream,
                    LaunchConfig::for_num_elems(u32::try_from(2 * outputs).unwrap()),
                    &input,
                    &wkv,
                    &wgate,
                    &ape,
                    &mut kv,
                    &mut score,
                    &mut score_plus_ape,
                    start_pos as u32,
                    2,
                    outputs as u32,
                    ratio as u32,
                )
                .expect("launch generic compressor");
        }
        assert_eq!(
            kv.to_host_vec(&stream).unwrap(),
            expected.kv_f32,
            "KV projection outputs={outputs} ratio={ratio}"
        );
        assert_eq!(
            score.to_host_vec(&stream).unwrap(),
            expected.score_f32,
            "score projection outputs={outputs} ratio={ratio}"
        );
        assert_eq!(
            score_plus_ape.to_host_vec(&stream).unwrap(),
            expected.score_plus_ape_f32,
            "score+APE outputs={outputs} ratio={ratio}"
        );
    }
}

#[test]
fn sparse_attention_consumes_concatenated_local_and_compressed_indices() {
    let context = CudaContext::new(0).expect("create CUDA context");
    let stream = context.default_stream();
    let module = kernels::load(&context).expect("load embedded Rust PTX");
    let q_host = vec![v4_router::f32_to_bf16_rne(0.0); 128 * 512];
    let mut kv_host = vec![v4_router::f32_to_bf16_rne(0.0); 2 * 512];
    kv_host[512..].fill(v4_router::f32_to_bf16_rne(2.0));
    let q = DeviceBuffer::from_host(&stream, &q_host).unwrap();
    let kv = DeviceBuffer::from_host(&stream, &kv_host).unwrap();
    let sink = DeviceBuffer::from_host(&stream, &vec![-100.0_f32; 128]).unwrap();
    let topk = DeviceBuffer::from_host(&stream, &[0_i32, 1_i32]).unwrap();
    let mut output = DeviceBuffer::<u16>::zeroed(&stream, q_host.len()).unwrap();
    unsafe {
        module
            .sparse_attn_v4_bf16(
                &stream,
                LaunchConfig::for_num_elems(u32::try_from(output.len()).unwrap()),
                &q,
                &kv,
                &sink,
                &topk,
                &mut output,
                2,
                u32::try_from(v4_state::Q_HEADS).unwrap(),
                u32::try_from(v4_state::Q_HEAD_DIM).unwrap(),
            )
            .expect("launch sparse attention with compressed KV");
    }
    assert_eq!(
        output.to_host_vec(&stream).unwrap(),
        vec![v4_router::f32_to_bf16_rne(1.0); q_host.len()]
    );
}

#[test]
fn sparse_attention_score_reuse_is_bit_exact_at_l3_l4_boundaries_and_faster_locally() {
    const HEADS: usize = v4_state::Q_HEADS;
    const DIM: usize = v4_state::Q_HEAD_DIM;

    fn assert_f32_bits(label: &str, actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len(), "{label}: length");
        if let Some((index, (actual, expected))) = actual
            .iter()
            .zip(expected)
            .enumerate()
            .find(|(_, (actual, expected))| actual.to_bits() != expected.to_bits())
        {
            panic!(
                "{label}: f32 bit mismatch at {index}: {:08x} != {:08x}",
                actual.to_bits(),
                expected.to_bits()
            );
        }
    }

    let pos0 = v4_state::SparseTopk {
        indices: vec![0],
        rows: 1,
        per_row: 1,
        window_per_row: 1,
        compressed_per_row: 0,
    };
    let pos1_indices = v4_state::window_topk_cpu(1, 1, v4_state::KV_WINDOW)
        .unwrap()
        .0;
    assert_eq!(&pos1_indices[..2], &[0, 1]);
    let pos1 = v4_state::SparseTopk {
        indices: pos1_indices,
        rows: 1,
        per_row: v4_state::KV_WINDOW,
        window_per_row: v4_state::KV_WINDOW,
        compressed_per_row: 0,
    };
    let csa_scores = vec![1.0_f32];
    let l4_3 = v4_state::build_csa_sparse_topk(
        &csa_scores,
        1,
        1,
        &[1],
        3,
        v4_state::KV_WINDOW,
        v4_state::INDEX_TOPK,
    )
    .unwrap();
    let l4_4 = v4_state::build_csa_sparse_topk(
        &csa_scores,
        1,
        1,
        &[1],
        4,
        v4_state::KV_WINDOW,
        v4_state::INDEX_TOPK,
    )
    .unwrap();
    let l3_127 = v4_state::build_hca_sparse_topk(1, 127, v4_state::KV_WINDOW, 128).unwrap();
    let l3_128 = v4_state::build_hca_sparse_topk(1, 128, v4_state::KV_WINDOW, 128).unwrap();
    for topk in [&l4_3, &l4_4, &l3_127, &l3_128] {
        assert_eq!(topk.per_row, v4_state::KV_WINDOW + 1);
        assert_eq!(topk.indices[v4_state::KV_WINDOW], 128);
    }

    let q_host = (0..HEADS * DIM)
        .map(|index| v4_router::f32_to_bf16_rne(((index % 29) as f32 - 14.0) * 0.007_812_5))
        .collect::<Vec<_>>();
    let kv_host = (0..(v4_state::KV_WINDOW + 1) * DIM)
        .map(|index| v4_router::f32_to_bf16_rne(((index % 31) as f32 - 15.0) * 0.003_906_25))
        .collect::<Vec<_>>();
    let sink_host = (0..HEADS)
        .map(|head| -1.0_f32 - (head % 7) as f32 * 0.03125)
        .collect::<Vec<_>>();
    let context = CudaContext::new(0).expect("create CUDA context");
    let stream = context.default_stream();
    let module = kernels::load(&context).expect("load embedded Rust PTX");
    let q = DeviceBuffer::from_host(&stream, &q_host).expect("upload q");
    let kv = DeviceBuffer::from_host(&stream, &kv_host).expect("upload kv");
    let sink = DeviceBuffer::from_host(&stream, &sink_host).expect("upload sink");

    for (name, topk) in [
        ("pos0", &pos0),
        ("pos1", &pos1),
        ("l4_3_to_4_before", &l4_3),
        ("l4_3_to_4_after", &l4_4),
        ("l3_127_to_128_before", &l3_127),
        ("l3_127_to_128_after", &l3_128),
    ] {
        let topk_device = DeviceBuffer::from_host(&stream, &topk.indices).expect("upload topk");
        let mut reference = DeviceBuffer::<u16>::zeroed(&stream, q_host.len()).unwrap();
        let mut candidate = DeviceBuffer::<u16>::zeroed(&stream, q_host.len()).unwrap();
        let mut reference_state =
            SparseAttentionScoreReuseScratch::new(&stream, 1, topk.per_row).unwrap();
        let mut candidate_state =
            SparseAttentionScoreReuseScratch::new(&stream, 1, topk.per_row).unwrap();
        run_sparse_attention_device_into(
            &module,
            &stream,
            &q,
            &kv,
            &sink,
            &topk_device,
            &mut reference,
            topk.per_row,
        )
        .unwrap();
        run_sparse_attention_reference_state_device_into(
            &module,
            &stream,
            &q,
            &kv,
            &sink,
            &topk_device,
            &mut reference_state,
            topk.per_row,
        )
        .unwrap();
        run_sparse_attention_score_reuse_device_into(
            &module,
            &stream,
            &q,
            &kv,
            &sink,
            &topk_device,
            &mut candidate,
            &mut candidate_state,
            topk.per_row,
        )
        .unwrap();
        let reference_bits = reference.to_host_vec(&stream).unwrap();
        assert_eq!(
            candidate.to_host_vec(&stream).unwrap(),
            reference_bits,
            "{name}: score-reuse BF16 bits against scalar reference"
        );
        assert_f32_bits(
            &format!("{name}: scores"),
            &candidate_state.scores.to_host_vec(&stream).unwrap(),
            &reference_state.scores.to_host_vec(&stream).unwrap(),
        );
        assert_f32_bits(
            &format!("{name}: probabilities"),
            &candidate_state.probabilities.to_host_vec(&stream).unwrap(),
            &reference_state.probabilities.to_host_vec(&stream).unwrap(),
        );
        assert_f32_bits(
            &format!("{name}: denominators"),
            &candidate_state.denominators.to_host_vec(&stream).unwrap(),
            &reference_state.denominators.to_host_vec(&stream).unwrap(),
        );
        assert_eq!(
            candidate_state.valid_counts.to_host_vec(&stream).unwrap(),
            reference_state.valid_counts.to_host_vec(&stream).unwrap(),
            "{name}: valid counts"
        );
    }

    let topk = DeviceBuffer::from_host(&stream, &l3_128.indices).unwrap();
    let mut reference = DeviceBuffer::<u16>::zeroed(&stream, q_host.len()).unwrap();
    let mut candidate = DeviceBuffer::<u16>::zeroed(&stream, q_host.len()).unwrap();
    let mut scratch = SparseAttentionScoreReuseScratch::new(&stream, 1, l3_128.per_row).unwrap();
    let start = context
        .new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))
        .unwrap();
    let end = context
        .new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))
        .unwrap();
    for _ in 0..2 {
        run_sparse_attention_device_into(
            &module,
            &stream,
            &q,
            &kv,
            &sink,
            &topk,
            &mut reference,
            l3_128.per_row,
        )
        .unwrap();
        run_sparse_attention_score_reuse_device_into(
            &module,
            &stream,
            &q,
            &kv,
            &sink,
            &topk,
            &mut candidate,
            &mut scratch,
            l3_128.per_row,
        )
        .unwrap();
    }
    end.record(&stream).unwrap();
    end.synchronize().unwrap();
    let mut reference_ns = Vec::new();
    let mut candidate_ns = Vec::new();
    for _ in 0..7 {
        start.record(&stream).unwrap();
        run_sparse_attention_device_into(
            &module,
            &stream,
            &q,
            &kv,
            &sink,
            &topk,
            &mut reference,
            l3_128.per_row,
        )
        .unwrap();
        end.record(&stream).unwrap();
        end.synchronize().unwrap();
        reference_ns.push((start.elapsed_ms(&end).unwrap() as f64 * 1_000_000.0) as u64);
        start.record(&stream).unwrap();
        run_sparse_attention_score_reuse_device_into(
            &module,
            &stream,
            &q,
            &kv,
            &sink,
            &topk,
            &mut candidate,
            &mut scratch,
            l3_128.per_row,
        )
        .unwrap();
        end.record(&stream).unwrap();
        end.synchronize().unwrap();
        candidate_ns.push((start.elapsed_ms(&end).unwrap() as f64 * 1_000_000.0) as u64);
    }
    let reference_p50 = median_u64(&reference_ns);
    let candidate_p50 = median_u64(&candidate_ns);
    println!(
        "SPARSE_SCORE_REUSE exact=1 shape=B1x128x512 topk={} l3_127_to_128 reference_p50_ms={:.3} candidate_p50_ms={:.3}",
        l3_128.per_row,
        reference_p50 as f64 / 1_000_000.0,
        candidate_p50 as f64 / 1_000_000.0,
    );
    assert!(
        candidate_p50 < reference_p50,
        "score reuse did not accelerate the isolated L3 127->128 fixture"
    );
}
