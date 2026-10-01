use super::*;

#[test]
fn csa_indexer_scores_match_cpu_oracle_and_select_causal_topk() {
    let (rows, heads, head_dim, keys, key_capacity) = (2_usize, 3_usize, 4_usize, 5_usize, 7_usize);
    let q = (0..rows * heads * head_dim)
        .map(|index| (index as f32 - 7.0) / 8.0)
        .collect::<Vec<_>>();
    let mut kv = (0..keys * head_dim)
        .map(|index| ((index * 3 % 17) as f32 - 8.0) / 7.0)
        .collect::<Vec<_>>();
    let active_kv = kv.clone();
    kv.resize(key_capacity * head_dim, 1234.0);
    let weights = vec![0.5, -0.25, 1.0, -0.75, 0.625, 0.375];
    let expected =
        v4_state::indexer_scores_cpu(&q, &active_kv, &weights, rows, heads, head_dim, keys)
            .expect("CPU CSA indexer score oracle");

    let context = CudaContext::new(0).expect("create CUDA context");
    let stream = context.default_stream();
    let module = kernels::load(&context).expect("load embedded Rust PTX");
    let q_device = DeviceBuffer::from_host(&stream, &q).expect("upload q");
    let kv_device = DeviceBuffer::from_host(&stream, &kv).expect("upload kv");
    let weights_device = DeviceBuffer::from_host(&stream, &weights).expect("upload weights");
    let untouched = -12345.0_f32;
    let mut scores = DeviceBuffer::from_host(&stream, &vec![untouched; rows * key_capacity])
        .expect("allocate scores");
    unsafe {
        module
            .indexer_scores_v4_f32(
                &stream,
                LaunchConfig::for_num_elems(u32::try_from(rows * keys).unwrap()),
                &q_device,
                &kv_device,
                &weights_device,
                &mut scores,
                rows as u32,
                heads as u32,
                head_dim as u32,
                keys as u32,
            )
            .expect("launch CSA indexer scores");
    }
    let actual = scores.to_host_vec(&stream).expect("download scores");
    assert_close("CSA indexer scores", &actual[..rows * keys], &expected);
    assert!(
        actual[rows * keys..]
            .iter()
            .all(|&value| value == untouched)
    );

    let undersized_kv =
        DeviceBuffer::from_host(&stream, &active_kv[..active_kv.len() - 1]).unwrap();
    let mut fail_closed =
        DeviceBuffer::from_host(&stream, &vec![untouched; rows * key_capacity]).unwrap();
    unsafe {
        module
            .indexer_scores_v4_f32(
                &stream,
                LaunchConfig::for_num_elems(u32::try_from(rows * keys).unwrap()),
                &q_device,
                &undersized_kv,
                &weights_device,
                &mut fail_closed,
                rows as u32,
                heads as u32,
                head_dim as u32,
                keys as u32,
            )
            .expect("launch undersized CSA indexer scores");
    }
    assert!(
        fail_closed
            .to_host_vec(&stream)
            .unwrap()
            .iter()
            .all(|&value| value == untouched)
    );

    let expected_topk = v4_state::indexer_topk_cpu(&expected, rows, keys, 2, &[3, 5], 128).unwrap();
    let actual_topk =
        v4_state::indexer_topk_cpu(&actual[..rows * keys], rows, keys, 2, &[3, 5], 128).unwrap();
    assert_eq!(actual_topk, expected_topk);
    let valid = DeviceBuffer::from_host(&stream, &[3_u32, 5_u32]).unwrap();
    let mut gpu_topk = DeviceBuffer::<i32>::zeroed(&stream, rows * 2).unwrap();
    unsafe {
        module
            .indexer_topk_v4_f32(
                &stream,
                LaunchConfig::for_num_elems(u32::try_from(rows * 2).unwrap()),
                &scores,
                &valid,
                &mut gpu_topk,
                rows as u32,
                keys as u32,
                2,
                128,
            )
            .expect("launch deterministic CSA top-k");
    }
    assert_eq!(gpu_topk.to_host_vec(&stream).unwrap(), expected_topk);
}

#[test]
fn csa_indexer_bf16_weight_projection_matches_cpu_exactly() {
    let input_f32 = [1.0_f32, -2.0, 0.5, 3.0, -1.0, 0.25, 2.0, -0.5];
    let weight_f32 = [
        0.5_f32, 1.0, -1.0, 0.25, -0.5, 0.75, 1.5, -2.0, 2.0, -1.0, 0.5, 1.0,
    ];
    let input = input_f32
        .into_iter()
        .map(v4_router::f32_to_bf16_rne)
        .collect::<Vec<_>>();
    let weight = weight_f32
        .into_iter()
        .map(v4_router::f32_to_bf16_rne)
        .collect::<Vec<_>>();
    let expected = input
        .chunks_exact(4)
        .flat_map(|row| {
            weight.chunks_exact(4).map(move |column| {
                row.iter()
                    .zip(column)
                    .map(|(&left, &right)| {
                        v4_router::bf16_to_f32(left) * v4_router::bf16_to_f32(right)
                    })
                    .sum::<f32>()
            })
        })
        .collect::<Vec<_>>();
    let context = CudaContext::new(0).expect("create CUDA context");
    let stream = context.default_stream();
    let module = kernels::load(&context).expect("load embedded Rust PTX");
    let weight = DeviceBuffer::from_host(&stream, &weight).unwrap();
    let actual = run_bf16_linear_f32(&module, &stream, &input, &weight, 4, 3).unwrap();
    assert_eq!(
        actual
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        expected
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
}

#[test]
#[ignore = "requires COLIBRI_DSV4_MODEL_DIR pinned snapshot and CUDA device"]
fn real_full_l0_k2_matches_two_s1_rows() {
    let context = CudaContext::new(0).expect("create CUDA context");
    let stream = context.default_stream();
    let module = kernels::load(&context).expect("load embedded Rust PTX");
    run_real_l0_k2_if_available(&context, &module, &stream).expect("real V4 full-L0 K2 contract");
}

#[test]
#[ignore = "requires COLIBRI_DSV4_MODEL_DIR pinned snapshot and CUDA device"]
fn real_common_runtime_loads_nonzero_hca_then_csa_with_layer_experts() {
    let root = v4_model_dir().expect("pinned model directory");
    let context = CudaContext::new(0).expect("create CUDA context");
    let stream = context.default_stream();
    let manifests = v4_state::serial_layer_manifests(0).unwrap();
    let mut visited = Vec::new();
    let (hidden, telemetry) = v4_state::run_serial_layer_slice(
        vec![v4_router::f32_to_bf16_rne(1.0)],
        &manifests[3..=4],
        |manifest, input| {
            let (runtime, _) = load_layer_runtime(&root, manifest.layer, &[0], &stream)?;
            assert_eq!(runtime.weights.layer, manifest.layer);
            assert_eq!(
                runtime.weights.compression_ratio,
                manifest.compression_ratio
            );
            assert!(!runtime.weights.hash_router);
            assert_eq!(
                runtime.weights.indexer.is_some(),
                manifest.compression_ratio == v4_state::CompressionRatio::Csa4
            );
            let expert =
                v4_expert::RoutedExpert::load_layer(&runtime.expert_shard, manifest.layer, 0)?;
            assert!(expert.h2d_bytes() > 0);
            visited.push(manifest.layer);
            Ok((
                vec![v4_router::f32_to_bf16_rne(
                    v4_router::bf16_to_f32(input[0]) + manifest.layer as f32 + 1.0,
                )],
                manifest.contract_range_bytes,
                u64::from(manifest.compression_ratio == v4_state::CompressionRatio::Csa4),
            ))
        },
    )
    .unwrap();
    assert_eq!(visited, vec![3, 4]);
    assert_eq!(telemetry.len(), 2);
    assert_ne!(hidden, vec![v4_router::f32_to_bf16_rne(1.0)]);
}

#[test]
#[ignore = "requires COLIBRI_DSV4_MODEL_DIR pinned snapshot and CUDA device"]
fn real_l0_compressor_official_cublas_exact_arm_matches_oracle() {
    let context = CudaContext::new(0).expect("create CUDA context");
    let stream = context.default_stream();
    let module = kernels::load(&context).expect("load embedded Rust PTX");
    run_real_l0_compressor_cublas_exact_if_available(&module, &stream)
        .expect("official cuBLAS L0 Compressor exact arm");
}

#[test]
#[ignore = "requires COLIBRI_DSV4_MODEL_DIR pinned snapshot and CUDA device"]
fn real_compressor_cuda_prepare_and_paired_commit_match_fixture() {
    let context = CudaContext::new(0).expect("create CUDA context");
    let stream = context.default_stream();
    let module = kernels::load(&context).expect("load embedded Rust PTX");
    run_real_v4_compressor_slice_if_available(&module, &stream)
        .expect("real V4 Compressor CUDA contract");
}

#[test]
#[ignore = "requires COLIBRI_DSV4_MODEL_DIR pinned snapshot and CUDA device"]
fn real_mhc_cuda_matches_canonical_fixture() {
    let context = CudaContext::new(0).expect("create CUDA context");
    let stream = context.default_stream();
    let module = kernels::load(&context).expect("load embedded Rust PTX");
    run_real_v4_state_slice_if_available(&module, &stream).expect("real V4 mHC CUDA contract");
}

#[test]
fn rows_add_contract_rejects_bad_rows_and_accepts_empty_work() {
    assert_eq!(validate_rows_add(8, 0, 0, &[], 2), Ok(None));
    assert_eq!(
        validate_rows_add(8, 0, 0, &[], 0),
        Err(RowsAddError::ZeroWidth)
    );
    assert_eq!(
        validate_rows_add(8, 2, 1, &[-1], 2),
        Err(RowsAddError::NegativeRow(-1))
    );
    assert_eq!(
        validate_rows_add(8, 4, 2, &[1, 1], 2),
        Err(RowsAddError::DuplicateRow(1))
    );
    assert_eq!(
        validate_rows_add(8, 2, 1, &[4], 2),
        Err(RowsAddError::RowOutOfRange {
            row: 4,
            row_count: 4,
        })
    );
}

#[test]
fn k2_fp4_microtile_router_and_contribution_match_active_rows1() {
    fn fnv1a_u16(values: &[u16]) -> u64 {
        let mut hash = 0xcbf2_9ce4_8422_2325_u64;
        for value in values {
            for byte in value.to_le_bytes() {
                hash = (hash ^ byte as u64).wrapping_mul(0x100_0000_01b3);
            }
        }
        hash
    }

    let context = CudaContext::new(0).expect("create CUDA context");
    let stream = context.default_stream();
    let module = kernels::load(&context).expect("load embedded Rust PTX");

    let router_row = deterministic_v4_bf16(1, 41);
    let router_batch = [router_row.clone(), deterministic_v4_bf16(1, 73)].concat();
    let norm = vec![v4_router::f32_to_bf16_rne(1.0); v4_router::V4_HIDDEN];
    let logits_row: Vec<f32> = (0..v4_router::V4_ROUTED_EXPERTS)
        .map(|expert| (expert as f32 - 192.0) / 64.0)
        .collect();
    let logits_batch = [logits_row.clone(), logits_row.clone()].concat();
    let hash_row = [1_i64, 3, 5, 7, 9, 11];
    let hash_batch = [hash_row, hash_row].concat();
    let bias = vec![0.0_f32; v4_router::V4_ROUTED_EXPERTS];
    let (normalized_s1, routes_s1) = run_v4_router_gpu_oracle(
        &module,
        &stream,
        &router_row,
        &norm,
        &logits_row,
        &bias,
        &hash_row,
        true,
    )
    .expect("S=1 router");
    let (normalized_k2, routes_k2) = run_v4_router_gpu_oracle(
        &module,
        &stream,
        &router_batch,
        &norm,
        &logits_batch,
        &bias,
        &hash_batch,
        true,
    )
    .expect("K2 router");
    assert_eq!(
        &normalized_k2[..v4_router::V4_HIDDEN],
        normalized_s1.as_slice(),
        "active_rows=2 changed row0 normalized BF16 bits"
    );
    assert_eq!(
        &routes_k2.ids[..v4_router::V4_TOPK],
        routes_s1.ids.as_slice(),
        "active_rows=2 changed row0 routes"
    );
    assert!(
        routes_k2.weights[..v4_router::V4_TOPK]
            .iter()
            .zip(&routes_s1.weights)
            .all(|(k2, s1)| k2.to_bits() == s1.to_bits()),
        "active_rows=2 changed row0 route weight bits"
    );
    assert!(
        routes_k2.weights[v4_router::V4_TOPK..]
            .iter()
            .zip(&routes_s1.weights)
            .all(|(k2, s1)| k2.to_bits() == s1.to_bits()),
        "test requires the second row to share the selected route weights"
    );
    let route_weight = routes_s1.weights[0];

    const INPUT: usize = 32;
    const INTERMEDIATE: usize = 32;
    const OUTPUT: usize = 2;
    let row0: Vec<f32> = (0..INPUT).map(|index| index as f32 / 16.0 - 1.0).collect();
    let row1: Vec<f32> = (0..INPUT)
        .map(|index| (31 - index) as f32 / 20.0 - 0.75)
        .collect();
    let input_k2 = [row0.clone(), row1].concat();
    let w1: Vec<u8> = (0..INTERMEDIATE * INPUT / 2)
        .map(|index| [0x10, 0x32, 0x54, 0x76, 0x89, 0xab, 0xcd, 0xef][index % 8])
        .collect();
    let w3: Vec<u8> = (0..INTERMEDIATE * INPUT / 2)
        .map(|index| [0x21, 0x43, 0x65, 0x87, 0x9a, 0xbc, 0xde, 0xf0][index % 8])
        .collect();
    let w2: Vec<u8> = (0..OUTPUT * INTERMEDIATE / 2)
        .map(|index| [0x12, 0x34, 0x56, 0x78, 0x90, 0xab, 0xcd, 0xef][index % 8])
        .collect();
    let w1_scale = vec![127_u8; INTERMEDIATE * INPUT / 32];
    let w3_scale = vec![127_u8; INTERMEDIATE * INPUT / 32];
    let w2_scale = vec![127_u8; OUTPUT * INTERMEDIATE / 32];

    let run = |input: &[f32], rows: usize| {
        let input = DeviceBuffer::from_host(&stream, input).expect("upload synthetic rows");
        let w1_device = DeviceBuffer::from_host(&stream, &w1).expect("upload w1");
        let w3_device = DeviceBuffer::from_host(&stream, &w3).expect("upload w3");
        let w2_device = DeviceBuffer::from_host(&stream, &w2).expect("upload w2");
        let w1_scale_device =
            DeviceBuffer::from_host(&stream, &w1_scale).expect("upload w1 scales");
        let w3_scale_device =
            DeviceBuffer::from_host(&stream, &w3_scale).expect("upload w3 scales");
        let w2_scale_device =
            DeviceBuffer::from_host(&stream, &w2_scale).expect("upload w2 scales");
        let mut gate = DeviceBuffer::<u16>::zeroed(&stream, rows * INTERMEDIATE).expect("gate");
        let mut up = DeviceBuffer::<u16>::zeroed(&stream, rows * INTERMEDIATE).expect("up");
        let mut activated =
            DeviceBuffer::<u16>::zeroed(&stream, rows * INTERMEDIATE).expect("activated");
        let mut activated_f32 =
            DeviceBuffer::<f32>::zeroed(&stream, rows * INTERMEDIATE).expect("activated f32");
        let mut output = DeviceBuffer::<u16>::zeroed(&stream, rows * OUTPUT).expect("output");
        launch_v4_fp4_matvec(
            &module,
            &stream,
            &input,
            &w1_device,
            &w1_scale_device,
            &mut gate,
            rows,
            INTERMEDIATE,
            INPUT,
        )
        .expect("w1 fixed-microtile");
        launch_v4_fp4_matvec(
            &module,
            &stream,
            &input,
            &w3_device,
            &w3_scale_device,
            &mut up,
            rows,
            INTERMEDIATE,
            INPUT,
        )
        .expect("w3 fixed-microtile");
        launch_v4_swiglu_route(&module, &stream, &gate, &up, &mut activated, route_weight)
            .expect("weighted SwiGLU");
        launch_v4_bf16_to_f32(&module, &stream, &activated, &mut activated_f32)
            .expect("BF16 activation to FP32");
        launch_v4_fp4_matvec(
            &module,
            &stream,
            &activated_f32,
            &w2_device,
            &w2_scale_device,
            &mut output,
            rows,
            OUTPUT,
            INTERMEDIATE,
        )
        .expect("w2 fixed-microtile");
        (
            gate.to_host_vec(&stream).expect("download gate"),
            up.to_host_vec(&stream).expect("download up"),
            activated
                .to_host_vec(&stream)
                .expect("download weighted activation"),
            output.to_host_vec(&stream).expect("download contribution"),
        )
    };

    let s1 = run(&row0, 1);
    let k2 = run(&input_k2, 2);
    assert_eq!(&k2.0[..INTERMEDIATE], s1.0.as_slice(), "w1 raw BF16");
    assert_eq!(&k2.1[..INTERMEDIATE], s1.1.as_slice(), "w3 raw BF16");
    assert_eq!(
        &k2.2[..INTERMEDIATE],
        s1.2.as_slice(),
        "weighted SwiGLU BF16"
    );
    assert_eq!(&k2.3[..OUTPUT], s1.3.as_slice(), "final contribution BF16");
    println!(
        "K2_ROW_INVARIANCE raw_bit_mismatch=0 route_id_mismatch=0 route_weight_bit_mismatch=0 contribution_bit_mismatch=0 contribution_fnv1a={:016x}",
        fnv1a_u16(&s1.3)
    );
}
