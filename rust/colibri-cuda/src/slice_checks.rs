use super::*;

pub(crate) fn run_real_v4_fp4_slice_if_available(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
) -> Result<(), Box<dyn Error>> {
    let path = v4_expert_shard()?;
    let shard = safetensors::ShardFile::open(path)?;
    let weight_name = "layers.0.ffn.experts.0.w1.weight";
    let scale_name = "layers.0.ffn.experts.0.w1.scale";
    let weight = shard.tensor(weight_name)?;
    let scale = shard.tensor(scale_name)?;
    if weight.dtype != safetensors::TensorDtype::Fp4E2M1x2
        || scale.dtype != safetensors::TensorDtype::F8E8M0
        || weight.shape.len() != 2
        || scale.shape != [weight.shape[0], weight.shape[1] / 16]
    {
        return Err("real V4 FP4 weight/scale metadata disagrees with the official layout".into());
    }
    let mut packed = [0_u8; 16];
    let mut scales = [0_u8; 1];
    shard.read_at(weight_name, 0, &mut packed)?;
    shard.read_at(scale_name, 0, &mut scales)?;
    assert_eq!(
        packed,
        [
            0xd2, 0x25, 0x19, 0x21, 0x19, 0x94, 0xac, 0x5b, 0xbb, 0x29, 0x18, 0xaa, 0xba, 0xe3,
            0xcd, 0x43,
        ],
        "pinned V4 expert range"
    );
    assert_eq!(scales, [0x79], "pinned V4 E8M0 scale");
    let layout = v4_fp4::Fp4E2M1x2Layout::new(&packed, &scales, 1, 32)?;
    let mut expected = [0.0; 32];
    layout.decode_block(0, 0, &mut expected)?;
    let official = [
        0.015625, -0.046875, 0.046875, 0.015625, -0.0078125, 0.0078125, 0.0078125, 0.015625,
        -0.0078125, 0.0078125, 0.03125, -0.0078125, -0.03125, -0.015625, -0.0234375, 0.046875,
        -0.0234375, -0.0234375, -0.0078125, 0.015625, 0.0, 0.0078125, -0.015625, -0.015625,
        -0.015625, -0.0234375, 0.0234375, -0.0625, -0.046875, -0.03125, 0.0234375, 0.03125,
    ];
    assert_close("decode_v4_fp4_real_cpu", &expected, &official);
    let device_packed = DeviceBuffer::from_host(stream, &packed)?;
    let device_scales = DeviceBuffer::from_host(stream, &scales)?;
    let mut device_output = DeviceBuffer::from_host(stream, &vec![0.0f32; expected.len()])?;
    launch_v4_fp4_decode(
        module,
        stream,
        &device_packed,
        &device_scales,
        &mut device_output,
        layout,
    )?;
    assert_close(
        "decode_v4_fp4_real_slice",
        &device_output.to_host_vec(stream)?,
        &expected,
    );
    println!("PASS: real V4 FP4 first 32-value range matches CPU and GPU decoding");
    Ok(())
}

pub(crate) fn read_v4_bf16(
    shard: &safetensors::ShardFile,
    name: &str,
    shape: &[u64],
) -> Result<Vec<u16>, Box<dyn Error>> {
    let info = shard.tensor(name)?;
    if info.dtype != safetensors::TensorDtype::Bf16 || info.shape != shape {
        return Err(format!("{name} metadata does not match the pinned V4 BF16 layout").into());
    }
    let mut bytes = vec![0_u8; usize::try_from(info.byte_len)?];
    shard.read_at(name, 0, &mut bytes)?;
    Ok(bytes
        .chunks_exact(2)
        .map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]]))
        .collect())
}

pub(crate) fn read_v4_f32(
    shard: &safetensors::ShardFile,
    name: &str,
    shape: &[u64],
) -> Result<Vec<f32>, Box<dyn Error>> {
    let info = shard.tensor(name)?;
    if info.dtype != safetensors::TensorDtype::F32 || info.shape != shape {
        return Err(format!("{name} metadata does not match the pinned V4 F32 layout").into());
    }
    let mut bytes = vec![0_u8; usize::try_from(info.byte_len)?];
    shard.read_at(name, 0, &mut bytes)?;
    Ok(bytes
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        .collect())
}

pub(crate) fn read_v4_hash_rows(
    shard: &safetensors::ShardFile,
    name: &str,
    input_ids: &[i64],
) -> Result<Vec<i64>, Box<dyn Error>> {
    read_v4_hash_rows_for(&v4_spec::PRO, shard, name, input_ids)
}

pub(crate) fn read_v4_hash_rows_for(
    spec: &v4_spec::V4Spec,
    shard: &safetensors::ShardFile,
    name: &str,
    input_ids: &[i64],
) -> Result<Vec<i64>, Box<dyn Error>> {
    let info = shard.tensor(name)?;
    if info.dtype != safetensors::TensorDtype::I64
        || info.shape != [spec.vocab as u64, spec.experts_per_token as u64]
    {
        return Err(format!("{name} metadata does not match the pinned V4 hash table").into());
    }
    let mut result = Vec::with_capacity(input_ids.len() * spec.experts_per_token);
    for &input_id in input_ids {
        if !(0..i64::try_from(spec.vocab)?).contains(&input_id) {
            return Err(format!("V4 token id {input_id} is outside the hash table").into());
        }
        let mut row = vec![0_u8; spec.experts_per_token * 8];
        shard.read_at(name, input_id as u64 * row.len() as u64, &mut row)?;
        result.extend(
            row.chunks_exact(8)
                .map(|bytes| i64::from_le_bytes(bytes.try_into().unwrap())),
        );
    }
    Ok(result)
}

pub(crate) fn load_layer_hash_ids(
    model_dir: &std::path::Path,
    layer: usize,
    input_ids: &[i64],
) -> Result<Vec<i64>, Box<dyn Error>> {
    load_layer_hash_ids_for(&v4_spec::PRO, model_dir, layer, input_ids)
}

pub(crate) fn load_layer_hash_ids_for(
    spec: &'static v4_spec::V4Spec,
    model_dir: &std::path::Path,
    layer: usize,
    input_ids: &[i64],
) -> Result<Vec<i64>, Box<dyn Error>> {
    if layer >= spec.hash_layers {
        return Ok(Vec::new());
    }
    let shard = safetensors::ShardFile::open(model_dir.join(spec.shard_name(layer + 2)?))?;
    read_v4_hash_rows_for(
        spec,
        &shard,
        &format!("layers.{layer}.ffn.gate.tid2eid"),
        input_ids,
    )
}

pub(crate) fn deterministic_v4_bf16(rows: usize, seed: usize) -> Vec<u16> {
    (0..rows * v4_router::V4_HIDDEN)
        .map(|index| {
            let value = ((index.wrapping_mul(17).wrapping_add(seed) % 113) as f32 - 56.0) / 64.0;
            v4_router::f32_to_bf16_rne(value)
        })
        .collect()
}

pub(crate) fn assert_router_selection_close(
    name: &str,
    actual: &v4_router::RouterSelection,
    expected: &v4_router::RouterSelection,
) {
    assert_eq!(actual.ids, expected.ids, "{name}: selected expert IDs");
    assert_close(name, &actual.weights, &expected.weights);
    for weights in actual.weights.chunks_exact(v4_router::V4_TOPK) {
        assert_close(name, &[weights.iter().sum()], &[v4_router::V4_ROUTE_SCALE]);
    }
}

pub(crate) fn run_real_v4_state_slice_if_available(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
) -> Result<(), Box<dyn Error>> {
    let root = v4_model_dir()?;
    let state = v4_state::load(&root, &v4_state::IDS)?;
    let rows = v4_state::IDS.len();
    let rows_u32 = u32::try_from(rows)?;
    let linear_rows_u32 = rows_u32
        .checked_mul(32)
        .ok_or("mHC linear launch overflow")?;
    let fixture = v4_state::fixture_stages()?;
    let cpu_mixes = v4_state::mhc_linear_cpu(&state, rows)?;
    let cpu = v4_state::mhc_sinkhorn_cpu(&state, &cpu_mixes, rows)?;
    println!("V4 mHC stage isolation A=CPUlinear+CPUsinkhorn");
    report_v4_state_f32("A mixes/fixture", &cpu_mixes, &fixture.mixes);
    report_v4_state_f32("A pre/fixture", &cpu.pre, &fixture.pre);
    report_v4_state_f32("A post/fixture", &cpu.post, &fixture.post);
    report_v4_state_f32("A comb/fixture", &cpu.comb, &fixture.comb);
    report_v4_state_bf16("A collapsed/fixture", &cpu.collapsed, &fixture.collapsed);
    report_v4_state_trace(
        "A sinkhorn/fixture",
        &cpu.sinkhorn_trace,
        &fixture.sinkhorn_trace,
    );
    report_v4_state_invariants("A sinkhorn", &cpu.sinkhorn_trace);
    let mut repeated = Vec::with_capacity(4 * 4 * v4_state::HIDDEN);
    for row in state.embedding.chunks_exact(v4_state::HIDDEN) {
        for _ in 0..4 {
            repeated.extend_from_slice(row);
        }
    }
    let input = DeviceBuffer::from_host(stream, &repeated)?;
    let hc_fn = DeviceBuffer::from_host(stream, &state.hc_fn)?;
    let hc_scale = DeviceBuffer::from_host(stream, &state.hc_scale)?;
    let hc_base = DeviceBuffer::from_host(stream, &state.hc_base)?;
    let norm = DeviceBuffer::from_host(stream, &state.attn_norm)?;
    let cpu_mixes_device = DeviceBuffer::from_host(stream, &cpu_mixes)?;
    let mut gpu_mixes = DeviceBuffer::from_host(stream, &vec![0_f32; rows * v4_state::MIX])?;
    let mut scalar_mixes = DeviceBuffer::from_host(stream, &vec![0_f32; rows * v4_state::MIX])?;
    unsafe {
        module.mhc_linear_v4_bf16_scalar(
            stream,
            LaunchConfig::for_num_elems(rows_u32),
            &input,
            &hc_fn,
            &mut scalar_mixes,
        )?;
    }
    unsafe {
        module.mhc_linear_v4_bf16(
            stream,
            LaunchConfig::for_num_elems(linear_rows_u32),
            &input,
            &hc_fn,
            &mut gpu_mixes,
        )?;
    }
    let gpu_mixes_host = gpu_mixes.to_host_vec(stream)?;
    let scalar_mixes_host = scalar_mixes.to_host_vec(stream)?;
    if gpu_mixes_host
        .iter()
        .zip(&scalar_mixes_host)
        .any(|(candidate, scalar)| candidate.to_bits() != scalar.to_bits())
    {
        return Err("SM89 mHC linear candidate diverged from same-revision scalar oracle".into());
    }
    println!("V4 mHC stage isolation B=GPUlinear+CPUsinkhorn");
    report_v4_state_f32("B mixes/fixture", &gpu_mixes_host, &fixture.mixes);
    let c = v4_state::mhc_sinkhorn_cpu(&state, &gpu_mixes_host, rows)?;
    report_v4_state_f32("B pre/fixture", &c.pre, &fixture.pre);
    report_v4_state_f32("B post/fixture", &c.post, &fixture.post);
    report_v4_state_f32("B comb/fixture", &c.comb, &fixture.comb);
    report_v4_state_bf16("B collapsed/fixture", &c.collapsed, &fixture.collapsed);
    report_v4_state_trace(
        "B sinkhorn/fixture",
        &c.sinkhorn_trace,
        &fixture.sinkhorn_trace,
    );
    report_v4_state_invariants("B sinkhorn", &c.sinkhorn_trace);

    let mut b_pre = DeviceBuffer::from_host(stream, &vec![0_f32; rows * v4_state::HC])?;
    let mut b_post = DeviceBuffer::from_host(stream, &vec![0_f32; rows * v4_state::HC])?;
    let mut b_comb =
        DeviceBuffer::from_host(stream, &vec![0_f32; rows * v4_state::HC * v4_state::HC])?;
    let mut b_collapsed = DeviceBuffer::from_host(stream, &vec![0_u16; rows * v4_state::HIDDEN])?;
    let mut b_trace = DeviceBuffer::from_host(
        stream,
        &vec![0_f32; rows * 41 * v4_state::HC * v4_state::HC],
    )?;
    unsafe {
        module.mhc_sinkhorn_v4_bf16(
            stream,
            LaunchConfig::for_num_elems(rows_u32),
            &input,
            &cpu_mixes_device,
            &hc_scale,
            &hc_base,
            &mut b_pre,
            &mut b_post,
            &mut b_comb,
            &mut b_collapsed,
            &mut b_trace,
        )?;
    }
    println!("V4 mHC stage isolation C=CPUlinear+GPUsinkhorn");
    report_v4_state_f32("C pre/fixture", &b_pre.to_host_vec(stream)?, &fixture.pre);
    report_v4_state_f32(
        "C post/fixture",
        &b_post.to_host_vec(stream)?,
        &fixture.post,
    );
    report_v4_state_f32(
        "C comb/fixture",
        &b_comb.to_host_vec(stream)?,
        &fixture.comb,
    );
    report_v4_state_bf16(
        "C collapsed/fixture",
        &b_collapsed.to_host_vec(stream)?,
        &fixture.collapsed,
    );
    let b_trace_host = b_trace.to_host_vec(stream)?;
    report_v4_state_trace("C sinkhorn/fixture", &b_trace_host, &fixture.sinkhorn_trace);
    report_v4_state_invariants("C sinkhorn", &b_trace_host);

    let mut d_pre = DeviceBuffer::from_host(stream, &vec![0_f32; rows * v4_state::HC])?;
    let mut d_post = DeviceBuffer::from_host(stream, &vec![0_f32; rows * v4_state::HC])?;
    let mut d_comb =
        DeviceBuffer::from_host(stream, &vec![0_f32; rows * v4_state::HC * v4_state::HC])?;
    let mut d_collapsed = DeviceBuffer::from_host(stream, &vec![0_u16; rows * v4_state::HIDDEN])?;
    let mut d_trace = DeviceBuffer::from_host(
        stream,
        &vec![0_f32; rows * 41 * v4_state::HC * v4_state::HC],
    )?;
    unsafe {
        module.mhc_sinkhorn_v4_bf16(
            stream,
            LaunchConfig::for_num_elems(rows_u32),
            &input,
            &gpu_mixes,
            &hc_scale,
            &hc_base,
            &mut d_pre,
            &mut d_post,
            &mut d_comb,
            &mut d_collapsed,
            &mut d_trace,
        )?;
    }
    println!("V4 mHC stage isolation D=GPUlinear+GPUsinkhorn");
    let d_pre_host = d_pre.to_host_vec(stream)?;
    let d_post_host = d_post.to_host_vec(stream)?;
    let d_comb_host = d_comb.to_host_vec(stream)?;
    let (_, mixes_max) = report_v4_state_f32("D mixes/fixture", &gpu_mixes_host, &fixture.mixes);
    let (_, pre_max) = report_v4_state_f32("D pre/fixture", &d_pre_host, &fixture.pre);
    let (_, post_max) = report_v4_state_f32("D post/fixture", &d_post_host, &fixture.post);
    let (_, comb_max) = report_v4_state_f32("D comb/fixture", &d_comb_host, &fixture.comb);
    let d_collapsed_host = d_collapsed.to_host_vec(stream)?;
    let (collapsed_mismatches, _) =
        report_v4_state_bf16("D collapsed/fixture", &d_collapsed_host, &fixture.collapsed);
    let d_trace_host = d_trace.to_host_vec(stream)?;
    report_v4_state_trace("D sinkhorn/fixture", &d_trace_host, &fixture.sinkhorn_trace);
    report_v4_state_invariants("D sinkhorn", &d_trace_host);
    let mhc_max = mixes_max.max(pre_max).max(post_max).max(comb_max);
    if mhc_max > 1e-7
        || collapsed_mismatches != 0
        || gpu_mixes_host
            .iter()
            .chain(&d_pre_host)
            .chain(&d_post_host)
            .chain(&d_comb_host)
            .chain(&d_trace_host)
            .any(|value| !value.is_finite())
    {
        return Err("Rust GPU mHC canonical-compatible gate failed".into());
    }

    let cpu_fixture_norm = v4_router::rmsnorm_bf16(&fixture.collapsed, &state.attn_norm)?;
    let fixture_collapsed_device = DeviceBuffer::from_host(stream, &fixture.collapsed)?;
    let mut gpu_fixture_norm =
        DeviceBuffer::from_host(stream, &vec![0_u16; rows * v4_state::HIDDEN])?;
    unsafe {
        module.rmsnorm_v4_bf16(
            stream,
            LaunchConfig::for_num_elems(rows_u32),
            &fixture_collapsed_device,
            &norm,
            &mut gpu_fixture_norm,
            v4_state::HIDDEN as u32,
        )?;
    }
    println!("V4 attn-norm isolated on fixture collapsed input");
    report_v4_state_bf16("norm CPU/fixture", &cpu_fixture_norm, &fixture.norm);
    let gpu_fixture_norm_host = gpu_fixture_norm.to_host_vec(stream)?;
    let (norm_mismatches, _) =
        report_v4_state_bf16("norm GPU/fixture", &gpu_fixture_norm_host, &fixture.norm);
    if norm_mismatches != 0 {
        return Err("Rust GPU mHC attn RMSNorm gate failed".into());
    }

    unsafe {
        module.mhc_linear_v4_bf16(
            stream,
            LaunchConfig::for_num_elems(linear_rows_u32),
            &input,
            &hc_fn,
            &mut gpu_mixes,
        )?;
        module.mhc_sinkhorn_v4_bf16(
            stream,
            LaunchConfig::for_num_elems(rows_u32),
            &input,
            &gpu_mixes,
            &hc_scale,
            &hc_base,
            &mut d_pre,
            &mut d_post,
            &mut d_comb,
            &mut d_collapsed,
            &mut d_trace,
        )?;
        module.rmsnorm_v4_bf16(
            stream,
            LaunchConfig::for_num_elems(rows_u32),
            &fixture_collapsed_device,
            &norm,
            &mut gpu_fixture_norm,
            v4_state::HIDDEN as u32,
        )?;
    }
    let deterministic = gpu_mixes
        .to_host_vec(stream)?
        .iter()
        .zip(&gpu_mixes_host)
        .all(|(left, right)| left.to_bits() == right.to_bits())
        && d_pre
            .to_host_vec(stream)?
            .iter()
            .zip(&d_pre_host)
            .all(|(left, right)| left.to_bits() == right.to_bits())
        && d_post
            .to_host_vec(stream)?
            .iter()
            .zip(&d_post_host)
            .all(|(left, right)| left.to_bits() == right.to_bits())
        && d_comb
            .to_host_vec(stream)?
            .iter()
            .zip(&d_comb_host)
            .all(|(left, right)| left.to_bits() == right.to_bits())
        && d_collapsed.to_host_vec(stream)? == d_collapsed_host
        && d_trace
            .to_host_vec(stream)?
            .iter()
            .zip(&d_trace_host)
            .all(|(left, right)| left.to_bits() == right.to_bits())
        && gpu_fixture_norm.to_host_vec(stream)? == gpu_fixture_norm_host;
    if !deterministic {
        return Err("Rust GPU mHC repeat was not raw-bit deterministic".into());
    }
    let mut mhc_hash = 0xcbf2_9ce4_8422_2325_u64;
    for value in d_pre_host.iter().chain(&d_post_host).chain(&d_comb_host) {
        for byte in value.to_bits().to_le_bytes() {
            mhc_hash = (mhc_hash ^ byte as u64).wrapping_mul(0x100_0000_01b3);
        }
    }
    let h2d = input.len() * 2
        + hc_fn.len() * 4
        + hc_scale.len() * 4
        + hc_base.len() * 4
        + norm.len() * 2
        + cpu_mixes_device.len() * 4
        + gpu_mixes.len() * 4
        + scalar_mixes.len() * 4
        + b_pre.len() * 4
        + b_post.len() * 4
        + b_comb.len() * 4
        + b_collapsed.len() * 2
        + b_trace.len() * 4
        + d_pre.len() * 4
        + d_post.len() * 4
        + d_comb.len() * 4
        + d_collapsed.len() * 2
        + d_trace.len() * 4
        + fixture_collapsed_device.len() * 2
        + gpu_fixture_norm.len() * 2;
    println!(
        "PASS: real V4 L0 mHC canonical-compatible on sm89, max_abs={mhc_max:.9e}, finite=true, deterministic=true, collapsed_bits=exact, norm_bits=exact, fnv1a={mhc_hash:016x}, range={} B, actual H2D={} B",
        state.range_bytes, h2d,
    );
    Ok(())
}

pub(crate) fn report_v4_state_bf16(name: &str, actual: &[u16], expected: &[u16]) -> (usize, f32) {
    assert_eq!(actual.len(), expected.len(), "{name}: length");
    let mut mismatches = 0;
    let mut max = 0.0_f32;
    for (&actual, &expected) in actual.iter().zip(expected) {
        let delta = (v4_router::bf16_to_f32(actual) - v4_router::bf16_to_f32(expected)).abs();
        max = max.max(delta);
        mismatches += (actual != expected) as usize;
    }
    println!("{name}: mismatches={mismatches}, max_abs={max:.6e}");
    (mismatches, max)
}

pub(crate) fn report_v4_state_f32(name: &str, actual: &[f32], expected: &[f32]) -> (usize, f32) {
    assert_eq!(actual.len(), expected.len(), "{name}: length");
    let mut mismatches = 0;
    let mut max = 0.0_f32;
    for (&actual, &expected) in actual.iter().zip(expected) {
        let delta = (actual - expected).abs();
        max = max.max(delta);
        mismatches += (actual.to_bits() != expected.to_bits()) as usize;
    }
    println!("{name}: mismatches={mismatches}, max_abs={max:.6e}");
    (mismatches, max)
}

pub(crate) fn report_v4_state_trace(name: &str, actual: &[f32], expected: &[f32]) {
    pub(crate) const STAGES: usize = 41;
    pub(crate) const CELLS: usize = v4_state::HC * v4_state::HC;
    assert_eq!(actual.len(), expected.len(), "{name}: trace lengths");
    for stage in 0..STAGES {
        let label = match stage {
            0 => "comb_logits".to_owned(),
            1 => "after_softmax_eps".to_owned(),
            n if n % 2 == 0 => format!("col_{:02}", (n - 2) / 2),
            n => format!("row_{:02}", (n - 3) / 2),
        };
        let mut bit_mismatches = 0;
        let mut max = 0.0_f32;
        for (a, b) in actual
            .chunks_exact(STAGES * CELLS)
            .zip(expected.chunks_exact(STAGES * CELLS))
        {
            for (&actual, &expected) in a[stage * CELLS..][..CELLS]
                .iter()
                .zip(&b[stage * CELLS..][..CELLS])
            {
                bit_mismatches += (actual.to_bits() != expected.to_bits()) as usize;
                max = max.max((actual - expected).abs());
            }
        }
        println!("{name} {label}: bit_mismatches={bit_mismatches}, max_abs={max:.6e}");
    }
}

pub(crate) fn report_v4_state_invariants(name: &str, trace: &[f32]) {
    pub(crate) const CELLS: usize = v4_state::HC * v4_state::HC;
    for stage in 2..=40 {
        let column = stage % 2 == 0;
        let mut max_deviation = 0.0_f32;
        for row in trace.chunks_exact(41 * CELLS) {
            let matrix = &row[stage * CELLS..][..CELLS];
            for axis in 0..v4_state::HC {
                let sum: f32 = (0..v4_state::HC)
                    .map(|other| {
                        if column {
                            matrix[other * v4_state::HC + axis]
                        } else {
                            matrix[axis * v4_state::HC + other]
                        }
                    })
                    .sum();
                max_deviation = max_deviation.max((sum - 1.0).abs());
            }
        }
        let label = if column {
            format!("col_{:02}", (stage - 2) / 2)
        } else {
            format!("row_{:02}", (stage - 3) / 2)
        };
        println!("{name} {label} sum_max_deviation={max_deviation:.6e}");
    }
}

pub(crate) fn run_real_v4_fp8_projection_slice_if_available(
    context: &std::sync::Arc<CudaContext>,
    module: &kernels::LoadedModule,
    stream: &std::sync::Arc<CudaStream>,
) -> Result<(), Box<dyn Error>> {
    pub(crate) const WARM_REQUESTS: usize = 100;
    let root = v4_model_dir()?;
    let fixture = v4_state::fp8_projection_fixture()?;
    let range_start = std::time::Instant::now();
    let projections = v4_state::load_fp8_projections(&root)?;
    let range_elapsed = range_start.elapsed();
    let activation = v4_state::quantize_fp8_activation(&fixture.input_bf16, v4_state::IDS.len())?;
    let cpu_start = std::time::Instant::now();
    let wq_cpu = v4_state::project_fp8_cpu(&projections.wq_a, &activation)?;
    let wkv_cpu = v4_state::project_fp8_cpu(&projections.wkv, &activation)?;
    let wq_b_activation =
        v4_state::quantize_fp8_activation(&fixture.q_norm_bf16, v4_state::IDS.len())?;
    let wq_b_cpu = v4_state::project_fp8_linear_cpu(&projections.wq_b, &wq_b_activation)?;
    let q_head_rms_cpu = v4_state::query_head_rms_bf16(&wq_b_cpu.bf16)?;
    let query_freqs = v4_state::query_freqs_cis_pos0_3();
    let q_rope_cpu = v4_state::query_rope_bf16(&q_head_rms_cpu, &query_freqs)?;
    let kv_ready_cpu = v4_state::finalize_kv_cpu(&fixture.kv_norm_bf16, &query_freqs)?;
    let cpu_elapsed = cpu_start.elapsed();

    let input_host = PinnedHostBuffer::from_slice(context, &fixture.input_bf16)?;
    let q_fixture_host = PinnedHostBuffer::from_slice(context, &fixture.q_norm_bf16)?;
    let kv_fixture_host = PinnedHostBuffer::from_slice(context, &fixture.kv_norm_bf16)?;
    let query_freqs_host = PinnedHostBuffer::from_slice(context, &query_freqs)?;
    let wq_weight_host = PinnedHostBuffer::from_slice(context, &projections.wq_a.matrix.weight)?;
    let wq_scale_host = PinnedHostBuffer::from_slice(context, &projections.wq_a.matrix.scale)?;
    let q_norm_host = PinnedHostBuffer::from_slice(context, &projections.wq_a.norm)?;
    let wq_b_weight_host = PinnedHostBuffer::from_slice(context, &projections.wq_b.weight)?;
    let wq_b_scale_host = PinnedHostBuffer::from_slice(context, &projections.wq_b.scale)?;
    let wkv_weight_host = PinnedHostBuffer::from_slice(context, &projections.wkv.matrix.weight)?;
    let wkv_scale_host = PinnedHostBuffer::from_slice(context, &projections.wkv.matrix.scale)?;
    let kv_norm_host = PinnedHostBuffer::from_slice(context, &projections.wkv.norm)?;
    let mut input = unsafe { DeviceBuffer::uninitialized_async(stream, fixture.input_bf16.len())? };
    let mut q_fixture =
        unsafe { DeviceBuffer::uninitialized_async(stream, fixture.q_norm_bf16.len())? };
    let mut kv_fixture =
        unsafe { DeviceBuffer::uninitialized_async(stream, fixture.kv_norm_bf16.len())? };
    let mut query_freqs_device =
        unsafe { DeviceBuffer::uninitialized_async(stream, query_freqs.len())? };
    let mut wq_weight =
        unsafe { DeviceBuffer::uninitialized_async(stream, projections.wq_a.matrix.weight.len())? };
    let mut wq_scale =
        unsafe { DeviceBuffer::uninitialized_async(stream, projections.wq_a.matrix.scale.len())? };
    let mut q_norm =
        unsafe { DeviceBuffer::uninitialized_async(stream, projections.wq_a.norm.len())? };
    let mut wq_b_weight =
        unsafe { DeviceBuffer::uninitialized_async(stream, projections.wq_b.weight.len())? };
    let mut wq_b_scale =
        unsafe { DeviceBuffer::uninitialized_async(stream, projections.wq_b.scale.len())? };
    let mut wkv_weight =
        unsafe { DeviceBuffer::uninitialized_async(stream, projections.wkv.matrix.weight.len())? };
    let mut wkv_scale =
        unsafe { DeviceBuffer::uninitialized_async(stream, projections.wkv.matrix.scale.len())? };
    let mut kv_norm =
        unsafe { DeviceBuffer::uninitialized_async(stream, projections.wkv.norm.len())? };
    let h2d_start = context.new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
    let h2d_end = context.new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
    h2d_start.record(stream)?;
    // SAFETY: every pinned source remains alive through h2d_end synchronization,
    // and each destination allocation has the exact same element count.
    unsafe {
        input.copy_from_pinned_host_async(stream, &input_host)?;
        q_fixture.copy_from_pinned_host_async(stream, &q_fixture_host)?;
        kv_fixture.copy_from_pinned_host_async(stream, &kv_fixture_host)?;
        query_freqs_device.copy_from_pinned_host_async(stream, &query_freqs_host)?;
        wq_weight.copy_from_pinned_host_async(stream, &wq_weight_host)?;
        wq_scale.copy_from_pinned_host_async(stream, &wq_scale_host)?;
        q_norm.copy_from_pinned_host_async(stream, &q_norm_host)?;
        wq_b_weight.copy_from_pinned_host_async(stream, &wq_b_weight_host)?;
        wq_b_scale.copy_from_pinned_host_async(stream, &wq_b_scale_host)?;
        wkv_weight.copy_from_pinned_host_async(stream, &wkv_weight_host)?;
        wkv_scale.copy_from_pinned_host_async(stream, &wkv_scale_host)?;
        kv_norm.copy_from_pinned_host_async(stream, &kv_norm_host)?;
    }
    h2d_end.record(stream)?;
    h2d_end.synchronize()?;
    let h2d_ms = h2d_start.elapsed_ms(&h2d_end)? as f64;
    let h2d_bytes = projections.range_bytes
        + fixture.input_bf16.len() * 2
        + fixture.q_norm_bf16.len() * 2
        + fixture.kv_norm_bf16.len() * 2
        + query_freqs.len() * 4;

    let rows = v4_state::IDS.len();
    let blocks = v4_state::HIDDEN / v4_state::FP8_BLOCK;
    let mut input_fp8 =
        unsafe { DeviceBuffer::uninitialized_async(stream, fixture.input_bf16.len())? };
    let mut input_scale = unsafe { DeviceBuffer::uninitialized_async(stream, rows * blocks)? };
    let mut wq_f32 = unsafe {
        DeviceBuffer::uninitialized_async(stream, rows * projections.wq_a.matrix.out_features)?
    };
    let mut wq_bf16 = unsafe {
        DeviceBuffer::uninitialized_async(stream, rows * projections.wq_a.matrix.out_features)?
    };
    let mut q_normalized = unsafe {
        DeviceBuffer::uninitialized_async(stream, rows * projections.wq_a.matrix.out_features)?
    };
    let mut wq_b_input_fp8 =
        unsafe { DeviceBuffer::uninitialized_async(stream, fixture.q_norm_bf16.len())? };
    let mut wq_b_input_scale = unsafe {
        DeviceBuffer::uninitialized_async(
            stream,
            rows * projections.wq_b.in_features / v4_state::FP8_BLOCK,
        )?
    };
    let mut wq_b_f32 =
        unsafe { DeviceBuffer::uninitialized_async(stream, rows * projections.wq_b.out_features)? };
    let mut wq_b_bf16 =
        unsafe { DeviceBuffer::uninitialized_async(stream, rows * projections.wq_b.out_features)? };
    let mut q_head_rms =
        unsafe { DeviceBuffer::uninitialized_async(stream, rows * projections.wq_b.out_features)? };
    let mut q_rope =
        unsafe { DeviceBuffer::uninitialized_async(stream, rows * projections.wq_b.out_features)? };
    let mut unit_norm = unsafe { DeviceBuffer::<u16>::uninitialized_async(stream, 0)? };
    let mut wkv_f32 = unsafe {
        DeviceBuffer::uninitialized_async(stream, rows * projections.wkv.matrix.out_features)?
    };
    let mut wkv_bf16 = unsafe {
        DeviceBuffer::uninitialized_async(stream, rows * projections.wkv.matrix.out_features)?
    };
    let mut kv_normalized = unsafe {
        DeviceBuffer::uninitialized_async(stream, rows * projections.wkv.matrix.out_features)?
    };
    let mut kv_rope =
        unsafe { DeviceBuffer::uninitialized_async(stream, fixture.kv_norm_bf16.len())? };
    let mut kv_ready =
        unsafe { DeviceBuffer::uninitialized_async(stream, fixture.kv_norm_bf16.len())? };
    let mut kv_scales =
        unsafe { DeviceBuffer::uninitialized_async(stream, fixture.kv_scale_e8m0.len())? };
    let mut empty_fp8 = unsafe { DeviceBuffer::<u8>::uninitialized_async(stream, 0)? };
    let mut causal_topk =
        unsafe { DeviceBuffer::uninitialized_async(stream, fixture.causal_topk_i32.len())? };
    let kv_cache = DeviceBuffer::<u16>::zeroed(stream, fixture.cache_initial_bf16.len())?;

    // SAFETY: fixed pinned V4 shapes prove all slice lengths and launch counts.
    unsafe {
        module.quantize_v4_fp8_bf16(
            stream,
            LaunchConfig::for_num_elems(u32::try_from(rows * blocks)?),
            &input,
            &mut input_fp8,
            &mut input_scale,
            &mut unit_norm,
            u32::try_from(v4_state::HIDDEN)?,
            u32::try_from(v4_state::HIDDEN)?,
            u32::try_from(v4_state::FP8_BLOCK)?,
        )?;
        launch_v4_fp8_matvec(
            module,
            stream,
            &input_fp8,
            &input_scale,
            &wq_weight,
            &wq_scale,
            &mut wq_f32,
            &mut wq_bf16,
            projections.wq_a.matrix.in_features,
            projections.wq_a.matrix.out_features,
            false,
        )?;
        launch_v4_fp8_matvec(
            module,
            stream,
            &input_fp8,
            &input_scale,
            &wkv_weight,
            &wkv_scale,
            &mut wkv_f32,
            &mut wkv_bf16,
            projections.wkv.matrix.in_features,
            projections.wkv.matrix.out_features,
            false,
        )?;
        module.rmsnorm_v4_bf16(
            stream,
            LaunchConfig::for_num_elems(u32::try_from(rows)?),
            &wq_bf16,
            &q_norm,
            &mut q_normalized,
            u32::try_from(projections.wq_a.matrix.out_features)?,
        )?;
        module.rmsnorm_v4_bf16(
            stream,
            LaunchConfig::for_num_elems(u32::try_from(rows)?),
            &wkv_bf16,
            &kv_norm,
            &mut kv_normalized,
            u32::try_from(projections.wkv.matrix.out_features)?,
        )?;
        module.quantize_v4_fp8_bf16(
            stream,
            LaunchConfig::for_num_elems(u32::try_from(wq_b_input_scale.len())?),
            &q_fixture,
            &mut wq_b_input_fp8,
            &mut wq_b_input_scale,
            &mut unit_norm,
            u32::try_from(projections.wq_b.in_features)?,
            u32::try_from(projections.wq_b.in_features)?,
            u32::try_from(v4_state::FP8_BLOCK)?,
        )?;
        launch_v4_fp8_matvec(
            module,
            stream,
            &wq_b_input_fp8,
            &wq_b_input_scale,
            &wq_b_weight,
            &wq_b_scale,
            &mut wq_b_f32,
            &mut wq_b_bf16,
            projections.wq_b.in_features,
            projections.wq_b.out_features,
            true,
        )?;
        module.rmsnorm_v4_bf16(
            stream,
            LaunchConfig::for_num_elems(u32::try_from(rows * v4_state::Q_HEADS)?),
            &wq_b_bf16,
            &unit_norm,
            &mut q_head_rms,
            u32::try_from(v4_state::Q_HEAD_DIM)?,
        )?;
        module.rope_v4_bf16(
            stream,
            LaunchConfig::for_num_elems(u32::try_from(q_rope.len())?),
            &q_head_rms,
            &query_freqs_device,
            &mut q_rope,
            u32::try_from(v4_state::Q_HEADS)?,
            u32::try_from(v4_state::Q_HEAD_DIM)?,
        )?;
        launch_v4_kv_ready(
            module,
            stream,
            &kv_fixture,
            &query_freqs_device,
            &mut kv_rope,
            &mut kv_ready,
            &mut empty_fp8,
            &mut kv_scales,
            &mut causal_topk,
        )?;
    }

    let input_fp8_host = input_fp8.to_host_vec(stream)?;
    let input_scale_host = input_scale.to_host_vec(stream)?;
    let activation_mismatches = report_v4_state_u8(
        "FP8 activation GPU/fixture",
        &input_fp8_host,
        &fixture.activation,
    );
    let scale_mismatches = report_v4_state_u8(
        "E8M0 activation scale GPU/fixture",
        &input_scale_host,
        &fixture.activation_scale,
    );
    let wq_gpu_f32 = wq_f32.to_host_vec(stream)?;
    let wq_gpu_bf16 = wq_bf16.to_host_vec(stream)?;
    let q_normalized_host = q_normalized.to_host_vec(stream)?;
    let wkv_gpu_f32 = wkv_f32.to_host_vec(stream)?;
    let wkv_gpu_bf16 = wkv_bf16.to_host_vec(stream)?;
    let kv_normalized_host = kv_normalized.to_host_vec(stream)?;
    let wq_b_input_fp8_host = wq_b_input_fp8.to_host_vec(stream)?;
    let wq_b_input_scale_host = wq_b_input_scale.to_host_vec(stream)?;
    let wq_b_gpu_f32 = wq_b_f32.to_host_vec(stream)?;
    let wq_b_gpu_bf16 = wq_b_bf16.to_host_vec(stream)?;
    let q_head_rms_gpu = q_head_rms.to_host_vec(stream)?;
    let q_rope_gpu = q_rope.to_host_vec(stream)?;
    let kv_rope_gpu = kv_rope.to_host_vec(stream)?;
    let kv_ready_gpu = kv_ready.to_host_vec(stream)?;
    let kv_scales_gpu = kv_scales.to_host_vec(stream)?;
    let causal_topk_gpu = causal_topk.to_host_vec(stream)?;
    let kv_cache_initial_gpu = kv_cache.to_host_vec(stream)?;
    let (_, wq_cpu_raw_max) =
        report_v4_state_f32("wq_a CPU/fixture raw", &wq_cpu.raw_f32, &fixture.wq_f32);
    let (wq_cpu_bf16_mismatches, _) =
        report_v4_state_bf16("wq_a CPU/fixture BF16", &wq_cpu.bf16, &fixture.wq_bf16);
    let (_, q_cpu_max) = report_v4_state_bf16(
        "q_norm CPU/fixture",
        &wq_cpu.norm_bf16,
        &fixture.q_norm_bf16,
    );
    let (_, wq_gpu_raw_max) =
        report_v4_state_f32("wq_a GPU/fixture raw", &wq_gpu_f32, &fixture.wq_f32);
    report_v4_state_f32("wq_a GPU/CPU raw", &wq_gpu_f32, &wq_cpu.raw_f32);
    let (wq_gpu_bf16_mismatches, _) =
        report_v4_state_bf16("wq_a GPU/fixture BF16", &wq_gpu_bf16, &fixture.wq_bf16);
    let (wq_gpu_cpu_bf16_mismatches, _) =
        report_v4_state_bf16("wq_a GPU/CPU BF16", &wq_gpu_bf16, &wq_cpu.bf16);
    let (_, q_gpu_max) = report_v4_state_bf16(
        "q_norm GPU/fixture",
        &q_normalized_host,
        &fixture.q_norm_bf16,
    );
    let (q_gpu_cpu_mismatches, _) =
        report_v4_state_bf16("q_norm GPU/CPU", &q_normalized_host, &wq_cpu.norm_bf16);
    let (_, wkv_cpu_raw_max) =
        report_v4_state_f32("wkv CPU/fixture raw", &wkv_cpu.raw_f32, &fixture.wkv_f32);
    let (wkv_cpu_bf16_mismatches, _) =
        report_v4_state_bf16("wkv CPU/fixture BF16", &wkv_cpu.bf16, &fixture.wkv_bf16);
    let (_, kv_cpu_max) = report_v4_state_bf16(
        "kv_norm CPU/fixture",
        &wkv_cpu.norm_bf16,
        &fixture.kv_norm_bf16,
    );
    let (_, wkv_gpu_raw_max) =
        report_v4_state_f32("wkv GPU/fixture raw", &wkv_gpu_f32, &fixture.wkv_f32);
    report_v4_state_f32("wkv GPU/CPU raw", &wkv_gpu_f32, &wkv_cpu.raw_f32);
    let (wkv_gpu_bf16_mismatches, _) =
        report_v4_state_bf16("wkv GPU/fixture BF16", &wkv_gpu_bf16, &fixture.wkv_bf16);
    let (wkv_gpu_cpu_bf16_mismatches, _) =
        report_v4_state_bf16("wkv GPU/CPU BF16", &wkv_gpu_bf16, &wkv_cpu.bf16);
    let (_, kv_gpu_max) = report_v4_state_bf16(
        "kv_norm GPU/fixture",
        &kv_normalized_host,
        &fixture.kv_norm_bf16,
    );
    let (kv_gpu_cpu_mismatches, _) =
        report_v4_state_bf16("kv_norm GPU/CPU", &kv_normalized_host, &wkv_cpu.norm_bf16);
    let wq_b_activation_mismatches = report_v4_state_u8(
        "wq_b activation GPU/fixture",
        &wq_b_input_fp8_host,
        &fixture.wq_b_activation,
    );
    let wq_b_scale_mismatches = report_v4_state_u8(
        "wq_b activation scale GPU/fixture",
        &wq_b_input_scale_host,
        &fixture.wq_b_activation_scale,
    );
    let (_, wq_b_cpu_raw_max) =
        report_v4_state_f32("wq_b CPU/fixture raw", &wq_b_cpu.raw_f32, &fixture.wq_b_f32);
    let (wq_b_cpu_bf16_mismatches, _) =
        report_v4_state_bf16("wq_b CPU/fixture BF16", &wq_b_cpu.bf16, &fixture.wq_b_bf16);
    let (_, wq_b_gpu_raw_max) =
        report_v4_state_f32("wq_b GPU/fixture raw", &wq_b_gpu_f32, &fixture.wq_b_f32);
    report_v4_state_f32("wq_b GPU/CPU raw", &wq_b_gpu_f32, &wq_b_cpu.raw_f32);
    let (wq_b_gpu_bf16_mismatches, _) =
        report_v4_state_bf16("wq_b GPU/fixture BF16", &wq_b_gpu_bf16, &fixture.wq_b_bf16);
    let (wq_b_gpu_cpu_bf16_mismatches, _) =
        report_v4_state_bf16("wq_b GPU/CPU BF16", &wq_b_gpu_bf16, &wq_b_cpu.bf16);
    let (q_head_cpu_mismatches, _) = report_v4_state_bf16(
        "query head RMS CPU/fixture",
        &q_head_rms_cpu,
        &fixture.q_head_rms_bf16,
    );
    let (q_head_gpu_mismatches, _) = report_v4_state_bf16(
        "query head RMS GPU/fixture",
        &q_head_rms_gpu,
        &fixture.q_head_rms_bf16,
    );
    let (q_head_gpu_cpu_mismatches, _) =
        report_v4_state_bf16("query head RMS GPU/CPU", &q_head_rms_gpu, &q_head_rms_cpu);
    let (_, query_freqs_max) = report_v4_state_f32(
        "query frequencies CPU/fixture",
        &query_freqs,
        &fixture.freqs_cis_f32_pairs,
    );
    let (q_rope_cpu_mismatches, _) =
        report_v4_state_bf16("query RoPE CPU/fixture", &q_rope_cpu, &fixture.q_rope_bf16);
    let (q_rope_gpu_mismatches, _) =
        report_v4_state_bf16("query RoPE GPU/fixture", &q_rope_gpu, &fixture.q_rope_bf16);
    let (q_rope_gpu_cpu_mismatches, _) =
        report_v4_state_bf16("query RoPE GPU/CPU", &q_rope_gpu, &q_rope_cpu);
    let (kv_rope_cpu_mismatches, _) = report_v4_state_bf16(
        "KV RoPE CPU/fixture",
        &kv_ready_cpu.rope_bf16,
        &fixture.kv_rope_bf16,
    );
    let kv_scale_cpu_mismatches = report_v4_state_u8(
        "KV E8M0 scale CPU/fixture",
        &kv_ready_cpu.scale_e8m0,
        &fixture.kv_scale_e8m0,
    );
    let (kv_ready_cpu_mismatches, _) = report_v4_state_bf16(
        "KV ready CPU/fixture",
        &kv_ready_cpu.ready_bf16,
        &fixture.kv_ready_bf16,
    );
    let topk_cpu_mismatches = report_v4_state_i32(
        "causal top-k CPU/fixture",
        &kv_ready_cpu.causal_topk_i32,
        &fixture.causal_topk_i32,
    );
    let (kv_rope_gpu_mismatches, _) =
        report_v4_state_bf16("KV RoPE GPU/fixture", &kv_rope_gpu, &fixture.kv_rope_bf16);
    let (kv_rope_gpu_cpu_mismatches, _) =
        report_v4_state_bf16("KV RoPE GPU/CPU", &kv_rope_gpu, &kv_ready_cpu.rope_bf16);
    let kv_scale_gpu_mismatches = report_v4_state_u8(
        "KV E8M0 scale GPU/fixture",
        &kv_scales_gpu,
        &fixture.kv_scale_e8m0,
    );
    let (kv_ready_gpu_mismatches, _) = report_v4_state_bf16(
        "KV ready GPU/fixture",
        &kv_ready_gpu,
        &fixture.kv_ready_bf16,
    );
    let (kv_ready_gpu_cpu_mismatches, _) =
        report_v4_state_bf16("KV ready GPU/CPU", &kv_ready_gpu, &kv_ready_cpu.ready_bf16);
    let kv_rope_tail_unchanged = (0..rows).all(|row| {
        let tail = row * v4_state::Q_HEAD_DIM + v4_state::Q_HEAD_DIM - v4_state::Q_ROPE_DIM;
        kv_ready_gpu[tail..tail + v4_state::Q_ROPE_DIM]
            == kv_rope_gpu[tail..tail + v4_state::Q_ROPE_DIM]
    });
    let topk_gpu_mismatches = report_v4_state_i32(
        "causal top-k GPU/fixture",
        &causal_topk_gpu,
        &fixture.causal_topk_i32,
    );
    let (kv_cache_initial_mismatches, _) = report_v4_state_bf16(
        "KV cache initial GPU/fixture",
        &kv_cache_initial_gpu,
        &fixture.cache_initial_bf16,
    );
    let rope_prefix_unchanged = (0..rows * v4_state::Q_HEADS).all(|position_head| {
        let start = position_head * v4_state::Q_HEAD_DIM;
        q_rope_gpu[start..start + v4_state::Q_HEAD_DIM - v4_state::Q_ROPE_DIM]
            == q_head_rms_gpu[start..start + v4_state::Q_HEAD_DIM - v4_state::Q_ROPE_DIM]
    });
    let rope_position_zero_unchanged = q_rope_gpu[..v4_state::Q_HEADS * v4_state::Q_HEAD_DIM]
        == q_head_rms_gpu[..v4_state::Q_HEADS * v4_state::Q_HEAD_DIM];
    println!(
        "query RoPE invariants: prefix_448_unchanged={rope_prefix_unchanged}, position_0_unchanged={rope_position_zero_unchanged}"
    );
    println!("KV fake-quant invariant: rotary_tail_64_unchanged={kv_rope_tail_unchanged}");
    if activation_mismatches != 0
        || scale_mismatches != 0
        || wq_cpu_raw_max != 0.0
        || wq_cpu_bf16_mismatches != 0
        || q_cpu_max > 2.45e-4
        || wq_gpu_raw_max > 1e-6
        || wq_gpu_bf16_mismatches != 0
        || wq_gpu_cpu_bf16_mismatches != 0
        || q_gpu_max > 2.45e-4
        || q_gpu_cpu_mismatches != 0
        || wkv_cpu_raw_max != 0.0
        || wkv_cpu_bf16_mismatches != 0
        || kv_cpu_max != 0.0
        || wkv_gpu_raw_max > 1e-6
        || wkv_gpu_bf16_mismatches != 0
        || wkv_gpu_cpu_bf16_mismatches != 0
        || kv_gpu_max != 0.0
        || kv_gpu_cpu_mismatches != 0
        || wq_b_activation_mismatches != 0
        || wq_b_scale_mismatches != 0
        || wq_b_cpu_raw_max != 0.0
        || wq_b_cpu_bf16_mismatches != 0
        || wq_b_gpu_raw_max > 1e-6
        || wq_b_gpu_bf16_mismatches != 0
        || wq_b_gpu_cpu_bf16_mismatches != 0
        || q_head_cpu_mismatches != 0
        || q_head_gpu_mismatches != 0
        || q_head_gpu_cpu_mismatches != 0
        || query_freqs_max > 1e-6
        || q_rope_cpu_mismatches != 0
        || q_rope_gpu_mismatches != 0
        || q_rope_gpu_cpu_mismatches != 0
        || kv_rope_cpu_mismatches != 0
        || kv_scale_cpu_mismatches != 0
        || kv_ready_cpu_mismatches != 0
        || topk_cpu_mismatches != 0
        || kv_rope_gpu_mismatches != 0
        || kv_rope_gpu_cpu_mismatches != 0
        || kv_scale_gpu_mismatches != 0
        || kv_ready_gpu_mismatches != 0
        || kv_ready_gpu_cpu_mismatches != 0
        || topk_gpu_mismatches != 0
        || kv_cache_initial_mismatches != 0
        || !kv_rope_tail_unchanged
        || !rope_prefix_unchanged
        || !rope_position_zero_unchanged
    {
        return Err("V4 FP8 projection/Phase A correctness gate failed before cache write".into());
    }

    let mut kv_cache_cpu = v4_state::V4KvCache::new();
    if kv_cache_cpu.as_slice() != fixture.cache_initial_bf16 {
        return Err("V4 CPU KV cache initial gate failed".into());
    }
    kv_cache_cpu.commit(&kv_ready_cpu.ready_bf16, 1, 0, rows)?;
    if kv_cache_cpu.as_slice() != fixture.cache_after_bf16 {
        return Err("V4 CPU KV cache Phase B gate failed".into());
    }
    let kv_cache_cpu_after = kv_cache_cpu.as_slice().to_vec();
    kv_cache_cpu.commit(&kv_ready_cpu.ready_bf16, 1, 0, rows)?;
    if kv_cache_cpu.as_slice() != kv_cache_cpu_after {
        return Err("V4 CPU KV cache repeat was not idempotent".into());
    }

    let kv_commit_bytes =
        v4_state::preflight_kv_cache_commit(kv_cache.len(), kv_ready.len(), 1, 0, rows)?;
    // SAFETY: preflight proves the one permitted contiguous cache write fits
    // both fixed BF16 buffers. Phase A has passed its exact gate above.
    unsafe {
        cuda_core::memory::memcpy_dtod_async(
            kv_cache.cu_deviceptr(),
            kv_ready.cu_deviceptr(),
            kv_commit_bytes,
            stream.cu_stream(),
        )?;
    }
    let kv_cache_after_gpu = kv_cache.to_host_vec(stream)?;
    let (kv_cache_after_mismatches, _) = report_v4_state_bf16(
        "KV cache after GPU/fixture",
        &kv_cache_after_gpu,
        &fixture.cache_after_bf16,
    );
    let kv_cache_prefix_matches = kv_cache_after_gpu[..kv_ready_gpu.len()] == kv_ready_gpu[..];
    let kv_cache_tail_zero = kv_cache_after_gpu[kv_ready_gpu.len()..]
        .iter()
        .all(|&bits| bits == 0);
    let kv_cache_compressed_zero = kv_cache_after_gpu[v4_state::KV_WINDOW * v4_state::Q_HEAD_DIM..]
        .iter()
        .all(|&bits| bits == 0);

    // An exact repeat is byte-idempotent; no alternate payload reaches this raw copy.
    unsafe {
        cuda_core::memory::memcpy_dtod_async(
            kv_cache.cu_deviceptr(),
            kv_ready.cu_deviceptr(),
            kv_commit_bytes,
            stream.cu_stream(),
        )?;
    }
    let kv_cache_repeat_gpu = kv_cache.to_host_vec(stream)?;
    let kv_cache_repeat_idempotent = kv_cache_repeat_gpu == kv_cache_after_gpu;
    let invalid_rejected = v4_state::preflight_kv_cache_commit(
        kv_cache.len(),
        kv_ready.len(),
        1,
        v4_state::KV_WINDOW - 1,
        rows,
    )
    .is_err();
    let kv_cache_after_reject_gpu = kv_cache.to_host_vec(stream)?;
    let invalid_unchanged = kv_cache_after_reject_gpu == kv_cache_after_gpu;
    println!(
        "KV cache invariants: prefix_matches={kv_cache_prefix_matches}, tail_zero={kv_cache_tail_zero}, compressed_zero={kv_cache_compressed_zero}, repeat_idempotent={kv_cache_repeat_idempotent}, invalid_rejected={invalid_rejected}, invalid_unchanged={invalid_unchanged}"
    );
    if kv_cache_after_mismatches != 0
        || !kv_cache_prefix_matches
        || !kv_cache_tail_zero
        || !kv_cache_compressed_zero
        || !kv_cache_repeat_idempotent
        || !invalid_rejected
        || !invalid_unchanged
    {
        return Err("V4 KV cache Phase B correctness gate failed".into());
    }

    let wq_samples = benchmark_v4_fp8_matvec(
        context,
        module,
        stream,
        &input_fp8,
        &input_scale,
        &wq_weight,
        &wq_scale,
        &mut wq_f32,
        &mut wq_bf16,
        projections.wq_a.matrix.in_features,
        projections.wq_a.matrix.out_features,
        false,
        WARM_REQUESTS,
    )?;
    let wkv_samples = benchmark_v4_fp8_matvec(
        context,
        module,
        stream,
        &input_fp8,
        &input_scale,
        &wkv_weight,
        &wkv_scale,
        &mut wkv_f32,
        &mut wkv_bf16,
        projections.wkv.matrix.in_features,
        projections.wkv.matrix.out_features,
        false,
        WARM_REQUESTS,
    )?;
    let wq_b_samples = benchmark_v4_fp8_matvec(
        context,
        module,
        stream,
        &wq_b_input_fp8,
        &wq_b_input_scale,
        &wq_b_weight,
        &wq_b_scale,
        &mut wq_b_f32,
        &mut wq_b_bf16,
        projections.wq_b.in_features,
        projections.wq_b.out_features,
        true,
        WARM_REQUESTS,
    )?;
    let kv_ready_samples = benchmark_v4_kv_ready(
        context,
        module,
        stream,
        &kv_fixture,
        &query_freqs_device,
        &mut kv_rope,
        &mut kv_ready,
        &mut empty_fp8,
        &mut kv_scales,
        &mut causal_topk,
        WARM_REQUESTS,
    )?;
    let kv_commit_samples = benchmark_v4_kv_commit(
        context,
        stream,
        &kv_ready,
        &kv_cache,
        kv_commit_bytes,
        WARM_REQUESTS,
    )?;
    let (wq_median, wq_p95) = median_p95_ms(&wq_samples);
    let (wkv_median, wkv_p95) = median_p95_ms(&wkv_samples);
    let (wq_b_median, wq_b_p95) = median_p95_ms(&wq_b_samples);
    let (kv_ready_median, kv_ready_p95) = median_p95_ms(&kv_ready_samples);
    let (kv_commit_median, kv_commit_p95) = median_p95_ms(&kv_commit_samples);
    let wq_gbps = (rows
        * (projections.wq_a.matrix.weight.len() + projections.wq_a.matrix.scale.len()))
        as f64
        / (wq_median * 1_000_000.0);
    let wkv_gbps =
        (rows * (projections.wkv.matrix.weight.len() + projections.wkv.matrix.scale.len())) as f64
            / (wkv_median * 1_000_000.0);
    let wq_b_gbps = (rows * (projections.wq_b.weight.len() + projections.wq_b.scale.len())) as f64
        / (wq_b_median * 1_000_000.0);
    println!(
        "V4_FP8_COLD range={:.3}ms bytes={} H2D={:.3}ms bytes={} CPU_oracle={:.3}ms",
        range_elapsed.as_secs_f64() * 1_000.0,
        projections.range_bytes,
        h2d_ms,
        h2d_bytes,
        cpu_elapsed.as_secs_f64() * 1_000.0,
    );
    println!(
        "V4_FP8_WARM wq_a rows={rows} n={WARM_REQUESTS} kernel_median/p95={wq_median:.3}/{wq_p95:.3}ms effective_weight_read={wq_gbps:.3}GB/s"
    );
    println!(
        "V4_FP8_WARM wkv rows={rows} n={WARM_REQUESTS} kernel_median/p95={wkv_median:.3}/{wkv_p95:.3}ms effective_weight_read={wkv_gbps:.3}GB/s"
    );
    println!(
        "V4_FP8_WARM wq_b rows={rows} n={WARM_REQUESTS} kernel_median/p95={wq_b_median:.3}/{wq_b_p95:.3}ms effective_weight_read={wq_b_gbps:.3}GB/s"
    );
    println!(
        "V4_KV_PHASE_A_WARM rows={rows} n={WARM_REQUESTS} rope_fake_quant_topk_median/p95={kv_ready_median:.3}/{kv_ready_p95:.3}ms"
    );
    println!(
        "V4_KV_PHASE_B_WARM bytes={kv_commit_bytes} n={WARM_REQUESTS} single_copy_median/p95={kv_commit_median:.3}/{kv_commit_p95:.3}ms"
    );
    println!(
        "PROVISIONAL: correctness-first V4 L0 FP8 projections; no performance optimization claim"
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn launch_v4_kv_ready(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    input: &DeviceBuffer<u16>,
    freqs: &DeviceBuffer<f32>,
    rope: &mut DeviceBuffer<u16>,
    ready: &mut DeviceBuffer<u16>,
    empty_fp8: &mut DeviceBuffer<u8>,
    scales: &mut DeviceBuffer<u8>,
    topk: &mut DeviceBuffer<i32>,
) -> Result<(), Box<dyn Error>> {
    // SAFETY: the caller provides the frozen [1,4,512] KV boundary, the
    // position-0..3 frequencies, and exact-size disjoint outputs.
    unsafe {
        module.rope_v4_bf16(
            stream,
            LaunchConfig::for_num_elems(u32::try_from(rope.len())?),
            input,
            freqs,
            rope,
            1,
            u32::try_from(v4_state::Q_HEAD_DIM)?,
        )?;
        cuda_core::memory::memcpy_dtod_async(
            ready.cu_deviceptr(),
            rope.cu_deviceptr(),
            rope.num_bytes(),
            stream.cu_stream(),
        )?;
        module.quantize_v4_fp8_bf16(
            stream,
            LaunchConfig::for_num_elems(u32::try_from(scales.len())?),
            rope,
            empty_fp8,
            scales,
            ready,
            u32::try_from(v4_state::Q_HEAD_DIM)?,
            u32::try_from(v4_state::Q_HEAD_DIM - v4_state::Q_ROPE_DIM)?,
            64,
        )?;
        module.causal_topk_v4_i32(
            stream,
            LaunchConfig::for_num_elems(u32::try_from(topk.len())?),
            topk,
        )?;
    }
    Ok(())
}

pub(crate) unsafe fn launch_v4_fp8_matvec(
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
) -> Result<(), Box<dyn Error>> {
    if in_features % 128 == 0 && out_features % 8 == 0 {
        let rows = input.len() / in_features;
        let threads = rows
            .checked_mul(out_features / 8)
            .and_then(|tiles| tiles.checked_mul(32))
            .ok_or("FP8 MMA projection launch overflow")?;
        // SAFETY: caller provides validated FP8 buffers; the launch contains
        // complete warps and one warp owns each 8-wide output tile.
        unsafe {
            module.matvec_v4_fp8_mma(
                stream,
                LaunchConfig::for_num_elems(u32::try_from(threads)?),
                input,
                input_scales,
                weight,
                weight_scales,
                output_f32,
                output_bf16,
                u32::try_from(in_features)?,
                u32::try_from(out_features)?,
            )
        }?;
        return Ok(());
    }
    let threads = output_f32
        .len()
        .checked_mul(32)
        .ok_or("FP8 projection launch overflow")?;
    // SAFETY: caller provides the validated pinned V4 projection buffers.
    unsafe {
        module.matvec_v4_fp8(
            stream,
            LaunchConfig::for_num_elems(u32::try_from(threads)?),
            input,
            input_scales,
            weight,
            weight_scales,
            output_f32,
            output_bf16,
            u32::try_from(in_features)?,
            u32::try_from(out_features)?,
            strict_k_order,
        )
    }?;
    Ok(())
}
