use super::*;

pub(crate) fn bench_telemetry_enabled() -> bool {
    std::env::var_os("COLIBRI_V4_BENCH_TELEMETRY").as_deref() == Some(std::ffi::OsStr::new("1"))
        || std::env::var_os("COLIBRI_V4_PHASE_PROFILE").as_deref()
            == Some(std::ffi::OsStr::new("1"))
}

pub(crate) fn load_layer_host(
    model_dir: &std::path::Path,
    layer: usize,
    token_ids: &[i64],
) -> Result<(DecodedLayerHost, FixedLoadTelemetry), Box<dyn Error>> {
    load_layer_host_inner(model_dir, layer, token_ids, false)
}

pub(crate) fn load_layer_host_raw_wo_a(
    model_dir: &std::path::Path,
    layer: usize,
    token_ids: &[i64],
) -> Result<(DecodedLayerHost, FixedLoadTelemetry), Box<dyn Error>> {
    load_layer_host_inner(model_dir, layer, token_ids, true)
}

pub(crate) fn load_layer_host_inner(
    model_dir: &std::path::Path,
    layer: usize,
    token_ids: &[i64],
    raw_wo_a: bool,
) -> Result<(DecodedLayerHost, FixedLoadTelemetry), Box<dyn Error>> {
    load_layer_host_inner_for(&v4_spec::PRO, model_dir, layer, token_ids, raw_wo_a)
}

pub(crate) fn load_layer_host_inner_for(
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

pub(crate) fn upload_layer_runtime(
    model_dir: &std::path::Path,
    layer: usize,
    host: &DecodedLayerHost,
    stream: &CudaStream,
) -> Result<LoadedLayerRuntime, Box<dyn Error>> {
    upload_layer_runtime_for(&v4_spec::PRO, model_dir, layer, host, stream)
}

pub(crate) fn upload_layer_runtime_for(
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

pub(crate) fn upload_layer_runtime_raw_wo_a(
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

pub(crate) fn upload_pinned_layer_runtime_raw(
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

pub(crate) fn read_raw_wo_a(
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

pub(crate) fn into_raw_layer_runtime(
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

pub(crate) fn into_raw_layer_host(
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

pub(crate) fn load_layer_runtime(
    model_dir: &std::path::Path,
    layer: usize,
    token_ids: &[i64],
    stream: &CudaStream,
) -> Result<(LoadedLayerRuntime, FixedLoadTelemetry), Box<dyn Error>> {
    load_layer_runtime_for(&v4_spec::PRO, model_dir, layer, token_ids, stream)
}

pub(crate) fn load_layer_runtime_for(
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
