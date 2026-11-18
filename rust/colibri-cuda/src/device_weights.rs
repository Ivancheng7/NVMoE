use super::*;

pub(crate) struct DeviceFp8Matrix {
    pub(crate) weight: DeviceBuffer<u8>,
    pub(crate) scale: DeviceBuffer<u8>,
    pub(crate) in_features: usize,
    pub(crate) out_features: usize,
}

impl DeviceFp8Matrix {
    pub(crate) fn new(
        stream: &CudaStream,
        matrix: &v4_state::Fp8Matrix,
    ) -> Result<Self, Box<dyn Error>> {
        Ok(Self {
            weight: DeviceBuffer::from_host(stream, &matrix.weight)?,
            scale: DeviceBuffer::from_host(stream, &matrix.scale)?,
            in_features: matrix.in_features,
            out_features: matrix.out_features,
        })
    }

    pub(crate) fn num_bytes(&self) -> usize {
        self.weight.num_bytes() + self.scale.num_bytes()
    }
}

pub(crate) struct DeviceLayerIndexer {
    pub(crate) wq_b: DeviceFp8Matrix,
    pub(crate) weights_proj: DeviceBuffer<u16>,
    pub(crate) compressor_wkv: DeviceBuffer<u16>,
    pub(crate) compressor_wgate: DeviceBuffer<u16>,
    pub(crate) compressor_ape: DeviceBuffer<f32>,
    pub(crate) compressor_norm: DeviceBuffer<u16>,
}

pub(crate) struct LayerDeviceWeights {
    pub(crate) spec: &'static v4_spec::V4Spec,
    pub(crate) layer: usize,
    pub(crate) compression_ratio: v4_state::CompressionRatio,
    pub(crate) hc_attn_fn: DeviceBuffer<f32>,
    pub(crate) hc_attn_base: DeviceBuffer<f32>,
    pub(crate) hc_attn_scale: DeviceBuffer<f32>,
    pub(crate) attn_norm: DeviceBuffer<u16>,
    pub(crate) wq_a: DeviceFp8Matrix,
    pub(crate) q_norm: DeviceBuffer<u16>,
    pub(crate) wq_b: DeviceFp8Matrix,
    pub(crate) wkv: DeviceFp8Matrix,
    pub(crate) kv_norm: DeviceBuffer<u16>,
    pub(crate) compressor_wkv: DeviceBuffer<u16>,
    pub(crate) compressor_wgate: DeviceBuffer<u16>,
    pub(crate) compressor_ape: DeviceBuffer<f32>,
    pub(crate) compressor_norm: DeviceBuffer<u16>,
    pub(crate) indexer: Option<DeviceLayerIndexer>,
    pub(crate) attn_sink: DeviceBuffer<f32>,
    pub(crate) wo_a: DeviceBuffer<u16>,
    pub(crate) wo_b: DeviceFp8Matrix,
    pub(crate) hc_ffn_fn: DeviceBuffer<f32>,
    pub(crate) hc_ffn_base: DeviceBuffer<f32>,
    pub(crate) hc_ffn_scale: DeviceBuffer<f32>,
    pub(crate) ffn_norm: DeviceBuffer<u16>,
    pub(crate) gate: DeviceBuffer<u16>,
    pub(crate) router_bias: DeviceBuffer<f32>,
    pub(crate) hash_router: bool,
    pub(crate) shared_w1: DeviceFp8Matrix,
    pub(crate) shared_w2: DeviceFp8Matrix,
    pub(crate) shared_w3: DeviceFp8Matrix,
}

impl LayerDeviceWeights {
    pub(crate) fn new_l0(
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

    pub(crate) fn new_layer(
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

    pub(crate) fn new_layer_with_wo_a(
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

    pub(crate) fn num_bytes(&self) -> usize {
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

    pub(crate) fn num_bytes_without_wo_a(&self) -> usize {
        self.num_bytes() - self.wo_a.num_bytes()
    }

    pub(crate) fn shared_num_bytes(&self) -> usize {
        self.shared_w1.num_bytes() + self.shared_w2.num_bytes() + self.shared_w3.num_bytes()
    }

    pub(crate) fn dense_num_bytes(&self) -> usize {
        self.num_bytes() - self.shared_num_bytes()
    }
}

/// MTP reuses the ordinary device weight layout only for the kernels that
/// consume the identical fixed tensors.  Its ratio-0 attention state is
/// `MtpDraftState`; the zero-length compressor fields are unreachable guards,
/// not a second persistent state.
pub(crate) fn upload_mtp_draft_weights(
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

pub(crate) struct MtpDraftPrefixDeviceWeights {
    pub(crate) hc_attn_fn: DeviceBuffer<f32>,
    pub(crate) hc_attn_base: DeviceBuffer<f32>,
    pub(crate) hc_attn_scale: DeviceBuffer<f32>,
    pub(crate) attn_norm: DeviceBuffer<u16>,
    pub(crate) wkv: DeviceFp8Matrix,
    pub(crate) kv_norm: DeviceBuffer<u16>,
}

pub(crate) fn upload_mtp_draft_prefix_weights(
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

pub(crate) struct LoadedLayerRuntime {
    pub(crate) weights: LayerDeviceWeights,
    pub(crate) expert_shard: safetensors::ShardFile,
}

pub(crate) struct RawLoadedLayerRuntime {
    pub(crate) runtime: LoadedLayerRuntime,
    pub(crate) wo_a_raw: DeviceBuffer<u8>,
    pub(crate) resident_bytes: usize,
}

pub(crate) struct RawDecodedLayerHost {
    pub(crate) host: DecodedLayerHost,
    pub(crate) wo_a_raw: Vec<u8>,
    pub(crate) resident_bytes: usize,
}

pub(crate) struct DecodedLayerHost {
    pub(crate) state: v4_state::LayerStateWeights,
    pub(crate) projections: v4_state::L0Fp8Projections,
    pub(crate) tail: v4_state::L0TailWeights,
}

pub(crate) fn drop_request_hash_ids_before_fixed_retention(
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

/// 把宿主负载拷进已有的设备缓冲。长度不一致直接报错，不静默读旧数据——槽位是
/// 跨层复用的，形状对不上就是层型配错了缓冲，必须炸。
fn restage_copy<T: cuda_core::DeviceCopy>(
    dst: &mut DeviceBuffer<T>,
    stream: &CudaStream,
    src: &[T],
) -> Result<(), Box<dyn Error>> {
    if dst.len() != src.len() {
        return Err(format!(
            "fixed layer restage length mismatch: {} != {}",
            dst.len(),
            src.len()
        )
        .into());
    }
    if dst.num_bytes() == 0 {
        return Ok(());
    }
    // SAFETY: src 属于常驻的宿主负载，活过整个请求；写 dst 与后续读它的 kernel
    // 在同一条 stream 上按序，所以这里不需要同步。
    unsafe { dst.copy_from_host_async_unchecked(stream, src) }
        .map_err(|error| Box::new(error) as Box<dyn Error>)
}

impl DeviceFp8Matrix {
    pub(crate) fn restage_from(
        &mut self,
        stream: &CudaStream,
        matrix: &v4_state::Fp8Matrix,
    ) -> Result<(), Box<dyn Error>> {
        restage_copy(&mut self.weight, stream, &matrix.weight)?;
        restage_copy(&mut self.scale, stream, &matrix.scale)?;
        self.in_features = matrix.in_features;
        self.out_features = matrix.out_features;
        Ok(())
    }
}

impl DeviceLayerIndexer {
    pub(crate) fn restage_from(
        &mut self,
        stream: &CudaStream,
        indexer: &v4_state::CsaIndexerWeights,
    ) -> Result<(), Box<dyn Error>> {
        self.wq_b.restage_from(stream, &indexer.wq_b)?;
        restage_copy(&mut self.weights_proj, stream, &indexer.weights_proj_bf16)?;
        restage_copy(
            &mut self.compressor_wkv,
            stream,
            &indexer.compressor.wkv_bf16,
        )?;
        restage_copy(
            &mut self.compressor_wgate,
            stream,
            &indexer.compressor.wgate_bf16,
        )?;
        restage_copy(
            &mut self.compressor_ape,
            stream,
            &indexer.compressor.ape_f32,
        )?;
        restage_copy(
            &mut self.compressor_norm,
            stream,
            &indexer.compressor.norm_bf16,
        )?;
        Ok(())
    }
}

impl LayerDeviceWeights {
    /// 按 `new_layer_with_wo_a` 的字段表把这一层的负载拷进已有缓冲。
    ///
    /// 这张表必须和构造函数逐项对齐：漏一项就是这一层拿着上一层的旧权重跑，
    /// 输出不会报错、只会悄悄变。`wo_a` 是唯一例外——raw 路径的 wo_a 由调用方
    /// 另行解码后换进来（见 `generate.rs` 的 swap），这里只在宿主侧给了 bf16
    /// 时才拷。
    pub(crate) fn restage_from_host(
        &mut self,
        stream: &CudaStream,
        state: &v4_state::LayerStateWeights,
        projections: &v4_state::L0Fp8Projections,
        tail: &v4_state::L0TailWeights,
        spare_indexer: &mut Option<DeviceLayerIndexer>,
    ) -> Result<(), Box<dyn Error>> {
        self.spec = state.spec;
        self.layer = state.layer;
        self.compression_ratio = state.compression_ratio;
        restage_copy(&mut self.hc_attn_fn, stream, &state.hc_fn)?;
        restage_copy(&mut self.hc_attn_base, stream, &state.hc_base)?;
        restage_copy(&mut self.hc_attn_scale, stream, &state.hc_scale)?;
        restage_copy(&mut self.attn_norm, stream, &state.attn_norm)?;
        self.wq_a.restage_from(stream, &projections.wq_a.matrix)?;
        restage_copy(&mut self.q_norm, stream, &projections.wq_a.norm)?;
        self.wq_b.restage_from(stream, &projections.wq_b)?;
        self.wkv.restage_from(stream, &projections.wkv.matrix)?;
        restage_copy(&mut self.kv_norm, stream, &projections.wkv.norm)?;
        restage_copy(&mut self.compressor_wkv, stream, &state.compressor.wkv_bf16)?;
        restage_copy(
            &mut self.compressor_wgate,
            stream,
            &state.compressor.wgate_bf16,
        )?;
        restage_copy(&mut self.compressor_ape, stream, &state.compressor.ape_f32)?;
        restage_copy(
            &mut self.compressor_norm,
            stream,
            &state.compressor.norm_bf16,
        )?;
        // indexer 只在 Csa4 层存在，其余层型要把缓冲交出来存着：三个槽位里只有
        // 一个带 indexer 缓冲，不存就会在层型交替时被丢掉。
        if let Some(indexer) = state.indexer.as_ref() {
            let mut device = self
                .indexer
                .take()
                .or_else(|| spare_indexer.take())
                .ok_or("fixed layer slot has no indexer buffers to restage into")?;
            device.restage_from(stream, indexer)?;
            self.indexer = Some(device);
        } else if let Some(device) = self.indexer.take() {
            *spare_indexer = Some(device);
        }
        restage_copy(&mut self.attn_sink, stream, &tail.attn_sink)?;
        if !tail.wo_a_bf16.is_empty() {
            restage_copy(&mut self.wo_a, stream, &tail.wo_a_bf16)?;
        }
        self.wo_b.restage_from(stream, &tail.wo_b)?;
        restage_copy(&mut self.hc_ffn_fn, stream, &tail.hc_ffn_fn)?;
        restage_copy(&mut self.hc_ffn_base, stream, &tail.hc_ffn_base)?;
        restage_copy(&mut self.hc_ffn_scale, stream, &tail.hc_ffn_scale)?;
        restage_copy(&mut self.ffn_norm, stream, &state.ffn_norm)?;
        restage_copy(&mut self.gate, stream, &state.gate)?;
        let (router_bias, hash_router) = match &state.router {
            v4_state::LayerRouterWeights::Hash { .. } => {
                (vec![0.0_f32; state.spec.routed_experts], true)
            }
            v4_state::LayerRouterWeights::Learned { bias } => (bias.clone(), false),
        };
        restage_copy(&mut self.router_bias, stream, &router_bias)?;
        self.hash_router = hash_router;
        self.shared_w1.restage_from(stream, &tail.shared.w1)?;
        self.shared_w2.restage_from(stream, &tail.shared.w2)?;
        self.shared_w3.restage_from(stream, &tail.shared.w3)?;
        Ok(())
    }
}
