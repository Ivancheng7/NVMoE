use super::*;

impl DeviceRequestScratch {
    pub(crate) fn new(
        stream: &CudaStream,
        weights: &LayerDeviceWeights,
        plan: v4_state::LayerStationaryRequestPlan,
    ) -> Result<Self, Box<dyn Error>> {
        if weights.wq_a.out_features != V4_Q_LORA_RANK
            || weights.wq_b.out_features != v4_state::Q_HEADS * v4_state::Q_HEAD_DIM
            || weights.wkv.out_features != v4_state::Q_HEAD_DIM
        {
            return Err(
                "generic stationary scratch projection shapes differ from its geometry".into(),
            );
        }
        Self::new_with_strict(stream, plan, strict_compressor_enabled())
    }

    pub(crate) fn new_with_strict(
        stream: &CudaStream,
        plan: v4_state::LayerStationaryRequestPlan,
        target_strict_compressor: bool,
    ) -> Result<Self, Box<dyn Error>> {
        let scratch_geometry = DeviceRequestScratchGeometry::new(plan, target_strict_compressor)?;
        let geometry = scratch_geometry.generic;
        let topk_rows = scratch_geometry.topk_rows;
        let scratch = Self {
            hc_hidden: DeviceBuffer::zeroed(stream, v4_state::HC * v4_state::HIDDEN)?,
            freqs: DeviceBuffer::zeroed(stream, v4_state::Q_ROPE_DIM)?,
            inverse_freqs: DeviceBuffer::zeroed(stream, v4_state::Q_ROPE_DIM)?,
            compressed_freqs: DeviceBuffer::zeroed(stream, v4_state::Q_ROPE_DIM)?,
            mixes: DeviceBuffer::zeroed(stream, v4_state::MIX)?,
            pre: DeviceBuffer::zeroed(stream, v4_state::HC)?,
            post: DeviceBuffer::zeroed(stream, v4_state::HC)?,
            comb: DeviceBuffer::zeroed(stream, v4_state::HC * v4_state::HC)?,
            trace: DeviceBuffer::zeroed(stream, 41 * v4_state::HC * v4_state::HC)?,
            collapsed: DeviceBuffer::zeroed(stream, v4_state::HIDDEN)?,
            attn_input: DeviceBuffer::zeroed(stream, v4_state::HIDDEN)?,
            fp8_7168: DeviceBuffer::zeroed(stream, v4_state::HIDDEN)?,
            fp8_7168_scales: DeviceBuffer::zeroed(stream, v4_state::HIDDEN / v4_state::FP8_BLOCK)?,
            fp8_1536: DeviceBuffer::zeroed(stream, V4_Q_LORA_RANK)?,
            fp8_1536_scales: DeviceBuffer::zeroed(stream, V4_Q_LORA_RANK / v4_state::FP8_BLOCK)?,
            fp8_16384: DeviceBuffer::zeroed(stream, v4_state::O_GROUPS * v4_state::O_LORA_RANK)?,
            fp8_16384_scales: DeviceBuffer::zeroed(
                stream,
                v4_state::O_GROUPS * v4_state::O_LORA_RANK / v4_state::FP8_BLOCK,
            )?,
            empty_u8: DeviceBuffer::zeroed(stream, 0)?,
            empty_u16: DeviceBuffer::zeroed(stream, 0)?,
            q_lora_f32: DeviceBuffer::zeroed(stream, V4_Q_LORA_RANK)?,
            q_lora_raw: DeviceBuffer::zeroed(stream, V4_Q_LORA_RANK)?,
            q_lora: DeviceBuffer::zeroed(stream, V4_Q_LORA_RANK)?,
            q_f32: DeviceBuffer::zeroed(stream, v4_state::Q_HEADS * v4_state::Q_HEAD_DIM)?,
            q_raw: DeviceBuffer::zeroed(stream, v4_state::Q_HEADS * v4_state::Q_HEAD_DIM)?,
            q_norm: DeviceBuffer::zeroed(stream, v4_state::Q_HEADS * v4_state::Q_HEAD_DIM)?,
            q_rope: DeviceBuffer::zeroed(stream, v4_state::Q_HEADS * v4_state::Q_HEAD_DIM)?,
            kv_f32: DeviceBuffer::zeroed(stream, v4_state::Q_HEAD_DIM)?,
            kv_raw: DeviceBuffer::zeroed(stream, v4_state::Q_HEAD_DIM)?,
            kv_norm: DeviceBuffer::zeroed(stream, v4_state::Q_HEAD_DIM)?,
            kv_rope: DeviceBuffer::zeroed(stream, v4_state::Q_HEAD_DIM)?,
            kv_ready: DeviceBuffer::zeroed(stream, v4_state::Q_HEAD_DIM)?,
            kv_scales: DeviceBuffer::zeroed(stream, 7)?,
            candidate_tail_kv: DeviceBuffer::zeroed(
                stream,
                geometry.tail_elements(GenericStationaryPath::Csa4),
            )?,
            candidate_tail_score: DeviceBuffer::zeroed(
                stream,
                geometry.tail_elements(GenericStationaryPath::Csa4),
            )?,
            candidate_hca_tail_kv: DeviceBuffer::zeroed(
                stream,
                geometry.tail_elements(GenericStationaryPath::Hca128),
            )?,
            candidate_hca_tail_score: DeviceBuffer::zeroed(
                stream,
                geometry.tail_elements(GenericStationaryPath::Hca128),
            )?,
            compressor_kv: DeviceBuffer::zeroed(
                stream,
                geometry.main_elements(GenericStationaryPath::Csa4),
            )?,
            compressor_score: DeviceBuffer::zeroed(
                stream,
                geometry.main_elements(GenericStationaryPath::Csa4),
            )?,
            compressor_score_plus_ape: DeviceBuffer::zeroed(
                stream,
                geometry.main_elements(GenericStationaryPath::Csa4),
            )?,
            hca_compressor_kv: DeviceBuffer::zeroed(
                stream,
                geometry.main_elements(GenericStationaryPath::Hca128),
            )?,
            hca_compressor_score: DeviceBuffer::zeroed(
                stream,
                geometry.main_elements(GenericStationaryPath::Hca128),
            )?,
            hca_compressor_score_plus_ape: DeviceBuffer::zeroed(
                stream,
                geometry.main_elements(GenericStationaryPath::Hca128),
            )?,
            exact_compressor: scratch_geometry
                .strict
                .then(|| GenericExactCompressorScratch::new_for_geometry(stream, scratch_geometry))
                .transpose()?,
            indexer_compressor_kv: Some(DeviceBuffer::zeroed(
                stream,
                geometry.csa_indexer_elements,
            )?),
            indexer_compressor_score: Some(DeviceBuffer::zeroed(
                stream,
                geometry.csa_indexer_elements,
            )?),
            indexer_compressor_score_plus_ape: Some(DeviceBuffer::zeroed(
                stream,
                geometry.csa_indexer_elements,
            )?),
            candidate_indexer_tail_kv: Some(DeviceBuffer::zeroed(
                stream,
                8 * geometry.csa_indexer_elements,
            )?),
            candidate_indexer_tail_score: Some(DeviceBuffer::zeroed(
                stream,
                8 * geometry.csa_indexer_elements,
            )?),
            compressed_pool: DeviceBuffer::zeroed(stream, v4_state::Q_HEAD_DIM)?,
            compressed_bf16: DeviceBuffer::zeroed(stream, v4_state::Q_HEAD_DIM)?,
            compressed_norm: DeviceBuffer::zeroed(stream, v4_state::Q_HEAD_DIM)?,
            compressed_rope: DeviceBuffer::zeroed(stream, v4_state::Q_HEAD_DIM)?,
            compressed_ready: DeviceBuffer::zeroed(stream, v4_state::Q_HEAD_DIM)?,
            compressed_scales: DeviceBuffer::zeroed(stream, 7)?,
            indexer_pool: DeviceBuffer::zeroed(stream, 128)?,
            indexer_bf16: DeviceBuffer::zeroed(stream, 128)?,
            indexer_norm: DeviceBuffer::zeroed(stream, 128)?,
            indexer_rope: DeviceBuffer::zeroed(stream, 128)?,
            indexer_ready: DeviceBuffer::zeroed(stream, 128)?,
            candidate_indexer_cache: DeviceBuffer::zeroed(
                stream,
                scratch_geometry
                    .csa_compressed_rows
                    .checked_mul(128)
                    .ok_or("generic stationary candidate indexer cache capacity overflow")?,
            )?,
            indexer_query_raw: DeviceBuffer::zeroed(stream, 64 * 128)?,
            indexer_query_rope: DeviceBuffer::zeroed(stream, 64 * 128)?,
            indexer_query_fp4: DeviceBuffer::zeroed(stream, 64 * 128)?,
            indexer_query_f32: DeviceBuffer::zeroed(stream, 64 * 128)?,
            indexer_weights_f32: DeviceBuffer::zeroed(stream, 64)?,
            indexer_kv_f32: DeviceBuffer::zeroed(
                stream,
                scratch_geometry
                    .csa_compressed_rows
                    .checked_mul(128)
                    .ok_or("generic stationary indexer scratch capacity overflow")?,
            )?,
            indexer_scores: DeviceBuffer::zeroed(stream, scratch_geometry.csa_compressed_rows)?,
            indexer_valid: DeviceBuffer::from_host(
                stream,
                &vec![1_u32; scratch_geometry.csa_compressed_rows],
            )?,
            indexer_selected: DeviceBuffer::zeroed(
                stream,
                scratch_geometry
                    .csa_compressed_rows
                    .min(v4_state::INDEX_TOPK),
            )?,
            topk: DeviceBuffer::zeroed(stream, topk_rows)?,
            attention_cache: DeviceBuffer::zeroed(stream, topk_rows * v4_state::Q_HEAD_DIM)?,
            sparse: DeviceBuffer::zeroed(stream, v4_state::Q_HEADS * v4_state::Q_HEAD_DIM)?,
            sparse_scratch: SparseAttentionScoreReuseScratch::new(stream, 1, topk_rows)?,
            sparse_inverse: DeviceBuffer::zeroed(stream, v4_state::Q_HEADS * v4_state::Q_HEAD_DIM)?,
            wo_a: DeviceBuffer::zeroed(stream, v4_state::O_GROUPS * v4_state::O_LORA_RANK)?,
            attention_f32: DeviceBuffer::zeroed(stream, scratch_geometry.attention_f32_elements())?,
            attention: DeviceBuffer::zeroed(stream, v4_state::HIDDEN)?,
            after_attention: DeviceBuffer::zeroed(stream, v4_state::HC * v4_state::HIDDEN)?,
        };
        let expected_bytes = scratch_geometry.num_bytes()?;
        let actual_bytes = scratch.num_bytes();
        if actual_bytes != expected_bytes {
            return Err(format!(
                "generic stationary scratch allocation diverged from its geometry contract: {}",
                serde_json::json!({
                    "plan": {"prompt_rows": plan.prompt_rows, "max_new_tokens": plan.max_new_tokens, "context_rows": plan.context_rows, "csa_rows": scratch_geometry.csa_compressed_rows, "hca_rows": plan.hca_compressed_rows, "topk_rows": scratch_geometry.topk_rows},
                    "target_strict_compressor": scratch_geometry.strict,
                    "runtime_strict_compressor": strict_compressor_enabled(),
                    "expected_bytes": expected_bytes,
                    "actual_bytes": actual_bytes,
                    "expected_breakdown": scratch_geometry.byte_breakdown()?.json(),
                    "actual_breakdown": scratch.byte_breakdown().json(),
                })
            )
            .into());
        }
        Ok(scratch)
    }

    pub(crate) fn main_compressor(
        &self,
        path: GenericStationaryPath,
    ) -> (&DeviceBuffer<f32>, &DeviceBuffer<f32>, &DeviceBuffer<f32>) {
        match path {
            GenericStationaryPath::Csa4 => (
                &self.compressor_kv,
                &self.compressor_score,
                &self.compressor_score_plus_ape,
            ),
            GenericStationaryPath::Hca128 => (
                &self.hca_compressor_kv,
                &self.hca_compressor_score,
                &self.hca_compressor_score_plus_ape,
            ),
        }
    }

    pub(crate) fn candidate_tails(
        &self,
        path: GenericStationaryPath,
    ) -> (&DeviceBuffer<f32>, &DeviceBuffer<f32>) {
        match path {
            GenericStationaryPath::Csa4 => (&self.candidate_tail_kv, &self.candidate_tail_score),
            GenericStationaryPath::Hca128 => {
                (&self.candidate_hca_tail_kv, &self.candidate_hca_tail_score)
            }
        }
    }

    pub(crate) fn num_bytes(&self) -> usize {
        [
            self.hc_hidden.num_bytes(),
            self.freqs.num_bytes(),
            self.inverse_freqs.num_bytes(),
            self.compressed_freqs.num_bytes(),
            self.mixes.num_bytes(),
            self.pre.num_bytes(),
            self.post.num_bytes(),
            self.comb.num_bytes(),
            self.trace.num_bytes(),
            self.collapsed.num_bytes(),
            self.attn_input.num_bytes(),
            self.fp8_7168.num_bytes(),
            self.fp8_7168_scales.num_bytes(),
            self.fp8_1536.num_bytes(),
            self.fp8_1536_scales.num_bytes(),
            self.fp8_16384.num_bytes(),
            self.fp8_16384_scales.num_bytes(),
            self.empty_u8.num_bytes(),
            self.empty_u16.num_bytes(),
            self.q_lora_f32.num_bytes(),
            self.q_lora_raw.num_bytes(),
            self.q_lora.num_bytes(),
            self.q_f32.num_bytes(),
            self.q_raw.num_bytes(),
            self.q_norm.num_bytes(),
            self.q_rope.num_bytes(),
            self.kv_f32.num_bytes(),
            self.kv_raw.num_bytes(),
            self.kv_norm.num_bytes(),
            self.kv_rope.num_bytes(),
            self.kv_ready.num_bytes(),
            self.kv_scales.num_bytes(),
            self.candidate_tail_kv.num_bytes(),
            self.candidate_tail_score.num_bytes(),
            self.candidate_hca_tail_kv.num_bytes(),
            self.candidate_hca_tail_score.num_bytes(),
            self.compressor_kv.num_bytes(),
            self.compressor_score.num_bytes(),
            self.compressor_score_plus_ape.num_bytes(),
            self.hca_compressor_kv.num_bytes(),
            self.hca_compressor_score.num_bytes(),
            self.hca_compressor_score_plus_ape.num_bytes(),
            self.compressed_pool.num_bytes(),
            self.compressed_bf16.num_bytes(),
            self.compressed_norm.num_bytes(),
            self.compressed_rope.num_bytes(),
            self.compressed_ready.num_bytes(),
            self.compressed_scales.num_bytes(),
            self.indexer_pool.num_bytes(),
            self.indexer_bf16.num_bytes(),
            self.indexer_norm.num_bytes(),
            self.indexer_rope.num_bytes(),
            self.indexer_ready.num_bytes(),
            self.candidate_indexer_cache.num_bytes(),
            self.indexer_query_raw.num_bytes(),
            self.indexer_query_rope.num_bytes(),
            self.indexer_query_fp4.num_bytes(),
            self.indexer_query_f32.num_bytes(),
            self.indexer_weights_f32.num_bytes(),
            self.indexer_kv_f32.num_bytes(),
            self.indexer_scores.num_bytes(),
            self.indexer_valid.num_bytes(),
            self.indexer_selected.num_bytes(),
            self.topk.num_bytes(),
            self.attention_cache.num_bytes(),
            self.sparse.num_bytes(),
            self.sparse_scratch.num_bytes(),
            self.sparse_inverse.num_bytes(),
            self.wo_a.num_bytes(),
            self.attention_f32.num_bytes(),
            self.attention.num_bytes(),
            self.after_attention.num_bytes(),
        ]
        .into_iter()
        .sum::<usize>()
            + self
                .exact_compressor
                .as_ref()
                .map_or(0, GenericExactCompressorScratch::num_bytes)
            + self
                .indexer_compressor_kv
                .as_ref()
                .map_or(0, DeviceBuffer::num_bytes)
            + self
                .indexer_compressor_score
                .as_ref()
                .map_or(0, DeviceBuffer::num_bytes)
            + self
                .indexer_compressor_score_plus_ape
                .as_ref()
                .map_or(0, DeviceBuffer::num_bytes)
            + self
                .candidate_indexer_tail_kv
                .as_ref()
                .map_or(0, DeviceBuffer::num_bytes)
            + self
                .candidate_indexer_tail_score
                .as_ref()
                .map_or(0, DeviceBuffer::num_bytes)
    }

    pub(crate) fn byte_breakdown(&self) -> DeviceRequestScratchByteBreakdown {
        let sum = |values: &[usize]| values.iter().sum::<usize>();
        DeviceRequestScratchByteBreakdown {
            core_mhc: sum(&[
                self.hc_hidden.num_bytes(),
                self.collapsed.num_bytes(),
                self.attn_input.num_bytes(),
                self.freqs.num_bytes(),
                self.inverse_freqs.num_bytes(),
                self.compressed_freqs.num_bytes(),
                self.mixes.num_bytes(),
                self.pre.num_bytes(),
                self.post.num_bytes(),
                self.comb.num_bytes(),
                self.trace.num_bytes(),
            ]),
            projections: sum(&[
                self.fp8_7168.num_bytes(),
                self.fp8_7168_scales.num_bytes(),
                self.fp8_1536.num_bytes(),
                self.fp8_1536_scales.num_bytes(),
                self.fp8_16384.num_bytes(),
                self.fp8_16384_scales.num_bytes(),
                self.q_lora_f32.num_bytes(),
                self.q_lora_raw.num_bytes(),
                self.q_lora.num_bytes(),
                self.q_f32.num_bytes(),
                self.q_raw.num_bytes(),
                self.q_norm.num_bytes(),
                self.q_rope.num_bytes(),
                self.kv_f32.num_bytes(),
                self.kv_raw.num_bytes(),
                self.kv_norm.num_bytes(),
                self.kv_rope.num_bytes(),
                self.kv_ready.num_bytes(),
            ]),
            compressor: sum(&[
                self.candidate_tail_kv.num_bytes(),
                self.candidate_tail_score.num_bytes(),
                self.candidate_hca_tail_kv.num_bytes(),
                self.candidate_hca_tail_score.num_bytes(),
                self.compressor_kv.num_bytes(),
                self.compressor_score.num_bytes(),
                self.compressor_score_plus_ape.num_bytes(),
                self.hca_compressor_kv.num_bytes(),
                self.hca_compressor_score.num_bytes(),
                self.hca_compressor_score_plus_ape.num_bytes(),
                self.compressed_pool.num_bytes(),
                self.compressed_bf16.num_bytes(),
                self.compressed_norm.num_bytes(),
                self.compressed_rope.num_bytes(),
                self.compressed_ready.num_bytes(),
                self.compressed_scales.num_bytes(),
                self.kv_scales.num_bytes(),
            ]),
            exact_compressor: self
                .exact_compressor
                .as_ref()
                .map_or(0, GenericExactCompressorScratch::num_bytes),
            indexer: sum(&[
                self.indexer_compressor_kv
                    .as_ref()
                    .map_or(0, DeviceBuffer::num_bytes),
                self.indexer_compressor_score
                    .as_ref()
                    .map_or(0, DeviceBuffer::num_bytes),
                self.indexer_compressor_score_plus_ape
                    .as_ref()
                    .map_or(0, DeviceBuffer::num_bytes),
                self.candidate_indexer_tail_kv
                    .as_ref()
                    .map_or(0, DeviceBuffer::num_bytes),
                self.candidate_indexer_tail_score
                    .as_ref()
                    .map_or(0, DeviceBuffer::num_bytes),
                self.indexer_pool.num_bytes(),
                self.indexer_bf16.num_bytes(),
                self.indexer_norm.num_bytes(),
                self.indexer_rope.num_bytes(),
                self.indexer_ready.num_bytes(),
                self.candidate_indexer_cache.num_bytes(),
                self.indexer_query_raw.num_bytes(),
                self.indexer_query_rope.num_bytes(),
                self.indexer_query_fp4.num_bytes(),
                self.indexer_query_f32.num_bytes(),
                self.indexer_weights_f32.num_bytes(),
                self.indexer_kv_f32.num_bytes(),
                self.indexer_scores.num_bytes(),
                self.indexer_valid.num_bytes(),
                self.indexer_selected.num_bytes(),
            ]),
            topk_composite: self.topk.num_bytes() + self.attention_cache.num_bytes(),
            sparse: self.sparse.num_bytes() + self.sparse_inverse.num_bytes(),
            score_reuse: self.sparse_scratch.num_bytes(),
            attention_output: self.wo_a.num_bytes()
                + self.attention_f32.num_bytes()
                + self.attention.num_bytes()
                + self.after_attention.num_bytes(),
        }
    }
}

/// Numerically identical to `DeviceRequestRopeLut` row construction, but
/// avoids a request-length table.  Generic execution uploads exactly the row
/// it needs into the reusable scratch buffer.
pub(crate) fn upload_request_rope_row(
    stream: &CudaStream,
    position: usize,
    inverse: bool,
    destination: &DeviceBuffer<f32>,
) -> Result<(), Box<dyn Error>> {
    if destination.len() != v4_state::Q_ROPE_DIM {
        return Err("generic stationary RoPE destination geometry mismatch".into());
    }
    let mut row = v4_state::compressed_rope_freqs(position);
    if inverse {
        for sine in row.iter_mut().skip(1).step_by(2) {
            *sine = -*sine;
        }
    }
    // SAFETY: the checked host row and preallocated device destination have
    // exactly the same Q_ROPE_DIM element count.
    unsafe {
        cuda_core::memory::memcpy_htod_async(
            destination.cu_deviceptr(),
            row.as_ptr(),
            destination.num_bytes(),
            stream.cu_stream(),
        )?;
    }
    Ok(())
}

/// Run one generic stationary attention row against the request's real layer
/// store.  Every persistent write is delayed until after the terminal hidden
/// D2H succeeds, so an error leaves this layer's p counter and every other
/// layer store untouched.
pub(crate) struct GenericDeviceAttentionPrepared {
    pub(crate) attention: RealAttentionRun,
    pub(crate) position: usize,
    pub(crate) boundary: bool,
    pub(crate) compressed_before: usize,
    pub(crate) path: GenericStationaryPath,
}

pub(crate) fn prepare_generic_device_attention_row(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    runtime: &LoadedLayerRuntime,
    state: &mut DeviceLayerPersistentState,
    scratch: &mut DeviceRequestScratch,
    hc_hidden: &[u16],
    position: usize,
    audit_expected_compressed_prefix: Option<&[u16]>,
) -> Result<GenericDeviceAttentionPrepared, Box<dyn Error>> {
    let weights = &runtime.weights;
    let path = GenericStationaryPath::from_ratio(weights.compression_ratio);
    let csa = path.is_csa();
    let ratio = path.ratio();
    if state.layer != weights.layer
        || state.committed_rows != position
        || hc_hidden.len() != v4_state::HC * v4_state::HIDDEN
        || state.compressor_width != path.compressor_width()
        || state.compressor_tail_kv.len() != path.tail_elements()
        || state.compressor_tail_score.len() != path.tail_elements()
        || scratch.candidate_tails(path).0.len() != path.tail_elements()
        || scratch.candidate_tails(path).1.len() != path.tail_elements()
    {
        return Err("generic stationary device attention state contract mismatch".into());
    }
    let started = std::time::Instant::now();
    // SAFETY: one fixed host hidden row matches the single reusable scratch.
    unsafe {
        cuda_core::memory::memcpy_htod_async(
            scratch.hc_hidden.cu_deviceptr(),
            hc_hidden.as_ptr(),
            scratch.hc_hidden.num_bytes(),
            stream.cu_stream(),
        )?;
    }
    upload_request_rope_row(stream, position, false, &scratch.freqs)?;
    run_l0_mhc_pre_device_into(
        module,
        stream,
        &scratch.hc_hidden,
        &weights.hc_attn_fn,
        &weights.hc_attn_scale,
        &weights.hc_attn_base,
        &mut scratch.mixes,
        &mut scratch.pre,
        &mut scratch.post,
        &mut scratch.comb,
        &mut scratch.collapsed,
        &mut scratch.trace,
        None,
    )?;
    run_l0_rmsnorm_device_into(
        module,
        stream,
        &scratch.collapsed,
        &weights.attn_norm,
        &mut scratch.attn_input,
        v4_state::HIDDEN,
    )?;
    run_l0_fp8_linear_device_into(
        module,
        stream,
        &scratch.attn_input,
        &weights.wkv,
        &mut scratch.fp8_7168,
        &mut scratch.fp8_7168_scales,
        &mut scratch.empty_u16,
        &mut scratch.kv_f32,
        &mut scratch.kv_raw,
        false,
    )?;
    run_l0_rmsnorm_device_into(
        module,
        stream,
        &scratch.kv_raw,
        &weights.kv_norm,
        &mut scratch.kv_norm,
        v4_state::Q_HEAD_DIM,
    )?;
    run_rope_device_into(
        module,
        stream,
        &scratch.kv_norm,
        &scratch.freqs,
        &mut scratch.kv_rope,
        1,
        v4_state::Q_HEAD_DIM,
    )?;
    run_l0_kv_quantize_device_into(
        module,
        stream,
        &scratch.kv_rope,
        &mut scratch.empty_u8,
        &mut scratch.kv_scales,
        &mut scratch.kv_ready,
    )?;
    run_generic_main_compressor_device(module, stream, scratch, weights, path, position)?;
    // Candidate tails are complete scratch copies.  Persistent tails are not
    // touched until all subsequent attention work and terminal D2H succeed.
    unsafe {
        let (candidate_tail_kv, candidate_tail_score) = scratch.candidate_tails(path);
        copy_device_range(
            stream,
            candidate_tail_kv,
            0,
            &state.compressor_tail_kv,
            0,
            state.compressor_tail_kv.len(),
        )?;
        copy_device_range(
            stream,
            candidate_tail_score,
            0,
            &state.compressor_tail_score,
            0,
            state.compressor_tail_score.len(),
        )?;
        let compressor_slot = path.compressor_slot(position);
        let (compressor_kv, _, compressor_score_plus_ape) = scratch.main_compressor(path);
        copy_device_row(stream, candidate_tail_kv, compressor_slot, compressor_kv)?;
        copy_device_row(
            stream,
            candidate_tail_score,
            compressor_slot,
            compressor_score_plus_ape,
        )?;
    }
    let boundary = (position + 1).is_multiple_of(ratio);
    let compressed_before = state.compressed_rows(weights.compression_ratio);
    let compressed_after = compressed_before + usize::from(boundary);
    if compressed_after * v4_state::Q_HEAD_DIM > state.compressed_kv.len() {
        return Err("generic stationary compressed KV capacity exhausted".into());
    }
    let composite_rows = path.composite_capacity_rows(position);
    if composite_rows != path.topk_rows(compressed_after)
        || scratch.topk.len() < composite_rows
        || scratch.attention_cache.len()
            < composite_rows
                .checked_mul(v4_state::Q_HEAD_DIM)
                .ok_or("generic stationary composite cache capacity overflow")?
    {
        return Err("generic stationary composite cache geometry mismatch".into());
    }
    if boundary {
        run_generic_compressor_pool_device(module, stream, scratch, path)?;
        run_f32_to_bf16_device_into(
            module,
            stream,
            &scratch.compressed_pool,
            &mut scratch.compressed_bf16,
        )?;
        upload_request_rope_row(
            stream,
            position + 1 - ratio,
            false,
            &scratch.compressed_freqs,
        )?;
        run_l0_rmsnorm_device_into(
            module,
            stream,
            &scratch.compressed_bf16,
            &weights.compressor_norm,
            &mut scratch.compressed_norm,
            v4_state::Q_HEAD_DIM,
        )?;
        run_rope_device_into(
            module,
            stream,
            &scratch.compressed_norm,
            &scratch.compressed_freqs,
            &mut scratch.compressed_rope,
            1,
            v4_state::Q_HEAD_DIM,
        )?;
        run_l0_kv_quantize_device_into(
            module,
            stream,
            &scratch.compressed_rope,
            &mut scratch.empty_u8,
            &mut scratch.compressed_scales,
            &mut scratch.compressed_ready,
        )?;
        if let Some(expected_prefix) = audit_expected_compressed_prefix {
            let expected_elements = (compressed_before + 1)
                .checked_mul(v4_state::Q_HEAD_DIM)
                .ok_or("generic prepared boundary audit prefix overflow")?;
            if expected_prefix.len() != expected_elements {
                return Err("generic prepared boundary audit host prefix is incomplete".into());
            }
            let expected_row =
                &expected_prefix[compressed_before * v4_state::Q_HEAD_DIM..expected_elements];
            let actual_row = scratch.compressed_ready.to_host_vec(stream)?;
            check_compressed_boundary_bf16(
                "prepared_compressed_row",
                weights.layer,
                position,
                compressed_before,
                v4_state::Q_HEAD_DIM,
                expected_row,
                &actual_row,
            )?;
        }
        if csa {
            // Preserve the official CSA overlap for the next four rows, still
            // inside the candidate tail until the semantic commit below.
            unsafe {
                copy_device_range(
                    stream,
                    &scratch.candidate_tail_kv,
                    0,
                    &scratch.candidate_tail_kv,
                    4 * 1_024,
                    4 * 1_024,
                )?;
                copy_device_range(
                    stream,
                    &scratch.candidate_tail_score,
                    0,
                    &scratch.candidate_tail_score,
                    4 * 1_024,
                    4 * 1_024,
                )?;
            }
        }
    }
    if csa {
        let indexer = weights
            .indexer
            .as_ref()
            .ok_or("generic CSA indexer weights disappeared")?;
        run_generic_indexer_compressor_device(
            module,
            stream,
            scratch,
            weights.layer,
            indexer,
            position,
        )?;
        let (index_kv, _index_score, index_score_plus_ape, candidate_kv, candidate_score) = (
            scratch
                .indexer_compressor_kv
                .as_mut()
                .ok_or("generic CSA indexer scratch disappeared")?,
            scratch
                .indexer_compressor_score
                .as_mut()
                .ok_or("generic CSA indexer scratch disappeared")?,
            scratch
                .indexer_compressor_score_plus_ape
                .as_mut()
                .ok_or("generic CSA indexer scratch disappeared")?,
            scratch
                .candidate_indexer_tail_kv
                .as_mut()
                .ok_or("generic CSA candidate indexer tail disappeared")?,
            scratch
                .candidate_indexer_tail_score
                .as_mut()
                .ok_or("generic CSA candidate indexer tail disappeared")?,
        );
        let state_indexer_kv = state
            .indexer_tail_kv
            .as_ref()
            .ok_or("generic CSA persistent indexer tail disappeared")?;
        let state_indexer_score = state
            .indexer_tail_score
            .as_ref()
            .ok_or("generic CSA persistent indexer tail disappeared")?;
        unsafe {
            copy_device_range(
                stream,
                candidate_kv,
                0,
                state_indexer_kv,
                0,
                state_indexer_kv.len(),
            )?;
            copy_device_range(
                stream,
                candidate_score,
                0,
                state_indexer_score,
                0,
                state_indexer_score.len(),
            )?;
            copy_device_row(stream, candidate_kv, 4 + position % 4, index_kv)?;
            copy_device_row(
                stream,
                candidate_score,
                4 + position % 4,
                index_score_plus_ape,
            )?;
        }
        if boundary {
            run_compressor_pool_ratio4_device_into(
                module,
                stream,
                candidate_kv,
                candidate_score,
                &mut scratch.indexer_pool,
                128,
            )?;
            run_f32_to_bf16_device_into(
                module,
                stream,
                &scratch.indexer_pool,
                &mut scratch.indexer_bf16,
            )?;
            run_l0_rmsnorm_device_into(
                module,
                stream,
                &scratch.indexer_bf16,
                &indexer.compressor_norm,
                &mut scratch.indexer_norm,
                128,
            )?;
            run_rope_device_into(
                module,
                stream,
                &scratch.indexer_norm,
                &scratch.compressed_freqs,
                &mut scratch.indexer_rope,
                1,
                128,
            )?;
            run_indexer_fp4_sim_device_into(
                module,
                stream,
                &scratch.indexer_rope,
                &mut scratch.indexer_ready,
            )?;
            unsafe {
                copy_device_range(stream, candidate_kv, 0, candidate_kv, 4 * 256, 4 * 256)?;
                copy_device_range(
                    stream,
                    candidate_score,
                    0,
                    candidate_score,
                    4 * 256,
                    4 * 256,
                )?;
            }
        }
    }
    // Query projection and the composite raw/compressed cache are also scratch
    // only.  The raw ring keeps physical p % 128 indices, matching top-k.
    run_l0_fp8_linear_device_into(
        module,
        stream,
        &scratch.attn_input,
        &weights.wq_a,
        &mut scratch.fp8_7168,
        &mut scratch.fp8_7168_scales,
        &mut scratch.empty_u16,
        &mut scratch.q_lora_f32,
        &mut scratch.q_lora_raw,
        false,
    )?;
    run_l0_rmsnorm_device_into(
        module,
        stream,
        &scratch.q_lora_raw,
        &weights.q_norm,
        &mut scratch.q_lora,
        weights.wq_a.out_features,
    )?;
    run_l0_fp8_linear_device_into(
        module,
        stream,
        &scratch.q_lora,
        &weights.wq_b,
        &mut scratch.fp8_1536,
        &mut scratch.fp8_1536_scales,
        &mut scratch.empty_u16,
        &mut scratch.q_f32,
        &mut scratch.q_raw,
        true,
    )?;
    run_l0_rmsnorm_device_into(
        module,
        stream,
        &scratch.q_raw,
        &scratch.empty_u16,
        &mut scratch.q_norm,
        v4_state::Q_HEAD_DIM,
    )?;
    run_rope_device_into(
        module,
        stream,
        &scratch.q_norm,
        &scratch.freqs,
        &mut scratch.q_rope,
        v4_state::Q_HEADS,
        v4_state::Q_HEAD_DIM,
    )?;
    unsafe {
        copy_device_range(
            stream,
            &scratch.attention_cache,
            0,
            &state.raw_kv_ring,
            0,
            state.raw_kv_ring.len(),
        )?;
        copy_device_row(
            stream,
            &scratch.attention_cache,
            state.raw_slot(position)?,
            &scratch.kv_ready,
        )?;
        if compressed_before != 0 {
            copy_device_range(
                stream,
                &scratch.attention_cache,
                v4_state::KV_WINDOW * v4_state::Q_HEAD_DIM,
                &state.compressed_kv,
                0,
                compressed_before * v4_state::Q_HEAD_DIM,
            )?;
        }
        if boundary {
            copy_device_range(
                stream,
                &scratch.attention_cache,
                (v4_state::KV_WINDOW + compressed_before) * v4_state::Q_HEAD_DIM,
                &scratch.compressed_ready,
                0,
                v4_state::Q_HEAD_DIM,
            )?;
        }
    }
    let (window, _) = v4_state::window_topk_cpu(1, position, v4_state::KV_WINDOW)?;
    // SAFETY: the dynamic top-k buffer is at least 128 entries by construction.
    unsafe {
        cuda_core::memory::memcpy_htod_async(
            scratch.topk.cu_deviceptr(),
            window.as_ptr(),
            window.len() * std::mem::size_of::<i32>(),
            stream.cu_stream(),
        )?;
    }
    let raw_topk_per_row = window.len();
    let topk_per_row = if csa && compressed_after != 0 {
        let indexer = weights
            .indexer
            .as_ref()
            .ok_or("generic CSA indexer weights disappeared")?;
        run_l0_fp8_linear_device_into(
            module,
            stream,
            &scratch.q_lora,
            &indexer.wq_b,
            &mut scratch.fp8_1536,
            &mut scratch.fp8_1536_scales,
            &mut scratch.empty_u16,
            &mut scratch.indexer_query_f32,
            &mut scratch.indexer_query_raw,
            true,
        )?;
        run_rope_device_into(
            module,
            stream,
            &scratch.indexer_query_raw,
            &scratch.freqs,
            &mut scratch.indexer_query_rope,
            64,
            128,
        )?;
        run_indexer_fp4_sim_device_into(
            module,
            stream,
            &scratch.indexer_query_rope,
            &mut scratch.indexer_query_fp4,
        )?;
        launch_v4_bf16_to_f32(
            module,
            stream,
            &scratch.indexer_query_fp4,
            &mut scratch.indexer_query_f32,
        )?;
        run_bf16_linear_f32_device_into(
            module,
            stream,
            &scratch.attn_input,
            &indexer.weights_proj,
            &mut scratch.indexer_weights_f32,
            v4_state::HIDDEN,
            64,
        )?;
        run_scale_f32_device_in_place(
            module,
            stream,
            &mut scratch.indexer_weights_f32,
            1.0_f32 / (128.0_f32 * 64.0).sqrt(),
        )?;
        let state_indexer_kv = state
            .indexer_kv
            .as_ref()
            .ok_or("generic CSA persistent indexer cache disappeared")?;
        unsafe {
            if compressed_before != 0 {
                copy_device_range(
                    stream,
                    &scratch.candidate_indexer_cache,
                    0,
                    state_indexer_kv,
                    0,
                    compressed_before * 128,
                )?;
            }
            if boundary {
                copy_device_range(
                    stream,
                    &scratch.candidate_indexer_cache,
                    compressed_before * 128,
                    &scratch.indexer_ready,
                    0,
                    128,
                )?;
            }
        }
        launch_v4_bf16_to_f32(
            module,
            stream,
            &scratch.candidate_indexer_cache,
            &mut scratch.indexer_kv_f32,
        )?;
        let valid = u32::try_from(compressed_after)?;
        unsafe {
            cuda_core::memory::memcpy_htod_async(
                scratch.indexer_valid.cu_deviceptr(),
                &valid,
                std::mem::size_of::<u32>(),
                stream.cu_stream(),
            )?;
            module.indexer_scores_v4_f32(
                stream,
                LaunchConfig::for_num_elems(u32::try_from(compressed_after)?),
                &scratch.indexer_query_f32,
                &scratch.indexer_kv_f32,
                &scratch.indexer_weights_f32,
                &mut scratch.indexer_scores,
                1,
                64,
                128,
                u32::try_from(compressed_after)?,
            )?;
        }
        let selected = compressed_after.min(v4_state::INDEX_TOPK);
        unsafe {
            module.indexer_topk_v4_f32(
                stream,
                LaunchConfig::for_num_elems(u32::try_from(selected)?),
                &scratch.indexer_scores,
                &scratch.indexer_valid,
                &mut scratch.indexer_selected,
                1,
                u32::try_from(compressed_after)?,
                u32::try_from(selected)?,
                u32::try_from(v4_state::KV_WINDOW)?,
            )?;
            copy_device_range(
                stream,
                &scratch.topk,
                v4_state::KV_WINDOW,
                &scratch.indexer_selected,
                0,
                selected,
            )?;
        }
        v4_state::KV_WINDOW + selected
    } else if !csa && compressed_after != 0 {
        let topk = v4_state::build_hca_sparse_topk(1, position, v4_state::KV_WINDOW, ratio)?;
        // SAFETY: the HCA composite list fits the request-planned reusable top-k buffer.
        unsafe {
            cuda_core::memory::memcpy_htod_async(
                scratch.topk.cu_deviceptr(),
                topk.indices.as_ptr(),
                topk.indices.len() * std::mem::size_of::<i32>(),
                stream.cu_stream(),
            )?;
        }
        topk.per_row
    } else {
        raw_topk_per_row
    };
    if sparse_attention_arm_from_env()?.uses_score_reuse(topk_per_row) {
        run_sparse_attention_score_reuse_device_into(
            module,
            stream,
            &scratch.q_rope,
            &scratch.attention_cache,
            &weights.attn_sink,
            &scratch.topk,
            &mut scratch.sparse,
            &mut scratch.sparse_scratch,
            topk_per_row,
        )?;
    } else {
        run_sparse_attention_device_into(
            module,
            stream,
            &scratch.q_rope,
            &scratch.attention_cache,
            &weights.attn_sink,
            &scratch.topk,
            &mut scratch.sparse,
            topk_per_row,
        )?;
    }
    upload_request_rope_row(stream, position, true, &scratch.inverse_freqs)?;
    run_rope_device_into(
        module,
        stream,
        &scratch.sparse,
        &scratch.inverse_freqs,
        &mut scratch.sparse_inverse,
        v4_state::Q_HEADS,
        v4_state::Q_HEAD_DIM,
    )?;
    run_l0_wo_a_device_into(
        module,
        stream,
        &scratch.sparse_inverse,
        &weights.wo_a,
        &mut scratch.wo_a,
    )?;
    run_l0_fp8_linear_device_into(
        module,
        stream,
        &scratch.wo_a,
        &weights.wo_b,
        &mut scratch.fp8_16384,
        &mut scratch.fp8_16384_scales,
        &mut scratch.empty_u16,
        &mut scratch.attention_f32,
        &mut scratch.attention,
        false,
    )?;
    run_l0_hc_post_device_into(
        module,
        stream,
        &scratch.attention,
        &scratch.hc_hidden,
        &scratch.post,
        &scratch.comb,
        &mut scratch.after_attention,
    )?;
    let output = scratch.after_attention.to_host_vec(stream)?;

    Ok(GenericDeviceAttentionPrepared {
        attention: RealAttentionRun {
            output,
            topk: None,
            stage_digests: Vec::new(),
            attention_topk_digest: 0,
            attention_topk_per_row: topk_per_row,
            indexer_topk_gpu_ns: 0,
            attention_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            commit_ns: 0,
            device_transfer: Some(DeviceAttentionTransfer {
                h2d_count: 1,
                h2d_bytes: scratch.hc_hidden.num_bytes(),
                rope_lut_d2d_count: 0,
                rope_lut_d2d_bytes: 0,
                terminal_d2h_count: 1,
                terminal_d2h_bytes: scratch.after_attention.num_bytes(),
                intermediate_or_audit_d2h_count: 0,
                intermediate_or_audit_d2h_bytes: 0,
            }),
            profile: None,
        },
        position,
        boundary,
        compressed_before,
        path,
    })
}

/// Commit the prepared delta only after the caller's complete FFN/mHC layer
/// succeeds.  This is intentionally separate from attention preparation.
pub(crate) fn commit_generic_device_attention_row(
    state: &mut DeviceLayerPersistentState,
    scratch: &DeviceRequestScratch,
    prepared: GenericDeviceAttentionPrepared,
    stream: &CudaStream,
) -> Result<RealAttentionRun, Box<dyn Error>> {
    if state.committed_rows != prepared.position {
        return Err("generic stationary commit lost its causal layer boundary".into());
    }
    if state.compressor_tail_kv.len() != prepared.path.tail_elements()
        || state.compressor_tail_score.len() != prepared.path.tail_elements()
        || scratch.candidate_tails(prepared.path).0.len() != prepared.path.tail_elements()
        || scratch.candidate_tails(prepared.path).1.len() != prepared.path.tail_elements()
    {
        return Err("generic stationary commit scratch geometry mismatch".into());
    }
    // Atomic semantic commit: all fallible layer work completed before this.
    unsafe {
        copy_device_row(
            stream,
            &state.raw_kv_ring,
            state.raw_slot(prepared.position)?,
            &scratch.kv_ready,
        )?;
        let (candidate_tail_kv, candidate_tail_score) = scratch.candidate_tails(prepared.path);
        copy_device_range(
            stream,
            &state.compressor_tail_kv,
            0,
            candidate_tail_kv,
            0,
            state.compressor_tail_kv.len(),
        )?;
        copy_device_range(
            stream,
            &state.compressor_tail_score,
            0,
            candidate_tail_score,
            0,
            state.compressor_tail_score.len(),
        )?;
        if prepared.boundary {
            copy_device_range(
                stream,
                &state.compressed_kv,
                prepared.compressed_before * v4_state::Q_HEAD_DIM,
                &scratch.compressed_ready,
                0,
                v4_state::Q_HEAD_DIM,
            )?;
        }
        if prepared.path.is_csa() {
            let persistent_kv = state
                .indexer_tail_kv
                .as_ref()
                .ok_or("generic CSA persistent indexer tail disappeared")?;
            let persistent_score = state
                .indexer_tail_score
                .as_ref()
                .ok_or("generic CSA persistent indexer tail disappeared")?;
            let candidate_kv = scratch
                .candidate_indexer_tail_kv
                .as_ref()
                .ok_or("generic CSA candidate indexer tail disappeared")?;
            let candidate_score = scratch
                .candidate_indexer_tail_score
                .as_ref()
                .ok_or("generic CSA candidate indexer tail disappeared")?;
            copy_device_range(
                stream,
                persistent_kv,
                0,
                candidate_kv,
                0,
                persistent_kv.len(),
            )?;
            copy_device_range(
                stream,
                persistent_score,
                0,
                candidate_score,
                0,
                persistent_score.len(),
            )?;
            if prepared.boundary {
                copy_device_range(
                    stream,
                    state
                        .indexer_kv
                        .as_ref()
                        .ok_or("generic CSA persistent indexer cache disappeared")?,
                    prepared.compressed_before * 128,
                    &scratch.indexer_ready,
                    0,
                    128,
                )?;
            }
        }
    }
    state.committed_rows = state
        .committed_rows
        .checked_add(1)
        .ok_or("generic stationary committed-row overflow")?;
    Ok(prepared.attention)
}

pub(crate) fn zero_device_buffer<T>(
    stream: &CudaStream,
    buffer: &DeviceBuffer<T>,
) -> Result<(), Box<dyn Error>> {
    if !buffer.is_empty() {
        // SAFETY: the complete owned device range is reset in stream order.
        unsafe {
            cuda_core::memory::memset_d8_async(
                buffer.cu_deviceptr(),
                0,
                buffer.num_bytes(),
                stream.cu_stream(),
            )?;
        }
    }
    Ok(())
}

/// Generic stationary shares one scratch across HCA and CSA layers.  Strict
/// conversion buffers therefore retain one exact-sized allocation per fixed
/// projection geometry; a layer-0 HCA allocation cannot stand in for CSA.
pub(crate) struct GenericExactCompressorScratch {
    pub(crate) cublas: cublas_exact::CublasExact,
    pub(crate) workspace: DeviceBuffer<u8>,
    pub(crate) input: DeviceBuffer<f32>,
    pub(crate) hca_wkv: DeviceBuffer<f32>,
    pub(crate) hca_wgate: DeviceBuffer<f32>,
    pub(crate) csa_wkv: DeviceBuffer<f32>,
    pub(crate) csa_wgate: DeviceBuffer<f32>,
    pub(crate) csa_indexer_wkv: DeviceBuffer<f32>,
    pub(crate) csa_indexer_wgate: DeviceBuffer<f32>,
    // This scratch is request-local, so layer identity is sufficient: a
    // request transition constructs fresh scratch.  Each buffer is a
    // one-owner cache, never an all-61-layer cache.
    pub(crate) hca_owner: Option<usize>,
    pub(crate) csa_owner: Option<usize>,
    pub(crate) csa_indexer_owner: Option<usize>,
    pub(crate) active_layer: Option<usize>,
    pub(crate) main_weight_conversions: u64,
    pub(crate) indexer_weight_conversions: u64,
}

impl GenericExactCompressorScratch {
    pub(crate) fn new(stream: &CudaStream) -> Result<Self, Box<dyn Error>> {
        Self::new_for_geometry(
            stream,
            DeviceRequestScratchGeometry::new(
                v4_state::LayerStationaryRequestPlan::new(5, 3)?,
                true,
            )?,
        )
    }

    pub(crate) fn new_for_geometry(
        stream: &CudaStream,
        geometry: DeviceRequestScratchGeometry,
    ) -> Result<Self, Box<dyn Error>> {
        if !geometry.strict {
            return Err("strict generic Compressor scratch requires strict geometry".into());
        }
        let weights = |width: usize| {
            width
                .checked_mul(v4_state::HIDDEN)
                .ok_or("strict generic Compressor weight size overflow")
        };
        Ok(Self {
            cublas: cublas_exact::CublasExact::load()?,
            workspace: DeviceBuffer::zeroed(stream, cublas_exact::WORKSPACE_BYTES)?,
            input: DeviceBuffer::zeroed(stream, v4_state::HIDDEN)?,
            hca_wkv: DeviceBuffer::zeroed(stream, weights(geometry.generic.hca_main_elements)?)?,
            hca_wgate: DeviceBuffer::zeroed(stream, weights(geometry.generic.hca_main_elements)?)?,
            csa_wkv: DeviceBuffer::zeroed(stream, weights(geometry.generic.csa_main_elements)?)?,
            csa_wgate: DeviceBuffer::zeroed(stream, weights(geometry.generic.csa_main_elements)?)?,
            csa_indexer_wkv: DeviceBuffer::zeroed(
                stream,
                weights(geometry.generic.csa_indexer_elements)?,
            )?,
            csa_indexer_wgate: DeviceBuffer::zeroed(
                stream,
                weights(geometry.generic.csa_indexer_elements)?,
            )?,
            hca_owner: None,
            csa_owner: None,
            csa_indexer_owner: None,
            active_layer: None,
            main_weight_conversions: 0,
            indexer_weight_conversions: 0,
        })
    }

    pub(crate) fn audit_telemetry(&self) -> serde_json::Value {
        serde_json::json!({
            "main_weight_conversions": self.main_weight_conversions,
            "indexer_weight_conversions": self.indexer_weight_conversions,
            "hca_owner": self.hca_owner,
            "csa_owner": self.csa_owner,
            "csa_indexer_owner": self.csa_indexer_owner,
            "active_layer": self.active_layer,
        })
    }

    pub(crate) fn begin_main_refresh(&mut self, path: GenericStationaryPath, layer: usize) -> bool {
        self.begin_layer(layer);
        let owner = match path {
            GenericStationaryPath::Hca128 => &mut self.hca_owner,
            GenericStationaryPath::Csa4 => &mut self.csa_owner,
        };
        let refresh = *owner != Some(layer);
        if refresh {
            *owner = None;
        }
        refresh
    }

    pub(crate) fn finish_main_refresh(&mut self, path: GenericStationaryPath, layer: usize, refresh: bool) {
        self.active_layer = Some(layer);
        if refresh {
            match path {
                GenericStationaryPath::Hca128 => self.hca_owner = Some(layer),
                GenericStationaryPath::Csa4 => self.csa_owner = Some(layer),
            }
            self.main_weight_conversions = self.main_weight_conversions.saturating_add(1);
        }
    }

    pub(crate) fn begin_indexer_refresh(&mut self, layer: usize) -> bool {
        self.begin_layer(layer);
        let refresh = self.csa_indexer_owner != Some(layer);
        if refresh {
            self.csa_indexer_owner = None;
        }
        refresh
    }

    pub(crate) fn finish_indexer_refresh(&mut self, layer: usize, refresh: bool) {
        self.active_layer = Some(layer);
        if refresh {
            self.csa_indexer_owner = Some(layer);
            self.indexer_weight_conversions = self.indexer_weight_conversions.saturating_add(1);
        }
    }

    pub(crate) fn begin_layer(&mut self, layer: usize) {
        if self.active_layer != Some(layer) {
            self.invalidate();
        }
    }

    pub(crate) fn invalidate(&mut self) {
        self.hca_owner = None;
        self.csa_owner = None;
        self.csa_indexer_owner = None;
        self.active_layer = None;
    }

    pub(crate) fn num_bytes(&self) -> usize {
        self.workspace.num_bytes()
            + self.input.num_bytes()
            + self.hca_wkv.num_bytes()
            + self.hca_wgate.num_bytes()
            + self.csa_wkv.num_bytes()
            + self.csa_wgate.num_bytes()
            + self.csa_indexer_wkv.num_bytes()
            + self.csa_indexer_wgate.num_bytes()
    }
}

impl DeviceExactCompressorScratch {
    pub(crate) fn new(stream: &CudaStream, weights: &LayerDeviceWeights) -> Result<Self, Box<dyn Error>> {
        let width = weights.compressor_ape.len() / weights.compression_ratio.get();
        if width == 0
            || weights.compressor_wkv.len() != width * v4_state::HIDDEN
            || weights.compressor_wgate.len() != width * v4_state::HIDDEN
        {
            return Err("strict device Compressor workspace geometry mismatch".into());
        }
        Ok(Self {
            cublas: cublas_exact::CublasExact::load()?,
            workspace: DeviceBuffer::zeroed(stream, cublas_exact::WORKSPACE_BYTES)?,
            input: DeviceBuffer::zeroed(stream, v4_state::HIDDEN)?,
            wkv: DeviceBuffer::zeroed(stream, weights.compressor_wkv.len())?,
            wgate: DeviceBuffer::zeroed(stream, weights.compressor_wgate.len())?,
            indexer_wkv: weights
                .indexer
                .as_ref()
                .map(|indexer| DeviceBuffer::zeroed(stream, indexer.compressor_wkv.len()))
                .transpose()?,
            indexer_wgate: weights
                .indexer
                .as_ref()
                .map(|indexer| DeviceBuffer::zeroed(stream, indexer.compressor_wgate.len()))
                .transpose()?,
        })
    }

    pub(crate) fn num_bytes(&self) -> usize {
        self.workspace.num_bytes()
            + self.input.num_bytes()
            + self.wkv.num_bytes()
            + self.wgate.num_bytes()
            + self.indexer_wkv.as_ref().map_or(0, DeviceBuffer::num_bytes)
            + self
                .indexer_wgate
                .as_ref()
                .map_or(0, DeviceBuffer::num_bytes)
    }
}

impl DeviceTokenState {
    pub(crate) fn new(stream: &CudaStream, weights: &LayerDeviceWeights) -> Result<Self, Box<dyn Error>> {
        let csa = weights.compression_ratio == v4_state::CompressionRatio::Csa4;
        if weights.wq_a.out_features != V4_Q_LORA_RANK {
            return Err("device token-state Q-LoRA rank differs from its capacity ledger".into());
        }
        let compressor_width = if csa { 1_024 } else { v4_state::Q_HEAD_DIM };
        let compressor_rows = weights.compression_ratio.get() * (1 + usize::from(csa));
        let mut topk_csa = Vec::with_capacity(V4_SHORT_DEVICE_DECODE_MAX_ROWS - 3);
        for position in 3..V4_SHORT_DEVICE_DECODE_MAX_ROWS {
            let (mut window, _) = v4_state::window_topk_cpu(1, position, v4_state::KV_WINDOW)?;
            window.resize(v4_state::KV_WINDOW + csa_compressed_rows_after(position), 0);
            topk_csa.push(DeviceBuffer::from_host(stream, &window)?);
        }
        let mut topk_raw = Vec::with_capacity(V4_SHORT_DEVICE_DECODE_MAX_ROWS);
        for position in 0..V4_SHORT_DEVICE_DECODE_MAX_ROWS {
            let (window, _) = v4_state::window_topk_cpu(1, position, v4_state::KV_WINDOW)?;
            topk_raw.push(DeviceBuffer::from_host(stream, &window)?);
        }
        let state = Self {
            layer: weights.layer,
            committed_rows: 0,
            compressor_width,
            hc_hidden: DeviceBuffer::zeroed(stream, v4_state::HC * v4_state::HIDDEN)?,
            freqs: DeviceBuffer::zeroed(stream, v4_state::Q_ROPE_DIM)?,
            inverse_freqs: DeviceBuffer::zeroed(stream, v4_state::Q_ROPE_DIM)?,
            compressed_freqs: DeviceBuffer::zeroed(stream, v4_state::Q_ROPE_DIM)?,
            mixes: DeviceBuffer::zeroed(stream, v4_state::MIX)?,
            pre: DeviceBuffer::zeroed(stream, v4_state::HC)?,
            post: DeviceBuffer::zeroed(stream, v4_state::HC)?,
            comb: DeviceBuffer::zeroed(stream, v4_state::HC * v4_state::HC)?,
            trace: DeviceBuffer::zeroed(stream, 41 * v4_state::HC * v4_state::HC)?,
            collapsed: DeviceBuffer::zeroed(stream, v4_state::HIDDEN)?,
            attn_input: DeviceBuffer::zeroed(stream, v4_state::HIDDEN)?,
            fp8_7168: DeviceBuffer::zeroed(stream, v4_state::HIDDEN)?,
            fp8_7168_scales: DeviceBuffer::zeroed(stream, v4_state::HIDDEN / v4_state::FP8_BLOCK)?,
            fp8_1536: DeviceBuffer::zeroed(stream, weights.wq_a.out_features)?,
            fp8_1536_scales: DeviceBuffer::zeroed(
                stream,
                weights.wq_a.out_features / v4_state::FP8_BLOCK,
            )?,
            fp8_16384: DeviceBuffer::zeroed(stream, v4_state::O_GROUPS * v4_state::O_LORA_RANK)?,
            fp8_16384_scales: DeviceBuffer::zeroed(
                stream,
                v4_state::O_GROUPS * v4_state::O_LORA_RANK / v4_state::FP8_BLOCK,
            )?,
            empty_u8: DeviceBuffer::zeroed(stream, 0)?,
            empty_u16: DeviceBuffer::zeroed(stream, 0)?,
            q_lora_f32: DeviceBuffer::zeroed(stream, weights.wq_a.out_features)?,
            q_lora_raw: DeviceBuffer::zeroed(stream, weights.wq_a.out_features)?,
            q_lora: DeviceBuffer::zeroed(stream, weights.wq_a.out_features)?,
            q_f32: DeviceBuffer::zeroed(stream, weights.wq_b.out_features)?,
            q_raw: DeviceBuffer::zeroed(stream, weights.wq_b.out_features)?,
            q_norm: DeviceBuffer::zeroed(stream, weights.wq_b.out_features)?,
            q_rope: DeviceBuffer::zeroed(stream, weights.wq_b.out_features)?,
            kv_f32: DeviceBuffer::zeroed(stream, weights.wkv.out_features)?,
            kv_raw: DeviceBuffer::zeroed(stream, weights.wkv.out_features)?,
            kv_norm: DeviceBuffer::zeroed(stream, weights.wkv.out_features)?,
            kv_rope: DeviceBuffer::zeroed(stream, weights.wkv.out_features)?,
            kv_ready: DeviceBuffer::zeroed(stream, weights.wkv.out_features)?,
            kv_scales: DeviceBuffer::zeroed(stream, 7)?,
            kv_cache: DeviceBuffer::zeroed(stream, v4_state::KV_WINDOW * v4_state::Q_HEAD_DIM)?,
            compressor: DeviceCompressorRows::new(stream, compressor_width, compressor_rows)?,
            exact_compressor: strict_compressor_enabled()
                .then(|| DeviceExactCompressorScratch::new(stream, weights))
                .transpose()?,
            indexer_compressor: weights
                .indexer
                .as_ref()
                .map(|_| DeviceCompressorRows::new(stream, 256, 8))
                .transpose()?,
            compressed_pool: DeviceBuffer::zeroed(stream, v4_state::Q_HEAD_DIM)?,
            compressed_bf16: DeviceBuffer::zeroed(stream, v4_state::Q_HEAD_DIM)?,
            compressed_norm: DeviceBuffer::zeroed(stream, v4_state::Q_HEAD_DIM)?,
            compressed_rope: DeviceBuffer::zeroed(stream, v4_state::Q_HEAD_DIM)?,
            compressed_ready: DeviceBuffer::zeroed(stream, v4_state::Q_HEAD_DIM)?,
            compressed_scales: DeviceBuffer::zeroed(stream, 7)?,
            compressed_cache: DeviceBuffer::zeroed(stream, 2 * v4_state::Q_HEAD_DIM)?,
            indexer_pool: DeviceBuffer::zeroed(stream, 128)?,
            indexer_bf16: DeviceBuffer::zeroed(stream, 128)?,
            indexer_norm: DeviceBuffer::zeroed(stream, 128)?,
            indexer_rope: DeviceBuffer::zeroed(stream, 128)?,
            indexer_ready: DeviceBuffer::zeroed(stream, 128)?,
            indexer_cache: DeviceBuffer::zeroed(stream, 2 * 128)?,
            indexer_query_raw: DeviceBuffer::zeroed(stream, 64 * 128)?,
            indexer_query_rope: DeviceBuffer::zeroed(stream, 64 * 128)?,
            indexer_query_fp4: DeviceBuffer::zeroed(stream, 64 * 128)?,
            indexer_query_f32: DeviceBuffer::zeroed(stream, 64 * 128)?,
            indexer_weights_f32: DeviceBuffer::zeroed(stream, 64)?,
            indexer_kv_f32: DeviceBuffer::zeroed(stream, 2 * 128)?,
            indexer_scores: DeviceBuffer::zeroed(stream, 2)?,
            indexer_valid: DeviceBuffer::from_host(stream, &[1_u32; 2])?,
            indexer_selected: DeviceBuffer::zeroed(stream, 2)?,
            topk_csa,
            attention_cache: DeviceBuffer::zeroed(
                stream,
                (v4_state::KV_WINDOW + 2) * v4_state::Q_HEAD_DIM,
            )?,
            compressed_rows: 0,
            topk_raw,
            sparse: DeviceBuffer::zeroed(stream, v4_state::Q_HEADS * v4_state::Q_HEAD_DIM)?,
            sparse_pos0: SparseAttentionScoreReuseScratch::new(stream, 1, 1)?,
            sparse_pos1: SparseAttentionScoreReuseScratch::new(stream, 1, v4_state::KV_WINDOW)?,
            sparse_csa: SparseAttentionScoreReuseScratch::new(stream, 1, v4_state::KV_WINDOW + 2)?,
            sparse_inverse: DeviceBuffer::zeroed(stream, v4_state::Q_HEADS * v4_state::Q_HEAD_DIM)?,
            wo_a: DeviceBuffer::zeroed(stream, v4_state::O_GROUPS * v4_state::O_LORA_RANK)?,
            attention_f32: DeviceBuffer::zeroed(stream, v4_state::HIDDEN)?,
            attention: DeviceBuffer::zeroed(stream, v4_state::HIDDEN)?,
            after_attention: DeviceBuffer::zeroed(stream, v4_state::HC * v4_state::HIDDEN)?,
        };
        let planned = device_token_state_ledger_bytes(csa, weights.wq_a.out_features);
        if state.num_bytes() != planned {
            return Err(format!(
                "device token-state constructor bytes {} != shared ledger {planned}",
                state.num_bytes()
            )
            .into());
        }
        Ok(state)
    }

    pub(crate) fn reset_for_layer(
        &mut self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        weights: &LayerDeviceWeights,
    ) -> Result<usize, Box<dyn Error>> {
        let csa = weights.compression_ratio == v4_state::CompressionRatio::Csa4;
        if self.compressor_width != if csa { 1_024 } else { v4_state::Q_HEAD_DIM }
            || self.indexer_compressor.is_some() != csa
        {
            return Err("device token-state layout reuse geometry mismatch".into());
        }
        let mut reset_calls = 0;
        reset_calls += self.compressor.reset_persistent(module, stream)?;
        if csa {
            reset_calls += self
                .indexer_compressor
                .as_mut()
                .ok_or("CSA device token-state indexer disappeared")?
                .reset_persistent(module, stream)?;
        }
        // All other device buffers are either fully overwritten before first
        // use or are immutable tables initialized once with the layout state.
        self.layer = weights.layer;
        self.committed_rows = 0;
        self.compressed_rows = 0;
        Ok(reset_calls)
    }

    pub(crate) fn num_bytes(&self) -> usize {
        [
            self.hc_hidden.num_bytes(),
            self.freqs.num_bytes(),
            self.inverse_freqs.num_bytes(),
            self.compressed_freqs.num_bytes(),
            self.mixes.num_bytes(),
            self.pre.num_bytes(),
            self.post.num_bytes(),
            self.comb.num_bytes(),
            self.trace.num_bytes(),
            self.collapsed.num_bytes(),
            self.attn_input.num_bytes(),
            self.fp8_7168.num_bytes(),
            self.fp8_7168_scales.num_bytes(),
            self.fp8_1536.num_bytes(),
            self.fp8_1536_scales.num_bytes(),
            self.fp8_16384.num_bytes(),
            self.fp8_16384_scales.num_bytes(),
            self.empty_u8.num_bytes(),
            self.empty_u16.num_bytes(),
            self.q_lora_f32.num_bytes(),
            self.q_lora_raw.num_bytes(),
            self.q_lora.num_bytes(),
            self.q_f32.num_bytes(),
            self.q_raw.num_bytes(),
            self.q_norm.num_bytes(),
            self.q_rope.num_bytes(),
            self.kv_f32.num_bytes(),
            self.kv_raw.num_bytes(),
            self.kv_norm.num_bytes(),
            self.kv_rope.num_bytes(),
            self.kv_ready.num_bytes(),
            self.kv_scales.num_bytes(),
            self.kv_cache.num_bytes(),
            self.compressor.num_bytes(),
            self.indexer_compressor
                .as_ref()
                .map_or(0, DeviceCompressorRows::num_bytes),
            self.compressed_pool.num_bytes(),
            self.compressed_bf16.num_bytes(),
            self.compressed_norm.num_bytes(),
            self.compressed_rope.num_bytes(),
            self.compressed_ready.num_bytes(),
            self.compressed_scales.num_bytes(),
            self.compressed_cache.num_bytes(),
            self.indexer_pool.num_bytes(),
            self.indexer_bf16.num_bytes(),
            self.indexer_norm.num_bytes(),
            self.indexer_rope.num_bytes(),
            self.indexer_ready.num_bytes(),
            self.indexer_cache.num_bytes(),
            self.indexer_query_raw.num_bytes(),
            self.indexer_query_rope.num_bytes(),
            self.indexer_query_fp4.num_bytes(),
            self.indexer_query_f32.num_bytes(),
            self.indexer_weights_f32.num_bytes(),
            self.indexer_kv_f32.num_bytes(),
            self.indexer_scores.num_bytes(),
            self.indexer_valid.num_bytes(),
            self.indexer_selected.num_bytes(),
            self.attention_cache.num_bytes(),
            self.sparse.num_bytes(),
            self.sparse_pos0.num_bytes(),
            self.sparse_pos1.num_bytes(),
            self.sparse_csa.num_bytes(),
            self.sparse_inverse.num_bytes(),
            self.wo_a.num_bytes(),
            self.attention_f32.num_bytes(),
            self.attention.num_bytes(),
            self.after_attention.num_bytes(),
        ]
        .into_iter()
        .sum::<usize>()
            + self
                .topk_csa
                .iter()
                .map(DeviceBuffer::num_bytes)
                .sum::<usize>()
            + self
                .topk_raw
                .iter()
                .map(DeviceBuffer::num_bytes)
                .sum::<usize>()
            + self
                .exact_compressor
                .as_ref()
                .map_or(0, DeviceExactCompressorScratch::num_bytes)
    }

    pub(crate) fn exact_compressor_bytes(&self) -> usize {
        self.exact_compressor
            .as_ref()
            .map_or(0, DeviceExactCompressorScratch::num_bytes)
    }
}

#[derive(Debug)]
pub(crate) struct DeviceAttentionRun {
    pub(crate) output: Vec<u16>,
    pub(crate) elapsed_ns: u64,
    pub(crate) cuda_event_ns: u64,
    pub(crate) h2d_count: usize,
    pub(crate) h2d_bytes: usize,
    pub(crate) rope_lut_d2d_count: usize,
    pub(crate) rope_lut_d2d_bytes: usize,
    pub(crate) d2h_count: usize,
    pub(crate) d2h_bytes: usize,
    pub(crate) alloc_count: usize,
}

pub(crate) unsafe fn copy_device_row<T>(
    stream: &CudaStream,
    destination: &DeviceBuffer<T>,
    destination_row: usize,
    source: &DeviceBuffer<T>,
) -> Result<(), Box<dyn Error>> {
    let byte_offset = destination_row
        .checked_mul(source.num_bytes())
        .ok_or("device row copy offset overflow")?;
    if byte_offset + source.num_bytes() > destination.num_bytes() {
        return Err("device row copy exceeds persistent allocation".into());
    }
    // SAFETY: the checked byte range fits the destination and the source owns
    // exactly one complete row for the duration of the stream operation.
    unsafe {
        cuda_core::memory::memcpy_dtod_async(
            destination.cu_deviceptr() + u64::try_from(byte_offset)?,
            source.cu_deviceptr(),
            source.num_bytes(),
            stream.cu_stream(),
        )?;
    }
    Ok(())
}

pub(crate) unsafe fn copy_device_range<T>(
    stream: &CudaStream,
    destination: &DeviceBuffer<T>,
    destination_offset: usize,
    source: &DeviceBuffer<T>,
    source_offset: usize,
    elements: usize,
) -> Result<(), Box<dyn Error>> {
    let element_bytes = std::mem::size_of::<T>();
    let byte_len = elements
        .checked_mul(element_bytes)
        .ok_or("device range copy byte length overflow")?;
    let destination_byte_offset = destination_offset
        .checked_mul(element_bytes)
        .ok_or("device range copy destination offset overflow")?;
    let source_byte_offset = source_offset
        .checked_mul(element_bytes)
        .ok_or("device range copy source offset overflow")?;
    if destination_offset
        .checked_add(elements)
        .is_none_or(|end| end > destination.len())
        || source_offset
            .checked_add(elements)
            .is_none_or(|end| end > source.len())
    {
        return Err("device range copy exceeds allocation".into());
    }
    // SAFETY: the validated typed element ranges are disjoint device ranges.
    unsafe {
        cuda_core::memory::memcpy_dtod_async(
            destination.cu_deviceptr() + u64::try_from(destination_byte_offset)?,
            source.cu_deviceptr() + u64::try_from(source_byte_offset)?,
            byte_len,
            stream.cu_stream(),
        )?;
    }
    Ok(())
}

pub(crate) fn run_compressor_pool_ratio4_device_into(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    persistent_kv: &DeviceBuffer<f32>,
    persistent_score: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<f32>,
    head_dim: usize,
) -> Result<(), Box<dyn Error>> {
    if !matches!(head_dim, 128 | v4_state::Q_HEAD_DIM)
        || output.len() != head_dim
        || persistent_kv.len() != 16 * head_dim
        || persistent_score.len() != 16 * head_dim
    {
        return Err("device ratio-4 compressor pool workspace contract mismatch".into());
    }
    // SAFETY: one thread owns each output dimension; geometry matches the
    // overlap layout created by DeviceCompressorRows for CSA decode.
    unsafe {
        module.compressor_pool_ratio4_v4_f32(
            stream,
            LaunchConfig::for_num_elems(u32::try_from(head_dim)?),
            persistent_kv,
            persistent_score,
            output,
            u32::try_from(head_dim)?,
        )?;
    }
    Ok(())
}

pub(crate) fn run_compressor_pool_ratio128_device_into(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    persistent_kv: &DeviceBuffer<f32>,
    persistent_score: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<f32>,
) -> Result<(), Box<dyn Error>> {
    if output.len() != v4_state::Q_HEAD_DIM
        || persistent_kv.len() != v4_state::KV_WINDOW * v4_state::Q_HEAD_DIM
        || persistent_score.len() != v4_state::KV_WINDOW * v4_state::Q_HEAD_DIM
    {
        return Err("device ratio-128 compressor pool workspace contract mismatch".into());
    }
    // SAFETY: each launch thread owns one 512-wide output dimension and scans
    // the bounded 128-slot HCA state in canonical order.
    unsafe {
        module.compressor_pool_ratio128_v4_f32(
            stream,
            LaunchConfig::for_num_elems(u32::try_from(v4_state::Q_HEAD_DIM)?),
            persistent_kv,
            persistent_score,
            output,
            u32::try_from(v4_state::Q_HEAD_DIM)?,
        )?;
    }
    Ok(())
}

pub(crate) fn run_f32_to_bf16_device_into(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    input: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<u16>,
) -> Result<(), Box<dyn Error>> {
    if input.is_empty() || output.len() != input.len() {
        return Err("device f32-to-BF16 workspace contract mismatch".into());
    }
    // SAFETY: the kernel writes one checked output element per input element.
    unsafe {
        module.f32_to_bf16_rne_v4(
            stream,
            LaunchConfig::for_num_elems(u32::try_from(input.len())?),
            input,
            output,
        )?;
    }
    Ok(())
}

pub(crate) fn run_indexer_fp4_sim_device_into(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    input: &DeviceBuffer<u16>,
    output: &mut DeviceBuffer<u16>,
) -> Result<(), Box<dyn Error>> {
    if input.is_empty() || !input.len().is_multiple_of(128) || output.len() != input.len() {
        return Err("device CSA indexer FP4 simulation workspace contract mismatch".into());
    }
    // SAFETY: one thread owns each complete 128-wide indexer vector.
    unsafe {
        module.indexer_fp4_sim_v4_bf16(
            stream,
            LaunchConfig::for_num_elems(u32::try_from(input.len() / 128)?),
            input,
            output,
        )?;
    }
    Ok(())
}

pub(crate) fn run_bf16_linear_f32_device_into(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    input: &DeviceBuffer<u16>,
    weight: &DeviceBuffer<u16>,
    output: &mut DeviceBuffer<f32>,
    in_features: usize,
    out_features: usize,
) -> Result<(), Box<dyn Error>> {
    if input.is_empty()
        || !input.len().is_multiple_of(in_features)
        || weight.len() != in_features * out_features
        || output.len() != input.len() / in_features * out_features
    {
        return Err("device BF16 linear FP32 workspace contract mismatch".into());
    }
    // SAFETY: one thread owns one checked row-major output element.
    unsafe {
        module.linear_v4_bf16_f32(
            stream,
            LaunchConfig::for_num_elems(u32::try_from(output.len())?),
            input,
            weight,
            output,
            u32::try_from(in_features)?,
            u32::try_from(out_features)?,
        )?;
    }
    Ok(())
}

pub(crate) fn run_scale_f32_device_in_place(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    values: &mut DeviceBuffer<f32>,
    scale: f32,
) -> Result<(), Box<dyn Error>> {
    if values.is_empty() || !scale.is_finite() {
        return Err("device FP32 scale workspace contract mismatch".into());
    }
    // SAFETY: each thread updates one disjoint checked FP32 value.
    unsafe {
        module.scale_f32_in_place_v4(
            stream,
            LaunchConfig::for_num_elems(u32::try_from(values.len())?),
            values,
            scale,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_exact_compressor_device_into(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    input: &DeviceBuffer<u16>,
    wkv: &DeviceBuffer<u16>,
    wgate: &DeviceBuffer<u16>,
    ape: &DeviceBuffer<f32>,
    compressor: &mut DeviceCompressorRows,
    cublas: &cublas_exact::CublasExact,
    workspace: &DeviceBuffer<u8>,
    input_f32: &mut DeviceBuffer<f32>,
    wkv_f32: &mut DeviceBuffer<f32>,
    wgate_f32: &mut DeviceBuffer<f32>,
    ratio: usize,
    start_pos: usize,
) -> Result<(), Box<dyn Error>> {
    run_exact_compressor_buffers_device_into(
        module,
        stream,
        input,
        wkv,
        wgate,
        ape,
        &mut compressor.kv,
        &mut compressor.score,
        &mut compressor.score_plus_ape,
        cublas,
        workspace,
        input_f32,
        wkv_f32,
        wgate_f32,
        ratio,
        start_pos,
        start_pos == 0,
    )
}

/// Strict projection shared by the short device fixture and generic stationary
/// path.  Both persist the same three FP32 row buffers, so the pinned cuBLAS
/// reduction must be selected before either path writes its rolling tail.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_exact_compressor_buffers_device_into(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    input: &DeviceBuffer<u16>,
    wkv: &DeviceBuffer<u16>,
    wgate: &DeviceBuffer<u16>,
    ape: &DeviceBuffer<f32>,
    kv: &mut DeviceBuffer<f32>,
    score: &mut DeviceBuffer<f32>,
    score_plus_ape: &mut DeviceBuffer<f32>,
    cublas: &cublas_exact::CublasExact,
    workspace: &DeviceBuffer<u8>,
    input_f32: &mut DeviceBuffer<f32>,
    wkv_f32: &mut DeviceBuffer<f32>,
    wgate_f32: &mut DeviceBuffer<f32>,
    ratio: usize,
    start_pos: usize,
    refresh_weights: bool,
) -> Result<(), Box<dyn Error>> {
    let outputs = kv.len();
    if input.len() != v4_state::HIDDEN
        || wkv.len() != outputs * v4_state::HIDDEN
        || wgate.len() != wkv.len()
        || ape.len() != ratio * outputs
        || score.len() != outputs
        || score_plus_ape.len() != outputs
        || input_f32.len() != input.len()
        || wkv_f32.len() != wkv.len()
        || wgate_f32.len() != wgate.len()
    {
        return Err("strict device Compressor launch geometry mismatch".into());
    }
    launch_v4_bf16_to_f32(module, stream, input, input_f32)?;
    if refresh_weights {
        launch_v4_bf16_to_f32(module, stream, wkv, wkv_f32)?;
        launch_v4_bf16_to_f32(module, stream, wgate, wgate_f32)?;
    }
    let device_const = |pointer| pointer as usize as *const std::ffi::c_void;
    let device_mut = |pointer| pointer as usize as *mut std::ffi::c_void;
    cublas.sgemm_mn1(
        stream.cu_stream().cast::<std::ffi::c_void>(),
        device_mut(workspace.cu_deviceptr()),
        workspace.num_bytes(),
        device_const(wkv_f32.cu_deviceptr()),
        device_const(input_f32.cu_deviceptr()),
        device_mut(kv.cu_deviceptr()),
        outputs,
        v4_state::HIDDEN,
    )?;
    cublas.sgemm_mn1(
        stream.cu_stream().cast::<std::ffi::c_void>(),
        device_mut(workspace.cu_deviceptr()),
        workspace.num_bytes(),
        device_const(wgate_f32.cu_deviceptr()),
        device_const(input_f32.cu_deviceptr()),
        device_mut(score.cu_deviceptr()),
        outputs,
        v4_state::HIDDEN,
    )?;
    let ape_offset = (start_pos % ratio)
        .checked_mul(outputs)
        .ok_or("strict device Compressor APE offset overflow")?;
    unsafe {
        copy_device_range(stream, score_plus_ape, 0, ape, ape_offset, outputs)?;
        module.add_in_place(
            stream,
            LaunchConfig::for_num_elems(u32::try_from(outputs)?),
            score_plus_ape,
            score,
        )?;
    }
    Ok(())
}

/// Candidate-only persistent-state update for the first CSA boundary.  This
/// deliberately stops before sparse attention: exact commit parity is the
/// gate, while pos3's CSA query/top-k path remains a separate follow-up.
pub(crate) fn run_device_token_state_commit_row(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    runtime: &LoadedLayerRuntime,
    state: &mut DeviceTokenState,
    rope_lut: &DeviceRequestRopeLut,
    hc_hidden: &[u16],
    start_pos: usize,
) -> Result<usize, Box<dyn Error>> {
    let weights = &runtime.weights;
    if weights.layer >= v4_state::V4_LAYER_COUNT
        || state.layer != weights.layer
        || start_pos != state.committed_rows
        || start_pos >= V4_SHORT_DEVICE_DECODE_MAX_ROWS
        || hc_hidden.len() != v4_state::HC * v4_state::HIDDEN
    {
        return Err("device token-state commit row contract mismatch".into());
    }
    // SAFETY: each fixed host row exactly matches its preallocated destination.
    unsafe {
        cuda_core::memory::memcpy_htod_async(
            state.hc_hidden.cu_deviceptr(),
            hc_hidden.as_ptr(),
            state.hc_hidden.num_bytes(),
            stream.cu_stream(),
        )?;
    }
    rope_lut.copy_forward_row(stream, start_pos, &state.freqs)?;
    run_l0_mhc_pre_device_into(
        module,
        stream,
        &state.hc_hidden,
        &weights.hc_attn_fn,
        &weights.hc_attn_scale,
        &weights.hc_attn_base,
        &mut state.mixes,
        &mut state.pre,
        &mut state.post,
        &mut state.comb,
        &mut state.collapsed,
        &mut state.trace,
        None,
    )?;
    run_l0_rmsnorm_device_into(
        module,
        stream,
        &state.collapsed,
        &weights.attn_norm,
        &mut state.attn_input,
        v4_state::HIDDEN,
    )?;
    run_l0_fp8_linear_device_into(
        module,
        stream,
        &state.attn_input,
        &weights.wkv,
        &mut state.fp8_7168,
        &mut state.fp8_7168_scales,
        &mut state.empty_u16,
        &mut state.kv_f32,
        &mut state.kv_raw,
        false,
    )?;
    run_l0_rmsnorm_device_into(
        module,
        stream,
        &state.kv_raw,
        &weights.kv_norm,
        &mut state.kv_norm,
        v4_state::Q_HEAD_DIM,
    )?;
    run_rope_device_into(
        module,
        stream,
        &state.kv_norm,
        &state.freqs,
        &mut state.kv_rope,
        1,
        v4_state::Q_HEAD_DIM,
    )?;
    run_l0_kv_quantize_device_into(
        module,
        stream,
        &state.kv_rope,
        &mut state.empty_u8,
        &mut state.kv_scales,
        &mut state.kv_ready,
    )?;
    if let Some(scratch) = state.exact_compressor.as_mut() {
        let DeviceExactCompressorScratch {
            cublas,
            workspace,
            input,
            wkv,
            wgate,
            ..
        } = scratch;
        run_exact_compressor_device_into(
            module,
            stream,
            &state.attn_input,
            &weights.compressor_wkv,
            &weights.compressor_wgate,
            &weights.compressor_ape,
            &mut state.compressor,
            cublas,
            workspace,
            input,
            wkv,
            wgate,
            weights.compression_ratio.get(),
            start_pos,
        )?;
    } else {
        run_compressor_device_into(
            module,
            stream,
            &state.attn_input,
            &weights.compressor_wkv,
            &weights.compressor_wgate,
            &weights.compressor_ape,
            &mut state.compressor.kv,
            &mut state.compressor.score,
            &mut state.compressor.score_plus_ape,
            state.compressor_width,
            weights.compression_ratio.get(),
            start_pos,
        )?;
    }
    if l0_pos0_compressor_probe_enabled(weights.layer, start_pos) {
        let kv = state.compressor.kv.to_host_vec(stream)?;
        let score = state.compressor.score.to_host_vec(stream)?;
        let score_plus_ape = state.compressor.score_plus_ape.to_host_vec(stream)?;
        print_l0_pos0_compressor_probe("device_kernel_pre_commit", &kv, &score, &score_plus_ape);
    }
    if let (Some(indexer), Some(indexer_state)) =
        (weights.indexer.as_ref(), state.indexer_compressor.as_mut())
    {
        if let Some(scratch) = state.exact_compressor.as_mut() {
            let DeviceExactCompressorScratch {
                cublas,
                workspace,
                input,
                indexer_wkv,
                indexer_wgate,
                ..
            } = scratch;
            run_exact_compressor_device_into(
                module,
                stream,
                &state.attn_input,
                &indexer.compressor_wkv,
                &indexer.compressor_wgate,
                &indexer.compressor_ape,
                indexer_state,
                cublas,
                workspace,
                input,
                indexer_wkv
                    .as_mut()
                    .ok_or("strict device indexer Compressor workspace disappeared")?,
                indexer_wgate
                    .as_mut()
                    .ok_or("strict device indexer Compressor workspace disappeared")?,
                4,
                start_pos,
            )?;
        } else {
            run_compressor_device_into(
                module,
                stream,
                &state.attn_input,
                &indexer.compressor_wkv,
                &indexer.compressor_wgate,
                &indexer.compressor_ape,
                &mut indexer_state.kv,
                &mut indexer_state.score,
                &mut indexer_state.score_plus_ape,
                256,
                4,
                start_pos,
            )?;
        }
    }
    unsafe {
        copy_device_row(stream, &state.kv_cache, start_pos, &state.kv_ready)?;
        let compressor_slot = if weights.compression_ratio == v4_state::CompressionRatio::Csa4 {
            4 + start_pos % 4
        } else {
            start_pos % v4_state::KV_WINDOW
        };
        copy_device_row(
            stream,
            &state.compressor.persistent_kv,
            compressor_slot,
            &state.compressor.kv,
        )?;
        copy_device_row(
            stream,
            &state.compressor.persistent_score,
            compressor_slot,
            &state.compressor.score_plus_ape,
        )?;
        if let Some(indexer_state) = state.indexer_compressor.as_ref() {
            copy_device_row(
                stream,
                &indexer_state.persistent_kv,
                4 + start_pos % 4,
                &indexer_state.kv,
            )?;
            copy_device_row(
                stream,
                &indexer_state.persistent_score,
                4 + start_pos % 4,
                &indexer_state.score_plus_ape,
            )?;
        }
    }
    let csa_boundary = weights.compression_ratio == v4_state::CompressionRatio::Csa4
        && (start_pos + 1).is_multiple_of(4);
    if csa_boundary {
        if state.compressed_rows >= 2 {
            return Err("short CSA decode exceeds its two-compressed-entry ceiling".into());
        }
        run_compressor_pool_ratio4_device_into(
            module,
            stream,
            &state.compressor.persistent_kv,
            &state.compressor.persistent_score,
            &mut state.compressed_pool,
            v4_state::Q_HEAD_DIM,
        )?;
        run_f32_to_bf16_device_into(
            module,
            stream,
            &state.compressed_pool,
            &mut state.compressed_bf16,
        )?;
        rope_lut.copy_forward_row(
            stream,
            start_pos + 1 - weights.compression_ratio.get(),
            &state.compressed_freqs,
        )?;
        run_l0_rmsnorm_device_into(
            module,
            stream,
            &state.compressed_bf16,
            &weights.compressor_norm,
            &mut state.compressed_norm,
            v4_state::Q_HEAD_DIM,
        )?;
        run_rope_device_into(
            module,
            stream,
            &state.compressed_norm,
            &state.compressed_freqs,
            &mut state.compressed_rope,
            1,
            v4_state::Q_HEAD_DIM,
        )?;
        run_l0_kv_quantize_device_into(
            module,
            stream,
            &state.compressed_rope,
            &mut state.empty_u8,
            &mut state.compressed_scales,
            &mut state.compressed_ready,
        )?;
        let indexer_weights = weights
            .indexer
            .as_ref()
            .ok_or("CSA indexer weights disappeared at pool boundary")?;
        let indexer_state = state
            .indexer_compressor
            .as_ref()
            .ok_or("CSA indexer candidate state disappeared at pool boundary")?;
        run_compressor_pool_ratio4_device_into(
            module,
            stream,
            &indexer_state.persistent_kv,
            &indexer_state.persistent_score,
            &mut state.indexer_pool,
            128,
        )?;
        run_f32_to_bf16_device_into(module, stream, &state.indexer_pool, &mut state.indexer_bf16)?;
        run_l0_rmsnorm_device_into(
            module,
            stream,
            &state.indexer_bf16,
            &indexer_weights.compressor_norm,
            &mut state.indexer_norm,
            128,
        )?;
        run_rope_device_into(
            module,
            stream,
            &state.indexer_norm,
            &state.compressed_freqs,
            &mut state.indexer_rope,
            1,
            128,
        )?;
        run_indexer_fp4_sim_device_into(
            module,
            stream,
            &state.indexer_rope,
            &mut state.indexer_ready,
        )?;
        unsafe {
            copy_device_range(
                stream,
                &state.compressed_cache,
                state.compressed_rows * v4_state::Q_HEAD_DIM,
                &state.compressed_ready,
                0,
                v4_state::Q_HEAD_DIM,
            )?;
            copy_device_range(
                stream,
                &state.indexer_cache,
                state.compressed_rows * 128,
                &state.indexer_ready,
                0,
                128,
            )?;
            copy_device_range(
                stream,
                &state.compressor.persistent_kv,
                0,
                &state.compressor.persistent_kv,
                4 * 1_024,
                4 * 1_024,
            )?;
            copy_device_range(
                stream,
                &state.compressor.persistent_score,
                0,
                &state.compressor.persistent_score,
                4 * 1_024,
                4 * 1_024,
            )?;
            let indexer_state = state
                .indexer_compressor
                .as_ref()
                .ok_or("CSA indexer candidate state disappeared during overlap copy")?;
            copy_device_range(
                stream,
                &indexer_state.persistent_kv,
                0,
                &indexer_state.persistent_kv,
                4 * 256,
                4 * 256,
            )?;
            copy_device_range(
                stream,
                &indexer_state.persistent_score,
                0,
                &indexer_state.persistent_score,
                4 * 256,
                4 * 256,
            )?;
        }
        state.compressed_rows += 1;
    }
    state.committed_rows += 1;
    Ok(1 + usize::from(csa_boundary))
}

/// Complete one attention row after its device-authoritative state commit for
/// both pinned compression layouts in the five-token stationary gate.
pub(crate) fn run_device_attention_after_commit(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    runtime: &LoadedLayerRuntime,
    state: &mut DeviceTokenState,
    rope_lut: &DeviceRequestRopeLut,
    position: usize,
    observe_topk: bool,
) -> Result<(Vec<u16>, Option<Vec<i32>>), Box<dyn Error>> {
    let weights = &runtime.weights;
    let csa_compressed_rows = if weights.compression_ratio == v4_state::CompressionRatio::Csa4 {
        csa_compressed_rows_after(position)
    } else {
        0
    };
    let csa_boundary = csa_compressed_rows != 0;
    if weights.layer >= v4_state::V4_LAYER_COUNT
        || state.layer != weights.layer
        || state.committed_rows != position + 1
        || state.compressed_rows != csa_compressed_rows
        || csa_boundary && state.indexer_compressor.is_none()
        || position >= V4_SHORT_DEVICE_DECODE_MAX_ROWS
    {
        return Err("device attention-after-commit contract mismatch".into());
    }
    run_l0_fp8_linear_device_into(
        module,
        stream,
        &state.attn_input,
        &weights.wq_a,
        &mut state.fp8_7168,
        &mut state.fp8_7168_scales,
        &mut state.empty_u16,
        &mut state.q_lora_f32,
        &mut state.q_lora_raw,
        false,
    )?;
    run_l0_rmsnorm_device_into(
        module,
        stream,
        &state.q_lora_raw,
        &weights.q_norm,
        &mut state.q_lora,
        weights.wq_a.out_features,
    )?;
    run_l0_fp8_linear_device_into(
        module,
        stream,
        &state.q_lora,
        &weights.wq_b,
        &mut state.fp8_1536,
        &mut state.fp8_1536_scales,
        &mut state.empty_u16,
        &mut state.q_f32,
        &mut state.q_raw,
        true,
    )?;
    run_l0_rmsnorm_device_into(
        module,
        stream,
        &state.q_raw,
        &state.empty_u16,
        &mut state.q_norm,
        v4_state::Q_HEAD_DIM,
    )?;
    run_rope_device_into(
        module,
        stream,
        &state.q_norm,
        &state.freqs,
        &mut state.q_rope,
        v4_state::Q_HEADS,
        v4_state::Q_HEAD_DIM,
    )?;
    if csa_boundary {
        let indexer = weights
            .indexer
            .as_ref()
            .ok_or("device CSA indexer weights disappeared")?;
        run_l0_fp8_linear_device_into(
            module,
            stream,
            &state.q_lora,
            &indexer.wq_b,
            &mut state.fp8_1536,
            &mut state.fp8_1536_scales,
            &mut state.empty_u16,
            &mut state.indexer_query_f32,
            &mut state.indexer_query_raw,
            true,
        )?;
        run_rope_device_into(
            module,
            stream,
            &state.indexer_query_raw,
            &state.freqs,
            &mut state.indexer_query_rope,
            64,
            128,
        )?;
        run_indexer_fp4_sim_device_into(
            module,
            stream,
            &state.indexer_query_rope,
            &mut state.indexer_query_fp4,
        )?;
        launch_v4_bf16_to_f32(
            module,
            stream,
            &state.indexer_query_fp4,
            &mut state.indexer_query_f32,
        )?;
        run_bf16_linear_f32_device_into(
            module,
            stream,
            &state.attn_input,
            &indexer.weights_proj,
            &mut state.indexer_weights_f32,
            v4_state::HIDDEN,
            64,
        )?;
        run_scale_f32_device_in_place(
            module,
            stream,
            &mut state.indexer_weights_f32,
            1.0_f32 / (128.0_f32 * 64.0).sqrt(),
        )?;
        launch_v4_bf16_to_f32(
            module,
            stream,
            &state.indexer_cache,
            &mut state.indexer_kv_f32,
        )?;
    }
    // SAFETY: all copies fit their preallocated cache; CSA contributes its
    // single first-boundary key only after that boundary has committed.
    unsafe {
        copy_device_range(
            stream,
            &state.attention_cache,
            0,
            &state.kv_cache,
            0,
            v4_state::KV_WINDOW * v4_state::Q_HEAD_DIM,
        )?;
        if csa_boundary {
            module.indexer_scores_v4_f32(
                stream,
                LaunchConfig::for_num_elems(u32::try_from(csa_compressed_rows)?),
                &state.indexer_query_f32,
                &state.indexer_kv_f32,
                &state.indexer_weights_f32,
                &mut state.indexer_scores,
                1,
                64,
                128,
                u32::try_from(csa_compressed_rows)?,
            )?;
            module.indexer_topk_v4_f32(
                stream,
                LaunchConfig::for_num_elems(u32::try_from(csa_compressed_rows)?),
                &state.indexer_scores,
                &state.indexer_valid,
                &mut state.indexer_selected,
                1,
                u32::try_from(csa_compressed_rows)?,
                u32::try_from(csa_compressed_rows)?,
                u32::try_from(v4_state::KV_WINDOW)?,
            )?;
            let topk = state
                .topk_csa
                .get(
                    position
                        .checked_sub(3)
                        .ok_or("device CSA top-k position underflow")?,
                )
                .ok_or("device CSA top-k position is outside short-decode gate")?;
            let indexer_topk = csa_compressed_rows.min(v4_state::INDEX_TOPK);
            copy_device_range(
                stream,
                topk,
                v4_state::KV_WINDOW,
                &state.indexer_selected,
                0,
                indexer_topk,
            )?;
            copy_device_range(
                stream,
                &state.attention_cache,
                v4_state::KV_WINDOW * v4_state::Q_HEAD_DIM,
                &state.compressed_cache,
                0,
                csa_compressed_rows * v4_state::Q_HEAD_DIM,
            )?;
        }
    }
    let raw_topk = state
        .topk_raw
        .get(position)
        .ok_or("device raw top-k position is outside short-decode gate")?;
    let topk = if csa_boundary {
        state
            .topk_csa
            .get(position - 3)
            .ok_or("device CSA sparse position is outside short-decode gate")?
    } else {
        raw_topk
    };
    let sparse_scratch = if csa_boundary {
        &mut state.sparse_csa
    } else if position == 0 {
        &mut state.sparse_pos0
    } else {
        &mut state.sparse_pos1
    };
    run_device_raw_window_attention(
        module,
        stream,
        &state.q_rope,
        &state.attention_cache,
        &weights.attn_sink,
        topk,
        &mut state.sparse,
        sparse_scratch,
    )?;
    rope_lut.copy_inverse_row(stream, position, &state.inverse_freqs)?;
    run_rope_device_into(
        module,
        stream,
        &state.sparse,
        &state.inverse_freqs,
        &mut state.sparse_inverse,
        v4_state::Q_HEADS,
        v4_state::Q_HEAD_DIM,
    )?;
    run_l0_wo_a_device_into(
        module,
        stream,
        &state.sparse_inverse,
        &weights.wo_a,
        &mut state.wo_a,
    )?;
    run_l0_fp8_linear_device_into(
        module,
        stream,
        &state.wo_a,
        &weights.wo_b,
        &mut state.fp8_16384,
        &mut state.fp8_16384_scales,
        &mut state.empty_u16,
        &mut state.attention_f32,
        &mut state.attention,
        false,
    )?;
    run_l0_hc_post_device_into(
        module,
        stream,
        &state.attention,
        &state.hc_hidden,
        &state.post,
        &state.comb,
        &mut state.after_attention,
    )?;
    Ok((
        state.after_attention.to_host_vec(stream)?,
        observe_topk.then(|| topk.to_host_vec(stream)).transpose()?,
    ))
}

/// Shared raw-window sparse-attention launch.  HCA/CSA still choose and fill
/// their own cache/top-k state above; MTP supplies its isolated ratio-0 cache.
pub(crate) fn run_device_raw_window_attention(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    q_rope: &DeviceBuffer<u16>,
    attention_cache: &DeviceBuffer<u16>,
    attn_sink: &DeviceBuffer<f32>,
    topk: &DeviceBuffer<i32>,
    sparse: &mut DeviceBuffer<u16>,
    sparse_scratch: &mut SparseAttentionScoreReuseScratch,
) -> Result<(), Box<dyn Error>> {
    let sparse_arm = sparse_attention_arm_from_env()?;
    if sparse_arm.uses_score_reuse(topk.len()) {
        run_sparse_attention_score_reuse_device_into(
            module,
            stream,
            q_rope,
            attention_cache,
            attn_sink,
            topk,
            sparse,
            sparse_scratch,
            topk.len(),
        )
    } else {
        run_sparse_attention_device_into(
            module,
            stream,
            q_rope,
            attention_cache,
            attn_sink,
            topk,
            sparse,
            topk.len(),
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_wavefront_l3_l4_attention_row(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    runtime: &LoadedLayerRuntime,
    state: &mut DeviceTokenState,
    rope_lut: &DeviceRequestRopeLut,
    hc_hidden: &[u16],
    start_pos: usize,
    sparse_arm: SparseAttentionArm,
) -> Result<DeviceAttentionRun, Box<dyn Error>> {
    let weights = &runtime.weights;
    if !matches!(weights.layer, 3 | 4)
        || state.layer != weights.layer
        || start_pos != state.committed_rows
        || start_pos > 1
        || hc_hidden.len() != v4_state::HC * v4_state::HIDDEN
    {
        return Err("CDQW L3/L4 attention row contract mismatch".into());
    }
    let started = std::time::Instant::now();
    let event_start = state
        .hc_hidden
        .context()
        .new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
    let event_end = state
        .hc_hidden
        .context()
        .new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
    event_start.record(stream)?;
    // SAFETY: the preallocated destination exactly matches this fixed one-row
    // host input and remains live through the terminal D2H sync.
    unsafe {
        cuda_core::memory::memcpy_htod_async(
            state.hc_hidden.cu_deviceptr(),
            hc_hidden.as_ptr(),
            state.hc_hidden.num_bytes(),
            stream.cu_stream(),
        )?;
    }
    rope_lut.copy_forward_row(stream, start_pos, &state.freqs)?;
    rope_lut.copy_inverse_row(stream, start_pos, &state.inverse_freqs)?;
    run_l0_mhc_pre_device_into(
        module,
        stream,
        &state.hc_hidden,
        &weights.hc_attn_fn,
        &weights.hc_attn_scale,
        &weights.hc_attn_base,
        &mut state.mixes,
        &mut state.pre,
        &mut state.post,
        &mut state.comb,
        &mut state.collapsed,
        &mut state.trace,
        None,
    )?;
    run_l0_rmsnorm_device_into(
        module,
        stream,
        &state.collapsed,
        &weights.attn_norm,
        &mut state.attn_input,
        v4_state::HIDDEN,
    )?;
    run_l0_fp8_linear_device_into(
        module,
        stream,
        &state.attn_input,
        &weights.wq_a,
        &mut state.fp8_7168,
        &mut state.fp8_7168_scales,
        &mut state.empty_u16,
        &mut state.q_lora_f32,
        &mut state.q_lora_raw,
        false,
    )?;
    run_l0_rmsnorm_device_into(
        module,
        stream,
        &state.q_lora_raw,
        &weights.q_norm,
        &mut state.q_lora,
        weights.wq_a.out_features,
    )?;
    run_l0_fp8_linear_device_into(
        module,
        stream,
        &state.q_lora,
        &weights.wq_b,
        &mut state.fp8_1536,
        &mut state.fp8_1536_scales,
        &mut state.empty_u16,
        &mut state.q_f32,
        &mut state.q_raw,
        true,
    )?;
    run_l0_rmsnorm_device_into(
        module,
        stream,
        &state.q_raw,
        &state.empty_u16,
        &mut state.q_norm,
        v4_state::Q_HEAD_DIM,
    )?;
    run_rope_device_into(
        module,
        stream,
        &state.q_norm,
        &state.freqs,
        &mut state.q_rope,
        v4_state::Q_HEADS,
        v4_state::Q_HEAD_DIM,
    )?;
    run_l0_fp8_linear_device_into(
        module,
        stream,
        &state.attn_input,
        &weights.wkv,
        &mut state.fp8_7168,
        &mut state.fp8_7168_scales,
        &mut state.empty_u16,
        &mut state.kv_f32,
        &mut state.kv_raw,
        false,
    )?;
    run_l0_rmsnorm_device_into(
        module,
        stream,
        &state.kv_raw,
        &weights.kv_norm,
        &mut state.kv_norm,
        v4_state::Q_HEAD_DIM,
    )?;
    run_rope_device_into(
        module,
        stream,
        &state.kv_norm,
        &state.freqs,
        &mut state.kv_rope,
        1,
        v4_state::Q_HEAD_DIM,
    )?;
    run_l0_kv_quantize_device_into(
        module,
        stream,
        &state.kv_rope,
        &mut state.empty_u8,
        &mut state.kv_scales,
        &mut state.kv_ready,
    )?;
    run_compressor_device_into(
        module,
        stream,
        &state.attn_input,
        &weights.compressor_wkv,
        &weights.compressor_wgate,
        &weights.compressor_ape,
        &mut state.compressor.kv,
        &mut state.compressor.score,
        &mut state.compressor.score_plus_ape,
        state.compressor_width,
        weights.compression_ratio.get(),
        start_pos,
    )?;
    if let (Some(indexer), Some(indexer_state)) =
        (weights.indexer.as_ref(), state.indexer_compressor.as_mut())
    {
        run_compressor_device_into(
            module,
            stream,
            &state.attn_input,
            &indexer.compressor_wkv,
            &indexer.compressor_wgate,
            &indexer.compressor_ape,
            &mut indexer_state.kv,
            &mut indexer_state.score,
            &mut indexer_state.score_plus_ape,
            256,
            4,
            start_pos,
        )?;
        // ponytail: before the 3->4 pool boundary there are no indexer keys,
        // so its q/weight projections cannot affect state, top-k, or output.
        unsafe {
            let indexer_slot = 4 + start_pos;
            copy_device_row(
                stream,
                &indexer_state.persistent_kv,
                indexer_slot,
                &indexer_state.kv,
            )?;
            copy_device_row(
                stream,
                &indexer_state.persistent_score,
                indexer_slot,
                &indexer_state.score_plus_ape,
            )?;
        }
    }
    unsafe {
        copy_device_row(stream, &state.kv_cache, start_pos, &state.kv_ready)?;
        let compressor_slot = if weights.compression_ratio == v4_state::CompressionRatio::Csa4 {
            weights.compression_ratio.get() + start_pos
        } else {
            start_pos
        };
        copy_device_row(
            stream,
            &state.compressor.persistent_kv,
            compressor_slot,
            &state.compressor.kv,
        )?;
        copy_device_row(
            stream,
            &state.compressor.persistent_score,
            compressor_slot,
            &state.compressor.score_plus_ape,
        )?;
    }
    let topk = state
        .topk_raw
        .get(start_pos)
        .ok_or("device attention gate top-k position is outside short-decode storage")?;
    if !sparse_arm.uses_score_reuse(topk.len()) {
        run_sparse_attention_device_into(
            module,
            stream,
            &state.q_rope,
            &state.kv_cache,
            &weights.attn_sink,
            topk,
            &mut state.sparse,
            topk.len(),
        )?;
    } else if start_pos == 0 {
        run_sparse_attention_score_reuse_device_into(
            module,
            stream,
            &state.q_rope,
            &state.kv_cache,
            &weights.attn_sink,
            topk,
            &mut state.sparse,
            &mut state.sparse_pos0,
            topk.len(),
        )?;
    } else {
        run_sparse_attention_score_reuse_device_into(
            module,
            stream,
            &state.q_rope,
            &state.kv_cache,
            &weights.attn_sink,
            topk,
            &mut state.sparse,
            &mut state.sparse_pos1,
            topk.len(),
        )?;
    }
    run_rope_device_into(
        module,
        stream,
        &state.sparse,
        &state.inverse_freqs,
        &mut state.sparse_inverse,
        v4_state::Q_HEADS,
        v4_state::Q_HEAD_DIM,
    )?;
    run_l0_wo_a_device_into(
        module,
        stream,
        &state.sparse_inverse,
        &weights.wo_a,
        &mut state.wo_a,
    )?;
    run_l0_fp8_linear_device_into(
        module,
        stream,
        &state.wo_a,
        &weights.wo_b,
        &mut state.fp8_16384,
        &mut state.fp8_16384_scales,
        &mut state.empty_u16,
        &mut state.attention_f32,
        &mut state.attention,
        false,
    )?;
    run_l0_hc_post_device_into(
        module,
        stream,
        &state.attention,
        &state.hc_hidden,
        &state.post,
        &state.comb,
        &mut state.after_attention,
    )?;
    event_end.record(stream)?;
    event_end.synchronize()?;
    let cuda_event_ns = (event_start.elapsed_ms(&event_end)? as f64 * 1_000_000.0) as u64;
    let output = state.after_attention.to_host_vec(stream)?;
    let elapsed_ns = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
    if !sparse_arm.uses_score_reuse(topk.len()) {
        let scratch = if start_pos == 0 {
            &mut state.sparse_pos0
        } else {
            &mut state.sparse_pos1
        };
        run_sparse_attention_reference_state_device_into(
            module,
            stream,
            &state.q_rope,
            &state.kv_cache,
            &weights.attn_sink,
            topk,
            scratch,
            topk.len(),
        )?;
    }
    state.committed_rows += 1;
    Ok(DeviceAttentionRun {
        output,
        elapsed_ns,
        cuda_event_ns,
        h2d_count: 1,
        h2d_bytes: state.hc_hidden.num_bytes(),
        rope_lut_d2d_count: 2,
        rope_lut_d2d_bytes: state.freqs.num_bytes() + state.inverse_freqs.num_bytes(),
        d2h_count: 1,
        d2h_bytes: state.after_attention.num_bytes(),
        alloc_count: 0,
    })
}
