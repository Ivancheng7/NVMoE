use super::*;

/// Default-on switch for the device-resident attention chain. The host path
/// remains fully available at `COLIBRI_V4_ATTENTION_DEVICE_CHAIN=0`, which is
/// what the A/B comparison and the parity dumps run against.
pub(crate) fn attention_device_chain_enabled() -> Result<bool, Box<dyn Error>> {
    let value = std::env::var_os("COLIBRI_V4_ATTENTION_DEVICE_CHAIN");
    parse_default_true_zero_one_flag("COLIBRI_V4_ATTENTION_DEVICE_CHAIN", value.as_deref())
}

/// The chain serves the delivery decode path, which always executes one row
/// per layer call (the prompt-chunk loop is position-inner, single row). The
/// ops derive row counts from buffer lengths, so the scratch is sized for
/// exactly one row and any other row count is a contract error.
pub(crate) const ATTENTION_DEVICE_CHAIN_MAX_ROWS: usize = 1;

/// One preallocated set of the sixteen small-step intermediates. Every buffer
/// lives for the whole run, so a layer's attention enqueues its kernels
/// without a single host round-trip until the compressor/commit boundary.
pub(crate) struct AttentionDeviceScratch {
    rows: usize,
    hidden: usize,
    q_heads: usize,
    q_head_dim: usize,
    q_lora_rank: usize,
    o_groups: usize,
    o_lora_rank: usize,
    hc_hidden: DeviceBuffer<u16>,
    mixes: DeviceBuffer<f32>,
    pre: DeviceBuffer<f32>,
    post: DeviceBuffer<f32>,
    comb: DeviceBuffer<f32>,
    trace: DeviceBuffer<f32>,
    attn_norm_trace: DeviceBuffer<f32>,
    collapsed: DeviceBuffer<u16>,
    attn_input: DeviceBuffer<u16>,
    fp8_in: DeviceBuffer<u8>,
    fp8_in_scales: DeviceBuffer<u8>,
    fp8_qb_in: DeviceBuffer<u8>,
    fp8_qb_in_scales: DeviceBuffer<u8>,
    fp8_wob_in: DeviceBuffer<u8>,
    fp8_wob_in_scales: DeviceBuffer<u8>,
    empty_u16: DeviceBuffer<u16>,
    empty_u8: DeviceBuffer<u8>,
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
    freqs: DeviceBuffer<f32>,
    inverse_freqs: DeviceBuffer<f32>,
    sparse: DeviceBuffer<u16>,
    sparse_reuse: SparseAttentionScoreReuseScratch,
    sparse_inverse: DeviceBuffer<u16>,
    wo_a: DeviceBuffer<u16>,
    attention_f32: DeviceBuffer<f32>,
    attention: DeviceBuffer<u16>,
    after_attention: DeviceBuffer<u16>,
    topk_device: DeviceBuffer<i32>,
    kv_cache_device: DeviceBuffer<u16>,
}

impl AttentionDeviceScratch {
    pub(crate) fn new(
        stream: &CudaStream,
        spec: &'static v4_spec::V4Spec,
    ) -> Result<Self, Box<dyn Error>> {
        let rows = ATTENTION_DEVICE_CHAIN_MAX_ROWS;
        let hidden = spec.hidden;
        let q_heads = spec.q_heads;
        let q_head_dim = spec.q_head_dim;
        let q_lora_rank = spec.q_lora_rank;
        let o_groups = spec.o_groups;
        let o_lora_rank = spec.o_lora_rank;
        let fp8_in_blocks = hidden / v4_state::FP8_BLOCK;
        let q_out = q_heads * q_head_dim;
        let wob_in = o_groups * o_lora_rank;
        Ok(Self {
            rows,
            hidden,
            q_heads,
            q_head_dim,
            q_lora_rank,
            o_groups,
            o_lora_rank,
            hc_hidden: DeviceBuffer::zeroed(stream, rows * v4_state::HC * hidden)?,
            mixes: DeviceBuffer::zeroed(stream, rows * v4_state::MIX)?,
            pre: DeviceBuffer::zeroed(stream, rows * v4_state::HC)?,
            post: DeviceBuffer::zeroed(stream, rows * v4_state::HC)?,
            comb: DeviceBuffer::zeroed(stream, rows * v4_state::HC * v4_state::HC)?,
            trace: DeviceBuffer::zeroed(stream, rows * 41 * v4_state::HC * v4_state::HC)?,
            attn_norm_trace: DeviceBuffer::zeroed(stream, 0)?,
            collapsed: DeviceBuffer::zeroed(stream, rows * hidden)?,
            attn_input: DeviceBuffer::zeroed(stream, rows * hidden)?,
            fp8_in: DeviceBuffer::zeroed(stream, rows * hidden)?,
            fp8_in_scales: DeviceBuffer::zeroed(stream, rows * fp8_in_blocks)?,
            fp8_qb_in: DeviceBuffer::zeroed(stream, rows * q_lora_rank)?,
            fp8_qb_in_scales: DeviceBuffer::zeroed(
                stream,
                rows * q_lora_rank / v4_state::FP8_BLOCK,
            )?,
            fp8_wob_in: DeviceBuffer::zeroed(stream, rows * wob_in)?,
            fp8_wob_in_scales: DeviceBuffer::zeroed(stream, rows * wob_in / v4_state::FP8_BLOCK)?,
            empty_u16: DeviceBuffer::zeroed(stream, 0)?,
            empty_u8: DeviceBuffer::zeroed(stream, 0)?,
            q_lora_f32: DeviceBuffer::zeroed(stream, rows * q_lora_rank)?,
            q_lora_raw: DeviceBuffer::zeroed(stream, rows * q_lora_rank)?,
            q_lora: DeviceBuffer::zeroed(stream, rows * q_lora_rank)?,
            q_f32: DeviceBuffer::zeroed(stream, rows * q_out)?,
            q_raw: DeviceBuffer::zeroed(stream, rows * q_out)?,
            q_norm: DeviceBuffer::zeroed(stream, rows * q_out)?,
            q_rope: DeviceBuffer::zeroed(stream, rows * q_out)?,
            kv_f32: DeviceBuffer::zeroed(stream, rows * q_head_dim)?,
            kv_raw: DeviceBuffer::zeroed(stream, rows * q_head_dim)?,
            kv_norm: DeviceBuffer::zeroed(stream, rows * q_head_dim)?,
            kv_rope: DeviceBuffer::zeroed(stream, rows * q_head_dim)?,
            kv_ready: DeviceBuffer::zeroed(stream, rows * q_head_dim)?,
            kv_scales: DeviceBuffer::zeroed(stream, rows * 7)?,
            freqs: DeviceBuffer::zeroed(stream, rows * v4_state::Q_ROPE_DIM)?,
            inverse_freqs: DeviceBuffer::zeroed(stream, rows * v4_state::Q_ROPE_DIM)?,
            sparse: DeviceBuffer::zeroed(stream, rows * q_out)?,
            sparse_reuse: SparseAttentionScoreReuseScratch::new_for_heads(
                stream,
                rows,
                q_heads,
                v4_state::KV_WINDOW + v4_state::KV_COMPRESSED,
            )?,
            sparse_inverse: DeviceBuffer::zeroed(stream, rows * q_out)?,
            wo_a: DeviceBuffer::zeroed(stream, rows * wob_in)?,
            attention_f32: DeviceBuffer::zeroed(stream, rows * hidden)?,
            attention: DeviceBuffer::zeroed(stream, rows * hidden)?,
            after_attention: DeviceBuffer::zeroed(stream, rows * v4_state::HC * hidden)?,
            topk_device: DeviceBuffer::zeroed(
                stream,
                rows * (v4_state::KV_WINDOW + v4_state::KV_COMPRESSED),
            )?,
            kv_cache_device: DeviceBuffer::zeroed(
                stream,
                (v4_state::KV_WINDOW + v4_state::KV_COMPRESSED) * q_head_dim,
            )?,
        })
    }

    pub(crate) fn num_bytes(&self) -> usize {
        [
            self.hc_hidden.num_bytes(),
            self.mixes.num_bytes(),
            self.pre.num_bytes(),
            self.post.num_bytes(),
            self.comb.num_bytes(),
            self.trace.num_bytes(),
            self.attn_norm_trace.num_bytes(),
            self.collapsed.num_bytes(),
            self.attn_input.num_bytes(),
            self.fp8_in.num_bytes(),
            self.fp8_in_scales.num_bytes(),
            self.fp8_qb_in.num_bytes(),
            self.fp8_qb_in_scales.num_bytes(),
            self.fp8_wob_in.num_bytes(),
            self.fp8_wob_in_scales.num_bytes(),
            self.empty_u16.num_bytes(),
            self.empty_u8.num_bytes(),
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
            self.freqs.num_bytes(),
            self.inverse_freqs.num_bytes(),
            self.sparse.num_bytes(),
            self.sparse_reuse.num_bytes(),
            self.sparse_inverse.num_bytes(),
            self.wo_a.num_bytes(),
            self.attention_f32.num_bytes(),
            self.attention.num_bytes(),
            self.after_attention.num_bytes(),
            self.topk_device.num_bytes(),
            self.kv_cache_device.num_bytes(),
        ]
        .into_iter()
        .fold(0, usize::saturating_add)
    }
}

/// Enqueues the mHC + attention projections with device-resident buffers.
///
/// Mirrors `step_real_layer_attention_rows`'s first eleven steps exactly (same
/// kernels, same argument order), minus the per-step allocations, host
/// round-trips, and their stream synchronizations. Returns nothing: every
/// intermediate stays in `scratch` until the commit boundary reads them.
struct AttentionDeviceChainRun<'scratch, 'weights> {
    weights: &'weights LayerDeviceWeights,
    spec: &'static v4_spec::V4Spec,
    module: &'weights kernels::LoadedModule,
    stream: &'weights CudaStream,
    scratch: &'scratch mut AttentionDeviceScratch,
    rows: usize,
}

impl<'scratch, 'weights> AttentionDeviceChainRun<'scratch, 'weights> {
    /// Step 1: mHC pre-projection (linear arm chosen by env, same as host).
    /// The three mHC tables come straight from the layer weights; they are
    /// already device-resident, so no upload happens here. The row input is
    /// the scratch's own copy, uploaded by the caller.
    fn mhc(
        &mut self,
        cublas_session: Option<&FlashCublasSession>,
        linear_arm: FlashMhcLinearArm,
    ) -> Result<(), Box<dyn Error>> {
        let weights = self.weights;
        let rows = self.rows;
        let scratch = &mut *self.scratch;
        let row_launch = LaunchConfig::for_num_elems(u32::try_from(rows)?);
        if linear_arm == FlashMhcLinearArm::Canonical {
            let linear_launch = LaunchConfig::for_num_elems(u32::try_from(
                rows.checked_mul(32).ok_or("mHC linear launch overflow")?,
            )?);
            // SAFETY: one row per validated launch, disjoint outputs.
            unsafe {
                self.module.mhc_linear_v4_bf16(
                    self.stream,
                    linear_launch,
                    &scratch.hc_hidden,
                    &weights.hc_attn_fn,
                    &mut scratch.mixes,
                )?;
            }
        } else {
            let session =
                cublas_session.ok_or("Flash cuBLAS mHC session is missing for the device chain")?;
            step_flash_mhc_cublas_linear_device_into(
                session,
                self.module,
                self.stream,
                &scratch.hc_hidden,
                &weights.hc_attn_fn,
                &mut scratch.mixes,
                rows,
                self.spec.hidden,
                false,
            )?;
        }
        // SAFETY: kernel owns complete validated rows; outputs are disjoint
        // workspace buffers sized for this row count.
        unsafe {
            self.module.mhc_sinkhorn_v4_bf16(
                self.stream,
                row_launch,
                &scratch.hc_hidden,
                &scratch.mixes,
                &weights.hc_attn_scale,
                &weights.hc_attn_base,
                &mut scratch.pre,
                &mut scratch.post,
                &mut scratch.comb,
                &mut scratch.collapsed,
                &mut scratch.trace,
            )?;
        }
        Ok(())
    }

    /// Steps 2-11: norm, q/kv projections, norms, RoPE, quantize. Same kernel
    /// order and arguments as the host path; buffers are preallocated.
    fn projections(&mut self) -> Result<(), Box<dyn Error>> {
        let spec = self.spec;
        let scratch = &mut *self.scratch;
        step_flash_learned_rmsnorm_4096_device_into(
            self.module,
            self.stream,
            &scratch.collapsed,
            &self.weights.attn_norm,
            &mut scratch.attn_input,
            &mut scratch.attn_norm_trace,
        )?;
        // q_a projection: quantize attn_input then fp8 matvec, device-resident.
        step_l0_fp8_linear_device_into(
            self.module,
            self.stream,
            &scratch.attn_input,
            &self.weights.wq_a,
            &mut scratch.fp8_in,
            &mut scratch.fp8_in_scales,
            &mut scratch.empty_u16,
            &mut scratch.q_lora_f32,
            &mut scratch.q_lora_raw,
            false,
        )?;
        step_l0_rmsnorm_device_into(
            self.module,
            self.stream,
            &scratch.q_lora_raw,
            &self.weights.q_norm,
            &mut scratch.q_lora,
            self.weights.wq_a.out_features,
        )?;
        step_l0_fp8_linear_device_into(
            self.module,
            self.stream,
            &scratch.q_lora,
            &self.weights.wq_b,
            &mut scratch.fp8_qb_in,
            &mut scratch.fp8_qb_in_scales,
            &mut scratch.empty_u16,
            &mut scratch.q_f32,
            &mut scratch.q_raw,
            true,
        )?;
        step_l0_rmsnorm_device_into(
            self.module,
            self.stream,
            &scratch.q_raw,
            &scratch.empty_u16,
            &mut scratch.q_norm,
            v4_state::Q_HEAD_DIM,
        )?;
        step_rope_device_into(
            self.module,
            self.stream,
            &scratch.q_norm,
            &scratch.freqs,
            &mut scratch.q_rope,
            spec.q_heads,
            spec.q_head_dim,
        )?;
        step_l0_fp8_linear_device_into(
            self.module,
            self.stream,
            &scratch.attn_input,
            &self.weights.wkv,
            &mut scratch.fp8_in,
            &mut scratch.fp8_in_scales,
            &mut scratch.empty_u16,
            &mut scratch.kv_f32,
            &mut scratch.kv_raw,
            false,
        )?;
        step_l0_rmsnorm_device_into(
            self.module,
            self.stream,
            &scratch.kv_raw,
            &self.weights.kv_norm,
            &mut scratch.kv_norm,
            v4_state::Q_HEAD_DIM,
        )?;
        step_rope_device_into(
            self.module,
            self.stream,
            &scratch.kv_norm,
            &scratch.freqs,
            &mut scratch.kv_rope,
            1,
            v4_state::Q_HEAD_DIM,
        )?;
        step_l0_kv_quantize_device_into(
            self.module,
            self.stream,
            &scratch.kv_rope,
            &mut scratch.empty_u8,
            &mut scratch.kv_scales,
            &mut scratch.kv_ready,
        )?;
        Ok(())
    }
}

/// Device-resident replacement for `step_real_layer_attention_rows` on the
/// delivery decode path. Same kernels in the same order as the host path;
/// the only structural differences are the preallocated scratch (no per-step
/// `malloc` + `from_host`/`to_host_vec` pairs) and where host values are
/// actually needed: the compressor/indexer inputs and the commit boundary.
///
/// Returns the same `RealAttentionRun` contract. `attention_profile` is `None`
/// — the step split no longer exists on this path, and the phase JSON keeps
/// rendering the totals (`attention_ns`, `commit_ns`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn step_real_layer_attention_rows_device(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    flash_cublas: Option<&FlashCublasSession>,
    runtime: &LoadedLayerRuntime,
    hc_hidden: &[u16],
    start_pos: usize,
    state: &mut v4_state::V4LayerPersistentState,
    cublas: Option<&cublas_exact::CublasExact>,
    strict_scratch: Option<&mut StrictCompressorScratch>,
    scratch: &mut AttentionDeviceScratch,
) -> Result<RealAttentionRun, Box<dyn Error>> {
    let weights = &runtime.weights;
    let spec = weights.spec;
    let row_width = v4_state::HC * spec.hidden;
    let rows = hc_hidden.len() / row_width;
    if rows != scratch.rows
        || hc_hidden.len() != rows * row_width
        || state.layer() != weights.layer
        || state.committed_kv_rows() != start_pos
        || spec != &v4_spec::FLASH_0731
        || scratch.hidden != spec.hidden
    {
        return Err(format!(
            "layer {} device attention chain input/state contract mismatch",
            weights.layer
        )
        .into());
    }
    // The diagnostic surfaces that read host intermediates stay on the host
    // path; they are mutually exclusive with this chain by construction.
    if std::env::var_os("COLIBRI_V4_FLASH_PARITY_DUMP_DIR").is_some()
        || std::env::var_os("COLIBRI_V4_WAVEFRONT_ATTENTION_BENCH").is_some()
        || std::env::var_os("COLIBRI_V4_AUDIT_LAYER").is_some()
    {
        return Err("diagnostic dump paths require COLIBRI_V4_ATTENTION_DEVICE_CHAIN=0".into());
    }
    let attention_started = std::time::Instant::now();
    let linear_arm = flash_mhc_linear_arm(spec)?;
    let mut strict_scratch = strict_scratch;
    // The row input upload is the chain's only H2D before the commit boundary.
    // It is enqueued without synchronizing: the borrowed `hc_hidden` slice
    // outlives this call, and the next stream synchronization (the D2H below)
    // orders the copy before anything reads it.
    //
    // SAFETY: `hc_hidden` is the caller's row input; nothing mutates or drops
    // it before the next `stream.synchronize()` inside this function.
    unsafe {
        scratch
            .hc_hidden
            .copy_from_host_async_unchecked(stream, hc_hidden)?;
    }
    let mut freqs = Vec::with_capacity(rows * v4_state::Q_ROPE_DIM);
    for row in 0..rows {
        freqs.extend(v4_state::attention_rope_freqs(
            spec,
            weights.layer,
            start_pos + row,
        )?);
    }
    let mut inverse_freqs = freqs.clone();
    for sine in inverse_freqs.iter_mut().skip(1).step_by(2) {
        *sine = -*sine;
    }
    // SAFETY: both slices are locals that stay alive until the D2H sync below.
    unsafe {
        scratch
            .freqs
            .copy_from_host_async_unchecked(stream, &freqs)?;
        scratch
            .inverse_freqs
            .copy_from_host_async_unchecked(stream, &inverse_freqs)?;
    }

    let mut chain = AttentionDeviceChainRun {
        weights,
        spec,
        module,
        stream,
        scratch,
        rows,
    };
    chain.mhc(flash_cublas, linear_arm)?;
    chain.projections()?;

    // Compressor/indexer still run from host slices; their kernels are
    // untouched. attn_input and q_lora come back once for them, and kv_ready
    // once for the commit. This is the chain's one D2H sync point.
    let attn_input = chain.scratch.attn_input.to_host_vec(stream)?;
    let q_lora = chain.scratch.q_lora.to_host_vec(stream)?;
    let kv = chain.scratch.kv_ready.to_host_vec(stream)?;
    let compressor = if weights.compression_ratio == v4_state::CompressionRatio::WindowOnly {
        None
    } else {
        Some(step_layer_compressor(
            module,
            stream,
            &attn_input,
            weights,
            start_pos,
            cublas,
            strict_scratch.as_deref_mut(),
        )?)
    };
    let indexer = if let Some(indexer) = weights.indexer.as_ref() {
        Some(step_csa_indexer_projections(
            spec,
            module,
            stream,
            &attn_input,
            &q_lora,
            &freqs,
            indexer,
            start_pos,
            cublas,
            strict_scratch,
        )?)
    } else {
        None
    };
    let commit_started = std::time::Instant::now();
    let (topk, indexer_topk_gpu_ns) = commit_layer_attention_state(
        module,
        stream,
        weights,
        state,
        &kv,
        compressor.as_ref(),
        indexer.as_ref(),
        start_pos,
    )?;
    let commit_ns = u64::try_from(commit_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
    let attention_topk_digest = fnv1a_i32(&topk.indices);
    let cache = state.attention_cache_bf16();
    let scratch = chain.scratch;
    // Csa4 top-k and the compressed cache grow with position. Grow the staging
    // buffers on demand instead of preallocating the request plan's cap; in a
    // normal run this reallocates a handful of times, early. The staged
    // buffers can be longer than the payload, so the copies are byte-exact
    // through the raw memcpy instead of the length-equal wrapper.
    if scratch.kv_cache_device.len() < cache.len() {
        scratch.kv_cache_device = DeviceBuffer::zeroed(stream, cache.len())?;
    }
    if scratch.topk_device.len() < topk.indices.len() {
        scratch.topk_device = DeviceBuffer::zeroed(stream, topk.indices.len())?;
    }
    if scratch.sparse_reuse.scores.len() < rows * scratch.q_heads * topk.per_row {
        scratch.sparse_reuse = SparseAttentionScoreReuseScratch::new_for_heads(
            stream,
            rows,
            scratch.q_heads,
            topk.per_row,
        )?;
    }
    // SAFETY: `cache` and `topk.indices` stay alive until the final
    // `to_host_vec` below synchronizes the stream; both destinations are
    // allocated with at least the copied byte counts.
    unsafe {
        cuda_core::memory::memcpy_htod_async(
            scratch.kv_cache_device.cu_deviceptr(),
            cache.as_ptr(),
            std::mem::size_of_val(&cache[..]),
            stream.cu_stream(),
        )?;
        cuda_core::memory::memcpy_htod_async(
            scratch.topk_device.cu_deviceptr(),
            topk.indices.as_ptr(),
            std::mem::size_of_val(&topk.indices[..]),
            stream.cu_stream(),
        )?;
    }
    // Steps 13-16: sparse attention, inverse RoPE, wo_a, wo_b — back on the
    // device, consuming the cache/topk just uploaded.
    let arm = sparse_attention_arm_from_env()?;
    match arm {
        SparseAttentionArm::Reference => step_sparse_attention_device_into(
            module,
            stream,
            &scratch.q_rope,
            &scratch.kv_cache_device,
            &weights.attn_sink,
            &scratch.topk_device,
            &mut scratch.sparse,
            topk.per_row,
        )?,
        SparseAttentionArm::ScoreReuse | SparseAttentionArm::Sm89Crossover => {
            step_sparse_attention_score_reuse_device_into(
                module,
                stream,
                &scratch.q_rope,
                &scratch.kv_cache_device,
                &weights.attn_sink,
                &scratch.topk_device,
                &mut scratch.sparse,
                &mut scratch.sparse_reuse,
                topk.per_row,
            )?
        }
    }
    step_rope_device_into(
        module,
        stream,
        &scratch.sparse,
        &scratch.inverse_freqs,
        &mut scratch.sparse_inverse,
        spec.q_heads,
        spec.q_head_dim,
    )?;
    step_l0_wo_a_device_into_for(
        spec,
        module,
        stream,
        &scratch.sparse_inverse,
        &weights.wo_a,
        &mut scratch.wo_a,
    )?;
    step_l0_fp8_linear_device_into(
        module,
        stream,
        &scratch.wo_a,
        &weights.wo_b,
        &mut scratch.fp8_wob_in,
        &mut scratch.fp8_wob_in_scales,
        &mut scratch.empty_u16,
        &mut scratch.attention_f32,
        &mut scratch.attention,
        false,
    )?;
    // Step 17: hc_post consumes the device-resident post/comb from step 1 and
    // the scratch's own residual copy of the row input.
    step_l0_hc_post_device_into(
        module,
        stream,
        &scratch.attention,
        &scratch.hc_hidden,
        &scratch.post,
        &scratch.comb,
        &mut scratch.after_attention,
    )?;
    let output = scratch.after_attention.to_host_vec(stream)?;
    let attention_ns = u64::try_from(attention_started.elapsed().as_nanos())
        .unwrap_or(u64::MAX)
        .saturating_sub(commit_ns);
    Ok(RealAttentionRun {
        output,
        topk: Some(topk.indices.clone()),
        stage_digests: Vec::new(),
        attention_topk_digest,
        attention_topk_per_row: topk.per_row,
        indexer_topk_gpu_ns,
        attention_ns,
        commit_ns,
        device_transfer: None,
        profile: None,
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn commit_layer_attention_state(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    weights: &LayerDeviceWeights,
    state: &mut v4_state::V4LayerPersistentState,
    kv: &[u16],
    compressor: Option<&v4_state::CompressorPrepared>,
    indexer: Option<&CsaIndexerPrepared>,
    start_pos: usize,
) -> Result<(v4_state::SparseTopk, u64), Box<dyn Error>> {
    let rows = kv.len() / v4_state::Q_HEAD_DIM;
    if weights.compression_ratio == v4_state::CompressionRatio::WindowOnly {
        if rows == 0
            || kv.len() != rows * v4_state::Q_HEAD_DIM
            || indexer.is_some()
            || compressor.is_some()
        {
            return Err(format!(
                "layer {} window-only attention state input contract mismatch",
                weights.layer
            )
            .into());
        }
        for row in 0..rows {
            state.append_kv_row(
                start_pos + row,
                &kv[row * v4_state::Q_HEAD_DIM..(row + 1) * v4_state::Q_HEAD_DIM],
            )?;
        }
        let (indices, per_row) = v4_state::window_topk_cpu(rows, start_pos, v4_state::KV_WINDOW)?;
        return Ok((
            v4_state::SparseTopk {
                indices,
                rows,
                per_row,
                window_per_row: per_row,
                compressed_per_row: 0,
            },
            0,
        ));
    }
    let compressor = compressor.ok_or("compressed attention omitted compressor projections")?;
    let compressor_width = if weights.compression_ratio == v4_state::CompressionRatio::Csa4 {
        1_024
    } else {
        v4_state::Q_HEAD_DIM
    };
    if rows == 0
        || kv.len() != rows * v4_state::Q_HEAD_DIM
        || compressor.kv_f32.len() != rows * compressor_width
        || compressor.score_f32.len() != rows * compressor_width
        || compressor.score_plus_ape_f32.len() != rows * compressor_width
        || (weights.compression_ratio == v4_state::CompressionRatio::Csa4) != indexer.is_some()
    {
        return Err(format!(
            "layer {} attention state input contract mismatch",
            weights.layer
        )
        .into());
    }
    let ratio = weights.compression_ratio.get();
    for row in 0..rows {
        let absolute = start_pos + row;
        state.append_kv_row(
            absolute,
            &kv[row * v4_state::Q_HEAD_DIM..(row + 1) * v4_state::Q_HEAD_DIM],
        )?;
        let main_range = row * compressor_width..(row + 1) * compressor_width;
        let main_pool = state.commit_compressor_row_and_pool(
            absolute,
            &v4_state::CompressorPrepared {
                kv_f32: compressor.kv_f32[main_range.clone()].to_vec(),
                score_f32: compressor.score_f32[main_range.clone()].to_vec(),
                score_plus_ape_f32: compressor.score_plus_ape_f32[main_range].to_vec(),
            },
        )?;
        let index_pool = if let Some(indexer) = indexer {
            let range = row * 256..(row + 1) * 256;
            state.commit_indexer_row_and_pool(
                absolute,
                &v4_state::CompressorPrepared {
                    kv_f32: indexer.compressor.kv_f32[range.clone()].to_vec(),
                    score_f32: indexer.compressor.score_f32[range.clone()].to_vec(),
                    score_plus_ape_f32: indexer.compressor.score_plus_ape_f32[range].to_vec(),
                },
            )?
        } else {
            None
        };
        match (main_pool, index_pool) {
            (None, None) => {}
            (Some(main), None)
                if weights.compression_ratio == v4_state::CompressionRatio::Hca128 =>
            {
                let main = main
                    .into_iter()
                    .map(v4_router::f32_to_bf16_rne)
                    .collect::<Vec<_>>();
                let main = step_l0_rmsnorm(
                    module,
                    stream,
                    &main,
                    &weights.compressor_norm,
                    v4_state::Q_HEAD_DIM,
                )?;
                let position = absolute + 1 - ratio;
                let main = step_l0_rope(
                    module,
                    stream,
                    &main,
                    &v4_state::compressed_rope_freqs(position),
                    1,
                    false,
                )?;
                let main = step_l0_kv_quantize(module, stream, &main)?;
                state.append_compressed_kv_row(absolute / ratio, &main)?;
            }
            (Some(main), Some(index))
                if weights.compression_ratio == v4_state::CompressionRatio::Csa4 =>
            {
                let indexer_weights = weights
                    .indexer
                    .as_ref()
                    .ok_or("CSA indexer device weights disappeared")?;
                let position = absolute + 1 - ratio;
                let freqs = v4_state::compressed_rope_freqs(position);
                let main = main
                    .into_iter()
                    .map(v4_router::f32_to_bf16_rne)
                    .collect::<Vec<_>>();
                let main = step_l0_rmsnorm(
                    module,
                    stream,
                    &main,
                    &weights.compressor_norm,
                    v4_state::Q_HEAD_DIM,
                )?;
                let main = step_l0_rope(module, stream, &main, &freqs, 1, false)?;
                let main = step_l0_kv_quantize(module, stream, &main)?;
                state.append_compressed_kv_row(absolute / ratio, &main)?;

                let index = index
                    .into_iter()
                    .map(v4_router::f32_to_bf16_rne)
                    .collect::<Vec<_>>();
                let index = step_l0_rmsnorm(
                    module,
                    stream,
                    &index,
                    &indexer_weights.compressor_norm,
                    128,
                )?;
                let index = run_rope(module, stream, &index, &freqs, 1, 128, false)?;
                let index = v4_state::rotate_fp4_sim_bf16(&index, 1, 128)?
                    .into_iter()
                    .map(v4_router::f32_to_bf16_rne)
                    .collect::<Vec<_>>();
                state.append_indexer_kv_row(absolute / ratio, &index)?;
            }
            _ => {
                return Err(format!(
                    "layer {} compressor/indexer boundary commit diverged",
                    weights.layer
                )
                .into());
            }
        }
    }

    if weights.compression_ratio == v4_state::CompressionRatio::Hca128 {
        return Ok((
            v4_state::build_hca_sparse_topk(rows, start_pos, v4_state::KV_WINDOW, ratio)?,
            0,
        ));
    }
    let (window, window_per_row) = v4_state::window_topk_cpu(rows, start_pos, v4_state::KV_WINDOW)?;
    let keys = state.indexer_kv_bf16().len() / 128;
    if keys == 0 {
        return Ok((
            v4_state::SparseTopk {
                indices: window,
                rows,
                per_row: window_per_row,
                window_per_row,
                compressed_per_row: 0,
            },
            0,
        ));
    }
    let indexer = indexer.ok_or("CSA indexer projection result disappeared")?;
    let valid_keys = (0..rows)
        .map(|row| (start_pos + row + 1) / ratio)
        .collect::<Vec<_>>();
    let (compressed, indexer_topk_gpu_ns) = run_indexer_topk_gpu(
        weights.spec,
        module,
        stream,
        &indexer.q_f32,
        &indexer.weights_f32,
        state.indexer_kv_bf16(),
        rows,
        &valid_keys,
        v4_state::KV_WINDOW,
    )?;
    let compressed_per_row = compressed.len() / rows;
    let mut indices = Vec::with_capacity(rows * (window_per_row + compressed_per_row));
    for row in 0..rows {
        indices.extend_from_slice(&window[row * window_per_row..(row + 1) * window_per_row]);
        indices.extend_from_slice(
            &compressed[row * compressed_per_row..(row + 1) * compressed_per_row],
        );
    }
    Ok((
        v4_state::SparseTopk {
            indices,
            rows,
            per_row: window_per_row + compressed_per_row,
            window_per_row,
            compressed_per_row,
        },
        indexer_topk_gpu_ns,
    ))
}
