use super::*;

pub(crate) const V4_Q_LORA_RANK: usize = 1_536;
pub(crate) const V4_DEVICE_TOKEN_STATE_CSA_WITHOUT_Q_LORA_F32_BYTES: usize =
    83_863_054 - V4_Q_LORA_RANK * std::mem::size_of::<f32>();
pub(crate) const V4_DEVICE_TOKEN_STATE_HCA_WITHOUT_Q_LORA_F32_BYTES: usize = 40_249_870;
pub(crate) const fn device_token_state_ledger_bytes(csa: bool, q_lora_rank: usize) -> usize {
    (if csa {
        V4_DEVICE_TOKEN_STATE_CSA_WITHOUT_Q_LORA_F32_BYTES
    } else {
        V4_DEVICE_TOKEN_STATE_HCA_WITHOUT_Q_LORA_F32_BYTES
    }) + q_lora_rank * std::mem::size_of::<f32>()
}
pub(crate) const V4_DEVICE_TOKEN_STATE_MAX_BYTES: usize =
    device_token_state_ledger_bytes(true, V4_Q_LORA_RANK);
pub(crate) const V4_DEVICE_TOKEN_STATE_HCA_BYTES: usize =
    device_token_state_ledger_bytes(false, V4_Q_LORA_RANK);
pub(crate) const V4_REQUEST_ROPE_LUT_ROWS: usize = 5;
pub(crate) const V4_SHORT_DEVICE_DECODE_MAX_NEW_TOKENS: usize = 5;
// ponytail: this is a short exact gate, not paged serving; lift only when a
// semantic gate covers a paged/rolling DeviceTokenState beyond nine rows.
pub(crate) const V4_SHORT_DEVICE_DECODE_MAX_ROWS: usize = 9;

pub(crate) fn csa_compressed_rows_after(position: usize) -> usize {
    (position + 1) / 4
}
pub(crate) const V4_REQUEST_ROPE_LUT_BYTES: usize =
    2 * V4_REQUEST_ROPE_LUT_ROWS * v4_state::Q_ROPE_DIM * std::mem::size_of::<f32>();
pub(crate) const V4_TID2EID_TRANSIENT_MAX_BYTES: usize = v4_router::V4_TOPK * std::mem::size_of::<i64>();

pub(crate) fn device_attention_expected_h2d_bytes(
    rope_lut_bytes: usize,
    dispatches: usize,
) -> Result<usize, Box<dyn Error>> {
    rope_lut_bytes
        .checked_add(
            dispatches
                .checked_mul(v4_state::HC * v4_state::HIDDEN * std::mem::size_of::<u16>())
                .ok_or("device attention H2D byte count overflow")?,
        )
        .ok_or("device attention H2D byte count overflow".into())
}
pub(crate) const V4_QD4_PINNED_HOST_BYTES: usize = V4_QD4_LANES
    * (v4_expert::V4_ROUTED_EXPERT_BYTES
        + v4_expert::V4_HIDDEN * std::mem::size_of::<u16>()
        + v4_expert::V4_HIDDEN * std::mem::size_of::<f32>());
pub(crate) const V4_QD4_DEVICE_BYTES: usize = V4_QD4_LANES
    * (v4_expert::V4_ROUTED_EXPERT_BYTES
        + v4_expert::V4_HIDDEN * std::mem::size_of::<u16>()
        + v4_expert::V4_HIDDEN * std::mem::size_of::<f32>()
        + 3 * v4_expert::V4_EXPERT_INTERMEDIATE * std::mem::size_of::<u16>()
        + v4_expert::V4_EXPERT_INTERMEDIATE * std::mem::size_of::<f32>()
        + v4_expert::V4_HIDDEN * std::mem::size_of::<u16>()
        + v4_expert::V4_HIDDEN * std::mem::size_of::<f32>());
pub(crate) const V4_QD4_IMMUTABLE_EXPERT_CACHE_SLOTS: usize = 32;
pub(crate) const V4_QD4_IMMUTABLE_EXPERT_CACHE_HOST_BYTES: usize = V4_QD4_IMMUTABLE_EXPERT_CACHE_SLOTS
    * (v4_expert::V4_ROUTED_EXPERT_BYTES
        + v4_expert::V4_HIDDEN * std::mem::size_of::<u16>()
        + v4_expert::V4_HIDDEN * std::mem::size_of::<f32>());
pub(crate) const V4_QD4_IMMUTABLE_EXPERT_CACHE_DEVICE_BYTES: usize = V4_QD4_IMMUTABLE_EXPERT_CACHE_SLOTS
    * (v4_expert::V4_ROUTED_EXPERT_BYTES
        + v4_expert::V4_HIDDEN * std::mem::size_of::<u16>()
        + v4_expert::V4_HIDDEN * std::mem::size_of::<f32>()
        + 3 * v4_expert::V4_EXPERT_INTERMEDIATE * std::mem::size_of::<u16>()
        + v4_expert::V4_EXPERT_INTERMEDIATE * std::mem::size_of::<f32>()
        + v4_expert::V4_HIDDEN * std::mem::size_of::<u16>()
        + v4_expert::V4_HIDDEN * std::mem::size_of::<f32>());

impl DeviceRequestScratchGeometry {
    pub(crate) fn new(
        plan: v4_state::LayerStationaryRequestPlan,
        strict: bool,
    ) -> Result<Self, Box<dyn Error>> {
        let topk_rows = GenericStationaryPath::Hca128
            .max_topk_rows(plan)
            .max(GenericStationaryPath::Csa4.max_topk_rows(plan));
        Ok(Self {
            generic: GenericStationaryScratchGeometry::max(),
            csa_compressed_rows: plan.csa_compressed_rows,
            topk_rows,
            strict,
        })
    }

    pub(crate) const fn attention_f32_elements(self) -> usize {
        v4_state::HIDDEN
    }

    pub(crate) fn byte_breakdown(self) -> Result<DeviceRequestScratchByteBreakdown, Box<dyn Error>> {
        let bytes = |elements: usize, element_bytes: usize| {
            elements
                .checked_mul(element_bytes)
                .ok_or("generic stationary scratch breakdown overflow")
        };
        let sum = |values: &[usize]| {
            values.iter().try_fold(0usize, |total, &value| {
                total
                    .checked_add(value)
                    .ok_or("generic stationary scratch breakdown overflow")
            })
        };
        Ok(DeviceRequestScratchByteBreakdown {
            core_mhc: sum(&[
                bytes(v4_state::HC * v4_state::HIDDEN, 2)?,
                bytes(2 * v4_state::HIDDEN, 2)?,
                bytes(3 * v4_state::Q_ROPE_DIM, 4)?,
                bytes(
                    v4_state::MIX + 2 * v4_state::HC + v4_state::HC * v4_state::HC,
                    4,
                )?,
                bytes(41 * v4_state::HC * v4_state::HC, 4)?,
            ])?,
            projections: sum(&[
                bytes(
                    v4_state::HIDDEN
                        + v4_state::HIDDEN / v4_state::FP8_BLOCK
                        + V4_Q_LORA_RANK
                        + V4_Q_LORA_RANK / v4_state::FP8_BLOCK
                        + v4_state::O_GROUPS * v4_state::O_LORA_RANK
                        + v4_state::O_GROUPS * v4_state::O_LORA_RANK / v4_state::FP8_BLOCK,
                    1,
                )?,
                bytes(
                    2 * V4_Q_LORA_RANK
                        + 3 * v4_state::Q_HEADS * v4_state::Q_HEAD_DIM
                        + 4 * v4_state::Q_HEAD_DIM,
                    2,
                )?,
                bytes(
                    V4_Q_LORA_RANK
                        + v4_state::Q_HEADS * v4_state::Q_HEAD_DIM
                        + v4_state::Q_HEAD_DIM,
                    4,
                )?,
            ])?,
            compressor: sum(&[
                bytes(4 * v4_state::Q_HEAD_DIM, 2)?,
                bytes(14, 1)?,
                bytes(
                    2 * self.generic.csa_tail_elements
                        + 2 * self.generic.hca_tail_elements
                        + 3 * self.generic.csa_main_elements
                        + 3 * self.generic.hca_main_elements
                        + v4_state::Q_HEAD_DIM,
                    4,
                )?,
            ])?,
            exact_compressor: if self.strict {
                sum(&[
                    cublas_exact::WORKSPACE_BYTES,
                    bytes(
                        v4_state::HIDDEN
                            + 2 * self.generic.hca_main_elements * v4_state::HIDDEN
                            + 2 * self.generic.csa_main_elements * v4_state::HIDDEN
                            + 2 * self.generic.csa_indexer_elements * v4_state::HIDDEN,
                        4,
                    )?,
                ])?
            } else {
                0
            },
            indexer: sum(&[
                bytes(4 * 128 + self.csa_compressed_rows * 128 + 3 * 64 * 128, 2)?,
                bytes(
                    3 * self.generic.csa_indexer_elements
                        + 16 * self.generic.csa_indexer_elements
                        + 128
                        + 64 * 128
                        + 64
                        + self.csa_compressed_rows * 128
                        + self.csa_compressed_rows,
                    4,
                )?,
                bytes(self.csa_compressed_rows.min(v4_state::INDEX_TOPK), 4)?,
                bytes(self.csa_compressed_rows, 4)?,
            ])?,
            topk_composite: sum(&[
                bytes(self.topk_rows * v4_state::Q_HEAD_DIM, 2)?,
                bytes(self.topk_rows, 4)?,
            ])?,
            sparse: bytes(2 * v4_state::Q_HEADS * v4_state::Q_HEAD_DIM, 2)?,
            score_reuse: sum(&[
                bytes(
                    2 * self.topk_rows * v4_state::Q_HEADS + v4_state::Q_HEADS,
                    4,
                )?,
                bytes(v4_state::Q_HEADS, 4)?,
            ])?,
            attention_output: sum(&[
                bytes(
                    v4_state::O_GROUPS * v4_state::O_LORA_RANK
                        + v4_state::HIDDEN
                        + v4_state::HC * v4_state::HIDDEN,
                    2,
                )?,
                bytes(self.attention_f32_elements(), 4)?,
            ])?,
        })
    }

    pub(crate) fn num_bytes(self) -> Result<usize, Box<dyn Error>> {
        let add = |total: usize, elements: usize, element_bytes: usize| {
            elements
                .checked_mul(element_bytes)
                .and_then(|value| total.checked_add(value))
                .ok_or("generic stationary scratch reserve overflow")
        };
        let mut total = 0usize;
        for elements in [
            v4_state::HC * v4_state::HIDDEN,
            v4_state::HIDDEN,
            v4_state::HIDDEN,
            V4_Q_LORA_RANK,
            V4_Q_LORA_RANK,
            3 * v4_state::Q_HEADS * v4_state::Q_HEAD_DIM,
            4 * v4_state::Q_HEAD_DIM,
            4 * v4_state::Q_HEAD_DIM,
            4 * 128,
            self.csa_compressed_rows * 128,
            3 * 64 * 128,
            self.topk_rows * v4_state::Q_HEAD_DIM,
            2 * v4_state::Q_HEADS * v4_state::Q_HEAD_DIM,
            v4_state::O_GROUPS * v4_state::O_LORA_RANK,
            v4_state::HIDDEN,
            v4_state::HC * v4_state::HIDDEN,
        ] {
            total = add(total, elements, 2)?;
        }
        for elements in [
            v4_state::HIDDEN,
            v4_state::HIDDEN / v4_state::FP8_BLOCK,
            V4_Q_LORA_RANK,
            V4_Q_LORA_RANK / v4_state::FP8_BLOCK,
            v4_state::O_GROUPS * v4_state::O_LORA_RANK,
            v4_state::O_GROUPS * v4_state::O_LORA_RANK / v4_state::FP8_BLOCK,
            14,
        ] {
            total = add(total, elements, 1)?;
        }
        for elements in [
            3 * v4_state::Q_ROPE_DIM,
            v4_state::MIX + 2 * v4_state::HC + v4_state::HC * v4_state::HC,
            41 * v4_state::HC * v4_state::HC,
            V4_Q_LORA_RANK,
            v4_state::Q_HEADS * v4_state::Q_HEAD_DIM,
            v4_state::Q_HEAD_DIM,
            2 * self.generic.csa_tail_elements + 2 * self.generic.hca_tail_elements,
            3 * self.generic.csa_main_elements + 3 * self.generic.hca_main_elements,
            3 * self.generic.csa_indexer_elements + 16 * self.generic.csa_indexer_elements,
            v4_state::Q_HEAD_DIM + 128 + 64 * 128 + 64,
            self.csa_compressed_rows * 128 + self.csa_compressed_rows,
            // Score/probability rows and denominator are f32 per head; the
            // attention projection retains its independent hidden-wide f32 row.
            2 * self.topk_rows * v4_state::Q_HEADS
                + v4_state::Q_HEADS
                + self.attention_f32_elements(),
        ] {
            total = add(total, elements, 4)?;
        }
        total = add(
            total,
            self.csa_compressed_rows.min(v4_state::INDEX_TOPK) + self.topk_rows,
            4,
        )?;
        // `SparseAttentionScoreReuseScratch::valid_counts` is one u32/head.
        total = add(total, self.csa_compressed_rows + v4_state::Q_HEADS, 4)?;
        if self.strict {
            total = add(total, cublas_exact::WORKSPACE_BYTES, 1)?;
            total = add(
                total,
                v4_state::HIDDEN
                    + 2 * self.generic.hca_main_elements * v4_state::HIDDEN
                    + 2 * self.generic.csa_main_elements * v4_state::HIDDEN
                    + 2 * self.generic.csa_indexer_elements * v4_state::HIDDEN,
                4,
            )?;
        }
        if total != self.byte_breakdown()?.total()? {
            return Err("generic stationary scratch total differs from its grouped ledger".into());
        }
        Ok(total)
    }
}

#[cfg(test)]
pub(crate) fn generic_stationary_range_fits(
    source_len: usize,
    source_offset: usize,
    destination_len: usize,
    destination_offset: usize,
    count: usize,
) -> bool {
    source_offset
        .checked_add(count)
        .is_some_and(|end| end <= source_len)
        && destination_offset
            .checked_add(count)
            .is_some_and(|end| end <= destination_len)
}

pub(crate) struct DeviceExactCompressorScratch {
    pub(crate) cublas: cublas_exact::CublasExact,
    pub(crate) workspace: DeviceBuffer<u8>,
    pub(crate) input: DeviceBuffer<f32>,
    pub(crate) wkv: DeviceBuffer<f32>,
    pub(crate) wgate: DeviceBuffer<f32>,
    pub(crate) indexer_wkv: Option<DeviceBuffer<f32>>,
    pub(crate) indexer_wgate: Option<DeviceBuffer<f32>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SparseAttentionArm {
    Reference,
    ScoreReuse,
    Sm89Crossover,
}

pub(crate) const SM89_SCORE_REUSE_MIN_TOPK: usize = 4;

impl SparseAttentionArm {
    pub(crate) fn uses_score_reuse(self, topk_per_row: usize) -> bool {
        matches!(self, Self::ScoreReuse)
            || matches!(self, Self::Sm89Crossover) && topk_per_row >= SM89_SCORE_REUSE_MIN_TOPK
    }
}

pub(crate) struct DeviceTokenState {
    pub(crate) layer: usize,
    pub(crate) committed_rows: usize,
    pub(crate) compressor_width: usize,
    pub(crate) hc_hidden: DeviceBuffer<u16>,
    pub(crate) freqs: DeviceBuffer<f32>,
    pub(crate) inverse_freqs: DeviceBuffer<f32>,
    pub(crate) compressed_freqs: DeviceBuffer<f32>,
    pub(crate) mixes: DeviceBuffer<f32>,
    pub(crate) pre: DeviceBuffer<f32>,
    pub(crate) post: DeviceBuffer<f32>,
    pub(crate) comb: DeviceBuffer<f32>,
    pub(crate) trace: DeviceBuffer<f32>,
    pub(crate) collapsed: DeviceBuffer<u16>,
    pub(crate) attn_input: DeviceBuffer<u16>,
    pub(crate) fp8_7168: DeviceBuffer<u8>,
    pub(crate) fp8_7168_scales: DeviceBuffer<u8>,
    pub(crate) fp8_1536: DeviceBuffer<u8>,
    pub(crate) fp8_1536_scales: DeviceBuffer<u8>,
    pub(crate) fp8_16384: DeviceBuffer<u8>,
    pub(crate) fp8_16384_scales: DeviceBuffer<u8>,
    pub(crate) empty_u8: DeviceBuffer<u8>,
    pub(crate) empty_u16: DeviceBuffer<u16>,
    pub(crate) q_lora_f32: DeviceBuffer<f32>,
    pub(crate) q_lora_raw: DeviceBuffer<u16>,
    pub(crate) q_lora: DeviceBuffer<u16>,
    pub(crate) q_f32: DeviceBuffer<f32>,
    pub(crate) q_raw: DeviceBuffer<u16>,
    pub(crate) q_norm: DeviceBuffer<u16>,
    pub(crate) q_rope: DeviceBuffer<u16>,
    pub(crate) kv_f32: DeviceBuffer<f32>,
    pub(crate) kv_raw: DeviceBuffer<u16>,
    pub(crate) kv_norm: DeviceBuffer<u16>,
    pub(crate) kv_rope: DeviceBuffer<u16>,
    pub(crate) kv_ready: DeviceBuffer<u16>,
    pub(crate) kv_scales: DeviceBuffer<u8>,
    pub(crate) kv_cache: DeviceBuffer<u16>,
    pub(crate) compressor: DeviceCompressorRows,
    pub(crate) exact_compressor: Option<DeviceExactCompressorScratch>,
    pub(crate) indexer_compressor: Option<DeviceCompressorRows>,
    pub(crate) compressed_pool: DeviceBuffer<f32>,
    pub(crate) compressed_bf16: DeviceBuffer<u16>,
    pub(crate) compressed_norm: DeviceBuffer<u16>,
    pub(crate) compressed_rope: DeviceBuffer<u16>,
    pub(crate) compressed_ready: DeviceBuffer<u16>,
    pub(crate) compressed_scales: DeviceBuffer<u8>,
    pub(crate) compressed_cache: DeviceBuffer<u16>,
    pub(crate) indexer_pool: DeviceBuffer<f32>,
    pub(crate) indexer_bf16: DeviceBuffer<u16>,
    pub(crate) indexer_norm: DeviceBuffer<u16>,
    pub(crate) indexer_rope: DeviceBuffer<u16>,
    pub(crate) indexer_ready: DeviceBuffer<u16>,
    pub(crate) indexer_cache: DeviceBuffer<u16>,
    pub(crate) indexer_query_raw: DeviceBuffer<u16>,
    pub(crate) indexer_query_rope: DeviceBuffer<u16>,
    pub(crate) indexer_query_fp4: DeviceBuffer<u16>,
    pub(crate) indexer_query_f32: DeviceBuffer<f32>,
    pub(crate) indexer_weights_f32: DeviceBuffer<f32>,
    pub(crate) indexer_kv_f32: DeviceBuffer<f32>,
    pub(crate) indexer_scores: DeviceBuffer<f32>,
    pub(crate) indexer_valid: DeviceBuffer<u32>,
    pub(crate) indexer_selected: DeviceBuffer<i32>,
    pub(crate) topk_csa: Vec<DeviceBuffer<i32>>,
    pub(crate) attention_cache: DeviceBuffer<u16>,
    pub(crate) compressed_rows: usize,
    pub(crate) topk_raw: Vec<DeviceBuffer<i32>>,
    pub(crate) sparse: DeviceBuffer<u16>,
    pub(crate) sparse_pos0: SparseAttentionScoreReuseScratch,
    pub(crate) sparse_pos1: SparseAttentionScoreReuseScratch,
    pub(crate) sparse_csa: SparseAttentionScoreReuseScratch,
    pub(crate) sparse_inverse: DeviceBuffer<u16>,
    pub(crate) wo_a: DeviceBuffer<u16>,
    pub(crate) attention_f32: DeviceBuffer<f32>,
    pub(crate) attention: DeviceBuffer<u16>,
    pub(crate) after_attention: DeviceBuffer<u16>,
}

/// Persistent state owned solely by the MTP draft.  Unlike `DeviceTokenState`,
/// this ratio-0 layout has exactly one BF16 sliding KV allocation and no
/// compressor/indexer state, so a rejected candidate cannot mutate the teacher.
pub(crate) struct MtpDraftState {
    pub(crate) position: usize,
    pub(crate) kv_cache: DeviceBuffer<u16>,
}

impl MtpDraftState {
    pub(crate) fn new(stream: &CudaStream) -> Result<Self, Box<dyn Error>> {
        let plan =
            v4_state::mtp1_swa::mtp1_swa_attention_state_plan(v4_state::mtp1_swa::MTP1_SWA_LAYER)?;
        let kv_cache = DeviceBuffer::zeroed(stream, plan.kv_elements())?;
        if kv_cache.num_bytes() != plan.capacity_contribution_bytes()
            || kv_cache.num_bytes() != 131_072
        {
            return Err("MTP draft BF16 sliding-KV allocation drifted from its plan".into());
        }
        Ok(Self {
            position: 0,
            kv_cache,
        })
    }

    pub(crate) fn persistent_bytes(&self) -> usize {
        self.kv_cache.num_bytes()
    }

    pub(crate) fn digest(&self, stream: &CudaStream) -> Result<u64, Box<dyn Error>> {
        Ok(fnv1a_u16(&self.kv_cache.to_host_vec(stream)?))
    }

    pub(crate) fn commit_kv_row(
        &mut self,
        stream: &CudaStream,
        kv_ready: &DeviceBuffer<u16>,
    ) -> Result<(), Box<dyn Error>> {
        if kv_ready.len() != v4_state::Q_HEAD_DIM
            || self.kv_cache.len() != v4_state::KV_WINDOW * v4_state::Q_HEAD_DIM
        {
            return Err("MTP draft sliding-KV commit contract mismatch".into());
        }
        let slot =
            v4_state::mtp1_swa::mtp1_swa_attention_state_plan(v4_state::mtp1_swa::MTP1_SWA_LAYER)?
                .slot_for_position(self.position)?;
        // SAFETY: the validated plan maps one complete 512-wide row into the
        // unique in-range ring slot owned by this MTP state.
        unsafe { copy_device_row(stream, &self.kv_cache, slot, kv_ready)? };
        self.position = self
            .position
            .checked_add(1)
            .ok_or("MTP draft KV position overflow")?;
        Ok(())
    }
}

pub(crate) struct DeviceRequestRopeLut {
    pub(crate) values: DeviceBuffer<f32>,
    pub(crate) rows: usize,
}

impl DeviceRequestRopeLut {
    pub(crate) fn new(stream: &CudaStream, rows: usize) -> Result<Self, Box<dyn Error>> {
        if !(V4_REQUEST_ROPE_LUT_ROWS..=V4_SHORT_DEVICE_DECODE_MAX_ROWS).contains(&rows) {
            return Err("request RoPE LUT rows are outside the supported decode ceiling".into());
        }
        let forward = (0..rows)
            .flat_map(v4_state::compressed_rope_freqs)
            .collect::<Vec<_>>();
        let mut inverse = forward.clone();
        for sine in inverse.iter_mut().skip(1).step_by(2) {
            *sine = -*sine;
        }
        let mut values = forward;
        values.extend(inverse);
        if values.len() * std::mem::size_of::<f32>()
            != 2 * rows * v4_state::Q_ROPE_DIM * std::mem::size_of::<f32>()
        {
            return Err("request RoPE LUT byte contract mismatch".into());
        }
        Ok(Self {
            values: DeviceBuffer::from_host(stream, &values)?,
            rows,
        })
    }

    pub(crate) fn num_bytes(&self) -> usize {
        self.values.num_bytes()
    }

    pub(crate) fn copy_forward_row(
        &self,
        stream: &CudaStream,
        position: usize,
        destination: &DeviceBuffer<f32>,
    ) -> Result<(), Box<dyn Error>> {
        self.copy_row(stream, position, destination)
    }

    pub(crate) fn copy_inverse_row(
        &self,
        stream: &CudaStream,
        position: usize,
        destination: &DeviceBuffer<f32>,
    ) -> Result<(), Box<dyn Error>> {
        self.copy_row(stream, self.rows + position, destination)
    }

    pub(crate) fn copy_row(
        &self,
        stream: &CudaStream,
        position: usize,
        destination: &DeviceBuffer<f32>,
    ) -> Result<(), Box<dyn Error>> {
        if position >= 2 * self.rows || destination.len() != v4_state::Q_ROPE_DIM {
            return Err("request RoPE LUT row copy contract mismatch".into());
        }
        unsafe {
            copy_device_range(
                stream,
                destination,
                0,
                &self.values,
                position * v4_state::Q_ROPE_DIM,
                v4_state::Q_ROPE_DIM,
            )?;
        }
        Ok(())
    }
}

impl DeviceCompressorRows {
    pub(crate) fn new(
        stream: &CudaStream,
        width: usize,
        capacity_rows: usize,
    ) -> Result<Self, Box<dyn Error>> {
        Ok(Self {
            kv: DeviceBuffer::zeroed(stream, width)?,
            score: DeviceBuffer::zeroed(stream, width)?,
            score_plus_ape: DeviceBuffer::zeroed(stream, width)?,
            persistent_kv: DeviceBuffer::zeroed(stream, capacity_rows * width)?,
            persistent_score: DeviceBuffer::from_host(
                stream,
                &vec![f32::NEG_INFINITY; capacity_rows * width],
            )?,
        })
    }

    pub(crate) fn num_bytes(&self) -> usize {
        self.kv.num_bytes()
            + self.score.num_bytes()
            + self.score_plus_ape.num_bytes()
            + self.persistent_kv.num_bytes()
            + self.persistent_score.num_bytes()
    }

    pub(crate) fn reset_persistent(
        &mut self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
    ) -> Result<usize, Box<dyn Error>> {
        zero_device_buffer(stream, &self.persistent_kv)?;
        // SAFETY: one checked thread writes each persistent score element.
        unsafe {
            module.fill_f32(
                stream,
                LaunchConfig::for_num_elems(u32::try_from(self.persistent_score.len())?),
                &mut self.persistent_score,
                f32::NEG_INFINITY,
            )?;
        }
        Ok(2)
    }
}

impl DeviceLayerPersistentState {
    pub(crate) fn new(
        stream: &CudaStream,
        weights: &LayerDeviceWeights,
        plan: v4_state::LayerStationaryRequestPlan,
    ) -> Result<Self, Box<dyn Error>> {
        let path = GenericStationaryPath::from_ratio(weights.compression_ratio);
        let csa = path.is_csa();
        let compressor_width = path.compressor_width();
        let compressed_rows = path.compressed_rows(plan);
        let tail_elements = path.tail_elements();
        let compressed_elements = compressed_rows
            .checked_mul(v4_state::Q_HEAD_DIM)
            .ok_or("generic stationary compressed KV capacity overflow")?;
        let indexer_elements = plan
            .csa_compressed_rows
            .checked_mul(128)
            .ok_or("generic stationary indexer KV capacity overflow")?;
        let state = Self {
            layer: weights.layer,
            committed_rows: 0,
            compressor_width,
            raw_kv_ring: DeviceBuffer::zeroed(stream, v4_state::KV_WINDOW * v4_state::Q_HEAD_DIM)?,
            compressor_tail_kv: DeviceBuffer::zeroed(stream, tail_elements)?,
            compressor_tail_score: DeviceBuffer::from_host(
                stream,
                &vec![f32::NEG_INFINITY; tail_elements],
            )?,
            compressed_kv: DeviceBuffer::zeroed(stream, compressed_elements)?,
            indexer_tail_kv: csa
                .then(|| DeviceBuffer::zeroed(stream, 8 * 256))
                .transpose()?,
            indexer_tail_score: csa
                .then(|| DeviceBuffer::from_host(stream, &vec![f32::NEG_INFINITY; 8 * 256]))
                .transpose()?,
            indexer_kv: csa
                .then(|| DeviceBuffer::zeroed(stream, indexer_elements))
                .transpose()?,
        };
        if state.raw_kv_ring.len() != v4_state::KV_WINDOW * v4_state::Q_HEAD_DIM
            || state.compressor_tail_kv.len() != tail_elements
            || state.compressor_tail_score.len() != tail_elements
            || state.compressed_kv.len() != compressed_elements
        {
            return Err("generic stationary persistent layout drifted from request plan".into());
        }
        Ok(state)
    }

    pub(crate) fn raw_slot(&self, position: usize) -> Result<usize, Box<dyn Error>> {
        if position != self.committed_rows {
            return Err("generic stationary raw-ring commit is not contiguous".into());
        }
        Ok(position % v4_state::KV_WINDOW)
    }

    pub(crate) fn compressed_rows(&self, ratio: v4_state::CompressionRatio) -> usize {
        self.committed_rows / ratio.get()
    }

    pub(crate) fn num_bytes(&self) -> usize {
        self.raw_kv_ring.num_bytes()
            + self.compressor_tail_kv.num_bytes()
            + self.compressor_tail_score.num_bytes()
            + self.compressed_kv.num_bytes()
            + self
                .indexer_tail_kv
                .as_ref()
                .map_or(0, DeviceBuffer::num_bytes)
            + self
                .indexer_tail_score
                .as_ref()
                .map_or(0, DeviceBuffer::num_bytes)
            + self.indexer_kv.as_ref().map_or(0, DeviceBuffer::num_bytes)
    }
}
