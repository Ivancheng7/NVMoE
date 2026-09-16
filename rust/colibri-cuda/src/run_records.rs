use super::*;

#[derive(Default)]
pub(crate) struct RouterTelemetry {
    pub(crate) alloc_h2d_sync_ns: u64,
    pub(crate) workspace_alloc_ns: u64,
    pub(crate) kernel_select_sync_ns: u64,
    pub(crate) kernel_select_gpu_ns: u64,
    pub(crate) d2h_sync_ns: u64,
}

pub(crate) struct LayerRouterRun {
    pub(crate) input: Vec<u16>,
    pub(crate) routes: v4_router::RouterSelection,
    pub(crate) debug: Option<RouterDebug>,
    pub(crate) telemetry: RouterTelemetry,
}

pub(crate) struct RouterDebug {
    pub(crate) rmsnorm: Vec<f32>,
    pub(crate) rmsnorm_squares: Vec<f32>,
    pub(crate) logits: Vec<f32>,
    pub(crate) softplus: Vec<f32>,
    pub(crate) post_scores: Vec<f32>,
    pub(crate) selected_scores: Vec<f32>,
    pub(crate) denom_steps: Vec<f32>,
}

pub(crate) struct RealLayerRun {
    pub(crate) output: Vec<u16>,
    pub(crate) attention_output: Vec<u16>,
    pub(crate) routes: v4_router::RouterSelection,
    pub(crate) attention_topk: Option<Vec<i32>>,
    pub(crate) attention_topk_digest: u64,
    pub(crate) attention_topk_per_row: usize,
    pub(crate) device_transfer: Option<DeviceAttentionTransfer>,
    pub(crate) indexer_topk_gpu_ns: u64,
    pub(crate) stage: LayerStageTelemetry,
    pub(crate) critical_path_timeline: Option<CriticalPathTimeline>,
    pub(crate) routed_audit: Option<RoutedExecutionAudit>,
}

/// Gate-only routed values. The production path leaves this absent so it does
/// not retain six expert outputs after canonical reduction.
pub(crate) struct RoutedExecutionAudit {
    pub(crate) after_attention_digest: u64,
    pub(crate) ffn_input_digest: u64,
    pub(crate) shared_digest: u64,
    pub(crate) moe_digest: u64,
    pub(crate) contributions: Vec<Vec<f32>>,
}

pub(crate) struct CriticalPathTimeline {
    pub(crate) origin: std::time::Instant,
    pub(crate) fixed_ready_ns: u64,
    pub(crate) attention_start_ns: u64,
    pub(crate) attention_end_ns: u64,
    pub(crate) router_start_ns: u64,
    pub(crate) router_end_ns: u64,
    pub(crate) shared_start_ns: u64,
    pub(crate) shared_end_ns: u64,
    pub(crate) routed_start_ns: u64,
    pub(crate) routed_end_ns: u64,
    pub(crate) hc_post_start_ns: u64,
    pub(crate) hc_post_end_ns: u64,
    pub(crate) output_ready_ns: u64,
    pub(crate) attention_commit_ns: u64,
    pub(crate) routed: Option<RoutedQd4Timeline>,
}

impl CriticalPathTimeline {
    pub(crate) fn new(origin: std::time::Instant) -> Self {
        let fixed_ready_ns = timeline_elapsed_ns(origin);
        Self {
            origin,
            fixed_ready_ns,
            attention_start_ns: fixed_ready_ns,
            attention_end_ns: fixed_ready_ns,
            router_start_ns: fixed_ready_ns,
            router_end_ns: fixed_ready_ns,
            shared_start_ns: fixed_ready_ns,
            shared_end_ns: fixed_ready_ns,
            routed_start_ns: fixed_ready_ns,
            routed_end_ns: fixed_ready_ns,
            hc_post_start_ns: fixed_ready_ns,
            hc_post_end_ns: fixed_ready_ns,
            output_ready_ns: fixed_ready_ns,
            attention_commit_ns: 0,
            routed: None,
        }
    }

    pub(crate) fn now(&self) -> u64 {
        timeline_elapsed_ns(self.origin)
    }

    pub(crate) fn as_json(&self) -> serde_json::Value {
        serde_json::json!({
            "clock": "monotonic_host_ns; CUDA event durations are nested diagnostics, not cross-stream timestamps",
            "fixed_ready_ns": self.fixed_ready_ns,
            "attention": {"start_ns": self.attention_start_ns, "end_ns": self.attention_end_ns, "state_commit_ns": self.attention_commit_ns},
            "router": {"start_ns": self.router_start_ns, "end_ns": self.router_end_ns},
            "shared": {"start_ns": self.shared_start_ns, "end_ns": self.shared_end_ns},
            "routed": self.routed.as_ref().map(RoutedQd4Timeline::as_json),
            "routed_scope": {"start_ns": self.routed_start_ns, "end_ns": self.routed_end_ns},
            "hc_post": {"start_ns": self.hc_post_start_ns, "end_ns": self.hc_post_end_ns},
            "output_ready_ns": self.output_ready_ns,
        })
    }
}

pub(crate) fn timeline_elapsed_ns(origin: std::time::Instant) -> u64 {
    u64::try_from(origin.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

#[derive(Clone)]
pub(crate) struct RealAttentionRun {
    pub(crate) output: Vec<u16>,
    pub(crate) topk: Option<Vec<i32>>,
    pub(crate) stage_digests: Vec<(&'static str, u64)>,
    pub(crate) attention_topk_digest: u64,
    pub(crate) attention_topk_per_row: usize,
    pub(crate) indexer_topk_gpu_ns: u64,
    pub(crate) attention_ns: u64,
    pub(crate) commit_ns: u64,
    pub(crate) device_transfer: Option<DeviceAttentionTransfer>,
    pub(crate) profile: Option<AttentionProfile>,
}

#[derive(Clone, Default)]
pub(crate) struct DeviceAttentionTransfer {
    pub(crate) h2d_count: usize,
    pub(crate) h2d_bytes: usize,
    pub(crate) rope_lut_d2d_count: usize,
    pub(crate) rope_lut_d2d_bytes: usize,
    pub(crate) terminal_d2h_count: usize,
    pub(crate) terminal_d2h_bytes: usize,
    pub(crate) intermediate_or_audit_d2h_count: usize,
    pub(crate) intermediate_or_audit_d2h_bytes: usize,
}

pub(crate) struct DeviceCompressorRows {
    pub(crate) kv: DeviceBuffer<f32>,
    pub(crate) score: DeviceBuffer<f32>,
    pub(crate) score_plus_ape: DeviceBuffer<f32>,
    pub(crate) persistent_kv: DeviceBuffer<f32>,
    pub(crate) persistent_score: DeviceBuffer<f32>,
}

/// Per-layer state for the generic stationary path.  It intentionally owns
/// only data that survives from row p to p+1; all projections and attention
/// intermediates live in the one `DeviceRequestScratch` below.
pub(crate) struct DeviceLayerPersistentState {
    pub(crate) layer: usize,
    pub(crate) committed_rows: usize,
    pub(crate) compressor_width: usize,
    pub(crate) raw_kv_ring: DeviceBuffer<u16>,
    pub(crate) compressor_tail_kv: DeviceBuffer<f32>,
    pub(crate) compressor_tail_score: DeviceBuffer<f32>,
    pub(crate) compressed_kv: DeviceBuffer<u16>,
    pub(crate) indexer_tail_kv: Option<DeviceBuffer<f32>>,
    pub(crate) indexer_tail_score: Option<DeviceBuffer<f32>>,
    pub(crate) indexer_kv: Option<DeviceBuffer<u16>>,
}

/// Exactly one request-local GPU workspace.  This is deliberately separate
/// from `DeviceTokenState`, which remains the frozen short/audit fixture.
pub(crate) struct DeviceRequestScratch {
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
    pub(crate) candidate_tail_kv: DeviceBuffer<f32>,
    pub(crate) candidate_tail_score: DeviceBuffer<f32>,
    pub(crate) candidate_hca_tail_kv: DeviceBuffer<f32>,
    pub(crate) candidate_hca_tail_score: DeviceBuffer<f32>,
    pub(crate) compressor_kv: DeviceBuffer<f32>,
    pub(crate) compressor_score: DeviceBuffer<f32>,
    pub(crate) compressor_score_plus_ape: DeviceBuffer<f32>,
    pub(crate) hca_compressor_kv: DeviceBuffer<f32>,
    pub(crate) hca_compressor_score: DeviceBuffer<f32>,
    pub(crate) hca_compressor_score_plus_ape: DeviceBuffer<f32>,
    pub(crate) exact_compressor: Option<GenericExactCompressorScratch>,
    pub(crate) indexer_compressor_kv: Option<DeviceBuffer<f32>>,
    pub(crate) indexer_compressor_score: Option<DeviceBuffer<f32>>,
    pub(crate) indexer_compressor_score_plus_ape: Option<DeviceBuffer<f32>>,
    pub(crate) candidate_indexer_tail_kv: Option<DeviceBuffer<f32>>,
    pub(crate) candidate_indexer_tail_score: Option<DeviceBuffer<f32>>,
    pub(crate) compressed_pool: DeviceBuffer<f32>,
    pub(crate) compressed_bf16: DeviceBuffer<u16>,
    pub(crate) compressed_norm: DeviceBuffer<u16>,
    pub(crate) compressed_rope: DeviceBuffer<u16>,
    pub(crate) compressed_ready: DeviceBuffer<u16>,
    pub(crate) compressed_scales: DeviceBuffer<u8>,
    pub(crate) indexer_pool: DeviceBuffer<f32>,
    pub(crate) indexer_bf16: DeviceBuffer<u16>,
    pub(crate) indexer_norm: DeviceBuffer<u16>,
    pub(crate) indexer_rope: DeviceBuffer<u16>,
    pub(crate) indexer_ready: DeviceBuffer<u16>,
    pub(crate) candidate_indexer_cache: DeviceBuffer<u16>,
    pub(crate) indexer_query_raw: DeviceBuffer<u16>,
    pub(crate) indexer_query_rope: DeviceBuffer<u16>,
    pub(crate) indexer_query_fp4: DeviceBuffer<u16>,
    pub(crate) indexer_query_f32: DeviceBuffer<f32>,
    pub(crate) indexer_weights_f32: DeviceBuffer<f32>,
    pub(crate) indexer_kv_f32: DeviceBuffer<f32>,
    pub(crate) indexer_scores: DeviceBuffer<f32>,
    pub(crate) indexer_valid: DeviceBuffer<u32>,
    pub(crate) indexer_selected: DeviceBuffer<i32>,
    pub(crate) topk: DeviceBuffer<i32>,
    pub(crate) attention_cache: DeviceBuffer<u16>,
    pub(crate) sparse: DeviceBuffer<u16>,
    pub(crate) sparse_scratch: SparseAttentionScoreReuseScratch,
    pub(crate) sparse_inverse: DeviceBuffer<u16>,
    pub(crate) wo_a: DeviceBuffer<u16>,
    pub(crate) attention_f32: DeviceBuffer<f32>,
    pub(crate) attention: DeviceBuffer<u16>,
    pub(crate) after_attention: DeviceBuffer<u16>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GenericStationaryPath {
    Hca128,
    Csa4,
}

impl GenericStationaryPath {
    pub(crate) const fn from_ratio(ratio: v4_state::CompressionRatio) -> Self {
        match ratio {
            v4_state::CompressionRatio::WindowOnly => {
                panic!("generic stationary path does not execute window-only layers")
            }
            v4_state::CompressionRatio::Hca128 => Self::Hca128,
            v4_state::CompressionRatio::Csa4 => Self::Csa4,
        }
    }

    pub(crate) const fn is_csa(self) -> bool {
        matches!(self, Self::Csa4)
    }

    pub(crate) const fn ratio(self) -> usize {
        match self {
            Self::Hca128 => 128,
            Self::Csa4 => 4,
        }
    }

    pub(crate) const fn compressor_width(self) -> usize {
        match self {
            Self::Hca128 => v4_state::Q_HEAD_DIM,
            Self::Csa4 => 1_024,
        }
    }

    pub(crate) const fn tail_rows(self) -> usize {
        match self {
            Self::Hca128 => v4_state::KV_WINDOW,
            Self::Csa4 => 8,
        }
    }

    pub(crate) const fn tail_elements(self) -> usize {
        self.tail_rows() * self.compressor_width()
    }

    pub(crate) const fn compressor_slot(self, position: usize) -> usize {
        match self {
            Self::Hca128 => position % v4_state::KV_WINDOW,
            Self::Csa4 => 4 + position % 4,
        }
    }

    pub(crate) const fn topk_rows(self, compressed_rows: usize) -> usize {
        v4_state::KV_WINDOW
            + match self {
                Self::Hca128 => compressed_rows,
                Self::Csa4 if compressed_rows < v4_state::INDEX_TOPK => compressed_rows,
                Self::Csa4 => v4_state::INDEX_TOPK,
            }
    }

    /// Scratch capacity, not the number of causal keys.  Before the raw ring
    /// fills, `window_topk_cpu` encodes uncommitted slots as `-1`.
    pub(crate) const fn composite_capacity_rows(self, position: usize) -> usize {
        self.topk_rows((position + 1) / self.ratio())
    }

    pub(crate) const fn max_topk_rows(self, plan: v4_state::LayerStationaryRequestPlan) -> usize {
        self.topk_rows(self.compressed_rows(plan))
    }

    pub(crate) const fn compressed_rows(self, plan: v4_state::LayerStationaryRequestPlan) -> usize {
        match self {
            Self::Hca128 => plan.hca_compressed_rows,
            Self::Csa4 => plan.csa_compressed_rows,
        }
    }
}

/// Element counts for the one request-shared generic stationary workspace.
/// The HCA and CSA rows/tails coexist, but every request path selects exactly
/// one row and one candidate tail from this table.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GenericStationaryScratchGeometry {
    pub(crate) hca_main_elements: usize,
    pub(crate) csa_main_elements: usize,
    pub(crate) csa_indexer_elements: usize,
    pub(crate) hca_tail_elements: usize,
    pub(crate) csa_tail_elements: usize,
}

impl GenericStationaryScratchGeometry {
    pub(crate) const fn max() -> Self {
        Self {
            hca_main_elements: v4_state::Q_HEAD_DIM,
            csa_main_elements: 1_024,
            csa_indexer_elements: 256,
            hca_tail_elements: GenericStationaryPath::Hca128.tail_elements(),
            csa_tail_elements: GenericStationaryPath::Csa4.tail_elements(),
        }
    }

    pub(crate) const fn main_elements(self, path: GenericStationaryPath) -> usize {
        match path {
            GenericStationaryPath::Hca128 => self.hca_main_elements,
            GenericStationaryPath::Csa4 => self.csa_main_elements,
        }
    }

    pub(crate) const fn tail_elements(self, path: GenericStationaryPath) -> usize {
        match path {
            GenericStationaryPath::Hca128 => self.hca_tail_elements,
            GenericStationaryPath::Csa4 => self.csa_tail_elements,
        }
    }
}

/// The one source of truth for the request-shared generic scratch.  Both the
/// pre-cache reservation and `DeviceRequestScratch::new` consume this exact
/// geometry, so a Full61 strict run cannot silently diverge from its ledger.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct DeviceRequestScratchGeometry {
    pub(crate) generic: GenericStationaryScratchGeometry,
    pub(crate) csa_compressed_rows: usize,
    pub(crate) topk_rows: usize,
    pub(crate) strict: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct DeviceRequestScratchByteBreakdown {
    pub(crate) core_mhc: usize,
    pub(crate) projections: usize,
    pub(crate) compressor: usize,
    pub(crate) exact_compressor: usize,
    pub(crate) indexer: usize,
    pub(crate) topk_composite: usize,
    pub(crate) sparse: usize,
    pub(crate) score_reuse: usize,
    pub(crate) attention_output: usize,
}

impl DeviceRequestScratchByteBreakdown {
    pub(crate) fn total(self) -> Result<usize, Box<dyn Error>> {
        [
            self.core_mhc,
            self.projections,
            self.compressor,
            self.exact_compressor,
            self.indexer,
            self.topk_composite,
            self.sparse,
            self.score_reuse,
            self.attention_output,
        ]
        .into_iter()
        .try_fold(0usize, |total, value| {
            total
                .checked_add(value)
                .ok_or_else(|| -> Box<dyn Error> { "generic scratch breakdown overflow".into() })
        })
    }

    pub(crate) fn json(self) -> serde_json::Value {
        serde_json::json!({
            "core_mhc": self.core_mhc,
            "projections": self.projections,
            "compressor": self.compressor,
            "exact_compressor": self.exact_compressor,
            "indexer": self.indexer,
            "topk_composite": self.topk_composite,
            "sparse": self.sparse,
            "score_reuse": self.score_reuse,
            "attention_output": self.attention_output,
        })
    }
}
