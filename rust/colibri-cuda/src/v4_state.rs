//! Pinned V4-Pro layer-0 state: embedding ranges, mHC pre-mix, and attn norm.

use std::error::Error;
use std::os::unix::fs::FileTypeExt;
use std::os::unix::fs::MetadataExt;
use std::path::Path;

use crate::safetensors::{
    IndexedSafetensors, ShardFile, SourceManifest, TensorDtype, TensorLocation, TensorRange,
};
use crate::v4_router::{bf16_to_f32, f32_to_bf16_rne, rmsnorm_bf16, rmsnorm_bf16_width};

pub const REVISION: &str = "b5968e9190ef611bbf34a7229255be88a0e937c1";
pub const PINNED_SHARD_BYTES: u64 = 864_721_029_744;
pub const V4_LAYER_COUNT: usize = 61;
pub const HIDDEN: usize = 7_168;
pub const VOCAB: usize = 129_280;
pub const HC: usize = 4;
pub const MIX: usize = 24;
pub const IDS: [i64; 4] = [0, 1, 2, 129_279];
const EPS: f32 = 1e-6;
pub const FP8_BLOCK: usize = 128;
pub const Q_HEADS: usize = 128;
pub const Q_HEAD_DIM: usize = 512;
pub const Q_ROPE_DIM: usize = 64;
pub const O_GROUPS: usize = 16;
pub const O_LORA_RANK: usize = 1_024;
pub const KV_WINDOW: usize = 128;
pub const KV_COMPRESSED: usize = 32;
pub const KV_CACHE_POSITIONS: usize = KV_WINDOW + KV_COMPRESSED;
pub const COMPRESS_RATIO: usize = 128;
pub const COMPRESS_STATE_ELEMENTS: usize = COMPRESS_RATIO * Q_HEAD_DIM;
pub const INDEX_TOPK: usize = 1_024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompressionRatio {
    Csa4,
    Hca128,
}

impl CompressionRatio {
    pub const fn get(self) -> usize {
        match self {
            Self::Csa4 => 4,
            Self::Hca128 => 128,
        }
    }
}

pub fn compression_ratio_for_layer(layer: usize) -> Result<CompressionRatio, Box<dyn Error>> {
    if layer >= V4_LAYER_COUNT {
        return Err(format!("V4 layer {layer} is outside 0..{V4_LAYER_COUNT}").into());
    }
    Ok(if layer >= 2 && layer.is_multiple_of(2) {
        CompressionRatio::Csa4
    } else {
        CompressionRatio::Hca128
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerTensorContract {
    pub name: String,
    pub dtype: TensorDtype,
    pub shape: Vec<u64>,
    pub byte_len: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerExecutionManifest {
    pub layer: usize,
    pub compression_ratio: CompressionRatio,
    pub shard: String,
    pub tensors: Vec<LayerTensorContract>,
    pub contract_range_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerTelemetry {
    pub layer: usize,
    pub compression_ratio: CompressionRatio,
    pub source_range_bytes: u64,
    pub indexer_topk_cpu_ns: u64,
    pub layer_elapsed_ns: u64,
    pub kv_rows_before: usize,
    pub kv_rows_after: usize,
}

pub fn layer_tensor_manifest(
    layer: usize,
    expert_id: u16,
) -> Result<Vec<LayerTensorContract>, Box<dyn Error>> {
    let ratio = compression_ratio_for_layer(layer)?;
    if expert_id as usize >= crate::v4_router::V4_ROUTED_EXPERTS {
        return Err(format!("invalid routed expert id {expert_id}").into());
    }
    let prefix = format!("layers.{layer}");
    let mut tensors = vec![
        LayerTensorContract {
            name: format!("{prefix}.hc_attn_fn"),
            dtype: TensorDtype::F32,
            shape: vec![24, 28_672],
            byte_len: 2_752_512,
        },
        LayerTensorContract {
            name: format!("{prefix}.hc_attn_base"),
            dtype: TensorDtype::F32,
            shape: vec![24],
            byte_len: 96,
        },
        LayerTensorContract {
            name: format!("{prefix}.hc_attn_scale"),
            dtype: TensorDtype::F32,
            shape: vec![3],
            byte_len: 12,
        },
        LayerTensorContract {
            name: format!("{prefix}.attn_norm.weight"),
            dtype: TensorDtype::Bf16,
            shape: vec![7_168],
            byte_len: 14_336,
        },
        LayerTensorContract {
            name: format!("{prefix}.attn.wq_a.weight"),
            dtype: TensorDtype::F8E4M3,
            shape: vec![1_536, 7_168],
            byte_len: 11_010_048,
        },
        LayerTensorContract {
            name: format!("{prefix}.attn.compressor.ape"),
            dtype: TensorDtype::F32,
            shape: match ratio {
                CompressionRatio::Csa4 => vec![4, 1_024],
                CompressionRatio::Hca128 => vec![128, 512],
            },
            byte_len: match ratio {
                CompressionRatio::Csa4 => 16_384,
                CompressionRatio::Hca128 => 262_144,
            },
        },
        LayerTensorContract {
            name: format!("{prefix}.attn.compressor.wkv.weight"),
            dtype: TensorDtype::Bf16,
            shape: vec![
                match ratio {
                    CompressionRatio::Csa4 => 1_024,
                    CompressionRatio::Hca128 => 512,
                },
                7_168,
            ],
            byte_len: match ratio {
                CompressionRatio::Csa4 => 14_680_064,
                CompressionRatio::Hca128 => 7_340_032,
            },
        },
        LayerTensorContract {
            name: format!("{prefix}.attn.compressor.wgate.weight"),
            dtype: TensorDtype::Bf16,
            shape: vec![
                match ratio {
                    CompressionRatio::Csa4 => 1_024,
                    CompressionRatio::Hca128 => 512,
                },
                7_168,
            ],
            byte_len: match ratio {
                CompressionRatio::Csa4 => 14_680_064,
                CompressionRatio::Hca128 => 7_340_032,
            },
        },
        LayerTensorContract {
            name: format!("{prefix}.attn.compressor.norm.weight"),
            dtype: TensorDtype::Bf16,
            shape: vec![512],
            byte_len: 1_024,
        },
        LayerTensorContract {
            name: format!("{prefix}.ffn_norm.weight"),
            dtype: TensorDtype::Bf16,
            shape: vec![7_168],
            byte_len: 14_336,
        },
        LayerTensorContract {
            name: format!("{prefix}.ffn.gate.weight"),
            dtype: TensorDtype::Bf16,
            shape: vec![384, 7_168],
            byte_len: 5_505_024,
        },
        LayerTensorContract {
            name: format!("{prefix}.ffn.experts.{expert_id}.w1.weight"),
            dtype: TensorDtype::Fp4E2M1x2,
            shape: vec![3_072, 3_584],
            byte_len: 11_010_048,
        },
    ];
    if ratio == CompressionRatio::Csa4 {
        tensors.extend([
            LayerTensorContract {
                name: format!("{prefix}.attn.indexer.wq_b.weight"),
                dtype: TensorDtype::F8E4M3,
                shape: vec![8_192, 1_536],
                byte_len: 12_582_912,
            },
            LayerTensorContract {
                name: format!("{prefix}.attn.indexer.wq_b.scale"),
                dtype: TensorDtype::F8E8M0,
                shape: vec![64, 12],
                byte_len: 768,
            },
            LayerTensorContract {
                name: format!("{prefix}.attn.indexer.weights_proj.weight"),
                dtype: TensorDtype::Bf16,
                shape: vec![64, 7_168],
                byte_len: 917_504,
            },
            LayerTensorContract {
                name: format!("{prefix}.attn.indexer.compressor.ape"),
                dtype: TensorDtype::F32,
                shape: vec![4, 256],
                byte_len: 4_096,
            },
            LayerTensorContract {
                name: format!("{prefix}.attn.indexer.compressor.wkv.weight"),
                dtype: TensorDtype::Bf16,
                shape: vec![256, 7_168],
                byte_len: 3_670_016,
            },
            LayerTensorContract {
                name: format!("{prefix}.attn.indexer.compressor.wgate.weight"),
                dtype: TensorDtype::Bf16,
                shape: vec![256, 7_168],
                byte_len: 3_670_016,
            },
            LayerTensorContract {
                name: format!("{prefix}.attn.indexer.compressor.norm.weight"),
                dtype: TensorDtype::Bf16,
                shape: vec![128],
                byte_len: 256,
            },
        ]);
    }
    if layer < 3 {
        tensors.push(LayerTensorContract {
            name: format!("{prefix}.ffn.gate.tid2eid"),
            dtype: TensorDtype::I64,
            shape: vec![129_280, 6],
            byte_len: 6_205_440,
        });
    } else {
        tensors.push(LayerTensorContract {
            name: format!("{prefix}.ffn.gate.bias"),
            dtype: TensorDtype::F32,
            shape: vec![384],
            byte_len: 1_536,
        });
    }
    Ok(tensors)
}

pub fn serial_layer_manifests(
    expert_id: u16,
) -> Result<Vec<LayerExecutionManifest>, Box<dyn Error>> {
    (0..V4_LAYER_COUNT)
        .map(|layer| {
            let tensors = layer_tensor_manifest(layer, expert_id)?;
            Ok(LayerExecutionManifest {
                layer,
                compression_ratio: compression_ratio_for_layer(layer)?,
                shard: format!("model-{:05}-of-00064.safetensors", layer + 2),
                contract_range_bytes: tensors.iter().map(|tensor| tensor.byte_len).sum(),
                tensors,
            })
        })
        .collect()
}

pub fn run_serial_layer_slice<F>(
    initial_hidden: Vec<u16>,
    manifests: &[LayerExecutionManifest],
    mut run_layer: F,
) -> Result<(Vec<u16>, Vec<LayerTelemetry>), Box<dyn Error>>
where
    F: FnMut(&LayerExecutionManifest, &[u16]) -> Result<(Vec<u16>, u64, u64), Box<dyn Error>>,
{
    if initial_hidden.is_empty()
        || manifests.is_empty()
        || manifests
            .windows(2)
            .any(|pair| pair[1].layer != pair[0].layer + 1)
    {
        return Err("V4 serial layer slice must be non-empty and contiguous".into());
    }
    let mut hidden = initial_hidden;
    let mut telemetry = Vec::with_capacity(manifests.len());
    for manifest in manifests {
        let (next_hidden, source_range_bytes, indexer_topk_cpu_ns) = run_layer(manifest, &hidden)?;
        if next_hidden.len() != hidden.len() || source_range_bytes != manifest.contract_range_bytes
        {
            return Err(format!("V4 layer {} commit contract mismatch", manifest.layer).into());
        }
        hidden = next_hidden;
        telemetry.push(LayerTelemetry {
            layer: manifest.layer,
            compression_ratio: manifest.compression_ratio,
            source_range_bytes,
            indexer_topk_cpu_ns,
            layer_elapsed_ns: 0,
            kv_rows_before: 0,
            kv_rows_after: 0,
        });
    }
    Ok((hidden, telemetry))
}

pub fn indexed_model(model_dir: &Path) -> Result<IndexedSafetensors, Box<dyn Error>> {
    validate_model_dir(model_dir)?;
    let shards = (1..=64).map(|number| {
        let name = format!("model-{number:05}-of-00064.safetensors");
        (name.clone(), model_dir.join(name))
    });
    let manifest = SourceManifest::new(REVISION, shards);
    Ok(IndexedSafetensors::from_index_reader(
        manifest,
        std::fs::File::open(model_dir.join("model.safetensors.index.json"))?,
    )?)
}

pub fn resolve_layer_tensor(
    index: &IndexedSafetensors,
    tensor: &LayerTensorContract,
) -> Result<TensorLocation, Box<dyn Error>> {
    Ok(index.resolve_tensor(&tensor.name)?)
}

pub fn resolve_layer_tensor_range(
    index: &IndexedSafetensors,
    tensor: &LayerTensorContract,
) -> Result<TensorRange, Box<dyn Error>> {
    let resolved = index.resolve_tensor_range(&tensor.name)?;
    if resolved.info.dtype != tensor.dtype
        || resolved.info.shape != tensor.shape
        || resolved.info.byte_len != tensor.byte_len
    {
        return Err(format!("{} pinned tensor range contract mismatch", tensor.name).into());
    }
    Ok(resolved)
}

fn validate_layer_locations(
    index: &IndexedSafetensors,
    tensors: &[LayerTensorContract],
    expected_shard: &str,
) -> Result<(), Box<dyn Error>> {
    for tensor in tensors {
        let location = resolve_layer_tensor(index, tensor)?;
        if location.shard != expected_shard {
            return Err(format!(
                "{} is in {}, expected {expected_shard}",
                tensor.name, location.shard
            )
            .into());
        }
    }
    Ok(())
}

struct TensorContract {
    name: &'static str,
    shard: &'static str,
    dtype: TensorDtype,
    shape: &'static [u64],
    bytes: u64,
}
const C: &[TensorContract] = &[
    TensorContract {
        name: "embed.weight",
        shard: "model-00001-of-00064.safetensors",
        dtype: TensorDtype::Bf16,
        shape: &[129280, 7168],
        bytes: 1853358080,
    },
    TensorContract {
        name: "layers.0.hc_attn_base",
        shard: "model-00002-of-00064.safetensors",
        dtype: TensorDtype::F32,
        shape: &[24],
        bytes: 96,
    },
    TensorContract {
        name: "layers.0.hc_ffn_base",
        shard: "model-00002-of-00064.safetensors",
        dtype: TensorDtype::F32,
        shape: &[24],
        bytes: 96,
    },
    TensorContract {
        name: "layers.0.hc_attn_fn",
        shard: "model-00002-of-00064.safetensors",
        dtype: TensorDtype::F32,
        shape: &[24, 28672],
        bytes: 2752512,
    },
    TensorContract {
        name: "layers.0.hc_ffn_fn",
        shard: "model-00002-of-00064.safetensors",
        dtype: TensorDtype::F32,
        shape: &[24, 28672],
        bytes: 2752512,
    },
    TensorContract {
        name: "layers.0.hc_attn_scale",
        shard: "model-00002-of-00064.safetensors",
        dtype: TensorDtype::F32,
        shape: &[3],
        bytes: 12,
    },
    TensorContract {
        name: "layers.0.hc_ffn_scale",
        shard: "model-00002-of-00064.safetensors",
        dtype: TensorDtype::F32,
        shape: &[3],
        bytes: 12,
    },
    TensorContract {
        name: "layers.0.attn.attn_sink",
        shard: "model-00002-of-00064.safetensors",
        dtype: TensorDtype::F32,
        shape: &[128],
        bytes: 512,
    },
    TensorContract {
        name: "layers.0.attn.wq_a.weight",
        shard: "model-00002-of-00064.safetensors",
        dtype: TensorDtype::F8E4M3,
        shape: &[1536, 7168],
        bytes: 11010048,
    },
    TensorContract {
        name: "layers.0.attn.wq_a.scale",
        shard: "model-00002-of-00064.safetensors",
        dtype: TensorDtype::F8E8M0,
        shape: &[12, 56],
        bytes: 672,
    },
    TensorContract {
        name: "layers.0.attn.wq_b.weight",
        shard: "model-00002-of-00064.safetensors",
        dtype: TensorDtype::F8E4M3,
        shape: &[65536, 1536],
        bytes: 100663296,
    },
    TensorContract {
        name: "layers.0.attn.wq_b.scale",
        shard: "model-00002-of-00064.safetensors",
        dtype: TensorDtype::F8E8M0,
        shape: &[512, 12],
        bytes: 6144,
    },
    TensorContract {
        name: "layers.0.attn.q_norm.weight",
        shard: "model-00002-of-00064.safetensors",
        dtype: TensorDtype::Bf16,
        shape: &[1536],
        bytes: 3072,
    },
    TensorContract {
        name: "layers.0.attn.wkv.weight",
        shard: "model-00002-of-00064.safetensors",
        dtype: TensorDtype::F8E4M3,
        shape: &[512, 7168],
        bytes: 3670016,
    },
    TensorContract {
        name: "layers.0.attn.wkv.scale",
        shard: "model-00002-of-00064.safetensors",
        dtype: TensorDtype::F8E8M0,
        shape: &[4, 56],
        bytes: 224,
    },
    TensorContract {
        name: "layers.0.attn.kv_norm.weight",
        shard: "model-00002-of-00064.safetensors",
        dtype: TensorDtype::Bf16,
        shape: &[512],
        bytes: 1024,
    },
    TensorContract {
        name: "layers.0.attn.wo_a.weight",
        shard: "model-00002-of-00064.safetensors",
        dtype: TensorDtype::F8E4M3,
        shape: &[16384, 4096],
        bytes: 67108864,
    },
    TensorContract {
        name: "layers.0.attn.wo_a.scale",
        shard: "model-00002-of-00064.safetensors",
        dtype: TensorDtype::F8E8M0,
        shape: &[128, 32],
        bytes: 4096,
    },
    TensorContract {
        name: "layers.0.attn.wo_b.weight",
        shard: "model-00002-of-00064.safetensors",
        dtype: TensorDtype::F8E4M3,
        shape: &[7168, 16384],
        bytes: 117440512,
    },
    TensorContract {
        name: "layers.0.attn.wo_b.scale",
        shard: "model-00002-of-00064.safetensors",
        dtype: TensorDtype::F8E8M0,
        shape: &[56, 128],
        bytes: 7168,
    },
    TensorContract {
        name: "layers.0.attn.compressor.ape",
        shard: "model-00002-of-00064.safetensors",
        dtype: TensorDtype::F32,
        shape: &[128, 512],
        bytes: 262144,
    },
    TensorContract {
        name: "layers.0.attn.compressor.wkv.weight",
        shard: "model-00002-of-00064.safetensors",
        dtype: TensorDtype::Bf16,
        shape: &[512, 7168],
        bytes: 7340032,
    },
    TensorContract {
        name: "layers.0.attn.compressor.wgate.weight",
        shard: "model-00002-of-00064.safetensors",
        dtype: TensorDtype::Bf16,
        shape: &[512, 7168],
        bytes: 7340032,
    },
    TensorContract {
        name: "layers.0.attn.compressor.norm.weight",
        shard: "model-00002-of-00064.safetensors",
        dtype: TensorDtype::Bf16,
        shape: &[512],
        bytes: 1024,
    },
    TensorContract {
        name: "layers.0.attn_norm.weight",
        shard: "model-00002-of-00064.safetensors",
        dtype: TensorDtype::Bf16,
        shape: &[7168],
        bytes: 14336,
    },
    TensorContract {
        name: "layers.0.ffn_norm.weight",
        shard: "model-00002-of-00064.safetensors",
        dtype: TensorDtype::Bf16,
        shape: &[7168],
        bytes: 14336,
    },
    TensorContract {
        name: "layers.0.ffn.gate.weight",
        shard: "model-00002-of-00064.safetensors",
        dtype: TensorDtype::Bf16,
        shape: &[384, 7168],
        bytes: 5505024,
    },
    TensorContract {
        name: "layers.0.ffn.gate.tid2eid",
        shard: "model-00002-of-00064.safetensors",
        dtype: TensorDtype::I64,
        shape: &[129280, 6],
        bytes: 6205440,
    },
];

pub struct L0State {
    pub embedding: Vec<u16>,
    pub hc_fn: Vec<f32>,
    pub hc_base: Vec<f32>,
    pub hc_scale: Vec<f32>,
    pub attn_norm: Vec<u16>,
    pub range_bytes: usize,
}

pub struct LayerCompressorWeights {
    pub wkv_bf16: Vec<u16>,
    pub wgate_bf16: Vec<u16>,
    pub ape_f32: Vec<f32>,
    pub norm_bf16: Vec<u16>,
}

pub struct CsaIndexerWeights {
    pub wq_b: Fp8Matrix,
    pub weights_proj_bf16: Vec<u16>,
    pub compressor: LayerCompressorWeights,
}

pub enum LayerRouterWeights {
    Hash { ids: Vec<i64> },
    Learned { bias: Vec<f32> },
}

pub struct LayerStateWeights {
    pub layer: usize,
    pub compression_ratio: CompressionRatio,
    pub hc_fn: Vec<f32>,
    pub hc_base: Vec<f32>,
    pub hc_scale: Vec<f32>,
    pub attn_norm: Vec<u16>,
    pub compressor: LayerCompressorWeights,
    pub indexer: Option<CsaIndexerWeights>,
    pub ffn_norm: Vec<u16>,
    pub gate: Vec<u16>,
    pub router: LayerRouterWeights,
}

pub struct MhcPre {
    pub pre: Vec<f32>,
    pub collapsed: Vec<u16>,
    pub post: Vec<f32>,
    pub comb: Vec<f32>,
    /// Per-row [comb logits, softmax, col 0, row 0, ..., col 19].
    pub sinkhorn_trace: Vec<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct E4M3Fn(pub u8);

impl E4M3Fn {
    pub fn decode(self) -> f32 {
        let sign = self.0 & 0x80 != 0;
        let magnitude = self.0 & 0x7f;
        if magnitude == 0x7f {
            return f32::NAN;
        }
        let exponent = magnitude >> 3;
        let mantissa = magnitude & 7;
        let value = if exponent == 0 {
            mantissa as f32 * 0.001953125
        } else {
            (1.0 + mantissa as f32 / 8.0)
                * f32::from_bits(((exponent as i32 - 7 + 127) as u32) << 23)
        };
        if sign { -value } else { value }
    }

    pub fn encode(value: f32) -> Result<Self, Box<dyn Error>> {
        if !value.is_finite() {
            return Err("E4M3FN input must be finite".into());
        }
        let negative = value.is_sign_negative();
        let value = value.abs().min(448.0);
        let mut best = 0_u8;
        let mut distance = f32::INFINITY;
        for bits in 0_u8..=126 {
            let candidate = Self(bits).decode();
            let candidate_distance = (value - candidate).abs();
            if candidate_distance < distance
                || (candidate_distance == distance && bits & 1 == 0 && best & 1 != 0)
            {
                best = bits;
                distance = candidate_distance;
            }
        }
        Ok(Self(best | if negative { 0x80 } else { 0 }))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct E8M0Fnu(pub u8);

impl E8M0Fnu {
    pub fn decode(self) -> f32 {
        match self.0 {
            0 => f32::from_bits(1 << 22),
            255 => f32::NAN,
            exponent => f32::from_bits((exponent as u32) << 23),
        }
    }

    fn encode_power_of_two(value: f32) -> Result<Self, Box<dyn Error>> {
        if value == f32::from_bits(1 << 22) {
            return Ok(Self(0));
        }
        let bits = value.to_bits();
        let exponent = ((bits >> 23) & 0xff) as u8;
        if !value.is_finite() || value <= 0.0 || bits & 0x7f_ffff != 0 || exponent == 255 {
            return Err("E8M0 scale must be a finite representable power of two".into());
        }
        Ok(Self(exponent))
    }
}

pub struct Fp8Matrix {
    pub weight: Vec<u8>,
    pub scale: Vec<u8>,
    pub in_features: usize,
    pub out_features: usize,
}

pub struct Fp8Projection {
    pub matrix: Fp8Matrix,
    pub norm: Vec<u16>,
}

pub struct L0Fp8Projections {
    pub wq_a: Fp8Projection,
    pub wq_b: Fp8Matrix,
    pub wkv: Fp8Projection,
    pub range_bytes: usize,
}

pub struct L0SharedExpert {
    pub w1: Fp8Matrix,
    pub w2: Fp8Matrix,
    pub w3: Fp8Matrix,
}

pub struct L0TailWeights {
    pub attn_sink: Vec<f32>,
    pub wo_a_bf16: Vec<u16>,
    pub wo_a_raw: Option<Vec<u8>>,
    pub wo_b: Fp8Matrix,
    pub hc_ffn_fn: Vec<f32>,
    pub hc_ffn_base: Vec<f32>,
    pub hc_ffn_scale: Vec<f32>,
    pub ffn_norm: Vec<u16>,
    pub gate: Vec<u16>,
    pub shared: L0SharedExpert,
    pub source_range_bytes: usize,
    pub load_elapsed_ms: f64,
    pub wo_a_decode_ms: f64,
}

pub struct QuantizedFp8Activation {
    pub values: Vec<u8>,
    pub scales: Vec<u8>,
    pub rows: usize,
    pub in_features: usize,
}

pub struct Fp8LinearOutput {
    pub raw_f32: Vec<f32>,
    pub bf16: Vec<u16>,
}

pub struct Fp8ProjectionOutput {
    pub raw_f32: Vec<f32>,
    pub bf16: Vec<u16>,
    pub norm_bf16: Vec<u16>,
}

pub struct Fp8ProjectionFixture {
    pub input_bf16: Vec<u16>,
    pub activation: Vec<u8>,
    pub activation_scale: Vec<u8>,
    pub wq_f32: Vec<f32>,
    pub wq_bf16: Vec<u16>,
    pub q_norm_bf16: Vec<u16>,
    pub wkv_f32: Vec<f32>,
    pub wkv_bf16: Vec<u16>,
    pub kv_norm_bf16: Vec<u16>,
    pub wq_b_activation: Vec<u8>,
    pub wq_b_activation_scale: Vec<u8>,
    pub wq_b_f32: Vec<f32>,
    pub wq_b_bf16: Vec<u16>,
    pub q_head_rms_bf16: Vec<u16>,
    pub freqs_cis_f32_pairs: Vec<f32>,
    pub q_rope_bf16: Vec<u16>,
    pub kv_rope_bf16: Vec<u16>,
    pub kv_scale_e8m0: Vec<u8>,
    pub kv_ready_bf16: Vec<u16>,
    pub causal_topk_i32: Vec<i32>,
    pub cache_initial_bf16: Vec<u16>,
    pub cache_after_bf16: Vec<u16>,
    pub compressor_kv_f32: Vec<f32>,
    pub compressor_score_f32: Vec<f32>,
    pub compressor_score_plus_ape_f32: Vec<f32>,
    pub compressor_kv_state_initial_f32: Vec<f32>,
    pub compressor_score_state_initial_f32: Vec<f32>,
    pub compressor_kv_state_after_f32: Vec<f32>,
    pub compressor_score_state_after_f32: Vec<f32>,
}

pub struct KvReadyOutput {
    pub rope_bf16: Vec<u16>,
    pub scale_e8m0: Vec<u8>,
    pub ready_bf16: Vec<u16>,
    pub causal_topk_i32: Vec<i32>,
}

#[derive(Clone)]
pub struct V4KvCache {
    data: Vec<u16>,
    committed: bool,
}

pub struct CompressorWeights {
    pub wkv_bf16: Vec<u16>,
    pub wgate_bf16: Vec<u16>,
    pub ape_f32: Vec<f32>,
    pub range_bytes: usize,
}

pub struct CompressorPrepared {
    pub kv_f32: Vec<f32>,
    pub score_f32: Vec<f32>,
    pub score_plus_ape_f32: Vec<f32>,
}

pub struct FinalHeadOutput {
    pub collapsed_bf16: Vec<u16>,
    pub normalized_bf16: Vec<u16>,
    pub logits_f32: Vec<f32>,
    pub argmax: Vec<usize>,
}

pub struct FinalHeadWeights {
    shard: ShardFile,
    pub hc_fn_f32: Vec<f32>,
    pub hc_base_f32: Vec<f32>,
    pub hc_scale_f32: f32,
    pub norm_bf16: Vec<u16>,
}

pub struct FinalHeadCpuRun {
    pub output: FinalHeadOutput,
    pub read_ns: u64,
    pub wall_ns: u64,
}

impl FinalHeadWeights {
    pub fn advise_head_dontneed(&self) -> Result<(), Box<dyn Error>> {
        Ok(self.shard.advise_tensor_dontneed("head.weight")?)
    }

    pub fn read_head_weight_bytes(
        &self,
        token_start: usize,
        token_rows: usize,
        output: &mut [u8],
    ) -> Result<(), Box<dyn Error>> {
        let elements = token_rows
            .checked_mul(HIDDEN)
            .ok_or("V4 final head chunk element overflow")?;
        if token_start > VOCAB
            || token_rows > VOCAB.saturating_sub(token_start)
            || output.len() != elements * 2
        {
            return Err("V4 final head chunk contract mismatch".into());
        }
        Ok(self.shard.read_at(
            "head.weight",
            u64::try_from(
                token_start
                    .checked_mul(HIDDEN * 2)
                    .ok_or("V4 final head chunk byte offset overflow")?,
            )?,
            output,
        )?)
    }
}

pub fn load_final_head(model_dir: &Path) -> Result<FinalHeadWeights, Box<dyn Error>> {
    validate_model_dir(model_dir)?;
    let index = indexed_model(model_dir)?;
    let expected_shard = "model-00063-of-00064.safetensors";
    for name in [
        "hc_head_fn",
        "hc_head_base",
        "hc_head_scale",
        "norm.weight",
        "head.weight",
    ] {
        let location = index.resolve_tensor(name)?;
        if location.shard != expected_shard {
            return Err(format!(
                "V4 final {name} is in {}, expected {expected_shard}",
                location.shard
            )
            .into());
        }
    }
    let shard = ShardFile::open(model_dir.join(expected_shard))?;
    for (name, dtype, shape) in [
        (
            "hc_head_fn",
            TensorDtype::F32,
            vec![HC as u64, (HC * HIDDEN) as u64],
        ),
        ("hc_head_base", TensorDtype::F32, vec![HC as u64]),
        ("hc_head_scale", TensorDtype::F32, vec![1]),
        ("norm.weight", TensorDtype::Bf16, vec![HIDDEN as u64]),
        (
            "head.weight",
            TensorDtype::Bf16,
            vec![129_280, HIDDEN as u64],
        ),
    ] {
        let tensor = shard.tensor(name)?;
        if tensor.dtype != dtype || tensor.shape != shape {
            return Err(format!("V4 final {name} tensor contract mismatch").into());
        }
    }
    let hc_scale = read_f32(&shard, "hc_head_scale", &[1])?[0];
    Ok(FinalHeadWeights {
        hc_fn_f32: read_f32(&shard, "hc_head_fn", &[HC, HC * HIDDEN])?,
        hc_base_f32: read_f32(&shard, "hc_head_base", &[HC])?,
        hc_scale_f32: hc_scale,
        norm_bf16: read_bf16(&shard, "norm.weight", &[HIDDEN])?,
        shard,
    })
}

pub fn final_hc_gate_cpu(
    hc_hidden_bf16: &[u16],
    rows: usize,
    hc: usize,
    dim: usize,
    hc_fn_f32: &[f32],
    hc_scale_f32: f32,
    hc_base_f32: &[f32],
) -> Result<Vec<f32>, Box<dyn Error>> {
    let width = hc.checked_mul(dim).ok_or("V4 final HC width overflow")?;
    if rows == 0
        || hc == 0
        || dim == 0
        || hc_hidden_bf16.len() != rows * width
        || hc_fn_f32.len() != hc * width
        || hc_base_f32.len() != hc
        || !hc_scale_f32.is_finite()
        || hc_hidden_bf16
            .iter()
            .any(|&bits| !bf16_to_f32(bits).is_finite())
        || hc_fn_f32
            .iter()
            .chain(hc_base_f32)
            .any(|value| !value.is_finite())
    {
        return Err("V4 final HC gate contract mismatch".into());
    }
    let mut pre = vec![0.0_f32; rows * hc];
    for row in 0..rows {
        let input = &hc_hidden_bf16[row * width..(row + 1) * width];
        let inverse = 1.0
            / (input
                .iter()
                .map(|&bits| {
                    let value = bf16_to_f32(bits);
                    value * value
                })
                .sum::<f32>()
                / width as f32
                + EPS)
                .sqrt();
        for copy in 0..hc {
            let mut mix = 0.0_f32;
            for column in 0..width {
                mix += bf16_to_f32(input[column]) * hc_fn_f32[copy * width + column];
            }
            let logit = mix * inverse * hc_scale_f32 + hc_base_f32[copy];
            pre[row * hc + copy] = 1.0 / (1.0 + (-logit).exp()) + EPS;
        }
    }
    Ok(pre)
}

pub fn final_hc_head_cpu(
    hc_hidden_bf16: &[u16],
    rows: usize,
    hc: usize,
    dim: usize,
    hc_pre_f32: &[f32],
    norm_weight_f32: &[f32],
    head_weight_f32: &[f32],
    vocab: usize,
) -> Result<FinalHeadOutput, Box<dyn Error>> {
    // `hc_pre_f32` is the already-computed official sigmoid+epsilon HC head
    // weight vector; the learned projection that produces it stays a separate gate.
    if rows == 0
        || hc == 0
        || dim == 0
        || vocab == 0
        || hc_hidden_bf16.len() != rows * hc * dim
        || !matches!(hc_pre_f32.len(), length if length == hc || length == rows * hc)
        || norm_weight_f32.len() != dim
        || head_weight_f32.len() != vocab * dim
        || hc_hidden_bf16
            .iter()
            .any(|&bits| !bf16_to_f32(bits).is_finite())
        || hc_pre_f32
            .iter()
            .chain(norm_weight_f32)
            .chain(head_weight_f32)
            .any(|value| !value.is_finite())
    {
        return Err("V4 final HC/head contract mismatch".into());
    }
    let mut collapsed_bf16 = vec![0_u16; rows * dim];
    for row in 0..rows {
        for column in 0..dim {
            let mut value = 0.0_f32;
            for copy in 0..hc {
                let pre = if hc_pre_f32.len() == hc {
                    hc_pre_f32[copy]
                } else {
                    hc_pre_f32[row * hc + copy]
                };
                value += pre * bf16_to_f32(hc_hidden_bf16[(row * hc + copy) * dim + column]);
            }
            collapsed_bf16[row * dim + column] = f32_to_bf16_rne(value);
        }
    }
    let mut normalized_bf16 = vec![0_u16; rows * dim];
    for row in 0..rows {
        let input = &collapsed_bf16[row * dim..(row + 1) * dim];
        let mean_square = input
            .iter()
            .map(|&bits| {
                let value = bf16_to_f32(bits);
                value * value
            })
            .sum::<f32>()
            / dim as f32;
        let inverse = 1.0 / (mean_square + EPS).sqrt();
        for column in 0..dim {
            normalized_bf16[row * dim + column] =
                f32_to_bf16_rne(bf16_to_f32(input[column]) * inverse * norm_weight_f32[column]);
        }
    }
    let mut logits_f32 = vec![0.0_f32; rows * vocab];
    let mut argmax = vec![0_usize; rows];
    for row in 0..rows {
        for token in 0..vocab {
            let mut value = 0.0_f32;
            for column in 0..dim {
                value += bf16_to_f32(normalized_bf16[row * dim + column])
                    * head_weight_f32[token * dim + column];
            }
            logits_f32[row * vocab + token] = value;
            if token == 0 || value > logits_f32[row * vocab + argmax[row]] {
                argmax[row] = token;
            }
        }
    }
    Ok(FinalHeadOutput {
        collapsed_bf16,
        normalized_bf16,
        logits_f32,
        argmax,
    })
}

pub fn final_checkpoint_head_cpu_timed(
    weights: &FinalHeadWeights,
    hc_hidden_bf16: &[u16],
    rows: usize,
) -> Result<FinalHeadCpuRun, Box<dyn Error>> {
    let wall_started = std::time::Instant::now();
    let pre = final_hc_gate_cpu(
        hc_hidden_bf16,
        rows,
        HC,
        HIDDEN,
        &weights.hc_fn_f32,
        weights.hc_scale_f32,
        &weights.hc_base_f32,
    )?;
    let mut base = final_hc_head_cpu(
        hc_hidden_bf16,
        rows,
        HC,
        HIDDEN,
        &pre,
        &weights
            .norm_bf16
            .iter()
            .map(|&bits| bf16_to_f32(bits))
            .collect::<Vec<_>>(),
        &vec![0.0; HIDDEN],
        1,
    )?;
    const CHUNK_ROWS: usize = 256;
    base.logits_f32 = vec![0.0; rows * VOCAB];
    base.argmax.fill(0);
    let mut bytes = vec![0_u8; CHUNK_ROWS * HIDDEN * 2];
    let mut read_ns = 0_u64;
    for token_start in (0..VOCAB).step_by(CHUNK_ROWS) {
        let chunk_rows = (VOCAB - token_start).min(CHUNK_ROWS);
        let chunk_bytes = chunk_rows * HIDDEN * 2;
        let read_started = std::time::Instant::now();
        weights.read_head_weight_bytes(token_start, chunk_rows, &mut bytes[..chunk_bytes])?;
        read_ns = read_ns
            .saturating_add(u64::try_from(read_started.elapsed().as_nanos()).unwrap_or(u64::MAX));
        for row in 0..rows {
            for token_offset in 0..chunk_rows {
                let mut logit = 0.0_f32;
                let weight_start = token_offset * HIDDEN * 2;
                for column in 0..HIDDEN {
                    let byte = weight_start + column * 2;
                    let weight = bf16_to_f32(u16::from_le_bytes([bytes[byte], bytes[byte + 1]]));
                    logit += bf16_to_f32(base.normalized_bf16[row * HIDDEN + column]) * weight;
                }
                let token = token_start + token_offset;
                base.logits_f32[row * VOCAB + token] = logit;
                if token == 0 || logit > base.logits_f32[row * VOCAB + base.argmax[row]] {
                    base.argmax[row] = token;
                }
            }
        }
    }
    Ok(FinalHeadCpuRun {
        output: base,
        read_ns,
        wall_ns: u64::try_from(wall_started.elapsed().as_nanos()).unwrap_or(u64::MAX),
    })
}

#[derive(Clone)]
pub struct V4CompressorState {
    kv: Vec<f32>,
    score: Vec<f32>,
    compress_ratio: usize,
    head_dim: usize,
    overlap: bool,
    committed: bool,
}

#[derive(Clone)]
pub struct V4LayerPersistentState {
    layer: usize,
    max_context: usize,
    kv_bf16: Vec<u16>,
    compressed_kv_bf16: Vec<u16>,
    indexer_kv_bf16: Vec<u16>,
    compressor: V4CompressorState,
    indexer: Option<V4CompressorState>,
}

pub struct V4SerialPersistentState {
    max_context: usize,
    layers: Vec<Option<V4LayerPersistentState>>,
}

impl V4LayerPersistentState {
    fn new(layer: usize, max_context: usize) -> Result<Self, Box<dyn Error>> {
        let ratio = compression_ratio_for_layer(layer)?.get();
        Ok(Self {
            layer,
            max_context,
            kv_bf16: Vec::new(),
            compressed_kv_bf16: Vec::new(),
            indexer_kv_bf16: Vec::new(),
            compressor: V4CompressorState::new_decode(ratio, Q_HEAD_DIM)?,
            indexer: (ratio == 4)
                .then(|| V4CompressorState::new_decode(4, 128))
                .transpose()?,
        })
    }

    pub fn layer(&self) -> usize {
        self.layer
    }

    pub fn committed_kv_rows(&self) -> usize {
        self.kv_bf16.len() / Q_HEAD_DIM
    }

    pub fn kv_bf16(&self) -> &[u16] {
        &self.kv_bf16
    }

    pub fn compressed_kv_bf16(&self) -> &[u16] {
        &self.compressed_kv_bf16
    }

    pub fn indexer_kv_bf16(&self) -> &[u16] {
        &self.indexer_kv_bf16
    }

    pub fn attention_cache_bf16(&self) -> Vec<u16> {
        let mut cache = vec![0_u16; KV_WINDOW * Q_HEAD_DIM];
        let rows = self.committed_kv_rows();
        for absolute in rows.saturating_sub(KV_WINDOW)..rows {
            let source = absolute * Q_HEAD_DIM;
            let destination = (absolute % KV_WINDOW) * Q_HEAD_DIM;
            cache[destination..destination + Q_HEAD_DIM]
                .copy_from_slice(&self.kv_bf16[source..source + Q_HEAD_DIM]);
        }
        cache.extend_from_slice(&self.compressed_kv_bf16);
        cache
    }

    pub fn compressor(&self) -> &V4CompressorState {
        &self.compressor
    }

    pub fn indexer(&self) -> Option<&V4CompressorState> {
        self.indexer.as_ref()
    }

    pub fn append_kv_row(
        &mut self,
        start_pos: usize,
        row_bf16: &[u16],
    ) -> Result<(), Box<dyn Error>> {
        if start_pos != self.committed_kv_rows()
            || start_pos >= self.max_context
            || row_bf16.len() != Q_HEAD_DIM
            || row_bf16.iter().any(|&bits| !bf16_to_f32(bits).is_finite())
        {
            return Err("V4 per-layer KV append contract mismatch".into());
        }
        self.kv_bf16.extend_from_slice(row_bf16);
        Ok(())
    }

    pub fn commit_compressor_row(
        &mut self,
        start_pos: usize,
        prepared: &CompressorPrepared,
    ) -> Result<(), Box<dyn Error>> {
        self.compressor = self.compressor.prepare_decode_row(prepared, start_pos)?;
        Ok(())
    }

    pub fn commit_compressor_row_and_pool(
        &mut self,
        start_pos: usize,
        prepared: &CompressorPrepared,
    ) -> Result<Option<Vec<f32>>, Box<dyn Error>> {
        let (next, pooled) = self
            .compressor
            .prepare_decode_row_and_pool(prepared, start_pos)?;
        self.compressor = next;
        Ok(pooled)
    }

    pub fn commit_indexer_row(
        &mut self,
        start_pos: usize,
        prepared: &CompressorPrepared,
    ) -> Result<(), Box<dyn Error>> {
        let indexer = self
            .indexer
            .as_ref()
            .ok_or("V4 HCA layer has no CSA indexer state")?;
        self.indexer = Some(indexer.prepare_decode_row(prepared, start_pos)?);
        Ok(())
    }

    pub fn commit_indexer_row_and_pool(
        &mut self,
        start_pos: usize,
        prepared: &CompressorPrepared,
    ) -> Result<Option<Vec<f32>>, Box<dyn Error>> {
        let indexer = self
            .indexer
            .as_ref()
            .ok_or("V4 HCA layer has no CSA indexer state")?;
        let (next, pooled) = indexer.prepare_decode_row_and_pool(prepared, start_pos)?;
        self.indexer = Some(next);
        Ok(pooled)
    }

    pub fn append_compressed_kv_row(
        &mut self,
        compressed_index: usize,
        row_bf16: &[u16],
    ) -> Result<(), Box<dyn Error>> {
        append_compressed_row(
            &mut self.compressed_kv_bf16,
            compressed_index,
            Q_HEAD_DIM,
            row_bf16,
            "main",
        )
    }

    pub fn append_indexer_kv_row(
        &mut self,
        compressed_index: usize,
        row_bf16: &[u16],
    ) -> Result<(), Box<dyn Error>> {
        if self.indexer.is_none() {
            return Err("V4 HCA layer has no CSA indexer cache".into());
        }
        append_compressed_row(
            &mut self.indexer_kv_bf16,
            compressed_index,
            128,
            row_bf16,
            "indexer",
        )
    }

    pub fn bits_equal(&self, other: &Self) -> bool {
        self.layer == other.layer
            && self.max_context == other.max_context
            && self.kv_bf16 == other.kv_bf16
            && self.compressed_kv_bf16 == other.compressed_kv_bf16
            && self.indexer_kv_bf16 == other.indexer_kv_bf16
            && self.compressor.bits_equal(&other.compressor)
            && match (&self.indexer, &other.indexer) {
                (Some(left), Some(right)) => left.bits_equal(right),
                (None, None) => true,
                _ => false,
            }
    }
}

fn append_compressed_row(
    cache: &mut Vec<u16>,
    compressed_index: usize,
    width: usize,
    row_bf16: &[u16],
    operation: &str,
) -> Result<(), Box<dyn Error>> {
    if compressed_index.checked_mul(width) != Some(cache.len())
        || row_bf16.len() != width
        || row_bf16.iter().any(|&bits| !bf16_to_f32(bits).is_finite())
    {
        return Err(format!("V4 {operation} compressed KV append contract mismatch").into());
    }
    cache.extend_from_slice(row_bf16);
    Ok(())
}

impl V4SerialPersistentState {
    pub fn new(max_context: usize) -> Result<Self, Box<dyn Error>> {
        if max_context == 0 || max_context > 1_048_576 {
            return Err("V4 max context must be within 1..=1048576".into());
        }
        Ok(Self {
            max_context,
            layers: (0..V4_LAYER_COUNT).map(|_| None).collect(),
        })
    }

    pub fn max_context(&self) -> usize {
        self.max_context
    }

    pub fn allocated_layers(&self) -> usize {
        self.layers.iter().filter(|state| state.is_some()).count()
    }

    pub fn layer(&self, layer: usize) -> Option<&V4LayerPersistentState> {
        self.layers.get(layer).and_then(Option::as_ref)
    }

    pub fn transaction(
        &mut self,
        layer: usize,
        apply: impl FnOnce(&mut V4LayerPersistentState) -> Result<(), Box<dyn Error>>,
    ) -> Result<(), Box<dyn Error>> {
        let slot = self
            .layers
            .get(layer)
            .ok_or_else(|| format!("V4 layer {layer} is outside 0..{V4_LAYER_COUNT}"))?;
        let mut scratch = match slot {
            Some(state) => state.clone(),
            None => V4LayerPersistentState::new(layer, self.max_context)?,
        };
        apply(&mut scratch)?;
        self.layers[layer] = Some(scratch);
        Ok(())
    }
}

pub struct TransactionalLayerOutput {
    pub next_hidden: Vec<u16>,
    pub source_range_bytes: u64,
    pub indexer_topk_cpu_ns: u64,
}

pub fn run_serial_layer_transactions<F>(
    state: &mut V4SerialPersistentState,
    initial_hidden: Vec<u16>,
    manifests: &[LayerExecutionManifest],
    mut run_layer: F,
) -> Result<(Vec<u16>, Vec<LayerTelemetry>), Box<dyn Error>>
where
    F: FnMut(
        &LayerExecutionManifest,
        &[u16],
        &mut V4LayerPersistentState,
    ) -> Result<TransactionalLayerOutput, Box<dyn Error>>,
{
    if initial_hidden.is_empty()
        || initial_hidden
            .iter()
            .any(|&bits| !bf16_to_f32(bits).is_finite())
    {
        return Err("V4 serial transactional runner requires finite hidden state".into());
    }
    let mut hidden = initial_hidden;
    let mut telemetry = Vec::with_capacity(manifests.len());
    for (position, manifest) in manifests.iter().enumerate() {
        if position > 0 && manifests[position - 1].layer + 1 != manifest.layer {
            return Err("V4 serial transactional runner requires contiguous layers".into());
        }
        let started = std::time::Instant::now();
        let mut committed_output = None;
        let kv_rows_before = state
            .layer(manifest.layer)
            .map_or(0, V4LayerPersistentState::committed_kv_rows);
        state.transaction(manifest.layer, |layer_state| {
            let output = run_layer(manifest, &hidden, layer_state)?;
            if output.source_range_bytes != manifest.contract_range_bytes
                || output.next_hidden.is_empty()
                || output.next_hidden == hidden
                || layer_state.committed_kv_rows() <= kv_rows_before
                || output
                    .next_hidden
                    .iter()
                    .any(|&bits| !bf16_to_f32(bits).is_finite())
            {
                return Err("V4 transactional layer output contract mismatch".into());
            }
            committed_output = Some(output);
            Ok(())
        })?;
        let output = committed_output.ok_or("V4 layer transaction returned no output")?;
        let kv_rows_after = state
            .layer(manifest.layer)
            .ok_or("V4 committed layer state is missing")?
            .committed_kv_rows();
        hidden = output.next_hidden;
        telemetry.push(LayerTelemetry {
            layer: manifest.layer,
            compression_ratio: manifest.compression_ratio,
            source_range_bytes: output.source_range_bytes,
            indexer_topk_cpu_ns: output.indexer_topk_cpu_ns,
            layer_elapsed_ns: started.elapsed().as_nanos().try_into().unwrap_or(u64::MAX),
            kv_rows_before,
            kv_rows_after,
        });
    }
    Ok((hidden, telemetry))
}

#[derive(Debug, Eq, PartialEq)]
pub enum K2ContractError {
    GuaranteedRowMismatch,
    RowEquivalence,
    PrefixOutOfRange,
    RejectedSuffix,
    DirtyAuthoritativeState,
}

impl std::fmt::Display for K2ContractError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl Error for K2ContractError {}

/// Three semantic snapshots: base, base+guaranteed row, and base+both rows.
/// Row 1 is prepared from row 0's scratch snapshot, never from `base`.
pub struct K2Prepared<T> {
    base: T,
    prefixes: [T; 2],
}

pub struct K2Validated<T> {
    prepared: K2Prepared<T>,
    accepted_rows: usize,
}

impl<T: Clone> K2Prepared<T> {
    pub fn prepare(
        base: &T,
        prepare_row0: impl FnOnce(&mut T) -> Result<(), Box<dyn Error>>,
        prepare_row1: impl FnOnce(&mut T) -> Result<(), Box<dyn Error>>,
    ) -> Result<Self, Box<dyn Error>> {
        let mut row0 = base.clone();
        prepare_row0(&mut row0)?;
        let mut row1 = row0.clone();
        prepare_row1(&mut row1)?;
        Ok(Self {
            base: base.clone(),
            prefixes: [row0, row1],
        })
    }

    pub fn validate(
        self,
        sequential: [&T; 2],
        proposed_tokens: [u32; 2],
        target_tokens: [u32; 2],
        exact: impl Fn(&T, &T) -> bool,
    ) -> Result<K2Validated<T>, K2ContractError> {
        if proposed_tokens[0] != target_tokens[0] {
            return Err(K2ContractError::GuaranteedRowMismatch);
        }
        if !exact(&self.prefixes[0], sequential[0]) || !exact(&self.prefixes[1], sequential[1]) {
            return Err(K2ContractError::RowEquivalence);
        }
        Ok(K2Validated {
            prepared: self,
            accepted_rows: 1 + usize::from(proposed_tokens[1] == target_tokens[1]),
        })
    }
}

impl<T: Clone> K2Validated<T> {
    pub fn accepted_rows(&self) -> usize {
        self.accepted_rows
    }

    pub fn commit_prefix(
        &self,
        authoritative: &mut T,
        rows: usize,
        exact: impl Fn(&T, &T) -> bool,
    ) -> Result<(), K2ContractError> {
        if rows > 2 {
            return Err(K2ContractError::PrefixOutOfRange);
        }
        if rows > self.accepted_rows {
            return Err(K2ContractError::RejectedSuffix);
        }
        if !exact(authoritative, &self.prepared.base) {
            return Err(K2ContractError::DirtyAuthoritativeState);
        }
        authoritative.clone_from(match rows {
            0 => &self.prepared.base,
            1 => &self.prepared.prefixes[0],
            2 => &self.prepared.prefixes[1],
            _ => unreachable!(),
        });
        Ok(())
    }
}

pub const K2_MIN_GATE_SAMPLES: u64 = 100;

#[derive(Clone, Copy, Debug)]
pub struct K2GateInput {
    pub draft_matches: u64,
    pub trials: u64,
    pub t1_lower: f64,
    pub t2_upper: f64,
    pub b1_lower: f64,
    pub b2_upper: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct K2GateReport {
    pub admit: bool,
    pub reason: &'static str,
    pub q_lower: Option<f64>,
    pub committed_rows_lower: Option<f64>,
    pub time_ratio_upper: Option<f64>,
    pub bytes_per_row_upper: Option<f64>,
}

/// Safe two-action minimax gate: the scheduler chooses AR/K2 and nature chooses
/// the worst q/time/byte point in the supplied confidence box. Exact semantic
/// parity is a hard prerequisite outside this payoff, not a weighted term.
pub fn evaluate_k2_renewal_gate(input: K2GateInput) -> K2GateReport {
    let invalid_measurement = input.trials < K2_MIN_GATE_SAMPLES
        || input.draft_matches > input.trials
        || [
            input.t1_lower,
            input.t2_upper,
            input.b1_lower,
            input.b2_upper,
        ]
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0);
    if invalid_measurement {
        return K2GateReport {
            admit: false,
            reason: "AR_INVALID_OR_INSUFFICIENT",
            q_lower: None,
            committed_rows_lower: None,
            time_ratio_upper: None,
            bytes_per_row_upper: None,
        };
    }

    let n = input.trials as f64;
    let p = input.draft_matches as f64 / n;
    let z = 1.959_963_984_540_054_f64;
    let z2 = z * z;
    let denominator = 1.0 + z2 / n;
    let q_lower =
        ((p + z2 / (2.0 * n)) - z * (p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt()) / denominator;
    let committed_rows_lower = 1.0 + q_lower;
    let time_ratio_upper = input.t2_upper / input.t1_lower;
    let bytes_per_row_upper = input.b2_upper / committed_rows_lower;
    let time_pass = committed_rows_lower > time_ratio_upper;
    let bytes_pass = bytes_per_row_upper <= 0.90 * input.b1_lower;
    K2GateReport {
        admit: time_pass && bytes_pass,
        reason: if time_pass && bytes_pass {
            "K2_ROBUST_WORST_CASE_PASS"
        } else {
            "AR_ROBUST_WORST_CASE_FAIL"
        },
        q_lower: Some(q_lower),
        committed_rows_lower: Some(committed_rows_lower),
        time_ratio_upper: Some(time_ratio_upper),
        bytes_per_row_upper: Some(bytes_per_row_upper),
    }
}

pub struct FixtureStages {
    pub collapsed: Vec<u16>,
    pub norm: Vec<u16>,
    pub mixes: Vec<f32>,
    pub pre: Vec<f32>,
    pub post: Vec<f32>,
    pub comb: Vec<f32>,
    pub sinkhorn_trace: Vec<f32>,
}

pub fn load(model_dir: &Path, ids: &[i64]) -> Result<L0State, Box<dyn Error>> {
    validate_ids(ids)?;
    validate_model_dir(model_dir)?;
    let embed = ShardFile::open(model_dir.join("model-00001-of-00064.safetensors"))?;
    let l0 = ShardFile::open(model_dir.join("model-00002-of-00064.safetensors"))?;
    for contract in C {
        let file = if contract.shard.starts_with("model-00001") {
            &embed
        } else {
            &l0
        };
        let tensor = file.tensor(contract.name)?;
        if tensor.dtype != contract.dtype
            || tensor.shape != contract.shape
            || tensor.byte_len != contract.bytes
        {
            return Err(format!("{} pinned contract mismatch", contract.name).into());
        }
    }
    checked(
        &embed,
        "embed.weight",
        TensorDtype::Bf16,
        &[129_280, HIDDEN],
    )?;
    let mut embedding = vec![0_u16; ids.len() * HIDDEN];
    for (row, &id) in ids.iter().enumerate() {
        let offset = usize::try_from(id)?
            .checked_mul(HIDDEN * 2)
            .ok_or("embedding offset overflow")?;
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(
                embedding[row * HIDDEN..][..HIDDEN].as_mut_ptr().cast(),
                HIDDEN * 2,
            )
        };
        embed.read_at("embed.weight", u64::try_from(offset)?, bytes)?;
    }
    let hc_fn = read_f32(&l0, "layers.0.hc_attn_fn", &[MIX, HC * HIDDEN])?;
    let hc_base = read_f32(&l0, "layers.0.hc_attn_base", &[MIX])?;
    let hc_scale = read_f32(&l0, "layers.0.hc_attn_scale", &[3])?;
    let attn_norm = read_bf16(&l0, "layers.0.attn_norm.weight", &[HIDDEN])?;
    Ok(L0State {
        embedding,
        hc_fn,
        hc_base,
        hc_scale,
        attn_norm,
        range_bytes: ids.len() * HIDDEN * 2 + MIX * HC * HIDDEN * 4 + MIX * 4 + 3 * 4 + HIDDEN * 2,
    })
}

pub fn validate_ids(ids: &[i64]) -> Result<(), Box<dyn Error>> {
    let vocab = i64::try_from(VOCAB)?;
    if ids.is_empty() || ids.iter().any(|id| !(0..vocab).contains(id)) {
        Err("V4 state requires nonempty in-range token ids".into())
    } else {
        Ok(())
    }
}

pub fn load_layer_state_weights(
    model_dir: &Path,
    layer: usize,
    ids: &[i64],
) -> Result<LayerStateWeights, Box<dyn Error>> {
    validate_ids(ids)?;
    let compression_ratio = compression_ratio_for_layer(layer)?;
    let contracts = layer_tensor_manifest(layer, 0)?;
    let index = indexed_model(model_dir)?;
    let expected_shard = format!("model-{:05}-of-00064.safetensors", layer + 2);
    validate_layer_locations(&index, &contracts, &expected_shard)?;
    let shard = ShardFile::open(model_dir.join(&expected_shard))?;
    for contract in &contracts {
        let tensor = shard.tensor(&contract.name)?;
        if tensor.dtype != contract.dtype
            || tensor.shape != contract.shape
            || tensor.byte_len != contract.byte_len
        {
            return Err(format!("{} pinned layer contract mismatch", contract.name).into());
        }
    }

    let prefix = format!("layers.{layer}");
    let compressor_width = match compression_ratio {
        CompressionRatio::Csa4 => 1_024,
        CompressionRatio::Hca128 => 512,
    };
    let compressor = LayerCompressorWeights {
        wkv_bf16: read_bf16(
            &shard,
            &format!("{prefix}.attn.compressor.wkv.weight"),
            &[compressor_width, HIDDEN],
        )?,
        wgate_bf16: read_bf16(
            &shard,
            &format!("{prefix}.attn.compressor.wgate.weight"),
            &[compressor_width, HIDDEN],
        )?,
        ape_f32: read_f32(
            &shard,
            &format!("{prefix}.attn.compressor.ape"),
            &[
                compression_ratio.get(),
                if compression_ratio == CompressionRatio::Csa4 {
                    1_024
                } else {
                    512
                },
            ],
        )?,
        norm_bf16: read_bf16(
            &shard,
            &format!("{prefix}.attn.compressor.norm.weight"),
            &[Q_HEAD_DIM],
        )?,
    };
    let indexer = if compression_ratio == CompressionRatio::Csa4 {
        Some(CsaIndexerWeights {
            wq_b: read_matrix(
                &shard,
                &format!("{prefix}.attn.indexer.wq_b.weight"),
                &format!("{prefix}.attn.indexer.wq_b.scale"),
                1_536,
                8_192,
            )?,
            weights_proj_bf16: read_bf16(
                &shard,
                &format!("{prefix}.attn.indexer.weights_proj.weight"),
                &[64, HIDDEN],
            )?,
            compressor: LayerCompressorWeights {
                wkv_bf16: read_bf16(
                    &shard,
                    &format!("{prefix}.attn.indexer.compressor.wkv.weight"),
                    &[256, HIDDEN],
                )?,
                wgate_bf16: read_bf16(
                    &shard,
                    &format!("{prefix}.attn.indexer.compressor.wgate.weight"),
                    &[256, HIDDEN],
                )?,
                ape_f32: read_f32(
                    &shard,
                    &format!("{prefix}.attn.indexer.compressor.ape"),
                    &[4, 256],
                )?,
                norm_bf16: read_bf16(
                    &shard,
                    &format!("{prefix}.attn.indexer.compressor.norm.weight"),
                    &[128],
                )?,
            },
        })
    } else {
        None
    };
    let router = if layer < 3 {
        LayerRouterWeights::Hash {
            ids: read_i64_rows(
                &shard,
                &format!("{prefix}.ffn.gate.tid2eid"),
                ids,
                crate::v4_router::V4_TOPK,
            )?,
        }
    } else {
        LayerRouterWeights::Learned {
            bias: read_f32(
                &shard,
                &format!("{prefix}.ffn.gate.bias"),
                &[crate::v4_router::V4_ROUTED_EXPERTS],
            )?,
        }
    };
    Ok(LayerStateWeights {
        layer,
        compression_ratio,
        hc_fn: read_f32(&shard, &format!("{prefix}.hc_attn_fn"), &[MIX, HC * HIDDEN])?,
        hc_base: read_f32(&shard, &format!("{prefix}.hc_attn_base"), &[MIX])?,
        hc_scale: read_f32(&shard, &format!("{prefix}.hc_attn_scale"), &[3])?,
        attn_norm: read_bf16(&shard, &format!("{prefix}.attn_norm.weight"), &[HIDDEN])?,
        compressor,
        indexer,
        ffn_norm: read_bf16(&shard, &format!("{prefix}.ffn_norm.weight"), &[HIDDEN])?,
        gate: read_bf16(
            &shard,
            &format!("{prefix}.ffn.gate.weight"),
            &[crate::v4_router::V4_ROUTED_EXPERTS, HIDDEN],
        )?,
        router,
    })
}

pub fn route_layer_cpu(
    weights: &LayerStateWeights,
    input: &[u16],
) -> Result<crate::v4_router::RouterSelection, Box<dyn Error>> {
    let normalized = rmsnorm_bf16(input, &weights.ffn_norm)?;
    let logits = crate::v4_router::project_gate_logits(&normalized, &weights.gate)?;
    Ok(match &weights.router {
        LayerRouterWeights::Hash { ids } => crate::v4_router::select_hash(&logits, ids)?,
        LayerRouterWeights::Learned { bias } => crate::v4_router::select_learned(&logits, bias)?,
    })
}

pub fn indexer_scores_cpu(
    q: &[f32],
    kv: &[f32],
    weights: &[f32],
    rows: usize,
    heads: usize,
    head_dim: usize,
    keys: usize,
) -> Result<Vec<f32>, Box<dyn Error>> {
    if rows == 0
        || heads == 0
        || head_dim == 0
        || keys == 0
        || q.len() != rows * heads * head_dim
        || kv.len() != keys * head_dim
        || weights.len() != rows * heads
        || q.iter()
            .chain(kv)
            .chain(weights)
            .any(|value| !value.is_finite())
    {
        return Err("V4 CSA indexer score contract mismatch".into());
    }
    let mut scores = vec![0.0_f32; rows * keys];
    for row in 0..rows {
        for key in 0..keys {
            let mut score = 0.0_f32;
            for head in 0..heads {
                let mut dot = 0.0_f32;
                for dimension in 0..head_dim {
                    dot += q[(row * heads + head) * head_dim + dimension]
                        * kv[key * head_dim + dimension];
                }
                score += dot.max(0.0) * weights[row * heads + head];
            }
            scores[row * keys + key] = score;
        }
    }
    Ok(scores)
}

pub fn rotate_fp4_sim_bf16(
    input_bf16: &[u16],
    vectors: usize,
    width: usize,
) -> Result<Vec<f32>, Box<dyn Error>> {
    if vectors == 0
        || width < 32
        || !width.is_power_of_two()
        || !width.is_multiple_of(32)
        || input_bf16.len() != vectors * width
        || input_bf16
            .iter()
            .any(|&bits| !bf16_to_f32(bits).is_finite())
    {
        return Err("V4 indexer rotate/FP4 simulation contract mismatch".into());
    }
    let scale = 1.0_f32 / (width as f32).sqrt();
    let mut output = input_bf16
        .iter()
        .map(|&bits| bf16_to_f32(bits))
        .collect::<Vec<_>>();
    for vector in output.chunks_exact_mut(width) {
        let mut stride = 1;
        while stride < width {
            for block in (0..width).step_by(stride * 2) {
                for offset in 0..stride {
                    let left = vector[block + offset];
                    let right = vector[block + stride + offset];
                    vector[block + offset] = left + right;
                    vector[block + stride + offset] = left - right;
                }
            }
            stride *= 2;
        }
        for value in vector.iter_mut() {
            *value = f32::from_bits((f32_to_bf16_rne(*value * scale) as u32) << 16);
        }
        for block in vector.chunks_exact_mut(32) {
            let amax = block
                .iter()
                .fold(0.0_f32, |maximum, value| maximum.max(value.abs()));
            let quant_scale = pow2_ceil((amax / 6.0).max(f32::from_bits(1 << 22)));
            for value in block {
                *value = quantize_e2m1(*value / quant_scale) * quant_scale;
            }
        }
    }
    Ok(output)
}

fn quantize_e2m1(value: f32) -> f32 {
    const LEVELS: [f32; 15] = [
        -6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    ];
    let mut best = 0.0_f32;
    let mut distance = f32::INFINITY;
    for candidate in LEVELS {
        let candidate_distance = (value - candidate).abs();
        if candidate_distance < distance
            || (candidate_distance == distance && e2m1_even(candidate) && !e2m1_even(best))
        {
            best = candidate;
            distance = candidate_distance;
        }
    }
    best
}

fn e2m1_even(value: f32) -> bool {
    matches!(
        value.abs().to_bits(),
        0 | 0x3f80_0000 | 0x4000_0000 | 0x4080_0000
    )
}

pub fn compressed_rope_freqs(position: usize) -> Vec<f32> {
    const DIM: usize = Q_ROPE_DIM;
    const BASE: f32 = 160_000.0;
    const FACTOR: f32 = 16.0;
    const ORIGINAL: f32 = 65_536.0;
    let correction = |rotations: f32| {
        DIM as f32 * (ORIGINAL / (rotations * 2.0 * std::f32::consts::PI)).ln() / (2.0 * BASE.ln())
    };
    let low = correction(32.0).floor().max(0.0);
    let high = correction(1.0).ceil().min((DIM - 1) as f32);
    (0..DIM / 2)
        .flat_map(|pair| {
            let smooth = 1.0 - ((pair as f32 - low) / (high - low)).clamp(0.0, 1.0);
            let base_frequency = BASE.powf(-((pair * 2) as f32) / DIM as f32);
            let frequency = base_frequency / FACTOR * (1.0 - smooth) + base_frequency * smooth;
            let angle = position as f32 * frequency;
            [angle.cos(), angle.sin()]
        })
        .collect()
}

pub fn indexer_topk_cpu(
    scores: &[f32],
    rows: usize,
    keys: usize,
    topk: usize,
    valid_keys: &[usize],
    offset: i32,
) -> Result<Vec<i32>, Box<dyn Error>> {
    if rows == 0
        || keys == 0
        || topk == 0
        || topk > keys
        || scores.len() != rows * keys
        || valid_keys.len() != rows
        || valid_keys.iter().any(|&valid| valid < topk || valid > keys)
        || scores.iter().any(|score| !score.is_finite())
    {
        return Err("V4 CSA indexer top-k contract mismatch".into());
    }
    let mut output = Vec::with_capacity(rows * topk);
    for row in 0..rows {
        let mut selected = Vec::with_capacity(topk);
        for key in 0..valid_keys[row] {
            let score = scores[row * keys + key];
            let position = selected
                .iter()
                .position(|&(selected_score, selected_key)| {
                    score > selected_score || (score == selected_score && key < selected_key)
                })
                .unwrap_or(selected.len());
            if position < topk {
                selected.insert(position, (score, key));
                selected.truncate(topk);
            }
        }
        output.extend(
            selected
                .into_iter()
                .map(|(_, key)| i32::try_from(key).unwrap() + offset),
        );
    }
    Ok(output)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SparseTopk {
    pub indices: Vec<i32>,
    pub rows: usize,
    pub per_row: usize,
    pub window_per_row: usize,
    pub compressed_per_row: usize,
}

pub fn window_topk_cpu(
    rows: usize,
    start_pos: usize,
    window: usize,
) -> Result<(Vec<i32>, usize), Box<dyn Error>> {
    if rows == 0 || window == 0 {
        return Err("V4 window top-k contract mismatch".into());
    }
    if start_pos > 0 {
        let mut output = Vec::with_capacity(rows * window);
        for row in 0..rows {
            let absolute = start_pos + row;
            if absolute >= window - 1 {
                let position = absolute % window;
                output.extend(
                    (position + 1..window)
                        .chain(0..=position)
                        .map(|index| index as i32),
                );
            } else {
                output.extend((0..=absolute).map(|index| index as i32));
                output.resize(output.len() + window - absolute - 1, -1);
            }
        }
        return Ok((output, window));
    }

    let width = rows.min(window);
    let mut output = Vec::with_capacity(rows * width);
    for row in 0..rows {
        let first = (row + 1).saturating_sub(window);
        for slot in 0..width {
            let key = first + slot;
            output.push(if key <= row { key as i32 } else { -1 });
        }
    }
    Ok((output, width))
}

pub fn hca_compress_topk_cpu(
    rows: usize,
    start_pos: usize,
    ratio: usize,
    offset: usize,
) -> Result<(Vec<i32>, usize), Box<dyn Error>> {
    if rows == 0 || ratio == 0 {
        return Err("V4 HCA compressed top-k contract mismatch".into());
    }
    if start_pos > 0 {
        let width = (start_pos + rows) / ratio;
        let mut output = Vec::with_capacity(rows * width);
        for row in 0..rows {
            let valid = (start_pos + row + 1) / ratio;
            for key in 0..width {
                output.push(if key < valid {
                    (offset + key) as i32
                } else {
                    -1
                });
            }
        }
        return Ok((output, width));
    }

    let width = rows / ratio;
    let mut output = Vec::with_capacity(rows * width);
    for row in 0..rows {
        let valid = (row + 1) / ratio;
        for key in 0..width {
            output.push(if key < valid {
                (offset + key) as i32
            } else {
                -1
            });
        }
    }
    Ok((output, width))
}

pub fn csa_compress_topk_cpu(
    scores: &[f32],
    rows: usize,
    keys: usize,
    valid_keys: &[usize],
    topk: usize,
    offset: usize,
) -> Result<(Vec<i32>, usize), Box<dyn Error>> {
    if rows == 0
        || topk == 0
        || scores.len() != rows * keys
        || valid_keys.len() != rows
        || valid_keys.iter().any(|&valid| valid > keys)
        || scores.iter().any(|score| !score.is_finite())
    {
        return Err("V4 CSA compressed top-k contract mismatch".into());
    }
    let width = topk.min(keys);
    let mut output = Vec::with_capacity(rows * width);
    for row in 0..rows {
        let mut candidates = (0..valid_keys[row])
            .map(|key| (scores[row * keys + key], key))
            .collect::<Vec<_>>();
        candidates.sort_by(|(left_score, left_key), (right_score, right_key)| {
            right_score
                .total_cmp(left_score)
                .then_with(|| left_key.cmp(right_key))
        });
        output.extend(
            candidates
                .into_iter()
                .take(width)
                .map(|(_, key)| (offset + key) as i32),
        );
        output.resize(
            output.len() + width.saturating_sub(valid_keys[row].min(width)),
            -1,
        );
    }
    Ok((output, width))
}

fn concatenate_sparse_topk(
    rows: usize,
    window: Vec<i32>,
    window_per_row: usize,
    compressed: Vec<i32>,
    compressed_per_row: usize,
) -> SparseTopk {
    let per_row = window_per_row + compressed_per_row;
    let mut indices = Vec::with_capacity(rows * per_row);
    for row in 0..rows {
        indices.extend_from_slice(&window[row * window_per_row..(row + 1) * window_per_row]);
        indices.extend_from_slice(
            &compressed[row * compressed_per_row..(row + 1) * compressed_per_row],
        );
    }
    SparseTopk {
        indices,
        rows,
        per_row,
        window_per_row,
        compressed_per_row,
    }
}

pub fn build_hca_sparse_topk(
    rows: usize,
    start_pos: usize,
    window: usize,
    ratio: usize,
) -> Result<SparseTopk, Box<dyn Error>> {
    let (window_indices, window_per_row) = window_topk_cpu(rows, start_pos, window)?;
    let offset = if start_pos == 0 { rows } else { window };
    let (compressed, compressed_per_row) = hca_compress_topk_cpu(rows, start_pos, ratio, offset)?;
    Ok(concatenate_sparse_topk(
        rows,
        window_indices,
        window_per_row,
        compressed,
        compressed_per_row,
    ))
}

pub fn build_csa_sparse_topk(
    scores: &[f32],
    rows: usize,
    keys: usize,
    valid_keys: &[usize],
    start_pos: usize,
    window: usize,
    topk: usize,
) -> Result<SparseTopk, Box<dyn Error>> {
    let (window_indices, window_per_row) = window_topk_cpu(rows, start_pos, window)?;
    let offset = if start_pos == 0 { rows } else { window };
    let (compressed, compressed_per_row) =
        csa_compress_topk_cpu(scores, rows, keys, valid_keys, topk, offset)?;
    Ok(concatenate_sparse_topk(
        rows,
        window_indices,
        window_per_row,
        compressed,
        compressed_per_row,
    ))
}

pub fn load_fp8_projections(model_dir: &Path) -> Result<L0Fp8Projections, Box<dyn Error>> {
    load_layer_fp8_projections(model_dir, 0)
}

pub fn load_layer_fp8_projections(
    model_dir: &Path,
    layer: usize,
) -> Result<L0Fp8Projections, Box<dyn Error>> {
    validate_model_dir(model_dir)?;
    compression_ratio_for_layer(layer)?;
    let shard =
        ShardFile::open(model_dir.join(format!("model-{:05}-of-00064.safetensors", layer + 2)))?;
    let prefix = format!("layers.{layer}.attn");
    let wq_a = read_projection(
        &shard,
        &format!("{prefix}.wq_a.weight"),
        &format!("{prefix}.wq_a.scale"),
        &format!("{prefix}.q_norm.weight"),
        1_536,
    )?;
    let wq_b = read_matrix(
        &shard,
        &format!("{prefix}.wq_b.weight"),
        &format!("{prefix}.wq_b.scale"),
        1_536,
        65_536,
    )?;
    let wkv = read_projection(
        &shard,
        &format!("{prefix}.wkv.weight"),
        &format!("{prefix}.wkv.scale"),
        &format!("{prefix}.kv_norm.weight"),
        512,
    )?;
    let range_bytes = wq_a.matrix.weight.len()
        + wq_a.matrix.scale.len()
        + wq_a.norm.len() * 2
        + wq_b.weight.len()
        + wq_b.scale.len()
        + wkv.matrix.weight.len()
        + wkv.matrix.scale.len()
        + wkv.norm.len() * 2;
    Ok(L0Fp8Projections {
        wq_a,
        wq_b,
        wkv,
        range_bytes,
    })
}

pub fn load_l0_tail(model_dir: &Path) -> Result<L0TailWeights, Box<dyn Error>> {
    load_layer_tail(model_dir, 0)
}

pub fn load_layer_tail(model_dir: &Path, layer: usize) -> Result<L0TailWeights, Box<dyn Error>> {
    load_layer_tail_inner(model_dir, layer, false)
}

pub fn load_layer_tail_raw(
    model_dir: &Path,
    layer: usize,
) -> Result<L0TailWeights, Box<dyn Error>> {
    load_layer_tail_inner(model_dir, layer, true)
}

fn load_layer_tail_inner(
    model_dir: &Path,
    layer: usize,
    retain_raw_wo_a: bool,
) -> Result<L0TailWeights, Box<dyn Error>> {
    let load_started = std::time::Instant::now();
    validate_model_dir(model_dir)?;
    compression_ratio_for_layer(layer)?;
    let shard =
        ShardFile::open(model_dir.join(format!("model-{:05}-of-00064.safetensors", layer + 2)))?;
    let prefix = format!("layers.{layer}");
    let attn_sink = read_f32(&shard, &format!("{prefix}.attn.attn_sink"), &[Q_HEADS])?;
    let wo_a = read_matrix(
        &shard,
        &format!("{prefix}.attn.wo_a.weight"),
        &format!("{prefix}.attn.wo_a.scale"),
        Q_HEADS * Q_HEAD_DIM / O_GROUPS,
        O_GROUPS * O_LORA_RANK,
    )?;
    let wo_a_weight_len = wo_a.weight.len();
    let wo_a_scale_len = wo_a.scale.len();
    let (wo_a_bf16, wo_a_raw, wo_a_decode_ms) = if retain_raw_wo_a {
        (Vec::new(), Some(wo_a.weight), 0.0)
    } else {
        let decode_started = std::time::Instant::now();
        let mut wo_a_bf16 = Vec::with_capacity(wo_a_weight_len);
        // The pinned official model declares wo_a as BF16, so checkpoint loading
        // casts the FP8 payload directly and does not apply its stored scale.
        for &value in &wo_a.weight {
            wo_a_bf16.push(f32_to_bf16_rne(E4M3Fn(value).decode()));
        }
        (
            wo_a_bf16,
            None,
            decode_started.elapsed().as_secs_f64() * 1_000.0,
        )
    };
    let wo_b = read_matrix(
        &shard,
        &format!("{prefix}.attn.wo_b.weight"),
        &format!("{prefix}.attn.wo_b.scale"),
        O_GROUPS * O_LORA_RANK,
        HIDDEN,
    )?;
    let hc_ffn_fn = read_f32(&shard, &format!("{prefix}.hc_ffn_fn"), &[MIX, HC * HIDDEN])?;
    let hc_ffn_base = read_f32(&shard, &format!("{prefix}.hc_ffn_base"), &[MIX])?;
    let hc_ffn_scale = read_f32(&shard, &format!("{prefix}.hc_ffn_scale"), &[3])?;
    let ffn_norm = read_bf16(&shard, &format!("{prefix}.ffn_norm.weight"), &[HIDDEN])?;
    let gate = read_bf16(
        &shard,
        &format!("{prefix}.ffn.gate.weight"),
        &[crate::v4_router::V4_ROUTED_EXPERTS, HIDDEN],
    )?;
    let shared = L0SharedExpert {
        w1: read_matrix(
            &shard,
            &format!("{prefix}.ffn.shared_experts.w1.weight"),
            &format!("{prefix}.ffn.shared_experts.w1.scale"),
            HIDDEN,
            crate::v4_expert::V4_EXPERT_INTERMEDIATE,
        )?,
        w2: read_matrix(
            &shard,
            &format!("{prefix}.ffn.shared_experts.w2.weight"),
            &format!("{prefix}.ffn.shared_experts.w2.scale"),
            crate::v4_expert::V4_EXPERT_INTERMEDIATE,
            HIDDEN,
        )?,
        w3: read_matrix(
            &shard,
            &format!("{prefix}.ffn.shared_experts.w3.weight"),
            &format!("{prefix}.ffn.shared_experts.w3.scale"),
            HIDDEN,
            crate::v4_expert::V4_EXPERT_INTERMEDIATE,
        )?,
    };
    let source_range_bytes = attn_sink.len() * 4
        + wo_a_weight_len
        + wo_a_scale_len
        + wo_b.weight.len()
        + wo_b.scale.len()
        + hc_ffn_fn.len() * 4
        + hc_ffn_base.len() * 4
        + hc_ffn_scale.len() * 4
        + ffn_norm.len() * 2
        + gate.len() * 2
        + shared.w1.weight.len()
        + shared.w1.scale.len()
        + shared.w2.weight.len()
        + shared.w2.scale.len()
        + shared.w3.weight.len()
        + shared.w3.scale.len();
    Ok(L0TailWeights {
        attn_sink,
        wo_a_bf16,
        wo_a_raw,
        wo_b,
        hc_ffn_fn,
        hc_ffn_base,
        hc_ffn_scale,
        ffn_norm,
        gate,
        shared,
        source_range_bytes,
        load_elapsed_ms: load_started.elapsed().as_secs_f64() * 1_000.0,
        wo_a_decode_ms,
    })
}

pub fn load_compressor_weights(model_dir: &Path) -> Result<CompressorWeights, Box<dyn Error>> {
    validate_model_dir(model_dir)?;
    let shard = ShardFile::open(model_dir.join("model-00002-of-00064.safetensors"))?;
    let wkv_bf16 = read_bf16(
        &shard,
        "layers.0.attn.compressor.wkv.weight",
        &[Q_HEAD_DIM, HIDDEN],
    )?;
    let wgate_bf16 = read_bf16(
        &shard,
        "layers.0.attn.compressor.wgate.weight",
        &[Q_HEAD_DIM, HIDDEN],
    )?;
    let ape_f32 = read_f32(
        &shard,
        "layers.0.attn.compressor.ape",
        &[COMPRESS_RATIO, Q_HEAD_DIM],
    )?;
    Ok(CompressorWeights {
        range_bytes: (wkv_bf16.len() + wgate_bf16.len()) * 2 + ape_f32.len() * 4,
        wkv_bf16,
        wgate_bf16,
        ape_f32,
    })
}

pub fn quantize_fp8_activation(
    input: &[u16],
    rows: usize,
) -> Result<QuantizedFp8Activation, Box<dyn Error>> {
    if rows == 0 || rows > IDS.len() || input.len() % rows != 0 {
        return Err("FP8 activation requires 1..=4 complete rows".into());
    }
    let in_features = input.len() / rows;
    if in_features == 0 || in_features % FP8_BLOCK != 0 {
        return Err("FP8 activation width must be a non-zero multiple of 128".into());
    }
    let mut values = vec![0_u8; input.len()];
    let mut scales = vec![0_u8; rows * (in_features / FP8_BLOCK)];
    for (group, (input, output)) in input
        .chunks_exact(FP8_BLOCK)
        .zip(values.chunks_exact_mut(FP8_BLOCK))
        .enumerate()
    {
        let mut amax = 0.0_f32;
        for (index, &bits) in input.iter().enumerate() {
            let value = bf16_to_f32(bits);
            if !value.is_finite() {
                return Err(format!(
                    "FP8 activation input {} is not finite",
                    group * FP8_BLOCK + index
                )
                .into());
            }
            amax = amax.max(value.abs());
        }
        let scale = pow2_ceil(amax.max(1e-4) / 448.0);
        scales[group] = E8M0Fnu::encode_power_of_two(scale)?.0;
        for (destination, &source) in output.iter_mut().zip(input) {
            *destination = E4M3Fn::encode(bf16_to_f32(source) / scale)?.0;
        }
    }
    Ok(QuantizedFp8Activation {
        values,
        scales,
        rows,
        in_features,
    })
}

pub fn project_fp8_linear_cpu(
    matrix: &Fp8Matrix,
    activation: &QuantizedFp8Activation,
) -> Result<Fp8LinearOutput, Box<dyn Error>> {
    let in_features = matrix.in_features;
    let out_features = matrix.out_features;
    let blocks = in_features / FP8_BLOCK;
    if in_features == 0
        || in_features % FP8_BLOCK != 0
        || activation.in_features != in_features
        || activation.values.len() != activation.rows * in_features
        || activation.scales.len() != activation.rows * blocks
        || matrix.weight.len() != out_features * in_features
        || matrix.scale.len() != out_features.div_ceil(FP8_BLOCK) * blocks
    {
        return Err("FP8 linear layout mismatch".into());
    }
    let mut raw_f32 = vec![0.0_f32; activation.rows * out_features];
    for row in 0..activation.rows {
        for output_row in 0..out_features {
            let mut accumulated = 0.0_f32;
            for block in 0..blocks {
                let input_start = row * in_features + block * FP8_BLOCK;
                let weight_start = output_row * in_features + block * FP8_BLOCK;
                let mut partial = 0.0_f32;
                for column in 0..FP8_BLOCK {
                    partial += E4M3Fn(activation.values[input_start + column]).decode()
                        * E4M3Fn(matrix.weight[weight_start + column]).decode();
                }
                let scale = E8M0Fnu(activation.scales[row * blocks + block]).decode()
                    * E8M0Fnu(matrix.scale[(output_row / FP8_BLOCK) * blocks + block]).decode();
                accumulated += partial * scale;
            }
            raw_f32[row * out_features + output_row] = accumulated;
        }
    }
    let bf16 = raw_f32
        .iter()
        .copied()
        .map(f32_to_bf16_rne)
        .collect::<Vec<_>>();
    Ok(Fp8LinearOutput { raw_f32, bf16 })
}

pub fn project_fp8_cpu(
    projection: &Fp8Projection,
    activation: &QuantizedFp8Activation,
) -> Result<Fp8ProjectionOutput, Box<dyn Error>> {
    let linear = project_fp8_linear_cpu(&projection.matrix, activation)?;
    if projection.norm.len() != projection.matrix.out_features {
        return Err("FP8 projection layout mismatch".into());
    }
    let norm_bf16 = rmsnorm_bf16_width(
        &linear.bf16,
        &projection.norm,
        projection.matrix.out_features,
    )?;
    Ok(Fp8ProjectionOutput {
        raw_f32: linear.raw_f32,
        bf16: linear.bf16,
        norm_bf16,
    })
}

pub fn query_head_rms_bf16(input: &[u16]) -> Result<Vec<u16>, Box<dyn Error>> {
    if input.len() != IDS.len() * Q_HEADS * Q_HEAD_DIM {
        return Err("query head RMS requires pinned [1,4,128,512] input".into());
    }
    let mut output = vec![0_u16; input.len()];
    for (input, output) in input
        .chunks_exact(Q_HEAD_DIM)
        .zip(output.chunks_exact_mut(Q_HEAD_DIM))
    {
        let mut square_sum = 0.0_f32;
        for &bits in input {
            let value = bf16_to_f32(bits);
            square_sum += bf16_to_f32(f32_to_bf16_rne(value * value));
        }
        let mean = bf16_to_f32(f32_to_bf16_rne(square_sum / Q_HEAD_DIM as f32));
        let mean_eps = bf16_to_f32(f32_to_bf16_rne(mean + 1e-6));
        let inv_rms = bf16_to_f32(f32_to_bf16_rne(mean_eps.sqrt().recip()));
        for (output, &input) in output.iter_mut().zip(input) {
            *output = f32_to_bf16_rne(bf16_to_f32(input) * inv_rms);
        }
    }
    Ok(output)
}

pub fn query_freqs_cis_pos0_3() -> Vec<f32> {
    let correction = |rotations: f64| {
        Q_ROPE_DIM as f64 * (65_536.0 / (rotations * 2.0 * std::f64::consts::PI)).ln()
            / (2.0 * 160_000.0_f64.ln())
    };
    let low = correction(32.0).floor().max(0.0) as usize;
    let high = correction(1.0).ceil().min((Q_ROPE_DIM - 1) as f64) as usize;
    let mut output = Vec::with_capacity(IDS.len() * Q_ROPE_DIM);
    for position in 0..IDS.len() {
        for pair in 0..Q_ROPE_DIM / 2 {
            let ramp = ((pair as f32 - low as f32) / (high - low) as f32).clamp(0.0, 1.0);
            let smooth = 1.0 - ramp;
            let base = 160_000.0_f32.powf(-((2 * pair) as f32) / Q_ROPE_DIM as f32);
            let frequency = base / 16.0 * (1.0 - smooth) + base * smooth;
            let angle = position as f32 * frequency;
            output.push(angle.cos());
            output.push(angle.sin());
        }
    }
    output
}

pub fn query_rope_bf16(input: &[u16], freqs: &[f32]) -> Result<Vec<u16>, Box<dyn Error>> {
    rope_bf16(input, freqs, Q_HEADS)
}

fn rope_bf16(input: &[u16], freqs: &[f32], heads: usize) -> Result<Vec<u16>, Box<dyn Error>> {
    if heads == 0
        || input.len() != IDS.len() * heads * Q_HEAD_DIM
        || freqs.len() != IDS.len() * Q_ROPE_DIM
        || freqs.iter().any(|value| !value.is_finite())
    {
        return Err("RoPE requires pinned position-0..3 BF16 rows".into());
    }
    let mut output = input.to_vec();
    for position in 0..IDS.len() {
        for head in 0..heads {
            let head_start = (position * heads + head) * Q_HEAD_DIM;
            for pair in 0..Q_ROPE_DIM / 2 {
                let index = head_start + Q_HEAD_DIM - Q_ROPE_DIM + pair * 2;
                let frequency = position * Q_ROPE_DIM + pair * 2;
                let real = bf16_to_f32(input[index]);
                let imaginary = bf16_to_f32(input[index + 1]);
                let cosine = freqs[frequency];
                let sine = freqs[frequency + 1];
                output[index] = f32_to_bf16_rne(real * cosine - imaginary * sine);
                output[index + 1] = f32_to_bf16_rne(real * sine + imaginary * cosine);
            }
        }
    }
    Ok(output)
}

pub fn finalize_kv_cpu(input: &[u16], freqs: &[f32]) -> Result<KvReadyOutput, Box<dyn Error>> {
    let rope_bf16 = rope_bf16(input, freqs, 1)?;
    let mut ready_bf16 = rope_bf16.clone();
    let mut scale_e8m0 = vec![0_u8; IDS.len() * 7];
    for row in 0..IDS.len() {
        for block in 0..7 {
            let start = row * Q_HEAD_DIM + block * 64;
            let mut amax = 0.0_f32;
            for &bits in &rope_bf16[start..start + 64] {
                amax = amax.max(bf16_to_f32(bits).abs());
            }
            let scale = pow2_ceil(amax.max(1e-4) / 448.0);
            scale_e8m0[row * 7 + block] = E8M0Fnu::encode_power_of_two(scale)?.0;
            for column in 0..64 {
                let bits = E4M3Fn::encode(bf16_to_f32(rope_bf16[start + column]) / scale)?;
                ready_bf16[start + column] = f32_to_bf16_rne(bits.decode() * scale);
            }
        }
    }
    Ok(KvReadyOutput {
        rope_bf16,
        scale_e8m0,
        ready_bf16,
        causal_topk_i32: vec![0, -1, -1, -1, 0, 1, -1, -1, 0, 1, 2, -1, 0, 1, 2, 3],
    })
}

impl V4KvCache {
    pub fn new() -> Self {
        Self {
            data: vec![0_u16; KV_CACHE_POSITIONS * Q_HEAD_DIM],
            committed: false,
        }
    }

    pub fn as_slice(&self) -> &[u16] {
        &self.data
    }

    pub fn bits_equal(&self, other: &Self) -> bool {
        self.committed == other.committed && self.data == other.data
    }

    pub fn prepare_decode_row(
        &self,
        payload: &[u16],
        start_pos: usize,
    ) -> Result<Self, Box<dyn Error>> {
        if payload.len() != Q_HEAD_DIM || payload.iter().any(|&bits| !bf16_to_f32(bits).is_finite())
        {
            return Err("V4 decode KV row is invalid".into());
        }
        let mut scratch = self.clone();
        let start = (start_pos % KV_WINDOW) * Q_HEAD_DIM;
        scratch.data[start..start + Q_HEAD_DIM].copy_from_slice(payload);
        scratch.committed = true;
        Ok(scratch)
    }

    pub fn prepare_compressed_row(
        &self,
        payload: &[u16],
        compressed_index: usize,
    ) -> Result<Self, Box<dyn Error>> {
        if payload.len() != Q_HEAD_DIM
            || compressed_index >= KV_COMPRESSED
            || payload.iter().any(|&bits| !bf16_to_f32(bits).is_finite())
        {
            return Err("V4 compressed KV row is invalid".into());
        }
        let mut scratch = self.clone();
        let start = (KV_WINDOW + compressed_index) * Q_HEAD_DIM;
        scratch.data[start..start + Q_HEAD_DIM].copy_from_slice(payload);
        scratch.committed = true;
        Ok(scratch)
    }

    pub fn commit(
        &mut self,
        payload: &[u16],
        batch_size: usize,
        start_pos: usize,
        sequence_len: usize,
    ) -> Result<(), Box<dyn Error>> {
        preflight_kv_cache_commit(
            self.data.len(),
            payload.len(),
            batch_size,
            start_pos,
            sequence_len,
        )?;
        if self.committed {
            if self.data[..payload.len()] == *payload {
                return Ok(());
            }
            return Err("V4 KV cache already contains a different payload".into());
        }
        self.data[..payload.len()].copy_from_slice(payload);
        self.committed = true;
        Ok(())
    }
}

pub fn prepare_compressor_cpu(
    input_bf16: &[u16],
    weights: &CompressorWeights,
) -> Result<CompressorPrepared, Box<dyn Error>> {
    prepare_compressor_cpu_with_geometry(input_bf16, weights, HIDDEN, Q_HEAD_DIM, COMPRESS_RATIO, 0)
}

pub fn prepare_compressor_cpu_with_geometry(
    input_bf16: &[u16],
    weights: &CompressorWeights,
    hidden: usize,
    outputs: usize,
    compress_ratio: usize,
    start_pos: usize,
) -> Result<CompressorPrepared, Box<dyn Error>> {
    if hidden == 0 || outputs == 0 || !matches!(compress_ratio, 4 | 128) {
        return Err("V4 compressor geometry is invalid".into());
    }
    let rows = input_bf16.len() / hidden;
    if input_bf16.len() % hidden != 0
        || rows == 0
        || weights.wkv_bf16.len() != outputs * hidden
        || weights.wgate_bf16.len() != outputs * hidden
        || weights.ape_f32.len() != compress_ratio * outputs
        || input_bf16
            .iter()
            .chain(&weights.wkv_bf16)
            .chain(&weights.wgate_bf16)
            .any(|&bits| !bf16_to_f32(bits).is_finite())
        || weights.ape_f32.iter().any(|value| !value.is_finite())
    {
        return Err("V4 compressor prepare contract mismatch or non-finite input".into());
    }
    let mut kv_f32 = vec![0.0_f32; rows * outputs];
    let mut score_f32 = vec![0.0_f32; rows * outputs];
    for row in 0..rows {
        for output in 0..outputs {
            let mut kv = 0.0_f32;
            let mut score = 0.0_f32;
            for block in (0..hidden).step_by(192) {
                let mut kv_block = 0.0_f32;
                let mut score_block = 0.0_f32;
                for column in block..(block + 192).min(hidden) {
                    let input = bf16_to_f32(input_bf16[row * hidden + column]);
                    kv_block = input.mul_add(
                        bf16_to_f32(weights.wkv_bf16[output * hidden + column]),
                        kv_block,
                    );
                    score_block = input.mul_add(
                        bf16_to_f32(weights.wgate_bf16[output * hidden + column]),
                        score_block,
                    );
                }
                kv += kv_block;
                score += score_block;
            }
            kv_f32[row * outputs + output] = kv;
            score_f32[row * outputs + output] = score;
        }
    }
    let score_plus_ape_f32 = score_f32
        .chunks_exact(outputs)
        .enumerate()
        .flat_map(|(row, scores)| {
            let ape_start = ((start_pos + row) % compress_ratio) * outputs;
            scores
                .iter()
                .zip(&weights.ape_f32[ape_start..ape_start + outputs])
                .map(|(&score, &ape)| score + ape)
        })
        .collect::<Vec<_>>();
    if kv_f32
        .iter()
        .chain(&score_f32)
        .chain(&score_plus_ape_f32)
        .any(|value| !value.is_finite())
    {
        return Err("V4 compressor prepare produced non-finite output".into());
    }
    Ok(CompressorPrepared {
        kv_f32,
        score_f32,
        score_plus_ape_f32,
    })
}

impl V4CompressorState {
    pub fn new() -> Self {
        Self::new_decode(COMPRESS_RATIO, Q_HEAD_DIM)
            .expect("fixed V4 HCA compressor geometry is valid")
    }

    pub fn new_decode(compress_ratio: usize, head_dim: usize) -> Result<Self, Box<dyn Error>> {
        if !matches!(compress_ratio, 4 | 128) || head_dim == 0 {
            return Err("V4 decode compressor geometry is invalid".into());
        }
        let overlap = compress_ratio == 4;
        let coefficient = 1 + usize::from(overlap);
        let elements = coefficient
            .checked_mul(compress_ratio)
            .and_then(|slots| slots.checked_mul(coefficient))
            .and_then(|value| value.checked_mul(head_dim))
            .ok_or("V4 decode compressor state overflow")?;
        Ok(Self {
            kv: vec![0.0; elements],
            score: vec![f32::NEG_INFINITY; elements],
            compress_ratio,
            head_dim,
            overlap,
            committed: false,
        })
    }

    pub fn kv(&self) -> &[f32] {
        &self.kv
    }

    pub fn score(&self) -> &[f32] {
        &self.score
    }

    pub fn bits_equal(&self, other: &Self) -> bool {
        self.compress_ratio == other.compress_ratio
            && self.head_dim == other.head_dim
            && self.overlap == other.overlap
            && self.committed == other.committed
            && f32_bits_equal(&self.kv, &other.kv)
            && f32_bits_equal(&self.score, &other.score)
    }

    /// Apply one already-projected decode row to a scratch copy. For ratio 4,
    /// the boundary copy mirrors the official overlapping-window state update.
    pub fn prepare_decode_row(
        &self,
        prepared: &CompressorPrepared,
        start_pos: usize,
    ) -> Result<Self, Box<dyn Error>> {
        Ok(self.prepare_decode_row_and_pool(prepared, start_pos)?.0)
    }

    pub fn prepare_decode_row_and_pool(
        &self,
        prepared: &CompressorPrepared,
        start_pos: usize,
    ) -> Result<(Self, Option<Vec<f32>>), Box<dyn Error>> {
        let next_pos = start_pos
            .checked_add(1)
            .ok_or("V4 decode compressor position overflow")?;
        let coefficient = 1 + usize::from(self.overlap);
        let value_width = coefficient
            .checked_mul(self.head_dim)
            .ok_or("V4 decode compressor row overflow")?;
        if prepared.kv_f32.len() != value_width
            || prepared.score_f32.len() != value_width
            || prepared.score_plus_ape_f32.len() != value_width
            || prepared
                .kv_f32
                .iter()
                .chain(&prepared.score_f32)
                .chain(&prepared.score_plus_ape_f32)
                .any(|value| !value.is_finite())
        {
            return Err("V4 decode compressor row contract mismatch".into());
        }
        let mut scratch = self.clone();
        let slot = if self.overlap {
            self.compress_ratio + start_pos % self.compress_ratio
        } else {
            start_pos % self.compress_ratio
        };
        let start = slot
            .checked_mul(value_width)
            .ok_or("V4 decode compressor slot overflow")?;
        scratch.kv[start..start + value_width].copy_from_slice(&prepared.kv_f32);
        scratch.score[start..start + value_width].copy_from_slice(&prepared.score_plus_ape_f32);

        let should_compress = next_pos % self.compress_ratio == 0;
        let pooled = should_compress.then(|| {
            let mut output = vec![0.0_f32; self.head_dim];
            for dimension in 0..self.head_dim {
                let mut maximum = f32::NEG_INFINITY;
                for slot in 0..self.compress_ratio {
                    let index = if self.overlap {
                        if slot < self.compress_ratio {
                            slot * value_width + dimension
                        } else {
                            unreachable!()
                        }
                    } else {
                        slot * value_width + dimension
                    };
                    maximum = maximum.max(scratch.score[index]);
                }
                if self.overlap {
                    for slot in 0..self.compress_ratio {
                        maximum = maximum.max(
                            scratch.score[(self.compress_ratio + slot) * value_width
                                + self.head_dim
                                + dimension],
                        );
                    }
                }
                let mut denominator = 0.0_f32;
                let mut numerator = 0.0_f32;
                for slot in 0..self.compress_ratio {
                    let index = slot * value_width + dimension;
                    let weight = (scratch.score[index] - maximum).exp();
                    denominator += weight;
                    numerator += scratch.kv[index] * weight;
                }
                if self.overlap {
                    for slot in 0..self.compress_ratio {
                        let index =
                            (self.compress_ratio + slot) * value_width + self.head_dim + dimension;
                        let weight = (scratch.score[index] - maximum).exp();
                        denominator += weight;
                        numerator += scratch.kv[index] * weight;
                    }
                }
                output[dimension] = numerator / denominator;
            }
            output
        });

        if self.overlap && should_compress {
            let half = self
                .compress_ratio
                .checked_mul(value_width)
                .ok_or("V4 overlap compressor copy overflow")?;
            let (previous_kv, current_kv) = scratch.kv.split_at_mut(half);
            previous_kv.copy_from_slice(&current_kv[..half]);
            let (previous_score, current_score) = scratch.score.split_at_mut(half);
            previous_score.copy_from_slice(&current_score[..half]);
        }
        scratch.committed = true;
        if pooled
            .as_ref()
            .is_some_and(|values| values.iter().any(|value| !value.is_finite()))
        {
            return Err("V4 compressor pooled boundary is non-finite".into());
        }
        Ok((scratch, pooled))
    }

    pub fn commit(
        &mut self,
        prepared: &CompressorPrepared,
        batch_size: usize,
        start_pos: usize,
        sequence_len: usize,
        compress_ratio: usize,
        overlap: bool,
    ) -> Result<(), Box<dyn Error>> {
        let write_elements = preflight_compressor_commit(
            self.kv.len(),
            self.score.len(),
            prepared.kv_f32.len(),
            prepared.score_plus_ape_f32.len(),
            batch_size,
            start_pos,
            sequence_len,
            compress_ratio,
            overlap,
        )?;
        if prepared
            .kv_f32
            .iter()
            .chain(&prepared.score_plus_ape_f32)
            .any(|value| !value.is_finite())
        {
            return Err("V4 compressor commit payload is non-finite".into());
        }
        if self.committed {
            if compressor_after_matches(
                &self.kv,
                &self.score,
                &prepared.kv_f32,
                &prepared.score_plus_ape_f32,
                write_elements,
            ) {
                return Ok(());
            }
            return Err("V4 compressor state already contains a conflict or is dirty".into());
        }
        if !compressor_initial_matches(&self.kv, &self.score) {
            return Err("V4 compressor initial state is dirty".into());
        }

        let mut scratch_kv = self.kv.clone();
        let mut scratch_score = self.score.clone();
        scratch_kv[..write_elements].copy_from_slice(&prepared.kv_f32);
        scratch_score[..write_elements].copy_from_slice(&prepared.score_plus_ape_f32);
        if !compressor_after_matches(
            &scratch_kv,
            &scratch_score,
            &prepared.kv_f32,
            &prepared.score_plus_ape_f32,
            write_elements,
        ) {
            return Err("V4 compressor paired scratch verification failed".into());
        }
        self.kv = scratch_kv;
        self.score = scratch_score;
        self.committed = true;
        Ok(())
    }
}

fn f32_bits_equal(left: &[f32], right: &[f32]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(left, right)| left.to_bits() == right.to_bits())
}

fn compressor_initial_matches(kv: &[f32], score: &[f32]) -> bool {
    kv.iter().all(|value| value.to_bits() == 0)
        && score
            .iter()
            .all(|value| value.to_bits() == f32::NEG_INFINITY.to_bits())
}

fn compressor_after_matches(
    kv: &[f32],
    score: &[f32],
    expected_kv: &[f32],
    expected_score: &[f32],
    write_elements: usize,
) -> bool {
    f32_bits_equal(&kv[..write_elements], expected_kv)
        && f32_bits_equal(&score[..write_elements], expected_score)
        && kv[write_elements..]
            .iter()
            .all(|value| value.to_bits() == 0)
        && score[write_elements..]
            .iter()
            .all(|value| value.to_bits() == f32::NEG_INFINITY.to_bits())
}

#[allow(clippy::too_many_arguments)]
pub fn preflight_compressor_commit(
    kv_state_elements: usize,
    score_state_elements: usize,
    kv_payload_elements: usize,
    score_payload_elements: usize,
    batch_size: usize,
    start_pos: usize,
    sequence_len: usize,
    compress_ratio: usize,
    overlap: bool,
) -> Result<usize, Box<dyn Error>> {
    let end = start_pos
        .checked_add(sequence_len)
        .ok_or("V4 compressor position overflow")?;
    let write_elements = batch_size
        .checked_mul(sequence_len)
        .and_then(|value| value.checked_mul(Q_HEAD_DIM))
        .ok_or("V4 compressor payload overflow")?;
    write_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or("V4 compressor byte overflow")?;
    if kv_state_elements != COMPRESS_STATE_ELEMENTS
        || score_state_elements != COMPRESS_STATE_ELEMENTS
        || kv_payload_elements != write_elements
        || score_payload_elements != write_elements
        || batch_size != 1
        || start_pos != 0
        || sequence_len != IDS.len()
        || compress_ratio != COMPRESS_RATIO
        || overlap
        || end > compress_ratio
    {
        return Err("V4 compressor commit contract mismatch".into());
    }
    Ok(write_elements)
}

pub fn preflight_kv_cache_commit(
    cache_elements: usize,
    payload_elements: usize,
    batch_size: usize,
    start_pos: usize,
    sequence_len: usize,
) -> Result<usize, Box<dyn Error>> {
    let end = start_pos
        .checked_add(sequence_len)
        .ok_or("V4 KV cache position overflow")?;
    let expected_payload = batch_size
        .checked_mul(sequence_len)
        .and_then(|value| value.checked_mul(Q_HEAD_DIM))
        .ok_or("V4 KV cache payload overflow")?;
    let bytes = expected_payload
        .checked_mul(std::mem::size_of::<u16>())
        .ok_or("V4 KV cache byte overflow")?;
    if cache_elements != KV_CACHE_POSITIONS * Q_HEAD_DIM
        || batch_size != 1
        || start_pos != 0
        || sequence_len != IDS.len()
        || end > KV_WINDOW
        || payload_elements != expected_payload
    {
        return Err("V4 KV cache commit contract mismatch".into());
    }
    Ok(bytes)
}

pub fn validate_model_dir(model_dir: &Path) -> Result<(), Box<dyn Error>> {
    let mountinfo = std::fs::read_to_string("/proc/self/mountinfo")?;
    let source = validate_model_dir_with(model_dir, &mountinfo, PINNED_SHARD_BYTES)?;
    if !std::fs::metadata(&source)?.file_type().is_block_device() {
        return Err(format!("V4 model ext4 source {source} is not a block device").into());
    }
    Ok(())
}

fn validate_model_dir_with(
    model_dir: &Path,
    mountinfo: &str,
    expected_shard_bytes: u64,
) -> Result<String, Box<dyn Error>> {
    if model_dir.starts_with("/mnt/c") || model_dir.starts_with("/mnt/d") {
        return Err("V4 model storage on DrvFS is forbidden".into());
    }
    let root_metadata = model_dir.symlink_metadata()?;
    if root_metadata.file_type().is_symlink() || !root_metadata.file_type().is_dir() {
        return Err("V4 model directory must be a regular directory, not a symlink".into());
    }
    let canonical = model_dir.canonicalize()?;
    if canonical.file_name().and_then(|name| name.to_str())
        != Some(&format!("deepseek-v4-pro-{REVISION}"))
    {
        return Err("V4 state model directory is not the pinned revision".into());
    }
    let mut best: Option<(&str, &str, usize)> = None;
    for line in mountinfo.lines() {
        let Some((mount, filesystem)) = line.split_once(" - ") else {
            continue;
        };
        let mountpoint = mount.split_whitespace().nth(4).unwrap_or("");
        let mut filesystem = filesystem.split_whitespace();
        let fstype = filesystem.next().unwrap_or("");
        let source = filesystem.next().unwrap_or("");
        if canonical.starts_with(Path::new(mountpoint))
            && best.is_none_or(|(_, _, length)| mountpoint.len() > length)
        {
            best = Some((fstype, source, mountpoint.len()));
        }
    }
    let source = match best {
        Some(("ext4", source, _)) if source.starts_with("/dev/") => source,
        Some((fstype, source, _)) => {
            return Err(format!(
                "V4 model path must be on block-backed ext4, got {source} {fstype}"
            )
            .into());
        }
        None => return Err("V4 model mount identity is unavailable".into()),
    };

    let root_device = canonical.metadata()?.dev();
    let mut shard_bytes = 0_u64;
    for name in std::iter::once("model.safetensors.index.json".to_owned())
        .chain((1..=64).map(|number| format!("model-{number:05}-of-00064.safetensors")))
    {
        let path = canonical.join(&name);
        let metadata = path
            .symlink_metadata()
            .map_err(|error| format!("{name}: missing pinned source: {error}"))?;
        if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
            return Err(format!("{name}: pinned source must be a regular non-symlink file").into());
        }
        let resolved = path.canonicalize()?;
        if resolved.parent() != Some(canonical.as_path()) || metadata.dev() != root_device {
            return Err(format!("{name}: pinned source escaped the model root").into());
        }
        if name.ends_with(".safetensors") {
            shard_bytes = shard_bytes
                .checked_add(metadata.len())
                .ok_or("V4 pinned shard byte count overflow")?;
        }
    }
    if shard_bytes != expected_shard_bytes {
        return Err(
            format!("V4 pinned shard bytes {shard_bytes} != {expected_shard_bytes}").into(),
        );
    }
    Ok(source.to_owned())
}

fn read_projection(
    shard: &ShardFile,
    weight_name: &str,
    scale_name: &str,
    norm_name: &str,
    out_features: usize,
) -> Result<Fp8Projection, Box<dyn Error>> {
    let matrix = read_matrix(shard, weight_name, scale_name, HIDDEN, out_features)?;
    Ok(Fp8Projection {
        matrix,
        norm: read_bf16(shard, norm_name, &[out_features])?,
    })
}

fn read_matrix(
    shard: &ShardFile,
    weight_name: &str,
    scale_name: &str,
    in_features: usize,
    out_features: usize,
) -> Result<Fp8Matrix, Box<dyn Error>> {
    let weight = read_u8(
        shard,
        weight_name,
        TensorDtype::F8E4M3,
        &[out_features, in_features],
    )?;
    let scale = read_u8(
        shard,
        scale_name,
        TensorDtype::F8E8M0,
        &[out_features.div_ceil(FP8_BLOCK), in_features / FP8_BLOCK],
    )?;
    if weight.iter().any(|bits| bits & 0x7f == 0x7f) {
        return Err(format!("{weight_name} contains E4M3FN NaN").into());
    }
    if scale.contains(&255) {
        return Err(format!("{scale_name} contains E8M0 NaN").into());
    }
    Ok(Fp8Matrix {
        weight,
        scale,
        in_features,
        out_features,
    })
}

fn read_u8(
    file: &ShardFile,
    name: &str,
    dtype: TensorDtype,
    shape: &[usize],
) -> Result<Vec<u8>, Box<dyn Error>> {
    checked(file, name, dtype, shape)?;
    let mut out = vec![0_u8; shape.iter().product()];
    file.read_at(name, 0, &mut out)?;
    Ok(out)
}

fn pow2_ceil(value: f32) -> f32 {
    let bits = value.to_bits();
    let exponent = ((bits >> 23) & 0xff) as i32 - 127;
    let has_fraction = bits & 0x007f_ffff != 0;
    f32::from_bits(((exponent + has_fraction as i32 + 127) as u32) << 23)
}

pub fn fp8_projection_fixture() -> Result<Fp8ProjectionFixture, Box<dyn Error>> {
    const B: &[u8] = include_bytes!("../fixtures/v4_fp8_projection_pytorch_oracle.bin");
    if B.len() != 4_189_556 || B.get(..8) != Some(b"CV4P8001") {
        return Err("FP8 projection fixture layout mismatch".into());
    }
    Ok(Fp8ProjectionFixture {
        input_bf16: bytes_bf16(&B[8..57_352]),
        activation: B[57_352..86_024].to_vec(),
        activation_scale: B[86_024..86_248].to_vec(),
        wq_f32: bytes_f32(&B[86_248..110_824]),
        wq_bf16: bytes_bf16(&B[110_824..123_112]),
        q_norm_bf16: bytes_bf16(&B[123_112..135_400]),
        wkv_f32: bytes_f32(&B[135_400..143_592]),
        wkv_bf16: bytes_bf16(&B[143_592..147_688]),
        kv_norm_bf16: bytes_bf16(&B[147_688..151_784]),
        wq_b_activation: B[151_784..157_928].to_vec(),
        wq_b_activation_scale: B[157_928..157_976].to_vec(),
        wq_b_f32: bytes_f32(&B[157_976..1_206_552]),
        wq_b_bf16: bytes_bf16(&B[1_206_552..1_730_840]),
        q_head_rms_bf16: bytes_bf16(&B[1_730_840..2_255_128]),
        freqs_cis_f32_pairs: bytes_f32(&B[2_255_128..2_256_152]),
        q_rope_bf16: bytes_bf16(&B[2_256_152..2_780_440]),
        kv_rope_bf16: bytes_bf16(&B[2_780_440..2_784_536]),
        kv_scale_e8m0: B[2_784_536..2_784_564].to_vec(),
        kv_ready_bf16: bytes_bf16(&B[2_784_564..2_788_660]),
        causal_topk_i32: bytes_i32(&B[2_788_660..2_788_724]),
        cache_initial_bf16: bytes_bf16(&B[2_788_724..2_952_564]),
        cache_after_bf16: bytes_bf16(&B[2_952_564..3_116_404]),
        compressor_kv_f32: bytes_f32(&B[3_116_404..3_124_596]),
        compressor_score_f32: bytes_f32(&B[3_124_596..3_132_788]),
        compressor_score_plus_ape_f32: bytes_f32(&B[3_132_788..3_140_980]),
        compressor_kv_state_initial_f32: bytes_f32(&B[3_140_980..3_403_124]),
        compressor_score_state_initial_f32: bytes_f32(&B[3_403_124..3_665_268]),
        compressor_kv_state_after_f32: bytes_f32(&B[3_665_268..3_927_412]),
        compressor_score_state_after_f32: bytes_f32(&B[3_927_412..4_189_556]),
    })
}

pub fn fixture_outputs() -> Result<(Vec<u16>, Vec<u16>), Box<dyn Error>> {
    let stages = fixture_stages()?;
    Ok((stages.collapsed, stages.norm))
}

pub fn fixture_stages() -> Result<FixtureStages, Box<dyn Error>> {
    const B: &[u8] = include_bytes!("../fixtures/v4_state_pytorch_oracle.bin");
    if B.len() != 183_304 || B.get(..8) != Some(b"CV4S0001") {
        return Err("state fixture layout mismatch".into());
    }
    Ok(FixtureStages {
        collapsed: bytes_bf16(&B[57_352..114_696]),
        norm: bytes_bf16(&B[114_696..172_040]),
        post: bytes_f32(&B[172_040..172_104]),
        comb: bytes_f32(&B[172_104..172_360]),
        mixes: bytes_f32(&B[172_360..172_744]),
        pre: bytes_f32(&B[172_744..172_808]),
        sinkhorn_trace: bytes_f32(&B[172_808..183_304]),
    })
}
fn bytes_bf16(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|x| u16::from_le_bytes([x[0], x[1]]))
        .collect()
}
fn bytes_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|x| f32::from_le_bytes([x[0], x[1], x[2], x[3]]))
        .collect()
}
fn bytes_i32(bytes: &[u8]) -> Vec<i32> {
    bytes
        .chunks_exact(4)
        .map(|x| i32::from_le_bytes([x[0], x[1], x[2], x[3]]))
        .collect()
}

pub fn run_cpu(state: &L0State, rows: usize) -> Result<MhcPre, Box<dyn Error>> {
    let mixes = mhc_linear_cpu(state, rows)?;
    mhc_sinkhorn_cpu(state, &mixes, rows)
}

pub fn mhc_linear_cpu(state: &L0State, rows: usize) -> Result<Vec<f32>, Box<dyn Error>> {
    if rows == 0
        || state.embedding.len() != rows * HIDDEN
        || state.hc_fn.len() != MIX * HC * HIDDEN
        || state.hc_base.len() != MIX
        || state.hc_scale.len() != 3
    {
        return Err("V4 state layout mismatch".into());
    }
    let mut mixes = vec![0.0; rows * MIX];
    for row in 0..rows {
        let input = &state.embedding[row * HIDDEN..][..HIDDEN];
        let mut squares = Vec::with_capacity(HC * HIDDEN);
        for _ in 0..HC {
            squares.extend(input.iter().map(|&bits| {
                let value = bf16_to_f32(bits);
                value * value
            }));
        }
        let square = crate::v4_router::pytorch_contiguous_sum_f32(&squares);
        let inv = 1.0 / (square / (HC * HIDDEN) as f32 + EPS).sqrt();
        for out in 0..MIX {
            let mut dot = 0.0;
            for block in (0..HC * HIDDEN).step_by(192) {
                let mut partial = 0.0;
                for column in block..(block + 192).min(HC * HIDDEN) {
                    partial = bf16_to_f32(input[column % HIDDEN])
                        .mul_add(state.hc_fn[out * HC * HIDDEN + column], partial);
                }
                dot += partial;
            }
            mixes[row * MIX + out] = dot * inv;
        }
    }
    Ok(mixes)
}

pub fn mhc_sinkhorn_cpu(
    state: &L0State,
    mixes: &[f32],
    rows: usize,
) -> Result<MhcPre, Box<dyn Error>> {
    if rows == 0
        || state.embedding.len() != rows * HIDDEN
        || mixes.len() != rows * MIX
        || state.hc_base.len() != MIX
        || state.hc_scale.len() != 3
    {
        return Err("V4 state layout mismatch".into());
    }
    let mut collapsed = vec![0_u16; rows * HIDDEN];
    let mut pre_out = vec![0.0; rows * HC];
    let mut post = vec![0.0; rows * HC];
    let mut comb = vec![0.0; rows * HC * HC];
    let mut sinkhorn_trace = vec![0.0; rows * 41 * HC * HC];
    for row in 0..rows {
        let input = &state.embedding[row * HIDDEN..][..HIDDEN];
        let mixes = &mixes[row * MIX..][..MIX];
        let mut pre = [0.0; HC];
        for i in 0..HC {
            pre[i] = sigmoid(mixes[i] * state.hc_scale[0] + state.hc_base[i]) + EPS;
            pre_out[row * HC + i] = pre[i];
            post[row * HC + i] =
                2.0 * sigmoid(mixes[HC + i] * state.hc_scale[1] + state.hc_base[HC + i]);
        }
        let base = row * HC * HC;
        for i in 0..HC {
            for j in 0..HC {
                comb[base + i * HC + j] = mixes[2 * HC + i * HC + j] * state.hc_scale[2]
                    + state.hc_base[2 * HC + i * HC + j];
            }
        }
        sinkhorn_trace[row * 41 * HC * HC..][..HC * HC].copy_from_slice(&comb[base..][..HC * HC]);
        for i in 0..HC {
            let mut max = f32::NEG_INFINITY;
            for j in 0..HC {
                max = max.max(comb[base + i * HC + j]);
            }
            let mut exponentials = [0.0_f32; HC];
            for j in 0..HC {
                exponentials[j] = pytorch_sleef_expf(comb[base + i * HC + j] - max);
            }
            let sum = (exponentials[0] + exponentials[2]) + (exponentials[1] + exponentials[3]);
            let reciprocal = 1.0 / sum;
            for j in 0..HC {
                let probability = exponentials[j] * reciprocal;
                comb[base + i * HC + j] = probability + EPS;
            }
        }
        sinkhorn_trace[row * 41 * HC * HC + HC * HC..][..HC * HC]
            .copy_from_slice(&comb[base..][..HC * HC]);
        for step in 0..20 {
            for j in 0..HC {
                let sum = (0..HC).map(|i| comb[base + i * HC + j]).sum::<f32>();
                for i in 0..HC {
                    comb[base + i * HC + j] /= sum + EPS;
                }
            }
            let trace = 2 + step * 2;
            sinkhorn_trace[row * 41 * HC * HC + trace * HC * HC..][..HC * HC]
                .copy_from_slice(&comb[base..][..HC * HC]);
            if step != 19 {
                for i in 0..HC {
                    let sum = (0..HC).map(|j| comb[base + i * HC + j]).sum::<f32>();
                    for j in 0..HC {
                        comb[base + i * HC + j] /= sum + EPS;
                    }
                }
                sinkhorn_trace[row * 41 * HC * HC + (trace + 1) * HC * HC..][..HC * HC]
                    .copy_from_slice(&comb[base..][..HC * HC]);
            }
        }
        for col in 0..HIDDEN {
            let x = bf16_to_f32(input[col]);
            collapsed[row * HIDDEN + col] = f32_to_bf16_rne((0..HC).map(|i| pre[i] * x).sum());
        }
    }
    Ok(MhcPre {
        pre: pre_out,
        collapsed,
        post,
        comb,
        sinkhorn_trace,
    })
}

pub fn run_cpu_norm(state: &L0State, rows: usize) -> Result<(MhcPre, Vec<u16>), Box<dyn Error>> {
    let mhc = run_cpu(state, rows)?;
    let norm = rmsnorm_bf16(&mhc.collapsed, &state.attn_norm)?;
    Ok((mhc, norm))
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// Adapted from SLEEF xexpf at revision 5a1d179d, Copyright Naoki Shibata and
// contributors, under the Boost Software License 1.0:
// Permission is hereby granted, free of charge, to any person or organization
// obtaining a copy of the software and accompanying documentation covered by
// this license (the "Software") to use, reproduce, display, distribute,
// execute, and transmit the Software, and to prepare derivative works of the
// Software, and to permit third-parties to whom the Software is furnished to
// do so, all subject to the following: The copyright notices in the Software
// and this entire statement, including the above license grant, this
// restriction and the following disclaimer, must be included in all copies of
// the Software, in whole or in part, and all derivative works of the Software,
// unless such copies or derivative works are solely in the form of
// machine-executable object code generated by a source language processor.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
fn pytorch_sleef_expf(value: f32) -> f32 {
    let exponent = (value * 1.442_695_040_888_963_4_f32).round_ties_even() as i32;
    let exponent_f32 = exponent as f32;
    let mut reduced = exponent_f32.mul_add(-0.693_145_751_953_125_f32, value);
    reduced = exponent_f32.mul_add(-1.428_606_765_330_187_e-6_f32, reduced);

    let mut polynomial = 0.000_198_527_617_612_853_65_f32;
    polynomial = polynomial.mul_add(reduced, 0.001_393_043_552_525_341_5_f32);
    polynomial = polynomial.mul_add(reduced, 0.008_333_360_776_305_199_f32);
    polynomial = polynomial.mul_add(reduced, 0.041_666_485_369_205_475_f32);
    polynomial = polynomial.mul_add(reduced, 0.166_666_671_633_720_4_f32);
    polynomial = polynomial.mul_add(reduced, 0.5);
    let mut result = (reduced * reduced).mul_add(polynomial, reduced) + 1.0;
    let high = exponent >> 1;
    result *= f32::from_bits(((high + 127) as u32) << 23);
    result *= f32::from_bits(((exponent - high + 127) as u32) << 23);
    if value < -104.0 {
        0.0
    } else if value > 100.0 {
        f32::INFINITY
    } else {
        result
    }
}

fn checked(
    file: &ShardFile,
    name: &str,
    dtype: TensorDtype,
    shape: &[usize],
) -> Result<(), Box<dyn Error>> {
    let t = file.tensor(name)?;
    if t.dtype != dtype || t.shape != shape.iter().map(|&x| x as u64).collect::<Vec<_>>() {
        return Err(format!("{name} metadata mismatch").into());
    }
    Ok(())
}
fn read_bf16(file: &ShardFile, name: &str, shape: &[usize]) -> Result<Vec<u16>, Box<dyn Error>> {
    checked(file, name, TensorDtype::Bf16, shape)?;
    let mut out = vec![0_u16; shape.iter().product()];
    let bytes = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr().cast(), out.len() * 2) };
    file.read_at(name, 0, bytes)?;
    Ok(out)
}
fn read_f32(file: &ShardFile, name: &str, shape: &[usize]) -> Result<Vec<f32>, Box<dyn Error>> {
    checked(file, name, TensorDtype::F32, shape)?;
    let mut out = vec![0_f32; shape.iter().product()];
    let bytes = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr().cast(), out.len() * 4) };
    file.read_at(name, 0, bytes)?;
    Ok(out)
}

fn read_i64_rows(
    file: &ShardFile,
    name: &str,
    rows: &[i64],
    width: usize,
) -> Result<Vec<i64>, Box<dyn Error>> {
    checked(file, name, TensorDtype::I64, &[129_280, width])?;
    let mut output = Vec::with_capacity(rows.len() * width);
    for &row in rows {
        let row = usize::try_from(row)?;
        let mut values = vec![0_i64; width];
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(
                values.as_mut_ptr().cast(),
                width * std::mem::size_of::<i64>(),
            )
        };
        file.read_at(
            name,
            u64::try_from(
                row.checked_mul(width * std::mem::size_of::<i64>())
                    .ok_or("V4 hash router row offset overflow")?,
            )?,
            bytes,
        )?;
        output.extend(values);
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    const JSON: &str = include_str!("../fixtures/v4_state_pytorch_oracle.json");
    const BIN: &[u8] = include_bytes!("../fixtures/v4_state_pytorch_oracle.bin");
    const FP8_JSON: &str = include_str!("../fixtures/v4_fp8_projection_pytorch_oracle.json");
    const FP8_BIN: &[u8] = include_bytes!("../fixtures/v4_fp8_projection_pytorch_oracle.bin");
    const MINI_SERIAL_JSON: &str = include_str!("../fixtures/v4_mini_serial_oracle_v1.json");

    #[test]
    fn token_id_validation_is_not_limited_to_fixture_batch_width() {
        assert!(validate_ids(&[0, 1, 2, 3, 4]).is_ok());
        assert!(validate_ids(&[]).is_err());
        assert!(validate_ids(&[-1]).is_err());
        assert!(validate_ids(&[i64::try_from(VOCAB).unwrap()]).is_err());
    }

    #[test]
    fn rejects_drvfs_model_paths_before_reading_metadata() {
        let error = validate_model_dir(Path::new(
            "/mnt/d/deepseek-v4-pro-b5968e9190ef611bbf34a7229255be88a0e937c1",
        ))
        .unwrap_err();
        assert!(error.to_string().contains("DrvFS"));
    }

    fn fake_pinned_model() -> std::path::PathBuf {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let parent = std::env::temp_dir().join(format!("colibri-v4-preflight-{unique}"));
        let root = parent.join(format!("deepseek-v4-pro-{REVISION}"));
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("model.safetensors.index.json"), b"{}").unwrap();
        for number in 1..=64 {
            std::fs::write(
                root.join(format!("model-{number:05}-of-00064.safetensors")),
                [u8::try_from(number).unwrap()],
            )
            .unwrap();
        }
        root
    }

    #[test]
    fn model_preflight_rejects_symlink_shard_and_drvfs_mount() {
        let root = fake_pinned_model();
        let ext4 = format!("1 0 0:1 / {} rw - ext4 /dev/sde rw", root.display());
        validate_model_dir_with(&root, &ext4, 64).unwrap();

        let shard = root.join("model-00064-of-00064.safetensors");
        std::fs::remove_file(&shard).unwrap();
        std::os::unix::fs::symlink(root.join("model-00063-of-00064.safetensors"), &shard).unwrap();
        assert!(
            validate_model_dir_with(&root, &ext4, 64)
                .unwrap_err()
                .to_string()
                .contains("non-symlink")
        );

        std::fs::remove_file(&shard).unwrap();
        std::fs::write(&shard, [64_u8]).unwrap();
        let drvfs = format!("1 0 0:1 / {} rw - 9p drvfs rw", root.display());
        assert!(
            validate_model_dir_with(&root, &drvfs, 64)
                .unwrap_err()
                .to_string()
                .contains("block-backed ext4")
        );
        std::fs::remove_dir_all(root.parent().unwrap()).unwrap();
    }

    #[test]
    fn enumerates_the_pinned_61_layer_compression_ratios() {
        let ratios = (0..V4_LAYER_COUNT)
            .map(|layer| compression_ratio_for_layer(layer).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(ratios.len(), 61);
        assert_eq!(ratios[0], CompressionRatio::Hca128);
        assert_eq!(ratios[1], CompressionRatio::Hca128);
        assert_eq!(ratios[2], CompressionRatio::Csa4);
        assert_eq!(ratios[60], CompressionRatio::Csa4);
        assert_eq!(
            ratios
                .iter()
                .filter(|&&ratio| ratio == CompressionRatio::Csa4)
                .count(),
            30
        );
        assert_eq!(
            ratios
                .iter()
                .filter(|&&ratio| ratio == CompressionRatio::Hca128)
                .count(),
            31
        );
        assert_eq!(CompressionRatio::Csa4.get(), 4);
        assert_eq!(CompressionRatio::Hca128.get(), 128);
        assert!(compression_ratio_for_layer(V4_LAYER_COUNT).is_err());
    }

    #[test]
    fn resolves_pinned_l0_l2_l3_l60_tensor_manifests_without_weight_reads() {
        let expected = [
            (
                0,
                "model-00002-of-00064.safetensors",
                [128, 512],
                false,
                true,
            ),
            (
                2,
                "model-00004-of-00064.safetensors",
                [4, 1_024],
                true,
                true,
            ),
            (
                3,
                "model-00005-of-00064.safetensors",
                [128, 512],
                false,
                false,
            ),
            (
                60,
                "model-00062-of-00064.safetensors",
                [4, 1_024],
                true,
                false,
            ),
        ];
        let mut weight_map = std::collections::BTreeMap::new();
        let mut manifests = Vec::new();
        for (layer, shard, _, _, _) in expected {
            let tensors = layer_tensor_manifest(layer, 0).unwrap();
            for tensor in &tensors {
                weight_map.insert(tensor.name.clone(), shard.to_owned());
            }
            manifests.push((layer, shard, tensors));
        }
        let index_bytes = serde_json::to_vec(&serde_json::json!({
            "weight_map": weight_map.clone()
        }))
        .unwrap();
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("colibri-v4-layer-headers-{unique}"));
        std::fs::create_dir(&root).unwrap();
        for (_, shard, tensors) in &manifests {
            let mut offset = 0_u64;
            let mut header = serde_json::Map::new();
            for tensor in tensors {
                let dtype = match tensor.dtype {
                    TensorDtype::Bf16 => "BF16",
                    TensorDtype::F32 => "F32",
                    TensorDtype::I64 => "I64",
                    TensorDtype::Fp4E2M1x2 => "I8",
                    TensorDtype::F8E4M3 => "F8_E4M3",
                    TensorDtype::F8E8M0 => "F8_E8M0",
                    _ => panic!("unexpected pinned layer dtype"),
                };
                header.insert(
                    tensor.name.clone(),
                    serde_json::json!({
                        "dtype": dtype,
                        "shape": tensor.shape,
                        "data_offsets": [offset, offset + tensor.byte_len],
                    }),
                );
                offset += tensor.byte_len;
            }
            let header = serde_json::to_vec(&header).unwrap();
            let mut file = std::fs::File::create(root.join(shard)).unwrap();
            use std::io::Write;
            file.write_all(&(header.len() as u64).to_le_bytes())
                .unwrap();
            file.write_all(&header).unwrap();
            file.set_len(8 + header.len() as u64 + offset).unwrap();
        }
        let source = SourceManifest::new(
            REVISION,
            expected.map(|(_, shard, _, _, _)| (shard.to_owned(), root.join(shard))),
        );
        let index = IndexedSafetensors::from_index_reader(source, index_bytes.as_slice()).unwrap();
        let mut wrong_map = weight_map;
        wrong_map.insert(
            manifests[0].2[0].name.clone(),
            "model-00005-of-00064.safetensors".to_owned(),
        );
        let wrong_index = IndexedSafetensors::from_index_reader(
            SourceManifest::new(
                REVISION,
                expected.map(|(_, shard, _, _, _)| (shard.to_owned(), root.join(shard))),
            ),
            serde_json::to_vec(&serde_json::json!({"weight_map": wrong_map}))
                .unwrap()
                .as_slice(),
        )
        .unwrap();
        assert!(
            validate_layer_locations(
                &wrong_index,
                &manifests[0].2,
                "model-00002-of-00064.safetensors"
            )
            .is_err()
        );

        for ((layer, expected_shard, ape_shape, has_indexer, has_hash), (_, _, tensors)) in
            expected.into_iter().zip(manifests)
        {
            assert!(
                tensors
                    .iter()
                    .all(|tensor| tensor.name.starts_with(&format!("layers.{layer}.")))
            );
            let ape = tensors
                .iter()
                .find(|tensor| tensor.name.ends_with("attn.compressor.ape"))
                .unwrap();
            assert_eq!(ape.dtype, TensorDtype::F32);
            assert_eq!(ape.shape, ape_shape);
            assert_eq!(
                tensors
                    .iter()
                    .any(|tensor| tensor.name.ends_with("attn.indexer.wq_b.weight")),
                has_indexer
            );
            assert_eq!(
                tensors
                    .iter()
                    .any(|tensor| tensor.name.ends_with("ffn.gate.tid2eid")),
                has_hash
            );
            let expert = tensors
                .iter()
                .find(|tensor| tensor.name.ends_with("ffn.experts.0.w1.weight"))
                .unwrap();
            assert_eq!(expert.dtype, TensorDtype::Fp4E2M1x2);
            assert_eq!(expert.shape, [3_072, 3_584]);
            for tensor in &tensors {
                let location = resolve_layer_tensor(&index, tensor).unwrap();
                assert_eq!(location.shard, expected_shard);
                assert_eq!(
                    location.path.file_name().and_then(|name| name.to_str()),
                    Some(expected_shard)
                );
                let range = resolve_layer_tensor_range(&index, tensor).unwrap();
                assert_eq!(range.location, location);
                assert_eq!(range.info.dtype, tensor.dtype);
                assert_eq!(range.info.shape, tensor.shape);
                assert_eq!(range.info.byte_len, tensor.byte_len);
                assert!(range.offset >= 8);
            }
        }
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn layer_router_boundary_executes_hash_and_learned_math() {
        let mut gate = vec![f32_to_bf16_rne(0.0); crate::v4_router::V4_ROUTED_EXPERTS * HIDDEN];
        for expert in 0..crate::v4_router::V4_ROUTED_EXPERTS {
            gate[expert * HIDDEN] = f32_to_bf16_rne(expert as f32 / 384.0);
        }
        let input = {
            let mut row = vec![f32_to_bf16_rne(0.0); HIDDEN];
            row[0] = f32_to_bf16_rne(1.0);
            row
        };
        let empty_compressor = || LayerCompressorWeights {
            wkv_bf16: Vec::new(),
            wgate_bf16: Vec::new(),
            ape_f32: Vec::new(),
            norm_bf16: Vec::new(),
        };
        let hash = LayerStateWeights {
            layer: 2,
            compression_ratio: CompressionRatio::Csa4,
            hc_fn: Vec::new(),
            hc_base: Vec::new(),
            hc_scale: Vec::new(),
            attn_norm: Vec::new(),
            compressor: empty_compressor(),
            indexer: None,
            ffn_norm: vec![f32_to_bf16_rne(1.0); HIDDEN],
            gate: gate.clone(),
            router: LayerRouterWeights::Hash {
                ids: vec![6, 1, 5, 2, 4, 3],
            },
        };
        assert_eq!(
            route_layer_cpu(&hash, &input).unwrap().ids,
            [6, 1, 5, 2, 4, 3]
        );

        let mut bias = vec![0.0; crate::v4_router::V4_ROUTED_EXPERTS];
        bias[1] = 100.0;
        let learned = LayerStateWeights {
            layer: 3,
            compression_ratio: CompressionRatio::Hca128,
            hc_fn: Vec::new(),
            hc_base: Vec::new(),
            hc_scale: Vec::new(),
            attn_norm: Vec::new(),
            compressor: empty_compressor(),
            indexer: None,
            ffn_norm: vec![f32_to_bf16_rne(1.0); HIDDEN],
            gate,
            router: LayerRouterWeights::Learned { bias },
        };
        assert_eq!(route_layer_cpu(&learned, &input).unwrap().ids[0], 1);
    }

    #[test]
    fn serial_manifest_covers_all_61_main_layers_in_order() {
        let manifests = serial_layer_manifests(0).unwrap();
        assert_eq!(manifests.len(), 61);
        for (layer, manifest) in manifests.iter().enumerate() {
            assert_eq!(manifest.layer, layer);
            assert_eq!(
                manifest.shard,
                format!("model-{:05}-of-00064.safetensors", layer + 2)
            );
            assert!(manifest.contract_range_bytes > 0);
        }
        assert_eq!(manifests[0].compression_ratio, CompressionRatio::Hca128);
        assert_eq!(manifests[2].compression_ratio, CompressionRatio::Csa4);
        assert_eq!(manifests[60].compression_ratio, CompressionRatio::Csa4);
    }

    #[test]
    fn mini_serial_oracle_schema_covers_state_routes_and_final_logits() {
        let fixture: serde_json::Value = serde_json::from_str(MINI_SERIAL_JSON).unwrap();
        assert_eq!(fixture["schema"], "colibri-v4-mini-serial-oracle-v1");
        assert_eq!(fixture["layers"][0]["ratio"], 128);
        assert!(fixture["layers"][0]["indexer"].is_null());
        assert_eq!(fixture["layers"][1]["ratio"], 4);
        assert!(fixture["layers"][1]["indexer"]["kv_state"]["fnv1a64"].is_string());
        for layer in fixture["layers"].as_array().unwrap() {
            assert!(layer["hidden_bf16"]["fnv1a64"].is_string());
            assert!(layer["kv_cache_bf16"]["sha256"].is_string());
            assert_eq!(layer["route_ids"].as_array().unwrap().len(), 2);
            assert_eq!(layer["route_weights_f32"].as_array().unwrap().len(), 2);
        }
        assert_eq!(fixture["final"]["argmax"].as_array().unwrap().len(), 4);
        assert!(fixture["final"]["logits_f32"]["sha256"].is_string());
    }

    #[test]
    fn final_hc_norm_head_logits_and_argmax_match_independent_oracle_exactly() {
        let fixture: serde_json::Value = serde_json::from_str(MINI_SERIAL_JSON).unwrap();
        let contract = &fixture["final"]["contract"];
        let ints = |name: &str| {
            contract[name]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_i64().unwrap())
                .collect::<Vec<_>>()
        };
        let floats = |name: &str| {
            contract[name]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_f64().unwrap() as f32)
                .collect::<Vec<_>>()
        };
        let hc_hidden = ints("hc_hidden_bf16_bits")
            .into_iter()
            .map(|bits| bits as u16)
            .collect::<Vec<_>>();
        let rows = contract["rows"].as_u64().unwrap() as usize;
        let hc = contract["hc"].as_u64().unwrap() as usize;
        let dim = contract["dim"].as_u64().unwrap() as usize;
        let pre = final_hc_gate_cpu(
            &hc_hidden,
            rows,
            hc,
            dim,
            &floats("hc_fn_f32"),
            contract["hc_scale_f32"].as_f64().unwrap() as f32,
            &floats("hc_base_f32"),
        )
        .unwrap();
        assert_eq!(
            pre.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
            floats("hc_pre_f32")
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        let output = final_hc_head_cpu(
            &hc_hidden,
            rows,
            hc,
            dim,
            &pre,
            &floats("norm_weight_f32"),
            &floats("head_weight_f32"),
            contract["vocab"].as_u64().unwrap() as usize,
        )
        .unwrap();
        assert_eq!(
            output.collapsed_bf16,
            ints("collapsed_bf16_bits")
                .into_iter()
                .map(|bits| bits as u16)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            output.normalized_bf16,
            ints("norm_bf16_bits")
                .into_iter()
                .map(|bits| bits as u16)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            output
                .logits_f32
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            ints("logits_f32_bits")
                .into_iter()
                .map(|bits| bits as u32)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            output.argmax,
            fixture["final"]["argmax"]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_u64().unwrap() as usize)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn generic_compressor_cpu_covers_hca_csa_and_indexer_geometries() {
        let oracle: serde_json::Value = serde_json::from_str(MINI_SERIAL_JSON).unwrap();
        let contract = &oracle["compressor_projection_contract"];
        assert_eq!(contract["hidden"], 2);
        assert_eq!(contract["rows"], 2);
        for case in contract["cases"].as_array().unwrap() {
            let outputs = case["outputs"].as_u64().unwrap() as usize;
            let ratio = case["ratio"].as_u64().unwrap() as usize;
            let start_pos = case["start_pos"].as_u64().unwrap() as usize;
            let mut ape = vec![0.0_f32; ratio * outputs];
            for phase in 0..ratio {
                ape[phase * outputs..(phase + 1) * outputs].fill(phase as f32);
            }
            let weights = CompressorWeights {
                wkv_bf16: [f32_to_bf16_rne(1.0), f32_to_bf16_rne(2.0)].repeat(outputs),
                wgate_bf16: [f32_to_bf16_rne(2.0), f32_to_bf16_rne(-1.0)].repeat(outputs),
                ape_f32: ape,
                range_bytes: 0,
            };
            let input = [
                f32_to_bf16_rne(1.0),
                f32_to_bf16_rne(2.0),
                f32_to_bf16_rne(3.0),
                f32_to_bf16_rne(4.0),
            ];
            let prepared = prepare_compressor_cpu_with_geometry(
                &input, &weights, 2, outputs, ratio, start_pos,
            )
            .unwrap();
            for (actual, expected) in [
                (&prepared.kv_f32, &case["row_values"]["kv"]),
                (&prepared.score_f32, &case["row_values"]["score"]),
                (
                    &prepared.score_plus_ape_f32,
                    &case["row_values"]["score_plus_ape"],
                ),
            ] {
                let expected = expected
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|value| value.as_f64().unwrap() as f32)
                    .collect::<Vec<_>>();
                assert_eq!(actual.len(), 2 * outputs);
                assert_eq!(&actual[..outputs], vec![expected[0]; outputs]);
                assert_eq!(&actual[outputs..], vec![expected[1]; outputs]);
            }
        }
    }

    #[test]
    fn compressor_boundary_pooling_precedes_csa_overlap_copy_exactly() {
        let oracle: serde_json::Value = serde_json::from_str(MINI_SERIAL_JSON).unwrap();
        let contract = &oracle["compressor_boundary_contract"];
        let row = |left: f32, right: f32| CompressorPrepared {
            kv_f32: vec![left, right],
            score_f32: vec![0.0, 0.0],
            score_plus_ape_f32: vec![0.0, 0.0],
        };
        let mut state = V4CompressorState::new_decode(4, 1).unwrap();
        for position in 0..3 {
            let (next, pooled) = state
                .prepare_decode_row_and_pool(
                    &row(100.0 + position as f32, 1.0 + position as f32),
                    position,
                )
                .unwrap();
            assert!(pooled.is_none());
            state = next;
        }
        let (next, pooled) = state
            .prepare_decode_row_and_pool(&row(103.0, 4.0), 3)
            .unwrap();
        assert_eq!(pooled.unwrap()[0].to_bits(), 2.5_f32.to_bits());
        state = next;
        for position in 4..7 {
            state = state
                .prepare_decode_row(
                    &row(100.0 + position as f32, 1.0 + position as f32),
                    position,
                )
                .unwrap();
        }
        let (_, pooled) = state
            .prepare_decode_row_and_pool(&row(107.0, 8.0), 7)
            .unwrap();
        assert_eq!(
            pooled.unwrap()[0].to_bits(),
            contract["csa_overlap_pool_f32_bits"][0].as_i64().unwrap() as u32
        );

        let mut hca = V4CompressorState::new_decode(128, 1).unwrap();
        for position in 0..127 {
            hca = hca
                .prepare_decode_row(
                    &CompressorPrepared {
                        kv_f32: vec![position as f32 + 1.0],
                        score_f32: vec![0.0],
                        score_plus_ape_f32: vec![0.0],
                    },
                    position,
                )
                .unwrap();
        }
        let (_, pooled) = hca
            .prepare_decode_row_and_pool(
                &CompressorPrepared {
                    kv_f32: vec![128.0],
                    score_f32: vec![0.0],
                    score_plus_ape_f32: vec![0.0],
                },
                127,
            )
            .unwrap();
        assert_eq!(
            pooled.unwrap()[0].to_bits(),
            contract["hca_ratio128_pool_f32_bits"][0].as_i64().unwrap() as u32
        );
    }

    #[test]
    fn compressed_rope_generation_matches_independent_pytorch_fixture() {
        let fixture = fp8_projection_fixture().unwrap();
        for position in 0..IDS.len() {
            let actual = compressed_rope_freqs(position);
            let expected =
                &fixture.freqs_cis_f32_pairs[position * Q_ROPE_DIM..(position + 1) * Q_ROPE_DIM];
            for (actual, expected) in actual.iter().zip(expected) {
                assert!(
                    (actual - expected).abs() <= 2e-6,
                    "position {position}: expected {expected}, got {actual}"
                );
            }
        }
    }

    #[test]
    fn csa_boundary_commits_main_and_indexer_cache_in_one_transaction() {
        let mut serial = V4SerialPersistentState::new(4).unwrap();
        for position in 0..4 {
            serial
                .transaction(2, |state| {
                    state.append_kv_row(
                        position,
                        &vec![f32_to_bf16_rne(position as f32); Q_HEAD_DIM],
                    )?;
                    let main = state.commit_compressor_row_and_pool(
                        position,
                        &CompressorPrepared {
                            kv_f32: vec![position as f32 + 1.0; 1_024],
                            score_f32: vec![0.0; 1_024],
                            score_plus_ape_f32: vec![0.0; 1_024],
                        },
                    )?;
                    let index = state.commit_indexer_row_and_pool(
                        position,
                        &CompressorPrepared {
                            kv_f32: vec![position as f32 + 1.0; 256],
                            score_f32: vec![0.0; 256],
                            score_plus_ape_f32: vec![0.0; 256],
                        },
                    )?;
                    if let (Some(main), Some(index)) = (main, index) {
                        state.append_compressed_kv_row(
                            position / 4,
                            &main.into_iter().map(f32_to_bf16_rne).collect::<Vec<_>>(),
                        )?;
                        state.append_indexer_kv_row(
                            position / 4,
                            &index.into_iter().map(f32_to_bf16_rne).collect::<Vec<_>>(),
                        )?;
                    }
                    Ok(())
                })
                .unwrap();
        }
        let layer = serial.layer(2).unwrap();
        assert_eq!(layer.committed_kv_rows(), 4);
        assert_eq!(layer.compressed_kv_bf16().len(), Q_HEAD_DIM);
        assert_eq!(layer.indexer_kv_bf16().len(), 128);
        assert_eq!(
            layer.attention_cache_bf16().len(),
            (KV_WINDOW + 1) * Q_HEAD_DIM
        );
        let before = layer.compressed_kv_bf16().to_vec();
        assert!(
            serial
                .transaction(2, |state| {
                    state.append_compressed_kv_row(7, &vec![0; Q_HEAD_DIM])
                })
                .is_err()
        );
        assert_eq!(serial.layer(2).unwrap().compressed_kv_bf16(), before);
    }

    #[test]
    fn official_sparse_indices_cover_short_context_and_compression_boundaries() {
        let short = build_hca_sparse_topk(3, 0, 128, 128).unwrap();
        assert_eq!(short.window_per_row, 3);
        assert_eq!(short.compressed_per_row, 0);
        assert_eq!(
            short.indices,
            vec![0, -1, -1, 0, 1, -1, 0, 1, 2],
            "short context degenerates to causal local attention"
        );

        let prefill = build_hca_sparse_topk(256, 0, 128, 128).unwrap();
        assert_eq!(prefill.window_per_row, 128);
        assert_eq!(prefill.compressed_per_row, 2);
        assert_eq!(
            &prefill.indices[127 * 130 + 128..127 * 130 + 130],
            &[256, -1]
        );
        assert_eq!(
            &prefill.indices[255 * 130 + 128..255 * 130 + 130],
            &[256, 257]
        );

        let boundary = build_hca_sparse_topk(1, 127, 128, 128).unwrap();
        assert_eq!(boundary.indices[..128], (0..128).collect::<Vec<_>>());
        assert_eq!(boundary.indices[128], 128);
        let wrapped = build_hca_sparse_topk(1, 128, 128, 128).unwrap();
        assert_eq!(
            wrapped.indices[..128],
            (1..128).chain(0..1).collect::<Vec<_>>()
        );
        assert_eq!(wrapped.indices[128], 128);
        let k2_boundary = build_hca_sparse_topk(2, 126, 128, 128).unwrap();
        assert_eq!(k2_boundary.compressed_per_row, 1);
        assert_eq!(k2_boundary.indices[128], -1);
        assert_eq!(k2_boundary.indices[129 + 128], 128);
    }

    #[test]
    fn csa_learned_top1024_is_offset_after_the_official_local_window() {
        let scores = (0..1_030).map(|key| key as f32).collect::<Vec<_>>();
        let topk =
            build_csa_sparse_topk(&scores, 1, 1_030, &[1_030], 4_119, 128, INDEX_TOPK).unwrap();
        assert_eq!(topk.window_per_row, 128);
        assert_eq!(topk.compressed_per_row, 1_024);
        assert_eq!(topk.per_row, 1_152);
        assert_eq!(
            &topk.indices[..128],
            &(24..128).chain(0..24).collect::<Vec<_>>()
        );
        assert_eq!(topk.indices[128], 128 + 1_029);
        assert_eq!(topk.indices[128 + 1_023], 128 + 6);

        let causal = build_csa_sparse_topk(
            &[0.5, 0.5, 0.5, 0.5],
            4,
            1,
            &[0, 0, 0, 1],
            0,
            128,
            INDEX_TOPK,
        )
        .unwrap();
        assert_eq!(causal.window_per_row, 4);
        assert_eq!(causal.compressed_per_row, 1);
        assert_eq!(causal.indices[4], -1);
        assert_eq!(causal.indices[9], -1);
        assert_eq!(causal.indices[14], -1);
        assert_eq!(causal.indices[19], 4);
    }

    #[test]
    fn serial_persistent_state_is_lazy_independent_and_transactional() {
        let mut state = V4SerialPersistentState::new(2).unwrap();
        assert_eq!(state.max_context(), 2);
        assert_eq!(state.allocated_layers(), 0);

        let kv_row = vec![f32_to_bf16_rne(1.0); Q_HEAD_DIM];
        let hca_projection = projected_row(Q_HEAD_DIM, 3.0);
        state
            .transaction(0, |layer| {
                layer.append_kv_row(0, &kv_row)?;
                layer.commit_compressor_row(0, &hca_projection)
            })
            .unwrap();
        assert_eq!(state.allocated_layers(), 1);
        assert_eq!(state.layer(0).unwrap().committed_kv_rows(), 1);
        assert!(state.layer(0).unwrap().indexer().is_none());
        let before = state.layer(0).unwrap().clone();
        let failed = state.transaction(0, |layer| {
            layer.append_kv_row(1, &kv_row)?;
            Err::<(), Box<dyn Error>>("forced rollback".into())
        });
        assert!(failed.is_err());
        assert!(state.layer(0).unwrap().bits_equal(&before));

        let failed = state.transaction(2, |layer| {
            layer.append_kv_row(0, &kv_row)?;
            Err::<(), Box<dyn Error>>("forced first-touch rollback".into())
        });
        assert!(failed.is_err());
        assert!(state.layer(2).is_none());

        let csa_projection = projected_row(2 * Q_HEAD_DIM, 5.0);
        let indexer_projection = projected_row(256, 7.0);
        state
            .transaction(2, |layer| {
                layer.append_kv_row(0, &kv_row)?;
                layer.commit_compressor_row(0, &csa_projection)?;
                layer.commit_indexer_row(0, &indexer_projection)
            })
            .unwrap();
        assert_eq!(state.allocated_layers(), 2);
        assert_eq!(state.layer(0).unwrap().committed_kv_rows(), 1);
        assert_eq!(state.layer(2).unwrap().committed_kv_rows(), 1);
        assert!(state.layer(2).unwrap().indexer().is_some());
        assert_ne!(
            state.layer(0).unwrap().kv_bf16().as_ptr(),
            state.layer(2).unwrap().kv_bf16().as_ptr()
        );
    }

    #[test]
    fn transactional_serial_runner_commits_each_layer_and_reports_cpu_indexer_time() {
        let manifests = serial_layer_manifests(0).unwrap();
        let slice = &manifests[1..=2];
        let mut state = V4SerialPersistentState::new(2).unwrap();
        let (hidden, telemetry) = run_serial_layer_transactions(
            &mut state,
            vec![f32_to_bf16_rne(1.0)],
            slice,
            |manifest, input, layer_state| {
                layer_state.append_kv_row(0, &vec![f32_to_bf16_rne(2.0); Q_HEAD_DIM])?;
                let width = match manifest.compression_ratio {
                    CompressionRatio::Hca128 => Q_HEAD_DIM,
                    CompressionRatio::Csa4 => 2 * Q_HEAD_DIM,
                };
                layer_state.commit_compressor_row(0, &projected_row(width, 3.0))?;
                let indexer_ns = if manifest.compression_ratio == CompressionRatio::Csa4 {
                    layer_state.commit_indexer_row(0, &projected_row(256, 4.0))?;
                    9
                } else {
                    0
                };
                Ok(TransactionalLayerOutput {
                    next_hidden: vec![f32_to_bf16_rne(
                        bf16_to_f32(input[0]) + manifest.layer as f32 + 1.0,
                    )],
                    source_range_bytes: manifest.contract_range_bytes,
                    indexer_topk_cpu_ns: indexer_ns,
                })
            },
        )
        .unwrap();
        assert_ne!(hidden, vec![f32_to_bf16_rne(1.0)]);
        assert_eq!(telemetry.len(), 2);
        assert_eq!(telemetry[0].kv_rows_before, 0);
        assert_eq!(telemetry[0].kv_rows_after, 1);
        assert_eq!(telemetry[0].indexer_topk_cpu_ns, 0);
        assert_eq!(telemetry[1].indexer_topk_cpu_ns, 9);
        assert!(telemetry.iter().all(|item| item.layer_elapsed_ns > 0));
        assert_eq!(state.allocated_layers(), 2);

        let mut rollback = V4SerialPersistentState::new(1).unwrap();
        let failed = run_serial_layer_transactions(
            &mut rollback,
            vec![f32_to_bf16_rne(1.0)],
            &manifests[1..=1],
            |manifest, input, layer_state| {
                layer_state.append_kv_row(0, &vec![f32_to_bf16_rne(2.0); Q_HEAD_DIM])?;
                Ok(TransactionalLayerOutput {
                    next_hidden: input.to_vec(),
                    source_range_bytes: manifest.contract_range_bytes,
                    indexer_topk_cpu_ns: 0,
                })
            },
        );
        assert!(failed.is_err());
        assert_eq!(rollback.allocated_layers(), 0);
    }

    #[test]
    fn two_layer_serial_slice_commits_hca_output_into_csa_input() {
        let manifests = serial_layer_manifests(0).unwrap();
        let slice = &manifests[1..=2];
        let initial = vec![f32_to_bf16_rne(1.0), f32_to_bf16_rne(2.0)];
        let mut received = Vec::new();
        let (final_hidden, telemetry) =
            run_serial_layer_slice(initial.clone(), slice, |manifest, hidden| {
                received.push(hidden.to_vec());
                let increment = match manifest.compression_ratio {
                    CompressionRatio::Hca128 => 3.0,
                    CompressionRatio::Csa4 => {
                        let scores = indexer_scores_cpu(
                            &[1.0, -1.0],
                            &[2.0, 1.0, -1.0, 3.0],
                            &[0.5],
                            1,
                            1,
                            2,
                            2,
                        )?;
                        let selected = indexer_topk_cpu(&scores, 1, 2, 1, &[2], 0)?;
                        selected[0] as f32 + 4.0
                    }
                };
                Ok((
                    hidden
                        .iter()
                        .map(|&bits| f32_to_bf16_rne(bf16_to_f32(bits) + increment))
                        .collect(),
                    manifest.contract_range_bytes,
                    u64::from(manifest.compression_ratio == CompressionRatio::Csa4),
                ))
            })
            .unwrap();
        assert_eq!(received[0], initial);
        assert_eq!(
            received[1],
            vec![f32_to_bf16_rne(4.0), f32_to_bf16_rne(5.0)]
        );
        assert_eq!(
            final_hidden,
            vec![f32_to_bf16_rne(8.0), f32_to_bf16_rne(9.0)]
        );
        assert_eq!(telemetry.len(), 2);
        assert_eq!(telemetry[0].layer, 1);
        assert_eq!(telemetry[1].layer, 2);
        assert_eq!(telemetry[0].indexer_topk_cpu_ns, 0);
        assert_eq!(telemetry[1].indexer_topk_cpu_ns, 1);
    }

    #[derive(Clone)]
    struct TestSemanticState {
        kv: V4KvCache,
        mhc: [u32; HC],
        compressor: V4CompressorState,
        indexer: Option<V4CompressorState>,
    }

    fn semantic_bits_equal(left: &TestSemanticState, right: &TestSemanticState) -> bool {
        left.kv.data == right.kv.data
            && left.kv.committed == right.kv.committed
            && left.mhc == right.mhc
            && left.compressor.bits_equal(&right.compressor)
            && match (&left.indexer, &right.indexer) {
                (Some(left), Some(right)) => left.bits_equal(right),
                (None, None) => true,
                _ => false,
            }
    }

    fn projected_row(width: usize, marker: f32) -> CompressorPrepared {
        CompressorPrepared {
            kv_f32: vec![marker; width],
            score_f32: vec![marker + 8.0; width],
            score_plus_ape_f32: vec![marker + 16.0; width],
        }
    }

    fn apply_test_decode_row(
        state: &mut TestSemanticState,
        ratio: usize,
        start_pos: usize,
        marker: u16,
    ) -> Result<(), Box<dyn Error>> {
        let row = vec![f32_to_bf16_rne(marker as f32); Q_HEAD_DIM];
        state.kv = state.kv.prepare_decode_row(&row, start_pos)?;
        if start_pos
            .checked_add(1)
            .ok_or("test position overflow")?
            .is_multiple_of(ratio)
        {
            let compressed = vec![f32_to_bf16_rne((marker + 32) as f32); Q_HEAD_DIM];
            state.kv = state
                .kv
                .prepare_compressed_row(&compressed, start_pos / ratio)?;
        }
        state.mhc[0] = state.mhc[0].wrapping_mul(17).wrapping_add(marker as u32);
        let width = (1 + usize::from(ratio == 4)) * state.compressor.head_dim;
        state.compressor = state
            .compressor
            .prepare_decode_row(&projected_row(width, marker as f32), start_pos)?;
        if let Some(indexer) = &state.indexer {
            let width = 2 * indexer.head_dim;
            state.indexer = Some(
                indexer
                    .prepare_decode_row(&projected_row(width, marker as f32 + 64.0), start_pos)?,
            );
        }
        Ok(())
    }

    fn exercise_k2_boundary(ratio: usize, start_pos: usize) {
        let base = TestSemanticState {
            kv: V4KvCache::new(),
            mhc: [0; HC],
            compressor: V4CompressorState::new_decode(ratio, 2).unwrap(),
            indexer: (ratio == 4).then(|| V4CompressorState::new_decode(4, 1).unwrap()),
        };
        let mut sequential_row0 = base.clone();
        apply_test_decode_row(&mut sequential_row0, ratio, start_pos, 3).unwrap();
        let mut sequential_row1 = sequential_row0.clone();
        apply_test_decode_row(&mut sequential_row1, ratio, start_pos + 1, 5).unwrap();

        let rejected = K2Prepared::prepare(
            &base,
            |state| apply_test_decode_row(state, ratio, start_pos, 3),
            |state| apply_test_decode_row(state, ratio, start_pos + 1, 5),
        )
        .unwrap()
        .validate(
            [&sequential_row0, &sequential_row1],
            [7, 11],
            [7, 13],
            semantic_bits_equal,
        )
        .unwrap();
        assert_eq!(rejected.accepted_rows(), 1);

        let mut r0 = base.clone();
        rejected
            .commit_prefix(&mut r0, 0, semantic_bits_equal)
            .unwrap();
        assert!(semantic_bits_equal(&r0, &base));

        let mut r1 = base.clone();
        rejected
            .commit_prefix(&mut r1, 1, semantic_bits_equal)
            .unwrap();
        assert!(semantic_bits_equal(&r1, &sequential_row0));
        let before_rejected_suffix = r1.clone();
        assert_eq!(
            rejected.commit_prefix(&mut r1, 2, semantic_bits_equal),
            Err(K2ContractError::RejectedSuffix)
        );
        assert!(
            semantic_bits_equal(&r1, &before_rejected_suffix),
            "rejected row 1 partially mutated KV/mHC/compressor/indexer/FP32 state"
        );

        let accepted = K2Prepared::prepare(
            &base,
            |state| apply_test_decode_row(state, ratio, start_pos, 3),
            |state| apply_test_decode_row(state, ratio, start_pos + 1, 5),
        )
        .unwrap()
        .validate(
            [&sequential_row0, &sequential_row1],
            [7, 13],
            [7, 13],
            semantic_bits_equal,
        )
        .unwrap();
        assert_eq!(accepted.accepted_rows(), 2);
        let mut r2 = base.clone();
        accepted
            .commit_prefix(&mut r2, 2, semantic_bits_equal)
            .unwrap();
        assert!(semantic_bits_equal(&r2, &sequential_row1));

        let mut dirty = base.clone();
        dirty.mhc[1] = 1;
        let dirty_before = dirty.clone();
        assert_eq!(
            accepted.commit_prefix(&mut dirty, 1, semantic_bits_equal),
            Err(K2ContractError::DirtyAuthoritativeState)
        );
        assert!(semantic_bits_equal(&dirty, &dirty_before));
    }

    #[test]
    fn rejects_bad_ids() {
        assert!(load(Path::new("."), &[129_280]).is_err());
    }

    #[test]
    fn k2_prefix_commit_is_exact_at_ratio4_pos3_and_ratio128_pos127() {
        exercise_k2_boundary(4, 3);
        exercise_k2_boundary(128, 127);
    }

    #[test]
    fn k2_safe_minimax_renewal_gate_uses_wilson_worst_case() {
        let pass = evaluate_k2_renewal_gate(K2GateInput {
            draft_matches: 80,
            trials: 100,
            t1_lower: 100.0,
            t2_upper: 150.0,
            b1_lower: 100.0,
            b2_upper: 150.0,
        });
        assert!(pass.admit);
        assert_eq!(pass.reason, "K2_ROBUST_WORST_CASE_PASS");
        assert!(pass.q_lower.is_some_and(|q| q > 0.70 && q < 0.72));
        assert!(pass.committed_rows_lower.unwrap() > pass.time_ratio_upper.unwrap());
        assert!(pass.bytes_per_row_upper.unwrap() <= 90.0);

        let time_fail = evaluate_k2_renewal_gate(K2GateInput {
            t2_upper: 180.0,
            ..K2GateInput {
                draft_matches: 80,
                trials: 100,
                t1_lower: 100.0,
                t2_upper: 150.0,
                b1_lower: 100.0,
                b2_upper: 150.0,
            }
        });
        assert!(!time_fail.admit);
        assert_eq!(time_fail.reason, "AR_ROBUST_WORST_CASE_FAIL");

        let bytes_fail = evaluate_k2_renewal_gate(K2GateInput {
            b2_upper: 160.0,
            ..K2GateInput {
                draft_matches: 80,
                trials: 100,
                t1_lower: 100.0,
                t2_upper: 150.0,
                b1_lower: 100.0,
                b2_upper: 150.0,
            }
        });
        assert!(!bytes_fail.admit);
        assert_eq!(bytes_fail.reason, "AR_ROBUST_WORST_CASE_FAIL");

        for invalid in [
            K2GateInput {
                draft_matches: 99,
                trials: K2_MIN_GATE_SAMPLES - 1,
                t1_lower: 100.0,
                t2_upper: 150.0,
                b1_lower: 100.0,
                b2_upper: 150.0,
            },
            K2GateInput {
                draft_matches: 101,
                trials: 100,
                t1_lower: 100.0,
                t2_upper: 150.0,
                b1_lower: 100.0,
                b2_upper: 150.0,
            },
            K2GateInput {
                draft_matches: 80,
                trials: 100,
                t1_lower: f64::NAN,
                t2_upper: 150.0,
                b1_lower: 100.0,
                b2_upper: 150.0,
            },
        ] {
            let report = evaluate_k2_renewal_gate(invalid);
            assert!(!report.admit);
            assert_eq!(report.reason, "AR_INVALID_OR_INSUFFICIENT");
            assert!(report.q_lower.is_none());
        }
    }

    #[test]
    fn pytorch_cascade_sum_matches_pinned_collapsed_rows() {
        let fixture = fixture_stages().unwrap();
        let expected = [0x44a4_0042_u32, 0x463f_b04f, 0x45d9_f64f, 0x3e82_0606];
        for (row, (&expected, values)) in expected
            .iter()
            .zip(fixture.collapsed.chunks_exact(HIDDEN))
            .enumerate()
        {
            let squares = values
                .iter()
                .map(|&bits| {
                    let value = bf16_to_f32(bits);
                    value * value
                })
                .collect::<Vec<_>>();
            assert_eq!(
                crate::v4_router::pytorch_contiguous_sum_f32(&squares).to_bits(),
                expected,
                "pinned PyTorch cascade sum row {row}"
            );
        }
    }

    #[test]
    fn pytorch_sleef_expf_matches_pinned_avx_vectors() {
        for (input, expected) in [
            (0x0000_0000_u32, 0x3f80_0000_u32),
            (0xbf80_0000, 0x3ebc_5ab2),
            (0x3f80_0000, 0x402d_f854),
            (0xc2d0_0000, 0x0000_0000),
            (0x42c8_0000, 0x7f80_0000),
            (0xc1b8_9a1f, 0x2ed1_4d68),
            (0xc015_93f8, 0x3dc5_d73f),
            (0xbe8a_9064, 0x3f43_4d30),
        ] {
            assert_eq!(
                pytorch_sleef_expf(f32::from_bits(input)).to_bits(),
                expected,
                "SLEEF exp input {input:08x}"
            );
        }
    }

    #[test]
    fn decodes_fp8_special_values_and_rejects_non_finite_activation() {
        assert_eq!(E4M3Fn(0x00).decode(), 0.0);
        assert!(E4M3Fn(0x80).decode().is_sign_negative());
        assert_eq!(E4M3Fn(0x01).decode(), 0.001953125);
        assert_eq!(E4M3Fn(0x81).decode(), -0.001953125);
        assert_eq!(E4M3Fn(0x08).decode(), 0.015625);
        assert_eq!(E4M3Fn(0x7e).decode(), 448.0);
        assert_eq!(E4M3Fn(0xfe).decode(), -448.0);
        assert!(E4M3Fn(0x7f).decode().is_nan());
        assert!(E4M3Fn(0xff).decode().is_nan());
        assert!(
            (0_u8..=u8::MAX)
                .map(|bits| E4M3Fn(bits).decode())
                .all(|value| !value.is_infinite())
        );
        assert_eq!(E4M3Fn::encode(-0.0).unwrap(), E4M3Fn(0x80));
        assert!(E4M3Fn::encode(f32::INFINITY).is_err());

        assert_eq!(E8M0Fnu(0).decode(), f32::from_bits(1 << 22));
        assert_eq!(E8M0Fnu(1).decode(), f32::MIN_POSITIVE);
        assert_eq!(E8M0Fnu(126).decode(), 0.5);
        assert_eq!(E8M0Fnu(127).decode(), 1.0);
        assert_eq!(E8M0Fnu(128).decode(), 2.0);
        assert_eq!(E8M0Fnu(254).decode(), f32::from_bits(254 << 23));
        assert!(E8M0Fnu(255).decode().is_nan());

        let mut bad = vec![0_u16; HIDDEN];
        bad[7] = f32_to_bf16_rne(f32::NAN);
        assert!(quantize_fp8_activation(&bad, 1).is_err());
        let q_rank = quantize_fp8_activation(&vec![0_u16; 1_536], 1).unwrap();
        assert_eq!(q_rank.in_features, 1_536);
        assert_eq!(q_rank.scales.len(), 12);
    }

    #[test]
    fn kv_finalize_and_single_cache_commit_match_fixture() {
        let fixture = fp8_projection_fixture().unwrap();
        let freqs = query_freqs_cis_pos0_3();
        let kv = finalize_kv_cpu(&fixture.kv_norm_bf16, &freqs).unwrap();
        assert_eq!(kv.rope_bf16, fixture.kv_rope_bf16, "KV RoPE fixture");
        assert_eq!(
            kv.scale_e8m0, fixture.kv_scale_e8m0,
            "KV E8M0 scale fixture"
        );
        assert_eq!(kv.ready_bf16, fixture.kv_ready_bf16, "KV ready fixture");
        assert_eq!(
            kv.causal_topk_i32, fixture.causal_topk_i32,
            "causal top-k fixture"
        );

        let mut cache = V4KvCache::new();
        assert_eq!(
            cache.as_slice(),
            fixture.cache_initial_bf16,
            "initial cache fixture"
        );
        cache.commit(&kv.ready_bf16, 1, 0, IDS.len()).unwrap();
        assert_eq!(
            cache.as_slice(),
            fixture.cache_after_bf16,
            "committed cache fixture"
        );
        let committed = cache.as_slice().to_vec();
        cache.commit(&kv.ready_bf16, 1, 0, IDS.len()).unwrap();
        assert_eq!(
            cache.as_slice(),
            committed,
            "exact repeat must be idempotent"
        );
        assert!(
            cache.as_slice()[IDS.len() * Q_HEAD_DIM..]
                .iter()
                .all(|&bits| bits == 0),
            "untouched cache tail"
        );
        assert!(
            cache.as_slice()[KV_WINDOW * Q_HEAD_DIM..]
                .iter()
                .all(|&bits| bits == 0),
            "compressed cache view"
        );

        let mut different = kv.ready_bf16.clone();
        different[0] ^= 1;
        assert!(cache.commit(&different, 1, 0, IDS.len()).is_err());
        assert_eq!(
            cache.as_slice(),
            committed,
            "different repeat must not mutate"
        );
    }

    #[test]
    fn kv_cache_preflight_rejects_invalid_geometry_without_mutation() {
        let mut cache = V4KvCache::new();
        cache.data.fill(0x5a5a);
        let sentinel = cache.as_slice().to_vec();
        let payload = vec![0_u16; IDS.len() * Q_HEAD_DIM];
        for invalid in [
            cache.commit(&payload, 2, 0, IDS.len()),
            cache.commit(&payload, 1, 1, IDS.len()),
            cache.commit(&payload, 1, KV_WINDOW - 1, IDS.len()),
            cache.commit(&payload, 1, 0, IDS.len() - 1),
            cache.commit(&payload[..payload.len() - 1], 1, 0, IDS.len()),
        ] {
            assert!(invalid.is_err());
            assert_eq!(cache.as_slice(), sentinel, "invalid commit mutated cache");
        }
        assert!(
            preflight_kv_cache_commit(
                KV_CACHE_POSITIONS * Q_HEAD_DIM,
                IDS.len() * Q_HEAD_DIM,
                usize::MAX,
                0,
                IDS.len(),
            )
            .is_err(),
            "overflow must fail closed"
        );
        assert_eq!(
            cache.as_slice(),
            sentinel,
            "overflow preflight mutated cache"
        );
    }

    #[test]
    fn compressor_paired_commit_is_idempotent_and_fails_closed() {
        let prepared = CompressorPrepared {
            kv_f32: vec![1.0; IDS.len() * Q_HEAD_DIM],
            score_f32: vec![2.0; IDS.len() * Q_HEAD_DIM],
            score_plus_ape_f32: vec![3.0; IDS.len() * Q_HEAD_DIM],
        };
        let mut state = V4CompressorState::new();
        let initial_kv = state.kv().to_vec();
        let initial_score = state.score().to_vec();
        for invalid in [
            state.commit(&prepared, 2, 0, IDS.len(), COMPRESS_RATIO, false),
            state.commit(&prepared, 1, 1, IDS.len(), COMPRESS_RATIO, false),
            state.commit(&prepared, 1, 0, IDS.len() - 1, COMPRESS_RATIO, false),
            state.commit(&prepared, 1, 0, IDS.len(), COMPRESS_RATIO - 1, false),
            state.commit(&prepared, 1, 0, IDS.len(), COMPRESS_RATIO, true),
        ] {
            assert!(invalid.is_err());
            assert!(f32_bits_equal(state.kv(), &initial_kv));
            assert!(f32_bits_equal(state.score(), &initial_score));
        }
        assert!(
            preflight_compressor_commit(
                COMPRESS_STATE_ELEMENTS,
                COMPRESS_STATE_ELEMENTS,
                IDS.len() * Q_HEAD_DIM,
                IDS.len() * Q_HEAD_DIM,
                usize::MAX,
                0,
                IDS.len(),
                COMPRESS_RATIO,
                false,
            )
            .is_err()
        );

        let mut non_finite = CompressorPrepared {
            kv_f32: prepared.kv_f32.clone(),
            score_f32: prepared.score_f32.clone(),
            score_plus_ape_f32: prepared.score_plus_ape_f32.clone(),
        };
        non_finite.kv_f32[0] = f32::NAN;
        assert!(
            state
                .commit(&non_finite, 1, 0, IDS.len(), COMPRESS_RATIO, false)
                .is_err()
        );
        assert!(f32_bits_equal(state.kv(), &initial_kv));
        assert!(f32_bits_equal(state.score(), &initial_score));

        state
            .commit(&prepared, 1, 0, IDS.len(), COMPRESS_RATIO, false)
            .unwrap();
        let committed_kv = state.kv().to_vec();
        let committed_score = state.score().to_vec();
        state
            .commit(&prepared, 1, 0, IDS.len(), COMPRESS_RATIO, false)
            .unwrap();
        assert!(f32_bits_equal(state.kv(), &committed_kv));
        assert!(f32_bits_equal(state.score(), &committed_score));
        let mut conflict = CompressorPrepared {
            kv_f32: prepared.kv_f32.clone(),
            score_f32: prepared.score_f32.clone(),
            score_plus_ape_f32: prepared.score_plus_ape_f32.clone(),
        };
        conflict.score_plus_ape_f32[0] = 4.0;
        assert!(
            state
                .commit(&conflict, 1, 0, IDS.len(), COMPRESS_RATIO, false)
                .is_err()
        );
        assert!(f32_bits_equal(state.kv(), &committed_kv));
        assert!(f32_bits_equal(state.score(), &committed_score));

        let mut dirty = V4CompressorState::new();
        dirty.kv[COMPRESS_STATE_ELEMENTS - 1] = 1.0;
        let dirty_kv = dirty.kv().to_vec();
        let dirty_score = dirty.score().to_vec();
        assert!(
            dirty
                .commit(&prepared, 1, 0, IDS.len(), COMPRESS_RATIO, false)
                .is_err()
        );
        assert!(f32_bits_equal(dirty.kv(), &dirty_kv));
        assert!(f32_bits_equal(dirty.score(), &dirty_score));
    }

    #[test]
    #[ignore = "requires COLIBRI_DSV4_MODEL_DIR pinned snapshot"]
    fn real_compressor_start0_state_commit_matches_pytorch_fixture() {
        let root = std::env::var_os("COLIBRI_DSV4_MODEL_DIR")
            .expect("set COLIBRI_DSV4_MODEL_DIR to the pinned snapshot");
        let fixture = fp8_projection_fixture().unwrap();
        let weights = load_compressor_weights(Path::new(&root)).unwrap();
        assert_eq!(weights.range_bytes, 14_942_208);
        let prepared = prepare_compressor_cpu(&fixture.input_bf16, &weights).unwrap();
        report_f32(
            "compressor kv CPU",
            &prepared.kv_f32,
            &fixture.compressor_kv_f32,
        );
        report_f32(
            "compressor score CPU",
            &prepared.score_f32,
            &fixture.compressor_score_f32,
        );
        report_f32(
            "compressor score+ape CPU",
            &prepared.score_plus_ape_f32,
            &fixture.compressor_score_plus_ape_f32,
        );
        assert!(f32_bits_equal(&prepared.kv_f32, &fixture.compressor_kv_f32));
        assert!(f32_bits_equal(
            &prepared.score_f32,
            &fixture.compressor_score_f32
        ));
        assert!(f32_bits_equal(
            &prepared.score_plus_ape_f32,
            &fixture.compressor_score_plus_ape_f32
        ));

        let cache_before = fixture.cache_after_bf16.clone();
        let mut state = V4CompressorState::new();
        assert!(f32_bits_equal(
            state.kv(),
            &fixture.compressor_kv_state_initial_f32
        ));
        assert!(f32_bits_equal(
            state.score(),
            &fixture.compressor_score_state_initial_f32
        ));
        state
            .commit(&prepared, 1, 0, IDS.len(), COMPRESS_RATIO, false)
            .unwrap();
        assert!(f32_bits_equal(
            state.kv(),
            &fixture.compressor_kv_state_after_f32
        ));
        assert!(f32_bits_equal(
            state.score(),
            &fixture.compressor_score_state_after_f32
        ));
        assert_eq!(fixture.cache_after_bf16, cache_before);
        assert!(
            fixture.cache_after_bf16[KV_WINDOW * Q_HEAD_DIM..]
                .iter()
                .all(|&bits| bits == 0)
        );
    }

    #[test]
    #[ignore = "requires COLIBRI_DSV4_MODEL_DIR pinned snapshot"]
    fn real_fp8_projections_match_pytorch_fixture() {
        let meta: serde_json::Value = serde_json::from_str(FP8_JSON).unwrap();
        assert_eq!(
            meta["schema"],
            "colibri-v4-fp8-projection-pytorch-oracle-v4"
        );
        assert_eq!(meta["model_revision"], REVISION);
        assert_eq!(meta["ids"], serde_json::json!(IDS));
        assert_eq!(
            meta["binary"]["sha256"],
            "b70ad7a4cb47c290eaa4f810566078c65cc0fb99bde06c27ac6f304500a790ee"
        );
        assert_eq!(
            meta["cache_contract"]["capacity_1_untouched_tail_sha256"],
            "30d4da31db85877f0102e0d56b9559e9d4400fde83c253a19c7c1988b4ed0c64"
        );
        assert_eq!(
            meta["cache_contract"]["capacity_1_compressed_view_sha256"],
            "c35020473aed1b4642cd726cad727b63fff2824ad68cedd7ffb73c7cbd890479"
        );
        assert_eq!(
            meta["cache_contract"]["capacity_4_initial_sha256"],
            "ff6335069b6e140eb47149d847aea80bf7e2b06bd80ae9708aa382efb3ae21ee"
        );
        assert_eq!(
            meta["cache_contract"]["capacity_4_after_sha256"],
            "525ba206025f57ccb2459086e2abd6d571551d61c37be9ca45067f784717b45c"
        );
        for (name, offset, bytes, dtype, shape) in [
            (
                "input_bf16",
                8,
                57_344,
                "bf16-le",
                serde_json::json!([4, HIDDEN]),
            ),
            (
                "activation_fp8",
                57_352,
                28_672,
                "e4m3fn",
                serde_json::json!([4, HIDDEN]),
            ),
            (
                "activation_scale",
                86_024,
                224,
                "e8m0fnu",
                serde_json::json!([4, 56]),
            ),
            (
                "wq_f32",
                86_248,
                24_576,
                "f32-le",
                serde_json::json!([4, 1_536]),
            ),
            (
                "wq_bf16",
                110_824,
                12_288,
                "bf16-le",
                serde_json::json!([4, 1_536]),
            ),
            (
                "q_norm_bf16",
                123_112,
                12_288,
                "bf16-le",
                serde_json::json!([4, 1_536]),
            ),
            (
                "wkv_f32",
                135_400,
                8_192,
                "f32-le",
                serde_json::json!([4, 512]),
            ),
            (
                "wkv_bf16",
                143_592,
                4_096,
                "bf16-le",
                serde_json::json!([4, 512]),
            ),
            (
                "kv_norm_bf16",
                147_688,
                4_096,
                "bf16-le",
                serde_json::json!([4, 512]),
            ),
            (
                "wq_b_activation",
                151_784,
                6_144,
                "e4m3fn",
                serde_json::json!([4, 1_536]),
            ),
            (
                "wq_b_activation_scale",
                157_928,
                48,
                "e8m0fnu",
                serde_json::json!([4, 12]),
            ),
            (
                "wq_b_f32",
                157_976,
                1_048_576,
                "f32-le",
                serde_json::json!([4, 65_536]),
            ),
            (
                "wq_b_bf16",
                1_206_552,
                524_288,
                "bf16-le",
                serde_json::json!([4, 65_536]),
            ),
            (
                "q_head_rms_bf16",
                1_730_840,
                524_288,
                "bf16-le",
                serde_json::json!([1, 4, 128, 512]),
            ),
            (
                "freqs_cis_f32_pairs",
                2_255_128,
                1_024,
                "f32-le",
                serde_json::json!([4, 32, 2]),
            ),
            (
                "q_rope_bf16",
                2_256_152,
                524_288,
                "bf16-le",
                serde_json::json!([1, 4, 128, 512]),
            ),
            (
                "kv_rope_bf16",
                2_780_440,
                4_096,
                "bf16-le",
                serde_json::json!([1, 4, 512]),
            ),
            (
                "kv_scale_e8m0",
                2_784_536,
                28,
                "e8m0fnu",
                serde_json::json!([1, 4, 7]),
            ),
            (
                "kv_ready_bf16",
                2_784_564,
                4_096,
                "bf16-le",
                serde_json::json!([1, 4, 512]),
            ),
            (
                "causal_topk_i32",
                2_788_660,
                64,
                "i32-le",
                serde_json::json!([1, 4, 4]),
            ),
            (
                "cache_initial_bf16",
                2_788_724,
                163_840,
                "bf16-le",
                serde_json::json!([1, 160, 512]),
            ),
            (
                "cache_after_bf16",
                2_952_564,
                163_840,
                "bf16-le",
                serde_json::json!([1, 160, 512]),
            ),
            (
                "compressor_kv_f32",
                3_116_404,
                8_192,
                "f32-le",
                serde_json::json!([1, 4, 512]),
            ),
            (
                "compressor_score_f32",
                3_124_596,
                8_192,
                "f32-le",
                serde_json::json!([1, 4, 512]),
            ),
            (
                "compressor_score_plus_ape_f32",
                3_132_788,
                8_192,
                "f32-le",
                serde_json::json!([1, 4, 512]),
            ),
            (
                "compressor_kv_state_initial_f32",
                3_140_980,
                262_144,
                "f32-le",
                serde_json::json!([1, 128, 512]),
            ),
            (
                "compressor_score_state_initial_f32",
                3_403_124,
                262_144,
                "f32-le",
                serde_json::json!([1, 128, 512]),
            ),
            (
                "compressor_kv_state_after_f32",
                3_665_268,
                262_144,
                "f32-le",
                serde_json::json!([1, 128, 512]),
            ),
            (
                "compressor_score_state_after_f32",
                3_927_412,
                262_144,
                "f32-le",
                serde_json::json!([1, 128, 512]),
            ),
        ] {
            assert_eq!(meta["binary"][name]["offset"], offset);
            assert_eq!(meta["binary"][name]["bytes"], bytes);
            assert_eq!(meta["binary"][name]["dtype"], dtype);
            assert_eq!(meta["binary"][name]["shape"], shape);
        }
        assert_eq!(
            meta["binary"]["compressor_score_state_after_f32"]["offset"]
                .as_u64()
                .unwrap()
                + meta["binary"]["compressor_score_state_after_f32"]["bytes"]
                    .as_u64()
                    .unwrap(),
            FP8_BIN.len() as u64
        );
        for (name, hash) in [
            (
                "input_bf16",
                "7627d8b18a5eb98d14a8985094e01cd178fbad8bbc3103cfef8e7dd38bd8185f",
            ),
            (
                "activation_fp8",
                "c7dcb5b1f75fc67053a6659ee8e6df59d7347e9f26198d8aba6b996c97727358",
            ),
            (
                "activation_scale",
                "f77fc18aea43e17694cdacdfe676ce7f18b68ce15f31cdd751201d0fc543a09a",
            ),
            (
                "wq_f32",
                "e1d3c01a491346ddb36c93f83b939d4b8036fc212de70c238b47f435048be2a9",
            ),
            (
                "wq_bf16",
                "b076e4cd4ed9b711d73693d62595804630c212f4d824beb146860d97c43df024",
            ),
            (
                "q_norm_bf16",
                "abe4e938487183728742c2adb5370839fbf63309397ecf8ae8f6f436b19a2397",
            ),
            (
                "wkv_f32",
                "dd89a3a25e6c3705d4e45c268b7557e42de3ff0a3ba3abfe0453a4f5dad4756e",
            ),
            (
                "wkv_bf16",
                "bf8b3fedbb3b23fece8f9cef75ba3faedfb6e84acd51e101f548f3a9dc2c2da7",
            ),
            (
                "kv_norm_bf16",
                "8e3eb47d233e2be25fb53b8453ca969c0400939444dc8271c5bfb85ad9d28b3b",
            ),
            (
                "wq_b_activation",
                "f1cd00c6cbf276f79383b367409c1bab002cee33c0b04dc5080d346f3bb9a0c7",
            ),
            (
                "wq_b_activation_scale",
                "6dab1934d3c41a145599c3de00c609bdc5da40ad44c0834bb677290dac346c72",
            ),
            (
                "wq_b_f32",
                "b1f75f61eb5a679528dc37a64bf45eba9eac09d06394da4e6dd313dad7792716",
            ),
            (
                "wq_b_bf16",
                "53af957538b5a9cb9f52eac2466b8bc236b270b27e94768bad1ecb56a511e179",
            ),
            (
                "q_head_rms_bf16",
                "eff8c5282170c26212abc8199df18f61a5513375590a73e64c6da0a33173da2a",
            ),
            (
                "freqs_cis_f32_pairs",
                "d8d903ccf3dff66933fe32d3dc01f48088011b7697124229bd54cde47e0a05c1",
            ),
            (
                "q_rope_bf16",
                "185eae6ca4339d4d66d532f7e88647b31a793d7db6f6d5e444465f87fb4e2621",
            ),
            (
                "kv_rope_bf16",
                "57e405931f753518bcb4292931a1e77f725b69b0b833071b95467c3240d278e8",
            ),
            (
                "kv_scale_e8m0",
                "884a47bc34eb98a71f783a70457f2126cec940333470386395edd90383a55579",
            ),
            (
                "kv_ready_bf16",
                "20dfb59da6da5b9a29463c3ce731de48f1c8ff4dca7df3105b11e9760df0b5a1",
            ),
            (
                "causal_topk_i32",
                "ebcca8a99005905d352d187e38023107b7bfc6611dd69499ad2b88288c594f00",
            ),
            (
                "cache_initial_bf16",
                "6cdd259c8ecbe61fbc369f3293c1961541386954a223b17a37899d7fd9ad42da",
            ),
            (
                "cache_after_bf16",
                "ebe6644642b05f6d23efa9f573d5483c498c7aba210c0a41f0611aca073edba6",
            ),
            (
                "compressor_kv_f32",
                "e3c9aac77d93f70cb1b0dc2242cfa8a67fe712d67142bc669eac7c0a224d9f3f",
            ),
            (
                "compressor_score_f32",
                "824861a6cd77a8eaf3997d2f4da5d00396507cfdff6ea97d390f08ff059fbbe7",
            ),
            (
                "compressor_score_plus_ape_f32",
                "941e5c6962754b0863ebe8ccb5ff97e69f41d03dd220836de1671ec0eb5df34a",
            ),
            (
                "compressor_kv_state_initial_f32",
                "8a39d2abd3999ab73c34db2476849cddf303ce389b35826850f9a700589b4a90",
            ),
            (
                "compressor_score_state_initial_f32",
                "6470bc26e7cc29bf2cc0672d57eb7062150933581c52927d2a4f7be0f5ed0778",
            ),
            (
                "compressor_kv_state_after_f32",
                "e95703723cea6e25b27c510cd92d652491a11d0b0ede02330ef0566aa00073dd",
            ),
            (
                "compressor_score_state_after_f32",
                "e056758653a459bc66f6d2da2a764b56aa8f06da528ed30c8cae3b48c21249a4",
            ),
        ] {
            assert_eq!(meta["hashes"][name], hash);
        }
        assert_eq!(FP8_BIN.get(..8), Some(b"CV4P8001".as_slice()));
        assert_eq!(FP8_BIN.len(), 4_189_556);
        assert_eq!(
            meta["source"]["compressor"]["wkv_sha256"],
            "caa6aca271c241ba79879ab6a310bd962003909fcd3399e5ebe56da7b3f7393c"
        );
        assert_eq!(
            meta["source"]["compressor"]["wgate_sha256"],
            "827ebd72c1cdf7b81ec6ba76725fe46e7b01f383466067e92111054e07f4c9f6"
        );
        assert_eq!(
            meta["source"]["compressor"]["ape_sha256"],
            "9b28a13ba864996147d41ec3f14f684ab18574ad0e6f7781a7a125b1e9a91af1"
        );
        assert_eq!(
            meta["compressor_contract"]["kv_state_tail_sha256"],
            "5132a24b01320fb085912c4da9e4c7cebe6c0f2a5dd43f277b4138e8d6d4e9b3"
        );
        assert_eq!(
            meta["compressor_contract"]["score_state_tail_sha256"],
            "0fe4793aa3e282837a93e5d34698915b3ab667ccbabd7fc21cfb01dda8252c37"
        );
        assert_eq!(
            meta["compressor_contract"]["written_byte_range"],
            serde_json::json!([0, 8192])
        );

        let root = std::env::var_os("COLIBRI_DSV4_MODEL_DIR")
            .expect("set COLIBRI_DSV4_MODEL_DIR to the pinned snapshot");
        let fixture = fp8_projection_fixture().unwrap();
        let projections = load_fp8_projections(Path::new(&root)).unwrap();
        assert_eq!(projections.range_bytes, 115_354_496);
        let activation = quantize_fp8_activation(&fixture.input_bf16, IDS.len()).unwrap();
        assert_eq!(activation.values, fixture.activation, "activation FP8 bits");
        assert_eq!(
            activation.scales, fixture.activation_scale,
            "activation E8M0 bits"
        );
        let wq = project_fp8_cpu(&projections.wq_a, &activation).unwrap();
        let wkv = project_fp8_cpu(&projections.wkv, &activation).unwrap();
        let wq_b_activation = quantize_fp8_activation(&fixture.q_norm_bf16, IDS.len()).unwrap();
        assert_eq!(
            wq_b_activation.values, fixture.wq_b_activation,
            "wq_b activation FP8 bits"
        );
        assert_eq!(
            wq_b_activation.scales, fixture.wq_b_activation_scale,
            "wq_b activation E8M0 bits"
        );
        let wq_b = project_fp8_linear_cpu(&projections.wq_b, &wq_b_activation).unwrap();
        let q_head_rms = query_head_rms_bf16(&wq_b.bf16).unwrap();
        let freqs = query_freqs_cis_pos0_3();
        let q_rope = query_rope_bf16(&q_head_rms, &freqs).unwrap();
        let kv = finalize_kv_cpu(&fixture.kv_norm_bf16, &freqs).unwrap();
        report_f32("wq_a raw", &wq.raw_f32, &fixture.wq_f32);
        report_bf16("wq_a BF16", &wq.bf16, &fixture.wq_bf16);
        report_bf16("q_norm BF16", &wq.norm_bf16, &fixture.q_norm_bf16);
        report_f32("wkv raw", &wkv.raw_f32, &fixture.wkv_f32);
        report_bf16("wkv BF16", &wkv.bf16, &fixture.wkv_bf16);
        report_bf16("kv_norm BF16", &wkv.norm_bf16, &fixture.kv_norm_bf16);
        report_f32("wq_b raw", &wq_b.raw_f32, &fixture.wq_b_f32);
        report_bf16("wq_b BF16", &wq_b.bf16, &fixture.wq_b_bf16);
        report_bf16("query head RMS BF16", &q_head_rms, &fixture.q_head_rms_bf16);
        report_f32_bounded(
            "query frequencies",
            &freqs,
            &fixture.freqs_cis_f32_pairs,
            1e-7,
        );
        report_bf16("query RoPE BF16", &q_rope, &fixture.q_rope_bf16);
        report_bf16("KV RoPE BF16", &kv.rope_bf16, &fixture.kv_rope_bf16);
        report_bf16("KV ready BF16", &kv.ready_bf16, &fixture.kv_ready_bf16);
        assert!(
            wq.raw_f32
                .iter()
                .zip(&fixture.wq_f32)
                .all(|(actual, expected)| actual.to_bits() == expected.to_bits()),
            "wq_a raw fixture mismatch"
        );
        assert_eq!(wq.bf16, fixture.wq_bf16, "wq_a BF16 fixture");
        let q_norm_max = wq
            .norm_bf16
            .iter()
            .zip(&fixture.q_norm_bf16)
            .map(|(&actual, &expected)| (bf16_to_f32(actual) - bf16_to_f32(expected)).abs())
            .fold(0.0_f32, f32::max);
        assert!(q_norm_max <= 2.45e-4, "q_norm max {q_norm_max}");
        assert!(
            wkv.raw_f32
                .iter()
                .zip(&fixture.wkv_f32)
                .all(|(actual, expected)| actual.to_bits() == expected.to_bits()),
            "wkv raw fixture mismatch"
        );
        assert_eq!(wkv.bf16, fixture.wkv_bf16, "wkv BF16 fixture");
        assert_eq!(wkv.norm_bf16, fixture.kv_norm_bf16, "kv_norm BF16 fixture");
        assert!(
            wq_b.raw_f32
                .iter()
                .zip(&fixture.wq_b_f32)
                .all(|(actual, expected)| actual.to_bits() == expected.to_bits()),
            "wq_b raw fixture mismatch"
        );
        assert_eq!(wq_b.bf16, fixture.wq_b_bf16, "wq_b BF16 fixture");
        assert_eq!(
            q_head_rms, fixture.q_head_rms_bf16,
            "query head RMS BF16 fixture"
        );
        assert_eq!(q_rope, fixture.q_rope_bf16, "query RoPE BF16 fixture");
        assert_eq!(kv.rope_bf16, fixture.kv_rope_bf16, "KV RoPE fixture");
        assert_eq!(kv.scale_e8m0, fixture.kv_scale_e8m0, "KV scale fixture");
        assert_eq!(kv.ready_bf16, fixture.kv_ready_bf16, "KV ready fixture");
        assert_eq!(
            kv.causal_topk_i32, fixture.causal_topk_i32,
            "causal top-k fixture"
        );
        for position_head in 0..IDS.len() * Q_HEADS {
            let start = position_head * Q_HEAD_DIM;
            assert_eq!(
                &q_rope[start..start + Q_HEAD_DIM - Q_ROPE_DIM],
                &q_head_rms[start..start + Q_HEAD_DIM - Q_ROPE_DIM],
                "query RoPE non-rotary prefix"
            );
        }
        assert_eq!(
            &q_rope[..Q_HEADS * Q_HEAD_DIM],
            &q_head_rms[..Q_HEADS * Q_HEAD_DIM],
            "position zero query RoPE"
        );
        println!(
            "PROVISIONAL: CPU FP8 projection fixture compared; range={} B",
            projections.range_bytes
        );
    }

    #[test]
    #[ignore = "requires COLIBRI_DSV4_MODEL_DIR pinned snapshot"]
    fn real_state_matches_pytorch_fixture() {
        let meta: serde_json::Value = serde_json::from_str(JSON).unwrap();
        assert_eq!(meta["schema"], "colibri-v4-state-pytorch-oracle-v2");
        assert_eq!(meta["model_revision"], REVISION);
        assert_eq!(meta["ids"], serde_json::json!(IDS));
        assert_eq!(
            meta["shape"],
            serde_json::json!({"rows":4,"hidden":HIDDEN,"hc":HC,"mix":MIX,"sinkhorn_iters":20,"sinkhorn_trace_stages":41})
        );
        assert_eq!(
            meta["binary"]["sha256"],
            "21bd8b40da54ed88689ba2dbfa32085b3502ab0fb07c57e2676be6166f8761df"
        );
        assert_eq!(
            meta["hashes"]["embedding_bf16"],
            "8917b6d9f86eee2fbae604b787cb75e41f0caefd92ee63201a8d385ae5cea62a"
        );
        assert_eq!(
            meta["hashes"]["norm_bf16"],
            "7627d8b18a5eb98d14a8985094e01cd178fbad8bbc3103cfef8e7dd38bd8185f"
        );
        for (name, offset, bytes, dtype, shape) in [
            (
                "embedding",
                8,
                57_344,
                "bf16-le",
                serde_json::json!([4, HIDDEN]),
            ),
            (
                "collapsed",
                57_352,
                57_344,
                "bf16-le",
                serde_json::json!([4, HIDDEN]),
            ),
            (
                "norm",
                114_696,
                57_344,
                "bf16-le",
                serde_json::json!([4, HIDDEN]),
            ),
            ("post", 172_040, 64, "f32-le", serde_json::json!([4, HC])),
            (
                "comb",
                172_104,
                256,
                "f32-le",
                serde_json::json!([4, HC, HC]),
            ),
            ("mixes", 172_360, 384, "f32-le", serde_json::json!([4, MIX])),
            ("pre", 172_744, 64, "f32-le", serde_json::json!([4, HC])),
            (
                "sinkhorn_trace",
                172_808,
                10_496,
                "f32-le",
                serde_json::json!([4, 41, HC, HC]),
            ),
        ] {
            assert_eq!(meta["binary"][name]["offset"], offset);
            assert_eq!(meta["binary"][name]["bytes"], bytes);
            assert_eq!(meta["binary"][name]["dtype"], dtype);
            assert_eq!(meta["binary"][name]["shape"], shape);
        }
        assert_eq!(
            meta["binary"]["sinkhorn_trace"]["offset"].as_u64().unwrap()
                + meta["binary"]["sinkhorn_trace"]["bytes"].as_u64().unwrap(),
            BIN.len() as u64
        );
        for (key, hash) in [
            (
                "collapsed_bf16",
                "fe56fec4389401097a4077eb074f7ec43e96df2f95adfebcf74dff05f9938762",
            ),
            (
                "mixes_f32",
                "da634bc2f3dd2c489e18907747261dec4b10ccbd670ba718e2cf3683576ae950",
            ),
            (
                "pre_f32",
                "d57e9bfc3daeb7f28dfffd00842128479c09960b2e014e4bc350ed9d4eb42fd4",
            ),
            (
                "sinkhorn_trace_f32",
                "5485b83635e461caa9dfc7ec04861f78b32db94dd0387c54fe5bfff82a26332f",
            ),
            (
                "post_f32",
                "af9d19b328633c5fdb690c6ad26c0eba96b42b49c78898bbb19939d85898d069",
            ),
            (
                "comb_f32",
                "06b0fc03e564148f18f8174449f78404bbc1eacecc8d89e5266281befee87983",
            ),
        ] {
            assert_eq!(meta["hashes"][key], hash);
        }
        for (key, hash) in [
            (
                "comb_logits_f32",
                "a4b1e0f057627b769d61a3d2532137d8f63de35c3218f98d770df6e7dae50dc7",
            ),
            (
                "after_softmax_eps_f32",
                "50a5d1c4b86298d06dad7278e6c7f94fae4fbd15803e78e34ab61b8942aefc83",
            ),
            (
                "col_00_f32",
                "bec9400f7cc707a9ed44a7268c0ab89a099b9ed342f0c62031c0c94979e3e74d",
            ),
            (
                "row_00_f32",
                "a5ef67cea891a1fbfea685e8083790e6af7bec9beb6491d2f0e93b2522fa7657",
            ),
            (
                "col_19_f32",
                "06b0fc03e564148f18f8174449f78404bbc1eacecc8d89e5266281befee87983",
            ),
        ] {
            assert_eq!(meta["sinkhorn_stage_hashes"][key], hash);
        }
        assert_eq!(&BIN[..8], b"CV4S0001");
        assert_eq!(BIN.len(), 183_304);
        let root = std::env::var_os("COLIBRI_DSV4_MODEL_DIR").expect("set COLIBRI_DSV4_MODEL_DIR");
        let state = load(Path::new(&root), &IDS).unwrap();
        let embed = bf16(&BIN[8..57_352]);
        assert_eq!(state.embedding, embed, "embedding range bits");
        let (mhc, norm) = run_cpu_norm(&state, 4).unwrap();
        assert_close(
            "collapsed",
            &mhc.collapsed,
            &bf16(&BIN[57_352..114_696]),
            0.0,
        );
        assert_close("norm", &norm, &bf16(&BIN[114_696..172_040]), 0.0);
        report_f32(
            "mixes",
            &mhc_linear_cpu(&state, 4).unwrap(),
            &f32s(&BIN[172_360..172_744]),
        );
        report_f32("pre", &mhc.pre, &f32s(&BIN[172_744..172_808]));
        report_f32("post", &mhc.post, &f32s(&BIN[172_040..172_104]));
        report_f32("comb", &mhc.comb, &f32s(&BIN[172_104..172_360]));
        report_f32(
            "sinkhorn trace",
            &mhc.sinkhorn_trace,
            &f32s(&BIN[172_808..]),
        );
        let expected_trace = f32s(&BIN[172_808..]);
        for stage in 0..41 {
            let mut mismatches = 0;
            let mut maximum = 0.0_f32;
            for row in 0..4 {
                let start = (row * 41 + stage) * HC * HC;
                for (&actual, &expected) in mhc.sinkhorn_trace[start..start + HC * HC]
                    .iter()
                    .zip(&expected_trace[start..start + HC * HC])
                {
                    mismatches += usize::from(actual.to_bits() != expected.to_bits());
                    maximum = maximum.max((actual - expected).abs());
                }
            }
            println!("sinkhorn stage {stage:02}: bit_mismatch={mismatches}, max_abs={maximum:.6e}");
            assert_eq!(mismatches, 0, "sinkhorn stage {stage:02} raw bits");
        }
        println!(
            "PASS: CPU state fixture raw-bit exact through mHC pre/Sinkhorn/attn norm; range={} B",
            state.range_bytes
        );
    }
    fn bf16(bytes: &[u8]) -> Vec<u16> {
        bytes
            .chunks_exact(2)
            .map(|x| u16::from_le_bytes([x[0], x[1]]))
            .collect()
    }
    fn f32s(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|x| f32::from_le_bytes([x[0], x[1], x[2], x[3]]))
            .collect()
    }
    fn assert_close(name: &str, a: &[u16], b: &[u16], tol: f32) {
        assert_eq!(a.len(), b.len());
        let mut bit_mismatch = 0;
        let mut over_tolerance = 0;
        let mut max = 0.0_f32;
        for (&x, &y) in a.iter().zip(b) {
            let d = (bf16_to_f32(x) - bf16_to_f32(y)).abs();
            max = max.max(d);
            bit_mismatch += (x != y) as usize;
            over_tolerance += (d > tol) as usize;
        }
        println!(
            "{name}: bit_mismatch={bit_mismatch}, over_tolerance={over_tolerance}, max_abs={max:.6e}"
        );
        assert_eq!(over_tolerance, 0, "{name}: max {max}");
    }
    fn report_f32(name: &str, a: &[f32], b: &[f32]) {
        assert_eq!(a.len(), b.len(), "{name}: length");
        let bit_mismatch = a
            .iter()
            .zip(b)
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();
        let max = a
            .iter()
            .zip(b)
            .map(|(&x, &y)| (x - y).abs())
            .fold(0., f32::max);
        println!("{name}: bit_mismatch={bit_mismatch}, max_abs={max:.6e}");
        assert_eq!(bit_mismatch, 0, "{name}: raw bits");
    }

    fn report_f32_bounded(name: &str, a: &[f32], b: &[f32], tolerance: f32) {
        assert_eq!(a.len(), b.len(), "{name}: length");
        let bit_mismatch = a
            .iter()
            .zip(b)
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();
        let max = a
            .iter()
            .zip(b)
            .map(|(&x, &y)| (x - y).abs())
            .fold(0.0_f32, f32::max);
        println!(
            "{name}: bit_mismatch={bit_mismatch}, max_abs={max:.6e}, tolerance={tolerance:.1e}"
        );
        assert!(
            a.iter().chain(b).all(|value| value.is_finite()) && max <= tolerance,
            "{name}: max {max} exceeds canonical bound {tolerance}"
        );
    }

    fn report_bf16(name: &str, actual: &[u16], expected: &[u16]) {
        assert_eq!(actual.len(), expected.len(), "{name}: length");
        let mismatches = actual
            .iter()
            .zip(expected)
            .filter(|(actual, expected)| actual != expected)
            .count();
        let max = actual
            .iter()
            .zip(expected)
            .map(|(&actual, &expected)| (bf16_to_f32(actual) - bf16_to_f32(expected)).abs())
            .fold(0.0_f32, f32::max);
        println!("{name}: bit_mismatch={mismatches}, max_abs={max:.6e}; PROVISIONAL");
    }
}
