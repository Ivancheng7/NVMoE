//! Exact V4 checkpoint profiles used by the range-backed static loader.
//!
//! Flash-0731 keeps base-only execution as its default. The fixed DSpark B=5
//! path is a separate, explicit opt-in mode.

use serde_json::Value;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct V4Spec {
    pub name: &'static str,
    pub revision: &'static str,
    pub shard_count: usize,
    pub shard_bytes: u64,
    pub layers: usize,
    pub hidden: usize,
    pub vocab: usize,
    pub q_heads: usize,
    pub q_head_dim: usize,
    pub q_rope_dim: usize,
    pub q_lora_rank: usize,
    pub o_groups: usize,
    pub o_lora_rank: usize,
    pub routed_experts: usize,
    pub experts_per_token: usize,
    pub moe_intermediate: usize,
    pub hash_layers: usize,
    pub index_topk: usize,
    pub route_scale_bits: u32,
    pub compression_ratios: &'static [usize],
    pub base_only_no_speculation: bool,
}

pub const PRO_REVISION: &str = "b5968e9190ef611bbf34a7229255be88a0e937c1";
pub const FLASH_0731_REVISION: &str = "9e165c30e2704aec5d9d593cce3eebd58bbef1cb";

// `compress_ratios` includes the legacy Pro MTP tail. Base-layer callers use
// only 0..layers, while config validation checks the complete upstream array.
const PRO_RATIOS: [usize; 62] = [
    128, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4,
    128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4,
    128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 0,
];
// Flash's three DSpark stages are an upstream config tail, not base layers.
const FLASH_0731_RATIOS: [usize; 46] = [
    0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
    4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 0, 0, 0,
];

pub const PRO: V4Spec = V4Spec {
    name: "deepseek-v4-pro",
    revision: PRO_REVISION,
    shard_count: 64,
    shard_bytes: 864_721_029_744,
    layers: 61,
    hidden: 7_168,
    vocab: 129_280,
    q_heads: 128,
    q_head_dim: 512,
    q_rope_dim: 64,
    q_lora_rank: 1_536,
    o_groups: 16,
    o_lora_rank: 1_024,
    routed_experts: 384,
    experts_per_token: 6,
    moe_intermediate: 3_072,
    hash_layers: 3,
    index_topk: 1_024,
    route_scale_bits: 2.5_f32.to_bits(),
    compression_ratios: &PRO_RATIOS,
    base_only_no_speculation: false,
};

pub const FLASH_0731: V4Spec = V4Spec {
    name: "deepseek-v4-flash-0731",
    revision: FLASH_0731_REVISION,
    shard_count: 48,
    // Sum of the 48 complete safetensors files, including their headers.
    shard_bytes: 166_886_535_336,
    layers: 43,
    hidden: 4_096,
    vocab: 129_280,
    q_heads: 64,
    q_head_dim: 512,
    q_rope_dim: 64,
    q_lora_rank: 1_024,
    o_groups: 8,
    o_lora_rank: 1_024,
    routed_experts: 256,
    experts_per_token: 6,
    moe_intermediate: 2_048,
    hash_layers: 3,
    index_topk: 512,
    route_scale_bits: 1.5_f32.to_bits(),
    compression_ratios: &FLASH_0731_RATIOS,
    base_only_no_speculation: true,
};

impl V4Spec {
    pub const fn route_scale(self) -> f32 {
        f32::from_bits(self.route_scale_bits)
    }

    pub fn compression_ratio(self, layer: usize) -> Result<usize, String> {
        self.compression_ratios
            .get(layer)
            .copied()
            .ok_or_else(|| format!("{} layer {layer} is outside 0..{}", self.name, self.layers))
    }

    pub fn shard_name(self, number: usize) -> Result<String, String> {
        if number == 0 || number > self.shard_count {
            return Err(format!(
                "{} shard {number} is outside 1..={}",
                self.name, self.shard_count
            ));
        }
        Ok(format!(
            "model-{number:05}-of-{:05}.safetensors",
            self.shard_count
        ))
    }

    pub fn model_dir_name(self) -> String {
        format!("{}-{}", self.name, self.revision)
    }

    pub fn validate_config(self, config: &Value) -> Result<(), String> {
        let field = |key: &str| config.get(key).ok_or_else(|| format!("config lacks {key}"));
        let number = |key: &str| {
            field(key)?
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| format!("config {key} is not a usize"))
        };
        let expected = [
            ("hidden_size", self.hidden),
            ("vocab_size", self.vocab),
            ("num_hidden_layers", self.layers),
            ("num_attention_heads", self.q_heads),
            ("head_dim", self.q_head_dim),
            ("qk_rope_head_dim", self.q_rope_dim),
            ("q_lora_rank", self.q_lora_rank),
            ("o_groups", self.o_groups),
            ("o_lora_rank", self.o_lora_rank),
            ("n_routed_experts", self.routed_experts),
            ("num_experts_per_tok", self.experts_per_token),
            ("moe_intermediate_size", self.moe_intermediate),
            ("num_hash_layers", self.hash_layers),
            ("index_topk", self.index_topk),
        ];
        if config.get("model_type").and_then(Value::as_str) != Some("deepseek_v4") {
            return Err("config model_type is not deepseek_v4".into());
        }
        for (key, expected) in expected {
            if number(key)? != expected {
                return Err(format!("config {key} does not match {}", self.name));
            }
        }
        let ratios = field("compress_ratios")?
            .as_array()
            .ok_or_else(|| "config compress_ratios is not an array".to_owned())?
            .iter()
            .map(|value| value.as_u64().and_then(|value| usize::try_from(value).ok()))
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| "config compress_ratios is not usize data".to_owned())?;
        if ratios.as_slice() != self.compression_ratios {
            return Err(format!(
                "config compress_ratios does not match {}",
                self.name
            ));
        }
        let route_scale = field("routed_scaling_factor")?
            .as_f64()
            .ok_or_else(|| "config routed_scaling_factor is not a number".to_owned())?
            as f32;
        if route_scale.to_bits() != self.route_scale_bits {
            return Err(format!(
                "config routed_scaling_factor does not match {}",
                self.name
            ));
        }
        if self.base_only_no_speculation {
            for (key, expected) in [
                ("dspark_block_size", 5_usize),
                ("dspark_noise_token_id", 128_799),
                ("dspark_markov_rank", 256),
            ] {
                if number(key)? != expected {
                    return Err(format!(
                        "config {key} does not match Flash-0731 DSpark metadata"
                    ));
                }
            }
            if config.get("dspark_target_layer_ids") != Some(&serde_json::json!([40, 41, 42])) {
                return Err("config dspark_target_layer_ids does not match Flash-0731".into());
            }
        }
        Ok(())
    }

    pub fn validate_config_bytes(self, bytes: &[u8]) -> Result<(), String> {
        let config: Value = serde_json::from_slice(bytes).map_err(|error| error.to_string())?;
        self.validate_config(&config)
    }
}

pub fn profile_from_config_bytes(bytes: &[u8]) -> Result<&'static V4Spec, String> {
    let config: Value = serde_json::from_slice(bytes).map_err(|error| error.to_string())?;
    for spec in [&PRO, &FLASH_0731] {
        if spec.validate_config(&config).is_ok() {
            return Ok(spec);
        }
    }
    Err("config does not match an enabled exact V4 profile".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flash_0731_profile_rejects_preview_geometry() {
        let flash = serde_json::json!({
            "model_type":"deepseek_v4", "hidden_size":4096, "vocab_size":129280,
            "num_hidden_layers":43, "num_attention_heads":64, "head_dim":512,
            "qk_rope_head_dim":64, "q_lora_rank":1024, "o_groups":8, "o_lora_rank":1024,
            "n_routed_experts":256, "num_experts_per_tok":6, "moe_intermediate_size":2048,
            "num_hash_layers":3, "index_topk":512, "routed_scaling_factor":1.5,
            "compress_ratios":[0,0,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,0,0,0],
            "dspark_block_size":5, "dspark_noise_token_id":128799,
            "dspark_target_layer_ids":[40,41,42], "dspark_markov_rank":256
        });
        assert_eq!(
            profile_from_config_bytes(flash.to_string().as_bytes()).unwrap(),
            &FLASH_0731
        );
        assert_eq!(FLASH_0731.compression_ratio(0).unwrap(), 0);
        assert_eq!(FLASH_0731.compression_ratio(42).unwrap(), 4);
        assert_eq!(FLASH_0731.shard_bytes, 166_886_535_336);
        assert_eq!(
            FLASH_0731.shard_name(48).unwrap(),
            "model-00048-of-00048.safetensors"
        );
    }
}
