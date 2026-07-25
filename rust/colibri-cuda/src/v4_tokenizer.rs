use std::error::Error;
use std::path::{Path, PathBuf};

use tokenizers::Tokenizer;

const TOKENIZER_SHA256: &str = "8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf";
const TOKENIZER_CONFIG_SHA256: &str =
    "6ac8c8dc065ed118161d02dd532749ae3f52c243deac27872134fae2f50d8547";
const BOS_TOKEN: &str = "<｜begin▁of▁sentence｜>";
const EOS_TOKEN: &str = "<｜end▁of▁sentence｜>";

pub struct V4Tokenizer {
    tokenizer: Tokenizer,
}

impl V4Tokenizer {
    pub fn open(root: &Path) -> Result<Self, Box<dyn Error>> {
        let tokenizer_path = regular_file(root.join("tokenizer.json"))?;
        let config_path = regular_file(root.join("tokenizer_config.json"))?;
        if super::sha256_file(&tokenizer_path)? != TOKENIZER_SHA256 {
            return Err("V4 tokenizer.json does not match the pinned digest".into());
        }
        if super::sha256_file(&config_path)? != TOKENIZER_CONFIG_SHA256 {
            return Err("V4 tokenizer_config.json does not match the pinned digest".into());
        }
        let config: serde_json::Value = serde_json::from_slice(&std::fs::read(&config_path)?)?;
        if config["add_bos_token"] != false || config["add_eos_token"] != false {
            return Err("V4 tokenizer config must disable automatic BOS/EOS insertion".into());
        }
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|error| format!("open pinned V4 tokenizer: {error}"))?;
        validate_special_token(&tokenizer, &config, "bos_token", BOS_TOKEN, 0)?;
        validate_special_token(&tokenizer, &config, "eos_token", EOS_TOKEN, 1)?;
        validate_special_token(&tokenizer, &config, "pad_token", EOS_TOKEN, 1)?;
        Ok(Self { tokenizer })
    }

    pub fn encode_raw(&self, text: &str) -> Result<Vec<i64>, Box<dyn Error>> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|error| format!("encode raw V4 prompt: {error}"))?;
        Ok(encoding.get_ids().iter().map(|&id| i64::from(id)).collect())
    }

    pub fn decode(&self, token_ids: &[i64]) -> Result<String, Box<dyn Error>> {
        let ids = token_ids
            .iter()
            .map(|&id| u32::try_from(id).map_err(|_| format!("invalid V4 token ID {id}")))
            .collect::<Result<Vec<_>, _>>()?;
        self.tokenizer
            .decode(&ids, false)
            .map_err(|error| format!("decode V4 token IDs: {error}").into())
    }
}

fn regular_file(path: PathBuf) -> Result<PathBuf, Box<dyn Error>> {
    let metadata = path.symlink_metadata()?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(format!(
            "V4 tokenizer input must be a regular non-symlink file: {}",
            path.display()
        )
        .into());
    }
    Ok(path)
}

fn validate_special_token(
    tokenizer: &Tokenizer,
    config: &serde_json::Value,
    config_key: &str,
    expected_content: &str,
    expected_id: u32,
) -> Result<(), Box<dyn Error>> {
    if config[config_key]["content"].as_str() != Some(expected_content) {
        return Err(format!("V4 tokenizer config has an unexpected {config_key}").into());
    }
    if tokenizer.token_to_id(expected_content) != Some(expected_id) {
        return Err(format!("V4 tokenizer {config_key} has an unexpected token ID").into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires pinned DeepSeek-V4-Pro snapshot"]
    fn pinned_v4_tokenizer_matches_raw_fixture() {
        let root = std::env::var_os("COLIBRI_DSV4_MODEL_DIR")
            .map(PathBuf::from)
            .expect("set COLIBRI_DSV4_MODEL_DIR");
        let tokenizer = V4Tokenizer::open(&root).expect("open pinned V4 tokenizer");
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/v4_tokenizer.json"))
                .expect("parse tokenizer fixture");
        for raw in fixture["raw"].as_array().expect("raw fixture entries") {
            let text = raw["text"].as_str().expect("fixture text");
            let ids = raw["ids"]
                .as_array()
                .expect("fixture IDs")
                .iter()
                .map(|id| id.as_i64().expect("signed fixture ID"))
                .collect::<Vec<_>>();
            assert_eq!(tokenizer.encode_raw(text).unwrap(), ids, "encode {text:?}");
            assert_eq!(
                tokenizer.decode(&ids).unwrap(),
                raw["decoded"].as_str().expect("fixture decoded text"),
                "decode {text:?}"
            );
        }
        for special in ["bos", "eos", "pad"] {
            let entry = &fixture["special"][special];
            let text = entry["text"].as_str().expect("special text");
            assert_eq!(
                tokenizer.encode_raw(text).unwrap(),
                vec![entry["id"].as_i64().expect("special ID")],
                "encode {special}"
            );
        }
    }
}
