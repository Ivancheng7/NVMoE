#[path = "../src/v4_chat.rs"]
mod v4_chat;

use std::path::PathBuf;

use v4_chat::{
    ThinkingMode, format_messages_json_0731, format_official_fixture_0731, format_single_user,
};

#[test]
fn matches_official_single_user_golden_prompts_byte_for_byte() {
    let fixture = include_str!("fixtures/v4_chat_golden.tsv");
    for line in fixture.lines().skip(1) {
        let mut fields = line.splitn(4, '\t');
        let case = fields.next().expect("case field");
        let mode = match fields.next().expect("mode field") {
            "chat" => ThinkingMode::Chat,
            "thinking" => ThinkingMode::Thinking,
            other => panic!("{case} has unsupported mode {other:?}"),
        };
        let content = fields.next().expect("content field");
        let expected = fields.next().expect("prompt field");
        let actual = format_single_user(content, mode);
        assert_eq!(
            actual.as_bytes(),
            expected.as_bytes(),
            "official prompt mismatch for {case}"
        );
        let messages = serde_json::to_string(&[serde_json::json!({
            "role": "user",
            "content": content,
        })])
        .expect("serialize single-user message");
        assert_eq!(
            format_messages_json_0731(&messages, mode).expect("encode single-user message"),
            expected,
            "general 0731 encoder mismatch for {case}"
        );
    }
}

#[test]
#[ignore = "requires the fixed official 0731 encoding fixtures and tokenizer"]
fn official_0731_goldens_match_byte_for_byte_and_token_by_token() {
    let fixture_root = PathBuf::from(
        std::env::var_os("COLIBRI_V4_ENCODING_FIXTURE_DIR")
            .expect("set COLIBRI_V4_ENCODING_FIXTURE_DIR"),
    );
    let tokenizer_root = PathBuf::from(
        std::env::var_os("COLIBRI_DSV4_MODEL_DIR").expect("set COLIBRI_DSV4_MODEL_DIR"),
    );
    let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_root.join("tokenizer.json"))
        .expect("open pinned 0731 tokenizer");
    for (case, mode) in [
        (1, ThinkingMode::Thinking),
        (2, ThinkingMode::Thinking),
        (3, ThinkingMode::Thinking),
        (4, ThinkingMode::Chat),
    ] {
        let input =
            std::fs::read_to_string(fixture_root.join(format!("tests/test_input_{case}.json")))
                .expect("read official input");
        let expected =
            std::fs::read_to_string(fixture_root.join(format!("tests/test_output_{case}.txt")))
                .expect("read official output");
        let actual = if case == 1 {
            format_official_fixture_0731(&input, mode)
        } else {
            format_messages_json_0731(&input, mode)
        }
        .unwrap_or_else(|error| panic!("encode official case {case}: {error}"));
        if actual.as_bytes() != expected.as_bytes() {
            let index = actual
                .bytes()
                .zip(expected.bytes())
                .position(|(actual, expected)| actual != expected)
                .unwrap_or_else(|| actual.len().min(expected.len()));
            let start = index.saturating_sub(32);
            let actual_end = (index + 64).min(actual.len());
            let expected_end = (index + 64).min(expected.len());
            panic!(
                "case {case} first byte mismatch at {index}: actual {:?}, expected {:?}",
                String::from_utf8_lossy(&actual.as_bytes()[start..actual_end]),
                String::from_utf8_lossy(&expected.as_bytes()[start..expected_end]),
            );
        }

        let actual_ids = tokenizer
            .encode(actual, false)
            .expect("tokenize actual prompt")
            .get_ids()
            .to_vec();
        let expected_ids = tokenizer
            .encode(expected, false)
            .expect("tokenize official prompt")
            .get_ids()
            .to_vec();
        assert_eq!(
            actual_ids.len(),
            expected_ids.len(),
            "case {case} token count"
        );
        for (index, (actual, expected)) in actual_ids.iter().zip(&expected_ids).enumerate() {
            assert_eq!(actual, expected, "case {case} token {index}");
        }
    }
}
