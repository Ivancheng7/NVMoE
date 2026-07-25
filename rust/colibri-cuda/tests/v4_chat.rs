#[path = "../src/v4_chat.rs"]
mod v4_chat;

use v4_chat::{ThinkingMode, format_single_user};

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
    }
}
