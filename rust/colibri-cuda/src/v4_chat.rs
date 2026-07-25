//! Minimal DeepSeek-V4 single-user chat prompt formatting.
//!
//! This mirrors `encoding/encoding_dsv4.py::encode_messages` for its narrow,
//! interactive single-user path. It deliberately keeps user content opaque.

/// Selects the marker appended after the assistant role token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThinkingMode {
    /// Start the assistant response outside a thinking block.
    Chat,
    /// Start the assistant response inside a thinking block.
    Thinking,
}

const BOS: &str = "<｜begin▁of▁sentence｜>";
const USER: &str = "<｜User｜>";
const ASSISTANT: &str = "<｜Assistant｜>";
const THINKING_START: &str = "<think>";
const THINKING_END: &str = "</think>";

// ponytail: this supports exactly one user message; add roles or history only
// when their official encoding contract has a separate golden suite.
/// Formats one raw user message for the official V4 chat entrypoint.
pub fn format_single_user(content: &str, mode: ThinkingMode) -> String {
    let suffix = match mode {
        ThinkingMode::Chat => THINKING_END,
        ThinkingMode::Thinking => THINKING_START,
    };

    [BOS, USER, content, ASSISTANT, suffix].concat()
}
