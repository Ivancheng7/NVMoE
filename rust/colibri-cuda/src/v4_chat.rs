//! DeepSeek-V4-Flash-0731 prompt encoding.
//!
//! The templates and transitions match `encoding/encoding_dsv4.py` at
//! `deepseek-ai/DeepSeek-V4-Flash-0731@9e165c30e2704aec5d9d593cce3eebd58bbef1cb`.

use std::collections::HashMap;
use std::fmt;

use serde::de::{Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};

/// Selects the marker appended after the assistant role token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThinkingMode {
    /// Start the assistant response outside a thinking block.
    Chat,
    /// Start the assistant response inside a thinking block.
    Thinking,
}

const BOS: &str = "<｜begin▁of▁sentence｜>";
const EOS: &str = "<｜end▁of▁sentence｜>";
const USER: &str = "<｜User｜>";
const ASSISTANT: &str = "<｜Assistant｜>";
const LATEST_REMINDER: &str = "<｜latest_reminder｜>";
const THINKING_START: &str = "<think>";
const THINKING_END: &str = "</think>";
const DSML: &str = "｜DSML｜";

const TOOLS_TEMPLATE: &str = r#"## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<{dsml_token}tool_calls>" block like the following:

<{dsml_token}tool_calls>
<{dsml_token}invoke name="$TOOL_NAME">
<{dsml_token}parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</{dsml_token}parameter>
...
</{dsml_token}invoke>
<{dsml_token}invoke name="$TOOL_NAME2">
...
</{dsml_token}invoke>
</{dsml_token}tool_calls>

String parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.

If thinking_mode is enabled (triggered by {thinking_start_token}), you MUST output your complete reasoning inside {thinking_start_token}...{thinking_end_token} BEFORE any tool calls or final response.

Otherwise, output directly after {thinking_end_token} with tool calls or final response.

### Available Tool Schemas

{tool_schemas}

You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.
"#;

#[derive(Clone, Debug, PartialEq)]
enum OrderedValue {
    Null,
    Bool(bool),
    Number(serde_json::Number),
    String(String),
    Array(Vec<Self>),
    Object(Vec<(String, Self)>),
}

impl OrderedValue {
    fn field(&self, key: &str) -> Option<&Self> {
        self.as_object()?
            .iter()
            .find_map(|(name, value)| (name == key).then_some(value))
    }

    fn field_mut(&mut self, key: &str) -> Option<&mut Self> {
        self.as_object_mut()?
            .iter_mut()
            .find_map(|(name, value)| (name == key).then_some(value))
    }

    fn insert(&mut self, key: &str, value: Self) -> Result<(), String> {
        let fields = self
            .as_object_mut()
            .ok_or_else(|| "V4 message must be a JSON object".to_owned())?;
        if let Some((_, old)) = fields.iter_mut().find(|(name, _)| name == key) {
            *old = value;
        } else {
            fields.push((key.to_owned(), value));
        }
        Ok(())
    }

    fn remove(&mut self, key: &str) {
        if let Some(fields) = self.as_object_mut()
            && let Some(index) = fields.iter().position(|(name, _)| name == key)
        {
            fields.remove(index);
        }
    }

    fn as_object(&self) -> Option<&[(String, Self)]> {
        match self {
            Self::Object(value) => Some(value),
            _ => None,
        }
    }

    fn as_object_mut(&mut self) -> Option<&mut Vec<(String, Self)>> {
        match self {
            Self::Object(value) => Some(value),
            _ => None,
        }
    }

    fn as_array(&self) -> Option<&[Self]> {
        match self {
            Self::Array(value) => Some(value),
            _ => None,
        }
    }

    fn as_array_mut(&mut self) -> Option<&mut Vec<Self>> {
        match self {
            Self::Array(value) => Some(value),
            _ => None,
        }
    }

    fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(value) => Some(value),
            _ => None,
        }
    }

    fn truthy(&self) -> bool {
        match self {
            Self::Null => false,
            Self::Bool(value) => *value,
            Self::Number(value) => value.as_f64() != Some(0.0),
            Self::String(value) => !value.is_empty(),
            Self::Array(value) => !value.is_empty(),
            Self::Object(value) => !value.is_empty(),
        }
    }
}

struct OrderedValueVisitor;

impl<'de> Visitor<'de> for OrderedValueVisitor {
    type Value = OrderedValue;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("JSON data")
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E> {
        Ok(OrderedValue::Null)
    }

    fn visit_none<E>(self) -> Result<Self::Value, E> {
        Ok(OrderedValue::Null)
    }

    fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E> {
        Ok(OrderedValue::Bool(value))
    }

    fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E> {
        Ok(OrderedValue::Number(value.into()))
    }

    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E> {
        Ok(OrderedValue::Number(value.into()))
    }

    fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        serde_json::Number::from_f64(value)
            .map(OrderedValue::Number)
            .ok_or_else(|| E::custom("non-finite JSON number"))
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E> {
        Ok(OrderedValue::String(value.to_owned()))
    }

    fn visit_string<E>(self, value: String) -> Result<Self::Value, E> {
        Ok(OrderedValue::String(value))
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut values = Vec::new();
        while let Some(value) = sequence.next_element()? {
            values.push(value);
        }
        Ok(OrderedValue::Array(values))
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut values = Vec::new();
        while let Some(entry) = map.next_entry()? {
            values.push(entry);
        }
        Ok(OrderedValue::Object(values))
    }
}

impl<'de> Deserialize<'de> for OrderedValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(OrderedValueVisitor)
    }
}

fn python_json(value: &OrderedValue) -> String {
    match value {
        OrderedValue::Null => "null".to_owned(),
        OrderedValue::Bool(value) => value.to_string(),
        OrderedValue::Number(value) => value.to_string(),
        OrderedValue::String(value) => serde_json::to_string(value).expect("serialize JSON string"),
        OrderedValue::Array(values) => format!(
            "[{}]",
            values
                .iter()
                .map(python_json)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        OrderedValue::Object(fields) => format!(
            "{{{}}}",
            fields
                .iter()
                .map(|(key, value)| format!(
                    "{}: {}",
                    serde_json::to_string(key).expect("serialize JSON key"),
                    python_json(value)
                ))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

fn role(message: &OrderedValue) -> Result<&str, String> {
    message
        .field("role")
        .and_then(OrderedValue::as_str)
        .ok_or_else(|| "V4 message lacks a string role".to_owned())
}

fn content(message: &OrderedValue) -> Result<&str, String> {
    match message.field("content") {
        None | Some(OrderedValue::Null) => Ok(""),
        Some(OrderedValue::String(value)) => Ok(value),
        Some(_) => Err("V4 message content must be a string".to_owned()),
    }
}

fn last_user_index(messages: &[OrderedValue]) -> Option<usize> {
    messages
        .iter()
        .rposition(|message| matches!(role(message), Ok("user" | "developer")))
}

fn merge_tool_messages(messages: Vec<OrderedValue>) -> Result<Vec<OrderedValue>, String> {
    let mut merged: Vec<OrderedValue> = Vec::new();
    for message in messages {
        match role(&message)? {
            "tool" => {
                let block = OrderedValue::Object(vec![
                    ("type".into(), OrderedValue::String("tool_result".into())),
                    (
                        "tool_use_id".into(),
                        message
                            .field("tool_call_id")
                            .cloned()
                            .unwrap_or_else(|| OrderedValue::String(String::new())),
                    ),
                    (
                        "content".into(),
                        message
                            .field("content")
                            .cloned()
                            .unwrap_or_else(|| OrderedValue::String(String::new())),
                    ),
                ]);
                let append = merged.last().is_some_and(|previous| {
                    role(previous) == Ok("user") && previous.field("content_blocks").is_some()
                });
                if append {
                    merged
                        .last_mut()
                        .and_then(|previous| previous.field_mut("content_blocks"))
                        .and_then(OrderedValue::as_array_mut)
                        .expect("checked merged user content blocks")
                        .push(block);
                } else {
                    merged.push(OrderedValue::Object(vec![
                        ("role".into(), OrderedValue::String("user".into())),
                        ("content_blocks".into(), OrderedValue::Array(vec![block])),
                    ]));
                }
            }
            "user" => {
                let text = content(&message)?.to_owned();
                let block = OrderedValue::Object(vec![
                    ("type".into(), OrderedValue::String("text".into())),
                    ("text".into(), OrderedValue::String(text.clone())),
                ]);
                let append = merged.last().is_some_and(|previous| {
                    role(previous) == Ok("user")
                        && previous.field("content_blocks").is_some()
                        && previous
                            .field("task")
                            .is_none_or(|task| matches!(task, OrderedValue::Null))
                });
                if append {
                    merged
                        .last_mut()
                        .and_then(|previous| previous.field_mut("content_blocks"))
                        .and_then(OrderedValue::as_array_mut)
                        .expect("checked merged user content blocks")
                        .push(block);
                } else {
                    let mut user = OrderedValue::Object(vec![
                        ("role".into(), OrderedValue::String("user".into())),
                        ("content".into(), OrderedValue::String(text)),
                        ("content_blocks".into(), OrderedValue::Array(vec![block])),
                    ]);
                    for key in ["task", "wo_eos", "mask"] {
                        if let Some(value) = message.field(key) {
                            user.insert(key, value.clone())?;
                        }
                    }
                    merged.push(user);
                }
            }
            _ => merged.push(message),
        }
    }
    Ok(merged)
}

fn sort_tool_results(messages: &mut [OrderedValue]) -> Result<(), String> {
    let mut order = HashMap::new();
    for message in messages {
        match role(message)? {
            "assistant" => {
                let Some(calls) = message.field("tool_calls").filter(|value| value.truthy()) else {
                    continue;
                };
                order.clear();
                for (index, call) in calls
                    .as_array()
                    .ok_or_else(|| "assistant tool_calls must be an array".to_owned())?
                    .iter()
                    .enumerate()
                {
                    let id = call.field("id").and_then(OrderedValue::as_str).or_else(|| {
                        call.field("function")
                            .and_then(|function| function.field("id"))
                            .and_then(OrderedValue::as_str)
                    });
                    if let Some(id) = id {
                        order.insert(id.to_owned(), index);
                    }
                }
            }
            "user" => {
                let Some(blocks) = message
                    .field_mut("content_blocks")
                    .and_then(OrderedValue::as_array_mut)
                else {
                    continue;
                };
                let positions = blocks
                    .iter()
                    .enumerate()
                    .filter_map(|(index, block)| {
                        (block.field("type").and_then(OrderedValue::as_str) == Some("tool_result"))
                            .then_some(index)
                    })
                    .collect::<Vec<_>>();
                if positions.len() > 1 && !order.is_empty() {
                    let mut sorted = positions
                        .iter()
                        .map(|&index| blocks[index].clone())
                        .collect::<Vec<_>>();
                    sorted.sort_by_key(|block| {
                        block
                            .field("tool_use_id")
                            .and_then(OrderedValue::as_str)
                            .and_then(|id| order.get(id))
                            .copied()
                            .unwrap_or(0)
                    });
                    for (position, block) in positions.into_iter().zip(sorted) {
                        blocks[position] = block;
                    }
                }
            }
            _ => {}
        }
    }
    Ok(())
}

fn drop_old_thinking(messages: &[OrderedValue]) -> Result<Vec<OrderedValue>, String> {
    let last_user = last_user_index(messages).unwrap_or(usize::MAX);
    let mut result = Vec::new();
    for (index, message) in messages.iter().enumerate() {
        match role(message)? {
            "user" | "system" | "tool" | "latest_reminder" | "direct_search_results" => {
                result.push(message.clone());
            }
            _ if index >= last_user => result.push(message.clone()),
            "assistant" => {
                let mut message = message.clone();
                message.remove("reasoning_content");
                result.push(message);
            }
            _ => {}
        }
    }
    Ok(result)
}

fn render_tools(tools: &OrderedValue) -> Result<String, String> {
    let schemas = tools
        .as_array()
        .ok_or_else(|| "V4 tools must be an array".to_owned())?
        .iter()
        .map(|tool| {
            tool.field("function")
                .map(python_json)
                .ok_or_else(|| "V4 tool lacks a function schema".to_owned())
        })
        .collect::<Result<Vec<_>, _>>()?
        .join("\n");
    Ok(TOOLS_TEMPLATE
        .replace("{tool_schemas}", &schemas)
        .replace("{dsml_token}", DSML)
        .replace("{thinking_start_token}", THINKING_START)
        .replace("{thinking_end_token}", THINKING_END))
}

fn render_arguments(arguments: &str) -> Result<String, String> {
    let value = serde_json::from_str::<OrderedValue>(arguments).unwrap_or_else(|_| {
        OrderedValue::Object(vec![(
            "arguments".into(),
            OrderedValue::String(arguments.to_owned()),
        )])
    });
    value
        .as_object()
        .ok_or_else(|| "V4 tool arguments must decode to a JSON object".to_owned())?
        .iter()
        .map(|(key, value)| {
            let (is_string, value) = match value {
                OrderedValue::String(value) => ("true", value.clone()),
                value => ("false", python_json(value)),
            };
            Ok(format!(
                "<{DSML}parameter name=\"{key}\" string=\"{is_string}\">{value}</{DSML}parameter>"
            ))
        })
        .collect::<Result<Vec<_>, String>>()
        .map(|parameters| parameters.join("\n"))
}

fn render_tool_calls(calls: &OrderedValue) -> Result<String, String> {
    let calls = calls
        .as_array()
        .ok_or_else(|| "assistant tool_calls must be an array".to_owned())?
        .iter()
        .map(|call| {
            let function = call
                .field("function")
                .ok_or_else(|| "assistant tool call lacks function".to_owned())?;
            let name = function
                .field("name")
                .and_then(OrderedValue::as_str)
                .ok_or_else(|| "assistant tool call lacks function name".to_owned())?;
            let arguments = function
                .field("arguments")
                .and_then(OrderedValue::as_str)
                .ok_or_else(|| "assistant tool call lacks string arguments".to_owned())?;
            Ok(format!(
                "<{DSML}invoke name=\"{name}\">\n{}\n</{DSML}invoke>",
                render_arguments(arguments)?
            ))
        })
        .collect::<Result<Vec<_>, String>>()?
        .join("\n");
    Ok(format!(
        "\n\n<{DSML}tool_calls>\n{calls}\n</{DSML}tool_calls>"
    ))
}

fn render_content_blocks(blocks: &OrderedValue) -> Result<String, String> {
    blocks
        .as_array()
        .ok_or_else(|| "user content_blocks must be an array".to_owned())?
        .iter()
        .map(|block| {
            let kind = block
                .field("type")
                .and_then(OrderedValue::as_str)
                .unwrap_or("null");
            match kind {
                "text" => Ok(block
                    .field("text")
                    .and_then(OrderedValue::as_str)
                    .unwrap_or("")
                    .to_owned()),
                "tool_result" => {
                    let value = block
                        .field("content")
                        .cloned()
                        .unwrap_or_else(|| OrderedValue::String(String::new()));
                    let text = match value {
                        OrderedValue::String(text) => text,
                        OrderedValue::Array(parts) => parts
                            .iter()
                            .map(|part| {
                                if part.field("type").and_then(OrderedValue::as_str) == Some("text")
                                {
                                    part.field("text")
                                        .and_then(OrderedValue::as_str)
                                        .unwrap_or("")
                                        .to_owned()
                                } else {
                                    format!(
                                        "[Unsupported {}]",
                                        part.field("type")
                                            .and_then(OrderedValue::as_str)
                                            .unwrap_or("null")
                                    )
                                }
                            })
                            .collect::<Vec<_>>()
                            .join("\n\n"),
                        _ => return Err("tool result content must be text blocks".to_owned()),
                    };
                    Ok(format!("<tool_result>{text}</tool_result>"))
                }
                other => Ok(format!("[Unsupported {other}]")),
            }
        })
        .collect::<Result<Vec<_>, _>>()
        .map(|parts| parts.join("\n\n"))
}

fn task_token(task: &str) -> Option<&'static str> {
    match task {
        "action" => Some("<｜action｜>"),
        "query" => Some("<｜query｜>"),
        "authority" => Some("<｜authority｜>"),
        "domain" => Some("<｜domain｜>"),
        "title" => Some("<｜title｜>"),
        "read_url" => Some("<｜read_url｜>"),
        _ => None,
    }
}

fn render_message(
    index: usize,
    messages: &[OrderedValue],
    mode: ThinkingMode,
    drop_thinking: bool,
) -> Result<String, String> {
    let message = &messages[index];
    let message_role = role(message)?;
    let last_user = last_user_index(messages).unwrap_or(usize::MAX);
    let mut prompt = String::new();

    match message_role {
        "system" => {
            prompt.push_str(content(message)?);
            if let Some(tools) = message.field("tools").filter(|value| value.truthy()) {
                prompt.push_str("\n\n");
                prompt.push_str(&render_tools(tools)?);
            }
            if let Some(format) = message
                .field("response_format")
                .filter(|value| value.truthy())
            {
                prompt.push_str("\n\n## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n");
                prompt.push_str(&python_json(format));
            }
        }
        "developer" => {
            let developer = content(message)?;
            if developer.is_empty() {
                return Err("V4 developer content cannot be empty".to_owned());
            }
            prompt.push_str(USER);
            prompt.push_str(developer);
            if let Some(tools) = message.field("tools").filter(|value| value.truthy()) {
                prompt.push_str("\n\n");
                prompt.push_str(&render_tools(tools)?);
            }
            if let Some(format) = message
                .field("response_format")
                .filter(|value| value.truthy())
            {
                prompt.push_str("\n\n## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n");
                prompt.push_str(&python_json(format));
            }
        }
        "user" => {
            prompt.push_str(USER);
            if let Some(blocks) = message
                .field("content_blocks")
                .filter(|value| value.truthy())
            {
                prompt.push_str(&render_content_blocks(blocks)?);
            } else {
                prompt.push_str(content(message)?);
            }
        }
        "latest_reminder" => {
            prompt.push_str(LATEST_REMINDER);
            prompt.push_str(content(message)?);
        }
        "assistant" => {
            let previous_has_task = index > 0
                && messages[index - 1]
                    .field("task")
                    .is_some_and(|value| !matches!(value, OrderedValue::Null));
            if mode == ThinkingMode::Thinking && !previous_has_task {
                if !drop_thinking || index > last_user {
                    prompt.push_str(
                        message
                            .field("reasoning_content")
                            .and_then(OrderedValue::as_str)
                            .unwrap_or(""),
                    );
                    prompt.push_str(THINKING_END);
                }
            }
            prompt.push_str(content(message)?);
            if let Some(calls) = message.field("tool_calls").filter(|value| value.truthy()) {
                prompt.push_str(&render_tool_calls(calls)?);
            }
            let without_eos = message
                .field("wo_eos")
                .is_some_and(|value| matches!(value, OrderedValue::Bool(true)));
            if !without_eos {
                prompt.push_str(EOS);
            }
        }
        "tool" => return Err("V4 tool message was not merged".to_owned()),
        other => return Err(format!("unsupported V4 message role {other:?}")),
    }

    if index + 1 < messages.len()
        && !matches!(role(&messages[index + 1])?, "assistant" | "latest_reminder")
    {
        return Ok(prompt);
    }

    if let Some(task) = message
        .field("task")
        .filter(|value| !matches!(value, OrderedValue::Null))
    {
        let task = task
            .as_str()
            .ok_or_else(|| "V4 task must be a string".to_owned())?;
        let token = task_token(task).ok_or_else(|| format!("unsupported V4 task {task:?}"))?;
        if task == "action" {
            prompt.push_str(ASSISTANT);
            prompt.push_str(if mode == ThinkingMode::Thinking {
                THINKING_START
            } else {
                THINKING_END
            });
        }
        prompt.push_str(token);
    } else if matches!(message_role, "user" | "developer") {
        prompt.push_str(ASSISTANT);
        prompt.push_str(
            if mode == ThinkingMode::Thinking && (!drop_thinking || index >= last_user) {
                THINKING_START
            } else {
                THINKING_END
            },
        );
    }

    Ok(prompt)
}

fn format_ordered_messages(
    messages: Vec<OrderedValue>,
    mode: ThinkingMode,
) -> Result<String, String> {
    let mut messages = merge_tool_messages(messages)?;
    sort_tool_results(&mut messages)?;
    let drop_thinking = !messages
        .iter()
        .any(|message| message.field("tools").is_some_and(OrderedValue::truthy));
    if mode == ThinkingMode::Thinking && drop_thinking {
        messages = drop_old_thinking(&messages)?;
    }
    let mut prompt = BOS.to_owned();
    for index in 0..messages.len() {
        prompt.push_str(&render_message(index, &messages, mode, drop_thinking)?);
    }
    Ok(prompt)
}

// ponytail: this is the official 0731 default, no-context path. Add context and
// reasoning-effort controls only when a runtime caller has matching official goldens.
/// Formats an OpenAI-compatible JSON message array using the official 0731 encoder defaults.
pub fn format_messages_json_0731(
    messages_json: &str,
    mode: ThinkingMode,
) -> Result<String, String> {
    let messages = serde_json::from_str::<OrderedValue>(messages_json)
        .map_err(|error| format!("parse V4 messages: {error}"))?;
    format_ordered_messages(
        messages
            .as_array()
            .ok_or_else(|| "V4 messages must be a JSON array".to_owned())?
            .to_vec(),
        mode,
    )
}

#[cfg(test)]
pub fn format_official_fixture_0731(
    input_json: &str,
    mode: ThinkingMode,
) -> Result<String, String> {
    let root = serde_json::from_str::<OrderedValue>(input_json)
        .map_err(|error| format!("parse official V4 fixture: {error}"))?;
    if let Some(messages) = root.as_array() {
        return format_ordered_messages(messages.to_vec(), mode);
    }
    let mut messages = root
        .field("messages")
        .and_then(OrderedValue::as_array)
        .ok_or_else(|| "official V4 fixture lacks messages".to_owned())?
        .to_vec();
    if let Some(tools) = root.field("tools").filter(|value| value.truthy()) {
        messages
            .first_mut()
            .ok_or_else(|| "official V4 fixture has no first message".to_owned())?
            .insert("tools", tools.clone())?;
    }
    format_ordered_messages(messages, mode)
}

/// Formats one raw user message for the official V4 chat entrypoint.
pub fn format_single_user(content: &str, mode: ThinkingMode) -> String {
    let suffix = match mode {
        ThinkingMode::Chat => THINKING_END,
        ThinkingMode::Thinking => THINKING_START,
    };
    [BOS, USER, content, ASSISTANT, suffix].concat()
}
