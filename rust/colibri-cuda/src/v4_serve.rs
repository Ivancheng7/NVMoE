use std::error::Error;
use std::io::{BufRead, Write};

pub const MAX_PAYLOAD_BYTES: usize = 1 << 20;
pub const MAX_PROMPT_TOKENS: usize = 512;
pub const MAX_NEW_TOKENS: usize = 64;

#[derive(Debug, PartialEq)]
pub struct Submit {
    pub request_id: String,
    pub cache_slot: usize,
    pub payload: String,
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub grammar_bytes: usize,
}

#[derive(Debug, PartialEq)]
pub enum Command {
    Submit(Submit),
    Cancel(String),
}

pub fn validate_submit(submit: &Submit, prompt_tokens: usize) -> Result<(), String> {
    if submit.cache_slot != 0 {
        return Err("KV_SLOTS_1_REQUIRED".to_owned());
    }
    if !submit.temperature.is_finite() || submit.temperature != 0.0 {
        return Err("TEMPERATURE_MUST_BE_0".to_owned());
    }
    if !submit.top_p.is_finite() || !(0.0..=1.0).contains(&submit.top_p) {
        return Err("TOP_P_OUT_OF_RANGE".to_owned());
    }
    if submit.grammar_bytes != 0 {
        return Err("DSPARK_AND_GRAMMAR_DEFERRED".to_owned());
    }
    if !(1..=MAX_NEW_TOKENS).contains(&submit.max_tokens) {
        return Err(format!("MAX_TOKENS_EXCEEDS_{}", MAX_NEW_TOKENS));
    }
    if !(1..=MAX_PROMPT_TOKENS).contains(&prompt_tokens) {
        return Err(format!(
            "CONTEXT_EXCEEDED {prompt_tokens} {MAX_PROMPT_TOKENS}"
        ));
    }
    Ok(())
}

pub fn read_command(reader: &mut impl BufRead) -> Result<Option<Command>, Box<dyn Error>> {
    let mut header = String::new();
    if reader.read_line(&mut header)? == 0 {
        return Ok(None);
    }
    let fields = header.split_whitespace().collect::<Vec<_>>();
    match fields.first().copied() {
        Some("CANCEL") if fields.len() == 2 => Ok(Some(Command::Cancel(fields[1].to_owned()))),
        Some("SUBMIT") if matches!(fields.len(), 7 | 8) => {
            let request_id = fields[1].to_owned();
            let cache_slot = fields[2].parse::<usize>()?;
            let payload_bytes = fields[3].parse::<usize>()?;
            let max_tokens = fields[4].parse::<usize>()?;
            let temperature = fields[5].parse::<f32>()?;
            let top_p = fields[6].parse::<f32>()?;
            let grammar_bytes = fields
                .get(7)
                .map_or(Ok(0), |value| value.parse::<usize>())?;
            let frame_bytes = payload_bytes
                .checked_add(grammar_bytes)
                .ok_or("Flash serve frame size overflow")?;
            if frame_bytes > MAX_PAYLOAD_BYTES {
                return Err("Flash serve frame exceeds 1 MiB".into());
            }
            let mut frame = vec![0_u8; frame_bytes];
            reader.read_exact(&mut frame)?;
            let mut terminator = [0_u8; 1];
            reader.read_exact(&mut terminator)?;
            if terminator != [b'\n'] {
                return Err("Flash serve frame lacks newline terminator".into());
            }
            let payload = String::from_utf8(frame[..payload_bytes].to_vec())?;
            Ok(Some(Command::Submit(Submit {
                request_id,
                cache_slot,
                payload,
                max_tokens,
                temperature,
                top_p,
                grammar_bytes,
            })))
        }
        _ => Err("invalid Flash serve command".into()),
    }
}

pub fn write_ready(writer: &mut impl Write) -> Result<(), Box<dyn Error>> {
    writer.write_all(b"\x01\x01READY\x01\x01\nSTAT 0 0 0 0\n")?;
    writer.flush()?;
    Ok(())
}

pub fn write_data(
    writer: &mut impl Write,
    request_id: &str,
    payload: &[u8],
) -> Result<(), Box<dyn Error>> {
    writeln!(writer, "DATA {request_id} {}", payload.len())?;
    writer.write_all(payload)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    Ok(())
}

pub fn write_error(
    writer: &mut impl Write,
    request_id: &str,
    message: &str,
) -> Result<(), Box<dyn Error>> {
    let message = message.replace(['\r', '\n'], " ");
    writeln!(writer, "ERROR {request_id} {message}")?;
    writer.flush()?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn write_done(
    writer: &mut impl Write,
    request_id: &str,
    completion_tokens: usize,
    tokens_per_second: f64,
    prompt_tokens: usize,
    length_limited: bool,
    ttft_ms: f64,
    itl_ms: f64,
    stop_reason: &str,
    generated_ids: &[i64],
) -> Result<(), Box<dyn Error>> {
    let generated_ids = generated_ids
        .iter()
        .map(i64::to_string)
        .collect::<Vec<_>>()
        .join(",");
    writeln!(
        writer,
        "DONE {request_id} STAT {completion_tokens} {tokens_per_second:.6} 0 0 {prompt_tokens} {} {ttft_ms:.3} {itl_ms:.3} {stop_reason} ids={generated_ids}",
        usize::from(length_limited)
    )?;
    writer.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufReader, Cursor};

    #[test]
    fn submit_payload_is_framed_without_losing_the_next_cancel() {
        let input = b"SUBMIT 7 0 14 3 0 1\n[{\"role\":\"u\"}]\nCANCEL 7\n";
        let mut reader = BufReader::new(Cursor::new(input));
        let Command::Submit(submit) = read_command(&mut reader).unwrap().unwrap() else {
            panic!("expected SUBMIT");
        };
        assert_eq!(submit.request_id, "7");
        assert_eq!(submit.payload, "[{\"role\":\"u\"}]");
        assert_eq!(
            read_command(&mut reader).unwrap(),
            Some(Command::Cancel("7".into()))
        );
    }

    #[test]
    fn normal_submit_data_done_matches_the_gateway_protocol() {
        let mut input = BufReader::new(Cursor::new(b"SUBMIT 2 0 2 1 0 1\n[]\n".as_slice()));
        assert!(matches!(
            read_command(&mut input).unwrap(),
            Some(Command::Submit(Submit { request_id, .. })) if request_id == "2"
        ));
        let mut output = Vec::new();
        write_ready(&mut output).unwrap();
        write_data(&mut output, "2", "好".as_bytes()).unwrap();
        write_done(&mut output, "2", 1, 2.5, 9, false, 400.0, 0.0, "eos", &[42]).unwrap();
        assert_eq!(
            String::from_utf8(output).unwrap(),
            "\u{1}\u{1}READY\u{1}\u{1}\nSTAT 0 0 0 0\nDATA 2 3\n好\nDONE 2 STAT 1 2.500000 0 0 9 0 400.000 0.000 eos ids=42\n"
        );
    }

    #[test]
    fn rejects_sampling_and_oversized_context_before_execution() {
        let mut submit = Submit {
            request_id: "1".into(),
            cache_slot: 0,
            payload: "[]".into(),
            max_tokens: 8,
            temperature: 0.1,
            top_p: 1.0,
            grammar_bytes: 0,
        };
        let temperature_error = validate_submit(&submit, 8).unwrap_err();
        assert_eq!(temperature_error, "TEMPERATURE_MUST_BE_0");
        let mut output = Vec::new();
        write_error(&mut output, &submit.request_id, &temperature_error).unwrap();
        assert_eq!(output, b"ERROR 1 TEMPERATURE_MUST_BE_0\n");
        submit.temperature = 0.0;
        let context_error = validate_submit(&submit, MAX_PROMPT_TOKENS + 1).unwrap_err();
        assert_eq!(context_error, "CONTEXT_EXCEEDED 513 512");
        output.clear();
        write_error(&mut output, &submit.request_id, &context_error).unwrap();
        assert_eq!(output, b"ERROR 1 CONTEXT_EXCEEDED 513 512\n");
    }
}
