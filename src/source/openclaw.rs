use chrono::{DateTime, Utc};
use serde::Deserialize;
use serde_json::Value;

use super::looks_like_resource;
use crate::event::{BehavioralEvent, EventType};

/// Parses OpenClaw JSONL transcript lines into `BehavioralEvent`s.
///
/// Maintains session state (session ID, sequence position) across lines.
/// One JSONL line may produce zero or more events (e.g. an assistant message
/// with multiple tool calls produces one event per tool call).
pub struct OpenClawParser {
    session_id: String,
    agent_id: String,
    sequence_position: u32,
}

// -- Raw JSONL deserialization types (private) --

#[derive(Deserialize)]
struct RawLine {
    #[serde(rename = "type")]
    line_type: String,
    // Session fields
    id: Option<String>,
    timestamp: Option<String>,
    // Message fields
    message: Option<RawMessage>,
}

#[derive(Deserialize)]
struct RawMessage {
    role: String,
    content: Option<Value>,
    #[serde(rename = "toolName")]
    tool_name: Option<String>,
    usage: Option<RawUsage>,
    #[serde(rename = "isError")]
    is_error: Option<bool>,
    provider: Option<String>,
    model: Option<String>,
}

#[derive(Deserialize)]
struct RawUsage {
    #[serde(rename = "totalTokens")]
    total_tokens: Option<u64>,
}

impl OpenClawParser {
    pub fn new(agent_id: impl Into<String>) -> Self {
        Self {
            session_id: String::new(),
            agent_id: agent_id.into(),
            sequence_position: 0,
        }
    }

    /// Parse a single JSONL line. Returns zero or more events.
    /// Skips lines that don't map to behavioral events (model changes, cache TTL, etc.).
    pub fn parse_line(&mut self, line: &str) -> Result<Vec<BehavioralEvent>, serde_json::Error> {
        let raw: RawLine = serde_json::from_str(line)?;

        match raw.line_type.as_str() {
            "session" => {
                if let Some(id) = raw.id {
                    self.session_id = id;
                }
                self.sequence_position = 0;
                Ok(vec![])
            }
            "message" => {
                let Some(msg) = raw.message else {
                    return Ok(vec![]);
                };
                self.parse_message(&msg, raw.timestamp.as_deref())
            }
            _ => Ok(vec![]),
        }
    }

    fn parse_message(
        &mut self,
        msg: &RawMessage,
        line_timestamp: Option<&str>,
    ) -> Result<Vec<BehavioralEvent>, serde_json::Error> {
        let timestamp = line_timestamp
            .and_then(|ts| ts.parse::<DateTime<Utc>>().ok())
            .unwrap_or_else(Utc::now);

        match msg.role.as_str() {
            "user" => {
                let data_in_bytes = msg
                    .content
                    .as_ref()
                    .map(|c| c.to_string().len() as u64)
                    .unwrap_or(0);

                self.sequence_position += 1;
                Ok(vec![BehavioralEvent {
                    timestamp,
                    session_id: self.session_id.clone(),
                    agent_id: self.agent_id.clone(),
                    event_type: EventType::UserMessage,
                    tool_name: None,
                    param_keys: vec![],
                    resource_ids: vec![],
                    data_in_bytes,
                    data_out_bytes: 0,
                    duration_ms: 0,
                    token_count: None,
                    sequence_position: self.sequence_position,
                    trace_id: None,
                    span_id: None,
                    provider: None,
                    model: None,
                }])
            }
            "assistant" => self.parse_assistant_message(msg, timestamp),
            "toolResult" => {
                let data_out_bytes = msg
                    .content
                    .as_ref()
                    .map(|c| c.to_string().len() as u64)
                    .unwrap_or(0);

                let event_type = if msg.is_error == Some(true) {
                    EventType::Error
                } else {
                    EventType::ToolResult
                };

                self.sequence_position += 1;
                Ok(vec![BehavioralEvent {
                    timestamp,
                    session_id: self.session_id.clone(),
                    agent_id: self.agent_id.clone(),
                    event_type,
                    tool_name: msg.tool_name.clone(),
                    param_keys: vec![],
                    resource_ids: vec![],
                    data_in_bytes: 0,
                    data_out_bytes,
                    duration_ms: 0,
                    token_count: None,
                    sequence_position: self.sequence_position,
                    trace_id: None,
                    span_id: None,
                    provider: None,
                    model: None,
                }])
            }
            _ => Ok(vec![]),
        }
    }

    fn parse_assistant_message(
        &mut self,
        msg: &RawMessage,
        timestamp: DateTime<Utc>,
    ) -> Result<Vec<BehavioralEvent>, serde_json::Error> {
        let token_count = msg.usage.as_ref().and_then(|u| u.total_tokens);

        let Some(Value::Array(content_blocks)) = &msg.content else {
            return Ok(vec![]);
        };

        let mut events = Vec::new();
        let mut has_text = false;

        for block in content_blocks {
            let Some(block_type) = block.get("type").and_then(Value::as_str) else {
                continue;
            };

            match block_type {
                "toolCall" => {
                    let tool_name = block.get("name").and_then(Value::as_str).map(String::from);
                    let (param_keys, resource_ids, data_in_bytes) =
                        extract_argument_metadata(block.get("arguments"));

                    self.sequence_position += 1;
                    events.push(BehavioralEvent {
                        timestamp,
                        session_id: self.session_id.clone(),
                        agent_id: self.agent_id.clone(),
                        event_type: EventType::ToolCall,
                        tool_name,
                        param_keys,
                        resource_ids,
                        data_in_bytes,
                        data_out_bytes: 0,
                        duration_ms: 0,
                        token_count,
                        sequence_position: self.sequence_position,
                        trace_id: None,
                        span_id: None,
                        provider: msg.provider.clone(),
                        model: msg.model.clone(),
                    });
                }
                "text" => {
                    has_text = true;
                }
                _ => {}
            }
        }

        // If the assistant message had text but no tool calls, emit an AgentMessage event.
        if has_text && events.is_empty() {
            let data_out_bytes = msg
                .content
                .as_ref()
                .map(|c| c.to_string().len() as u64)
                .unwrap_or(0);

            self.sequence_position += 1;
            events.push(BehavioralEvent {
                timestamp,
                session_id: self.session_id.clone(),
                agent_id: self.agent_id.clone(),
                event_type: EventType::AgentMessage,
                tool_name: None,
                param_keys: vec![],
                resource_ids: vec![],
                data_in_bytes: 0,
                data_out_bytes,
                duration_ms: 0,
                token_count,
                sequence_position: self.sequence_position,
                trace_id: None,
                span_id: None,
                provider: msg.provider.clone(),
                model: msg.model.clone(),
            });
        }

        Ok(events)
    }
}

/// Extract param keys, resource IDs, and data size from a tool call's arguments.
///
/// - `param_keys`: the keys of the arguments object (never values).
/// - `resource_ids`: values that look like file paths or URLs.
/// - `data_in_bytes`: serialized size of the arguments.
fn extract_argument_metadata(arguments: Option<&Value>) -> (Vec<String>, Vec<String>, u64) {
    let Some(Value::Object(args)) = arguments else {
        return (vec![], vec![], 0);
    };

    let param_keys: Vec<String> = args.keys().cloned().collect();
    let data_in_bytes = serde_json::to_string(args)
        .map(|s| s.len() as u64)
        .unwrap_or(0);

    let mut resource_ids = Vec::new();
    for value in args.values() {
        if let Some(s) = value.as_str()
            && looks_like_resource(s)
        {
            resource_ids.push(s.to_string());
        }
    }

    (param_keys, resource_ids, data_in_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn session_line() -> String {
        r#"{"type":"session","version":3,"id":"test-session-123","timestamp":"2026-02-20T21:55:55.066Z","cwd":"/tmp"}"#.to_string()
    }

    fn user_message_line() -> String {
        r#"{"type":"message","id":"msg1","parentId":"prev","timestamp":"2026-02-20T21:56:49.041Z","message":{"role":"user","content":[{"type":"text","text":"hello"}],"timestamp":1771624609040}}"#.to_string()
    }

    fn assistant_text_line() -> String {
        r#"{"type":"message","id":"msg2","parentId":"msg1","timestamp":"2026-02-20T21:56:02.723Z","message":{"role":"assistant","content":[{"type":"text","text":"Hello! How can I help?"}],"api":"anthropic-messages","provider":"anthropic","model":"claude-sonnet-4-6","usage":{"input":10,"output":256,"cacheRead":0,"cacheWrite":14247,"totalTokens":14513,"cost":{"input":0.00003,"output":0.00384,"total":0.05729625}},"stopReason":"stop","timestamp":1771624555076}}"#.to_string()
    }

    fn assistant_tool_call_line() -> String {
        r##"{"type":"message","id":"msg3","parentId":"msg2","timestamp":"2026-02-20T21:56:58.914Z","message":{"role":"assistant","content":[{"type":"text","text":"Let me write that file."},{"type":"toolCall","id":"toolu_123","name":"write","arguments":{"file_path":"/Users/oliver/test.md","content":"# Hello"}}],"api":"anthropic-messages","provider":"anthropic","model":"claude-sonnet-4-6","usage":{"input":10,"output":470,"totalTokens":15105},"stopReason":"toolUse","timestamp":1771624609041}}"##.to_string()
    }

    fn tool_result_line() -> String {
        r#"{"type":"message","id":"msg4","parentId":"msg3","timestamp":"2026-02-20T21:56:58.928Z","message":{"role":"toolResult","toolCallId":"toolu_123","toolName":"write","content":[{"type":"text","text":"Successfully wrote 7 bytes to /Users/oliver/test.md"}],"isError":false,"timestamp":1771624618925}}"#.to_string()
    }

    fn tool_result_error_line() -> String {
        r#"{"type":"message","id":"msg5","parentId":"msg3","timestamp":"2026-02-20T21:57:00.000Z","message":{"role":"toolResult","toolCallId":"toolu_456","toolName":"exec","content":[{"type":"text","text":"Permission denied"}],"isError":true,"timestamp":1771624620000}}"#.to_string()
    }

    fn model_change_line() -> String {
        r#"{"type":"model_change","id":"mc1","parentId":null,"timestamp":"2026-02-20T21:55:55.069Z","provider":"anthropic","modelId":"claude-sonnet-4-6"}"#.to_string()
    }

    #[test]
    fn session_line_sets_session_id() {
        let mut parser = OpenClawParser::new("test-agent");
        let events = parser.parse_line(&session_line()).unwrap();
        assert!(events.is_empty());
        assert_eq!(parser.session_id, "test-session-123");
        assert_eq!(parser.sequence_position, 0);
    }

    #[test]
    fn user_message_produces_event() {
        let mut parser = OpenClawParser::new("test-agent");
        parser.parse_line(&session_line()).unwrap();

        let events = parser.parse_line(&user_message_line()).unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, EventType::UserMessage);
        assert_eq!(events[0].session_id, "test-session-123");
        assert_eq!(events[0].agent_id, "test-agent");
        assert_eq!(events[0].sequence_position, 1);
        assert!(events[0].data_in_bytes > 0);
    }

    #[test]
    fn assistant_text_only_produces_agent_message() {
        let mut parser = OpenClawParser::new("test-agent");
        parser.parse_line(&session_line()).unwrap();

        let events = parser.parse_line(&assistant_text_line()).unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, EventType::AgentMessage);
        assert_eq!(events[0].token_count, Some(14513));
    }

    #[test]
    fn assistant_tool_call_produces_tool_call_event() {
        let mut parser = OpenClawParser::new("test-agent");
        parser.parse_line(&session_line()).unwrap();

        let events = parser.parse_line(&assistant_tool_call_line()).unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, EventType::ToolCall);
        assert_eq!(events[0].tool_name.as_deref(), Some("write"));
        assert!(events[0].param_keys.contains(&"file_path".to_string()));
        assert!(events[0].param_keys.contains(&"content".to_string()));
        assert!(
            events[0]
                .resource_ids
                .contains(&"/Users/oliver/test.md".to_string())
        );
        assert!(events[0].data_in_bytes > 0);
        assert_eq!(events[0].token_count, Some(15105));
    }

    #[test]
    fn tool_result_produces_event() {
        let mut parser = OpenClawParser::new("test-agent");
        parser.parse_line(&session_line()).unwrap();

        let events = parser.parse_line(&tool_result_line()).unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, EventType::ToolResult);
        assert_eq!(events[0].tool_name.as_deref(), Some("write"));
        assert!(events[0].data_out_bytes > 0);
    }

    #[test]
    fn tool_result_error_produces_error_event() {
        let mut parser = OpenClawParser::new("test-agent");
        parser.parse_line(&session_line()).unwrap();

        let events = parser.parse_line(&tool_result_error_line()).unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, EventType::Error);
        assert_eq!(events[0].tool_name.as_deref(), Some("exec"));
    }

    #[test]
    fn skips_irrelevant_line_types() {
        let mut parser = OpenClawParser::new("test-agent");
        let events = parser.parse_line(&model_change_line()).unwrap();
        assert!(events.is_empty());
    }

    #[test]
    fn sequence_position_increments() {
        let mut parser = OpenClawParser::new("test-agent");
        parser.parse_line(&session_line()).unwrap();

        parser.parse_line(&user_message_line()).unwrap();
        let events = parser.parse_line(&assistant_tool_call_line()).unwrap();
        assert_eq!(events[0].sequence_position, 2);

        let events = parser.parse_line(&tool_result_line()).unwrap();
        assert_eq!(events[0].sequence_position, 3);
    }

    #[test]
    fn resource_extraction_heuristics() {
        use super::looks_like_resource;
        assert!(looks_like_resource("/usr/bin/env"));
        assert!(looks_like_resource("~/Documents/file.txt"));
        assert!(looks_like_resource("https://example.com"));
        assert!(looks_like_resource("http://localhost:8080"));
        assert!(!looks_like_resource("just some text"));
        assert!(!looks_like_resource("file.txt"));
    }

    #[test]
    fn assistant_null_content() {
        let mut parser = OpenClawParser::new("test-agent");
        parser.parse_line(&session_line()).unwrap();

        let line = r#"{"type":"message","id":"m1","parentId":null,"timestamp":"2026-02-20T10:00:00.000Z","message":{"role":"assistant","content":null,"usage":{"totalTokens":10},"timestamp":1771624609040}}"#;
        let events = parser.parse_line(line).unwrap();
        assert!(events.is_empty(), "null content should produce no events");
    }

    #[test]
    fn session_line_without_id() {
        let mut parser = OpenClawParser::new("test-agent");

        // First set a known session_id.
        parser.parse_line(&session_line()).unwrap();
        assert_eq!(parser.session_id, "test-session-123");

        // Session line without "id" field — session_id should stay unchanged.
        let no_id_line =
            r#"{"type":"session","version":3,"timestamp":"2026-02-20T12:00:00.000Z","cwd":"/tmp"}"#;
        let events = parser.parse_line(no_id_line).unwrap();
        assert!(events.is_empty());
        assert_eq!(
            parser.session_id, "test-session-123",
            "session_id should remain unchanged when no id in session line"
        );
        assert_eq!(
            parser.sequence_position, 0,
            "sequence_position should reset"
        );
    }

    #[test]
    fn bigram_serialization_roundtrips() {
        use crate::baseline::Bigram;
        use std::collections::HashMap;

        let mut bigrams: HashMap<Bigram, u64> = HashMap::new();
        bigrams.insert(Bigram::new("read", "write"), 5);
        bigrams.insert(Bigram::new("exec", "read"), 3);

        let json = serde_json::to_string(&bigrams).unwrap();
        let roundtripped: HashMap<Bigram, u64> = serde_json::from_str(&json).unwrap();
        assert_eq!(bigrams, roundtripped);
    }
}
