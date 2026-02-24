use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// The normalized event type that all sources produce.
/// Captures the shape of agent actions, not content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehavioralEvent {
    pub timestamp: DateTime<Utc>,
    pub session_id: String,
    pub agent_id: String,
    pub event_type: EventType,
    pub tool_name: Option<String>,
    /// Parameter names only — never values.
    pub param_keys: Vec<String>,
    /// File paths, URLs, hostnames, table names.
    pub resource_ids: Vec<String>,
    pub data_in_bytes: u64,
    pub data_out_bytes: u64,
    pub duration_ms: u64,
    pub token_count: Option<u64>,
    /// Position within the current session.
    pub sequence_position: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trace_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub span_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum EventType {
    ToolCall,
    ToolResult,
    UserMessage,
    AgentMessage,
    Error,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backward_compat_old_json_deserializes_with_none() {
        let old_json = r#"{"timestamp":"2026-02-20T10:00:00Z","session_id":"s1","agent_id":"a1","event_type":"ToolCall","tool_name":"read","param_keys":[],"resource_ids":[],"data_in_bytes":0,"data_out_bytes":0,"duration_ms":0,"token_count":null,"sequence_position":1}"#;
        let event: BehavioralEvent = serde_json::from_str(old_json).unwrap();
        assert!(event.trace_id.is_none());
        assert!(event.span_id.is_none());
        assert!(event.provider.is_none());
        assert!(event.model.is_none());
    }

    #[test]
    fn new_fields_roundtrip() {
        let event = BehavioralEvent {
            timestamp: chrono::Utc::now(),
            session_id: "s1".into(),
            agent_id: "a1".into(),
            event_type: EventType::ToolCall,
            tool_name: Some("read".into()),
            param_keys: vec![],
            resource_ids: vec![],
            data_in_bytes: 0,
            data_out_bytes: 0,
            duration_ms: 0,
            token_count: None,
            sequence_position: 1,
            trace_id: Some("abc123".into()),
            span_id: Some("def456".into()),
            provider: Some("openai".into()),
            model: Some("gpt-4".into()),
        };
        let json = serde_json::to_string(&event).unwrap();
        let decoded: BehavioralEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.trace_id.as_deref(), Some("abc123"));
        assert_eq!(decoded.span_id.as_deref(), Some("def456"));
        assert_eq!(decoded.provider.as_deref(), Some("openai"));
        assert_eq!(decoded.model.as_deref(), Some("gpt-4"));
    }

    #[test]
    fn none_fields_omitted_from_json() {
        let event = BehavioralEvent {
            timestamp: chrono::Utc::now(),
            session_id: "s1".into(),
            agent_id: "a1".into(),
            event_type: EventType::ToolCall,
            tool_name: None,
            param_keys: vec![],
            resource_ids: vec![],
            data_in_bytes: 0,
            data_out_bytes: 0,
            duration_ms: 0,
            token_count: None,
            sequence_position: 1,
            trace_id: None,
            span_id: None,
            provider: None,
            model: None,
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(!json.contains("trace_id"));
        assert!(!json.contains("span_id"));
        assert!(!json.contains("provider"));
        assert!(!json.contains("model"));
    }
}
