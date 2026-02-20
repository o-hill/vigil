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
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum EventType {
    ToolCall,
    ToolResult,
    UserMessage,
    AgentMessage,
    Error,
}
