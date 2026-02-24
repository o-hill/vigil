use std::collections::HashMap;

use chrono::{DateTime, TimeZone, Utc};
use serde::Deserialize;

use super::looks_like_resource;
use crate::event::{BehavioralEvent, EventType};

// ---------------------------------------------------------------------------
// OTLP JSON deserialization types (private)
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ExportTraceServiceRequest {
    #[serde(default)]
    resource_spans: Vec<ResourceSpans>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ResourceSpans {
    #[serde(default)]
    scope_spans: Vec<ScopeSpans>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ScopeSpans {
    #[serde(default)]
    spans: Vec<OtlpSpan>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct OtlpSpan {
    #[serde(default)]
    trace_id: String,
    #[serde(default)]
    span_id: String,
    #[serde(default)]
    name: String,
    #[serde(default)]
    start_time_unix_nano: Option<StringOrU64>,
    #[serde(default)]
    end_time_unix_nano: Option<StringOrU64>,
    #[serde(default)]
    attributes: Vec<KeyValue>,
    #[serde(default)]
    status: Option<SpanStatus>,
}

#[derive(Deserialize)]
struct SpanStatus {
    #[serde(default)]
    code: u32,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum StringOrU64 {
    String(String),
    Number(u64),
}

impl StringOrU64 {
    fn as_u64(&self) -> u64 {
        match self {
            StringOrU64::String(s) => s.parse().unwrap_or(0),
            StringOrU64::Number(n) => *n,
        }
    }
}

impl Default for StringOrU64 {
    fn default() -> Self {
        StringOrU64::Number(0)
    }
}

#[derive(Deserialize)]
struct KeyValue {
    key: String,
    value: Option<AnyValue>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct AnyValue {
    string_value: Option<String>,
    int_value: Option<StringOrU64>,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn attributes_map(attrs: &[KeyValue]) -> HashMap<&str, String> {
    let mut map = HashMap::new();
    for kv in attrs {
        if let Some(ref val) = kv.value {
            let s = if let Some(ref sv) = val.string_value {
                sv.clone()
            } else if let Some(ref iv) = val.int_value {
                iv.as_u64().to_string()
            } else {
                continue;
            };
            map.insert(kv.key.as_str(), s);
        }
    }
    map
}

fn nanos_to_datetime(nanos: u64) -> DateTime<Utc> {
    let secs = (nanos / 1_000_000_000) as i64;
    let nsec = (nanos % 1_000_000_000) as u32;
    Utc.timestamp_opt(secs, nsec)
        .single()
        .unwrap_or_else(Utc::now)
}

// ---------------------------------------------------------------------------
// OtelParser
// ---------------------------------------------------------------------------

/// Parses OTLP JSON (ExportTraceServiceRequest) into `BehavioralEvent`s.
pub struct OtelParser {
    default_agent_id: String,
}

impl OtelParser {
    pub fn new(default_agent_id: impl Into<String>) -> Self {
        Self {
            default_agent_id: default_agent_id.into(),
        }
    }

    pub fn parse(&self, input: &str) -> Result<Vec<BehavioralEvent>, serde_json::Error> {
        let request: ExportTraceServiceRequest = serde_json::from_str(input)?;

        // Collect all spans, grouped by trace_id.
        let mut spans_by_trace: HashMap<String, Vec<OtlpSpan>> = HashMap::new();
        for rs in request.resource_spans {
            for ss in rs.scope_spans {
                for span in ss.spans {
                    spans_by_trace
                        .entry(span.trace_id.clone())
                        .or_default()
                        .push(span);
                }
            }
        }

        let mut all_events = Vec::new();

        for (trace_id, mut spans) in spans_by_trace {
            // Sort by start_time.
            spans.sort_by_key(|s| {
                s.start_time_unix_nano
                    .as_ref()
                    .map(|t| t.as_u64())
                    .unwrap_or(0)
            });

            let mut seq: u32 = 0;
            for span in &spans {
                if let Some(event) = self.convert_span(span, &trace_id, &mut seq) {
                    all_events.push(event);
                }
            }
        }

        // Sort all events by timestamp for deterministic output.
        all_events.sort_by_key(|e| e.timestamp);

        Ok(all_events)
    }

    fn convert_span(
        &self,
        span: &OtlpSpan,
        trace_id: &str,
        seq: &mut u32,
    ) -> Option<BehavioralEvent> {
        let attrs = attributes_map(&span.attributes);

        // Determine event type using dual-namespace resolution.
        let event_type = self.resolve_event_type(&attrs, span)?;

        // session_id: gen_ai.conversation.id > session.id > trace_id
        let session_id = attrs
            .get("gen_ai.conversation.id")
            .or_else(|| attrs.get("session.id"))
            .cloned()
            .unwrap_or_else(|| trace_id.to_string());

        // agent_id
        let agent_id = attrs
            .get("gen_ai.agent.name")
            .or_else(|| attrs.get("gen_ai.agent.id"))
            .or_else(|| attrs.get("agent.name"))
            .cloned()
            .unwrap_or_else(|| self.default_agent_id.clone());

        // tool_name
        let tool_name = attrs
            .get("gen_ai.tool.name")
            .or_else(|| attrs.get("tool.name"))
            .cloned();

        // param_keys + resource_ids + data_in_bytes from tool arguments
        let (param_keys, resource_ids, data_in_bytes) =
            if let Some(args_str) = attrs.get("gen_ai.tool.call.arguments") {
                extract_from_args_json(args_str)
            } else {
                (vec![], vec![], 0)
            };

        // data_out_bytes from tool result
        let data_out_bytes = attrs
            .get("gen_ai.tool.call.result")
            .map(|s| s.len() as u64)
            .unwrap_or(0);

        // token_count
        let token_count = {
            let input_tokens: u64 = attrs
                .get("gen_ai.usage.input_tokens")
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
            let output_tokens: u64 = attrs
                .get("gen_ai.usage.output_tokens")
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
            let total = input_tokens + output_tokens;
            if total > 0 {
                Some(total)
            } else {
                attrs
                    .get("llm.token_count.total")
                    .and_then(|s| s.parse().ok())
            }
        };

        // duration_ms from span timestamps
        let start_nanos = span
            .start_time_unix_nano
            .as_ref()
            .map(|t| t.as_u64())
            .unwrap_or(0);
        let end_nanos = span
            .end_time_unix_nano
            .as_ref()
            .map(|t| t.as_u64())
            .unwrap_or(0);
        let duration_ms = if end_nanos > start_nanos {
            (end_nanos - start_nanos) / 1_000_000
        } else {
            0
        };

        let timestamp = nanos_to_datetime(start_nanos);

        let provider = attrs.get("gen_ai.provider.name").cloned();
        let model = attrs.get("gen_ai.request.model").cloned();

        *seq += 1;

        Some(BehavioralEvent {
            timestamp,
            session_id,
            agent_id,
            event_type,
            tool_name,
            param_keys,
            resource_ids,
            data_in_bytes,
            data_out_bytes,
            duration_ms,
            token_count,
            sequence_position: *seq,
            trace_id: Some(trace_id.to_string()),
            span_id: Some(span.span_id.clone()),
            provider,
            model,
        })
    }

    fn resolve_event_type(
        &self,
        attrs: &HashMap<&str, String>,
        span: &OtlpSpan,
    ) -> Option<EventType> {
        // Check for error status on tool spans first.
        let is_error = span.status.as_ref().is_some_and(|s| s.code == 2);

        // 1. gen_ai.operation.name (preferred namespace)
        if let Some(op) = attrs.get("gen_ai.operation.name") {
            return match op.as_str() {
                "execute_tool" => {
                    if is_error {
                        Some(EventType::Error)
                    } else {
                        Some(EventType::ToolCall)
                    }
                }
                "chat" => {
                    // Distinguish user vs agent by checking for role or content attributes.
                    if attrs.get("gen_ai.message.role").map(|s| s.as_str()) == Some("user") {
                        Some(EventType::UserMessage)
                    } else {
                        Some(EventType::AgentMessage)
                    }
                }
                "invoke_agent" | "create_agent" => None, // structural, skip
                _ => Some(EventType::AgentMessage),      // default for unknown gen_ai ops
            };
        }

        // 2. Fallback: openinference.span.kind
        if let Some(kind) = attrs.get("openinference.span.kind") {
            return match kind.as_str() {
                "TOOL" => {
                    if is_error {
                        Some(EventType::Error)
                    } else {
                        Some(EventType::ToolCall)
                    }
                }
                "LLM" | "AGENT" => Some(EventType::AgentMessage),
                _ => None,
            };
        }

        // 3. Heuristic: span name
        if span.name.contains("tool") {
            return Some(if is_error {
                EventType::Error
            } else {
                EventType::ToolCall
            });
        }

        None
    }
}

fn extract_from_args_json(args_str: &str) -> (Vec<String>, Vec<String>, u64) {
    let data_in_bytes = args_str.len() as u64;

    let Ok(parsed) = serde_json::from_str::<serde_json::Value>(args_str) else {
        return (vec![], vec![], data_in_bytes);
    };

    let Some(obj) = parsed.as_object() else {
        return (vec![], vec![], data_in_bytes);
    };

    let param_keys: Vec<String> = obj.keys().cloned().collect();
    let mut resource_ids = Vec::new();
    for value in obj.values() {
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

    fn make_otlp_json(spans_json: &str) -> String {
        format!(r#"{{"resourceSpans":[{{"scopeSpans":[{{"spans":{spans_json}}}]}}]}}"#)
    }

    fn tool_span(
        trace_id: &str,
        span_id: &str,
        name: &str,
        start_ns: u64,
        end_ns: u64,
        attrs: &[(&str, &str)],
    ) -> String {
        let attrs_json: Vec<String> = attrs
            .iter()
            .map(|(k, v)| format!(r#"{{"key":"{k}","value":{{"stringValue":"{v}"}}}}"#))
            .collect();
        format!(
            r#"{{"traceId":"{trace_id}","spanId":"{span_id}","name":"{name}","startTimeUnixNano":"{start_ns}","endTimeUnixNano":"{end_ns}","attributes":[{attrs}]}}"#,
            attrs = attrs_json.join(",")
        )
    }

    fn tool_span_with_status(
        trace_id: &str,
        span_id: &str,
        name: &str,
        start_ns: u64,
        end_ns: u64,
        attrs: &[(&str, &str)],
        status_code: u32,
    ) -> String {
        let attrs_json: Vec<String> = attrs
            .iter()
            .map(|(k, v)| format!(r#"{{"key":"{k}","value":{{"stringValue":"{v}"}}}}"#))
            .collect();
        format!(
            r#"{{"traceId":"{trace_id}","spanId":"{span_id}","name":"{name}","startTimeUnixNano":"{start_ns}","endTimeUnixNano":"{end_ns}","attributes":[{attrs}],"status":{{"code":{status_code}}}}}"#,
            attrs = attrs_json.join(",")
        )
    }

    #[test]
    fn gen_ai_tool_call() {
        let span = tool_span(
            "t1",
            "s1",
            "execute_tool",
            1_000_000_000,
            2_000_000_000,
            &[
                ("gen_ai.operation.name", "execute_tool"),
                ("gen_ai.tool.name", "read_file"),
                (
                    "gen_ai.tool.call.arguments",
                    r#"{\"file_path\":\"/tmp/foo.txt\"}"#,
                ),
                ("gen_ai.tool.call.result", "file contents here"),
            ],
        );
        let input = make_otlp_json(&format!("[{span}]"));
        let parser = OtelParser::new("default-agent");
        let events = parser.parse(&input).unwrap();

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, EventType::ToolCall);
        assert_eq!(events[0].tool_name.as_deref(), Some("read_file"));
        assert!(events[0].param_keys.contains(&"file_path".to_string()));
        assert!(events[0].resource_ids.contains(&"/tmp/foo.txt".to_string()));
        assert!(events[0].data_in_bytes > 0);
        assert!(events[0].data_out_bytes > 0);
        assert_eq!(events[0].duration_ms, 1000);
    }

    #[test]
    fn gen_ai_chat_agent_message() {
        let span = tool_span(
            "t1",
            "s1",
            "chat",
            1_000_000_000,
            2_000_000_000,
            &[
                ("gen_ai.operation.name", "chat"),
                ("gen_ai.message.role", "assistant"),
                ("gen_ai.usage.input_tokens", "100"),
                ("gen_ai.usage.output_tokens", "200"),
            ],
        );
        let input = make_otlp_json(&format!("[{span}]"));
        let parser = OtelParser::new("default-agent");
        let events = parser.parse(&input).unwrap();

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, EventType::AgentMessage);
        assert_eq!(events[0].token_count, Some(300));
    }

    #[test]
    fn gen_ai_chat_user_message() {
        let span = tool_span(
            "t1",
            "s1",
            "chat",
            1_000_000_000,
            2_000_000_000,
            &[
                ("gen_ai.operation.name", "chat"),
                ("gen_ai.message.role", "user"),
            ],
        );
        let input = make_otlp_json(&format!("[{span}]"));
        let parser = OtelParser::new("default-agent");
        let events = parser.parse(&input).unwrap();

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, EventType::UserMessage);
    }

    #[test]
    fn openinference_tool() {
        let span = tool_span(
            "t1",
            "s1",
            "run_tool",
            1_000_000_000,
            2_000_000_000,
            &[("openinference.span.kind", "TOOL"), ("tool.name", "search")],
        );
        let input = make_otlp_json(&format!("[{span}]"));
        let parser = OtelParser::new("default-agent");
        let events = parser.parse(&input).unwrap();

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, EventType::ToolCall);
        assert_eq!(events[0].tool_name.as_deref(), Some("search"));
    }

    #[test]
    fn openinference_llm() {
        let span = tool_span(
            "t1",
            "s1",
            "llm_call",
            1_000_000_000,
            2_000_000_000,
            &[
                ("openinference.span.kind", "LLM"),
                ("llm.token_count.total", "500"),
            ],
        );
        let input = make_otlp_json(&format!("[{span}]"));
        let parser = OtelParser::new("default-agent");
        let events = parser.parse(&input).unwrap();

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, EventType::AgentMessage);
        assert_eq!(events[0].token_count, Some(500));
    }

    #[test]
    fn sequence_positions_within_trace() {
        let s1 = tool_span(
            "t1",
            "s1",
            "tool1",
            1_000_000_000,
            2_000_000_000,
            &[
                ("gen_ai.operation.name", "execute_tool"),
                ("gen_ai.tool.name", "a"),
            ],
        );
        let s2 = tool_span(
            "t1",
            "s2",
            "tool2",
            3_000_000_000,
            4_000_000_000,
            &[
                ("gen_ai.operation.name", "execute_tool"),
                ("gen_ai.tool.name", "b"),
            ],
        );
        let input = make_otlp_json(&format!("[{s1},{s2}]"));
        let parser = OtelParser::new("default-agent");
        let events = parser.parse(&input).unwrap();

        assert_eq!(events.len(), 2);
        assert_eq!(events[0].sequence_position, 1);
        assert_eq!(events[1].sequence_position, 2);
    }

    #[test]
    fn multiple_traces() {
        let s1 = tool_span(
            "trace-a",
            "s1",
            "tool",
            1_000_000_000,
            2_000_000_000,
            &[
                ("gen_ai.operation.name", "execute_tool"),
                ("gen_ai.tool.name", "a"),
            ],
        );
        let s2 = tool_span(
            "trace-b",
            "s2",
            "tool",
            3_000_000_000,
            4_000_000_000,
            &[
                ("gen_ai.operation.name", "execute_tool"),
                ("gen_ai.tool.name", "b"),
            ],
        );
        let input = make_otlp_json(&format!("[{s1},{s2}]"));
        let parser = OtelParser::new("default-agent");
        let events = parser.parse(&input).unwrap();

        assert_eq!(events.len(), 2);
        // Each trace gets its own sequence counter starting at 1.
        assert!(events.iter().all(|e| e.sequence_position == 1));
    }

    #[test]
    fn session_id_fallback_to_trace_id() {
        let span = tool_span(
            "my-trace-id",
            "s1",
            "tool",
            1_000_000_000,
            2_000_000_000,
            &[
                ("gen_ai.operation.name", "execute_tool"),
                ("gen_ai.tool.name", "a"),
            ],
        );
        let input = make_otlp_json(&format!("[{span}]"));
        let parser = OtelParser::new("default-agent");
        let events = parser.parse(&input).unwrap();

        assert_eq!(events[0].session_id, "my-trace-id");
    }

    #[test]
    fn error_span() {
        let span = tool_span_with_status(
            "t1",
            "s1",
            "execute_tool",
            1_000_000_000,
            2_000_000_000,
            &[
                ("gen_ai.operation.name", "execute_tool"),
                ("gen_ai.tool.name", "dangerous_tool"),
            ],
            2, // ERROR status
        );
        let input = make_otlp_json(&format!("[{span}]"));
        let parser = OtelParser::new("default-agent");
        let events = parser.parse(&input).unwrap();

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, EventType::Error);
    }

    #[test]
    fn invoke_agent_skipped() {
        let span = tool_span(
            "t1",
            "s1",
            "invoke",
            1_000_000_000,
            2_000_000_000,
            &[("gen_ai.operation.name", "invoke_agent")],
        );
        let input = make_otlp_json(&format!("[{span}]"));
        let parser = OtelParser::new("default-agent");
        let events = parser.parse(&input).unwrap();

        assert!(events.is_empty());
    }

    #[test]
    fn gen_ai_preferred_over_openinference() {
        // Both namespaces present — gen_ai should win.
        let span = tool_span(
            "t1",
            "s1",
            "tool",
            1_000_000_000,
            2_000_000_000,
            &[
                ("gen_ai.operation.name", "execute_tool"),
                ("gen_ai.tool.name", "read"),
                ("openinference.span.kind", "LLM"),
            ],
        );
        let input = make_otlp_json(&format!("[{span}]"));
        let parser = OtelParser::new("default-agent");
        let events = parser.parse(&input).unwrap();

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, EventType::ToolCall);
    }

    #[test]
    fn duration_ms_calculated() {
        // 1.5 seconds = 1500ms
        let span = tool_span(
            "t1",
            "s1",
            "tool",
            1_000_000_000,
            2_500_000_000,
            &[
                ("gen_ai.operation.name", "execute_tool"),
                ("gen_ai.tool.name", "a"),
            ],
        );
        let input = make_otlp_json(&format!("[{span}]"));
        let parser = OtelParser::new("default-agent");
        let events = parser.parse(&input).unwrap();

        assert_eq!(events[0].duration_ms, 1500);
    }

    #[test]
    fn trace_id_and_span_id_populated() {
        let span = tool_span(
            "my-trace-123",
            "my-span-456",
            "tool",
            1_000_000_000,
            2_000_000_000,
            &[
                ("gen_ai.operation.name", "execute_tool"),
                ("gen_ai.tool.name", "a"),
            ],
        );
        let input = make_otlp_json(&format!("[{span}]"));
        let parser = OtelParser::new("default-agent");
        let events = parser.parse(&input).unwrap();

        assert_eq!(events[0].trace_id.as_deref(), Some("my-trace-123"));
        assert_eq!(events[0].span_id.as_deref(), Some("my-span-456"));
    }

    #[test]
    fn empty_resource_spans() {
        let input = r#"{"resourceSpans":[]}"#;
        let parser = OtelParser::new("default-agent");
        let events = parser.parse(input).unwrap();
        assert!(events.is_empty());
    }

    #[test]
    fn provider_and_model_populated() {
        let span = tool_span(
            "t1",
            "s1",
            "chat",
            1_000_000_000,
            2_000_000_000,
            &[
                ("gen_ai.operation.name", "chat"),
                ("gen_ai.provider.name", "openai"),
                ("gen_ai.request.model", "gpt-4o"),
            ],
        );
        let input = make_otlp_json(&format!("[{span}]"));
        let parser = OtelParser::new("default-agent");
        let events = parser.parse(&input).unwrap();

        assert_eq!(events[0].provider.as_deref(), Some("openai"));
        assert_eq!(events[0].model.as_deref(), Some("gpt-4o"));
    }

    #[test]
    fn agent_id_from_attributes() {
        let span = tool_span(
            "t1",
            "s1",
            "tool",
            1_000_000_000,
            2_000_000_000,
            &[
                ("gen_ai.operation.name", "execute_tool"),
                ("gen_ai.tool.name", "a"),
                ("gen_ai.agent.name", "my-custom-agent"),
            ],
        );
        let input = make_otlp_json(&format!("[{span}]"));
        let parser = OtelParser::new("default-agent");
        let events = parser.parse(&input).unwrap();

        assert_eq!(events[0].agent_id, "my-custom-agent");
    }

    #[test]
    fn session_id_from_conversation_id() {
        let span = tool_span(
            "t1",
            "s1",
            "tool",
            1_000_000_000,
            2_000_000_000,
            &[
                ("gen_ai.operation.name", "execute_tool"),
                ("gen_ai.tool.name", "a"),
                ("gen_ai.conversation.id", "conv-789"),
            ],
        );
        let input = make_otlp_json(&format!("[{span}]"));
        let parser = OtelParser::new("default-agent");
        let events = parser.parse(&input).unwrap();

        assert_eq!(events[0].session_id, "conv-789");
    }
}
