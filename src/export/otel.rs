use serde::Serialize;

use crate::detect::{Anomaly, AnomalyType};

// ---------------------------------------------------------------------------
// OTLP JSON serialization types
// ---------------------------------------------------------------------------

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ExportTraceServiceRequest {
    resource_spans: Vec<ResourceSpans>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ResourceSpans {
    scope_spans: Vec<ScopeSpans>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ScopeSpans {
    spans: Vec<OtlpSpan>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct OtlpSpan {
    trace_id: String,
    span_id: String,
    name: String,
    start_time_unix_nano: String,
    end_time_unix_nano: String,
    attributes: Vec<KeyValue>,
}

#[derive(Serialize)]
struct KeyValue {
    key: String,
    value: AnyValue,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct AnyValue {
    string_value: String,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Convert anomalies to OTLP JSON (ExportTraceServiceRequest).
pub fn anomalies_to_otlp(anomalies: &[Anomaly]) -> String {
    let spans: Vec<OtlpSpan> = anomalies.iter().map(anomaly_to_span).collect();

    let request = ExportTraceServiceRequest {
        resource_spans: vec![ResourceSpans {
            scope_spans: vec![ScopeSpans { spans }],
        }],
    };

    serde_json::to_string(&request).unwrap_or_else(|_| r#"{"resourceSpans":[]}"#.to_string())
}

fn anomaly_to_span(anomaly: &Anomaly) -> OtlpSpan {
    let timestamp_nanos = anomaly.timestamp.timestamp_nanos_opt().unwrap_or(0);
    let ts_str = timestamp_nanos.to_string();

    // trace_id: from evidence event's trace_id if available, otherwise generate from timestamp
    let trace_id = anomaly
        .evidence
        .event
        .trace_id
        .clone()
        .unwrap_or_else(|| format!("{:032x}", timestamp_nanos as u128));

    // span_id: generate from timestamp + anomaly type
    let span_id = format!(
        "{:016x}",
        (timestamp_nanos as u64).wrapping_add(anomaly_type_hash(&anomaly.anomaly_type))
    );

    let mut attributes = vec![
        kv("vigil.anomaly.type", &format!("{:?}", anomaly.anomaly_type)),
        kv("vigil.severity", &format!("{:?}", anomaly.severity)),
        kv("vigil.description", &anomaly.description),
        kv(
            "vigil.confidence",
            &format!("{}", anomaly.evidence.confidence),
        ),
        kv("vigil.baseline_value", &anomaly.evidence.baseline_value),
        kv("vigil.observed_value", &anomaly.evidence.observed_value),
        kv("vigil.agent_id", &anomaly.evidence.event.agent_id),
    ];

    if let Some(z) = anomaly.evidence.z_score {
        attributes.push(kv("vigil.z_score", &format!("{z}")));
    }

    if let Some(ref tool) = anomaly.evidence.event.tool_name {
        attributes.push(kv("vigil.tool_name", tool));
    }

    OtlpSpan {
        trace_id,
        span_id,
        name: format!("vigil.anomaly {:?}", anomaly.anomaly_type),
        start_time_unix_nano: ts_str.clone(),
        end_time_unix_nano: ts_str,
        attributes,
    }
}

fn kv(key: &str, value: &str) -> KeyValue {
    KeyValue {
        key: key.to_string(),
        value: AnyValue {
            string_value: value.to_string(),
        },
    }
}

fn anomaly_type_hash(t: &AnomalyType) -> u64 {
    match t {
        AnomalyType::UnknownTool => 1,
        AnomalyType::UnknownSequence => 2,
        AnomalyType::UnknownResource => 3,
        AnomalyType::VolumeSpike => 4,
        AnomalyType::RateSpike => 5,
        AnomalyType::TemporalAnomaly => 6,
        AnomalyType::UnknownParamPattern => 7,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detect::{Evidence, Severity};
    use crate::event::{BehavioralEvent, EventType};
    use chrono::{TimeZone, Utc};

    fn make_test_anomaly(anomaly_type: AnomalyType, severity: Severity) -> Anomaly {
        Anomaly {
            timestamp: Utc.with_ymd_and_hms(2026, 2, 20, 10, 0, 0).unwrap(),
            severity,
            anomaly_type,
            description: "test anomaly".to_string(),
            evidence: Evidence {
                event: BehavioralEvent {
                    timestamp: Utc.with_ymd_and_hms(2026, 2, 20, 10, 0, 0).unwrap(),
                    session_id: "s1".to_string(),
                    agent_id: "test-agent".to_string(),
                    event_type: EventType::ToolCall,
                    tool_name: Some("evil_tool".to_string()),
                    param_keys: vec![],
                    resource_ids: vec![],
                    data_in_bytes: 0,
                    data_out_bytes: 0,
                    duration_ms: 0,
                    token_count: None,
                    sequence_position: 1,
                    trace_id: Some("trace-abc".to_string()),
                    span_id: Some("span-def".to_string()),
                    provider: None,
                    model: None,
                },
                baseline_value: "known tools: [read, write]".to_string(),
                observed_value: "evil_tool".to_string(),
                z_score: None,
                confidence: 0.95,
            },
        }
    }

    #[test]
    fn otlp_output_is_valid_json() {
        let anomalies = vec![make_test_anomaly(AnomalyType::UnknownTool, Severity::High)];
        let output = anomalies_to_otlp(&anomalies);
        let parsed: serde_json::Value = serde_json::from_str(&output).unwrap();
        assert!(parsed.get("resourceSpans").is_some());
    }

    #[test]
    fn otlp_output_contains_anomaly_attributes() {
        let anomalies = vec![make_test_anomaly(AnomalyType::UnknownTool, Severity::High)];
        let output = anomalies_to_otlp(&anomalies);
        assert!(output.contains("vigil.anomaly.type"));
        assert!(output.contains("UnknownTool"));
        assert!(output.contains("vigil.severity"));
        assert!(output.contains("High"));
        assert!(output.contains("vigil.agent_id"));
        assert!(output.contains("test-agent"));
        assert!(output.contains("vigil.tool_name"));
        assert!(output.contains("evil_tool"));
    }

    #[test]
    fn otlp_output_uses_event_trace_id() {
        let anomalies = vec![make_test_anomaly(AnomalyType::UnknownTool, Severity::High)];
        let output = anomalies_to_otlp(&anomalies);
        assert!(output.contains("trace-abc"));
    }

    #[test]
    fn otlp_output_span_name() {
        let anomalies = vec![make_test_anomaly(AnomalyType::RateSpike, Severity::Medium)];
        let output = anomalies_to_otlp(&anomalies);
        assert!(output.contains("vigil.anomaly RateSpike"));
    }

    #[test]
    fn otlp_output_empty_anomalies() {
        let output = anomalies_to_otlp(&[]);
        let parsed: serde_json::Value = serde_json::from_str(&output).unwrap();
        let spans = &parsed["resourceSpans"][0]["scopeSpans"][0]["spans"];
        assert!(spans.as_array().unwrap().is_empty());
    }

    #[test]
    fn otlp_output_z_score_included_when_present() {
        let mut anomaly = make_test_anomaly(AnomalyType::RateSpike, Severity::High);
        anomaly.evidence.z_score = Some(4.5);
        let output = anomalies_to_otlp(&[anomaly]);
        assert!(output.contains("vigil.z_score"));
        assert!(output.contains("4.5"));
    }

    #[test]
    fn otlp_output_generates_trace_id_when_missing() {
        let mut anomaly = make_test_anomaly(AnomalyType::UnknownTool, Severity::High);
        anomaly.evidence.event.trace_id = None;
        let output = anomalies_to_otlp(&[anomaly]);
        let parsed: serde_json::Value = serde_json::from_str(&output).unwrap();
        let trace_id = parsed["resourceSpans"][0]["scopeSpans"][0]["spans"][0]["traceId"]
            .as_str()
            .unwrap();
        // Should be a hex string, not empty.
        assert!(!trace_id.is_empty());
        assert!(trace_id.chars().all(|c| c.is_ascii_hexdigit()));
    }
}
