use std::sync::Mutex;

use chrono::{DateTime, Utc};

use crate::baseline::{Baseline, Bigram};
use crate::event::{BehavioralEvent, EventType};

use super::{Anomaly, AnomalyType, Detector, Evidence, Severity};

fn confidence(baseline: &Baseline) -> f64 {
    (baseline.event_count as f64 / 100.0).min(1.0)
}

/// Flags tool calls with tool names not seen in the baseline.
pub struct UnknownToolDetector;

impl Detector for UnknownToolDetector {
    fn detect(&self, event: &BehavioralEvent, baseline: &Baseline) -> Vec<Anomaly> {
        if event.event_type != EventType::ToolCall {
            return vec![];
        }
        let tool_name = match &event.tool_name {
            Some(name) => name,
            None => return vec![],
        };
        if baseline.tool_stats.contains_key(tool_name) {
            return vec![];
        }

        vec![Anomaly {
            timestamp: event.timestamp,
            severity: Severity::High,
            anomaly_type: AnomalyType::UnknownTool,
            description: format!("unknown tool: {tool_name}"),
            evidence: Evidence {
                event: event.clone(),
                baseline_value: format!(
                    "known tools: [{}]",
                    baseline
                        .tool_stats
                        .keys()
                        .cloned()
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
                observed_value: tool_name.clone(),
                z_score: None,
                confidence: confidence(baseline),
            },
        }]
    }
}

/// Flags tool call bigrams not seen in the baseline.
pub struct UnknownSequenceDetector {
    state: Mutex<SequenceState>,
}

struct SequenceState {
    last_tool_call: Option<String>,
    last_session_id: Option<String>,
}

impl Default for UnknownSequenceDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl UnknownSequenceDetector {
    pub fn new() -> Self {
        Self {
            state: Mutex::new(SequenceState {
                last_tool_call: None,
                last_session_id: None,
            }),
        }
    }
}

impl Detector for UnknownSequenceDetector {
    fn detect(&self, event: &BehavioralEvent, baseline: &Baseline) -> Vec<Anomaly> {
        if event.event_type != EventType::ToolCall {
            return vec![];
        }
        let tool_name = match &event.tool_name {
            Some(name) => name,
            None => return vec![],
        };

        let mut state = self.state.lock().unwrap();

        // Reset on session boundary.
        if state.last_session_id.as_ref() != Some(&event.session_id) {
            state.last_tool_call = None;
            state.last_session_id = Some(event.session_id.clone());
        }

        let result = if let Some(ref prev) = state.last_tool_call {
            let bigram = Bigram::new(prev.clone(), tool_name.clone());
            if baseline.bigrams.contains_key(&bigram) {
                vec![]
            } else {
                vec![Anomaly {
                    timestamp: event.timestamp,
                    severity: Severity::Medium,
                    anomaly_type: AnomalyType::UnknownSequence,
                    description: format!("unseen sequence: {bigram}"),
                    evidence: Evidence {
                        event: event.clone(),
                        baseline_value: format!("{} known bigrams", baseline.bigrams.len()),
                        observed_value: bigram.to_string(),
                        z_score: None,
                        confidence: confidence(baseline),
                    },
                }]
            }
        } else {
            vec![]
        };

        state.last_tool_call = Some(tool_name.clone());
        result
    }
}

/// Flags tool call rates that exceed a z-score threshold.
pub struct RateSpikeDetector {
    threshold: f64,
    state: Mutex<RateState>,
}

struct RateState {
    last_tool_call_time: Option<DateTime<Utc>>,
    last_session_id: Option<String>,
}

impl RateSpikeDetector {
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold,
            state: Mutex::new(RateState {
                last_tool_call_time: None,
                last_session_id: None,
            }),
        }
    }
}

impl Detector for RateSpikeDetector {
    fn detect(&self, event: &BehavioralEvent, baseline: &Baseline) -> Vec<Anomaly> {
        if event.event_type != EventType::ToolCall {
            return vec![];
        }

        let mut state = self.state.lock().unwrap();

        // Reset on session boundary.
        if state.last_session_id.as_ref() != Some(&event.session_id) {
            state.last_tool_call_time = None;
            state.last_session_id = Some(event.session_id.clone());
        }

        let result = if let Some(prev_time) = state.last_tool_call_time {
            let interval_minutes = event
                .timestamp
                .signed_duration_since(prev_time)
                .num_milliseconds() as f64
                / 60_000.0;

            if interval_minutes <= 0.0 {
                vec![]
            } else {
                let rate = 1.0 / interval_minutes;
                match baseline.rate_stats.z_score(rate) {
                    Some(z) if z > self.threshold => {
                        let severity = if z > self.threshold + 2.0 {
                            Severity::High
                        } else {
                            Severity::Medium
                        };
                        vec![Anomaly {
                            timestamp: event.timestamp,
                            severity,
                            anomaly_type: AnomalyType::RateSpike,
                            description: format!("rate spike: {rate:.1} calls/min (z={z:.1})"),
                            evidence: Evidence {
                                event: event.clone(),
                                baseline_value: format!(
                                    "{:.1} calls/min (mean)",
                                    baseline.rate_stats.mean
                                ),
                                observed_value: format!("{rate:.1} calls/min"),
                                z_score: Some(z),
                                confidence: confidence(baseline),
                            },
                        }]
                    }
                    _ => vec![],
                }
            }
        } else {
            vec![]
        };

        state.last_tool_call_time = Some(event.timestamp);
        result
    }
}

/// Flags data volumes that exceed a z-score threshold for a given tool.
pub struct VolumeSpikeDetector {
    threshold: f64,
}

impl VolumeSpikeDetector {
    pub fn new(threshold: f64) -> Self {
        Self { threshold }
    }
}

impl Detector for VolumeSpikeDetector {
    fn detect(&self, event: &BehavioralEvent, baseline: &Baseline) -> Vec<Anomaly> {
        if event.event_type != EventType::ToolCall {
            return vec![];
        }
        let tool_name = match &event.tool_name {
            Some(name) => name,
            None => return vec![],
        };
        let stats = match baseline.volume_stats.get(tool_name) {
            Some(s) => s,
            None => return vec![],
        };

        let volume = (event.data_in_bytes + event.data_out_bytes) as f64;
        match stats.z_score(volume) {
            Some(z) if z > self.threshold => {
                let severity = if z > self.threshold + 2.0 {
                    Severity::High
                } else {
                    Severity::Medium
                };
                vec![Anomaly {
                    timestamp: event.timestamp,
                    severity,
                    anomaly_type: AnomalyType::VolumeSpike,
                    description: format!(
                        "volume spike on {tool_name}: {volume:.0} bytes (z={z:.1})"
                    ),
                    evidence: Evidence {
                        event: event.clone(),
                        baseline_value: format!("{:.0} bytes (mean)", stats.mean),
                        observed_value: format!("{volume:.0} bytes"),
                        z_score: Some(z),
                        confidence: confidence(baseline),
                    },
                }]
            }
            _ => vec![],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::baseline::StreamingStats;
    use chrono::TimeZone;

    fn ts(hour: u32, minute: u32, second: u32) -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2025, 1, 15, hour, minute, second)
            .unwrap()
    }

    fn make_event(tool: &str, session_id: &str, timestamp: DateTime<Utc>) -> BehavioralEvent {
        BehavioralEvent {
            timestamp,
            session_id: session_id.to_string(),
            agent_id: "test-agent".to_string(),
            event_type: EventType::ToolCall,
            tool_name: Some(tool.to_string()),
            param_keys: vec![],
            resource_ids: vec![],
            data_in_bytes: 0,
            data_out_bytes: 0,
            duration_ms: 0,
            token_count: None,
            sequence_position: 1,
        }
    }

    fn baseline_with_tools(tools: &[&str]) -> Baseline {
        let mut b = Baseline::new("test-agent");
        b.event_count = 200; // high confidence
        for tool in tools {
            b.tool_stats.insert(
                tool.to_string(),
                crate::baseline::ToolStats {
                    call_count: 10,
                    first_seen: ts(10, 0, 0),
                    last_seen: ts(10, 0, 0),
                    param_key_sets: Default::default(),
                },
            );
        }
        b
    }

    #[test]
    fn unknown_tool_fires_for_novel_tool() {
        let detector = UnknownToolDetector;
        let baseline = baseline_with_tools(&["read", "write"]);
        let event = make_event("evil_exfiltrate", "s1", ts(10, 0, 0));

        let anomalies = detector.detect(&event, &baseline);
        assert_eq!(anomalies.len(), 1);
        assert_eq!(anomalies[0].anomaly_type, AnomalyType::UnknownTool);
        assert_eq!(anomalies[0].severity, Severity::High);
    }

    #[test]
    fn unknown_tool_does_not_fire_for_known_tool() {
        let detector = UnknownToolDetector;
        let baseline = baseline_with_tools(&["read", "write"]);
        let event = make_event("read", "s1", ts(10, 0, 0));

        let anomalies = detector.detect(&event, &baseline);
        assert!(anomalies.is_empty());
    }

    #[test]
    fn unknown_sequence_fires_for_unseen_bigram() {
        let detector = UnknownSequenceDetector::new();
        let mut baseline = baseline_with_tools(&["read", "write", "exec"]);
        baseline.bigrams.insert(Bigram::new("read", "write"), 5);

        // read → exec is not in bigrams.
        let e1 = make_event("read", "s1", ts(10, 0, 0));
        let e2 = make_event("exec", "s1", ts(10, 1, 0));

        detector.detect(&e1, &baseline);
        let anomalies = detector.detect(&e2, &baseline);
        assert_eq!(anomalies.len(), 1);
        assert_eq!(anomalies[0].anomaly_type, AnomalyType::UnknownSequence);
    }

    #[test]
    fn unknown_sequence_does_not_fire_for_known_bigram() {
        let detector = UnknownSequenceDetector::new();
        let mut baseline = baseline_with_tools(&["read", "write"]);
        baseline.bigrams.insert(Bigram::new("read", "write"), 5);

        let e1 = make_event("read", "s1", ts(10, 0, 0));
        let e2 = make_event("write", "s1", ts(10, 1, 0));

        detector.detect(&e1, &baseline);
        let anomalies = detector.detect(&e2, &baseline);
        assert!(anomalies.is_empty());
    }

    #[test]
    fn unknown_sequence_resets_across_sessions() {
        let detector = UnknownSequenceDetector::new();
        let mut baseline = baseline_with_tools(&["read", "write"]);
        baseline.bigrams.insert(Bigram::new("read", "write"), 5);

        let e1 = make_event("read", "s1", ts(10, 0, 0));
        // New session — should not form bigram with e1.
        let e2 = make_event("write", "s2", ts(11, 0, 0));

        detector.detect(&e1, &baseline);
        let anomalies = detector.detect(&e2, &baseline);
        assert!(anomalies.is_empty()); // no bigram formed, so no anomaly
    }

    #[test]
    fn rate_spike_fires_for_extreme_rate() {
        let detector = RateSpikeDetector::new(3.0);

        // Baseline: mean rate of 1.0 calls/min, std_dev ~0.1
        let mut baseline = baseline_with_tools(&["read", "write"]);
        baseline.rate_stats = StreamingStats::new();
        for _ in 0..50 {
            baseline.rate_stats.update(1.0);
        }
        // std_dev is 0 with identical values, so add slight variance.
        for _ in 0..50 {
            baseline.rate_stats.update(1.1);
        }

        // Two calls 1 second apart = 60 calls/min — massive spike.
        let e1 = make_event("read", "s1", ts(10, 0, 0));
        let e2 = make_event("write", "s1", ts(10, 0, 1));

        detector.detect(&e1, &baseline);
        let anomalies = detector.detect(&e2, &baseline);
        assert_eq!(anomalies.len(), 1);
        assert_eq!(anomalies[0].anomaly_type, AnomalyType::RateSpike);
    }

    #[test]
    fn rate_spike_does_not_fire_for_normal_rate() {
        let detector = RateSpikeDetector::new(3.0);

        let mut baseline = baseline_with_tools(&["read", "write"]);
        baseline.rate_stats = StreamingStats::new();
        for _ in 0..50 {
            baseline.rate_stats.update(1.0);
        }
        for _ in 0..50 {
            baseline.rate_stats.update(1.1);
        }

        // Two calls 1 minute apart = ~1.0 calls/min — normal.
        let e1 = make_event("read", "s1", ts(10, 0, 0));
        let e2 = make_event("write", "s1", ts(10, 1, 0));

        detector.detect(&e1, &baseline);
        let anomalies = detector.detect(&e2, &baseline);
        assert!(anomalies.is_empty());
    }

    #[test]
    fn unknown_tool_ignores_non_toolcall() {
        let detector = UnknownToolDetector;
        let baseline = baseline_with_tools(&["read", "write"]);

        let mut event = make_event("read", "s1", ts(10, 0, 0));
        event.event_type = EventType::UserMessage;

        let anomalies = detector.detect(&event, &baseline);
        assert!(
            anomalies.is_empty(),
            "UserMessage should not trigger UnknownTool"
        );
    }

    #[test]
    fn volume_spike_no_history() {
        let detector = VolumeSpikeDetector::new(3.0);
        // Baseline has tool_stats for "read" but no volume_stats entry.
        let baseline = baseline_with_tools(&["read"]);

        let mut event = make_event("read", "s1", ts(10, 0, 0));
        event.data_in_bytes = 100_000;

        let anomalies = detector.detect(&event, &baseline);
        assert!(
            anomalies.is_empty(),
            "no volume_stats should mean no anomaly"
        );
    }

    #[test]
    fn rate_spike_same_timestamp_skipped() {
        let detector = RateSpikeDetector::new(3.0);

        let mut baseline = baseline_with_tools(&["read"]);
        baseline.rate_stats = StreamingStats::new();
        for _ in 0..50 {
            baseline.rate_stats.update(1.0);
        }
        for _ in 0..50 {
            baseline.rate_stats.update(1.1);
        }

        // Two calls with identical timestamps → interval_minutes = 0 → skipped.
        let e1 = make_event("read", "s1", ts(10, 0, 0));
        let e2 = make_event("read", "s1", ts(10, 0, 0));

        detector.detect(&e1, &baseline);
        let anomalies = detector.detect(&e2, &baseline);
        assert!(
            anomalies.is_empty(),
            "identical timestamps should produce no rate anomaly"
        );
    }

    #[test]
    fn volume_spike_fires_for_extreme_volume() {
        let detector = VolumeSpikeDetector::new(3.0);

        let mut baseline = baseline_with_tools(&["read"]);
        let mut vol = StreamingStats::new();
        for _ in 0..100 {
            vol.update(500.0);
        }
        for _ in 0..100 {
            vol.update(550.0);
        }
        baseline.volume_stats.insert("read".to_string(), vol);

        let mut event = make_event("read", "s1", ts(10, 0, 0));
        event.data_in_bytes = 0;
        event.data_out_bytes = 100_000; // way above mean of ~525

        let anomalies = detector.detect(&event, &baseline);
        assert_eq!(anomalies.len(), 1);
        assert_eq!(anomalies[0].anomaly_type, AnomalyType::VolumeSpike);
    }
}
