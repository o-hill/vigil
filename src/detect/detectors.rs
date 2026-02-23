use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;

use chrono::{DateTime, Timelike, Utc};

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

/// A tool call signature: (tool_name, param_keys).
type ToolSignature = (String, Vec<String>);

/// Detects repeated identical tool calls (same tool + same param_keys)
/// within a session, indicating the agent is stuck in a reasoning loop.
pub struct ReasoningLoopDetector {
    loop_threshold: usize,
    window_size: usize,
    state: Mutex<HashMap<String, VecDeque<ToolSignature>>>,
}

impl ReasoningLoopDetector {
    pub fn new(loop_threshold: usize, window_size: usize) -> Self {
        Self {
            loop_threshold,
            window_size,
            state: Mutex::new(HashMap::new()),
        }
    }
}

impl Detector for ReasoningLoopDetector {
    fn detect(&self, event: &BehavioralEvent, baseline: &Baseline) -> Vec<Anomaly> {
        if event.event_type != EventType::ToolCall {
            return vec![];
        }
        let tool_name = match &event.tool_name {
            Some(name) => name,
            None => return vec![],
        };

        let mut state = self.state.lock().unwrap();
        let history = state.entry(event.session_id.clone()).or_default();
        let current = (tool_name.clone(), event.param_keys.clone());

        history.push_back(current.clone());
        if history.len() > self.window_size {
            history.pop_front();
        }

        let consecutive = history.iter().rev().take_while(|c| **c == current).count();
        if consecutive >= self.loop_threshold {
            let severity = if consecutive >= 5 {
                Severity::High
            } else {
                Severity::Medium
            };
            vec![Anomaly {
                timestamp: event.timestamp,
                severity,
                anomaly_type: AnomalyType::ReasoningLoop,
                description: format!(
                    "reasoning loop: {tool_name} called {consecutive} times consecutively"
                ),
                evidence: Evidence {
                    event: event.clone(),
                    baseline_value: "no repeated calls expected".to_string(),
                    observed_value: format!("{consecutive} consecutive identical calls"),
                    z_score: None,
                    confidence: confidence(baseline),
                },
            }]
        } else {
            vec![]
        }
    }
}

/// Detects repeated ToolCall → Error cycles on the same tool,
/// indicating the agent is retrying a failing operation.
pub struct ErrorRetryDetector {
    cycle_threshold: u32,
    state: Mutex<HashMap<String, ErrorRetryState>>,
}

struct ErrorRetryState {
    last_tool_call: Option<(String, Vec<String>)>,
    consecutive_retries: u32,
    awaiting_retry: bool,
}

impl ErrorRetryDetector {
    pub fn new(cycle_threshold: u32) -> Self {
        Self {
            cycle_threshold,
            state: Mutex::new(HashMap::new()),
        }
    }
}

impl Detector for ErrorRetryDetector {
    fn detect(&self, event: &BehavioralEvent, baseline: &Baseline) -> Vec<Anomaly> {
        let mut state = self.state.lock().unwrap();
        let session = state
            .entry(event.session_id.clone())
            .or_insert(ErrorRetryState {
                last_tool_call: None,
                consecutive_retries: 0,
                awaiting_retry: false,
            });

        match event.event_type {
            EventType::ToolCall => {
                let current = (
                    event.tool_name.clone().unwrap_or_default(),
                    event.param_keys.clone(),
                );

                if session.awaiting_retry {
                    // After an error, we got another tool call.
                    if session.last_tool_call.as_ref() == Some(&current) {
                        // Same tool+params retried after error.
                        session.awaiting_retry = false;
                        // The retry count increments when we see the next Error.
                    } else {
                        // Different tool — reset.
                        session.last_tool_call = Some(current);
                        session.consecutive_retries = 0;
                        session.awaiting_retry = false;
                    }
                } else {
                    // Fresh tool call or first after success.
                    if session.last_tool_call.as_ref() != Some(&current) {
                        session.consecutive_retries = 0;
                    }
                    session.last_tool_call = Some(current);
                    session.awaiting_retry = false;
                }
                vec![]
            }
            EventType::Error => {
                if session.last_tool_call.is_some() {
                    session.consecutive_retries += 1;
                    session.awaiting_retry = true;

                    if session.consecutive_retries >= self.cycle_threshold {
                        let tool = session
                            .last_tool_call
                            .as_ref()
                            .map(|(t, _)| t.as_str())
                            .unwrap_or("unknown");
                        let retries = session.consecutive_retries;
                        return vec![Anomaly {
                            timestamp: event.timestamp,
                            severity: Severity::Medium,
                            anomaly_type: AnomalyType::ErrorRetryLoop,
                            description: format!(
                                "error-retry loop: {tool} failed {retries} times consecutively"
                            ),
                            evidence: Evidence {
                                event: event.clone(),
                                baseline_value: "successful tool execution expected".to_string(),
                                observed_value: format!("{retries} consecutive failures"),
                                z_score: None,
                                confidence: confidence(baseline),
                            },
                        }];
                    }
                }
                vec![]
            }
            // ToolResult (success) or other events reset the retry state.
            _ => {
                if let Some(s) = state.get_mut(&event.session_id) {
                    s.consecutive_retries = 0;
                    s.awaiting_retry = false;
                }
                vec![]
            }
        }
    }
}

/// GUARDIAN-inspired combined anomaly scoring.
/// Aggregates bigram transition surprise, volume z-score, and temporal
/// surprise into a single score via negative log probability.
pub struct CombinedAnomalyDetector {
    threshold: f64,
    state: Mutex<HashMap<String, String>>, // session_id → last tool name
}

impl CombinedAnomalyDetector {
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold,
            state: Mutex::new(HashMap::new()),
        }
    }
}

impl Detector for CombinedAnomalyDetector {
    fn detect(&self, event: &BehavioralEvent, baseline: &Baseline) -> Vec<Anomaly> {
        if event.event_type != EventType::ToolCall {
            return vec![];
        }
        let tool_name = match &event.tool_name {
            Some(name) => name,
            None => return vec![],
        };

        let mut scores: Vec<f64> = Vec::new();

        // 1. Bigram transition surprise.
        let mut state = self.state.lock().unwrap();
        let prev_tool = state.get(&event.session_id).cloned();
        state.insert(event.session_id.clone(), tool_name.clone());
        drop(state);

        if let Some(prev) = prev_tool {
            let bigram = Bigram::new(&prev, tool_name);
            let total: u64 = baseline.bigrams.values().sum();
            if total > 0 {
                let count = baseline.bigrams.get(&bigram).copied().unwrap_or(0);
                if count > 0 {
                    let p = count as f64 / total as f64;
                    scores.push(-p.ln());
                } else {
                    // Unseen bigram — max surprise, use Laplace smoothing.
                    let p = 1.0 / (total + baseline.bigrams.len() as u64 + 1) as f64;
                    scores.push(-p.ln());
                }
            }
        }

        // 2. Volume surprise (z-score).
        if let Some(vol_stats) = baseline.volume_stats.get(tool_name) {
            let volume = (event.data_in_bytes + event.data_out_bytes) as f64;
            if let Some(z) = vol_stats.z_score(volume) {
                scores.push(z.abs());
            }
        }

        // 3. Temporal surprise (hourly distribution).
        let hour = event.timestamp.hour() as usize;
        let total_hourly: u64 = baseline.hourly_distribution.iter().sum();
        if total_hourly > 0 {
            let count = baseline.hourly_distribution[hour];
            if count > 0 {
                let p = count as f64 / total_hourly as f64;
                scores.push(-p.ln());
            } else {
                // Never-seen hour — high surprise.
                let p = 1.0 / (total_hourly + 24) as f64;
                scores.push(-p.ln());
            }
        }

        if scores.is_empty() {
            return vec![];
        }

        let combined = scores.iter().sum::<f64>() / scores.len() as f64;
        if combined > self.threshold {
            let severity = if combined > self.threshold + 3.0 {
                Severity::High
            } else {
                Severity::Medium
            };
            vec![Anomaly {
                timestamp: event.timestamp,
                severity,
                anomaly_type: AnomalyType::CombinedAnomaly,
                description: format!(
                    "combined anomaly score {combined:.2} (threshold {:.1})",
                    self.threshold
                ),
                evidence: Evidence {
                    event: event.clone(),
                    baseline_value: format!("threshold: {:.1}", self.threshold),
                    observed_value: format!("combined={combined:.2} from {} signals", scores.len()),
                    z_score: Some(combined),
                    confidence: confidence(baseline),
                },
            }]
        } else {
            vec![]
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

    // --- ReasoningLoopDetector tests ---

    #[test]
    fn reasoning_loop_fires_after_three_identical_calls() {
        let detector = ReasoningLoopDetector::new(3, 10);
        let baseline = baseline_with_tools(&["read"]);

        let e1 = make_event("read", "s1", ts(10, 0, 0));
        let e2 = make_event("read", "s1", ts(10, 0, 1));
        let e3 = make_event("read", "s1", ts(10, 0, 2));

        assert!(detector.detect(&e1, &baseline).is_empty());
        assert!(detector.detect(&e2, &baseline).is_empty());
        let anomalies = detector.detect(&e3, &baseline);
        assert_eq!(anomalies.len(), 1);
        assert_eq!(anomalies[0].anomaly_type, AnomalyType::ReasoningLoop);
        assert_eq!(anomalies[0].severity, Severity::Medium);
    }

    #[test]
    fn reasoning_loop_does_not_fire_for_two_identical() {
        let detector = ReasoningLoopDetector::new(3, 10);
        let baseline = baseline_with_tools(&["read"]);

        let e1 = make_event("read", "s1", ts(10, 0, 0));
        let e2 = make_event("read", "s1", ts(10, 0, 1));

        assert!(detector.detect(&e1, &baseline).is_empty());
        assert!(detector.detect(&e2, &baseline).is_empty());
    }

    #[test]
    fn reasoning_loop_does_not_fire_for_different_tools() {
        let detector = ReasoningLoopDetector::new(3, 10);
        let baseline = baseline_with_tools(&["read", "write", "exec"]);

        let e1 = make_event("read", "s1", ts(10, 0, 0));
        let e2 = make_event("write", "s1", ts(10, 0, 1));
        let e3 = make_event("exec", "s1", ts(10, 0, 2));

        assert!(detector.detect(&e1, &baseline).is_empty());
        assert!(detector.detect(&e2, &baseline).is_empty());
        assert!(detector.detect(&e3, &baseline).is_empty());
    }

    #[test]
    fn reasoning_loop_resets_on_session_boundary() {
        let detector = ReasoningLoopDetector::new(3, 10);
        let baseline = baseline_with_tools(&["read"]);

        // Two calls in session s1.
        let e1 = make_event("read", "s1", ts(10, 0, 0));
        let e2 = make_event("read", "s1", ts(10, 0, 1));
        // One call in session s2 — should NOT combine with s1.
        let e3 = make_event("read", "s2", ts(10, 0, 2));

        assert!(detector.detect(&e1, &baseline).is_empty());
        assert!(detector.detect(&e2, &baseline).is_empty());
        assert!(detector.detect(&e3, &baseline).is_empty());
    }

    #[test]
    fn reasoning_loop_considers_param_keys() {
        let detector = ReasoningLoopDetector::new(3, 10);
        let baseline = baseline_with_tools(&["read"]);

        let mut e1 = make_event("read", "s1", ts(10, 0, 0));
        e1.param_keys = vec!["path".to_string()];
        let mut e2 = make_event("read", "s1", ts(10, 0, 1));
        e2.param_keys = vec!["path".to_string()];
        // Different params — breaks the chain.
        let mut e3 = make_event("read", "s1", ts(10, 0, 2));
        e3.param_keys = vec!["url".to_string()];

        assert!(detector.detect(&e1, &baseline).is_empty());
        assert!(detector.detect(&e2, &baseline).is_empty());
        assert!(detector.detect(&e3, &baseline).is_empty());
    }

    #[test]
    fn reasoning_loop_high_severity_at_five() {
        let detector = ReasoningLoopDetector::new(3, 10);
        let baseline = baseline_with_tools(&["read"]);

        for i in 0..5 {
            let e = make_event("read", "s1", ts(10, 0, i));
            let anomalies = detector.detect(&e, &baseline);
            if i >= 4 {
                assert_eq!(anomalies[0].severity, Severity::High);
            }
        }
    }

    // --- ErrorRetryDetector tests ---

    #[test]
    fn error_retry_fires_after_two_cycles() {
        let detector = ErrorRetryDetector::new(2);
        let baseline = baseline_with_tools(&["read"]);

        // Cycle 1: ToolCall → Error
        let e1 = make_event("read", "s1", ts(10, 0, 0));
        let mut e2 = make_event("read", "s1", ts(10, 0, 1));
        e2.event_type = EventType::Error;

        // Cycle 2: same ToolCall → Error
        let e3 = make_event("read", "s1", ts(10, 0, 2));
        let mut e4 = make_event("read", "s1", ts(10, 0, 3));
        e4.event_type = EventType::Error;

        assert!(detector.detect(&e1, &baseline).is_empty());
        assert!(detector.detect(&e2, &baseline).is_empty()); // first error, count=1
        assert!(detector.detect(&e3, &baseline).is_empty()); // retry
        let anomalies = detector.detect(&e4, &baseline); // second error, count=2
        assert_eq!(anomalies.len(), 1);
        assert_eq!(anomalies[0].anomaly_type, AnomalyType::ErrorRetryLoop);
    }

    #[test]
    fn error_retry_does_not_fire_for_different_tools() {
        let detector = ErrorRetryDetector::new(2);
        let baseline = baseline_with_tools(&["read", "write"]);

        // ToolCall(read) → Error → ToolCall(write) → Error
        let e1 = make_event("read", "s1", ts(10, 0, 0));
        let mut e2 = make_event("read", "s1", ts(10, 0, 1));
        e2.event_type = EventType::Error;
        let e3 = make_event("write", "s1", ts(10, 0, 2)); // different tool
        let mut e4 = make_event("write", "s1", ts(10, 0, 3));
        e4.event_type = EventType::Error;

        assert!(detector.detect(&e1, &baseline).is_empty());
        assert!(detector.detect(&e2, &baseline).is_empty());
        assert!(detector.detect(&e3, &baseline).is_empty());
        // First error for "write", count resets to 1.
        assert!(detector.detect(&e4, &baseline).is_empty());
    }

    #[test]
    fn error_retry_resets_on_success() {
        let detector = ErrorRetryDetector::new(2);
        let baseline = baseline_with_tools(&["read"]);

        // ToolCall → Error → ToolResult (success!) → ToolCall → Error
        let e1 = make_event("read", "s1", ts(10, 0, 0));
        let mut e2 = make_event("read", "s1", ts(10, 0, 1));
        e2.event_type = EventType::Error;
        let mut e3 = make_event("read", "s1", ts(10, 0, 2));
        e3.event_type = EventType::ToolResult; // success resets
        let e4 = make_event("read", "s1", ts(10, 0, 3));
        let mut e5 = make_event("read", "s1", ts(10, 0, 4));
        e5.event_type = EventType::Error;

        assert!(detector.detect(&e1, &baseline).is_empty());
        assert!(detector.detect(&e2, &baseline).is_empty());
        assert!(detector.detect(&e3, &baseline).is_empty()); // resets
        assert!(detector.detect(&e4, &baseline).is_empty());
        assert!(detector.detect(&e5, &baseline).is_empty()); // count=1, not 2
    }

    // --- CombinedAnomalyDetector tests ---

    #[test]
    fn combined_anomaly_fires_for_high_surprise() {
        let detector = CombinedAnomalyDetector::new(3.0);

        let mut baseline = baseline_with_tools(&["read", "write"]);
        baseline.event_count = 200;
        // Set up hourly distribution heavily weighted to hour 14.
        baseline.hourly_distribution[14] = 100;

        // Event at hour 3 (never seen) with no bigram context.
        let event = make_event("read", "s1", ts(3, 0, 0));

        let anomalies = detector.detect(&event, &baseline);
        // Should fire because hour 3 has 0 events → high temporal surprise.
        assert_eq!(anomalies.len(), 1);
        assert_eq!(anomalies[0].anomaly_type, AnomalyType::CombinedAnomaly);
    }

    #[test]
    fn combined_anomaly_does_not_fire_for_normal_event() {
        let detector = CombinedAnomalyDetector::new(3.0);

        let mut baseline = baseline_with_tools(&["read"]);
        baseline.event_count = 200;
        // Concentrate distribution on hour 10 so surprise is low.
        baseline.hourly_distribution[10] = 500;
        for h in 0..24 {
            if h != 10 {
                baseline.hourly_distribution[h] = 20;
            }
        }

        // Event at hour 10 — very common hour, low surprise.
        let event = make_event("read", "s1", ts(10, 0, 0));

        let anomalies = detector.detect(&event, &baseline);
        assert!(anomalies.is_empty());
    }

    #[test]
    fn combined_anomaly_handles_empty_baseline() {
        let detector = CombinedAnomalyDetector::new(3.0);
        let baseline = Baseline::new("test-agent");

        let event = make_event("read", "s1", ts(10, 0, 0));
        let anomalies = detector.detect(&event, &baseline);
        // No hourly data, no bigrams, no volume — no signals → no anomaly.
        assert!(anomalies.is_empty());
    }
}
