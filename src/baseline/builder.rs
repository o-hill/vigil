use chrono::{DateTime, Timelike, Utc};

use crate::baseline::{Baseline, Bigram, ToolStats};
use crate::event::{BehavioralEvent, EventType};

/// Ingests `BehavioralEvent`s and builds a per-agent `Baseline`.
///
/// Tracks ephemeral session-local state (last tool call, session boundaries)
/// that doesn't belong in the persisted baseline.
pub struct BaselineBuilder {
    baseline: Baseline,
    current_session_id: Option<String>,
    last_tool_call: Option<String>,
    last_tool_call_time: Option<DateTime<Utc>>,
}

impl BaselineBuilder {
    /// Start building a fresh baseline for the given agent.
    pub fn new(agent_id: impl Into<String>) -> Self {
        Self {
            baseline: Baseline::new(agent_id),
            current_session_id: None,
            last_tool_call: None,
            last_tool_call_time: None,
        }
    }

    /// Resume building from an existing baseline.
    pub fn from_baseline(baseline: Baseline) -> Self {
        Self {
            baseline,
            current_session_id: None,
            last_tool_call: None,
            last_tool_call_time: None,
        }
    }

    /// Ingest a single event into the baseline.
    /// Returns `false` if the event was skipped as a duplicate.
    pub fn process(&mut self, event: &BehavioralEvent) -> bool {
        // 0. Dedup: skip events at or below the high-water mark for this session.
        if let Some(&hwm) = self.baseline.processed_through.get(&event.session_id)
            && event.sequence_position <= hwm
        {
            return false;
        }

        // 1. Handle session boundary.
        let session_changed = match &self.current_session_id {
            Some(current) => current != &event.session_id,
            None => true,
        };
        if session_changed {
            if self.current_session_id.is_some() {
                self.baseline.session_count += 1;
            }
            self.current_session_id = Some(event.session_id.clone());
            self.last_tool_call = None;
            self.last_tool_call_time = None;
        }

        // 2. Increment event_count, update first_seen/last_updated.
        self.baseline.event_count += 1;
        if self.baseline.event_count == 1 {
            self.baseline.first_seen = event.timestamp;
        }
        self.baseline.last_updated = event.timestamp;

        // 3. Update hourly distribution.
        let hour = event.timestamp.hour() as usize;
        self.baseline.hourly_distribution[hour] += 1;

        // 4. Add resource_ids to known_resources.
        for resource in &event.resource_ids {
            self.baseline.known_resources.insert(resource.clone());
        }

        // 5. ToolCall-specific processing.
        if event.event_type == EventType::ToolCall {
            let tool_name = event.tool_name.as_deref().unwrap_or("unknown");

            // Update tool_stats.
            let stats = self
                .baseline
                .tool_stats
                .entry(tool_name.to_string())
                .or_insert_with(|| ToolStats {
                    call_count: 0,
                    first_seen: event.timestamp,
                    last_seen: event.timestamp,
                    param_key_sets: Default::default(),
                });
            stats.call_count += 1;
            stats.last_seen = event.timestamp;

            let mut sorted_keys = event.param_keys.clone();
            sorted_keys.sort();
            stats.param_key_sets.insert(sorted_keys);

            // Record bigram if there was a previous tool call in this session.
            if let Some(ref prev) = self.last_tool_call {
                let bigram = Bigram::new(prev.clone(), tool_name);
                *self.baseline.bigrams.entry(bigram).or_insert(0) += 1;
            }

            // Update volume_stats.
            let volume = (event.data_in_bytes + event.data_out_bytes) as f64;
            self.baseline
                .volume_stats
                .entry(tool_name.to_string())
                .or_default()
                .update(volume);

            // Compute rate from previous tool call time.
            if let Some(prev_time) = self.last_tool_call_time {
                let interval = event
                    .timestamp
                    .signed_duration_since(prev_time)
                    .num_milliseconds() as f64
                    / 60_000.0;
                if interval > 0.0 {
                    self.baseline.rate_stats.update(1.0 / interval);
                }
            }

            self.last_tool_call = Some(tool_name.to_string());
            self.last_tool_call_time = Some(event.timestamp);
        }

        // Update high-water mark.
        let hwm = self
            .baseline
            .processed_through
            .entry(event.session_id.clone())
            .or_insert(0);
        if event.sequence_position > *hwm {
            *hwm = event.sequence_position;
        }

        true
    }

    /// Borrow the baseline for inspection.
    pub fn baseline(&self) -> &Baseline {
        &self.baseline
    }

    /// Consume the builder and return the finished baseline.
    /// Counts the final session if any events were processed.
    pub fn finish(mut self) -> Baseline {
        if self.current_session_id.is_some() {
            self.baseline.session_count += 1;
        }
        self.baseline
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn make_event_at(
        event_type: EventType,
        tool_name: Option<&str>,
        session_id: &str,
        timestamp: DateTime<Utc>,
        seq: u32,
    ) -> BehavioralEvent {
        BehavioralEvent {
            timestamp,
            session_id: session_id.to_string(),
            agent_id: "test-agent".to_string(),
            event_type,
            tool_name: tool_name.map(|s| s.to_string()),
            param_keys: vec![],
            resource_ids: vec![],
            data_in_bytes: 0,
            data_out_bytes: 0,
            duration_ms: 0,
            token_count: None,
            sequence_position: seq,
        }
    }

    fn make_event(
        event_type: EventType,
        tool_name: Option<&str>,
        session_id: &str,
        timestamp: DateTime<Utc>,
    ) -> BehavioralEvent {
        // Use unique sequence positions by hashing the timestamp minute
        // so multi-event tests don't self-dedup.
        let seq = (timestamp.minute() * 60 + timestamp.second()) + timestamp.hour() * 3600;
        make_event_at(event_type, tool_name, session_id, timestamp, seq)
    }

    fn ts(hour: u32, minute: u32, second: u32) -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2025, 1, 15, hour, minute, second)
            .unwrap()
    }

    #[test]
    fn new_baseline_has_correct_defaults() {
        let builder = BaselineBuilder::new("agent-1");
        let b = builder.baseline();
        assert_eq!(b.agent_id, "agent-1");
        assert_eq!(b.session_count, 0);
        assert_eq!(b.event_count, 0);
        assert!(b.tool_stats.is_empty());
        assert!(b.bigrams.is_empty());
        assert!(b.known_resources.is_empty());
        assert!(b.volume_stats.is_empty());
        assert_eq!(b.hourly_distribution, [0; 24]);
        assert_eq!(b.rate_stats.count, 0);
    }

    #[test]
    fn single_tool_call_updates_stats() {
        let mut builder = BaselineBuilder::new("agent-1");
        let mut event = make_event(EventType::ToolCall, Some("read_file"), "s1", ts(10, 0, 0));
        event.param_keys = vec!["path".to_string()];
        event.resource_ids = vec!["/tmp/foo.txt".to_string()];
        event.data_in_bytes = 100;
        event.data_out_bytes = 200;

        builder.process(&event);
        let b = builder.baseline();

        assert_eq!(b.event_count, 1);
        assert_eq!(b.first_seen, ts(10, 0, 0));
        assert_eq!(b.last_updated, ts(10, 0, 0));

        let tool = &b.tool_stats["read_file"];
        assert_eq!(tool.call_count, 1);
        assert!(tool.param_key_sets.contains(&vec!["path".to_string()]));

        assert!(b.known_resources.contains("/tmp/foo.txt"));

        let vol = &b.volume_stats["read_file"];
        assert_eq!(vol.count, 1);
        assert_eq!(vol.mean, 300.0);

        assert_eq!(b.hourly_distribution[10], 1);
    }

    #[test]
    fn bigrams_tracked_for_consecutive_tool_calls() {
        let mut builder = BaselineBuilder::new("agent-1");

        builder.process(&make_event(
            EventType::ToolCall,
            Some("read_file"),
            "s1",
            ts(10, 0, 0),
        ));
        builder.process(&make_event(
            EventType::ToolCall,
            Some("write_file"),
            "s1",
            ts(10, 1, 0),
        ));
        builder.process(&make_event(
            EventType::ToolCall,
            Some("read_file"),
            "s1",
            ts(10, 2, 0),
        ));

        let b = builder.baseline();
        assert_eq!(b.bigrams[&Bigram::new("read_file", "write_file")], 1);
        assert_eq!(b.bigrams[&Bigram::new("write_file", "read_file")], 1);
    }

    #[test]
    fn bigrams_not_formed_across_sessions() {
        let mut builder = BaselineBuilder::new("agent-1");

        builder.process(&make_event(
            EventType::ToolCall,
            Some("read_file"),
            "s1",
            ts(10, 0, 0),
        ));
        // New session — should NOT form a bigram with previous.
        builder.process(&make_event(
            EventType::ToolCall,
            Some("write_file"),
            "s2",
            ts(11, 0, 0),
        ));

        let b = builder.baseline();
        assert!(b.bigrams.is_empty());
    }

    #[test]
    fn bigrams_skip_non_tool_call_events() {
        let mut builder = BaselineBuilder::new("agent-1");

        builder.process(&make_event(
            EventType::ToolCall,
            Some("read_file"),
            "s1",
            ts(10, 0, 0),
        ));
        // UserMessage between two ToolCalls shouldn't break the chain.
        builder.process(&make_event(
            EventType::UserMessage,
            None,
            "s1",
            ts(10, 0, 30),
        ));
        builder.process(&make_event(
            EventType::ToolCall,
            Some("write_file"),
            "s1",
            ts(10, 1, 0),
        ));

        let b = builder.baseline();
        assert_eq!(b.bigrams[&Bigram::new("read_file", "write_file")], 1);
    }

    #[test]
    fn rate_stats_computed_from_intervals() {
        let mut builder = BaselineBuilder::new("agent-1");

        // Two tool calls 1 minute apart → rate = 1.0 calls/min.
        builder.process(&make_event(
            EventType::ToolCall,
            Some("read_file"),
            "s1",
            ts(10, 0, 0),
        ));
        builder.process(&make_event(
            EventType::ToolCall,
            Some("write_file"),
            "s1",
            ts(10, 1, 0),
        ));

        let b = builder.baseline();
        assert_eq!(b.rate_stats.count, 1);
        assert!((b.rate_stats.mean - 1.0).abs() < 1e-9);
    }

    #[test]
    fn rate_stats_reset_across_sessions() {
        let mut builder = BaselineBuilder::new("agent-1");

        builder.process(&make_event(
            EventType::ToolCall,
            Some("read_file"),
            "s1",
            ts(10, 0, 0),
        ));
        // New session — rate should NOT be computed from previous session's last call.
        builder.process(&make_event(
            EventType::ToolCall,
            Some("write_file"),
            "s2",
            ts(11, 0, 0),
        ));

        let b = builder.baseline();
        assert_eq!(b.rate_stats.count, 0);
    }

    #[test]
    fn volume_stats_tracked_per_tool() {
        let mut builder = BaselineBuilder::new("agent-1");

        let mut e1 = make_event(EventType::ToolCall, Some("read_file"), "s1", ts(10, 0, 0));
        e1.data_in_bytes = 100;
        e1.data_out_bytes = 500;

        let mut e2 = make_event(EventType::ToolCall, Some("write_file"), "s1", ts(10, 1, 0));
        e2.data_in_bytes = 1000;
        e2.data_out_bytes = 0;

        builder.process(&e1);
        builder.process(&e2);

        let b = builder.baseline();
        assert_eq!(b.volume_stats["read_file"].mean, 600.0);
        assert_eq!(b.volume_stats["write_file"].mean, 1000.0);
    }

    #[test]
    fn session_count_increments_on_new_session() {
        let mut builder = BaselineBuilder::new("agent-1");

        builder.process(&make_event(
            EventType::ToolCall,
            Some("read_file"),
            "s1",
            ts(10, 0, 0),
        ));
        builder.process(&make_event(
            EventType::ToolCall,
            Some("read_file"),
            "s2",
            ts(11, 0, 0),
        ));
        builder.process(&make_event(
            EventType::ToolCall,
            Some("read_file"),
            "s3",
            ts(12, 0, 0),
        ));

        // During processing, session_count is incremented on boundary crossings.
        // finish() counts the final session.
        let baseline = builder.finish();
        assert_eq!(baseline.session_count, 3);
    }

    #[test]
    fn from_baseline_resumes_correctly() {
        let mut builder = BaselineBuilder::new("agent-1");
        builder.process(&make_event(
            EventType::ToolCall,
            Some("read_file"),
            "s1",
            ts(10, 0, 0),
        ));
        let baseline = builder.finish();

        assert_eq!(baseline.event_count, 1);
        assert_eq!(baseline.session_count, 1);

        // Resume from saved baseline.
        let mut builder2 = BaselineBuilder::from_baseline(baseline);
        builder2.process(&make_event(
            EventType::ToolCall,
            Some("write_file"),
            "s2",
            ts(12, 0, 0),
        ));
        let baseline2 = builder2.finish();

        assert_eq!(baseline2.event_count, 2);
        assert_eq!(baseline2.session_count, 2);
        assert_eq!(baseline2.tool_stats["read_file"].call_count, 1);
        assert_eq!(baseline2.tool_stats["write_file"].call_count, 1);
    }

    #[test]
    fn param_keys_sorted_before_insertion() {
        let mut builder = BaselineBuilder::new("agent-1");

        let mut event = make_event(EventType::ToolCall, Some("api_call"), "s1", ts(10, 0, 0));
        event.param_keys = vec![
            "z_param".to_string(),
            "a_param".to_string(),
            "m_param".to_string(),
        ];

        builder.process(&event);

        let b = builder.baseline();
        let sets = &b.tool_stats["api_call"].param_key_sets;
        let expected = vec![
            "a_param".to_string(),
            "m_param".to_string(),
            "z_param".to_string(),
        ];
        assert!(sets.contains(&expected));
    }

    #[test]
    fn hourly_distribution_correct() {
        let mut builder = BaselineBuilder::new("agent-1");

        // Events at hours 10, 10, 14.
        builder.process(&make_event(
            EventType::ToolCall,
            Some("a"),
            "s1",
            ts(10, 0, 0),
        ));
        builder.process(&make_event(
            EventType::UserMessage,
            None,
            "s1",
            ts(10, 30, 0),
        ));
        builder.process(&make_event(
            EventType::ToolCall,
            Some("b"),
            "s1",
            ts(14, 0, 0),
        ));

        let b = builder.baseline();
        assert_eq!(b.hourly_distribution[10], 2);
        assert_eq!(b.hourly_distribution[14], 1);
        assert_eq!(b.hourly_distribution[0], 0);
    }

    #[test]
    fn duplicates_skipped_on_replay() {
        let mut builder = BaselineBuilder::new("agent-1");

        let e1 = make_event_at(EventType::ToolCall, Some("read"), "s1", ts(10, 0, 0), 1);
        let e2 = make_event_at(EventType::ToolCall, Some("write"), "s1", ts(10, 1, 0), 2);

        assert!(builder.process(&e1));
        assert!(builder.process(&e2));
        assert_eq!(builder.baseline().event_count, 2);

        // Replay the same events — should be skipped.
        assert!(!builder.process(&e1));
        assert!(!builder.process(&e2));
        assert_eq!(builder.baseline().event_count, 2);
    }

    #[test]
    fn duplicates_skipped_after_resume() {
        let mut builder = BaselineBuilder::new("agent-1");

        let e1 = make_event_at(EventType::ToolCall, Some("read"), "s1", ts(10, 0, 0), 1);
        let e2 = make_event_at(EventType::ToolCall, Some("write"), "s1", ts(10, 1, 0), 2);

        builder.process(&e1);
        builder.process(&e2);
        let baseline = builder.finish();
        assert_eq!(baseline.event_count, 2);

        // Resume from persisted baseline, replay same events.
        let mut builder2 = BaselineBuilder::from_baseline(baseline);
        assert!(!builder2.process(&e1));
        assert!(!builder2.process(&e2));

        // New event in same session passes through.
        let e3 = make_event_at(EventType::ToolCall, Some("exec"), "s1", ts(10, 2, 0), 3);
        assert!(builder2.process(&e3));

        let baseline2 = builder2.finish();
        assert_eq!(baseline2.event_count, 3);
    }

    #[test]
    fn finish_with_no_events() {
        let builder = BaselineBuilder::new("agent-1");
        let baseline = builder.finish();
        assert_eq!(baseline.session_count, 0);
        assert_eq!(baseline.event_count, 0);
    }

    #[test]
    fn tool_call_with_no_tool_name() {
        let mut builder = BaselineBuilder::new("agent-1");
        let event = make_event_at(EventType::ToolCall, None, "s1", ts(10, 0, 0), 1);
        builder.process(&event);

        let b = builder.baseline();
        assert!(
            b.tool_stats.contains_key("unknown"),
            "tool_name: None should be keyed as 'unknown'"
        );
        assert_eq!(b.tool_stats["unknown"].call_count, 1);
    }

    #[test]
    fn non_tool_events_skip_bigrams_and_rate() {
        let mut builder = BaselineBuilder::new("agent-1");

        builder.process(&make_event_at(
            EventType::UserMessage,
            None,
            "s1",
            ts(10, 0, 0),
            1,
        ));
        builder.process(&make_event_at(
            EventType::AgentMessage,
            None,
            "s1",
            ts(10, 1, 0),
            2,
        ));

        let b = builder.baseline();
        assert_eq!(b.event_count, 2);
        assert!(b.bigrams.is_empty(), "no bigrams from non-tool events");
        assert_eq!(b.rate_stats.count, 0, "no rate stats from non-tool events");
        assert!(
            b.tool_stats.is_empty(),
            "no tool_stats from non-tool events"
        );
    }

    #[test]
    fn dedup_is_per_session() {
        let mut builder = BaselineBuilder::new("agent-1");

        let e1 = make_event_at(EventType::ToolCall, Some("read"), "s1", ts(10, 0, 0), 1);
        let e2 = make_event_at(EventType::ToolCall, Some("read"), "s2", ts(11, 0, 0), 1);

        // Same sequence_position but different sessions — both should process.
        assert!(builder.process(&e1));
        assert!(builder.process(&e2));
        assert_eq!(builder.baseline().event_count, 2);
    }
}
