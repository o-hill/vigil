use std::collections::{HashMap, HashSet};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Behavioral profile built from historical events.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Baseline {
    pub agent_id: String,
    pub session_count: u64,
    pub event_count: u64,

    /// Per-tool usage frequency and stats.
    pub tool_stats: HashMap<String, ToolStats>,

    /// Bigrams of consecutive tool calls.
    pub bigrams: HashMap<(String, String), u64>,

    /// Resources the agent has accessed.
    pub known_resources: HashSet<String>,

    /// Data volume statistics per tool.
    pub volume_stats: HashMap<String, StreamingStats>,

    /// Hourly activity distribution (0-23).
    pub hourly_distribution: [u64; 24],

    /// Tool call rate (calls per minute).
    pub rate_stats: StreamingStats,

    pub first_seen: DateTime<Utc>,
    pub last_updated: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolStats {
    pub call_count: u64,
    pub first_seen: DateTime<Utc>,
    pub last_seen: DateTime<Utc>,
    /// Known parameter key combinations.
    pub param_key_sets: HashSet<Vec<String>>,
}

/// Welford's online algorithm for streaming mean/variance.
/// Updates incrementally — no need to store historical values.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingStats {
    pub count: u64,
    pub mean: f64,
    /// Sum of squares of differences from the mean.
    pub m2: f64,
}

impl StreamingStats {
    pub fn new() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            m2: 0.0,
        }
    }

    pub fn update(&mut self, value: f64) {
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
    }

    pub fn variance(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }
        self.m2 / (self.count - 1) as f64
    }

    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    pub fn z_score(&self, value: f64) -> Option<f64> {
        let sd = self.std_dev();
        if sd == 0.0 {
            return None;
        }
        Some((value - self.mean) / sd)
    }
}

impl Default for StreamingStats {
    fn default() -> Self {
        Self::new()
    }
}
