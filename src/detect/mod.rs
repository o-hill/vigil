pub mod detectors;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::baseline::Baseline;
use crate::event::BehavioralEvent;

/// A detected behavioral anomaly.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Anomaly {
    pub timestamp: DateTime<Utc>,
    pub severity: Severity,
    pub anomaly_type: AnomalyType,
    pub description: String,
    pub evidence: Evidence,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Severity {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnomalyType {
    UnknownTool,
    UnknownSequence,
    UnknownResource,
    VolumeSpike,
    RateSpike,
    TemporalAnomaly,
    UnknownParamPattern,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Evidence {
    pub event: BehavioralEvent,
    /// Human-readable baseline context.
    pub baseline_value: String,
    /// Human-readable observed value.
    pub observed_value: String,
    pub z_score: Option<f64>,
    /// 0.0-1.0 based on baseline maturity.
    pub confidence: f64,
}

/// Individual anomaly detection strategy.
pub trait Detector: Send + Sync {
    fn detect(&self, event: &BehavioralEvent, baseline: &Baseline) -> Vec<Anomaly>;
}

/// Returns the default set of detectors.
pub fn default_detectors(threshold: f64) -> Vec<Box<dyn Detector>> {
    vec![
        Box::new(detectors::UnknownToolDetector),
        Box::new(detectors::UnknownSequenceDetector::new()),
        Box::new(detectors::RateSpikeDetector::new(threshold)),
        Box::new(detectors::VolumeSpikeDetector::new(threshold)),
    ]
}
