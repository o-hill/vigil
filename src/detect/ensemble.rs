use std::collections::HashMap;
use std::f64::consts::{PI, SQRT_2};

use super::{Anomaly, AnomalyType, Evidence, Severity};
use crate::event::BehavioralEvent;

/// Combines independent anomaly signals using Fisher's method and
/// Stouffer's Z. Emits a single EnsembleScore anomaly per event when
/// multiple detectors fire.
pub struct EnsembleScorer {
    weights: HashMap<AnomalyType, f64>,
}

impl Default for EnsembleScorer {
    fn default() -> Self {
        let weights = HashMap::from([
            (AnomalyType::UnknownTool, 3.0),
            (AnomalyType::UnknownSequence, 2.0),
            (AnomalyType::VolumeSpike, 2.0),
            (AnomalyType::UnknownResource, 1.5),
            (AnomalyType::RateSpike, 1.5),
            (AnomalyType::UnknownParamPattern, 1.5),
            (AnomalyType::TemporalAnomaly, 1.0),
            (AnomalyType::ReasoningLoop, 2.5),
            (AnomalyType::ErrorRetryLoop, 2.0),
            (AnomalyType::CombinedAnomaly, 2.0),
        ]);
        Self { weights }
    }
}

impl EnsembleScorer {
    /// Score a batch of anomalies from a single event.
    /// Returns an EnsembleScore anomaly if the combined signal is strong enough.
    pub fn score(&self, anomalies: &[Anomaly], event: &BehavioralEvent) -> Option<Anomaly> {
        // Need at least 2 independent signals to combine.
        if anomalies.len() < 2 {
            return None;
        }

        let mut weighted_scores: Vec<(f64, f64)> = Vec::new(); // (score, weight)

        for anomaly in anomalies {
            // Skip ensemble scores from previous runs.
            if anomaly.anomaly_type == AnomalyType::EnsembleScore {
                continue;
            }
            let score = severity_to_score(anomaly.severity);
            let weight = self
                .weights
                .get(&anomaly.anomaly_type)
                .copied()
                .unwrap_or(1.0);
            weighted_scores.push((score, weight));
        }

        if weighted_scores.len() < 2 {
            return None;
        }

        let fisher_p = fisher_combined(&weighted_scores);
        let stouffer_p = stouffer_combined(&weighted_scores);
        let combined_p = fisher_p.max(stouffer_p);

        if combined_p > 0.8 {
            let severity = if combined_p > 0.95 {
                Severity::Critical
            } else if combined_p > 0.9 {
                Severity::High
            } else {
                Severity::Medium
            };

            Some(Anomaly {
                timestamp: event.timestamp,
                severity,
                anomaly_type: AnomalyType::EnsembleScore,
                description: format!(
                    "ensemble score: {combined_p:.3} from {} signals (fisher={fisher_p:.3}, stouffer={stouffer_p:.3})",
                    weighted_scores.len()
                ),
                evidence: Evidence {
                    event: event.clone(),
                    baseline_value: "threshold: 0.8".to_string(),
                    observed_value: format!("combined p-value: {combined_p:.3}"),
                    z_score: Some(combined_p),
                    confidence: 1.0,
                },
            })
        } else {
            None
        }
    }
}

fn severity_to_score(severity: Severity) -> f64 {
    match severity {
        Severity::Low => 0.3,
        Severity::Medium => 0.6,
        Severity::High => 0.8,
        Severity::Critical => 0.95,
    }
}

/// Fisher's method: X² = -2 * Σ(w_i * ln(1 - s_i))
/// Under H0, X² ~ χ²(2k). Returns 1 - P(χ²(2k) >= X²).
fn fisher_combined(scores: &[(f64, f64)]) -> f64 {
    let k = scores.len();
    let chi2: f64 = scores
        .iter()
        .map(|&(s, w)| {
            let clamped = s.clamp(0.001, 0.999);
            -2.0 * w * (1.0 - clamped).ln()
        })
        .sum();

    let total_weight: f64 = scores.iter().map(|&(_, w)| w).sum();
    let mean_weight = total_weight / k as f64;
    // Scale degrees of freedom by mean weight.
    let df = 2.0 * k as f64 * mean_weight;

    1.0 - chi2_survival(chi2, df)
}

/// Stouffer's Z: Z = Σ(w_i * Φ⁻¹(s_i)) / √(Σ(w_i²))
/// Returns Φ(Z) — the combined p-value.
fn stouffer_combined(scores: &[(f64, f64)]) -> f64 {
    let numerator: f64 = scores
        .iter()
        .map(|&(s, w)| {
            let clamped = s.clamp(0.001, 0.999);
            w * probit(clamped)
        })
        .sum();

    let denominator: f64 = scores.iter().map(|&(_, w)| w * w).sum::<f64>().sqrt();

    if denominator == 0.0 {
        return 0.0;
    }

    normal_cdf(numerator / denominator)
}

// --- Pure-Rust math approximations ---

/// Error function approximation (Abramowitz & Stegun 7.1.26).
/// Max error ~1.5e-7.
fn erf(x: f64) -> f64 {
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));

    sign * (1.0 - poly * (-x * x).exp())
}

/// Standard normal CDF: Φ(x) = 0.5 * (1 + erf(x / √2))
fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / SQRT_2))
}

/// Probit function (inverse normal CDF).
/// Uses initial approximation + Newton-Raphson refinement.
fn probit(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    if (p - 0.5).abs() < 1e-15 {
        return 0.0;
    }

    // Initial approximation (Abramowitz & Stegun 26.2.23).
    let sign = if p < 0.5 { -1.0 } else { 1.0 };
    let pp = if p < 0.5 { p } else { 1.0 - p };
    let t = (-2.0 * pp.ln()).sqrt();

    let c0 = 2.515517;
    let c1 = 0.802853;
    let c2 = 0.010328;
    let d1 = 1.432788;
    let d2 = 0.189269;
    let d3 = 0.001308;

    let mut x =
        sign * (t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t));

    // Newton-Raphson refinement (2 iterations suffice for ~1e-9 accuracy).
    let inv_sqrt_2pi = 1.0 / (2.0 * PI).sqrt();
    for _ in 0..2 {
        let phi = normal_cdf(x);
        let pdf = inv_sqrt_2pi * (-0.5 * x * x).exp();
        if pdf.abs() < 1e-30 {
            break;
        }
        x -= (phi - p) / pdf;
    }

    x
}

/// Chi-squared survival function: P(χ²(k) >= x).
/// Uses the regularized lower incomplete gamma function.
fn chi2_survival(x: f64, k: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    // P(χ²(k) >= x) = 1 - γ(k/2, x/2) / Γ(k/2)
    // = 1 - regularized_lower_gamma(k/2, x/2)
    1.0 - regularized_lower_gamma(k / 2.0, x / 2.0)
}

/// Regularized lower incomplete gamma function P(a, x) = γ(a,x)/Γ(a).
/// Uses series expansion for x < a+1, continued fraction otherwise.
fn regularized_lower_gamma(a: f64, x: f64) -> f64 {
    if x < 0.0 {
        return 0.0;
    }
    if x == 0.0 {
        return 0.0;
    }

    if x < a + 1.0 {
        // Series expansion.
        gamma_series(a, x)
    } else {
        // Continued fraction (Lentz's method).
        1.0 - gamma_cf(a, x)
    }
}

/// Series expansion for regularized lower incomplete gamma.
fn gamma_series(a: f64, x: f64) -> f64 {
    let ln_gamma_a = ln_gamma(a);
    let mut sum = 1.0 / a;
    let mut term = 1.0 / a;

    for n in 1..200 {
        term *= x / (a + n as f64);
        sum += term;
        if term.abs() < sum.abs() * 1e-14 {
            break;
        }
    }

    sum * (-x + a * x.ln() - ln_gamma_a).exp()
}

/// Continued fraction for upper incomplete gamma Q(a,x) = 1 - P(a,x).
fn gamma_cf(a: f64, x: f64) -> f64 {
    let ln_gamma_a = ln_gamma(a);

    // Lentz's method.
    let mut c = 1e-30_f64;
    let mut d = 1.0 / (x + 1.0 - a);
    let mut f = d;

    for n in 1..200 {
        let an = -(n as f64) * (n as f64 - a);
        let bn = x + 2.0 * n as f64 + 1.0 - a;
        d = bn + an * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = bn + an / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        let delta = c * d;
        f *= delta;
        if (delta - 1.0).abs() < 1e-14 {
            break;
        }
    }

    f * (-x + a * x.ln() - ln_gamma_a).exp()
}

/// Lanczos approximation for ln(Γ(x)).
fn ln_gamma(x: f64) -> f64 {
    let coefficients = [
        76.18009172947146,
        -86.50532032941677,
        24.01409824083091,
        -1.231739572450155,
        0.1208650973866179e-2,
        -0.5395239384953e-5,
    ];

    let y = x;
    let tmp = x + 5.5;
    let tmp = tmp - (x + 0.5) * tmp.ln();

    let mut ser = 1.000000000190015;
    for (i, &coeff) in coefficients.iter().enumerate() {
        ser += coeff / (y + 1.0 + i as f64);
    }

    -tmp + ((2.0 * PI).sqrt() * ser / x).ln()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::baseline::Baseline;
    use crate::event::EventType;
    use chrono::{TimeZone, Utc};

    fn make_test_event(tool: &str) -> BehavioralEvent {
        BehavioralEvent {
            timestamp: Utc.with_ymd_and_hms(2025, 1, 15, 10, 0, 0).unwrap(),
            session_id: "s1".to_string(),
            agent_id: "test".to_string(),
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

    #[test]
    fn erf_known_values() {
        assert!((erf(0.0)).abs() < 1e-6);
        assert!((erf(1.0) - 0.8427007929).abs() < 1e-5);
        assert!((erf(-1.0) + 0.8427007929).abs() < 1e-5);
        assert!((erf(2.0) - 0.9953222650).abs() < 1e-5);
    }

    #[test]
    fn normal_cdf_known_values() {
        assert!((normal_cdf(0.0) - 0.5).abs() < 1e-6);
        assert!((normal_cdf(1.96) - 0.975).abs() < 1e-3);
        assert!((normal_cdf(-1.96) - 0.025).abs() < 1e-3);
    }

    #[test]
    fn probit_known_values() {
        assert!(probit(0.5).abs() < 1e-6);
        assert!((probit(0.975) - 1.96).abs() < 0.01);
        assert!((probit(0.025) + 1.96).abs() < 0.01);
    }

    #[test]
    fn chi2_survival_known_values() {
        // χ²(2) >= 5.991 should be ~0.05
        assert!((chi2_survival(5.991, 2.0) - 0.05).abs() < 0.01);
        // χ²(2) >= 0 should be 1.0
        assert!((chi2_survival(0.0, 2.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn ensemble_needs_two_signals() {
        let scorer = EnsembleScorer::default();
        let event = make_test_event("read");
        let _baseline = Baseline::new("test");

        // Single anomaly.
        let anomalies = vec![Anomaly {
            timestamp: event.timestamp,
            severity: Severity::High,
            anomaly_type: AnomalyType::UnknownTool,
            description: "test".to_string(),
            evidence: Evidence {
                event: event.clone(),
                baseline_value: "".to_string(),
                observed_value: "".to_string(),
                z_score: None,
                confidence: 1.0,
            },
        }];

        assert!(scorer.score(&anomalies, &event).is_none());
    }

    #[test]
    fn ensemble_fires_for_multiple_high_severity() {
        let scorer = EnsembleScorer::default();
        let event = make_test_event("evil");

        let anomalies = vec![
            Anomaly {
                timestamp: event.timestamp,
                severity: Severity::High,
                anomaly_type: AnomalyType::UnknownTool,
                description: "unknown tool".to_string(),
                evidence: Evidence {
                    event: event.clone(),
                    baseline_value: "".to_string(),
                    observed_value: "".to_string(),
                    z_score: None,
                    confidence: 1.0,
                },
            },
            Anomaly {
                timestamp: event.timestamp,
                severity: Severity::High,
                anomaly_type: AnomalyType::UnknownSequence,
                description: "unknown sequence".to_string(),
                evidence: Evidence {
                    event: event.clone(),
                    baseline_value: "".to_string(),
                    observed_value: "".to_string(),
                    z_score: None,
                    confidence: 1.0,
                },
            },
        ];

        let result = scorer.score(&anomalies, &event);
        assert!(result.is_some());
        let ensemble = result.unwrap();
        assert_eq!(ensemble.anomaly_type, AnomalyType::EnsembleScore);
    }

    #[test]
    fn ensemble_empty_returns_none() {
        let scorer = EnsembleScorer::default();
        let event = make_test_event("read");
        assert!(scorer.score(&[], &event).is_none());
    }
}
