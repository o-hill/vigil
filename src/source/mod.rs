pub mod openclaw;
pub mod otel;

use std::pin::Pin;

use futures_core::Stream;

use crate::event::BehavioralEvent;

pub type EventError = Box<dyn std::error::Error + Send + Sync>;

/// Generic interface for getting behavioral events into Vigil.
///
/// Each agent framework or integration implements this trait.
/// The stream may replay historical events, emit live events,
/// or both — that's the consumer's responsibility.
pub trait EventSource: Send {
    /// Stream of behavioral events from this source.
    fn events(
        &mut self,
    ) -> Pin<Box<dyn Stream<Item = Result<BehavioralEvent, EventError>> + Send + '_>>;

    /// Human-readable name for this source.
    fn name(&self) -> &str;
}

/// Heuristic: does this string look like a file path or URL?
pub(crate) fn looks_like_resource(s: &str) -> bool {
    s.starts_with('/')
        || s.starts_with("~/")
        || s.starts_with("http://")
        || s.starts_with("https://")
}
