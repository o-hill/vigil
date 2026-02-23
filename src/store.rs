use std::fs::{self, OpenOptions};
use std::io::{self, BufRead, Write};
use std::path::PathBuf;

use crate::baseline::Baseline;
use crate::event::BehavioralEvent;

/// Backend-agnostic storage for baselines and events.
pub trait Store {
    type Error: std::error::Error + Send + Sync + 'static;

    fn load_baseline(&self, agent_id: &str) -> Result<Option<Baseline>, Self::Error>;
    fn save_baseline(&self, baseline: &Baseline) -> Result<(), Self::Error>;

    fn append_events(&self, agent_id: &str, events: &[BehavioralEvent]) -> Result<(), Self::Error>;
    fn load_events(&self, agent_id: &str) -> Result<Vec<BehavioralEvent>, Self::Error>;

    fn list_agents(&self) -> Result<Vec<String>, Self::Error>;
}

/// Filesystem-backed store. Layout:
///
/// ```text
/// root/
///   <agent_id>/
///     baseline.json
///     events.jsonl
/// ```
pub struct FileStore {
    root: PathBuf,
}

impl FileStore {
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    /// Default store at `~/.vigil/`.
    pub fn default_location() -> io::Result<Self> {
        let home = dirs::home_dir().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                "could not determine home directory",
            )
        })?;
        Ok(Self::new(home.join(".vigil")))
    }

    fn agent_dir(&self, agent_id: &str) -> PathBuf {
        self.root.join(agent_id)
    }

    fn baseline_path(&self, agent_id: &str) -> PathBuf {
        self.agent_dir(agent_id).join("baseline.json")
    }

    fn events_path(&self, agent_id: &str) -> PathBuf {
        self.agent_dir(agent_id).join("events.jsonl")
    }
}

impl Store for FileStore {
    type Error = io::Error;

    fn load_baseline(&self, agent_id: &str) -> Result<Option<Baseline>, io::Error> {
        let path = self.baseline_path(agent_id);
        match fs::read_to_string(&path) {
            Ok(contents) => {
                let baseline: Baseline = serde_json::from_str(&contents)
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
                Ok(Some(baseline))
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e),
        }
    }

    fn save_baseline(&self, baseline: &Baseline) -> Result<(), io::Error> {
        let dir = self.agent_dir(&baseline.agent_id);
        fs::create_dir_all(&dir)?;
        let json = serde_json::to_string_pretty(baseline)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        fs::write(self.baseline_path(&baseline.agent_id), json)?;
        Ok(())
    }

    fn append_events(&self, agent_id: &str, events: &[BehavioralEvent]) -> Result<(), io::Error> {
        let dir = self.agent_dir(agent_id);
        fs::create_dir_all(&dir)?;
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.events_path(agent_id))?;
        for event in events {
            let line = serde_json::to_string(event)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
            writeln!(file, "{line}")?;
        }
        Ok(())
    }

    fn load_events(&self, agent_id: &str) -> Result<Vec<BehavioralEvent>, io::Error> {
        let path = self.events_path(agent_id);
        match fs::File::open(&path) {
            Ok(file) => {
                let reader = io::BufReader::new(file);
                let mut events = Vec::new();
                for line in reader.lines() {
                    let line = line?;
                    if line.trim().is_empty() {
                        continue;
                    }
                    let event: BehavioralEvent = serde_json::from_str(&line)
                        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
                    events.push(event);
                }
                Ok(events)
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(Vec::new()),
            Err(e) => Err(e),
        }
    }

    fn list_agents(&self) -> Result<Vec<String>, io::Error> {
        match fs::read_dir(&self.root) {
            Ok(entries) => {
                let mut agents: Vec<String> = entries
                    .filter_map(|entry| {
                        let entry = entry.ok()?;
                        if entry.path().is_dir() {
                            entry.file_name().into_string().ok()
                        } else {
                            None
                        }
                    })
                    .collect();
                agents.sort();
                Ok(agents)
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(Vec::new()),
            Err(e) => Err(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::EventType;
    use chrono::{TimeZone, Utc};
    use std::sync::atomic::{AtomicU32, Ordering};

    static TEST_COUNTER: AtomicU32 = AtomicU32::new(0);

    fn temp_store() -> FileStore {
        let id = TEST_COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir()
            .join("vigil-test")
            .join(format!("{}_{id}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        FileStore::new(dir)
    }

    fn make_baseline(agent_id: &str) -> Baseline {
        let mut b = Baseline::new(agent_id);
        b.event_count = 42;
        b.session_count = 3;
        b
    }

    fn make_event(agent_id: &str, tool: &str) -> BehavioralEvent {
        BehavioralEvent {
            timestamp: Utc.with_ymd_and_hms(2025, 1, 15, 10, 0, 0).unwrap(),
            session_id: "s1".to_string(),
            agent_id: agent_id.to_string(),
            event_type: EventType::ToolCall,
            tool_name: Some(tool.to_string()),
            param_keys: vec!["path".to_string()],
            resource_ids: vec![],
            data_in_bytes: 100,
            data_out_bytes: 200,
            duration_ms: 0,
            token_count: None,
            sequence_position: 0,
        }
    }

    #[test]
    fn save_then_load_baseline_roundtrips() {
        let store = temp_store();
        let baseline = make_baseline("agent-1");
        store.save_baseline(&baseline).unwrap();

        let loaded = store.load_baseline("agent-1").unwrap().unwrap();
        assert_eq!(loaded.agent_id, "agent-1");
        assert_eq!(loaded.event_count, 42);
        assert_eq!(loaded.session_count, 3);
    }

    #[test]
    fn load_baseline_returns_none_for_nonexistent() {
        let store = temp_store();
        let result = store.load_baseline("no-such-agent").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn append_then_load_events_roundtrips() {
        let store = temp_store();
        let events = vec![
            make_event("agent-1", "read"),
            make_event("agent-1", "write"),
        ];
        store.append_events("agent-1", &events).unwrap();

        let loaded = store.load_events("agent-1").unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].tool_name.as_deref(), Some("read"));
        assert_eq!(loaded[1].tool_name.as_deref(), Some("write"));
    }

    #[test]
    fn append_events_is_additive() {
        let store = temp_store();
        store
            .append_events("agent-1", &[make_event("agent-1", "read")])
            .unwrap();
        store
            .append_events("agent-1", &[make_event("agent-1", "write")])
            .unwrap();

        let loaded = store.load_events("agent-1").unwrap();
        assert_eq!(loaded.len(), 2);
    }

    #[test]
    fn list_agents_returns_saved_ids() {
        let store = temp_store();
        store.save_baseline(&make_baseline("alpha")).unwrap();
        store.save_baseline(&make_baseline("beta")).unwrap();

        let agents = store.list_agents().unwrap();
        assert_eq!(agents, vec!["alpha", "beta"]);
    }

    #[test]
    fn save_baseline_creates_parent_directories() {
        let store = temp_store();
        // Root doesn't exist yet — save should create it.
        store.save_baseline(&make_baseline("deep-agent")).unwrap();
        assert!(store.baseline_path("deep-agent").exists());
    }

    #[test]
    fn append_empty_slice() {
        let store = temp_store();
        store.append_events("agent-1", &[]).unwrap();

        let events_path = store.events_path("agent-1");
        // The file may or may not exist, but if it does, it should be empty.
        if events_path.exists() {
            let content = fs::read_to_string(&events_path).unwrap();
            assert!(content.is_empty(), "empty slice should not write data");
        }
        // Either way, loading events should return empty vec.
        let loaded = store.load_events("agent-1").unwrap();
        assert!(loaded.is_empty());
    }

    #[test]
    fn list_agents_ignores_files() {
        let store = temp_store();
        // Create the root directory with a regular file (not a subdirectory).
        fs::create_dir_all(&store.root).unwrap();
        fs::write(store.root.join("not-a-dir.txt"), "hello").unwrap();

        // Also create a real agent dir to verify it IS returned.
        store.save_baseline(&make_baseline("real-agent")).unwrap();

        let agents = store.list_agents().unwrap();
        assert_eq!(agents, vec!["real-agent"]);
        assert!(
            !agents.contains(&"not-a-dir.txt".to_string()),
            "regular files should be excluded from list_agents"
        );
    }

    #[test]
    fn list_agents_empty_for_nonexistent_directory() {
        let store = FileStore::new(PathBuf::from("/tmp/vigil-nonexistent-dir-test"));
        let agents = store.list_agents().unwrap();
        assert!(agents.is_empty());
    }
}
