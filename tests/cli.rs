use std::fs;
use std::path::Path;

use assert_cmd::Command;
use predicates::prelude::*;
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn vigil() -> Command {
    Command::cargo_bin("vigil").unwrap()
}

fn temp_dir() -> TempDir {
    tempfile::tempdir().unwrap()
}

fn write_file(dir: &Path, name: &str, content: &str) {
    let path = dir.join(name);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(path, content).unwrap();
}

/// Minimal valid OpenClaw session: session line + user message + assistant tool call + tool result.
fn fixture_session_lines() -> String {
    [
        r#"{"type":"session","version":3,"id":"sess-1","timestamp":"2026-02-20T10:00:00.000Z","cwd":"/tmp"}"#,
        r#"{"type":"message","id":"m1","parentId":null,"timestamp":"2026-02-20T10:00:01.000Z","message":{"role":"user","content":[{"type":"text","text":"hello"}],"timestamp":1771624609040}}"#,
        r#"{"type":"message","id":"m2","parentId":"m1","timestamp":"2026-02-20T10:00:02.000Z","message":{"role":"assistant","content":[{"type":"text","text":"Let me read that."},{"type":"toolCall","id":"tc1","name":"read_file","arguments":{"file_path":"/tmp/foo.txt"}}],"api":"anthropic-messages","provider":"anthropic","model":"claude-sonnet-4-6","usage":{"input":10,"output":50,"totalTokens":60},"stopReason":"toolUse","timestamp":1771624609041}}"#,
        r#"{"type":"message","id":"m3","parentId":"m2","timestamp":"2026-02-20T10:00:03.000Z","message":{"role":"toolResult","toolCallId":"tc1","toolName":"read_file","content":[{"type":"text","text":"file contents here"}],"isError":false,"timestamp":1771624609042}}"#,
    ].join("\n")
}

/// Pre-serialized BehavioralEvent JSONL (for baseline build / detect stdin).
fn fixture_events() -> String {
    [
        r#"{"timestamp":"2026-02-20T10:00:01.000Z","session_id":"sess-1","agent_id":"test-agent","event_type":"UserMessage","tool_name":null,"param_keys":[],"resource_ids":[],"data_in_bytes":10,"data_out_bytes":0,"duration_ms":0,"token_count":null,"sequence_position":1}"#,
        r#"{"timestamp":"2026-02-20T10:00:02.000Z","session_id":"sess-1","agent_id":"test-agent","event_type":"ToolCall","tool_name":"read_file","param_keys":["file_path"],"resource_ids":["/tmp/foo.txt"],"data_in_bytes":50,"data_out_bytes":0,"duration_ms":0,"token_count":60,"sequence_position":2}"#,
        r#"{"timestamp":"2026-02-20T10:00:03.000Z","session_id":"sess-1","agent_id":"test-agent","event_type":"ToolResult","tool_name":"read_file","param_keys":[],"resource_ids":[],"data_in_bytes":0,"data_out_bytes":100,"duration_ms":0,"token_count":null,"sequence_position":3}"#,
    ].join("\n")
}

/// Serialized Baseline JSON with given tool names pre-populated.
fn make_baseline_json(agent_id: &str, tools: &[&str]) -> String {
    let mut tool_stats = String::from("{");
    for (i, tool) in tools.iter().enumerate() {
        if i > 0 {
            tool_stats.push(',');
        }
        tool_stats.push_str(&format!(
            r#""{tool}":{{"call_count":10,"first_seen":"2026-02-20T10:00:00Z","last_seen":"2026-02-20T10:00:00Z","param_key_sets":[]}}"#
        ));
    }
    tool_stats.push('}');

    format!(
        r#"{{"agent_id":"{agent_id}","session_count":5,"event_count":200,"tool_stats":{tool_stats},"bigrams":{{}},"known_resources":[],"volume_stats":{{}},"hourly_distribution":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"rate_stats":{{"count":0,"mean":0.0,"m2":0.0}},"first_seen":"2026-02-20T10:00:00Z","last_updated":"2026-02-20T10:00:00Z","processed_through":{{}}}}"#
    )
}

// ===========================================================================
// vigil openclaw parse
// ===========================================================================

#[test]
fn parse_single_file() {
    let dir = temp_dir();
    write_file(dir.path(), "session.jsonl", &fixture_session_lines());

    let assert = vigil()
        .args(["openclaw", "parse", dir.path().to_str().unwrap()])
        .assert()
        .success();

    let stdout = String::from_utf8(assert.get_output().stdout.clone()).unwrap();
    let stderr = String::from_utf8(assert.get_output().stderr.clone()).unwrap();

    // Each stdout line should be valid JSON with expected fields.
    let lines: Vec<&str> = stdout.lines().collect();
    assert!(!lines.is_empty(), "expected events on stdout");
    for line in &lines {
        let v: serde_json::Value = serde_json::from_str(line).expect("each line should be JSON");
        assert!(v.get("event_type").is_some(), "missing event_type field");
        assert!(v.get("session_id").is_some(), "missing session_id field");
    }

    assert!(
        stderr.contains("done:"),
        "stderr should contain 'done:' summary"
    );
}

#[test]
fn parse_nonexistent_path() {
    vigil()
        .args(["openclaw", "parse", "/nonexistent/path/to/sessions"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("error:"));
}

#[test]
fn parse_empty_dir() {
    let dir = temp_dir();

    vigil()
        .args(["openclaw", "parse", dir.path().to_str().unwrap()])
        .assert()
        .success()
        .stderr(predicate::str::contains("no .jsonl files found"));
}

#[test]
fn parse_skips_bad_lines() {
    let dir = temp_dir();
    let content = format!("{}\nthis is not valid json at all", fixture_session_lines());
    write_file(dir.path(), "mixed.jsonl", &content);

    let assert = vigil()
        .args(["openclaw", "parse", dir.path().to_str().unwrap()])
        .assert()
        .success();

    let stdout = String::from_utf8(assert.get_output().stdout.clone()).unwrap();
    let stderr = String::from_utf8(assert.get_output().stderr.clone()).unwrap();

    // Valid events should still appear on stdout.
    assert!(!stdout.is_empty(), "should have events from valid lines");
    // Warning about bad line on stderr.
    assert!(
        stderr.contains("warning"),
        "stderr should warn about bad line"
    );
}

#[test]
fn parse_agent_id_flag() {
    let dir = temp_dir();
    write_file(dir.path(), "session.jsonl", &fixture_session_lines());

    let assert = vigil()
        .args([
            "openclaw",
            "parse",
            "--agent-id",
            "custom-agent",
            dir.path().to_str().unwrap(),
        ])
        .assert()
        .success();

    let stdout = String::from_utf8(assert.get_output().stdout.clone()).unwrap();
    for line in stdout.lines() {
        let v: serde_json::Value = serde_json::from_str(line).unwrap();
        assert_eq!(
            v["agent_id"].as_str().unwrap(),
            "custom-agent",
            "all events should have custom agent_id"
        );
    }
}

// ===========================================================================
// vigil baseline build
// ===========================================================================

#[test]
fn build_from_stdin() {
    let store_dir = temp_dir();

    vigil()
        .args([
            "baseline",
            "build",
            "--store",
            store_dir.path().to_str().unwrap(),
        ])
        .write_stdin(fixture_events())
        .assert()
        .success();

    // baseline.json should exist under the agent's directory.
    let baseline_path = store_dir.path().join("test-agent").join("baseline.json");
    assert!(baseline_path.exists(), "baseline.json should be created");

    let baseline_content = fs::read_to_string(&baseline_path).unwrap();
    let baseline: serde_json::Value = serde_json::from_str(&baseline_content).unwrap();
    assert_eq!(baseline["agent_id"].as_str().unwrap(), "test-agent");
    assert!(
        baseline["event_count"].as_u64().unwrap() > 0,
        "event_count should be > 0"
    );
}

#[test]
fn build_empty_stdin() {
    let store_dir = temp_dir();

    vigil()
        .args([
            "baseline",
            "build",
            "--store",
            store_dir.path().to_str().unwrap(),
        ])
        .write_stdin("")
        .assert()
        .success()
        .stderr(predicate::str::contains("no events processed"));

    // No agent subdirectories should be created.
    let entries: Vec<_> = fs::read_dir(store_dir.path())
        .ok()
        .map(|rd| rd.filter_map(|e| e.ok()).collect())
        .unwrap_or_default();
    assert!(
        entries.is_empty(),
        "no files should be created for empty stdin"
    );
}

#[test]
fn build_resumes_existing() {
    let store_dir = temp_dir();

    // First build.
    vigil()
        .args([
            "baseline",
            "build",
            "--store",
            store_dir.path().to_str().unwrap(),
        ])
        .write_stdin(fixture_events())
        .assert()
        .success();

    let baseline_path = store_dir.path().join("test-agent").join("baseline.json");
    let b1: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&baseline_path).unwrap()).unwrap();
    let count1 = b1["event_count"].as_u64().unwrap();

    // Second build with NEW events (different session so they won't be deduped).
    let new_events = [
        r#"{"timestamp":"2026-02-20T11:00:01.000Z","session_id":"sess-2","agent_id":"test-agent","event_type":"ToolCall","tool_name":"write_file","param_keys":["path"],"resource_ids":[],"data_in_bytes":200,"data_out_bytes":0,"duration_ms":0,"token_count":null,"sequence_position":1}"#,
    ].join("\n");

    vigil()
        .args([
            "baseline",
            "build",
            "--store",
            store_dir.path().to_str().unwrap(),
        ])
        .write_stdin(new_events)
        .assert()
        .success()
        .stderr(predicate::str::contains("resuming baseline"));

    let b2: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&baseline_path).unwrap()).unwrap();
    let count2 = b2["event_count"].as_u64().unwrap();

    assert!(count2 > count1, "event_count should increase on resume");
}

#[test]
fn build_dedup_on_replay() {
    let store_dir = temp_dir();

    // First build.
    vigil()
        .args([
            "baseline",
            "build",
            "--store",
            store_dir.path().to_str().unwrap(),
        ])
        .write_stdin(fixture_events())
        .assert()
        .success();

    let baseline_path = store_dir.path().join("test-agent").join("baseline.json");
    let b1: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&baseline_path).unwrap()).unwrap();
    let count1 = b1["event_count"].as_u64().unwrap();

    // Replay the exact same events.
    vigil()
        .args([
            "baseline",
            "build",
            "--store",
            store_dir.path().to_str().unwrap(),
        ])
        .write_stdin(fixture_events())
        .assert()
        .success();

    let b2: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&baseline_path).unwrap()).unwrap();
    let count2 = b2["event_count"].as_u64().unwrap();

    assert_eq!(
        count1, count2,
        "replaying same events should not change event_count"
    );
}

#[test]
fn build_appends_events() {
    let store_dir = temp_dir();

    vigil()
        .args([
            "baseline",
            "build",
            "--store",
            store_dir.path().to_str().unwrap(),
        ])
        .write_stdin(fixture_events())
        .assert()
        .success();

    let events_path = store_dir.path().join("test-agent").join("events.jsonl");
    assert!(events_path.exists(), "events.jsonl should be written");

    let content = fs::read_to_string(&events_path).unwrap();
    let line_count = content.lines().count();
    assert!(line_count > 0, "events.jsonl should have lines");
}

// ===========================================================================
// vigil detect
// ===========================================================================

#[test]
fn detect_unknown_tool() {
    let store_dir = temp_dir();

    // Pre-write a baseline that only knows "read_file".
    let agent_dir = store_dir.path().join("test-agent");
    fs::create_dir_all(&agent_dir).unwrap();
    fs::write(
        agent_dir.join("baseline.json"),
        make_baseline_json("test-agent", &["read_file"]),
    )
    .unwrap();

    // Pipe an event with a novel tool "evil_exfiltrate".
    let event = r#"{"timestamp":"2026-02-20T10:00:01.000Z","session_id":"sess-1","agent_id":"test-agent","event_type":"ToolCall","tool_name":"evil_exfiltrate","param_keys":[],"resource_ids":[],"data_in_bytes":0,"data_out_bytes":0,"duration_ms":0,"token_count":null,"sequence_position":1}"#;

    let assert = vigil()
        .args(["detect", "--store", store_dir.path().to_str().unwrap()])
        .write_stdin(event)
        .assert()
        .success();

    let stdout = String::from_utf8(assert.get_output().stdout.clone()).unwrap();
    assert!(
        stdout.contains("UnknownTool"),
        "should detect unknown tool anomaly"
    );
}

#[test]
fn detect_clean_event() {
    let store_dir = temp_dir();

    let agent_dir = store_dir.path().join("test-agent");
    fs::create_dir_all(&agent_dir).unwrap();
    fs::write(
        agent_dir.join("baseline.json"),
        make_baseline_json("test-agent", &["read_file"]),
    )
    .unwrap();

    // Pipe an event with a known tool.
    let event = r#"{"timestamp":"2026-02-20T10:00:01.000Z","session_id":"sess-1","agent_id":"test-agent","event_type":"ToolCall","tool_name":"read_file","param_keys":["file_path"],"resource_ids":[],"data_in_bytes":50,"data_out_bytes":0,"duration_ms":0,"token_count":null,"sequence_position":1}"#;

    let assert = vigil()
        .args(["detect", "--store", store_dir.path().to_str().unwrap()])
        .write_stdin(event)
        .assert()
        .success();

    let stdout = String::from_utf8(assert.get_output().stdout.clone()).unwrap();
    assert!(
        stdout.trim().is_empty(),
        "no anomalies expected for known tool"
    );
}

#[test]
fn detect_no_baseline() {
    let store_dir = temp_dir();

    // No baseline file — agent is unknown.
    let event = r#"{"timestamp":"2026-02-20T10:00:01.000Z","session_id":"sess-1","agent_id":"unknown-agent","event_type":"ToolCall","tool_name":"read","param_keys":[],"resource_ids":[],"data_in_bytes":0,"data_out_bytes":0,"duration_ms":0,"token_count":null,"sequence_position":1}"#;

    vigil()
        .args(["detect", "--store", store_dir.path().to_str().unwrap()])
        .write_stdin(event)
        .assert()
        .success()
        .stderr(predicate::str::contains("no baseline"));
}

#[test]
fn detect_empty_stdin() {
    let store_dir = temp_dir();

    vigil()
        .args(["detect", "--store", store_dir.path().to_str().unwrap()])
        .write_stdin("")
        .assert()
        .success()
        .stderr(predicate::str::contains("events scanned: 0"));
}

#[test]
fn detect_summary_on_stderr() {
    let store_dir = temp_dir();

    let agent_dir = store_dir.path().join("test-agent");
    fs::create_dir_all(&agent_dir).unwrap();
    fs::write(
        agent_dir.join("baseline.json"),
        make_baseline_json("test-agent", &["read_file"]),
    )
    .unwrap();

    let event = r#"{"timestamp":"2026-02-20T10:00:01.000Z","session_id":"sess-1","agent_id":"test-agent","event_type":"ToolCall","tool_name":"read_file","param_keys":[],"resource_ids":[],"data_in_bytes":0,"data_out_bytes":0,"duration_ms":0,"token_count":null,"sequence_position":1}"#;

    vigil()
        .args(["detect", "--store", store_dir.path().to_str().unwrap()])
        .write_stdin(event)
        .assert()
        .success()
        .stderr(predicate::str::contains("Detection Summary"));
}
