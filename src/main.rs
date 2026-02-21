use std::collections::HashMap;
use std::fs;
use std::io::{self, BufRead};
use std::path::PathBuf;

use clap::{Parser, Subcommand};

use vigil::baseline::builder::BaselineBuilder;
use vigil::event::BehavioralEvent;
use vigil::source::openclaw::OpenClawParser;

#[derive(Parser)]
#[command(name = "vigil", about = "Behavioral anomaly detection for AI agents")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// OpenClaw integration commands.
    Openclaw {
        #[command(subcommand)]
        command: OpenclawCommand,
    },
    /// Baseline profiling commands.
    Baseline {
        #[command(subcommand)]
        command: BaselineCommand,
    },
}

#[derive(Subcommand)]
enum OpenclawCommand {
    /// Parse session transcripts and print extracted events as JSONL.
    Parse {
        /// Path to sessions directory or a single .jsonl file.
        /// Defaults to ~/.openclaw/agents/main/sessions/
        path: Option<PathBuf>,

        /// Agent ID to tag events with.
        #[arg(long, default_value = "openclaw")]
        agent_id: String,
    },
}

#[derive(Subcommand)]
enum BaselineCommand {
    /// Build a behavioral baseline from events on stdin (JSONL).
    Build {
        /// Only build a baseline for this agent ID (default: all agents).
        #[arg(long)]
        agent_id: Option<String>,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Command::Openclaw { command } => match command {
            OpenclawCommand::Parse { path, agent_id } => {
                if let Err(e) = run_parse(path, &agent_id) {
                    eprintln!("error: {e}");
                    std::process::exit(1);
                }
            }
        },
        Command::Baseline { command } => match command {
            BaselineCommand::Build { agent_id } => {
                if let Err(e) = run_baseline_build(agent_id.as_deref()) {
                    eprintln!("error: {e}");
                    std::process::exit(1);
                }
            }
        },
    }
}

fn run_parse(path: Option<PathBuf>, agent_id: &str) -> Result<(), Box<dyn std::error::Error>> {
    let path = match path {
        Some(p) => p,
        None => {
            let home = dirs::home_dir().ok_or("could not determine home directory")?;
            home.join(".openclaw/agents/main/sessions")
        }
    };

    let files = find_jsonl_files(&path)?;

    if files.is_empty() {
        eprintln!("no .jsonl files found");
        return Ok(());
    }

    let mut parser = OpenClawParser::new(agent_id);
    let mut total_events = 0;

    for file in &files {
        eprintln!("parsing: {}", file.display());
        let reader = io::BufReader::new(fs::File::open(file)?);

        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }

            match parser.parse_line(&line) {
                Ok(events) => {
                    for event in &events {
                        println!("{}", serde_json::to_string(event)?);
                        total_events += 1;
                    }
                }
                Err(e) => {
                    eprintln!("warning: skipping unparseable line: {e}");
                }
            }
        }
    }

    eprintln!("done: {total_events} events from {} file(s)", files.len());
    Ok(())
}

fn run_baseline_build(filter_agent_id: Option<&str>) -> Result<(), Box<dyn std::error::Error>> {
    let stdin = io::stdin();
    let reader = stdin.lock();

    let mut builders: HashMap<String, BaselineBuilder> = HashMap::new();
    let mut line_count: u64 = 0;
    let mut skipped: u64 = 0;

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        let event: BehavioralEvent = match serde_json::from_str(&line) {
            Ok(e) => e,
            Err(e) => {
                skipped += 1;
                eprintln!("warning: skipping line {}: {e}", line_count + 1);
                continue;
            }
        };

        if let Some(filter) = filter_agent_id
            && event.agent_id != filter
        {
            continue;
        }

        let builder = builders
            .entry(event.agent_id.clone())
            .or_insert_with(|| BaselineBuilder::new(&event.agent_id));

        builder.process(&event);
        line_count += 1;
    }

    if builders.is_empty() {
        eprintln!("no events processed");
        return Ok(());
    }

    for (agent_id, builder) in builders {
        let baseline = builder.finish();
        eprintln!("\n--- Baseline: {agent_id} ---");
        eprintln!("  sessions:  {}", baseline.session_count);
        eprintln!("  events:    {}", baseline.event_count);
        eprintln!("  tools:     {}", baseline.tool_stats.len());
        eprintln!("  resources: {}", baseline.known_resources.len());
        eprintln!("  bigrams:   {}", baseline.bigrams.len());
        eprintln!(
            "  rate:      {:.2} calls/min (mean, n={})",
            baseline.rate_stats.mean, baseline.rate_stats.count
        );
        eprintln!("  first:     {}", baseline.first_seen);
        eprintln!("  last:      {}", baseline.last_updated);

        for (tool, stats) in &baseline.tool_stats {
            eprintln!("  tool {tool}: {} calls", stats.call_count);
        }

        // Output the baseline as JSON to stdout for downstream consumption.
        println!("{}", serde_json::to_string(&baseline)?);
    }

    if skipped > 0 {
        eprintln!("\nskipped {skipped} unparseable line(s)");
    }

    Ok(())
}

fn find_jsonl_files(path: &PathBuf) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    if path.is_dir() {
        let mut jsonl_files: Vec<PathBuf> = fs::read_dir(path)?
            .filter_map(|entry| {
                let entry = entry.ok()?;
                let path = entry.path();
                if path.extension().is_some_and(|ext| ext == "jsonl") {
                    Some(path)
                } else {
                    None
                }
            })
            .collect();
        jsonl_files.sort();
        Ok(jsonl_files)
    } else {
        Ok(vec![path.clone()])
    }
}
