use std::fs;
use std::io::{self, BufRead};
use std::path::PathBuf;

use clap::{Parser, Subcommand};

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

    let files = if path.is_dir() {
        let mut jsonl_files: Vec<PathBuf> = fs::read_dir(&path)?
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
        jsonl_files
    } else {
        vec![path]
    };

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
