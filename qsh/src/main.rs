use anyhow::Result;
use clap::{Parser, Subcommand};
use colored::Colorize;
use std::io::{self, BufRead, Write};
use std::process::Command;

use qsh::bridge::{Bridge, InferenceRequest, LlamaCppBridge, PythonBridge, RustBridge};
use qsh::config::Config;
use qsh::history::History;
use qsh::safety;

#[derive(Parser)]
#[command(name = "qsh")]
#[command(about = "Qwen Shell: AI Coreutils for the modern terminal", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// The natural-language prompt (English to Bash). Quotes are optional.
    #[arg(trailing_var_arg = true, num_args = 0..)]
    prompt: Vec<String>,

    /// The engine to use (python, rust, or llamacpp)
    #[arg(short, long)]
    engine: Option<Engine>,

    /// The model ID to use from HuggingFace
    #[arg(short, long)]
    model: Option<String>,

    /// Clear the chat history
    #[arg(long)]
    clear_history: bool,
}

#[derive(clap::ValueEnum, Clone, Debug, PartialEq)]
enum Engine {
    Python,
    Rust,
    LlamaCpp,
}

#[derive(Subcommand)]
enum Commands {
    /// The Semantic Text Filter: Acts as a smart pipe for text data.
    Filter {
        /// The query to filter the text (e.g., "is this about a due date?")
        query: String,
    },
    /// The Vision Filter: Acts as a smart pipe for image data.
    Vision {
        /// The query to analyze the images (e.g., "is this a screenshot of software code?")
        query: String,
    },
    /// Fine-tune the model with LoRA based on your execution history.
    Lora,
}

fn main() -> Result<()> {
    colored::control::set_override(true);
    let cli = Cli::parse();
    let config = Config::load();
    let history = History::open()?;

    if cli.clear_history {
        history.clear()?;
        println!("History cleared.");
        return Ok(());
    }

    let engine = cli.engine.unwrap_or(match config.default_engine.as_str() {
        "rust" => Engine::Rust,
        "llamacpp" => Engine::LlamaCpp,
        _ => Engine::Python,
    });

    let model_id = cli.model.as_deref().or(Some(&config.default_model));

    let mut bridge: Box<dyn Bridge> = match engine {
        Engine::Python => Box::new(PythonBridge::spawn(model_id)?),
        Engine::Rust => Box::new(RustBridge::spawn(model_id)?),
        Engine::LlamaCpp => Box::new(LlamaCppBridge::spawn(&config)?),
    };

    match cli.command {
        Some(Commands::Lora) => {
            let request = InferenceRequest {
                mode: "lora".to_string(),
                query: None,
                path: Some(History::get_path().to_string_lossy().to_string()),
                text: None,
                prompt: None,
                command: None,
                history: None,
            };
            println!("{}", "Starting LoRA fine-tuning...".bold().cyan());
            bridge.query_text(&request, true)?;
            println!("{}", "\nFine-tuning complete!".bold().green());
        }
        Some(Commands::Filter { query }) => {
            let stdin = io::stdin();
            for line in stdin.lock().lines() {
                let line = line?;
                if line.trim().is_empty() {
                    continue;
                }
                let request = InferenceRequest {
                    mode: "filter".to_string(),
                    query: Some(query.clone()),
                    path: None,
                    text: Some(line.clone()),
                    prompt: None,
                    command: None,
                    history: None,
                };
                if bridge.query_bool(&request)? {
                    println!("{}", line);
                }
            }
        }
        Some(Commands::Vision { query }) => {
            let stdin = io::stdin();
            for path in stdin.lock().lines() {
                let path = path?;
                let path = path.trim().to_string();
                if path.is_empty() {
                    continue;
                }
                if !std::path::Path::new(&path).exists() {
                    continue;
                }
                let request = InferenceRequest {
                    mode: "vision".to_string(),
                    query: Some(query.clone()),
                    path: Some(path.clone()),
                    text: None,
                    prompt: None,
                    command: None,
                    history: None,
                };
                if bridge.query_bool(&request)? {
                    println!("{}", path);
                }
            }
        }
        None => {
            let prompt = cli.prompt.join(" ");
            if prompt.is_empty() {
                println!("Use --help for usage instructions.");
                return Ok(());
            }

            let recent = history.get_recent_messages(10)?;
            let request = InferenceRequest {
                mode: "bash".to_string(),
                query: None,
                path: None,
                text: None,
                prompt: Some(prompt.clone()),
                command: None,
                history: Some(recent),
            };

            print!("\nSuggested Command: ");
            io::stdout().flush()?;
            let command = bridge.query_text(&request, true)?;
            println!();

            if command.is_empty() {
                println!("Error: Model failed to generate a command.");
                return Ok(());
            }

            history.add_message("user", &prompt, None)?;
            history.add_message("assistant", &command, None)?;

            if config.safety_check && !safety::check_safety(&command) {
                safety::print_warning(&command);
                print!(
                    "{}",
                    "Are you absolutely sure you want to proceed? [y/N] "
                        .bold()
                        .red()
                );
                io::stdout().flush()?;
                let mut confirm = String::new();
                io::stdin().read_line(&mut confirm)?;
                if !confirm.trim().to_lowercase().starts_with('y') {
                    history.update_last_outcome("abort")?;
                    println!("Aborted.");
                    return Ok(());
                }
            }

            loop {
                print!("[E]xecute, [e]xplain, [a]bort? ");
                io::stdout().flush()?;
                let mut input = String::new();
                io::stdin().read_line(&mut input)?;
                let choice = input.trim();

                if choice == "e" {
                    let req = InferenceRequest {
                        mode: "explain".to_string(),
                        query: None,
                        path: None,
                        text: None,
                        prompt: None,
                        command: Some(command.clone()),
                        history: None,
                    };
                    println!("\n> Explanation:");
                    bridge.query_text(&req, true)?;
                    println!("\n");
                    continue;
                } else if choice == "E" || choice.is_empty() {
                    println!("Executing: {}", command);
                    history.update_last_outcome("execute")?;
                    Command::new("bash")
                        .arg("-c")
                        .arg(&command)
                        .spawn()?
                        .wait()?;
                    break;
                } else if choice.to_lowercase() == "a" {
                    history.update_last_outcome("abort")?;
                    println!("Aborted.");
                    break;
                } else {
                    println!("Invalid option.");
                }
            }
        }
    }
    Ok(())
}
