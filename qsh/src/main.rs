use clap::{Parser, Subcommand};
use anyhow::{Result, Context};
use std::io::{self, BufRead, Write};
use std::process::{Command, Stdio, Child};
use serde::{Serialize, Deserialize};

#[derive(Parser)]
#[command(name = "qsh")]
#[command(about = "Qwen Shell: AI Coreutils for the modern terminal", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// The commander prompt (English to Bash)
    prompt: Option<String>,
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
}

#[derive(Serialize)]
struct InferenceRequest {
    mode: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    query: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    command: Option<String>,
}

#[derive(Deserialize, Debug)]
struct InferenceResponse {
    result: Option<bool>,
    text: Option<String>,
    error: Option<String>,
    info: Option<String>,
}

struct PythonBridge {
    child: Child,
    reader: io::BufReader<std::process::ChildStdout>,
}

impl PythonBridge {
    fn spawn() -> Result<Self> {
        let mut child = Command::new("/Users/woodj/Desktop/qsh/qenv/bin/python3")
            .arg("src/inference.py")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .context("Failed to start Python inference script. Ensure transformers, torch, qwen-vl-utils are installed in qenv.")?;
        
        let stdout = child.stdout.take().context("Failed to capture stdout")?;
        let mut reader = io::BufReader::new(stdout);
        
        loop {
            let mut line = String::new();
            if reader.read_line(&mut line)? == 0 { break; }
            if let Ok(resp) = serde_json::from_str::<InferenceResponse>(&line) {
                if let Some(info) = resp.info {
                    eprintln!("Info: {}", info);
                    if info == "Model loaded successfully!" {
                        break;
                    }
                }
                if let Some(err) = resp.error {
                    anyhow::bail!("Python Init Error: {}", err);
                }
            } else {
                eprint!("{}", line);
            }
        }

        Ok(Self { child, reader })
    }

    fn query_bool(&mut self, request: &InferenceRequest) -> Result<bool> {
        let stdin = self.child.stdin.as_mut().context("Failed to open stdin")?;
        let json = serde_json::to_string(request)?;
        writeln!(stdin, "{}", json)?;
        stdin.flush()?;

        loop {
            let mut line = String::new();
            if self.reader.read_line(&mut line)? == 0 {
                anyhow::bail!("Python process exited unexpectedly");
            }
            
            if let Ok(response) = serde_json::from_str::<InferenceResponse>(&line) {
                if let Some(err) = response.error {
                    anyhow::bail!("Python Error: {}", err);
                }
                if let Some(info) = response.info {
                    eprintln!("Info: {}", info);
                    continue;
                }
                if let Some(res) = response.result {
                    return Ok(res);
                }
            }
        }
    }

    fn query_text(&mut self, request: &InferenceRequest) -> Result<String> {
        let stdin = self.child.stdin.as_mut().context("Failed to open stdin")?;
        let json = serde_json::to_string(request)?;
        writeln!(stdin, "{}", json)?;
        stdin.flush()?;

        loop {
            let mut line = String::new();
            if self.reader.read_line(&mut line)? == 0 {
                anyhow::bail!("Python process exited unexpectedly");
            }
            
            if let Ok(response) = serde_json::from_str::<InferenceResponse>(&line) {
                if let Some(err) = response.error {
                    anyhow::bail!("Python Error: {}", err);
                }
                if let Some(info) = response.info {
                    eprintln!("Info: {}", info);
                    continue;
                }
                if let Some(txt) = response.text {
                    return Ok(txt);
                }
            }
        }
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Filter { query }) => {
            let mut bridge = PythonBridge::spawn()?;
            let stdin = io::stdin();
            for line in stdin.lock().lines() {
                let line = line?;
                if line.trim().is_empty() { continue; }
                let request = InferenceRequest {
                    mode: "filter".to_string(),
                    query: Some(query.clone()),
                    path: None,
                    text: Some(line.clone()),
                    prompt: None,
                    command: None,
                };
                if bridge.query_bool(&request)? {
                    println!("{}", line);
                }
            }
        }
        Some(Commands::Vision { query }) => {
            let mut bridge = PythonBridge::spawn()?;
            let stdin = io::stdin();
            for path in stdin.lock().lines() {
                let path = path?;
                if path.trim().is_empty() { continue; }
                let request = InferenceRequest {
                    mode: "vision".to_string(),
                    query: Some(query.clone()),
                    path: Some(path.clone()),
                    text: None,
                    prompt: None,
                    command: None,
                };
                if bridge.query_bool(&request)? {
                    println!("{}", path);
                }
            }
        }
        None => {
            if let Some(prompt) = cli.prompt {
                let mut bridge = PythonBridge::spawn()?;
                let request = InferenceRequest {
                    mode: "bash".to_string(),
                    query: None,
                    path: None,
                    text: None,
                    prompt: Some(prompt),
                    command: None,
                };
                let command = bridge.query_text(&request)?;
                
                if command.is_empty() {
                    println!("Error: Model failed to generate a command.");
                    return Ok(());
                }
                
                println!("\nSuggested Command: {}", command);
                
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
                        };
                        println!("\n> Explanation:");
                        let explanation = bridge.query_text(&req)?;
                        println!("{}", explanation);
                        println!();
                        continue;
                    } else if choice == "E" || choice.is_empty() {
                        println!("Executing: {}", command);
                        Command::new("bash").arg("-c").arg(&command).spawn()?.wait()?;
                        break;
                    } else if choice.to_lowercase() == "a" {
                        println!("Aborted.");
                        break;
                    } else {
                        println!("Invalid option.");
                    }
                }
            } else {
                println!("Use --help for usage instructions.");
            }
        }
    }
    Ok(())
}
