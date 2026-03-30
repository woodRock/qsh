use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use std::io::{self, BufRead, Write};
use std::process::{Child, Command, Stdio};

#[derive(Parser)]
#[command(name = "qsh")]
#[command(about = "Qwen Shell: AI Coreutils for the modern terminal", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// The commander prompt (English to Bash)
    prompt: Option<String>,

    /// The engine to use (python or rust)
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

use colored::Colorize;
use qsh::config::Config;
use qsh::history::History;
use qsh::model;
use qsh::safety;

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
    #[serde(skip_serializing_if = "Option::is_none")]
    history: Option<Vec<(String, String)>>,
}

#[derive(Deserialize, Debug)]
struct InferenceResponse {
    result: Option<bool>,
    text: Option<String>,
    error: Option<String>,
    info: Option<String>,
    chunk: Option<String>,
}

struct PythonBridge {
    child: Child,
    reader: io::BufReader<std::process::ChildStdout>,
}

impl PythonBridge {
    fn spawn(model_id: Option<&str>) -> Result<Self> {
        let exe_path = std::env::current_exe()?;
        let exe_dir = exe_path.parent().context("Failed to get exe directory")?;

        let mut python_path = exe_dir.join("qenv/bin/python3");
        let mut inference_path = exe_dir.join("src/inference.py");

        // First, check if we're in a workspace and prefer that inference.py
        if let Some(project_root) = exe_dir.parent().and_then(|p| p.parent()).and_then(|p| p.parent()) {
             let ws_inference = project_root.join("qsh/src/inference.py");
             let ws_python = project_root.join("qenv/bin/python3");
             if ws_inference.exists() && ws_python.exists() {
                 python_path = ws_python;
                 inference_path = ws_inference;
             }
        }

        if !python_path.exists() {
            python_path = exe_dir
                .parent()
                .context("Failed to get parent dir")?
                .join("qenv/bin/python3");
            inference_path = exe_dir
                .parent()
                .context("Failed to get parent dir")?
                .join("src/inference.py");
        }

        if !python_path.exists() {
            // Check for qsh/src/inference.py from workspace root
            // exe_dir: qsh/target/debug
            // parent: qsh/target
            // parent.parent: qsh
            // parent.parent.parent: project root
            if let Some(project_root) = exe_dir
                .parent()
                .and_then(|p| p.parent())
                .and_then(|p| p.parent())
            {
                python_path = project_root.join("qenv/bin/python3");
                inference_path = project_root.join("qsh/src/inference.py");
            }
        }

        if !python_path.exists() {
            python_path = std::path::PathBuf::from("qenv/bin/python3");
            inference_path = std::path::PathBuf::from("src/inference.py");
        }

        if !python_path.exists() {
            python_path = std::path::PathBuf::from("/Users/woodj/Desktop/qsh/qenv/bin/python3");
            inference_path =
                std::path::PathBuf::from("/Users/woodj/Desktop/qsh/qsh/src/inference.py");
        }

        let mut cmd = Command::new(&python_path);
        cmd.arg(&inference_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped());

        if let Some(id) = model_id {
            cmd.env("QSH_MODEL", id);
        }

        let mut child = cmd.spawn()
            .context(format!("Failed to start Python inference script at {:?}. Ensure transformers, torch, qwen-vl-utils are installed in qenv.", inference_path))?;

        let stdout = child.stdout.take().context("Failed to capture stdout")?;
        let mut reader = io::BufReader::new(stdout);

        loop {
            let mut line = String::new();
            if reader.read_line(&mut line)? == 0 {
                break;
            }
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
}

struct RustBridge {
    model: model::ModelForCausalLM,
    tokenizer: tokenizers::Tokenizer,
    device: candle_core::Device,
    _model_id: String,
}

impl RustBridge {
    fn spawn(model_id: Option<&str>) -> Result<Self> {
        use hf_hub::{Repo, api::sync::Api};
        let device = {
            #[cfg(target_vendor = "apple")]
            {
                if candle_core::utils::metal_is_available() {
                    candle_core::Device::new_metal(0)?
                } else {
                    candle_core::Device::Cpu
                }
            }
            #[cfg(not(target_vendor = "apple"))]
            {
                candle_core::Device::Cpu
            }
        };

        let id = model_id.unwrap_or("Qwen/Qwen3.5-0.8B");
        eprintln!("Loading Rust model ({}) onto {:?}...", id, device);

        let api = Api::new()?;
        let repo = api.repo(Repo::model(id.to_string()));

        let tokenizer_filename = repo.get("tokenizer.json")?;
        let tokenizer =
            tokenizers::Tokenizer::from_file(tokenizer_filename).map_err(anyhow::Error::msg)?;

        let config_filename = repo.get("config.json")?;
        let config_str = std::fs::read_to_string(config_filename)?;
        let config: model::Config = serde_json::from_str(&config_str)?;

        let mut weights_filenames = vec![];
        if let Ok(index_file) = repo.get("model.safetensors.index.json") {
            let index_str = std::fs::read_to_string(index_file)?;
            let index: serde_json::Value = serde_json::from_str(&index_str)?;
            if let Some(weight_map) = index.get("weight_map").and_then(|m| m.as_object()) {
                let mut unique_files = std::collections::HashSet::new();
                for file in weight_map.values() {
                    if let Some(file_str) = file.as_str() {
                        unique_files.insert(file_str.to_string());
                    }
                }
                for file in unique_files {
                    weights_filenames.push(repo.get(&file)?);
                }
            }
        } else {
            match repo.get("model.safetensors") {
                Ok(f) => weights_filenames.push(f),
                Err(_) => {
                    weights_filenames
                        .push(repo.get("model.safetensors-00001-of-00001.safetensors")?);
                }
            }
        }

        let vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(
                &weights_filenames,
                candle_core::DType::BF16,
                &device,
            )?
        };

        let model = model::ModelForCausalLM::new(&config, vb)?;

        Ok(Self {
            model,
            tokenizer,
            device,
            _model_id: id.to_string(),
        })
    }

    fn generate(&mut self, prompt: &str, max_tokens: usize, stream: bool) -> Result<String> {
        let mut tokens = self
            .tokenizer
            .encode(prompt, true)
            .map_err(anyhow::Error::msg)?
            .get_ids()
            .to_vec();
        let mut generated_text = String::new();

        for i in 0..max_tokens {
            let input = candle_core::Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, None, 0)?;
            let logits = logits.squeeze(0)?;

            let next_token = logits.argmax(0)?.to_scalar::<u32>()?;

            if next_token
                == self
                    .tokenizer
                    .get_vocab(true)
                    .get("<|endoftext|>")
                    .copied()
                    .unwrap_or(0)
                || next_token
                    == self
                        .tokenizer
                        .get_vocab(true)
                        .get("<|im_end|>")
                        .copied()
                        .unwrap_or(0)
            {
                break;
            }

            tokens.push(next_token);
            let decoded = self
                .tokenizer
                .decode(&[next_token], true)
                .map_err(anyhow::Error::msg)?;

            if stream {
                print!("{}", decoded);
                io::stdout().flush()?;
            }

            generated_text.push_str(&decoded);

            if i == 0 {
                self.model.clear_kv_cache();
            }
        }

        self.model.clear_kv_cache();

        let mut text = generated_text.trim().to_string();
        if let Some(_start) = text.find("<think>")
            && let Some(end) = text.find("</think>")
        {
            text = text[end + 8..].trim().to_string();
        }

        if text.starts_with('`') && text.ends_with('`') {
            text = text[1..text.len() - 1].trim().to_string();
        }

        Ok(text)
    }
}

struct LlamaCppBridge {
    child: Option<Child>,
    url: String,
}

impl LlamaCppBridge {
    fn spawn(config: &Config) -> Result<Self> {
        let url = config.llama_cpp.server_url.clone();

        // Check if server is already running
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(2))
            .build()?;

        if let Ok(resp) = client.get(format!("{}/health", url)).send() {
            if resp.status().is_success() {
                eprintln!("Connected to existing llama-server at {}", url);
                return Ok(Self { child: None, url });
            }
        }

        // If not running, attempt to spawn
        if let Some(bin) = &config.llama_cpp.server_binary {
            let model = config.llama_cpp.model_path.as_ref().context("model_path is required for LlamaCpp engine if server is not running")?;
            
            eprintln!("Starting persistent llama-server with TurboQuant+ optimizations...");
            let mut cmd = Command::new(bin);
            cmd.arg("-m").arg(model)
               .arg("--cache-type-k").arg(&config.llama_cpp.turbo_k)
               .arg("--cache-type-v").arg(&config.llama_cpp.turbo_v)
               .arg("--port").arg(url.split(':').last().unwrap_or("8080"));

            if config.llama_cpp.flash_attn {
                cmd.arg("-fa").arg("on");
            }

            // Disconnect from terminal to allow persistence
            cmd.stdout(Stdio::null());
            cmd.stderr(Stdio::null());
            cmd.stdin(Stdio::null());

            // On Unix-like systems, we want the process to survive after qsh exits
            #[cfg(unix)]
            {
                use std::os::unix::process::CommandExt;
                unsafe {
                    cmd.pre_exec(|| {
                        // Create a new session to detach from the terminal
                        if libc::setsid() == -1 {
                            return Err(std::io::Error::last_os_error());
                        }
                        Ok(())
                    });
                }
            }

            let child = cmd.spawn().context(format!("Failed to start llama-server at {}", bin))?;
            
            // Wait for health check
            eprintln!("Waiting for llama-server to initialize (this can take 30-60s for large models)...");
            let start = std::time::Instant::now();
            loop {
                // ... rest of health check loop
                if let Ok(resp) = client.get(format!("{}/health", url)).send() {
                    if resp.status().is_success() {
                        eprintln!("llama-server is ready!");
                        break;
                    }
                }
                if start.elapsed().as_secs() > 120 {
                    anyhow::bail!("llama-server failed to start within 120 seconds.");
                }
                std::thread::sleep(std::time::Duration::from_secs(2));
            }

            Ok(Self { child: Some(child), url })
        } else {
            anyhow::bail!("No llama-server running at {} and llama_cpp.server_binary is not set in config.", url);
        }
    }
}

impl Bridge for LlamaCppBridge {
    fn query_bool(&mut self, request: &InferenceRequest) -> Result<bool> {
        let prompt = match request.mode.as_str() {
            "filter" => format!(
                "<|im_start|>system\nYou are a text filter. Answer YES or NO.<|im_end|>\n<|im_start|>user\nIs the following line related to '{}'?\nLine: {}\nAnswer:<|im_end|>\n<|im_start|>assistant\n",
                request.query.as_ref().unwrap(),
                request.text.as_ref().unwrap()
            ),
            "vision" => anyhow::bail!("Vision is not yet supported in LlamaCpp engine"),
            _ => anyhow::bail!("Unsupported bool mode"),
        };

        let client = reqwest::blocking::Client::new();
        let body = serde_json::json!({
            "prompt": prompt,
            "n_predict": 5,
            "stop": ["<|im_end|>"]
        });

        let resp = client.post(format!("{}/completion", self.url))
            .json(&body)
            .send()?;
        
        let json: serde_json::Value = resp.json()?;
        let content = json["content"].as_str().unwrap_or("");
        
        Ok(content.to_uppercase().contains("YES"))
    }

    fn query_text(&mut self, request: &InferenceRequest, stream: bool) -> Result<String> {
        let (prompt, max_tokens) = match request.mode.as_str() {
            "bash" => {
                let mut p = String::from(
                    "<|im_start|>system\nYou are a Unix shell expert. Provide the valid Bash command for the user's request. Output ONLY the command, no reasoning, no explanation.<|im_end|>\n",
                );
                if let Some(hist) = &request.history {
                    for (role, content) in hist {
                        p.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", role, content));
                    }
                }
                p.push_str(&format!(
                    "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                    request.prompt.as_ref().unwrap()
                ));
                (p, 128)
            }
            "explain" => (
                format!(
                    "<|im_start|>system\nYou are a Unix shell expert.<|im_end|>\n<|im_start|>user\nExplain this Bash command briefly: {}<|im_end|>\n<|im_start|>assistant\n",
                    request.command.as_ref().unwrap()
                ),
                200,
            ),
            _ => anyhow::bail!("Unsupported text mode"),
        };

        let client = reqwest::blocking::Client::new();
        let body = serde_json::json!({
            "prompt": prompt,
            "n_predict": max_tokens,
            "stop": ["<|im_end|>"],
            "stream": stream
        });

        let resp = client.post(format!("{}/completion", self.url))
            .json(&body)
            .send()?;

        if stream {
            let mut full_text = String::new();
            let reader = io::BufReader::new(resp);
            for line in reader.lines() {
                let line = line?;
                if line.starts_with("data: ") {
                    let json_str = &line[6..];
                    if let Ok(json) = serde_json::from_str::<serde_json::Value>(json_str) {
                        if let Some(content) = json["content"].as_str() {
                            print!("{}", content);
                            io::stdout().flush()?;
                            full_text.push_str(content);
                        }
                    }
                }
            }
            
            // Clean up thinking tags
            let mut text = full_text.trim().to_string();
            if let Some(_start) = text.find("<think>")
                && let Some(end) = text.find("</think>")
            {
                text = text[end + 8..].trim().to_string();
            }
            Ok(text)
        } else {
            let json: serde_json::Value = resp.json()?;
            let mut text = json["content"].as_str().unwrap_or("").trim().to_string();
            if let Some(_start) = text.find("<think>")
                && let Some(end) = text.find("</think>")
            {
                text = text[end + 8..].trim().to_string();
            }
            Ok(text)
        }
    }
}

trait Bridge {
    fn query_bool(&mut self, request: &InferenceRequest) -> Result<bool>;
    fn query_text(&mut self, request: &InferenceRequest, stream: bool) -> Result<String>;
}

impl Bridge for PythonBridge {
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

    fn query_text(&mut self, request: &InferenceRequest, stream: bool) -> Result<String> {
        let stdin = self.child.stdin.as_mut().context("Failed to open stdin")?;
        let json = serde_json::to_string(request)?;
        writeln!(stdin, "{}", json)?;
        stdin.flush()?;

        let mut full_text = String::new();
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
                if let Some(chunk) = response.chunk {
                    if stream {
                        print!("{}", chunk);
                        io::stdout().flush()?;
                    }
                    full_text.push_str(&chunk);
                    continue;
                }
                if let Some(txt) = response.text {
                    let mut text = txt.trim().to_string();
                    if let Some(_start) = text.find("<think>")
                        && let Some(end) = text.find("</think>")
                    {
                        text = text[end + 8..].trim().to_string();
                    }
                    return Ok(text);
                }
            }
        }
    }
}

impl Bridge for RustBridge {
    fn query_bool(&mut self, request: &InferenceRequest) -> Result<bool> {
        let prompt = match request.mode.as_str() {
            "filter" => {
                let mut p = String::from(
                    "<|im_start|>system\nYou are a text filter. Answer YES or NO.<|im_end|>\n",
                );
                if let Some(hist) = &request.history {
                    for (role, content) in hist {
                        p.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", role, content));
                    }
                }
                p.push_str(&format!(
                    "<|im_start|>user\nIs the following line related to '{}'?\nLine: {}\nAnswer:<|im_end|>\n<|im_start|>assistant\n",
                    request.query.as_ref().unwrap(),
                    request.text.as_ref().unwrap()
                ));
                p
            }
            "vision" => {
                format!(
                    "<|im_start|>system\nYou are a vision assistant. Answer YES or NO.<|im_end|>\n<|im_start|>user\nAnalyze the image at path '{}'. Question: {} Answer YES or NO.<|im_end|>\n<|im_start|>assistant\n",
                    request.path.as_ref().unwrap(),
                    request.query.as_ref().unwrap()
                )
            }
            _ => anyhow::bail!("Unsupported bool mode"),
        };

        let response = self.generate(&prompt, 5, false)?;
        Ok(response.to_uppercase().contains("YES"))
    }

    fn query_text(&mut self, request: &InferenceRequest, stream: bool) -> Result<String> {
        let (prompt, max_tokens) = match request.mode.as_str() {
            "bash" => {
                let mut p = String::from(
                    "<|im_start|>system\nYou are a Unix shell expert. Provide the valid Bash command for the user's request. Output ONLY the command, no reasoning, no explanation.<|im_end|>\n",
                );
                if let Some(hist) = &request.history {
                    for (role, content) in hist {
                        p.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", role, content));
                    }
                }
                p.push_str(&format!(
                    "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                    request.prompt.as_ref().unwrap()
                ));
                (p, 128)
            }
            "explain" => (
                format!(
                    "<|im_start|>system\nYou are a Unix shell expert.<|im_end|>\n<|im_start|>user\nExplain this Bash command briefly: {}<|im_end|>\n<|im_start|>assistant\n",
                    request.command.as_ref().unwrap()
                ),
                200,
            ),
            _ => anyhow::bail!("Unsupported text mode"),
        };

        self.generate(&prompt, max_tokens, stream)
    }
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
                    history: None, // Filter doesn't use history for performance
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
                let path = path.trim();
                if path.is_empty() {
                    continue;
                }

                // Check if file exists to avoid crashes
                if !std::path::Path::new(path).exists() {
                    continue;
                }

                let request = InferenceRequest {
                    mode: "vision".to_string(),
                    query: Some(query.clone()),
                    path: Some(path.to_string()),
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
            if let Some(prompt) = cli.prompt {
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
            } else {
                println!("Use --help for usage instructions.");
            }
        }
    }
    Ok(())
}
