use anyhow::{Context, Result};
use std::io::{self, BufRead, IsTerminal, Write};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use super::{Bridge, InferenceRequest, InferenceResponse};

const SPINNER_FRAMES: [char; 10] = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];

/// Ticks a "<frame> <label> (<elapsed>s)" line on stderr until told to stop,
/// so model loading (which can take 30-60s for larger models) doesn't look
/// like qsh has hung. Only runs when stderr is an interactive terminal.
struct LoadingSpinner {
    label: Arc<Mutex<String>>,
    stop: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl LoadingSpinner {
    fn start(initial_label: &str) -> Self {
        let label = Arc::new(Mutex::new(initial_label.to_string()));
        let stop = Arc::new(AtomicBool::new(false));

        let handle = if io::stderr().is_terminal() {
            let label = label.clone();
            let stop = stop.clone();
            Some(std::thread::spawn(move || {
                let start = Instant::now();
                let mut i = 0usize;
                while !stop.load(Ordering::Relaxed) {
                    let text = label.lock().unwrap().clone();
                    eprint!(
                        "\r\x1b[2K{} {} ({}s)",
                        SPINNER_FRAMES[i % SPINNER_FRAMES.len()],
                        text,
                        start.elapsed().as_secs()
                    );
                    let _ = io::stderr().flush();
                    i += 1;
                    std::thread::sleep(std::time::Duration::from_millis(120));
                }
                eprint!("\r\x1b[2K");
                let _ = io::stderr().flush();
            }))
        } else {
            eprintln!("{}", initial_label);
            None
        };

        Self {
            label,
            stop,
            handle,
        }
    }

    fn set_label(&self, text: &str) {
        if self.handle.is_some() {
            *self.label.lock().unwrap() = text.to_string();
        } else {
            eprintln!("{}", text);
        }
    }

    fn stop(mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

pub struct PythonBridge {
    child: Child,
    reader: io::BufReader<std::process::ChildStdout>,
}

impl PythonBridge {
    pub fn spawn(model_id: Option<&str>) -> Result<Self> {
        let exe_path = std::env::current_exe()?;
        let exe_dir = exe_path.parent().context("Failed to get exe directory")?;

        let mut python_path = exe_dir.join("qenv/bin/python3");
        let mut inference_path = exe_dir.join("src/inference.py");

        // Prefer workspace-root paths when running from a dev build
        if let Some(project_root) = exe_dir
            .parent()
            .and_then(|p| p.parent())
            .and_then(|p| p.parent())
        {
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

        if !python_path.exists()
            && let Some(project_root) = exe_dir
                .parent()
                .and_then(|p| p.parent())
                .and_then(|p| p.parent())
        {
            python_path = project_root.join("qenv/bin/python3");
            inference_path = project_root.join("qsh/src/inference.py");
        }

        if !python_path.exists() {
            python_path = std::path::PathBuf::from("qenv/bin/python3");
            inference_path = std::path::PathBuf::from("src/inference.py");
        }

        if !python_path.exists() {
            anyhow::bail!(
                "Could not find the Python inference environment (qenv/bin/python3). \
                 Run ./setup.sh from the repo root, or reinstall qsh."
            );
        }

        let mut cmd = Command::new(&python_path);
        cmd.arg(&inference_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped());

        if let Some(id) = model_id {
            cmd.env("QSH_MODEL", id);
        }

        let mut child = cmd.spawn().context(format!(
            "Failed to start Python inference script at {:?}. \
             Ensure transformers, torch, qwen-vl-utils are installed in qenv.",
            inference_path
        ))?;

        let stdout = child.stdout.take().context("Failed to capture stdout")?;
        let mut reader = io::BufReader::new(stdout);

        // Wait for the model-loaded signal before returning. Model loading
        // can take 30-60s for the larger Qwen3.5 sizes, so show a spinner
        // rather than leaving the terminal looking stuck.
        let spinner = LoadingSpinner::start("Starting inference engine...");
        loop {
            let mut line = String::new();
            if reader.read_line(&mut line)? == 0 {
                break;
            }
            if let Ok(resp) = serde_json::from_str::<InferenceResponse>(&line) {
                if let Some(info) = resp.info {
                    spinner.set_label(&info);
                    if info == "Model loaded successfully!" {
                        break;
                    }
                }
                if let Some(err) = resp.error {
                    spinner.stop();
                    anyhow::bail!("Python Init Error: {}", err);
                }
            } else {
                spinner.set_label(line.trim());
            }
        }
        spinner.stop();

        Ok(Self { child, reader })
    }
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
        let mut inside_think = false;
        let mut buffer = String::new();
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
                    full_text.push_str(&chunk);

                    if stream {
                        buffer.push_str(&chunk);
                        loop {
                            if !inside_think {
                                if let Some(start) = buffer.find("<think>") {
                                    print!("{}", &buffer[..start]);
                                    io::stdout().flush()?;
                                    inside_think = true;
                                    buffer = buffer[start + 7..].to_string();
                                } else {
                                    let mut print_until = buffer.len();
                                    if let Some(last_lt) = buffer.rfind('<')
                                        && "<think>".starts_with(&buffer[last_lt..])
                                    {
                                        print_until = last_lt;
                                    }
                                    if print_until > 0 {
                                        print!("{}", &buffer[..print_until]);
                                        io::stdout().flush()?;
                                        buffer = buffer[print_until..].to_string();
                                    }
                                    break;
                                }
                            } else if let Some(end) = buffer.find("</think>") {
                                inside_think = false;
                                buffer = buffer[end + 8..].to_string();
                            } else {
                                let mut keep_from = buffer.len();
                                if let Some(last_lt) = buffer.rfind('<')
                                    && "</think>".starts_with(&buffer[last_lt..])
                                {
                                    keep_from = last_lt;
                                }
                                buffer = buffer[keep_from..].to_string();
                                break;
                            }
                        }
                    }
                    continue;
                }
                // Final text response (non-streaming path from Python)
                if let Some(text) = response.text {
                    if stream && !buffer.is_empty() && !inside_think {
                        print!("{}", buffer);
                        io::stdout().flush()?;
                    }
                    return Ok(text);
                }
            }
        }
    }
}
