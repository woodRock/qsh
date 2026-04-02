use anyhow::{Context, Result};
use std::io::{self, BufRead, Write};
use std::process::{Child, Command, Stdio};

use super::{Bridge, InferenceRequest, InferenceResponse};

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

        if !python_path.exists() {
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
            python_path =
                std::path::PathBuf::from("/Users/woodj/Desktop/qsh/qenv/bin/python3");
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

        let mut child = cmd.spawn().context(format!(
            "Failed to start Python inference script at {:?}. \
             Ensure transformers, torch, qwen-vl-utils are installed in qenv.",
            inference_path
        ))?;

        let stdout = child.stdout.take().context("Failed to capture stdout")?;
        let mut reader = io::BufReader::new(stdout);

        // Wait for the model-loaded signal before returning
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
                                    if let Some(last_lt) = buffer.rfind('<') {
                                        if "<think>".starts_with(&buffer[last_lt..]) {
                                            print_until = last_lt;
                                        }
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
                                if let Some(last_lt) = buffer.rfind('<') {
                                    if "</think>".starts_with(&buffer[last_lt..]) {
                                        keep_from = last_lt;
                                    }
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
