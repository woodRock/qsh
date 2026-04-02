use anyhow::{Context, Result};
use std::io::{self, BufRead, Write};
use std::process::{Child, Command, Stdio};

use super::{Bridge, InferenceRequest, strip_code_fence, strip_think_tags};
use crate::config::Config;

pub struct LlamaCppBridge {
    child: Option<Child>,
    url: String,
    client: reqwest::blocking::Client,
}

impl LlamaCppBridge {
    pub fn spawn(config: &Config) -> Result<Self> {
        let url = config.llama_cpp.server_url.clone();

        let health_client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(2))
            .build()?;

        let client = reqwest::blocking::Client::builder()
            .connection_verbose(false)
            .build()?;

        if let Ok(resp) = health_client.get(format!("{}/health", url)).send() {
            if resp.status().is_success() {
                eprintln!("Connected to existing llama-server at {}", url);
                return Ok(Self { child: None, url, client });
            }
        }

        if let Some(bin) = &config.llama_cpp.server_binary {
            let model = config
                .llama_cpp
                .model_path
                .as_ref()
                .context("model_path is required for LlamaCpp engine if server is not running")?;

            eprintln!("Starting persistent llama-server with TurboQuant+ optimizations...");
            let mut cmd = Command::new(bin);
            cmd.arg("-m")
                .arg(model)
                .arg("--cache-type-k")
                .arg(&config.llama_cpp.turbo_k)
                .arg("--cache-type-v")
                .arg(&config.llama_cpp.turbo_v)
                .arg("--port")
                .arg(url.split(':').last().unwrap_or("8080"))
                .arg("--ctx-size")
                .arg("8192")
                .arg("--threads")
                .arg(num_cpus::get().to_string());

            if let Some(mmproj) = &config.llama_cpp.mmproj_path {
                if std::path::Path::new(mmproj).exists() {
                    cmd.arg("--mmproj").arg(mmproj);
                }
            }

            if config.llama_cpp.flash_attn {
                cmd.arg("--flash-attn").arg("on");
            }

            cmd.stdout(Stdio::null());
            let log_file = std::fs::File::create("/tmp/qsh_llama_server.log")?;
            cmd.stderr(Stdio::from(log_file));
            cmd.stdin(Stdio::null());

            #[cfg(unix)]
            {
                use std::os::unix::process::CommandExt;
                unsafe {
                    cmd.pre_exec(|| {
                        if libc::setsid() == -1 {
                            return Err(std::io::Error::last_os_error());
                        }
                        Ok(())
                    });
                }
            }

            let child = cmd
                .spawn()
                .context(format!("Failed to start llama-server at {}", bin))?;

            eprintln!(
                "Waiting for llama-server to initialize (this can take 30-60s for large models)..."
            );
            let start = std::time::Instant::now();
            loop {
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

            Ok(Self { child: Some(child), url, client })
        } else {
            anyhow::bail!(
                "No llama-server running at {} and llama_cpp.server_binary is not set in config.",
                url
            );
        }
    }
}

impl Bridge for LlamaCppBridge {
    fn query_bool(&mut self, request: &InferenceRequest) -> Result<bool> {
        let mut body = serde_json::json!({
            "n_predict": 10,
            "stop": ["<|im_end|>"],
            "cache_prompt": true
        });

        match request.mode.as_str() {
            "filter" => {
                let prompt = format!(
                    "<|im_start|>system\nYou are a text filter. Answer YES or NO.<|im_end|>\n\
                     <|im_start|>user\nIs the following line related to '{}'?\nLine: {}\nAnswer:<|im_end|>\n\
                     <|im_start|>assistant\n",
                    request.query.as_ref().unwrap(),
                    request.text.as_ref().unwrap()
                );
                body["prompt"] = serde_json::Value::String(prompt);
            }
            "vision" => {
                let path = request.path.as_ref().unwrap();
                let query = request.query.as_ref().unwrap();

                let image_data = std::fs::read(path)?;
                let base64_image = base64::Engine::encode(
                    &base64::engine::general_purpose::STANDARD,
                    image_data,
                );

                let prompt = format!(
                    "<|im_start|>system\nYou are a vision assistant. Answer YES or NO.<|im_end|>\n\
                     <|im_start|>user\nAnalyze the attached image. Question: {} Answer YES or NO.<|im_end|>\n\
                     <|im_start|>assistant\n",
                    query
                );

                body["prompt"] = serde_json::Value::String(prompt);
                body["image_data"] = serde_json::json!([{
                    "data": base64_image,
                    "id": 1
                }]);
            }
            _ => anyhow::bail!("Unsupported bool mode"),
        };

        let resp = self
            .client
            .post(format!("{}/completion", self.url))
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
                    "<|im_start|>system\nYou are a Unix shell expert. Provide the valid Bash \
                     command for the user's request. Output ONLY the command, no reasoning, \
                     no explanation.<|im_end|>\n",
                );
                if let Some(hist) = &request.history {
                    for (role, content) in hist {
                        p.push_str(&format!(
                            "<|im_start|>{}\n{}<|im_end|>\n",
                            role, content
                        ));
                    }
                }
                p.push_str(&format!(
                    "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                    request.prompt.as_ref().unwrap()
                ));
                (p, 4096)
            }
            "explain" => (
                format!(
                    "<|im_start|>system\nYou are a Unix shell expert.<|im_end|>\n\
                     <|im_start|>user\nExplain this Bash command briefly: {}<|im_end|>\n\
                     <|im_start|>assistant\n",
                    request.command.as_ref().unwrap()
                ),
                200,
            ),
            _ => anyhow::bail!("Unsupported text mode"),
        };

        let body = serde_json::json!({
            "prompt": prompt,
            "n_predict": max_tokens,
            "stop": ["<|im_end|>"],
            "stream": stream,
            "cache_prompt": true
        });

        let resp = self
            .client
            .post(format!("{}/completion", self.url))
            .json(&body)
            .send()?;

        if stream {
            let mut full_text = String::new();
            let reader = io::BufReader::new(resp);
            let mut inside_think = false;
            let mut buffer = String::new();
            for line in reader.lines() {
                let line = line?;
                if line.starts_with("data: ") {
                    let json_str = &line[6..];
                    if let Ok(json) = serde_json::from_str::<serde_json::Value>(json_str) {
                        if let Some(content) = json["content"].as_str() {
                            full_text.push_str(content);
                            buffer.push_str(content);

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
                    }
                }
            }
            if !buffer.is_empty() && !inside_think {
                print!("{}", buffer);
                io::stdout().flush()?;
            }

            let text = full_text.trim().to_string();
            Ok(strip_code_fence(&strip_think_tags(&text)))
        } else {
            let json: serde_json::Value = resp.json()?;
            let text = json["content"].as_str().unwrap_or("").trim().to_string();
            Ok(strip_code_fence(&strip_think_tags(&text)))
        }
    }
}
