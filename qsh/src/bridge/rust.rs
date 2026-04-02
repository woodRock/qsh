use anyhow::Result;
use std::io::{self, Write};

use super::{Bridge, InferenceRequest, strip_code_fence, strip_think_tags};
use crate::model;

pub struct RustBridge {
    model: model::ModelForCausalLM,
    tokenizer: tokenizers::Tokenizer,
    device: candle_core::Device,
    _model_id: String,
}

impl RustBridge {
    pub fn spawn(model_id: Option<&str>) -> Result<Self> {
        use hf_hub::{api::sync::Api, Repo};

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

    pub fn generate(&mut self, prompt: &str, max_tokens: usize, stream: bool) -> Result<String> {
        let mut tokens = self
            .tokenizer
            .encode(prompt, true)
            .map_err(anyhow::Error::msg)?
            .get_ids()
            .to_vec();
        let mut generated_text = String::new();
        let mut inside_think = false;
        let mut buffer = String::new();

        for i in 0..max_tokens {
            let input_tokens = if i == 0 {
                tokens.clone()
            } else {
                vec![*tokens.last().unwrap()]
            };
            let seqlen_offset = if i == 0 { 0 } else { tokens.len() - 1 };

            let input =
                candle_core::Tensor::new(input_tokens.as_slice(), &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, None, seqlen_offset)?;
            let logits = logits.squeeze(0)?;

            let next_token = logits.argmax(0)?.to_scalar::<u32>()?;

            let vocab = self.tokenizer.get_vocab(true);
            let eot = vocab.get("<|endoftext|>").copied().unwrap_or(0);
            let im_end = vocab.get("<|im_end|>").copied().unwrap_or(0);
            if next_token == eot || next_token == im_end {
                break;
            }

            tokens.push(next_token);
            let decoded = self
                .tokenizer
                .decode(&[next_token], true)
                .map_err(anyhow::Error::msg)?;

            generated_text.push_str(&decoded);

            if stream {
                buffer.push_str(&decoded);
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

        self.model.clear_kv_cache();
        if stream && !buffer.is_empty() && !inside_think {
            print!("{}", buffer);
            io::stdout().flush()?;
        }

        let text = generated_text.trim();
        Ok(strip_code_fence(&strip_think_tags(text)))
    }
}

impl Bridge for RustBridge {
    fn query_bool(&mut self, request: &InferenceRequest) -> Result<bool> {
        let prompt = match request.mode.as_str() {
            "filter" => format!(
                "<|im_start|>system\nYou are a text filter. Answer YES or NO.<|im_end|>\n\
                 <|im_start|>user\nIs the following line related to '{}'?\nLine: {}\nAnswer:<|im_end|>\n\
                 <|im_start|>assistant\n",
                request.query.as_ref().unwrap(),
                request.text.as_ref().unwrap()
            ),
            "vision" => format!(
                "<|im_start|>system\nYou are a vision assistant. Answer YES or NO.<|im_end|>\n\
                 <|im_start|>user\nAnalyze the image at path '{}'. Question: {} Answer YES or NO.<|im_end|>\n\
                 <|im_start|>assistant\n",
                request.path.as_ref().unwrap(),
                request.query.as_ref().unwrap()
            ),
            _ => anyhow::bail!("Unsupported bool mode"),
        };

        let response = self.generate(&prompt, 5, false)?;
        Ok(response.to_uppercase().contains("YES"))
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

        self.generate(&prompt, max_tokens, stream)
    }
}
