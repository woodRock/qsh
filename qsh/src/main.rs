use clap::{Parser, Subcommand};
use anyhow::{Result, Context};
use std::io::{self, BufRead, Write};
use hf_hub::{api::sync::Api, Repo, RepoType};
use candle_core::{Device, Tensor};
use candle_transformers::generation::LogitsProcessor;

mod model;
use model::{Qwen35Model, Qwen35Config, LayerState};

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
    /// Download the required model (Qwen 3.5-0.8B)
    Setup,
}

struct AIModel {
    device: Device,
    tokenizer: tokenizers::Tokenizer,
    model: Qwen35Model,
}

impl AIModel {
    fn load() -> Result<Self> {
        let device = Device::new_metal(0).or_else(|_| Device::new_cuda(0)).unwrap_or(Device::Cpu);
        
        let api = Api::new()?;
        let repo = api.repo(Repo::with_revision(
            "Qwen/Qwen3.5-0.8B".to_string(),
            RepoType::Model,
            "main".to_string(),
        ));

        let tokenizer_filename = repo.get("tokenizer.json")
            .context("Failed to get tokenizer.json. Run 'qsh setup' first.")?;
        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_filename)
            .map_err(anyhow::Error::msg)?;

        println!("Loading weights into {:?}...", device);
        let weights_filename = repo.get("model.safetensors-00001-of-00001.safetensors")?;
        let vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(&[weights_filename], candle_core::DType::F32, &device)?
        };
        
        let config = Qwen35Config::default();
        let model = Qwen35Model::load(vb, &config)?;
        
        Ok(AIModel { device, tokenizer, model })
    }

    fn generate(&self, prompt: &str, image: Option<&Tensor>, max_tokens: usize, quiet: bool) -> Result<String> {
        let tokens = self.tokenizer.encode(prompt, true).map_err(anyhow::Error::msg)?;
        let mut token_ids = tokens.get_ids().to_vec();
        let mut generated = String::new();
        // Greedy decoding for better stability
        let mut logits_processor = LogitsProcessor::new(299792458, None, None);
        
        let config = Qwen35Config::default();
        let mut layer_states: Vec<LayerState> = vec![LayerState::None; config.num_hidden_layers];
        
        let eos_token = self.tokenizer.get_vocab(true).get("<|im_end|>").copied().unwrap_or(151645);

        // Pre-fill
        let input_ids = Tensor::new(&token_ids[..], &self.device)?.unsqueeze(0)?;
        let mut logits = self.model.forward(Some(&input_ids), image, &mut layer_states)?;

        for _ in 0..max_tokens {
            let next_token = logits_processor.sample(&logits.squeeze(0)?)?;
            token_ids.push(next_token);
            
            if next_token == eos_token {
                break;
            }

            if let Some(text) = self.tokenizer.decode(&[next_token], true).ok() {
                if !quiet {
                    print!("{}", text);
                    io::stdout().flush()?;
                }
                generated.push_str(&text);
            }

            // Recurrent step
            let input_ids = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
            logits = self.model.forward(Some(&input_ids), None, &mut layer_states)?;
        }
        if !quiet {
            println!();
        }
        
        Ok(generated)
    }

    fn classify_image(&self, path: &str, query: &str) -> Result<bool> {
        let img = image::open(path).context(format!("Failed to open image: {}", path))?;
        let img = img.resize_exact(224, 224, image::imageops::FilterType::Triangle);
        let rgb = img.to_rgb8();
        let data = rgb.into_raw();
        let tensor = Tensor::from_vec(data, (224, 224, 3), &self.device)?
            .permute((2, 0, 1))?
            .to_dtype(candle_core::DType::F32)?
            .affine(1.0 / 255.0, 0.0)?;
        let tensor = Tensor::cat(&[&tensor, &tensor], 0)?.unsqueeze(0)?; // [1, 6, 224, 224]
            
        let prompt = format!("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nQuestion: {} Answer YES or NO.<|im_end|>\n<|im_start|>assistant\n", query);
        let response = self.generate(&prompt, Some(&tensor), 5, true)?;
        Ok(response.trim().to_uppercase().contains("YES"))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Setup) => {
            println!("Downloading Qwen 3.5-0.8B from Hugging Face...");
            let api = Api::new()?;
            let repo = api.repo(Repo::model("Qwen/Qwen3.5-0.8B".to_string()));
            repo.get("tokenizer.json")?;
            repo.get("model.safetensors-00001-of-00001.safetensors")?;
            println!("Setup complete!");
        }
        Some(Commands::Filter { query }) => {
            let model = AIModel::load()?;
            let stdin = io::stdin();
            for line in stdin.lock().lines() {
                let line = line?;
                if line.trim().is_empty() { continue; }
                let prompt = format!("<|im_start|>system\nYou are a text filter. Answer YES or NO. NO Chinese.<|im_end|>\n<|im_start|>user\nIs the following line related to '{}'?\nLine: {}<|im_end|>\n<|im_start|>assistant\n", query, line);
                let response = model.generate(&prompt, None, 5, true)?;
                if response.trim().to_uppercase().contains("YES") {
                    println!("{}", line);
                }
            }
        }
        Some(Commands::Vision { query }) => {
            let model = AIModel::load()?;
            let stdin = io::stdin();
            for path in stdin.lock().lines() {
                let path = path?;
                if path.trim().is_empty() { continue; }
                if model.classify_image(&path, &query)? {
                    println!("{}", path);
                }
            }
        }
        None => {
            if let Some(prompt) = cli.prompt {
                let model = AIModel::load()?;
                let system_prompt = "You are a Unix shell expert. Output ONLY valid Bash code. NO explanation, NO markdown, NO Chinese. If you think, keep it extremely brief.";
                let full_prompt = format!("<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n", system_prompt, prompt);

                let command = model.generate(&full_prompt, None, 512, false)?;

                
                loop {
                    print!("\n[E]xecute? [e]xplain? [A]bort? ");
                    io::stdout().flush()?;
                    let mut input = String::new();
                    io::stdin().read_line(&mut input)?;
                    match input.trim().to_lowercase().as_str() {
                        "e" => {
                            if input.trim() == "e" {
                                let explain_prompt = format!("<|im_start|>system\nYou are a Unix shell expert.<|im_end|>\n<|im_start|>user\nExplain this Bash command briefly: {}<|im_end|>\n<|im_start|>assistant\n", command);
                                println!("\n> Explanation:");
                                model.generate(&explain_prompt, None, 100, false)?;
                            } else {
                                println!("Executing: {}", command);
                                std::process::Command::new("bash").arg("-c").arg(&command).spawn()?.wait()?;
                                break;
                            }
                        }
                        "a" => { println!("Aborted."); break; }
                        _ => {
                            if input.trim().to_uppercase() == "E" || input.trim().is_empty() {
                                println!("Executing: {}", command);
                                std::process::Command::new("bash").arg("-c").arg(&command).spawn()?.wait()?;
                                break;
                            }
                            println!("Aborted.");
                            break;
                        }
                    }
                }
            }
        }
    }
    Ok(())
}
