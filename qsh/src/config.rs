use anyhow::Result;
use directories::ProjectDirs;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Config {
    pub default_engine: String,
    pub default_model: String,
    pub llama_cpp: LlamaCppConfig,
    pub safety_check: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LlamaCppConfig {
    pub server_url: String,
    pub server_binary: Option<String>,
    pub model_path: Option<String>,
    pub mmproj_path: Option<String>,
    pub turbo_k: String,
    pub turbo_v: String,
    pub flash_attn: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            default_engine: "python".to_string(),
            default_model: "Qwen/Qwen3.5-0.8B".to_string(),
            llama_cpp: LlamaCppConfig {
                // Use the literal loopback address, not "localhost": on systems where
                // "localhost" resolves to ::1 before 127.0.0.1, a request can be routed to
                // a *different* process that also happens to be listening on this port over
                // IPv6 (e.g. another app's web UI), instead of to llama-server, which binds
                // IPv4-only. That silently misroutes the health check and every inference
                // request without llama-server ever being at fault.
                server_url: "http://127.0.0.1:8080".to_string(),
                server_binary: None,
                model_path: None,
                mmproj_path: None,
                turbo_k: "q8_0".to_string(),
                turbo_v: "turbo4".to_string(),
                flash_attn: true,
            },
            safety_check: true,
        }
    }
}

impl Config {
    pub fn load() -> Self {
        Self::load_from(Self::get_path())
    }

    pub fn load_from(path: PathBuf) -> Self {
        if let Ok(content) = std::fs::read_to_string(&path)
            && let Ok(config) = toml::from_str(&content)
        {
            return config;
        }
        // If no config, return default but don't save yet
        Self::default()
    }

    #[allow(dead_code)]
    pub fn save(&self) -> Result<()> {
        self.save_to(Self::get_path())
    }

    pub fn save_to(&self, path: PathBuf) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let content = toml::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }

    fn get_path() -> PathBuf {
        if let Some(dirs) = ProjectDirs::from("com", "qwen", "qsh") {
            return dirs.config_dir().join("config.toml");
        }
        PathBuf::from(".qsh_config.toml")
    }
}
