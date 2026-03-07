use serde::{Serialize, Deserialize};
use std::path::PathBuf;
use directories::ProjectDirs;
use anyhow::Result;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Config {
    pub default_engine: String,
    pub default_model: String,
    pub safety_check: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            default_engine: "python".to_string(),
            default_model: "Qwen/Qwen3.5-0.8B".to_string(),
            safety_check: true,
        }
    }
}

impl Config {
    pub fn load() -> Self {
        let config_path = Self::get_path();
        if let Ok(content) = std::fs::read_to_string(&config_path) {
            if let Ok(config) = toml::from_str(&content) {
                return config;
            }
        }
        // If no config, return default but don't save yet
        Self::default()
    }

    pub fn save(&self) -> Result<()> {
        let config_path = Self::get_path();
        if let Some(parent) = config_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let content = toml::to_string_pretty(self)?;
        std::fs::write(config_path, content)?;
        Ok(())
    }

    fn get_path() -> PathBuf {
        if let Some(dirs) = ProjectDirs::from("com", "qwen", "qsh") {
            return dirs.config_dir().join("config.toml");
        }
        PathBuf::from(".qsh_config.toml")
    }
}
