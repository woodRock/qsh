use qsh::config::Config;
use std::path::PathBuf;
use tempfile::NamedTempFile;

#[test]
fn test_default_config() {
    let config = Config::default();
    assert_eq!(config.default_engine, "python");
    assert_eq!(config.default_model, "Qwen/Qwen3.5-0.8B");
    assert!(config.safety_check);
}

#[test]
fn test_load_save_config() {
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path().to_path_buf();

    let mut config = Config::default();
    config.default_engine = "rust".to_string();
    config.default_model = "Custom/Model".to_string();
    config.safety_check = false;

    config.save_to(path.clone()).expect("Failed to save config");

    let loaded = Config::load_from(path);
    assert_eq!(loaded.default_engine, "rust");
    assert_eq!(loaded.default_model, "Custom/Model");
    assert!(!loaded.safety_check);
}

#[test]
fn test_load_non_existent() {
    let path = PathBuf::from("non_existent_config.toml");
    let config = Config::load_from(path);
    assert_eq!(config.default_engine, Config::default().default_engine);
}

#[test]
fn test_load_invalid_toml() {
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path().to_path_buf();
    std::fs::write(&path, "invalid toml content").unwrap();

    let config = Config::load_from(path);
    assert_eq!(config.default_engine, Config::default().default_engine);
}
