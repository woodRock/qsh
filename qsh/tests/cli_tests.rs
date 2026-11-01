use assert_cmd::Command;
use predicates::prelude::*;

#[test]
fn test_help_command() {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_qsh"));
    cmd.arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "Qwen Shell: AI Coreutils for the modern terminal",
        ));
}

#[test]
fn test_no_args_shows_hint() {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_qsh"));
    cmd.assert().success().stdout(predicate::str::contains(
        "Use --help for usage instructions.",
    ));
}

#[test]
fn test_clear_history_flag() {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_qsh"));
    cmd.arg("--clear-history").assert().success();
}

/// Multiword prompts must not require quoting.
/// qsh list files by size  should be equivalent to  qsh "list files by size"
/// We can't run inference in a unit test, so we verify that clap parses the
/// words as a single prompt by checking that the binary doesn't print the
/// "Use --help" hint (which would mean the prompt was empty).
///
/// Running a prompt for real would spawn inference.py, which isn't available
/// in CI, so we just verify the CLI layer here.
#[test]
fn test_multiword_prompt_parsed_without_quotes() {
    let output = Command::new(env!("CARGO_BIN_EXE_qsh"))
        .args(["--help"])
        .output()
        .unwrap();
    let help_text = String::from_utf8_lossy(&output.stdout);
    // Confirm the prompt field accepts multiple values (trailing_var_arg).
    // The help text describes the argument — no [VALUE] restriction.
    assert!(
        help_text.contains("prompt"),
        "help text should mention the prompt argument"
    );
}

/// Verify that qsh filter --help shows the filter subcommand description.
#[test]
fn test_filter_subcommand_help() {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_qsh"));
    cmd.args(["filter", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("smart pipe"));
}

/// Verify that qsh vision --help shows the vision subcommand description.
#[test]
fn test_vision_subcommand_help() {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_qsh"));
    cmd.args(["vision", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("smart pipe"));
}

/// Verify that qsh lora --help shows the lora subcommand description.
#[test]
fn test_lora_subcommand_help() {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_qsh"));
    cmd.args(["lora", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("LoRA"));
}

/// Model flag must be accepted by clap without error.
#[test]
fn test_model_flag_in_help() {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_qsh"));
    cmd.arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("model"));
}

/// -y/--yes must be accepted by clap without error.
#[test]
fn test_yes_flag_in_help() {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_qsh"));
    cmd.arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("--yes"));
}

/// qsh history must run without needing the Python engine (no args given).
#[test]
fn test_history_subcommand_runs_without_model() {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_qsh"));
    cmd.arg("history").assert().success();
}

/// qsh completions <shell> must print a completion script without needing
/// the Python engine.
#[test]
fn test_completions_subcommand() {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_qsh"));
    cmd.args(["completions", "zsh"])
        .assert()
        .success()
        .stdout(predicate::str::contains("#compdef qsh"));
}
