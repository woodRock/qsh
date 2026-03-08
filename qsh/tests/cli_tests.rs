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
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Use --help for usage instructions."));
}

#[test]
fn test_unknown_command_treated_as_prompt() {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_qsh"));
    // Since it tries to run inference, it might fail if model not found, 
    // but here we see it successfully starts and asks for confirmation.
    cmd.arg("invalid-subcommand")
        .write_stdin("n\n")
        .assert()
        .success();
}


#[test]
fn test_clear_history_flag() {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_qsh"));
    cmd.arg("--clear-history").assert().success();
}
