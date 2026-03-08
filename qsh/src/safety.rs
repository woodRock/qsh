use colored::*;

pub fn check_safety(command: &str) -> bool {
    let cmd_lower = command.to_lowercase();
    let dangerous_patterns = [
        "rm -rf",
        "rm -f",
        "rm  -rf",
        ":(){",
        "mkfs",
        "dd if=",
        "> /dev/sda",
        ">/dev/sda",
        "chmod -r 777",
        "chown -r",
        "shutdown",
        "reboot",
        "mkswap",
        "shred",
    ];

    for pattern in dangerous_patterns {
        if cmd_lower.contains(pattern) {
            return false;
        }
    }
    true
}

pub fn print_warning(command: &str) {
    println!(
        "\n{}",
        "⚠️  WARNING: This command looks potentially destructive!"
            .bold()
            .yellow()
    );
    println!("Suggested Command: {}", command.bold().red());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rm_rf() {
        assert!(!check_safety("rm -rf /"));
        assert!(!check_safety("rm -rf ."));
        assert!(!check_safety("sudo rm -rf /etc"));
    }

    #[test]
    fn test_safe() {
        assert!(check_safety("ls -la"));
        assert!(check_safety("cp file.txt backup.txt"));
    }
}
