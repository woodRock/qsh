use colored::*;

pub fn check_safety(command: &str) -> bool {
    let dangerous_patterns = [
        "rm -rf /",
        ":(){ :|:& };:",
        "mkfs",
        "dd if=",
        "> /dev/sda",
        "chmod -R 777 /",
        "chown -R",
        "shutdown",
        "reboot",
    ];

    for pattern in dangerous_patterns {
        if command.contains(pattern) {
            return false;
        }
    }
    true
}

pub fn print_warning(command: &str) {
    println!("{}", "⚠️  WARNING: This command looks potentially destructive!".bold().yellow());
    println!("Suggested Command: {}", command.bold().red());
}
