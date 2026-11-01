use colored::*;

/// Base commands that are inherently destructive regardless of their arguments.
const DANGEROUS_BASE_COMMANDS: &[&str] = &["mkswap", "shred", "shutdown", "reboot"];

/// Wrapper commands to skip when looking for the real command being run
/// (e.g. `sudo rm -rf /` should still be flagged as `rm`).
const WRAPPER_COMMANDS: &[&str] = &["sudo", "env", "nice", "nohup"];

pub fn check_safety(command: &str) -> bool {
    let cmd_lower = command.to_lowercase();

    // Shell syntax that doesn't tokenize as a normal command.
    if cmd_lower.contains(":(){") {
        return false;
    }
    if redirects_to_block_device(&cmd_lower) {
        return false;
    }

    for segment in split_top_level(&cmd_lower) {
        let tokens: Vec<&str> = segment.split_whitespace().collect();

        let mut idx = 0;
        while idx < tokens.len() && WRAPPER_COMMANDS.contains(&tokens[idx]) {
            idx += 1;
        }
        let Some(base_raw) = tokens.get(idx) else {
            continue;
        };
        let base = base_raw.rsplit('/').next().unwrap_or(base_raw);
        let args = &tokens[idx + 1..];

        if base.starts_with("mkfs") || DANGEROUS_BASE_COMMANDS.contains(&base) {
            return false;
        }

        let is_dangerous = match base {
            // Any `-f`/`--force` is enough: force-deleting even a single file
            // is irreversible, with or without `-r`.
            "rm" => args.iter().any(|a| is_force_flag(a)),
            "dd" => args.iter().any(|a| a.starts_with("if=")),
            "chmod" => {
                args.iter().any(|a| is_recursive_flag(a))
                    && args.iter().any(|a| matches!(*a, "777" | "000"))
            }
            "chown" => args.iter().any(|a| is_recursive_flag(a)),
            _ => false,
        };
        if is_dangerous {
            return false;
        }
    }
    true
}

fn is_force_flag(arg: &str) -> bool {
    arg == "--force" || (arg.starts_with('-') && !arg.starts_with("--") && arg.contains('f'))
}

fn is_recursive_flag(arg: &str) -> bool {
    arg == "--recursive" || (arg.starts_with('-') && !arg.starts_with("--") && arg.contains('r'))
}

/// Catches redirecting output straight onto a block device, e.g.
/// `echo data > /dev/sda` or `cat file >/dev/nvme0n1`.
fn redirects_to_block_device(cmd_lower: &str) -> bool {
    for marker in [">>", ">"] {
        if let Some(pos) = cmd_lower.find(marker) {
            let rest = cmd_lower[pos + marker.len()..].trim_start();
            if ["/dev/sd", "/dev/nvme", "/dev/disk", "/dev/hd"]
                .iter()
                .any(|prefix| rest.starts_with(prefix))
            {
                return true;
            }
        }
    }
    false
}

/// Split a shell command on top-level `;`, `&&`, `||`, `|`, ignoring
/// separators inside single or double quotes.
fn split_top_level(cmd: &str) -> Vec<String> {
    let mut segments = Vec::new();
    let mut current = String::new();
    let mut chars = cmd.chars().peekable();
    let mut in_single = false;
    let mut in_double = false;

    while let Some(c) = chars.next() {
        match c {
            '\'' if !in_double => {
                in_single = !in_single;
                current.push(c);
            }
            '"' if !in_single => {
                in_double = !in_double;
                current.push(c);
            }
            ';' if !in_single && !in_double => segments.push(std::mem::take(&mut current)),
            '|' if !in_single && !in_double => {
                if chars.peek() == Some(&'|') {
                    chars.next();
                }
                segments.push(std::mem::take(&mut current));
            }
            '&' if !in_single && !in_double => {
                if chars.peek() == Some(&'&') {
                    chars.next();
                }
                segments.push(std::mem::take(&mut current));
            }
            _ => current.push(c),
        }
    }
    segments.push(current);
    segments
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

    #[test]
    fn test_split_short_flags_still_caught() {
        // These evade a plain substring match on "rm -rf" / "chmod -r 777".
        assert!(!check_safety("rm -r -f /tmp/x"));
        assert!(!check_safety("rm --recursive --force /tmp/x"));
        assert!(!check_safety("chmod 777 -R /"));
    }

    #[test]
    fn test_compound_command_checked_per_segment() {
        assert!(!check_safety("ls -la && rm -rf /tmp"));
        assert!(!check_safety("echo hi; shutdown -h now"));
        assert!(check_safety("cd /tmp && ls -la"));
    }
}
