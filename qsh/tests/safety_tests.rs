use qsh::safety;

#[test]
fn test_destructive_commands() {
    let destructive = [
        "rm -rf /",
        "rm -f important.txt",
        "rm  -rf .",
        ":(){ :|:& };:",
        "mkfs.ext4 /dev/sda1",
        "dd if=/dev/zero of=/dev/sda",
        "echo data > /dev/sda",
        "cat file >/dev/sda",
        "chmod -R 777 /",
        "chown -R root:root /",
        "shutdown -h now",
        "reboot",
        "mkswap /dev/sdb1",
        "shred -u secret.txt",
    ];

    for cmd in destructive {
        assert!(
            !safety::check_safety(cmd),
            "Command '{}' should be flagged as dangerous",
            cmd
        );
    }
}

#[test]
fn test_safe_commands() {
    let safe = [
        "ls -la",
        "cp source.txt dest.txt",
        "mv old.txt new.txt",
        "grep 'pattern' file.log",
        "cat README.md",
        "mkdir my_project",
        "touch new_file.rs",
        "git status",
        "cargo build",
        "npm install",
    ];

    for cmd in safe {
        assert!(
            safety::check_safety(cmd),
            "Command '{}' should be flagged as safe",
            cmd
        );
    }
}

#[test]
fn test_case_insensitivity() {
    assert!(!safety::check_safety("RM -RF /"));
    assert!(!safety::check_safety("ShUtDoWn"));
}
