use qsh::bridge::{strip_code_fence, strip_think_tags};

// ── strip_think_tags ──────────────────────────────────────────────────────────

#[test]
fn test_strip_think_tags_no_tags() {
    assert_eq!(strip_think_tags("ls -la"), "ls -la");
}

#[test]
fn test_strip_think_tags_complete_block() {
    let input = "<think>I should use find here.</think>find . -name '*.rs'";
    assert_eq!(strip_think_tags(input), "find . -name '*.rs'");
}

#[test]
fn test_strip_think_tags_multiline_block() {
    let input = "<think>\nLet me reason about this.\nThe answer is ls.\n</think>ls -la";
    assert_eq!(strip_think_tags(input), "ls -la");
}

#[test]
fn test_strip_think_tags_incomplete_returns_prelude() {
    // No closing tag — model hit token limit mid-think.
    // Should return text before <think>, not empty string.
    let input = "some prelude <think>reasoning that never finishes";
    assert_eq!(strip_think_tags(input), "some prelude");
}

#[test]
fn test_strip_think_tags_incomplete_no_prelude_returns_empty() {
    // Nothing before <think> and no closing tag — return empty.
    let input = "<think>reasoning that never finishes";
    assert_eq!(strip_think_tags(input), "");
}

#[test]
fn test_strip_think_tags_trims_whitespace() {
    let input = "<think>thought</think>  git status  ";
    assert_eq!(strip_think_tags(input), "git status");
}

#[test]
fn test_strip_think_tags_no_think_content_preserved() {
    let input = "echo hello world";
    assert_eq!(strip_think_tags(input), "echo hello world");
}

// ── strip_code_fence ─────────────────────────────────────────────────────────

#[test]
fn test_strip_code_fence_bash_lang() {
    let input = "```bash\nls -la\n```";
    assert_eq!(strip_code_fence(input), "ls -la");
}

#[test]
fn test_strip_code_fence_no_lang() {
    let input = "```\nls -la\n```";
    assert_eq!(strip_code_fence(input), "ls -la");
}

#[test]
fn test_strip_code_fence_sh_lang() {
    let input = "```sh\necho hello\n```";
    assert_eq!(strip_code_fence(input), "echo hello");
}

#[test]
fn test_strip_code_fence_multiline_command() {
    let input = "```bash\nfind . -name '*.rs' \\\n  -exec wc -l {} \\;\n```";
    assert_eq!(
        strip_code_fence(input),
        "find . -name '*.rs' \\\n  -exec wc -l {} \\;"
    );
}

#[test]
fn test_strip_code_fence_missing_close_returns_body() {
    // No closing ``` — return whatever is inside anyway.
    let input = "```bash\nls -la";
    assert_eq!(strip_code_fence(input), "ls -la");
}

#[test]
fn test_strip_code_fence_single_backtick() {
    let input = "`git status`";
    assert_eq!(strip_code_fence(input), "git status");
}

#[test]
fn test_strip_code_fence_plain_text_unchanged() {
    let input = "git log --oneline";
    assert_eq!(strip_code_fence(input), "git log --oneline");
}

#[test]
fn test_strip_code_fence_trims_outer_whitespace() {
    let input = "  ```bash\nls\n```  ";
    assert_eq!(strip_code_fence(input), "ls");
}

// ── composition ──────────────────────────────────────────────────────────────

#[test]
fn test_strip_think_then_fence() {
    // Model outputs think block then a fenced command — the real-world case.
    let input = "<think>Let me think.</think>```bash\nls -la\n```";
    let after_think = strip_think_tags(input);
    let result = strip_code_fence(&after_think);
    assert_eq!(result, "ls -la");
}

#[test]
fn test_strip_think_and_fence_plain_command() {
    let input = "<think>reasoning</think>ls -la";
    let result = strip_code_fence(&strip_think_tags(input));
    assert_eq!(result, "ls -la");
}
