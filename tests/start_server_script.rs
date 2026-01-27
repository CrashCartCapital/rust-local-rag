use std::process::Command;

#[test]
fn test_prd_t1_2_start_server_script_has_no_grep_xargs_env_parsing() {
    let script = include_str!("../start-server.sh");
    let non_comment_lines: String = script
        .lines()
        .filter(|line| !line.trim_start().starts_with('#'))
        .collect::<Vec<_>>()
        .join("\n");

    assert!(
        !non_comment_lines.contains("xargs"),
        "start-server.sh contains `xargs` in non-comment lines; avoid `.env` parsing via `grep|xargs` and use a safe loader (e.g., `set -a; source .env; set +a`)."
    );
    assert!(
        !non_comment_lines.contains("grep "),
        "start-server.sh contains `grep` in non-comment lines; avoid `.env` parsing via `grep|xargs` and use a safe loader (e.g., `set -a; source .env; set +a`)."
    );
}

#[test]
fn test_prd_t1_2_start_server_script_is_valid_bash() {
    let script_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("start-server.sh");
    let status = Command::new("bash")
        .arg("-n")
        .arg(&script_path)
        .status()
        .expect("failed to run `bash -n start-server.sh`");

    assert!(status.success(), "`bash -n` reported a syntax error");
}
