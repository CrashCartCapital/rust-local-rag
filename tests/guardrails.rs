use serial_test::serial;

struct EnvVarGuard {
    key: &'static str,
    original: Option<std::ffi::OsString>,
}

impl EnvVarGuard {
    fn set(key: &'static str, value: &str) -> Self {
        let original = std::env::var_os(key);
        unsafe {
            std::env::set_var(key, value);
        }
        Self { key, original }
    }

    fn unset(key: &'static str) -> Self {
        let original = std::env::var_os(key);
        unsafe {
            std::env::remove_var(key);
        }
        Self { key, original }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        unsafe {
            match &self.original {
                Some(value) => std::env::set_var(self.key, value),
                None => std::env::remove_var(self.key),
            }
        }
    }
}

#[test]
#[serial]
fn test_prd_t0_1_ollama_url_loopback_allowed() {
    let _guard = EnvVarGuard::unset("RAG_ALLOW_REMOTE_OLLAMA");

    rust_local_rag::guardrails::check_ollama_url("http://127.0.0.1:11434").unwrap();
    rust_local_rag::guardrails::check_ollama_url("http://[::1]:11434").unwrap();
    rust_local_rag::guardrails::check_ollama_url("http://localhost:11434").unwrap();
}

#[test]
#[serial]
fn test_prd_t0_1_ollama_url_remote_rejected_without_override() {
    let _guard = EnvVarGuard::unset("RAG_ALLOW_REMOTE_OLLAMA");

    let err = rust_local_rag::guardrails::check_ollama_url("http://example.com:11434").unwrap_err();
    let msg = err.to_string();

    assert!(msg.contains("example.com"));
    assert!(msg.contains("RAG_ALLOW_REMOTE_OLLAMA"));
    let msg_lower = msg.to_lowercase();
    assert!(msg_lower.contains("document"));
    assert!(msg_lower.contains("query"));
    assert!(msg_lower.contains("text"));
}

#[test]
#[serial]
fn test_prd_t0_1_ollama_url_remote_allowed_with_override() {
    let _guard = EnvVarGuard::set("RAG_ALLOW_REMOTE_OLLAMA", "1");
    rust_local_rag::guardrails::check_ollama_url("http://example.com:11434").unwrap();
}

#[test]
#[serial]
fn test_prd_t0_1_ollama_url_invalid_url_rejected() {
    let _guard = EnvVarGuard::unset("RAG_ALLOW_REMOTE_OLLAMA");

    let err = rust_local_rag::guardrails::check_ollama_url("not-a-url").unwrap_err();
    assert!(err.to_string().to_lowercase().contains("invalid"));
}

#[test]
#[serial]
fn test_prd_t0_2_mcp_http_bind_loopback_allowed() {
    let _guard = EnvVarGuard::unset("RAG_ALLOW_REMOTE_BIND");

    rust_local_rag::guardrails::check_mcp_http_bind("127.0.0.1:8140".parse().unwrap()).unwrap();
    rust_local_rag::guardrails::check_mcp_http_bind("[::1]:8140".parse().unwrap()).unwrap();
    rust_local_rag::guardrails::check_mcp_http_bind("127.0.0.1:0".parse().unwrap()).unwrap();
}

#[test]
#[serial]
fn test_prd_t0_2_mcp_http_bind_non_loopback_rejected_without_override() {
    let _guard = EnvVarGuard::unset("RAG_ALLOW_REMOTE_BIND");

    let err = rust_local_rag::guardrails::check_mcp_http_bind("0.0.0.0:8140".parse().unwrap())
        .unwrap_err();
    let msg = err.to_string();

    assert!(msg.contains("0.0.0.0:8140"));
    assert!(msg.contains("RAG_ALLOW_REMOTE_BIND"));
    assert!(msg.to_lowercase().contains("no auth"));
}

#[test]
#[serial]
fn test_prd_t0_2_mcp_http_bind_non_loopback_allowed_with_override() {
    let _guard = EnvVarGuard::set("RAG_ALLOW_REMOTE_BIND", "1");
    rust_local_rag::guardrails::check_mcp_http_bind("0.0.0.0:8140".parse().unwrap()).unwrap();
}
