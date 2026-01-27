use std::net::IpAddr;
use std::net::SocketAddr;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GuardrailKind {
    OllamaUrl,
    McpHttpBind,
}

impl GuardrailKind {
    fn override_env_var(self) -> &'static str {
        match self {
            GuardrailKind::OllamaUrl => "RAG_ALLOW_REMOTE_OLLAMA",
            GuardrailKind::McpHttpBind => "RAG_ALLOW_REMOTE_BIND",
        }
    }
}

#[derive(Debug, Clone)]
pub struct GuardrailError {
    message: String,
}

impl GuardrailError {
    fn invalid_url(_kind: GuardrailKind, value: &str, source: impl std::fmt::Display) -> Self {
        Self {
            message: format!("Invalid URL '{value}': {source}"),
        }
    }

    fn missing_host(_kind: GuardrailKind, value: &str) -> Self {
        Self {
            message: format!("Invalid URL '{value}': missing host"),
        }
    }

    fn non_loopback(kind: GuardrailKind, value: &str) -> Self {
        let override_env_var = kind.override_env_var();
        let kind_name = match kind {
            GuardrailKind::OllamaUrl => "OLLAMA_URL",
            GuardrailKind::McpHttpBind => "MCP_HTTP_BIND",
        };
        let risk = match kind {
            GuardrailKind::OllamaUrl => "This will send document/query text to that endpoint.",
            GuardrailKind::McpHttpBind => {
                "This will expose the server over the network with no auth."
            }
        };
        Self {
            message: format!(
                "Refusing non-loopback {kind_name} '{value}'. {risk} If you intended this, set {override_env_var}=1."
            ),
        }
    }
}

impl std::fmt::Display for GuardrailError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for GuardrailError {}

pub fn is_loopback_url(url: &str) -> Result<bool, GuardrailError> {
    let parsed = reqwest::Url::parse(url)
        .map_err(|e| GuardrailError::invalid_url(GuardrailKind::OllamaUrl, url, e))?;
    let Some(host) = parsed.host_str() else {
        return Err(GuardrailError::missing_host(GuardrailKind::OllamaUrl, url));
    };
    let host = host.trim_start_matches('[').trim_end_matches(']');

    if host.eq_ignore_ascii_case("localhost") {
        return Ok(true);
    }

    Ok(host
        .parse::<IpAddr>()
        .map(|ip| ip.is_loopback())
        .unwrap_or(false))
}

fn is_remote_ollama_override_enabled() -> bool {
    std::env::var(GuardrailKind::OllamaUrl.override_env_var())
        .ok()
        .is_some_and(|v| v == "1")
}

pub fn check_ollama_url(ollama_url: &str) -> Result<(), GuardrailError> {
    if is_loopback_url(ollama_url)? {
        return Ok(());
    }

    if is_remote_ollama_override_enabled() {
        return Ok(());
    }

    Err(GuardrailError::non_loopback(
        GuardrailKind::OllamaUrl,
        ollama_url,
    ))
}

pub fn is_loopback_bind(bind_addr: SocketAddr) -> bool {
    bind_addr.ip().is_loopback()
}

fn is_remote_bind_override_enabled() -> bool {
    std::env::var(GuardrailKind::McpHttpBind.override_env_var())
        .ok()
        .is_some_and(|v| v == "1")
}

pub fn check_mcp_http_bind(bind_addr: SocketAddr) -> Result<(), GuardrailError> {
    if is_loopback_bind(bind_addr) {
        return Ok(());
    }

    if is_remote_bind_override_enabled() {
        return Ok(());
    }

    let bind_value = bind_addr.to_string();
    Err(GuardrailError::non_loopback(
        GuardrailKind::McpHttpBind,
        &bind_value,
    ))
}
