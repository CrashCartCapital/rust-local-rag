use std::net::IpAddr;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GuardrailKind {
    OllamaUrl,
}

impl GuardrailKind {
    fn override_env_var(self) -> &'static str {
        match self {
            GuardrailKind::OllamaUrl => "RAG_ALLOW_REMOTE_OLLAMA",
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
        };
        Self {
            message: format!(
                "Refusing non-loopback {kind_name} '{value}'. This will send document/query text to that endpoint. If you intended this, set {override_env_var}=1."
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
