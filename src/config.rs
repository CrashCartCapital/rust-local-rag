use std::num::NonZeroUsize;
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct Config {
    pub embedding_timeout: Duration,
    pub embedding_cache_size: NonZeroUsize,
    pub reranker_timeout: Duration,
    pub reranker_concurrency: NonZeroUsize,
    pub default_logprob_fallback: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            embedding_timeout: Duration::from_secs(1200),
            embedding_cache_size: NonZeroUsize::new(1000).expect("1000 is non-zero"),
            reranker_timeout: Duration::from_secs(60),
            reranker_concurrency: NonZeroUsize::new(1).expect("1 is non-zero"),
            default_logprob_fallback: -10.0,
        }
    }
}

impl Config {
    pub fn from_env() -> Result<Self, ConfigError> {
        let mut config = Self::default();

        if let Some(timeout_secs) = parse_env_u64("RAG_EMBEDDING_TIMEOUT_SECS")? {
            config.embedding_timeout = Duration::from_secs(timeout_secs);
        }
        if let Some(cache_size) = parse_env_nonzero_usize("RAG_EMBEDDING_CACHE_SIZE")? {
            config.embedding_cache_size = cache_size;
        }

        if let Some(timeout_secs) = parse_env_u64("RAG_RERANKER_TIMEOUT_SECS")? {
            config.reranker_timeout = Duration::from_secs(timeout_secs);
        }
        if let Some(concurrency) = parse_env_nonzero_usize("RAG_RERANKER_CONCURRENCY")? {
            config.reranker_concurrency = concurrency;
        }

        if let Some(value) = parse_env_f64("RAG_DEFAULT_LOGPROB")? {
            config.default_logprob_fallback = value;
        }

        Ok(config)
    }

    pub fn log_active(&self) {
        tracing::info!(
            embedding_timeout_secs = self.embedding_timeout.as_secs(),
            embedding_cache_size = self.embedding_cache_size.get(),
            reranker_timeout_secs = self.reranker_timeout.as_secs(),
            reranker_concurrency = self.reranker_concurrency.get(),
            default_logprob_fallback = self.default_logprob_fallback,
            "Configuration loaded"
        );
    }
}

#[derive(Debug, Clone)]
pub struct ConfigError {
    pub var: &'static str,
    pub value: String,
    pub message: String,
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Invalid value '{}' for {}: {}",
            self.value, self.var, self.message
        )
    }
}

impl std::error::Error for ConfigError {}

fn parse_env_u64(var: &'static str) -> Result<Option<u64>, ConfigError> {
    match std::env::var(var) {
        Ok(val) => val.parse::<u64>().map(Some).map_err(|e| ConfigError {
            var,
            value: val,
            message: e.to_string(),
        }),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(e) => Err(ConfigError {
            var,
            value: String::new(),
            message: e.to_string(),
        }),
    }
}

fn parse_env_f64(var: &'static str) -> Result<Option<f64>, ConfigError> {
    match std::env::var(var) {
        Ok(val) => val.parse::<f64>().map(Some).map_err(|e| ConfigError {
            var,
            value: val,
            message: e.to_string(),
        }),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(e) => Err(ConfigError {
            var,
            value: String::new(),
            message: e.to_string(),
        }),
    }
}

fn parse_env_nonzero_usize(var: &'static str) -> Result<Option<NonZeroUsize>, ConfigError> {
    match std::env::var(var) {
        Ok(val) => {
            let parsed = val.parse::<usize>().map_err(|e| ConfigError {
                var,
                value: val.clone(),
                message: e.to_string(),
            })?;
            NonZeroUsize::new(parsed)
                .ok_or_else(|| ConfigError {
                    var,
                    value: val,
                    message: "must be > 0".to_string(),
                })
                .map(Some)
        }
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(e) => Err(ConfigError {
            var,
            value: String::new(),
            message: e.to_string(),
        }),
    }
}
