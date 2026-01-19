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
            if timeout_secs == 0 || timeout_secs > 3600 {
                return Err(ConfigError {
                    var: "RAG_EMBEDDING_TIMEOUT_SECS",
                    value: timeout_secs.to_string(),
                    message: "must be > 0 and <= 3600".to_string(),
                });
            }
            config.embedding_timeout = Duration::from_secs(timeout_secs);
        }
        if let Some(cache_size) = parse_env_nonzero_usize("RAG_EMBEDDING_CACHE_SIZE")? {
            if cache_size.get() > 10000 {
                return Err(ConfigError {
                    var: "RAG_EMBEDDING_CACHE_SIZE",
                    value: cache_size.to_string(),
                    message: "must be <= 10000".to_string(),
                });
            }
            config.embedding_cache_size = cache_size;
        }

        if let Some(timeout_secs) = parse_env_u64("RAG_RERANKER_TIMEOUT_SECS")? {
            if timeout_secs == 0 || timeout_secs > 600 {
                return Err(ConfigError {
                    var: "RAG_RERANKER_TIMEOUT_SECS",
                    value: timeout_secs.to_string(),
                    message: "must be > 0 and <= 600".to_string(),
                });
            }
            config.reranker_timeout = Duration::from_secs(timeout_secs);
        }
        if let Some(concurrency) = parse_env_nonzero_usize("RAG_RERANKER_CONCURRENCY")? {
            config.reranker_concurrency = concurrency;
        }

        if let Some(value) = parse_env_f64("RAG_DEFAULT_LOGPROB")? {
            if !value.is_finite() || value > 0.0 {
                return Err(ConfigError {
                    var: "RAG_DEFAULT_LOGPROB",
                    value: value.to_string(),
                    message: "must be finite and <= 0.0".to_string(),
                });
            }
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static ENV_MUTEX: Mutex<()> = Mutex::new(());

    fn with_env_var<F>(key: &str, value: &str, f: F)
    where
        F: FnOnce(),
    {
        let _lock = ENV_MUTEX.lock().unwrap();
        // SAFETY: We hold the mutex so no other test in this module can access the environment.
        // However, this is still not thread-safe if other threads outside this module access env.
        // Given this is a unit test and we don't have other threads accessing env in tests, it's acceptable.
        unsafe {
            std::env::set_var(key, value);
        }
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
        unsafe {
            std::env::remove_var(key);
        }
        if let Err(e) = result {
            std::panic::resume_unwind(e);
        }
    }

    #[test]
    fn test_rag_embedding_timeout_valid() {
        with_env_var("RAG_EMBEDDING_TIMEOUT_SECS", "60", || {
            let config = Config::from_env().unwrap();
            assert_eq!(config.embedding_timeout.as_secs(), 60);
        });
    }

    #[test]
    fn test_rag_embedding_timeout_zero() {
        with_env_var("RAG_EMBEDDING_TIMEOUT_SECS", "0", || {
            let err = Config::from_env().unwrap_err();
            assert!(err.message.contains("must be > 0 and <= 3600"));
        });
    }

    #[test]
    fn test_rag_embedding_timeout_too_large() {
        with_env_var("RAG_EMBEDDING_TIMEOUT_SECS", "3601", || {
            let err = Config::from_env().unwrap_err();
            assert!(err.message.contains("must be > 0 and <= 3600"));
        });
    }

    #[test]
    fn test_rag_default_logprob_valid() {
        with_env_var("RAG_DEFAULT_LOGPROB", "-5.5", || {
            let config = Config::from_env().unwrap();
            assert_eq!(config.default_logprob_fallback, -5.5);
        });
    }

    #[test]
    fn test_rag_default_logprob_zero() {
        with_env_var("RAG_DEFAULT_LOGPROB", "0.0", || {
            let config = Config::from_env().unwrap();
            assert_eq!(config.default_logprob_fallback, 0.0);
        });
    }

    #[test]
    fn test_rag_default_logprob_positive() {
        with_env_var("RAG_DEFAULT_LOGPROB", "0.1", || {
            let err = Config::from_env().unwrap_err();
            assert!(err.message.contains("must be finite and <= 0.0"));
        });
    }

    #[test]
    fn test_rag_default_logprob_nan() {
        with_env_var("RAG_DEFAULT_LOGPROB", "NaN", || {
            let err = Config::from_env().unwrap_err();
            assert!(err.message.contains("must be finite and <= 0.0"));
        });
    }

    #[test]
    fn test_rag_default_logprob_inf() {
        with_env_var("RAG_DEFAULT_LOGPROB", "inf", || {
            let err = Config::from_env().unwrap_err();
            assert!(err.message.contains("must be finite and <= 0.0"));
        });
    }

    #[test]
    fn test_rag_reranker_timeout_valid() {
        with_env_var("RAG_RERANKER_TIMEOUT_SECS", "30", || {
            let config = Config::from_env().unwrap();
            assert_eq!(config.reranker_timeout.as_secs(), 30);
        });
    }

    #[test]
    fn test_rag_reranker_timeout_zero() {
        with_env_var("RAG_RERANKER_TIMEOUT_SECS", "0", || {
            let err = Config::from_env().unwrap_err();
            assert!(err.message.contains("must be > 0 and <= 600"));
        });
    }

    #[test]
    fn test_rag_reranker_timeout_too_large() {
        with_env_var("RAG_RERANKER_TIMEOUT_SECS", "601", || {
            let err = Config::from_env().unwrap_err();
            assert!(err.message.contains("must be > 0 and <= 600"));
        });
    }
}
