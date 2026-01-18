use std::num::NonZeroUsize;
use std::time::Duration;

use crate::RagError;

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
    pub fn from_env() -> Result<Self, RagError> {
        let mut config = Self::default();

        if let Some(timeout_secs) = parse_env_u64("RAG_EMBEDDING_TIMEOUT_SECS")? {
            config.embedding_timeout = Duration::from_secs(timeout_secs);
        }
        if let Some(cache_size) = parse_env_nonzero_usize("RAG_EMBEDDING_CACHE_SIZE")? {
            if cache_size.get() > 10000 {
                return Err(RagError::config(
                    "Invalid value for RAG_EMBEDDING_CACHE_SIZE",
                    format!("got {}, which exceeds 10000", cache_size.get()),
                    "Set RAG_EMBEDDING_CACHE_SIZE to an integer in [1, 10000]",
                ));
            }
            config.embedding_cache_size = cache_size;
        }

        if let Some(timeout_secs) = parse_env_u64("RAG_RERANKER_TIMEOUT_SECS")? {
            config.reranker_timeout = Duration::from_secs(timeout_secs);
        }
        if let Some(concurrency) = parse_env_nonzero_usize("RAG_RERANKER_CONCURRENCY")? {
            config.reranker_concurrency = concurrency;
        }

        if let Some(value) = parse_env_f64("RAG_DEFAULT_LOGPROB")? {
            if !value.is_finite() || value > 0.0 {
                return Err(RagError::config(
                    "Invalid value for RAG_DEFAULT_LOGPROB",
                    format!("got {value}, expected a finite number <= 0.0"),
                    "Set RAG_DEFAULT_LOGPROB to a finite number <= 0.0 (example: -10.0)",
                ));
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

fn parse_env_u64(var: &'static str) -> Result<Option<u64>, RagError> {
    match std::env::var(var) {
        Ok(val) => val.parse::<u64>().map(Some).map_err(|e| {
            RagError::config(
                format!("Invalid value for {var}"),
                format!("got '{val}': {e}"),
                format!("Set {var} to a positive integer (example: 1200)"),
            )
        }),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(e) => Err(RagError::config(
            format!("Failed to read environment variable {var}"),
            e.to_string(),
            format!("Ensure {var} is set to a valid UTF-8 value, or unset it"),
        )),
    }
}

fn parse_env_f64(var: &'static str) -> Result<Option<f64>, RagError> {
    match std::env::var(var) {
        Ok(val) => val.parse::<f64>().map(Some).map_err(|e| {
            RagError::config(
                format!("Invalid value for {var}"),
                format!("got '{val}': {e}"),
                format!("Set {var} to a valid number (example: -10.0)"),
            )
        }),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(e) => Err(RagError::config(
            format!("Failed to read environment variable {var}"),
            e.to_string(),
            format!("Ensure {var} is set to a valid UTF-8 value, or unset it"),
        )),
    }
}

fn parse_env_nonzero_usize(var: &'static str) -> Result<Option<NonZeroUsize>, RagError> {
    match std::env::var(var) {
        Ok(val) => {
            let parsed = val.parse::<usize>().map_err(|e| {
                RagError::config(
                    format!("Invalid value for {var}"),
                    format!("got '{val}': {e}"),
                    format!("Set {var} to a positive integer (example: 1000)"),
                )
            })?;
            NonZeroUsize::new(parsed)
                .ok_or_else(|| {
                    RagError::config(
                        format!("Invalid value for {var}"),
                        format!("got '{val}': must be > 0"),
                        format!("Set {var} to a positive integer (example: 1)"),
                    )
                })
                .map(Some)
        }
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(e) => Err(RagError::config(
            format!("Failed to read environment variable {var}"),
            e.to_string(),
            format!("Ensure {var} is set to a valid UTF-8 value, or unset it"),
        )),
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
        let _lock = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
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
            let msg = err.to_string();
            assert!(
                msg.contains("Invalid value for RAG_DEFAULT_LOGPROB"),
                "msg was: {msg}"
            );
            assert!(
                msg.contains("expected a finite number <= 0.0"),
                "msg was: {msg}"
            );
            assert!(msg.contains("Fix:"), "msg was: {msg}");
        });
    }

    #[test]
    fn test_rag_default_logprob_nan() {
        with_env_var("RAG_DEFAULT_LOGPROB", "NaN", || {
            let err = Config::from_env().unwrap_err();
            let msg = err.to_string();
            assert!(
                msg.contains("Invalid value for RAG_DEFAULT_LOGPROB"),
                "msg was: {msg}"
            );
            assert!(
                msg.contains("expected a finite number <= 0.0"),
                "msg was: {msg}"
            );
        });
    }

    #[test]
    fn test_rag_default_logprob_inf() {
        with_env_var("RAG_DEFAULT_LOGPROB", "inf", || {
            let err = Config::from_env().unwrap_err();
            let msg = err.to_string();
            assert!(
                msg.contains("Invalid value for RAG_DEFAULT_LOGPROB"),
                "msg was: {msg}"
            );
            assert!(
                msg.contains("expected a finite number <= 0.0"),
                "msg was: {msg}"
            );
        });
    }

    #[test]
    fn test_rag_embedding_cache_size_too_large() {
        with_env_var("RAG_EMBEDDING_CACHE_SIZE", "10001", || {
            let err = Config::from_env().unwrap_err();
            let msg = err.to_string();
            assert!(
                msg.contains("Invalid value for RAG_EMBEDDING_CACHE_SIZE"),
                "msg was: {msg}"
            );
            assert!(msg.contains("exceeds 10000"), "msg was: {msg}");
            assert!(msg.contains("Fix:"), "msg was: {msg}");
        });
    }
}
