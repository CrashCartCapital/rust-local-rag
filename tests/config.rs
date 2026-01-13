use rust_local_rag::Config;
use serial_test::serial;
use std::time::Duration;

#[test]
#[serial]
fn test_config_default_values() {
    let config = Config::default();
    assert_eq!(config.embedding_timeout, Duration::from_secs(1200));
    assert_eq!(config.embedding_cache_size.get(), 1000);
    assert_eq!(config.reranker_timeout, Duration::from_secs(60));
    assert_eq!(config.reranker_concurrency.get(), 1);
    assert_eq!(config.default_logprob_fallback, -10.0);
}

#[test]
#[serial]
fn test_config_env_override() {
    unsafe {
        std::env::set_var("RAG_EMBEDDING_TIMEOUT_SECS", "500");
        std::env::set_var("RAG_EMBEDDING_CACHE_SIZE", "123");
    }
    let config = Config::from_env().unwrap();
    assert_eq!(config.embedding_timeout, Duration::from_secs(500));
    assert_eq!(config.embedding_cache_size.get(), 123);
    unsafe {
        std::env::remove_var("RAG_EMBEDDING_TIMEOUT_SECS");
        std::env::remove_var("RAG_EMBEDDING_CACHE_SIZE");
    }
}

#[test]
#[serial]
fn test_config_invalid_env_returns_error() {
    unsafe {
        std::env::set_var("RAG_RERANKER_CONCURRENCY", "0");
    }
    let result = Config::from_env();
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("RAG_RERANKER_CONCURRENCY"));
    unsafe {
        std::env::remove_var("RAG_RERANKER_CONCURRENCY");
    }
}
