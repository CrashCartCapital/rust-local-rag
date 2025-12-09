//! Configuration loading for the RAG TUI
//!
//! Centralizes environment variable reading into a single struct.

use crate::constants::{DEFAULT_POLL_INTERVAL_SECS, DEFAULT_TOP_K};

/// TUI configuration loaded from environment variables
#[derive(Debug, Clone)]
pub struct Config {
    /// RAG server URL
    pub server_url: String,

    /// Data directory path
    pub data_dir: String,

    /// Documents directory path
    pub documents_dir: String,

    /// Ollama server URL
    pub ollama_url: String,

    /// Polling interval for stats updates (seconds)
    pub poll_interval_secs: u64,

    /// Default number of search results
    pub top_k: usize,

    /// Theme name (dark, light, high-contrast)
    pub theme: String,
}

impl Config {
    /// Load configuration from environment variables with defaults
    pub fn from_env() -> Self {
        Self {
            server_url: std::env::var("RAG_TUI_SERVER_URL")
                .unwrap_or_else(|_| "http://localhost:3046".to_string()),
            data_dir: std::env::var("DATA_DIR")
                .unwrap_or_else(|_| "./data".to_string()),
            documents_dir: std::env::var("DOCUMENTS_DIR")
                .unwrap_or_else(|_| "./documents".to_string()),
            ollama_url: std::env::var("OLLAMA_URL")
                .unwrap_or_else(|_| "localhost:11434".to_string()),
            poll_interval_secs: std::env::var("RAG_TUI_POLL_INTERVAL_S")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(DEFAULT_POLL_INTERVAL_SECS),
            top_k: std::env::var("RAG_TUI_TOP_K")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(DEFAULT_TOP_K),
            theme: std::env::var("RAG_TUI_THEME")
                .unwrap_or_else(|_| "dark".to_string()),
        }
    }

    /// Build a summary string for display
    pub fn summary(&self) -> String {
        format!(
            "DATA_DIR={}  DOCS_DIR={}  OLLAMA={}",
            self.data_dir, self.documents_dir, self.ollama_url
        )
    }
}

impl Default for Config {
    fn default() -> Self {
        Self::from_env()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        // This test verifies defaults when env vars are not set
        // Note: In practice, other env vars might be set
        let config = Config::from_env();

        // These should have sensible defaults
        assert!(!config.server_url.is_empty());
        assert!(!config.data_dir.is_empty());
        assert!(!config.documents_dir.is_empty());
        assert!(!config.ollama_url.is_empty());
        assert!(config.poll_interval_secs > 0);
        assert!(config.top_k > 0);
    }

    #[test]
    fn test_config_summary() {
        let config = Config {
            server_url: "http://localhost:3046".to_string(),
            data_dir: "./data".to_string(),
            documents_dir: "./documents".to_string(),
            ollama_url: "localhost:11434".to_string(),
            poll_interval_secs: 2,
            top_k: 10,
            theme: "dark".to_string(),
        };

        let summary = config.summary();
        assert!(summary.contains("DATA_DIR=./data"));
        assert!(summary.contains("DOCS_DIR=./documents"));
        assert!(summary.contains("OLLAMA=localhost:11434"));
    }

    #[test]
    fn test_config_theme_default() {
        let config = Config::from_env();
        // Theme should have a value (dark is default)
        assert!(!config.theme.is_empty());
    }
}
