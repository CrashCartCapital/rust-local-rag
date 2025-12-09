//! Ollama API client for fetching available models
//!
//! Provides async model discovery from a running Ollama instance.

use reqwest::Client;
use serde::Deserialize;
use std::time::Duration;

/// Model information from Ollama API
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct OllamaModel {
    /// Model name (e.g., "nomic-embed-text:latest")
    pub name: String,
    /// Model size in bytes
    #[serde(default)]
    pub size: u64,
    /// Model details (contains family info)
    #[serde(default)]
    pub details: Option<ModelDetails>,
}

/// Nested details from Ollama API response
#[derive(Debug, Clone, Deserialize, Default)]
#[allow(dead_code)]
pub struct ModelDetails {
    /// Model family (e.g., "nomic-bert", "llama", "phi")
    pub family: Option<String>,
    /// Parameter size (e.g., "137M", "7B")
    pub parameter_size: Option<String>,
}

#[allow(dead_code)]
impl OllamaModel {
    /// Get the model family, if available
    pub fn family(&self) -> Option<&str> {
        self.details.as_ref().and_then(|d| d.family.as_deref())
    }

    /// Format size as human-readable string
    pub fn size_display(&self) -> String {
        if self.size == 0 {
            "?".to_string()
        } else if self.size >= 1_000_000_000 {
            format!("{:.1} GB", self.size as f64 / 1_000_000_000.0)
        } else if self.size >= 1_000_000 {
            format!("{:.0} MB", self.size as f64 / 1_000_000.0)
        } else {
            format!("{:.0} KB", self.size as f64 / 1_000.0)
        }
    }
}

/// Response wrapper for /api/tags endpoint
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct TagsResponse {
    models: Vec<OllamaModel>,
}

/// Fetch available models from Ollama
///
/// # Arguments
/// * `client` - Reusable reqwest client (created once, cloned for tasks)
/// * `base_url` - Ollama server URL (e.g., "localhost:11434" or "http://localhost:11434")
///
/// # Returns
/// * `Ok(Vec<OllamaModel>)` - List of available models
/// * `Err(String)` - Human-readable error message
#[allow(dead_code)]
pub async fn fetch_models(client: &Client, base_url: &str) -> Result<Vec<OllamaModel>, String> {
    // Normalize URL - add http:// if not present
    let base = base_url.trim_end_matches('/');
    let url = if base.starts_with("http://") || base.starts_with("https://") {
        format!("{base}/api/tags")
    } else {
        format!("http://{base}/api/tags")
    };

    // Fetch with timeout (5s to accommodate slow/loaded Ollama servers)
    let response = client
        .get(&url)
        .timeout(Duration::from_secs(5))
        .send()
        .await
        .map_err(|e| {
            if e.is_timeout() {
                "Ollama not responding (timeout)".to_string()
            } else if e.is_connect() {
                format!("Cannot connect to Ollama at {base_url}")
            } else {
                format!("Request failed: {e}")
            }
        })?;

    // Check status
    let response = response.error_for_status().map_err(|e| {
        format!(
            "Ollama returned error: {}",
            e.status().map_or("unknown".to_string(), |s| s.to_string())
        )
    })?;

    // Parse JSON
    let tags: TagsResponse = response
        .json()
        .await
        .map_err(|e| format!("Invalid response from Ollama: {e}"))?;

    Ok(tags.models)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_size_display() {
        let model = OllamaModel {
            name: "test".to_string(),
            size: 274_000_000,
            details: None,
        };
        assert_eq!(model.size_display(), "274 MB");

        let model = OllamaModel {
            name: "test".to_string(),
            size: 2_200_000_000,
            details: None,
        };
        assert_eq!(model.size_display(), "2.2 GB");

        let model = OllamaModel {
            name: "test".to_string(),
            size: 0,
            details: None,
        };
        assert_eq!(model.size_display(), "?");
    }

    #[test]
    fn test_family_accessor() {
        let model = OllamaModel {
            name: "test".to_string(),
            size: 0,
            details: Some(ModelDetails {
                family: Some("llama".to_string()),
                parameter_size: None,
            }),
        };
        assert_eq!(model.family(), Some("llama"));

        let model_no_details = OllamaModel {
            name: "test".to_string(),
            size: 0,
            details: None,
        };
        assert_eq!(model_no_details.family(), None);
    }
}
