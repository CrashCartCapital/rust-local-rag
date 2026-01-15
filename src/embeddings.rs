use anyhow::Result;
use lru::LruCache;
use serde::{Deserialize, Serialize};
use std::num::NonZeroUsize;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::instrument;

use crate::config::Config;

#[derive(Serialize)]
#[serde(untagged)]
enum OllamaEmbeddingRequest<'a> {
    Single { model: &'a str, input: &'a str },
    Batch { model: &'a str, input: &'a [String] },
}

#[derive(Deserialize)]
struct OllamaEmbeddingResponse {
    #[serde(default)]
    embedding: Option<Vec<f32>>,
    #[serde(default)]
    embeddings: Option<Vec<Vec<f32>>>,
}

/// Embedding service using Ollama API with LRU query caching.
/// Supports both single and batch embedding operations.
pub struct EmbeddingService {
    client: reqwest::Client,
    ollama_url: String,
    model: String,
    query_cache: RwLock<LruCache<String, Vec<f32>>>,
    embedding_timeout: Duration,
}

impl EmbeddingService {
    pub async fn new(config: &Config) -> Result<Self> {
        let ollama_url =
            std::env::var("OLLAMA_URL").unwrap_or_else(|_| "http://localhost:11434".to_string());
        let model = std::env::var("OLLAMA_EMBEDDING_MODEL")
            .unwrap_or_else(|_| "nomic-embed-text".to_string());

        Self::new_with_settings(ollama_url, model, config).await
    }

    #[instrument(skip(ollama_url))]
    pub async fn new_with_config(ollama_url: String, model: String) -> Result<Self> {
        Self::new_with_settings(ollama_url, model, &Config::default()).await
    }

    #[instrument(skip(ollama_url, config))]
    pub async fn new_with_settings(
        ollama_url: String,
        model: String,
        config: &Config,
    ) -> Result<Self> {
        tracing::info!("Ollama URL: {}", ollama_url);
        tracing::info!("Ollama Model: {}", model);

        let service = Self {
            client: reqwest::Client::builder()
                .timeout(config.embedding_timeout) // per-batch for large documents
                .build()?,
            ollama_url,
            model,
            query_cache: RwLock::new(LruCache::new(
                NonZeroUsize::new(config.embedding_cache_size.get())
                    .expect("embedding_cache_size is non-zero"),
            )),
            embedding_timeout: config.embedding_timeout,
        };

        service.test_connection().await?;
        service.verify_model().await?;

        Ok(service)
    }

    pub fn model_name(&self) -> &str {
        &self.model
    }

    #[instrument(skip(self, text), fields(text_len = text.len()))]
    pub async fn get_embedding(&self, text: &str) -> Result<Vec<f32>> {
        let start = std::time::Instant::now();
        let request = OllamaEmbeddingRequest::Single {
            model: &self.model,
            input: text,
        };

        let request_future = self
            .client
            .post(format!("{}/api/embed", self.ollama_url))
            .json(&request)
            .send();

        // HARD TIMEOUT: Wrap request in tokio::time::timeout to prevent indefinite hangs
        let response = match tokio::time::timeout(self.embedding_timeout, request_future).await {
            Ok(Ok(resp)) => resp,
            Ok(Err(e)) => return Err(e.into()),
            Err(_) => {
                return Err(anyhow::anyhow!(
                    "Embedding request timed out after {} seconds. The Ollama server may be overloaded.",
                    self.embedding_timeout.as_secs()
                ));
            }
        };

        if !response.status().is_success() {
            return Err(anyhow::anyhow!(
                "Ollama API error: {} - {}",
                response.status(),
                response.text().await.unwrap_or_default()
            ));
        }
        let embedding_response: OllamaEmbeddingResponse = response.json().await?;

        tracing::debug!("Ollama embedding generated in {:?}", start.elapsed());

        if let Some(embedding) = embedding_response.embedding {
            Ok(embedding)
        } else if let Some(embeddings) = embedding_response.embeddings {
            embeddings
                .into_iter()
                .next()
                .ok_or_else(|| anyhow::anyhow!("Empty embeddings array from Ollama"))
        } else {
            Err(anyhow::anyhow!("No embedding returned from Ollama"))
        }
    }

    #[instrument(skip(self, text), fields(text_len = text.len()))]
    pub async fn get_query_embedding(&self, text: &str) -> Result<Vec<f32>> {
        if let Some(cached) = self.query_cache.write().await.get(text) {
            tracing::debug!("Query embedding cache hit");
            return Ok(cached.clone());
        }

        tracing::debug!("Query embedding cache miss");
        let embedding = self.get_embedding(text).await?;
        self.query_cache
            .write()
            .await
            .put(text.to_string(), embedding.clone());
        Ok(embedding)
    }

    #[instrument(skip(self, texts), fields(count = texts.len()))]
    pub async fn embed_texts(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let start = std::time::Instant::now();
        if texts.is_empty() {
            return Ok(vec![]);
        }

        // Try batch embedding first for multiple texts
        if texts.len() > 1 {
            let request = OllamaEmbeddingRequest::Batch {
                model: &self.model,
                input: texts,
            };

            // HARD TIMEOUT: Wrap request in tokio::time::timeout to prevent indefinite hangs
            // This creates an external "stopwatch" that will cancel the operation if it takes too long
            let request_future = self
                .client
                .post(format!("{}/api/embed", self.ollama_url))
                .json(&request)
                .send();

            let response = match tokio::time::timeout(self.embedding_timeout, request_future).await
            {
                Ok(Ok(resp)) => resp,
                Ok(Err(e)) => return Err(e.into()),
                Err(_) => {
                    return Err(anyhow::anyhow!(
                        "Batch embedding request timed out after {} seconds for {} texts. The Ollama server may be overloaded.",
                        self.embedding_timeout.as_secs(),
                        texts.len()
                    ));
                }
            };

            if !response.status().is_success() {
                return Err(anyhow::anyhow!(
                    "Ollama API error: {} - {}",
                    response.status(),
                    response.text().await.unwrap_or_default()
                ));
            }

            let embedding_response: OllamaEmbeddingResponse = response.json().await?;

            // Check if we got the expected number of embeddings
            if let Some(embeddings) = embedding_response.embeddings {
                if embeddings.len() == texts.len() {
                    return Ok(embeddings);
                }
                tracing::warn!(
                    "Batch embedding returned {} embeddings for {} texts, falling back to sequential",
                    embeddings.len(),
                    texts.len()
                );
            } else if embedding_response.embedding.is_some() {
                tracing::warn!(
                    "Model '{}' doesn't support batch embeddings, falling back to sequential",
                    self.model
                );
            }

            // Fall back to sequential embedding
            tracing::info!("Processing {} embeddings sequentially", texts.len());
            let mut result = Vec::with_capacity(texts.len());
            for text in texts {
                let embedding = self.get_embedding(text).await?;
                result.push(embedding);
            }
            tracing::info!("Batch embedding completed in {:?}", start.elapsed());
            return Ok(result);
        }

        // Single text - use standard single embedding request
        let embedding = self.get_embedding(&texts[0]).await?;
        tracing::info!("Batch embedding completed in {:?}", start.elapsed());
        Ok(vec![embedding])
    }

    #[instrument(skip(self))]
    async fn test_connection(&self) -> Result<()> {
        let response = self
            .client
            .get(format!("{}/api/tags", self.ollama_url))
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(anyhow::anyhow!(
                "Cannot connect to Ollama at {}. Make sure Ollama is running.",
                self.ollama_url
            ));
        }

        tracing::info!("Successfully connected to Ollama at {}", self.ollama_url);
        Ok(())
    }

    #[instrument(skip(self))]
    async fn verify_model(&self) -> Result<()> {
        let response = self
            .client
            .get(format!("{}/api/tags", self.ollama_url))
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!(
                "Failed to list models from Ollama: {} - {}",
                status,
                body
            ));
        }

        let tags: serde_json::Value = response.json().await?;
        let models = tags["models"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("Cannot list models"))?;

        let exists = models.iter().any(|m| {
            m.get("name")
                .and_then(|n| n.as_str())
                .is_some_and(|s| s.starts_with(&self.model))
        });

        if !exists {
            let available: Vec<_> = models.iter().filter_map(|m| m["name"].as_str()).collect();
            return Err(anyhow::anyhow!(
                "Model '{}' not found. Available: {:?}. Run: ollama pull {}",
                self.model,
                available,
                self.model
            ));
        }

        tracing::info!("✅ Model '{}' verified", self.model);
        Ok(())
    }
}

impl rag_core::EmbeddingBackend for EmbeddingService {
    fn model_id(&self) -> &str {
        self.model_name()
    }

    async fn embed(&self, text: &str) -> std::result::Result<Vec<f32>, rag_core::EmbeddingError> {
        self.get_query_embedding(text)
            .await
            .map_err(|e| rag_core::EmbeddingError::Api(e.to_string()))
    }

    async fn embed_batch(
        &self,
        texts: &[String],
    ) -> std::result::Result<Vec<Vec<f32>>, rag_core::EmbeddingError> {
        self.embed_texts(texts)
            .await
            .map_err(|e| rag_core::EmbeddingError::Api(e.to_string()))
    }

    fn dimension(&self) -> usize {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn test_get_embedding_timeout_enforced() {
        // Start a mock server
        let mock_server = MockServer::start().await;

        // Mock a delayed response for embedding
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({ "embedding": vec![0.1, 0.2] }))
                    .set_delay(Duration::from_millis(200)), // Delay longer than timeout
            )
            .mount(&mock_server)
            .await;

        // Mock tags endpoint for connection check
        Mock::given(method("GET"))
            .and(path("/api/tags"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "models": [ { "name": "nomic-embed-text:latest" } ]
            })))
            .mount(&mock_server)
            .await;

        // Create config with very short timeout
        let config = Config {
            embedding_timeout: Duration::from_millis(50),
            ..Config::default()
        };

        // Initialize service
        let service = EmbeddingService::new_with_settings(
            mock_server.uri(),
            "nomic-embed-text".to_string(),
            &config,
        )
        .await
        .expect("Failed to create service");

        // Perform request - should timeout
        let result = service.get_embedding("test text").await;

        // Assert timeout error
        assert!(result.is_err());
        let err = result.unwrap_err();

        // Check for our custom timeout message
        let is_custom_timeout = err.to_string().contains("Embedding request timed out");

        // Check for reqwest timeout (race condition means either can trigger)
        let is_reqwest_timeout = err
            .downcast_ref::<reqwest::Error>()
            .map(|e| e.is_timeout())
            .unwrap_or(false);

        // Fallback check on string message
        let err_msg = err.to_string();
        let is_generic_timeout = err_msg.contains("timed out") || err_msg.contains("timeout");

        assert!(
            is_custom_timeout || is_reqwest_timeout || is_generic_timeout,
            "Expected timeout error, got: {:?}",
            err
        );
    }
}
