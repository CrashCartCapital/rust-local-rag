use anyhow::Result;
use lru::LruCache;
use serde::{Deserialize, Serialize};
use std::num::NonZeroUsize;
use tokio::sync::RwLock;

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
}

impl EmbeddingService {
    pub async fn new() -> Result<Self> {
        let ollama_url =
            std::env::var("OLLAMA_URL").unwrap_or_else(|_| "http://localhost:11434".to_string());
        let model = std::env::var("OLLAMA_EMBEDDING_MODEL")
            .unwrap_or_else(|_| "nomic-embed-text".to_string());

        tracing::info!("Ollama URL: {}", ollama_url);
        tracing::info!("Ollama Model: {}", model);

        let service = Self {
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(1200)) // 20 minutes per batch for large documents
                .build()?,
            ollama_url,
            model,
            query_cache: RwLock::new(LruCache::new(NonZeroUsize::new(1000).unwrap())),
        };

        service.test_connection().await?;
        service.verify_model().await?;

        Ok(service)
    }

    pub fn model_name(&self) -> &str {
        &self.model
    }

    pub async fn get_embedding(&self, text: &str) -> Result<Vec<f32>> {
        let request = OllamaEmbeddingRequest::Single {
            model: &self.model,
            input: text,
        };
        let response = self
            .client
            .post(format!("{}/api/embed", self.ollama_url))
            .json(&request)
            .send()
            .await?;
        if !response.status().is_success() {
            return Err(anyhow::anyhow!(
                "Ollama API error: {} - {}",
                response.status(),
                response.text().await.unwrap_or_default()
            ));
        }
        let embedding_response: OllamaEmbeddingResponse = response.json().await?;
        if let Some(embedding) = embedding_response.embedding {
            Ok(embedding)
        } else if let Some(embeddings) = embedding_response.embeddings {
            embeddings.into_iter().next()
                .ok_or_else(|| anyhow::anyhow!("Empty embeddings array from Ollama"))
        } else {
            Err(anyhow::anyhow!("No embedding returned from Ollama"))
        }
    }

    pub async fn get_query_embedding(&self, text: &str) -> Result<Vec<f32>> {
        if let Some(cached) = self.query_cache.write().await.get(text) {
            return Ok(cached.clone());
        }

        let embedding = self.get_embedding(text).await?;
        self.query_cache
            .write()
            .await
            .put(text.to_string(), embedding.clone());
        Ok(embedding)
    }

    pub async fn embed_texts(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
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
            const BATCH_TIMEOUT_SECS: u64 = 1200; // 20 minutes per batch
            let request_future = self
                .client
                .post(format!("{}/api/embed", self.ollama_url))
                .json(&request)
                .send();

            let response = match tokio::time::timeout(
                tokio::time::Duration::from_secs(BATCH_TIMEOUT_SECS),
                request_future,
            )
            .await
            {
                Ok(Ok(resp)) => resp,
                Ok(Err(e)) => return Err(e.into()),
                Err(_) => {
                    return Err(anyhow::anyhow!(
                        "Batch embedding request timed out after {} seconds for {} texts. The Ollama server may be overloaded.",
                        BATCH_TIMEOUT_SECS,
                        texts.len()
                    ))
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
            return Ok(result);
        }

        // Single text - use standard single embedding request
        let embedding = self.get_embedding(&texts[0]).await?;
        Ok(vec![embedding])
    }

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

        let exists = models
            .iter()
            .any(|m| m["name"].as_str().unwrap_or("").starts_with(&self.model));

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
