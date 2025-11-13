use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio::sync::RwLock;

#[derive(Serialize)]
#[serde(untagged)]
enum OllamaEmbeddingRequest<'a> {
    Single { model: &'a str, prompt: &'a str },
    Batch { model: &'a str, input: &'a [String] },
}

#[derive(Deserialize)]
struct OllamaEmbeddingResponse {
    #[serde(default)]
    embedding: Option<Vec<f32>>,
    #[serde(default)]
    embeddings: Option<Vec<Vec<f32>>>,
}

pub struct EmbeddingService {
    client: reqwest::Client,
    ollama_url: String,
    model: String,
    query_cache: RwLock<HashMap<String, Vec<f32>>>,
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
            client: reqwest::Client::new(),
            ollama_url,
            model,
            query_cache: RwLock::new(HashMap::new()),
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
            prompt: text,
        };
        let response = self
            .client
            .post(format!("{}/api/embeddings", self.ollama_url))
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
        } else {
            Err(anyhow::anyhow!("No embedding returned from Ollama"))
        }
    }

    pub async fn get_query_embedding(&self, text: &str) -> Result<Vec<f32>> {
        if let Some(cached) = self.query_cache.read().await.get(text) {
            return Ok(cached.clone());
        }

        let embedding = self.get_embedding(text).await?;
        self.query_cache
            .write()
            .await
            .insert(text.to_string(), embedding.clone());
        Ok(embedding)
    }

    pub async fn embed_texts(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let request = if texts.len() == 1 {
            OllamaEmbeddingRequest::Single {
                model: &self.model,
                prompt: &texts[0],
            }
        } else {
            OllamaEmbeddingRequest::Batch {
                model: &self.model,
                input: texts,
            }
        };

        let response = self
            .client
            .post(format!("{}/api/embeddings", self.ollama_url))
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
            Ok(vec![embedding])
        } else if let Some(embeddings) = embedding_response.embeddings {
            Ok(embeddings)
        } else {
            Err(anyhow::anyhow!(
                "Ollama returned no embeddings for the provided input"
            ))
        }
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
