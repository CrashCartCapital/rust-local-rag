use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Serialize)]
struct OllamaEmbeddingRequest {
    model: String,
    prompt: String,
}

#[derive(Deserialize)]
struct OllamaEmbeddingResponse {
    embedding: Vec<f32>,
}

pub struct EmbeddingService {
    client: reqwest::Client,
    ollama_url: String,
    model: String,
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
        };

        service.test_connection().await?;
        service.verify_model().await?;

        Ok(service)
    }

    pub fn model_name(&self) -> &str {
        &self.model
    }

    pub async fn get_embedding(&self, text: &str) -> Result<Vec<f32>> {
        let request = OllamaEmbeddingRequest {
            model: self.model.clone(),
            prompt: text.to_string(),
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
        Ok(embedding_response.embedding)
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
