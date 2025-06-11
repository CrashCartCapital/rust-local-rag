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
        let service = Self {
            client: reqwest::Client::new(),
            ollama_url: "http://localhost:11434".to_string(),
            model: "nomic-embed-text".to_string(),
        };

        service.test_connection().await?;

        Ok(service)
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
}
