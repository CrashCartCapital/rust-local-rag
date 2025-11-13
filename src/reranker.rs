use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone)]
pub struct RerankerCandidate {
    pub chunk_id: String,
    pub document: String,
    pub text: String,
    pub page_number: usize,
    pub section: Option<String>,
    pub initial_score: f32,
}

/// Represents the result of reranking a candidate chunk using an LLM.
///
/// The `relevance` field is the LLM-based reranking score (from 0.0 to 1.0),
/// which differs from the embedding similarity score.
pub struct RerankedResult {
    /// The identifier of the chunk that was reranked.
    pub chunk_id: String,
    /// The LLM-based relevance score (0.0 to 1.0) assigned during reranking.
    /// This is distinct from the embedding similarity score.
    pub relevance: f32,
}

#[derive(Serialize)]
struct OllamaGenerateRequest {
    model: String,
    prompt: String,
    #[serde(default)]
    stream: bool,
}

#[derive(Deserialize)]
struct OllamaGenerateResponse {
    response: String,
}

pub struct RerankerService {
    client: reqwest::Client,
    ollama_url: String,
    model: String,
}

impl RerankerService {
    /// Creates a new reranker service backed by Ollama.
    /// 
    /// This method initializes the service, validates the connection to Ollama,
    /// and verifies that the specified model is available.
    /// 
    /// # Configuration
    /// 
    /// The service is configured via environment variables:
    /// * `OLLAMA_URL` - The Ollama API endpoint (default: `http://localhost:11434`)
    /// * `OLLAMA_RERANK_MODEL` - The model to use for reranking (default: `llama3.1`)
    /// 
    /// # Errors
    /// 
    /// Returns an error if:
    /// * Cannot connect to Ollama at the configured URL
    /// * The specified model is not available (not pulled)
    pub async fn new() -> Result<Self> {
        let ollama_url =
            std::env::var("OLLAMA_URL").unwrap_or_else(|_| "http://localhost:11434".to_string());
        let model = std::env::var("OLLAMA_RERANK_MODEL").unwrap_or_else(|_| "llama3.1".to_string());

        let service = Self {
            client: reqwest::Client::new(),
            ollama_url,
            model,
        };

        service.test_connection().await?;
        service.verify_model().await?;

        Ok(service)
    }

    /// Performs second-stage reranking of search candidates using an LLM.
    ///
    /// This method scores each candidate's relevance to the query using an LLM,
    /// sorts the results by relevance score (highest first), and gracefully falls
    /// back to the initial embedding score if LLM scoring fails for any candidate.
    ///
    /// # Arguments
    ///
    /// * `query` - The search query to evaluate candidates against
    /// * `candidates` - The list of candidates to rerank
    ///
    /// # Returns
    ///
    /// A vector of reranked results sorted by relevance score in descending order
    pub async fn rerank(
        &self,
        query: &str,
        candidates: &[RerankerCandidate],
    ) -> Result<Vec<RerankedResult>> {
        let mut reranked = Vec::with_capacity(candidates.len());

        for candidate in candidates {
            let score = match self.score_candidate(query, candidate).await {
                Ok(score) => score,
                Err(err) => {
                    tracing::warn!(
                        "Falling back to embedding score for chunk {}: {}",
                        candidate.chunk_id,
                        err
                    );
                    candidate.initial_score
                }
            };

            reranked.push(RerankedResult {
                chunk_id: candidate.chunk_id.clone(),
                relevance: score,
            });
        }

        reranked.sort_by(|a, b| {
            b.relevance
                .partial_cmp(&a.relevance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(reranked)
    }

    async fn score_candidate(&self, query: &str, candidate: &RerankerCandidate) -> Result<f32> {
        let prompt = self.build_prompt(query, candidate);

        let request = OllamaGenerateRequest {
            model: self.model.clone(),
            prompt,
            stream: false,
        };

        let response = self
            .client
            .post(format!("{}/api/generate", self.ollama_url))
            .json(&request)
            .send()
            .await
            .context("Failed to contact Ollama reranker")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!("Reranker API error: {} - {}", status, body));
        }

        let payload: OllamaGenerateResponse = response
            .json()
            .await
            .context("Failed to parse reranker response")?;

        self.parse_score(&payload.response)
            .ok_or_else(|| anyhow::anyhow!("No numeric score in reranker response"))
    }

    fn build_prompt(&self, query: &str, candidate: &RerankerCandidate) -> String {
        let mut prompt = format!(
            "You are a retrieval relevance scorer. Given a search query and a document chunk, respond with a single number between 0 and 1 indicating how relevant the chunk is to the query.\n\nQuery: {query}\nDocument: {doc}\nPage: {page}\n",
            query = query.trim(),
            doc = candidate.document,
            page = if candidate.page_number == 0 {
                "unknown".to_string()
            } else {
                candidate.page_number.to_string()
            }
        );

        if let Some(section) = &candidate.section {
            if !section.trim().is_empty() {
                prompt.push_str(&format!("Section heading: {}\n", section.trim()));
            }
        }

        prompt.push_str("\nChunk:\n");
        prompt.push_str(candidate.text.trim());
        prompt.push_str(
            "\n\nRespond with only the numeric relevance score between 0 and 1, using decimal format.",
        );

        prompt
    }

    fn parse_score(&self, response: &str) -> Option<f32> {
        let mut number = String::new();
        let mut found_digit = false;

        for ch in response.chars() {
            if ch.is_ascii_digit() || ch == '.' {
                number.push(ch);
                found_digit = true;
            } else if found_digit {
                break;
            }
        }

        number
            .parse::<f32>()
            .ok()
            .map(|score| score.clamp(0.0, 1.0))
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
                "Rerank model '{}' not found. Available: {:?}. Run: ollama pull {}",
                self.model,
                available,
                self.model
            ));
        }

        tracing::info!("✅ Rerank model '{}' verified", self.model);
        Ok(())
    }
}
