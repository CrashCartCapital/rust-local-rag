use anyhow::{Context, Result};
use futures::stream::{FuturesUnordered, StreamExt};
use serde::{Deserialize, Serialize};
use tokio::time::{timeout, Duration, Instant};

#[derive(Clone)]
/// Represents a candidate chunk for reranking, containing metadata and initial retrieval score.
pub struct RerankerCandidate {
    /// Unique identifier for the chunk.
    pub chunk_id: String,
    /// Name or identifier of the source document.
    pub document: String,
    /// The text content of the chunk.
    pub text: String,
    /// The page number in the source document where the chunk is located.
    pub page_number: usize,
    /// The section name or identifier within the document, if available.
    pub section: Option<String>,
    /// Embedding similarity score from the first-stage retrieval.
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

/// Calibration statistics for determining optimal timeout
pub struct CalibrationStats {
    pub mean_ms: f64,
    pub median_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub max_ms: f64,
    pub sample_size: usize,
}

#[derive(Serialize)]
struct OllamaGenerateRequest {
    model: String,
    prompt: String,
    #[serde(default)]
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<OllamaOptions>,
}

#[derive(Serialize)]
struct OllamaOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
}

#[derive(Deserialize)]
struct OllamaGenerateResponse {
    response: String,
}

/// LLM-based relevance reranking service using Ollama.
/// Scores search candidates using structured JSON output with Phi chat template.
pub struct RerankerService {
    client: reqwest::Client,
    ollama_url: String,
    model: String,
    prompt_template: String,
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

        // Load prompt template from file or use default
        let prompt_template = Self::load_prompt_template();

        // Configure client with optimized connection pooling settings
        // to avoid connection overhead on each request
        let client = reqwest::Client::builder()
            .pool_max_idle_per_host(10)  // Keep up to 10 idle connections per host
            .pool_idle_timeout(Some(Duration::from_secs(300)))  // Keep connections alive for 5 minutes
            .tcp_keepalive(Some(Duration::from_secs(30)))  // TCP keepalive every 30s
            .timeout(Duration::from_secs(120))  // 2-minute timeout for reranking requests
            .build()
            .context("Failed to build HTTP client")?;

        let service = Self {
            client,
            ollama_url,
            model,
            prompt_template,
        };

        service.test_connection().await?;
        service.verify_model().await?;

        Ok(service)
    }

    /// Load prompt template from external file or fall back to default
    fn load_prompt_template() -> String {
        let prompts_dir = std::env::var("PROMPTS_DIR").unwrap_or_else(|_| "./prompts".to_string());
        let prompt_path = std::path::Path::new(&prompts_dir).join("reranker.txt");

        match std::fs::read_to_string(&prompt_path) {
            Ok(template) => {
                tracing::info!("Loaded reranker prompt from {}", prompt_path.display());
                template
            }
            Err(_) => {
                tracing::info!("Using default reranker prompt (no external file found at {})", prompt_path.display());
                Self::default_prompt_template()
            }
        }
    }

    /// Default prompt template (compiled in as fallback)
    fn default_prompt_template() -> String {
        r#"You are a semantic relevance judge for a RAG system. Your job is to score how well a document chunk ANSWERS the user's underlying question - not just whether it contains matching keywords.

Scoring guidelines (0-100):
- 90-100: Directly answers the question with actionable information
- 70-89: Highly relevant context that helps answer the question
- 50-69: Related topic but doesn't directly address the question
- 30-49: Tangentially related, mostly background information
- 0-29: Not relevant to what the user is actually asking

IMPORTANT: A chunk mentioning the same words as the query is NOT automatically relevant. Judge whether the chunk would actually help someone who asked this question.

Query: {query}
Document: {document}
Page: {page}
Section: {section}

Chunk:
{text}

Score (0-100):"#.to_string()
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
        // Use sequential processing to avoid memory saturation
        // M2 Max memory-bound: concurrent requests cause KV cache contention
        let concurrency_limit = 1;
        // 60s timeout to handle p99 latency (36s observed) with 66% buffer
        let timeout_duration = Duration::from_secs(60);

        let mut futures = FuturesUnordered::new();
        let mut results = Vec::with_capacity(candidates.len());
        let mut candidate_iter = candidates.iter();

        // Seed the futures pool with initial batch
        for _ in 0..concurrency_limit.min(candidates.len()) {
            if let Some(candidate) = candidate_iter.next() {
                futures.push(self.score_with_timeout(query, candidate, timeout_duration));
            }
        }

        // Process remaining candidates as futures complete
        while let Some(result) = futures.next().await {
            results.push(result);

            // Add next candidate if available
            if let Some(candidate) = candidate_iter.next() {
                futures.push(self.score_with_timeout(query, candidate, timeout_duration));
            }
        }

        // Sort by relevance score (highest first)
        results.sort_by(|a, b| {
            b.relevance
                .partial_cmp(&a.relevance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(results)
    }

    async fn score_with_timeout(
        &self,
        query: &str,
        candidate: &RerankerCandidate,
        timeout_duration: Duration,
    ) -> RerankedResult {
        let chunk_id = candidate.chunk_id.clone();
        let initial_score = candidate.initial_score;

        let score_result = timeout(
            timeout_duration,
            self.score_candidate(query, candidate),
        ).await;

        let score = match score_result {
            Ok(Ok(score)) => score,
            Ok(Err(err)) => {
                tracing::warn!(
                    "Reranking failed for chunk {}, falling back to embedding score: {}",
                    chunk_id,
                    err
                );
                initial_score
            }
            Err(_) => {
                tracing::warn!(
                    "Reranking timeout for chunk {}, falling back to embedding score",
                    chunk_id
                );
                initial_score
            }
        };

        RerankedResult { chunk_id, relevance: score }
    }

    async fn score_candidate(&self, query: &str, candidate: &RerankerCandidate) -> Result<f32> {
        let overall_start = Instant::now();

        // Phase 1: Build prompt
        let phase1_start = Instant::now();
        let prompt = self.build_prompt(query, candidate);
        let phase1_elapsed = phase1_start.elapsed().as_millis();

        // Phase 2: Create request with stop sequences for Phi template
        let phase2_start = Instant::now();
        let request = OllamaGenerateRequest {
            model: self.model.clone(),
            prompt,
            stream: false,
            options: Some(OllamaOptions {
                stop: Some(vec!["<|end|>".to_string(), "<|user|>".to_string()]),
                temperature: Some(0.1), // Low temperature for consistent scoring
            }),
        };
        let phase2_elapsed = phase2_start.elapsed().as_millis();

        // Phase 3: Send request and get response
        // Request-level timeout ensures clean socket cancellation
        let phase3_start = Instant::now();
        let response = self
            .client
            .post(format!("{}/api/generate", self.ollama_url))
            .timeout(Duration::from_secs(60))
            .json(&request)
            .send()
            .await
            .context("Failed to contact Ollama reranker")?;
        let phase3_elapsed = phase3_start.elapsed().as_millis();

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!("Reranker API error: {} - {}", status, body));
        }

        // Phase 4: Parse response
        let phase4_start = Instant::now();
        let payload: OllamaGenerateResponse = response
            .json()
            .await
            .context("Failed to parse reranker response")?;
        let phase4_elapsed = phase4_start.elapsed().as_millis();

        // Phase 5: Extract score
        let phase5_start = Instant::now();
        let score = self.parse_score(&payload.response)
            .ok_or_else(|| anyhow::anyhow!("No numeric score in reranker response"))?;
        let phase5_elapsed = phase5_start.elapsed().as_millis();

        let total_elapsed = overall_start.elapsed().as_millis();

        // Log detailed timing breakdown
        tracing::debug!(
            "Reranking timing for chunk {}: total={}ms [build_prompt={}ms, create_request={}ms, api_send={}ms, parse_response={}ms, extract_score={}ms]",
            candidate.chunk_id,
            total_elapsed,
            phase1_elapsed,
            phase2_elapsed,
            phase3_elapsed,
            phase4_elapsed,
            phase5_elapsed
        );

        Ok(score)
    }

    fn build_prompt(&self, query: &str, candidate: &RerankerCandidate) -> String {
        let page = if candidate.page_number == 0 {
            "unknown".to_string()
        } else {
            candidate.page_number.to_string()
        };

        let section = candidate
            .section
            .as_ref()
            .filter(|s| !s.trim().is_empty())
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|| "N/A".to_string());

        self.prompt_template
            .replace("{query}", query.trim())
            .replace("{document}", &candidate.document)
            .replace("{page}", &page)
            .replace("{section}", &section)
            .replace("{text}", candidate.text.trim())
    }

    /// Parse score from LLM response, handling JSON format with pre-fill.
    /// Reconstructs JSON from pre-filled prompt, extracts "score" field.
    /// Falls back to finding first number if JSON parsing fails.
    /// Normalizes all scores to 0.0-1.0 range.
    fn parse_score(&self, response: &str) -> Option<f32> {
        tracing::debug!(
            response_preview = %response.chars().take(200).collect::<String>(),
            "Parsing reranker response"
        );

        // Reconstruct full JSON by prepending the pre-fill from the prompt
        // Prompt ends with: {"classification": "
        // Model completes: DIRECT_ANSWER", "reasoning": "...", "score": 95}
        let full_json = format!(r#"{{"classification": "{response}"#);

        // Try to parse as JSON and extract score
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&full_json) {
            if let Some(score) = json.get("score").and_then(|v| v.as_f64()) {
                let score = score as f32;
                tracing::debug!(raw_score = score, "Extracted score from JSON");
                // Normalize: if > 1.0, assume 0-100 scale
                return Some(if score > 1.0 {
                    (score / 100.0).clamp(0.0, 1.0)
                } else {
                    score.clamp(0.0, 1.0)
                });
            }
        }

        // Fallback: find "score" followed by a number in the response
        tracing::debug!("JSON parse failed, trying text fallback");
        let lower = response.to_lowercase();
        if let Some(score_pos) = lower.find("score") {
            let after_score = &response[score_pos..];
            let mut num = String::new();
            let mut found_digit = false;
            for ch in after_score.chars() {
                if ch.is_ascii_digit() || ch == '.' {
                    num.push(ch);
                    found_digit = true;
                } else if found_digit {
                    break;
                }
            }
            if let Ok(score) = num.trim().parse::<f32>() {
                tracing::debug!(raw_score = score, "Extracted score via text fallback");
                return Some(if score > 1.0 {
                    (score / 100.0).clamp(0.0, 1.0)
                } else {
                    score.clamp(0.0, 1.0)
                });
            }
        }

        // Last resort: first number in response
        let mut num = String::new();
        let mut found_digit = false;
        for ch in response.chars() {
            if ch.is_ascii_digit() || ch == '.' {
                num.push(ch);
                found_digit = true;
            } else if found_digit {
                break;
            }
        }

        num.trim().parse::<f32>().ok().map(|score| {
            tracing::debug!(raw_score = score, "Extracted score via last resort");
            if score > 1.0 {
                (score / 100.0).clamp(0.0, 1.0)
            } else {
                score.clamp(0.0, 1.0)
            }
        })
    }

    /// Calibrate timeout by measuring actual LLM latencies
    ///
    /// Runs a sample of reranking requests without timeout to measure
    /// actual inference times, then calculates statistics to determine
    /// an appropriate timeout that allows 99% of requests to complete.
    ///
    /// # Warm-up Phase
    ///
    /// The first few LLM requests are often slower due to cold starts.
    /// This method runs 2-3 warm-up requests before measurement to avoid bias.
    ///
    /// # Sample Size
    ///
    /// For reliable p99 estimation, use at least 50-100 samples. Smaller
    /// samples will produce unstable estimates sensitive to outliers.
    pub async fn calibrate_timeout(
        &self,
        query: &str,
        candidates: &[RerankerCandidate],
        sample_size: usize,
    ) -> Result<CalibrationStats> {
        let sample_size = sample_size.min(candidates.len());
        if sample_size == 0 {
            return Err(anyhow::anyhow!("No candidates provided for calibration"));
        }

        tracing::info!(
            "Starting reranker calibration with {} samples (warm-up phase + measurement)",
            sample_size
        );

        // Warm-up phase: run 2 untimed requests to avoid cold-start bias
        let warmup_count = 2.min(candidates.len());
        if warmup_count > 0 {
            tracing::debug!("Running {} warm-up requests...", warmup_count);
            for candidate in candidates.iter().take(warmup_count) {
                let _ = self.score_candidate(query, candidate).await;
            }
        }

        let mut durations_ms = Vec::with_capacity(sample_size);

        // Measure latencies for sample sequentially (concurrency=1)
        let mut futures = FuturesUnordered::new();
        let mut candidate_iter = candidates.iter().take(sample_size);

        // Seed with first candidate (sequential processing)
        for _ in 0..1.min(sample_size) {
            if let Some(candidate) = candidate_iter.next() {
                futures.push(self.score_with_timing(query, candidate));
            }
        }

        // Process remaining candidates as futures complete
        while let Some((duration, result)) = futures.next().await {
            let duration_ms = duration.as_millis() as f64;
            durations_ms.push(duration_ms);

            if result.is_ok() {
                tracing::debug!("Calibration sample completed in {:.0}ms", duration_ms);
            } else {
                tracing::warn!("Calibration sample failed in {:.0}ms: {:?}", duration_ms, result.unwrap_err());
            }

            // Add next candidate if available
            if let Some(candidate) = candidate_iter.next() {
                futures.push(self.score_with_timing(query, candidate));
            }
        }

        // Calculate statistics
        durations_ms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mean_ms = durations_ms.iter().sum::<f64>() / durations_ms.len() as f64;
        let median_ms = durations_ms[durations_ms.len() / 2];

        // Correct percentile calculation using nearest-rank method
        // Formula: index = round(percentile * (N - 1))
        let n = durations_ms.len();
        let p95_idx = ((0.95 * (n - 1) as f64).round() as usize).min(n - 1);
        let p99_idx = ((0.99 * (n - 1) as f64).round() as usize).min(n - 1);
        let p95_ms = durations_ms[p95_idx];
        let p99_ms = durations_ms[p99_idx];
        let max_ms = *durations_ms.last().unwrap();

        let stats = CalibrationStats {
            mean_ms,
            median_ms,
            p95_ms,
            p99_ms,
            max_ms,
            sample_size: durations_ms.len(),
        };

        tracing::info!(
            "Calibration complete: mean={:.0}ms, median={:.0}ms, p95={:.0}ms, p99={:.0}ms, max={:.0}ms (sample_size={})",
            stats.mean_ms,
            stats.median_ms,
            stats.p95_ms,
            stats.p99_ms,
            stats.max_ms,
            stats.sample_size
        );

        Ok(stats)
    }

    async fn score_with_timing(
        &self,
        query: &str,
        candidate: &RerankerCandidate,
    ) -> (std::time::Duration, Result<f32>) {
        let start = Instant::now();
        let result = self.score_candidate(query, candidate).await;
        let duration = start.elapsed();
        (duration, result)
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
