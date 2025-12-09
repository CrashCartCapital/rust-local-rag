use anyhow::{Context, Result};
use futures::stream::{FuturesUnordered, StreamExt};
use serde::{Deserialize, Serialize};
use tokio::time::{Duration, Instant, timeout};

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
    /// Log probability of "yes" token (for softmax scoring transparency)
    #[allow(dead_code)] // Will be used by TUI display
    pub yes_logprob: Option<f64>,
    /// Log probability of "no" token (for softmax scoring transparency)
    #[allow(dead_code)] // Will be used by TUI display
    pub no_logprob: Option<f64>,
}

/// Detailed score result including logprobs for transparency
struct DetailedScore {
    score: f32,
    yes_logprob: Option<f64>,
    no_logprob: Option<f64>,
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
    /// Request log probabilities for output tokens (requires Ollama v0.12.11+)
    #[serde(skip_serializing_if = "Option::is_none")]
    logprobs: Option<bool>,
    /// Number of top logprobs to return per token position (0-20)
    #[serde(skip_serializing_if = "Option::is_none")]
    top_logprobs: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<OllamaOptions>,
}

#[derive(Serialize)]
struct OllamaOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    num_predict: Option<i32>,
}

#[derive(Deserialize, Debug)]
struct OllamaGenerateResponse {
    response: String,
    /// Log probability information for generated tokens (Ollama v0.12.11+)
    #[serde(default)]
    logprobs: Option<Vec<TokenLogprob>>,
}

#[derive(Deserialize, Debug)]
struct TokenLogprob {
    /// The generated token
    token: String,
    /// Log probability of this token
    logprob: f64,
    /// Top alternative tokens with their logprobs
    #[serde(default)]
    top_logprobs: Option<Vec<TopLogprob>>,
}

#[derive(Deserialize, Debug)]
struct TopLogprob {
    /// Alternative token
    token: String,
    /// Log probability of this alternative
    logprob: f64,
}

/// LLM-based relevance reranking service using Ollama.
/// Scores search candidates using Yes/No classification for binary relevance.
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
            .pool_max_idle_per_host(10) // Keep up to 10 idle connections per host
            .pool_idle_timeout(Some(Duration::from_secs(300))) // Keep connections alive for 5 minutes
            .tcp_keepalive(Some(Duration::from_secs(30))) // TCP keepalive every 30s
            .timeout(Duration::from_secs(120)) // 2-minute timeout for reranking requests
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

    /// Returns the name of the reranking model being used.
    pub fn model_name(&self) -> &str {
        &self.model
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
                tracing::info!(
                    "Using default reranker prompt (no external file found at {})",
                    prompt_path.display()
                );
                Self::default_prompt_template()
            }
        }
    }

    /// Default prompt template (compiled in as fallback)
    /// Uses Yes/No format for logprobs-based scoring with Qwen3-Reranker
    /// Enhanced for semantic nuance capture beyond keyword matching
    fn default_prompt_template() -> String {
        r#"Query: {query}
Document: {document}
Page: {page}

Chunk:
{text}

Consider semantic meaning, not just keyword matches. A chunk is relevant if it:
- Directly answers the query
- Provides essential context or definitions
- Contains logically related information that helps address the query

Does this chunk contain relevant information for the query?
Answer:"#
            .to_string()
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

        let score_result = timeout(timeout_duration, self.score_candidate(query, candidate)).await;

        match score_result {
            Ok(Ok(detailed)) => RerankedResult {
                chunk_id,
                relevance: detailed.score,
                yes_logprob: detailed.yes_logprob,
                no_logprob: detailed.no_logprob,
            },
            Ok(Err(err)) => {
                tracing::warn!(
                    "Reranking failed for chunk {}, falling back to embedding score: {}",
                    chunk_id,
                    err
                );
                RerankedResult {
                    chunk_id,
                    relevance: initial_score,
                    yes_logprob: None,
                    no_logprob: None,
                }
            }
            Err(_) => {
                tracing::warn!(
                    "Reranking timeout for chunk {}, falling back to embedding score",
                    chunk_id
                );
                RerankedResult {
                    chunk_id,
                    relevance: initial_score,
                    yes_logprob: None,
                    no_logprob: None,
                }
            }
        }
    }

    async fn score_candidate(
        &self,
        query: &str,
        candidate: &RerankerCandidate,
    ) -> Result<DetailedScore> {
        let overall_start = Instant::now();

        // Phase 1: Build prompt
        let phase1_start = Instant::now();
        let prompt = self.build_prompt(query, candidate);
        let phase1_elapsed = phase1_start.elapsed().as_millis();

        // DEBUG: Log the full prompt being sent
        tracing::info!(
            "\n=== RERANKER DEBUG [{}] ===\nQuery: {}\nDocument: {}\nPage: {}\n--- FULL PROMPT ---\n{}\n--- END PROMPT ---",
            &candidate.chunk_id[..8],
            query,
            candidate.document,
            candidate.page_number,
            prompt
        );

        // Phase 2: Create request for Yes/No classification with logprobs
        let phase2_start = Instant::now();
        let request = OllamaGenerateRequest {
            model: self.model.clone(),
            prompt,
            stream: false,
            logprobs: Some(true),  // Enable logprobs for softmax scoring
            top_logprobs: Some(5), // Get top 5 alternatives to find yes/no
            options: Some(OllamaOptions {
                stop: Some(vec!["\n".to_string()]), // Stop at newline after Yes/No
                temperature: Some(0.0),             // Zero temperature for deterministic scoring
                num_predict: Some(3),               // Only need 1-3 tokens for Yes/No
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

        // DEBUG: Log the raw response and logprobs from Ollama
        tracing::info!(
            "\n=== RERANKER RESPONSE [{}] ===\nRaw response: '{}'\nLogprobs: {:?}\n--- END RESPONSE ---",
            &candidate.chunk_id[..8],
            payload.response,
            payload.logprobs
        );

        // Phase 5: Extract score from logprobs (preferred) or fall back to text parsing
        let phase5_start = Instant::now();
        let detailed = if let Some(ref logprobs) = payload.logprobs {
            self.compute_score_from_logprobs(logprobs, &candidate.chunk_id)
                .unwrap_or_else(|| {
                    tracing::warn!(
                        "Logprobs parsing failed for chunk {}, falling back to text",
                        &candidate.chunk_id[..8]
                    );
                    DetailedScore {
                        score: self.parse_score(&payload.response).unwrap_or(0.5),
                        yes_logprob: None,
                        no_logprob: None,
                    }
                })
        } else {
            tracing::warn!(
                "No logprobs in response for chunk {}, using text parsing",
                &candidate.chunk_id[..8]
            );
            let score = self
                .parse_score(&payload.response)
                .ok_or_else(|| anyhow::anyhow!("No score in reranker response"))?;
            DetailedScore {
                score,
                yes_logprob: None,
                no_logprob: None,
            }
        };
        let phase5_elapsed = phase5_start.elapsed().as_millis();

        // DEBUG: Log the parsed score
        tracing::info!(
            "=== RERANKER SCORE [{}] === Parsed score: {:.4} (yes_lp: {:?}, no_lp: {:?}) ===",
            &candidate.chunk_id[..8],
            detailed.score,
            detailed.yes_logprob,
            detailed.no_logprob
        );

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

        Ok(detailed)
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

    /// Parse score from LLM response by detecting Yes/No answer.
    /// Returns 1.0 for "Yes" responses, 0.0 for "No" responses.
    /// Falls back to 0.5 if neither is detected.
    fn parse_score(&self, response: &str) -> Option<f32> {
        let response_lower = response.to_lowercase();
        let response_trimmed = response_lower.trim();

        tracing::debug!(
            response = %response_trimmed,
            "Parsing Yes/No reranker response"
        );

        // Check for Yes/No at the start of the response
        if response_trimmed.starts_with("yes") {
            tracing::debug!("Detected YES - score 1.0");
            return Some(1.0);
        }

        if response_trimmed.starts_with("no") {
            tracing::debug!("Detected NO - score 0.0");
            return Some(0.0);
        }

        // Fallback: check if yes/no appears anywhere
        if response_trimmed.contains("yes") && !response_trimmed.contains("no") {
            tracing::debug!("Found 'yes' in response - score 1.0");
            return Some(1.0);
        }

        if response_trimmed.contains("no") && !response_trimmed.contains("yes") {
            tracing::debug!("Found 'no' in response - score 0.0");
            return Some(0.0);
        }

        // Ambiguous response - return middle score
        tracing::warn!(
            response = %response,
            "Ambiguous reranker response, defaulting to 0.5"
        );
        Some(0.5)
    }

    /// Compute relevance score from logprobs using softmax over yes/no probabilities.
    ///
    /// This implements the official Qwen3-Reranker scoring formula:
    /// score = exp(yes_logprob) / (exp(yes_logprob) + exp(no_logprob))
    ///
    /// The function aggregates all yes-like tokens ("yes", "Yes", "YES") and
    /// no-like tokens ("no", "No", "NO") from the top_logprobs to get robust estimates.
    ///
    /// Returns a DetailedScore with the softmax score and the raw logprobs for transparency.
    fn compute_score_from_logprobs(
        &self,
        logprobs: &[TokenLogprob],
        chunk_id: &str,
    ) -> Option<DetailedScore> {
        // We only need the first token's logprobs (the yes/no decision)
        let first_token = logprobs.first()?;
        let top_probs = first_token.top_logprobs.as_ref()?;

        // Find the best yes-like and no-like logprobs
        let mut yes_logprob: Option<f64> = None;
        let mut no_logprob: Option<f64> = None;

        for prob in top_probs {
            let token_lower = prob.token.to_lowercase();
            let token_trimmed = token_lower.trim();
            // Strip punctuation to handle "yes." or "no," variants
            let clean_token = token_trimmed.trim_matches(|c: char| !c.is_alphabetic());

            if clean_token == "yes" {
                // Take the highest (least negative) yes logprob
                if yes_logprob.is_none() || prob.logprob > yes_logprob.unwrap() {
                    yes_logprob = Some(prob.logprob);
                }
            } else if clean_token == "no" {
                // Take the highest (least negative) no logprob
                if no_logprob.is_none() || prob.logprob > no_logprob.unwrap() {
                    no_logprob = Some(prob.logprob);
                }
            }
        }

        // Also check the actual generated token
        let generated_lower = first_token.token.to_lowercase();
        let generated_trimmed = generated_lower.trim();
        let clean_generated = generated_trimmed.trim_matches(|c: char| !c.is_alphabetic());

        if clean_generated == "yes"
            && (yes_logprob.is_none() || first_token.logprob > yes_logprob.unwrap())
        {
            yes_logprob = Some(first_token.logprob);
        } else if clean_generated == "no"
            && (no_logprob.is_none() || first_token.logprob > no_logprob.unwrap())
        {
            no_logprob = Some(first_token.logprob);
        }

        // Need both yes and no logprobs to compute softmax
        let yes_lp = yes_logprob.unwrap_or(-10.0); // Default to very unlikely if not found
        let no_lp = no_logprob.unwrap_or(-10.0); // Default to very unlikely if not found

        // Compute softmax: score = exp(yes) / (exp(yes) + exp(no))
        let yes_exp = yes_lp.exp();
        let no_exp = no_lp.exp();
        let score = yes_exp / (yes_exp + no_exp);

        tracing::info!(
            "Logprobs scoring for chunk {}: yes_logprob={:.4}, no_logprob={:.4}, softmax_score={:.4}",
            &chunk_id[..8.min(chunk_id.len())],
            yes_lp,
            no_lp,
            score
        );

        Some(DetailedScore {
            score: score as f32,
            yes_logprob: Some(yes_lp),
            no_logprob: Some(no_lp),
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
                tracing::warn!(
                    "Calibration sample failed in {:.0}ms: {:?}",
                    duration_ms,
                    result.unwrap_err()
                );
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
        let result = self
            .score_candidate(query, candidate)
            .await
            .map(|d| d.score);
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
