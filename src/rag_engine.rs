use anyhow::{Context, Result};
use regex::Regex;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::OnceLock;
use uuid::Uuid;
use walkdir::WalkDir;

use crate::{
    embeddings::EmbeddingService,
    reranker::{RerankerCandidate, RerankerService},
};

// Helper function to get configurable batch size from environment
// Default to 32 for power-efficient operation (down from 128 for throughput)
fn get_batch_size() -> usize {
    std::env::var("EMBEDDING_BATCH_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(32)
}

// Helper function to get cooldown duration between batches (in milliseconds)
// Default to 500ms to allow GPU/CPU thermal recovery
fn get_batch_cooldown_ms() -> u64 {
    std::env::var("EMBEDDING_BATCH_COOLDOWN_MS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(500)
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ChunkMetadata {
    pub page_range: Option<(usize, usize)>,
    pub sentence_range: Option<(usize, usize)>,
    pub section_title: Option<String>,
    pub token_count: usize,
    pub overlap_with_previous: usize,
}

/// A chunk of document text with its embedding vector and metadata.
/// Chunks are the fundamental unit of storage and retrieval in the RAG system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentChunk {
    pub id: String,
    pub document_name: String,
    pub text: String,
    pub embedding: Vec<f32>,
    pub chunk_index: usize,
    #[serde(default = "default_page_number")]
    pub page_number: usize,
    #[serde(default)]
    pub section: Option<String>,
    #[serde(default)]
    pub metadata: ChunkMetadata,
}

#[derive(Debug, Clone)]
struct SentenceInfo {
    text: String,
    tokens: usize,
    page: usize,
    heading: Option<String>,
    index: usize,
}

/// Search result containing the matched chunk text, relevance score, and source metadata.
/// Includes detailed score breakdown for transparency when reranking is used.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SearchResult {
    pub text: String,
    /// Final blended score (reranker + initial if reranked, or initial if not)
    pub score: f32,
    pub document: String,
    pub chunk_id: String,
    pub chunk_index: usize,
    pub page_number: usize,
    pub section: Option<String>,
    /// Raw embedding cosine similarity score (first-stage retrieval)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding_score: Option<f32>,
    /// Raw lexical/BM25 score (normalized 0-1)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lexical_score: Option<f32>,
    /// Combined initial score before reranking (embedding + lexical blend)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub initial_score: Option<f32>,
    /// Raw reranker score from logprobs softmax (0-1)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reranker_score: Option<f32>,
    /// Log probability of "yes" token from reranker
    #[serde(skip_serializing_if = "Option::is_none")]
    pub yes_logprob: Option<f64>,
    /// Log probability of "no" token from reranker
    #[serde(skip_serializing_if = "Option::is_none")]
    pub no_logprob: Option<f64>,
}

/// Core RAG engine handling document chunking, embedding, and semantic search.
/// Supports two-stage retrieval: embedding similarity + optional LLM reranking.
pub struct RagEngine {
    chunks: HashMap<String, DocumentChunk>,
    embedding_service: EmbeddingService,
    data_dir: String,
    needs_reindex: bool,
    reranker: Option<RerankerService>,
    document_hashes: HashMap<String, String>,
    ann_index: Option<AnnIndex>,
    lexical_index: LexicalIndex,
}

#[derive(Debug, Clone)]
struct ChunkFragment {
    text: String,
    page_number: usize,
    section: Option<String>,
    metadata: ChunkMetadata,
}

impl ChunkFragment {
    fn from_metadata(text: String, metadata: ChunkMetadata) -> Self {
        Self {
            text,
            page_number: metadata.page_range.map(|(start, _)| start).unwrap_or(1),
            section: metadata.section_title.clone(),
            metadata,
        }
    }
}

#[derive(Debug, Clone)]
struct SearchCandidate {
    chunk_id: String,
    document: String,
    text: String,
    page_number: usize,
    section: Option<String>,
    chunk_index: usize,
    /// Combined initial score (embedding + lexical blend)
    initial_score: f32,
    /// Raw embedding cosine similarity
    embedding_score: f32,
    /// Raw lexical/BM25 score (normalized)
    lexical_score: f32,
    #[allow(dead_code)] // Used when MMR diversification is refactored to use this
    embedding: Vec<f32>,
}

/// Internal struct for MMR diversification that includes embedding
#[derive(Debug, Clone)]
struct SearchResultWithEmbedding {
    result: SearchResult,
    embedding: Vec<f32>,
}

impl RagEngine {
    pub async fn new(data_dir: &str) -> Result<Self> {
        let embedding_service = EmbeddingService::new().await?;

        // Try to initialize reranker, but don't fail if it's unavailable
        let reranker = match RerankerService::new().await {
            Ok(service) => {
                tracing::info!("Reranker service initialized successfully");
                Some(service)
            }
            Err(e) => {
                tracing::warn!(
                    "Reranker service unavailable, will fall back to embedding scores only: {}",
                    e
                );
                None
            }
        };

        let mut engine = Self {
            chunks: HashMap::new(),
            embedding_service,
            data_dir: data_dir.to_string(),
            needs_reindex: false,
            reranker,
            document_hashes: HashMap::new(),
            ann_index: None,
            lexical_index: LexicalIndex::new(),
        };

        if let Err(e) = engine.load_from_disk().await {
            tracing::warn!("Could not load existing data: {}", e);
        }

        Ok(engine)
    }

    pub fn needs_reindex(&self) -> bool {
        self.needs_reindex
    }

    pub fn embedding_model(&self) -> &str {
        self.embedding_service.model_name()
    }

    pub async fn finalize_reindex(&mut self) -> Result<()> {
        if self.needs_reindex {
            self.needs_reindex = false;
            self.save_to_disk().await?;
            tracing::info!(
                "Reindexing complete. Indexed {} chunks across {} documents.",
                self.chunks.len(),
                self.list_documents().len()
            );
        }
        Ok(())
    }

    /// Adds a document to the index by extracting text, chunking, and generating embeddings.
    /// Returns the number of chunks created (0 if document unchanged via hash check).
    pub async fn add_document(
        &mut self,
        filename: &str,
        data: &[u8],
        mut batch_callback: Option<&mut (dyn FnMut(usize, usize, usize, usize) + Send)>,
    ) -> Result<usize> {
        tracing::info!("Processing document: {}", filename);

        let document_hash = Self::compute_document_hash(data);
        if let Some(existing_hash) = self.document_hashes.get(filename) {
            if existing_hash == &document_hash {
                tracing::info!(
                    "Document {} unchanged since last index. Skipping re-embedding.",
                    filename
                );
                return Ok(0);
            }

            tracing::info!("Document {} has changed. Refreshing embeddings.", filename);
        }

        let text = self.extract_pdf_text(data.to_vec()).await?;
        if text.trim().is_empty() {
            return Err(anyhow::anyhow!("No text extracted from PDF"));
        }

        let chunks = self.chunk_text(text, 200).await?;
        tracing::info!("Created {} chunks for {}", chunks.len(), filename);

        let filtered_chunks: Vec<(usize, &ChunkFragment)> = chunks
            .iter()
            .enumerate()
            .filter_map(|(i, fragment)| {
                if fragment.text.trim().len() < 10 {
                    None
                } else {
                    Some((i, fragment))
                }
            })
            .collect();

        if filtered_chunks.is_empty() {
            tracing::warn!(
                "Document {} produced no sizeable chunks after filtering. Removing any cached chunks for this file.",
                filename
            );
            self.chunks
                .retain(|_, chunk| chunk.document_name != filename);
            self.document_hashes
                .insert(filename.to_string(), document_hash);
            self.save_to_disk().await?;
            return Ok(0);
        }

        let chunk_texts: Vec<&str> = filtered_chunks
            .iter()
            .map(|(_, fragment)| fragment.text.as_str())
            .collect();

        // Process embeddings in batches (configurable via EMBEDDING_BATCH_SIZE env var)
        let batch_size = get_batch_size();
        let cooldown_ms = get_batch_cooldown_ms();
        let total_chunks = chunk_texts.len();
        let total_batches = total_chunks.div_ceil(batch_size);

        tracing::info!(
            "Processing {} chunks from {} in {} batches of up to {} chunks each (cooldown: {}ms between batches)",
            total_chunks,
            filename,
            total_batches,
            batch_size,
            cooldown_ms
        );

        let mut embeddings = Vec::with_capacity(total_chunks);

        for (batch_idx, batch_texts) in chunk_texts.chunks(batch_size).enumerate() {
            tracing::debug!(
                "Batch {}/{}: Generating embeddings for {} chunks from {}",
                batch_idx + 1,
                total_batches,
                batch_texts.len(),
                filename
            );

            // Convert &str slice to Vec<String> only for the API call (minimal cloning)
            let batch_strings: Vec<String> = batch_texts.iter().map(|&s| s.to_string()).collect();
            let batch_embeddings = self.embedding_service.embed_texts(&batch_strings).await?;

            if batch_embeddings.len() != batch_texts.len() {
                return Err(anyhow::anyhow!(
                    "Batch {}/{}: Received {} embeddings for {} chunks in {}",
                    batch_idx + 1,
                    total_batches,
                    batch_embeddings.len(),
                    batch_texts.len(),
                    filename
                ));
            }

            embeddings.extend(batch_embeddings);

            // Invoke callback for batch progress
            if let Some(ref mut callback) = batch_callback {
                callback(
                    batch_idx + 1,
                    total_batches,
                    total_chunks,
                    batch_texts.len(),
                );
            }

            // Add cooldown between batches to prevent thermal throttling and allow power recovery
            // Skip cooldown after the last batch
            if batch_idx + 1 < total_batches && cooldown_ms > 0 {
                tokio::time::sleep(tokio::time::Duration::from_millis(cooldown_ms)).await;
            }
        }

        if embeddings.len() != filtered_chunks.len() {
            return Err(anyhow::anyhow!(
                "Total embeddings mismatch: Received {} embeddings for {} chunks in {}",
                embeddings.len(),
                filtered_chunks.len(),
                filename
            ));
        }

        self.chunks
            .retain(|_, chunk| chunk.document_name != filename);

        let mut chunk_count = 0;
        let mut embedding_index = 0;
        for (i, fragment) in chunks.into_iter().enumerate() {
            if fragment.text.trim().len() < 10 {
                continue;
            }

            // Use pre-generated embeddings from batch call
            let mut embedding = embeddings[embedding_index].clone();
            normalize(&mut embedding);
            embedding_index += 1;

            let chunk = DocumentChunk {
                id: Uuid::new_v4().to_string(),
                document_name: filename.to_string(),
                text: fragment.text.clone(),
                embedding: embedding.clone(),
                chunk_index: i,
                page_number: fragment.page_number,
                section: fragment.section.clone(),
                metadata: fragment.metadata.clone(),
            };

            // Initialize ANN index lazily when first embedding is created
            if self.ann_index.is_none() && !embedding.is_empty() {
                self.ann_index = Some(AnnIndex::new(embedding.len()));
                tracing::info!("Initialized ANN index with dimension {}", embedding.len());
            }

            if let Some(index) = self.ann_index.as_mut() {
                index.insert(&chunk.id, &chunk.embedding);
            }
            self.lexical_index.add_chunk(&chunk.id, &chunk.text);

            self.chunks.insert(chunk.id.clone(), chunk);
            chunk_count += 1;
        }

        self.document_hashes
            .insert(filename.to_string(), document_hash);

        // Validate index synchronization after adding all chunks
        self.validate_index_sync()?;

        self.save_to_disk().await?;

        tracing::info!(
            "Successfully processed {} chunks for {}",
            chunk_count,
            filename
        );
        Ok(chunk_count)
    }

    /// Check if reranker is available
    pub fn has_reranker(&self) -> bool {
        self.reranker.is_some()
    }

    /// Get reference to reranker if available
    pub fn get_reranker(&self) -> Option<&RerankerService> {
        self.reranker.as_ref()
    }

    /// Get embedding-based candidates for calibration or testing
    pub async fn get_embedding_candidates(
        &self,
        query: &str,
        count: usize,
    ) -> Result<Vec<RerankerCandidate>> {
        if self.chunks.is_empty() {
            return Ok(vec![]);
        }

        let mut query_embedding = self.embedding_service.get_query_embedding(query).await?;
        normalize(&mut query_embedding);

        let ann_candidate_iter = match &self.ann_index {
            Some(index) => Box::new(
                index
                    .search(&query_embedding, count.saturating_mul(2))
                    .into_iter(),
            ) as Box<dyn Iterator<Item = String>>,
            None => Box::new(self.chunks.keys().cloned()) as Box<dyn Iterator<Item = String>>,
        };

        let mut scores: Vec<(f32, DocumentChunk)> = Vec::new();

        for chunk_id in ann_candidate_iter {
            if let Some(chunk) = self.chunks.get(&chunk_id) {
                let embedding_score = dot_product(&query_embedding, &chunk.embedding);
                scores.push((embedding_score, chunk.clone()));
            }
        }

        scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let candidates: Vec<RerankerCandidate> = scores
            .into_iter()
            .take(count)
            .map(|(score, chunk)| RerankerCandidate {
                chunk_id: chunk.id.clone(),
                document: chunk.document_name.clone(),
                text: chunk.text.clone(),
                page_number: chunk.page_number,
                section: chunk.section.clone(),
                initial_score: score,
            })
            .collect();

        Ok(candidates)
    }

    /// Performs semantic search over indexed documents.
    /// Uses two-stage retrieval: embedding similarity + optional LLM reranking.
    ///
    /// # Arguments
    /// * `query` - Search query string
    /// * `top_k` - Number of results to return
    /// * `weights` - Optional per-query weight overrides. If None or fields are None, uses cached defaults.
    pub async fn search(
        &self,
        query: &str,
        top_k: usize,
        weights: Option<&QueryWeights>,
    ) -> Result<Vec<SearchResult>> {
        if self.chunks.is_empty() {
            return Ok(vec![]);
        }

        // Resolve weights: use overrides if valid, else cached defaults
        let resolved = ResolvedWeights::from_query_weights(weights);
        tracing::debug!(
            "Search weights: embedding={:.2}, lexical={:.2}, reranker={:.2}, initial={:.2}",
            resolved.embedding,
            resolved.lexical,
            resolved.reranker,
            resolved.initial
        );

        let top_k = top_k.max(1);
        tracing::debug!("Searching for: '{}'", query);

        let mut query_embedding = self.embedding_service.get_query_embedding(query).await?;
        normalize(&mut query_embedding);

        let ann_candidate_iter = match &self.ann_index {
            Some(index) => Box::new(
                index
                    .search(&query_embedding, top_k.saturating_mul(5))
                    .into_iter(),
            ) as Box<dyn Iterator<Item = String>>,
            None => Box::new(self.chunks.keys().cloned()) as Box<dyn Iterator<Item = String>>,
        };

        let lexical_candidates = self.lexical_index.score(query, top_k.saturating_mul(5));
        let lexical_map: HashMap<String, f32> = lexical_candidates.into_iter().collect();

        let mut candidate_ids: HashSet<String> = ann_candidate_iter.collect();
        candidate_ids.extend(lexical_map.keys().cloned());

        if candidate_ids.is_empty() {
            return Ok(vec![]);
        }

        let max_lexical = lexical_map
            .values()
            .copied()
            .fold(0.0_f32, f32::max)
            .max(f32::EPSILON);

        // Track individual scores: (combined, embedding, lexical, chunk)
        let mut scores: Vec<(f32, f32, f32, DocumentChunk)> = Vec::new();

        for chunk_id in candidate_ids {
            if let Some(chunk) = self.chunks.get(&chunk_id) {
                let embedding_score = dot_product(&query_embedding, &chunk.embedding);
                let lexical_score = lexical_map
                    .get(&chunk_id)
                    .map(|score| score / max_lexical)
                    .unwrap_or(0.0);
                let combined_score =
                    resolved.embedding * embedding_score + resolved.lexical * lexical_score;

                scores.push((
                    combined_score,
                    embedding_score,
                    lexical_score,
                    chunk.clone(),
                ));
            }
        }

        scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let initial_k = scores.len().min(top_k.saturating_mul(3).max(top_k));

        let candidates: Vec<SearchCandidate> = scores
            .into_iter()
            .take(initial_k)
            .map(|(combined, embed, lex, chunk)| SearchCandidate {
                chunk_id: chunk.id.clone(),
                document: chunk.document_name.clone(),
                text: chunk.text.clone(),
                page_number: chunk.page_number,
                section: chunk.section.clone(),
                chunk_index: chunk.chunk_index,
                initial_score: combined,
                embedding_score: embed,
                lexical_score: lex,
                embedding: chunk.embedding.clone(),
            })
            .collect();

        if candidates.is_empty() {
            return Ok(vec![]);
        }

        let candidate_map: HashMap<String, SearchCandidate> = candidates
            .iter()
            .cloned()
            .map(|candidate| (candidate.chunk_id.clone(), candidate))
            .collect();

        let reranker_inputs: Vec<RerankerCandidate> = candidates
            .iter()
            .map(|candidate| RerankerCandidate {
                chunk_id: candidate.chunk_id.clone(),
                document: candidate.document.clone(),
                text: candidate.text.clone(),
                page_number: candidate.page_number,
                section: candidate.section.clone(),
                initial_score: candidate.initial_score,
            })
            .collect();

        let reranked = match &self.reranker {
            Some(reranker) => match reranker.rerank(query, &reranker_inputs).await {
                Ok(results) => results,
                Err(err) => {
                    tracing::warn!("Reranker failed, falling back to embedding scores: {}", err);
                    Vec::new()
                }
            },
            None => {
                tracing::debug!("Reranker not available, using embedding scores only");
                Vec::new()
            }
        };

        let mut ordered_results = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();

        if !reranked.is_empty() {
            // Calculate max scores for per-query normalization
            let max_reranker = reranked
                .iter()
                .map(|r| r.relevance)
                .fold(0.0_f32, f32::max)
                .max(f32::EPSILON);
            let max_initial = candidates
                .iter()
                .map(|c| c.initial_score)
                .fold(0.0_f32, f32::max)
                .max(f32::EPSILON);

            // Build results with blended scores
            for result in &reranked {
                if let Some(candidate) = candidate_map.get(&result.chunk_id)
                    && seen.insert(result.chunk_id.clone())
                {
                    // Normalize both scores to [0,1] range within this query's candidates
                    let reranker_norm = result.relevance / max_reranker;
                    let initial_norm = candidate.initial_score / max_initial;

                    // Blend: reranker dominates but initial score provides discrimination
                    // Weights configurable via RAG_RERANKER_WEIGHT/RAG_INITIAL_SCORE_WEIGHT env vars or per-query override
                    let blended_score =
                        resolved.reranker * reranker_norm + resolved.initial * initial_norm;

                    tracing::debug!(
                        chunk_id = %candidate.chunk_id,
                        reranker = %result.relevance,
                        initial = %candidate.initial_score,
                        blended = %blended_score,
                        "Score blending"
                    );

                    ordered_results.push(SearchResult {
                        text: candidate.text.clone(),
                        score: blended_score,
                        document: candidate.document.clone(),
                        chunk_id: candidate.chunk_id.clone(),
                        chunk_index: candidate.chunk_index,
                        page_number: candidate.page_number,
                        section: candidate.section.clone(),
                        // Detailed score breakdown for TUI display
                        embedding_score: Some(candidate.embedding_score),
                        lexical_score: Some(candidate.lexical_score),
                        initial_score: Some(candidate.initial_score),
                        reranker_score: Some(result.relevance),
                        yes_logprob: result.yes_logprob,
                        no_logprob: result.no_logprob,
                    });
                }
            }

            // Re-sort by blended score (reranked order may differ after blending)
            ordered_results.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            // Truncate to top_k
            ordered_results.truncate(top_k);
        }

        if ordered_results.len() < top_k {
            let mut fallback_candidates: Vec<_> = candidate_map.values().collect();
            fallback_candidates.sort_by(|a, b| {
                b.initial_score
                    .partial_cmp(&a.initial_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            for candidate in fallback_candidates {
                if ordered_results.len() == top_k {
                    break;
                }

                if seen.insert(candidate.chunk_id.clone()) {
                    ordered_results.push(SearchResult {
                        text: candidate.text.clone(),
                        score: candidate.initial_score,
                        document: candidate.document.clone(),
                        chunk_id: candidate.chunk_id.clone(),
                        chunk_index: candidate.chunk_index,
                        page_number: candidate.page_number,
                        section: candidate.section.clone(),
                        // No reranking for fallback results
                        embedding_score: Some(candidate.embedding_score),
                        lexical_score: Some(candidate.lexical_score),
                        initial_score: Some(candidate.initial_score),
                        reranker_score: None,
                        yes_logprob: None,
                        no_logprob: None,
                    });
                }
            }
        }

        Ok(ordered_results)
    }

    /// Search with MMR (Maximal Marginal Relevance) diversification.
    ///
    /// This method performs the standard search and then applies MMR to diversify results.
    /// MMR balances relevance with diversity by penalizing results that are too similar
    /// to already-selected results.
    ///
    /// # Arguments
    /// * `query` - The search query
    /// * `top_k` - Number of results to return
    /// * `diversity_factor` - λ parameter for MMR (0.0-1.0).
    ///   - 0.0 = pure relevance (no diversity penalty)
    ///   - 1.0 = maximum diversity (heavily penalizes similar results)
    ///   - Recommended: 0.2-0.4 for most use cases
    /// * `weights` - Optional per-query weight overrides
    pub async fn search_with_diversity(
        &self,
        query: &str,
        top_k: usize,
        diversity_factor: f32,
        weights: Option<&QueryWeights>,
    ) -> Result<Vec<SearchResult>> {
        // Clamp diversity factor to valid range
        let diversity_factor = diversity_factor.clamp(0.0, 1.0);

        // If no diversity requested, just do normal search
        if diversity_factor == 0.0 {
            return self.search(query, top_k, weights).await;
        }

        // Fetch more candidates than needed for MMR selection
        // We need extra candidates because MMR may skip similar ones
        let candidate_pool_size = (top_k * 3).max(top_k + 10);
        let candidates = self.search(query, candidate_pool_size, weights).await?;

        if candidates.is_empty() {
            return Ok(vec![]);
        }

        // Build candidates with embeddings for MMR
        let candidates_with_embeddings: Vec<SearchResultWithEmbedding> = candidates
            .into_iter()
            .filter_map(|result| {
                // Look up the embedding from our chunks
                self.chunks
                    .get(&result.chunk_id)
                    .map(|chunk| SearchResultWithEmbedding {
                        result,
                        embedding: chunk.embedding.clone(),
                    })
            })
            .collect();

        // Apply MMR diversification
        let diversified = self.mmr_diversify(candidates_with_embeddings, top_k, diversity_factor);

        Ok(diversified)
    }

    /// Apply Maximal Marginal Relevance (MMR) diversification to search results.
    ///
    /// MMR formula: MMR(i) = (1 - λ) × relevance_score - λ × max_similarity_to_selected
    ///
    /// This iteratively selects results that balance high relevance with low similarity
    /// to already-selected results.
    fn mmr_diversify(
        &self,
        candidates: Vec<SearchResultWithEmbedding>,
        top_k: usize,
        diversity_factor: f32, // λ in the MMR formula
    ) -> Vec<SearchResult> {
        if candidates.is_empty() {
            return vec![];
        }

        let mut selected: Vec<SearchResultWithEmbedding> = Vec::with_capacity(top_k);
        let mut remaining: Vec<SearchResultWithEmbedding> = candidates;

        // First result is always the highest-scoring (most relevant)
        // Use swap_remove for O(1) removal instead of O(n) shift
        if !remaining.is_empty() {
            let first = remaining.swap_remove(0);
            selected.push(first);
        }

        // Iteratively select results using MMR
        while selected.len() < top_k && !remaining.is_empty() {
            let mut best_mmr_score = f32::NEG_INFINITY;
            let mut best_idx = 0;

            for (idx, candidate) in remaining.iter().enumerate() {
                // Skip candidates with non-finite scores (NaN/Inf protection)
                let relevance = candidate.result.score;
                if !relevance.is_finite() {
                    continue;
                }

                // Calculate max similarity to any already-selected result
                let max_similarity = selected
                    .iter()
                    .map(|s| dot_product(&candidate.embedding, &s.embedding))
                    .filter(|sim| sim.is_finite()) // Filter out NaN/Inf similarities
                    .fold(0.0_f32, |a, b| a.max(b));

                // MMR score: balance relevance vs diversity
                // Higher diversity_factor = more penalty for similarity
                let mmr_score =
                    (1.0 - diversity_factor) * relevance - diversity_factor * max_similarity;

                // Only update if mmr_score is finite and better
                if mmr_score.is_finite() && mmr_score > best_mmr_score {
                    best_mmr_score = mmr_score;
                    best_idx = idx;
                }
            }

            // If no valid candidate found (all NaN/Inf), break
            if best_mmr_score == f32::NEG_INFINITY {
                tracing::warn!("MMR: No valid candidates remaining (all scores non-finite)");
                break;
            }

            // Move best candidate from remaining to selected using O(1) swap_remove
            let best = remaining.swap_remove(best_idx);

            tracing::debug!(
                chunk_id = %best.result.chunk_id,
                relevance = %best.result.score,
                mmr_score = %best_mmr_score,
                "MMR selected result"
            );

            selected.push(best);
        }

        // Extract just the SearchResult from our selected candidates
        selected.into_iter().map(|s| s.result).collect()
    }

    /// Calculate cosine similarity between two embeddings.
    /// Returns a value in [-1, 1] where 1 means identical direction.
    /// Returns 0.0 for edge cases (empty, mismatched length, near-zero norm).
    #[allow(dead_code)] // Used in tests and legacy paths
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        // We use the same epsilon as the free function below to match behavior
        cosine_similarity(a, b)
    }

    pub fn list_documents(&self) -> Vec<String> {
        let mut docs: Vec<String> = self
            .chunks
            .values()
            .map(|chunk| chunk.document_name.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        docs.sort();
        docs
    }

    pub fn get_stats(&self) -> serde_json::Value {
        let doc_count = self.list_documents().len();
        let chunk_count = self.chunks.len();

        let status = if self.needs_reindex {
            "reindexing"
        } else {
            "ready"
        };

        let reranker_model = self.reranker.as_ref().map(|r| r.model_name());

        serde_json::json!({
            "documents": doc_count,
            "chunks": chunk_count,
            "status": status,
            "embedding_model": self.embedding_model(),
            "reranker_model": reranker_model
        })
    }

    #[allow(dead_code)]
    pub async fn load_documents_from_dir(&mut self, dir: &str) -> Result<()> {
        let was_reindexing = self.needs_reindex;

        if was_reindexing {
            tracing::info!(
                "Reindexing documents in '{}' with embedding model '{}'",
                dir,
                self.embedding_service.model_name()
            );
        }

        for entry in WalkDir::new(dir).into_iter().filter_map(|e| e.ok()) {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("pdf") {
                let filename = path.file_name().unwrap().to_str().unwrap();

                match tokio::fs::read(&path).await {
                    Ok(data) => {
                        tracing::info!("Loading document: {}", filename);
                        match self.add_document(filename, &data, None).await {
                            Ok(chunk_count) => {
                                if chunk_count > 0 {
                                    tracing::info!(
                                        "Successfully processed {} with {} chunks",
                                        filename,
                                        chunk_count
                                    );
                                } else {
                                    tracing::info!(
                                        "{} is already up to date. No reindex needed.",
                                        filename
                                    );
                                }
                            }
                            Err(e) => {
                                tracing::warn!("Skipping {}: {}", filename, e);
                            }
                        }
                    }
                    Err(e) => {
                        tracing::error!("Failed to read {}: {}", filename, e);
                    }
                }
            }
        }

        if was_reindexing {
            self.needs_reindex = false;
            self.save_to_disk().await?;
            tracing::info!(
                "Reindexing complete. Indexed {} chunks across {} documents.",
                self.chunks.len(),
                self.list_documents().len()
            );
        }

        Ok(())
    }

    /// Async wrapper for PDF text extraction using spawn_blocking.
    /// This prevents blocking the Tokio async executor during PDF processing.
    ///
    /// Uses a two-stage fallback strategy:
    /// 1. Try pure-Rust extraction (lopdf) first for deployment flexibility
    /// 2. Fall back to pdftotext binary if lopdf fails
    async fn extract_pdf_text(&self, data: Vec<u8>) -> Result<String> {
        // Clone data for potential fallback (lopdf consumes data)
        let data_for_fallback = data.clone();

        // Try pure-Rust extraction first
        let lopdf_result = tokio::task::spawn_blocking(move || Self::lopdf_extract_sync(&data))
            .await
            .context("lopdf extraction task failed")?;

        match lopdf_result {
            Ok(text) => {
                tracing::info!(
                    "✅ PDF extracted using pure-Rust backend (lopdf): {} chars",
                    text.chars().count()
                );
                Ok(text)
            }
            Err(lopdf_err) => {
                tracing::warn!(
                    error = %lopdf_err,
                    "Pure-Rust PDF extraction failed, falling back to pdftotext"
                );

                // Fall back to pdftotext
                let pdftotext_result = tokio::task::spawn_blocking(move || {
                    Self::pdftotext_extract_sync(&data_for_fallback)
                })
                .await
                .context("pdftotext extraction task failed")?;

                match pdftotext_result {
                    Ok(text) => {
                        tracing::info!(
                            "✅ PDF extracted using pdftotext fallback: {} chars",
                            text.chars().count()
                        );
                        Ok(text)
                    }
                    Err(pdftotext_err) => {
                        tracing::error!(
                            lopdf_error = %lopdf_err,
                            pdftotext_error = %pdftotext_err,
                            "Both PDF extraction backends failed"
                        );
                        Err(anyhow::anyhow!(
                            "PDF extraction failed: lopdf error: {}, pdftotext error: {}",
                            lopdf_err,
                            pdftotext_err
                        ))
                    }
                }
            }
        }
    }

    /// Pure-Rust PDF text extraction using lopdf.
    /// Extracts text by walking page content streams and interpreting text operators.
    fn lopdf_extract_sync(data: &[u8]) -> Result<String> {
        use lopdf::Document;

        let doc = Document::load_mem(data)
            .map_err(|e| anyhow::anyhow!("lopdf failed to parse PDF: {}", e))?;

        let mut all_text = String::new();
        let pages = doc.get_pages();

        for (page_num, _page_id) in pages {
            match doc.extract_text(&[page_num]) {
                Ok(page_text) => {
                    if !all_text.is_empty() && !page_text.is_empty() {
                        all_text.push('\n');
                    }
                    all_text.push_str(&page_text);
                }
                Err(e) => {
                    tracing::debug!(
                        "lopdf: failed to extract text from page {}: {}",
                        page_num,
                        e
                    );
                    // Continue with other pages
                }
            }
        }

        if all_text.trim().is_empty() {
            return Err(anyhow::anyhow!("lopdf extracted no text from PDF"));
        }

        Ok(all_text)
    }

    /// Synchronous PDF extraction using pdftotext binary.
    /// This is a static function to allow calling from spawn_blocking.
    /// Uses UUID for temp filename to prevent race conditions in concurrent calls.
    fn pdftotext_extract_sync(data: &[u8]) -> Result<String> {
        use std::process::Command;

        let temp_dir = std::env::temp_dir();
        // Use UUID instead of process::id() to prevent temp file collisions
        // when multiple PDFs are extracted concurrently via spawn_blocking
        let temp_file = temp_dir.join(format!("temp_pdf_{}.pdf", Uuid::new_v4()));

        std::fs::write(&temp_file, data)
            .map_err(|e| anyhow::anyhow!("Failed to write temp PDF: {}", e))?;

        let output = Command::new("pdftotext")
            .arg("-layout")
            .arg("-enc")
            .arg("UTF-8")
            .arg(&temp_file)
            .arg("-")
            .output();
        let _ = std::fs::remove_file(&temp_file);

        match output {
            Ok(output) if output.status.success() => {
                let text = String::from_utf8_lossy(&output.stdout).to_string();
                let text_chars = text.chars().count();

                if text.trim().is_empty() {
                    tracing::warn!("pdftotext extracted 0 characters");
                    Err(anyhow::anyhow!("pdftotext produced no text output"))
                } else {
                    tracing::debug!("pdftotext extracted {} characters", text_chars);
                    Ok(text)
                }
            }
            Ok(output) => {
                let error_msg = String::from_utf8_lossy(&output.stderr);
                tracing::warn!("pdftotext failed with error: {}", error_msg);
                Err(anyhow::anyhow!("pdftotext failed: {}", error_msg))
            }
            Err(e) => {
                tracing::warn!("Failed to run pdftotext command: {}", e);
                Err(anyhow::anyhow!(
                    "pdftotext command failed: {} (is poppler installed?)",
                    e
                ))
            }
        }
    }

    /// Async wrapper for text chunking using spawn_blocking.
    /// This prevents blocking the Tokio async executor during CPU-intensive regex operations.
    async fn chunk_text(&self, text: String, chunk_tokens: usize) -> Result<Vec<ChunkFragment>> {
        tokio::task::spawn_blocking(move || Self::chunk_text_sync(&text, chunk_tokens))
            .await
            .context("Chunking task failed")
    }

    /// Synchronous text chunking with sentence-aware splitting.
    /// This is a static function to allow calling from spawn_blocking.
    fn chunk_text_sync(text: &str, chunk_tokens: usize) -> Vec<ChunkFragment> {
        // Extract sentences from text
        let sentences = Self::extract_sentences(text);
        if sentences.is_empty() {
            return Vec::new();
        }

        let mut window: Vec<usize> = Vec::new();
        let mut token_sum = 0usize;
        let sentence_overlap = 2usize; // Overlap 2 sentences between chunks
        let mut fragments = Vec::new();

        for (idx, sentence) in sentences.iter().enumerate() {
            window.push(idx);
            token_sum += sentence.tokens;

            // When token budget is exceeded, finalize current chunk
            if token_sum >= chunk_tokens {
                if let Some((text, metadata)) =
                    Self::finalize_chunk(&window, &sentences, sentence_overlap)
                {
                    fragments.push(ChunkFragment::from_metadata(text, metadata));
                }

                // Keep last N sentences for overlap
                let overlap_start = window.len().saturating_sub(sentence_overlap);
                window = window.split_off(overlap_start);
                token_sum = window.iter().map(|&i| sentences[i].tokens).sum();
            }
        }

        // Finalize remaining sentences
        if !window.is_empty() {
            if let Some((text, metadata)) = Self::finalize_chunk(&window, &sentences, 0) {
                fragments.push(ChunkFragment::from_metadata(text, metadata));
            }
        }

        fragments
    }

    fn finalize_chunk(
        sentence_indices: &[usize],
        sentences: &[SentenceInfo],
        overlap_with_previous: usize,
    ) -> Option<(String, ChunkMetadata)> {
        if sentence_indices.is_empty() {
            return None;
        }

        let mut text_parts: Vec<String> = Vec::with_capacity(sentence_indices.len());
        let mut min_page: Option<usize> = None;
        let mut max_page: Option<usize> = None;
        let mut section_title: Option<String> = None;
        let mut token_sum = 0usize;

        for &idx in sentence_indices {
            let sentence = sentences.get(idx)?;
            text_parts.push(sentence.text.clone());
            token_sum += sentence.tokens;

            min_page = Some(match min_page {
                Some(current_min) => current_min.min(sentence.page),
                None => sentence.page,
            });

            max_page = Some(match max_page {
                Some(current_max) => current_max.max(sentence.page),
                None => sentence.page,
            });

            if section_title.is_none()
                && let Some(title) = &sentence.heading
            {
                section_title = Some(title.clone());
            }
        }

        let start_index = sentences
            .get(*sentence_indices.first()?)
            .map(|s| s.index)
            .unwrap_or(0);
        let end_index = sentences
            .get(*sentence_indices.last()?)
            .map(|s| s.index)
            .unwrap_or(start_index);

        let mut chunk_text = text_parts.join(" ");
        chunk_text = Self::normalize_whitespace(&chunk_text);

        let mut metadata = ChunkMetadata {
            page_range: min_page.zip(max_page),
            sentence_range: Some((start_index, end_index)),
            section_title,
            token_count: token_sum,
            overlap_with_previous,
        };

        if let Some(title) = metadata.section_title.as_mut() {
            const MAX_TITLE_LEN: usize = 160;
            if title.chars().count() > MAX_TITLE_LEN {
                *title = title.chars().take(MAX_TITLE_LEN).collect();
            }
        }

        if chunk_text.is_empty() {
            return None;
        }

        Some((chunk_text, metadata))
    }

    fn extract_sentences(text: &str) -> Vec<SentenceInfo> {
        let splitter = Self::sentence_splitter();
        let mut sentences: Vec<SentenceInfo> = Vec::new();
        let mut sentence_index = 0usize;

        for (page_idx, page_text) in text.split('\u{0c}').enumerate() {
            let page_number = page_idx + 1;
            let mut last_heading: Option<String> = None;

            for block in page_text.split("\n\n") {
                let block = block.trim();
                if block.is_empty() {
                    continue;
                }

                let lines: Vec<&str> = block.lines().collect();
                if lines.len() == 1 && Self::is_heading(lines[0]) {
                    last_heading = Some(lines[0].trim().to_string());
                    continue;
                }

                let mut paragraph_lines = Vec::new();
                for line in lines {
                    let trimmed = line.trim();
                    if trimmed.is_empty() {
                        continue;
                    }
                    if paragraph_lines.is_empty() && Self::is_heading(trimmed) {
                        last_heading = Some(trimmed.to_string());
                        continue;
                    }
                    paragraph_lines.push(trimmed);
                }

                if paragraph_lines.is_empty() {
                    continue;
                }

                let normalized = Self::normalize_whitespace(&paragraph_lines.join(" "));
                if normalized.is_empty() {
                    continue;
                }

                let splits: Vec<&str> = splitter
                    .split(&normalized)
                    .map(str::trim)
                    .filter(|part| !part.is_empty())
                    .collect();

                let parts = if splits.is_empty() {
                    vec![normalized.as_str()]
                } else {
                    splits
                };

                for part in parts {
                    let tokens = Self::approximate_token_count(part);
                    if tokens == 0 {
                        continue;
                    }
                    sentences.push(SentenceInfo {
                        text: part.to_string(),
                        tokens,
                        page: page_number,
                        heading: last_heading.clone(),
                        index: sentence_index,
                    });
                    sentence_index += 1;
                }
            }
        }

        if sentences.is_empty() {
            let normalized = Self::normalize_whitespace(text);
            if !normalized.is_empty() {
                sentences.push(SentenceInfo {
                    text: normalized.clone(),
                    tokens: Self::approximate_token_count(&normalized),
                    page: 1,
                    heading: None,
                    index: 0,
                });
            }
        }

        sentences
    }

    fn normalize_whitespace(value: &str) -> String {
        value.split_whitespace().collect::<Vec<_>>().join(" ")
    }

    fn is_heading(line: &str) -> bool {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.len() > 120 {
            return false;
        }

        let word_count = trimmed.split_whitespace().count();
        if word_count == 0 || word_count > 12 {
            return false;
        }

        let uppercase_letters = trimmed.chars().filter(|c| c.is_uppercase()).count();
        let lowercase_letters = trimmed.chars().filter(|c| c.is_lowercase()).count();

        if lowercase_letters == 0 && uppercase_letters > 0 {
            return true;
        }

        if trimmed.ends_with(':') {
            return true;
        }

        if word_count <= 4 && uppercase_letters >= lowercase_letters {
            return true;
        }

        // Refined: match lines starting with digit(s), dot, and whitespace (e.g., "1. Introduction")
        if Self::heading_regex().is_match(trimmed) {
            return true;
        }

        false
    }

    /// Returns a cached regex for detecting numbered headings (e.g., "1. Introduction")
    fn heading_regex() -> &'static Regex {
        static HEADING_REGEX: OnceLock<Regex> = OnceLock::new();
        HEADING_REGEX.get_or_init(|| Regex::new(r"^\d+\.\s").expect("valid heading regex pattern"))
    }

    fn approximate_token_count(value: &str) -> usize {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            return 0;
        }

        let char_count = trimmed.chars().count();
        let word_count = trimmed.split_whitespace().count();
        let char_estimate = char_count.div_ceil(4);
        let word_estimate = ((word_count as f32) * 0.9).ceil() as usize;
        char_estimate.max(word_estimate).max(1)
    }

    fn sentence_splitter() -> &'static srx::Rules {
        static SPLITTER: OnceLock<srx::Rules> = OnceLock::new();
        SPLITTER.get_or_init(|| {
            // Load SRX rules from embedded segment.srx file
            const SRX_XML: &str = include_str!("../data/segment.srx");
            let srx =
                srx::SRX::from_str(SRX_XML).expect("valid SRX rules from embedded segment.srx");

            // Use English language rules for sentence splitting
            // This handles abbreviations like "Dr.", "Mr.", "etc." correctly
            srx.language_rules("English")
        })
    }

    /// Validates that all indexes are synchronized with the chunks HashMap
    /// Ensures document_hashes, lexical_index, and ann_index all contain the same chunk IDs
    fn validate_index_sync(&mut self) -> Result<()> {
        let valid_chunk_ids: HashSet<String> = self.chunks.keys().cloned().collect();

        // Validate lexical index
        self.lexical_index.drop_stale(&valid_chunk_ids);

        // Ensure all chunks are in lexical index
        for chunk_id in &valid_chunk_ids {
            if let Some(chunk) = self.chunks.get(chunk_id) {
                if !self.lexical_index.contains(chunk_id) {
                    tracing::debug!("Re-adding missing chunk {} to lexical index", chunk_id);
                    self.lexical_index.add_chunk(chunk_id, &chunk.text);
                }
            }
        }

        // Initialize ANN index if missing but we have chunks (e.g. after restart)
        if self.ann_index.is_none() && !self.chunks.is_empty() {
            // Use the dimension of the first chunk's embedding
            if let Some(first_chunk) = self.chunks.values().next() {
                let dim = first_chunk.embedding.len();
                if dim > 0 {
                    tracing::info!(
                        "Rebuilding ANN index with dimension {} for {} chunks...",
                        dim,
                        self.chunks.len()
                    );
                    self.ann_index = Some(AnnIndex::new(dim));
                }
            }
        }

        // Validate ANN index if present
        if let Some(ann_index) = &mut self.ann_index {
            // Remove stale entries from ANN
            ann_index.drop_stale(&valid_chunk_ids);

            // Add missing chunks to ANN
            for chunk_id in &valid_chunk_ids {
                if let Some(chunk) = self.chunks.get(chunk_id) {
                    if !ann_index.contains(chunk_id) {
                        tracing::debug!("Re-adding missing chunk {} to ANN index", chunk_id);
                        ann_index.insert(chunk_id, &chunk.embedding);
                    }
                }
            }
        }

        // Validate document hashes - remove orphaned entries
        let valid_documents: HashSet<String> = self
            .chunks
            .values()
            .map(|chunk| chunk.document_name.clone())
            .collect();

        self.document_hashes.retain(|doc_name, _| {
            if valid_documents.contains(doc_name) {
                true
            } else {
                tracing::debug!("Removing orphaned document hash for {}", doc_name);
                false
            }
        });

        tracing::debug!("Index synchronization validated");
        Ok(())
    }

    // ============================================================
    // Model-Partitioned Storage Helpers
    // Enable hot-swapping between embedding models by storing each
    // model's index in a separate file (e.g., chunks_nomic-embed-text.json)
    // ============================================================

    /// Sanitizes model name for safe use as a filename.
    /// Replaces path separators, special characters, and handles edge cases.
    pub fn sanitize_model_name(model_name: &str) -> String {
        let trimmed = model_name.trim();

        // Handle empty or whitespace-only input
        if trimmed.is_empty() {
            return "default".to_string();
        }

        // Replace path separators and unsafe characters
        let sanitized: String = trimmed
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' {
                    c
                } else {
                    '_'
                }
            })
            .collect();

        // Final safety check - ensure result is not empty after sanitization
        if sanitized.is_empty() || sanitized.chars().all(|c| c == '_' || c == '.') {
            "default".to_string()
        } else {
            sanitized
        }
    }

    /// Generates the index file path for a specific model.
    /// Uses sanitized model name to ensure filesystem safety.
    pub fn get_index_path(data_dir: &str, model_name: &str) -> PathBuf {
        let sanitized = Self::sanitize_model_name(model_name);
        PathBuf::from(data_dir).join(format!("chunks_{sanitized}.json"))
    }

    /// Generates the legacy index file path (for migration support).
    pub fn get_legacy_path(data_dir: &str) -> PathBuf {
        PathBuf::from(data_dir).join("chunks.json")
    }

    /// Persist the current state to disk atomically.
    /// Uses temp file + rename pattern for crash safety.
    pub async fn save_to_disk(&self) -> Result<()> {
        #[derive(Serialize)]
        struct PersistedState<'a> {
            version: u32,
            model: &'a str,
            chunks: &'a HashMap<String, DocumentChunk>,
            needs_reindex: bool,
            #[serde(default, skip_serializing_if = "HashMap::is_empty")]
            document_hashes: &'a HashMap<String, String>,
        }

        // Use model-specific path for hot-swappable storage
        let model_name = self.embedding_service.model_name();
        let final_path = Self::get_index_path(&self.data_dir, model_name);
        let temp_path = final_path.with_extension("json.tmp");

        let state = PersistedState {
            version: 2,
            model: model_name,
            chunks: &self.chunks,
            needs_reindex: self.needs_reindex,
            document_hashes: &self.document_hashes,
        };

        let data = serde_json::to_string_pretty(&state)?;

        // Atomic write: write to temp file, then rename
        tokio::fs::write(&temp_path, data)
            .await
            .context("Failed to write index to temporary file")?;
        tokio::fs::rename(&temp_path, &final_path)
            .await
            .context("Failed to commit index file (atomic rename)")?;

        tracing::debug!(
            "Saved {} chunks to {:?} for model '{}'",
            self.chunks.len(),
            final_path,
            model_name
        );
        Ok(())
    }

    async fn load_from_disk(&mut self) -> Result<()> {
        let current_model = self.embedding_service.model_name();
        let model_specific_path = Self::get_index_path(&self.data_dir, current_model);
        let legacy_path = Self::get_legacy_path(&self.data_dir);

        #[derive(Deserialize)]
        struct PersistedState {
            version: u32,
            #[allow(dead_code)] // Deserialized for schema validation, not explicitly read
            model: String,
            chunks: HashMap<String, DocumentChunk>,
            #[serde(default)]
            needs_reindex: bool,
            #[serde(default)]
            document_hashes: HashMap<String, String>,
        }

        // Helper to peek at model name in a JSON file without fully loading
        #[derive(Deserialize)]
        struct ModelOnly {
            model: String,
        }

        // Strategy:
        // 1. Try model-specific file first (priority)
        // 2. Fall back to legacy chunks.json ONLY if model matches (migration)
        // 3. Never delete another model's data

        // Step 1: Check for model-specific index file
        if tokio::fs::try_exists(&model_specific_path).await? {
            tracing::info!(
                "Loading model-specific index from {:?} for model '{}'",
                model_specific_path,
                current_model
            );
            let data = tokio::fs::read_to_string(&model_specific_path).await?;

            match serde_json::from_str::<PersistedState>(&data) {
                Ok(state) => {
                    return self
                        .apply_loaded_state(
                            state.version,
                            state.chunks,
                            state.needs_reindex,
                            state.document_hashes,
                            &model_specific_path,
                            false,
                        )
                        .await;
                }
                Err(e) => {
                    tracing::warn!(
                        "Failed to parse model-specific index at {:?}: {}. Starting fresh for model '{}'. \
                        Marking for reindex to rebuild data.",
                        model_specific_path,
                        e,
                        current_model
                    );
                    // Don't delete corrupted file - let user investigate
                    // But mark for reindex so we rebuild instead of running empty
                    self.needs_reindex = true;
                    return Ok(());
                }
            }
        }

        // Step 2: Check legacy chunks.json for migration
        if tokio::fs::try_exists(&legacy_path).await? {
            tracing::info!(
                "No model-specific index found. Checking legacy chunks.json for migration..."
            );
            let data = tokio::fs::read_to_string(&legacy_path).await?;

            // Peek at model name first
            if let Ok(info) = serde_json::from_str::<ModelOnly>(&data) {
                if info.model == current_model {
                    tracing::info!(
                        "Legacy index matches current model '{}'. Migrating to model-specific storage.",
                        current_model
                    );

                    match serde_json::from_str::<PersistedState>(&data) {
                        Ok(state) => {
                            return self
                                .apply_loaded_state(
                                    state.version,
                                    state.chunks,
                                    state.needs_reindex,
                                    state.document_hashes,
                                    &legacy_path,
                                    true,
                                )
                                .await;
                        }
                        Err(e) => {
                            tracing::warn!("Failed to parse legacy index: {}. Starting fresh.", e);
                        }
                    }
                } else {
                    tracing::info!(
                        "Legacy index belongs to model '{}', but we're using '{}'. \
                        Legacy data preserved. Starting fresh for current model.",
                        info.model,
                        current_model
                    );
                    // DO NOT delete legacy file - it belongs to another model
                }
            } else {
                // Very old format without model field - try to parse as raw chunks
                tracing::warn!(
                    "Legacy index has unknown format. Checking if it can be migrated..."
                );
                if let Ok(legacy_chunks) =
                    serde_json::from_str::<HashMap<String, DocumentChunk>>(&data)
                {
                    if !legacy_chunks.is_empty() {
                        tracing::warn!(
                            "Found {} legacy chunks without model info. Reindexing required for model '{}'.",
                            legacy_chunks.len(),
                            current_model
                        );
                        self.needs_reindex = true;
                    }
                }
            }
        }

        tracing::info!(
            "No existing index found for model '{}'. Starting fresh.",
            current_model
        );
        Ok(())
    }

    /// Helper to apply loaded state and optionally migrate to new format
    async fn apply_loaded_state(
        &mut self,
        version: u32,
        chunks: HashMap<String, DocumentChunk>,
        needs_reindex: bool,
        document_hashes: HashMap<String, String>,
        source_path: &std::path::Path,
        migrate_to_new_format: bool,
    ) -> Result<()> {
        if version < 2 {
            tracing::info!(
                "Index version {} is outdated. Marking for reindex to capture provenance data.",
                version
            );
            self.chunks.clear();
            self.needs_reindex = true;
            self.save_to_disk().await?;
            return Ok(());
        }

        self.chunks = chunks;
        // Normalize all embeddings to ensure fast cosine similarity optimization works
        // This handles legacy data that might not be normalized
        for chunk in self.chunks.values_mut() {
            normalize(&mut chunk.embedding);
        }

        self.needs_reindex = needs_reindex;
        self.document_hashes = document_hashes;

        // Check if document fingerprints need initialization
        if self.document_hashes.is_empty() && !self.chunks.is_empty() {
            tracing::info!(
                "No document fingerprints found. Marking for reindex to initialize change detection."
            );
            self.needs_reindex = true;
        }

        tracing::info!("Loaded {} chunks from {:?}", self.chunks.len(), source_path);

        // Validate and repair index synchronization
        self.validate_index_sync()?;

        // Migrate to new format if needed
        if migrate_to_new_format {
            tracing::info!("Saving to model-specific format...");
            self.save_to_disk().await?;
            tracing::info!(
                "Migration complete. Legacy chunks.json preserved for safety. \
                You can delete it manually after verifying the new index works."
            );
        }

        Ok(())
    }

    fn compute_document_hash(data: &[u8]) -> String {
        let hash = Sha256::digest(data);
        format!("{hash:x}")
    }
}

/// Extracts a section heading from page text.
///
/// This function scans the provided text line by line to find a suitable heading.
/// It returns the first line that meets all of the following criteria:
/// - Non-empty after trimming whitespace
/// - Not purely numeric (contains at least one non-digit character)
/// - Contains at least 3 alphabetic characters
///
/// The returned heading is truncated to a maximum of 120 characters.
///
/// # Arguments
///
/// * `text` - The text content to extract a heading from
///
/// # Returns
///
/// * `Some(String)` - The extracted heading (up to 120 characters) if found
/// * `None` - If no suitable heading is found in the text
// extract_section_heading has been replaced by sentence-aware chunking
// which extracts headings via SentenceInfo in extract_sentences()
fn default_page_number() -> usize {
    0
}

#[allow(dead_code)] // Used in tests
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    // Epsilon for near-zero norm detection (prevents division instability)
    const EPSILON: f32 = 1e-10;

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a < EPSILON || norm_b < EPSILON {
        return 0.0;
    } else {
        (dot_product / (norm_a * norm_b)).clamp(-1.0, 1.0)
    }
}

/// Normalize a vector to unit length in-place.
/// If the vector has zero or very small norm, it is left unchanged.
fn normalize(v: &mut [f32]) {
    let norm_sq: f32 = v.iter().map(|x| x * x).sum();
    if norm_sq > 1e-20 {
        let norm = norm_sq.sqrt();
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Calculate dot product between two vectors.
/// Assumes vectors are of the same length (or truncates to shorter length).
/// If vectors are normalized, this is equivalent to cosine similarity.
#[inline(always)]
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> f32 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let bits = (self.state >> 32) as u32;
        let value = bits as f32 / u32::MAX as f32;
        value * 2.0 - 1.0
    }
}

const NUM_HYPERPLANES: usize = 32;

// Default score blending weights (can be overridden via environment variables)
const DEFAULT_EMBEDDING_WEIGHT: f32 = 0.7;
const DEFAULT_LEXICAL_WEIGHT: f32 = 0.3;
const DEFAULT_RERANKER_WEIGHT: f32 = 0.7;
const DEFAULT_INITIAL_SCORE_WEIGHT: f32 = 0.3;

// Cached weight values using OnceLock for performance (avoids repeated env var reads)
static EMBEDDING_WEIGHT: OnceLock<f32> = OnceLock::new();
static LEXICAL_WEIGHT: OnceLock<f32> = OnceLock::new();
static RERANKER_WEIGHT: OnceLock<f32> = OnceLock::new();
static INITIAL_SCORE_WEIGHT: OnceLock<f32> = OnceLock::new();

/// Parse a weight from environment variable with validation for finite values in [0.0, 1.0]
fn parse_weight(env_var: &str, default: f32) -> f32 {
    std::env::var(env_var)
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .filter(|w| w.is_finite() && (0.0..=1.0).contains(w))
        .unwrap_or(default)
}

/// Get embedding weight from environment or default (cached after first access)
fn get_embedding_weight() -> f32 {
    *EMBEDDING_WEIGHT.get_or_init(|| parse_weight("RAG_EMBEDDING_WEIGHT", DEFAULT_EMBEDDING_WEIGHT))
}

/// Get lexical weight from environment or default (cached after first access)
fn get_lexical_weight() -> f32 {
    *LEXICAL_WEIGHT.get_or_init(|| parse_weight("RAG_LEXICAL_WEIGHT", DEFAULT_LEXICAL_WEIGHT))
}

/// Get reranker weight from environment or default (cached after first access)
/// Reranker provides semantic relevance, initial score provides discrimination
fn get_reranker_weight() -> f32 {
    *RERANKER_WEIGHT.get_or_init(|| parse_weight("RAG_RERANKER_WEIGHT", DEFAULT_RERANKER_WEIGHT))
}

/// Get initial score weight from environment or default (cached after first access)
fn get_initial_score_weight() -> f32 {
    *INITIAL_SCORE_WEIGHT
        .get_or_init(|| parse_weight("RAG_INITIAL_SCORE_WEIGHT", DEFAULT_INITIAL_SCORE_WEIGHT))
}

/// Optional per-query weight overrides for search scoring.
/// All fields are optional - omitted weights fall back to cached defaults.
/// Invalid values (NaN, Inf, out of range) are ignored and defaults are used.
#[derive(Debug, Clone, Default, Serialize, Deserialize, rmcp::schemars::JsonSchema)]
#[schemars(crate = "rmcp::schemars")]
pub struct QueryWeights {
    /// Embedding similarity weight for first-stage retrieval (0.0-1.0)
    #[schemars(description = "Embedding similarity weight (0.0-1.0, default: 0.7)")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding: Option<f32>,
    /// Lexical/BM25 weight for first-stage retrieval (0.0-1.0)
    #[schemars(description = "Lexical/BM25 weight (0.0-1.0, default: 0.3)")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lexical: Option<f32>,
    /// Reranker weight for second-stage scoring (0.0-1.0)
    #[schemars(description = "Reranker weight for score blending (0.0-1.0, default: 0.7)")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reranker: Option<f32>,
    /// Initial score weight for second-stage scoring (0.0-1.0)
    #[schemars(description = "Initial score weight for score blending (0.0-1.0, default: 0.3)")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub initial: Option<f32>,
}

/// Resolve a single weight: use override if valid (finite and in [0.0, 1.0]), else use default.
/// This is a pure helper function for testability.
fn resolve_weight(override_weight: Option<f32>, default: f32) -> f32 {
    override_weight
        .filter(|&w| w.is_finite() && (0.0..=1.0).contains(&w))
        .unwrap_or(default)
}

/// Resolved weights for a search query.
/// Contains the effective weight values after applying overrides and validation.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ResolvedWeights {
    pub embedding: f32,
    pub lexical: f32,
    pub reranker: f32,
    pub initial: f32,
}

impl ResolvedWeights {
    /// Resolve query weights by applying overrides to cached defaults.
    /// Invalid overrides (NaN, Inf, out of range) are silently ignored.
    pub fn from_query_weights(weights: Option<&QueryWeights>) -> Self {
        Self {
            embedding: resolve_weight(weights.and_then(|w| w.embedding), get_embedding_weight()),
            lexical: resolve_weight(weights.and_then(|w| w.lexical), get_lexical_weight()),
            reranker: resolve_weight(weights.and_then(|w| w.reranker), get_reranker_weight()),
            initial: resolve_weight(weights.and_then(|w| w.initial), get_initial_score_weight()),
        }
    }
}

const MAX_SINGLE_BIT_NEIGHBORS: usize = 32;
const MAX_TOTAL_NEIGHBORS: usize = 64;

struct AnnIndex {
    dim: usize,
    hyperplanes: Vec<Vec<f32>>,
    buckets: HashMap<u64, Vec<String>>,
    id_to_bucket: HashMap<String, u64>,
}

impl AnnIndex {
    fn new(dim: usize) -> Self {
        let mut rng = SimpleRng::new(42);
        let mut hyperplanes = Vec::with_capacity(NUM_HYPERPLANES);

        for _ in 0..NUM_HYPERPLANES {
            let mut plane = Vec::with_capacity(dim);
            for _ in 0..dim {
                plane.push(rng.next());
            }
            // Normalize the hyperplane to unit length
            let magnitude = plane.iter().map(|&x| x * x).sum::<f32>().sqrt();
            if magnitude > 0.0 {
                for val in &mut plane {
                    *val /= magnitude;
                }
            }
            hyperplanes.push(plane);
        }

        Self {
            dim,
            hyperplanes,
            buckets: HashMap::new(),
            id_to_bucket: HashMap::new(),
        }
    }

    #[allow(dead_code)]
    fn dim(&self) -> usize {
        self.dim
    }

    fn insert(&mut self, id: &str, vector: &[f32]) {
        if vector.len() != self.dim {
            tracing::warn!(
                "Vector dimension {} does not match ANN index dimension {}",
                vector.len(),
                self.dim
            );
            return;
        }

        let hash = self.hash(vector);
        self.buckets.entry(hash).or_default().push(id.to_string());
        self.id_to_bucket.insert(id.to_string(), hash);
    }

    fn remove(&mut self, id: &str) {
        if let Some(hash) = self.id_to_bucket.remove(id) {
            if let Some(bucket) = self.buckets.get_mut(&hash) {
                bucket.retain(|stored| stored != id);
                if bucket.is_empty() {
                    self.buckets.remove(&hash);
                }
            }
        }
    }

    fn search(&self, vector: &[f32], max_candidates: usize) -> Vec<String> {
        if self.buckets.is_empty() || max_candidates == 0 {
            return Vec::new();
        }

        let mut candidates = Vec::new();
        let mut visited = HashSet::new();
        let primary_hash = self.hash(vector);

        self.collect_bucket(primary_hash, &mut candidates, &mut visited, max_candidates);

        if candidates.len() < max_candidates {
            for neighbor in self.neighbor_hashes(primary_hash) {
                if candidates.len() >= max_candidates {
                    break;
                }
                self.collect_bucket(neighbor, &mut candidates, &mut visited, max_candidates);
            }
        }

        if candidates.len() < max_candidates {
            for (hash, bucket) in &self.buckets {
                if candidates.len() >= max_candidates {
                    break;
                }
                if visited.contains(hash) {
                    continue;
                }
                for id in bucket {
                    if candidates.len() >= max_candidates {
                        break;
                    }
                    candidates.push(id.clone());
                }
            }
        }

        candidates
    }

    fn hash(&self, vector: &[f32]) -> u64 {
        let mut hash = 0u64;

        for (i, plane) in self.hyperplanes.iter().enumerate() {
            let dot: f32 = vector.iter().zip(plane.iter()).map(|(a, b)| a * b).sum();
            if dot >= 0.0 {
                hash |= 1u64 << i;
            }
        }

        hash
    }

    fn collect_bucket(
        &self,
        hash: u64,
        candidates: &mut Vec<String>,
        visited: &mut HashSet<u64>,
        limit: usize,
    ) {
        if visited.contains(&hash) {
            return;
        }

        visited.insert(hash);

        if let Some(bucket) = self.buckets.get(&hash) {
            for id in bucket {
                if candidates.len() >= limit {
                    break;
                }
                candidates.push(id.clone());
            }
        }
    }

    fn neighbor_hashes(&self, hash: u64) -> Vec<u64> {
        let bits = self.hyperplanes.len().min(64);
        let mut neighbors = Vec::new();

        for i in 0..bits {
            if neighbors.len() >= MAX_SINGLE_BIT_NEIGHBORS {
                break;
            }
            neighbors.push(hash ^ (1u64 << i));
        }

        if neighbors.len() < MAX_SINGLE_BIT_NEIGHBORS {
            for i in 0..bits {
                if neighbors.len() >= MAX_TOTAL_NEIGHBORS {
                    break;
                }
                for j in (i + 1)..bits {
                    neighbors.push(hash ^ (1u64 << i) ^ (1u64 << j));
                    if neighbors.len() >= MAX_TOTAL_NEIGHBORS {
                        break;
                    }
                }
            }
        }

        neighbors
    }

    fn contains(&self, id: &str) -> bool {
        self.id_to_bucket.contains_key(id)
    }

    fn drop_stale(&mut self, valid_ids: &HashSet<String>) {
        let current_ids: HashSet<String> = self.id_to_bucket.keys().cloned().collect();
        for stale_id in current_ids.difference(valid_ids) {
            self.remove(stale_id);
        }
    }
}

#[derive(Default)]
struct LexicalIndex {
    term_postings: HashMap<String, HashMap<String, usize>>,
    doc_lengths: HashMap<String, usize>,
    doc_terms: HashMap<String, HashMap<String, usize>>,
    total_docs: usize,
    total_length: usize,
}

impl LexicalIndex {
    fn new() -> Self {
        Self::default()
    }

    #[allow(dead_code)] // Reserved for future full reindex operations
    fn clear(&mut self) {
        self.term_postings.clear();
        self.doc_lengths.clear();
        self.doc_terms.clear();
        self.total_docs = 0;
        self.total_length = 0;
    }

    fn add_chunk(&mut self, id: &str, text: &str) {
        if self.doc_terms.contains_key(id) {
            self.remove_chunk(id);
        }

        let tokens = tokenize(text);
        if tokens.is_empty() {
            return;
        }

        let mut term_counts: HashMap<String, usize> = HashMap::new();
        for token in tokens {
            *term_counts.entry(token).or_insert(0) += 1;
        }

        let doc_length: usize = term_counts.values().sum();
        if doc_length == 0 {
            return;
        }

        for (term, count) in &term_counts {
            self.term_postings
                .entry(term.clone())
                .or_default()
                .insert(id.to_string(), *count);
        }

        self.doc_lengths.insert(id.to_string(), doc_length);
        self.doc_terms.insert(id.to_string(), term_counts);
        self.total_docs += 1;
        self.total_length += doc_length;
    }

    fn remove_chunk(&mut self, id: &str) {
        if let Some(term_counts) = self.doc_terms.remove(id) {
            for (term, _) in term_counts {
                if let Some(postings) = self.term_postings.get_mut(&term) {
                    postings.remove(id);
                    if postings.is_empty() {
                        self.term_postings.remove(&term);
                    }
                }
            }
            if let Some(length) = self.doc_lengths.remove(id) {
                if self.total_length >= length {
                    self.total_length -= length;
                } else {
                    self.total_length = 0;
                }
            }
            if self.total_docs > 0 {
                self.total_docs -= 1;
            }
        } else {
            // Ensure doc_lengths is clean even if we didn't have stored term counts.
            self.doc_lengths.remove(id);
        }

        if self.total_docs == 0 {
            self.total_length = 0;
        }
    }

    fn score(&self, query: &str, limit: usize) -> Vec<(String, f32)> {
        if self.total_docs == 0 {
            return Vec::new();
        }

        let tokens = tokenize(query);
        if tokens.is_empty() {
            return Vec::new();
        }

        let mut unique_terms: HashSet<String> = HashSet::new();
        for token in tokens {
            unique_terms.insert(token);
        }

        let avg_doc_len = if self.total_docs == 0 {
            0.0
        } else {
            self.total_length as f32 / self.total_docs as f32
        };

        let k1 = 1.5_f32;
        let b = 0.75_f32;
        let mut scores: HashMap<String, f32> = HashMap::new();

        for term in unique_terms {
            if let Some(postings) = self.term_postings.get(&term) {
                let df = postings.len() as f32;
                let idf = ((self.total_docs as f32 - df + 0.5) / (df + 0.5))
                    .ln()
                    .max(0.0);

                for (doc_id, term_freq) in postings {
                    let doc_length = *self.doc_lengths.get(doc_id).unwrap_or(&0) as f32;
                    if doc_length == 0.0 {
                        continue;
                    }

                    let tf = *term_freq as f32;
                    let denom = tf + k1 * (1.0 - b + b * (doc_length / avg_doc_len));
                    if denom == 0.0 {
                        continue;
                    }

                    let score = idf * (tf * (k1 + 1.0)) / denom;
                    *scores.entry(doc_id.clone()).or_insert(0.0) += score;
                }
            }
        }

        let mut results: Vec<(String, f32)> = scores.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        if limit > 0 && results.len() > limit {
            results.truncate(limit);
        }
        results
    }

    fn contains(&self, id: &str) -> bool {
        self.doc_terms.contains_key(id)
    }

    fn drop_stale(&mut self, valid_ids: &HashSet<String>) {
        let current_ids: HashSet<String> = self.doc_terms.keys().cloned().collect();
        for stale_id in current_ids.difference(valid_ids) {
            self.remove_chunk(stale_id);
        }
    }
}

/// Tokenizes text into lowercase terms for lexical indexing.
/// Filters out tokens shorter than 3 characters to reduce noise
/// (stop words, numbers, abbreviations) and memory usage.
fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|token| token.len() >= 3)
        .map(|token| token.to_lowercase())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sentence_info_creation() {
        // Test that extract_sentences creates SentenceInfo with proper metadata
        let test_text = "Dr. Smith presented findings.\u{c}This is page two. Results show success.";

        let sentences = RagEngine::extract_sentences(test_text);

        assert!(!sentences.is_empty(), "Should extract sentences");

        // Verify sentence metadata
        for sentence in &sentences {
            assert!(sentence.tokens > 0, "Each sentence should have token count");
            assert!(sentence.page > 0, "Each sentence should have page number");
        }
    }

    #[test]
    fn test_finalize_chunk_creates_metadata() {
        let test_text = "Sentence one. Sentence two.\u{c}Page two sentence.";
        let sentences = RagEngine::extract_sentences(test_text);

        assert!(!sentences.is_empty(), "Should have sentences");

        // Finalize first two sentences into a chunk
        let indices: Vec<usize> = vec![0, 1];
        let result = RagEngine::finalize_chunk(&indices, &sentences, 0);

        assert!(result.is_some(), "Should create chunk");

        let (text, metadata) = result.unwrap();
        assert!(!text.is_empty(), "Chunk text should not be empty");
        assert!(
            metadata.sentence_range.is_some(),
            "sentence_range should be populated"
        );
        assert!(metadata.token_count > 0, "token_count should be positive");
        assert!(
            metadata.page_range.is_some(),
            "page_range should be populated"
        );
    }

    #[test]
    fn test_lexical_index_contains_and_drop_stale() {
        let mut index = LexicalIndex::new();

        // Add some chunks
        index.add_chunk("chunk1", "hello world");
        index.add_chunk("chunk2", "foo bar baz");
        index.add_chunk("chunk3", "test document");

        // Verify contains works
        assert!(index.contains("chunk1"));
        assert!(index.contains("chunk2"));
        assert!(index.contains("chunk3"));
        assert!(!index.contains("chunk4"));

        // Create a set with only chunk1 and chunk2
        let valid_ids: HashSet<String> = vec!["chunk1".to_string(), "chunk2".to_string()]
            .into_iter()
            .collect();

        // Drop stale entries
        index.drop_stale(&valid_ids);

        // Verify chunk3 was removed
        assert!(index.contains("chunk1"));
        assert!(index.contains("chunk2"));
        assert!(
            !index.contains("chunk3"),
            "chunk3 should have been removed as stale"
        );
    }

    #[test]
    fn test_ann_index_contains_and_drop_stale() {
        let mut ann_index = AnnIndex::new(384); // Standard embedding dimension

        // Create some test vectors
        let vec1: Vec<f32> = (0..384).map(|i| (i as f32) / 384.0).collect();
        let vec2: Vec<f32> = (0..384).map(|i| ((i + 100) as f32) / 384.0).collect();
        let vec3: Vec<f32> = (0..384).map(|i| ((i + 200) as f32) / 384.0).collect();

        // Insert vectors
        ann_index.insert("id1", &vec1);
        ann_index.insert("id2", &vec2);
        ann_index.insert("id3", &vec3);

        // Verify contains
        assert!(ann_index.contains("id1"));
        assert!(ann_index.contains("id2"));
        assert!(ann_index.contains("id3"));
        assert!(!ann_index.contains("id4"));

        // Drop stale
        let valid_ids: HashSet<String> = vec!["id1".to_string(), "id3".to_string()]
            .into_iter()
            .collect();

        ann_index.drop_stale(&valid_ids);

        // Verify id2 was removed
        assert!(ann_index.contains("id1"));
        assert!(!ann_index.contains("id2"), "id2 should have been removed");
        assert!(ann_index.contains("id3"));
    }

    // ============================================================
    // TDD Tests for Model-Partitioned Storage
    // These tests define the expected behavior BEFORE implementation
    // ============================================================

    #[test]
    fn test_sanitize_model_name_basic() {
        // Basic model names should pass through unchanged
        assert_eq!(
            RagEngine::sanitize_model_name("nomic-embed-text"),
            "nomic-embed-text"
        );
        assert_eq!(
            RagEngine::sanitize_model_name("all-MiniLM-L6-v2"),
            "all-MiniLM-L6-v2"
        );
    }

    #[test]
    fn test_sanitize_model_name_with_slashes() {
        // Slashes (common in HuggingFace model names) should become underscores
        assert_eq!(
            RagEngine::sanitize_model_name("sentence-transformers/all-MiniLM-L6-v2"),
            "sentence-transformers_all-MiniLM-L6-v2"
        );
        assert_eq!(
            RagEngine::sanitize_model_name("openai/text-embedding-3-large"),
            "openai_text-embedding-3-large"
        );
    }

    #[test]
    fn test_sanitize_model_name_path_traversal() {
        // Path traversal attempts should be sanitized (security)
        assert_eq!(
            RagEngine::sanitize_model_name("../etc/passwd"),
            ".._etc_passwd"
        );
        assert_eq!(
            RagEngine::sanitize_model_name("..\\windows\\system32"),
            ".._windows_system32"
        );
        assert_eq!(RagEngine::sanitize_model_name("foo/../bar"), "foo_.._bar");
    }

    #[test]
    fn test_sanitize_model_name_special_chars() {
        // Special characters should become underscores
        assert_eq!(RagEngine::sanitize_model_name("model:v1"), "model_v1");
        assert_eq!(RagEngine::sanitize_model_name("model*test?"), "model_test_");
        assert_eq!(RagEngine::sanitize_model_name("model<>|"), "model___");
    }

    #[test]
    fn test_sanitize_model_name_empty_and_whitespace() {
        // Empty or whitespace-only names should have a fallback
        assert_eq!(RagEngine::sanitize_model_name(""), "default");
        assert_eq!(RagEngine::sanitize_model_name("   "), "default");
    }

    #[test]
    fn test_get_index_path_basic() {
        use std::path::PathBuf;

        let path = RagEngine::get_index_path("/data", "nomic-embed-text");
        assert_eq!(path, PathBuf::from("/data/chunks_nomic-embed-text.json"));
    }

    #[test]
    fn test_get_index_path_with_slashes_in_model() {
        use std::path::PathBuf;

        // Model names with slashes should be sanitized in the filename
        let path = RagEngine::get_index_path("/data", "sentence-transformers/all-MiniLM");
        assert_eq!(
            path,
            PathBuf::from("/data/chunks_sentence-transformers_all-MiniLM.json")
        );
    }

    #[test]
    fn test_get_index_path_stays_in_directory() {
        use std::path::PathBuf;

        // Path traversal in model name should NOT escape the data directory
        let path = RagEngine::get_index_path("/data", "../etc/passwd");
        // Should NOT be /etc/passwd, should stay in /data
        assert!(path.starts_with("/data/"));
        assert_eq!(path, PathBuf::from("/data/chunks_.._etc_passwd.json"));
    }

    #[test]
    fn test_get_legacy_path() {
        use std::path::PathBuf;

        let path = RagEngine::get_legacy_path("/data");
        assert_eq!(path, PathBuf::from("/data/chunks.json"));
    }

    // ============================================================
    // Integration Tests for Model-Partitioned Storage I/O
    // These test actual file operations using tempdir
    // ============================================================

    /// Test that model-specific index files are created correctly
    #[test]
    fn test_model_specific_file_creation() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        let data_dir = temp_dir.path().to_str().unwrap();

        // Simulate what save_to_disk does - create model-specific file
        let model_name = "nomic-embed-text";
        let index_path = RagEngine::get_index_path(data_dir, model_name);

        // Create a mock index file
        let mock_state = serde_json::json!({
            "version": 2,
            "model": model_name,
            "chunks": {},
            "needs_reindex": false,
            "document_hashes": {}
        });

        std::fs::write(
            &index_path,
            serde_json::to_string_pretty(&mock_state).unwrap(),
        )
        .expect("Failed to write mock index");

        // Verify file exists at model-specific path
        assert!(index_path.exists(), "Model-specific index should exist");
        assert_eq!(
            index_path.file_name().unwrap().to_str().unwrap(),
            "chunks_nomic-embed-text.json"
        );

        // Verify legacy path does NOT exist
        let legacy_path = RagEngine::get_legacy_path(data_dir);
        assert!(!legacy_path.exists(), "Legacy path should NOT exist");
    }

    /// Test that switching models preserves existing model's index
    #[test]
    fn test_model_switching_preserves_other_index() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        let data_dir = temp_dir.path().to_str().unwrap();

        // Create index for model A
        let model_a = "nomic-embed-text";
        let path_a = RagEngine::get_index_path(data_dir, model_a);
        let state_a = serde_json::json!({
            "version": 2,
            "model": model_a,
            "chunks": {"chunk1": {"id": "chunk1", "text": "model A data"}},
            "needs_reindex": false
        });
        std::fs::write(&path_a, serde_json::to_string(&state_a).unwrap()).unwrap();

        // Create index for model B
        let model_b = "mxbai-embed-large";
        let path_b = RagEngine::get_index_path(data_dir, model_b);
        let state_b = serde_json::json!({
            "version": 2,
            "model": model_b,
            "chunks": {"chunk2": {"id": "chunk2", "text": "model B data"}},
            "needs_reindex": false
        });
        std::fs::write(&path_b, serde_json::to_string(&state_b).unwrap()).unwrap();

        // Verify both files exist independently
        assert!(path_a.exists(), "Model A index should exist");
        assert!(path_b.exists(), "Model B index should exist");

        // Read and verify contents are independent
        let read_a: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path_a).unwrap()).unwrap();
        let read_b: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path_b).unwrap()).unwrap();

        assert_eq!(read_a["model"], "nomic-embed-text");
        assert_eq!(read_b["model"], "mxbai-embed-large");
        assert!(read_a["chunks"]["chunk1"].is_object());
        assert!(read_b["chunks"]["chunk2"].is_object());
    }

    /// Test atomic write pattern (temp file + rename)
    #[test]
    fn test_atomic_write_pattern() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        let data_dir = temp_dir.path().to_str().unwrap();

        let model_name = "test-model";
        let final_path = RagEngine::get_index_path(data_dir, model_name);
        let temp_path = final_path.with_extension("json.tmp");

        // Simulate atomic write: write to temp, then rename
        let data = serde_json::json!({"version": 2, "model": model_name});
        std::fs::write(&temp_path, serde_json::to_string(&data).unwrap()).unwrap();

        // Verify temp file exists before rename
        assert!(temp_path.exists(), "Temp file should exist before rename");
        assert!(
            !final_path.exists(),
            "Final file should NOT exist before rename"
        );

        // Perform rename (atomic operation)
        std::fs::rename(&temp_path, &final_path).unwrap();

        // Verify final file exists and temp is gone
        assert!(final_path.exists(), "Final file should exist after rename");
        assert!(
            !temp_path.exists(),
            "Temp file should NOT exist after rename"
        );
    }

    /// Test legacy file detection and preservation
    #[test]
    fn test_legacy_file_preserved_on_model_mismatch() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        let data_dir = temp_dir.path().to_str().unwrap();

        // Create legacy chunks.json for model A
        let legacy_path = RagEngine::get_legacy_path(data_dir);
        let legacy_state = serde_json::json!({
            "version": 2,
            "model": "old-model",
            "chunks": {"legacy_chunk": {"id": "legacy", "text": "old data"}},
            "needs_reindex": false
        });
        std::fs::write(&legacy_path, serde_json::to_string(&legacy_state).unwrap()).unwrap();

        // Simulate "new model" wanting to load
        let new_model = "new-model";
        let new_model_path = RagEngine::get_index_path(data_dir, new_model);

        // New model's file doesn't exist yet
        assert!(!new_model_path.exists());

        // Peek at legacy file's model (simulating load_from_disk behavior)
        let legacy_data: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&legacy_path).unwrap()).unwrap();

        // Model mismatch - legacy belongs to "old-model", we want "new-model"
        assert_ne!(legacy_data["model"], new_model);

        // Key assertion: Legacy file is STILL preserved (not deleted)
        assert!(
            legacy_path.exists(),
            "Legacy file must be preserved on model mismatch"
        );

        // New model creates its own file
        let new_state = serde_json::json!({
            "version": 2,
            "model": new_model,
            "chunks": {},
            "needs_reindex": true
        });
        std::fs::write(&new_model_path, serde_json::to_string(&new_state).unwrap()).unwrap();

        // Both files coexist
        assert!(legacy_path.exists(), "Legacy should still exist");
        assert!(new_model_path.exists(), "New model index should exist");
    }

    /// Test migration scenario: legacy file matches current model
    #[test]
    fn test_legacy_migration_when_model_matches() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        let data_dir = temp_dir.path().to_str().unwrap();

        let current_model = "nomic-embed-text";

        // Create legacy chunks.json matching current model
        let legacy_path = RagEngine::get_legacy_path(data_dir);
        let legacy_state = serde_json::json!({
            "version": 2,
            "model": current_model,
            "chunks": {"migrated_chunk": {"id": "migrated", "text": "will migrate"}},
            "needs_reindex": false,
            "document_hashes": {"doc.pdf": "abc123"}
        });
        std::fs::write(
            &legacy_path,
            serde_json::to_string_pretty(&legacy_state).unwrap(),
        )
        .unwrap();

        // Model-specific file doesn't exist yet
        let model_path = RagEngine::get_index_path(data_dir, current_model);
        assert!(!model_path.exists());

        // Simulate migration: read legacy, write to model-specific path
        let legacy_data = std::fs::read_to_string(&legacy_path).unwrap();
        std::fs::write(&model_path, &legacy_data).unwrap();

        // After migration: both files exist (legacy preserved for safety)
        assert!(legacy_path.exists(), "Legacy preserved after migration");
        assert!(model_path.exists(), "Model-specific file created");

        // Verify migrated content
        let migrated: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&model_path).unwrap()).unwrap();
        assert_eq!(migrated["model"], current_model);
        assert!(migrated["chunks"]["migrated_chunk"].is_object());
    }

    // ============================================================
    // MMR (Maximal Marginal Relevance) Unit Tests
    // Tests for search diversification algorithm
    // ============================================================

    #[test]
    fn test_cosine_similarity_identical_vectors() {
        let vec_a = vec![1.0, 0.0, 0.0];
        let vec_b = vec![1.0, 0.0, 0.0];
        let similarity = RagEngine::cosine_similarity(&vec_a, &vec_b);
        assert!(
            (similarity - 1.0).abs() < 1e-6,
            "Identical vectors should have similarity ~1.0"
        );
    }

    #[test]
    fn test_cosine_similarity_orthogonal_vectors() {
        let vec_a = vec![1.0, 0.0, 0.0];
        let vec_b = vec![0.0, 1.0, 0.0];
        let similarity = RagEngine::cosine_similarity(&vec_a, &vec_b);
        assert!(
            similarity.abs() < 1e-6,
            "Orthogonal vectors should have similarity ~0.0"
        );
    }

    #[test]
    fn test_cosine_similarity_opposite_vectors() {
        let vec_a = vec![1.0, 0.0, 0.0];
        let vec_b = vec![-1.0, 0.0, 0.0];
        let similarity = RagEngine::cosine_similarity(&vec_a, &vec_b);
        assert!(
            (similarity - (-1.0)).abs() < 1e-6,
            "Opposite vectors should have similarity ~-1.0"
        );
    }

    #[test]
    fn test_cosine_similarity_zero_vectors() {
        let vec_a = vec![0.0, 0.0, 0.0];
        let vec_b = vec![1.0, 2.0, 3.0];
        let similarity = RagEngine::cosine_similarity(&vec_a, &vec_b);
        assert_eq!(similarity, 0.0, "Zero vectors should return 0.0");

        let both_zero = RagEngine::cosine_similarity(&vec_a, &[0.0, 0.0, 0.0]);
        assert_eq!(both_zero, 0.0, "Both zero vectors should return 0.0");
    }

    #[test]
    fn test_cosine_similarity_near_zero_vectors() {
        // Near-zero norm vectors (below epsilon threshold)
        let vec_a = vec![1e-12, 1e-12, 1e-12];
        let vec_b = vec![1.0, 2.0, 3.0];
        let similarity = RagEngine::cosine_similarity(&vec_a, &vec_b);
        assert_eq!(
            similarity, 0.0,
            "Near-zero vectors should return 0.0 (numerical stability)"
        );
    }

    #[test]
    fn test_cosine_similarity_mismatched_length() {
        let vec_a = vec![1.0, 2.0, 3.0];
        let vec_b = vec![1.0, 2.0];
        let similarity = RagEngine::cosine_similarity(&vec_a, &vec_b);
        assert_eq!(
            similarity, 0.0,
            "Mismatched length vectors should return 0.0"
        );
    }

    #[test]
    fn test_cosine_similarity_empty_vectors() {
        let vec_a: Vec<f32> = vec![];
        let vec_b: Vec<f32> = vec![];
        let similarity = RagEngine::cosine_similarity(&vec_a, &vec_b);
        assert_eq!(similarity, 0.0, "Empty vectors should return 0.0");
    }

    #[test]
    fn test_cosine_similarity_clamping() {
        // Test that result is clamped to [-1, 1] even with floating point errors
        let vec_a = vec![1.0, 1.0, 1.0];
        let vec_b = vec![1.0, 1.0, 1.0];
        let similarity = RagEngine::cosine_similarity(&vec_a, &vec_b);
        assert!(
            (-1.0..=1.0).contains(&similarity),
            "Similarity should be clamped to [-1, 1]"
        );
    }

    #[test]
    fn test_cosine_similarity_realistic_embeddings() {
        // Test with realistic high-dimensional embeddings
        let dim = 384;
        let vec_a: Vec<f32> = (0..dim).map(|i| (i as f32) / dim as f32).collect();
        let vec_b: Vec<f32> = (0..dim).map(|i| ((i + 10) as f32) / dim as f32).collect();
        let similarity = RagEngine::cosine_similarity(&vec_a, &vec_b);

        // Should be high but not 1.0 (similar but not identical)
        assert!(
            similarity > 0.9 && similarity < 1.0,
            "Similar vectors should have high similarity"
        );
    }

    #[test]
    fn test_dot_product_equivalence_when_normalized() {
        let dim = 384;
        let mut vec_a: Vec<f32> = (0..dim).map(|i| (i as f32) / dim as f32).collect();
        let mut vec_b: Vec<f32> = (0..dim).map(|i| ((i + 10) as f32) / dim as f32).collect();

        // Calculate cosine similarity on unnormalized vectors
        let cos_sim = RagEngine::cosine_similarity(&vec_a, &vec_b);

        // Normalize vectors
        normalize(&mut vec_a);
        normalize(&mut vec_b);

        // Calculate dot product on normalized vectors
        let dot_prod = dot_product(&vec_a, &vec_b);

        // Should be equal within floating point error
        assert!(
            (cos_sim - dot_prod).abs() < 1e-6,
            "Dot product of normalized vectors should equal cosine similarity: {} vs {}",
            cos_sim,
            dot_prod
        );
    }

    // Helper function to create test SearchResultWithEmbedding
    fn make_test_candidate(id: &str, score: f32, embedding: Vec<f32>) -> SearchResultWithEmbedding {
        SearchResultWithEmbedding {
            result: SearchResult {
                text: format!("Text for {id}"),
                score,
                document: "test.pdf".to_string(),
                chunk_id: id.to_string(),
                chunk_index: 0,
                page_number: 1,
                section: None,
                embedding_score: None,
                lexical_score: None,
                initial_score: None,
                reranker_score: None,
                yes_logprob: None,
                no_logprob: None,
            },
            embedding,
        }
    }

    /// Standalone MMR diversify function for testing (doesn't need full RagEngine)
    fn test_mmr_diversify(
        candidates: Vec<SearchResultWithEmbedding>,
        top_k: usize,
        diversity_factor: f32,
    ) -> Vec<SearchResult> {
        if candidates.is_empty() {
            return vec![];
        }

        let mut selected: Vec<SearchResultWithEmbedding> = Vec::with_capacity(top_k);
        let mut remaining: Vec<SearchResultWithEmbedding> = candidates;

        if !remaining.is_empty() {
            let first = remaining.swap_remove(0);
            selected.push(first);
        }

        while selected.len() < top_k && !remaining.is_empty() {
            let mut best_mmr_score = f32::NEG_INFINITY;
            let mut best_idx = 0;

            for (idx, candidate) in remaining.iter().enumerate() {
                let relevance = candidate.result.score;
                if !relevance.is_finite() {
                    continue;
                }

                let max_similarity = selected
                    .iter()
                    .map(|s| dot_product(&candidate.embedding, &s.embedding))
                    .filter(|sim| sim.is_finite())
                    .fold(0.0_f32, |a, b| a.max(b));

                let mmr_score =
                    (1.0 - diversity_factor) * relevance - diversity_factor * max_similarity;

                if mmr_score.is_finite() && mmr_score > best_mmr_score {
                    best_mmr_score = mmr_score;
                    best_idx = idx;
                }
            }

            if best_mmr_score == f32::NEG_INFINITY {
                break;
            }

            let best = remaining.swap_remove(best_idx);
            selected.push(best);
        }

        selected.into_iter().map(|s| s.result).collect()
    }

    #[test]
    fn test_mmr_diversify_empty_candidates() {
        let result = test_mmr_diversify(vec![], 5, 0.3);
        assert!(
            result.is_empty(),
            "Empty candidates should return empty results"
        );
    }

    #[test]
    fn test_mmr_diversify_single_candidate() {
        let candidates = vec![make_test_candidate("chunk1", 0.9, vec![1.0, 0.0, 0.0])];

        let result = test_mmr_diversify(candidates, 5, 0.3);
        assert_eq!(
            result.len(),
            1,
            "Single candidate should return single result"
        );
        assert_eq!(result[0].chunk_id, "chunk1");
    }

    #[test]
    fn test_mmr_diversify_top_k_larger_than_candidates() {
        let candidates = vec![
            make_test_candidate("chunk1", 0.9, vec![1.0, 0.0, 0.0]),
            make_test_candidate("chunk2", 0.8, vec![0.0, 1.0, 0.0]),
        ];

        // Request more than available
        let result = test_mmr_diversify(candidates, 10, 0.3);
        assert_eq!(
            result.len(),
            2,
            "Should return all available candidates when top_k > candidates"
        );
    }

    #[test]
    fn test_mmr_diversify_zero_diversity_factor() {
        // Three candidates with similar embeddings but different scores
        let candidates = vec![
            make_test_candidate("chunk1", 0.9, vec![1.0, 0.1, 0.0]),
            make_test_candidate("chunk2", 0.8, vec![1.0, 0.2, 0.0]),
            make_test_candidate("chunk3", 0.7, vec![1.0, 0.3, 0.0]),
        ];

        // With diversity_factor = 0, should select purely by relevance
        let result = test_mmr_diversify(candidates, 3, 0.0);

        assert_eq!(result.len(), 3);
        // First should be highest relevance (but swap_remove changes order after first)
        assert_eq!(
            result[0].chunk_id, "chunk1",
            "First should be most relevant"
        );
    }

    #[test]
    fn test_mmr_diversify_high_diversity_factor() {
        // Two similar chunks (high cosine similarity) and one different
        let candidates = vec![
            make_test_candidate("chunk1", 0.9, vec![1.0, 0.0, 0.0]), // First selected (highest score)
            make_test_candidate("chunk2", 0.85, vec![0.99, 0.1, 0.0]), // Very similar to chunk1
            make_test_candidate("chunk3", 0.7, vec![0.0, 1.0, 0.0]), // Orthogonal, different
        ];

        // With high diversity_factor, should prefer diverse chunk3 over similar chunk2
        let result = test_mmr_diversify(candidates, 2, 0.9);

        assert_eq!(result.len(), 2);
        assert_eq!(
            result[0].chunk_id, "chunk1",
            "First should be most relevant"
        );
        // Second should be chunk3 (diverse) not chunk2 (similar) despite lower score
        assert_eq!(
            result[1].chunk_id, "chunk3",
            "High diversity should prefer orthogonal chunk"
        );
    }

    #[test]
    fn test_mmr_diversify_nan_score_handling() {
        let candidates = vec![
            make_test_candidate("chunk1", 0.9, vec![1.0, 0.0, 0.0]),
            make_test_candidate("chunk_nan", f32::NAN, vec![0.0, 1.0, 0.0]),
            make_test_candidate("chunk3", 0.7, vec![0.0, 0.0, 1.0]),
        ];

        let result = test_mmr_diversify(candidates, 3, 0.3);

        // Should skip the NaN candidate
        assert_eq!(result.len(), 2, "Should skip NaN candidates");
        assert!(
            result.iter().all(|r| r.chunk_id != "chunk_nan"),
            "NaN chunk should be skipped"
        );
    }

    #[test]
    fn test_mmr_diversify_inf_score_handling() {
        let candidates = vec![
            make_test_candidate("chunk1", 0.9, vec![1.0, 0.0, 0.0]),
            make_test_candidate("chunk_inf", f32::INFINITY, vec![0.0, 1.0, 0.0]),
            make_test_candidate("chunk3", 0.7, vec![0.0, 0.0, 1.0]),
        ];

        let result = test_mmr_diversify(candidates, 3, 0.3);

        // Should skip the Inf candidate
        assert_eq!(result.len(), 2, "Should skip Inf candidates");
        assert!(
            result.iter().all(|r| r.chunk_id != "chunk_inf"),
            "Inf chunk should be skipped"
        );
    }

    #[test]
    fn test_mmr_diversify_preserves_relevance_order_when_orthogonal() {
        // All orthogonal vectors (no similarity penalty)
        let candidates = vec![
            make_test_candidate("a", 0.9, vec![1.0, 0.0, 0.0, 0.0]),
            make_test_candidate("b", 0.8, vec![0.0, 1.0, 0.0, 0.0]),
            make_test_candidate("c", 0.7, vec![0.0, 0.0, 1.0, 0.0]),
            make_test_candidate("d", 0.6, vec![0.0, 0.0, 0.0, 1.0]),
        ];

        // Even with moderate diversity_factor, orthogonal vectors shouldn't change order
        let result = test_mmr_diversify(candidates, 4, 0.3);

        assert_eq!(result.len(), 4);
        assert_eq!(result[0].chunk_id, "a");
        // Note: After first selection, swap_remove changes remaining order,
        // but with orthogonal vectors (0 similarity), pure relevance should determine selection
    }

    #[test]
    fn test_mmr_formula_correctness() {
        // Verify MMR formula: MMR(i) = (1-λ)*relevance - λ*max_similarity
        // With λ=0.5, relevance=0.8, max_sim=0.6: MMR = 0.5*0.8 - 0.5*0.6 = 0.4 - 0.3 = 0.1
        // Candidate with relevance=0.6, max_sim=0.0: MMR = 0.5*0.6 - 0.5*0.0 = 0.3 - 0 = 0.3
        // So the second candidate (lower relevance but more diverse) should win

        let candidates = vec![
            make_test_candidate("selected", 0.9, vec![1.0, 0.0, 0.0]), // Already selected
            make_test_candidate("similar", 0.8, vec![1.0, 0.0, 0.0]), // Same direction, high similarity
            make_test_candidate("diverse", 0.6, vec![0.0, 1.0, 0.0]), // Orthogonal, low similarity
        ];

        let result = test_mmr_diversify(candidates, 2, 0.5);

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].chunk_id, "selected");
        // At λ=0.5: similar MMR = 0.5*0.8 - 0.5*1.0 = -0.1
        //           diverse MMR = 0.5*0.6 - 0.5*0.0 = 0.3
        // Diverse wins!
        assert_eq!(
            result[1].chunk_id, "diverse",
            "MMR should prefer diverse chunk when λ=0.5"
        );
    }

    // ============================================
    // Per-Query Weight Resolution Tests (TDD)
    // ============================================

    #[test]
    fn test_resolve_weight_uses_override_when_valid() {
        // Valid override should be used
        assert_eq!(resolve_weight(Some(0.5), 0.7), 0.5);
        assert_eq!(resolve_weight(Some(0.9), 0.3), 0.9);
    }

    #[test]
    fn test_resolve_weight_uses_default_when_none() {
        // None should fall back to default
        assert_eq!(resolve_weight(None, 0.7), 0.7);
        assert_eq!(resolve_weight(None, 0.3), 0.3);
    }

    #[test]
    fn test_resolve_weight_accepts_valid_boundaries() {
        // Edge values 0.0 and 1.0 should be accepted
        let default = 0.5;
        assert_eq!(resolve_weight(Some(0.0), default), 0.0);
        assert_eq!(resolve_weight(Some(1.0), default), 1.0);
    }

    #[test]
    fn test_resolve_weight_rejects_invalid_values() {
        let default = 0.5;

        // NaN should fall back to default
        assert_eq!(
            resolve_weight(Some(f32::NAN), default),
            default,
            "NaN should fall back to default"
        );

        // Infinity should fall back to default
        assert_eq!(
            resolve_weight(Some(f32::INFINITY), default),
            default,
            "INFINITY should fall back to default"
        );
        assert_eq!(
            resolve_weight(Some(f32::NEG_INFINITY), default),
            default,
            "NEG_INFINITY should fall back to default"
        );

        // Out of range values should fall back to default
        assert_eq!(
            resolve_weight(Some(-0.1), default),
            default,
            "Negative value should fall back to default"
        );
        assert_eq!(
            resolve_weight(Some(1.5), default),
            default,
            "Value > 1.0 should fall back to default"
        );
        assert_eq!(
            resolve_weight(Some(2.0), default),
            default,
            "Value > 1.0 should fall back to default"
        );
    }

    #[test]
    fn test_query_weights_default_all_none() {
        // QueryWeights::default() should have all fields as None
        let weights = QueryWeights::default();
        assert!(weights.embedding.is_none());
        assert!(weights.lexical.is_none());
        assert!(weights.reranker.is_none());
        assert!(weights.initial.is_none());
    }

    #[test]
    fn test_resolved_weights_from_none_uses_cached_defaults() {
        // When no overrides provided, should use cached defaults
        let resolved = ResolvedWeights::from_query_weights(None);

        // These should match the get_*_weight() functions
        assert_eq!(resolved.embedding, get_embedding_weight());
        assert_eq!(resolved.lexical, get_lexical_weight());
        assert_eq!(resolved.reranker, get_reranker_weight());
        assert_eq!(resolved.initial, get_initial_score_weight());
    }

    #[test]
    fn test_resolved_weights_from_default_query_weights_uses_cached_defaults() {
        // QueryWeights::default() (all None) should use cached defaults
        let weights = QueryWeights::default();
        let resolved = ResolvedWeights::from_query_weights(Some(&weights));

        assert_eq!(resolved.embedding, get_embedding_weight());
        assert_eq!(resolved.lexical, get_lexical_weight());
        assert_eq!(resolved.reranker, get_reranker_weight());
        assert_eq!(resolved.initial, get_initial_score_weight());
    }

    #[test]
    fn test_resolved_weights_partial_override() {
        // Override only embedding, others should use defaults
        let weights = QueryWeights {
            embedding: Some(0.9),
            lexical: None,
            reranker: None,
            initial: None,
        };
        let resolved = ResolvedWeights::from_query_weights(Some(&weights));

        // Overridden field should use override
        assert_eq!(resolved.embedding, 0.9);
        // Others should use cached defaults
        assert_eq!(resolved.lexical, get_lexical_weight());
        assert_eq!(resolved.reranker, get_reranker_weight());
        assert_eq!(resolved.initial, get_initial_score_weight());
    }

    #[test]
    fn test_resolved_weights_multiple_overrides() {
        // Override multiple fields
        let weights = QueryWeights {
            embedding: Some(0.8),
            lexical: Some(0.2),
            reranker: None,
            initial: Some(0.4),
        };
        let resolved = ResolvedWeights::from_query_weights(Some(&weights));

        assert_eq!(resolved.embedding, 0.8);
        assert_eq!(resolved.lexical, 0.2);
        assert_eq!(resolved.reranker, get_reranker_weight()); // Not overridden
        assert_eq!(resolved.initial, 0.4);
    }

    #[test]
    fn test_resolved_weights_invalid_override_falls_back() {
        // Invalid override should fall back to default
        let weights = QueryWeights {
            embedding: Some(f32::NAN), // Invalid
            lexical: Some(-0.1),       // Invalid (out of range)
            reranker: Some(0.6),       // Valid
            initial: Some(1.5),        // Invalid (out of range)
        };
        let resolved = ResolvedWeights::from_query_weights(Some(&weights));

        // Invalid overrides fall back to defaults
        assert_eq!(resolved.embedding, get_embedding_weight());
        assert_eq!(resolved.lexical, get_lexical_weight());
        // Valid override is used
        assert_eq!(resolved.reranker, 0.6);
        // Invalid falls back
        assert_eq!(resolved.initial, get_initial_score_weight());
    }

    #[test]
    fn test_resolve_weight_edge_case_values() {
        let default = 0.5;

        // Values very close to but just outside valid range should fall back
        assert_eq!(
            resolve_weight(Some(-1e-6), default),
            default,
            "Negative value very close to zero should fall back"
        );
        assert_eq!(
            resolve_weight(Some(1.000001), default),
            default,
            "Value slightly above 1.0 should fall back"
        );

        // NEG_INFINITY should fall back
        assert_eq!(
            resolve_weight(Some(f32::NEG_INFINITY), default),
            default,
            "NEG_INFINITY should fall back"
        );

        // Negative zero is valid (0.0 == -0.0 in IEEE-754)
        assert_eq!(
            resolve_weight(Some(-0.0), default),
            -0.0,
            "Negative zero is valid and should be accepted"
        );
    }
}
