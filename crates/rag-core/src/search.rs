use crate::error::{EngineError, ValidationKind};
use crate::types::SearchResult;
use std::collections::{HashMap, HashSet};
#[cfg(feature = "tracing")]
use tracing::instrument;

/// Normalize a vector to unit length in-place.
/// If the vector has zero or very small norm, it is left unchanged.
pub(crate) fn normalize(v: &mut [f32]) {
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
pub(crate) fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Calculate cosine similarity between two embeddings.
/// Returns a value in [-1, 1] where 1 means identical direction.
/// Returns 0.0 for edge cases (empty, mismatched length, near-zero norm).
///
/// Note: This is a re-export of the canonical implementation in tags.rs
/// for crate-internal use.
#[allow(dead_code)] // used by tests and legacy paths
pub(crate) fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    crate::tags::cosine_similarity(a, b)
}

#[derive(Debug, Clone)]
pub(crate) struct SearchResultWithEmbedding {
    pub(crate) result: SearchResult,
    pub(crate) embedding: Vec<f32>,
}

#[cfg_attr(
    feature = "tracing",
    instrument(
        skip(candidates),
        fields(
            candidate_count = candidates.len(),
            top_k = top_k,
            diversity = diversity_factor
        )
    )
)]
pub(crate) fn mmr_diversify(
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
            #[cfg(feature = "tracing")]
            tracing::warn!("MMR: No valid candidates remaining (all scores non-finite)");
            break;
        }

        let best = remaining.swap_remove(best_idx);

        #[cfg(feature = "tracing")]
        tracing::debug!(
            chunk_id = %best.result.chunk_id,
            relevance = %best.result.score,
            mmr_score = %best_mmr_score,
            "MMR selected result"
        );

        selected.push(best);
    }

    selected.into_iter().map(|s| s.result).collect()
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
const MAX_SINGLE_BIT_NEIGHBORS: usize = 32;
const MAX_TOTAL_NEIGHBORS: usize = 64;

pub(crate) struct AnnIndex {
    dim: usize,
    hyperplanes: Vec<Vec<f32>>,
    buckets: HashMap<u64, Vec<String>>,
    id_to_bucket: HashMap<String, u64>,
}

impl AnnIndex {
    pub(crate) fn new(dim: usize) -> Self {
        let mut rng = SimpleRng::new(42);
        let mut hyperplanes = Vec::with_capacity(NUM_HYPERPLANES);

        for _ in 0..NUM_HYPERPLANES {
            let mut plane = Vec::with_capacity(dim);
            for _ in 0..dim {
                plane.push(rng.next());
            }
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

    pub(crate) fn insert(&mut self, id: &str, vector: &[f32]) {
        if vector.len() != self.dim {
            #[cfg(feature = "tracing")]
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

    pub(crate) fn remove(&mut self, id: &str) {
        if let Some(hash) = self.id_to_bucket.remove(id)
            && let Some(bucket) = self.buckets.get_mut(&hash)
        {
            bucket.retain(|stored| stored != id);
            if bucket.is_empty() {
                self.buckets.remove(&hash);
            }
        }
    }

    pub(crate) fn search(&self, vector: &[f32], max_candidates: usize) -> Vec<String> {
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
            // Sort keys for deterministic fallback iteration
            let mut all_hashes: Vec<&u64> = self.buckets.keys().collect();
            all_hashes.sort();

            for hash in all_hashes {
                if candidates.len() >= max_candidates {
                    break;
                }
                if visited.contains(hash) {
                    continue;
                }
                if let Some(bucket) = self.buckets.get(hash) {
                    for id in bucket {
                        if candidates.len() >= max_candidates {
                            break;
                        }
                        candidates.push(id.clone());
                    }
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

    pub(crate) fn contains(&self, id: &str) -> bool {
        self.id_to_bucket.contains_key(id)
    }

    pub(crate) fn drop_stale(&mut self, valid_ids: &HashSet<String>) {
        let current_ids: HashSet<String> = self.id_to_bucket.keys().cloned().collect();
        for stale_id in current_ids.difference(valid_ids) {
            self.remove(stale_id);
        }
    }
}

#[derive(Default)]
pub(crate) struct LexicalIndex {
    term_postings: HashMap<String, HashMap<String, usize>>,
    doc_lengths: HashMap<String, usize>,
    doc_terms: HashMap<String, HashMap<String, usize>>,
    total_docs: usize,
    total_length: usize,
}

impl LexicalIndex {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn add_chunk(&mut self, id: &str, text: &str) {
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

    pub(crate) fn remove_chunk(&mut self, id: &str) {
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
                self.total_length = self.total_length.saturating_sub(length);
            } else if self.total_docs == 0 {
                self.total_length = 0;
            }
            self.total_docs = self.total_docs.saturating_sub(1);
        } else {
            self.doc_lengths.remove(id);
        }

        if self.total_docs == 0 {
            self.total_length = 0;
        }
    }

    pub(crate) fn score(&self, query: &str, limit: usize) -> Vec<(String, f32)> {
        if self.total_docs == 0 {
            return Vec::new();
        }

        let tokens = tokenize(query);
        if tokens.is_empty() {
            return Vec::new();
        }

        let unique_terms: HashSet<String> = tokens.into_iter().collect();

        let avg_doc_len = if self.total_docs == 0 {
            0.0
        } else {
            self.total_length as f32 / self.total_docs as f32
        };

        let k1 = 1.5_f32;
        let b = 0.75_f32;
        let mut scores: HashMap<&String, f32> = HashMap::new();

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
                    *scores.entry(doc_id).or_insert(0.0) += score;
                }
            }
        }

        let mut results: Vec<(String, f32)> = scores
            .into_iter()
            // Optimization: Clone keys only once at the end instead of during the loop
            .map(|(k, v)| (k.to_string(), v))
            .collect();

        // Optimization: Use select_nth_unstable to find top-k in O(N) time
        // instead of sorting the whole vector in O(N log N).
        if limit > 0 && results.len() > limit {
            results.select_nth_unstable_by(limit, |a, b| {
                b.1.partial_cmp(&a.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.0.cmp(&b.0))
            });
            results.truncate(limit);
        }

        results.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        results
    }

    pub(crate) fn contains(&self, id: &str) -> bool {
        self.doc_terms.contains_key(id)
    }

    pub(crate) fn drop_stale(&mut self, valid_ids: &HashSet<String>) {
        let current_ids: HashSet<String> = self.doc_terms.keys().cloned().collect();
        for stale_id in current_ids.difference(valid_ids) {
            self.remove_chunk(stale_id);
        }
    }
}

/// Tokenizes text into lowercase terms for lexical indexing.
/// Filters out tokens shorter than 3 characters to reduce noise and memory usage.
fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|token| token.len() >= 3)
        .map(|token| token.to_lowercase())
        .collect()
}

// ============================================================================
// IndexSet: Unified wrapper for ANN and Lexical indexes
// ============================================================================

/// Update operation for atomic batch processing.
#[derive(Debug, Clone)]
pub enum ChunkUpdate {
    /// Add a chunk with its text and embedding vector
    Add {
        id: String,
        text: String,
        embedding: Vec<f32>,
    },
    /// Remove a chunk by ID
    Remove { id: String },
}

/// Candidate result from index search with component scores.
#[derive(Debug, Clone)]
pub struct CandidateScore {
    pub chunk_id: String,
    pub embedding_score: f32,
    pub lexical_score: f32,
    pub combined_score: f32,
}

/// Unified index set wrapping ANN (vector) and Lexical (BM25) indexes.
///
/// This struct provides atomic batch operations and maintains a single
/// authoritative chunk registry to ensure both indexes stay synchronized.
///
/// # Invariants
///
/// - Both indexes always contain identical chunk ID sets
/// - Embedding dimension is locked after the first chunk is added
/// - Batch operations are all-or-nothing (atomic)
pub struct IndexSet {
    ann_index: Option<AnnIndex>,
    lexical_index: LexicalIndex,
    /// Authoritative registry of all chunk IDs
    chunk_ids: HashSet<String>,
    /// Embedding dimension, locked after first insert
    embedding_dim: Option<usize>,
}

impl Default for IndexSet {
    fn default() -> Self {
        Self::new()
    }
}

impl IndexSet {
    /// Create a new empty IndexSet.
    pub fn new() -> Self {
        Self {
            ann_index: None,
            lexical_index: LexicalIndex::new(),
            chunk_ids: HashSet::new(),
            embedding_dim: None,
        }
    }

    /// Create an IndexSet with a known embedding dimension.
    pub fn with_dimension(dim: usize) -> Self {
        Self {
            ann_index: Some(AnnIndex::new(dim)),
            lexical_index: LexicalIndex::new(),
            chunk_ids: HashSet::new(),
            embedding_dim: Some(dim),
        }
    }

    /// Apply a batch of updates atomically.
    ///
    /// All updates succeed or all fail. This ensures index consistency
    /// by validating all operations before applying any of them.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Embedding dimension doesn't match (after first insert)
    /// - Embedding contains NaN or Inf values
    /// - Embedding is empty (zero dimension)
    /// - Text is empty or whitespace-only
    pub fn apply_batch(&mut self, updates: &[ChunkUpdate]) -> Result<(), EngineError> {
        // Phase 1: Validate all updates WITHOUT mutating self
        // Determine expected dimension from self or first Add in batch
        let mut expected_dim = self.embedding_dim;

        for update in updates {
            if let ChunkUpdate::Add {
                id,
                text,
                embedding,
            } = update
            {
                // Validate embedding (non-mutating)
                Self::validate_embedding_pure(id, embedding, &mut expected_dim)?;

                // Validate text
                if text.trim().is_empty() {
                    return Err(EngineError::validation(id, ValidationKind::EmptyText));
                }
            }
        }

        // Phase 2: Now that all validation passed, commit dimension if needed
        if let Some(dim) = expected_dim {
            if self.embedding_dim.is_none() {
                self.embedding_dim = Some(dim);
                self.ann_index = Some(AnnIndex::new(dim));
            }
        }

        // Phase 3: Apply all updates (validated, won't fail)
        for update in updates {
            match update {
                ChunkUpdate::Add {
                    id,
                    text,
                    embedding,
                } => {
                    self.add_chunk_internal(id, text, embedding);
                }
                ChunkUpdate::Remove { id } => {
                    self.remove_chunk_internal(id);
                }
            }
        }

        Ok(())
    }

    /// Add a single chunk. For bulk operations, prefer `apply_batch`.
    pub fn add_chunk(
        &mut self,
        id: &str,
        text: &str,
        embedding: &[f32],
    ) -> Result<(), EngineError> {
        // Validate without mutating
        let mut expected_dim = self.embedding_dim;
        Self::validate_embedding_pure(id, embedding, &mut expected_dim)?;

        if text.trim().is_empty() {
            return Err(EngineError::validation(id, ValidationKind::EmptyText));
        }

        // Commit dimension if this is the first insert
        if let Some(dim) = expected_dim {
            if self.embedding_dim.is_none() {
                self.embedding_dim = Some(dim);
                self.ann_index = Some(AnnIndex::new(dim));
            }
        }

        self.add_chunk_internal(id, text, embedding);
        Ok(())
    }

    /// Remove a single chunk.
    pub fn remove_chunk(&mut self, id: &str) {
        self.remove_chunk_internal(id);
    }

    /// Search both indexes and merge results with configurable weights.
    ///
    /// Returns candidates sorted by combined score in descending order.
    /// Uses deterministic tie-breaking (chunk_id) for stable ordering.
    pub fn search_candidates(
        &self,
        query_embedding: &[f32],
        query_text: &str,
        embedding_weight: f32,
        lexical_weight: f32,
        max_candidates: usize,
    ) -> Vec<CandidateScore> {
        let mut scores: HashMap<String, (f32, f32)> = HashMap::new();

        // Get ANN candidates
        if let Some(ann_index) = &self.ann_index {
            // Request more candidates for better coverage before filtering
            let ann_candidates = ann_index.search(query_embedding, max_candidates * 3);
            for chunk_id in ann_candidates {
                scores.insert(chunk_id, (1.0, 0.0)); // Placeholder embedding score
            }
        }

        // Get lexical candidates
        let lexical_results = self.lexical_index.score(query_text, max_candidates * 3);
        let max_lexical = lexical_results
            .iter()
            .map(|(_, s)| *s)
            .fold(0.0_f32, f32::max);

        for (chunk_id, score) in lexical_results {
            let normalized = if max_lexical > 0.0 {
                score / max_lexical
            } else {
                0.0
            };
            scores
                .entry(chunk_id)
                .and_modify(|(_, lex)| *lex = normalized)
                .or_insert((0.0, normalized));
        }

        // Build candidate list with combined scores
        let mut candidates: Vec<CandidateScore> = scores
            .into_iter()
            .map(|(chunk_id, (emb, lex))| {
                let combined = embedding_weight * emb + lexical_weight * lex;
                CandidateScore {
                    chunk_id,
                    embedding_score: emb,
                    lexical_score: lex,
                    combined_score: combined,
                }
            })
            .collect();

        // Sort by combined score descending, with deterministic tie-breaker
        candidates.sort_by(|a, b| {
            b.combined_score
                .partial_cmp(&a.combined_score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.chunk_id.cmp(&b.chunk_id))
        });

        candidates.truncate(max_candidates);
        candidates
    }

    /// Validate that both indexes have identical chunk sets.
    ///
    /// This is a consistency check that should pass under normal operation.
    /// Checks both directions: registry → indexes AND indexes → registry.
    pub fn validate_sync(&self) -> Result<(), EngineError> {
        // Forward check: all chunk_ids must be in lexical index
        for id in &self.chunk_ids {
            if !self.lexical_index.contains(id) {
                return Err(EngineError::IndexSync {
                    message: format!("Chunk '{id}' missing from lexical index"),
                });
            }
        }

        // Reverse check: lexical index shouldn't have extra IDs
        for id in self.lexical_index.doc_terms.keys() {
            if !self.chunk_ids.contains(id) {
                return Err(EngineError::IndexSync {
                    message: format!("Lexical index contains unknown chunk '{id}'"),
                });
            }
        }

        // Check ANN index if it exists
        if let Some(ann_index) = &self.ann_index {
            // Forward check: all chunk_ids must be in ANN index
            for id in &self.chunk_ids {
                if !ann_index.contains(id) {
                    return Err(EngineError::IndexSync {
                        message: format!("Chunk '{id}' missing from ANN index"),
                    });
                }
            }

            // Reverse check: ANN index shouldn't have extra IDs
            for id in ann_index.id_to_bucket.keys() {
                if !self.chunk_ids.contains(id) {
                    return Err(EngineError::IndexSync {
                        message: format!("ANN index contains unknown chunk '{id}'"),
                    });
                }
            }
        }

        Ok(())
    }

    /// Check if a chunk exists in the index.
    pub fn contains(&self, id: &str) -> bool {
        self.chunk_ids.contains(id)
    }

    /// Get the number of indexed chunks.
    pub fn chunk_count(&self) -> usize {
        self.chunk_ids.len()
    }

    /// Get the embedding dimension, if known.
    pub fn embedding_dim(&self) -> Option<usize> {
        self.embedding_dim
    }

    /// Clear all indexes.
    pub fn clear(&mut self) {
        self.ann_index = self.embedding_dim.map(AnnIndex::new);
        self.lexical_index = LexicalIndex::new();
        self.chunk_ids.clear();
    }

    /// Remove chunks not in the valid set.
    pub fn drop_stale(&mut self, valid_ids: &HashSet<String>) {
        let stale: Vec<String> = self
            .chunk_ids
            .iter()
            .filter(|id| !valid_ids.contains(*id))
            .cloned()
            .collect();

        for id in stale {
            self.remove_chunk_internal(&id);
        }
    }

    // Internal helpers

    /// Pure validation of embedding - does not mutate self.
    /// Updates expected_dim in-place if it was None and embedding is valid.
    fn validate_embedding_pure(
        id: &str,
        embedding: &[f32],
        expected_dim: &mut Option<usize>,
    ) -> Result<(), EngineError> {
        let dim = embedding.len();

        // Reject empty embeddings
        if dim == 0 {
            return Err(EngineError::validation(
                id,
                ValidationKind::DimensionMismatch {
                    expected: 1, // At least 1
                    got: 0,
                },
            ));
        }

        // Check for NaN/Inf
        for val in embedding {
            if val.is_nan() {
                return Err(EngineError::validation(id, ValidationKind::NaN));
            }
            if val.is_infinite() {
                return Err(EngineError::validation(id, ValidationKind::Inf));
            }
        }

        // Check/set dimension
        match *expected_dim {
            Some(expected) if expected != dim => {
                return Err(EngineError::validation(
                    id,
                    ValidationKind::DimensionMismatch { expected, got: dim },
                ));
            }
            None => {
                // Set expected dimension for this batch
                *expected_dim = Some(dim);
            }
            _ => {}
        }

        Ok(())
    }

    fn add_chunk_internal(&mut self, id: &str, text: &str, embedding: &[f32]) {
        // Remove existing if present (for upsert behavior)
        if self.chunk_ids.contains(id) {
            self.remove_chunk_internal(id);
        }

        // Add to both indexes
        if let Some(ann_index) = &mut self.ann_index {
            ann_index.insert(id, embedding);
        }
        self.lexical_index.add_chunk(id, text);
        self.chunk_ids.insert(id.to_string());
    }

    fn remove_chunk_internal(&mut self, id: &str) {
        if let Some(ann_index) = &mut self.ann_index {
            ann_index.remove(id);
        }
        self.lexical_index.remove_chunk(id);
        self.chunk_ids.remove(id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lexical_index_contains_and_drop_stale() {
        let mut index = LexicalIndex::new();

        index.add_chunk("chunk1", "hello world");
        index.add_chunk("chunk2", "foo bar baz");
        index.add_chunk("chunk3", "test document");

        assert!(index.contains("chunk1"));
        assert!(index.contains("chunk2"));
        assert!(index.contains("chunk3"));
        assert!(!index.contains("chunk4"));

        let valid_ids: HashSet<String> = vec!["chunk1".to_string(), "chunk2".to_string()]
            .into_iter()
            .collect();

        index.drop_stale(&valid_ids);

        assert!(index.contains("chunk1"));
        assert!(index.contains("chunk2"));
        assert!(
            !index.contains("chunk3"),
            "chunk3 should have been removed as stale"
        );
    }

    #[test]
    fn test_ann_index_contains_and_drop_stale() {
        let mut ann_index = AnnIndex::new(384);

        let vec1: Vec<f32> = (0..384).map(|i| (i as f32) / 384.0).collect();
        let vec2: Vec<f32> = (0..384).map(|i| ((i + 100) as f32) / 384.0).collect();
        let vec3: Vec<f32> = (0..384).map(|i| ((i + 200) as f32) / 384.0).collect();

        ann_index.insert("id1", &vec1);
        ann_index.insert("id2", &vec2);
        ann_index.insert("id3", &vec3);

        assert!(ann_index.contains("id1"));
        assert!(ann_index.contains("id2"));
        assert!(ann_index.contains("id3"));
        assert!(!ann_index.contains("id4"));

        let valid_ids: HashSet<String> = vec!["id1".to_string(), "id3".to_string()]
            .into_iter()
            .collect();

        ann_index.drop_stale(&valid_ids);

        assert!(ann_index.contains("id1"));
        assert!(!ann_index.contains("id2"), "id2 should have been removed");
        assert!(ann_index.contains("id3"));
    }

    fn build_candidate(id: &str, score: f32, embedding: Vec<f32>) -> SearchResultWithEmbedding {
        SearchResultWithEmbedding {
            result: SearchResult {
                text: format!("text-{id}"),
                score,
                document: "doc".to_string(),
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

    #[test]
    fn test_mmr_diversify_empty_candidates() {
        let result = mmr_diversify(vec![], 5, 0.3);
        assert!(result.is_empty());
    }

    #[test]
    fn test_mmr_diversify_single_candidate() {
        let candidates = vec![build_candidate("a", 1.0, vec![1.0, 0.0])];
        let result = mmr_diversify(candidates, 5, 0.3);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].chunk_id, "a");
    }

    #[test]
    fn test_mmr_diversify_top_k_larger_than_candidates() {
        let candidates = vec![
            build_candidate("a", 1.0, vec![1.0, 0.0]),
            build_candidate("b", 0.9, vec![0.0, 1.0]),
        ];
        let result = mmr_diversify(candidates, 10, 0.3);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_mmr_diversify_zero_diversity_factor() {
        let candidates = vec![
            build_candidate("a", 1.0, vec![1.0, 0.0]),
            build_candidate("b", 0.9, vec![1.0, 0.0]),
            build_candidate("c", 0.8, vec![1.0, 0.0]),
        ];
        let result = mmr_diversify(candidates, 3, 0.0);
        let ids: Vec<String> = result.into_iter().map(|r| r.chunk_id).collect();
        assert_eq!(ids, vec!["a".to_string(), "b".to_string(), "c".to_string()]);
    }

    #[test]
    fn test_mmr_diversify_high_diversity_factor() {
        let candidates = vec![
            build_candidate("a", 1.0, vec![1.0, 0.0]),
            build_candidate("b", 0.9, vec![1.0, 0.0]),
            build_candidate("c", 0.8, vec![0.0, 1.0]),
        ];
        let result = mmr_diversify(candidates, 2, 0.9);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].chunk_id, "a");
        assert_eq!(result[1].chunk_id, "c");
    }

    #[test]
    fn test_mmr_diversify_nan_score_handling() {
        let candidates = vec![
            build_candidate("a", 1.0, vec![1.0, 0.0]),
            build_candidate("b", f32::NAN, vec![0.0, 1.0]),
            build_candidate("c", 0.8, vec![0.0, 1.0]),
        ];
        let result = mmr_diversify(candidates, 3, 0.3);
        let ids: Vec<String> = result.into_iter().map(|r| r.chunk_id).collect();
        assert!(!ids.contains(&"b".to_string()));
        assert!(ids.contains(&"a".to_string()));
        assert!(ids.contains(&"c".to_string()));
    }

    #[test]
    fn test_mmr_diversify_inf_score_handling() {
        let candidates = vec![
            build_candidate("a", 1.0, vec![1.0, 0.0]),
            build_candidate("b", f32::INFINITY, vec![0.0, 1.0]),
            build_candidate("c", 0.8, vec![0.0, 1.0]),
        ];
        let result = mmr_diversify(candidates, 3, 0.3);
        let ids: Vec<String> = result.into_iter().map(|r| r.chunk_id).collect();
        assert!(!ids.contains(&"b".to_string()));
        assert!(ids.contains(&"a".to_string()));
        assert!(ids.contains(&"c".to_string()));
    }

    #[test]
    fn test_mmr_diversify_preserves_relevance_order_when_orthogonal() {
        let candidates = vec![
            build_candidate("a", 1.0, vec![1.0, 0.0]),
            build_candidate("b", 0.9, vec![0.0, 1.0]),
            build_candidate("c", 0.8, vec![-1.0, 0.0]),
            build_candidate("d", 0.7, vec![0.0, -1.0]),
        ];
        let result = mmr_diversify(candidates, 4, 0.3);
        let ids: Vec<String> = result.into_iter().map(|r| r.chunk_id).collect();
        assert_eq!(
            ids,
            vec![
                "a".to_string(),
                "b".to_string(),
                "c".to_string(),
                "d".to_string()
            ]
        );
    }

    #[test]
    fn test_mmr_formula_correctness() {
        let candidates = vec![
            build_candidate("a", 1.0, vec![1.0, 0.0]),
            build_candidate("b", 0.8, vec![1.0, 0.0]),
            build_candidate("c", 0.7, vec![0.0, 1.0]),
        ];
        let result = mmr_diversify(candidates, 2, 0.5);
        let ids: Vec<String> = result.into_iter().map(|r| r.chunk_id).collect();
        assert_eq!(ids, vec!["a".to_string(), "c".to_string()]);
    }

    // ============================================================================
    // IndexSet tests
    // ============================================================================

    #[test]
    fn test_index_set_new_empty() {
        let index = IndexSet::new();
        assert_eq!(index.chunk_count(), 0);
        assert!(index.embedding_dim().is_none());
    }

    #[test]
    fn test_index_set_with_dimension() {
        let index = IndexSet::with_dimension(384);
        assert_eq!(index.embedding_dim(), Some(384));
        assert_eq!(index.chunk_count(), 0);
    }

    #[test]
    fn test_index_set_add_chunk() {
        let mut index = IndexSet::new();
        let embedding: Vec<f32> = (0..384).map(|i| i as f32 / 384.0).collect();

        index
            .add_chunk("chunk1", "hello world test", &embedding)
            .unwrap();

        assert!(index.contains("chunk1"));
        assert_eq!(index.chunk_count(), 1);
        assert_eq!(index.embedding_dim(), Some(384));
        assert!(index.validate_sync().is_ok());
    }

    #[test]
    fn test_index_set_remove_chunk() {
        let mut index = IndexSet::new();
        let embedding: Vec<f32> = (0..384).map(|i| i as f32 / 384.0).collect();

        index
            .add_chunk("chunk1", "hello world test", &embedding)
            .unwrap();
        index.remove_chunk("chunk1");

        assert!(!index.contains("chunk1"));
        assert_eq!(index.chunk_count(), 0);
        assert!(index.validate_sync().is_ok());
    }

    #[test]
    fn test_index_set_apply_batch() {
        let mut index = IndexSet::new();
        let embedding1: Vec<f32> = (0..384).map(|i| i as f32 / 384.0).collect();
        let embedding2: Vec<f32> = (0..384).map(|i| (i + 100) as f32 / 384.0).collect();

        let updates = vec![
            ChunkUpdate::Add {
                id: "chunk1".to_string(),
                text: "hello world".to_string(),
                embedding: embedding1,
            },
            ChunkUpdate::Add {
                id: "chunk2".to_string(),
                text: "foo bar baz".to_string(),
                embedding: embedding2,
            },
        ];

        index.apply_batch(&updates).unwrap();

        assert!(index.contains("chunk1"));
        assert!(index.contains("chunk2"));
        assert_eq!(index.chunk_count(), 2);
        assert!(index.validate_sync().is_ok());
    }

    #[test]
    fn test_index_set_rejects_nan_embedding() {
        let mut index = IndexSet::new();
        let mut embedding: Vec<f32> = (0..384).map(|i| i as f32 / 384.0).collect();
        embedding[10] = f32::NAN;

        let result = index.add_chunk("chunk1", "hello world", &embedding);

        assert!(result.is_err());
        let err_str = result.unwrap_err().to_string();
        assert!(err_str.contains("NaN"));
    }

    #[test]
    fn test_index_set_rejects_inf_embedding() {
        let mut index = IndexSet::new();
        let mut embedding: Vec<f32> = (0..384).map(|i| i as f32 / 384.0).collect();
        embedding[10] = f32::INFINITY;

        let result = index.add_chunk("chunk1", "hello world", &embedding);

        assert!(result.is_err());
        let err_str = result.unwrap_err().to_string();
        assert!(err_str.contains("Inf"));
    }

    #[test]
    fn test_index_set_rejects_dimension_mismatch() {
        let mut index = IndexSet::new();
        let embedding1: Vec<f32> = (0..384).map(|i| i as f32 / 384.0).collect();
        let embedding2: Vec<f32> = (0..768).map(|i| i as f32 / 768.0).collect(); // Wrong dim

        index
            .add_chunk("chunk1", "hello world", &embedding1)
            .unwrap();
        let result = index.add_chunk("chunk2", "foo bar", &embedding2);

        assert!(result.is_err());
        let err_str = result.unwrap_err().to_string();
        assert!(err_str.contains("dimension"));
        assert!(err_str.contains("384"));
        assert!(err_str.contains("768"));
    }

    #[test]
    fn test_index_set_rejects_empty_text() {
        let mut index = IndexSet::new();
        let embedding: Vec<f32> = (0..384).map(|i| i as f32 / 384.0).collect();

        let result = index.add_chunk("chunk1", "   ", &embedding);

        assert!(result.is_err());
        let err_str = result.unwrap_err().to_string();
        assert!(err_str.contains("empty"));
    }

    #[test]
    fn test_index_set_batch_atomic_on_validation_failure() {
        let mut index = IndexSet::new();
        let embedding1: Vec<f32> = (0..384).map(|i| i as f32 / 384.0).collect();
        let mut embedding2: Vec<f32> = (0..384).map(|i| i as f32 / 384.0).collect();
        embedding2[0] = f32::NAN; // Invalid

        let updates = vec![
            ChunkUpdate::Add {
                id: "chunk1".to_string(),
                text: "hello world".to_string(),
                embedding: embedding1,
            },
            ChunkUpdate::Add {
                id: "chunk2".to_string(),
                text: "foo bar".to_string(),
                embedding: embedding2,
            },
        ];

        let result = index.apply_batch(&updates);

        // Batch should fail and no chunks should be added
        assert!(result.is_err());
        assert!(
            !index.contains("chunk1"),
            "chunk1 should not be added on batch failure"
        );
        assert!(!index.contains("chunk2"));
        assert_eq!(index.chunk_count(), 0);
    }

    #[test]
    fn test_index_set_search_candidates() {
        let mut index = IndexSet::new();

        // Add chunks with distinct embeddings
        let embedding1: Vec<f32> = (0..384).map(|i| i as f32 / 384.0).collect();
        let embedding2: Vec<f32> = (0..384).map(|i| (384 - i) as f32 / 384.0).collect();

        index
            .add_chunk("chunk1", "machine learning algorithms", &embedding1)
            .unwrap();
        index
            .add_chunk("chunk2", "deep learning neural networks", &embedding2)
            .unwrap();

        let query_embedding: Vec<f32> = (0..384).map(|i| i as f32 / 384.0).collect();
        let candidates =
            index.search_candidates(&query_embedding, "machine learning", 0.7, 0.3, 10);

        assert!(!candidates.is_empty());
        // Results should be sorted by combined score
        for i in 1..candidates.len() {
            assert!(candidates[i - 1].combined_score >= candidates[i].combined_score);
        }
    }

    #[test]
    fn test_index_set_drop_stale() {
        let mut index = IndexSet::new();
        let embedding: Vec<f32> = (0..384).map(|i| i as f32 / 384.0).collect();

        index
            .add_chunk("chunk1", "hello world", &embedding)
            .unwrap();
        index
            .add_chunk("chunk2", "foo bar baz", &embedding)
            .unwrap();
        index
            .add_chunk("chunk3", "test document", &embedding)
            .unwrap();

        let valid_ids: HashSet<String> =
            ["chunk1", "chunk2"].iter().map(|s| s.to_string()).collect();
        index.drop_stale(&valid_ids);

        assert!(index.contains("chunk1"));
        assert!(index.contains("chunk2"));
        assert!(!index.contains("chunk3"));
        assert_eq!(index.chunk_count(), 2);
        assert!(index.validate_sync().is_ok());
    }

    #[test]
    fn test_index_set_clear() {
        let mut index = IndexSet::new();
        let embedding: Vec<f32> = (0..384).map(|i| i as f32 / 384.0).collect();

        index
            .add_chunk("chunk1", "hello world", &embedding)
            .unwrap();
        index.add_chunk("chunk2", "foo bar", &embedding).unwrap();

        index.clear();

        assert_eq!(index.chunk_count(), 0);
        assert!(!index.contains("chunk1"));
        // Dimension should be preserved after clear
        assert_eq!(index.embedding_dim(), Some(384));
    }

    #[test]
    fn test_index_set_upsert_behavior() {
        let mut index = IndexSet::new();
        let embedding1: Vec<f32> = (0..384).map(|i| i as f32 / 384.0).collect();
        let embedding2: Vec<f32> = (0..384).map(|i| (i + 100) as f32 / 384.0).collect();

        index
            .add_chunk("chunk1", "original text", &embedding1)
            .unwrap();
        index
            .add_chunk("chunk1", "updated text", &embedding2)
            .unwrap(); // Same ID

        assert!(index.contains("chunk1"));
        assert_eq!(index.chunk_count(), 1);
        assert!(index.validate_sync().is_ok());
    }

    #[test]
    fn test_index_set_deterministic_ordering() {
        let mut index = IndexSet::new();
        let embedding: Vec<f32> = (0..384).map(|i| i as f32 / 384.0).collect();

        // Add chunks in random order
        index.add_chunk("c", "test document", &embedding).unwrap();
        index.add_chunk("a", "test document", &embedding).unwrap();
        index.add_chunk("b", "test document", &embedding).unwrap();

        let candidates = index.search_candidates(&embedding, "test document", 0.5, 0.5, 10);

        // With equal scores, should be sorted alphabetically by chunk_id
        let ids: Vec<&str> = candidates.iter().map(|c| c.chunk_id.as_str()).collect();
        assert_eq!(ids, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_lexical_index_stability() {
        let mut index = LexicalIndex::new();

        // Add multiple chunks with identical content (same score)
        let chunks = vec!["a", "b", "c", "d", "e", "f", "g", "h"];
        for id in &chunks {
            index.add_chunk(id, "common text content");
        }

        // Search with limit < total chunks
        // With deterministic tie-breaking (ID asc), we should get the first 3 alphabetically
        let limit = 3;
        let results = index.score("common", limit);

        assert_eq!(results.len(), limit);

        let ids: Vec<String> = results.into_iter().map(|(id, _)| id).collect();
        assert_eq!(ids, vec!["a".to_string(), "b".to_string(), "c".to_string()]);
    }

    #[test]
    fn test_ann_index_deterministic_fallback() {
        // This test verifies that the fallback search (scanning all buckets)
        // returns candidates in a deterministic order.

        // We run the experiment multiple times with new indices to ensure
        // we are not just getting lucky with the same HashMap seed.
        // (Though SimpleRng for hyperplanes is seeded constantly).

        let query = vec![0.5, 0.5, 0.5, 0.5];
        let mut first_results = None;

        for _ in 0..5 {
            let mut index = AnnIndex::new(4);
            // Insert enough items to create multiple buckets
            for i in 0..100 {
                let v = vec![
                    (i % 2) as f32,
                    ((i / 2) % 2) as f32,
                    ((i / 4) % 2) as f32,
                    ((i / 8) % 2) as f32,
                ];
                index.insert(&format!("id-{}", i), &v);
            }

            // Limit = 20 forces us to pick a subset of buckets in the fallback loop
            let results = index.search(&query, 20);

            if let Some(ref first) = first_results {
                assert_eq!(
                    first, &results,
                    "Results should be deterministic across runs"
                );
            } else {
                first_results = Some(results);
            }
        }
    }
}
