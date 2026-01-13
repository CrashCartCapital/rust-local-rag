//! Tag management and indexing for document organization.
//!
//! This module provides:
//! - Hierarchical tag explosion (e.g., "a/b/c" → ["a", "a/b", "a/b/c"])
//! - Inverted index for fast tag-based lookup
//! - Filter expression evaluation against the tag index

use crate::types::FilterExpr;
use std::collections::{HashMap, HashSet};

/// Type alias for tag strings.
pub type Tag = String;

/// Type alias for chunk IDs.
pub type ChunkId = String;

/// Explode a hierarchical tag into all its prefixes.
///
/// For example, "finance/invoice/2024" becomes:
/// - "finance"
/// - "finance/invoice"
/// - "finance/invoice/2024"
///
/// # Examples
///
/// ```rust
/// use rag_core::explode_tag;
///
/// let tags = explode_tag("a/b/c");
/// assert_eq!(tags, vec!["a", "a/b", "a/b/c"]);
/// ```
pub fn explode_tag(tag: &str) -> Vec<String> {
    let mut result = Vec::new();
    let mut current = String::new();

    for part in tag.split('/') {
        if !current.is_empty() {
            current.push('/');
        }
        current.push_str(part);
        result.push(current.clone());
    }

    result
}

/// Explode multiple tags, returning all unique prefixes.
pub fn explode_tags(tags: &[String]) -> HashSet<String> {
    tags.iter().flat_map(|t| explode_tag(t)).collect()
}

/// Inverted index mapping tags to chunk IDs.
///
/// Provides O(1) lookup for chunks matching a specific tag and efficient
/// set operations for filter expression evaluation.
#[derive(Debug, Clone, Default)]
pub struct TagIndex {
    /// Maps tag -> set of chunk IDs that have this tag
    tag_to_chunks: HashMap<Tag, HashSet<ChunkId>>,
    /// Maps chunk ID -> set of tags for that chunk (for removal)
    chunk_to_tags: HashMap<ChunkId, HashSet<Tag>>,
}

impl TagIndex {
    /// Create a new empty tag index.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert tags for a chunk. Tags are stored as-is (explosion should happen before calling).
    pub fn insert(&mut self, chunk_id: &str, tags: impl IntoIterator<Item = impl AsRef<str>>) {
        let tags: HashSet<String> = tags.into_iter().map(|t| t.as_ref().to_string()).collect();

        // Update chunk_to_tags
        self.chunk_to_tags
            .insert(chunk_id.to_string(), tags.clone());

        // Update tag_to_chunks
        for tag in tags {
            self.tag_to_chunks
                .entry(tag)
                .or_default()
                .insert(chunk_id.to_string());
        }
    }

    /// Remove a chunk from the index.
    pub fn remove_chunk(&mut self, chunk_id: &str) {
        if let Some(tags) = self.chunk_to_tags.remove(chunk_id) {
            for tag in tags {
                if let Some(chunks) = self.tag_to_chunks.get_mut(&tag) {
                    chunks.remove(chunk_id);
                    // Clean up empty tag entries
                    if chunks.is_empty() {
                        self.tag_to_chunks.remove(&tag);
                    }
                }
            }
        }
    }

    /// Get all chunk IDs that have a specific tag.
    pub fn chunks_with_tag(&self, tag: &str) -> HashSet<ChunkId> {
        self.tag_to_chunks.get(tag).cloned().unwrap_or_default()
    }

    /// Get all chunk IDs that have tags matching a prefix.
    pub fn chunks_with_prefix(&self, prefix: &str) -> HashSet<ChunkId> {
        let mut result = HashSet::new();
        for (tag, chunks) in &self.tag_to_chunks {
            if tag.starts_with(prefix) || tag == prefix.trim_end_matches('/') {
                result.extend(chunks.iter().cloned());
            }
        }
        result
    }

    /// Get all unique tags in the index.
    pub fn all_tags(&self) -> impl Iterator<Item = &Tag> {
        self.tag_to_chunks.keys()
    }

    /// Get the number of unique tags.
    pub fn tag_count(&self) -> usize {
        self.tag_to_chunks.len()
    }

    /// Get the number of indexed chunks.
    pub fn chunk_count(&self) -> usize {
        self.chunk_to_tags.len()
    }

    /// Check if a chunk has a specific tag.
    pub fn chunk_has_tag(&self, chunk_id: &str, tag: &str) -> bool {
        self.chunk_to_tags
            .get(chunk_id)
            .map(|tags| tags.contains(tag))
            .unwrap_or(false)
    }

    /// Get all tags for a specific chunk.
    pub fn tags_for_chunk(&self, chunk_id: &str) -> HashSet<Tag> {
        self.chunk_to_tags
            .get(chunk_id)
            .cloned()
            .unwrap_or_default()
    }

    /// Evaluate a filter expression and return matching chunk IDs.
    ///
    /// This is the core method for filter-based search. It returns the set
    /// of chunk IDs that match the given filter expression.
    ///
    /// For DocumentNameEquals filters, this returns an empty set (handled separately).
    pub fn evaluate(&self, filter: &FilterExpr, all_chunks: &HashSet<ChunkId>) -> HashSet<ChunkId> {
        match filter {
            FilterExpr::DocumentNameEquals(_) => {
                // Document name filtering is handled at a different level
                // Return all chunks (no filtering at tag level)
                all_chunks.clone()
            }
            FilterExpr::Tag(tag) => self.chunks_with_tag(tag),
            FilterExpr::Prefix(prefix) => self.chunks_with_prefix(prefix),
            FilterExpr::And(exprs) => {
                if exprs.is_empty() {
                    return all_chunks.clone();
                }
                let mut result = self.evaluate(&exprs[0], all_chunks);
                for expr in &exprs[1..] {
                    let other = self.evaluate(expr, all_chunks);
                    result = result.intersection(&other).cloned().collect();
                }
                result
            }
            FilterExpr::Or(exprs) => {
                let mut result = HashSet::new();
                for expr in exprs {
                    result.extend(self.evaluate(expr, all_chunks));
                }
                result
            }
            FilterExpr::Not(expr) => {
                let excluded = self.evaluate(expr, all_chunks);
                all_chunks.difference(&excluded).cloned().collect()
            }
        }
    }
}

// ============================================================================
// Phase 3: Semantic Tag Expansion
// ============================================================================

/// Mode for expanding tags based on semantic similarity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExpansionMode {
    /// Add similar tags as score boosts (safer, preserves recall)
    #[default]
    SoftBoost,
    /// Expand filter expression with OR clauses (more aggressive)
    HardFilter,
}

/// Configuration for semantic tag expansion.
#[derive(Debug, Clone, PartialEq)]
pub struct TagExpansionConfig {
    /// Whether semantic expansion is enabled
    pub enabled: bool,
    /// Minimum cosine similarity threshold for expansion (0.0-1.0)
    pub threshold: f32,
    /// Maximum number of tags to expand to
    pub max_expanded_tags: usize,
    /// Expansion strategy
    pub mode: ExpansionMode,
}

impl Default for TagExpansionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            threshold: 0.8,
            max_expanded_tags: 5,
            mode: ExpansionMode::default(),
        }
    }
}

impl TagExpansionConfig {
    /// Create a new enabled config with default settings.
    pub fn enabled() -> Self {
        Self {
            enabled: true,
            ..Default::default()
        }
    }

    /// Set the similarity threshold.
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set the maximum expanded tags.
    pub fn with_max_tags(mut self, max: usize) -> Self {
        self.max_expanded_tags = max;
        self
    }

    /// Set the expansion mode.
    pub fn with_mode(mut self, mode: ExpansionMode) -> Self {
        self.mode = mode;
        self
    }
}

/// A tag found through semantic similarity search.
#[derive(Debug, Clone, PartialEq)]
pub struct ExpandedTag {
    /// The semantically similar tag
    pub tag: String,
    /// Cosine similarity score (0.0-1.0)
    pub score: f32,
}

impl ExpandedTag {
    /// Create a new expanded tag.
    pub fn new(tag: impl Into<String>, score: f32) -> Self {
        Self {
            tag: tag.into(),
            score,
        }
    }
}

/// Index for storing tag embeddings and performing similarity search.
///
/// This is separate from TagIndex to allow optional semantic expansion
/// without affecting the core tag->chunk mapping.
#[derive(Debug, Clone, Default)]
pub struct TagEmbeddingIndex {
    /// Maps tag -> embedding vector
    embeddings: HashMap<Tag, Vec<f32>>,
    /// Embedding dimension (set on first insert)
    dimension: Option<usize>,
}

impl TagEmbeddingIndex {
    /// Create a new empty embedding index.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert or update an embedding for a tag.
    ///
    /// Returns an error if the dimension doesn't match existing embeddings.
    pub fn insert(&mut self, tag: impl Into<String>, embedding: Vec<f32>) -> Result<(), String> {
        if embedding.is_empty() {
            return Err("Embedding cannot be empty".into());
        }

        match self.dimension {
            Some(dim) if dim != embedding.len() => {
                return Err(format!(
                    "Dimension mismatch: expected {}, got {}",
                    dim,
                    embedding.len()
                ));
            }
            None => {
                self.dimension = Some(embedding.len());
            }
            _ => {}
        }

        self.embeddings.insert(tag.into(), embedding);
        Ok(())
    }

    /// Get the embedding for a tag.
    pub fn get(&self, tag: &str) -> Option<&Vec<f32>> {
        self.embeddings.get(tag)
    }

    /// Check if a tag has an embedding.
    pub fn contains(&self, tag: &str) -> bool {
        self.embeddings.contains_key(tag)
    }

    /// Get all tags with embeddings.
    pub fn tags(&self) -> impl Iterator<Item = &Tag> {
        self.embeddings.keys()
    }

    /// Number of embedded tags.
    pub fn len(&self) -> usize {
        self.embeddings.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.embeddings.is_empty()
    }

    /// Get the embedding dimension.
    pub fn dimension(&self) -> Option<usize> {
        self.dimension
    }

    /// Find tags similar to a query embedding.
    ///
    /// Returns tags with cosine similarity >= threshold, sorted by score descending,
    /// limited to max_results.
    pub fn find_similar(
        &self,
        query_embedding: &[f32],
        threshold: f32,
        max_results: usize,
    ) -> Vec<ExpandedTag> {
        if query_embedding.is_empty() || self.is_empty() {
            return Vec::new();
        }

        // Check dimension match
        if let Some(dim) = self.dimension {
            if dim != query_embedding.len() {
                return Vec::new();
            }
        }

        let mut results: Vec<ExpandedTag> = self
            .embeddings
            .iter()
            .filter_map(|(tag, embedding)| {
                let score = cosine_similarity(query_embedding, embedding);
                if score >= threshold && score.is_finite() {
                    Some(ExpandedTag::new(tag.clone(), score))
                } else {
                    None
                }
            })
            .collect();

        // Sort by score descending (using total_cmp for deterministic float ordering)
        results.sort_by(|a, b| b.score.total_cmp(&a.score));

        // Limit results
        results.truncate(max_results);
        results
    }

    /// Remove a tag's embedding.
    ///
    /// If this removes the last embedding, dimension is reset to allow
    /// a different dimension on subsequent inserts.
    pub fn remove(&mut self, tag: &str) -> Option<Vec<f32>> {
        let result = self.embeddings.remove(tag);
        if self.embeddings.is_empty() {
            self.dimension = None;
        }
        result
    }

    /// Clear all embeddings.
    pub fn clear(&mut self) {
        self.embeddings.clear();
        self.dimension = None;
    }
}

/// Compute cosine similarity between two vectors.
///
/// Returns a value in [-1.0, 1.0] where 1 means identical direction.
/// Returns 0.0 for edge cases (empty, mismatched length, near-zero norm).
///
/// This is the canonical cosine similarity implementation for rag-core.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    const EPSILON: f32 = 1e-10;

    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }

    let norm_a = norm_a.sqrt();
    let norm_b = norm_b.sqrt();

    if norm_a < EPSILON || norm_b < EPSILON {
        0.0
    } else {
        (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // Phase 2 TDD Tests: Tag Explosion
    // ========================================================================

    #[test]
    fn test_explode_tag_simple() {
        let tags = explode_tag("research");
        assert_eq!(tags, vec!["research"]);
    }

    #[test]
    fn test_explode_tag_hierarchical() {
        let tags = explode_tag("finance/invoice/2024");
        assert_eq!(
            tags,
            vec!["finance", "finance/invoice", "finance/invoice/2024"]
        );
    }

    #[test]
    fn test_explode_tag_two_levels() {
        let tags = explode_tag("ml/nlp");
        assert_eq!(tags, vec!["ml", "ml/nlp"]);
    }

    #[test]
    fn test_explode_tags_multiple() {
        let tags = explode_tags(&["ml/nlp".into(), "research".into()]);
        assert!(tags.contains("ml"));
        assert!(tags.contains("ml/nlp"));
        assert!(tags.contains("research"));
        assert_eq!(tags.len(), 3);
    }

    // ========================================================================
    // Phase 2 TDD Tests: TagIndex Basic Operations
    // ========================================================================

    #[test]
    fn test_tag_index_new_empty() {
        let index = TagIndex::new();
        assert_eq!(index.tag_count(), 0);
        assert_eq!(index.chunk_count(), 0);
    }

    #[test]
    fn test_tag_index_insert() {
        let mut index = TagIndex::new();
        index.insert("chunk-1", vec!["research", "ml"]);

        assert_eq!(index.chunk_count(), 1);
        assert_eq!(index.tag_count(), 2);
        assert!(index.chunk_has_tag("chunk-1", "research"));
        assert!(index.chunk_has_tag("chunk-1", "ml"));
    }

    #[test]
    fn test_tag_index_chunks_with_tag() {
        let mut index = TagIndex::new();
        index.insert("chunk-1", vec!["research"]);
        index.insert("chunk-2", vec!["research", "ml"]);
        index.insert("chunk-3", vec!["ml"]);

        let research_chunks = index.chunks_with_tag("research");
        assert_eq!(research_chunks.len(), 2);
        assert!(research_chunks.contains("chunk-1"));
        assert!(research_chunks.contains("chunk-2"));
    }

    #[test]
    fn test_tag_index_remove_chunk() {
        let mut index = TagIndex::new();
        index.insert("chunk-1", vec!["research", "ml"]);
        index.insert("chunk-2", vec!["research"]);

        index.remove_chunk("chunk-1");

        assert_eq!(index.chunk_count(), 1);
        assert!(!index.chunk_has_tag("chunk-1", "research"));
        assert!(index.chunk_has_tag("chunk-2", "research"));
        // "ml" tag should be removed since no chunks have it
        assert_eq!(index.chunks_with_tag("ml").len(), 0);
    }

    #[test]
    fn test_tag_index_tags_for_chunk() {
        let mut index = TagIndex::new();
        index.insert("chunk-1", vec!["research", "ml", "python"]);

        let tags = index.tags_for_chunk("chunk-1");
        assert_eq!(tags.len(), 3);
        assert!(tags.contains("research"));
        assert!(tags.contains("ml"));
        assert!(tags.contains("python"));
    }

    // ========================================================================
    // Phase 2 TDD Tests: Prefix Matching
    // ========================================================================

    #[test]
    fn test_tag_index_prefix_match() {
        let mut index = TagIndex::new();
        index.insert("chunk-1", vec!["ml", "ml/nlp", "ml/nlp/transformers"]);
        index.insert("chunk-2", vec!["ml", "ml/vision"]);
        index.insert("chunk-3", vec!["research"]);

        let ml_chunks = index.chunks_with_prefix("ml/");
        assert_eq!(ml_chunks.len(), 2); // chunk-1 and chunk-2
        assert!(ml_chunks.contains("chunk-1"));
        assert!(ml_chunks.contains("chunk-2"));
    }

    #[test]
    fn test_tag_index_prefix_match_exact() {
        let mut index = TagIndex::new();
        index.insert("chunk-1", vec!["ml"]);

        // "ml/" prefix should also match exact "ml" tag
        let ml_chunks = index.chunks_with_prefix("ml/");
        assert_eq!(ml_chunks.len(), 1);
    }

    // ========================================================================
    // Phase 2 TDD Tests: Filter Expression Evaluation
    // ========================================================================

    #[test]
    fn test_filter_tag_simple() {
        let mut index = TagIndex::new();
        index.insert("chunk-1", vec!["research"]);
        index.insert("chunk-2", vec!["ml"]);

        let all: HashSet<_> = vec!["chunk-1".to_string(), "chunk-2".to_string()]
            .into_iter()
            .collect();

        let result = index.evaluate(&FilterExpr::Tag("research".into()), &all);
        assert_eq!(result.len(), 1);
        assert!(result.contains("chunk-1"));
    }

    #[test]
    fn test_filter_prefix() {
        let mut index = TagIndex::new();
        index.insert("chunk-1", vec!["ml/nlp"]);
        index.insert("chunk-2", vec!["ml/vision"]);
        index.insert("chunk-3", vec!["research"]);

        let all: HashSet<_> = vec![
            "chunk-1".to_string(),
            "chunk-2".to_string(),
            "chunk-3".to_string(),
        ]
        .into_iter()
        .collect();

        let result = index.evaluate(&FilterExpr::Prefix("ml/".into()), &all);
        assert_eq!(result.len(), 2);
        assert!(result.contains("chunk-1"));
        assert!(result.contains("chunk-2"));
    }

    #[test]
    fn test_filter_and() {
        let mut index = TagIndex::new();
        index.insert("chunk-1", vec!["research", "ml"]);
        index.insert("chunk-2", vec!["research"]);
        index.insert("chunk-3", vec!["ml"]);

        let all: HashSet<_> = vec![
            "chunk-1".to_string(),
            "chunk-2".to_string(),
            "chunk-3".to_string(),
        ]
        .into_iter()
        .collect();

        let filter = FilterExpr::And(vec![
            FilterExpr::Tag("research".into()),
            FilterExpr::Tag("ml".into()),
        ]);

        let result = index.evaluate(&filter, &all);
        assert_eq!(result.len(), 1);
        assert!(result.contains("chunk-1"));
    }

    #[test]
    fn test_filter_or() {
        let mut index = TagIndex::new();
        index.insert("chunk-1", vec!["research"]);
        index.insert("chunk-2", vec!["ml"]);
        index.insert("chunk-3", vec!["finance"]);

        let all: HashSet<_> = vec![
            "chunk-1".to_string(),
            "chunk-2".to_string(),
            "chunk-3".to_string(),
        ]
        .into_iter()
        .collect();

        let filter = FilterExpr::Or(vec![
            FilterExpr::Tag("research".into()),
            FilterExpr::Tag("ml".into()),
        ]);

        let result = index.evaluate(&filter, &all);
        assert_eq!(result.len(), 2);
        assert!(result.contains("chunk-1"));
        assert!(result.contains("chunk-2"));
    }

    #[test]
    fn test_filter_not() {
        let mut index = TagIndex::new();
        index.insert("chunk-1", vec!["research"]);
        index.insert("chunk-2", vec!["research", "outdated"]);
        index.insert("chunk-3", vec!["ml"]);

        let all: HashSet<_> = vec![
            "chunk-1".to_string(),
            "chunk-2".to_string(),
            "chunk-3".to_string(),
        ]
        .into_iter()
        .collect();

        // research AND NOT outdated
        let filter = FilterExpr::And(vec![
            FilterExpr::Tag("research".into()),
            FilterExpr::Not(Box::new(FilterExpr::Tag("outdated".into()))),
        ]);

        let result = index.evaluate(&filter, &all);
        assert_eq!(result.len(), 1);
        assert!(result.contains("chunk-1"));
    }

    #[test]
    fn test_filter_complex_nested() {
        let mut index = TagIndex::new();
        index.insert("chunk-1", vec!["research", "ml"]);
        index.insert("chunk-2", vec!["research", "finance"]);
        index.insert("chunk-3", vec!["ml", "production"]);
        index.insert("chunk-4", vec!["finance", "production"]);

        let all: HashSet<_> = vec![
            "chunk-1".to_string(),
            "chunk-2".to_string(),
            "chunk-3".to_string(),
            "chunk-4".to_string(),
        ]
        .into_iter()
        .collect();

        // (research OR production) AND ml
        let filter = FilterExpr::And(vec![
            FilterExpr::Or(vec![
                FilterExpr::Tag("research".into()),
                FilterExpr::Tag("production".into()),
            ]),
            FilterExpr::Tag("ml".into()),
        ]);

        let result = index.evaluate(&filter, &all);
        assert_eq!(result.len(), 2);
        assert!(result.contains("chunk-1")); // research AND ml
        assert!(result.contains("chunk-3")); // production AND ml
    }

    #[test]
    fn test_filter_empty_and() {
        let index = TagIndex::new();
        let all: HashSet<_> = vec!["chunk-1".to_string()].into_iter().collect();

        // Empty AND should return all chunks
        let filter = FilterExpr::And(vec![]);
        let result = index.evaluate(&filter, &all);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_filter_empty_or() {
        let index = TagIndex::new();
        let all: HashSet<_> = vec!["chunk-1".to_string()].into_iter().collect();

        // Empty OR should return no chunks
        let filter = FilterExpr::Or(vec![]);
        let result = index.evaluate(&filter, &all);
        assert!(result.is_empty());
    }

    #[test]
    fn test_filter_document_name_equals_passthrough() {
        let index = TagIndex::new();
        let all: HashSet<_> = vec!["chunk-1".to_string(), "chunk-2".to_string()]
            .into_iter()
            .collect();

        // DocumentNameEquals should return all (filtered at different level)
        let filter = FilterExpr::DocumentNameEquals("doc.pdf".into());
        let result = index.evaluate(&filter, &all);
        assert_eq!(result, all);
    }

    // ========================================================================
    // Phase 3 TDD Tests: Semantic Tag Expansion
    // ========================================================================

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let score = cosine_similarity(&a, &b);
        assert!((score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let score = cosine_similarity(&a, &b);
        assert!(score.abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![-1.0, 0.0, 0.0];
        let score = cosine_similarity(&a, &b);
        assert!((score - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_different_lengths() {
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let score = cosine_similarity(&a, &b);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let score = cosine_similarity(&a, &b);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_expansion_mode_default() {
        assert_eq!(ExpansionMode::default(), ExpansionMode::SoftBoost);
    }

    #[test]
    fn test_tag_expansion_config_defaults() {
        let config = TagExpansionConfig::default();
        assert!(!config.enabled);
        assert!((config.threshold - 0.8).abs() < f32::EPSILON);
        assert_eq!(config.max_expanded_tags, 5);
        assert_eq!(config.mode, ExpansionMode::SoftBoost);
    }

    #[test]
    fn test_tag_expansion_config_builder() {
        let config = TagExpansionConfig::enabled()
            .with_threshold(0.9)
            .with_max_tags(10)
            .with_mode(ExpansionMode::HardFilter);

        assert!(config.enabled);
        assert!((config.threshold - 0.9).abs() < f32::EPSILON);
        assert_eq!(config.max_expanded_tags, 10);
        assert_eq!(config.mode, ExpansionMode::HardFilter);
    }

    #[test]
    fn test_tag_expansion_config_threshold_clamping() {
        let config = TagExpansionConfig::enabled().with_threshold(1.5);
        assert!((config.threshold - 1.0).abs() < f32::EPSILON);

        let config = TagExpansionConfig::enabled().with_threshold(-0.5);
        assert!(config.threshold.abs() < f32::EPSILON);
    }

    #[test]
    fn test_expanded_tag_creation() {
        let tag = ExpandedTag::new("finance", 0.95);
        assert_eq!(tag.tag, "finance");
        assert!((tag.score - 0.95).abs() < f32::EPSILON);
    }

    #[test]
    fn test_tag_embedding_index_new_empty() {
        let index = TagEmbeddingIndex::new();
        assert!(index.is_empty());
        assert_eq!(index.len(), 0);
        assert!(index.dimension().is_none());
    }

    #[test]
    fn test_tag_embedding_index_insert() {
        let mut index = TagEmbeddingIndex::new();
        let result = index.insert("finance", vec![1.0, 0.0, 0.0]);
        assert!(result.is_ok());
        assert_eq!(index.len(), 1);
        assert!(index.contains("finance"));
        assert_eq!(index.dimension(), Some(3));
    }

    #[test]
    fn test_tag_embedding_index_dimension_mismatch() {
        let mut index = TagEmbeddingIndex::new();
        index.insert("a", vec![1.0, 0.0, 0.0]).unwrap();
        let result = index.insert("b", vec![1.0, 0.0]);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Dimension mismatch"));
    }

    #[test]
    fn test_tag_embedding_index_empty_embedding() {
        let mut index = TagEmbeddingIndex::new();
        let result = index.insert("a", vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_tag_embedding_index_get() {
        let mut index = TagEmbeddingIndex::new();
        index.insert("finance", vec![1.0, 0.0, 0.0]).unwrap();
        let embedding = index.get("finance");
        assert!(embedding.is_some());
        assert_eq!(embedding.unwrap(), &vec![1.0, 0.0, 0.0]);
        assert!(index.get("nonexistent").is_none());
    }

    #[test]
    fn test_tag_embedding_index_find_similar_exact_match() {
        let mut index = TagEmbeddingIndex::new();
        index.insert("finance", vec![1.0, 0.0, 0.0]).unwrap();
        index.insert("legal", vec![0.0, 1.0, 0.0]).unwrap();

        let query = vec![1.0, 0.0, 0.0];
        let results = index.find_similar(&query, 0.9, 10);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].tag, "finance");
        assert!((results[0].score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_tag_embedding_index_find_similar_threshold() {
        let mut index = TagEmbeddingIndex::new();
        // Create similar but not identical vectors
        index.insert("tax", vec![0.9, 0.1, 0.0]).unwrap();
        index.insert("budget", vec![0.8, 0.2, 0.0]).unwrap();
        index.insert("legal", vec![0.0, 0.0, 1.0]).unwrap();

        let query = vec![1.0, 0.0, 0.0];
        let results = index.find_similar(&query, 0.85, 10);

        // "tax" should be above 0.85 threshold, "budget" might be below
        assert!(results.iter().any(|r| r.tag == "tax"));
        assert!(!results.iter().any(|r| r.tag == "legal"));
    }

    #[test]
    fn test_tag_embedding_index_find_similar_max_results() {
        let mut index = TagEmbeddingIndex::new();
        // Create multiple similar vectors
        for i in 0..10 {
            let mut vec = vec![0.0; 3];
            vec[0] = 1.0 - (i as f32 * 0.05);
            index.insert(format!("tag{}", i), vec).unwrap();
        }

        let query = vec![1.0, 0.0, 0.0];
        let results = index.find_similar(&query, 0.5, 3);

        assert_eq!(results.len(), 3);
        // Should be sorted by score descending
        assert!(results[0].score >= results[1].score);
        assert!(results[1].score >= results[2].score);
    }

    #[test]
    fn test_tag_embedding_index_find_similar_empty_query() {
        let mut index = TagEmbeddingIndex::new();
        index.insert("a", vec![1.0, 0.0]).unwrap();

        let results = index.find_similar(&[], 0.5, 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_tag_embedding_index_find_similar_dimension_mismatch() {
        let mut index = TagEmbeddingIndex::new();
        index.insert("a", vec![1.0, 0.0, 0.0]).unwrap();

        let query = vec![1.0, 0.0]; // Wrong dimension
        let results = index.find_similar(&query, 0.5, 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_tag_embedding_index_remove() {
        let mut index = TagEmbeddingIndex::new();
        index.insert("a", vec![1.0, 0.0]).unwrap();
        index.insert("b", vec![0.0, 1.0]).unwrap();

        let removed = index.remove("a");
        assert!(removed.is_some());
        assert!(!index.contains("a"));
        assert!(index.contains("b"));
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn test_tag_embedding_index_remove_last_resets_dimension() {
        let mut index = TagEmbeddingIndex::new();

        // Insert 2D embedding
        index.insert("a", vec![1.0, 0.0]).unwrap();
        assert_eq!(index.dimension(), Some(2));

        // Remove it - dimension should reset
        index.remove("a");
        assert!(index.is_empty());
        assert_eq!(index.dimension(), None);

        // Now we can insert a 3D embedding (different dimension)
        let result = index.insert("b", vec![1.0, 2.0, 3.0]);
        assert!(result.is_ok());
        assert_eq!(index.dimension(), Some(3));
    }

    #[test]
    fn test_tag_embedding_index_clear() {
        let mut index = TagEmbeddingIndex::new();
        index.insert("a", vec![1.0, 0.0]).unwrap();
        index.insert("b", vec![0.0, 1.0]).unwrap();

        index.clear();
        assert!(index.is_empty());
        assert!(index.dimension().is_none());
    }

    #[test]
    fn test_tag_embedding_index_tags_iterator() {
        let mut index = TagEmbeddingIndex::new();
        index.insert("a", vec![1.0, 0.0]).unwrap();
        index.insert("b", vec![0.0, 1.0]).unwrap();

        let tags: HashSet<_> = index.tags().cloned().collect();
        assert_eq!(tags.len(), 2);
        assert!(tags.contains("a"));
        assert!(tags.contains("b"));
    }
}
