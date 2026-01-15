#[cfg(feature = "persistence")]
use serde::{Deserialize, Serialize};

pub type DocumentName = String;

#[cfg_attr(
    feature = "persistence",
    derive(Debug, Clone, Serialize, Deserialize, Default)
)]
#[cfg_attr(not(feature = "persistence"), derive(Debug, Clone, Default))]
pub struct ChunkMetadata {
    pub page_range: Option<(usize, usize)>,
    pub sentence_range: Option<(usize, usize)>,
    pub section_title: Option<String>,
    pub token_count: usize,
    pub overlap_with_previous: usize,
}

#[cfg_attr(feature = "persistence", derive(Debug, Clone, Serialize, Deserialize))]
#[cfg_attr(not(feature = "persistence"), derive(Debug, Clone))]
pub struct DocumentChunk {
    pub id: String,
    pub document_name: DocumentName,
    pub text: String,
    pub embedding: Vec<f32>,
    pub chunk_index: usize,
    #[cfg_attr(feature = "persistence", serde(default = "default_page_number"))]
    pub page_number: usize,
    #[cfg_attr(feature = "persistence", serde(default))]
    pub section: Option<String>,
    #[cfg_attr(feature = "persistence", serde(default))]
    pub metadata: ChunkMetadata,
    /// Tags associated with this chunk (Phase 2). Hierarchical tags are exploded
    /// at ingestion: "a/b/c" becomes ["a", "a/b", "a/b/c"].
    #[cfg_attr(feature = "persistence", serde(default))]
    pub tags: std::collections::HashSet<String>,
    /// Resolution level of this chunk (Phase 5). Defaults to Chunk.
    #[cfg_attr(feature = "persistence", serde(default))]
    pub resolution: Resolution,
    /// ID of the parent chunk in the hierarchy (Phase 5).
    /// Chunk → Section → Document
    #[cfg_attr(feature = "persistence", serde(default))]
    pub parent_id: Option<String>,
}

#[cfg_attr(feature = "persistence", derive(Debug, Clone, Serialize))]
#[cfg_attr(not(feature = "persistence"), derive(Debug, Clone))]
pub struct SearchResult {
    pub text: String,
    pub score: f32,
    pub document: DocumentName,
    pub chunk_id: String,
    pub chunk_index: usize,
    pub page_number: usize,
    pub section: Option<String>,
    #[cfg_attr(
        feature = "persistence",
        serde(skip_serializing_if = "Option::is_none")
    )]
    pub embedding_score: Option<f32>,
    #[cfg_attr(
        feature = "persistence",
        serde(skip_serializing_if = "Option::is_none")
    )]
    pub lexical_score: Option<f32>,
    #[cfg_attr(
        feature = "persistence",
        serde(skip_serializing_if = "Option::is_none")
    )]
    pub initial_score: Option<f32>,
    #[cfg_attr(
        feature = "persistence",
        serde(skip_serializing_if = "Option::is_none")
    )]
    pub reranker_score: Option<f32>,
    #[cfg_attr(
        feature = "persistence",
        serde(skip_serializing_if = "Option::is_none")
    )]
    pub yes_logprob: Option<f64>,
    #[cfg_attr(
        feature = "persistence",
        serde(skip_serializing_if = "Option::is_none")
    )]
    pub no_logprob: Option<f64>,
}

/// Weights for hybrid search scoring.
///
/// rag-core uses a two-stage retrieval process:
///
/// ## Stage 1: Initial Retrieval
///
/// Combines embedding similarity and lexical matching:
/// ```text
/// initial_score = (embedding * embedding_score) + (lexical * lexical_score)
/// ```
///
/// - **`embedding`**: Weight for semantic similarity (cosine distance). Higher values
///   favor documents with similar meaning even if words differ.
/// - **`lexical`**: Weight for keyword matching (BM25-style). Higher values favor
///   documents containing the exact query terms.
///
/// ## Stage 2: Reranking (if enabled)
///
/// Blends initial scores with reranker judgments:
/// ```text
/// final_score = (initial * initial_score) + (reranker * reranker_score)
/// ```
///
/// - **`initial`**: Weight for the Stage 1 score.
/// - **`reranker`**: Weight for LLM-based relevance scoring.
///
/// ## Tuning Guidelines
///
/// | Scenario | Recommended Weights |
/// |----------|---------------------|
/// | General search | `embedding: 0.7, lexical: 0.3` (default) |
/// | Exact term matching (code, IDs) | `embedding: 0.3, lexical: 0.7` |
/// | Semantic-only (concepts) | `embedding: 1.0, lexical: 0.0` |
/// | Trust reranker fully | `reranker: 1.0, initial: 0.0` |
/// | Skip reranking | `reranker: 0.0, initial: 1.0` |
///
/// ## Example
///
/// ```rust,ignore
/// use rag_core::SearchWeights;
///
/// // Favor exact keyword matches for code search
/// let code_weights = SearchWeights {
///     embedding: 0.3,
///     lexical: 0.7,
///     ..Default::default()
/// };
///
/// // Semantic-only for conceptual queries
/// let semantic_weights = SearchWeights {
///     embedding: 1.0,
///     lexical: 0.0,
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "persistence", derive(serde::Serialize, serde::Deserialize))]
pub struct SearchWeights {
    /// Weight for embedding (semantic) similarity in Stage 1. Default: 0.7
    pub embedding: f32,
    /// Weight for lexical (keyword) matching in Stage 1. Default: 0.3
    pub lexical: f32,
    /// Weight for reranker score in Stage 2. Default: 0.7
    pub reranker: f32,
    /// Weight for initial score in Stage 2. Default: 0.3
    pub initial: f32,
}

impl Default for SearchWeights {
    fn default() -> Self {
        Self {
            embedding: 0.7,
            lexical: 0.3,
            reranker: 0.7,
            initial: 0.3,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RagConfig {
    pub chunk_tokens: usize,
    pub sentence_overlap: usize,
    pub weights: SearchWeights,
    pub embedding_batch_size: usize,
}

impl Default for RagConfig {
    fn default() -> Self {
        Self {
            chunk_tokens: 200,
            sentence_overlap: 2,
            weights: SearchWeights::default(),
            embedding_batch_size: 32,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RerankerCandidate {
    pub chunk_id: String,
    pub document: DocumentName,
    pub text: String,
    pub page_number: usize,
    pub section: Option<String>,
    pub initial_score: f32,
}

#[derive(Debug, Clone)]
pub struct RerankedResult {
    pub chunk_id: String,
    pub relevance: f32,
    pub yes_logprob: Option<f64>,
    pub no_logprob: Option<f64>,
}

/// Health status for monitoring and observability.
///
/// Returned by [`RagEngine::health()`](crate::RagEngine::health) for container
/// health probes or monitoring dashboards.
#[derive(Debug, Clone)]
pub struct HealthStatus {
    /// Whether the engine is healthy and ready to serve requests
    pub is_healthy: bool,
    /// The embedding model identifier
    pub embedding_model: String,
    /// The embedding dimension
    pub embedding_dim: usize,
    /// Number of indexed documents
    pub document_count: usize,
    /// Number of indexed chunks
    pub chunk_count: usize,
    /// Whether a reindex is needed (e.g., after model change)
    pub needs_reindex: bool,
    /// Whether a reranker is configured
    pub has_reranker: bool,
}

#[cfg(feature = "persistence")]
fn default_page_number() -> usize {
    0
}

// ============================================================================
// Phase 0: QuerySpec Contract Types
// ============================================================================

/// Defines which documents to search across.
///
/// Used in [`QuerySpec`] to scope searches to specific documents, collections, or all.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum SearchScope {
    /// Search all indexed documents across all collections
    #[default]
    All,
    /// Search only the specified documents (by name)
    Documents(Vec<DocumentName>),
    /// Search within a specific collection (Phase 1)
    Collection(crate::collections::CollectionId),
    /// Search across multiple collections (Phase 1)
    Multi(Vec<crate::collections::CollectionId>),
}

/// Filter expressions for document and tag-based filtering.
///
/// These filters are evaluated before scoring to reduce the candidate set.
/// Supports logical operators (AND, OR, NOT) and tag-based filtering with
/// hierarchical prefix matching.
///
/// # Examples
///
/// ```rust
/// use rag_core::FilterExpr;
///
/// // Simple tag filter
/// let filter = FilterExpr::Tag("research".into());
///
/// // Prefix match (matches "ml", "ml/nlp", "ml/vision", etc.)
/// let filter = FilterExpr::Prefix("ml/".into());
///
/// // Combined: research AND ml
/// let filter = FilterExpr::And(vec![
///     FilterExpr::Tag("research".into()),
///     FilterExpr::Tag("ml".into()),
/// ]);
///
/// // Negation: research but NOT outdated
/// let filter = FilterExpr::And(vec![
///     FilterExpr::Tag("research".into()),
///     FilterExpr::Not(Box::new(FilterExpr::Tag("outdated".into()))),
/// ]);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum FilterExpr {
    /// Match chunks from documents with this exact name (Phase 0)
    DocumentNameEquals(DocumentName),
    /// Match chunks with this exact tag (Phase 2)
    Tag(String),
    /// Match chunks with tags starting with this prefix (Phase 2)
    /// e.g., Prefix("ml/") matches "ml", "ml/nlp", "ml/vision"
    Prefix(String),
    /// All conditions must match (logical AND)
    And(Vec<FilterExpr>),
    /// Any condition matches (logical OR)
    Or(Vec<FilterExpr>),
    /// Negate the inner expression (logical NOT)
    Not(Box<FilterExpr>),
}

/// Score boosting configuration for specific documents.
#[derive(Debug, Clone, PartialEq)]
pub struct BoostSpec {
    /// Document to boost
    pub document_name: DocumentName,
    /// Multiplicative boost factor (1.0 = no change, >1.0 = boost, <1.0 = demote)
    pub factor: f32,
}

impl BoostSpec {
    /// Create a new boost specification.
    pub fn new(document_name: impl Into<DocumentName>, factor: f32) -> Self {
        Self {
            document_name: document_name.into(),
            factor,
        }
    }
}

// ============================================================================
// Phase 5: Multi-Resolution Indexing Types
// ============================================================================

/// Resolution level for multi-resolution indexing.
///
/// Documents can be indexed at multiple granularities:
/// - Document: High-level summary (title + first 5 sentences)
/// - Section: Chapter/section summary (heading + first 2 sentences)
/// - Chunk: Fine-grained text (200 tokens, current default)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "persistence", derive(Serialize, Deserialize))]
pub enum Resolution {
    /// Document-level summary for "what is this about?" queries
    Document,
    /// Section-level summary for "find the right chapter" queries
    Section,
    /// Chunk-level text for precise fact retrieval (default)
    #[default]
    Chunk,
}

impl Resolution {
    /// Returns the parent resolution level, if any.
    pub fn parent(&self) -> Option<Resolution> {
        match self {
            Resolution::Document => None,
            Resolution::Section => Some(Resolution::Document),
            Resolution::Chunk => Some(Resolution::Section),
        }
    }

    /// Returns all resolutions from coarsest to finest.
    pub fn all_coarse_to_fine() -> [Resolution; 3] {
        [Resolution::Document, Resolution::Section, Resolution::Chunk]
    }
}

/// Search strategy for multi-resolution queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SearchStrategy {
    /// Search coarse levels first, then filter to fine levels.
    /// More efficient but may miss matches that only appear at fine level.
    CoarseToFine,
    /// Search all resolution levels in parallel, then merge scores.
    /// More comprehensive but higher latency.
    #[default]
    Parallel,
}

/// Configuration for multi-resolution search behavior.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolutionSpec {
    /// Which strategy to use for multi-resolution search.
    pub strategy: SearchStrategy,
    /// Weights for each resolution level [document, section, chunk].
    /// Used by Parallel strategy to merge scores.
    pub weights: [f32; 3],
    /// Which resolution levels to include in search.
    pub enabled_resolutions: Vec<Resolution>,
}

impl Default for ResolutionSpec {
    fn default() -> Self {
        Self {
            strategy: SearchStrategy::default(),
            weights: [0.1, 0.2, 0.7], // Favor chunk-level by default
            enabled_resolutions: vec![Resolution::Document, Resolution::Section, Resolution::Chunk],
        }
    }
}

impl ResolutionSpec {
    /// Create a new resolution spec with defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the search strategy.
    pub fn with_strategy(mut self, strategy: SearchStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set weights for [document, section, chunk] resolution levels.
    ///
    /// Weights are normalized internally; they don't need to sum to 1.0.
    pub fn with_weights(mut self, weights: [f32; 3]) -> Self {
        // Normalize weights to sum to 1.0
        let sum: f32 = weights.iter().sum();
        if sum > 0.0 {
            self.weights = [weights[0] / sum, weights[1] / sum, weights[2] / sum];
        } else {
            // Default to equal weights if all zero
            self.weights = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];
        }
        self
    }

    /// Enable only specific resolution levels.
    pub fn with_resolutions(mut self, resolutions: Vec<Resolution>) -> Self {
        self.enabled_resolutions = resolutions;
        self
    }

    /// Shorthand for chunk-only search (current behavior).
    pub fn chunk_only() -> Self {
        Self {
            strategy: SearchStrategy::Parallel,
            weights: [0.0, 0.0, 1.0],
            enabled_resolutions: vec![Resolution::Chunk],
        }
    }
}

/// The universal query specification.
///
/// All search features plug into this contract. Provides a stable interface
/// that can be extended without breaking existing code.
///
/// # Example
///
/// ```rust
/// use rag_core::{QuerySpec, SearchScope, SearchWeights};
///
/// // Simple query with defaults
/// let spec = QuerySpec::new("machine learning");
///
/// // Query with custom settings
/// let spec = QuerySpec::new("machine learning")
///     .with_top_k(10)
///     .with_scope(SearchScope::Documents(vec!["paper.pdf".into()]));
/// ```
#[derive(Debug, Clone)]
pub struct QuerySpec {
    /// The semantic query text
    pub query: String,
    /// Maximum number of results to return
    pub top_k: usize,
    /// Which documents to search
    pub scope: SearchScope,
    /// Optional weight overrides for scoring
    pub weights: Option<SearchWeights>,
    /// Filters to apply before scoring
    pub filters: Vec<FilterExpr>,
    /// Score boosts to apply
    pub boosts: Vec<BoostSpec>,
    /// MMR diversity factor (0.0 = pure relevance, 1.0 = max diversity)
    pub diversity_factor: f32,
    /// Multi-resolution search configuration (Phase 5).
    /// If None, defaults to chunk-only search for backward compatibility.
    pub resolution_spec: Option<ResolutionSpec>,
}

impl QuerySpec {
    /// Create a new query specification with default settings.
    pub fn new(query: impl Into<String>) -> Self {
        Self {
            query: query.into(),
            top_k: 5,
            scope: SearchScope::default(),
            weights: None,
            filters: Vec::new(),
            boosts: Vec::new(),
            diversity_factor: 0.3,
            resolution_spec: None, // Defaults to chunk-only for backward compatibility
        }
    }

    /// Set the maximum number of results.
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    /// Set the search scope.
    pub fn with_scope(mut self, scope: SearchScope) -> Self {
        self.scope = scope;
        self
    }

    /// Set custom weights for scoring.
    pub fn with_weights(mut self, weights: SearchWeights) -> Self {
        self.weights = Some(weights);
        self
    }

    /// Add a filter expression.
    pub fn with_filter(mut self, filter: FilterExpr) -> Self {
        self.filters.push(filter);
        self
    }

    /// Add a document boost.
    pub fn with_boost(mut self, boost: BoostSpec) -> Self {
        self.boosts.push(boost);
        self
    }

    /// Set the MMR diversity factor.
    pub fn with_diversity(mut self, factor: f32) -> Self {
        self.diversity_factor = factor;
        self
    }

    /// Set multi-resolution search configuration (Phase 5).
    pub fn with_resolution(mut self, spec: ResolutionSpec) -> Self {
        self.resolution_spec = Some(spec);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // Phase 0 TDD Tests: QuerySpec Contract
    // ========================================================================

    #[test]
    fn test_query_spec_new_defaults() {
        let spec = QuerySpec::new("test query");

        assert_eq!(spec.query, "test query");
        assert_eq!(spec.top_k, 5);
        assert_eq!(spec.scope, SearchScope::All);
        assert!(spec.weights.is_none());
        assert!(spec.filters.is_empty());
        assert!(spec.boosts.is_empty());
        assert!((spec.diversity_factor - 0.3).abs() < f32::EPSILON);
    }

    #[test]
    fn test_query_spec_builder_chain() {
        let spec = QuerySpec::new("machine learning")
            .with_top_k(10)
            .with_scope(SearchScope::Documents(vec!["paper.pdf".into()]))
            .with_diversity(0.5);

        assert_eq!(spec.query, "machine learning");
        assert_eq!(spec.top_k, 10);
        assert_eq!(spec.scope, SearchScope::Documents(vec!["paper.pdf".into()]));
        assert!((spec.diversity_factor - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_query_spec_with_weights() {
        let weights = SearchWeights {
            embedding: 0.9,
            lexical: 0.1,
            ..Default::default()
        };

        let spec = QuerySpec::new("test").with_weights(weights);

        assert!(spec.weights.is_some());
        let w = spec.weights.unwrap();
        assert!((w.embedding - 0.9).abs() < f32::EPSILON);
        assert!((w.lexical - 0.1).abs() < f32::EPSILON);
    }

    #[test]
    fn test_query_spec_with_filter() {
        let spec =
            QuerySpec::new("test").with_filter(FilterExpr::DocumentNameEquals("doc.pdf".into()));

        assert_eq!(spec.filters.len(), 1);
        assert_eq!(
            spec.filters[0],
            FilterExpr::DocumentNameEquals("doc.pdf".into())
        );
    }

    #[test]
    fn test_query_spec_with_boost() {
        let spec = QuerySpec::new("test").with_boost(BoostSpec::new("important.pdf", 2.0));

        assert_eq!(spec.boosts.len(), 1);
        assert_eq!(spec.boosts[0].document_name, "important.pdf");
        assert!((spec.boosts[0].factor - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_query_spec_multiple_filters_and_boosts() {
        let spec = QuerySpec::new("test")
            .with_filter(FilterExpr::DocumentNameEquals("a.pdf".into()))
            .with_filter(FilterExpr::DocumentNameEquals("b.pdf".into()))
            .with_boost(BoostSpec::new("a.pdf", 1.5))
            .with_boost(BoostSpec::new("c.pdf", 0.5));

        assert_eq!(spec.filters.len(), 2);
        assert_eq!(spec.boosts.len(), 2);
    }

    #[test]
    fn test_search_scope_default() {
        let scope = SearchScope::default();
        assert_eq!(scope, SearchScope::All);
    }

    #[test]
    fn test_search_scope_documents() {
        let scope = SearchScope::Documents(vec!["a.pdf".into(), "b.pdf".into()]);
        if let SearchScope::Documents(docs) = scope {
            assert_eq!(docs.len(), 2);
            assert_eq!(docs[0], "a.pdf");
            assert_eq!(docs[1], "b.pdf");
        } else {
            panic!("Expected Documents variant");
        }
    }

    #[test]
    fn test_boost_spec_creation() {
        let boost = BoostSpec::new("test.pdf", 1.5);
        assert_eq!(boost.document_name, "test.pdf");
        assert!((boost.factor - 1.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_filter_expr_document_name_equals() {
        let filter = FilterExpr::DocumentNameEquals("target.pdf".into());
        assert_eq!(filter, FilterExpr::DocumentNameEquals("target.pdf".into()));
    }

    // ========================================================================
    // Phase 1 TDD Tests: SearchScope Collection variants
    // ========================================================================

    #[test]
    fn test_search_scope_collection() {
        use crate::collections::CollectionId;
        let scope = SearchScope::Collection(CollectionId::new("research"));
        if let SearchScope::Collection(id) = scope {
            assert_eq!(id.as_str(), "research");
        } else {
            panic!("Expected Collection variant");
        }
    }

    #[test]
    fn test_search_scope_multi() {
        use crate::collections::CollectionId;
        let scope = SearchScope::Multi(vec![
            CollectionId::new("research"),
            CollectionId::new("legal"),
        ]);
        if let SearchScope::Multi(ids) = scope {
            assert_eq!(ids.len(), 2);
            assert_eq!(ids[0].as_str(), "research");
            assert_eq!(ids[1].as_str(), "legal");
        } else {
            panic!("Expected Multi variant");
        }
    }

    #[test]
    fn test_query_spec_with_collection_scope() {
        use crate::collections::CollectionId;
        let spec = QuerySpec::new("test")
            .with_scope(SearchScope::Collection(CollectionId::new("research")));

        if let SearchScope::Collection(id) = &spec.scope {
            assert_eq!(id.as_str(), "research");
        } else {
            panic!("Expected Collection scope");
        }
    }

    // ========================================================================
    // Phase 5 TDD Tests: Multi-Resolution Indexing
    // ========================================================================

    #[test]
    fn test_resolution_default() {
        assert_eq!(Resolution::default(), Resolution::Chunk);
    }

    #[test]
    fn test_resolution_parent_hierarchy() {
        assert_eq!(Resolution::Document.parent(), None);
        assert_eq!(Resolution::Section.parent(), Some(Resolution::Document));
        assert_eq!(Resolution::Chunk.parent(), Some(Resolution::Section));
    }

    #[test]
    fn test_resolution_all_coarse_to_fine() {
        let all = Resolution::all_coarse_to_fine();
        assert_eq!(
            all,
            [Resolution::Document, Resolution::Section, Resolution::Chunk]
        );
    }

    #[test]
    fn test_search_strategy_default() {
        assert_eq!(SearchStrategy::default(), SearchStrategy::Parallel);
    }

    #[test]
    fn test_resolution_spec_defaults() {
        let spec = ResolutionSpec::default();

        assert_eq!(spec.strategy, SearchStrategy::Parallel);
        assert!((spec.weights[0] - 0.1).abs() < 1e-6); // document
        assert!((spec.weights[1] - 0.2).abs() < 1e-6); // section
        assert!((spec.weights[2] - 0.7).abs() < 1e-6); // chunk
        assert_eq!(spec.enabled_resolutions.len(), 3);
    }

    #[test]
    fn test_resolution_spec_builder() {
        let spec = ResolutionSpec::new()
            .with_strategy(SearchStrategy::CoarseToFine)
            .with_weights([0.5, 0.3, 0.2])
            .with_resolutions(vec![Resolution::Document, Resolution::Chunk]);

        assert_eq!(spec.strategy, SearchStrategy::CoarseToFine);
        // Weights are normalized to sum to 1.0
        assert!((spec.weights[0] - 0.5).abs() < 1e-6);
        assert!((spec.weights[1] - 0.3).abs() < 1e-6);
        assert!((spec.weights[2] - 0.2).abs() < 1e-6);
        assert_eq!(spec.enabled_resolutions.len(), 2);
    }

    #[test]
    fn test_resolution_spec_weight_normalization() {
        // Non-normalized weights
        let spec = ResolutionSpec::new().with_weights([2.0, 4.0, 4.0]);

        // Should be normalized to sum to 1.0
        assert!((spec.weights[0] - 0.2).abs() < 1e-6);
        assert!((spec.weights[1] - 0.4).abs() < 1e-6);
        assert!((spec.weights[2] - 0.4).abs() < 1e-6);
    }

    #[test]
    fn test_resolution_spec_zero_weights_fallback() {
        // All zero weights should default to equal
        let spec = ResolutionSpec::new().with_weights([0.0, 0.0, 0.0]);

        let third = 1.0 / 3.0;
        assert!((spec.weights[0] - third).abs() < 1e-6);
        assert!((spec.weights[1] - third).abs() < 1e-6);
        assert!((spec.weights[2] - third).abs() < 1e-6);
    }

    #[test]
    fn test_resolution_spec_chunk_only() {
        let spec = ResolutionSpec::chunk_only();

        assert_eq!(spec.weights, [0.0, 0.0, 1.0]);
        assert_eq!(spec.enabled_resolutions, vec![Resolution::Chunk]);
    }

    #[test]
    fn test_query_spec_with_resolution() {
        let spec = QuerySpec::new("test")
            .with_resolution(ResolutionSpec::new().with_strategy(SearchStrategy::CoarseToFine));

        assert!(spec.resolution_spec.is_some());
        let res_spec = spec.resolution_spec.unwrap();
        assert_eq!(res_spec.strategy, SearchStrategy::CoarseToFine);
    }

    #[test]
    fn test_query_spec_resolution_defaults_to_none() {
        let spec = QuerySpec::new("test");

        // Should be None for backward compatibility
        assert!(spec.resolution_spec.is_none());
    }
}
