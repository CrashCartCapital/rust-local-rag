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
#[derive(Debug, Clone, Copy)]
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
