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

#[derive(Debug, Clone, Copy)]
pub struct SearchWeights {
    pub embedding: f32,
    pub lexical: f32,
    pub reranker: f32,
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

#[cfg(feature = "persistence")]
fn default_page_number() -> usize {
    0
}
