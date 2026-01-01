pub mod chunking;
pub mod engine;
pub mod error;
pub mod search;
pub mod traits;
pub mod types;

#[cfg(feature = "persistence")]
pub mod persistence;

pub use crate::engine::RagEngine;
pub use crate::error::{EmbeddingError, EngineError, RerankError};
pub use crate::traits::{EmbeddingBackend, Rerank};
pub use crate::types::{
    ChunkMetadata, DocumentChunk, RagConfig, RerankedResult, RerankerCandidate, SearchResult,
    SearchWeights,
};
