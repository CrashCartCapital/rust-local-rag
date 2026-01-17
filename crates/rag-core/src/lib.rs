pub mod chunking;
pub mod collections;
pub mod engine;
pub mod error;
#[cfg(feature = "persistence")]
pub mod persistence;
pub mod process_timeout;
pub mod relationships;
pub mod search;
pub mod tags;
pub mod traits;
pub mod types;

pub use chunking::chunk_text;
pub use engine::RagEngine;
pub use error::{EmbeddingError, EngineError};
#[cfg(feature = "persistence")]
pub use persistence::{JsonFileBackend, PersistenceBackend};
pub use search::SearchResultWithEmbedding;
pub use tags::explode_tag;
pub use traits::{EmbeddingBackend, Rerank};
pub use types::{
    ChunkMetadata, DocumentChunk, FilterExpr, QuerySpec, RagConfig, SearchResult, SearchScope,
    SearchWeights,
};
