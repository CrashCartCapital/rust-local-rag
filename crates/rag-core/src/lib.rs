//! # rag-core
//!
//! A batteries-included-but-swappable embedded RAG (Retrieval-Augmented Generation)
//! engine for Rust.
//!
//! ## Overview
//!
//! rag-core provides the core abstractions for building local-first, privacy-focused
//! document retrieval systems. It handles:
//!
//! - **Sentence-aware chunking** of documents
//! - **Hybrid search** combining embedding similarity and lexical matching
//! - **Optional reranking** with pluggable reranker backends
//! - **Persistence** with atomic writes and legacy format migration
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use rag_core::{RagEngine, EmbeddingBackend, EmbeddingError};
//!
//! // Implement EmbeddingBackend for your model
//! struct MyEmbedder;
//! impl EmbeddingBackend for MyEmbedder {
//!     async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
//!         Ok(vec![0.1, 0.2, 0.3])
//!     }
//!     fn model_id(&self) -> &str { "my-model" }
//!     fn dimension(&self) -> usize { 3 }
//! }
//!
//! let mut engine = RagEngine::new(MyEmbedder);
//! ```
//!
//! ## Feature Flags
//!
//! - **`persistence`**: Enables [`JsonFileBackend`] and disk save/load operations
//! - **`tracing`**: Adds instrumentation spans for observability
//!
//! ## Modules
//!
//! - [`chunking`]: Sentence-aware text chunking with metadata
//! - [`engine`]: The main [`RagEngine`] coordinator
//! - [`error`]: Structured error types with context
//! - [`search`]: Hybrid ANN + lexical search with MMR diversification
//! - [`traits`]: Core traits ([`EmbeddingBackend`], [`Rerank`])
//! - [`types`]: Data structures for chunks, results, and configuration
//! - [`persistence`]: Save/load engine state (feature-gated)

pub mod chunking;
pub mod collections;
pub mod engine;
pub mod error;
pub mod relationships;
pub mod search;
pub mod tags;
pub mod traits;
pub mod types;

#[cfg(feature = "persistence")]
pub mod persistence;

#[cfg(feature = "persistence")]
pub use crate::persistence::{EngineState, JsonFileBackend, PersistenceBackend};

pub use crate::engine::RagEngine;
pub use crate::error::{
    EmbeddingError, EngineError, PersistenceError, PersistenceOp, RerankError, ValidationKind,
};
pub use crate::collections::{Collection, CollectionConfig, CollectionId, CollectionManifest};
pub use crate::relationships::{
    Edge, RelationIndex, Relationship, RelationshipConfig, RelationshipType,
};
pub use crate::tags::{
    cosine_similarity, explode_tag, explode_tags, ExpandedTag, ExpansionMode, TagEmbeddingIndex,
    TagExpansionConfig, TagIndex,
};
pub use crate::traits::{EmbeddingBackend, Rerank};
pub use crate::search::{CandidateScore, ChunkUpdate, IndexSet};
pub use crate::types::{
    BoostSpec, ChunkMetadata, DocumentChunk, FilterExpr, HealthStatus, QuerySpec, RagConfig,
    RerankedResult, RerankerCandidate, Resolution, ResolutionSpec, SearchResult, SearchScope,
    SearchStrategy, SearchWeights,
};
