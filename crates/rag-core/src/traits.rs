//! Core traits for embedding and reranking backends.
//!
//! This module defines the pluggable abstractions that allow rag-core to work
//! with any embedding model or reranker.
//!
//! # Concurrency Model
//!
//! All traits in this module require `Send + Sync` bounds:
//!
//! - **`Send`**: Implementations can be transferred between threads
//! - **`Sync`**: Implementations can be shared between threads via `&T`
//!
//! This enables concurrent embedding and search operations. If your backend
//! uses internal mutability (e.g., connection pools, caches), use appropriate
//! synchronization primitives:
//!
//! ```rust,ignore
//! use std::sync::RwLock;
//!
//! struct CachingEmbedder {
//!     cache: RwLock<HashMap<String, Vec<f32>>>,
//!     // ...
//! }
//! ```
//!
//! # Blocking I/O
//!
//! Embedding and reranking often involve network I/O. The async methods in
//! these traits are designed for non-blocking execution. If your backend uses
//! blocking I/O internally, wrap calls in `tokio::task::spawn_blocking`:
//!
//! ```rust,ignore
//! async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
//!     let text = text.to_string();
//!     tokio::task::spawn_blocking(move || {
//!         // Blocking HTTP call here
//!     }).await.map_err(|_| EmbeddingError::Connection("task failed".into()))?
//! }
//! ```
//!
//! # Implementing an Embedding Backend
//!
//! ```rust,ignore
//! use rag_core::{EmbeddingBackend, EmbeddingError};
//!
//! struct OllamaEmbedder {
//!     model: String,
//!     dim: usize,
//! }
//!
//! impl EmbeddingBackend for OllamaEmbedder {
//!     fn model_id(&self) -> &str { &self.model }
//!     fn dimension(&self) -> usize { self.dim }
//!
//!     async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
//!         // Call your embedding service here
//!         Ok(vec![0.0; self.dim])
//!     }
//! }
//! ```
//!
//! # Error Mapping
//!
//! When integrating external HTTP clients (reqwest, ureq, etc.), map their
//! errors to [`EmbeddingError`] variants for consistent error handling:
//!
//! ```rust,ignore
//! use rag_core::EmbeddingError;
//! use std::time::Duration;
//!
//! // Example: mapping reqwest errors to EmbeddingError
//! fn map_reqwest_error(e: reqwest::Error) -> EmbeddingError {
//!     if e.is_timeout() {
//!         EmbeddingError::Timeout(Duration::from_secs(30))
//!     } else if e.is_connect() {
//!         EmbeddingError::Connection(format!("Connection failed: {}", e))
//!     } else if e.is_status() {
//!         EmbeddingError::Api(format!("HTTP error: {}", e))
//!     } else {
//!         EmbeddingError::Api(e.to_string())
//!     }
//! }
//!
//! // Usage in embed():
//! async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
//!     let response = self.client
//!         .post(&self.url)
//!         .json(&request)
//!         .send()
//!         .await
//!         .map_err(map_reqwest_error)?;  // Convert to EmbeddingError
//!
//!     // ... parse response
//! }
//! ```

use crate::error::{EmbeddingError, RerankError};
use crate::types::{RerankedResult, RerankerCandidate};
use std::future::Future;

/// Trait for embedding providers.
///
/// Implement this trait to integrate any embedding model (Ollama, OpenAI,
/// HuggingFace, etc.) with rag-core.
///
/// # Thread Safety
///
/// Implementations must be `Send + Sync` for concurrent embedding operations.
///
/// # Batch Processing
///
/// The default [`embed_batch`](EmbeddingBackend::embed_batch) implementation
/// calls [`embed`](EmbeddingBackend::embed) sequentially. Override for better
/// performance if your backend supports native batching.
pub trait EmbeddingBackend: Send + Sync {
    /// Returns the model identifier (e.g., "nomic-embed-text").
    ///
    /// This is used for persistence and model-switching detection.
    fn model_id(&self) -> &str;

    /// Embed a single text into a vector.
    ///
    /// # Errors
    ///
    /// Returns [`EmbeddingError`] if the embedding service fails.
    fn embed(&self, text: &str) -> impl Future<Output = Result<Vec<f32>, EmbeddingError>> + Send;

    /// Embed multiple texts in batch.
    ///
    /// Default implementation calls [`embed`](EmbeddingBackend::embed) for each
    /// text sequentially. Override this if your backend supports native batching
    /// for better performance.
    ///
    /// # Errors
    ///
    /// Returns [`EmbeddingError`] if any embedding fails. Partial results are
    /// not returned.
    fn embed_batch(
        &self,
        texts: &[String],
    ) -> impl Future<Output = Result<Vec<Vec<f32>>, EmbeddingError>> + Send {
        async move {
            let mut results = Vec::with_capacity(texts.len());
            for text in texts {
                results.push(self.embed(text).await?);
            }
            Ok(results)
        }
    }

    /// Returns the embedding dimension (e.g., 768 for many models).
    ///
    /// This must be consistent for all embeddings produced by this backend.
    fn dimension(&self) -> usize;
}

/// Trait for second-stage reranking.
///
/// Rerankers improve search quality by using more expensive models to
/// re-score candidates after initial retrieval.
///
/// # No-op Implementation
///
/// The unit type `()` implements `Rerank` as a no-op, returning empty results.
/// Use this when reranking is not needed.
pub trait Rerank: Send + Sync {
    /// Rerank candidates by relevance to the query.
    ///
    /// # Arguments
    ///
    /// * `query` - The search query
    /// * `candidates` - Chunks to rerank with their metadata
    ///
    /// # Returns
    ///
    /// Reranked results with updated scores, sorted by relevance.
    fn rerank(
        &self,
        query: &str,
        candidates: &[RerankerCandidate],
    ) -> impl Future<Output = Result<Vec<RerankedResult>, RerankError>> + Send;
}

/// No-op reranker implementation.
///
/// Returns empty results, effectively disabling reranking.
impl Rerank for () {
    async fn rerank(
        &self,
        _query: &str,
        _candidates: &[RerankerCandidate],
    ) -> Result<Vec<RerankedResult>, RerankError> {
        Ok(Vec::new())
    }
}
