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
use futures_core::future::BoxFuture;
use std::future::Future;
use std::sync::Arc;

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

// =============================================================================
// Object-Safe Dynamic Dispatch Traits
// =============================================================================
//
// The traits above use RPITIT (`impl Future`) which is NOT object-safe.
// The traits below use `BoxFuture` for object-safe dynamic dispatch.
// Use these when you need `dyn Trait` (e.g., Python bindings, plugin systems).

/// Object-safe version of [`EmbeddingBackend`] for dynamic dispatch.
///
/// This trait uses [`BoxFuture`] instead of `impl Future`, enabling use as
/// `dyn DynEmbeddingBackend`. Use this when you need runtime polymorphism
/// (e.g., Python bindings, plugin systems).
///
/// # Blanket Implementation
///
/// All types implementing [`EmbeddingBackend`] automatically implement
/// `DynEmbeddingBackend` via a blanket impl. You don't need to implement
/// this trait manually.
///
/// # Type Alias
///
/// Use [`BoxedEmbedder`] as a convenient alias for `Arc<dyn DynEmbeddingBackend>`.
///
/// # Example
///
/// ```rust,ignore
/// use rag_core::{BoxedEmbedder, DynEmbeddingBackend};
///
/// fn create_embedder(config: &Config) -> BoxedEmbedder {
///     if config.use_ollama {
///         Arc::new(OllamaEmbedder::new())
///     } else {
///         Arc::new(OpenAIEmbedder::new())
///     }
/// }
/// ```
pub trait DynEmbeddingBackend: Send + Sync {
    /// Returns the model identifier (e.g., "nomic-embed-text").
    fn model_id(&self) -> &str;

    /// Embed a single text into a vector.
    ///
    /// Returns a boxed future for object safety.
    fn embed<'a>(&'a self, text: &'a str) -> BoxFuture<'a, Result<Vec<f32>, EmbeddingError>>;

    /// Embed multiple texts in batch.
    ///
    /// Returns a boxed future for object safety.
    fn embed_batch<'a>(
        &'a self,
        texts: &'a [String],
    ) -> BoxFuture<'a, Result<Vec<Vec<f32>>, EmbeddingError>>;

    /// Returns the embedding dimension (e.g., 768 for many models).
    fn dimension(&self) -> usize;
}

/// Blanket implementation: any `EmbeddingBackend` is also a `DynEmbeddingBackend`.
///
/// This allows static-dispatch backends to be used with dynamic dispatch
/// by simply wrapping them in `Arc<dyn DynEmbeddingBackend>`.
impl<T: EmbeddingBackend> DynEmbeddingBackend for T {
    fn model_id(&self) -> &str {
        <T as EmbeddingBackend>::model_id(self)
    }

    fn embed<'a>(&'a self, text: &'a str) -> BoxFuture<'a, Result<Vec<f32>, EmbeddingError>> {
        Box::pin(<T as EmbeddingBackend>::embed(self, text))
    }

    fn embed_batch<'a>(
        &'a self,
        texts: &'a [String],
    ) -> BoxFuture<'a, Result<Vec<Vec<f32>>, EmbeddingError>> {
        Box::pin(<T as EmbeddingBackend>::embed_batch(self, texts))
    }

    fn dimension(&self) -> usize {
        <T as EmbeddingBackend>::dimension(self)
    }
}

/// Object-safe version of [`Rerank`] for dynamic dispatch.
///
/// Similar to [`DynEmbeddingBackend`], this trait uses [`BoxFuture`] for
/// object safety. All types implementing [`Rerank`] automatically implement
/// `DynRerank` via a blanket impl.
pub trait DynRerank: Send + Sync {
    /// Rerank candidates by relevance to the query.
    ///
    /// Returns a boxed future for object safety.
    fn rerank<'a>(
        &'a self,
        query: &'a str,
        candidates: &'a [RerankerCandidate],
    ) -> BoxFuture<'a, Result<Vec<RerankedResult>, RerankError>>;
}

/// Blanket implementation: any `Rerank` is also a `DynRerank`.
impl<T: Rerank> DynRerank for T {
    fn rerank<'a>(
        &'a self,
        query: &'a str,
        candidates: &'a [RerankerCandidate],
    ) -> BoxFuture<'a, Result<Vec<RerankedResult>, RerankError>> {
        Box::pin(<T as Rerank>::rerank(self, query, candidates))
    }
}

/// Type alias for a boxed, thread-safe embedding backend.
///
/// Use this for dynamic dispatch when you need runtime polymorphism.
/// The `'static` bound is required for storage in structs like `RagEngine`.
///
/// # Example
///
/// ```rust,ignore
/// use rag_core::BoxedEmbedder;
/// use std::sync::Arc;
///
/// let embedder: BoxedEmbedder = Arc::new(MyEmbedder::new());
/// ```
pub type BoxedEmbedder = Arc<dyn DynEmbeddingBackend + 'static>;

/// Type alias for a boxed, thread-safe reranker.
///
/// Use this for dynamic dispatch when you need runtime polymorphism.
pub type BoxedReranker = Arc<dyn DynRerank + 'static>;

// =============================================================================
// Reverse Direction: Enable BoxedEmbedder/BoxedReranker to work with RagEngine
// =============================================================================
//
// RagEngine<E, R> requires E: EmbeddingBackend. These impls allow using
// BoxedEmbedder (Arc<dyn DynEmbeddingBackend>) directly as the E parameter.

/// Implement `EmbeddingBackend` for `Arc<dyn DynEmbeddingBackend>`.
///
/// This enables `BoxedEmbedder` to be used with `RagEngine<BoxedEmbedder, R>`,
/// completing the bidirectional bridge between static and dynamic dispatch.
impl EmbeddingBackend for Arc<dyn DynEmbeddingBackend + 'static> {
    fn model_id(&self) -> &str {
        DynEmbeddingBackend::model_id(&**self)
    }

    fn embed(&self, text: &str) -> impl Future<Output = Result<Vec<f32>, EmbeddingError>> + Send {
        // Clone Arc and owned text to create 'static future
        let backend = Arc::clone(self);
        let text = text.to_string();
        async move { DynEmbeddingBackend::embed(&*backend, &text).await }
    }

    fn embed_batch(
        &self,
        texts: &[String],
    ) -> impl Future<Output = Result<Vec<Vec<f32>>, EmbeddingError>> + Send {
        let backend = Arc::clone(self);
        let texts = texts.to_vec();
        async move { DynEmbeddingBackend::embed_batch(&*backend, &texts).await }
    }

    fn dimension(&self) -> usize {
        DynEmbeddingBackend::dimension(&**self)
    }
}

/// Implement `Rerank` for `Arc<dyn DynRerank>`.
///
/// This enables `BoxedReranker` to be used with `RagEngine<E, BoxedReranker>`.
impl Rerank for Arc<dyn DynRerank + 'static> {
    fn rerank(
        &self,
        query: &str,
        candidates: &[RerankerCandidate],
    ) -> impl Future<Output = Result<Vec<RerankedResult>, RerankError>> + Send {
        let reranker = Arc::clone(self);
        let query = query.to_string();
        let candidates = candidates.to_vec();
        async move { DynRerank::rerank(&*reranker, &query, &candidates).await }
    }
}
