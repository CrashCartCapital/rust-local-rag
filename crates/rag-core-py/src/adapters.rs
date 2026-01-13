//! Python backend adapters for rag-core traits.
//!
//! This module provides adapters that wrap Python classes implementing
//! embedding and reranking backends, bridging them to rag-core's Rust traits.

use pyo3::prelude::*;
use pyo3::types::PyList;
use std::future::Future;
use std::sync::Arc;
use tokio::sync::Semaphore;

use pyo3::types::PyDict;
use rag_core::{
    EmbeddingBackend, EmbeddingError, Rerank, RerankError, RerankedResult, RerankerCandidate,
};

/// Default number of concurrent Python embedding calls.
#[allow(dead_code)]
const DEFAULT_CONCURRENCY: usize = 4;

/// Result of calling a Python embed method - either sync result or async future.
enum EmbedCallResult {
    Sync(Vec<f32>),
    Async(std::pin::Pin<Box<dyn std::future::Future<Output = PyResult<Py<PyAny>>> + Send>>),
}

/// Result of calling Python embed_batch - either sync or async.
enum EmbedBatchCallResult {
    Sync(Vec<Vec<f32>>),
    Async(std::pin::Pin<Box<dyn std::future::Future<Output = PyResult<Py<PyAny>>> + Send>>),
}

/// Result of calling Python rerank - either sync or async.
enum RerankCallResult {
    Sync(Vec<RerankedResult>),
    Async(std::pin::Pin<Box<dyn std::future::Future<Output = PyResult<Py<PyAny>>> + Send>>),
}

/// Adapter that wraps a Python embedding backend class.
///
/// This adapter implements [`EmbeddingBackend`] by delegating to Python methods.
/// The Python class must implement:
/// - `model_id() -> str`
/// - `dimension() -> int`
/// - `embed(text: str) -> list[float]` (sync or async)
/// - `embed_batch(texts: list[str]) -> list[list[float]]` (optional, sync or async)
///
/// # Thread Safety
///
/// The adapter is `Send + Sync` and can be shared across threads. Python calls
/// are serialized via GIL acquisition, and a semaphore limits concurrent calls
/// to avoid overwhelming the Python interpreter.
///
/// # Example (Python)
///
/// ```python
/// class MyEmbedder:
///     def model_id(self) -> str:
///         return "my-model"
///
///     def dimension(self) -> int:
///         return 768
///
///     def embed(self, text: str) -> list[float]:
///         # Your embedding logic
///         return [0.0] * 768
///
/// engine = RagEngine.create("/tmp/index", backend=MyEmbedder())
/// ```
/// Inner state that can be shared via Arc.
struct AdapterInner {
    /// Python backend object wrapped for safe sharing.
    backend: BackendRef,
    /// Cached model identifier.
    model_id: String,
    /// Cached embedding dimension.
    dimension: usize,
    /// Whether the Python backend has embed_batch method.
    has_batch: bool,
    /// Semaphore for concurrency control.
    semaphore: Semaphore,
}

/// Wrapper for Py<PyAny> that is Send + Sync.
/// Safety: Python calls are always made with the GIL held.
struct BackendRef(Py<PyAny>);

unsafe impl Send for BackendRef {}
unsafe impl Sync for BackendRef {}

pub struct PyEmbeddingBackendAdapter {
    /// Shared inner state, wrapped in Arc for safe async capture.
    inner: Arc<AdapterInner>,
}

#[allow(dead_code)]
impl PyEmbeddingBackendAdapter {
    /// Create a new adapter wrapping a Python embedding backend.
    ///
    /// # Arguments
    ///
    /// * `py` - Python GIL token
    /// * `backend` - Python object implementing embedding backend protocol
    /// * `max_concurrency` - Maximum concurrent Python calls (default: 4)
    ///
    /// # Errors
    ///
    /// Returns error if the Python object doesn't have required methods
    /// or if `model_id()`/`dimension()` calls fail.
    pub fn new(
        _py: Python<'_>,
        backend: &Bound<'_, PyAny>,
        max_concurrency: Option<usize>,
    ) -> PyResult<Self> {
        // Validate required methods exist
        if !backend.hasattr("model_id")? {
            return Err(pyo3::exceptions::PyAttributeError::new_err(
                "Backend must have model_id() method",
            ));
        }
        if !backend.hasattr("dimension")? {
            return Err(pyo3::exceptions::PyAttributeError::new_err(
                "Backend must have dimension() method",
            ));
        }
        if !backend.hasattr("embed")? {
            return Err(pyo3::exceptions::PyAttributeError::new_err(
                "Backend must have embed() method",
            ));
        }

        // Cache model_id and dimension
        let model_id: String = backend.call_method0("model_id")?.extract()?;
        let dimension: usize = backend.call_method0("dimension")?.extract()?;

        // Check for optional embed_batch
        let has_batch = backend.hasattr("embed_batch")?;

        let concurrency = max_concurrency.unwrap_or(DEFAULT_CONCURRENCY);

        Ok(Self {
            inner: Arc::new(AdapterInner {
                backend: BackendRef(backend.clone().unbind()),
                model_id,
                dimension,
                has_batch,
                semaphore: Semaphore::new(concurrency),
            }),
        })
    }

    /// Create adapter with default concurrency.
    pub fn with_defaults(py: Python<'_>, backend: &Bound<'_, PyAny>) -> PyResult<Self> {
        Self::new(py, backend, None)
    }

    /// Call Python embed method and handle sync/async.
    async fn call_embed(
        inner: Arc<AdapterInner>,
        text: String,
    ) -> Result<Vec<f32>, EmbeddingError> {
        // Acquire semaphore permit
        let _permit = inner
            .semaphore
            .acquire()
            .await
            .map_err(|e| EmbeddingError::Api(format!("Semaphore error: {}", e)))?;

        // Call Python and check for coroutine - returns either sync result or async future
        let call_result = Python::with_gil(|py| -> PyResult<EmbedCallResult> {
            let backend = inner.backend.0.bind(py);
            let result = backend.call_method1("embed", (&text,))?;

            // Check if result is a coroutine
            if is_coroutine(py, &result)? {
                // Convert coroutine to future (only once, no re-call)
                let future = pyo3_async_runtimes::tokio::into_future(result)?;
                Ok(EmbedCallResult::Async(Box::pin(future)))
            } else {
                // Sync result - extract directly
                Ok(EmbedCallResult::Sync(extract_embedding(&result)?))
            }
        })
        .map_err(pyerr_to_embedding_error)?;

        match call_result {
            EmbedCallResult::Sync(embedding) => Ok(embedding),
            EmbedCallResult::Async(future) => {
                // Await outside GIL
                let result = future.await.map_err(pyerr_to_embedding_error)?;
                // Extract result with GIL
                Python::with_gil(|py| extract_embedding(result.bind(py)))
                    .map_err(pyerr_to_embedding_error)
            }
        }
    }

    /// Call Python embed_batch method.
    async fn call_embed_batch(
        inner: Arc<AdapterInner>,
        texts: Vec<String>,
    ) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        // Acquire semaphore permit
        let _permit = inner
            .semaphore
            .acquire()
            .await
            .map_err(|e| EmbeddingError::Api(format!("Semaphore error: {}", e)))?;

        // Call Python and check for coroutine
        let call_result = Python::with_gil(|py| -> PyResult<EmbedBatchCallResult> {
            let backend = inner.backend.0.bind(py);
            let texts_list = PyList::new(py, &texts)?;
            let result = backend.call_method1("embed_batch", (texts_list,))?;

            // Check if result is a coroutine
            if is_coroutine(py, &result)? {
                let future = pyo3_async_runtimes::tokio::into_future(result)?;
                Ok(EmbedBatchCallResult::Async(Box::pin(future)))
            } else {
                Ok(EmbedBatchCallResult::Sync(extract_embeddings_batch(
                    &result,
                )?))
            }
        })
        .map_err(pyerr_to_embedding_error)?;

        match call_result {
            EmbedBatchCallResult::Sync(embeddings) => Ok(embeddings),
            EmbedBatchCallResult::Async(future) => {
                let result = future.await.map_err(pyerr_to_embedding_error)?;
                Python::with_gil(|py| extract_embeddings_batch(result.bind(py)))
                    .map_err(pyerr_to_embedding_error)
            }
        }
    }
}

impl EmbeddingBackend for PyEmbeddingBackendAdapter {
    fn model_id(&self) -> &str {
        &self.inner.model_id
    }

    fn embed(&self, text: &str) -> impl Future<Output = Result<Vec<f32>, EmbeddingError>> + Send {
        let text = text.to_string();
        let inner = Arc::clone(&self.inner);
        async move { Self::call_embed(inner, text).await }
    }

    fn embed_batch(
        &self,
        texts: &[String],
    ) -> impl Future<Output = Result<Vec<Vec<f32>>, EmbeddingError>> + Send {
        let texts = texts.to_vec();
        let has_batch = self.inner.has_batch;
        let inner = Arc::clone(&self.inner);

        async move {
            if has_batch {
                Self::call_embed_batch(inner, texts).await
            } else {
                // Fall back to sequential embedding
                let mut results = Vec::with_capacity(texts.len());
                for text in texts {
                    results.push(Self::call_embed(Arc::clone(&inner), text).await?);
                }
                Ok(results)
            }
        }
    }

    fn dimension(&self) -> usize {
        self.inner.dimension
    }
}

/// Check if a Python object is a coroutine.
fn is_coroutine(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    let inspect = py.import("inspect")?;
    let is_coro: bool = inspect.call_method1("iscoroutine", (obj,))?.extract()?;
    Ok(is_coro)
}

/// Extract embedding vector from Python result.
fn extract_embedding(result: &Bound<'_, PyAny>) -> PyResult<Vec<f32>> {
    // Handle list or numpy array
    if let Ok(list) = result.downcast::<PyList>() {
        let mut embedding = Vec::with_capacity(list.len());
        for item in list.iter() {
            embedding.push(item.extract::<f32>()?);
        }
        Ok(embedding)
    } else {
        // Try to extract as sequence
        result.extract::<Vec<f32>>()
    }
}

/// Extract batch embeddings from Python result.
fn extract_embeddings_batch(result: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<f32>>> {
    if let Ok(list) = result.downcast::<PyList>() {
        let mut embeddings = Vec::with_capacity(list.len());
        for item in list.iter() {
            embeddings.push(extract_embedding(&item)?);
        }
        Ok(embeddings)
    } else {
        result.extract::<Vec<Vec<f32>>>()
    }
}

/// Convert PyErr to EmbeddingError.
fn pyerr_to_embedding_error(e: PyErr) -> EmbeddingError {
    Python::with_gil(|py| {
        let err_str = e.to_string();
        if e.is_instance_of::<pyo3::exceptions::PyAttributeError>(py) {
            EmbeddingError::ModelNotFound(err_str)
        } else if e.is_instance_of::<pyo3::exceptions::PyConnectionError>(py) {
            EmbeddingError::Connection(err_str)
        } else if e.is_instance_of::<pyo3::exceptions::PyTimeoutError>(py) {
            EmbeddingError::Timeout(std::time::Duration::from_secs(30))
        } else {
            EmbeddingError::Api(err_str)
        }
    })
}

// =============================================================================
// Reranker Adapter
// =============================================================================

/// Inner state for PyRerankerAdapter.
struct RerankerInner {
    /// Python reranker object.
    backend: BackendRef,
    /// Semaphore for concurrency control.
    semaphore: Semaphore,
}

/// Adapter that wraps a Python reranker class.
///
/// This adapter implements [`Rerank`] by delegating to Python methods.
/// The Python class must implement:
/// - `rerank(query: str, candidates: list[dict]) -> list[dict]` (sync or async)
///
/// Each candidate dict should have:
/// - `chunk_id: str`
/// - `document: str`
/// - `text: str`
/// - `page_number: int`
/// - `section: str | None`
/// - `initial_score: float`
///
/// Each result dict should have:
/// - `chunk_id: str`
/// - `relevance: float`
/// - `yes_logprob: float | None` (optional)
/// - `no_logprob: float | None` (optional)
///
/// # Example (Python)
///
/// ```python
/// class MyReranker:
///     def rerank(self, query: str, candidates: list[dict]) -> list[dict]:
///         # Your reranking logic
///         return [{"chunk_id": c["chunk_id"], "relevance": 0.9} for c in candidates]
///
/// engine = RagEngine.create("/tmp/index", reranker=MyReranker())
/// ```
pub struct PyRerankerAdapter {
    inner: Arc<RerankerInner>,
}

#[allow(dead_code)]
impl PyRerankerAdapter {
    /// Create a new adapter wrapping a Python reranker.
    ///
    /// # Arguments
    ///
    /// * `_py` - Python GIL token
    /// * `backend` - Python object implementing reranker protocol
    /// * `max_concurrency` - Maximum concurrent Python calls (default: 4)
    ///
    /// # Errors
    ///
    /// Returns error if the Python object doesn't have the required `rerank` method.
    pub fn new(
        _py: Python<'_>,
        backend: &Bound<'_, PyAny>,
        max_concurrency: Option<usize>,
    ) -> PyResult<Self> {
        // Validate required method exists
        if !backend.hasattr("rerank")? {
            return Err(pyo3::exceptions::PyAttributeError::new_err(
                "Reranker must have rerank() method",
            ));
        }

        let concurrency = max_concurrency.unwrap_or(DEFAULT_CONCURRENCY);

        Ok(Self {
            inner: Arc::new(RerankerInner {
                backend: BackendRef(backend.clone().unbind()),
                semaphore: Semaphore::new(concurrency),
            }),
        })
    }

    /// Create adapter with default concurrency.
    pub fn with_defaults(py: Python<'_>, backend: &Bound<'_, PyAny>) -> PyResult<Self> {
        Self::new(py, backend, None)
    }

    /// Call Python rerank method.
    async fn call_rerank(
        inner: Arc<RerankerInner>,
        query: String,
        candidates: Vec<RerankerCandidate>,
    ) -> Result<Vec<RerankedResult>, RerankError> {
        // Acquire semaphore permit
        let _permit = inner
            .semaphore
            .acquire()
            .await
            .map_err(|e| RerankError::Error(format!("Semaphore error: {}", e)))?;

        // Call Python and check for coroutine
        let call_result = Python::with_gil(|py| -> PyResult<RerankCallResult> {
            let backend = inner.backend.0.bind(py);
            let py_candidates = candidates_to_py(py, &candidates)?;
            let result = backend.call_method1("rerank", (&query, py_candidates))?;

            // Check if result is a coroutine
            if is_coroutine(py, &result)? {
                let future = pyo3_async_runtimes::tokio::into_future(result)?;
                Ok(RerankCallResult::Async(Box::pin(future)))
            } else {
                Ok(RerankCallResult::Sync(extract_rerank_results(py, &result)?))
            }
        })
        .map_err(pyerr_to_rerank_error)?;

        match call_result {
            RerankCallResult::Sync(results) => Ok(results),
            RerankCallResult::Async(future) => {
                let result = future.await.map_err(pyerr_to_rerank_error)?;
                Python::with_gil(|py| extract_rerank_results(py, result.bind(py)))
                    .map_err(pyerr_to_rerank_error)
            }
        }
    }
}

impl Rerank for PyRerankerAdapter {
    fn rerank(
        &self,
        query: &str,
        candidates: &[RerankerCandidate],
    ) -> impl Future<Output = Result<Vec<RerankedResult>, RerankError>> + Send {
        let query = query.to_string();
        let candidates = candidates.to_vec();
        let inner = Arc::clone(&self.inner);
        async move { Self::call_rerank(inner, query, candidates).await }
    }
}

/// Convert RerankerCandidates to Python list of dicts.
fn candidates_to_py<'py>(
    py: Python<'py>,
    candidates: &[RerankerCandidate],
) -> PyResult<Bound<'py, PyList>> {
    let py_list = PyList::empty(py);
    for candidate in candidates {
        let dict = PyDict::new(py);
        dict.set_item("chunk_id", &candidate.chunk_id)?;
        dict.set_item("document", &candidate.document)?;
        dict.set_item("text", &candidate.text)?;
        dict.set_item("page_number", candidate.page_number)?;
        dict.set_item("section", &candidate.section)?;
        dict.set_item("initial_score", candidate.initial_score)?;
        py_list.append(dict)?;
    }
    Ok(py_list)
}

/// Extract reranked results from Python result.
fn extract_rerank_results(
    _py: Python<'_>,
    result: &Bound<'_, PyAny>,
) -> PyResult<Vec<RerankedResult>> {
    let list = result.downcast::<PyList>()?;
    let mut results = Vec::with_capacity(list.len());

    for item in list.iter() {
        let dict = item.downcast::<PyDict>()?;

        let chunk_id: String = dict
            .get_item("chunk_id")?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("missing 'chunk_id'"))?
            .extract()?;

        let relevance: f32 = dict
            .get_item("relevance")?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("missing 'relevance'"))?
            .extract()?;

        let yes_logprob: Option<f64> = dict
            .get_item("yes_logprob")?
            .map(|v| v.extract())
            .transpose()?;

        let no_logprob: Option<f64> = dict
            .get_item("no_logprob")?
            .map(|v| v.extract())
            .transpose()?;

        results.push(RerankedResult {
            chunk_id,
            relevance,
            yes_logprob,
            no_logprob,
        });
    }

    Ok(results)
}

/// Convert PyErr to RerankError.
fn pyerr_to_rerank_error(e: PyErr) -> RerankError {
    Python::with_gil(|py| {
        let err_str = e.to_string();
        if e.is_instance_of::<pyo3::exceptions::PyConnectionError>(py) {
            RerankError::Unavailable(err_str)
        } else {
            RerankError::Error(err_str)
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adapter_requires_methods() {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            // Empty class should fail
            let empty = py.eval(c"type('Empty', (), {})()", None, None).unwrap();
            let result = PyEmbeddingBackendAdapter::new(py, &empty, None);
            assert!(result.is_err());
        });
    }

    #[test]
    fn test_adapter_requires_model_id() {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            // Class with dimension and embed but missing model_id
            let code = c"type('NoModelId', (), {'dimension': lambda self: 768, 'embed': lambda self, text: [0.0] * 768})()";
            let obj = py.eval(code, None, None).unwrap();
            let result = PyEmbeddingBackendAdapter::new(py, &obj, None);
            match result {
                Err(e) => assert!(e.to_string().contains("model_id")),
                Ok(_) => panic!("Expected error for missing model_id"),
            }
        });
    }

    #[test]
    fn test_adapter_requires_dimension() {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            // Class with model_id and embed but missing dimension
            let code = c"type('NoDimension', (), {'model_id': lambda self: 'test', 'embed': lambda self, text: [0.0] * 768})()";
            let obj = py.eval(code, None, None).unwrap();
            let result = PyEmbeddingBackendAdapter::new(py, &obj, None);
            match result {
                Err(e) => assert!(e.to_string().contains("dimension")),
                Ok(_) => panic!("Expected error for missing dimension"),
            }
        });
    }

    #[test]
    fn test_adapter_requires_embed() {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            // Class with model_id and dimension but missing embed
            let code = c"type('NoEmbed', (), {'model_id': lambda self: 'test', 'dimension': lambda self: 768})()";
            let obj = py.eval(code, None, None).unwrap();
            let result = PyEmbeddingBackendAdapter::new(py, &obj, None);
            match result {
                Err(e) => assert!(e.to_string().contains("embed")),
                Ok(_) => panic!("Expected error for missing embed"),
            }
        });
    }

    #[test]
    fn test_adapter_creation_caches_values() {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            // Valid backend class
            let code = c"type('ValidBackend', (), {'model_id': lambda self: 'test-model', 'dimension': lambda self: 384, 'embed': lambda self, text: [0.0] * 384})()";
            let obj = py.eval(code, None, None).unwrap();
            let adapter = PyEmbeddingBackendAdapter::new(py, &obj, None).unwrap();

            // Values should be cached in inner
            assert_eq!(adapter.inner.model_id, "test-model");
            assert_eq!(adapter.inner.dimension, 384);
            assert!(!adapter.inner.has_batch); // No embed_batch method
        });
    }

    #[test]
    fn test_adapter_detects_batch_method() {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            // Backend with embed_batch method
            let code = c"type('BatchBackend', (), {'model_id': lambda self: 'batch-model', 'dimension': lambda self: 512, 'embed': lambda self, text: [0.0] * 512, 'embed_batch': lambda self, texts: [[0.0] * 512 for _ in texts]})()";
            let obj = py.eval(code, None, None).unwrap();
            let adapter = PyEmbeddingBackendAdapter::new(py, &obj, None).unwrap();

            assert!(adapter.inner.has_batch);
        });
    }

    #[test]
    fn test_adapter_custom_concurrency() {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            let code = c"type('Backend', (), {'model_id': lambda self: 'test', 'dimension': lambda self: 768, 'embed': lambda self, text: [0.0] * 768})()";
            let obj = py.eval(code, None, None).unwrap();
            let adapter = PyEmbeddingBackendAdapter::new(py, &obj, Some(8)).unwrap();

            // Semaphore should have 8 permits (internal check via available permits)
            assert_eq!(adapter.inner.semaphore.available_permits(), 8);
        });
    }

    #[test]
    fn test_adapter_implements_embedding_backend_trait() {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            let code = c"type('Backend', (), {'model_id': lambda self: 'trait-test', 'dimension': lambda self: 256, 'embed': lambda self, text: [0.0] * 256})()";
            let obj = py.eval(code, None, None).unwrap();
            let adapter = PyEmbeddingBackendAdapter::new(py, &obj, None).unwrap();

            // Test EmbeddingBackend trait methods (sync parts)
            assert_eq!(adapter.model_id(), "trait-test");
            assert_eq!(adapter.dimension(), 256);
        });
    }

    #[tokio::test]
    async fn test_adapter_embed_sync_python() {
        pyo3::prepare_freethreaded_python();

        // Create adapter
        let adapter = Python::with_gil(|py| {
            // Python backend that returns deterministic embeddings based on text length
            let code = c"type('SyncBackend', (), {'model_id': lambda self: 'sync-test', 'dimension': lambda self: 4, 'embed': lambda self, text: [float(len(text)), 0.5, 0.5, 0.5]})()";
            let obj = py.eval(code, None, None).unwrap();
            PyEmbeddingBackendAdapter::new(py, &obj, None).unwrap()
        });

        // Call embed via EmbeddingBackend trait
        let embedding = adapter.embed("hello").await.unwrap();

        // Should have 4 dimensions
        assert_eq!(embedding.len(), 4);
        // First element should be text length (5.0 for "hello")
        assert!((embedding[0] - 5.0).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_adapter_embed_batch_sync_python() {
        pyo3::prepare_freethreaded_python();

        // Create adapter with batch support
        let adapter = Python::with_gil(|py| {
            let code = c"type('BatchBackend', (), {'model_id': lambda self: 'batch-test', 'dimension': lambda self: 3, 'embed': lambda self, text: [float(len(text)), 0.5, 0.5], 'embed_batch': lambda self, texts: [[float(len(t)), 0.5, 0.5] for t in texts]})()";
            let obj = py.eval(code, None, None).unwrap();
            PyEmbeddingBackendAdapter::new(py, &obj, None).unwrap()
        });

        // Call embed_batch
        let texts = vec!["hi".to_string(), "hello".to_string(), "goodbye".to_string()];
        let embeddings = adapter.embed_batch(&texts).await.unwrap();

        // Should have 3 embeddings
        assert_eq!(embeddings.len(), 3);
        // First values should be text lengths
        assert!((embeddings[0][0] - 2.0).abs() < 0.001); // "hi"
        assert!((embeddings[1][0] - 5.0).abs() < 0.001); // "hello"
        assert!((embeddings[2][0] - 7.0).abs() < 0.001); // "goodbye"
    }

    #[tokio::test]
    async fn test_adapter_embed_batch_fallback_without_batch_method() {
        pyo3::prepare_freethreaded_python();

        // Create adapter WITHOUT batch support
        let adapter = Python::with_gil(|py| {
            let code = c"type('NoBatchBackend', (), {'model_id': lambda self: 'no-batch', 'dimension': lambda self: 2, 'embed': lambda self, text: [float(len(text)), 1.0]})()";
            let obj = py.eval(code, None, None).unwrap();
            PyEmbeddingBackendAdapter::new(py, &obj, None).unwrap()
        });

        // Should not have batch
        assert!(!adapter.inner.has_batch);

        // Call embed_batch - should fall back to sequential embed calls
        let texts = vec!["a".to_string(), "bb".to_string()];
        let embeddings = adapter.embed_batch(&texts).await.unwrap();

        assert_eq!(embeddings.len(), 2);
        assert!((embeddings[0][0] - 1.0).abs() < 0.001); // "a"
        assert!((embeddings[1][0] - 2.0).abs() < 0.001); // "bb"
    }

    // =========================================================================
    // PyRerankerAdapter Tests
    // =========================================================================

    #[test]
    fn test_reranker_requires_rerank_method() {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            // Empty class should fail
            let empty = py.eval(c"type('Empty', (), {})()", None, None).unwrap();
            let result = PyRerankerAdapter::new(py, &empty, None);
            match result {
                Err(e) => assert!(e.to_string().contains("rerank")),
                Ok(_) => panic!("Expected error for missing rerank method"),
            }
        });
    }

    #[test]
    fn test_reranker_creation_succeeds() {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            // Valid reranker class
            let code = c"type('ValidReranker', (), {'rerank': lambda self, query, candidates: [{'chunk_id': c['chunk_id'], 'relevance': 0.9} for c in candidates]})()";
            let obj = py.eval(code, None, None).unwrap();
            let adapter = PyRerankerAdapter::new(py, &obj, None);
            assert!(adapter.is_ok());
        });
    }

    #[test]
    fn test_reranker_custom_concurrency() {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            let code = c"type('Reranker', (), {'rerank': lambda self, query, candidates: []})()";
            let obj = py.eval(code, None, None).unwrap();
            let adapter = PyRerankerAdapter::new(py, &obj, Some(2)).unwrap();
            assert_eq!(adapter.inner.semaphore.available_permits(), 2);
        });
    }

    #[tokio::test]
    async fn test_reranker_rerank_sync_python() {
        pyo3::prepare_freethreaded_python();

        // Create reranker
        let adapter = Python::with_gil(|py| {
            // Python reranker that returns relevance based on text length
            let code = c"type('SyncReranker', (), {'rerank': lambda self, query, candidates: [{'chunk_id': c['chunk_id'], 'relevance': float(len(c['text'])) / 100.0} for c in candidates]})()";
            let obj = py.eval(code, None, None).unwrap();
            PyRerankerAdapter::new(py, &obj, None).unwrap()
        });

        // Create test candidates
        let candidates = vec![
            RerankerCandidate {
                chunk_id: "chunk1".to_string(),
                document: "doc1".to_string(),
                text: "short".to_string(),
                page_number: 1,
                section: None,
                initial_score: 0.5,
            },
            RerankerCandidate {
                chunk_id: "chunk2".to_string(),
                document: "doc1".to_string(),
                text: "this is a longer text".to_string(),
                page_number: 2,
                section: Some("Section A".to_string()),
                initial_score: 0.3,
            },
        ];

        // Call rerank via Rerank trait
        let results = adapter.rerank("test query", &candidates).await.unwrap();

        // Should have 2 results
        assert_eq!(results.len(), 2);
        // First result: "short" = 5 chars -> 0.05 relevance
        assert_eq!(results[0].chunk_id, "chunk1");
        assert!((results[0].relevance - 0.05).abs() < 0.001);
        // Second result: "this is a longer text" = 21 chars -> 0.21 relevance
        assert_eq!(results[1].chunk_id, "chunk2");
        assert!((results[1].relevance - 0.21).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_reranker_with_logprobs() {
        pyo3::prepare_freethreaded_python();

        // Create reranker that returns logprobs
        let adapter = Python::with_gil(|py| {
            let code = c"type('LogprobReranker', (), {'rerank': lambda self, query, candidates: [{'chunk_id': c['chunk_id'], 'relevance': 0.8, 'yes_logprob': -0.5, 'no_logprob': -2.0} for c in candidates]})()";
            let obj = py.eval(code, None, None).unwrap();
            PyRerankerAdapter::new(py, &obj, None).unwrap()
        });

        let candidates = vec![RerankerCandidate {
            chunk_id: "c1".to_string(),
            document: "doc".to_string(),
            text: "test".to_string(),
            page_number: 1,
            section: None,
            initial_score: 0.5,
        }];

        let results = adapter.rerank("query", &candidates).await.unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].chunk_id, "c1");
        assert!((results[0].relevance - 0.8).abs() < 0.001);
        assert_eq!(results[0].yes_logprob, Some(-0.5));
        assert_eq!(results[0].no_logprob, Some(-2.0));
    }
}
