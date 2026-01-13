//! PyRagEngine: Python wrapper for rag-core's RagEngine.
//!
//! This module provides the main entry point for Python applications.
//! It wraps the Rust RagEngine with PyO3 bindings.

use once_cell::sync::Lazy;
use pyo3::prelude::*;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::runtime::Runtime;
use tokio::sync::Mutex;

use crate::adapters::{PyEmbeddingBackendAdapter, PyRerankerAdapter};
use crate::errors::engine_error_to_pyerr;
use crate::mock_backend::MockEmbeddingBackend;
use crate::types::{PyEngineStats, PyQuerySpec, PySearchResult};
use rag_core::{BoxedEmbedder, BoxedReranker};

/// Shared Tokio runtime for async operations.
///
/// Using a lazy static runtime avoids creating multiple thread pools
/// and aligns with pyo3-async-runtimes patterns.
#[allow(dead_code)] // Used in Phase 2 Task 2.2 for search operations
static TOKIO_RUNTIME: Lazy<Runtime> = Lazy::new(|| {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .expect("Failed to create Tokio runtime")
});

/// Type alias for the dynamic engine type using boxed trait objects.
/// This allows the engine to work with any embedding backend or reranker
/// that implements the required traits (via adapters for Python backends).
type DynamicEngine = rag_core::RagEngine<BoxedEmbedder, BoxedReranker>;

/// Helper macro for catch_unwind at FFI boundary.
///
/// Converts Rust panics into Python exceptions to prevent UB.
macro_rules! catch_panic {
    ($py:expr, $body:expr) => {{
        match catch_unwind(AssertUnwindSafe(|| $body)) {
            Ok(result) => result,
            Err(panic) => {
                let msg = if let Some(s) = panic.downcast_ref::<&str>() {
                    s.to_string()
                } else if let Some(s) = panic.downcast_ref::<String>() {
                    s.clone()
                } else {
                    "Unknown panic".to_string()
                };
                Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Internal error (panic): {}",
                    msg
                )))
            }
        }
    }};
}

/// The main RAG engine for Python.
///
/// Provides semantic search over documents with configurable embedding
/// and reranking backends.
///
/// # Examples
///
/// ```python
/// from ragcore import RagEngine
///
/// # Create a mock engine for testing
/// engine = RagEngine.create_mock("/tmp/index")
/// stats = engine.stats()
/// print(f"Documents: {stats.document_count}")
///
/// # Create an engine with a Python backend
/// class MyEmbedder:
///     def model_id(self) -> str:
///         return "my-model"
///     def dimension(self) -> int:
///         return 768
///     def embed(self, text: str) -> list[float]:
///         return [0.0] * 768
///
/// engine = RagEngine.create("/tmp/index", backend=MyEmbedder())
/// ```
/// Construction mode for pickle support.
enum ConstructionMode {
    /// Mock backend with specified dimension
    Mock { dimension: usize },
    /// Python backend (stores references for pickle reconstruction)
    PythonBackend {
        backend: Py<PyAny>,
        reranker: Option<Py<PyAny>>,
    },
}

#[pyclass(name = "RagEngine", module = "ragcore")]
pub struct PyRagEngine {
    /// Directory for storing the index
    index_dir: PathBuf,
    /// The underlying rag-core engine with dynamic dispatch.
    /// Using Arc<tokio::sync::Mutex> to support both sync (blocking_lock)
    /// and async (.lock().await) access patterns.
    engine: Arc<Mutex<DynamicEngine>>,
    /// How the engine was constructed (for pickle support)
    construction_mode: ConstructionMode,
}

#[pymethods]
impl PyRagEngine {
    /// Create a new engine with a Python embedding backend.
    ///
    /// This is the primary constructor for production use. You provide
    /// your own embedding backend (and optionally a reranker) that will
    /// be called during document indexing and search operations.
    ///
    /// # Arguments
    /// * `index_dir` - Directory for storing the index
    /// * `backend` - Python object implementing embedding backend protocol:
    ///   - `model_id() -> str`
    ///   - `dimension() -> int`
    ///   - `embed(text: str) -> list[float]` (sync or async)
    ///   - `embed_batch(texts: list[str]) -> list[list[float]]` (optional)
    /// * `reranker` - Optional Python object implementing reranker protocol:
    ///   - `rerank(query: str, candidates: list[dict]) -> list[dict]` (sync or async)
    ///
    /// # Returns
    /// A new RagEngine instance with the provided backends.
    ///
    /// # Errors
    /// Raises AttributeError if backend/reranker doesn't have required methods.
    ///
    /// # Examples
    ///
    /// ```python
    /// class MyEmbedder:
    ///     def model_id(self) -> str:
    ///         return "my-model"
    ///     def dimension(self) -> int:
    ///         return 768
    ///     def embed(self, text: str) -> list[float]:
    ///         return [0.0] * 768
    ///
    /// engine = RagEngine.create("/tmp/index", backend=MyEmbedder())
    /// ```
    #[staticmethod]
    #[pyo3(signature = (index_dir, backend, reranker=None))]
    fn create(
        py: Python<'_>,
        index_dir: &str,
        backend: &Bound<'_, PyAny>,
        reranker: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        catch_panic!(py, {
            // Wrap Python backend in adapter
            let adapter = PyEmbeddingBackendAdapter::new(py, backend, None)?;
            let boxed_embedder: BoxedEmbedder = Arc::new(adapter);

            // Wrap reranker if provided, otherwise use no-op
            let boxed_reranker: BoxedReranker = if let Some(r) = reranker {
                let reranker_adapter = PyRerankerAdapter::new(py, r, None)?;
                Arc::new(reranker_adapter)
            } else {
                // No-op reranker: () implements Rerank returning empty vec
                Arc::new(()) as BoxedReranker
            };

            let engine = rag_core::RagEngine::with_reranker(boxed_embedder, boxed_reranker);

            // Store references for pickle reconstruction
            let construction_mode = ConstructionMode::PythonBackend {
                backend: backend.clone().unbind(),
                reranker: reranker.map(|r| r.clone().unbind()),
            };

            Ok(Self {
                index_dir: PathBuf::from(index_dir),
                engine: Arc::new(Mutex::new(engine)),
                construction_mode,
            })
        })
    }

    /// Create a new engine with a mock backend (for testing).
    ///
    /// This constructor is primarily for testing and development.
    /// It creates an engine with deterministic hash-based embeddings.
    ///
    /// # Arguments
    /// * `index_dir` - Directory for storing the index
    /// * `dimension` - Embedding dimension (default: 768)
    ///
    /// # Returns
    /// A new RagEngine instance with mock embeddings.
    ///
    /// # Examples
    ///
    /// ```python
    /// engine = RagEngine.create_mock("/tmp/index", dimension=384)
    /// assert engine.stats().embedding_dimension == 384
    /// ```
    #[staticmethod]
    #[pyo3(signature = (index_dir, dimension=768))]
    fn create_mock(_py: Python<'_>, index_dir: &str, dimension: usize) -> PyResult<Self> {
        catch_panic!(_py, {
            // Wrap mock backend in BoxedEmbedder for dynamic dispatch
            let mock_backend = MockEmbeddingBackend::new(dimension);
            let boxed_embedder: BoxedEmbedder = Arc::new(mock_backend);

            // No-op reranker
            let boxed_reranker: BoxedReranker = Arc::new(()) as BoxedReranker;

            let engine = rag_core::RagEngine::with_reranker(boxed_embedder, boxed_reranker);

            Ok(Self {
                index_dir: PathBuf::from(index_dir),
                engine: Arc::new(Mutex::new(engine)),
                construction_mode: ConstructionMode::Mock { dimension },
            })
        })
    }

    /// Get engine statistics.
    ///
    /// Returns health status and statistics about the engine including
    /// document count, chunk count, and embedding model info.
    ///
    /// # Returns
    /// EngineStats with current engine state.
    fn stats(&self, _py: Python<'_>) -> PyResult<PyEngineStats> {
        catch_panic!(_py, {
            let engine = self.engine.blocking_lock();
            let health = engine.health();

            Ok(PyEngineStats {
                document_count: health.document_count,
                chunk_count: health.chunk_count,
                embedding_model: health.embedding_model,
                embedding_dimension: health.embedding_dim,
            })
        })
    }

    /// Save the engine state to disk.
    ///
    /// Persists all indexed documents and embeddings to the index directory.
    /// Releases the GIL during disk I/O operations.
    ///
    /// # Errors
    /// Raises IndexError if persistence fails.
    fn save(&self, py: Python<'_>) -> PyResult<()> {
        let index_dir = self.index_dir.clone();
        let engine = Arc::clone(&self.engine);

        // Release GIL during potentially slow disk I/O
        py.allow_threads(move || {
            let engine = engine.blocking_lock();
            engine
                .save_to_dir(&index_dir)
                .map_err(engine_error_to_pyerr)
        })
    }

    /// Load engine state from disk.
    ///
    /// Restores previously indexed documents and embeddings from the index directory.
    /// Releases the GIL during disk I/O operations.
    ///
    /// # Errors
    /// Raises IndexError if loading fails.
    fn load(&self, py: Python<'_>) -> PyResult<()> {
        let index_dir = self.index_dir.clone();
        let engine = Arc::clone(&self.engine);

        // Release GIL during potentially slow disk I/O
        py.allow_threads(move || {
            let mut engine = engine.blocking_lock();
            engine
                .load_from_dir(&index_dir)
                .map_err(engine_error_to_pyerr)
        })
    }

    /// List all indexed documents.
    ///
    /// # Returns
    /// List of document names currently in the index.
    fn list_documents(&self, _py: Python<'_>) -> PyResult<Vec<String>> {
        catch_panic!(_py, {
            let engine = self.engine.blocking_lock();
            Ok(engine.list_documents())
        })
    }

    /// Check if the engine needs reindexing.
    ///
    /// Returns true if the embedding model has changed since the last index.
    fn needs_reindex(&self, _py: Python<'_>) -> PyResult<bool> {
        catch_panic!(_py, {
            let engine = self.engine.blocking_lock();
            Ok(engine.needs_reindex())
        })
    }

    /// Add or update a document in the index.
    ///
    /// The document text is automatically chunked and embedded using the
    /// engine's embedding backend. If a document with the same name exists,
    /// it is replaced.
    ///
    /// # Arguments
    /// * `name` - Document name/identifier (e.g., "paper.pdf")
    /// * `text` - Full text content of the document
    /// * `content_hash` - Optional content hash for change detection.
    ///   If provided and matches the stored hash, the document is skipped.
    ///
    /// # Returns
    /// Number of chunks created (0 if skipped due to unchanged hash or empty text).
    ///
    /// # Errors
    /// Raises EmbeddingError if embedding fails.
    ///
    /// # Examples
    ///
    /// ```python
    /// chunks = engine.upsert_document("doc.txt", "Hello world!")
    /// print(f"Created {chunks} chunks")
    /// ```
    #[pyo3(signature = (name, text, content_hash=None))]
    fn upsert_document(
        &self,
        py: Python<'_>,
        name: &str,
        text: &str,
        content_hash: Option<String>,
    ) -> PyResult<usize> {
        let name = name.to_string();
        let text = text.to_string();
        let engine = Arc::clone(&self.engine);

        // Release GIL and perform async embedding + upsert
        py.allow_threads(move || {
            let mut engine = engine.blocking_lock();
            runtime()
                .block_on(engine.upsert_document(&name, &text, content_hash))
                .map_err(engine_error_to_pyerr)
        })
    }

    /// Remove a document from the index.
    ///
    /// Removes all chunks associated with the document name.
    ///
    /// # Arguments
    /// * `name` - Document name/identifier to remove
    ///
    /// # Returns
    /// Number of chunks removed (0 if document was not found).
    ///
    /// # Errors
    /// Raises IndexError if removal fails.
    ///
    /// # Examples
    ///
    /// ```python
    /// removed = engine.remove_document("old_doc.txt")
    /// if removed > 0:
    ///     print(f"Removed {removed} chunks")
    /// ```
    fn remove_document(&self, py: Python<'_>, name: &str) -> PyResult<usize> {
        let name = name.to_string();
        let engine = Arc::clone(&self.engine);

        py.allow_threads(move || {
            let mut engine = engine.blocking_lock();
            engine.remove_document(&name).map_err(engine_error_to_pyerr)
        })
    }

    /// Search for documents matching the query.
    ///
    /// Performs semantic search using the engine's embedding backend,
    /// returning results sorted by relevance score.
    ///
    /// # Arguments
    /// * `query` - The search query string
    /// * `top_k` - Maximum number of results to return (default: 10)
    /// * `spec` - Optional QuerySpec for advanced configuration
    ///
    /// # Returns
    /// List of SearchResult objects, sorted by score descending.
    ///
    /// # Errors
    /// Raises EmbeddingError if embedding fails.
    /// Raises ValidationError if parameters are invalid.
    ///
    /// # Examples
    ///
    /// ```python
    /// results = engine.search("machine learning", top_k=5)
    /// for r in results:
    ///     print(f"{r.document}: {r.score:.2f}")
    /// ```
    #[pyo3(signature = (query, top_k=10, spec=None))]
    fn search(
        &self,
        py: Python<'_>,
        query: &str,
        top_k: usize,
        spec: Option<&PyQuerySpec>,
    ) -> PyResult<Vec<PySearchResult>> {
        // Validate parameters
        if query.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Query cannot be empty",
            ));
        }

        // Build QuerySpec from optional spec or Python defaults.
        // Note: We explicitly set diversity_factor=0.0 when no spec provided,
        // matching Python QuerySpec defaults rather than rag-core defaults.
        let query_spec = if let Some(s) = spec {
            s.to_query_spec(query).with_top_k(top_k)
        } else {
            rag_core::QuerySpec::new(query)
                .with_top_k(top_k)
                .with_diversity(0.0) // Match Python QuerySpec default
        };

        let engine = Arc::clone(&self.engine);

        // Release GIL and perform async search
        py.allow_threads(move || {
            let engine = engine.blocking_lock();

            // Use diversity search if diversity_factor > 0
            let search_result = if query_spec.diversity_factor > 0.0 {
                runtime().block_on(engine.search_with_diversity(
                    &query_spec.query,
                    query_spec.top_k,
                    query_spec.diversity_factor,
                    query_spec.weights.clone(),
                ))
            } else {
                runtime().block_on(engine.search(
                    &query_spec.query,
                    query_spec.top_k,
                    query_spec.weights.clone(),
                ))
            };

            search_result
                .map(|rs| rs.into_iter().map(PySearchResult::from).collect())
                .map_err(engine_error_to_pyerr)
        })
    }

    /// Async version of search.
    ///
    /// Returns a Python coroutine that can be awaited. This is the preferred
    /// method when calling from async Python code (asyncio).
    ///
    /// # Arguments
    /// * `query` - The search query string
    /// * `top_k` - Maximum number of results to return (default: 10)
    /// * `spec` - Optional QuerySpec for advanced configuration
    ///
    /// # Returns
    /// A coroutine that resolves to List[SearchResult].
    ///
    /// # Examples
    ///
    /// ```python
    /// async def search_async(engine):
    ///     results = await engine.asearch("machine learning", top_k=5)
    ///     return results
    /// ```
    #[pyo3(signature = (query, top_k=10, spec=None))]
    fn asearch<'py>(
        &self,
        py: Python<'py>,
        query: &str,
        top_k: usize,
        spec: Option<&PyQuerySpec>,
    ) -> PyResult<Bound<'py, PyAny>> {
        // Validate parameters
        if query.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Query cannot be empty",
            ));
        }

        // Build QuerySpec from optional spec or Python defaults.
        let query_spec = if let Some(s) = spec {
            s.to_query_spec(query).with_top_k(top_k)
        } else {
            rag_core::QuerySpec::new(query)
                .with_top_k(top_k)
                .with_diversity(0.0)
        };

        let engine = Arc::clone(&self.engine);

        // Convert Rust future to Python coroutine.
        // We use spawn_blocking because RagEngine::search returns a future
        // that borrows from &self, which doesn't satisfy 'static bounds.
        // spawn_blocking moves the work to a thread pool where we can safely
        // use blocking_lock + block_on.
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            tokio::task::spawn_blocking(move || {
                let engine = engine.blocking_lock();

                // Use diversity search if diversity_factor > 0
                let search_result = if query_spec.diversity_factor > 0.0 {
                    runtime().block_on(engine.search_with_diversity(
                        &query_spec.query,
                        query_spec.top_k,
                        query_spec.diversity_factor,
                        query_spec.weights.clone(),
                    ))
                } else {
                    runtime().block_on(engine.search(
                        &query_spec.query,
                        query_spec.top_k,
                        query_spec.weights.clone(),
                    ))
                };

                search_result
                    .map(|rs| rs.into_iter().map(PySearchResult::from).collect::<Vec<_>>())
                    .map_err(engine_error_to_pyerr)
            })
            .await
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
        })
    }

    /// Async version of upsert_document.
    ///
    /// Returns a Python coroutine that can be awaited. This is the preferred
    /// method when calling from async Python code (asyncio).
    ///
    /// # Arguments
    /// * `name` - Document name/identifier (e.g., "paper.pdf")
    /// * `text` - Full text content of the document
    /// * `content_hash` - Optional content hash for change detection.
    ///
    /// # Returns
    /// A coroutine that resolves to the number of chunks created.
    ///
    /// # Examples
    ///
    /// ```python
    /// async def add_doc_async(engine):
    ///     chunks = await engine.aupsert_document("doc.txt", "Hello world!")
    ///     return chunks
    /// ```
    #[pyo3(signature = (name, text, content_hash=None))]
    fn aupsert_document<'py>(
        &self,
        py: Python<'py>,
        name: &str,
        text: &str,
        content_hash: Option<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let name = name.to_string();
        let text = text.to_string();
        let engine = Arc::clone(&self.engine);

        // Convert Rust future to Python coroutine.
        // Uses spawn_blocking pattern for same reason as asearch.
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            tokio::task::spawn_blocking(move || {
                let mut engine = engine.blocking_lock();
                runtime()
                    .block_on(engine.upsert_document(&name, &text, content_hash))
                    .map_err(engine_error_to_pyerr)
            })
            .await
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
        })
    }

    /// Get the index directory path.
    #[getter]
    fn index_dir(&self) -> String {
        self.index_dir.to_string_lossy().to_string()
    }

    /// Pickle support via __reduce__.
    ///
    /// Returns (callable, args) tuple that reconstructs the engine.
    /// For mock backends: (RagEngine.create_mock, (index_dir, dimension))
    /// For Python backends: (RagEngine.create, (index_dir, backend, reranker))
    ///
    /// Note: Python backends must themselves be picklable for this to work.
    /// If the backend is not picklable, pickle.dumps() will raise.
    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        let cls = py.get_type::<Self>();
        let index_dir = self.index_dir.to_string_lossy().to_string();

        match &self.construction_mode {
            ConstructionMode::Mock { dimension } => {
                // Return (RagEngine.create_mock, (index_dir, dimension))
                let callable = cls.getattr("create_mock")?;
                let args = (index_dir, *dimension).into_pyobject(py)?;
                Ok((callable, args.into_any()))
            }
            ConstructionMode::PythonBackend { backend, reranker } => {
                // Return (RagEngine.create, (index_dir, backend, reranker))
                let callable = cls.getattr("create")?;
                let args = (
                    index_dir,
                    backend.clone_ref(py),
                    reranker.as_ref().map(|r| r.clone_ref(py)),
                )
                    .into_pyobject(py)?;
                Ok((callable, args.into_any()))
            }
        }
    }

    fn __repr__(&self) -> String {
        format!("RagEngine(index_dir='{}')", self.index_dir.display())
    }
}

/// Get a reference to the shared Tokio runtime.
///
/// This is used by async methods to execute futures.
#[allow(dead_code)] // Used in Phase 2 Task 2.2 for search operations
pub(crate) fn runtime() -> &'static Runtime {
    &TOKIO_RUNTIME
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    /// Helper to create a DynamicEngine with MockEmbeddingBackend.
    fn create_mock_dynamic_engine(dimension: usize) -> DynamicEngine {
        let mock_backend = MockEmbeddingBackend::new(dimension);
        let boxed_embedder: BoxedEmbedder = Arc::new(mock_backend);
        let boxed_reranker: BoxedReranker = Arc::new(()) as BoxedReranker;
        rag_core::RagEngine::with_optional_reranker(boxed_embedder, Some(boxed_reranker), rag_core::RagConfig::default())
    }

    #[test]
    fn test_mock_engine_creation() {
        let _dir = tempdir().unwrap();
        let engine = create_mock_dynamic_engine(768);
        let health = engine.health();

        assert_eq!(health.chunk_count, 0);
        assert_eq!(health.document_count, 0);
        assert_eq!(health.embedding_dim, 768);
        assert!(health.embedding_model.contains("mock"));
    }

    #[test]
    fn test_mock_engine_custom_dimension() {
        let engine = create_mock_dynamic_engine(384);
        let health = engine.health();

        assert_eq!(health.embedding_dim, 384);
    }

    #[test]
    fn test_runtime_accessible() {
        let rt = runtime();
        rt.block_on(async {
            // Simple async test
            tokio::time::sleep(std::time::Duration::from_millis(1)).await;
        });
    }
}
