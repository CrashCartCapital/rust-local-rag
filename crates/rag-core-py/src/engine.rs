//! PyRagEngine: Python wrapper for rag-core's RagEngine.
//!
//! This module provides the main entry point for Python applications.
//! It wraps the Rust RagEngine with PyO3 bindings.

use once_cell::sync::Lazy;
use pyo3::prelude::*;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;
use std::sync::Mutex;
use tokio::runtime::Runtime;

use crate::errors::engine_error_to_pyerr;
use crate::mock_backend::MockEmbeddingBackend;
use crate::types::{PyEngineStats, PyQuerySpec, PySearchResult};

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

/// Type alias for the concrete engine type using MockEmbeddingBackend.
/// In Phase 3, this will be replaced with trait object support.
type MockEngine = rag_core::RagEngine<MockEmbeddingBackend, ()>;

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
/// ```
#[pyclass(name = "RagEngine", module = "ragcore")]
pub struct PyRagEngine {
    /// Directory for storing the index
    index_dir: PathBuf,
    /// The underlying rag-core engine (using mock backend for Phase 2)
    engine: Mutex<MockEngine>,
}

#[pymethods]
impl PyRagEngine {
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
            let backend = MockEmbeddingBackend::new(dimension);
            let engine = rag_core::RagEngine::new(backend);

            Ok(Self {
                index_dir: PathBuf::from(index_dir),
                engine: Mutex::new(engine),
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
            let engine = self
                .engine
                .lock()
                .map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("Lock poisoned"))?;

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
        // Release GIL during potentially slow disk I/O
        py.allow_threads(|| {
            let engine = match self.engine.lock() {
                Ok(e) => e,
                Err(_) => {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err("Lock poisoned"))
                }
            };

            engine
                .save_to_dir(&self.index_dir)
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
        // Release GIL during potentially slow disk I/O
        py.allow_threads(|| {
            let mut engine = match self.engine.lock() {
                Ok(e) => e,
                Err(_) => {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err("Lock poisoned"))
                }
            };

            engine
                .load_from_dir(&self.index_dir)
                .map_err(engine_error_to_pyerr)
        })
    }

    /// List all indexed documents.
    ///
    /// # Returns
    /// List of document names currently in the index.
    fn list_documents(&self, _py: Python<'_>) -> PyResult<Vec<String>> {
        catch_panic!(_py, {
            let engine = self
                .engine
                .lock()
                .map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("Lock poisoned"))?;

            Ok(engine.list_documents())
        })
    }

    /// Check if the engine needs reindexing.
    ///
    /// Returns true if the embedding model has changed since the last index.
    fn needs_reindex(&self, _py: Python<'_>) -> PyResult<bool> {
        catch_panic!(_py, {
            let engine = self
                .engine
                .lock()
                .map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("Lock poisoned"))?;

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
        // Release GIL and perform async embedding + upsert
        py.allow_threads(|| {
            let mut engine = match self.engine.lock() {
                Ok(e) => e,
                Err(_) => {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err("Lock poisoned"))
                }
            };

            runtime()
                .block_on(engine.upsert_document(name, text, content_hash))
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
        py.allow_threads(|| {
            let mut engine = match self.engine.lock() {
                Ok(e) => e,
                Err(_) => {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err("Lock poisoned"))
                }
            };

            engine.remove_document(name).map_err(engine_error_to_pyerr)
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

        // Release GIL and perform async search
        let result = py.allow_threads(|| {
            // Acquire engine lock
            let engine = match self.engine.lock() {
                Ok(e) => e,
                Err(_) => {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err("Lock poisoned"))
                }
            };

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
        });

        result
    }

    /// Get the index directory path.
    #[getter]
    fn index_dir(&self) -> String {
        self.index_dir.to_string_lossy().to_string()
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

    #[test]
    fn test_mock_engine_creation() {
        let _dir = tempdir().unwrap();
        let backend = MockEmbeddingBackend::new(768);
        let engine: MockEngine = rag_core::RagEngine::new(backend);
        let health = engine.health();

        assert_eq!(health.chunk_count, 0);
        assert_eq!(health.document_count, 0);
        assert_eq!(health.embedding_dim, 768);
        assert!(health.embedding_model.contains("mock"));
    }

    #[test]
    fn test_mock_engine_custom_dimension() {
        let backend = MockEmbeddingBackend::new(384);
        let engine: MockEngine = rag_core::RagEngine::new(backend);
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
