//! PyRagEngine: Python wrapper for rag-core's RagEngine.
//!
//! This module provides the main entry point for Python applications.

use pyo3::prelude::*;
use rag_core::EmbeddingBackend;

use crate::mock_backend::MockEmbeddingBackend;
use crate::types::PyEngineStats;

/// The main RAG engine for Python.
///
/// Provides semantic search over documents with configurable embedding
/// and reranking backends.
#[pyclass(name = "RagEngine", module = "ragcore")]
pub struct PyRagEngine {
    index_dir: String,
    // Mock backend for testing (Phase 1)
    // Will be replaced with Arc<dyn EmbeddingBackend> in Phase 2
    mock_backend: Option<MockEmbeddingBackend>,
    // Track document/chunk counts for stats
    document_count: usize,
    chunk_count: usize,
}

#[pymethods]
impl PyRagEngine {
    /// Create a new engine with a mock backend (for testing).
    ///
    /// # Arguments
    /// * `index_dir` - Directory for storing the index
    /// * `dimension` - Embedding dimension (default: 768)
    #[staticmethod]
    #[pyo3(signature = (index_dir, dimension=768))]
    fn create_mock(index_dir: &str, dimension: usize) -> PyResult<Self> {
        Ok(Self {
            index_dir: index_dir.to_string(),
            mock_backend: Some(MockEmbeddingBackend::new(dimension)),
            document_count: 0,
            chunk_count: 0,
        })
    }

    /// Get engine statistics.
    fn stats(&self) -> PyEngineStats {
        let (model, dim) = if let Some(ref backend) = self.mock_backend {
            (
                backend.model_id().to_string(),
                backend.dimension(),
            )
        } else {
            ("unknown".to_string(), 0)
        };

        PyEngineStats {
            document_count: self.document_count,
            chunk_count: self.chunk_count,
            embedding_model: model,
            embedding_dimension: dim,
        }
    }

    fn __repr__(&self) -> String {
        format!("RagEngine(index_dir='{}')", self.index_dir)
    }
}
