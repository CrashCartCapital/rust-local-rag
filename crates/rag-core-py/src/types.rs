//! Python type wrappers for rag-core types.
//!
//! Provides #[pyclass] wrappers for QuerySpec, SearchResult, ScoreBreakdown, etc.

use pyo3::prelude::*;

/// Search configuration parameters.
#[pyclass(name = "QuerySpec", module = "ragcore")]
#[derive(Clone, Debug)]
pub struct PyQuerySpec {
    #[pyo3(get, set)]
    pub top_k: usize,
    #[pyo3(get, set)]
    pub embedding_weight: f32,
    #[pyo3(get, set)]
    pub lexical_weight: f32,
    #[pyo3(get, set)]
    pub reranker_weight: f32,
    #[pyo3(get, set)]
    pub initial_weight: f32,
    #[pyo3(get, set)]
    pub diversity_factor: f32,
}

#[pymethods]
impl PyQuerySpec {
    #[new]
    #[pyo3(signature = (*, top_k=10, embedding_weight=0.7, lexical_weight=0.3, reranker_weight=0.7, initial_weight=0.3, diversity_factor=0.0))]
    fn new(
        top_k: usize,
        embedding_weight: f32,
        lexical_weight: f32,
        reranker_weight: f32,
        initial_weight: f32,
        diversity_factor: f32,
    ) -> Self {
        Self {
            top_k,
            embedding_weight,
            lexical_weight,
            reranker_weight,
            initial_weight,
            diversity_factor,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "QuerySpec(top_k={}, embedding_weight={}, lexical_weight={}, diversity_factor={})",
            self.top_k, self.embedding_weight, self.lexical_weight, self.diversity_factor
        )
    }
}

/// Detailed scoring components for a search result.
#[pyclass(name = "ScoreBreakdown", module = "ragcore")]
#[derive(Clone, Debug)]
pub struct PyScoreBreakdown {
    #[pyo3(get)]
    pub embedding: Option<f32>,
    #[pyo3(get)]
    pub lexical: Option<f32>,
    #[pyo3(get)]
    pub initial: Option<f32>,
    #[pyo3(get)]
    pub reranker: Option<f32>,
    #[pyo3(get)]
    pub total: f32,
}

#[pymethods]
impl PyScoreBreakdown {
    fn __repr__(&self) -> String {
        format!(
            "ScoreBreakdown(total={:.4}, embedding={:?}, lexical={:?}, reranker={:?})",
            self.total, self.embedding, self.lexical, self.reranker
        )
    }
}

/// A single search result with metadata and scores.
#[pyclass(name = "SearchResult", module = "ragcore")]
#[derive(Clone, Debug)]
pub struct PySearchResult {
    #[pyo3(get)]
    pub id: String,
    #[pyo3(get)]
    pub text: String,
    #[pyo3(get)]
    pub document: String,
    #[pyo3(get)]
    pub chunk_index: usize,
    #[pyo3(get)]
    pub page_number: Option<usize>,
    #[pyo3(get)]
    pub section: Option<String>,
    #[pyo3(get)]
    pub score: f32,
    #[pyo3(get)]
    pub scores: PyScoreBreakdown,
}

#[pymethods]
impl PySearchResult {
    fn __repr__(&self) -> String {
        format!(
            "SearchResult(id='{}', document='{}', score={:.4})",
            self.id, self.document, self.score
        )
    }
}

/// Engine health and statistics.
#[pyclass(name = "EngineStats", module = "ragcore")]
#[derive(Clone, Debug)]
pub struct PyEngineStats {
    #[pyo3(get)]
    pub document_count: usize,
    #[pyo3(get)]
    pub chunk_count: usize,
    #[pyo3(get)]
    pub embedding_model: String,
    #[pyo3(get)]
    pub embedding_dimension: usize,
}

#[pymethods]
impl PyEngineStats {
    fn __repr__(&self) -> String {
        format!(
            "EngineStats(documents={}, chunks={}, model='{}')",
            self.document_count, self.chunk_count, self.embedding_model
        )
    }
}
