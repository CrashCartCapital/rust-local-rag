//! Error types and exception mapping for Python bindings.
//!
//! Maps rag-core errors to Python exception hierarchy.

use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::{PyErr, create_exception};

// Define Python exception hierarchy
create_exception!(ragcore, RagError, PyException);
create_exception!(ragcore, EmbeddingError, RagError);
create_exception!(ragcore, RerankError, RagError);
create_exception!(ragcore, IndexError, RagError);
create_exception!(ragcore, ConfigError, RagError);
create_exception!(ragcore, ValidationError, RagError);

/// Register exception types with the Python module.
pub fn register_exceptions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("RagError", m.py().get_type::<RagError>())?;
    m.add("EmbeddingError", m.py().get_type::<EmbeddingError>())?;
    m.add("RerankError", m.py().get_type::<RerankError>())?;
    m.add("IndexError", m.py().get_type::<IndexError>())?;
    m.add("ConfigError", m.py().get_type::<ConfigError>())?;
    m.add("ValidationError", m.py().get_type::<ValidationError>())?;
    Ok(())
}

/// Convert rag-core EngineError to Python exception.
#[allow(dead_code)] // Will be used in Task 2.1+
pub fn engine_error_to_pyerr(err: rag_core::EngineError) -> PyErr {
    match &err {
        rag_core::EngineError::Embedding(_) => EmbeddingError::new_err(err.to_string()),
        rag_core::EngineError::Rerank(_) => RerankError::new_err(err.to_string()),
        rag_core::EngineError::Persistence { .. } => IndexError::new_err(err.to_string()),
        rag_core::EngineError::DocumentNotFound(_) => IndexError::new_err(err.to_string()),
        rag_core::EngineError::IndexSync { .. } => IndexError::new_err(err.to_string()),
        rag_core::EngineError::Config(_) => ConfigError::new_err(err.to_string()),
        rag_core::EngineError::Validation { .. } => ValidationError::new_err(err.to_string()),
    }
}
