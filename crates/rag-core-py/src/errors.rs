//! Error types and exception mapping for Python bindings.
//!
//! Maps rag-core errors to Python exception hierarchy with structured metadata (TD-4).
//!
//! ## Exception Hierarchy
//!
//! ```text
//! PyException
//! └── RagError (base for all ragcore errors)
//!     ├── EmbeddingError - embedding service failures
//!     ├── RerankError - reranker service failures
//!     ├── IndexError - persistence/document errors
//!     ├── ConfigError - configuration errors
//!     └── ValidationError - input validation failures
//! ```
//!
//! ## Structured Attributes (TD-4)
//!
//! Exceptions include structured metadata for programmatic error handling:
//!
//! ```python
//! try:
//!     engine.search("query")
//! except ragcore.ValidationError as e:
//!     print(e.chunk_id)  # "chunk-123" or None
//!     print(e.kind)      # "dimension_mismatch"
//!     print(e.expected)  # 384
//!     print(e.got)       # 768
//! except ragcore.IndexError as e:
//!     print(e.path)      # "/tmp/index.json"
//!     print(e.operation) # "save"
//! ```

use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::{PyErr, create_exception};
use rag_core::{EngineError, ValidationKind};

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

/// Convert rag-core EngineError to Python exception with structured attributes (TD-4).
///
/// Extracts structured metadata from error variants and attaches them as
/// exception attributes for programmatic error handling in Python.
pub fn engine_error_to_pyerr(err: EngineError) -> PyErr {
    Python::with_gil(|py| {
        match &err {
            EngineError::Embedding(emb_err) => {
                let pyerr = EmbeddingError::new_err(err.to_string());
                // Set embedding error subtype
                if let Ok(exc) = pyerr.value(py).extract::<Bound<'_, PyAny>>() {
                    let kind = match emb_err {
                        rag_core::EmbeddingError::Unavailable(_) => "unavailable",
                        rag_core::EmbeddingError::ModelNotFound(_) => "model_not_found",
                        rag_core::EmbeddingError::Api(_) => "api",
                        rag_core::EmbeddingError::Timeout(_) => "timeout",
                        rag_core::EmbeddingError::Validation(_) => "validation",
                    };
                    let _ = exc.setattr("kind", kind);

                    // Set timeout duration if applicable
                    if let rag_core::EmbeddingError::Timeout(duration) = emb_err {
                        let _ = exc.setattr("timeout_secs", duration.as_secs_f64());
                    }
                }
                pyerr
            }

            EngineError::Rerank(rerank_err) => {
                let pyerr = RerankError::new_err(err.to_string());
                if let Ok(exc) = pyerr.value(py).extract::<Bound<'_, PyAny>>() {
                    let kind = match rerank_err {
                        rag_core::RerankError::Error(_) => "error",
                        rag_core::RerankError::Unavailable(_) => "unavailable",
                        rag_core::RerankError::Api(_) => "api",
                        rag_core::RerankError::InvalidResponse(_) => "invalid_response",
                        rag_core::RerankError::Timeout(_) => "timeout",
                    };
                    let _ = exc.setattr("kind", kind);

                    // Set timeout duration if applicable
                    if let rag_core::RerankError::Timeout(duration) = rerank_err {
                        let _ = exc.setattr("timeout_secs", duration.as_secs_f64());
                    }
                }
                pyerr
            }

            EngineError::Persistence {
                path,
                operation,
                source,
            } => {
                let pyerr = IndexError::new_err(err.to_string());
                if let Ok(exc) = pyerr.value(py).extract::<Bound<'_, PyAny>>() {
                    let _ = exc.setattr("path", path.to_string_lossy().to_string());
                    let _ = exc.setattr("operation", operation.to_string());

                    // Set source error info
                    // Note: rag-core-py always enables persistence feature in rag-core
                    let source_kind = match source {
                        rag_core::PersistenceError::Io(_) => "io",
                        rag_core::PersistenceError::Json(_) => "json",
                        rag_core::PersistenceError::VersionMismatch { .. } => "version_mismatch",
                        rag_core::PersistenceError::NotFound(_) => "not_found",
                    };
                    let _ = exc.setattr("source_kind", source_kind);

                    // Chain __cause__ with source error string
                    let cause_err = PyErr::new::<RagError, _>(source.to_string());
                    let _ = exc.setattr("__cause__", cause_err.value(py));
                }
                pyerr
            }

            EngineError::DocumentNotFound(doc_name) => {
                let pyerr = IndexError::new_err(err.to_string());
                if let Ok(exc) = pyerr.value(py).extract::<Bound<'_, PyAny>>() {
                    let _ = exc.setattr("document_name", doc_name.as_str());
                    let _ = exc.setattr("kind", "document_not_found");
                }
                pyerr
            }

            EngineError::IndexSync { message } => {
                let pyerr = IndexError::new_err(err.to_string());
                if let Ok(exc) = pyerr.value(py).extract::<Bound<'_, PyAny>>() {
                    let _ = exc.setattr("kind", "sync");
                    let _ = exc.setattr("sync_message", message.as_str());
                }
                pyerr
            }

            EngineError::Config(msg) => {
                let pyerr = ConfigError::new_err(err.to_string());
                if let Ok(exc) = pyerr.value(py).extract::<Bound<'_, PyAny>>() {
                    let _ = exc.setattr("config_message", msg.as_str());
                }
                pyerr
            }

            EngineError::Validation { chunk_id, kind } => {
                let pyerr = ValidationError::new_err(err.to_string());
                if let Ok(exc) = pyerr.value(py).extract::<Bound<'_, PyAny>>() {
                    // Set chunk_id (may be None)
                    match chunk_id {
                        Some(id) => {
                            let _ = exc.setattr("chunk_id", id.as_str());
                        }
                        None => {
                            let _ = exc.setattr("chunk_id", py.None());
                        }
                    }

                    // Set validation kind and specific fields
                    match kind {
                        ValidationKind::NaN => {
                            let _ = exc.setattr("kind", "nan");
                        }
                        ValidationKind::Inf => {
                            let _ = exc.setattr("kind", "inf");
                        }
                        ValidationKind::DimensionMismatch { expected, got } => {
                            let _ = exc.setattr("kind", "dimension_mismatch");
                            let _ = exc.setattr("expected", *expected);
                            let _ = exc.setattr("got", *got);
                        }
                        ValidationKind::EmptyText => {
                            let _ = exc.setattr("kind", "empty_text");
                        }
                        ValidationKind::HashMismatch { expected, got } => {
                            let _ = exc.setattr("kind", "hash_mismatch");
                            let _ = exc.setattr("expected_hash", expected.as_str());
                            let _ = exc.setattr("got_hash", got.as_str());
                        }
                    }
                }
                pyerr
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rag_core::{EmbeddingError as CoreEmbeddingError, PersistenceError, PersistenceOp};
    use std::path::PathBuf;

    #[test]
    fn test_validation_error_has_structured_attrs() {
        let err = EngineError::Validation {
            chunk_id: Some("chunk-123".to_string()),
            kind: ValidationKind::DimensionMismatch {
                expected: 384,
                got: 768,
            },
        };

        let pyerr = engine_error_to_pyerr(err);

        Python::with_gil(|py| {
            let exc = pyerr.value(py);
            // Verify attributes are set
            assert!(exc.getattr("chunk_id").is_ok());
            assert!(exc.getattr("kind").is_ok());
            assert!(exc.getattr("expected").is_ok());
            assert!(exc.getattr("got").is_ok());
        });
    }

    #[test]
    fn test_persistence_error_has_path_and_operation() {
        let err = EngineError::Persistence {
            path: PathBuf::from("/tmp/test.json"),
            operation: PersistenceOp::Save,
            source: PersistenceError::NotFound(PathBuf::from("/tmp/test.json")),
        };

        let pyerr = engine_error_to_pyerr(err);

        Python::with_gil(|py| {
            let exc = pyerr.value(py);
            assert!(exc.getattr("path").is_ok());
            assert!(exc.getattr("operation").is_ok());
            assert!(exc.getattr("source_kind").is_ok());
        });
    }

    #[test]
    fn test_embedding_error_has_kind() {
        let err = EngineError::Embedding(CoreEmbeddingError::Timeout(
            std::time::Duration::from_secs(30),
        ));

        let pyerr = engine_error_to_pyerr(err);

        Python::with_gil(|py| {
            let exc = pyerr.value(py);
            assert!(exc.getattr("kind").is_ok());
            assert!(exc.getattr("timeout_secs").is_ok());
        });
    }

    #[test]
    fn test_document_not_found_has_name() {
        let err = EngineError::DocumentNotFound("test.pdf".into());

        let pyerr = engine_error_to_pyerr(err);

        Python::with_gil(|py| {
            let exc = pyerr.value(py);
            assert!(exc.getattr("document_name").is_ok());
            assert!(exc.getattr("kind").is_ok());
        });
    }
}
