//! Error types and exception mapping for Python bindings.
//!
//! Maps rag-core errors to Python exception hierarchy.

use pyo3::exceptions::{PyException, PyFileNotFoundError, PyIOError, PyTimeoutError, PyValueError};
use pyo3::prelude::*;
use pyo3::{create_exception, PyErr};

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
    match err {
        rag_core::EngineError::Persistence {
            path,
            operation,
            source,
        } => {
            let msg = format!("{} failed for {}: {}", operation, path.display(), source);
            match source {
                rag_core::PersistenceError::Io(io_err) => {
                    if io_err.kind() == std::io::ErrorKind::NotFound {
                        PyFileNotFoundError::new_err(msg)
                    } else {
                        PyIOError::new_err(msg)
                    }
                }
                rag_core::PersistenceError::NotFound(_) => PyFileNotFoundError::new_err(msg),
                _ => IndexError::new_err(msg),
            }
        }
        rag_core::EngineError::Validation { chunk_id, kind } => {
            let mut msg = kind.to_string();
            if let Some(cid) = chunk_id {
                msg = format!("{} (chunk_id: {})", msg, cid);
            }
            PyValueError::new_err(msg)
        }
        rag_core::EngineError::Config(msg) => PyValueError::new_err(msg),
        rag_core::EngineError::DocumentNotFound(name) => {
            PyValueError::new_err(format!("Document not found: {}", name))
        }
        rag_core::EngineError::Embedding(err) => match err {
            rag_core::EmbeddingError::Validation(kind) => PyValueError::new_err(kind.to_string()),
            rag_core::EmbeddingError::Timeout(duration) => {
                PyTimeoutError::new_err(format!("Timeout after {:?}", duration))
            }
            _ => EmbeddingError::new_err(err.to_string()),
        },
        rag_core::EngineError::Rerank(err) => RerankError::new_err(err.to_string()),
        rag_core::EngineError::IndexSync { message } => IndexError::new_err(message),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rag_core::{EngineError, PersistenceError, PersistenceOp, ValidationKind};
    use std::path::PathBuf;

    // Helper to check if a PyErr matches an expected Python type name
    fn assert_is_instance(py: Python, err: &PyErr, expected_type: &str) {
        let err_type = err.get_type(py);
        let name = err_type.name().unwrap();
        // Handle fully qualified names if needed, but often checking if it ends with or is equivalent
        // Checking if it inherits or is the type.
        // For standard exceptions, name is just the name (e.g. "ValueError").
        // For custom, it might be "rag_core_py._native.EmbeddingError" depending on how it's initialized.
        // But in simple unit tests, just checking the name is often robust enough for basic checks.
        assert_eq!(name, expected_type, "Expected exception type {}, got {}", expected_type, name);
    }

    // We can't easily check for inheritance in rust unit tests without full python runtime init,
    // but pyo3 test harness handles some of it.

    #[test]
    fn test_persistence_io_mapping() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let io_err = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied");
            let err = EngineError::Persistence {
                path: PathBuf::from("/test"),
                operation: PersistenceOp::Save,
                source: PersistenceError::Io(io_err),
            };

            let py_err = engine_error_to_pyerr(err);
            assert_is_instance(py, &py_err, "OSError");

            let msg = py_err.value(py).to_string();
            assert!(msg.contains("save failed for /test"));
            assert!(msg.contains("denied"));
        });
    }

    #[test]
    fn test_persistence_not_found_mapping() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "missing");
            let err = EngineError::Persistence {
                path: PathBuf::from("/test"),
                operation: PersistenceOp::Load,
                source: PersistenceError::Io(io_err),
            };

            let py_err = engine_error_to_pyerr(err);
            assert_is_instance(py, &py_err, "FileNotFoundError");
        });
    }

    #[test]
    fn test_validation_mapping() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let err = EngineError::Validation {
                chunk_id: Some("c1".to_string()),
                kind: ValidationKind::NaN,
            };

            let py_err = engine_error_to_pyerr(err);
            assert_is_instance(py, &py_err, "ValueError");

            let msg = py_err.value(py).to_string();
            assert!(msg.contains("chunk_id: c1"));
            assert!(msg.contains("NaN"));
        });
    }

    #[test]
    fn test_document_not_found_mapping() {
         pyo3::prepare_freethreaded_python();
         Python::with_gil(|py| {
             let err = EngineError::DocumentNotFound("doc1".into());
             let py_err = engine_error_to_pyerr(err);
             assert_is_instance(py, &py_err, "ValueError");
             assert!(py_err.value(py).to_string().contains("Document not found: doc1"));
         });
    }

    #[test]
    fn test_timeout_mapping() {
         pyo3::prepare_freethreaded_python();
         Python::with_gil(|py| {
             let err = EngineError::Embedding(rag_core::EmbeddingError::Timeout(std::time::Duration::from_secs(5)));
             let py_err = engine_error_to_pyerr(err);
             assert_is_instance(py, &py_err, "TimeoutError");
         });
    }
}
