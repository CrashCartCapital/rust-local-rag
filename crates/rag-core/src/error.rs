//! Structured error types with full context.
//!
//! rag-core uses typed errors that preserve context for debugging and
//! machine-actionable error handling.
//!
//! # Error Hierarchy
//!
//! ```text
//! EngineError (top-level)
//! ├── Embedding(EmbeddingError)     - Embedding service failures
//! ├── Rerank(RerankError)           - Reranker service failures
//! ├── DocumentNotFound              - Missing document
//! ├── Persistence { ... }           - I/O errors with path and operation
//! ├── Validation { ... }            - Input validation failures
//! ├── Config                        - Configuration errors
//! └── IndexSync                     - Index consistency errors
//! ```
//!
//! # Pattern Matching
//!
//! Errors are designed for pattern matching:
//!
//! ```rust,ignore
//! use rag_core::{EngineError, ValidationKind};
//!
//! match error {
//!     EngineError::Validation { kind: ValidationKind::NaN, .. } => {
//!         // Handle NaN embedding specifically
//!     }
//!     EngineError::Validation { kind: ValidationKind::DimensionMismatch { expected, got }, .. } => {
//!         eprintln!("Expected {expected} dims, got {got}");
//!     }
//!     _ => {}
//! }
//! ```

use crate::types::DocumentName;
use std::path::PathBuf;

/// Operation type for persistence errors, enabling machine-actionable error handling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PersistenceOp {
    /// Saving engine state to disk
    Save,
    /// Loading engine state from disk
    Load,
    /// Migrating between schema versions
    Migrate,
}

impl std::fmt::Display for PersistenceOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PersistenceOp::Save => write!(f, "save"),
            PersistenceOp::Load => write!(f, "load"),
            PersistenceOp::Migrate => write!(f, "migrate"),
        }
    }
}

/// Validation failure kind for embedding and chunk operations.
#[derive(Debug, Clone, PartialEq)]
pub enum ValidationKind {
    /// Embedding contains NaN values
    NaN,
    /// Embedding contains Inf values
    Inf,
    /// Embedding dimension doesn't match expected
    DimensionMismatch { expected: usize, got: usize },
    /// Chunk text is empty or whitespace-only
    EmptyText,
    /// Document hash mismatch (corruption or concurrent modification)
    HashMismatch { expected: String, got: String },
}

impl std::fmt::Display for ValidationKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ValidationKind::NaN => write!(f, "embedding contains NaN values"),
            ValidationKind::Inf => write!(f, "embedding contains Inf values"),
            ValidationKind::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
            ValidationKind::EmptyText => write!(f, "text is empty or whitespace-only"),
            ValidationKind::HashMismatch { expected, got } => {
                write!(f, "hash mismatch: expected {expected}, got {got}")
            }
        }
    }
}

/// Errors from persistence operations (I/O and serialization).
#[derive(Debug, thiserror::Error)]
pub enum PersistenceError {
    /// Filesystem I/O error
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON serialization/deserialization error (feature-gated)
    #[cfg(feature = "persistence")]
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Schema version mismatch requiring migration or reindex
    #[error("version mismatch: file has version {file_version}, expected {expected_version}")]
    VersionMismatch {
        file_version: u32,
        expected_version: u32,
        path: PathBuf,
    },

    /// Index file not found (not necessarily an error for new engines)
    #[error("index file not found: {0}")]
    NotFound(PathBuf),
}

#[derive(Debug, thiserror::Error)]
pub enum EmbeddingError {
    #[error("connection failed: {0}")]
    Connection(String),

    #[error("model not found: {0}")]
    ModelNotFound(String),

    #[error("API error: {0}")]
    Api(String),

    #[error("timeout after {0:?}")]
    Timeout(std::time::Duration),

    #[error("validation error: {0}")]
    Validation(ValidationKind),
}

/// Errors from reranking operations.
#[derive(Debug, thiserror::Error)]
pub enum RerankError {
    /// Generic reranker error
    #[error("reranker error: {0}")]
    Error(String),

    /// Reranker service unavailable
    #[error("reranker unavailable: {0}")]
    Unavailable(String),

    /// Reranker API error
    #[error("reranker API error: {0}")]
    Api(String),

    /// Reranker returned invalid response
    #[error("invalid reranker response: {0}")]
    InvalidResponse(String),

    /// Reranker operation timed out
    #[error("timeout after {0:?}")]
    Timeout(std::time::Duration),
}

#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("embedding failed: {0}")]
    Embedding(#[from] EmbeddingError),

    #[error("reranking failed: {0}")]
    Rerank(#[from] RerankError),

    #[error("document not found: {0}")]
    DocumentNotFound(DocumentName),

    #[error("{operation} failed for {path}: {source}")]
    Persistence {
        path: PathBuf,
        operation: PersistenceOp,
        #[source]
        source: PersistenceError,
    },

    #[error("validation error{}: {kind}", chunk_id.as_ref().map(|id| format!(" in chunk {id}")).unwrap_or_default())]
    Validation {
        chunk_id: Option<String>,
        kind: ValidationKind,
    },

    #[error("invalid configuration: {0}")]
    Config(String),

    #[error("index synchronization error: {message}")]
    IndexSync { message: String },
}

pub type Result<T> = std::result::Result<T, EngineError>;

// Helper constructors for common error patterns
impl EngineError {
    /// Create a persistence save error with context
    pub fn save_failed(path: impl Into<PathBuf>, source: PersistenceError) -> Self {
        EngineError::Persistence {
            path: path.into(),
            operation: PersistenceOp::Save,
            source,
        }
    }

    /// Create a persistence load error with context
    pub fn load_failed(path: impl Into<PathBuf>, source: PersistenceError) -> Self {
        EngineError::Persistence {
            path: path.into(),
            operation: PersistenceOp::Load,
            source,
        }
    }

    /// Create a validation error for a specific chunk
    pub fn validation(chunk_id: impl Into<String>, kind: ValidationKind) -> Self {
        EngineError::Validation {
            chunk_id: Some(chunk_id.into()),
            kind,
        }
    }

    /// Create a validation error without chunk context
    pub fn validation_no_chunk(kind: ValidationKind) -> Self {
        EngineError::Validation {
            chunk_id: None,
            kind,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_persistence_op_display() {
        assert_eq!(PersistenceOp::Save.to_string(), "save");
        assert_eq!(PersistenceOp::Load.to_string(), "load");
        assert_eq!(PersistenceOp::Migrate.to_string(), "migrate");
    }

    #[test]
    fn test_validation_kind_display() {
        assert_eq!(
            ValidationKind::NaN.to_string(),
            "embedding contains NaN values"
        );
        assert_eq!(
            ValidationKind::EmptyText.to_string(),
            "text is empty or whitespace-only"
        );
        assert_eq!(
            ValidationKind::DimensionMismatch {
                expected: 384,
                got: 768
            }
            .to_string(),
            "dimension mismatch: expected 384, got 768"
        );
    }

    #[test]
    fn test_engine_error_persistence_display() {
        let err = EngineError::save_failed(
            PathBuf::from("/tmp/test.json"),
            PersistenceError::Io(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "access denied",
            )),
        );
        let msg = err.to_string();
        assert!(msg.contains("save"));
        assert!(msg.contains("/tmp/test.json"));
    }

    #[test]
    fn test_engine_error_validation_with_chunk() {
        let err = EngineError::validation("chunk-123", ValidationKind::NaN);
        let msg = err.to_string();
        assert!(msg.contains("chunk-123"));
        assert!(msg.contains("NaN"));
    }

    #[test]
    fn test_engine_error_validation_no_chunk() {
        let err = EngineError::validation_no_chunk(ValidationKind::EmptyText);
        let msg = err.to_string();
        assert!(!msg.contains("chunk"));
        assert!(msg.contains("empty"));
    }

    #[test]
    fn test_persistence_error_version_mismatch() {
        let err = PersistenceError::VersionMismatch {
            file_version: 1,
            expected_version: 2,
            path: PathBuf::from("/tmp/index.json"),
        };
        let msg = err.to_string();
        assert!(msg.contains("version 1"));
        assert!(msg.contains("expected 2"));
    }

    #[test]
    fn test_error_source_chain() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let persistence_err = PersistenceError::Io(io_err);
        let engine_err = EngineError::load_failed("/tmp/missing.json", persistence_err);

        // Verify source chain is preserved
        use std::error::Error;
        assert!(engine_err.source().is_some());
    }

    #[test]
    fn test_validation_kind_pattern_matching() {
        // Verify errors are machine-actionable via pattern matching
        let err = EngineError::validation(
            "chunk-1",
            ValidationKind::DimensionMismatch {
                expected: 384,
                got: 768,
            },
        );

        match &err {
            EngineError::Validation {
                kind: ValidationKind::DimensionMismatch { expected, got },
                ..
            } => {
                assert_eq!(*expected, 384);
                assert_eq!(*got, 768);
            }
            _ => panic!("Expected DimensionMismatch variant"),
        }
    }
}
