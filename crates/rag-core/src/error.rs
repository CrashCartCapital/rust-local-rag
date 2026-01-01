use crate::types::DocumentName;

#[derive(Debug, thiserror::Error)]
pub enum EmbeddingError {
    #[error("Connection failed: {0}")]
    Connection(String),
    #[error("Model not found: {0}")]
    ModelNotFound(String),
    #[error("API error: {0}")]
    Api(String),
    #[error("Timeout after {0:?}")]
    Timeout(std::time::Duration),
}

#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("Embedding failed: {0}")]
    Embedding(#[from] EmbeddingError),
    #[error("Document not found: {0}")]
    DocumentNotFound(DocumentName),
    #[error("Persistence error: {0}")]
    Persistence(String),
    #[error("Invalid configuration: {0}")]
    Config(String),
}

pub type Result<T> = std::result::Result<T, EngineError>;

#[derive(Debug, thiserror::Error)]
pub enum RerankError {
    #[error("Reranker error: {0}")]
    Error(String),
}
