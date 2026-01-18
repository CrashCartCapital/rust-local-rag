use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RagError {
    Config {
        message: String,
        cause: String,
        fix: String,
    },
    PdfExtraction {
        filename: String,
        reason: String,
        fix: String,
    },
    Search {
        message: String,
        fix: String,
    },
    Embedding {
        message: String,
        fix: String,
    },
    ModelMismatch {
        stored: String,
        configured: String,
        fix: String,
    },
    PartialIndexFailure {
        failures_count: usize,
        total_count: usize,
        details: String,
        fix: String,
    },
}

impl RagError {
    pub fn config(message: impl Into<String>, cause: impl Into<String>, fix: impl Into<String>) -> Self {
        Self::Config {
            message: message.into(),
            cause: cause.into(),
            fix: fix.into(),
        }
    }

    pub fn pdf_extraction(
        filename: impl Into<String>,
        reason: impl Into<String>,
        fix: impl Into<String>,
    ) -> Self {
        Self::PdfExtraction {
            filename: filename.into(),
            reason: reason.into(),
            fix: fix.into(),
        }
    }

    pub fn search(message: impl Into<String>, fix: impl Into<String>) -> Self {
        Self::Search {
            message: message.into(),
            fix: fix.into(),
        }
    }

    pub fn embedding(message: impl Into<String>, fix: impl Into<String>) -> Self {
        Self::Embedding {
            message: message.into(),
            fix: fix.into(),
        }
    }

    pub fn model_mismatch(
        stored: impl Into<String>,
        configured: impl Into<String>,
        fix: impl Into<String>,
    ) -> Self {
        Self::ModelMismatch {
            stored: stored.into(),
            configured: configured.into(),
            fix: fix.into(),
        }
    }

    pub fn partial_index_failure(
        failures_count: usize,
        total_count: usize,
        details: impl Into<String>,
        fix: impl Into<String>,
    ) -> Self {
        Self::PartialIndexFailure {
            failures_count,
            total_count,
            details: details.into(),
            fix: fix.into(),
        }
    }
}

impl fmt::Display for RagError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RagError::Config { message, cause, fix } => write!(
                f,
                "Configuration error: {message}\n  Cause: {cause}\n  Fix: {fix}"
            ),
            RagError::PdfExtraction {
                filename,
                reason,
                fix,
            } => write!(
                f,
                "PDF extraction failed for '{filename}': {reason}\n  Fix: {fix}"
            ),
            RagError::Search { message, fix } => {
                write!(f, "Search failed: {message}\n  Fix: {fix}")
            }
            RagError::Embedding { message, fix } => {
                write!(f, "Embedding service error: {message}\n  Fix: {fix}")
            }
            RagError::ModelMismatch {
                stored,
                configured,
                fix,
            } => write!(
                f,
                "Index mismatch: stored model '{stored}' differs from configured '{configured}'\n  Fix: {fix}"
            ),
            RagError::PartialIndexFailure {
                failures_count,
                total_count,
                details,
                fix,
            } => write!(
                f,
                "Document processing failed: {failures_count} of {total_count} documents failed\n  Details: {details}\n  Fix: {fix}"
            ),
        }
    }
}

impl std::error::Error for RagError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_formatting() {
        let err = RagError::config(
            "invalid value for RAG_DEFAULT_LOGPROB",
            "invalid float",
            "Set RAG_DEFAULT_LOGPROB to a finite number <= 0.0",
        );
        let msg = err.to_string();
        assert!(msg.contains("Configuration error: invalid value for RAG_DEFAULT_LOGPROB"));
        assert!(msg.contains("Cause: invalid float"));
        assert!(msg.contains("Fix: Set RAG_DEFAULT_LOGPROB"));
    }

    #[test]
    fn test_pdf_extraction_formatting() {
        let err = RagError::pdf_extraction(
            "broken.pdf",
            "lopdf failed to parse PDF: ...",
            "Ensure the file is a valid PDF (try re-downloading or regenerating it)",
        );
        let msg = err.to_string();
        assert!(msg.contains("PDF extraction failed for 'broken.pdf':"));
        assert!(msg.contains("Fix: Ensure the file is a valid PDF"));
    }

    #[test]
    fn test_model_mismatch_formatting() {
        let err = RagError::model_mismatch(
            "model-a",
            "model-b",
            "Run reindex or update OLLAMA_EMBEDDING_MODEL",
        );
        assert!(err
            .to_string()
            .contains("stored model 'model-a' differs from configured 'model-b'"));
    }

    #[test]
    fn test_anyhow_compatibility() {
        fn returns_anyhow_error() -> anyhow::Result<()> {
            Err(anyhow::Error::new(RagError::search(
                "backend unavailable",
                "Start Ollama and retry",
            )))
        }

        let err = returns_anyhow_error().unwrap_err();
        assert!(err.is::<RagError>());
        assert!(err.to_string().contains("Search failed: backend unavailable"));
    }
}

