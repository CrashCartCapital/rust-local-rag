use serde::{Deserialize, Serialize};

/// Structured payload for reindex jobs.
///
/// Stored as JSON in `jobs.payload` to make resumes model-safe.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReindexJobPayload {
    pub documents_dir: String,
    pub model_id: String,
    pub embedding_dim: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParsedReindexJobPayload {
    Structured(ReindexJobPayload),
    LegacyDocumentsDir(String),
}

pub fn parse_reindex_job_payload(raw: &str) -> ParsedReindexJobPayload {
    match serde_json::from_str::<ReindexJobPayload>(raw) {
        Ok(payload) => ParsedReindexJobPayload::Structured(payload),
        Err(_) => ParsedReindexJobPayload::LegacyDocumentsDir(raw.trim().to_string()),
    }
}

