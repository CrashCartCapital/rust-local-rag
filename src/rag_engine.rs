use anyhow::{Context, Result};
use sha2::{Digest, Sha256};
use std::path::Path;
use std::sync::Arc;
use std::sync::OnceLock;
use tracing::instrument;
use uuid::Uuid;

use crate::{
    config::Config,
    embeddings::EmbeddingService,
    index_store::SqliteIndexStore,
    reranker::{RerankerCandidate, RerankerService},
};

pub use rag_core::SearchResult;
pub type PreparedDocument = rag_core::engine::PreparedDocument;

/// Core RAG engine for the server binary.
///
/// This is a thin wrapper over `rag-core` plus server-specific concerns:
/// - PDF extraction (lopdf + pdftotext fallback)
/// - Environment-driven configuration
/// - Stats formatting for MCP/HTTP
pub struct RagEngine {
    core: rag_core::RagEngine<EmbeddingService, RerankerService>,
    start_time: std::time::Instant,
    index_store: Arc<SqliteIndexStore>,
    incompatible_index_reason: Option<String>,
    embedding_weight: f32,
}

impl RagEngine {
    pub async fn new(data_dir: &str, config: &Config) -> Result<Self> {
        let embedding_service = EmbeddingService::new(config).await?;
        Self::new_with_embedding_service(data_dir, embedding_service, config).await
    }

    pub async fn new_with_embedding_service(
        data_dir: &str,
        embedding_service: EmbeddingService,
        config: &Config,
    ) -> Result<Self> {
        // Try to initialize reranker, but don't fail if it's unavailable.
        let reranker = match RerankerService::new(config).await {
            Ok(service) => {
                tracing::info!("Reranker service initialized successfully");
                Some(service)
            }
            Err(e) => {
                tracing::warn!(
                    "Reranker service unavailable, will fall back to embedding scores only: {}",
                    e
                );
                None
            }
        };

        let rag_config = rag_core::RagConfig {
            embedding_batch_size: config.embedding_batch_size.get(),
            ..rag_core::RagConfig::default()
        };

        let mut core =
            rag_core::RagEngine::with_optional_reranker(embedding_service, reranker, rag_config);

        let db_path = format!("sqlite:{data_dir}/jobs.db");
        let index_store = Arc::new(SqliteIndexStore::new(&db_path).await?);

        // One-time JSON -> SQLite migration (idempotent).
        let model_id = core.embedding_model().to_string();
        let backend_dim = core.backend_embedding_dim();
        index_store
            .migrate_json_if_needed(Path::new(data_dir), &model_id, backend_dim)
            .await?;

        // Hydrate the in-memory engine from SQLite (source of truth).
        match index_store.load_model_state(&model_id).await {
            Ok(Some(state)) => {
                core.load_state(state).map_err(anyhow::Error::new)?;
            }
            Ok(None) => {}
            Err(e) => {
                tracing::warn!("Could not load existing SQLite index state: {}", e);
                core.set_needs_reindex(true);
                let backend_dim = core.backend_embedding_dim();
                if let Err(db_err) = index_store
                    .set_model_needs_reindex(&model_id, backend_dim, true)
                    .await
                {
                    tracing::warn!(
                        "Could not persist needs_reindex state to SQLite after load failure: {}",
                        db_err
                    );
                }
            }
        }

        let mut incompatible_index_reason = None;
        let persisted_dim = core.persisted_embedding_dim();
        let backend_dim = core.backend_embedding_dim();
        if persisted_dim != 0 && backend_dim != 0 && persisted_dim != backend_dim {
            core.set_needs_reindex(true);
            incompatible_index_reason = Some(format!(
                "Embedding dimension mismatch: persisted index dim {persisted_dim} vs backend dim {backend_dim}. Reindex required."
            ));
            tracing::warn!(
                persisted_dim,
                backend_dim,
                "Embedding dimension mismatch detected; marking for reindex"
            );
        }

        Ok(Self {
            core,
            start_time: std::time::Instant::now(),
            index_store,
            incompatible_index_reason,
            embedding_weight: config.embedding_weight,
        })
    }

    pub fn needs_reindex(&self) -> bool {
        self.core.needs_reindex()
    }

    pub fn embedding_model(&self) -> &str {
        self.core.embedding_model()
    }

    pub fn backend_embedding_dim(&self) -> usize {
        self.core.backend_embedding_dim()
    }

    pub fn incompatible_index_reason(&self) -> Option<&str> {
        self.incompatible_index_reason.as_deref()
    }

    pub fn has_reranker(&self) -> bool {
        self.core.has_reranker()
    }

    pub fn get_reranker(&self) -> Option<&RerankerService> {
        self.core.reranker()
    }

    pub async fn finalize_reindex(&mut self) -> Result<()> {
        if self.core.needs_reindex() {
            self.core.set_needs_reindex(false);
            self.incompatible_index_reason = None;
            tracing::info!(
                "Reindexing complete. Indexed {} chunks across {} documents.",
                self.core.chunk_count(),
                self.list_documents().len()
            );
        }
        Ok(())
    }

    pub fn reset_state_for_reindex(&mut self, embedding_dim: usize) -> Result<()> {
        let mut state = rag_core::EngineState::new(self.core.embedding_model(), embedding_dim);
        state.needs_reindex = true;
        self.core.load_state(state).map_err(anyhow::Error::new)?;
        Ok(())
    }

    /// Adds a document to the index by extracting text, chunking, and generating embeddings.
    /// Returns the number of chunks created (0 if document unchanged via hash check).
    #[instrument(skip(self, data, batch_callback), fields(filename = %filename))]
    pub async fn add_document(
        &mut self,
        filename: &str,
        data: &[u8],
        batch_callback: Option<&mut (dyn FnMut(usize, usize, usize, usize) + Send)>,
    ) -> Result<usize> {
        tracing::info!("Processing document: {}", filename);
        let start = std::time::Instant::now();

        let prepared = self
            .prepare_document(filename, data, batch_callback)
            .await?;
        let Some(prepared) = prepared else {
            return Ok(0);
        };

        let document_hash = prepared.document_hash.as_deref().unwrap_or("");
        if document_hash.is_empty() {
            return Err(anyhow::anyhow!(
                "Prepared document missing hash; cannot persist safely"
            ));
        }

        self.index_store
            .upsert_document_atomic(
                self.core.embedding_model(),
                self.core.backend_embedding_dim(),
                self.core.needs_reindex(),
                &prepared.document_name,
                document_hash,
                &prepared.chunks,
            )
            .await?;

        let chunk_count = self.apply_prepared_document(prepared).await?;

        tracing::info!(
            "Successfully processed {} chunks for {} in {:?}",
            chunk_count,
            filename,
            start.elapsed()
        );

        Ok(chunk_count)
    }

    #[instrument(skip(self, data, batch_callback), fields(filename = %filename))]
    pub async fn prepare_document(
        &self,
        filename: &str,
        data: &[u8],
        batch_callback: Option<&mut (dyn FnMut(usize, usize, usize, usize) + Send)>,
    ) -> Result<Option<PreparedDocument>> {
        let document_hash = compute_document_hash(data);
        if !self.core.needs_reindex() && self.core.is_document_unchanged(filename, &document_hash) {
            tracing::info!(
                "Document {} unchanged since last index. Skipping re-embedding.",
                filename
            );
            return Ok(None);
        }

        let text = self.extract_pdf_text(data.to_vec()).await?;
        if text.trim().is_empty() {
            return Err(rag_core::EngineError::validation_no_chunk(
                rag_core::ValidationKind::EmptyText,
            )
            .into());
        }

        self.core
            .prepare_document(filename, &text, Some(&document_hash), batch_callback)
            .await
            .map_err(anyhow::Error::new)
    }

    #[instrument(skip(self, prepared), fields(doc = %prepared.document_name, chunks = prepared.chunks.len()))]
    pub async fn apply_prepared_document(&mut self, prepared: PreparedDocument) -> Result<usize> {
        let chunk_count = self
            .core
            .upsert_prepared_document(prepared)
            .map_err(anyhow::Error::new)?;

        Ok(chunk_count)
    }

    #[instrument(skip(self), fields(query = %query, count = count))]
    pub async fn get_embedding_candidates(
        &self,
        query: &str,
        count: usize,
    ) -> Result<Vec<RerankerCandidate>> {
        if let Some(reason) = &self.incompatible_index_reason {
            return Err(anyhow::anyhow!(reason.clone()));
        }
        self.core
            .embedding_candidates(query, count)
            .await
            .map_err(anyhow::Error::new)
    }

    #[instrument(skip(self, weights), fields(query = %query, top_k = top_k, ?weights))]
    pub async fn search(
        &self,
        query: &str,
        top_k: usize,
        weights: Option<&QueryWeights>,
    ) -> Result<Vec<SearchResult>> {
        if let Some(reason) = &self.incompatible_index_reason {
            return Err(anyhow::anyhow!(reason.clone()));
        }
        let start = std::time::Instant::now();
        let resolved = ResolvedWeights::from_query_weights(weights, self.embedding_weight);
        let weights = rag_core::SearchWeights {
            embedding: resolved.embedding,
            lexical: resolved.lexical,
            reranker: resolved.reranker,
            initial: resolved.initial,
        };

        let results = self
            .core
            .search(query, top_k, Some(weights))
            .await
            .map_err(anyhow::Error::new)?;

        tracing::info!(
            "Engine search completed in {:?} with {} results",
            start.elapsed(),
            results.len()
        );
        Ok(results)
    }

    #[instrument(skip(self, weights), fields(query = %query, top_k = top_k, diversity = diversity_factor, ?weights))]
    pub async fn search_with_diversity(
        &self,
        query: &str,
        top_k: usize,
        diversity_factor: f32,
        weights: Option<&QueryWeights>,
    ) -> Result<Vec<SearchResult>> {
        if let Some(reason) = &self.incompatible_index_reason {
            return Err(anyhow::anyhow!(reason.clone()));
        }
        let start = std::time::Instant::now();
        let resolved = ResolvedWeights::from_query_weights(weights, self.embedding_weight);
        let weights = rag_core::SearchWeights {
            embedding: resolved.embedding,
            lexical: resolved.lexical,
            reranker: resolved.reranker,
            initial: resolved.initial,
        };

        let results = self
            .core
            .search_with_diversity(query, top_k, diversity_factor, Some(weights))
            .await
            .map_err(anyhow::Error::new)?;

        tracing::info!(
            "Engine search (w/ diversity) completed in {:?} with {} results",
            start.elapsed(),
            results.len()
        );
        Ok(results)
    }

    pub fn list_documents(&self) -> Vec<String> {
        self.core.list_documents()
    }

    pub fn index_store(&self) -> Arc<SqliteIndexStore> {
        Arc::clone(&self.index_store)
    }

    pub fn remove_document(&mut self, document_name: &str) -> Result<usize> {
        self.core
            .remove_document(document_name)
            .map_err(anyhow::Error::new)
    }

    pub fn get_stats(&self) -> serde_json::Value {
        let doc_count = self.list_documents().len();
        let chunk_count = self.core.chunk_count();

        let status = if self.core.needs_reindex() {
            "reindexing"
        } else {
            "ready"
        };

        let reranker_model = self.core.reranker().map(|r| r.model_name());

        serde_json::json!({
            "documents": doc_count,
            "chunks": chunk_count,
            "status": status,
            "embedding_model": self.embedding_model(),
            "embedding_dim": self.core.health().embedding_dim,
            "needs_reindex_reason": self.incompatible_index_reason.as_deref(),
            "reranker_model": reranker_model,
            "uptime_seconds": self.start_time.elapsed().as_secs(),
            "version": env!("CARGO_PKG_VERSION")
        })
    }

    /// Async wrapper for PDF text extraction using spawn_blocking.
    ///
    /// Uses a two-stage fallback strategy:
    /// 1. Try pure-Rust extraction (lopdf) first for deployment flexibility
    /// 2. Fall back to pdftotext binary if lopdf fails
    async fn extract_pdf_text(&self, data: Vec<u8>) -> Result<String> {
        // Use Arc to share data between fallback tasks without cloning the underlying buffer
        let shared_data = std::sync::Arc::new(data);
        let data_for_lopdf = std::sync::Arc::clone(&shared_data);
        let data_for_fallback = std::sync::Arc::clone(&shared_data);

        let lopdf_result =
            tokio::task::spawn_blocking(move || Self::lopdf_extract_sync(&data_for_lopdf))
                .await
                .context("lopdf extraction task failed")?;

        match lopdf_result {
            Ok(text) => {
                tracing::info!(
                    "✅ PDF extracted using pure-Rust backend (lopdf): {} chars",
                    text.chars().count()
                );
                Ok(text)
            }
            Err(lopdf_err) => {
                tracing::warn!(
                    error = %lopdf_err,
                    "Pure-Rust PDF extraction failed, falling back to pdftotext"
                );

                let lopdf_empty = lopdf_err
                    .downcast_ref::<rag_core::EngineError>()
                    .map(|e| {
                        matches!(
                            e,
                            rag_core::EngineError::Validation {
                                kind: rag_core::ValidationKind::EmptyText,
                                ..
                            }
                        )
                    })
                    .unwrap_or(false);

                let pdftotext_result = tokio::task::spawn_blocking(move || {
                    Self::pdftotext_extract_sync(&data_for_fallback)
                })
                .await
                .context("pdftotext extraction task failed")?;

                match pdftotext_result {
                    Ok(text) => {
                        tracing::info!(
                            "✅ PDF extracted using pdftotext fallback: {} chars",
                            text.chars().count()
                        );
                        Ok(text)
                    }
                    Err(pdftotext_err) => {
                        // Check if pdftotext also returned EmptyText
                        let pdftotext_empty = pdftotext_err
                            .downcast_ref::<rag_core::EngineError>()
                            .map(|e| {
                                matches!(
                                    e,
                                    rag_core::EngineError::Validation {
                                        kind: rag_core::ValidationKind::EmptyText,
                                        ..
                                    }
                                )
                            })
                            .unwrap_or(false);

                        if pdftotext_empty {
                            return Err(pdftotext_err);
                        }

                        if lopdf_empty {
                            return Err(rag_core::EngineError::validation_no_chunk(
                                rag_core::ValidationKind::EmptyText,
                            )
                            .into());
                        }

                        tracing::error!(
                            lopdf_error = %lopdf_err,
                            pdftotext_error = %pdftotext_err,
                            "Both PDF extraction backends failed"
                        );
                        Err(anyhow::anyhow!(
                            "PDF extraction failed: lopdf error: {}, pdftotext error: {}",
                            lopdf_err,
                            pdftotext_err
                        ))
                    }
                }
            }
        }
    }

    fn lopdf_extract_sync(data: &[u8]) -> Result<String> {
        use lopdf::Document;

        let doc = Document::load_mem(data)
            .map_err(|e| anyhow::anyhow!("lopdf failed to parse PDF: {}", e))?;

        let pages = doc.get_pages();
        let mut all_text = String::with_capacity(pages.len() * 500);

        for (page_num, _page_id) in pages {
            match doc.extract_text(&[page_num]) {
                Ok(page_text) => {
                    if !all_text.is_empty() && !page_text.is_empty() {
                        all_text.push('\n');
                    }
                    all_text.push_str(&page_text);
                }
                Err(e) => {
                    tracing::debug!(
                        "lopdf: failed to extract text from page {}: {}",
                        page_num,
                        e
                    );
                }
            }
        }

        if all_text.trim().is_empty() {
            return Err(rag_core::EngineError::validation_no_chunk(
                rag_core::ValidationKind::EmptyText,
            )
            .into());
        }

        Ok(all_text)
    }

    /// Synchronous PDF extraction using pdftotext binary.
    /// Uses UUID for temp filename to prevent race conditions in concurrent calls.
    fn pdftotext_extract_sync(data: &[u8]) -> Result<String> {
        use std::process::Command;

        let temp_dir = std::env::temp_dir();
        let temp_file = temp_dir.join(format!("temp_pdf_{}.pdf", Uuid::new_v4()));

        std::fs::write(&temp_file, data)
            .map_err(|e| anyhow::anyhow!("Failed to write temp PDF: {}", e))?;

        let output = Command::new("pdftotext")
            .arg("-layout")
            .arg("-enc")
            .arg("UTF-8")
            .arg(&temp_file)
            .arg("-")
            .output();
        if let Err(e) = std::fs::remove_file(&temp_file) {
            tracing::debug!(
                error = %e,
                path = %temp_file.display(),
                "Failed to remove temp file after pdftotext"
            );
        }

        match output {
            Ok(output) if output.status.success() => {
                let text = String::from_utf8(output.stdout)
                    .unwrap_or_else(|e| String::from_utf8_lossy(&e.into_bytes()).to_string());

                if text.trim().is_empty() {
                    tracing::warn!("pdftotext extracted 0 characters");
                    Err(rag_core::EngineError::validation_no_chunk(
                        rag_core::ValidationKind::EmptyText,
                    )
                    .into())
                } else {
                    Ok(text)
                }
            }
            Ok(output) => {
                let error_msg = String::from_utf8_lossy(&output.stderr);
                tracing::warn!("pdftotext failed with error: {}", error_msg);
                Err(anyhow::anyhow!("pdftotext failed: {}", error_msg))
            }
            Err(e) => {
                tracing::warn!("Failed to run pdftotext command: {}", e);
                Err(anyhow::anyhow!(
                    "pdftotext command failed: {} (is poppler installed?)",
                    e
                ))
            }
        }
    }
}

fn compute_document_hash(data: &[u8]) -> String {
    let hash = Sha256::digest(data);
    format!("{hash:x}")
}

// Default score blending weights (can be overridden via environment variables)
const DEFAULT_LEXICAL_WEIGHT: f32 = 0.3;
const DEFAULT_RERANKER_WEIGHT: f32 = 0.7;
const DEFAULT_INITIAL_SCORE_WEIGHT: f32 = 0.3;

// Cached weight values using OnceLock for performance (avoids repeated env var reads)
static LEXICAL_WEIGHT: OnceLock<f32> = OnceLock::new();
static RERANKER_WEIGHT: OnceLock<f32> = OnceLock::new();
static INITIAL_SCORE_WEIGHT: OnceLock<f32> = OnceLock::new();

/// Parse a weight from environment variable with validation for finite values in [0.0, 1.0]
fn parse_weight(env_var: &str, default: f32) -> f32 {
    std::env::var(env_var)
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .filter(|w| w.is_finite() && (0.0..=1.0).contains(w))
        .unwrap_or(default)
}

fn get_lexical_weight() -> f32 {
    *LEXICAL_WEIGHT.get_or_init(|| parse_weight("RAG_LEXICAL_WEIGHT", DEFAULT_LEXICAL_WEIGHT))
}

fn get_reranker_weight() -> f32 {
    *RERANKER_WEIGHT.get_or_init(|| parse_weight("RAG_RERANKER_WEIGHT", DEFAULT_RERANKER_WEIGHT))
}

fn get_initial_score_weight() -> f32 {
    *INITIAL_SCORE_WEIGHT
        .get_or_init(|| parse_weight("RAG_INITIAL_SCORE_WEIGHT", DEFAULT_INITIAL_SCORE_WEIGHT))
}

/// Optional per-query weight overrides for search scoring.
/// All fields are optional - omitted weights fall back to cached defaults.
/// Invalid values (NaN, Inf, out of range) are ignored and defaults are used.
#[derive(
    Debug, Clone, Default, serde::Serialize, serde::Deserialize, rmcp::schemars::JsonSchema,
)]
#[schemars(crate = "rmcp::schemars")]
pub struct QueryWeights {
    /// Embedding similarity weight for first-stage retrieval (0.0-1.0)
    #[schemars(description = "Embedding similarity weight (0.0-1.0, default: 0.7)")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding: Option<f32>,
    /// Lexical/BM25 weight for first-stage retrieval (0.0-1.0)
    #[schemars(description = "Lexical/BM25 weight (0.0-1.0, default: 0.3)")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lexical: Option<f32>,
    /// Reranker weight for second-stage scoring (0.0-1.0)
    #[schemars(description = "Reranker weight for score blending (0.0-1.0, default: 0.7)")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reranker: Option<f32>,
    /// Initial score weight for second-stage scoring (0.0-1.0)
    #[schemars(description = "Initial score weight for score blending (0.0-1.0, default: 0.3)")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub initial: Option<f32>,
}

fn resolve_weight(override_weight: Option<f32>, default: f32) -> f32 {
    override_weight
        .filter(|&w| w.is_finite() && (0.0..=1.0).contains(&w))
        .unwrap_or(default)
}

#[derive(Debug, Clone, Copy)]
struct ResolvedWeights {
    embedding: f32,
    lexical: f32,
    reranker: f32,
    initial: f32,
}

impl ResolvedWeights {
    fn from_query_weights(weights: Option<&QueryWeights>, default_embedding_weight: f32) -> Self {
        Self {
            embedding: resolve_weight(weights.and_then(|w| w.embedding), default_embedding_weight),
            lexical: resolve_weight(weights.and_then(|w| w.lexical), get_lexical_weight()),
            reranker: resolve_weight(weights.and_then(|w| w.reranker), get_reranker_weight()),
            initial: resolve_weight(weights.and_then(|w| w.initial), get_initial_score_weight()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lopdf::content::Content;
    use lopdf::{Dictionary, Document, Object, Stream};
    use serial_test::serial;

    // Helper to create a PDF with no text
    fn create_empty_pdf() -> Vec<u8> {
        let mut doc = Document::with_version("1.7");
        let pages_id = doc.new_object_id();

        // Create an empty page content stream
        let content = Content { operations: vec![] };
        let stream = Stream::new(Dictionary::new(), content.encode().unwrap());
        let stream_id = doc.add_object(stream);

        // Create page dictionary
        let page_id = doc.add_object(Dictionary::from_iter(vec![
            ("Type", "Page".into()),
            ("Parent", pages_id.into()),
            ("Contents", stream_id.into()),
        ]));

        // Create pages dictionary
        let pages = Dictionary::from_iter(vec![
            ("Type", "Pages".into()),
            ("Kids", vec![page_id.into()].into()),
            ("Count", 1.into()),
        ]);
        doc.objects.insert(pages_id, Object::Dictionary(pages));

        // Create catalog
        let catalog_id = doc.add_object(Dictionary::from_iter(vec![
            ("Type", "Catalog".into()),
            ("Pages", pages_id.into()),
        ]));

        doc.trailer.set("Root", catalog_id);

        let mut buffer = Vec::new();
        doc.save_to(&mut buffer).unwrap();
        buffer
    }

    #[tokio::test]
    #[serial]
    async fn test_prepare_document_empty_text_returns_validation_error() {
        struct EnvVarGuard {
            key: &'static str,
            original: Option<std::ffi::OsString>,
        }

        impl EnvVarGuard {
            fn set(key: &'static str, value: &str) -> Self {
                let original = std::env::var_os(key);
                unsafe {
                    std::env::set_var(key, value);
                }
                Self { key, original }
            }
        }

        impl Drop for EnvVarGuard {
            fn drop(&mut self) {
                unsafe {
                    match &self.original {
                        Some(value) => std::env::set_var(self.key, value),
                        None => std::env::remove_var(self.key),
                    }
                }
            }
        }

        // Setup simple config
        let config = Config::default();
        let temp_dir = tempfile::tempdir().unwrap();
        let data_dir = temp_dir.path().to_str().unwrap();

        // Use wiremock to mock Ollama if needed.
        use wiremock::matchers::method;
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let mock_server = MockServer::start().await;

        let _ollama_url = EnvVarGuard::set("OLLAMA_URL", &mock_server.uri());
        let _ollama_embedding_model = EnvVarGuard::set("OLLAMA_EMBEDDING_MODEL", "test-model");

        // Mock response for any embedding call
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "embedding": vec![0.0; 384]
            })))
            .mount(&mock_server)
            .await;

        // Mock tags response for initialization
        Mock::given(method("GET"))
            .and(wiremock::matchers::path("/api/tags"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "models": [
                    { "name": "test-model:latest" }
                ]
            })))
            .mount(&mock_server)
            .await;

        let engine = RagEngine::new(data_dir, &config)
            .await
            .expect("Failed to create engine");

        let pdf_data = create_empty_pdf();

        let result = engine.prepare_document("empty.pdf", &pdf_data, None).await;

        match result {
            Ok(_) => panic!("Should have returned error"),
            Err(e) => {
                // Check if it is EngineError::Validation
                if let Some(engine_error) = e.downcast_ref::<rag_core::EngineError>() {
                    match engine_error {
                        rag_core::EngineError::Validation {
                            kind: rag_core::ValidationKind::EmptyText,
                            ..
                        } => {
                            // Success
                        }
                        _ => panic!("Expected Validation(EmptyText), got {engine_error:?}"),
                    }
                } else {
                    let msg = e.to_string();
                    if msg.contains("No text extracted from PDF")
                        || msg.contains("lopdf extracted no text")
                        || msg.contains("pdftotext produced no text")
                    {
                        panic!("Got old string error, expected EngineError::Validation: {msg}");
                    } else {
                        panic!("Got unexpected error type/msg: {e:?}");
                    }
                }
            }
        }
    }
}
