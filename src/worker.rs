use crate::job_manager::{JobManager, JobStatus};
use crate::job_payload::{ParsedReindexJobPayload, parse_reindex_job_payload};
use crate::progress_logger::{ProgressLogger, ProgressState, Stage};
use crate::rag_engine::RagEngine;
use anyhow::{Context, Result};
use std::collections::HashSet;
use std::ops::{Deref, DerefMut};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, RwLockWriteGuard, Semaphore, mpsc};
use tracing::Instrument;
use tracing::instrument;

/// Global timeout for lock acquisition in milliseconds.
/// Can be reduced in tests to verify timeout behavior.
static LOCK_TIMEOUT_MS: AtomicU64 = AtomicU64::new(30_000);

struct ReindexOutcome {
    total_docs: i64,
    failed_documents: Vec<String>,
}

#[derive(Clone)]
struct ProcessDocumentContext {
    rag_engine: Arc<RwLock<RagEngine>>,
    index_store: Arc<crate::SqliteIndexStore>,
    model_id: String,
    embedding_dim: usize,
    needs_reindex: bool,
    total_docs: i64,
}

impl ReindexOutcome {
    fn failure_summary(&self) -> Option<String> {
        if self.failed_documents.is_empty() {
            return None;
        }

        Some(format!(
            "Job completed with {} failures out of {} documents. Failed documents:\n{}",
            self.failed_documents.len(),
            self.total_docs,
            self.failed_documents.join("\n")
        ))
    }
}

/// Maximum allowed write lock duration in milliseconds.
/// This is an enforced design contract - locks held longer indicate
/// heavy work that should be moved out of the critical section.
pub const WRITE_LOCK_MAX_MS: u64 = 1000;

/// Test-visible metrics for lock duration verification.
#[cfg(test)]
pub mod lock_metrics {
    use std::sync::atomic::{AtomicU64, Ordering};

    static MAX_WRITE_LOCK_HELD_MS: AtomicU64 = AtomicU64::new(0);

    /// Reset the maximum lock duration metric. Call before tests.
    pub fn reset() {
        MAX_WRITE_LOCK_HELD_MS.store(0, Ordering::Relaxed);
    }

    /// Get the maximum observed write lock duration in milliseconds.
    pub fn max_held_ms() -> u64 {
        MAX_WRITE_LOCK_HELD_MS.load(Ordering::Relaxed)
    }

    /// Update max if new value is greater.
    pub(super) fn record_held_ms(ms: u64) {
        MAX_WRITE_LOCK_HELD_MS.fetch_max(ms, Ordering::Relaxed);
    }
}

/// Instrumented write lock guard that measures hold duration.
/// Logs timing info and warns if lock is held beyond threshold.
/// Generic over T to allow testing with simple types.
pub struct TimedWriteLockGuard<'a, T> {
    guard: RwLockWriteGuard<'a, T>,
    start: Instant,
    context: String,
}

impl<'a, T> TimedWriteLockGuard<'a, T> {
    /// Acquire an instrumented write lock.
    pub async fn acquire(lock: &'a RwLock<T>, context: impl Into<String>) -> Result<Self> {
        let context_str = context.into();
        let wait_start = Instant::now();
        // Enforce timeout for lock acquisition to prevent deadlocks
        let timeout_ms = LOCK_TIMEOUT_MS.load(Ordering::Relaxed);
        let guard =
            match tokio::time::timeout(Duration::from_millis(timeout_ms), lock.write()).await {
                Ok(g) => g,
                Err(_) => {
                    tracing::error!(
                        context = %context_str,
                        timeout_ms = timeout_ms,
                        "Failed to acquire write lock within timeout - potential deadlock"
                    );
                    return Err(anyhow::anyhow!(
                        "Timed out acquiring write lock for '{context_str}' after {timeout_ms}ms"
                    ));
                }
            };
        let wait_ms = wait_start.elapsed().as_millis() as u64;

        if wait_ms > 100 {
            tracing::debug!(wait_ms = wait_ms, "Write lock wait time");
        }

        Ok(Self {
            guard,
            start: Instant::now(),
            context: context_str,
        })
    }
}

impl<'a, T> Deref for TimedWriteLockGuard<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.guard
    }
}

impl<'a, T> DerefMut for TimedWriteLockGuard<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.guard
    }
}

impl<'a, T> Drop for TimedWriteLockGuard<'a, T> {
    fn drop(&mut self) {
        let held_ms = self.start.elapsed().as_millis() as u64;

        // Update test-visible metrics
        #[cfg(test)]
        lock_metrics::record_held_ms(held_ms);

        // Log with appropriate level based on threshold
        if held_ms > WRITE_LOCK_MAX_MS {
            tracing::warn!(
                lock_held_ms = held_ms,
                threshold_ms = WRITE_LOCK_MAX_MS,
                context = %self.context,
                "Write lock held beyond threshold"
            );
        } else {
            tracing::debug!(
                lock_held_ms = held_ms,
                context = %self.context,
                "Write lock released"
            );
        }
    }
}

#[derive(Debug)]
pub enum JobRequest {
    StartReindex {
        job_id: String,
        documents_dir: String,
    },
}

/// Background worker supervisor for processing async reindex jobs.
/// Manages job lifecycle, progress tracking, and document processing.
pub struct WorkerSupervisor {
    job_manager: Arc<JobManager>,
    rag_engine: Arc<RwLock<RagEngine>>,
    job_rx: mpsc::Receiver<JobRequest>,
    /// Semaphore limiting concurrent reindex workers to 2 (memory constraint: 32GB cap)
    worker_permits: Arc<Semaphore>,
}

impl WorkerSupervisor {
    pub fn new(
        job_manager: Arc<JobManager>,
        rag_engine: Arc<RwLock<RagEngine>>,
        job_rx: mpsc::Receiver<JobRequest>,
    ) -> Self {
        Self {
            job_manager,
            rag_engine,
            job_rx,
            // Start with 1 worker to validate memory usage (conservative approach)
            // Can scale to 2 workers after confirming memory stays under 32GB cap
            // (12GB base + 10-13GB per worker = 22-25GB total with 1 worker)
            worker_permits: Arc::new(Semaphore::new(1)),
        }
    }

    #[instrument(skip(self), name = "worker_supervisor_run")]
    pub async fn run(mut self) {
        // Resume any in-progress or pending jobs from DB
        if let Ok(mut jobs) = self.job_manager.find_resumable_jobs().await {
            jobs.sort_by(|a, b| {
                b.updated_at
                    .cmp(&a.updated_at)
                    .then_with(|| b.started_at.cmp(&a.started_at))
                    .then_with(|| b.job_id.cmp(&a.job_id))
            });

            if !jobs.is_empty() {
                // Guardrail: only allow one active reindex job on startup.
                // Choose the newest resumable job that has a valid payload and is compatible
                // with the current embedding model; mark the rest as failed to avoid confusion.
                let (current_model_id, current_dim) = {
                    let engine = self.rag_engine.read().await;
                    (
                        engine.embedding_model().to_string(),
                        engine.backend_embedding_dim(),
                    )
                };

                let mut handled_job_ids: HashSet<String> = HashSet::new();
                let mut selected: Option<(String, String, JobStatus, bool)> = None;

                for job in &jobs {
                    let Some(payload) = job.payload.as_deref() else {
                        handled_job_ids.insert(job.job_id.clone());
                        let _ = self
                            .job_manager
                            .update_status(
                                &job.job_id,
                                JobStatus::Failed,
                                Some("Missing payload (documents_dir); cannot resume".to_string()),
                            )
                            .await;
                        continue;
                    };

                    let parsed = parse_reindex_job_payload(payload.trim());
                    let documents_dir = match &parsed {
                        ParsedReindexJobPayload::Structured(payload) => {
                            payload.documents_dir.trim().to_string()
                        }
                        ParsedReindexJobPayload::LegacyDocumentsDir(dir) => dir.clone(),
                    };

                    if documents_dir.is_empty() || !Path::new(&documents_dir).exists() {
                        handled_job_ids.insert(job.job_id.clone());
                        let _ = self
                            .job_manager
                            .update_status(
                                &job.job_id,
                                JobStatus::Failed,
                                Some(format!(
                                    "Invalid documents_dir in payload: '{documents_dir}'"
                                )),
                            )
                            .await;
                        continue;
                    }

                    if let ParsedReindexJobPayload::Structured(payload) = parsed {
                        if payload.model_id != current_model_id {
                            handled_job_ids.insert(job.job_id.clone());
                            let _ = self
                                .job_manager
                                .update_status(
                                    &job.job_id,
                                    JobStatus::Failed,
                                    Some(format!(
                                        "reindex job model mismatch: job started with '{}' but server is running '{}'. reindex required under the current model.",
                                        payload.model_id, current_model_id
                                    )),
                                )
                                .await;
                            continue;
                        }
                        if payload.embedding_dim != current_dim {
                            handled_job_ids.insert(job.job_id.clone());
                            let _ = self
                                .job_manager
                                .update_status(
                                    &job.job_id,
                                    JobStatus::Failed,
                                    Some(format!(
                                        "reindex job embedding dimension mismatch: job expects {} but server backend is {} for model '{}'. reindex required.",
                                        payload.embedding_dim, current_dim, current_model_id
                                    )),
                                )
                                .await;
                            continue;
                        }
                    }

                    let already_complete = job.total > 0 && job.progress >= job.total;
                    selected = Some((
                        job.job_id.clone(),
                        documents_dir,
                        job.status.clone(),
                        already_complete,
                    ));
                    break;
                }

                if let Some((selected_job_id, documents_dir, status, already_complete)) = selected {
                    for job in &jobs {
                        if job.job_id == selected_job_id || handled_job_ids.contains(&job.job_id) {
                            continue;
                        }
                        let msg =
                            format!("Superseded by newer reindex job {selected_job_id} on startup");
                        let _ = self
                            .job_manager
                            .update_status(&job.job_id, JobStatus::Failed, Some(msg))
                            .await;
                    }

                    if already_complete {
                        let _ = self
                            .job_manager
                            .update_status(
                                &selected_job_id,
                                JobStatus::Completed,
                                Some("Recovered on startup: index already up to date".to_string()),
                            )
                            .await;
                    } else {
                        tracing::info!(
                            "Resuming job {} (status: {:?}) from restart",
                            selected_job_id,
                            status
                        );
                        self.spawn_reindex_worker(selected_job_id, documents_dir)
                            .await;
                    }
                } else {
                    for job in jobs {
                        if handled_job_ids.contains(&job.job_id) {
                            continue;
                        }
                        let _ = self
                            .job_manager
                            .update_status(
                                &job.job_id,
                                JobStatus::Failed,
                                Some("Missing payload (documents_dir); cannot resume".to_string()),
                            )
                            .await;
                    }
                }
            }
        }

        // Listen for new job requests
        while let Some(request) = self.job_rx.recv().await {
            match request {
                JobRequest::StartReindex {
                    job_id,
                    documents_dir,
                } => {
                    self.spawn_reindex_worker(job_id, documents_dir).await;
                }
            }
        }
    }

    #[instrument(skip(self), fields(job_id = %job_id))]
    async fn spawn_reindex_worker(&self, job_id: String, documents_dir: String) {
        let job_manager = self.job_manager.clone();
        let rag_engine = self.rag_engine.clone();
        let permits = self.worker_permits.clone();

        // Get log directory from environment
        let log_dir = std::env::var("LOG_DIR").unwrap_or_else(|_| {
            if std::path::Path::new("/var/log").exists() {
                "/var/log/rust-local-rag".to_string()
            } else {
                "./logs".to_string()
            }
        });

        // Create progress logger
        let progress_logger = match ProgressLogger::new(&log_dir).await {
            Ok(logger) => Some(logger),
            Err(e) => {
                tracing::error!("Failed to create progress logger: {}", e);
                None
            }
        };

        // Acquire permit - will wait naturally if all workers busy
        // No timeout: job stays queued until worker available (prevents job loss)
        let permit = match permits.acquire_owned().await {
            Ok(permit) => permit,
            Err(_) => {
                // Semaphore closed (should only happen during shutdown)
                tracing::warn!(
                    "Worker semaphore closed, job {} will not be processed",
                    job_id
                );
                // Mark job as failed so it's visible in status
                if let Err(e) = job_manager
                    .update_status(
                        &job_id,
                        JobStatus::Failed,
                        Some("Server shutdown".to_string()),
                    )
                    .await
                {
                    tracing::error!(
                        "Failed to mark job {} as failed during shutdown: {}",
                        job_id,
                        e
                    );
                }
                return;
            }
        };

        tokio::spawn(async move {
            // Hold permit for entire task lifetime (automatic cleanup on drop)
            let _permit = permit;

            tracing::info!("Starting reindex job {} (acquired worker permit)", job_id);

            // Mark as in progress
            if let Err(e) = job_manager
                .update_status(&job_id, JobStatus::InProgress, None)
                .await
            {
                tracing::error!("Failed to mark job {} as in progress: {}", job_id, e);
                return;
            }

            // Perform reindexing work WITHOUT holding the write lock for hours
            // Strategy: Process each document individually with brief locks per document
            let result = Self::reindex_documents(
                rag_engine.clone(),
                &documents_dir,
                job_manager.clone(),
                &job_id,
                progress_logger,
            )
            .await;

            match result {
                Ok(outcome) => {
                    let error = outcome.failure_summary();
                    if let Err(e) = job_manager
                        .update_status(&job_id, JobStatus::Completed, error)
                        .await
                    {
                        tracing::error!("Failed to mark job {} as completed: {}", job_id, e);
                    } else {
                        tracing::info!(
                            "Job {} completed successfully (releasing worker permit)",
                            job_id
                        );
                    }
                }
                Err(e) => {
                    let error_msg = e.to_string();
                    if let Err(update_err) = job_manager
                        .update_status(&job_id, JobStatus::Failed, Some(error_msg.clone()))
                        .await
                    {
                        tracing::error!("Failed to mark job {} as failed: {}", job_id, update_err);
                    }
                    tracing::error!(
                        "Job {} failed: {} (releasing worker permit)",
                        job_id,
                        error_msg
                    );
                }
            }
        });
    }

    #[instrument(skip(job_manager), fields(job_id = %job_id, total_docs = total_docs))]
    async fn init_job_tracking(job_manager: &JobManager, job_id: &str, total_docs: i64) {
        if let Err(e) = job_manager.update_total(job_id, total_docs).await {
            tracing::error!("Failed to set job total: {}", e);
        }
        if let Err(e) = job_manager.update_progress(job_id, 0).await {
            tracing::error!("Failed to initialize job progress: {}", e);
        }
    }

    async fn discover_pdf_paths(documents_dir: &str) -> Result<Vec<PathBuf>> {
        use walkdir::WalkDir;

        let allow_symlinks = std::env::var("RAG_ALLOW_SYMLINKS")
            .ok()
            .is_some_and(|v| v == "1");

        tokio::task::spawn_blocking({
            let dir = documents_dir.to_string();
            move || {
                let walker = WalkDir::new(&dir).follow_links(allow_symlinks);
                let mut paths = Vec::new();
                let mut skipped_symlinks = 0usize;

                for entry in walker.into_iter().filter_map(|e| e.ok()) {
                    if entry.path().extension().and_then(|s| s.to_str()) != Some("pdf") {
                        continue;
                    }

                    if !allow_symlinks && entry.file_type().is_symlink() {
                        skipped_symlinks += 1;
                        continue;
                    }

                    paths.push(entry.path().to_path_buf());
                }

                (paths, skipped_symlinks)
            }
        })
        .await
        .context("discover_pdf_paths task failed")
        .map(|(paths, skipped_symlinks)| {
            if skipped_symlinks > 0 {
                tracing::warn!(
                    skipped_symlinks,
                    "Skipped symlinked PDFs during ingestion. Set RAG_ALLOW_SYMLINKS=1 to allow indexing symlinks."
                );
            }
            paths
        })
    }

    async fn emit_logger(
        progress_logger: &Option<ProgressLogger>,
        progress_state: &ProgressState,
        kind: &str,
        note: Option<&str>,
    ) {
        #[allow(clippy::collapsible_if)]
        if let Some(logger) = progress_logger.as_ref() {
            if let Err(e) = logger.emit(progress_state, kind, note).await {
                tracing::error!("Failed to log progress: {}", e);
            }
        }
    }

    async fn finalize_reindex(
        rag_engine: &Arc<RwLock<RagEngine>>,
        progress_logger: &Option<ProgressLogger>,
        progress_state: &mut ProgressState,
    ) -> Result<()> {
        progress_state.stage = Stage::Finalize;
        Self::emit_logger(
            progress_logger,
            progress_state,
            "stage",
            Some("finalizing reindex"),
        )
        .await;

        let (index_store, model_id, embedding_dim) = {
            let engine = rag_engine.read().await;
            (
                engine.index_store(),
                engine.embedding_model().to_string(),
                engine.backend_embedding_dim(),
            )
        };

        {
            let mut engine = TimedWriteLockGuard::acquire(rag_engine, "finalize_reindex").await?;
            engine.finalize_reindex().await?;
        }

        index_store
            .set_model_needs_reindex(&model_id, embedding_dim, false)
            .await?;
        Ok(())
    }

    async fn try_process_single_pdf(
        progress_logger: &Option<ProgressLogger>,
        progress_state: &ProgressState,
        ctx: &ProcessDocumentContext,
        path: &Path,
        idx: usize,
    ) -> Result<Option<(String, Result<usize>)>> {
        let span = tracing::info_span!("process_document", ?path, idx);

        async move {
            let filename = match path.file_name().and_then(|n| n.to_str()) {
                Some(name) => name.to_string(),
                None => {
                    tracing::warn!("Skipping file with invalid path: {:?}", path);
                    return Ok(None);
                }
            };

            let data = match tokio::fs::read(path).await {
                Ok(data) => data,
                Err(e) => {
                    tracing::error!("Failed to read {}: {}", filename, e);
                    return Ok(None);
                }
            };

            tracing::info!(
                "Processing document {} ({}/{})",
                filename,
                idx + 1,
                ctx.total_docs
            );

            let logger_clone = progress_logger.clone();
            let filename_clone = filename.clone();
            let mut progress_state_clone = progress_state.clone();
            let current_idx = idx;

            let mut batch_callback = |batch_idx: usize,
                                      batch_count: usize,
                                      total_chunks: usize,
                                      chunks_in_batch: usize| {
                progress_state_clone.current_batch = Some(batch_idx);
                progress_state_clone.total_batches = Some(batch_count);
                progress_state_clone.current_chunks = Some(total_chunks);
                progress_state_clone.last_doc = Some(filename_clone.clone());
                progress_state_clone.done_docs = (current_idx + 1) as i64;

                if let Some(ref logger) = logger_clone {
                    let batch_progress = crate::progress_logger::BatchProgress {
                        document_name: filename_clone.clone(),
                        batch_index: batch_idx,
                        batch_count,
                        chunks_in_batch,
                        total_chunks,
                    };
                    let logger = logger.clone();
                    let state = progress_state_clone.clone();
                    tokio::spawn(async move {
                        if let Err(e) = logger.emit_batch(&state, &batch_progress).await {
                            tracing::error!("Failed to log batch progress: {}", e);
                        }
                    });
                }
            };

            let prepared = {
                let engine = ctx.rag_engine.read().await;
                engine
                    .prepare_document(&filename, &data, Some(&mut batch_callback))
                    .await
            };

            let result = match prepared {
                Ok(None) => Ok(0),
                Ok(Some(prepared)) => {
                    let document_hash = prepared.document_hash.as_deref().unwrap_or("");
                    if document_hash.is_empty() {
                        return Ok(Some((
                            filename,
                            Err(anyhow::anyhow!(
                                "Prepared document missing hash; cannot persist safely"
                            )),
                        )));
                    }

                    if let Err(e) = ctx
                        .index_store
                        .upsert_document_atomic(
                            &ctx.model_id,
                            ctx.embedding_dim,
                            ctx.needs_reindex,
                            &prepared.document_name,
                            document_hash,
                            &prepared.chunks,
                        )
                        .await
                    {
                        return Ok(Some((filename, Err(e))));
                    }

                    let chunk_count = {
                        let mut engine = match TimedWriteLockGuard::acquire(
                            &ctx.rag_engine,
                            format!("apply_document:{filename}"),
                        )
                        .await
                        {
                            Ok(engine) => engine,
                            Err(e) => return Ok(Some((filename, Err(e)))),
                        };

                        match engine.apply_prepared_document(prepared).await {
                            Ok(chunk_count) => chunk_count,
                            Err(e) => return Ok(Some((filename, Err(e)))),
                        }
                    };

                    Ok(chunk_count)
                }
                Err(e) => Err(e),
            };

            Ok(Some((filename, result)))
        }
        .instrument(span)
        .await
    }

    #[instrument(skip(rag_engine, job_manager, progress_logger), fields(job_id = %job_id, documents_dir = %documents_dir))]
    async fn reindex_documents(
        rag_engine: Arc<RwLock<RagEngine>>,
        documents_dir: &str,
        job_manager: Arc<JobManager>,
        job_id: &str,
        progress_logger: Option<ProgressLogger>,
    ) -> Result<ReindexOutcome> {
        let pdf_paths = Self::discover_pdf_paths(documents_dir).await?;
        let total_docs = pdf_paths.len() as i64;
        tracing::info!("Found {} PDF files to process", total_docs);

        Self::init_job_tracking(&job_manager, job_id, total_docs).await;

        let mut progress_state = ProgressState::new(job_id.to_string(), total_docs);
        let discovery_note = format!("discovered {total_docs} PDFs");
        Self::emit_logger(
            &progress_logger,
            &progress_state,
            "stage",
            Some(discovery_note.as_str()),
        )
        .await;

        let mut failed_documents = Vec::new();

        let (index_store, model_id, embedding_dim, mut needs_reindex, incompatible_index) = {
            let engine = rag_engine.read().await;
            (
                engine.index_store(),
                engine.embedding_model().to_string(),
                engine.backend_embedding_dim(),
                engine.needs_reindex(),
                engine.incompatible_index_reason().is_some(),
            )
        };

        // If we detected an incompatible index (e.g., embedding dimension mismatch), reset the
        // durable and in-memory state before indexing to avoid mixed-dimension rows on crash/resume.
        if incompatible_index {
            index_store
                .reset_model_state_atomic(&model_id, embedding_dim, true)
                .await?;
            let mut engine =
                TimedWriteLockGuard::acquire(&rag_engine, "reset_model_state_for_reindex").await?;
            engine.reset_state_for_reindex(embedding_dim)?;
            needs_reindex = true;
        }

        // Phase 2: Deletion synchronization - prune indexed docs missing from disk
        let current_docs: HashSet<String> = pdf_paths
            .iter()
            .filter_map(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .map(|s| s.to_string())
            })
            .collect();

        let indexed_docs = {
            let engine = rag_engine.read().await;
            engine.list_documents()
        };

        let docs_to_prune: Vec<String> = indexed_docs
            .into_iter()
            .filter(|doc| !current_docs.contains(doc))
            .collect();

        if !docs_to_prune.is_empty() {
            let note = format!("pruning {} deleted docs", docs_to_prune.len());
            Self::emit_logger(&progress_logger, &progress_state, "stage", Some(&note)).await;

            let mut removed_docs = Vec::new();
            let mut removed_chunks = 0usize;

            for doc in docs_to_prune {
                match index_store.delete_document_atomic(&model_id, &doc).await {
                    Ok(chunks_removed) => {
                        removed_docs.push(doc);
                        removed_chunks += chunks_removed;
                    }
                    Err(e) => {
                        failed_documents.push(format!("prune {doc}: {e}"));
                        progress_state.failed_docs += 1;
                    }
                }
            }

            if !removed_docs.is_empty() {
                let mut engine =
                    TimedWriteLockGuard::acquire(&rag_engine, "prune_deleted_documents").await?;
                for doc in &removed_docs {
                    if let Err(e) = engine.remove_document(doc) {
                        failed_documents.push(format!("prune {doc}: {e}"));
                        progress_state.failed_docs += 1;
                    }
                }
            }

            tracing::info!(
                "Pruned {} deleted documents ({} chunks) from index",
                removed_docs.len(),
                removed_chunks
            );
        }

        {
            let engine = rag_engine.read().await;
            if engine.needs_reindex() {
                tracing::info!(
                    "Reindexing {} documents with embedding model '{}'",
                    total_docs,
                    engine.embedding_model()
                );
            }
        }

        let mut successful_docs = 0i64;

        progress_state.stage = Stage::Embedding;
        Self::emit_logger(
            &progress_logger,
            &progress_state,
            "stage",
            Some("starting document embedding"),
        )
        .await;

        let process_ctx = ProcessDocumentContext {
            rag_engine: rag_engine.clone(),
            index_store: index_store.clone(),
            model_id: model_id.clone(),
            embedding_dim,
            needs_reindex,
            total_docs,
        };

        for (idx, path) in pdf_paths.iter().enumerate() {
            let Some((filename, result)) = Self::try_process_single_pdf(
                &progress_logger,
                &progress_state,
                &process_ctx,
                path,
                idx,
            )
            .await?
            else {
                continue;
            };

            progress_state.current_batch = None;
            progress_state.total_batches = None;
            progress_state.current_chunks = None;

            let progress_note = match &result {
                Ok(chunk_count) => format!("{chunk_count} chunks"),
                Err(_) => "failed".to_string(),
            };

            match result {
                Ok(chunk_count) => {
                    successful_docs += 1;
                    progress_state.success_docs += 1;

                    if chunk_count > 0 {
                        progress_state.embedded_docs += 1;
                        tracing::info!(
                            "Successfully processed {} with {} chunks ({}/{})",
                            filename,
                            chunk_count,
                            idx + 1,
                            total_docs
                        );
                    } else {
                        progress_state.skipped_docs += 1;
                        tracing::info!(
                            "{} is already up to date. No reindex needed. ({}/{})",
                            filename,
                            idx + 1,
                            total_docs
                        );
                    }
                }
                Err(e) => {
                    let error_msg = format!("{filename}: {e}");
                    failed_documents.push(error_msg);
                    progress_state.failed_docs += 1;
                    tracing::warn!(
                        "Failed to process {} ({}/{}): {}. Continuing with remaining documents.",
                        filename,
                        idx + 1,
                        total_docs,
                        e
                    );
                }
            }

            let progress = (idx + 1) as i64;
            progress_state.done_docs = progress;
            progress_state.last_doc = Some(filename);

            if let Err(e) = job_manager.update_progress(job_id, progress).await {
                tracing::error!("Failed to update job progress: {}", e);
            }

            Self::emit_logger(
                &progress_logger,
                &progress_state,
                "progress",
                Some(progress_note.as_str()),
            )
            .await;
        }

        Self::finalize_reindex(&rag_engine, &progress_logger, &mut progress_state).await?;

        if let Some(logger) = progress_logger.as_ref() {
            let completion_note = if failed_documents.is_empty() {
                format!("completed successfully - {total_docs} docs")
            } else {
                format!(
                    "completed with {} failures out of {total_docs}",
                    failed_documents.len()
                )
            };
            if let Err(e) = logger
                .emit(&progress_state, "done", Some(&completion_note))
                .await
            {
                tracing::error!("Failed to log completion: {e}");
            }
        }

        if failed_documents.is_empty() {
            tracing::info!("All {} documents processed successfully", total_docs);
        } else {
            tracing::info!(
                "Successfully processed {}/{} documents",
                successful_docs,
                total_docs
            );
        }

        Ok(ReindexOutcome {
            total_docs,
            failed_documents,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;
    use std::os::unix::fs::symlink;
    use std::time::Duration;

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

        fn unset(key: &'static str) -> Self {
            let original = std::env::var_os(key);
            unsafe {
                std::env::remove_var(key);
            }
            Self { key, original }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            match &self.original {
                Some(value) => unsafe {
                    std::env::set_var(self.key, value);
                },
                None => unsafe {
                    std::env::remove_var(self.key);
                },
            }
        }
    }

    #[tokio::test]
    #[serial]
    async fn test_prd_t1_3_discover_pdf_paths_skips_symlinks_by_default() {
        let _env = EnvVarGuard::unset("RAG_ALLOW_SYMLINKS");

        let documents_dir = tempfile::tempdir().unwrap();
        let real_pdf = documents_dir.path().join("real.pdf");
        std::fs::write(&real_pdf, b"%PDF-1.7\n").unwrap();

        let outside_dir = tempfile::tempdir().unwrap();
        let outside_pdf = outside_dir.path().join("outside.pdf");
        std::fs::write(&outside_pdf, b"%PDF-1.7\n").unwrap();

        let linked_pdf = documents_dir.path().join("linked.pdf");
        symlink(&outside_pdf, &linked_pdf).unwrap();

        let outside_subdir = tempfile::tempdir().unwrap();
        let outside_folder = outside_subdir.path().join("folder");
        std::fs::create_dir_all(&outside_folder).unwrap();
        let outside_folder_pdf = outside_folder.join("inside.pdf");
        std::fs::write(&outside_folder_pdf, b"%PDF-1.7\n").unwrap();

        let linked_dir = documents_dir.path().join("linked_dir");
        symlink(&outside_folder, &linked_dir).unwrap();
        let linked_dir_pdf = linked_dir.join("inside.pdf");

        let paths = WorkerSupervisor::discover_pdf_paths(documents_dir.path().to_str().unwrap())
            .await
            .unwrap();

        assert!(paths.contains(&real_pdf));
        assert!(!paths.contains(&linked_pdf));
        assert!(!paths.contains(&linked_dir_pdf));
    }

    #[tokio::test]
    #[serial]
    async fn test_prd_t1_3_discover_pdf_paths_allows_symlinks_with_override() {
        let _env = EnvVarGuard::set("RAG_ALLOW_SYMLINKS", "1");

        let documents_dir = tempfile::tempdir().unwrap();
        let real_pdf = documents_dir.path().join("real.pdf");
        std::fs::write(&real_pdf, b"%PDF-1.7\n").unwrap();

        let outside_dir = tempfile::tempdir().unwrap();
        let outside_pdf = outside_dir.path().join("outside.pdf");
        std::fs::write(&outside_pdf, b"%PDF-1.7\n").unwrap();

        let linked_pdf = documents_dir.path().join("linked.pdf");
        symlink(&outside_pdf, &linked_pdf).unwrap();

        let outside_subdir = tempfile::tempdir().unwrap();
        let outside_folder = outside_subdir.path().join("folder");
        std::fs::create_dir_all(&outside_folder).unwrap();
        let outside_folder_pdf = outside_folder.join("inside.pdf");
        std::fs::write(&outside_folder_pdf, b"%PDF-1.7\n").unwrap();

        let linked_dir = documents_dir.path().join("linked_dir");
        symlink(&outside_folder, &linked_dir).unwrap();
        let linked_dir_pdf = linked_dir.join("inside.pdf");

        let paths = WorkerSupervisor::discover_pdf_paths(documents_dir.path().to_str().unwrap())
            .await
            .unwrap();

        assert!(paths.contains(&real_pdf));
        assert!(paths.contains(&linked_pdf));
        assert!(paths.contains(&linked_dir_pdf));
    }

    /// Test that TimedWriteLockGuard correctly records lock duration.
    #[tokio::test]
    async fn test_timed_lock_guard_records_duration() {
        // Reset metrics before test
        lock_metrics::reset();

        let data = Arc::new(RwLock::new(42u64));

        // Acquire instrumented lock and hold for ~50ms
        {
            let mut guard = TimedWriteLockGuard::acquire(&data, "test_hold")
                .await
                .unwrap();
            *guard += 1;
            tokio::time::sleep(Duration::from_millis(50)).await;
        }

        // Verify metric was recorded (should be ~50ms)
        let max_ms = lock_metrics::max_held_ms();
        assert!(
            max_ms >= 50,
            "Lock should have been held at least 50ms, got {max_ms}ms"
        );
        assert!(
            max_ms < 200,
            "Lock should not have been held more than 200ms, got {max_ms}ms"
        );
    }

    /// Test that TimedWriteLockGuard Deref works correctly.
    #[tokio::test]
    async fn test_timed_lock_guard_deref() {
        let data = Arc::new(RwLock::new(vec!["a".to_string(), "b".to_string()]));

        // Acquire lock and read via Deref
        {
            let guard = TimedWriteLockGuard::acquire(&data, "test_deref")
                .await
                .unwrap();
            assert_eq!(guard.len(), 2);
            assert_eq!(guard[0], "a");
        }
    }

    /// Test that TimedWriteLockGuard DerefMut works correctly.
    #[tokio::test]
    async fn test_timed_lock_guard_deref_mut() {
        let data = Arc::new(RwLock::new(0u64));

        // Acquire lock and modify via DerefMut
        {
            let mut guard = TimedWriteLockGuard::acquire(&data, "test_deref_mut")
                .await
                .unwrap();
            *guard = 100;
        }

        // Verify value changed
        let value = *data.read().await;
        assert_eq!(value, 100);
    }

    /// Test that quick operations stay under threshold.
    #[tokio::test]
    async fn test_quick_lock_under_threshold() {
        lock_metrics::reset();

        let data = Arc::new(RwLock::new(String::new()));

        // Perform quick operation (no sleep)
        {
            let mut guard = TimedWriteLockGuard::acquire(&data, "quick_op")
                .await
                .unwrap();
            guard.push_str("hello");
        }

        // Should be well under 1 second
        let max_ms = lock_metrics::max_held_ms();
        assert!(
            max_ms < WRITE_LOCK_MAX_MS,
            "Quick op should be under {WRITE_LOCK_MAX_MS}ms threshold, got {max_ms}ms"
        );
    }

    /// Test that metrics track maximum across multiple locks.
    #[tokio::test]
    async fn test_metrics_track_maximum() {
        lock_metrics::reset();

        let data = Arc::new(RwLock::new(0u64));

        // First lock - 10ms
        {
            let _guard = TimedWriteLockGuard::acquire(&data, "lock1").await.unwrap();
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        // Second lock - 30ms (should update max)
        {
            let _guard = TimedWriteLockGuard::acquire(&data, "lock2").await.unwrap();
            tokio::time::sleep(Duration::from_millis(30)).await;
        }

        // Third lock - 15ms (should NOT update max)
        {
            let _guard = TimedWriteLockGuard::acquire(&data, "lock3").await.unwrap();
            tokio::time::sleep(Duration::from_millis(15)).await;
        }

        // Max should be from second lock (~30ms)
        let max_ms = lock_metrics::max_held_ms();
        assert!(
            max_ms >= 30,
            "Max should be at least 30ms from second lock, got {max_ms}ms"
        );
        assert!(max_ms < 60, "Max should be less than 60ms, got {max_ms}ms");
    }

    #[tokio::test]
    async fn test_lock_acquisition_timeout() {
        let old_timeout = LOCK_TIMEOUT_MS.swap(100, Ordering::Relaxed);
        struct ResetTimeout(u64);
        impl Drop for ResetTimeout {
            fn drop(&mut self) {
                LOCK_TIMEOUT_MS.store(self.0, Ordering::Relaxed);
            }
        }
        let _reset = ResetTimeout(old_timeout);

        let data = Arc::new(RwLock::new(0));
        let data_clone = data.clone();

        // Hold lock indefinitely
        let _guard = data.write().await;

        // Spawn task to acquire lock - should return an error after timeout
        let handle = tokio::spawn(async move {
            TimedWriteLockGuard::acquire(&data_clone, "test_timeout")
                .await
                .is_err()
        });

        let timed_out = handle.await.unwrap();
        assert!(timed_out, "Expected lock acquisition to time out");
    }
}
