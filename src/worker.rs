use crate::job_manager::{JobManager, JobStatus};
use crate::progress_logger::{ProgressLogger, ProgressState, Stage};
use crate::rag_engine::RagEngine;
use anyhow::Result;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{RwLock, RwLockWriteGuard, Semaphore, mpsc};

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
    pub async fn acquire(lock: &'a RwLock<T>, context: impl Into<String>) -> Self {
        let wait_start = Instant::now();
        let guard = lock.write().await;
        let wait_ms = wait_start.elapsed().as_millis() as u64;

        if wait_ms > 100 {
            tracing::debug!(wait_ms = wait_ms, "Write lock wait time");
        }

        Self {
            guard,
            start: Instant::now(),
            context: context.into(),
        }
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

    pub async fn run(mut self) {
        // Resume any in-progress or pending jobs from DB
        if let Ok(jobs) = self.job_manager.find_resumable_jobs().await {
            for job in jobs {
                tracing::info!(
                    "Resuming job {} (status: {:?}) from restart",
                    job.job_id,
                    job.status
                );
                if let Some(payload) = job.payload {
                    self.spawn_reindex_worker(job.job_id, payload).await;
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
        let progress_logger = match ProgressLogger::new(&log_dir) {
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
                Ok(_) => {
                    if let Err(e) = job_manager
                        .update_status(&job_id, JobStatus::Completed, None)
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

    async fn reindex_documents(
        rag_engine: Arc<RwLock<RagEngine>>,
        documents_dir: &str,
        job_manager: Arc<JobManager>,
        job_id: &str,
        progress_logger: Option<ProgressLogger>,
    ) -> Result<()> {
        use walkdir::WalkDir;

        // Step 1: Discover PDF files (blocking operation, no lock needed)
        let pdf_paths: Vec<_> = tokio::task::spawn_blocking({
            let dir = documents_dir.to_string();
            move || {
                WalkDir::new(&dir)
                    .into_iter()
                    .filter_map(|e| e.ok())
                    .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("pdf"))
                    .map(|e| e.path().to_path_buf())
                    .collect::<Vec<_>>()
            }
        })
        .await?;

        let total_docs = pdf_paths.len() as i64;
        tracing::info!("Found {} PDF files to process", total_docs);

        // Update job total and progress in JobManager
        if let Err(e) = job_manager.update_total(job_id, total_docs).await {
            tracing::error!("Failed to set job total: {}", e);
        }
        if let Err(e) = job_manager.update_progress(job_id, 0).await {
            tracing::error!("Failed to initialize job progress: {}", e);
        }

        // Initialize progress state
        let mut progress_state = ProgressState::new(job_id.to_string(), total_docs);

        // Log discovery stage completion
        if let Some(ref logger) = progress_logger {
            if let Err(e) = logger
                .emit(
                    &progress_state,
                    "stage",
                    Some(&format!("discovered {total_docs} PDFs")),
                )
                .await
            {
                tracing::error!("Failed to log discovery stage: {}", e);
            }
        }

        // Update job total
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

        // Track failed documents for poison pill handling
        let mut failed_documents = Vec::new();
        let mut successful_count = 0;

        // Change to embedding stage
        progress_state.stage = Stage::Embedding;
        if let Some(ref logger) = progress_logger {
            if let Err(e) = logger
                .emit(
                    &progress_state,
                    "stage",
                    Some("starting document embedding"),
                )
                .await
            {
                tracing::error!("Failed to log embedding stage: {}", e);
            }
        }

        // Step 2: Process each document with brief locks per document
        for (idx, path) in pdf_paths.iter().enumerate() {
            // Handle potential file path issues gracefully
            let filename = match path.file_name().and_then(|n| n.to_str()) {
                Some(name) => name,
                None => {
                    tracing::warn!("Skipping file with invalid path: {:?}", path);
                    continue;
                }
            };

            // Read file (async, no lock)
            let data = match tokio::fs::read(&path).await {
                Ok(data) => data,
                Err(e) => {
                    tracing::error!("Failed to read {}: {}", filename, e);
                    continue;
                }
            };

            tracing::info!(
                "Processing document {} ({}/{})",
                filename,
                idx + 1,
                total_docs
            );

            // Add document with brief write lock (minutes per document, not hours total)
            // Create batch progress callback
            let logger_clone = progress_logger.clone();
            let filename_clone = filename.to_string();
            let mut progress_state_clone = progress_state.clone();
            let current_idx = idx;

            let result = {
                // Use instrumented lock guard for timing visibility
                let mut engine =
                    TimedWriteLockGuard::acquire(&rag_engine, format!("add_document:{filename}"))
                        .await;

                // Define batch callback
                let mut batch_callback =
                    |batch_idx: usize,
                     batch_count: usize,
                     total_chunks: usize,
                     chunks_in_batch: usize| {
                        // Update state with batch progress and current document position
                        progress_state_clone.current_batch = Some(batch_idx);
                        progress_state_clone.total_batches = Some(batch_count);
                        progress_state_clone.current_chunks = Some(total_chunks);
                        progress_state_clone.last_doc = Some(filename_clone.clone());
                        // Update done_docs to show monotonic progress during batch embedding
                        progress_state_clone.done_docs = (current_idx + 1) as i64;

                        // Emit batch progress asynchronously (spawn to avoid blocking)
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

                engine
                    .add_document(filename, &data, Some(&mut batch_callback))
                    .await
            };

            // Clear batch progress after document completes
            progress_state.current_batch = None;
            progress_state.total_batches = None;
            progress_state.current_chunks = None;

            // Capture progress note before consuming result
            let progress_note = match &result {
                Ok(chunk_count) => format!("{chunk_count} chunks"),
                Err(_) => "failed".to_string(),
            };

            match result {
                Ok(chunk_count) => {
                    successful_count += 1;
                    progress_state.success_docs += 1;

                    // Track skipped vs embedded separately
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
                    failed_documents.push(error_msg.clone());
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

            // Update progress
            let progress = (idx + 1) as i64;
            progress_state.done_docs = progress;
            progress_state.last_doc = Some(filename.to_string());

            if let Err(e) = job_manager.update_progress(job_id, progress).await {
                tracing::error!("Failed to update job progress: {}", e);
            }

            // Log progress
            if let Some(ref logger) = progress_logger {
                if let Err(e) = logger
                    .emit(&progress_state, "progress", Some(&progress_note))
                    .await
                {
                    tracing::error!("Failed to log progress: {}", e);
                }
            }
        }

        // Finalize reindex
        progress_state.stage = Stage::Finalize;
        if let Some(ref logger) = progress_logger {
            if let Err(e) = logger
                .emit(&progress_state, "stage", Some("finalizing reindex"))
                .await
            {
                tracing::error!("Failed to log finalize stage: {}", e);
            }
        }

        {
            // Use instrumented lock guard for timing visibility
            let mut engine = TimedWriteLockGuard::acquire(&rag_engine, "finalize_reindex").await;
            engine.finalize_reindex().await?;
        }

        // Log completion
        if let Some(ref logger) = progress_logger {
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
                tracing::error!("Failed to log completion: {}", e);
            }
        }

        // Report poison pill documents if any
        if !failed_documents.is_empty() {
            let failure_summary = format!(
                "Job completed with {} failures out of {} documents. Failed documents:\n{}",
                failed_documents.len(),
                total_docs,
                failed_documents.join("\n")
            );
            tracing::warn!("{}", failure_summary);

            // Update job with partial failure status
            if let Err(e) = job_manager
                .update_status(job_id, JobStatus::Completed, Some(failure_summary))
                .await
            {
                tracing::error!("Failed to update job with failure summary: {}", e);
            }

            tracing::info!(
                "Successfully processed {}/{} documents",
                successful_count,
                total_docs
            );
        } else {
            tracing::info!("All {} documents processed successfully", total_docs);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    /// Test that TimedWriteLockGuard correctly records lock duration.
    #[tokio::test]
    async fn test_timed_lock_guard_records_duration() {
        // Reset metrics before test
        lock_metrics::reset();

        let data = Arc::new(RwLock::new(42u64));

        // Acquire instrumented lock and hold for ~50ms
        {
            let mut guard = TimedWriteLockGuard::acquire(&data, "test_hold").await;
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
            let guard = TimedWriteLockGuard::acquire(&data, "test_deref").await;
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
            let mut guard = TimedWriteLockGuard::acquire(&data, "test_deref_mut").await;
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
            let mut guard = TimedWriteLockGuard::acquire(&data, "quick_op").await;
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
            let _guard = TimedWriteLockGuard::acquire(&data, "lock1").await;
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        // Second lock - 30ms (should update max)
        {
            let _guard = TimedWriteLockGuard::acquire(&data, "lock2").await;
            tokio::time::sleep(Duration::from_millis(30)).await;
        }

        // Third lock - 15ms (should NOT update max)
        {
            let _guard = TimedWriteLockGuard::acquire(&data, "lock3").await;
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
}
