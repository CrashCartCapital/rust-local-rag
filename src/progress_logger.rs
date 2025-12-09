use anyhow::Result;
use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::Mutex;

/// Processing stage for reindexing
#[derive(Debug, Clone, Copy)]
pub enum Stage {
    Discover,
    Embedding,
    Finalize,
}

impl fmt::Display for Stage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Stage::Discover => write!(f, "discover"),
            Stage::Embedding => write!(f, "embedding"),
            Stage::Finalize => write!(f, "finalize"),
        }
    }
}

/// Batch progress information for incremental updates during embedding
#[derive(Debug, Clone)]
pub struct BatchProgress {
    pub document_name: String,
    pub batch_index: usize,
    pub batch_count: usize,
    pub chunks_in_batch: usize,
    pub total_chunks: usize,
}

/// Progress state for a job
#[derive(Clone)]
pub struct ProgressState {
    pub job_id: String,
    pub stage: Stage,
    pub total_docs: i64,
    pub done_docs: i64,
    pub success_docs: i64,
    pub failed_docs: i64,
    pub skipped_docs: i64,  // Documents skipped (hash match)
    pub embedded_docs: i64, // Documents actually embedded
    pub last_doc: Option<String>,
    pub started: Instant,
    // Current batch progress (if embedding)
    pub current_batch: Option<usize>,
    pub total_batches: Option<usize>,
    pub current_chunks: Option<usize>,
}

impl ProgressState {
    pub fn new(job_id: String, total_docs: i64) -> Self {
        Self {
            job_id,
            stage: Stage::Discover,
            total_docs,
            done_docs: 0,
            success_docs: 0,
            failed_docs: 0,
            skipped_docs: 0,
            embedded_docs: 0,
            last_doc: None,
            started: Instant::now(),
            current_batch: None,
            total_batches: None,
            current_chunks: None,
        }
    }

    /// Calculate docs per second
    pub fn docs_per_sec(&self) -> f64 {
        let elapsed = self.started.elapsed().as_secs_f64();
        if elapsed > 0.0 && self.done_docs > 0 {
            self.done_docs as f64 / elapsed
        } else {
            0.0
        }
    }

    /// Calculate ETA in seconds
    pub fn eta_seconds(&self) -> u64 {
        let dps = self.docs_per_sec();
        if dps > 0.0 {
            let remaining = self.total_docs - self.done_docs;
            (remaining as f64 / dps) as u64
        } else {
            0
        }
    }

    /// Calculate progress percentage
    pub fn percent(&self) -> i64 {
        if self.total_docs > 0 {
            (self.done_docs * 100) / self.total_docs
        } else {
            0
        }
    }
}

/// Logger for progress tracking
#[derive(Clone)]
pub struct ProgressLogger {
    writer: Arc<Mutex<BufWriter<File>>>,
}

impl ProgressLogger {
    /// Create a new progress logger in the specified directory
    pub fn new(log_dir: &str) -> Result<Self> {
        std::fs::create_dir_all(log_dir)?;
        let log_path = format!("{log_dir}/progress_tracking.log");

        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)?;

        let writer = BufWriter::new(file);

        Ok(Self {
            writer: Arc::new(Mutex::new(writer)),
        })
    }

    /// Emit a progress event
    /// Event types: progress | stage | done | error | batch
    pub async fn emit(&self, state: &ProgressState, event: &str, note: Option<&str>) -> Result<()> {
        let ts = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis();

        let dps = state.docs_per_sec();
        let eta = state.eta_seconds();
        let pct = state.percent();

        let last_doc = state.last_doc.as_deref().unwrap_or("");
        let note_str = note.unwrap_or("");

        // URL-encode fields that may have spaces
        let last_doc_encoded = urlencoding::encode(last_doc);
        let note_encoded = urlencoding::encode(note_str);

        // Include batch progress if available
        let batch_info =
            if let (Some(cur), Some(total)) = (state.current_batch, state.total_batches) {
                format!(" current_batch={cur} total_batches={total}")
            } else {
                String::new()
            };

        let line = format!(
            "ts={} job={} event={} stage={} done={} total={} success={} failed={} skipped={} embedded={} pct={} dps={:.2} eta_s={} last_doc={} note={}{}\n",
            ts,
            state.job_id,
            event,
            state.stage,
            state.done_docs,
            state.total_docs,
            state.success_docs,
            state.failed_docs,
            state.skipped_docs,
            state.embedded_docs,
            pct,
            dps,
            eta,
            last_doc_encoded,
            note_encoded,
            batch_info,
        );

        let mut guard = self.writer.lock().await;
        guard.write_all(line.as_bytes())?;
        guard.flush()?;

        Ok(())
    }

    /// Emit a batch progress event for incremental updates during embedding
    pub async fn emit_batch(
        &self,
        state: &ProgressState,
        batch_progress: &BatchProgress,
    ) -> Result<()> {
        let ts = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis();

        let doc_encoded = urlencoding::encode(&batch_progress.document_name);
        let batch_pct = if batch_progress.batch_count > 0 {
            (batch_progress.batch_index * 100) / batch_progress.batch_count
        } else {
            0
        };

        // Include full counter set for consistency with progress events
        let pct = state.percent();

        let line = format!(
            "ts={} job={} event=batch stage=embedding done={} total={} success={} failed={} skipped={} embedded={} pct={} last_doc={} current_batch={} total_batches={} batch_pct={} total_chunks={} chunks_in_batch={} note=batch%20{}/{}%20complete\n",
            ts,
            state.job_id,
            state.done_docs,
            state.total_docs,
            state.success_docs,
            state.failed_docs,
            state.skipped_docs,
            state.embedded_docs,
            pct,
            doc_encoded,
            batch_progress.batch_index,
            batch_progress.batch_count,
            batch_pct,
            batch_progress.total_chunks,
            batch_progress.chunks_in_batch,
            batch_progress.batch_index,
            batch_progress.batch_count,
        );

        let mut guard = self.writer.lock().await;
        guard.write_all(line.as_bytes())?;
        guard.flush()?;

        Ok(())
    }
}
