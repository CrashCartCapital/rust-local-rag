use anyhow::Result;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing_subscriber::{EnvFilter, layer::SubscriberExt, util::SubscriberInitExt};

mod embeddings;
mod job_manager;
mod mcp_server;
mod progress_logger;
mod rag_engine;
mod reranker;
mod worker;

use job_manager::JobManager;
use rag_engine::RagEngine;
use worker::{JobRequest, WorkerSupervisor};
use tokio::sync::mpsc;

fn get_data_dir() -> String {
    std::env::var("DATA_DIR").unwrap_or_else(|_| "./data".to_string())
}

fn get_documents_dir() -> String {
    std::env::var("DOCUMENTS_DIR").unwrap_or_else(|_| "./documents".to_string())
}

fn get_log_dir() -> String {
    std::env::var("LOG_DIR").unwrap_or_else(|_| {
        if std::path::Path::new("/var/log").exists() && is_writable("/var/log") {
            "/var/log/rust-local-rag".to_string()
        } else {
            "./logs".to_string()
        }
    })
}

fn get_log_level() -> String {
    std::env::var("LOG_LEVEL").unwrap_or_else(|_| "info".to_string())
}

fn get_log_max_mb() -> u64 {
    std::env::var("LOG_MAX_MB")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5)
}

fn is_writable(path: &str) -> bool {
    std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .open(format!("{path}/test_write"))
        .map(|_| {
            let _ = std::fs::remove_file(format!("{path}/test_write"));
            true
        })
        .unwrap_or(false)
}

fn setup_logging() -> Result<()> {
    let log_dir = get_log_dir();
    let log_level = get_log_level();
    let log_max_mb = get_log_max_mb();

    std::fs::create_dir_all(&log_dir)?;

    let env_filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(&log_level));

    let is_development = std::env::var("DEVELOPMENT").is_ok() || std::env::var("DEV").is_ok();
    let force_console = std::env::var("CONSOLE_LOGS").is_ok();

    // Always write to file for tracker compatibility
    let log_file = format!("{log_dir}/rust-local-rag.log");
    let file_appender = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_file)?;

    let file_layer = tracing_subscriber::fmt::layer()
        .with_writer(file_appender)
        .json();

    if is_development || force_console {
        // Development mode: log to BOTH console and file
        let console_layer = tracing_subscriber::fmt::layer()
            .compact()
            .with_writer(std::io::stdout);

        tracing_subscriber::registry()
            .with(env_filter)
            .with(file_layer)
            .with(console_layer)
            .init();

        tracing::info!("Development mode: logging to console AND file");
    } else {
        // Production mode: log to file only
        tracing_subscriber::registry()
            .with(env_filter)
            .with(file_layer)
            .init();
    }

    tracing::info!("Logging initialized");
    tracing::info!("Log directory: {}", log_dir);
    tracing::info!("Log level: {}", log_level);
    tracing::info!("Log max size: {}MB (auto-truncate)", log_max_mb);
    tracing::info!("Development mode: {}", is_development || force_console);

    Ok(())
}

async fn start_log_cleanup_task(log_dir: String, max_mb: u64) {
    let max_bytes = max_mb * 1024 * 1024;
    let log_file = format!("{log_dir}/rust-local-rag.log");

    tokio::spawn(async move {
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(300));

        loop {
            interval.tick().await;

            if let Ok(metadata) = std::fs::metadata(&log_file) {
                if metadata.len() <= max_bytes {
                    continue;
                }

                if let Err(e) = std::fs::write(
                    &log_file,
                    format!("[LOG TRUNCATED - Size exceeded {max_mb}MB]\n"),
                ) {
                    tracing::warn!("Failed to truncate log file: {e}");
                }
            }
        }
    });
}

#[tokio::main]
async fn main() -> Result<()> {
    if let Err(e) = dotenv::dotenv() {
        eprintln!("Warning: Could not load .env file: {e}");
    }
    setup_logging()?;

    let data_dir = get_data_dir();
    let documents_dir = get_documents_dir();
    let log_dir = get_log_dir();
    let log_max_mb = get_log_max_mb();

    tokio::fs::create_dir_all(&data_dir).await?;
    tokio::fs::create_dir_all(&documents_dir).await?;

    start_log_cleanup_task(log_dir, log_max_mb).await;
    tracing::info!("Started automatic log cleanup task (max: {}MB)", log_max_mb);

    let rag_engine = RagEngine::new(&data_dir).await?;

    if rag_engine.needs_reindex() {
        tracing::warn!(
            "Embedding model changed to '{}'. Existing embeddings were cleared and a full reindex will be available via start_reindex tool.",
            rag_engine.embedding_model()
        );
    }

    let rag_state = Arc::new(RwLock::new(rag_engine));

    // Initialize job system
    let job_db_path = format!("sqlite://{data_dir}/jobs.db");
    let job_manager = Arc::new(JobManager::new(&job_db_path).await?);
    tracing::info!("Job manager initialized with database at {}", job_db_path);

    // Create job request channel
    let (job_tx, job_rx) = mpsc::channel::<JobRequest>(100);

    // Spawn worker supervisor and monitor it
    let supervisor = WorkerSupervisor::new(
        job_manager.clone(),
        rag_state.clone(),
        job_rx,
    );
    let supervisor_handle = tokio::spawn(supervisor.run());
    tracing::info!("Worker supervisor started");

    // Monitor supervisor for panics
    let supervisor_monitor = tokio::spawn(async move {
        match supervisor_handle.await {
            Ok(_) => {
                tracing::error!("Worker supervisor task exited unexpectedly");
            }
            Err(e) => {
                tracing::error!("Worker supervisor task panicked: {}", e);
            }
        }
    });

    tracing::info!("Data directory: {}", data_dir);
    tracing::info!("Documents directory: {}", documents_dir);
    tracing::info!(
        "Ollama Model: {}",
        std::env::var("OLLAMA_EMBEDDING_MODEL")
            .unwrap_or_else(|_| "nomic-embed-text (default)".to_string())
    );
    tracing::info!("Use start_reindex tool to begin document indexing");

    // Clone rag_state for server, keeping original for shutdown cleanup
    let server_rag_state = rag_state.clone();

    // Start MCP server (blocks until shutdown, handles Ctrl+C internally)
    // Propagate errors to ensure non-zero exit code on failure
    let server_result: Result<()> = tokio::select! {
        result = mcp_server::start_mcp_server(server_rag_state, job_manager, job_tx, documents_dir) => {
            result
        }
        _ = supervisor_monitor => {
            Err(anyhow::anyhow!("Worker supervisor exited unexpectedly - critical system failure"))
        }
    };

    // --- Graceful Shutdown: Flush state to disk with timeouts ---
    tracing::info!("Initiating graceful shutdown...");

    // Acquire lock with 10s timeout
    tracing::info!("Acquiring lock for flush (10s timeout)...");
    match tokio::time::timeout(
        std::time::Duration::from_secs(10),
        rag_state.write()
    ).await {
        Ok(engine) => {
            // Flush to disk with 5s timeout
            tracing::info!("Lock acquired. Flushing state to disk (5s timeout)...");
            match tokio::time::timeout(
                std::time::Duration::from_secs(5),
                engine.save_to_disk()
            ).await {
                Ok(Ok(())) => tracing::info!("✅ RAG state successfully saved to disk"),
                Ok(Err(e)) => tracing::error!("❌ Failed to save state: {}", e),
                Err(_) => tracing::error!("⚠️ Save operation timed out after 5s"),
            }
        }
        Err(_) => {
            // Lock held by stuck task - exit without saving
            tracing::error!("⚠️ Could not acquire lock within 10s. Exiting without save.");
        }
    }

    tracing::info!("MCP server shut down gracefully");
    server_result
}
