use anyhow::Result;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing_subscriber::EnvFilter;

mod embeddings;
mod mcp_server;
mod rag_engine;

use rag_engine::RagEngine;

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
        .open(format!("{}/test_write", path))
        .map(|_| {
            let _ = std::fs::remove_file(format!("{}/test_write", path));
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

    if is_development || force_console {
        tracing_subscriber::fmt()
            .with_env_filter(env_filter)
            .compact()
            .init();
        tracing::info!("Development mode: logging to console");
    } else {
        let log_file = format!("{}/rust-local-rag.log", log_dir);
        let file_appender = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_file)?;

        tracing_subscriber::fmt()
            .with_env_filter(env_filter)
            .with_writer(file_appender)
            .json()
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
    let log_file = format!("{}/rust-local-rag.log", log_dir);

    tokio::spawn(async move {
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(300));

        loop {
            interval.tick().await;

            if let Ok(metadata) = std::fs::metadata(&log_file) {
                if metadata.len() > max_bytes {
                    if let Err(e) = std::fs::write(
                        &log_file,
                        format!("[LOG TRUNCATED - Size exceeded {}MB]\n", max_mb),
                    ) {
                        eprintln!("Failed to truncate log file: {}", e);
                    }
                }
            }
        }
    });
}

#[tokio::main]
async fn main() -> Result<()> {
    if let Err(e) = dotenv::dotenv() {
        eprintln!("Warning: Could not load .env file: {}", e);
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
            "Embedding model changed to '{}'. Existing embeddings were cleared and a full reindex will start shortly.",
            rag_engine.embedding_model()
        );
    }

    let rag_state = Arc::new(RwLock::new(rag_engine));

    let document_loading_state = rag_state.clone();
    let docs_dir = documents_dir.clone();
    tokio::spawn(async move {
        tracing::info!("Starting document loading in background...");
        let mut engine = document_loading_state.write().await;

        if engine.needs_reindex() {
            tracing::info!(
                "Starting reindex to rebuild embeddings using model '{}'...",
                engine.embedding_model()
            );
        }

        if let Err(e) = engine.load_documents_from_dir(&docs_dir).await {
            tracing::error!("Failed to load documents: {}", e);
        } else {
            tracing::info!("Document loading completed successfully");
        }
    });

    tracing::info!("Starting MCP server (stdin/stdout mode)");
    tracing::info!("Data directory: {}", data_dir);
    tracing::info!("Documents directory: {}", documents_dir);
    tracing::info!(
        "Ollama Model: {}",
        std::env::var("OLLAMA_EMBEDDING_MODEL")
            .unwrap_or_else(|_| "nomic-embed-text (default)".to_string())
    );

    mcp_server::start_mcp_server(rag_state).await?;

    Ok(())
}
