use anyhow::Result;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing_subscriber::{EnvFilter, layer::SubscriberExt, util::SubscriberInitExt};

use rust_local_rag::{Config, JobManager, RagEngine, WorkerSupervisor};
use tokio::sync::mpsc;

fn get_data_dir() -> String {
    std::env::var("DATA_DIR").unwrap_or_else(|_| "./data".to_string())
}

fn get_documents_dir() -> String {
    std::env::var("DOCUMENTS_DIR").unwrap_or_else(|_| "./documents".to_string())
}

fn get_log_dir() -> String {
    std::env::var("LOG_DIR").unwrap_or_else(|_| {
        if std::path::Path::new("/var/log").exists() {
            "/var/log/rust-local-rag".to_string()
        } else {
            "./logs".to_string()
        }
    })
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenv::dotenv().ok();

    // Check for development mode or force console logs
    let is_dev = std::env::var("DEV").is_ok()
        || std::env::var("DEVELOPMENT").is_ok()
        || std::env::var("CONSOLE_LOGS").is_ok();

    let log_level = std::env::var("LOG_LEVEL").unwrap_or_else(|_| "info".to_string());
    let filter = EnvFilter::new(&log_level);

    if is_dev {
        // Console logging for development
        tracing_subscriber::registry()
            .with(tracing_subscriber::fmt::layer())
            .with(filter)
            .init();
        tracing::info!("Starting in DEVELOPMENT mode (console logs only)");
    } else {
        // File logging for production
        let log_dir = get_log_dir();
        std::fs::create_dir_all(&log_dir)?;

        let file_appender = tracing_appender::rolling::daily(&log_dir, "rust-local-rag.log");
        let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);

        tracing_subscriber::registry()
            .with(
                tracing_subscriber::fmt::layer()
                    .with_writer(non_blocking)
                    .json(),
            )
            .with(filter)
            .init();
        tracing::info!("Starting in PRODUCTION mode (logs to {})", log_dir);
    }

    let data_dir = get_data_dir();
    let documents_dir = get_documents_dir();

    tracing::info!("Data directory: {}", data_dir);
    tracing::info!("Documents directory: {}", documents_dir);

    std::fs::create_dir_all(&data_dir)?;
    std::fs::create_dir_all(&documents_dir)?;

    let config = Config::from_env()?;
    config.log_active();

    // Initialize Job Manager (SQLite)
    let db_path = format!("sqlite:{data_dir}/jobs.db");
    let job_manager = Arc::new(JobManager::new(&db_path).await?);

    // Initialize RagEngine (Chunks + Index)
    let rag_engine = Arc::new(RwLock::new(RagEngine::new(&data_dir, &config).await?));

    // Create channel for job requests
    let (job_tx, job_rx) = mpsc::channel(100);

    // Start Worker Supervisor
    let supervisor = WorkerSupervisor::new(job_manager.clone(), rag_engine.clone(), job_rx);
    tokio::spawn(async move {
        supervisor.run().await;
    });

    // Start MCP Server
    let bind_addr: std::net::SocketAddr = std::env::var("MCP_HTTP_BIND")
        .unwrap_or_else(|_| "127.0.0.1:8140".to_string())
        .parse()?;

    rust_local_rag::guardrails::check_mcp_http_bind(bind_addr)?;
    let listener = tokio::net::TcpListener::bind(bind_addr).await?;

    rust_local_rag::mcp_server::start_mcp_server(
        rag_engine,
        job_manager,
        job_tx,
        documents_dir,
        listener,
    )
    .await?;

    Ok(())
}
