use anyhow::Result;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::job_manager::JobManager;
use crate::rag_engine::RagEngine;
use crate::worker::JobRequest;
use tokio::sync::mpsc;

mod formatting;
mod http;
mod models;
mod responses;
mod tools;

pub use models::{CalibrateRerankerRequest, GetJobStatusRequest, SearchRequest};

/// Maximum allowed top_k to prevent DoS via memory exhaustion
pub(crate) const MAX_TOP_K: usize = 100;

pub async fn start_mcp_server(
    rag_state: Arc<RwLock<RagEngine>>,
    job_manager: Arc<JobManager>,
    job_tx: mpsc::Sender<JobRequest>,
    documents_dir: String,
    tcp_listener: tokio::net::TcpListener,
) -> Result<()> {
    use rmcp::transport::streamable_http_server::{
        StreamableHttpService, session::local::LocalSessionManager,
    };

    let endpoint_path = std::env::var("MCP_HTTP_ENDPOINT").unwrap_or_else(|_| "/mcp".to_string());
    let local_addr = tcp_listener.local_addr()?;

    tracing::info!(
        "Starting MCP Streamable HTTP server on http://{}{}",
        local_addr,
        endpoint_path
    );
    tracing::info!("Health endpoints: /healthz (liveness), /readyz (readiness)");

    let service = StreamableHttpService::new(
        {
            let rag_state = rag_state.clone();
            let job_manager = job_manager.clone();
            let job_tx = job_tx.clone();
            let documents_dir = documents_dir.clone();
            move || {
                Ok(tools::RagMcpServer::new(
                    rag_state.clone(),
                    job_manager.clone(),
                    job_tx.clone(),
                    documents_dir.clone(),
                ))
            }
        },
        LocalSessionManager::default().into(),
        Default::default(),
    );

    let app_state = http::AppState {
        rag_state: rag_state.clone(),
        job_manager: job_manager.clone(),
        job_tx: job_tx.clone(),
        documents_dir: documents_dir.clone(),
    };

    let router = http::create_api_router()
        .route(&endpoint_path, axum::routing::any_service(service))
        .with_state(app_state);

    tracing::info!(
        "HTTP evaluation endpoints: POST /search, GET /stats, POST /reindex, GET /jobs/active, GET /jobs/:id"
    );

    axum::serve(tcp_listener, router)
        .with_graceful_shutdown(async {
            tokio::signal::ctrl_c().await.ok();
        })
        .await?;

    Ok(())
}

#[cfg(test)]
mod tests;
