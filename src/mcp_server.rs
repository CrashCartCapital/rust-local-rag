#![allow(dead_code)]
use anyhow::Result;
use rmcp::{
    ErrorData as McpError, ServerHandler, model::*, schemars, tool, tool_router, tool_handler,
};
use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use std::net::SocketAddr;

use std::sync::Arc;
use tokio::sync::RwLock;

use crate::job_manager::JobManager;
use crate::rag_engine::RagEngine;
use crate::worker::JobRequest;
use tokio::sync::mpsc;

#[derive(Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct SearchRequest {
    #[schemars(description = "The search query")]
    pub query: String,
    #[schemars(description = "Number of results to return (default: 5)")]
    pub top_k: Option<usize>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct GetJobStatusRequest {
    #[schemars(description = "Job ID to query")]
    pub job_id: String,
}

#[derive(Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct CalibrateRerankerRequest {
    #[schemars(description = "Sample query to use for calibration")]
    pub query: String,
    #[schemars(description = "Number of samples to test (default: 20)")]
    pub sample_size: Option<usize>,
}

// Empty param structs for tools with no parameters
#[derive(Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct EmptyParams {}

#[derive(Clone)]
pub struct RagMcpServer {
    tool_router: ToolRouter<Self>,
    rag_state: Arc<RwLock<RagEngine>>,
    job_manager: Arc<JobManager>,
    job_tx: mpsc::Sender<JobRequest>,
    documents_dir: String,
}

#[tool_router]
impl RagMcpServer {
    pub fn new(
        rag_state: Arc<RwLock<RagEngine>>,
        job_manager: Arc<JobManager>,
        job_tx: mpsc::Sender<JobRequest>,
        documents_dir: String,
    ) -> Self {
        Self {
            tool_router: Self::tool_router(),
            rag_state,
            job_manager,
            job_tx,
            documents_dir,
        }
    }

    #[tool(description = "Search through uploaded documents using semantic similarity")]
    async fn search_documents(
        &self,
        Parameters(params): Parameters<SearchRequest>,
    ) -> Result<CallToolResult, McpError> {
        let top_k = params.top_k.unwrap_or(5);
        let query = params.query;
        let engine = self.rag_state.read().await;

        match engine.search(&query, top_k).await {
            Ok(results) => {
                let formatted_results = if results.is_empty() {
                    "No results found.".to_string()
                } else {
                    results
                        .iter()
                        .enumerate()
                        .map(|(i, result)| {
                            let provenance = if result.page_number > 0 {
                                format!("{} (page {})", result.document, result.page_number)
                            } else {
                                result.document.clone()
                            };
                            let section = result
                                .section
                                .as_ref()
                                .map(|s| format!("Section: {s}\n"))
                                .unwrap_or_default();
                            format!(
                                "**Result {}** (Relevance: {:.3}) [{}] (Chunk: {} / idx {})\n{}{}\n",
                                i + 1,
                                result.score,
                                provenance,
                                result.chunk_id,
                                result.chunk_index,
                                section,
                                result.text
                            )
                        })
                        .collect::<Vec<_>>()
                        .join("\n---\n\n")
                };

                Ok(CallToolResult::success(vec![Content::text(format!(
                    "Found {} results for '{}':\n\n{}",
                    results.len(),
                    query,
                    formatted_results
                ))]))
            }
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!("Search error: {e}"))])),
        }
    }

    #[tool(description = "List all uploaded documents")]
    async fn list_documents(&self) -> Result<CallToolResult, McpError> {
        let engine = self.rag_state.read().await;
        let documents = engine.list_documents();

        let response = if documents.is_empty() {
            "No documents uploaded yet.".to_string()
        } else {
            format!(
                "Uploaded documents ({}):\n{}",
                documents.len(),
                documents
                    .iter()
                    .enumerate()
                    .map(|(i, doc)| format!("{}. {}", i + 1, doc))
                    .collect::<Vec<_>>()
                    .join("\n")
            )
        };

        Ok(CallToolResult::success(vec![Content::text(response)]))
    }

    #[tool(description = "Get RAG system statistics")]
    async fn get_stats(&self) -> Result<CallToolResult, McpError> {
        let engine = self.rag_state.read().await;
        let stats = engine.get_stats();

        let stats_text = serde_json::to_string_pretty(&stats)
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;

        Ok(CallToolResult::success(vec![Content::text(format!(
            "RAG System Stats:\n{stats_text}"
        ))]))
    }

    #[tool(description = "Start a background reindexing job and return immediately with job ID")]
    async fn start_reindex(&self) -> Result<CallToolResult, McpError> {
        // Atomically create job if no active job exists (prevents race conditions)
        let job = match self
            .job_manager
            .create_reindex_job_if_not_active(Some(self.documents_dir.clone()), 0)
            .await
            .map_err(|e| McpError::internal_error(e.to_string(), None))?
        {
            Some(job) => job,
            None => {
                // Active job already exists
                return Ok(CallToolResult::error(vec![Content::text(
                    "A reindex job is already in progress. Please wait for it to complete or check its status with get_job_status."
                        .to_string(),
                )]));
            }
        };

        // Send job request to worker supervisor
        self.job_tx
            .send(JobRequest::StartReindex {
                job_id: job.job_id.clone(),
                documents_dir: self.documents_dir.clone(),
            })
            .await
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;

        let response = serde_json::json!({
            "job_id": job.job_id,
            "status": "pending",
            "documents_dir": self.documents_dir,
            "message": "Reindexing job started in background. Use get_job_status to check progress."
        });

        let response_text = serde_json::to_string_pretty(&response)
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;

        Ok(CallToolResult::success(vec![Content::text(format!(
            "Reindexing started:\n{response_text}"
        ))]))
    }

    #[tool(description = "Get the status of a job (reindexing, etc.)")]
    async fn get_job_status(
        &self,
        Parameters(params): Parameters<GetJobStatusRequest>,
    ) -> Result<CallToolResult, McpError> {
        let job_id = params.job_id;
        let job = self
            .job_manager
            .get_job(&job_id)
            .await
            .map_err(|e| McpError::internal_error(e.to_string(), None))?
            .ok_or_else(|| McpError::resource_not_found(format!("Job {job_id} not found"), None))?;

        let response = serde_json::json!({
            "job_id": job.job_id,
            "status": job.status,
            "job_type": job.job_type,
            "progress": job.progress,
            "total": job.total,
            "error": job.error,
            "started_at": job.started_at,
            "updated_at": job.updated_at
        });

        let response_text = serde_json::to_string_pretty(&response)
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;

        Ok(CallToolResult::success(vec![Content::text(format!(
            "Job Status:\n{response_text}"
        ))]))
    }

    #[tool(description = "Calibrate reranker timeout by measuring actual LLM latencies and computing p99 statistics")]
    async fn calibrate_reranker(
        &self,
        Parameters(params): Parameters<CalibrateRerankerRequest>,
    ) -> Result<CallToolResult, McpError> {
        let sample_size = params.sample_size.unwrap_or(100);
        let query = params.query;

        // Get engine and check if reranker is available
        let engine = self.rag_state.read().await;

        if !engine.has_reranker() {
            return Ok(CallToolResult::error(vec![Content::text(
                "Reranker is not enabled. Set OLLAMA_RERANK_MODEL environment variable to enable reranking.".to_string()
            )]));
        }

        // Get candidates for calibration (use more than sample_size to have enough)
        let candidates_result = engine.get_embedding_candidates(&query, sample_size * 2).await;

        match candidates_result {
            Ok(candidates) if candidates.is_empty() => {
                Ok(CallToolResult::error(vec![Content::text(
                    "No candidates found for calibration. Index some documents first using start_reindex.".to_string()
                )]))
            }
            Ok(candidates) => {
                // Run calibration
                let reranker = engine.get_reranker().unwrap();
                match reranker.calibrate_timeout(&query, &candidates, sample_size).await {
                    Ok(stats) => {
                        // Apply 1.2x safety margin as recommended by ensemble
                        // Enforce minimum 10-second timeout as baseline buffer
                        let safety_margin = 1.2;
                        let recommended_timeout_ms = ((stats.p99_ms * safety_margin).ceil() as u64).max(10_000);

                        let response = serde_json::json!({
                            "calibration_stats": {
                                "mean_ms": stats.mean_ms,
                                "median_ms": stats.median_ms,
                                "p95_ms": stats.p95_ms,
                                "p99_ms": stats.p99_ms,
                                "max_ms": stats.max_ms,
                                "sample_size": stats.sample_size
                            },
                            "safety_margin": safety_margin,
                            "recommended_timeout_ms": recommended_timeout_ms,
                            "current_timeout_ms": 10000,
                            "query": query
                        });

                        let response_text = serde_json::to_string_pretty(&response)
                            .map_err(|e| McpError::internal_error(e.to_string(), None))?;

                        Ok(CallToolResult::success(vec![Content::text(format!(
                            "Reranker Calibration Results:\n{response_text}\n\n\
                            Recommendation: Based on p99 latency ({:.0}ms) with {}x safety margin \
                            (minimum 10 seconds baseline), set timeout to {} seconds (currently 10 seconds).\n\
                            Note: For reliable p99 estimation, use sample_size ≥ 50-100.",
                            stats.p99_ms,
                            safety_margin,
                            recommended_timeout_ms / 1000
                        ))]))
                    }
                    Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                        "Calibration failed: {e}"
                    ))])),
                }
            }
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "Failed to get candidates for calibration: {e}"
            ))])),
        }
    }
}

#[tool_handler]
impl ServerHandler for RagMcpServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            protocol_version: ProtocolVersion::V_2025_03_26,
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            server_info: Implementation {
                name: "rust-rag-server".to_string(),
                version: "0.1.0".to_string(),
                title: None,
                icons: None,
                website_url: None,
            },
            instructions: Some(
                "A Rust-based RAG server for document search and analysis.".to_string(),
            ),
        }
    }
}

/// Liveness probe handler - always returns 200 OK if process is alive
async fn healthz() -> axum::http::StatusCode {
    axum::http::StatusCode::OK
}

/// Readiness probe handler - returns 200 when server is ready to serve requests
async fn readyz(
    axum::extract::State(rag_state): axum::extract::State<Arc<RwLock<RagEngine>>>,
) -> axum::http::StatusCode {
    // Simple readiness check: can we acquire a read lock on the engine?
    // This confirms the engine is initialized and not stuck
    match tokio::time::timeout(
        std::time::Duration::from_millis(100),
        rag_state.read()
    ).await {
        Ok(_guard) => axum::http::StatusCode::OK,
        Err(_) => axum::http::StatusCode::SERVICE_UNAVAILABLE,
    }
}

pub async fn start_mcp_server(
    rag_state: Arc<RwLock<RagEngine>>,
    job_manager: Arc<JobManager>,
    job_tx: mpsc::Sender<JobRequest>,
    documents_dir: String,
) -> Result<()> {
    use rmcp::transport::streamable_http_server::{StreamableHttpService, session::local::LocalSessionManager};

    let bind: SocketAddr = std::env::var("MCP_HTTP_BIND")
        .unwrap_or_else(|_| "127.0.0.1:3046".to_string())
        .parse()?;

    let endpoint_path = std::env::var("MCP_HTTP_ENDPOINT")
        .unwrap_or_else(|_| "/mcp".to_string());

    tracing::info!("Starting MCP Streamable HTTP server on http://{}{}", bind, endpoint_path);
    tracing::info!("Health endpoints: /healthz (liveness), /readyz (readiness)");

    let service = StreamableHttpService::new(
        {
            let rag_state = rag_state.clone();
            let job_manager = job_manager.clone();
            let job_tx = job_tx.clone();
            let documents_dir = documents_dir.clone();
            move || Ok(RagMcpServer::new(rag_state.clone(), job_manager.clone(), job_tx.clone(), documents_dir.clone()))
        },
        LocalSessionManager::default().into(),
        Default::default(), // StreamableHttpServerConfig
    );

    let router = axum::Router::new()
        .route("/healthz", axum::routing::get(healthz))
        .route("/readyz", axum::routing::get(readyz))
        .route(&endpoint_path, axum::routing::any_service(service))
        .with_state(rag_state.clone());

    let tcp_listener = tokio::net::TcpListener::bind(bind).await?;

    axum::serve(tcp_listener, router)
        .with_graceful_shutdown(async {
            tokio::signal::ctrl_c().await.ok();
        })
        .await?;

    Ok(())
}
