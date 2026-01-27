use anyhow::Result;
use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::{ErrorData as McpError, ServerHandler, model::*, tool, tool_handler, tool_router};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::instrument;

use crate::job_manager::JobManager;
use crate::job_payload::ReindexJobPayload;
use crate::rag_engine::RagEngine;
use crate::worker::JobRequest;
use tokio::sync::mpsc;

use super::formatting::format_search_results;
use super::models::{CalibrateRerankerRequest, GetJobStatusRequest, SearchRequest};
use super::responses::{JobStatusResponse, ReindexResponse};
use super::validation::validate_search_request;

#[derive(Clone)]
pub(crate) struct RagMcpServer {
    tool_router: ToolRouter<Self>,
    rag_state: Arc<RwLock<RagEngine>>,
    job_manager: Arc<JobManager>,
    job_tx: mpsc::Sender<JobRequest>,
    documents_dir: String,
}

#[tool_router]
impl RagMcpServer {
    pub(crate) fn new(
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

    #[tool(
        description = "Search through uploaded documents using semantic similarity with optional MMR diversification and per-query weight customization"
    )]
    #[instrument(skip(self), fields(query = %params.query, top_k = ?params.top_k, diversity = ?params.diversity_factor))]
    async fn search_documents(
        &self,
        Parameters(params): Parameters<SearchRequest>,
    ) -> Result<CallToolResult, McpError> {
        let start = std::time::Instant::now();
        let validated = validate_search_request(params);
        let engine = self.rag_state.read().await;

        match engine
            .search_with_diversity(
                &validated.query,
                validated.top_k,
                validated.diversity_factor,
                validated.weights.as_ref(),
            )
            .await
        {
            Ok(results) => {
                let duration = start.elapsed();
                let count = results.len();
                tracing::info!("Search completed in {:?} with {} results", duration, count);
                let formatted_results = format_search_results(&results, &validated.query);

                Ok(CallToolResult::success(vec![Content::text(format!(
                    "Found {count} results for '{}':\n\n{formatted_results}",
                    validated.query
                ))]))
            }
            Err(e) => {
                tracing::error!("Search failed: {}", e);
                Ok(CallToolResult::error(vec![Content::text(format!(
                    "Search error: {e}"
                ))]))
            }
        }
    }

    #[tool(description = "List all uploaded documents")]
    #[instrument(skip(self))]
    async fn list_documents(&self) -> Result<CallToolResult, McpError> {
        let engine = self.rag_state.read().await;
        let documents = engine.list_documents();

        let response = if documents.is_empty() {
            "No documents uploaded yet. Please add PDF files to your documents directory and run `start_reindex`."
                .to_string()
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
    #[instrument(skip(self))]
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
    #[instrument(skip(self))]
    async fn start_reindex(&self) -> Result<CallToolResult, McpError> {
        tracing::info!("Starting reindex job");
        let (model_id, embedding_dim) = {
            let engine = self.rag_state.read().await;
            (
                engine.embedding_model().to_string(),
                engine.backend_embedding_dim(),
            )
        };
        let payload = serde_json::to_string(&ReindexJobPayload {
            documents_dir: self.documents_dir.clone(),
            model_id,
            embedding_dim,
        })
        .map_err(|e| McpError::internal_error(e.to_string(), None))?;

        // Atomically create job if no active job exists (prevents race conditions)
        let job = match self
            .job_manager
            .create_reindex_job_if_not_active(Some(payload), 0)
            .await
            .map_err(|e| McpError::internal_error(e.to_string(), None))?
        {
            Some(job) => job,
            None => {
                tracing::warn!("Reindex job already in progress");
                return Ok(CallToolResult::error(vec![Content::text(
                    "A reindex job is already in progress. Please wait for it to complete or check its status with get_job_status."
                        .to_string(),
                )]));
            }
        };

        self.job_tx
            .send(JobRequest::StartReindex {
                job_id: job.job_id.clone(),
                documents_dir: self.documents_dir.clone(),
            })
            .await
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;

        tracing::info!(job_id = %job.job_id, "Reindex job started successfully");

        let response = ReindexResponse {
            job_id: job.job_id,
            status: job.status.as_str().to_string(),
            documents_dir: self.documents_dir.clone(),
            message: "Reindexing job started in background. Use get_job_status to check progress."
                .to_string(),
        };

        let response_text = serde_json::to_string_pretty(&response)
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;

        Ok(CallToolResult::success(vec![Content::text(format!(
            "Reindexing started:\n{response_text}"
        ))]))
    }

    #[tool(description = "Get the status of a job (reindexing, etc.)")]
    #[instrument(skip(self), fields(job_id = %params.job_id))]
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

        let response = JobStatusResponse::from(&job);
        let response_text = serde_json::to_string_pretty(&response)
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;

        Ok(CallToolResult::success(vec![Content::text(format!(
            "Job Status:\n{response_text}"
        ))]))
    }

    #[tool(
        description = "Calibrate reranker timeout by measuring actual LLM latencies and computing p99 statistics"
    )]
    #[instrument(skip(self), fields(query = %params.query, sample_size = ?params.sample_size))]
    async fn calibrate_reranker(
        &self,
        Parameters(params): Parameters<CalibrateRerankerRequest>,
    ) -> Result<CallToolResult, McpError> {
        let sample_size = params.sample_size.unwrap_or(100);
        let query = params.query;

        let engine = self.rag_state.read().await;

        if !engine.has_reranker() {
            return Ok(CallToolResult::error(vec![Content::text(
                "Reranker is not enabled. Set OLLAMA_RERANK_MODEL environment variable to enable reranking."
                    .to_string(),
            )]));
        }

        let candidates_result = engine
            .get_embedding_candidates(&query, sample_size * 2)
            .await;

        match candidates_result {
            Ok(candidates) if candidates.is_empty() => Ok(CallToolResult::error(vec![
                Content::text(
                    "No candidates found for calibration. Index some documents first using start_reindex."
                        .to_string(),
                ),
            ])),
            Ok(candidates) => {
                let Some(reranker) = engine.get_reranker() else {
                    return Ok(CallToolResult::error(vec![Content::text(
                        "Reranker is not available. Set OLLAMA_RERANK_MODEL environment variable to enable reranking."
                            .to_string(),
                    )]));
                };

                match reranker.calibrate_timeout(&query, &candidates, sample_size).await {
                    Ok(stats) => {
                        let safety_margin = 1.2;
                        let recommended_timeout_ms =
                            ((stats.p99_ms * safety_margin).ceil() as u64).max(10_000);
                        let current_timeout_ms = reranker.timeout_duration().as_millis() as u64;

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
                            "current_timeout_ms": current_timeout_ms,
                            "query": query
                        });

                        let response_text = serde_json::to_string_pretty(&response)
                            .map_err(|e| McpError::internal_error(e.to_string(), None))?;

                        Ok(CallToolResult::success(vec![Content::text(format!(
                            "Reranker Calibration Results:\n{response_text}\n\n\
                            Recommendation: Based on p99 latency ({:.0}ms) with {}x safety margin \
                            (minimum 10 seconds baseline), set timeout to {} seconds (currently {} seconds).\n\
                            Note: For reliable p99 estimation, use sample_size ≥ 50-100.",
                            stats.p99_ms,
                            safety_margin,
                            recommended_timeout_ms / 1000,
                            current_timeout_ms / 1000
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings::EmbeddingService;
    use crate::job_manager::JobManager;
    use crate::rag_engine::RagEngine;
    use rmcp::handler::server::wrapper::Parameters;
    use serial_test::serial;
    use std::sync::Arc;
    use tokio::sync::RwLock;
    use tokio::sync::mpsc;

    #[tokio::test]
    #[serial]
    async fn test_calibrate_reranker_without_reranker_returns_error() {
        use wiremock::matchers::{body_string_contains, method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let mock_server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/api/tags"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "models": [
                    { "name": "nomic-embed-text:latest" }
                ]
            })))
            .mount(&mock_server)
            .await;

        // Startup dimension discovery canary embed
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .and(body_string_contains("quick brown fox"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "embedding": vec![0.1f32; 384]
            })))
            .mount(&mock_server)
            .await;

        unsafe {
            std::env::set_var("OLLAMA_URL", mock_server.uri());
            std::env::set_var("OLLAMA_RERANK_MODEL", "missing-rerank-model");
        }

        let temp_dir = tempfile::tempdir().unwrap();
        let embedding_service =
            EmbeddingService::new_with_config(mock_server.uri(), "nomic-embed-text".to_string())
                .await
                .expect("EmbeddingService init failed");

        let rag_engine = RagEngine::new_with_embedding_service(
            temp_dir.path().to_str().unwrap(),
            embedding_service,
            &crate::config::Config::default(),
        )
        .await
        .expect("RagEngine init failed");

        assert!(!rag_engine.has_reranker());

        let rag_state = Arc::new(RwLock::new(rag_engine));
        let job_manager = Arc::new(JobManager::new("sqlite::memory:").await.unwrap());
        let (job_tx, _rx) = mpsc::channel(1);

        let documents_dir = temp_dir.path().to_string_lossy().to_string();
        let server = RagMcpServer::new(rag_state, job_manager, job_tx, documents_dir);

        let result = server
            .calibrate_reranker(Parameters(CalibrateRerankerRequest {
                query: "test".to_string(),
                sample_size: None,
            }))
            .await;

        assert!(result.is_ok());
        let tool_result = result.unwrap();
        let value = serde_json::to_value(&tool_result).unwrap();
        assert_eq!(value.get("isError").and_then(|v| v.as_bool()), Some(true));

        unsafe {
            std::env::remove_var("OLLAMA_URL");
            std::env::remove_var("OLLAMA_RERANK_MODEL");
        }
    }
}
