use crate::job_manager::JobManager;
use crate::rag_engine::RagEngine;
use crate::worker::JobRequest;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{Instrument, instrument};
use uuid::Uuid;

use super::responses::{JobStatusResponse, ReindexResponse};
use super::MAX_TOP_K;
use tokio::sync::mpsc;

/// Shared application state for HTTP handlers
#[derive(Clone)]
pub(crate) struct AppState {
    pub(crate) rag_state: Arc<RwLock<RagEngine>>,
    pub(crate) job_manager: Arc<JobManager>,
    pub(crate) job_tx: mpsc::Sender<JobRequest>,
    pub(crate) documents_dir: String,
}

/// Liveness probe handler - always returns 200 OK if process is alive
#[instrument]
async fn healthz() -> axum::http::StatusCode {
    axum::http::StatusCode::OK
}

/// Readiness probe handler - returns 200 when server is ready to serve requests
#[instrument(skip(app_state))]
async fn readyz(
    axum::extract::State(app_state): axum::extract::State<AppState>,
) -> axum::http::StatusCode {
    match tokio::time::timeout(
        std::time::Duration::from_millis(100),
        app_state.rag_state.read(),
    )
    .await
    {
        Ok(_guard) => axum::http::StatusCode::OK,
        Err(_) => axum::http::StatusCode::SERVICE_UNAVAILABLE,
    }
}

#[derive(Debug, serde::Deserialize)]
struct HttpSearchRequest {
    query: String,
    #[serde(default = "default_top_k")]
    top_k: usize,
    #[serde(default = "default_diversity_factor")]
    diversity_factor: f32,
}

fn default_top_k() -> usize {
    5
}
fn default_diversity_factor() -> f32 {
    0.3
}

#[derive(Debug, serde::Serialize)]
struct HttpSearchResponse {
    results: Vec<crate::rag_engine::SearchResult>,
}

fn api_error(
    code: axum::http::StatusCode,
    message: impl Into<String>,
) -> (axum::http::StatusCode, axum::Json<serde_json::Value>) {
    (
        code,
        axum::Json(serde_json::json!({ "error": message.into() })),
    )
}

#[instrument(skip(app_state), fields(query = %request.query, top_k = %request.top_k, diversity = %request.diversity_factor))]
async fn http_search(
    axum::extract::State(app_state): axum::extract::State<AppState>,
    axum::extract::Json(request): axum::extract::Json<HttpSearchRequest>,
) -> Result<axum::Json<HttpSearchResponse>, (axum::http::StatusCode, axum::Json<serde_json::Value>)>
{
    let start = std::time::Instant::now();
    let top_k = request.top_k.min(MAX_TOP_K);
    let diversity_factor = request.diversity_factor.clamp(0.0, 1.0);
    let engine = app_state.rag_state.read().await;
    match engine
        .search_with_diversity(&request.query, top_k, diversity_factor, None)
        .await
    {
        Ok(results) => {
            let duration = start.elapsed();
            tracing::info!(
                "HTTP Search completed in {:?} with {} results",
                duration,
                results.len()
            );
            Ok(axum::Json(HttpSearchResponse { results }))
        }
        Err(e) => {
            tracing::error!("Search error: {}", e);
            Err(api_error(
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                format!("Search error: {e}"),
            ))
        }
    }
}

#[instrument(skip(app_state))]
async fn http_stats(
    axum::extract::State(app_state): axum::extract::State<AppState>,
) -> axum::Json<serde_json::Value> {
    let engine = app_state.rag_state.read().await;
    axum::Json(engine.get_stats())
}

#[instrument(skip(app_state))]
async fn http_start_reindex(
    axum::extract::State(app_state): axum::extract::State<AppState>,
) -> Result<axum::Json<ReindexResponse>, (axum::http::StatusCode, axum::Json<serde_json::Value>)>
{
    tracing::info!("HTTP request to start reindex job");
    let job = match app_state
        .job_manager
        .create_reindex_job_if_not_active(Some(app_state.documents_dir.clone()), 0)
        .await
    {
        Ok(Some(job)) => job,
        Ok(None) => {
            tracing::warn!("Reindex job already in progress");
            return Err(api_error(
                axum::http::StatusCode::CONFLICT,
                "A reindex job is already in progress",
            ));
        }
        Err(e) => {
            tracing::error!("Failed to create reindex job: {e}");
            return Err(api_error(
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to create job: {e}"),
            ));
        }
    };

    if let Err(e) = app_state
        .job_tx
        .send(JobRequest::StartReindex {
            job_id: job.job_id.clone(),
            documents_dir: app_state.documents_dir.clone(),
        })
        .await
    {
        tracing::error!("Failed to send job request: {e}");
        return Err(api_error(
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to start job: {e}"),
        ));
    }

    tracing::info!(job_id = %job.job_id, "Reindex job started successfully");

    Ok(axum::Json(ReindexResponse {
        job_id: job.job_id,
        status: job.status.as_str().to_string(),
        documents_dir: app_state.documents_dir.clone(),
        message: "Reindexing started".to_string(),
    }))
}

#[instrument(skip(app_state), fields(job_id = %job_id))]
async fn http_get_job_status(
    axum::extract::State(app_state): axum::extract::State<AppState>,
    axum::extract::Path(job_id): axum::extract::Path<String>,
) -> Result<axum::Json<JobStatusResponse>, (axum::http::StatusCode, axum::Json<serde_json::Value>)>
{
    match app_state.job_manager.get_job(&job_id).await {
        Ok(Some(job)) => Ok(axum::Json(JobStatusResponse::from(&job))),
        Ok(None) => Err(api_error(
            axum::http::StatusCode::NOT_FOUND,
            format!("Job {job_id} not found"),
        )),
        Err(e) => {
            tracing::error!("Failed to get job status: {e}");
            Err(api_error(
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to get job: {e}"),
            ))
        }
    }
}

#[instrument(skip(app_state))]
async fn http_get_active_job(
    axum::extract::State(app_state): axum::extract::State<AppState>,
) -> Result<
    axum::Json<Option<JobStatusResponse>>,
    (axum::http::StatusCode, axum::Json<serde_json::Value>),
> {
    match app_state.job_manager.find_active_reindex_job().await {
        Ok(Some(job)) => Ok(axum::Json(Some(JobStatusResponse::from(&job)))),
        Ok(None) => Ok(axum::Json(None)),
        Err(e) => {
            tracing::error!("Failed to get active job: {e}");
            Err(api_error(
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to get active job: {e}"),
            ))
        }
    }
}

async fn trace_request(
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    let request_id = Uuid::new_v4().to_string();
    let method = req.method().clone();
    let uri = req.uri().clone();

    let span = tracing::info_span!("http_request", %request_id, %method, %uri);

    async move {
        tracing::info!("Request started");
        let start = std::time::Instant::now();
        let response = next.run(req).await;
        let duration = start.elapsed();
        let status = response.status();
        tracing::info!(%status, ?duration, "Request completed");
        response
    }
    .instrument(span)
    .await
}

pub(crate) fn create_api_router() -> axum::Router<AppState> {
    axum::Router::new()
        .route("/healthz", axum::routing::get(healthz))
        .route("/health", axum::routing::get(healthz))
        .route("/readyz", axum::routing::get(readyz))
        .route("/search", axum::routing::post(http_search))
        .route("/stats", axum::routing::get(http_stats))
        .route("/reindex", axum::routing::post(http_start_reindex))
        .route("/jobs/active", axum::routing::get(http_get_active_job))
        .route("/jobs/{job_id}", axum::routing::get(http_get_job_status))
        .layer(axum::middleware::from_fn(trace_request))
}

