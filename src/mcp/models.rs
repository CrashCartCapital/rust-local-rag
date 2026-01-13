use rmcp::schemars;

#[derive(Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
#[schemars(crate = "rmcp::schemars")]
pub struct SearchRequest {
    #[schemars(description = "The search query")]
    pub query: String,
    #[schemars(description = "Number of results to return (default: 5)")]
    pub top_k: Option<usize>,
    #[schemars(
        description = "Diversity factor for MMR reranking (0.0-1.0, default: 0.3). Higher values increase result diversity."
    )]
    pub diversity_factor: Option<f32>,
    #[schemars(
        description = "Optional per-query weight overrides for scoring. Omitted weights use cached defaults."
    )]
    pub weights: Option<crate::rag_engine::QueryWeights>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
#[schemars(crate = "rmcp::schemars")]
pub struct GetJobStatusRequest {
    #[schemars(description = "Job ID to query")]
    pub job_id: String,
}

#[derive(Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
#[schemars(crate = "rmcp::schemars")]
pub struct CalibrateRerankerRequest {
    #[schemars(description = "Sample query to use for calibration")]
    pub query: String,
    #[schemars(description = "Number of samples to test (default: 100)")]
    pub sample_size: Option<usize>,
}
