use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Clone)]
pub struct ApiClient {
    client: reqwest::Client,
    base_url: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Stats {
    pub documents: usize,
    pub chunks: usize,
    pub status: String,
    #[serde(default)]
    pub embedding_model: Option<String>,
    #[serde(default)]
    pub reranker_model: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct SearchRequest {
    query: String,
    top_k: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SearchResponse {
    pub results: Vec<SearchResult>,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct SearchResult {
    pub document: String,
    pub text: String,
    pub score: f32,
    #[serde(default)]
    pub page_number: u32,
    #[serde(default)]
    pub chunk_id: String,
    #[serde(default)]
    pub section: Option<String>,
}

impl ApiClient {
    pub fn new(base_url: String) -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .build()
            .unwrap_or_default();

        Self { client, base_url }
    }

    pub async fn health_check(&self) -> Result<bool> {
        let url = format!("{}/healthz", self.base_url);
        match self.client.get(&url).send().await {
            Ok(resp) => Ok(resp.status().is_success()),
            Err(_) => Ok(false),
        }
    }

    pub async fn get_stats(&self) -> Result<Stats> {
        let url = format!("{}/stats", self.base_url);
        let resp = self.client.get(&url).send().await?;
        let stats = resp.json::<Stats>().await?;
        Ok(stats)
    }

    pub async fn search(&self, query: &str, top_k: usize) -> Result<Vec<SearchResult>> {
        let url = format!("{}/search", self.base_url);
        let request = SearchRequest {
            query: query.to_string(),
            top_k,
        };

        let resp = self.client.post(&url).json(&request).send().await?;
        let search_resp = resp.json::<SearchResponse>().await?;
        Ok(search_resp.results)
    }
}
