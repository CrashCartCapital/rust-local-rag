use crate::rag_engine::QueryWeights;

use super::models::SearchRequest;
use super::MAX_TOP_K;

const DEFAULT_TOP_K: usize = 5;
const DEFAULT_DIVERSITY_FACTOR: f32 = 0.3;

#[derive(Debug, Clone)]
pub(crate) struct ValidatedSearch {
    pub(crate) query: String,
    pub(crate) top_k: usize,
    pub(crate) diversity_factor: f32,
    pub(crate) weights: Option<QueryWeights>,
}

pub(crate) fn validate_search_request(request: SearchRequest) -> ValidatedSearch {
    let top_k = request.top_k.unwrap_or(DEFAULT_TOP_K).min(MAX_TOP_K);
    let diversity_factor = request
        .diversity_factor
        .unwrap_or(DEFAULT_DIVERSITY_FACTOR)
        .clamp(0.0, 1.0);

    ValidatedSearch {
        query: request.query,
        top_k,
        diversity_factor,
        weights: request.weights,
    }
}

