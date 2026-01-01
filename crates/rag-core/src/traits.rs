use crate::error::{EmbeddingError, RerankError};
use crate::types::{RerankedResult, RerankerCandidate};
use std::future::Future;

pub trait EmbeddingBackend: Send + Sync {
    fn model_id(&self) -> &str;

    fn embed(&self, text: &str) -> impl Future<Output = Result<Vec<f32>, EmbeddingError>> + Send;

    fn embed_batch(
        &self,
        texts: &[String],
    ) -> impl Future<Output = Result<Vec<Vec<f32>>, EmbeddingError>> + Send {
        async move {
            let mut results = Vec::with_capacity(texts.len());
            for text in texts {
                results.push(self.embed(text).await?);
            }
            Ok(results)
        }
    }

    fn dimension(&self) -> usize;
}

pub trait Rerank: Send + Sync {
    fn rerank(
        &self,
        query: &str,
        candidates: &[RerankerCandidate],
    ) -> impl Future<Output = Result<Vec<RerankedResult>, RerankError>> + Send;
}

impl Rerank for () {
    async fn rerank(
        &self,
        _query: &str,
        _candidates: &[RerankerCandidate],
    ) -> Result<Vec<RerankedResult>, RerankError> {
        Ok(Vec::new())
    }
}
