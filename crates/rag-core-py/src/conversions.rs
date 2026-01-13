//! Type conversions between rag-core and Python types.
//!
//! This module provides From/Into implementations for converting
//! between rag-core types and their Python wrapper equivalents.

use crate::types::{PyQuerySpec, PyScoreBreakdown, PySearchResult};

/// Convert from rag_core::SearchResult to PySearchResult.
///
/// Note: rag-core uses page_number=0 as the default/unknown value.
/// We convert this to None for clearer Python semantics.
impl From<rag_core::SearchResult> for PySearchResult {
    fn from(result: rag_core::SearchResult) -> Self {
        Self {
            id: result.chunk_id,
            text: result.text,
            document: result.document,
            chunk_index: result.chunk_index,
            // rag-core uses 0 as "unknown page", map to None for Python
            page_number: if result.page_number == 0 {
                None
            } else {
                Some(result.page_number)
            },
            section: result.section,
            score: result.score,
            scores: PyScoreBreakdown {
                embedding: result.embedding_score,
                lexical: result.lexical_score,
                initial: result.initial_score,
                reranker: result.reranker_score,
                total: result.score,
            },
        }
    }
}

impl PyQuerySpec {
    /// Convert to rag_core::QuerySpec with the given query string.
    ///
    /// This creates a new QuerySpec with the weights and top_k from
    /// this PyQuerySpec, plus defaults for other fields.
    pub fn to_query_spec(&self, query: &str) -> rag_core::QuerySpec {
        let weights = rag_core::SearchWeights {
            embedding: self.embedding_weight,
            lexical: self.lexical_weight,
            reranker: self.reranker_weight,
            initial: self.initial_weight,
        };

        rag_core::QuerySpec::new(query)
            .with_top_k(self.top_k)
            .with_weights(weights)
            .with_diversity(self.diversity_factor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_result_conversion() {
        let rust_result = rag_core::SearchResult {
            text: "test text".to_string(),
            score: 0.95,
            document: "test.pdf".to_string(),
            chunk_id: "chunk-123".to_string(),
            chunk_index: 0,
            page_number: 1,
            section: Some("Introduction".to_string()),
            embedding_score: Some(0.9),
            lexical_score: Some(0.8),
            initial_score: Some(0.85),
            reranker_score: Some(0.95),
            yes_logprob: None,
            no_logprob: None,
        };

        let py_result: PySearchResult = rust_result.into();

        assert_eq!(py_result.id, "chunk-123");
        assert_eq!(py_result.text, "test text");
        assert_eq!(py_result.document, "test.pdf");
        assert_eq!(py_result.chunk_index, 0);
        assert_eq!(py_result.page_number, Some(1));
        assert_eq!(py_result.section, Some("Introduction".to_string()));
        assert!((py_result.score - 0.95).abs() < 0.0001);
        assert_eq!(py_result.scores.embedding, Some(0.9));
        assert_eq!(py_result.scores.lexical, Some(0.8));
        assert_eq!(py_result.scores.reranker, Some(0.95));
    }

    #[test]
    fn test_page_number_zero_becomes_none() {
        // rag-core uses page_number=0 as "unknown", should convert to None
        let rust_result = rag_core::SearchResult {
            text: "test".to_string(),
            score: 0.5,
            document: "doc.pdf".to_string(),
            chunk_id: "chunk-0".to_string(),
            chunk_index: 0,
            page_number: 0, // Unknown page
            section: None,
            embedding_score: None,
            lexical_score: None,
            initial_score: None,
            reranker_score: None,
            yes_logprob: None,
            no_logprob: None,
        };

        let py_result: PySearchResult = rust_result.into();
        assert_eq!(py_result.page_number, None); // Should be None, not Some(0)
    }

    #[test]
    fn test_query_spec_conversion() {
        let py_spec = PyQuerySpec {
            top_k: 5,
            embedding_weight: 0.6,
            lexical_weight: 0.4,
            reranker_weight: 0.8,
            initial_weight: 0.2,
            diversity_factor: 0.3,
        };

        let rust_spec = py_spec.to_query_spec("test query");

        assert_eq!(rust_spec.query, "test query");
        assert_eq!(rust_spec.top_k, 5);
        assert!((rust_spec.diversity_factor - 0.3).abs() < 0.0001);

        // Check weights
        let weights = rust_spec.weights.unwrap();
        assert!((weights.embedding - 0.6).abs() < 0.0001);
        assert!((weights.lexical - 0.4).abs() < 0.0001);
    }
}
