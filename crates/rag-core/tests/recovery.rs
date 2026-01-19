use rag_core::{RagEngine, error::EmbeddingError, traits::EmbeddingBackend};
use tempfile::TempDir;

struct MockBackend;

impl EmbeddingBackend for MockBackend {
    fn model_id(&self) -> &str {
        "mock-embed"
    }

    async fn embed(&self, _text: &str) -> std::result::Result<Vec<f32>, EmbeddingError> {
        Ok(vec![0.0])
    }

    async fn embed_batch(
        &self,
        texts: &[String],
    ) -> std::result::Result<Vec<Vec<f32>>, EmbeddingError> {
        Ok(vec![vec![0.0]; texts.len()])
    }

    fn dimension(&self) -> usize {
        1
    }
}

#[cfg(feature = "persistence")]
#[tokio::test]
async fn test_load_corrupted_index_triggers_reindex() {
    // 1. Setup temp dir
    let temp_dir = TempDir::new().unwrap();
    let data_dir = temp_dir.path();

    // 2. Create a corrupted index file
    // The engine uses "chunks_{sanitized_model}.json"
    // Our mock model ID is "mock-embed", so "chunks_mock-embed.json"
    let index_path = data_dir.join("chunks_mock-embed.json");
    std::fs::write(&index_path, "{ corrupted_json: true, ").unwrap();

    // 3. Initialize engine and attempt load
    let mut engine = RagEngine::new(MockBackend);

    let result = engine.load_from_dir(data_dir);

    // 4. Verification
    assert!(
        result.is_ok(),
        "Loading corrupted index should not error out"
    );

    // Should default to empty state
    assert_eq!(
        engine.chunk_count(),
        0,
        "Engine should be empty after load failure"
    );

    // Should signal reindex required
    assert!(
        engine.needs_reindex(),
        "Engine should flag needs_reindex on corruption"
    );
}
