#![cfg(feature = "persistence")]

use rag_core::{EmbeddingBackend, EmbeddingError, RagEngine};
use tempfile::TempDir;

struct MockBackend;

impl EmbeddingBackend for MockBackend {
    fn model_id(&self) -> &str {
        "mock-model"
    }

    fn dimension(&self) -> usize {
        2
    }

    async fn embed(&self, _text: &str) -> Result<Vec<f32>, EmbeddingError> {
        Ok(vec![0.0, 0.0])
    }

    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        Ok(vec![vec![0.0, 0.0]; texts.len()])
    }
}

#[tokio::test]
async fn test_load_from_dir_handles_corrupted_index() {
    let temp_dir = TempDir::new().unwrap();
    let backend = MockBackend;
    let mut engine = RagEngine::new(backend);

    // Create a corrupted index file
    // The index path is determined by rag_core::persistence::index_path
    // model_id is "mock-model", so "chunks_mock-model.json"
    let index_path = rag_core::persistence::index_path(temp_dir.path(), "mock-model");

    // Ensure parent dir exists (though temp_dir exists)
    std::fs::create_dir_all(index_path.parent().unwrap()).unwrap();

    // Write invalid JSON
    std::fs::write(&index_path, "{ invalid json").unwrap();

    // Verify file exists and has content
    assert!(index_path.exists());
    let content = std::fs::read_to_string(&index_path).unwrap();
    assert_eq!(content, "{ invalid json");

    // Attempt to load
    let result = engine.load_from_dir(temp_dir.path());

    // Assertions
    assert!(
        result.is_ok(),
        "load_from_dir should not error on corruption, but handle it properly"
    );
    assert!(
        engine.needs_reindex(),
        "Engine should be marked for reindex upon corruption"
    );
    assert_eq!(
        engine.chunk_count(),
        0,
        "Engine should be empty (clean slate)"
    );
}

#[tokio::test]
async fn test_load_from_dir_handles_invalid_utf8() {
    let temp_dir = TempDir::new().unwrap();
    let backend = MockBackend;
    let mut engine = RagEngine::new(backend);

    let index_path = rag_core::persistence::index_path(temp_dir.path(), "mock-model");
    std::fs::create_dir_all(index_path.parent().unwrap()).unwrap();

    // Write invalid UTF-8 bytes (0xFF is not valid in UTF-8)
    std::fs::write(&index_path, b"\xff\xff\xff").unwrap();

    assert!(index_path.exists());

    // Attempt to load
    let result = engine.load_from_dir(temp_dir.path());

    // Assertions
    assert!(
        result.is_ok(),
        "load_from_dir should not error on invalid UTF-8, but treat as corruption"
    );
    assert!(
        engine.needs_reindex(),
        "Engine should be marked for reindex upon corruption"
    );
    assert_eq!(engine.chunk_count(), 0, "Engine should be empty");
}
