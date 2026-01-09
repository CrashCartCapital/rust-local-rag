use rust_local_rag::rag_engine::RagEngine;
use rust_local_rag::Config;
use serial_test::serial;
use std::io::Write;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

async fn setup_mock_ollama() -> MockServer {
    let mock_server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/tags"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "models": [{ "name": "nomic-embed-text:latest" }]
        })))
        .mount(&mock_server)
        .await;
    mock_server
}

#[tokio::test]
#[serial]
async fn test_recover_from_corrupt_index() {
    let mock_server = setup_mock_ollama().await;
    let temp_dir = tempfile::tempdir().unwrap();
    let data_dir = temp_dir.path().to_str().unwrap();

    // 1. Manually create a corrupt index file
    // The engine uses "chunks_{model}.json"
    let model_name = "nomic-embed-text";
    // We must simulate the environment variable being set so RagEngine looks for this model
    unsafe {
        std::env::set_var("OLLAMA_URL", mock_server.uri());
        std::env::set_var("OLLAMA_EMBEDDING_MODEL", model_name);
    }

    // We need to use the sanitization logic to match the filename
    // "nomic-embed-text" -> "nomic-embed-text"
    let index_filename = format!("chunks_{model_name}.json");
    let index_path = temp_dir.path().join(index_filename);

    let mut file = std::fs::File::create(&index_path).unwrap();
    // Write garbage JSON
    writeln!(file, "{{ \"version\": 2, \"chunks\": [INVALID JSON HERE...").unwrap();

    // 2. Initialize RagEngine
    // This should log a warning but NOT panic
    let engine = RagEngine::new(data_dir, &Config::default()).await;

    // 3. Verify
    assert!(
        engine.is_ok(),
        "RagEngine should recover from corrupt index"
    );
    let engine = engine.unwrap();

    // It should have marked itself for reindex
    assert!(
        engine.needs_reindex(),
        "Should mark for reindex after corruption"
    );
    assert!(
        engine.list_documents().is_empty(),
        "Should have empty state"
    );
}

#[tokio::test]
#[serial]
async fn test_load_existing_valid_index() {
    let mock_server = setup_mock_ollama().await;
    let temp_dir = tempfile::tempdir().unwrap();
    let data_dir = temp_dir.path().to_str().unwrap();
    let model_name = "nomic-embed-text";
    unsafe {
        std::env::set_var("OLLAMA_URL", mock_server.uri());
        std::env::set_var("OLLAMA_EMBEDDING_MODEL", model_name);
    }

    // 1. Create a valid index file manually
    let index_path = temp_dir.path().join(format!("chunks_{model_name}.json"));
    let valid_json = serde_json::json!({
        "version": 2,
        "model": model_name,
        "chunks": {
            "chunk-1": {
                "id": "chunk-1",
                "document_name": "test.pdf",
                "text": "Hello World",
                "embedding": vec![0.0f32; 384],
                "chunk_index": 0,
                "page_number": 1,
                "metadata": {
                    "token_count": 2,
                    "overlap_with_previous": 0
                }
            }
        },
        "needs_reindex": false,
        "document_hashes": {
            "test.pdf": "hash123"
        }
    });
    std::fs::write(&index_path, serde_json::to_string(&valid_json).unwrap()).unwrap();

    // 2. Initialize RagEngine
    let engine = RagEngine::new(data_dir, &Config::default())
        .await
        .expect("Failed to load valid index");

    // 3. Verify loaded data
    assert!(!engine.needs_reindex(), "Should not need reindex");
    let docs = engine.list_documents();
    assert_eq!(docs.len(), 1);
    assert_eq!(docs[0], "test.pdf");
}
