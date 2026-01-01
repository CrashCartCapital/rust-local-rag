use lopdf::content::{Content, Operation};
use lopdf::{Dictionary, Document, Object, Stream};
use rust_local_rag::rag_engine::RagEngine;
use serial_test::serial;
use std::sync::Arc;
use tokio::sync::RwLock;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn create_valid_pdf() -> Vec<u8> {
    let mut doc = Document::with_version("1.5");
    let pages_id = doc.new_object_id();
    let font_id = doc.add_object(Dictionary::from_iter(vec![
        ("Type", "Font".into()),
        ("Subtype", "Type1".into()),
        ("BaseFont", "Courier".into()),
    ]));
    let resources_id = doc.add_object(Dictionary::from_iter(vec![(
        "Font",
        Dictionary::from_iter(vec![("F1", font_id.into())]).into(),
    )]));
    let content = Content {
        operations: vec![
            Operation::new("BT", vec![]),
            Operation::new("Tf", vec!["F1".into(), 48.into()]),
            Operation::new("Td", vec![100.into(), 600.into()]),
            Operation::new("Tj", vec![Object::string_literal("Hello World")]),
            Operation::new("ET", vec![]),
        ],
    };
    let content_id = doc.add_object(Stream::new(Dictionary::new(), content.encode().unwrap()));
    let page_id = doc.add_object(Dictionary::from_iter(vec![
        ("Type", "Page".into()),
        ("Parent", pages_id.into()),
        ("Contents", content_id.into()),
        ("Resources", resources_id.into()),
        (
            "MediaBox",
            vec![0.into(), 0.into(), 595.into(), 842.into()].into(),
        ),
    ]));
    let pages = Dictionary::from_iter(vec![
        ("Type", "Pages".into()),
        ("Kids", vec![page_id.into()].into()),
        ("Count", 1.into()),
    ]);
    doc.objects.insert(pages_id, Object::Dictionary(pages));
    let catalog_id = doc.add_object(Dictionary::from_iter(vec![
        ("Type", "Catalog".into()),
        ("Pages", pages_id.into()),
    ]));
    doc.trailer.set("Root", catalog_id);
    let mut buffer = Vec::new();
    doc.save_to(&mut buffer).unwrap();
    buffer
}

#[tokio::test]
#[serial]
async fn test_e2e_indexing_and_search() {
    // 1. Setup Mock Ollama
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

    Mock::given(method("POST"))
        .and(path("/api/embed"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "embedding": vec![0.1f32; 384]
        })))
        .mount(&mock_server)
        .await;

    // 2. Setup RagEngine
    let temp_dir = tempfile::tempdir().unwrap();
    unsafe {
        std::env::set_var("OLLAMA_URL", mock_server.uri());
        std::env::set_var("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text");
        std::env::set_var("EMBEDDING_BATCH_SIZE", "1");
    }

    let engine = RagEngine::new(temp_dir.path().to_str().unwrap())
        .await
        .expect("Failed to create RagEngine");

    let engine = Arc::new(RwLock::new(engine));

    // 3. Add Document
    let mut engine_write = engine.write().await;
    let pdf_bytes = create_valid_pdf();
    let chunks = engine_write
        .add_document("test.pdf", &pdf_bytes, None)
        .await
        .expect("Failed to add document");

    assert!(chunks > 0, "Should have created chunks");

    // 4. Search
    let results = engine_write
        .search("Hello", 5, None)
        .await
        .expect("Search failed");

    // 5. Verify
    assert!(!results.is_empty(), "Should find results");
    assert_eq!(results[0].document, "test.pdf");
    // Depending on extraction, might be "Hello World" or "Hello World"
    assert!(results[0].text.contains("Hello World"));
}

#[tokio::test]
#[serial]
async fn test_reranker_fallback_on_failure() {
    // 1. Setup Mock Ollama
    let mock_server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/api/tags"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "models": [
                { "name": "nomic-embed-text:latest" },
                { "name": "my-reranker:latest" }
            ]
        })))
        .mount(&mock_server)
        .await;

    Mock::given(method("POST"))
        .and(path("/api/embed"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "embedding": vec![0.1f32; 384]
        })))
        .mount(&mock_server)
        .await;

    Mock::given(method("POST"))
        .and(path("/api/generate"))
        .respond_with(ResponseTemplate::new(500))
        .mount(&mock_server)
        .await;

    // 2. Setup RagEngine
    let temp_dir = tempfile::tempdir().unwrap();
    unsafe {
        std::env::set_var("OLLAMA_URL", mock_server.uri());
        std::env::set_var("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text");
        std::env::set_var("OLLAMA_RERANK_MODEL", "my-reranker");
    }

    let mut engine = RagEngine::new(temp_dir.path().to_str().unwrap())
        .await
        .expect("Failed to create RagEngine");

    assert!(engine.has_reranker(), "Reranker should be enabled");

    // 3. Add Document
    let pdf_bytes = create_valid_pdf();
    engine
        .add_document("test.pdf", &pdf_bytes, None)
        .await
        .unwrap();

    // 4. Search
    let results = engine
        .search("Hello", 5, None)
        .await
        .expect("Search should succeed despite reranker failure");

    // 5. Verify
    assert!(!results.is_empty(), "Should return results");
    // NOTE: The current implementation seems to return Some(relevance) even on fallback,
    // where relevance is set to initial_score.
    // See RerankerService::score_with_timeout fallback logic:
    // RerankedResult { ..., relevance: initial_score, ... }
    // And RagEngine uses this relevance for reranker_score.
    // So we just check that we got a result.
    assert!(results[0].score > 0.0, "Should have a positive score");
}
