use lopdf::content::{Content, Operation};
use lopdf::{Dictionary, Document, Object, Stream};
use rust_local_rag::Config;
use rust_local_rag::rag_engine::RagEngine;
use serial_test::serial;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn create_pdf_with_text(text: &str) -> Vec<u8> {
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
            Operation::new("Tj", vec![Object::string_literal(text)]),
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

fn create_valid_pdf() -> Vec<u8> {
    create_pdf_with_text("Hello World")
}

fn create_multi_chunk_pdf() -> Vec<u8> {
    let long_text = (0..200)
        .map(|i| format!("Sentence {i:03} has enough text to force chunking."))
        .collect::<Vec<_>>()
        .join(" ");
    create_pdf_with_text(&long_text)
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
        std::env::set_var("RAG_EMBEDDING_BATCH_SIZE", "1");
    }

    let config = Config::from_env().unwrap();
    let engine = RagEngine::new(temp_dir.path().to_str().unwrap(), &config)
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
async fn test_prd_t0_3_batch_size_produces_batch_embed_request_shape() {
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

    // Return a single-embedding payload for all requests; we only assert request shape.
    Mock::given(method("POST"))
        .and(path("/api/embed"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "embedding": vec![0.1f32; 384]
        })))
        .mount(&mock_server)
        .await;

    let temp_dir = tempfile::tempdir().unwrap();
    unsafe {
        std::env::set_var("OLLAMA_URL", mock_server.uri());
        std::env::set_var("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text");
        std::env::set_var("RAG_EMBEDDING_BATCH_SIZE", "2");
    }

    let config = Config::from_env().unwrap();
    let mut engine = RagEngine::new(temp_dir.path().to_str().unwrap(), &config)
        .await
        .expect("Failed to create RagEngine");

    let pdf_bytes = create_multi_chunk_pdf();
    let chunks = engine
        .add_document("multi-chunk.pdf", &pdf_bytes, None)
        .await
        .expect("Failed to add document");

    assert!(chunks >= 2, "Expected multi-chunk PDF to yield >= 2 chunks");

    let requests = mock_server.received_requests().await.unwrap();
    let mut saw_batch_request = false;
    for req in requests {
        if req.method.as_str() != "POST" || req.url.path() != "/api/embed" {
            continue;
        }
        let Ok(body) = serde_json::from_slice::<serde_json::Value>(&req.body) else {
            continue;
        };
        if body.get("input").is_some_and(|v| v.is_array()) {
            saw_batch_request = true;
            break;
        }
    }

    assert!(
        saw_batch_request,
        "Expected at least one /api/embed request with array `input` when RAG_EMBEDDING_BATCH_SIZE > 1"
    );
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

    let mut engine = RagEngine::new(temp_dir.path().to_str().unwrap(), &Config::default())
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

#[tokio::test]
#[serial]
async fn test_reranker_timeout_fallback() {
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

    // Deliberately slow reranker response to trigger client timeout.
    Mock::given(method("POST"))
        .and(path("/api/generate"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_delay(Duration::from_millis(1000))
                .set_body_json(serde_json::json!({
                    "response": "Yes",
                    "logprobs": []
                })),
        )
        .mount(&mock_server)
        .await;

    let temp_dir = tempfile::tempdir().unwrap();
    unsafe {
        std::env::set_var("OLLAMA_URL", mock_server.uri());
        std::env::set_var("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text");
        std::env::set_var("OLLAMA_RERANK_MODEL", "my-reranker");
        std::env::set_var("RAG_EMBEDDING_BATCH_SIZE", "1");
    }

    let mut config = Config::from_env().unwrap();
    config.reranker_timeout = Duration::from_millis(200);

    let mut engine = RagEngine::new(temp_dir.path().to_str().unwrap(), &config)
        .await
        .expect("Failed to create RagEngine");

    assert!(engine.has_reranker(), "Reranker should be enabled");

    let pdf_bytes = create_valid_pdf();
    engine
        .add_document("test.pdf", &pdf_bytes, None)
        .await
        .unwrap();

    let results = engine
        .search("Hello", 1, None)
        .await
        .expect("Search should succeed despite reranker timeout");

    assert!(!results.is_empty(), "Should return results");
    assert!(results[0].score > 0.0, "Should have a positive score");
}
