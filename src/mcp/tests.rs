use super::formatting::format_search_results;
use crate::embeddings::EmbeddingService;
use crate::job_manager::JobManager;
use crate::rag_engine::RagEngine;
use crate::rag_engine::SearchResult;
use serial_test::serial;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::sync::mpsc;

#[test]
fn test_format_search_results() {
    let results = vec![
        SearchResult {
            text: "The quick brown fox jumps over the lazy dog.".to_string(),
            score: 0.8531,
            document: "fox.pdf".to_string(),
            chunk_id: "chunk-123".to_string(),
            chunk_index: 0,
            page_number: 1,
            section: Some("Intro".to_string()),
            embedding_score: Some(0.82),
            lexical_score: Some(0.65),
            initial_score: Some(0.7),
            reranker_score: None,
            yes_logprob: None,
            no_logprob: None,
        },
        SearchResult {
            text: "Lorem ipsum dolor sit amet.".to_string(),
            score: 0.725,
            document: "lorem.pdf".to_string(),
            chunk_id: "chunk-456".to_string(),
            chunk_index: 5,
            page_number: 10,
            section: None,
            embedding_score: Some(0.72),
            lexical_score: None,
            initial_score: None,
            reranker_score: Some(0.95),
            yes_logprob: None,
            no_logprob: None,
        },
    ];

    let formatted = format_search_results(&results, "fox");

    assert!(formatted.contains("**1. [85%] fox.pdf (page 1)**"));
    assert!(formatted.contains("*Section: Intro*"));
    assert!(formatted.contains("Scores: Semantic: 0.82 | Keyword: 0.65"));
    assert!(formatted.contains("The quick brown **fox**"));

    assert!(formatted.contains("---\n\n"));

    assert!(formatted.contains("**2. [73%] lorem.pdf (page 10)**"));
    assert!(formatted.contains("Scores: Semantic: 0.72 | Reranker: 0.95"));
}

#[test]
fn test_format_search_results_boundary() {
    let results = vec![SearchResult {
        text: "The bobcat matches feline but should not highlight.".to_string(),
        score: 0.8,
        document: "cat.pdf".to_string(),
        chunk_id: "id".to_string(),
        chunk_index: 0,
        page_number: 1,
        section: None,
        embedding_score: None,
        lexical_score: None,
        initial_score: None,
        reranker_score: None,
        yes_logprob: None,
        no_logprob: None,
    }];

    let formatted = format_search_results(&results, "cat");
    assert!(!formatted.contains("**cat**"));
    assert!(formatted.contains("The bobcat matches"));

    let results_match = vec![SearchResult {
        text: "My cat is nice.".to_string(),
        score: 0.8,
        document: "cat.pdf".to_string(),
        chunk_id: "id".to_string(),
        chunk_index: 0,
        page_number: 1,
        section: None,
        embedding_score: None,
        lexical_score: None,
        initial_score: None,
        reranker_score: None,
        yes_logprob: None,
        no_logprob: None,
    }];

    let formatted_match = format_search_results(&results_match, "cat");
    assert!(formatted_match.contains("My **cat** is nice"));
}

#[test]
fn test_format_search_results_short_terms() {
    let results = vec![SearchResult {
        text: "The AI is coming.".to_string(),
        score: 0.8,
        document: "ai.pdf".to_string(),
        chunk_id: "id".to_string(),
        chunk_index: 0,
        page_number: 1,
        section: None,
        embedding_score: None,
        lexical_score: None,
        initial_score: None,
        reranker_score: None,
        yes_logprob: None,
        no_logprob: None,
    }];

    let formatted = format_search_results(&results, "AI");
    assert!(formatted.contains("The **AI** is coming"));
}

#[test]
fn test_format_search_results_does_not_highlight_two_letter_stopwords() {
    let results = vec![SearchResult {
        text: "This is a test.".to_string(),
        score: 0.8,
        document: "stopwords.pdf".to_string(),
        chunk_id: "id".to_string(),
        chunk_index: 0,
        page_number: 1,
        section: None,
        embedding_score: None,
        lexical_score: None,
        initial_score: None,
        reranker_score: None,
        yes_logprob: None,
        no_logprob: None,
    }];

    let formatted = format_search_results(&results, "is");
    assert!(!formatted.contains("**is**"));
}

#[tokio::test]
#[serial]
async fn test_health_endpoint() {
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

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

    let temp_dir = tempfile::tempdir().unwrap();

    let embedding_service =
        EmbeddingService::new_with_config(mock_server.uri(), "nomic-embed-text".to_string())
            .await
            .expect("EmbeddingService init failed");

    let rag_engine = RagEngine::new_with_embedding_service(
        temp_dir.path().to_str().unwrap(),
        embedding_service,
        &crate::config::Config::default(),
    )
    .await
    .expect("RagEngine init failed");
    let rag_state = Arc::new(RwLock::new(rag_engine));

    let job_manager = Arc::new(JobManager::new("sqlite::memory:").await.unwrap());
    let (job_tx, _rx) = mpsc::channel(1);

    let documents_dir = temp_dir.path().to_string_lossy().to_string();

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();

    let server_handle = tokio::spawn(async move {
        super::start_mcp_server(rag_state, job_manager, job_tx, documents_dir, listener)
            .await
            .unwrap();
    });

    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    let client = reqwest::Client::new();
    let response = client
        .get(format!("http://127.0.0.1:{}/health", port))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), reqwest::StatusCode::OK);
    server_handle.abort();
}
