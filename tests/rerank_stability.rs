use rag_core::RerankerCandidate;
use rust_local_rag::config::Config;
use rust_local_rag::reranker::RerankerService;
use serial_test::serial;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
#[serial]
async fn test_rerank_stability_ties() {
    // Start mock server
    let mock_server = MockServer::start().await;

    // Mock /api/tags (used by verify_model)
    Mock::given(method("GET"))
        .and(path("/api/tags"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "models": [
                { "name": "dengcao/Qwen3-Reranker-4B:Q5_K_M" }
            ]
        })))
        .mount(&mock_server)
        .await;

    // Mock /api/generate (used by rerank)
    // Return "Yes" (score 1.0) for all requests to force ties.
    Mock::given(method("POST"))
        .and(path("/api/generate"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "response": "Yes",
            "logprobs": [
                {
                    "token": "Yes",
                    "logprob": 0.0,
                    "top_logprobs": []
                }
            ]
        })))
        .mount(&mock_server)
        .await;

    // Set ENV var for this test to point to mock server
    unsafe {
        std::env::set_var("OLLAMA_URL", mock_server.uri());
    }

    let config = Config::default();

    let service = RerankerService::new(&config)
        .await
        .expect("Failed to create service");

    // Create candidates with different IDs but same score (due to mock response)
    // Use IDs that are not sorted alphabetically initially to verify sort works.
    let candidates = vec![
        RerankerCandidate {
            chunk_id: "c".to_string(),
            document: "doc1".to_string(),
            text: "text".to_string(),
            page_number: 1,
            section: None,
            initial_score: 0.5,
        },
        RerankerCandidate {
            chunk_id: "a".to_string(),
            document: "doc1".to_string(),
            text: "text".to_string(),
            page_number: 1,
            section: None,
            initial_score: 0.5,
        },
        RerankerCandidate {
            chunk_id: "b".to_string(),
            document: "doc1".to_string(),
            text: "text".to_string(),
            page_number: 1,
            section: None,
            initial_score: 0.5,
        },
    ];

    let results = service
        .rerank("query", &candidates)
        .await
        .expect("Rerank failed");

    // Check order. Should be a, b, c.
    let ids: Vec<String> = results.into_iter().map(|r| r.chunk_id).collect();

    // Clean up env var
    unsafe {
        std::env::remove_var("OLLAMA_URL");
    }

    assert_eq!(
        ids,
        vec!["a", "b", "c"],
        "Order should be deterministic and sorted by chunk_id for ties"
    );
}
