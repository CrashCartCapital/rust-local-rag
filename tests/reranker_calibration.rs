use rust_local_rag::{
    Config,
    reranker::{RerankerCandidate, RerankerService},
};
use serial_test::serial;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
#[serial]
async fn test_calibrate_timeout_all_failed() {
    // Start mock server
    let mock_server = MockServer::start().await;

    // Mock tags endpoint (needed for new())
    Mock::given(method("GET"))
        .and(path("/api/tags"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "models": [
                { "name": "dengcao/Qwen3-Reranker-4B:Q5_K_M" }
            ]
        })))
        .mount(&mock_server)
        .await;

    // Mock generate endpoint to always fail (500 Internal Server Error)
    Mock::given(method("POST"))
        .and(path("/api/generate"))
        .respond_with(ResponseTemplate::new(500))
        .mount(&mock_server)
        .await;

    // Setup env
    let original_url = std::env::var("OLLAMA_URL").ok();
    unsafe {
        std::env::set_var("OLLAMA_URL", mock_server.uri());
    }

    // Create service
    let config = Config::default();
    let service = RerankerService::new(&config)
        .await
        .expect("Service init failed");

    // Create candidates
    let candidates = vec![RerankerCandidate {
        chunk_id: "1".to_string(),
        text: "text".to_string(),
        document: "doc".to_string(),
        page_number: 1,
        section: None,
        initial_score: 0.5,
    }];

    // Run calibration
    let result = service.calibrate_timeout("query", &candidates, 1).await;

    // Restore env
    unsafe {
        if let Some(url) = original_url {
            std::env::set_var("OLLAMA_URL", url);
        } else {
            std::env::remove_var("OLLAMA_URL");
        }
    }

    // Assert error
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("All calibration samples failed"),
        "Unexpected error: {err_msg}"
    );
}
