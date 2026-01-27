use std::process::Command;

use serial_test::serial;

#[tokio::test]
#[serial]
async fn test_prd_t2_1_rag_doctor_happy_path_exit_zero() {
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let mock_server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/tags"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "models": [
                { "name": "test-embed:latest" },
                { "name": "test-rerank:latest" }
            ]
        })))
        .mount(&mock_server)
        .await;

    let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
    let data_dir = temp_dir.path().join("data");
    let docs_dir = temp_dir.path().join("docs");
    std::fs::create_dir_all(&data_dir).expect("Failed to create data dir");
    std::fs::create_dir_all(&docs_dir).expect("Failed to create docs dir");

    let bin_path = env!("CARGO_BIN_EXE_rag-doctor");
    let output = Command::new(bin_path)
        .env("OLLAMA_URL", mock_server.uri())
        .env("OLLAMA_EMBEDDING_MODEL", "test-embed")
        .env("OLLAMA_RERANK_MODEL", "test-rerank")
        .env("DATA_DIR", data_dir.to_str().unwrap())
        .env("DOCUMENTS_DIR", docs_dir.to_str().unwrap())
        .output()
        .expect("Failed to run rag-doctor");

    assert!(
        output.status.success(),
        "Expected rag-doctor to exit 0, got: {:?}\nstdout:\n{}\nstderr:\n{}",
        output.status.code(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("[PASS] DATA_DIR"));
    assert!(stdout.contains("[PASS] DOCUMENTS_DIR"));
    assert!(stdout.contains("[PASS] OLLAMA_URL"));
    assert!(stdout.contains("[PASS] OLLAMA_EMBEDDING_MODEL"));
    assert!(stdout.contains("[PASS] OLLAMA_RERANK_MODEL"));
}

#[tokio::test]
#[serial]
async fn test_prd_t2_1_rag_doctor_fails_when_embedding_model_missing() {
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let mock_server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/tags"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "models": [
                { "name": "present:latest" }
            ]
        })))
        .mount(&mock_server)
        .await;

    let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
    let data_dir = temp_dir.path().join("data");
    let docs_dir = temp_dir.path().join("docs");
    std::fs::create_dir_all(&data_dir).expect("Failed to create data dir");
    std::fs::create_dir_all(&docs_dir).expect("Failed to create docs dir");

    let bin_path = env!("CARGO_BIN_EXE_rag-doctor");
    let output = Command::new(bin_path)
        .env("OLLAMA_URL", mock_server.uri())
        .env("OLLAMA_EMBEDDING_MODEL", "missing")
        .env("DATA_DIR", data_dir.to_str().unwrap())
        .env("DOCUMENTS_DIR", docs_dir.to_str().unwrap())
        .output()
        .expect("Failed to run rag-doctor");

    assert!(
        !output.status.success(),
        "Expected rag-doctor to exit non-zero when embedding model missing.\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("[FAIL] OLLAMA_EMBEDDING_MODEL"));
}
