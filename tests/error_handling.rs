use rust_local_rag::{Config, RagEngine, RagError};
use serial_test::serial;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

struct EnvVarGuard {
    key: &'static str,
    original: Option<std::ffi::OsString>,
}

impl EnvVarGuard {
    fn set(key: &'static str, value: &str) -> Self {
        let original = std::env::var_os(key);
        unsafe {
            std::env::set_var(key, value);
        }
        Self { key, original }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        unsafe {
            match &self.original {
                Some(value) => std::env::set_var(self.key, value),
                None => std::env::remove_var(self.key),
            }
        }
    }
}

#[tokio::test]
#[serial]
async fn test_bad_pdf_returns_actionable_error() {
    let temp_dir = tempfile::tempdir().unwrap();
    let data_dir = temp_dir.path().to_str().unwrap();

    let mock_server = MockServer::start().await;
    let _ollama_url = EnvVarGuard::set("OLLAMA_URL", &mock_server.uri());
    let _ollama_model = EnvVarGuard::set("OLLAMA_EMBEDDING_MODEL", "test-model");

    Mock::given(method("GET"))
        .and(path("/api/tags"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "models": [
                { "name": "test-model:latest" }
            ]
        })))
        .mount(&mock_server)
        .await;

    let engine = RagEngine::new(data_dir, &Config::default()).await.unwrap();

    let err = engine
        .prepare_document("bad.pdf", b"not a pdf", None)
        .await
        .unwrap_err();

    let rag_err = err.downcast_ref::<RagError>().expect("expected RagError");
    assert!(matches!(rag_err, RagError::PdfExtraction { .. }));
    let msg = rag_err.to_string();
    assert!(msg.contains("bad.pdf"));
    assert!(msg.contains("Fix:"));
}

#[tokio::test]
#[serial]
async fn test_missing_ollama_returns_actionable_error() {
    let temp_dir = tempfile::tempdir().unwrap();
    let data_dir = temp_dir.path().to_str().unwrap();

    let _ollama_url = EnvVarGuard::set("OLLAMA_URL", "http://127.0.0.1:1");
    let _ollama_model = EnvVarGuard::set("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text");

    let err = RagEngine::new(data_dir, &Config::default())
        .await
        .err()
        .expect("expected engine init to fail");

    let rag_err = err.downcast_ref::<RagError>().expect("expected RagError");
    assert!(matches!(rag_err, RagError::Embedding { .. }));

    let msg = rag_err.to_string();
    assert!(msg.contains("Start Ollama"), "msg was: {msg}");
    assert!(msg.contains("OLLAMA_URL"), "msg was: {msg}");
}

#[tokio::test]
#[serial]
async fn test_corrupt_index_json_marks_needs_reindex() {
    let temp_dir = tempfile::tempdir().unwrap();
    let data_dir = temp_dir.path().to_str().unwrap();

    let mock_server = MockServer::start().await;
    let _ollama_url = EnvVarGuard::set("OLLAMA_URL", &mock_server.uri());
    let _ollama_model = EnvVarGuard::set("OLLAMA_EMBEDDING_MODEL", "test-model");

    Mock::given(method("GET"))
        .and(path("/api/tags"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "models": [
                { "name": "test-model:latest" }
            ]
        })))
        .mount(&mock_server)
        .await;

    let index_path = rag_core::persistence::index_path(data_dir, "test-model");
    std::fs::write(index_path, "{ not valid json").unwrap();

    let engine = RagEngine::new(data_dir, &Config::default()).await.unwrap();
    assert!(engine.needs_reindex());
}
