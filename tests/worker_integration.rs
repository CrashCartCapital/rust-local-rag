use lopdf::content::{Content, Operation};
use lopdf::{Dictionary, Document, Object, Stream};
use rust_local_rag::job_manager::{JobStatus, JobType};
use rust_local_rag::{Config, JobManager, JobRequest, RagEngine, WorkerSupervisor};
use serial_test::serial;
use std::sync::Arc;
use tokio::sync::{RwLock, mpsc};
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
async fn test_worker_completes_job() {
    // 1. Setup Environment & Mock Ollama
    let mock_server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/tags"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "models": [{ "name": "nomic-embed-text:latest" }]
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

    unsafe {
        std::env::set_var("OLLAMA_URL", mock_server.uri());
        std::env::set_var("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text");
        std::env::set_var("EMBEDDING_BATCH_SIZE", "1");
    }

    let temp_dir = tempfile::tempdir().unwrap();
    let data_dir = temp_dir.path().join("data");
    let docs_dir = temp_dir.path().join("docs");
    let log_dir = temp_dir.path().join("logs");
    std::fs::create_dir_all(&data_dir).unwrap();
    std::fs::create_dir_all(&docs_dir).unwrap();
    std::fs::create_dir_all(&log_dir).unwrap();

    // Set LOG_DIR for progress logger
    unsafe {
        std::env::set_var("LOG_DIR", log_dir.to_str().unwrap());
    }

    // 2. Write a dummy PDF
    let pdf_path = docs_dir.join("test.pdf");
    std::fs::write(&pdf_path, create_valid_pdf()).unwrap();

    // 3. Initialize Components
    let db_path = format!("sqlite:{}/jobs.db", data_dir.to_str().unwrap());
    let job_manager = Arc::new(JobManager::new(&db_path).await.unwrap());
    let rag_engine = Arc::new(RwLock::new(
        RagEngine::new(data_dir.to_str().unwrap(), &Config::default())
            .await
            .unwrap(),
    ));
    let (job_tx, job_rx) = mpsc::channel(10);

    // 4. Start Worker
    let supervisor = WorkerSupervisor::new(job_manager.clone(), rag_engine.clone(), job_rx);
    tokio::spawn(async move {
        supervisor.run().await;
    });

    // 5. Create Job
    // Ideally we use the method that creates and sends, but we are testing worker integration
    // so we manually create job in DB and send request.
    let job = job_manager
        .create_job(JobType::Reindex, None, 0)
        .await
        .unwrap();

    job_tx
        .send(JobRequest::StartReindex {
            job_id: job.job_id.clone(),
            documents_dir: docs_dir.to_str().unwrap().to_string(),
        })
        .await
        .unwrap();

    // 6. Wait for Completion
    let mut attempts = 0;
    loop {
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        let current_job = job_manager.get_job(&job.job_id).await.unwrap().unwrap();

        if current_job.status == JobStatus::Completed {
            break;
        }
        if current_job.status == JobStatus::Failed {
            panic!("Job failed: {:?}", current_job.error);
        }

        attempts += 1;
        if attempts > 150 {
            // 15 seconds timeout
            panic!("Job timed out. Status: {:?}", current_job.status);
        }
    }

    // 7. Verify Results
    let engine = rag_engine.read().await;
    let docs = engine.list_documents();
    assert_eq!(docs.len(), 1, "Should have indexed 1 document");
    assert_eq!(docs[0], "test.pdf");
}

#[tokio::test]
#[serial]
async fn test_worker_completes_job_with_corrupt_pdf() {
    let mock_server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/tags"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "models": [{ "name": "nomic-embed-text:latest" }]
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

    unsafe {
        std::env::set_var("OLLAMA_URL", mock_server.uri());
        std::env::set_var("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text");
        std::env::set_var("EMBEDDING_BATCH_SIZE", "1");
    }

    let temp_dir = tempfile::tempdir().unwrap();
    let data_dir = temp_dir.path().join("data");
    let docs_dir = temp_dir.path().join("docs");
    let log_dir = temp_dir.path().join("logs");
    std::fs::create_dir_all(&data_dir).unwrap();
    std::fs::create_dir_all(&docs_dir).unwrap();
    std::fs::create_dir_all(&log_dir).unwrap();

    unsafe {
        std::env::set_var("LOG_DIR", log_dir.to_str().unwrap());
    }

    std::fs::write(docs_dir.join("test.pdf"), create_valid_pdf()).unwrap();
    std::fs::write(docs_dir.join("corrupt.pdf"), b"not a pdf").unwrap();

    let db_path = format!("sqlite:{}/jobs.db", data_dir.to_str().unwrap());
    let job_manager = Arc::new(JobManager::new(&db_path).await.unwrap());
    let rag_engine = Arc::new(RwLock::new(
        RagEngine::new(data_dir.to_str().unwrap(), &Config::default())
            .await
            .unwrap(),
    ));
    let (job_tx, job_rx) = mpsc::channel(10);

    let supervisor = WorkerSupervisor::new(job_manager.clone(), rag_engine.clone(), job_rx);
    tokio::spawn(async move {
        supervisor.run().await;
    });

    let job = job_manager
        .create_job(JobType::Reindex, None, 0)
        .await
        .unwrap();

    job_tx
        .send(JobRequest::StartReindex {
            job_id: job.job_id.clone(),
            documents_dir: docs_dir.to_str().unwrap().to_string(),
        })
        .await
        .unwrap();

    let mut attempts = 0;
    loop {
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        let current_job = job_manager.get_job(&job.job_id).await.unwrap().unwrap();

        if current_job.status == JobStatus::Completed {
            assert!(
                current_job
                    .error
                    .as_deref()
                    .unwrap_or("")
                    .contains("corrupt.pdf"),
                "Expected job error summary to mention corrupt.pdf"
            );
            break;
        }
        if current_job.status == JobStatus::Failed {
            panic!("Job failed: {:?}", current_job.error);
        }

        attempts += 1;
        if attempts > 200 {
            panic!("Job timed out. Status: {:?}", current_job.status);
        }
    }

    let engine = rag_engine.read().await;
    let docs = engine.list_documents();
    assert_eq!(docs.len(), 1, "Should have indexed only the valid document");
    assert_eq!(docs[0], "test.pdf");
}

#[tokio::test]
#[serial]
async fn test_worker_prunes_deleted_documents() {
    // 1. Setup Environment & Mock Ollama
    let mock_server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/tags"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "models": [{ "name": "nomic-embed-text:latest" }]
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

    unsafe {
        std::env::set_var("OLLAMA_URL", mock_server.uri());
        std::env::set_var("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text");
        std::env::set_var("EMBEDDING_BATCH_SIZE", "1");
    }

    let temp_dir = tempfile::tempdir().unwrap();
    let data_dir = temp_dir.path().join("data");
    let docs_dir = temp_dir.path().join("docs");
    let log_dir = temp_dir.path().join("logs");
    std::fs::create_dir_all(&data_dir).unwrap();
    std::fs::create_dir_all(&docs_dir).unwrap();
    std::fs::create_dir_all(&log_dir).unwrap();

    unsafe {
        std::env::set_var("LOG_DIR", log_dir.to_str().unwrap());
    }

    // 2. Write a dummy PDF
    let pdf_path = docs_dir.join("test.pdf");
    std::fs::write(&pdf_path, create_valid_pdf()).unwrap();

    // 3. Initialize Components
    let db_path = format!("sqlite:{}/jobs.db", data_dir.to_str().unwrap());
    let job_manager = Arc::new(JobManager::new(&db_path).await.unwrap());
    let rag_engine = Arc::new(RwLock::new(
        RagEngine::new(data_dir.to_str().unwrap(), &Config::default())
            .await
            .unwrap(),
    ));
    let (job_tx, job_rx) = mpsc::channel(10);

    let supervisor = WorkerSupervisor::new(job_manager.clone(), rag_engine.clone(), job_rx);
    tokio::spawn(async move {
        supervisor.run().await;
    });

    // 4. Run initial reindex to create index state
    let job1 = job_manager
        .create_job(JobType::Reindex, None, 0)
        .await
        .unwrap();

    job_tx
        .send(JobRequest::StartReindex {
            job_id: job1.job_id.clone(),
            documents_dir: docs_dir.to_str().unwrap().to_string(),
        })
        .await
        .unwrap();

    let mut attempts = 0;
    loop {
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        let current_job = job_manager.get_job(&job1.job_id).await.unwrap().unwrap();

        if current_job.status == JobStatus::Completed {
            break;
        }
        if current_job.status == JobStatus::Failed {
            panic!("Job failed: {:?}", current_job.error);
        }

        attempts += 1;
        if attempts > 150 {
            panic!("Job timed out. Status: {:?}", current_job.status);
        }
    }

    {
        let engine = rag_engine.read().await;
        assert_eq!(engine.list_documents(), vec!["test.pdf".to_string()]);
    }

    // 5. Delete the PDF and run reindex again; index should be pruned
    std::fs::remove_file(&pdf_path).unwrap();

    let job2 = job_manager
        .create_job(JobType::Reindex, None, 0)
        .await
        .unwrap();

    job_tx
        .send(JobRequest::StartReindex {
            job_id: job2.job_id.clone(),
            documents_dir: docs_dir.to_str().unwrap().to_string(),
        })
        .await
        .unwrap();

    let mut attempts = 0;
    loop {
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        let current_job = job_manager.get_job(&job2.job_id).await.unwrap().unwrap();

        if current_job.status == JobStatus::Completed {
            break;
        }
        if current_job.status == JobStatus::Failed {
            panic!("Job failed: {:?}", current_job.error);
        }

        attempts += 1;
        if attempts > 150 {
            panic!("Job timed out. Status: {:?}", current_job.status);
        }
    }

    let engine = rag_engine.read().await;
    assert!(
        engine.list_documents().is_empty(),
        "Deleted documents should be pruned from index"
    );
}

#[tokio::test]
#[serial]
async fn test_supervisor_fails_stale_jobs_on_startup() {
    // Setup mock Ollama (tags + canary embed)
    let mock_server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/tags"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "models": [{ "name": "nomic-embed-text:latest" }]
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

    unsafe {
        std::env::set_var("OLLAMA_URL", mock_server.uri());
        std::env::set_var("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text");
        std::env::set_var("EMBEDDING_BATCH_SIZE", "1");
    }

    let temp_dir = tempfile::tempdir().unwrap();
    let data_dir = temp_dir.path().join("data");
    let docs_dir = temp_dir.path().join("docs");
    let log_dir = temp_dir.path().join("logs");
    std::fs::create_dir_all(&data_dir).unwrap();
    std::fs::create_dir_all(&docs_dir).unwrap();
    std::fs::create_dir_all(&log_dir).unwrap();

    unsafe {
        std::env::set_var("LOG_DIR", log_dir.to_str().unwrap());
    }

    let db_path = format!("sqlite:{}/jobs.db", data_dir.to_str().unwrap());
    let job_manager = Arc::new(JobManager::new(&db_path).await.unwrap());
    let rag_engine = Arc::new(RwLock::new(
        RagEngine::new(data_dir.to_str().unwrap(), &Config::default())
            .await
            .unwrap(),
    ));
    let (_job_tx, job_rx) = mpsc::channel(10);

    // Create two resumable jobs before supervisor starts; the older one should be failed.
    let job1 = job_manager
        .create_job(
            JobType::Reindex,
            Some(docs_dir.to_str().unwrap().to_string()),
            0,
        )
        .await
        .unwrap();

    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    let job2 = job_manager
        .create_job(
            JobType::Reindex,
            Some(docs_dir.to_str().unwrap().to_string()),
            0,
        )
        .await
        .unwrap();

    let supervisor = WorkerSupervisor::new(job_manager.clone(), rag_engine.clone(), job_rx);
    tokio::spawn(async move {
        supervisor.run().await;
    });

    // Wait for job1 to be failed as stale/superseded
    let mut attempts = 0;
    loop {
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        let j1 = job_manager.get_job(&job1.job_id).await.unwrap().unwrap();
        let j2 = job_manager.get_job(&job2.job_id).await.unwrap().unwrap();

        if j1.status == JobStatus::Failed {
            assert!(
                j1.error
                    .as_deref()
                    .unwrap_or("")
                    .contains(&job2.job_id),
                "Expected stale job error to mention newer job id"
            );
            assert_ne!(j2.status, JobStatus::Failed, "Newer job should not fail");
            break;
        }

        attempts += 1;
        if attempts > 150 {
            panic!("Timed out waiting for stale job to be failed. j1={j1:?} j2={j2:?}");
        }
    }
}

#[tokio::test]
#[serial]
async fn test_supervisor_marks_job_completed_when_progress_reached_total() {
    // Setup mock Ollama (tags + canary embed)
    let mock_server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/tags"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "models": [{ "name": "nomic-embed-text:latest" }]
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

    unsafe {
        std::env::set_var("OLLAMA_URL", mock_server.uri());
        std::env::set_var("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text");
        std::env::set_var("EMBEDDING_BATCH_SIZE", "1");
    }

    let temp_dir = tempfile::tempdir().unwrap();
    let data_dir = temp_dir.path().join("data");
    let docs_dir = temp_dir.path().join("docs");
    let log_dir = temp_dir.path().join("logs");
    std::fs::create_dir_all(&data_dir).unwrap();
    std::fs::create_dir_all(&docs_dir).unwrap();
    std::fs::create_dir_all(&log_dir).unwrap();

    unsafe {
        std::env::set_var("LOG_DIR", log_dir.to_str().unwrap());
    }

    let db_path = format!("sqlite:{}/jobs.db", data_dir.to_str().unwrap());
    let job_manager = Arc::new(JobManager::new(&db_path).await.unwrap());
    let rag_engine = Arc::new(RwLock::new(
        RagEngine::new(data_dir.to_str().unwrap(), &Config::default())
            .await
            .unwrap(),
    ));
    let (_job_tx, job_rx) = mpsc::channel(10);

    let job = job_manager
        .create_job(
            JobType::Reindex,
            Some(docs_dir.to_str().unwrap().to_string()),
            0,
        )
        .await
        .unwrap();
    job_manager.update_total(&job.job_id, 1).await.unwrap();
    job_manager.update_progress(&job.job_id, 1).await.unwrap();

    let supervisor = WorkerSupervisor::new(job_manager.clone(), rag_engine.clone(), job_rx);
    tokio::spawn(async move {
        supervisor.run().await;
    });

    let mut attempts = 0;
    loop {
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        let current_job = job_manager.get_job(&job.job_id).await.unwrap().unwrap();
        if current_job.status == JobStatus::Completed {
            assert!(
                current_job
                    .error
                    .as_deref()
                    .unwrap_or("")
                    .contains("Recovered on startup"),
                "Expected completion note in job.error"
            );
            break;
        }
        if current_job.status == JobStatus::Failed {
            panic!("Job unexpectedly failed: {:?}", current_job.error);
        }

        attempts += 1;
        if attempts > 150 {
            panic!("Timed out waiting for job to be marked completed: {current_job:?}");
        }
    }
}

#[tokio::test]
#[serial]
async fn test_reindex_resets_incompatible_index_to_avoid_mixed_dimensions() {
    let mock_server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/tags"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "models": [{ "name": "nomic-embed-text:latest" }]
        })))
        .mount(&mock_server)
        .await;

    // Canary + document embeddings return 768D (backend dim).
    Mock::given(method("POST"))
        .and(path("/api/embed"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "embedding": vec![0.1f32; 768]
        })))
        .mount(&mock_server)
        .await;

    unsafe {
        std::env::set_var("OLLAMA_URL", mock_server.uri());
        std::env::set_var("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text");
        std::env::set_var("EMBEDDING_BATCH_SIZE", "1");
    }

    let temp_dir = tempfile::tempdir().unwrap();
    let data_dir = temp_dir.path().join("data");
    let docs_dir = temp_dir.path().join("docs");
    let log_dir = temp_dir.path().join("logs");
    std::fs::create_dir_all(&data_dir).unwrap();
    std::fs::create_dir_all(&docs_dir).unwrap();
    std::fs::create_dir_all(&log_dir).unwrap();

    unsafe {
        std::env::set_var("LOG_DIR", log_dir.to_str().unwrap());
    }

    // Write a PDF that will be reindexed.
    let pdf_bytes = create_valid_pdf();
    std::fs::write(docs_dir.join("test.pdf"), &pdf_bytes).unwrap();

    let db_path = format!("sqlite:{}/jobs.db", data_dir.to_str().unwrap());
    let job_manager = Arc::new(JobManager::new(&db_path).await.unwrap());

    // Seed an incompatible 384D index state in SQLite (persisted dim).
    let index_store = rust_local_rag::SqliteIndexStore::new(&db_path).await.unwrap();
    index_store
        .upsert_document_atomic(
            "nomic-embed-text",
            384,
            false,
            "test.pdf",
            &sha256_hex(&pdf_bytes),
            &[rag_core::DocumentChunk {
                id: "old-chunk".to_string(),
                document_name: "test.pdf".to_string(),
                text: "old".to_string(),
                embedding: vec![0.0f32; 384],
                chunk_index: 0,
                page_number: 1,
                section: None,
                metadata: rag_core::ChunkMetadata {
                    token_count: 0,
                    overlap_with_previous: 0,
                    ..Default::default()
                },
                tags: std::collections::HashSet::new(),
                resolution: rag_core::Resolution::Chunk,
                parent_id: None,
            }],
        )
        .await
        .unwrap();

    let rag_engine = Arc::new(RwLock::new(
        RagEngine::new(data_dir.to_str().unwrap(), &Config::default())
            .await
            .unwrap(),
    ));
    assert!(
        rag_engine.read().await.needs_reindex(),
        "Expected needs_reindex due to dim mismatch"
    );

    let (job_tx, job_rx) = mpsc::channel(10);
    let supervisor = WorkerSupervisor::new(job_manager.clone(), rag_engine.clone(), job_rx);
    tokio::spawn(async move {
        supervisor.run().await;
    });

    let job = job_manager
        .create_job(JobType::Reindex, None, 0)
        .await
        .unwrap();
    job_tx
        .send(JobRequest::StartReindex {
            job_id: job.job_id.clone(),
            documents_dir: docs_dir.to_str().unwrap().to_string(),
        })
        .await
        .unwrap();

    let mut attempts = 0;
    loop {
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        let current_job = job_manager.get_job(&job.job_id).await.unwrap().unwrap();
        if current_job.status == JobStatus::Completed {
            break;
        }
        if current_job.status == JobStatus::Failed {
            panic!("Job failed: {:?}", current_job.error);
        }
        attempts += 1;
        if attempts > 200 {
            panic!("Job timed out. Status: {:?}", current_job.status);
        }
    }

    // After reindex, rag_models.embedding_dim should match backend (768),
    // and no 384D embeddings should remain.
    let index_store = rust_local_rag::SqliteIndexStore::new(&db_path).await.unwrap();
    let stored_dim: i64 =
        sqlx::query_scalar("SELECT embedding_dim FROM rag_models WHERE model_id = ?")
            .bind("nomic-embed-text")
            .fetch_one(&index_store.pool())
            .await
            .unwrap();
    assert_eq!(stored_dim, 768);

    let blob_lens: Vec<i64> =
        sqlx::query_scalar("SELECT LENGTH(embedding) FROM rag_chunks WHERE model_id = ?")
            .bind("nomic-embed-text")
            .fetch_all(&index_store.pool())
            .await
            .unwrap();
    assert!(!blob_lens.is_empty(), "Expected chunks after reindex");
    for len in blob_lens {
        assert_eq!(len, 768 * 4);
    }
}

#[derive(serde::Serialize)]
struct ReindexJobPayload {
    documents_dir: String,
    model_id: String,
    embedding_dim: usize,
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let hash = Sha256::digest(bytes);
    format!("{hash:x}")
}

#[tokio::test]
#[serial]
async fn test_resume_mid_reindex_completes_without_duplicate_chunks() {
    // 1. Setup Environment & Mock Ollama
    let mock_server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/tags"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "models": [{ "name": "nomic-embed-text:latest" }]
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

    unsafe {
        std::env::set_var("OLLAMA_URL", mock_server.uri());
        std::env::set_var("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text");
        std::env::set_var("EMBEDDING_BATCH_SIZE", "1");
    }

    let temp_dir = tempfile::tempdir().unwrap();
    let data_dir = temp_dir.path().join("data");
    let docs_dir = temp_dir.path().join("docs");
    let log_dir = temp_dir.path().join("logs");
    std::fs::create_dir_all(&data_dir).unwrap();
    std::fs::create_dir_all(&docs_dir).unwrap();
    std::fs::create_dir_all(&log_dir).unwrap();

    unsafe {
        std::env::set_var("LOG_DIR", log_dir.to_str().unwrap());
    }

    // 2. Write two PDFs
    let a_bytes = create_valid_pdf();
    std::fs::write(docs_dir.join("a.pdf"), &a_bytes).unwrap();
    std::fs::write(docs_dir.join("b.pdf"), create_valid_pdf()).unwrap();

    // 3. Seed SQLite with only a.pdf (simulate crash mid-job after a.pdf committed)
    let db_path = format!("sqlite:{}/jobs.db", data_dir.to_str().unwrap());
    let index_store = rust_local_rag::SqliteIndexStore::new(&db_path).await.unwrap();
    index_store
        .upsert_document_atomic(
            "nomic-embed-text",
            384,
            false,
            "a.pdf",
            &sha256_hex(&a_bytes),
            &[rag_core::DocumentChunk {
                id: "chunk-a-1".to_string(),
                document_name: "a.pdf".to_string(),
                text: "Hello World".to_string(),
                embedding: vec![0.0f32; 384],
                chunk_index: 0,
                page_number: 1,
                section: None,
                metadata: rag_core::ChunkMetadata {
                    token_count: 0,
                    overlap_with_previous: 0,
                    ..Default::default()
                },
                tags: std::collections::HashSet::new(),
                resolution: rag_core::Resolution::Chunk,
                parent_id: None,
            }],
        )
        .await
        .unwrap();

    // 4. Create a resumable job with structured payload (to be parsed on resume).
    let job_manager = Arc::new(JobManager::new(&db_path).await.unwrap());
    let payload = serde_json::to_string(&ReindexJobPayload {
        documents_dir: docs_dir.to_str().unwrap().to_string(),
        model_id: "nomic-embed-text".to_string(),
        embedding_dim: 384,
    })
    .unwrap();
    let job = job_manager
        .create_job(JobType::Reindex, Some(payload), 2)
        .await
        .unwrap();
    job_manager
        .update_status(&job.job_id, JobStatus::InProgress, None)
        .await
        .unwrap();
    job_manager.update_progress(&job.job_id, 1).await.unwrap();

    // 5. Start supervisor (simulates restart) and wait for completion.
    let rag_engine = Arc::new(RwLock::new(
        RagEngine::new(data_dir.to_str().unwrap(), &Config::default())
            .await
            .unwrap(),
    ));
    let (_job_tx, job_rx) = mpsc::channel(10);
    let supervisor = WorkerSupervisor::new(job_manager.clone(), rag_engine.clone(), job_rx);
    tokio::spawn(async move {
        supervisor.run().await;
    });

    let mut attempts = 0;
    loop {
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        let current_job = job_manager.get_job(&job.job_id).await.unwrap().unwrap();

        if current_job.status == JobStatus::Completed {
            break;
        }
        if current_job.status == JobStatus::Failed {
            panic!("Job failed: {:?}", current_job.error);
        }

        attempts += 1;
        if attempts > 200 {
            panic!("Job timed out. Status: {:?}", current_job.status);
        }
    }

    // 6. Verify both documents are present and per-document rows are not duplicated.
    let engine = rag_engine.read().await;
    let docs = engine.list_documents();
    assert_eq!(docs.len(), 2, "Should have indexed 2 documents");
    assert!(docs.contains(&"a.pdf".to_string()));
    assert!(docs.contains(&"b.pdf".to_string()));

    let index_store = rust_local_rag::SqliteIndexStore::new(&db_path).await.unwrap();
    let doc_count: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM rag_documents WHERE model_id = ?")
            .bind("nomic-embed-text")
            .fetch_one(&index_store.pool())
            .await
            .unwrap();
    assert_eq!(doc_count, 2);
}

#[tokio::test]
#[serial]
async fn test_resume_job_with_model_mismatch_is_marked_failed() {
    let mock_server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/tags"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "models": [{ "name": "nomic-embed-text:latest" }]
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

    unsafe {
        std::env::set_var("OLLAMA_URL", mock_server.uri());
        std::env::set_var("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text");
    }

    let temp_dir = tempfile::tempdir().unwrap();
    let data_dir = temp_dir.path().join("data");
    let docs_dir = temp_dir.path().join("docs");
    let log_dir = temp_dir.path().join("logs");
    std::fs::create_dir_all(&data_dir).unwrap();
    std::fs::create_dir_all(&docs_dir).unwrap();
    std::fs::create_dir_all(&log_dir).unwrap();

    unsafe {
        std::env::set_var("LOG_DIR", log_dir.to_str().unwrap());
    }

    let db_path = format!("sqlite:{}/jobs.db", data_dir.to_str().unwrap());
    let job_manager = Arc::new(JobManager::new(&db_path).await.unwrap());
    let rag_engine = Arc::new(RwLock::new(
        RagEngine::new(data_dir.to_str().unwrap(), &Config::default())
            .await
            .unwrap(),
    ));
    let (_job_tx, job_rx) = mpsc::channel(10);

    let payload = serde_json::to_string(&ReindexJobPayload {
        documents_dir: docs_dir.to_str().unwrap().to_string(),
        model_id: "some-other-model".to_string(),
        embedding_dim: 384,
    })
    .unwrap();
    let job = job_manager
        .create_job(JobType::Reindex, Some(payload), 0)
        .await
        .unwrap();
    job_manager
        .update_status(&job.job_id, JobStatus::InProgress, None)
        .await
        .unwrap();

    let supervisor = WorkerSupervisor::new(job_manager.clone(), rag_engine, job_rx);
    tokio::spawn(async move {
        supervisor.run().await;
    });

    let mut attempts = 0;
    loop {
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        let current_job = job_manager.get_job(&job.job_id).await.unwrap().unwrap();
        if current_job.status == JobStatus::Failed {
            let msg = current_job.error.as_deref().unwrap_or("");
            assert!(
                msg.contains("model") && msg.contains("reindex"),
                "Unexpected error message: {msg}"
            );
            break;
        }
        if current_job.status == JobStatus::Completed {
            panic!("Job unexpectedly completed");
        }

        attempts += 1;
        if attempts > 200 {
            panic!("Timed out waiting for job to be marked failed: {current_job:?}");
        }
    }
}
