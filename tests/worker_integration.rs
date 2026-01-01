use lopdf::content::{Content, Operation};
use lopdf::{Dictionary, Document, Object, Stream};
use rust_local_rag::job_manager::{JobStatus, JobType};
use rust_local_rag::{JobManager, JobRequest, RagEngine, WorkerSupervisor};
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
        RagEngine::new(data_dir.to_str().unwrap()).await.unwrap(),
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
