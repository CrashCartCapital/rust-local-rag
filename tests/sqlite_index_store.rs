use rag_core::{ChunkMetadata, DocumentChunk, Resolution};
use rust_local_rag::job_manager::{JobStatus, JobType};
use rust_local_rag::{JobManager, SqliteIndexStore};
use sqlx::Row;
use std::collections::HashSet;

fn make_chunk(
    chunk_id: &str,
    document_name: &str,
    text: &str,
    embedding_dim: usize,
    chunk_index: usize,
) -> DocumentChunk {
    DocumentChunk {
        id: chunk_id.to_string(),
        document_name: document_name.to_string(),
        text: text.to_string(),
        embedding: vec![0.0f32; embedding_dim],
        chunk_index,
        page_number: 1,
        section: None,
        metadata: ChunkMetadata {
            token_count: 0,
            overlap_with_previous: 0,
            ..Default::default()
        },
        tags: HashSet::new(),
        resolution: Resolution::Chunk,
        parent_id: None,
    }
}

#[tokio::test]
async fn test_upsert_document_atomic_rolls_back_on_error_and_preserves_previous_state() {
    let temp_dir = tempfile::tempdir().unwrap();
    let db_path = format!("sqlite:{}/jobs.db", temp_dir.path().display());
    let store = SqliteIndexStore::new(&db_path).await.unwrap();

    let model_id = "model-a";
    let embedding_dim = 3;
    let doc = "doc.pdf";

    // Seed with a valid document.
    let initial_chunks = vec![make_chunk("chunk-1", doc, "hello", embedding_dim, 0)];
    store
        .upsert_document_atomic(model_id, embedding_dim, false, doc, "hash1", &initial_chunks)
        .await
        .unwrap();

    // Attempt an update that will fail mid-transaction due to duplicate chunk_id.
    let failing_chunks = vec![
        make_chunk("dup", doc, "a", embedding_dim, 0),
        make_chunk("dup", doc, "b", embedding_dim, 1),
    ];
    let err = store
        .upsert_document_atomic(model_id, embedding_dim, false, doc, "hash2", &failing_chunks)
        .await
        .unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("UNIQUE") || msg.contains("constraint") || msg.contains("PRIMARY KEY"),
        "Unexpected error: {msg}"
    );

    // Verify previous state is intact (rollback worked).
    let row = sqlx::query(
        "SELECT document_hash, chunk_count FROM rag_documents WHERE model_id = ? AND document_name = ?",
    )
    .bind(model_id)
    .bind(doc)
    .fetch_one(&store.pool())
    .await
    .unwrap();
    let document_hash: String = row.get("document_hash");
    let chunk_count: i64 = row.get("chunk_count");
    assert_eq!(document_hash, "hash1");
    assert_eq!(chunk_count, 1);

    let chunk_ids: Vec<String> = sqlx::query_scalar(
        "SELECT chunk_id FROM rag_chunks WHERE model_id = ? AND document_name = ? ORDER BY chunk_index",
    )
    .bind(model_id)
    .bind(doc)
    .fetch_all(&store.pool())
    .await
    .unwrap();
    assert_eq!(chunk_ids, vec!["chunk-1".to_string()]);
}

#[tokio::test]
async fn test_model_isolation_loads_only_requested_model() {
    let temp_dir = tempfile::tempdir().unwrap();
    let db_path = format!("sqlite:{}/jobs.db", temp_dir.path().display());
    let store = SqliteIndexStore::new(&db_path).await.unwrap();

    let dim = 4;
    store
        .upsert_document_atomic(
            "model-a",
            dim,
            false,
            "a.pdf",
            "hash-a",
            &[make_chunk("a-1", "a.pdf", "a", dim, 0)],
        )
        .await
        .unwrap();
    store
        .upsert_document_atomic(
            "model-b",
            dim,
            false,
            "b.pdf",
            "hash-b",
            &[make_chunk("b-1", "b.pdf", "b", dim, 0)],
        )
        .await
        .unwrap();

    let state_a = store.load_model_state("model-a").await.unwrap().unwrap();
    assert_eq!(state_a.document_hashes.len(), 1);
    assert!(state_a.document_hashes.contains_key("a.pdf"));
    assert!(!state_a.document_hashes.contains_key("b.pdf"));

    let state_b = store.load_model_state("model-b").await.unwrap().unwrap();
    assert_eq!(state_b.document_hashes.len(), 1);
    assert!(state_b.document_hashes.contains_key("b.pdf"));
    assert!(!state_b.document_hashes.contains_key("a.pdf"));
}

#[tokio::test]
async fn test_delete_document_atomic_cascades_chunk_rows() {
    let temp_dir = tempfile::tempdir().unwrap();
    let db_path = format!("sqlite:{}/jobs.db", temp_dir.path().display());
    let store = SqliteIndexStore::new(&db_path).await.unwrap();

    let model_id = "model-a";
    let embedding_dim = 5;
    let doc = "doc.pdf";
    let chunks = vec![
        make_chunk("chunk-1", doc, "a", embedding_dim, 0),
        make_chunk("chunk-2", doc, "b", embedding_dim, 1),
    ];
    store
        .upsert_document_atomic(model_id, embedding_dim, false, doc, "hash", &chunks)
        .await
        .unwrap();

    let before: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM rag_chunks WHERE model_id = ? AND document_name = ?",
    )
    .bind(model_id)
    .bind(doc)
    .fetch_one(&store.pool())
    .await
    .unwrap();
    assert_eq!(before, 2);

    store.delete_document_atomic(model_id, doc).await.unwrap();

    let docs_left: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM rag_documents WHERE model_id = ? AND document_name = ?",
    )
    .bind(model_id)
    .bind(doc)
    .fetch_one(&store.pool())
    .await
    .unwrap();
    assert_eq!(docs_left, 0);

    let chunks_left: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM rag_chunks WHERE model_id = ? AND document_name = ?",
    )
    .bind(model_id)
    .bind(doc)
    .fetch_one(&store.pool())
    .await
    .unwrap();
    assert_eq!(chunks_left, 0);
}

#[tokio::test]
async fn test_upsert_rejects_embedding_dim_mismatch() {
    let temp_dir = tempfile::tempdir().unwrap();
    let db_path = format!("sqlite:{}/jobs.db", temp_dir.path().display());
    let store = SqliteIndexStore::new(&db_path).await.unwrap();

    let model_id = "model-a";
    let doc = "doc.pdf";
    let embedding_dim = 8;

    // Create a chunk with the wrong embedding length (7 instead of 8).
    let mut chunk = make_chunk("chunk-1", doc, "x", embedding_dim - 1, 0);
    chunk.embedding = vec![0.0f32; embedding_dim - 1];

    let err = store
        .upsert_document_atomic(model_id, embedding_dim, false, doc, "hash", &[chunk])
        .await
        .unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("Embedding dimension mismatch"),
        "Unexpected error: {msg}"
    );

    let doc_count: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM rag_documents WHERE model_id = ?")
            .bind(model_id)
            .fetch_one(&store.pool())
            .await
            .unwrap();
    assert_eq!(doc_count, 0);
}

#[tokio::test]
async fn test_json_migration_imports_and_is_idempotent_and_creates_backup() {
    use rag_core::{EngineState, JsonFileBackend, PersistenceBackend};

    let temp_dir = tempfile::tempdir().unwrap();
    let data_dir = temp_dir.path();
    let db_path = format!("sqlite:{}/jobs.db", data_dir.display());
    let store = SqliteIndexStore::new(&db_path).await.unwrap();

    let model_id = "model-a";
    let embedding_dim = 6;

    // Create a JSON index using rag-core's own writer (ensures valid format).
    let mut state = EngineState::new(model_id, embedding_dim);
    state.document_hashes.insert("a.pdf".to_string(), "hash-a".to_string());
    state.document_hashes.insert("b.pdf".to_string(), "hash-b".to_string());
    state.chunks.insert(
        "a-1".to_string(),
        make_chunk("a-1", "a.pdf", "hello", embedding_dim, 0),
    );
    state.chunks.insert(
        "b-1".to_string(),
        make_chunk("b-1", "b.pdf", "world", embedding_dim, 0),
    );

    let backend = JsonFileBackend::new(data_dir.to_path_buf(), model_id.to_string());
    let json_path = backend.index_path();
    backend.save(&state).unwrap();
    assert!(json_path.exists());

    store
        .migrate_json_if_needed(data_dir, model_id, embedding_dim)
        .await
        .unwrap();

    // Backup created and original JSON removed.
    let backup_path = json_path.with_file_name(format!(
        "{}.migrated.bak",
        json_path.file_name().and_then(|s| s.to_str()).unwrap()
    ));
    assert!(!json_path.exists(), "JSON should be renamed after migration");
    assert!(backup_path.exists(), "Backup should exist: {:?}", backup_path);

    // Counts match imported state.
    let doc_count: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM rag_documents WHERE model_id = ?")
            .bind(model_id)
            .fetch_one(&store.pool())
            .await
            .unwrap();
    let chunk_count: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM rag_chunks WHERE model_id = ?")
            .bind(model_id)
            .fetch_one(&store.pool())
            .await
            .unwrap();
    assert_eq!(doc_count, 2);
    assert_eq!(chunk_count, 2);

    // Idempotent: running again is a no-op (marker is rag_models row).
    store
        .migrate_json_if_needed(data_dir, model_id, embedding_dim)
        .await
        .unwrap();
    let doc_count2: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM rag_documents WHERE model_id = ?")
            .bind(model_id)
            .fetch_one(&store.pool())
            .await
            .unwrap();
    let chunk_count2: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM rag_chunks WHERE model_id = ?")
            .bind(model_id)
            .fetch_one(&store.pool())
            .await
            .unwrap();
    assert_eq!(doc_count2, 2);
    assert_eq!(chunk_count2, 2);
}

#[tokio::test]
async fn test_json_migration_marks_needs_reindex_when_json_is_unreadable() {
    use rag_core::JsonFileBackend;

    let temp_dir = tempfile::tempdir().unwrap();
    let data_dir = temp_dir.path();
    let db_path = format!("sqlite:{}/jobs.db", data_dir.display());
    let store = SqliteIndexStore::new(&db_path).await.unwrap();

    let model_id = "model-a";
    let embedding_dim = 12;

    let backend = JsonFileBackend::new(data_dir.to_path_buf(), model_id.to_string());
    let json_path = backend.index_path();
    std::fs::write(&json_path, "{ not valid json").unwrap();

    store
        .migrate_json_if_needed(data_dir, model_id, embedding_dim)
        .await
        .unwrap();

    let state = store.load_model_state(model_id).await.unwrap().unwrap();
    assert!(state.needs_reindex);
    assert_eq!(state.embedding_dim, embedding_dim);
    assert!(state.document_hashes.is_empty());
    assert!(state.chunks.is_empty());
}

#[tokio::test]
async fn test_sqlite_busy_timeout_prevents_lock_errors_under_concurrent_writes() {
    use tokio::time::{timeout, Duration};

    let temp_dir = tempfile::tempdir().unwrap();
    let db_path = format!("sqlite:{}/jobs.db", temp_dir.path().display());

    let store = SqliteIndexStore::new(&db_path).await.unwrap();
    let job_manager = JobManager::new(&db_path).await.unwrap();

    let job = job_manager
        .create_job(JobType::Reindex, Some("/tmp".to_string()), 50)
        .await
        .unwrap();
    job_manager
        .update_status(&job.job_id, JobStatus::InProgress, None)
        .await
        .unwrap();

    let writer = {
        let store = store;
        async move {
            let dim = 8;
            for i in 0..25usize {
                let doc = format!("doc-{i}.pdf");
                let chunk_id = format!("chunk-{i}");
                let chunks = vec![make_chunk(&chunk_id, &doc, "x", dim, 0)];
                store
                    .upsert_document_atomic("model-a", dim, false, &doc, "hash", &chunks)
                    .await
                    .unwrap();
            }
        }
    };

    let job_updates = async {
        for i in 0..25i64 {
            job_manager.update_progress(&job.job_id, i).await.unwrap();
        }
    };

    timeout(Duration::from_secs(10), async move {
        tokio::join!(writer, job_updates);
    })
    .await
    .expect("concurrent writes should not time out");
}
