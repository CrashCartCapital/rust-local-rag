#![allow(clippy::collapsible_if)]
use anyhow::{Context, Result};
use chrono::Utc;
use futures::TryStreamExt;
use rag_core::PersistenceBackend;
use rag_core::{ChunkMetadata, DocumentChunk, EngineState, Resolution};
use sqlx::{Pool, Row, Sqlite};
use std::collections::{HashMap, HashSet};
use std::path::Path;

pub struct SqliteIndexStore {
    pool: Pool<Sqlite>,
}

impl SqliteIndexStore {
    pub async fn new(db_path: &str) -> Result<Self> {
        use sqlx::sqlite::{SqliteConnectOptions, SqliteJournalMode, SqliteSynchronous};
        use std::str::FromStr;

        let options = SqliteConnectOptions::from_str(db_path)?
            .busy_timeout(std::time::Duration::from_secs(30))
            .journal_mode(SqliteJournalMode::Wal)
            .synchronous(SqliteSynchronous::Normal)
            .foreign_keys(true)
            .create_if_missing(true);

        let pool = sqlx::SqlitePool::connect_with(options).await?;
        let store = Self { pool };
        store.ensure_schema().await?;
        Ok(store)
    }

    pub fn pool(&self) -> Pool<Sqlite> {
        self.pool.clone()
    }

    async fn ensure_schema(&self) -> Result<()> {
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS rag_models (
                model_id TEXT PRIMARY KEY NOT NULL,
                embedding_dim INTEGER NOT NULL,
                schema_version INTEGER NOT NULL,
                needs_reindex INTEGER NOT NULL DEFAULT 0,
                canary_embedding BLOB,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            )
            "#,
        )
        .execute(&self.pool)
        .await?;

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS rag_documents (
                model_id TEXT NOT NULL,
                document_name TEXT NOT NULL,
                document_hash TEXT NOT NULL,
                chunk_count INTEGER NOT NULL DEFAULT 0,
                updated_at INTEGER NOT NULL,
                PRIMARY KEY (model_id, document_name),
                FOREIGN KEY (model_id) REFERENCES rag_models(model_id) ON DELETE CASCADE
            )
            "#,
        )
        .execute(&self.pool)
        .await?;

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS rag_chunks (
                model_id TEXT NOT NULL,
                chunk_id TEXT NOT NULL,
                document_name TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                embedding BLOB NOT NULL,
                page_number INTEGER NOT NULL,
                section TEXT,
                metadata_json TEXT NOT NULL,
                tags_json TEXT NOT NULL,
                resolution TEXT NOT NULL,
                parent_id TEXT,
                PRIMARY KEY (model_id, chunk_id),
                FOREIGN KEY (model_id, document_name)
                    REFERENCES rag_documents(model_id, document_name)
                    ON DELETE CASCADE
            )
            "#,
        )
        .execute(&self.pool)
        .await?;

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_rag_chunks_model_doc ON rag_chunks(model_id, document_name)",
        )
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn load_model_state(&self, model_id: &str) -> Result<Option<EngineState>> {
        let model_row = sqlx::query(
            "SELECT embedding_dim, schema_version, needs_reindex FROM rag_models WHERE model_id = ?",
        )
        .bind(model_id)
        .fetch_optional(&self.pool)
        .await?;

        let Some(model_row) = model_row else {
            return Ok(None);
        };

        let embedding_dim: i64 = model_row.get("embedding_dim");
        let schema_version: i64 = model_row.get("schema_version");
        let needs_reindex: i64 = model_row.get("needs_reindex");
        let embedding_dim_usize: usize = embedding_dim
            .try_into()
            .context("rag_models.embedding_dim is not a valid usize")?;

        let mut document_hashes: HashMap<String, String> = HashMap::new();
        let mut docs_rows = sqlx::query(
            "SELECT document_name, document_hash FROM rag_documents WHERE model_id = ?",
        )
        .bind(model_id)
        .fetch(&self.pool);
        while let Some(row) = docs_rows.try_next().await? {
            let document_name: String = row.get("document_name");
            let document_hash: String = row.get("document_hash");
            document_hashes.insert(document_name, document_hash);
        }

        let mut chunks: HashMap<String, DocumentChunk> = HashMap::new();
        let mut chunk_rows = sqlx::query(
            r#"
            SELECT
                chunk_id,
                document_name,
                chunk_index,
                text,
                embedding,
                page_number,
                section,
                metadata_json,
                tags_json,
                resolution,
                parent_id
            FROM rag_chunks
            WHERE model_id = ?
            "#,
        )
        .bind(model_id)
        .fetch(&self.pool);

        while let Some(row) = chunk_rows.try_next().await? {
            let chunk_id: String = row.get("chunk_id");
            let document_name: String = row.get("document_name");
            let chunk_index: i64 = row.get("chunk_index");
            let text: String = row.get("text");
            let embedding_blob: Vec<u8> = row.get("embedding");
            let page_number: i64 = row.get("page_number");
            let section: Option<String> = row.get("section");
            let metadata_json: String = row.get("metadata_json");
            let tags_json: String = row.get("tags_json");
            let resolution_str: String = row.get("resolution");
            let parent_id: Option<String> = row.get("parent_id");

            let embedding = decode_f32_blob(&embedding_blob)
                .with_context(|| format!("Invalid embedding blob for chunk {chunk_id}"))?;
            if embedding.len() != embedding_dim_usize {
                return Err(anyhow::anyhow!(
                    "Embedding dimension mismatch for chunk {}: expected {} floats but found {}",
                    chunk_id,
                    embedding_dim_usize,
                    embedding.len()
                ));
            }

            let metadata: ChunkMetadata = serde_json::from_str(&metadata_json)
                .with_context(|| format!("Invalid metadata_json for chunk {chunk_id}"))?;

            let tags_vec: Vec<String> = serde_json::from_str(&tags_json)
                .with_context(|| format!("Invalid tags_json for chunk {chunk_id}"))?;
            let tags: HashSet<String> = tags_vec.into_iter().collect();

            let resolution = resolution_from_str(&resolution_str).with_context(|| {
                format!("Invalid resolution '{resolution_str}' for chunk {chunk_id}")
            })?;

            chunks.insert(
                chunk_id.clone(),
                DocumentChunk {
                    id: chunk_id,
                    document_name,
                    text,
                    embedding,
                    chunk_index: chunk_index as usize,
                    page_number: page_number as usize,
                    section,
                    metadata,
                    tags,
                    resolution,
                    parent_id,
                },
            );
        }

        let mut state = EngineState::new(model_id, embedding_dim_usize);
        state.schema_version = schema_version as u32;
        state.needs_reindex = needs_reindex != 0;
        state.document_hashes = document_hashes;
        state.chunks = chunks;

        Ok(Some(state))
    }

    pub async fn upsert_document_atomic(
        &self,
        model_id: &str,
        embedding_dim: usize,
        needs_reindex: bool,
        document_name: &str,
        document_hash: &str,
        chunks: &[DocumentChunk],
    ) -> Result<()> {
        for chunk in chunks {
            if chunk.embedding.len() != embedding_dim {
                return Err(anyhow::anyhow!(
                    "Embedding dimension mismatch for chunk {}: expected {} floats but found {}",
                    chunk.id,
                    embedding_dim,
                    chunk.embedding.len()
                ));
            }
        }

        let now = Utc::now().timestamp();
        let mut tx = self.pool.begin().await?;

        sqlx::query(
            r#"
            INSERT INTO rag_models (model_id, embedding_dim, schema_version, needs_reindex, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(model_id) DO UPDATE SET
                schema_version = excluded.schema_version,
                needs_reindex = excluded.needs_reindex,
                updated_at = excluded.updated_at
            "#,
        )
        .bind(model_id)
        .bind(embedding_dim as i64)
        .bind(rag_core::persistence::INDEX_VERSION as i64)
        .bind(if needs_reindex { 1_i64 } else { 0_i64 })
        .bind(now)
        .bind(now)
        .execute(&mut *tx)
        .await?;

        let existing_dim: i64 =
            sqlx::query_scalar("SELECT embedding_dim FROM rag_models WHERE model_id = ?")
                .bind(model_id)
                .fetch_one(&mut *tx)
                .await?;
        let existing_usize: usize = existing_dim
            .try_into()
            .context("rag_models.embedding_dim is not a valid usize")?;
        if existing_usize != embedding_dim {
            let doc_count: i64 =
                sqlx::query_scalar("SELECT COUNT(*) FROM rag_documents WHERE model_id = ?")
                    .bind(model_id)
                    .fetch_one(&mut *tx)
                    .await?;
            if doc_count > 0 {
                return Err(anyhow::anyhow!(
                    "Embedding dimension changed for model_id '{}': stored {} vs requested {}. Reset/reindex required before upserts.",
                    model_id,
                    existing_usize,
                    embedding_dim
                ));
            }

            sqlx::query(
                "UPDATE rag_models SET embedding_dim = ?, updated_at = ? WHERE model_id = ?",
            )
            .bind(embedding_dim as i64)
            .bind(now)
            .bind(model_id)
            .execute(&mut *tx)
            .await?;
        }

        sqlx::query("DELETE FROM rag_documents WHERE model_id = ? AND document_name = ?")
            .bind(model_id)
            .bind(document_name)
            .execute(&mut *tx)
            .await?;

        sqlx::query(
            r#"
            INSERT INTO rag_documents (model_id, document_name, document_hash, chunk_count, updated_at)
            VALUES (?, ?, ?, ?, ?)
            "#,
        )
        .bind(model_id)
        .bind(document_name)
        .bind(document_hash)
        .bind(chunks.len() as i64)
        .bind(now)
        .execute(&mut *tx)
        .await?;

        for chunk in chunks {
            let embedding_blob = encode_f32_blob(&chunk.embedding);
            let metadata_json = serde_json::to_string(&chunk.metadata)?;

            let mut tags_vec: Vec<String> = chunk.tags.iter().cloned().collect();
            tags_vec.sort();
            let tags_json = serde_json::to_string(&tags_vec)?;

            sqlx::query(
                r#"
                INSERT INTO rag_chunks (
                    model_id,
                    chunk_id,
                    document_name,
                    chunk_index,
                    text,
                    embedding,
                    page_number,
                    section,
                    metadata_json,
                    tags_json,
                    resolution,
                    parent_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                "#,
            )
            .bind(model_id)
            .bind(&chunk.id)
            .bind(&chunk.document_name)
            .bind(chunk.chunk_index as i64)
            .bind(&chunk.text)
            .bind(embedding_blob)
            .bind(chunk.page_number as i64)
            .bind(&chunk.section)
            .bind(metadata_json)
            .bind(tags_json)
            .bind(resolution_to_str(chunk.resolution))
            .bind(&chunk.parent_id)
            .execute(&mut *tx)
            .await?;
        }

        tx.commit().await?;
        Ok(())
    }

    pub async fn delete_document_atomic(
        &self,
        model_id: &str,
        document_name: &str,
    ) -> Result<usize> {
        let mut tx = self.pool.begin().await?;

        let chunk_count: Option<i64> = sqlx::query_scalar(
            "SELECT chunk_count FROM rag_documents WHERE model_id = ? AND document_name = ?",
        )
        .bind(model_id)
        .bind(document_name)
        .fetch_optional(&mut *tx)
        .await?;

        sqlx::query("DELETE FROM rag_documents WHERE model_id = ? AND document_name = ?")
            .bind(model_id)
            .bind(document_name)
            .execute(&mut *tx)
            .await?;

        tx.commit().await?;
        Ok(chunk_count.unwrap_or(0) as usize)
    }

    pub async fn set_model_needs_reindex(
        &self,
        model_id: &str,
        embedding_dim: usize,
        needs_reindex: bool,
    ) -> Result<()> {
        let now = Utc::now().timestamp();
        let mut tx = self.pool.begin().await?;

        sqlx::query(
            r#"
            INSERT INTO rag_models (model_id, embedding_dim, schema_version, needs_reindex, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(model_id) DO UPDATE SET
                schema_version = excluded.schema_version,
                needs_reindex = excluded.needs_reindex,
                updated_at = excluded.updated_at
            "#,
        )
        .bind(model_id)
        .bind(embedding_dim as i64)
        .bind(rag_core::persistence::INDEX_VERSION as i64)
        .bind(if needs_reindex { 1_i64 } else { 0_i64 })
        .bind(now)
        .bind(now)
        .execute(&mut *tx)
        .await?;

        let existing_dim: i64 =
            sqlx::query_scalar("SELECT embedding_dim FROM rag_models WHERE model_id = ?")
                .bind(model_id)
                .fetch_one(&mut *tx)
                .await?;
        let existing_usize: usize = existing_dim
            .try_into()
            .context("rag_models.embedding_dim is not a valid usize")?;
        if existing_usize != embedding_dim {
            let doc_count: i64 =
                sqlx::query_scalar("SELECT COUNT(*) FROM rag_documents WHERE model_id = ?")
                    .bind(model_id)
                    .fetch_one(&mut *tx)
                    .await?;
            if doc_count == 0 {
                sqlx::query(
                    "UPDATE rag_models SET embedding_dim = ?, updated_at = ? WHERE model_id = ?",
                )
                .bind(embedding_dim as i64)
                .bind(now)
                .bind(model_id)
                .execute(&mut *tx)
                .await?;
            }
        }

        tx.commit().await?;
        Ok(())
    }

    pub async fn reset_model_state_atomic(
        &self,
        model_id: &str,
        embedding_dim: usize,
        needs_reindex: bool,
    ) -> Result<()> {
        let now = Utc::now().timestamp();
        let mut tx = self.pool.begin().await?;

        sqlx::query(
            r#"
            INSERT INTO rag_models (model_id, embedding_dim, schema_version, needs_reindex, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(model_id) DO UPDATE SET
                embedding_dim = excluded.embedding_dim,
                schema_version = excluded.schema_version,
                needs_reindex = excluded.needs_reindex,
                updated_at = excluded.updated_at
            "#,
        )
        .bind(model_id)
        .bind(embedding_dim as i64)
        .bind(rag_core::persistence::INDEX_VERSION as i64)
        .bind(if needs_reindex { 1_i64 } else { 0_i64 })
        .bind(now)
        .bind(now)
        .execute(&mut *tx)
        .await?;

        sqlx::query("DELETE FROM rag_documents WHERE model_id = ?")
            .bind(model_id)
            .execute(&mut *tx)
            .await?;

        tx.commit().await?;
        Ok(())
    }

    pub async fn migrate_json_if_needed(
        &self,
        data_dir: &Path,
        model_id: &str,
        embedding_dim: usize,
    ) -> Result<()> {
        let already = sqlx::query("SELECT 1 FROM rag_models WHERE model_id = ?")
            .bind(model_id)
            .fetch_optional(&self.pool)
            .await?
            .is_some();
        if already {
            return Ok(());
        }

        let backend = rag_core::JsonFileBackend::new(data_dir.to_path_buf(), model_id.to_string());
        let json_path = backend.index_path();
        let legacy_path = backend.legacy_path();
        if !json_path.exists() && !legacy_path.exists() {
            return Ok(());
        }

        let state = tokio::task::spawn_blocking(move || backend.load())
            .await
            .context("json load task failed")??;

        let Some(mut state) = state else {
            // JSON exists but couldn't be loaded; preserve behavior by marking for reindex.
            self.set_model_needs_reindex(model_id, embedding_dim, true)
                .await?;
            return Ok(());
        };

        if state.embedding_dim == 0 {
            if let Some(first) = state.chunks.values().next() {
                state.embedding_dim = first.embedding.len();
            }
        }
        if state.embedding_dim == 0 {
            self.set_model_needs_reindex(model_id, embedding_dim, true)
                .await?;
            return Ok(());
        }
        if state
            .chunks
            .values()
            .any(|chunk| chunk.embedding.len() != state.embedding_dim)
        {
            self.set_model_needs_reindex(model_id, embedding_dim, true)
                .await?;
            return Ok(());
        }

        let mut doc_counts: HashMap<String, usize> = HashMap::new();
        for chunk in state.chunks.values() {
            *doc_counts.entry(chunk.document_name.clone()).or_default() += 1;
        }

        let now = Utc::now().timestamp();
        let mut tx = self.pool.begin().await?;

        sqlx::query(
            r#"
            INSERT INTO rag_models (model_id, embedding_dim, schema_version, needs_reindex, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            "#,
        )
        .bind(model_id)
        .bind(state.embedding_dim as i64)
        .bind(state.schema_version as i64)
        .bind(if state.needs_reindex { 1_i64 } else { 0_i64 })
        .bind(now)
        .bind(now)
        .execute(&mut *tx)
        .await?;

        for (doc, hash) in &state.document_hashes {
            let count = doc_counts.get(doc).copied().unwrap_or(0);
            sqlx::query(
                r#"
                INSERT INTO rag_documents (model_id, document_name, document_hash, chunk_count, updated_at)
                VALUES (?, ?, ?, ?, ?)
                "#,
            )
            .bind(model_id)
            .bind(doc)
            .bind(hash)
            .bind(count as i64)
            .bind(now)
            .execute(&mut *tx)
            .await?;
        }

        for chunk in state.chunks.values() {
            let embedding_blob = encode_f32_blob(&chunk.embedding);
            let metadata_json = serde_json::to_string(&chunk.metadata)?;

            let mut tags_vec: Vec<String> = chunk.tags.iter().cloned().collect();
            tags_vec.sort();
            let tags_json = serde_json::to_string(&tags_vec)?;

            sqlx::query(
                r#"
                INSERT INTO rag_chunks (
                    model_id,
                    chunk_id,
                    document_name,
                    chunk_index,
                    text,
                    embedding,
                    page_number,
                    section,
                    metadata_json,
                    tags_json,
                    resolution,
                    parent_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                "#,
            )
            .bind(model_id)
            .bind(&chunk.id)
            .bind(&chunk.document_name)
            .bind(chunk.chunk_index as i64)
            .bind(&chunk.text)
            .bind(embedding_blob)
            .bind(chunk.page_number as i64)
            .bind(&chunk.section)
            .bind(metadata_json)
            .bind(tags_json)
            .bind(resolution_to_str(chunk.resolution))
            .bind(&chunk.parent_id)
            .execute(&mut *tx)
            .await?;
        }

        tx.commit().await?;

        // Best-effort backup rename
        let backup_name = format!(
            "{}.migrated.bak",
            json_path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("chunks.json")
        );
        let backup_path = json_path.with_file_name(backup_name);
        if json_path.exists() && !backup_path.exists() {
            let from = json_path.clone();
            let to = backup_path.clone();
            let _ = tokio::task::spawn_blocking(move || std::fs::rename(&from, &to)).await;
        } else if legacy_path.exists() {
            let legacy_backup = legacy_path.with_extension("json.migrated.bak");
            if !legacy_backup.exists() {
                let from = legacy_path.clone();
                let to = legacy_backup.clone();
                let _ = tokio::task::spawn_blocking(move || std::fs::rename(&from, &to)).await;
            }
        }

        Ok(())
    }
}

fn encode_f32_blob(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn decode_f32_blob(bytes: &[u8]) -> Result<Vec<f32>> {
    if !bytes.len().is_multiple_of(4) {
        return Err(anyhow::anyhow!(
            "Embedding blob length {} is not divisible by 4",
            bytes.len()
        ));
    }
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

fn resolution_to_str(resolution: Resolution) -> &'static str {
    match resolution {
        Resolution::Document => "document",
        Resolution::Section => "section",
        Resolution::Chunk => "chunk",
    }
}

fn resolution_from_str(s: &str) -> Option<Resolution> {
    match s {
        "document" => Some(Resolution::Document),
        "section" => Some(Resolution::Section),
        "chunk" => Some(Resolution::Chunk),
        _ => None,
    }
}
