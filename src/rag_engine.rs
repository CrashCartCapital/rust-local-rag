use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use uuid::Uuid;
use walkdir::WalkDir;

use crate::embeddings::EmbeddingService;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentChunk {
    pub id: String,
    pub document_name: String,
    pub text: String,
    pub embedding: Vec<f32>,
    pub chunk_index: usize,
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub text: String,
    pub score: f32,
    pub document: String,
    pub chunk_id: String,
}

pub struct RagEngine {
    chunks: HashMap<String, DocumentChunk>,
    embedding_service: EmbeddingService,
    data_dir: String,
    needs_reindex: bool,
    document_hashes: HashMap<String, String>,
}

impl RagEngine {
    pub async fn new(data_dir: &str) -> Result<Self> {
        let embedding_service = EmbeddingService::new().await?;

        let mut engine = Self {
            chunks: HashMap::new(),
            embedding_service,
            data_dir: data_dir.to_string(),
            needs_reindex: false,
            document_hashes: HashMap::new(),
        };

        if let Err(e) = engine.load_from_disk().await {
            tracing::warn!("Could not load existing data: {}", e);
        }

        Ok(engine)
    }

    pub fn needs_reindex(&self) -> bool {
        self.needs_reindex
    }

    pub fn embedding_model(&self) -> &str {
        self.embedding_service.model_name()
    }

    pub async fn add_document(&mut self, filename: &str, data: &[u8]) -> Result<usize> {
        tracing::info!("Processing document: {}", filename);

        let document_hash = Self::compute_document_hash(data);
        if let Some(existing_hash) = self.document_hashes.get(filename) {
            if existing_hash == &document_hash {
                tracing::info!(
                    "Document {} unchanged since last index. Skipping re-embedding.",
                    filename
                );
                return Ok(0);
            }

            tracing::info!("Document {} has changed. Refreshing embeddings.", filename);
        }

        let text = self.extract_pdf_text(data)?;
        if text.trim().is_empty() {
            return Err(anyhow::anyhow!("No text extracted from PDF"));
        }

        let chunks = self.chunk_text(&text, 500);
        tracing::info!("Created {} chunks for {}", chunks.len(), filename);

        let filtered_chunks: Vec<(usize, String)> = chunks
            .into_iter()
            .enumerate()
            .filter_map(|(i, chunk_text)| {
                if chunk_text.trim().len() < 10 {
                    None
                } else {
                    Some((i, chunk_text))
                }
            })
            .collect();

        if filtered_chunks.is_empty() {
            tracing::warn!(
                "Document {} produced no sizeable chunks after filtering. Removing any cached chunks for this file.",
                filename
            );
            self.chunks
                .retain(|_, chunk| chunk.document_name != filename);
            self.document_hashes
                .insert(filename.to_string(), document_hash);
            self.save_to_disk().await?;
            return Ok(0);
        }

        let chunk_texts: Vec<String> = filtered_chunks
            .iter()
            .map(|(_, text)| text.clone())
            .collect();

        tracing::debug!(
            "Generating embeddings for {} chunks from {} in a single request",
            chunk_texts.len(),
            filename
        );

        let embeddings = self.embedding_service.embed_texts(&chunk_texts).await?;

        if embeddings.len() != filtered_chunks.len() {
            return Err(anyhow::anyhow!(
                "Received {} embeddings for {} chunks in {}",
                embeddings.len(),
                filtered_chunks.len(),
                filename
            ));
        }

        self.chunks
            .retain(|_, chunk| chunk.document_name != filename);

        let mut chunk_count = 0;
        for ((i, chunk_text), embedding) in filtered_chunks.into_iter().zip(embeddings.into_iter()) {
            let chunk = DocumentChunk {
                id: Uuid::new_v4().to_string(),
                document_name: filename.to_string(),
                text: chunk_text,
                embedding,
                chunk_index: i,
            };

            self.chunks.insert(chunk.id.clone(), chunk);
            chunk_count += 1;
        }

        self.document_hashes
            .insert(filename.to_string(), document_hash);
        self.save_to_disk().await?;

        tracing::info!(
            "Successfully processed {} chunks for {}",
            chunk_count,
            filename
        );
        Ok(chunk_count)
    }

    pub async fn search(&self, query: &str, top_k: usize) -> Result<Vec<SearchResult>> {
        if self.chunks.is_empty() {
            return Ok(vec![]);
        }

        tracing::debug!("Searching for: '{}'", query);

        let query_embedding = self.embedding_service.get_query_embedding(query).await?;

        let mut scores: Vec<(f32, &DocumentChunk)> = self
            .chunks
            .values()
            .map(|chunk| {
                let similarity = cosine_similarity(&query_embedding, &chunk.embedding);
                (similarity, chunk)
            })
            .collect();

        scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        Ok(scores
            .into_iter()
            .take(top_k)
            .map(|(score, chunk)| SearchResult {
                text: chunk.text.clone(),
                score,
                document: chunk.document_name.clone(),
                chunk_id: chunk.id.clone(),
            })
            .collect())
    }

    pub fn list_documents(&self) -> Vec<String> {
        let mut docs: Vec<String> = self
            .chunks
            .values()
            .map(|chunk| chunk.document_name.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        docs.sort();
        docs
    }

    pub fn get_stats(&self) -> serde_json::Value {
        let doc_count = self.list_documents().len();
        let chunk_count = self.chunks.len();

        let status = if self.needs_reindex {
            "reindexing"
        } else {
            "ready"
        };

        serde_json::json!({
            "documents": doc_count,
            "chunks": chunk_count,
            "status": status
        })
    }

    pub async fn load_documents_from_dir(&mut self, dir: &str) -> Result<()> {
        let was_reindexing = self.needs_reindex;

        if was_reindexing {
            tracing::info!(
                "Reindexing documents in '{}' with embedding model '{}'",
                dir,
                self.embedding_service.model_name()
            );
        }

        for entry in WalkDir::new(dir).into_iter().filter_map(|e| e.ok()) {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("pdf") {
                let filename = path.file_name().unwrap().to_str().unwrap();

                match tokio::fs::read(&path).await {
                    Ok(data) => {
                        tracing::info!("Loading document: {}", filename);
                        match self.add_document(filename, &data).await {
                            Ok(chunk_count) => {
                                if chunk_count > 0 {
                                    tracing::info!(
                                        "Successfully processed {} with {} chunks",
                                        filename,
                                        chunk_count
                                    );
                                } else {
                                    tracing::info!(
                                        "{} is already up to date. No reindex needed.",
                                        filename
                                    );
                                }
                            }
                            Err(e) => {
                                tracing::warn!("Skipping {}: {}", filename, e);
                            }
                        }
                    }
                    Err(e) => {
                        tracing::error!("Failed to read {}: {}", filename, e);
                    }
                }
            }
        }

        if was_reindexing {
            self.needs_reindex = false;
            self.save_to_disk().await?;
            tracing::info!(
                "Reindexing complete. Indexed {} chunks across {} documents.",
                self.chunks.len(),
                self.list_documents().len()
            );
        }

        Ok(())
    }

    fn extract_pdf_text(&self, data: &[u8]) -> Result<String> {
        tracing::info!("Extracting PDF text using pdftotext system binary");
        self.extract_pdf_with_pdftotext(data)
    }

    fn extract_pdf_with_pdftotext(&self, data: &[u8]) -> Result<String> {
        use std::process::Command;

        let temp_dir = std::env::temp_dir();
        let temp_file = temp_dir.join(format!("temp_pdf_{}.pdf", std::process::id()));

        std::fs::write(&temp_file, data)
            .map_err(|e| anyhow::anyhow!("Failed to write temp PDF: {}", e))?;

        let output = Command::new("pdftotext")
            .arg("-layout")
            .arg("-enc")
            .arg("UTF-8")
            .arg(&temp_file)
            .arg("-")
            .output();
        let _ = std::fs::remove_file(&temp_file);

        match output {
            Ok(output) if output.status.success() => {
                let text = String::from_utf8_lossy(&output.stdout).to_string();
                let text_chars = text.chars().count();

                if text.trim().is_empty() {
                    tracing::warn!("pdftotext extracted 0 characters");
                    Err(anyhow::anyhow!("pdftotext produced no text output"))
                } else {
                    tracing::info!("✅ pdftotext extracted {} characters", text_chars);
                    Ok(text)
                }
            }
            Ok(output) => {
                let error_msg = String::from_utf8_lossy(&output.stderr);
                tracing::warn!("pdftotext failed with error: {}", error_msg);
                Err(anyhow::anyhow!("pdftotext failed: {}", error_msg))
            }
            Err(e) => {
                tracing::warn!("Failed to run pdftotext command: {}", e);
                Err(anyhow::anyhow!(
                    "pdftotext command failed: {} (is poppler installed?)",
                    e
                ))
            }
        }
    }

    fn chunk_text(&self, text: &str, chunk_size: usize) -> Vec<String> {
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut chunks = Vec::new();

        for chunk in words.chunks(chunk_size) {
            let chunk_text = chunk.join(" ");
            if !chunk_text.trim().is_empty() {
                chunks.push(chunk_text);
            }
        }

        chunks
    }

    async fn save_to_disk(&self) -> Result<()> {
        #[derive(Serialize)]
        struct PersistedState<'a> {
            version: u32,
            model: &'a str,
            chunks: &'a HashMap<String, DocumentChunk>,
            needs_reindex: bool,
            #[serde(default, skip_serializing_if = "HashMap::is_empty")]
            document_hashes: &'a HashMap<String, String>,
        }

        let path = format!("{}/chunks.json", self.data_dir);
        let state = PersistedState {
            version: 1,
            model: self.embedding_service.model_name(),
            chunks: &self.chunks,
            needs_reindex: self.needs_reindex,
            document_hashes: &self.document_hashes,
        };

        let data = serde_json::to_string_pretty(&state)?;
        tokio::fs::write(path, data).await?;
        tracing::debug!("Saved {} chunks to disk", self.chunks.len());
        Ok(())
    }

    async fn load_from_disk(&mut self) -> Result<()> {
        let path = format!("{}/chunks.json", self.data_dir);
        if tokio::fs::try_exists(&path).await? {
            let data = tokio::fs::read_to_string(path).await?;
            #[derive(Deserialize)]
            struct PersistedState {
                #[serde(rename = "version")]
                _version: u32,
                model: String,
                chunks: HashMap<String, DocumentChunk>,
                #[serde(default)]
                needs_reindex: bool,
                #[serde(default)]
                document_hashes: HashMap<String, String>,
            }

            match serde_json::from_str::<PersistedState>(&data) {
                Ok(PersistedState {
                    model,
                    chunks,
                    needs_reindex,
                    document_hashes,
                    ..
                }) => {
                    if model != self.embedding_service.model_name() {
                        tracing::warn!(
                            "Embedding model changed from '{}' to '{}'. Existing embeddings will be reindexed.",
                            model,
                            self.embedding_service.model_name()
                        );
                        self.chunks.clear();
                        self.needs_reindex = true;
                        self.save_to_disk().await?;
                    } else {
                        self.chunks = chunks;
                        self.needs_reindex = needs_reindex;
                        self.document_hashes = document_hashes;
                        // Do not filter document_hashes based on existing_docs.
                        // This preserves hashes for documents that have zero chunks due to filtering.
                        // If you need to clean up hashes for deleted documents, implement that logic separately.
                        if self.document_hashes.is_empty() && !self.chunks.is_empty() {
                            tracing::info!(
                                "No document fingerprints found in cache. Existing documents will be reindexed to initialize change detection."
                            );
                            self.needs_reindex = true;
                        }
                        tracing::info!("Loaded {} chunks from disk", self.chunks.len());
                    }
                }
                Err(_) => {
                    let legacy_chunks: HashMap<String, DocumentChunk> = serde_json::from_str(&data)
                        .context("Failed to parse legacy chunks.json")?;
                    if !legacy_chunks.is_empty() {
                        tracing::warn!(
                            "Existing embeddings were created before model tracking was added. Reindexing with model '{}' is required.",
                            self.embedding_service.model_name()
                        );
                        tracing::info!(
                            "Discarding {} legacy chunks so they can be regenerated.",
                            legacy_chunks.len()
                        );
                        self.needs_reindex = true;
                    }
                    self.chunks.clear();
                    self.document_hashes.clear();
                    self.save_to_disk().await?;
                }
            }
        }
        Ok(())
    }

    fn compute_document_hash(data: &[u8]) -> String {
        let hash = Sha256::digest(data);
        format!("{:x}", hash)
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}
