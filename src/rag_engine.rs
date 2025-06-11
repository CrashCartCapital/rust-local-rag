use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
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
}

impl RagEngine {
    pub async fn new(data_dir: &str) -> Result<Self> {
        let embedding_service = EmbeddingService::new().await?;

        let mut engine = Self {
            chunks: HashMap::new(),
            embedding_service,
            data_dir: data_dir.to_string(),
        };

        if let Err(e) = engine.load_from_disk().await {
            tracing::warn!("Could not load existing data: {}", e);
        }

        Ok(engine)
    }

    pub async fn add_document(&mut self, filename: &str, data: &[u8]) -> Result<usize> {
        tracing::info!("Processing document: {}", filename);

        let text = self.extract_pdf_text(data)?;
        if text.trim().is_empty() {
            return Err(anyhow::anyhow!("No text extracted from PDF"));
        }

        let chunks = self.chunk_text(&text, 500);
        tracing::info!("Created {} chunks for {}", chunks.len(), filename);

        self.chunks
            .retain(|_, chunk| chunk.document_name != filename);

        let mut chunk_count = 0;
        for (i, chunk_text) in chunks.into_iter().enumerate() {
            if chunk_text.trim().len() < 10 {
                continue;
            }

            tracing::debug!("Generating embedding for chunk {} of {}", i + 1, filename);
            let embedding = self.embedding_service.get_embedding(&chunk_text).await?;

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

        let query_embedding = self.embedding_service.get_embedding(query).await?;

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

        serde_json::json!({
            "documents": doc_count,
            "chunks": chunk_count,
            "status": "ready"
        })
    }

    pub async fn load_documents_from_dir(&mut self, dir: &str) -> Result<()> {
        for entry in WalkDir::new(dir).into_iter().filter_map(|e| e.ok()) {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("pdf") {
                let filename = path.file_name().unwrap().to_str().unwrap();

                if self.chunks.values().any(|c| c.document_name == filename) {
                    tracing::info!("Document {} already processed, skipping", filename);
                    continue;
                }

                match tokio::fs::read(&path).await {
                    Ok(data) => {
                        tracing::info!("Loading document: {}", filename);
                        match self.add_document(filename, &data).await {
                            Ok(chunk_count) => {
                                tracing::info!(
                                    "Successfully processed {} with {} chunks",
                                    filename,
                                    chunk_count
                                );
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
        let path = format!("{}/chunks.json", self.data_dir);
        let data = serde_json::to_string_pretty(&self.chunks)?;
        tokio::fs::write(path, data).await?;
        tracing::debug!("Saved {} chunks to disk", self.chunks.len());
        Ok(())
    }

    async fn load_from_disk(&mut self) -> Result<()> {
        let path = format!("{}/chunks.json", self.data_dir);
        if tokio::fs::try_exists(&path).await? {
            let data = tokio::fs::read_to_string(path).await?;
            self.chunks = serde_json::from_str(&data)?;
            tracing::info!("Loaded {} chunks from disk", self.chunks.len());
        }
        Ok(())
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
