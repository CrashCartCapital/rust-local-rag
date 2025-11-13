use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use uuid::Uuid;
use walkdir::WalkDir;

use crate::{
    embeddings::EmbeddingService,
    reranker::{RerankerCandidate, RerankerService},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentChunk {
    pub id: String,
    pub document_name: String,
    pub text: String,
    pub embedding: Vec<f32>,
    pub chunk_index: usize,
    #[serde(default = "default_page_number")]
    pub page_number: usize,
    #[serde(default)]
    pub section: Option<String>,
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub text: String,
    pub score: f32,
    pub document: String,
    pub chunk_id: String,
    pub chunk_index: usize,
    pub page_number: usize,
    pub section: Option<String>,
}

pub struct RagEngine {
    chunks: HashMap<String, DocumentChunk>,
    embedding_service: EmbeddingService,
    data_dir: String,
    needs_reindex: bool,
    reranker: RerankerService,
}

#[derive(Debug, Clone)]
struct ChunkFragment {
    text: String,
    page_number: usize,
    section: Option<String>,
}

#[derive(Debug, Clone)]
struct SearchCandidate {
    chunk_id: String,
    document: String,
    text: String,
    page_number: usize,
    section: Option<String>,
    chunk_index: usize,
    initial_score: f32,
}

impl RagEngine {
    pub async fn new(data_dir: &str) -> Result<Self> {
        let embedding_service = EmbeddingService::new().await?;
        let reranker = RerankerService::new().await?;

        let mut engine = Self {
            chunks: HashMap::new(),
            embedding_service,
            data_dir: data_dir.to_string(),
            needs_reindex: false,
            reranker,
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

        let text = self.extract_pdf_text(data)?;
        if text.trim().is_empty() {
            return Err(anyhow::anyhow!("No text extracted from PDF"));
        }

        let chunks = self.chunk_text(&text, 500);
        tracing::info!("Created {} chunks for {}", chunks.len(), filename);

        self.chunks
            .retain(|_, chunk| chunk.document_name != filename);

        let mut chunk_count = 0;
        for (i, fragment) in chunks.into_iter().enumerate() {
            if fragment.text.trim().len() < 10 {
                continue;
            }

            tracing::debug!("Generating embedding for chunk {} of {}", i + 1, filename);
            let embedding = self.embedding_service.get_embedding(&fragment.text).await?;

            let chunk = DocumentChunk {
                id: Uuid::new_v4().to_string(),
                document_name: filename.to_string(),
                text: fragment.text,
                embedding,
                chunk_index: i,
                page_number: fragment.page_number,
                section: fragment.section,
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

        let top_k = top_k.max(1);
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

        scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let initial_k = scores.len().min(top_k.saturating_mul(3).max(top_k));

        let mut candidates: Vec<SearchCandidate> = scores
            .into_iter()
            .take(initial_k)
            .map(|(score, chunk)| SearchCandidate {
                chunk_id: chunk.id.clone(),
                document: chunk.document_name.clone(),
                text: chunk.text.clone(),
                page_number: chunk.page_number,
                section: chunk.section.clone(),
                chunk_index: chunk.chunk_index,
                initial_score: score,
            })
            .collect();

        if candidates.is_empty() {
            return Ok(vec![]);
        }

        let mut candidate_map: HashMap<String, SearchCandidate> = HashMap::new();
        for candidate in candidates {
            candidate_map.insert(candidate.chunk_id.clone(), candidate);
        }

        let reranker_inputs: Vec<RerankerCandidate> = candidates
            .iter()
            .map(|candidate| RerankerCandidate {
                chunk_id: candidate.chunk_id.clone(),
                document: candidate.document.clone(),
                text: candidate.text.clone(),
                page_number: candidate.page_number,
                section: candidate.section.clone(),
                initial_score: candidate.initial_score,
            })
            .collect();

        let reranked = match self.reranker.rerank(query, &reranker_inputs).await {
            Ok(results) => results,
            Err(err) => {
                tracing::warn!("Reranker failed, falling back to embedding scores: {}", err);
                Vec::new()
            }
        };

        let mut ordered_results = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();

        if !reranked.is_empty() {
            for result in reranked {
                if let Some(candidate) = candidate_map.get(&result.chunk_id) {
                    if seen.insert(result.chunk_id.clone()) {
                        ordered_results.push(SearchResult {
                            text: candidate.text.clone(),
                            score: result.relevance,
                            document: candidate.document.clone(),
                            chunk_id: candidate.chunk_id.clone(),
                            chunk_index: candidate.chunk_index,
                            page_number: candidate.page_number,
                            section: candidate.section.clone(),
                        });
                    }
                }

                if ordered_results.len() == top_k {
                    break;
                }
            }
        }

        if ordered_results.len() < top_k {
            candidates.sort_by(|a, b| {
                b.initial_score
                    .partial_cmp(&a.initial_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            for candidate in candidates {
                if ordered_results.len() == top_k {
                    break;
                }

                if seen.insert(candidate.chunk_id.clone()) {
                    ordered_results.push(SearchResult {
                        text: candidate.text.clone(),
                        score: candidate.initial_score,
                        document: candidate.document.clone(),
                        chunk_id: candidate.chunk_id.clone(),
                        chunk_index: candidate.chunk_index,
                        page_number: candidate.page_number,
                        section: candidate.section.clone(),
                    });
                }
            }
        }

        Ok(ordered_results)
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

    fn chunk_text(&self, text: &str, chunk_size: usize) -> Vec<ChunkFragment> {
        let mut fragments = Vec::new();

        for (page_idx, page_text) in text.split('\u{c}').enumerate() {
            let words: Vec<&str> = page_text.split_whitespace().collect();
            if words.is_empty() {
                continue;
            }

            let page_number = page_idx + 1;

            for chunk in words.chunks(chunk_size) {
                let chunk_text = chunk.join(" ");
                if !chunk_text.trim().is_empty() {
                    let section = extract_section_heading(&chunk_text);
                    fragments.push(ChunkFragment {
                        text: chunk_text,
                        page_number,
                        section,
                    });
                }
            }
        }

        if fragments.is_empty() {
            let words: Vec<&str> = text.split_whitespace().collect();
            for chunk in words.chunks(chunk_size) {
                let chunk_text = chunk.join(" ");
                if !chunk_text.trim().is_empty() {
                    let section = extract_section_heading(&chunk_text);
                    fragments.push(ChunkFragment {
                        text: chunk_text,
                        page_number: 1,
                        section,
                    });
                }
            }
        }

        fragments
    }

    async fn save_to_disk(&self) -> Result<()> {
        #[derive(Serialize)]
        struct PersistedState<'a> {
            version: u32,
            model: &'a str,
            chunks: &'a HashMap<String, DocumentChunk>,
            needs_reindex: bool,
        }

        let path = format!("{}/chunks.json", self.data_dir);
        let state = PersistedState {
            version: 2,
            model: self.embedding_service.model_name(),
            chunks: &self.chunks,
            needs_reindex: self.needs_reindex,
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
                version: u32,
                model: String,
                chunks: HashMap<String, DocumentChunk>,
                #[serde(default)]
                needs_reindex: bool,
            }

            match serde_json::from_str::<PersistedState>(&data) {
                Ok(PersistedState {
                    version,
                    model,
                    chunks,
                    needs_reindex,
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
                    } else if version < 2 {
                        tracing::info!(
                            "Chunk metadata version {} is outdated. Marking for reindex to capture provenance data.",
                            version
                        );
                        self.chunks.clear();
                        self.needs_reindex = true;
                        self.save_to_disk().await?;
                    } else {
                        self.chunks = chunks;
                        self.needs_reindex = needs_reindex;
                        tracing::info!("Loaded {} chunks from disk", self.chunks.len());
                    }
                }
                Err(_) => {
                    let legacy_chunks: HashMap<String, DocumentChunk> = serde_json::from_str(&data)
                        .context("Failed to parse legacy chunks.json")?;
                    if !legacy_chunks.is_empty() {
                        tracing::warn!(
                            "Existing embeddings were created before provenance metadata was tracked. Reindexing with model '{}' is required.",
                            self.embedding_service.model_name()
                        );
                        tracing::info!(
                            "Discarding {} legacy chunks so they can be regenerated.",
                            legacy_chunks.len()
                        );
                        self.needs_reindex = true;
                    }
                    self.chunks.clear();
                    self.save_to_disk().await?;
                }
            }
        }
        Ok(())
    }
}

/// Extracts a section heading from page text.
///
/// This function scans the provided text line by line to find a suitable heading.
/// It returns the first line that meets all of the following criteria:
/// - Non-empty after trimming whitespace
/// - Not purely numeric (contains at least one non-digit character)
/// - Contains at least 3 alphabetic characters
///
/// The returned heading is truncated to a maximum of 120 characters.
///
/// # Arguments
///
/// * `text` - The text content to extract a heading from
///
/// # Returns
///
/// * `Some(String)` - The extracted heading (up to 120 characters) if found
/// * `None` - If no suitable heading is found in the text
fn extract_section_heading(text: &str) -> Option<String> {
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        if trimmed.chars().all(|c| c.is_ascii_digit()) {
            continue;
        }

        let alphabetic_count = trimmed.chars().filter(|c| c.is_alphabetic()).count();
        if alphabetic_count < 3 {
            continue;
        }

        let truncated: String = trimmed.chars().take(120).collect();
        return Some(truncated);
    }

    None
}

fn default_page_number() -> usize {
    0
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
