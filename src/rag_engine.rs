use anyhow::{Context, Result};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::OnceLock};
use uuid::Uuid;
use walkdir::WalkDir;

use crate::embeddings::EmbeddingService;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ChunkMetadata {
    pub page_range: Option<(usize, usize)>,
    pub sentence_range: Option<(usize, usize)>,
    pub section_title: Option<String>,
    pub token_count: usize,
    pub overlap_with_previous: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentChunk {
    pub id: String,
    pub document_name: String,
    pub text: String,
    pub embedding: Vec<f32>,
    pub chunk_index: usize,
    #[serde(default)]
    pub metadata: ChunkMetadata,
}

#[derive(Debug, Clone)]
struct SentenceInfo {
    text: String,
    tokens: usize,
    page: usize,
    heading: Option<String>,
    index: usize,
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
}

impl RagEngine {
    pub async fn new(data_dir: &str) -> Result<Self> {
        let embedding_service = EmbeddingService::new().await?;

        let mut engine = Self {
            chunks: HashMap::new(),
            embedding_service,
            data_dir: data_dir.to_string(),
            needs_reindex: false,
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

        let chunks = self.chunk_text(&text);
        tracing::info!("Created {} chunks for {}", chunks.len(), filename);

        self.chunks
            .retain(|_, chunk| chunk.document_name != filename);

        let mut chunk_count = 0;
        for (i, (chunk_text, metadata)) in chunks.into_iter().enumerate() {
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
                metadata,
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

    fn chunk_text(&self, text: &str) -> Vec<(String, ChunkMetadata)> {
        let sentences = Self::extract_sentences(text);
        if sentences.is_empty() {
            return Vec::new();
        }

        let total_tokens: usize = sentences.iter().map(|s| s.tokens).sum();
        let max_tokens_per_chunk = 800usize;
        let min_tokens_per_chunk = 120usize;

        let estimated_chunks = if total_tokens == 0 {
            1usize
        } else {
            ((total_tokens as f32) / (max_tokens_per_chunk as f32))
                .ceil()
                .max(1.0) as usize
        };

        let mut target_tokens = if total_tokens == 0 {
            min_tokens_per_chunk
        } else {
            (total_tokens as f32 / estimated_chunks as f32).ceil() as usize
        };
        target_tokens = target_tokens.clamp(min_tokens_per_chunk, max_tokens_per_chunk);

        let mut overlap_tokens = (target_tokens as f32 * 0.2).round() as usize;
        if overlap_tokens == 0 {
            overlap_tokens = 20;
        }
        overlap_tokens = overlap_tokens.min(target_tokens.saturating_sub(1));

        let mut chunks: Vec<(String, ChunkMetadata)> = Vec::new();
        let mut current_indices: Vec<usize> = Vec::new();
        let mut current_token_total = 0usize;
        let mut overlap_with_previous = 0usize;

        for (idx, sentence) in sentences.iter().enumerate() {
            let sentence_tokens = sentence.tokens.max(1);
            if !current_indices.is_empty() && current_token_total + sentence_tokens > target_tokens
            {
                if let Some(chunk) =
                    Self::finalize_chunk(&current_indices, &sentences, overlap_with_previous)
                {
                    chunks.push(chunk);
                }

                let mut new_indices = Vec::new();
                let mut carried_tokens = 0usize;
                for &rev_idx in current_indices.iter().rev() {
                    let tokens = sentences[rev_idx].tokens.max(1);
                    if carried_tokens + tokens > overlap_tokens {
                        break;
                    }
                    new_indices.push(rev_idx);
                    carried_tokens += tokens;
                }
                new_indices.reverse();
                overlap_with_previous = carried_tokens;
                current_indices = new_indices;
                current_token_total = current_indices
                    .iter()
                    .map(|&i| sentences[i].tokens.max(1))
                    .sum();
            }

            if current_indices.is_empty() {
                overlap_with_previous = 0;
            }

            current_indices.push(idx);
            current_token_total += sentence_tokens;
        }

        if let Some(chunk) =
            Self::finalize_chunk(&current_indices, &sentences, overlap_with_previous)
        {
            chunks.push(chunk);
        }

        chunks
    }

    fn finalize_chunk(
        sentence_indices: &[usize],
        sentences: &[SentenceInfo],
        overlap_with_previous: usize,
    ) -> Option<(String, ChunkMetadata)> {
        if sentence_indices.is_empty() {
            return None;
        }

        let mut text_parts: Vec<String> = Vec::with_capacity(sentence_indices.len());
        let mut min_page: Option<usize> = None;
        let mut max_page: Option<usize> = None;
        let mut section_title: Option<String> = None;
        let mut token_sum = 0usize;

        for &idx in sentence_indices {
            let sentence = sentences.get(idx)?;
            text_parts.push(sentence.text.clone());
            token_sum += sentence.tokens;

            min_page = Some(match min_page {
                Some(current_min) => current_min.min(sentence.page),
                None => sentence.page,
            });

            max_page = Some(match max_page {
                Some(current_max) => current_max.max(sentence.page),
                None => sentence.page,
            });

            if let Some(title) = &sentence.heading {
                section_title = Some(title.clone());
            }
        }

        let start_index = sentences
            .get(*sentence_indices.first()?)
            .map(|s| s.index)
            .unwrap_or(0);
        let end_index = sentences
            .get(*sentence_indices.last()?)
            .map(|s| s.index)
            .unwrap_or(start_index);

        let mut chunk_text = text_parts.join(" ");
        chunk_text = Self::normalize_whitespace(&chunk_text);

        let mut metadata = ChunkMetadata {
            page_range: min_page.zip(max_page),
            sentence_range: Some((start_index, end_index)),
            section_title,
            token_count: token_sum,
            overlap_with_previous,
        };

        if let Some(title) = metadata.section_title.as_mut() {
            const MAX_TITLE_LEN: usize = 160;
            if title.len() > MAX_TITLE_LEN {
                title.truncate(MAX_TITLE_LEN);
            }
        }

        if chunk_text.is_empty() {
            return None;
        }

        Some((chunk_text, metadata))
    }

    fn extract_sentences(text: &str) -> Vec<SentenceInfo> {
        let splitter = Self::sentence_splitter();
        let mut sentences: Vec<SentenceInfo> = Vec::new();
        let mut sentence_index = 0usize;

        for (page_idx, page_text) in text.split('\u{0c}').enumerate() {
            let page_number = page_idx + 1;
            let mut last_heading: Option<String> = None;

            for block in page_text.split("\n\n") {
                let block = block.trim();
                if block.is_empty() {
                    continue;
                }

                let lines: Vec<&str> = block.lines().collect();
                if lines.len() == 1 && Self::is_heading(lines[0]) {
                    last_heading = Some(lines[0].trim().to_string());
                    continue;
                }

                let mut paragraph_lines = Vec::new();
                for line in lines {
                    let trimmed = line.trim();
                    if trimmed.is_empty() {
                        continue;
                    }
                    if paragraph_lines.is_empty() && Self::is_heading(trimmed) {
                        last_heading = Some(trimmed.to_string());
                        continue;
                    }
                    paragraph_lines.push(trimmed);
                }

                if paragraph_lines.is_empty() {
                    continue;
                }

                let normalized = Self::normalize_whitespace(&paragraph_lines.join(" "));
                if normalized.is_empty() {
                    continue;
                }

                let splits: Vec<&str> = splitter
                    .split(&normalized)
                    .map(str::trim)
                    .filter(|part| !part.is_empty())
                    .collect();

                let parts = if splits.is_empty() {
                    vec![normalized.as_str()]
                } else {
                    splits
                };

                for part in parts {
                    let tokens = Self::approximate_token_count(part);
                    if tokens == 0 {
                        continue;
                    }
                    sentences.push(SentenceInfo {
                        text: part.to_string(),
                        tokens,
                        page: page_number,
                        heading: last_heading.clone(),
                        index: sentence_index,
                    });
                    sentence_index += 1;
                }
            }
        }

        if sentences.is_empty() {
            let normalized = Self::normalize_whitespace(text);
            if !normalized.is_empty() {
                sentences.push(SentenceInfo {
                    text: normalized.clone(),
                    tokens: Self::approximate_token_count(&normalized),
                    page: 1,
                    heading: None,
                    index: 0,
                });
            }
        }

        sentences
    }

    fn normalize_whitespace(value: &str) -> String {
        value
            .split_whitespace()
            .filter(|segment| !segment.is_empty())
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn is_heading(line: &str) -> bool {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.len() > 120 {
            return false;
        }

        let word_count = trimmed.split_whitespace().count();
        if word_count == 0 || word_count > 12 {
            return false;
        }

        let uppercase_letters = trimmed.chars().filter(|c| c.is_uppercase()).count();
        let lowercase_letters = trimmed.chars().filter(|c| c.is_lowercase()).count();

        if lowercase_letters == 0 && uppercase_letters > 0 {
            return true;
        }

        if trimmed.ends_with(':') {
            return true;
        }

        if word_count <= 4 && uppercase_letters >= lowercase_letters {
            return true;
        }

        // Refined: match lines starting with digit(s), dot, and whitespace (e.g., "1. Introduction")
        if Regex::new(r"^\d+\.\s").unwrap().is_match(trimmed) {
            return true;
        }

        false
    }

    fn approximate_token_count(value: &str) -> usize {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            return 0;
        }

        let char_count = trimmed.chars().count();
        let word_count = trimmed.split_whitespace().count();
        let char_estimate = (char_count + 3) / 4;
        let word_estimate = ((word_count as f32) * 0.9).ceil() as usize;
        char_estimate.max(word_estimate).max(1)
    }

    fn sentence_splitter() -> &'static Regex {
        static SPLITTER: OnceLock<Regex> = OnceLock::new();
        SPLITTER.get_or_init(|| {
            Regex::new("(?ms)(?<=[.!?])\\s+(?=[A-Z0-9\"'])")
                .expect("valid sentence splitting regex")
        })
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
            version: 1,
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
                #[serde(rename = "version")]
                _version: u32,
                model: String,
                chunks: HashMap<String, DocumentChunk>,
                #[serde(default)]
                needs_reindex: bool,
            }

            match serde_json::from_str::<PersistedState>(&data) {
                Ok(PersistedState {
                    model,
                    chunks,
                    needs_reindex,
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
                    self.save_to_disk().await?;
                }
            }
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
