use anyhow::{Context, Result};
use regex::Regex;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use uuid::Uuid;
use walkdir::WalkDir;

use crate::{
    embeddings::EmbeddingService,
    reranker::{RerankerCandidate, RerankerService},
};

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
    #[serde(default = "default_page_number")]
    pub page_number: usize,
    #[serde(default)]
    pub section: Option<String>,
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
    pub chunk_index: usize,
    pub page_number: usize,
    pub section: Option<String>,
}

pub struct RagEngine {
    chunks: HashMap<String, DocumentChunk>,
    embedding_service: EmbeddingService,
    data_dir: String,
    needs_reindex: bool,
    reranker: Option<RerankerService>,
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
        
        // Try to initialize reranker, but don't fail if it's unavailable
        let reranker = match RerankerService::new().await {
            Ok(service) => {
                tracing::info!("Reranker service initialized successfully");
                Some(service)
            }
            Err(e) => {
                tracing::warn!(
                    "Reranker service unavailable, will fall back to embedding scores only: {}",
                    e
                );
                None
            }
        };

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

        let chunks = self.chunk_text(&text);
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

            if let Some(index) = self.ann_index.as_mut() {
                index.insert(&chunk.id, &chunk.embedding);
            }
            self.lexical_index.add_chunk(&chunk.id, &chunk.text);

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

        let top_k = top_k.max(1);
        tracing::debug!("Searching for: '{}'", query);

        let query_embedding = self.embedding_service.get_query_embedding(query).await?;

        let ann_candidate_iter = match &self.ann_index {
            Some(index) => Box::new(index.search(&query_embedding, top_k.saturating_mul(5)).into_iter()) as Box<dyn Iterator<Item = String>>,
            None => Box::new(self.chunks.keys().cloned()) as Box<dyn Iterator<Item = String>>,
        };

        let lexical_candidates = self.lexical_index.score(query, top_k.saturating_mul(5));
        let lexical_map: HashMap<String, f32> = lexical_candidates.into_iter().collect();

        let mut candidate_ids: HashSet<String> = ann_candidate_iter.collect();
        candidate_ids.extend(lexical_map.keys().cloned());

        if candidate_ids.is_empty() {
            return Ok(vec![]);
        }

        let max_lexical = lexical_map
            .values()
            .copied()
            .fold(0.0_f32, f32::max)
            .max(f32::EPSILON);

        let mut results: Vec<(f32, SearchResult)> = Vec::new();

        for chunk_id in candidate_ids {
            if let Some(chunk) = self.chunks.get(&chunk_id) {
                let embedding_score = cosine_similarity(&query_embedding, &chunk.embedding);
                let lexical_score = lexical_map
                    .get(&chunk_id)
                    .map(|score| score / max_lexical)
                    .unwrap_or(0.0);
                let combined_score =
                    EMBEDDING_WEIGHT * embedding_score + LEXICAL_WEIGHT * lexical_score;

                results.push((
                    combined_score,
                    SearchResult {
                        text: chunk.text.clone(),
                        score: combined_score,
                        document: chunk.document_name.clone(),
                        chunk_id: chunk.id.clone(),
                    },
                ));
            }
        }

        scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let initial_k = scores.len().min(top_k.saturating_mul(3).max(top_k));

        let candidates: Vec<SearchCandidate> = scores
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

        let reranked = match &self.reranker {
            Some(reranker) => match reranker.rerank(query, &reranker_inputs).await {
                Ok(results) => results,
                Err(err) => {
                    tracing::warn!("Reranker failed, falling back to embedding scores: {}", err);
                    Vec::new()
                }
            },
            None => {
                tracing::debug!("Reranker not available, using embedding scores only");
                Vec::new()
            }
        };

        let mut ordered_results = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();

        if !reranked.is_empty() {
            for result in reranked {
                if let Some(candidate) = candidate_map.get(&result.chunk_id)
                    && seen.insert(result.chunk_id.clone()) {
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

                if ordered_results.len() == top_k {
                    break;
                }
            }
        }

        if ordered_results.len() < top_k {
            let mut fallback_candidates: Vec<_> = candidate_map.values().collect();
            fallback_candidates.sort_by(|a, b| {
                b.initial_score
                    .partial_cmp(&a.initial_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            for candidate in fallback_candidates {
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

            current_indices.push(idx);
            current_token_total += sentence_tokens;
        }

        if let Some(chunk) =
            Self::finalize_chunk(&current_indices, &sentences, overlap_with_previous)
        {
            chunks.push(chunk);
        }

        fragments
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

            if section_title.is_none()
                && let Some(title) = &sentence.heading
            {
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
            if title.chars().count() > MAX_TITLE_LEN {
                *title = title.chars().take(MAX_TITLE_LEN).collect();
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
        let char_estimate = char_count.div_ceil(4);
        let word_estimate = ((word_count as f32) * 0.9).ceil() as usize;
        char_estimate.max(word_estimate).max(1)
    }

    fn sentence_splitter() -> &'static srx::Rules {
        static SPLITTER: OnceLock<srx::Rules> = OnceLock::new();
        SPLITTER.get_or_init(|| {
            // Load SRX rules from embedded segment.srx file
            const SRX_XML: &str = include_str!("../data/segment.srx");
            let srx = SRX::from_str(SRX_XML)
                .expect("valid SRX rules from embedded segment.srx");
            
            // Use English language rules for sentence splitting
            // This handles abbreviations like "Dr.", "Mr.", "etc." correctly
            srx.language_rules("English")
        })
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
            version: 2,
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
                version: u32,
                model: String,
                chunks: HashMap<String, DocumentChunk>,
                #[serde(default)]
                needs_reindex: bool,
                #[serde(default)]
                document_hashes: HashMap<String, String>,
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
                        self.lexical_index.clear();
                        self.ann_index = None;
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

struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> f32 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        let bits = (self.state >> 32) as u32;
        let value = bits as f32 / u32::MAX as f32;
        value * 2.0 - 1.0
    }
}

const NUM_HYPERPLANES: usize = 32;
const EMBEDDING_WEIGHT: f32 = 0.7;
const LEXICAL_WEIGHT: f32 = 0.3;
const MAX_SINGLE_BIT_NEIGHBORS: usize = 32;
const MAX_TOTAL_NEIGHBORS: usize = 64;

struct AnnIndex {
    dim: usize,
    hyperplanes: Vec<Vec<f32>>,
    buckets: HashMap<u64, Vec<String>>,
    id_to_bucket: HashMap<String, u64>,
}

impl AnnIndex {
    fn new(dim: usize) -> Self {
        let mut rng = SimpleRng::new(42);
        let mut hyperplanes = Vec::with_capacity(NUM_HYPERPLANES);

        for _ in 0..NUM_HYPERPLANES {
            let mut plane = Vec::with_capacity(dim);
            for _ in 0..dim {
                plane.push(rng.next());
            }
            // Normalize the hyperplane to unit length
            let magnitude = plane.iter().map(|&x| x * x).sum::<f32>().sqrt();
            if magnitude > 0.0 {
                for val in &mut plane {
                    *val /= magnitude;
                }
            }
            hyperplanes.push(plane);
        }

        Self {
            dim,
            hyperplanes,
            buckets: HashMap::new(),
            id_to_bucket: HashMap::new(),
        }
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn insert(&mut self, id: &str, vector: &[f32]) {
        if vector.len() != self.dim {
            tracing::warn!(
                "Vector dimension {} does not match ANN index dimension {}",
                vector.len(),
                self.dim
            );
            return;
        }

        let hash = self.hash(vector);
        self.buckets.entry(hash).or_default().push(id.to_string());
        self.id_to_bucket.insert(id.to_string(), hash);
    }

    fn remove(&mut self, id: &str) {
        if let Some(hash) = self.id_to_bucket.remove(id) {
            if let Some(bucket) = self.buckets.get_mut(&hash) {
                bucket.retain(|stored| stored != id);
                if bucket.is_empty() {
                    self.buckets.remove(&hash);
                }
            }
        }
    }

    fn search(&self, vector: &[f32], max_candidates: usize) -> Vec<String> {
        if self.buckets.is_empty() || max_candidates == 0 {
            return Vec::new();
        }

        let mut candidates = Vec::new();
        let mut visited = HashSet::new();
        let primary_hash = self.hash(vector);

        self.collect_bucket(primary_hash, &mut candidates, &mut visited, max_candidates);

        if candidates.len() < max_candidates {
            for neighbor in self.neighbor_hashes(primary_hash) {
                if candidates.len() >= max_candidates {
                    break;
                }
                self.collect_bucket(neighbor, &mut candidates, &mut visited, max_candidates);
            }
        }

        if candidates.len() < max_candidates {
            for (hash, bucket) in &self.buckets {
                if candidates.len() >= max_candidates {
                    break;
                }
                if visited.contains(hash) {
                    continue;
                }
                for id in bucket {
                    if candidates.len() >= max_candidates {
                        break;
                    }
                    candidates.push(id.clone());
                }
            }
        }

        candidates
    }

    fn hash(&self, vector: &[f32]) -> u64 {
        let mut hash = 0u64;

        for (i, plane) in self.hyperplanes.iter().enumerate() {
            let dot: f32 = vector.iter().zip(plane.iter()).map(|(a, b)| a * b).sum();
            if dot >= 0.0 {
                hash |= 1u64 << i;
            }
        }

        hash
    }

    fn collect_bucket(
        &self,
        hash: u64,
        candidates: &mut Vec<String>,
        visited: &mut HashSet<u64>,
        limit: usize,
    ) {
        if visited.contains(&hash) {
            return;
        }

        visited.insert(hash);

        if let Some(bucket) = self.buckets.get(&hash) {
            for id in bucket {
                if candidates.len() >= limit {
                    break;
                }
                candidates.push(id.clone());
            }
        }
    }

    fn neighbor_hashes(&self, hash: u64) -> Vec<u64> {
        let bits = self.hyperplanes.len().min(64);
        let mut neighbors = Vec::new();

        for i in 0..bits {
            if neighbors.len() >= MAX_SINGLE_BIT_NEIGHBORS {
                break;
            }
            neighbors.push(hash ^ (1u64 << i));
        }

        if neighbors.len() < MAX_SINGLE_BIT_NEIGHBORS {
            for i in 0..bits {
                if neighbors.len() >= MAX_TOTAL_NEIGHBORS {
                    break;
                }
                for j in (i + 1)..bits {
                    neighbors.push(hash ^ (1u64 << i) ^ (1u64 << j));
                    if neighbors.len() >= MAX_TOTAL_NEIGHBORS {
                        break;
                    }
                }
            }
        }

        neighbors
    }
}

#[derive(Default)]
struct LexicalIndex {
    term_postings: HashMap<String, HashMap<String, usize>>,
    doc_lengths: HashMap<String, usize>,
    doc_terms: HashMap<String, HashMap<String, usize>>,
    total_docs: usize,
    total_length: usize,
}

impl LexicalIndex {
    fn new() -> Self {
        Self::default()
    }

    fn clear(&mut self) {
        self.term_postings.clear();
        self.doc_lengths.clear();
        self.doc_terms.clear();
        self.total_docs = 0;
        self.total_length = 0;
    }

    fn add_chunk(&mut self, id: &str, text: &str) {
        if self.doc_terms.contains_key(id) {
            self.remove_chunk(id);
        }

        let tokens = tokenize(text);
        if tokens.is_empty() {
            return;
        }

        let mut term_counts: HashMap<String, usize> = HashMap::new();
        for token in tokens {
            *term_counts.entry(token).or_insert(0) += 1;
        }

        let doc_length: usize = term_counts.values().sum();
        if doc_length == 0 {
            return;
        }

        for (term, count) in &term_counts {
            self.term_postings
                .entry(term.clone())
                .or_default()
                .insert(id.to_string(), *count);
        }

        self.doc_lengths.insert(id.to_string(), doc_length);
        self.doc_terms.insert(id.to_string(), term_counts);
        self.total_docs += 1;
        self.total_length += doc_length;
    }

    fn remove_chunk(&mut self, id: &str) {
        if let Some(term_counts) = self.doc_terms.remove(id) {
            for (term, _) in term_counts {
                if let Some(postings) = self.term_postings.get_mut(&term) {
                    postings.remove(id);
                    if postings.is_empty() {
                        self.term_postings.remove(&term);
                    }
                }
            }
            if let Some(length) = self.doc_lengths.remove(id) {
                if self.total_length >= length {
                    self.total_length -= length;
                } else {
                    self.total_length = 0;
                }
            }
            if self.total_docs > 0 {
                self.total_docs -= 1;
            }
        } else {
            // Ensure doc_lengths is clean even if we didn't have stored term counts.
            self.doc_lengths.remove(id);
        }

        if self.total_docs == 0 {
            self.total_length = 0;
        }
    }

    fn score(&self, query: &str, limit: usize) -> Vec<(String, f32)> {
        if self.total_docs == 0 {
            return Vec::new();
        }

        let tokens = tokenize(query);
        if tokens.is_empty() {
            return Vec::new();
        }

        let mut unique_terms: HashSet<String> = HashSet::new();
        for token in tokens {
            unique_terms.insert(token);
        }

        let avg_doc_len = if self.total_docs == 0 {
            0.0
        } else {
            self.total_length as f32 / self.total_docs as f32
        };

        let k1 = 1.5_f32;
        let b = 0.75_f32;
        let mut scores: HashMap<String, f32> = HashMap::new();

        for term in unique_terms {
            if let Some(postings) = self.term_postings.get(&term) {
                let df = postings.len() as f32;
                let idf = ((self.total_docs as f32 - df + 0.5) / (df + 0.5))
                    .ln()
                    .max(0.0);

                for (doc_id, term_freq) in postings {
                    let doc_length = *self.doc_lengths.get(doc_id).unwrap_or(&0) as f32;
                    if doc_length == 0.0 {
                        continue;
                    }

                    let tf = *term_freq as f32;
                    let denom = tf + k1 * (1.0 - b + b * (doc_length / avg_doc_len));
                    if denom == 0.0 {
                        continue;
                    }

                    let score = idf * (tf * (k1 + 1.0)) / denom;
                    *scores.entry(doc_id.clone()).or_insert(0.0) += score;
                }
            }
        }

        let mut results: Vec<(String, f32)> = scores.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        if limit > 0 && results.len() > limit {
            results.truncate(limit);
        }
        results
    }
}

fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|token| !token.is_empty())
        .map(|token| token.to_lowercase())
        .collect()
}
