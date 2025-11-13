use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
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
    ann_index: Option<AnnIndex>,
    lexical_index: LexicalIndex,
}

impl RagEngine {
    pub async fn new(data_dir: &str) -> Result<Self> {
        let embedding_service = EmbeddingService::new().await?;

        let mut engine = Self {
            chunks: HashMap::new(),
            embedding_service,
            data_dir: data_dir.to_string(),
            needs_reindex: false,
            ann_index: None,
            lexical_index: LexicalIndex::new(),
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

        let existing_ids: Vec<String> = self
            .chunks
            .values()
            .filter(|chunk| chunk.document_name == filename)
            .map(|chunk| chunk.id.clone())
            .collect();

        for chunk_id in existing_ids {
            self.remove_chunk_by_id(&chunk_id);
        }

        let mut chunk_count = 0;
        for (i, chunk_text) in chunks.into_iter().enumerate() {
            if chunk_text.trim().len() < 10 {
                continue;
            }

            tracing::debug!("Generating embedding for chunk {} of {}", i + 1, filename);
            let embedding = self.embedding_service.get_embedding(&chunk_text).await?;

            self.ensure_ann_index(embedding.len());

            let chunk = DocumentChunk {
                id: Uuid::new_v4().to_string(),
                document_name: filename.to_string(),
                text: chunk_text,
                embedding,
                chunk_index: i,
            };

            if let Some(index) = self.ann_index.as_mut() {
                index.insert(&chunk.id, &chunk.embedding);
            }
            self.lexical_index.add_chunk(&chunk.id, &chunk.text);

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

        results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        Ok(results
            .into_iter()
            .take(top_k)
            .map(|(_, result)| result)
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
                        self.lexical_index.clear();
                        self.ann_index = None;
                        self.save_to_disk().await?;
                    } else {
                        self.chunks = chunks;
                        self.needs_reindex = needs_reindex;
                        self.rebuild_indexes();
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
                    self.lexical_index.clear();
                    self.ann_index = None;
                    self.save_to_disk().await?;
                }
            }
        }
        Ok(())
    }

    fn ensure_ann_index(&mut self, dim: usize) {
        let rebuild = match &self.ann_index {
            Some(index) => index.dim() != dim,
            None => true,
        };

        if rebuild {
            let mut index = AnnIndex::new(dim);
            for chunk in self.chunks.values() {
                if chunk.embedding.len() == dim {
                    index.insert(&chunk.id, &chunk.embedding);
                }
            }
            self.ann_index = Some(index);
        }
    }

    fn remove_chunk_by_id(&mut self, chunk_id: &str) {
        if self.chunks.remove(chunk_id).is_some() {
            if let Some(index) = self.ann_index.as_mut() {
                index.remove(chunk_id);
            }
            self.lexical_index.remove_chunk(chunk_id);
        }
    }

    fn rebuild_indexes(&mut self) {
        self.lexical_index.clear();
        let dim = self
            .chunks
            .values()
            .next()
            .map(|chunk| chunk.embedding.len());

        self.ann_index = dim.map(|dim| {
            let mut index = AnnIndex::new(dim);
            for chunk in self.chunks.values() {
                if chunk.embedding.len() == dim {
                    index.insert(&chunk.id, &chunk.embedding);
                }
            }
            index
        });

        for chunk in self.chunks.values() {
            self.lexical_index.add_chunk(&chunk.id, &chunk.text);
        }
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
