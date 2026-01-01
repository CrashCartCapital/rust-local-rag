use crate::chunking::{ChunkFragment, chunk_text};
use crate::error::{EmbeddingError, EngineError, Result};
use crate::search::{AnnIndex, LexicalIndex, SearchResultWithEmbedding, mmr_diversify};
use crate::traits::{EmbeddingBackend, Rerank};
use crate::types::{
    DocumentChunk, DocumentName, RagConfig, RerankedResult, RerankerCandidate, SearchResult,
    SearchWeights,
};
use std::collections::{HashMap, HashSet};
use uuid::Uuid;

pub struct RagEngine<B: EmbeddingBackend, R = ()> {
    backend: B,
    reranker: Option<R>,
    config: RagConfig,

    chunks: HashMap<String, DocumentChunk>,
    document_hashes: HashMap<DocumentName, String>,
    needs_reindex: bool,

    ann_index: Option<AnnIndex>,
    lexical_index: LexicalIndex,
}

#[derive(Debug)]
pub struct PreparedDocument {
    pub document_name: DocumentName,
    pub document_hash: Option<String>,
    pub chunks: Vec<DocumentChunk>,
}

#[derive(Debug, Clone)]
struct SearchCandidate {
    chunk_id: String,
    document: DocumentName,
    text: String,
    page_number: usize,
    section: Option<String>,
    chunk_index: usize,
    initial_score: f32,
    embedding_score: f32,
    lexical_score: f32,
    embedding: Vec<f32>,
}

impl<B: EmbeddingBackend> RagEngine<B, ()> {
    pub fn new(backend: B) -> Self {
        Self::with_config(backend, RagConfig::default())
    }

    pub fn with_config(backend: B, config: RagConfig) -> Self {
        Self {
            backend,
            reranker: None,
            config,
            chunks: HashMap::new(),
            document_hashes: HashMap::new(),
            needs_reindex: false,
            ann_index: None,
            lexical_index: LexicalIndex::new(),
        }
    }
}

impl<B: EmbeddingBackend, R> RagEngine<B, R> {
    pub fn embedding_model(&self) -> &str {
        self.backend.model_id()
    }

    pub fn has_reranker(&self) -> bool {
        self.reranker.is_some()
    }

    pub fn reranker(&self) -> Option<&R> {
        self.reranker.as_ref()
    }

    pub fn needs_reindex(&self) -> bool {
        self.needs_reindex
    }

    pub fn set_needs_reindex(&mut self, value: bool) {
        self.needs_reindex = value;
    }

    pub fn is_document_unchanged(&self, document_name: &str, document_hash: &str) -> bool {
        self.document_hashes
            .get(document_name)
            .is_some_and(|h| h == document_hash)
    }

    pub fn list_documents(&self) -> Vec<DocumentName> {
        let mut docs: Vec<DocumentName> = self
            .chunks
            .values()
            .map(|chunk| chunk.document_name.clone())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        docs.sort();
        docs
    }

    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    pub fn remove_document(&mut self, document_name: &str) -> Result<usize> {
        let mut removed_ids = Vec::new();
        self.chunks.retain(|chunk_id, chunk| {
            if chunk.document_name == document_name {
                removed_ids.push(chunk_id.clone());
                false
            } else {
                true
            }
        });

        for chunk_id in &removed_ids {
            self.lexical_index.remove_chunk(chunk_id);
            if let Some(index) = self.ann_index.as_mut() {
                index.remove(chunk_id);
            }
        }

        self.document_hashes.remove(document_name);

        self.validate_index_sync()?;
        Ok(removed_ids.len())
    }

    pub async fn prepare_document(
        &self,
        document_name: &str,
        text: &str,
        document_hash: Option<&str>,
        mut batch_callback: Option<&mut (dyn FnMut(usize, usize, usize, usize) + Send)>,
    ) -> Result<Option<PreparedDocument>> {
        if let Some(hash) = document_hash {
            if self.is_document_unchanged(document_name, hash) {
                return Ok(None);
            }
        }

        let fragments = chunk_text(text, self.config.chunk_tokens, self.config.sentence_overlap);

        let filtered: Vec<(usize, ChunkFragment)> = fragments
            .into_iter()
            .enumerate()
            .filter_map(|(i, fragment)| {
                if fragment.text.trim().len() < 10 {
                    None
                } else {
                    Some((i, fragment))
                }
            })
            .collect();

        if filtered.is_empty() {
            return Ok(Some(PreparedDocument {
                document_name: document_name.to_string(),
                document_hash: document_hash.map(ToString::to_string),
                chunks: Vec::new(),
            }));
        }

        let chunk_texts: Vec<String> = filtered.iter().map(|(_, f)| f.text.clone()).collect();

        let batch_size = self.config.embedding_batch_size.max(1);
        let total_chunks = chunk_texts.len();
        let batch_count = total_chunks.div_ceil(batch_size);
        let mut embeddings: Vec<Vec<f32>> = Vec::with_capacity(total_chunks);

        for (batch_idx, batch) in chunk_texts.chunks(batch_size).enumerate() {
            if let Some(cb) = batch_callback.as_deref_mut() {
                cb(batch_idx + 1, batch_count, total_chunks, batch.len());
            }

            let batch_embeddings = self.backend.embed_batch(batch).await?;
            if batch_embeddings.len() != batch.len() {
                return Err(EngineError::Embedding(EmbeddingError::Api(format!(
                    "Received {} embeddings for {} texts",
                    batch_embeddings.len(),
                    batch.len()
                ))));
            }
            embeddings.extend(batch_embeddings);
        }

        if embeddings.len() != total_chunks {
            return Err(EngineError::Embedding(EmbeddingError::Api(format!(
                "Total embeddings mismatch: received {} for {} texts",
                embeddings.len(),
                total_chunks
            ))));
        }

        let mut chunks: Vec<DocumentChunk> = Vec::with_capacity(total_chunks);
        for ((chunk_index, fragment), embedding) in filtered.into_iter().zip(embeddings) {
            let mut embedding = embedding;
            crate::search::normalize(&mut embedding);

            chunks.push(DocumentChunk {
                id: Uuid::new_v4().to_string(),
                document_name: document_name.to_string(),
                text: fragment.text,
                embedding,
                chunk_index,
                page_number: fragment.page_number,
                section: fragment.section.clone(),
                metadata: fragment.metadata,
            });
        }

        Ok(Some(PreparedDocument {
            document_name: document_name.to_string(),
            document_hash: document_hash.map(ToString::to_string),
            chunks,
        }))
    }

    pub fn upsert_prepared_document(&mut self, prepared: PreparedDocument) -> Result<usize> {
        // Remove existing chunks even if we end up with zero usable chunks.
        let _ = self.remove_document(&prepared.document_name);

        if prepared.chunks.is_empty() {
            if let Some(hash) = prepared.document_hash {
                self.document_hashes.insert(prepared.document_name, hash);
            }

            self.validate_index_sync()?;
            return Ok(0);
        }

        let mut chunk_count = 0usize;
        for chunk in prepared.chunks {
            if self.ann_index.is_none() && !chunk.embedding.is_empty() {
                self.ann_index = Some(AnnIndex::new(chunk.embedding.len()));
            }
            if let Some(index) = self.ann_index.as_mut() {
                index.insert(&chunk.id, &chunk.embedding);
            }
            self.lexical_index.add_chunk(&chunk.id, &chunk.text);
            self.chunks.insert(chunk.id.clone(), chunk);
            chunk_count += 1;
        }

        if let Some(hash) = prepared.document_hash {
            self.document_hashes.insert(prepared.document_name, hash);
        }

        self.validate_index_sync()?;
        Ok(chunk_count)
    }

    pub async fn upsert_document(
        &mut self,
        document_name: &str,
        text: &str,
        document_hash: Option<String>,
    ) -> Result<usize> {
        let prepared = self
            .prepare_document(document_name, text, document_hash.as_deref(), None)
            .await?;
        let Some(prepared) = prepared else {
            return Ok(0);
        };
        self.upsert_prepared_document(prepared)
    }

    pub async fn search(
        &self,
        query: &str,
        top_k: usize,
        weights: Option<SearchWeights>,
    ) -> Result<Vec<SearchResult>>
    where
        R: Rerank,
    {
        let resolved = weights.unwrap_or(self.config.weights);
        let results = self.search_internal(query, top_k, resolved).await?;
        Ok(results.into_iter().map(|r| r.result).collect())
    }

    pub async fn search_with_diversity(
        &self,
        query: &str,
        top_k: usize,
        diversity_factor: f32,
        weights: Option<SearchWeights>,
    ) -> Result<Vec<SearchResult>>
    where
        R: Rerank,
    {
        let diversity_factor = diversity_factor.clamp(0.0, 1.0);
        if diversity_factor == 0.0 {
            return self.search(query, top_k, weights).await;
        }

        let top_k = top_k.max(1);
        let candidate_pool_size = (top_k * 3).max(top_k + 10);
        let resolved = weights.unwrap_or(self.config.weights);
        let candidates_with_embeddings = self
            .search_internal(query, candidate_pool_size, resolved)
            .await?;

        if candidates_with_embeddings.is_empty() {
            return Ok(vec![]);
        }

        Ok(mmr_diversify(
            candidates_with_embeddings,
            top_k,
            diversity_factor,
        ))
    }

    pub async fn embedding_candidates(
        &self,
        query: &str,
        count: usize,
    ) -> Result<Vec<RerankerCandidate>> {
        if self.chunks.is_empty() || count == 0 {
            return Ok(vec![]);
        }

        let mut query_embedding = self.backend.embed(query).await?;
        crate::search::normalize(&mut query_embedding);

        let candidate_iter: Box<dyn Iterator<Item = String>> = match &self.ann_index {
            Some(index) => Box::new(
                index
                    .search(&query_embedding, count.saturating_mul(2))
                    .into_iter(),
            ),
            None => Box::new(self.chunks.keys().cloned()),
        };

        let mut scores: Vec<(f32, &DocumentChunk)> = Vec::new();
        for chunk_id in candidate_iter {
            if let Some(chunk) = self.chunks.get(&chunk_id) {
                let embedding_score =
                    crate::search::dot_product(&query_embedding, &chunk.embedding);
                scores.push((embedding_score, chunk));
            }
        }

        scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let candidates: Vec<RerankerCandidate> = scores
            .into_iter()
            .take(count)
            .map(|(score, chunk)| RerankerCandidate {
                chunk_id: chunk.id.clone(),
                document: chunk.document_name.clone(),
                text: chunk.text.clone(),
                page_number: chunk.page_number,
                section: chunk.section.clone(),
                initial_score: score,
            })
            .collect();

        Ok(candidates)
    }

    async fn search_internal(
        &self,
        query: &str,
        top_k: usize,
        weights: SearchWeights,
    ) -> Result<Vec<SearchResultWithEmbedding>>
    where
        R: Rerank,
    {
        if self.chunks.is_empty() {
            return Ok(vec![]);
        }

        let top_k = top_k.max(1);

        let mut query_embedding = self.backend.embed(query).await?;
        crate::search::normalize(&mut query_embedding);

        let ann_candidate_iter: Box<dyn Iterator<Item = String>> = match &self.ann_index {
            Some(index) => Box::new(
                index
                    .search(&query_embedding, top_k.saturating_mul(5))
                    .into_iter(),
            ),
            None => Box::new(self.chunks.keys().cloned()),
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

        let mut scores: Vec<(f32, f32, f32, DocumentChunk)> = Vec::new();

        for chunk_id in candidate_ids {
            if let Some(chunk) = self.chunks.get(&chunk_id) {
                let embedding_score =
                    crate::search::dot_product(&query_embedding, &chunk.embedding);
                let lexical_score = lexical_map
                    .get(&chunk_id)
                    .map(|score| score / max_lexical)
                    .unwrap_or(0.0);
                let combined_score =
                    weights.embedding * embedding_score + weights.lexical * lexical_score;

                scores.push((
                    combined_score,
                    embedding_score,
                    lexical_score,
                    chunk.clone(),
                ));
            }
        }

        scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let initial_k = scores.len().min(top_k.saturating_mul(3).max(top_k));

        let candidates: Vec<SearchCandidate> = scores
            .into_iter()
            .take(initial_k)
            .map(|(combined, embed, lex, chunk)| SearchCandidate {
                chunk_id: chunk.id.clone(),
                document: chunk.document_name.clone(),
                text: chunk.text.clone(),
                page_number: chunk.page_number,
                section: chunk.section.clone(),
                chunk_index: chunk.chunk_index,
                initial_score: combined,
                embedding_score: embed,
                lexical_score: lex,
                embedding: chunk.embedding.clone(),
            })
            .collect();

        if candidates.is_empty() {
            return Ok(vec![]);
        }

        let candidate_map: HashMap<String, SearchCandidate> = candidates
            .iter()
            .cloned()
            .map(|candidate| (candidate.chunk_id.clone(), candidate))
            .collect();

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

        let reranked: Vec<RerankedResult> = match &self.reranker {
            Some(reranker) => reranker
                .rerank(query, &reranker_inputs)
                .await
                .unwrap_or_else(|_err| {
                    #[cfg(feature = "tracing")]
                    tracing::warn!("Reranker failed, falling back to initial scores: {}", _err);
                    Vec::new()
                }),
            None => Vec::new(),
        };

        let mut ordered_results = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();

        if !reranked.is_empty() {
            let max_reranker = reranked
                .iter()
                .map(|r| r.relevance)
                .fold(0.0_f32, f32::max)
                .max(f32::EPSILON);
            let max_initial = candidates
                .iter()
                .map(|c| c.initial_score)
                .fold(0.0_f32, f32::max)
                .max(f32::EPSILON);

            for result in &reranked {
                if let Some(candidate) = candidate_map.get(&result.chunk_id)
                    && seen.insert(result.chunk_id.clone())
                {
                    let reranker_norm = result.relevance / max_reranker;
                    let initial_norm = candidate.initial_score / max_initial;
                    let blended_score =
                        weights.reranker * reranker_norm + weights.initial * initial_norm;

                    ordered_results.push(SearchResultWithEmbedding {
                        result: SearchResult {
                            text: candidate.text.clone(),
                            score: blended_score,
                            document: candidate.document.clone(),
                            chunk_id: candidate.chunk_id.clone(),
                            chunk_index: candidate.chunk_index,
                            page_number: candidate.page_number,
                            section: candidate.section.clone(),
                            embedding_score: Some(candidate.embedding_score),
                            lexical_score: Some(candidate.lexical_score),
                            initial_score: Some(candidate.initial_score),
                            reranker_score: Some(result.relevance),
                            yes_logprob: result.yes_logprob,
                            no_logprob: result.no_logprob,
                        },
                        embedding: candidate.embedding.clone(),
                    });
                }
            }

            ordered_results.sort_by(|a, b| {
                b.result
                    .score
                    .partial_cmp(&a.result.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            ordered_results.truncate(top_k);
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
                    ordered_results.push(SearchResultWithEmbedding {
                        result: SearchResult {
                            text: candidate.text.clone(),
                            score: candidate.initial_score,
                            document: candidate.document.clone(),
                            chunk_id: candidate.chunk_id.clone(),
                            chunk_index: candidate.chunk_index,
                            page_number: candidate.page_number,
                            section: candidate.section.clone(),
                            embedding_score: Some(candidate.embedding_score),
                            lexical_score: Some(candidate.lexical_score),
                            initial_score: Some(candidate.initial_score),
                            reranker_score: None,
                            yes_logprob: None,
                            no_logprob: None,
                        },
                        embedding: candidate.embedding.clone(),
                    });
                }
            }
        }

        Ok(ordered_results)
    }

    fn validate_index_sync(&mut self) -> Result<()> {
        let valid_chunk_ids: HashSet<String> = self.chunks.keys().cloned().collect();

        self.lexical_index.drop_stale(&valid_chunk_ids);

        for chunk_id in &valid_chunk_ids {
            if let Some(chunk) = self.chunks.get(chunk_id) {
                if !self.lexical_index.contains(chunk_id) {
                    #[cfg(feature = "tracing")]
                    tracing::debug!("Re-adding missing chunk {} to lexical index", chunk_id);
                    self.lexical_index.add_chunk(chunk_id, &chunk.text);
                }
            }
        }

        if self.ann_index.is_none() && !self.chunks.is_empty() {
            if let Some(first_chunk) = self.chunks.values().next() {
                let dim = first_chunk.embedding.len();
                if dim > 0 {
                    self.ann_index = Some(AnnIndex::new(dim));
                }
            }
        }

        if let Some(ann_index) = &mut self.ann_index {
            ann_index.drop_stale(&valid_chunk_ids);

            for chunk_id in &valid_chunk_ids {
                if let Some(chunk) = self.chunks.get(chunk_id) {
                    if !ann_index.contains(chunk_id) {
                        #[cfg(feature = "tracing")]
                        tracing::debug!("Re-adding missing chunk {} to ANN index", chunk_id);
                        ann_index.insert(chunk_id, &chunk.embedding);
                    }
                }
            }
        }

        let valid_documents: HashSet<String> = self
            .chunks
            .values()
            .map(|chunk| chunk.document_name.clone())
            .collect();
        self.document_hashes.retain(|doc_name, _| {
            let keep = valid_documents.contains(doc_name);
            if !keep {
                #[cfg(feature = "tracing")]
                tracing::debug!("Removing orphaned document hash for {}", doc_name);
            }
            keep
        });

        Ok(())
    }
}

impl<B: EmbeddingBackend, R: Rerank> RagEngine<B, R> {
    pub fn with_optional_reranker(backend: B, reranker: Option<R>, config: RagConfig) -> Self {
        Self {
            backend,
            reranker,
            config,
            chunks: HashMap::new(),
            document_hashes: HashMap::new(),
            needs_reindex: false,
            ann_index: None,
            lexical_index: LexicalIndex::new(),
        }
    }

    pub fn with_reranker(backend: B, reranker: R) -> Self {
        Self {
            backend,
            reranker: Some(reranker),
            config: RagConfig::default(),
            chunks: HashMap::new(),
            document_hashes: HashMap::new(),
            needs_reindex: false,
            ann_index: None,
            lexical_index: LexicalIndex::new(),
        }
    }

    pub fn set_reranker(&mut self, reranker: Option<R>) {
        self.reranker = reranker;
    }
}

#[cfg(feature = "persistence")]
impl<B: EmbeddingBackend, R> RagEngine<B, R>
where
    R: Rerank,
{
    pub fn save_to_dir(&self, data_dir: impl AsRef<std::path::Path>) -> Result<()> {
        use crate::persistence::{INDEX_VERSION, index_path};
        use serde::Serialize;

        #[derive(Serialize)]
        struct PersistedState<'a> {
            version: u32,
            model: &'a str,
            chunks: &'a HashMap<String, DocumentChunk>,
            needs_reindex: bool,
            #[serde(default, skip_serializing_if = "HashMap::is_empty")]
            document_hashes: &'a HashMap<String, String>,
        }

        let model_name = self.backend.model_id();
        let final_path = index_path(data_dir.as_ref(), model_name);
        let temp_path = final_path.with_extension("json.tmp");

        let state = PersistedState {
            version: INDEX_VERSION,
            model: model_name,
            chunks: &self.chunks,
            needs_reindex: self.needs_reindex,
            document_hashes: &self.document_hashes,
        };

        let data = serde_json::to_string_pretty(&state)
            .map_err(|e| EngineError::Persistence(e.to_string()))?;

        std::fs::write(&temp_path, data).map_err(|e| EngineError::Persistence(e.to_string()))?;
        std::fs::rename(&temp_path, &final_path)
            .map_err(|e| EngineError::Persistence(e.to_string()))?;
        Ok(())
    }

    pub fn load_from_dir(&mut self, data_dir: impl AsRef<std::path::Path>) -> Result<()> {
        use crate::persistence::{index_path, legacy_path};
        use serde::Deserialize;

        #[derive(Deserialize)]
        struct PersistedState {
            version: u32,
            #[allow(dead_code)]
            model: String,
            chunks: HashMap<String, DocumentChunk>,
            #[serde(default)]
            needs_reindex: bool,
            #[serde(default)]
            document_hashes: HashMap<String, String>,
        }

        #[derive(Deserialize)]
        struct ModelOnly {
            model: String,
        }

        let current_model = self.backend.model_id();
        let model_specific_path = index_path(data_dir.as_ref(), current_model);
        let legacy_path = legacy_path(data_dir.as_ref());

        if model_specific_path.exists() {
            let data = std::fs::read_to_string(&model_specific_path)
                .map_err(|e| EngineError::Persistence(e.to_string()))?;

            match serde_json::from_str::<PersistedState>(&data) {
                Ok(state) => {
                    return self.apply_loaded_state(
                        state.version,
                        state.chunks,
                        state.needs_reindex,
                        state.document_hashes,
                        false,
                        data_dir.as_ref(),
                    );
                }
                Err(_e) => {
                    #[cfg(feature = "tracing")]
                    tracing::warn!(
                        "Failed to parse model-specific index at {:?}: {}. Marking for reindex.",
                        model_specific_path,
                        _e
                    );
                    self.needs_reindex = true;
                    return Ok(());
                }
            }
        }

        if legacy_path.exists() {
            let data = std::fs::read_to_string(&legacy_path)
                .map_err(|e| EngineError::Persistence(e.to_string()))?;

            if let Ok(info) = serde_json::from_str::<ModelOnly>(&data) {
                if info.model == current_model {
                    match serde_json::from_str::<PersistedState>(&data) {
                        Ok(state) => {
                            return self.apply_loaded_state(
                                state.version,
                                state.chunks,
                                state.needs_reindex,
                                state.document_hashes,
                                true,
                                data_dir.as_ref(),
                            );
                        }
                        Err(_e) => {
                            #[cfg(feature = "tracing")]
                            tracing::warn!("Failed to parse legacy index: {}. Starting fresh.", _e);
                        }
                    }
                } else {
                    #[cfg(feature = "tracing")]
                    tracing::info!(
                        "Legacy index belongs to model '{}', current model is '{}'. Preserving legacy file.",
                        info.model,
                        current_model
                    );
                }
            } else if let Ok(legacy_chunks) =
                serde_json::from_str::<HashMap<String, DocumentChunk>>(&data)
            {
                if !legacy_chunks.is_empty() {
                    #[cfg(feature = "tracing")]
                    tracing::warn!(
                        "Found legacy chunks without model info. Reindex required for model '{}'.",
                        current_model
                    );
                    self.needs_reindex = true;
                }
            }
        }

        Ok(())
    }

    fn apply_loaded_state(
        &mut self,
        version: u32,
        chunks: HashMap<String, DocumentChunk>,
        needs_reindex: bool,
        document_hashes: HashMap<String, String>,
        migrate_to_new_format: bool,
        data_dir: &std::path::Path,
    ) -> Result<()> {
        if version < crate::persistence::INDEX_VERSION {
            self.chunks.clear();
            self.needs_reindex = true;
            self.save_to_dir(data_dir)?;
            return Ok(());
        }

        self.chunks = chunks;
        for chunk in self.chunks.values_mut() {
            crate::search::normalize(&mut chunk.embedding);
        }

        self.needs_reindex = needs_reindex;
        self.document_hashes = document_hashes;

        if self.document_hashes.is_empty() && !self.chunks.is_empty() {
            self.needs_reindex = true;
        }

        self.validate_index_sync()?;

        if migrate_to_new_format {
            self.save_to_dir(data_dir)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::EmbeddingBackend;

    struct MockBackend;

    impl EmbeddingBackend for MockBackend {
        fn model_id(&self) -> &str {
            "mock-embed"
        }

        fn embed(
            &self,
            text: &str,
        ) -> impl std::future::Future<Output = std::result::Result<Vec<f32>, EmbeddingError>> + Send
        {
            let text = text.to_lowercase();
            async move {
                if text.contains("cats") {
                    Ok(vec![1.0, 0.0])
                } else if text.contains("dogs") {
                    Ok(vec![0.0, 1.0])
                } else {
                    Ok(vec![0.0, 0.0])
                }
            }
        }

        fn embed_batch(
            &self,
            texts: &[String],
        ) -> impl std::future::Future<Output = std::result::Result<Vec<Vec<f32>>, EmbeddingError>> + Send
        {
            let texts: Vec<String> = texts.to_vec();
            async move {
                let mut out = Vec::with_capacity(texts.len());
                for t in texts {
                    if t.to_lowercase().contains("cats") {
                        out.push(vec![1.0, 0.0]);
                    } else if t.to_lowercase().contains("dogs") {
                        out.push(vec![0.0, 1.0]);
                    } else {
                        out.push(vec![0.0, 0.0]);
                    }
                }
                Ok(out)
            }
        }

        fn dimension(&self) -> usize {
            2
        }
    }

    #[tokio::test]
    async fn test_upsert_and_search_with_mock_backend() {
        let mut engine = RagEngine::new(MockBackend);

        let inserted = engine
            .upsert_document("doc1.txt", "Cats are great.", Some("h1".to_string()))
            .await
            .unwrap();
        assert_eq!(inserted, 1);

        let inserted = engine
            .upsert_document("doc2.txt", "Dogs are great.", Some("h2".to_string()))
            .await
            .unwrap();
        assert_eq!(inserted, 1);

        let results = engine.search("cats", 1, None).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].document, "doc1.txt");

        let skipped = engine
            .upsert_document("doc1.txt", "Cats are great.", Some("h1".to_string()))
            .await
            .unwrap();
        assert_eq!(
            skipped, 0,
            "Should skip unchanged documents when hash matches"
        );
    }
}
