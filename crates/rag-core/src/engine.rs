use crate::chunking::{ChunkFragment, chunk_text};
use crate::error::{EmbeddingError, EngineError, Result};
use crate::persistence::EngineState;
use crate::search::{AnnIndex, LexicalIndex, SearchResultWithEmbedding, mmr_diversify};
use crate::traits::{EmbeddingBackend, Rerank};
use crate::types::{
    DocumentChunk, DocumentName, HealthStatus, RagConfig, RerankedResult, RerankerCandidate,
    SearchResult, SearchWeights,
};
use std::collections::{HashMap, HashSet};
#[cfg(feature = "tracing")]
use tracing::{field, instrument};
use uuid::Uuid;

pub struct RagEngine<B: EmbeddingBackend, R = ()> {
    backend: B,
    reranker: Option<R>,
    config: RagConfig,

    state: EngineState,

    ann_index: Option<AnnIndex>,
    lexical_index: LexicalIndex,
}

#[derive(Debug)]
pub struct PreparedDocument {
    pub document_name: DocumentName,
    pub document_hash: Option<String>,
    pub chunks: Vec<DocumentChunk>,
}

impl<B: EmbeddingBackend> RagEngine<B, ()> {
    pub fn new(backend: B) -> Self {
        Self::with_config(backend, RagConfig::default())
    }

    pub fn with_config(backend: B, config: RagConfig) -> Self {
        let embedding_model_id = backend.model_id().to_string();
        let embedding_dim = backend.dimension();
        Self {
            backend,
            reranker: None,
            config,
            state: EngineState::new(embedding_model_id, embedding_dim),
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
        self.state.needs_reindex
    }

    pub fn set_needs_reindex(&mut self, value: bool) {
        self.state.needs_reindex = value;
    }

    pub fn is_document_unchanged(&self, document_name: &str, document_hash: &str) -> bool {
        self.state
            .document_hashes
            .get(document_name)
            .is_some_and(|h| h == document_hash)
    }

    pub fn list_documents(&self) -> Vec<DocumentName> {
        let mut docs: Vec<DocumentName> = self
            .state
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
        self.state.chunks.len()
    }

    pub fn persisted_embedding_dim(&self) -> usize {
        self.state.embedding_dim
    }

    pub fn backend_embedding_dim(&self) -> usize {
        self.backend.dimension()
    }

    /// Returns health status for monitoring/observability.
    ///
    /// This method is useful for container health probes or monitoring dashboards.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let health = engine.health();
    /// if health.is_healthy {
    ///     println!("Engine ready: {} docs, {} chunks", health.document_count, health.chunk_count);
    /// }
    /// ```
    pub fn health(&self) -> HealthStatus {
        let embedding_dim = match self.backend.dimension() {
            0 => self.state.embedding_dim,
            dim => dim,
        };
        HealthStatus {
            is_healthy: true,
            embedding_model: self.backend.model_id().to_string(),
            embedding_dim,
            document_count: self.list_documents().len(),
            chunk_count: self.state.chunks.len(),
            needs_reindex: self.state.needs_reindex,
            has_reranker: self.reranker.is_some(),
        }
    }

    pub fn remove_document(&mut self, document_name: &str) -> Result<usize> {
        let mut removed_ids = Vec::new();
        self.state.chunks.retain(|chunk_id, chunk| {
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

        self.state.document_hashes.remove(document_name);

        self.validate_index_sync()?;
        Ok(removed_ids.len())
    }

    #[cfg_attr(
        feature = "tracing",
        instrument(
            skip(self, text, batch_callback),
            fields(
                doc = document_name,
                text_len = text.len(),
                chunk_tokens = self.config.chunk_tokens,
                overlap = self.config.sentence_overlap,
                chunks_generated = field::Empty,
                chunks_filtered = field::Empty,
                chunking_ms = field::Empty,
                embedding_ms = field::Empty
            ),
            err
        )
    )]
    pub async fn prepare_document(
        &self,
        document_name: &str,
        text: &str,
        document_hash: Option<&str>,
        mut batch_callback: Option<&mut (dyn FnMut(usize, usize, usize, usize) + Send)>,
    ) -> Result<Option<PreparedDocument>> {
        if let Some(hash) = document_hash
            && self.is_document_unchanged(document_name, hash)
        {
            #[cfg(feature = "tracing")]
            tracing::debug!("Document unchanged, skipping");
            return Ok(None);
        }

        let start_time = std::time::Instant::now();

        #[cfg(feature = "tracing")]
        let chunk_start = std::time::Instant::now();

        let fragments = chunk_text(text, self.config.chunk_tokens, self.config.sentence_overlap)?;

        #[cfg(feature = "tracing")]
        {
            let elapsed = chunk_start.elapsed().as_millis();
            tracing::Span::current().record("chunking_ms", elapsed);
            tracing::Span::current().record("chunks_generated", fragments.len());
        }

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

        #[cfg(feature = "tracing")]
        tracing::Span::current().record("chunks_filtered", filtered.len());

        if filtered.is_empty() {
            #[cfg(feature = "tracing")]
            tracing::debug!("No valid chunks after filtering (min length 10)");
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

        #[cfg(feature = "tracing")]
        let embed_start = std::time::Instant::now();

        for (batch_idx, batch) in chunk_texts.chunks(batch_size).enumerate() {
            if let Some(timeout) = self.config.document_prep_timeout {
                if start_time.elapsed() > timeout {
                    return Err(EngineError::Embedding(EmbeddingError::Timeout(timeout)));
                }
            }

            if let Some(cb) = batch_callback.as_deref_mut() {
                cb(batch_idx + 1, batch_count, total_chunks, batch.len());
            }

            let batch_embeddings = self.backend.embed_batch(batch).await.inspect_err(|_e| {
                #[cfg(feature = "tracing")]
                tracing::error!(
                    error = ?_e,
                    batch_idx,
                    batch_size = batch.len(),
                    "Failed to embed batch"
                );
            })?;

            if batch_embeddings.len() != batch.len() {
                return Err(EngineError::Embedding(EmbeddingError::Api(format!(
                    "Received {} embeddings for {} texts",
                    batch_embeddings.len(),
                    batch.len()
                ))));
            }
            embeddings.extend(batch_embeddings);
        }

        #[cfg(feature = "tracing")]
        {
            let elapsed = embed_start.elapsed().as_millis();
            tracing::Span::current().record("embedding_ms", elapsed);
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
                tags: std::collections::HashSet::new(),
                resolution: crate::types::Resolution::default(), // Chunk-level by default
                parent_id: None,
            });
        }

        #[cfg(feature = "tracing")]
        tracing::debug!(
            "Prepared document {} with {} chunks",
            document_name,
            chunks.len()
        );

        Ok(Some(PreparedDocument {
            document_name: document_name.to_string(),
            document_hash: document_hash.map(ToString::to_string),
            chunks,
        }))
    }

    #[cfg_attr(feature = "tracing", instrument(skip(self, prepared), fields(doc = %prepared.document_name, chunks = prepared.chunks.len())))]
    pub fn upsert_prepared_document(&mut self, prepared: PreparedDocument) -> Result<usize> {
        // Pass 1: Validate all chunks before modifying state
        let mut target_dim = self.state.embedding_dim;

        // If engine dim is not set, infer it from the first non-empty chunk
        if target_dim == 0 {
            for chunk in &prepared.chunks {
                if !chunk.embedding.is_empty() {
                    target_dim = chunk.embedding.len();
                    break;
                }
            }
        }

        if target_dim > 0 {
            for chunk in &prepared.chunks {
                if !chunk.embedding.is_empty() && chunk.embedding.len() != target_dim {
                    return Err(EngineError::validation(
                        chunk.id.clone(),
                        crate::error::ValidationKind::DimensionMismatch {
                            expected: target_dim,
                            got: chunk.embedding.len(),
                        },
                    ));
                }
            }
        }

        // Pass 2: Apply changes (remove old, insert new)
        // Remove existing chunks even if we end up with zero usable chunks.
        let _ = self.remove_document(&prepared.document_name);

        if prepared.chunks.is_empty() {
            if let Some(hash) = prepared.document_hash {
                self.state
                    .document_hashes
                    .insert(prepared.document_name, hash);
            }

            self.validate_index_sync()?;
            return Ok(0);
        }

        // Initialize engine state from inferred dim if needed
        if self.state.embedding_dim == 0 && target_dim > 0 {
            self.state.embedding_dim = target_dim;
        }

        // Initialize ANN index if needed
        if self.ann_index.is_none() && target_dim > 0 {
            self.ann_index = Some(AnnIndex::new(target_dim));
        }

        let mut chunk_count = 0usize;
        for chunk in prepared.chunks {
            if let Some(index) = self.ann_index.as_mut() {
                index.insert(&chunk.id, &chunk.embedding);
            }
            self.lexical_index.add_chunk(&chunk.id, &chunk.text);
            self.state.chunks.insert(chunk.id.clone(), chunk);
            chunk_count += 1;
        }

        if let Some(hash) = prepared.document_hash {
            self.state
                .document_hashes
                .insert(prepared.document_name, hash);
        }

        self.validate_index_sync()?;
        Ok(chunk_count)
    }

    #[cfg_attr(feature = "tracing", instrument(skip(self, text)))]
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

    #[cfg_attr(feature = "tracing", instrument(skip(self, query)))]
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
        #[cfg(feature = "tracing")]
        tracing::debug!("Search returned {} results", results.len());
        Ok(results.into_iter().map(|r| r.result).collect())
    }

    #[cfg_attr(feature = "tracing", instrument(skip(self, query)))]
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
            #[cfg(feature = "tracing")]
            tracing::debug!("MMR search: no candidates found");
            return Ok(vec![]);
        }

        let results = mmr_diversify(candidates_with_embeddings, top_k, diversity_factor);
        #[cfg(feature = "tracing")]
        tracing::debug!("MMR search returned {} results", results.len());
        Ok(results)
    }

    #[cfg_attr(feature = "tracing", instrument(skip(self, query)))]
    pub async fn embedding_candidates(
        &self,
        query: &str,
        count: usize,
    ) -> Result<Vec<RerankerCandidate>> {
        if self.state.chunks.is_empty() || count == 0 {
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
            None => Box::new(self.state.chunks.keys().cloned()),
        };

        let mut scores: Vec<(f32, &DocumentChunk)> = Vec::new();
        for chunk_id in candidate_iter {
            if let Some(chunk) = self.state.chunks.get(&chunk_id) {
                let embedding_score =
                    crate::search::dot_product(&query_embedding, &chunk.embedding);
                scores.push((embedding_score, chunk));
            }
        }

        scores.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.id.cmp(&b.1.id)));

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

        #[cfg(feature = "tracing")]
        tracing::debug!("Found {} embedding candidates", candidates.len());

        Ok(candidates)
    }

    #[cfg_attr(feature = "tracing", instrument(skip(self, query), fields(query_len = query.len(), top_k = top_k, weights = ?weights), err))]
    async fn search_internal(
        &self,
        query: &str,
        top_k: usize,
        weights: SearchWeights,
    ) -> Result<Vec<SearchResultWithEmbedding>>
    where
        R: Rerank,
    {
        if self.state.chunks.is_empty() {
            #[cfg(feature = "tracing")]
            tracing::debug!("Search internal: index empty");
            return Ok(vec![]);
        }

        let top_k = top_k.max(1);

        #[cfg(feature = "tracing")]
        let start = std::time::Instant::now();

        let mut query_embedding = self.backend.embed(query).await.inspect_err(|_e| {
            #[cfg(feature = "tracing")]
            tracing::error!(error = ?_e, "Failed to generate query embedding");
        })?;

        #[cfg(feature = "tracing")]
        tracing::debug!(duration_ms = ?start.elapsed().as_millis(), "Generated query embedding");

        crate::search::normalize(&mut query_embedding);

        let ann_candidate_iter: Box<dyn Iterator<Item = String>> = match &self.ann_index {
            Some(index) => Box::new(
                index
                    .search(&query_embedding, top_k.saturating_mul(5))
                    .into_iter(),
            ),
            None => Box::new(self.state.chunks.keys().cloned()),
        };

        let lexical_candidates = self.lexical_index.score(query, top_k.saturating_mul(5));
        let lexical_map: HashMap<String, f32> = lexical_candidates.into_iter().collect();

        let mut candidate_ids: HashSet<String> = ann_candidate_iter.collect();

        #[cfg(feature = "tracing")]
        let ann_count = candidate_ids.len();

        candidate_ids.extend(lexical_map.keys().cloned());

        #[cfg(feature = "tracing")]
        tracing::debug!(
            ann_candidates = ann_count,
            lexical_candidates = lexical_map.len(),
            candidates_overlap =
                (ann_count + lexical_map.len()).saturating_sub(candidate_ids.len()),
            total_unique_candidates = candidate_ids.len(),
            "Candidate selection complete"
        );

        if candidate_ids.is_empty() {
            return Ok(vec![]);
        }

        let max_lexical = lexical_map
            .values()
            .copied()
            .fold(0.0_f32, f32::max)
            .max(f32::EPSILON);

        let mut scores: Vec<(f32, f32, f32, &DocumentChunk)> = Vec::new();

        for chunk_id in candidate_ids {
            if let Some(chunk) = self.state.chunks.get(&chunk_id) {
                let embedding_score =
                    crate::search::dot_product(&query_embedding, &chunk.embedding);
                let lexical_score = lexical_map
                    .get(&chunk_id)
                    .map(|score| score / max_lexical)
                    .unwrap_or(0.0);
                let combined_score =
                    weights.embedding * embedding_score + weights.lexical * lexical_score;

                scores.push((combined_score, embedding_score, lexical_score, chunk));
            }
        }

        let initial_k = scores.len().min(top_k.saturating_mul(3).max(top_k));
        if initial_k < scores.len() {
            scores.select_nth_unstable_by(initial_k, |a, b| {
                b.0.total_cmp(&a.0).then_with(|| a.3.id.cmp(&b.3.id))
            });
            scores.truncate(initial_k);
        }
        scores.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.3.id.cmp(&b.3.id)));

        let candidates: Vec<SearchResultWithEmbedding> = scores
            .into_iter()
            .take(initial_k)
            .map(|(combined, embed, lex, chunk)| SearchResultWithEmbedding {
                result: SearchResult {
                    chunk_id: chunk.id.clone(),
                    document: chunk.document_name.clone(),
                    text: chunk.text.clone(),
                    page_number: chunk.page_number,
                    section: chunk.section.clone(),
                    chunk_index: chunk.chunk_index,
                    score: combined,
                    initial_score: Some(combined),
                    embedding_score: Some(embed),
                    lexical_score: Some(lex),
                    reranker_score: None,
                    yes_logprob: None,
                    no_logprob: None,
                },
                embedding: chunk.embedding.clone(),
            })
            .collect();

        if candidates.is_empty() {
            return Ok(vec![]);
        }

        let candidate_map: HashMap<String, SearchResultWithEmbedding> = candidates
            .iter()
            .cloned()
            .map(|candidate| (candidate.result.chunk_id.clone(), candidate))
            .collect();

        let reranker_inputs: Vec<RerankerCandidate> = candidates
            .iter()
            .map(|candidate| RerankerCandidate {
                chunk_id: candidate.result.chunk_id.clone(),
                document: candidate.result.document.clone(),
                text: candidate.result.text.clone(),
                page_number: candidate.result.page_number,
                section: candidate.result.section.clone(),
                initial_score: candidate.result.score,
            })
            .collect();

        let reranked: Vec<RerankedResult> = match &self.reranker {
            Some(reranker) => {
                #[cfg(feature = "tracing")]
                let rerank_start = std::time::Instant::now();

                let res = reranker
                    .rerank(query, &reranker_inputs)
                    .await
                    .unwrap_or_else(|_err| {
                        #[cfg(feature = "tracing")]
                        tracing::error!(error = ?_err, "Reranker failed, falling back to initial scores");
                        Vec::new()
                    });

                #[cfg(feature = "tracing")]
                tracing::debug!(
                    duration_ms = ?rerank_start.elapsed().as_millis(),
                    input_count = reranker_inputs.len(),
                    output_count = res.len(),
                    "Reranking complete"
                );
                res
            }
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
                .map(|c| c.result.score)
                .fold(0.0_f32, f32::max)
                .max(f32::EPSILON);

            for result in &reranked {
                if let Some(candidate) = candidate_map.get(&result.chunk_id)
                    && seen.insert(result.chunk_id.clone())
                {
                    let reranker_norm = result.relevance / max_reranker;
                    // candidate.result.score holds the initial score
                    let initial_norm = candidate.result.score / max_initial;
                    let blended_score =
                        weights.reranker * reranker_norm + weights.initial * initial_norm;

                    ordered_results.push(SearchResultWithEmbedding {
                        result: SearchResult {
                            text: candidate.result.text.clone(),
                            score: blended_score,
                            document: candidate.result.document.clone(),
                            chunk_id: candidate.result.chunk_id.clone(),
                            chunk_index: candidate.result.chunk_index,
                            page_number: candidate.result.page_number,
                            section: candidate.result.section.clone(),
                            embedding_score: candidate.result.embedding_score,
                            lexical_score: candidate.result.lexical_score,
                            initial_score: candidate.result.initial_score,
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
                    .total_cmp(&a.result.score)
                    .then_with(|| a.result.chunk_id.cmp(&b.result.chunk_id))
            });

            ordered_results.truncate(top_k);
        }

        if ordered_results.len() < top_k {
            let mut fallback_candidates: Vec<_> = candidate_map.values().collect();
            fallback_candidates.sort_by(|a, b| {
                b.result
                    .score
                    .total_cmp(&a.result.score)
                    .then_with(|| a.result.chunk_id.cmp(&b.result.chunk_id))
            });
            for candidate in fallback_candidates {
                if ordered_results.len() == top_k {
                    break;
                }
                if seen.insert(candidate.result.chunk_id.clone()) {
                    // It's already a SearchResultWithEmbedding, just clone it
                    ordered_results.push(candidate.clone());
                }
            }
        }

        #[cfg(feature = "tracing")]
        if let Some(first) = ordered_results.first() {
            tracing::debug!(
                top_match_id = %first.result.chunk_id,
                top_match_score = %first.result.score,
                top_match_doc = %first.result.document,
                result_count = ordered_results.len(),
                "Search internal finished"
            );
        } else {
            tracing::debug!("Search internal finished with 0 results");
        }

        Ok(ordered_results)
    }

    #[cfg_attr(feature = "tracing", instrument(skip(self)))]
    fn validate_index_sync(&mut self) -> Result<()> {
        let valid_chunk_ids: HashSet<String> = self.state.chunks.keys().cloned().collect();

        self.lexical_index.drop_stale(&valid_chunk_ids);

        for chunk_id in &valid_chunk_ids {
            if let Some(chunk) = self.state.chunks.get(chunk_id)
                && !self.lexical_index.contains(chunk_id)
            {
                #[cfg(feature = "tracing")]
                tracing::debug!("Re-adding missing chunk {} to lexical index", chunk_id);
                self.lexical_index.add_chunk(chunk_id, &chunk.text);
            }
        }

        if self.ann_index.is_none()
            && !self.state.chunks.is_empty()
            && let Some(first_chunk) = self.state.chunks.values().next()
        {
            let dim = first_chunk.embedding.len();
            if dim > 0 {
                self.ann_index = Some(AnnIndex::new(dim));
            }
        }

        if let Some(ann_index) = &mut self.ann_index {
            ann_index.drop_stale(&valid_chunk_ids);

            for chunk_id in &valid_chunk_ids {
                if let Some(chunk) = self.state.chunks.get(chunk_id)
                    && !ann_index.contains(chunk_id)
                {
                    #[cfg(feature = "tracing")]
                    tracing::debug!("Re-adding missing chunk {} to ANN index", chunk_id);
                    ann_index.insert(chunk_id, &chunk.embedding);
                }
            }
        }

        let valid_documents: HashSet<String> = self
            .state
            .chunks
            .values()
            .map(|chunk| chunk.document_name.clone())
            .collect();
        self.state.document_hashes.retain(|doc_name, _| {
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
        let embedding_model_id = backend.model_id().to_string();
        let embedding_dim = backend.dimension();
        Self {
            backend,
            reranker,
            config,
            state: EngineState::new(embedding_model_id, embedding_dim),
            ann_index: None,
            lexical_index: LexicalIndex::new(),
        }
    }

    pub fn with_reranker(backend: B, reranker: R) -> Self {
        let embedding_model_id = backend.model_id().to_string();
        let embedding_dim = backend.dimension();
        Self {
            backend,
            reranker: Some(reranker),
            config: RagConfig::default(),
            state: EngineState::new(embedding_model_id, embedding_dim),
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
    pub fn load_state(&mut self, mut state: crate::persistence::EngineState) -> Result<()> {
        let current_model = self.backend.model_id();
        if state.embedding_model_id != current_model {
            #[cfg(feature = "tracing")]
            tracing::warn!(
                persisted_model = %state.embedding_model_id,
                current_model = %current_model,
                "Persisted state model mismatch. Marking for reindex."
            );
            self.state.needs_reindex = true;
            return Ok(());
        }

        if state.schema_version < crate::persistence::INDEX_VERSION {
            self.state.chunks.clear();
            self.state.document_hashes.clear();
            self.state.needs_reindex = true;
            self.state.schema_version = crate::persistence::INDEX_VERSION;
            self.state.embedding_model_id = current_model.to_string();
            return Ok(());
        }

        if state.embedding_dim == 0 && !state.chunks.is_empty() {
            state.embedding_dim = state
                .chunks
                .values()
                .next()
                .map(|c| c.embedding.len())
                .unwrap_or(0);
        }

        for chunk in state.chunks.values_mut() {
            crate::search::normalize(&mut chunk.embedding);
        }

        self.state = state;

        if self.state.document_hashes.is_empty() && !self.state.chunks.is_empty() {
            self.state.needs_reindex = true;
        }

        self.ann_index = None;
        self.lexical_index = LexicalIndex::new();
        self.validate_index_sync()?;
        Ok(())
    }

    pub fn save_to_dir(&self, data_dir: impl AsRef<std::path::Path>) -> Result<()> {
        use crate::persistence::{JsonFileBackend, PersistenceBackend};

        let backend = JsonFileBackend::new(data_dir.as_ref(), self.backend.model_id());
        backend
            .save(&self.state)
            .map_err(|e| EngineError::save_failed(backend.index_path(), e))?;
        Ok(())
    }

    pub fn load_from_dir(&mut self, data_dir: impl AsRef<std::path::Path>) -> Result<()> {
        use crate::persistence::{JsonFileBackend, PersistenceBackend};
        use serde::Deserialize;

        let data_dir = data_dir.as_ref();
        let current_model = self.backend.model_id();
        let backend = JsonFileBackend::new(data_dir, current_model);

        let model_specific_path = backend.index_path();
        let legacy_path = backend.legacy_path();
        let model_specific_exists = model_specific_path.exists();
        let legacy_exists = legacy_path.exists();

        let loaded = backend
            .load()
            .map_err(|e| EngineError::load_failed(model_specific_path.clone(), e))?;

        let Some(mut state) = loaded else {
            if model_specific_exists {
                #[cfg(feature = "tracing")]
                tracing::warn!(
                    "Failed to parse model-specific index at {:?}. Marking for reindex.",
                    model_specific_path
                );
                self.state.needs_reindex = true;
                return Ok(());
            }

            if legacy_exists {
                #[derive(Deserialize)]
                struct ModelOnly {
                    model: String,
                }

                match std::fs::read_to_string(&legacy_path) {
                    Ok(data) => {
                        if let Ok(info) = serde_json::from_str::<ModelOnly>(&data) {
                            if info.model == current_model {
                                #[cfg(feature = "tracing")]
                                tracing::warn!(
                                    "Failed to parse legacy index at {:?}. Marking for reindex.",
                                    legacy_path
                                );
                                self.state.needs_reindex = true;
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
                            && !legacy_chunks.is_empty()
                        {
                            #[cfg(feature = "tracing")]
                            tracing::warn!(
                                "Found legacy chunks without model info. Reindex required for model '{}'.",
                                current_model
                            );
                            self.state.needs_reindex = true;
                        }
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::InvalidData => {
                        #[cfg(feature = "tracing")]
                        tracing::warn!(
                            "Legacy index at {:?} contains invalid UTF-8. Marking for reindex.",
                            legacy_path
                        );
                        self.state.needs_reindex = true;
                    }
                    Err(e) => {
                        return Err(EngineError::load_failed(
                            &legacy_path,
                            crate::error::PersistenceError::Io(e),
                        ));
                    }
                }
            }

            return Ok(());
        };

        // If we successfully parsed state but it belongs to another model, treat it as incompatible.
        if state.embedding_model_id != current_model {
            #[cfg(feature = "tracing")]
            tracing::warn!(
                persisted_model = %state.embedding_model_id,
                current_model = %current_model,
                "Persisted state model mismatch. Marking for reindex."
            );
            self.state.needs_reindex = true;
            return Ok(());
        }

        // Invalidate state if it is older than what we support.
        if state.schema_version < crate::persistence::INDEX_VERSION {
            self.state.chunks.clear();
            self.state.document_hashes.clear();
            self.state.needs_reindex = true;
            self.state.schema_version = crate::persistence::INDEX_VERSION;
            self.state.embedding_model_id = current_model.to_string();
            self.save_to_dir(data_dir)?;
            return Ok(());
        }

        if state.embedding_dim == 0 && !state.chunks.is_empty() {
            state.embedding_dim = state
                .chunks
                .values()
                .next()
                .map(|c| c.embedding.len())
                .unwrap_or(0);
        }

        self.state = state;

        for chunk in self.state.chunks.values_mut() {
            crate::search::normalize(&mut chunk.embedding);
        }

        if self.state.document_hashes.is_empty() && !self.state.chunks.is_empty() {
            self.state.needs_reindex = true;
        }

        // Rebuild in-memory indexes from persisted chunk state.
        self.ann_index = None;
        self.lexical_index = LexicalIndex::new();
        self.validate_index_sync()?;

        // If we loaded from the legacy path, write out the model-specific file.
        if !model_specific_exists && legacy_exists {
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

    #[tokio::test]
    async fn test_search_stability_tie_breaker() {
        let mut engine = RagEngine::new(MockBackend);

        let docs = vec!["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"];
        // Text must be > 10 chars
        let content = "This is a sufficiently long text content to be indexed by the engine.";

        for doc in &docs {
            engine.upsert_document(doc, content, None).await.unwrap();
        }

        let results = engine.search(content, 10, None).await.unwrap();
        assert_eq!(results.len(), 10);

        for i in 0..results.len() - 1 {
            let current = &results[i];
            let next = &results[i + 1];

            assert!(
                (current.score - next.score).abs() < f32::EPSILON,
                "Scores differ: {} vs {}",
                current.score,
                next.score
            );

            assert!(
                current.chunk_id < next.chunk_id,
                "Results not sorted by ID at index {}: {} comes before {} with same score {}",
                i,
                current.chunk_id,
                next.chunk_id,
                current.score
            );
        }
    }

    #[tokio::test]
    async fn test_search_truncation_optimization() {
        let mut engine = RagEngine::new(MockBackend);

        // Add 20 documents
        for i in 0..20 {
            engine
                .upsert_document(
                    &format!("doc{}.txt", i),
                    "Cats are great pets.",
                    Some(format!("h{}", i)),
                )
                .await
                .unwrap();
        }

        // Search for top 5.
        // This triggers the path where candidates (20) > top_k (5).
        // It ensures that sorting/truncating of reference-based `scores` works correctly
        // before converting to owned `SearchCandidate`s.
        let results = engine.search("cats", 5, None).await.unwrap();

        assert_eq!(results.len(), 5);
    }

    #[tokio::test]
    async fn test_embedding_candidates_stability() {
        let mut engine = RagEngine::new(MockBackend);

        // Insert documents that will have identical embeddings ("cats" -> [1.0, 0.0])
        for i in 0..10 {
            engine
                .upsert_document(&format!("doc{}.txt", i), "Cats are great pets.", None)
                .await
                .unwrap();
        }

        // Get all candidates
        let candidates = engine.embedding_candidates("cats", 10).await.unwrap();
        assert_eq!(candidates.len(), 10);

        // Verify deterministic ordering: Score DESC, then ChunkID ASC
        for i in 0..candidates.len() - 1 {
            let current = &candidates[i];
            let next = &candidates[i + 1];

            // Scores should be identical for this test case
            assert!(
                (current.initial_score - next.initial_score).abs() < f32::EPSILON,
                "Scores should be identical"
            );

            // IDs should be sorted strictly ASC
            assert!(
                current.chunk_id < next.chunk_id,
                "Candidates not sorted by ID at index {}: {} vs {}",
                i,
                current.chunk_id,
                next.chunk_id
            );
        }
    }

    #[tokio::test]
    async fn test_prepare_document_timeout() {
        // Slow backend that sleeps longer than the timeout
        struct SlowBackend;
        impl EmbeddingBackend for SlowBackend {
            fn model_id(&self) -> &str {
                "slow-model"
            }
            fn dimension(&self) -> usize {
                2
            }
            async fn embed(&self, _: &str) -> std::result::Result<Vec<f32>, EmbeddingError> {
                Ok(vec![0.0, 0.0])
            }
            async fn embed_batch(
                &self,
                texts: &[String],
            ) -> std::result::Result<Vec<Vec<f32>>, EmbeddingError> {
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                Ok(vec![vec![0.0, 0.0]; texts.len()])
            }
        }

        let config = RagConfig {
            chunk_tokens: 10,
            sentence_overlap: 0,
            embedding_batch_size: 1, // Force multiple batches if text is long enough
            document_prep_timeout: Some(std::time::Duration::from_millis(50)),
            ..Default::default()
        };

        let engine = RagEngine::with_config(SlowBackend, config);

        // Create enough text to generate multiple chunks/batches
        // Since batch_size is 1, and we have multiple sentences, we should get multiple batches.
        // The first batch will sleep 100ms.
        // The second batch check should see > 50ms elapsed and fail.
        // OR the first batch itself takes 100ms, and we return.
        // Wait, the check is BEFORE the batch.
        // 1. Check time (0ms) -> OK.
        // 2. Embed batch (sleep 100ms).
        // 3. Loop.
        // 4. Check time (100ms > 50ms) -> Fail.
        let text = "Sentence 1. Sentence 2. Sentence 3. Sentence 4. Sentence 5. Sentence 6. Sentence 7. Sentence 8. Sentence 9. Sentence 10.";

        let result = engine.prepare_document("doc.txt", text, None, None).await;

        match result {
            Err(EngineError::Embedding(EmbeddingError::Timeout(t))) => {
                assert_eq!(t, std::time::Duration::from_millis(50));
            }
            Err(e) => panic!("Expected Timeout error, got {:?}", e),
            Ok(res) => panic!(
                "Expected Timeout error, got Ok. Generated {} chunks.",
                res.map(|d| d.chunks.len()).unwrap_or(0)
            ),
        }
    }

    #[tokio::test]
    async fn test_upsert_prepared_document_atomic_failure() {
        let mut engine = RagEngine::new(MockBackend);

        // 1. Setup: Insert a valid document "doc1"
        let valid_chunk = DocumentChunk {
            id: "chunk1".to_string(),
            document_name: "doc1".to_string(),
            text: "Valid content".to_string(),
            embedding: vec![1.0, 0.0],
            chunk_index: 0,
            page_number: 1,
            section: None,
            metadata: crate::types::ChunkMetadata::default(),
            tags: std::collections::HashSet::new(),
            resolution: crate::types::Resolution::default(),
            parent_id: None,
        };

        let prepared_valid = PreparedDocument {
            document_name: "doc1".to_string(),
            document_hash: Some("hash1".to_string()),
            chunks: vec![valid_chunk],
        };

        engine.upsert_prepared_document(prepared_valid).unwrap();

        assert_eq!(engine.chunk_count(), 1);
        assert_eq!(engine.list_documents(), vec!["doc1".to_string()]);

        // 2. Attempt to update "doc1" with mixed valid/invalid chunks
        let mixed_chunks = vec![
            DocumentChunk {
                id: "chunk2".to_string(),
                document_name: "doc1".to_string(),
                text: "New valid content".to_string(),
                embedding: vec![0.0, 1.0], // Valid dim (2)
                chunk_index: 0,
                page_number: 1,
                section: None,
                metadata: crate::types::ChunkMetadata::default(),
                tags: std::collections::HashSet::new(),
                resolution: crate::types::Resolution::default(),
                parent_id: None,
            },
            DocumentChunk {
                id: "chunk3".to_string(),
                document_name: "doc1".to_string(),
                text: "Invalid content".to_string(),
                embedding: vec![1.0, 0.0, 0.0], // Invalid dim (3)
                chunk_index: 1,
                page_number: 1,
                section: None,
                metadata: crate::types::ChunkMetadata::default(),
                tags: std::collections::HashSet::new(),
                resolution: crate::types::Resolution::default(),
                parent_id: None,
            },
        ];

        let prepared_mixed = PreparedDocument {
            document_name: "doc1".to_string(),
            document_hash: Some("hash2".to_string()),
            chunks: mixed_chunks,
        };

        // This should fail due to dimension mismatch
        let result = engine.upsert_prepared_document(prepared_mixed);
        assert!(result.is_err());

        match result {
            Err(EngineError::Validation {
                kind: crate::error::ValidationKind::DimensionMismatch { .. },
                ..
            }) => {}
            _ => panic!("Expected Validation::DimensionMismatch error"),
        }

        // 3. Verify INVARIANT: The engine state should be unchanged
        // If atomic, we should still have the original chunk1, and NOT chunk2 or chunk3.
        assert_eq!(engine.chunk_count(), 1, "Chunk count should remain 1");
        assert_eq!(
            engine.list_documents(),
            vec!["doc1".to_string()],
            "Document should still exist"
        );

        let results = engine.search("Valid", 1, None).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].chunk_id, "chunk1");
    }
}
