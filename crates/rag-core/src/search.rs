use crate::types::SearchResult;
use std::collections::{HashMap, HashSet};

/// Normalize a vector to unit length in-place.
/// If the vector has zero or very small norm, it is left unchanged.
pub(crate) fn normalize(v: &mut [f32]) {
    let norm_sq: f32 = v.iter().map(|x| x * x).sum();
    if norm_sq > 1e-20 {
        let norm = norm_sq.sqrt();
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Calculate dot product between two vectors.
/// Assumes vectors are of the same length (or truncates to shorter length).
/// If vectors are normalized, this is equivalent to cosine similarity.
#[inline(always)]
pub(crate) fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Calculate cosine similarity between two embeddings.
/// Returns a value in [-1, 1] where 1 means identical direction.
/// Returns 0.0 for edge cases (empty, mismatched length, near-zero norm).
#[allow(dead_code)] // used by tests and legacy paths
pub(crate) fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    const EPSILON: f32 = 1e-10;

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a < EPSILON || norm_b < EPSILON {
        0.0
    } else {
        (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct SearchResultWithEmbedding {
    pub(crate) result: SearchResult,
    pub(crate) embedding: Vec<f32>,
}

pub(crate) fn mmr_diversify(
    candidates: Vec<SearchResultWithEmbedding>,
    top_k: usize,
    diversity_factor: f32,
) -> Vec<SearchResult> {
    if candidates.is_empty() {
        return vec![];
    }

    let mut selected: Vec<SearchResultWithEmbedding> = Vec::with_capacity(top_k);
    let mut remaining: Vec<SearchResultWithEmbedding> = candidates;

    if !remaining.is_empty() {
        let first = remaining.swap_remove(0);
        selected.push(first);
    }

    while selected.len() < top_k && !remaining.is_empty() {
        let mut best_mmr_score = f32::NEG_INFINITY;
        let mut best_idx = 0;

        for (idx, candidate) in remaining.iter().enumerate() {
            let relevance = candidate.result.score;
            if !relevance.is_finite() {
                continue;
            }

            let max_similarity = selected
                .iter()
                .map(|s| dot_product(&candidate.embedding, &s.embedding))
                .filter(|sim| sim.is_finite())
                .fold(0.0_f32, |a, b| a.max(b));

            let mmr_score =
                (1.0 - diversity_factor) * relevance - diversity_factor * max_similarity;

            if mmr_score.is_finite() && mmr_score > best_mmr_score {
                best_mmr_score = mmr_score;
                best_idx = idx;
            }
        }

        if best_mmr_score == f32::NEG_INFINITY {
            #[cfg(feature = "tracing")]
            tracing::warn!("MMR: No valid candidates remaining (all scores non-finite)");
            break;
        }

        let best = remaining.swap_remove(best_idx);

        #[cfg(feature = "tracing")]
        tracing::debug!(
            chunk_id = %best.result.chunk_id,
            relevance = %best.result.score,
            mmr_score = %best_mmr_score,
            "MMR selected result"
        );

        selected.push(best);
    }

    selected.into_iter().map(|s| s.result).collect()
}

struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> f32 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let bits = (self.state >> 32) as u32;
        let value = bits as f32 / u32::MAX as f32;
        value * 2.0 - 1.0
    }
}

const NUM_HYPERPLANES: usize = 32;
const MAX_SINGLE_BIT_NEIGHBORS: usize = 32;
const MAX_TOTAL_NEIGHBORS: usize = 64;

pub(crate) struct AnnIndex {
    dim: usize,
    hyperplanes: Vec<Vec<f32>>,
    buckets: HashMap<u64, Vec<String>>,
    id_to_bucket: HashMap<String, u64>,
}

impl AnnIndex {
    pub(crate) fn new(dim: usize) -> Self {
        let mut rng = SimpleRng::new(42);
        let mut hyperplanes = Vec::with_capacity(NUM_HYPERPLANES);

        for _ in 0..NUM_HYPERPLANES {
            let mut plane = Vec::with_capacity(dim);
            for _ in 0..dim {
                plane.push(rng.next());
            }
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

    pub(crate) fn insert(&mut self, id: &str, vector: &[f32]) {
        if vector.len() != self.dim {
            #[cfg(feature = "tracing")]
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

    pub(crate) fn remove(&mut self, id: &str) {
        if let Some(hash) = self.id_to_bucket.remove(id) {
            if let Some(bucket) = self.buckets.get_mut(&hash) {
                bucket.retain(|stored| stored != id);
                if bucket.is_empty() {
                    self.buckets.remove(&hash);
                }
            }
        }
    }

    pub(crate) fn search(&self, vector: &[f32], max_candidates: usize) -> Vec<String> {
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

    pub(crate) fn contains(&self, id: &str) -> bool {
        self.id_to_bucket.contains_key(id)
    }

    pub(crate) fn drop_stale(&mut self, valid_ids: &HashSet<String>) {
        let current_ids: HashSet<String> = self.id_to_bucket.keys().cloned().collect();
        for stale_id in current_ids.difference(valid_ids) {
            self.remove(stale_id);
        }
    }
}

#[derive(Default)]
pub(crate) struct LexicalIndex {
    term_postings: HashMap<String, HashMap<String, usize>>,
    doc_lengths: HashMap<String, usize>,
    doc_terms: HashMap<String, HashMap<String, usize>>,
    total_docs: usize,
    total_length: usize,
}

impl LexicalIndex {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn add_chunk(&mut self, id: &str, text: &str) {
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

    pub(crate) fn remove_chunk(&mut self, id: &str) {
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
                self.total_length = self.total_length.saturating_sub(length);
            } else if self.total_docs == 0 {
                self.total_length = 0;
            }
            self.total_docs = self.total_docs.saturating_sub(1);
        } else {
            self.doc_lengths.remove(id);
        }

        if self.total_docs == 0 {
            self.total_length = 0;
        }
    }

    pub(crate) fn score(&self, query: &str, limit: usize) -> Vec<(String, f32)> {
        if self.total_docs == 0 {
            return Vec::new();
        }

        let tokens = tokenize(query);
        if tokens.is_empty() {
            return Vec::new();
        }

        let unique_terms: HashSet<String> = tokens.into_iter().collect();

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

    pub(crate) fn contains(&self, id: &str) -> bool {
        self.doc_terms.contains_key(id)
    }

    pub(crate) fn drop_stale(&mut self, valid_ids: &HashSet<String>) {
        let current_ids: HashSet<String> = self.doc_terms.keys().cloned().collect();
        for stale_id in current_ids.difference(valid_ids) {
            self.remove_chunk(stale_id);
        }
    }
}

/// Tokenizes text into lowercase terms for lexical indexing.
/// Filters out tokens shorter than 3 characters to reduce noise and memory usage.
fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|token| token.len() >= 3)
        .map(|token| token.to_lowercase())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lexical_index_contains_and_drop_stale() {
        let mut index = LexicalIndex::new();

        index.add_chunk("chunk1", "hello world");
        index.add_chunk("chunk2", "foo bar baz");
        index.add_chunk("chunk3", "test document");

        assert!(index.contains("chunk1"));
        assert!(index.contains("chunk2"));
        assert!(index.contains("chunk3"));
        assert!(!index.contains("chunk4"));

        let valid_ids: HashSet<String> = vec!["chunk1".to_string(), "chunk2".to_string()]
            .into_iter()
            .collect();

        index.drop_stale(&valid_ids);

        assert!(index.contains("chunk1"));
        assert!(index.contains("chunk2"));
        assert!(
            !index.contains("chunk3"),
            "chunk3 should have been removed as stale"
        );
    }

    #[test]
    fn test_ann_index_contains_and_drop_stale() {
        let mut ann_index = AnnIndex::new(384);

        let vec1: Vec<f32> = (0..384).map(|i| (i as f32) / 384.0).collect();
        let vec2: Vec<f32> = (0..384).map(|i| ((i + 100) as f32) / 384.0).collect();
        let vec3: Vec<f32> = (0..384).map(|i| ((i + 200) as f32) / 384.0).collect();

        ann_index.insert("id1", &vec1);
        ann_index.insert("id2", &vec2);
        ann_index.insert("id3", &vec3);

        assert!(ann_index.contains("id1"));
        assert!(ann_index.contains("id2"));
        assert!(ann_index.contains("id3"));
        assert!(!ann_index.contains("id4"));

        let valid_ids: HashSet<String> = vec!["id1".to_string(), "id3".to_string()]
            .into_iter()
            .collect();

        ann_index.drop_stale(&valid_ids);

        assert!(ann_index.contains("id1"));
        assert!(!ann_index.contains("id2"), "id2 should have been removed");
        assert!(ann_index.contains("id3"));
    }

    fn build_candidate(id: &str, score: f32, embedding: Vec<f32>) -> SearchResultWithEmbedding {
        SearchResultWithEmbedding {
            result: SearchResult {
                text: format!("text-{id}"),
                score,
                document: "doc".to_string(),
                chunk_id: id.to_string(),
                chunk_index: 0,
                page_number: 1,
                section: None,
                embedding_score: None,
                lexical_score: None,
                initial_score: None,
                reranker_score: None,
                yes_logprob: None,
                no_logprob: None,
            },
            embedding,
        }
    }

    #[test]
    fn test_mmr_diversify_empty_candidates() {
        let result = mmr_diversify(vec![], 5, 0.3);
        assert!(result.is_empty());
    }

    #[test]
    fn test_mmr_diversify_single_candidate() {
        let candidates = vec![build_candidate("a", 1.0, vec![1.0, 0.0])];
        let result = mmr_diversify(candidates, 5, 0.3);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].chunk_id, "a");
    }

    #[test]
    fn test_mmr_diversify_top_k_larger_than_candidates() {
        let candidates = vec![
            build_candidate("a", 1.0, vec![1.0, 0.0]),
            build_candidate("b", 0.9, vec![0.0, 1.0]),
        ];
        let result = mmr_diversify(candidates, 10, 0.3);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_mmr_diversify_zero_diversity_factor() {
        let candidates = vec![
            build_candidate("a", 1.0, vec![1.0, 0.0]),
            build_candidate("b", 0.9, vec![1.0, 0.0]),
            build_candidate("c", 0.8, vec![1.0, 0.0]),
        ];
        let result = mmr_diversify(candidates, 3, 0.0);
        let ids: Vec<String> = result.into_iter().map(|r| r.chunk_id).collect();
        assert_eq!(ids, vec!["a".to_string(), "b".to_string(), "c".to_string()]);
    }

    #[test]
    fn test_mmr_diversify_high_diversity_factor() {
        let candidates = vec![
            build_candidate("a", 1.0, vec![1.0, 0.0]),
            build_candidate("b", 0.9, vec![1.0, 0.0]),
            build_candidate("c", 0.8, vec![0.0, 1.0]),
        ];
        let result = mmr_diversify(candidates, 2, 0.9);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].chunk_id, "a");
        assert_eq!(result[1].chunk_id, "c");
    }

    #[test]
    fn test_mmr_diversify_nan_score_handling() {
        let candidates = vec![
            build_candidate("a", 1.0, vec![1.0, 0.0]),
            build_candidate("b", f32::NAN, vec![0.0, 1.0]),
            build_candidate("c", 0.8, vec![0.0, 1.0]),
        ];
        let result = mmr_diversify(candidates, 3, 0.3);
        let ids: Vec<String> = result.into_iter().map(|r| r.chunk_id).collect();
        assert!(!ids.contains(&"b".to_string()));
        assert!(ids.contains(&"a".to_string()));
        assert!(ids.contains(&"c".to_string()));
    }

    #[test]
    fn test_mmr_diversify_inf_score_handling() {
        let candidates = vec![
            build_candidate("a", 1.0, vec![1.0, 0.0]),
            build_candidate("b", f32::INFINITY, vec![0.0, 1.0]),
            build_candidate("c", 0.8, vec![0.0, 1.0]),
        ];
        let result = mmr_diversify(candidates, 3, 0.3);
        let ids: Vec<String> = result.into_iter().map(|r| r.chunk_id).collect();
        assert!(!ids.contains(&"b".to_string()));
        assert!(ids.contains(&"a".to_string()));
        assert!(ids.contains(&"c".to_string()));
    }

    #[test]
    fn test_mmr_diversify_preserves_relevance_order_when_orthogonal() {
        let candidates = vec![
            build_candidate("a", 1.0, vec![1.0, 0.0]),
            build_candidate("b", 0.9, vec![0.0, 1.0]),
            build_candidate("c", 0.8, vec![-1.0, 0.0]),
            build_candidate("d", 0.7, vec![0.0, -1.0]),
        ];
        let result = mmr_diversify(candidates, 4, 0.3);
        let ids: Vec<String> = result.into_iter().map(|r| r.chunk_id).collect();
        assert_eq!(
            ids,
            vec![
                "a".to_string(),
                "b".to_string(),
                "c".to_string(),
                "d".to_string()
            ]
        );
    }

    #[test]
    fn test_mmr_formula_correctness() {
        let candidates = vec![
            build_candidate("a", 1.0, vec![1.0, 0.0]),
            build_candidate("b", 0.8, vec![1.0, 0.0]),
            build_candidate("c", 0.7, vec![0.0, 1.0]),
        ];
        let result = mmr_diversify(candidates, 2, 0.5);
        let ids: Vec<String> = result.into_iter().map(|r| r.chunk_id).collect();
        assert_eq!(ids, vec!["a".to_string(), "c".to_string()]);
    }
}
