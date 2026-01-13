//! Example: Implementing a custom EmbeddingBackend.
//!
//! This example shows how to create your own embedding backend
//! that integrates with external services like Ollama or OpenAI.
//!
//! Run with: `cargo run --example custom_backend -p rag-core`

use rag_core::{EmbeddingBackend, EmbeddingError, RagEngine};
use std::collections::HashMap;
use std::sync::RwLock;

/// A caching embedding backend that wraps another backend.
///
/// This pattern is useful for reducing API calls to expensive
/// embedding services by caching results.
struct CachingEmbedder<B: EmbeddingBackend> {
    inner: B,
    cache: RwLock<HashMap<String, Vec<f32>>>,
}

impl<B: EmbeddingBackend> CachingEmbedder<B> {
    fn new(inner: B) -> Self {
        Self {
            inner,
            cache: RwLock::new(HashMap::new()),
        }
    }

    #[allow(dead_code)]
    fn cache_size(&self) -> usize {
        self.cache.read().unwrap().len()
    }
}

impl<B: EmbeddingBackend> EmbeddingBackend for CachingEmbedder<B> {
    fn model_id(&self) -> &str {
        self.inner.model_id()
    }

    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        // Check cache first
        if let Some(cached) = self.cache.read().unwrap().get(text) {
            return Ok(cached.clone());
        }

        // Cache miss - call inner backend
        let embedding = self.inner.embed(text).await?;

        // Store in cache
        self.cache
            .write()
            .unwrap()
            .insert(text.to_string(), embedding.clone());

        Ok(embedding)
    }

    fn dimension(&self) -> usize {
        self.inner.dimension()
    }
}

/// A simple deterministic embedder for demonstration.
struct SimpleEmbedder {
    dim: usize,
}

impl SimpleEmbedder {
    fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl EmbeddingBackend for SimpleEmbedder {
    fn model_id(&self) -> &str {
        "simple-embedder"
    }

    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        // Simulate some work
        println!("  [SimpleEmbedder] Computing embedding for: {}...", &text[..text.len().min(30)]);

        // Create deterministic embedding from text hash
        let hash = text.bytes().fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
        let mut embedding = vec![0.0f32; self.dim];

        for (i, val) in embedding.iter_mut().enumerate() {
            let h = hash.wrapping_add(i as u64);
            *val = ((h % 1000) as f32 / 1000.0) - 0.5;
        }

        // Normalize
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for val in &mut embedding {
                *val /= norm;
            }
        }

        Ok(embedding)
    }

    fn dimension(&self) -> usize {
        self.dim
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Custom Backend Example ===\n");

    // Create a caching wrapper around our simple embedder
    let inner = SimpleEmbedder::new(128);
    let embedder = CachingEmbedder::new(inner);

    let mut engine = RagEngine::new(embedder);

    // Index documents
    let docs = [
        ("doc1.txt", "Rust is a systems programming language."),
        ("doc2.txt", "Python is great for data science."),
    ];

    for (name, content) in &docs {
        if let Some(prepared) = engine.prepare_document(name, content, None, None).await? {
            engine.upsert_prepared_document(prepared)?;
        }
    }

    println!("\nCache entries after indexing: {}", engine.embedding_model());

    // Search - first call computes embedding
    println!("\nFirst search (cache miss):");
    let _ = engine.search("systems programming", 2, None).await?;

    // Search again - should use cached embedding
    println!("\nSecond search (cache hit):");
    let results = engine.search("systems programming", 2, None).await?;

    println!("\nResults:");
    for r in &results {
        println!("  {} (score: {:.3})", r.document, r.score);
    }

    Ok(())
}
