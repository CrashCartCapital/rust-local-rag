//! Basic usage example for rag-core.
//!
//! This example demonstrates:
//! - Implementing a simple EmbeddingBackend
//! - Creating a RagEngine
//! - Indexing documents
//! - Performing search
//!
//! Run with: `cargo run --example basic_usage -p rag-core`

use rag_core::{EmbeddingBackend, EmbeddingError, RagConfig, RagEngine};

/// A mock embedding backend that produces deterministic embeddings.
///
/// In a real application, this would call an embedding service like Ollama,
/// OpenAI, or a local HuggingFace model.
struct MockEmbedder {
    dim: usize,
}

impl MockEmbedder {
    fn new(dimension: usize) -> Self {
        Self { dim: dimension }
    }

    /// Generate a deterministic embedding based on text hash.
    fn hash_embed(&self, text: &str) -> Vec<f32> {
        let mut embedding = vec![0.0f32; self.dim];

        // Simple hash-based embedding for demonstration
        let hash = text
            .bytes()
            .fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));

        for (i, val) in embedding.iter_mut().enumerate() {
            let h = hash.wrapping_add(i as u64);
            *val = ((h % 1000) as f32 / 1000.0) - 0.5;
        }

        // Normalize to unit length
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for val in &mut embedding {
                *val /= norm;
            }
        }

        embedding
    }
}

impl EmbeddingBackend for MockEmbedder {
    fn model_id(&self) -> &str {
        "mock-embedder-v1"
    }

    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        Ok(self.hash_embed(text))
    }

    fn dimension(&self) -> usize {
        self.dim
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== rag-core Basic Usage Example ===\n");

    // 1. Create the embedding backend
    let embedder = MockEmbedder::new(384); // 384-dim like many real models

    // 2. Configure and create the engine
    let config = RagConfig {
        chunk_tokens: 200,        // tokens per chunk (approx 4 chars/token)
        sentence_overlap: 1,      // sentences to overlap between chunks
        ..Default::default()      // use defaults for weights and batch_size
    };

    let mut engine = RagEngine::with_config(embedder, config);
    println!("Created RagEngine with mock embedder\n");

    // 3. Index some documents
    let documents = vec![
        (
            "rust_intro.txt",
            "Rust is a systems programming language focused on safety, speed, and concurrency. \
             It achieves memory safety without garbage collection. The borrow checker is a key \
             feature that prevents data races at compile time.",
        ),
        (
            "python_intro.txt",
            "Python is a high-level programming language known for its readability and simplicity. \
             It uses dynamic typing and garbage collection. Python is popular for web development, \
             data science, and machine learning applications.",
        ),
        (
            "go_intro.txt",
            "Go, also known as Golang, is a statically typed language designed at Google. \
             It features garbage collection, structural typing, and CSP-style concurrency. \
             Go is often used for cloud services and command-line tools.",
        ),
    ];

    for (name, content) in &documents {
        // prepare_document returns Option<PreparedDocument> - None if unchanged
        if let Some(prepared) = engine.prepare_document(name, content, None, None).await? {
            engine.upsert_prepared_document(prepared)?;
            println!("Indexed: {name}");
        }
    }
    println!();

    // 4. Perform searches
    let queries = vec![
        "memory safety without garbage collection",
        "dynamic typing and machine learning",
        "cloud services and concurrency",
    ];

    for query in queries {
        println!("Query: \"{query}\"");
        // search takes query, top_k, and optional weights
        let results = engine.search(query, 2, None).await?;

        for (i, result) in results.iter().enumerate() {
            println!(
                "  {}. {} (score: {:.3})",
                i + 1,
                result.document,
                result.score
            );
        }
        println!();
    }

    // 5. Show engine stats
    let chunk_count = engine.chunk_count();
    let doc_count = engine.list_documents().len();
    println!("Engine stats: {doc_count} documents, {chunk_count} chunks");

    Ok(())
}
