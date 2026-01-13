//! Example: Saving and loading engine state.
//!
//! This example demonstrates how to persist the RAG engine
//! to disk and restore it later.
//!
//! Run with: `cargo run --example persistence -p rag-core --features persistence`

#[cfg(feature = "persistence")]
mod example {
    use rag_core::{
        EmbeddingBackend, EmbeddingError, JsonFileBackend, PersistenceBackend, RagEngine,
    };

    /// Simple embedder for demonstration.
    struct MockEmbedder {
        dim: usize,
    }

    impl EmbeddingBackend for MockEmbedder {
        fn model_id(&self) -> &str {
            "mock-v1"
        }

        async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
            let hash = text
                .bytes()
                .fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
            let mut emb = vec![0.0f32; self.dim];
            for (i, v) in emb.iter_mut().enumerate() {
                *v = ((hash.wrapping_add(i as u64) % 1000) as f32 / 1000.0) - 0.5;
            }
            let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                emb.iter_mut().for_each(|v| *v /= norm);
            }
            Ok(emb)
        }

        fn dimension(&self) -> usize {
            self.dim
        }
    }

    pub async fn run() -> Result<(), Box<dyn std::error::Error>> {
        let temp_dir = std::env::temp_dir().join("rag-core-example");
        std::fs::create_dir_all(&temp_dir)?;
        println!("Using temp directory: {:?}\n", temp_dir);

        // === Part 1: Create and populate engine ===
        println!("=== Creating and populating engine ===");
        {
            let embedder = MockEmbedder { dim: 64 };
            let mut engine = RagEngine::new(embedder);

            // Index some documents
            let docs = [
                (
                    "physics.txt",
                    "Quantum mechanics describes the behavior of particles at atomic scales.",
                ),
                (
                    "biology.txt",
                    "DNA contains the genetic instructions for all living organisms.",
                ),
                (
                    "history.txt",
                    "The Roman Empire was one of the largest empires in ancient history.",
                ),
            ];

            for (name, content) in &docs {
                if let Some(prepared) = engine.prepare_document(name, content, None, None).await? {
                    engine.upsert_prepared_document(prepared)?;
                    println!("  Indexed: {}", name);
                }
            }

            println!("  Total chunks: {}", engine.chunk_count());

            // Save to disk
            engine.save_to_dir(&temp_dir)?;
            println!("\n  Saved engine state to disk.");
        }

        // === Part 2: Load engine from disk ===
        println!("\n=== Loading engine from disk ===");
        {
            let embedder = MockEmbedder { dim: 64 };
            let mut engine = RagEngine::new(embedder);

            // Load persisted state
            engine.load_from_dir(&temp_dir)?;
            println!(
                "  Loaded {} documents, {} chunks",
                engine.list_documents().len(),
                engine.chunk_count()
            );

            // Search to verify data is intact
            println!("\n  Searching for 'atomic particles':");
            let results = engine.search("atomic particles", 2, None).await?;
            for r in &results {
                println!("    {} (score: {:.3})", r.document, r.score);
            }
        }

        // === Part 3: Using JsonFileBackend directly ===
        println!("\n=== Using JsonFileBackend directly ===");
        {
            let backend = JsonFileBackend::new(&temp_dir, "mock-v1");
            println!("  Index file: {:?}", backend.index_path());

            if let Some(state) = backend.load()? {
                println!("  Loaded state:");
                println!("    Schema version: {}", state.schema_version);
                println!("    Model: {}", state.embedding_model_id);
                println!("    Chunks: {}", state.chunks.len());
            }
        }

        // Cleanup
        std::fs::remove_dir_all(&temp_dir)?;
        println!("\nCleaned up temp directory.");

        Ok(())
    }
}

#[cfg(feature = "persistence")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Persistence Example ===\n");
    example::run().await
}

#[cfg(not(feature = "persistence"))]
fn main() {
    eprintln!("This example requires the 'persistence' feature.");
    eprintln!("Run with: cargo run --example persistence -p rag-core --features persistence");
}
