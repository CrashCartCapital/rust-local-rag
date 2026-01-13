//! Example: Real-world Ollama integration.
//!
//! This example demonstrates how to implement `EmbeddingBackend` for Ollama,
//! including proper error mapping from HTTP client errors.
//!
//! ## Prerequisites
//!
//! 1. Install Ollama: https://ollama.ai
//! 2. Pull an embedding model: `ollama pull nomic-embed-text`
//! 3. Start Ollama: `ollama serve`
//!
//! ## Dependencies
//!
//! Add to your Cargo.toml:
//! ```toml
//! [dependencies]
//! rag-core = "0.1"
//! reqwest = { version = "0.12", features = ["json"] }
//! serde = { version = "1.0", features = ["derive"] }
//! tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
//! ```
//!
//! Run with: `cargo run --example ollama_backend -p rag-core`
//!
//! Note: This example uses a mock embedder since rag-core doesn't depend on reqwest.
//! Copy the `OllamaEmbedder` implementation below into your project with reqwest.

use rag_core::{EmbeddingBackend, EmbeddingError, RagEngine};

// =============================================================================
// COPY THIS INTO YOUR PROJECT (requires reqwest dependency)
// =============================================================================
//
// use reqwest::Client;
// use serde::Deserialize;
//
// pub struct OllamaEmbedder {
//     client: Client,
//     model: String,
//     base_url: String,
//     dimension: usize,
// }
//
// impl OllamaEmbedder {
//     pub fn new(model: impl Into<String>, dimension: usize) -> Self {
//         Self {
//             client: Client::builder()
//                 .timeout(Duration::from_secs(30))
//                 .build()
//                 .expect("Failed to create HTTP client"),
//             model: model.into(),
//             base_url: "http://localhost:11434".to_string(),
//             dimension,
//         }
//     }
//
//     pub fn with_url(mut self, url: impl Into<String>) -> Self {
//         self.base_url = url.into();
//         self
//     }
// }
//
// #[derive(Deserialize)]
// struct OllamaResponse {
//     embedding: Vec<f32>,
// }
//
// impl EmbeddingBackend for OllamaEmbedder {
//     fn model_id(&self) -> &str {
//         &self.model
//     }
//
//     fn dimension(&self) -> usize {
//         self.dimension
//     }
//
//     async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
//         let url = format!("{}/api/embeddings", self.base_url);
//
//         let response = self.client
//             .post(&url)
//             .json(&serde_json::json!({
//                 "model": self.model,
//                 "prompt": text,
//             }))
//             .send()
//             .await
//             .map_err(|e| {
//                 // Map reqwest errors to EmbeddingError variants
//                 if e.is_timeout() {
//                     EmbeddingError::Timeout(Duration::from_secs(30))
//                 } else if e.is_connect() {
//                     EmbeddingError::Connection(format!(
//                         "Cannot connect to Ollama at {}: {}",
//                         self.base_url, e
//                     ))
//                 } else {
//                     EmbeddingError::Api(format!("HTTP request failed: {}", e))
//                 }
//             })?;
//
//         // Check HTTP status
//         if !response.status().is_success() {
//             let status = response.status();
//             let body = response.text().await.unwrap_or_default();
//             return Err(EmbeddingError::Api(format!(
//                 "Ollama returned {}: {}",
//                 status, body
//             )));
//         }
//
//         // Parse JSON response
//         let body: OllamaResponse = response.json().await.map_err(|e| {
//             EmbeddingError::Api(format!("Invalid JSON response: {}", e))
//         })?;
//
//         Ok(body.embedding)
//     }
// }
// =============================================================================

/// Mock embedder for demonstration (no external dependencies).
/// Replace with OllamaEmbedder above for real usage.
struct MockOllamaEmbedder {
    model: String,
    dimension: usize,
}

impl MockOllamaEmbedder {
    fn new(model: impl Into<String>, dimension: usize) -> Self {
        Self {
            model: model.into(),
            dimension,
        }
    }
}

impl EmbeddingBackend for MockOllamaEmbedder {
    fn model_id(&self) -> &str {
        &self.model
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        // Simulate what Ollama would return
        println!("  [MockOllama] Would call: POST http://localhost:11434/api/embeddings");
        println!("  [MockOllama] Model: {}, Text length: {} chars", self.model, text.len());

        // Generate deterministic embedding for demo
        let hash = text.bytes().fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
        let mut embedding = vec![0.0f32; self.dimension];
        for (i, v) in embedding.iter_mut().enumerate() {
            *v = ((hash.wrapping_add(i as u64) % 1000) as f32 / 1000.0) - 0.5;
        }

        // Normalize
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            embedding.iter_mut().for_each(|v| *v /= norm);
        }

        Ok(embedding)
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Ollama Backend Example ===\n");
    println!("This example shows how to integrate Ollama with rag-core.");
    println!("See the commented code above for a real implementation.\n");

    // Use mock for this example (replace with OllamaEmbedder for real usage)
    // let embedder = OllamaEmbedder::new("nomic-embed-text", 768);
    let embedder = MockOllamaEmbedder::new("nomic-embed-text", 768);

    let mut engine = RagEngine::new(embedder);
    println!("Created RagEngine with {} embedder\n", engine.embedding_model());

    // Index documents
    let docs = [
        ("quantum.txt", "Quantum computing uses qubits that can exist in superposition."),
        ("ml.txt", "Machine learning models learn patterns from training data."),
    ];

    for (name, content) in &docs {
        if let Some(prepared) = engine.prepare_document(name, content, None, None).await? {
            engine.upsert_prepared_document(prepared)?;
            println!("Indexed: {}", name);
        }
    }

    // Search
    println!("\nSearching for 'quantum superposition':");
    let results = engine.search("quantum superposition", 2, None).await?;
    for (i, r) in results.iter().enumerate() {
        println!("  {}. {} (score: {:.3})", i + 1, r.document, r.score);
    }

    println!("\n=== Error Handling Demo ===\n");
    println!("The OllamaEmbedder above maps errors like this:");
    println!("  - Connection refused  → EmbeddingError::Connection");
    println!("  - Request timeout     → EmbeddingError::Timeout");
    println!("  - HTTP 4xx/5xx        → EmbeddingError::Api");
    println!("  - Invalid JSON        → EmbeddingError::Api");

    Ok(())
}
