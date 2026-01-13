# rag-core

A batteries-included-but-swappable embedded RAG (Retrieval-Augmented Generation) engine for Rust.

[![Crates.io](https://img.shields.io/crates/v/rag-core.svg)](https://crates.io/crates/rag-core)
[![Documentation](https://docs.rs/rag-core/badge.svg)](https://docs.rs/rag-core)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Philosophy

**Local-first, privacy-focused retrieval.** rag-core runs entirely on your machine with no external API calls required. Your documents stay local, embeddings are computed locally, and search happens in-memory.

## Features

- **Async/Await**: Full async support using Tokio for non-blocking I/O
- **Pluggable Embedding Backends**: Bring your own embedding model via the `EmbeddingBackend` trait
- **Sentence-Aware Chunking**: Intelligent text chunking that respects sentence boundaries
- **Hybrid Search**: Combined embedding similarity + lexical (BM25-style) scoring
- **MMR Diversification**: Maximal Marginal Relevance for diverse search results
- **Optional Reranking**: Second-stage LLM-based relevance reranking
- **Persistence**: Save/load engine state to disk with atomic writes
- **Robust Error Handling**: Structured errors with full context for debugging

## Quick Start

Add rag-core to your `Cargo.toml`:

```toml
[dependencies]
rag-core = "0.1"
```

### Minimal Example

```rust
use rag_core::{RagEngine, EmbeddingBackend, EmbeddingError};

// 1. Implement your embedding backend
struct MyEmbedder;

impl EmbeddingBackend for MyEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        // Your embedding logic here (e.g., call Ollama, HuggingFace, etc.)
        Ok(vec![0.1, 0.2, 0.3]) // Placeholder 3-dim embedding
    }

    fn model_id(&self) -> &str {
        "my-embedder-v1"
    }

    fn dimension(&self) -> usize {
        3
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 2. Create the engine
    let embedder = MyEmbedder;
    let mut engine = RagEngine::new(embedder);

    // 3. Index a document
    if let Some(prepared) = engine.prepare_document("doc.txt", "Your document text here.", None, None).await? {
        engine.upsert_prepared_document(prepared)?;
    }

    // 4. Search
    let results = engine.search("query text", 5, None).await?;
    for result in results {
        println!("{}: {:.3}", result.document, result.score);
    }
    Ok(())
}
```

See the [`examples/`](examples/) directory for more complete examples.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        RagEngine                            │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐  │
│  │ EmbeddingBackend│  │    IndexSet     │  │  Reranker   │  │
│  │   (pluggable)   │  │ (ANN + Lexical) │  │ (optional)  │  │
│  └─────────────────┘  └─────────────────┘  └─────────────┘  │
│  ┌─────────────────────────────────────────────────────────┐│
│  │                  PersistenceBackend                     ││
│  │               (JsonFileBackend default)                 ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

### Core Components

| Component | Description |
|-----------|-------------|
| `RagEngine` | Main entry point. Coordinates indexing, search, and persistence. |
| `EmbeddingBackend` | Trait for embedding providers. Implement to integrate any model. |
| `IndexSet` | Unified index wrapping ANN (approximate nearest neighbor) and lexical search. |
| `PersistenceBackend` | Trait for storage. Default `JsonFileBackend` uses atomic file writes. |
| `Rerank` | Optional trait for second-stage LLM-based reranking. |

## Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `persistence` | Off | Enables `JsonFileBackend`, `EngineState`, and disk save/load |
| `tracing` | Off | Adds `tracing` instrumentation spans |

Enable features in your `Cargo.toml`:

```toml
[dependencies]
rag-core = { version = "0.1", features = ["persistence", "tracing"] }
```

## API Reference

### Document Processing

```rust
// Prepare a document into chunks (returns Option<PreparedDocument>)
// Args: name, text, optional_hash, optional_callback
let prepared = engine.prepare_document("name.pdf", "full text content", None, None).await?;

// Index the prepared chunks (if document changed)
if let Some(prepared) = prepared {
    engine.upsert_prepared_document(prepared)?;
}

// Remove a document
engine.remove_document("name.pdf")?;
```

### Search

```rust
// Basic search (top_k results, optional weights)
let results = engine.search("query", 10, None).await?;

// Search with custom weights (embedding vs lexical blend)
let weights = SearchWeights {
    embedding: 0.7,  // Weight for embedding similarity
    lexical: 0.3,    // Weight for lexical/BM25 matching
    ..Default::default()
};
let results = engine.search("query", 10, Some(weights)).await?;
```

### Persistence

**Requires the `persistence` feature**:

```toml
[dependencies]
rag-core = { version = "0.1", features = ["persistence"] }
```

```rust
// Save engine state to directory
engine.save_to_dir("./data")?;

// Load engine state from directory
engine.load_from_dir("./data")?;
```

For advanced use (custom backends), use the `PersistenceBackend` trait directly:

```rust
use rag_core::{JsonFileBackend, PersistenceBackend};

let backend = JsonFileBackend::new("./data", engine.embedding_model());
if let Some(state) = backend.load()? {
    // Access raw state for inspection
    println!("Loaded {} chunks", state.chunks.len());
}
```

## Error Handling

rag-core uses structured errors with full context:

```rust
use rag_core::{EngineError, ValidationKind, PersistenceOp};

match engine.upsert_prepared_document("doc.txt", chunks) {
    Ok(_) => println!("Indexed successfully"),
    Err(EngineError::Validation { chunk_id, kind }) => {
        match kind {
            ValidationKind::NaN => eprintln!("Embedding contains NaN"),
            ValidationKind::DimensionMismatch { expected, got } => {
                eprintln!("Dimension mismatch: expected {expected}, got {got}");
            }
            _ => eprintln!("Validation error: {kind}"),
        }
    }
    Err(EngineError::Persistence { path, operation, source }) => {
        eprintln!("Persistence {operation} failed at {path:?}: {source}");
    }
    Err(e) => eprintln!("Error: {e}"),
}
```

## Input Validation

rag-core validates all inputs to prevent garbage-in-garbage-out:

- **NaN/Inf Detection**: Embeddings containing NaN or Inf values are rejected
- **Dimension Enforcement**: All embeddings must match the engine's dimension
- **Empty Text Rejection**: Chunks with empty or whitespace-only text are rejected
- **Atomic Batch Operations**: Batch updates validate all items before committing any

## Privacy & Security

- **No Network Calls**: rag-core itself makes no network requests
- **Local Storage**: All data stays on disk in your chosen directory
- **No Telemetry**: Zero analytics or tracking
- **Atomic Writes**: Persistence uses temp-file-then-rename to prevent corruption

## Observability

### Health Check

```rust
let health = engine.health();
println!("Model: {}, Chunks: {}, Healthy: {}",
    health.embedding_model, health.chunk_count, health.is_healthy);
```

### Tracing

Enable the `tracing` feature for instrumentation spans:

```toml
[dependencies]
rag-core = { version = "0.1", features = ["tracing"] }
```

Major operations emit tracing spans compatible with `tracing-subscriber`:

```rust
use tracing_subscriber::prelude::*;

tracing_subscriber::registry()
    .with(tracing_subscriber::fmt::layer())
    .init();

// Now rag-core operations will emit spans
```

## Concurrency

`RagEngine` is **not** thread-safe by default. For concurrent access:

```rust
use std::sync::{Arc, RwLock};

let engine = Arc::new(RwLock::new(RagEngine::new(embedder, config)));

// Read access (concurrent)
let results = engine.read().unwrap().search("query", 10)?;

// Write access (exclusive)
engine.write().unwrap().upsert_prepared_document("doc.txt", chunks)?;
```

## Target Use Cases

| Persona | Use Case |
|---------|----------|
| **CLI Toolsmith** | Embed rag-core in command-line tools for log analysis, code search |
| **Desktop App Dev** | Privacy-focused apps with Tauri/Iced for local document retrieval |
| **Microservice Architect** | Embed specialized RAG in microservices with custom backends |

## Minimum Supported Rust Version

rag-core requires **Rust 1.85+** (Edition 2024).

Check your version: `rustc --version`

## License

MIT License. See [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please see the main repository for guidelines.
