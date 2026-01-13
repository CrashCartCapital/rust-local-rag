# PRD: rust-local-rag Library Extraction

**Date**: 2025-12-31
**Status**: Final Draft
**Author**: Claude (AI-assisted)
**MSRV**: Rust 1.75+ (for native async traits / RPITIT)
**Edition**: 2024

---

## Executive Summary

Extract the core RAG functionality from `rust-local-rag` into a reusable Rust library (`rag-core`) that other projects can embed to add local RAG capabilities without requiring the MCP server layer.

**Goal**: Enable modular, portable RAG functionality that can be integrated into existing Rust projects with minimal coupling.

---

## 1. Requirements

### 1.1 Problem Statement

The current `rust-local-rag` is a monolithic MCP server. Other projects wanting RAG functionality must either:
1. Copy-paste components (fragile, no updates)
2. Run rust-local-rag as a separate service (operational overhead)
3. Reimplement from scratch (wasted effort)

### 1.2 Success Criteria

| Criterion | Measure |
|-----------|---------|
| **Embeddable** | Any Rust project can add RAG via `rag-core = "0.1"` in Cargo.toml |
| **Pluggable Embeddings** | Swap Ollama for OpenAI, HuggingFace, or custom embedders |
| **Optional Persistence** | In-memory only mode for ephemeral use cases |
| **Minimal Dependencies** | Core crate compiles without heavy deps (no reqwest/sqlx unless opted in) |
| **Backward Compatible** | See detailed compatibility criteria below |

#### Backward Compatibility Definition

The existing MCP server must remain fully functional after extraction:

| Aspect | Compatibility Requirement | Validation |
|--------|---------------------------|------------|
| **MCP API Surface** | All 6 tools unchanged: `search_documents`, `list_documents`, `get_stats`, `start_reindex`, `get_job_status`, `calibrate_reranker` | MCP integration tests pass |
| **Persistence Format** | Existing `chunks_{model}.json` files readable | Load existing index, search returns identical results |
| **Search Behavior** | Same scoring algorithm (embedding + reranker weights) | Eval harness: Hit Rate@5 within ±2% of baseline (77.8%) |
| **CLI Behavior** | `rust-local-rag` binary starts and serves MCP | Smoke test: start server, call search |
| **Environment Variables** | All existing env vars honored | Config test with current `.env` |

### 1.3 Non-Goals (Explicitly Out of Scope)

- Enterprise features (multi-tenancy, auth, rate limiting)
- Cloud deployment abstractions
- Web UI or REST API in core
- Support for non-Rust languages (FFI, WASM bindings)
- Complex plugin architectures

---

## 2. Design

### 2.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Consumer Projects                     │
│  (your-app, another-tool, etc.)                         │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                      rag-core                            │
│  ┌─────────────────┐  ┌──────────────┐  ┌────────────┐  │
│  │  RagEngine      │  │ Chunking     │  │ Search     │  │
│  │  (orchestrator) │  │ (sentence-   │  │ (cosine +  │  │
│  │                 │  │  aware)      │  │  MMR)      │  │
│  └─────────────────┘  └──────────────┘  └────────────┘  │
│                                                          │
│  Traits:                                                 │
│  ┌─────────────────┐  ┌──────────────┐                  │
│  │ EmbeddingBackend│  │ Reranker     │  (optional)      │
│  │ (required)      │  │ (optional)   │                  │
│  └─────────────────┘  └──────────────┘                  │
└─────────────────────────────────────────────────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         ▼                 ▼                 ▼
┌─────────────┐   ┌─────────────┐   ┌─────────────────────┐
│rag-ollama  │   │rag-persist │   │ rust-local-rag      │
│(impl crate)│   │(impl crate)│   │ (MCP binary, thin   │
│            │   │            │   │  wrapper over core) │
└─────────────┘   └─────────────┘   └─────────────────────┘
```

### 2.2 Crate Structure (Minimal Workspace)

**Phase 1 (MVP)**:
```
rust-local-rag/
├── Cargo.toml              # Workspace root
├── crates/
│   └── rag-core/           # Core library
│       ├── Cargo.toml
│       └── src/
│           ├── lib.rs
│           ├── engine.rs       # RagEngine
│           ├── chunking.rs     # Sentence-aware chunking
│           ├── search.rs       # Cosine similarity + MMR
│           ├── traits.rs       # EmbeddingBackend trait
│           └── types.rs        # DocumentChunk, SearchResult, etc.
└── src/                    # Existing binary (becomes thin wrapper)
    └── main.rs
```

**Phase 2 (When Needed)**:
```
crates/
├── rag-core/               # Core library
├── rag-ollama/             # Ollama EmbeddingBackend impl
├── rag-persist/            # SQLite/JSON persistence
└── rag-pdf/                # PDF extraction (lopdf)
```

### 2.3 Core Trait: EmbeddingBackend

This is the **only essential trait** for library consumers. Everything else stays concrete with configuration options.

```rust
// crates/rag-core/src/traits.rs

use std::future::Future;

/// Embedding backend trait for generating vector embeddings from text.
///
/// Implementors provide the actual embedding computation (Ollama, OpenAI, etc.)
pub trait EmbeddingBackend: Send + Sync {
    /// Unique identifier for this model (used for persistence partitioning)
    fn model_id(&self) -> &str;

    /// Embed a single text. Prefer `embed_batch` for multiple texts.
    fn embed(&self, text: &str) -> impl Future<Output = Result<Vec<f32>, EmbeddingError>> + Send;

    /// Embed multiple texts in a batch (more efficient).
    /// Default implementation calls `embed` sequentially.
    fn embed_batch(&self, texts: &[String]) -> impl Future<Output = Result<Vec<Vec<f32>>, EmbeddingError>> + Send {
        async {
            let mut results = Vec::with_capacity(texts.len());
            for text in texts {
                results.push(self.embed(text).await?);
            }
            Ok(results)
        }
    }

    /// Embedding dimension (for validation)
    fn dimension(&self) -> usize;
}

#[derive(Debug, thiserror::Error)]
pub enum EmbeddingError {
    #[error("Connection failed: {0}")]
    Connection(String),
    #[error("Model not found: {0}")]
    ModelNotFound(String),
    #[error("API error: {0}")]
    Api(String),
    #[error("Timeout after {0:?}")]
    Timeout(std::time::Duration),
}
```

### 2.4 RagEngine API

```rust
// crates/rag-core/src/engine.rs

/// Core RAG engine, generic over embedding backend.
/// Optional second type parameter R for reranker (defaults to () for no reranking).
pub struct RagEngine<B: EmbeddingBackend, R = ()> {
    backend: B,
    reranker: Option<R>,
    chunks: HashMap<String, DocumentChunk>,
    document_hashes: HashMap<String, String>,  // SHA-256 for change detection
    config: RagConfig,
}

impl<B: EmbeddingBackend> RagEngine<B, ()> {
    /// Create a new in-memory RAG engine (no reranking)
    pub fn new(backend: B) -> Self { ... }

    /// Create with custom configuration (no reranking)
    pub fn with_config(backend: B, config: RagConfig) -> Self { ... }
}

impl<B: EmbeddingBackend, R: Rerank> RagEngine<B, R> {
    /// Create with reranker
    pub fn with_reranker(backend: B, reranker: R) -> Self { ... }
}

impl<B: EmbeddingBackend, R> RagEngine<B, R> {
    /// Add a document (text already extracted). Returns chunk count.
    pub async fn add_document(&mut self, name: &str, text: &str) -> Result<usize, EngineError> { ... }

    /// Update or insert a document (replaces if exists)
    pub async fn upsert_document(&mut self, name: &str, text: &str) -> Result<usize, EngineError> { ... }

    /// Remove a document and its chunks
    pub fn remove_document(&mut self, name: &str) -> Result<usize, EngineError> { ... }

    /// Search for relevant chunks (embedding similarity only)
    pub async fn search(&self, query: &str, top_k: usize) -> Result<Vec<SearchResult>, EngineError> { ... }

    /// Search with MMR diversification
    pub async fn search_diverse(&self, query: &str, top_k: usize, diversity: f32) -> Result<Vec<SearchResult>, EngineError> { ... }

    /// List indexed documents
    pub fn documents(&self) -> Vec<&str> { ... }

    /// Get statistics
    pub fn stats(&self) -> EngineStats { ... }
}

// Rerank-specific methods only available when R implements Rerank
impl<B: EmbeddingBackend, R: Rerank> RagEngine<B, R> {
    /// Search with reranking (embedding + LLM reranker)
    pub async fn search_reranked(&self, query: &str, top_k: usize) -> Result<Vec<SearchResult>, EngineError> { ... }
}

#[derive(Default, Clone)]
pub struct RagConfig {
    pub chunk_size_chars: usize,     // Default: 2000 chars (~500 tokens)
    pub chunk_overlap_chars: usize,  // Default: 200 chars (~50 tokens)
    pub similarity_threshold: f32,   // Default: 0.0 (no threshold)
}
// Note: Sizing is in characters, not tokens. Approximate token count = chars / 4.
// This avoids requiring a tokenizer dependency in core.
```

### 2.5 Optional Reranker (Concrete Type, No Trait)

Reranking is optional. Most users will use the default Ollama-based reranker or none.

**Design Decision**: Reranker is a **concrete type**, not a trait object. This avoids object-safety issues with async methods (`-> impl Future` is not object-safe for `Box<dyn Trait>`).

```rust
// In rag-ollama crate (not rag-core)
pub struct OllamaReranker {
    client: reqwest::Client,
    model: String,
    ollama_url: String,
}

impl OllamaReranker {
    pub async fn new(ollama_url: &str, model: &str) -> Result<Self, RerankerError> { ... }

    pub async fn rerank(&self, query: &str, candidates: Vec<RerankerCandidate>)
        -> Result<Vec<RankedResult>, RerankerError> { ... }
}

// RagEngine in rag-core is generic over an optional reranker type
pub struct RagEngine<B: EmbeddingBackend, R = ()> {
    backend: B,
    reranker: Option<R>,
    // ...
}

// Search method conditionally uses reranker if present
impl<B: EmbeddingBackend, R: Rerank> RagEngine<B, R> {
    pub async fn search_with_rerank(&self, query: &str, top_k: usize) -> Result<Vec<SearchResult>> {
        // Uses self.reranker if Some
    }
}

// Simple Rerank trait (object-safe version if needed later)
pub trait Rerank: Send + Sync {
    fn rerank<'a>(&'a self, query: &'a str, candidates: Vec<RerankerCandidate>)
        -> Pin<Box<dyn Future<Output = Result<Vec<RankedResult>, RerankerError>> + Send + 'a>>;
}
```

**Pragmatic Path**: For MVP, make reranker a second generic parameter with a simple bound. Only introduce `async_trait` if we need true dynamic dispatch later.

### 2.6 Error Handling Strategy

**Library Error Types** (not `anyhow`):

```rust
// crates/rag-core/src/error.rs

#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("Embedding failed: {0}")]
    Embedding(#[from] EmbeddingError),

    #[error("Document not found: {0}")]
    DocumentNotFound(String),

    #[error("Persistence error: {0}")]
    Persistence(String),

    #[error("Invalid configuration: {0}")]
    Config(String),
}

pub type Result<T> = std::result::Result<T, EngineError>;
```

**Rationale**: Typed errors allow library consumers to handle specific cases. Use `anyhow` only at binary/app boundaries (MCP server).

### 2.7 Feature Flags

```toml
# crates/rag-core/Cargo.toml
[features]
default = []
persistence = ["serde", "serde_json"]  # Enable save/load to disk
tracing = ["tracing"]                  # Structured logging hooks
```

No heavy dependencies (reqwest, sqlx, lopdf) in core. These live in implementation crates.

### 2.8 Persistence Format & Versioning

```rust
// Persistence schema (when `persistence` feature enabled)
#[derive(Serialize, Deserialize)]
pub struct PersistedIndex {
    pub version: u32,                              // Schema version (start at 2 to match existing)
    pub model: String,                             // Embedding model identifier
    pub chunks: HashMap<String, DocumentChunk>,
    pub document_hashes: HashMap<String, String>,  // SHA-256 hashes
}
```

**Migration Strategy**:
- Version 2 = current `chunks_{model}.json` format (maintained for backward compat)
- Future versions: add migration functions in persistence module
- Breaking changes increment major version

### 2.9 Code Location Matrix

What stays in the MCP server (`src/`) vs moves to `rag-core`:

| Component | Location | Rationale |
|-----------|----------|-----------|
| `DocumentChunk`, `SearchResult`, `ChunkMetadata` | `rag-core` | Core domain types |
| Sentence-aware chunking | `rag-core` | Core algorithm |
| Cosine similarity, MMR | `rag-core` | Core search logic |
| `EmbeddingBackend` trait | `rag-core` | Pluggability interface |
| `RagEngine<B, R>` | `rag-core` | Main orchestrator |
| Persistence (save/load) | `rag-core` (feature) | Optional |
| `OllamaEmbedder` | `rag-ollama` | Implementation |
| `OllamaReranker` | `rag-ollama` | Implementation |
| PDF extraction | `rag-pdf` or server | Implementation |
| `JobManager`, `WorkerSupervisor` | server (`src/`) | MCP-specific async job handling |
| `RagMcpServer` | server (`src/`) | MCP protocol layer |
| Health endpoints (`/healthz`, `/readyz`) | server (`src/`) | HTTP concerns |
| Progress logging | server (`src/`) | Job-specific |
| Per-document locking, `TimedWriteLockGuard` | server (`src/`) | Concurrency for MCP |

---

## 3. Tasks

### Phase 1: Core Library Extraction (MVP)

| # | Task | Test Criteria |
|---|------|---------------|
| 1.1 | Create workspace structure with `crates/rag-core` | `cargo build -p rag-core` succeeds |
| 1.2 | Extract types: `DocumentChunk`, `ChunkMetadata`, `SearchResult` | Unit tests pass for serialization |
| 1.3 | Define `EmbeddingBackend` trait with associated `Future` | Compiles with mock impl |
| 1.4 | Extract chunking logic (sentence-aware) to `chunking.rs` | Unit tests: chunk boundaries at sentences |
| 1.5 | Extract search logic (cosine + MMR) to `search.rs` | Unit tests: correct ranking, MMR diversification |
| 1.6 | Create `RagEngine<B>` struct with core methods | Integration test with mock backend |
| 1.7 | Add `persistence` feature for save/load | Test: save, reload, search returns same results |
| 1.8 | Refactor existing `src/` to use `rag-core` | All existing tests pass |

### Phase 2: Implementation Crates (When Needed)

| # | Task | Test Criteria |
|---|------|---------------|
| 2.1 | Create `rag-ollama` crate with `OllamaEmbedder` | Integration test against Ollama |
| 2.2 | Create `rag-pdf` crate with PDF extraction | Test: extract text from sample PDFs |
| 2.3 | Create `rag-persist` crate with SQLite backend | Test: persist and reload index |

### Phase 3: Documentation & Examples

| # | Task | Test Criteria |
|---|------|---------------|
| 3.1 | Write `rag-core` README with quick start | Builds in example project |
| 3.2 | Add `examples/` directory with common use cases | Examples compile and run |
| 3.3 | Document trait implementations for custom backends | Doctest passes |

---

## 4. API Usage Examples

### 4.1 Basic Usage (In-Memory)

```rust
use rag_core::{RagEngine, EmbeddingBackend};
use rag_ollama::OllamaEmbedder;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Create embedder (Ollama, OpenAI, or custom)
    let embedder = OllamaEmbedder::new("http://localhost:11434", "nomic-embed-text").await?;

    // Create RAG engine
    let mut engine = RagEngine::new(embedder);

    // Add documents
    engine.add_document("doc1.txt", "Machine learning is a subset of AI...").await?;
    engine.add_document("doc2.txt", "Deep learning uses neural networks...").await?;

    // Search
    let results = engine.search("What is machine learning?", 5).await?;
    for result in results {
        println!("[{:.2}] {}: {}", result.score, result.document, result.text);
    }

    Ok(())
}
```

### 4.2 Custom Embedding Backend

```rust
use rag_core::{EmbeddingBackend, EmbeddingError};

struct MyCustomEmbedder {
    model: String,
    client: reqwest::Client,
}

impl EmbeddingBackend for MyCustomEmbedder {
    fn model_id(&self) -> &str {
        &self.model
    }

    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        // Your embedding logic here
        todo!()
    }

    fn dimension(&self) -> usize {
        768 // Your model's dimension
    }
}
```

### 4.3 With Persistence

```rust
use rag_core::{RagEngine, RagConfig};

// Save to disk
engine.save("./my_index.json")?;

// Load from disk
let engine = RagEngine::load("./my_index.json", embedder)?;
```

---

## 5. Migration Path

### For Existing rust-local-rag

1. **No breaking changes** to MCP server functionality
2. Binary in `src/main.rs` becomes thin wrapper:
   ```rust
   use rag_core::RagEngine;
   use rag_ollama::OllamaEmbedder;

   // Initialize engine
   let embedder = OllamaEmbedder::from_env().await?;
   let engine = RagEngine::with_persistence(embedder, &data_dir).await?;

   // MCP server wraps engine
   let mcp_server = RagMcpServer::new(engine, job_manager);
   ```

### For Projects Importing Components

Instead of copy-pasting from rust-local-rag:
```toml
[dependencies]
rag-core = { path = "../rust-local-rag/crates/rag-core" }
rag-ollama = { path = "../rust-local-rag/crates/rag-ollama" }
```

Or eventually (if published):
```toml
[dependencies]
rag-core = "0.1"
rag-ollama = "0.1"
```

---

## 6. Design Decisions & Rationale

### Why Single Essential Trait (EmbeddingBackend)?

- **Codex recommendation**: "Make traits only where you truly need pluggability"
- Embedding provider is the main variability axis (Ollama today, OpenAI tomorrow)
- Chunking, search, MMR are algorithmic choices better served by config, not traits

### Why Workspace over Single Crate with Features?

- **Gemini recommendation**: "Workspace structure facilitates easier testing and downstream embedding"
- Heavy dependencies (reqwest, lopdf, sqlx) isolated to implementation crates
- Consumers only compile what they need

### Why No DocumentLoader/Parser Trait Initially?

- **Pragmatic start**: Most users will call `add_document(name, text)` with pre-extracted text
- PDF extraction is a separate concern - provide `rag-pdf` crate for those who need it
- Avoids coupling core to file I/O

### Why Associated Future Instead of async_trait?

- **Codex recommendation**: "Use GAT/RPITIT-style returned futures when you can keep RagEngine generic"
- Avoids boxing overhead for performance-sensitive embedding calls
- Rust 2024 edition supports this natively

---

## 7. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Over-engineering with too many traits | Medium | High | Start minimal, add abstractions only when 2+ impls exist |
| Breaking existing MCP server | Low | High | Extensive test coverage before refactor |
| Feature flag complexity | Medium | Medium | Keep features orthogonal, test common combinations in CI |
| API churn during extraction | Medium | Low | Mark 0.x versions as unstable |

---

## 8. Technical Decisions (Resolved)

| Question | Decision | Rationale |
|----------|----------|-----------|
| **Publish to crates.io?** | No (path deps) | Personal use; avoid maintenance burden |
| **Async runtime coupling?** | Tokio-native | RAG is I/O heavy; Tokio is de facto standard |
| **Error types?** | Typed (`thiserror`) | Library consumers need specific handling |
| **MSRV?** | Rust 1.75+ | Required for native async traits (RPITIT) |

## 9. Open Questions (For Implementation)

1. **Metadata filtering?** - Add filter API for document attributes (date, source)?
2. **Observability hooks?** - Expose metrics (search latency, embedding token count)?
3. **Streaming results?** - Support `Stream<Item=SearchResult>` for large result sets?

---

## Appendix A: Consultant Feedback Summary

### Initial Design Review

#### Gemini (gemini-3-flash-preview) - Approved

**Strengths Identified**:
- Modern Rust async design using RPITIT
- Excellent separation of concerns
- Pragmatic phased approach
- Strong focus on testability

**Recommendations Adopted**:
- ✅ Add tracing integration (section 2.7)
- ✅ EmbeddingBackend with batch_size hint consideration
- ✅ Document update/remove operations (section 2.4)

#### Codex (GPT-5.2-xhigh) - Needs Revision → Fixed

**Issues Identified & Fixes Applied**:

| Issue | Fix |
|-------|-----|
| Object safety: `Box<dyn Reranker>` with `impl Future` won't compile | Changed to generic `R: Rerank` with object-safe fallback (section 2.5) |
| "Backward compatible" not measurable | Added detailed compatibility table (section 1.2) |
| Chunk sizing in "tokens" ambiguous | Changed to characters with note (section 2.4) |
| Missing upsert/remove operations | Added `upsert_document`, `remove_document` (section 2.4) |
| Error handling underspecified | Added typed errors with `thiserror` (section 2.6) |
| Persistence versioning missing | Added schema version strategy (section 2.8) |
| MSRV/edition undefined | Added to header (Rust 1.75+, Edition 2024) |
| Code location unclear | Added Code Location Matrix (section 2.9) |

### Final Review Status

Both consultants' concerns have been addressed. PRD is ready for implementation.
