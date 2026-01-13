# Product Requirements Document: rag-core Library Portability

**Version**: 1.1
**Date**: 2026-01-12
**Authors**: Claude Code (AI-assisted with Gemini/Codex/CRASH ensemble validation)
**Status**: Ready for Implementation

---

## Executive Summary

This PRD specifies the requirements, design, and implementation tasks to transform `rag-core` from an internal extraction into a production-ready, externally-adoptable Rust library for local-first RAG (Retrieval-Augmented Generation).

### Vision

`rag-core` will be a **batteries-included-but-swappable** embedded RAG engine for Rust developers who need:
- Local, privacy-first document retrieval
- Minimal external dependencies
- Pluggable embedding backends and storage
- Production-grade reliability

### Target Personas

| Persona | Description | Key Needs |
|---------|-------------|-----------|
| **CLI Toolsmith** | Builds command-line utilities for dev productivity, log analysis | Minimal deps, static binaries, simple config |
| **Desktop App Dev** | Creates privacy-focused apps with Tauri/Iced | Offline capability, low memory, cross-platform |
| **Microservice Architect** | Embeds RAG in specialized services | Thread safety, observability, customizable storage |

### Anti-Requirements (What We Won't Do)

| Exclusion | Rationale |
|-----------|-----------|
| Built-in HTTP server | Library, not service—consumers choose their web framework |
| Heavy vector DB bindings | Local-first focus—external adapters can be separate crates |
| Python/JS bindings | Focus on Rust API ergonomics first; FFI adds complexity |
| Agent/orchestration framework | Focus on retrieval mechanics, not execution chains |

---

# PART 1: REQUIREMENTS

## R1: Documentation & Adoption (Phase 1 Priority)

### R1.1: Crate README (MUST)

**Problem**: No README exists in `crates/rag-core/`.

**Requirement**: Create comprehensive README.md with:
1. Introduction & local-first philosophy
2. Quick Start (5-minute Hello World)
3. Core architecture overview (Engine, Backends, Reranking)
4. Feature flags reference
5. Installation instructions (cargo, git dependency)
6. Privacy & security guarantees

**Acceptance Criteria**:
- [ ] README renders correctly on GitHub and crates.io
- [ ] Quick Start example compiles and runs
- [ ] All feature flags documented with use cases

### R1.2: Rustdoc Coverage (MUST)

**Problem**: Zero doc comments on public types/functions.

**Requirement**: All public API surface MUST have:
1. Module-level `//!` documentation with usage examples
2. Function-level `///` documentation with:
   - Description of purpose
   - Parameter explanations
   - Return value documentation
   - `# Examples` section with runnable code
   - `# Panics` section where applicable
3. Intra-doc links for all trait implementations

**Acceptance Criteria**:
- [ ] `cargo doc --no-deps` produces complete documentation
- [ ] All public items have doc comments
- [ ] Doc tests pass: `cargo test --doc`

### R1.3: Usage Examples (MUST)

**Problem**: No examples directory; users must reverse-engineer from main crate.

**Requirement**: Create `examples/` directory with:
1. `basic_usage.rs` - Minimal RagEngine setup with mock backend
2. `custom_backend.rs` - Implementing EmbeddingBackend trait
3. `search_with_reranking.rs` - Full search flow with optional reranker
4. `persistence.rs` - Save/load engine state

**Acceptance Criteria**:
- [ ] All examples compile: `cargo build --examples`
- [ ] Examples are referenced in README
- [ ] Examples demonstrate common use cases per persona

### R1.4: Distribution Setup (MUST)

**Problem**: `publish = false` blocks cargo distribution.

**Requirement**:
1. Set `publish = true` in Cargo.toml (or explicitly document git dependency approach)
2. Complete crate metadata:
   - `license = "MIT"`
   - `authors`
   - `repository`
   - `documentation`
   - `categories` and `keywords`
3. README integration for docs.rs

**Acceptance Criteria**:
- [ ] Crate passes `cargo publish --dry-run`
- [ ] Metadata visible on crates.io preview
- [ ] docs.rs compatible

### R1.5: API Stabilization (MUST)

**Problem**: Internal types may be accidentally exposed; no stability guarantees.

**Requirement**:
1. Audit all `pub` exports in `lib.rs`
2. Ensure internal types remain `pub(crate)`
3. Define "happy path" API: `RagEngine::new()`, `prepare_document()`, `upsert_prepared_document()`, `search()`, `save_to_dir()`, `load_from_dir()`
4. Document any unstable APIs with `#[doc(hidden)]` or feature gates

**Acceptance Criteria**:
- [ ] Only intentional types are publicly exported
- [ ] Breaking change surface is minimized
- [ ] CHANGELOG.md documents public API

---

## R2: Error Handling & Robustness (Phase 2 Priority)

### R2.1: Structured Error Context (MUST)

**Problem**: `EngineError::Persistence(String)` loses structured context.

**Requirement**: Enhance error types to include:
1. File paths for I/O errors
2. Underlying error types (wrapped with `#[from]` or explicit)
3. Operation context (what was being attempted)

**Example**:
```rust
pub enum PersistenceOp {
    Save,
    Load,
    Migrate,
}

pub enum EngineError {
    Persistence {
        path: PathBuf,
        operation: PersistenceOp,
        source: std::io::Error,
    },
    // ...
}
```

**Note**: Use typed operation enums (not strings) for machine-actionable, testable errors. Prefer `thiserror` for library errors; reserve `anyhow` for application layers.

**Acceptance Criteria**:
- [ ] All error variants include actionable context
- [ ] Errors implement `std::error::Error` with `source()`
- [ ] Error messages are user-friendly and debuggable

### R2.2: Input Validation (MUST)

**Problem**: NaN/Inf floats can propagate to scores; dimension mismatches silently ignored.

**Requirement**:
1. Validate embeddings at ingestion (reject NaN/Inf)
2. Error on dimension mismatch (not silent skip)
3. Validate chunk text is non-empty after trimming

**Acceptance Criteria**:
- [ ] `upsert_prepared_document` returns error for invalid embeddings
- [ ] Tests cover NaN, Inf, empty, and mismatched dimension cases
- [ ] Error messages indicate which validation failed

### R2.3: Deterministic Testing Strategy (MUST)

**Problem**: Embedding/scoring can introduce nondeterminism; tests may be flaky.

**Requirement**:
1. Mock backend uses seeded RNG for reproducible embeddings
2. Scoring tests use tolerance bands for floating-point comparison
3. Fixtures for expected search results in integration tests

**Acceptance Criteria**:
- [ ] All tests pass deterministically across runs
- [ ] No external API calls in unit tests
- [ ] CI runs same tests without flakiness

### R2.4: Concurrency Contract (MUST)

**Problem**: Threading model not documented; runtime assumptions may leak.

**Requirement**:
1. Document `Send + Sync` requirements for all traits
2. Clarify blocking I/O expectations
3. Document recommended usage with Tokio/async-std

**Acceptance Criteria**:
- [ ] Traits document thread-safety requirements
- [ ] README includes concurrency section
- [ ] No hidden `block_on` or blocking I/O in async paths

---

## R3: Storage Abstraction (Phase 2-3)

### R3.1: PersistenceBackend Trait (SHOULD - Phase 2)

**Problem**: Hardcoded JSON persistence; no alternative storage options.

**Requirement**: Create `PersistenceBackend` trait for save/load operations:
```rust
pub trait PersistenceBackend: Send + Sync {
    fn save(&self, state: &EngineState) -> Result<(), PersistenceError>;
    fn load(&self) -> Result<Option<EngineState>, PersistenceError>;
    fn schema_version(&self) -> u32;
}

pub struct EngineState {
    pub schema_version: u32,
    pub embedding_model_id: String,
    pub embedding_dim: usize,
    pub chunking_config_hash: u64,
    pub chunks: HashMap<ChunkId, ChunkData>,
    pub document_hashes: HashMap<DocumentId, String>,
    // Optional cached embeddings (can be regenerated)
    pub embeddings: Option<HashMap<ChunkId, Vec<f32>>>,
}
```

**Implementations**:
1. `JsonFileBackend` (default, current behavior) - atomic via temp+rename
2. `SqliteBackend` (feature-gated, Phase 3) - transactional

**Design Decision** (validated by Codex): Keep indexes as derived in-memory caches; abstract persistence only, not index storage. Include identity/invalidation keys in persisted state.

**Acceptance Criteria**:
- [ ] Trait defined in `persistence` module
- [ ] `JsonFileBackend` implements trait
- [ ] `RagEngine` accepts optional custom backend

### R3.2: IndexSet Abstraction (SHOULD - Phase 2)

**Problem**: ANN + Lexical indexes are tightly coupled to engine; hard to test independently.

**Requirement**: Create `IndexSet` struct that:
1. Wraps `AnnIndex` + `LexicalIndex` together
2. Exposes atomic batch operations: `apply_batch(chunks: &[ChunkUpdate])` for all-or-nothing updates
3. Exposes query operations: `search_candidates(query_embedding, query_text, top_k)`
4. Maintains single authoritative chunk registry (indexes reference, not own)
5. Guarantees stable ordering with deterministic tie-breakers
6. Preserves `validate_index_sync` semantics
7. Enables future backend swapping (e.g., HNSW library)

**Design Decision** (validated by Codex): Do NOT split into 3 separate traits; this would introduce atomicity bugs. Use atomic `apply_batch` to update both indexes together.

**Acceptance Criteria**:
- [ ] `IndexSet` struct encapsulates both indexes
- [ ] Engine delegates to `IndexSet` for index operations
- [ ] Existing tests continue to pass

### R3.3: Persistence Format Versioning (MUST - Phase 2)

**Problem**: Format changes could break existing indexes without migration path.

**Requirement**:
1. Increment `INDEX_VERSION` on breaking changes
2. Document migration strategy in CHANGELOG
3. Add version mismatch handling (clear warning, offer reindex)

**Acceptance Criteria**:
- [ ] Version changes trigger clear user notification
- [ ] Migration path documented for each version bump
- [ ] Tests verify version upgrade handling

---

## R4: Enterprise Features (Phase 3)

### R4.1: Alternative Storage Backends (SHOULD)

**Requirement**: Implement at least one alternative `PersistenceBackend`:
1. `SqliteBackend` - transactional, crash-consistent
2. Optionally: `SledBackend` - embedded key-value store

**Acceptance Criteria**:
- [ ] Backend is feature-gated: `features = ["sqlite"]`
- [ ] Backend passes same test suite as JSON backend
- [ ] Performance benchmarks included

### R4.2: Observability Hooks (SHOULD)

**Requirement**: Add structured observability support:
1. Metrics hooks for search latency, chunk count, cache hits
2. Span integration with `tracing` crate
3. Health check method: `engine.health() -> HealthStatus`

**Acceptance Criteria**:
- [ ] Tracing spans cover major operations
- [ ] Metrics can be exported to Prometheus/StatsD
- [ ] Health check available for container probes

### R4.3: CI/CD Pipeline (MUST)

**Requirement**: Dedicated CI for rag-core crate:
1. Run tests on PR
2. Enforce clippy/fmt
3. Check MSRV compatibility
4. Publish to crates.io on release tags

**Acceptance Criteria**:
- [ ] GitHub Actions workflow exists for rag-core
- [ ] MSRV tested (e.g., 1.75+)
- [ ] Release automation functional

---

# PART 2: DESIGN

## D1: Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         rag-core                                │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    RagEngine<B, R>                       │   │
│  │  • B: EmbeddingBackend (pluggable)                      │   │
│  │  • R: Rerank (optional, defaults to ())                 │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│         ┌────────────────────┼────────────────────┐            │
│         ▼                    ▼                    ▼            │
│  ┌────────────┐      ┌────────────┐      ┌────────────────┐   │
│  │  IndexSet  │      │   Chunks   │      │ Persistence    │   │
│  │ (ANN+BM25) │      │  HashMap   │      │   Backend      │   │
│  └────────────┘      └────────────┘      └────────────────┘   │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  Traits:                                                        │
│  • EmbeddingBackend: embed(), embed_batch(), dimension()       │
│  • Rerank: rerank() (optional)                                 │
│  • PersistenceBackend: save(), load() [Phase 2]                │
└─────────────────────────────────────────────────────────────────┘
```

## D2: Module Structure (Target State)

```
crates/rag-core/
├── Cargo.toml           # Features: persistence, tracing, sqlite
├── README.md            # [Phase 1] Quick start & architecture
├── CHANGELOG.md         # [Phase 1] Version history
├── src/
│   ├── lib.rs           # Public exports only
│   ├── engine.rs        # RagEngine<B, R> implementation
│   ├── traits.rs        # EmbeddingBackend, Rerank traits
│   ├── types.rs         # DocumentChunk, SearchResult, RagConfig
│   ├── error.rs         # Structured error types
│   ├── chunking.rs      # Sentence-aware text chunking
│   ├── search/          # [Phase 2] Search module
│   │   ├── mod.rs
│   │   ├── index_set.rs # IndexSet abstraction
│   │   ├── ann.rs       # AnnIndex (LSH)
│   │   ├── lexical.rs   # LexicalIndex (BM25)
│   │   └── mmr.rs       # MMR diversification
│   └── persistence/     # [Phase 2] Storage module
│       ├── mod.rs       # PersistenceBackend trait
│       ├── json.rs      # JsonFileBackend (default)
│       └── sqlite.rs    # [Phase 3] SqliteBackend
└── examples/            # [Phase 1] Usage examples
    ├── basic_usage.rs
    ├── custom_backend.rs
    └── persistence.rs
```

## D3: Key Design Decisions

| Decision | Choice | Rationale | Validated By |
|----------|--------|-----------|--------------|
| Storage abstraction scope | Persistence only, not indexes | Indexes are derived caches; splitting causes atomicity bugs | Codex |
| Trait dispatch | Generics for Engine, dyn for backends | Stable API while enabling runtime swapping | Codex |
| Index structure | Single IndexSet wrapping ANN+Lexical | Preserve validate_index_sync semantics | Codex |
| Default persistence | JSON files | Zero dependencies, human-readable | N/A |
| Threading model | Send + Sync required | Enable concurrent access | Gemini |
| Runtime agnosticism | No Tokio types in core traits | Enable use with any async runtime | Codex |
| Error types | Typed enums with operation enum | Machine-actionable, testable errors | Codex |

## D4: Technical Invariants (Codex-Validated)

These invariants MUST be maintained across all implementations:

### D4.1: Index Consistency
- For any committed batch, both ANN and lexical indexes contain identical ChunkId sets
- Single authoritative chunk registry; indexes reference, not own, membership
- Stable ordering and deterministic tie-breakers in merged search results

### D4.2: Identity/Invalidation Keys
All cache partitioning MUST include:
- `embedding_model_id` + `embedding_dim`
- `chunking_config_hash` (parameters that affect chunk boundaries)
- `reranker_model_id` (if applicable)

### D4.3: Persistence Schema
- `schema_version` field required in all persisted state
- Migration policy: detect mismatch, surface actionable error
- Atomic write semantics: temp file + rename pattern

### D4.4: Runtime Agnosticism
- Core traits use associated `Future` (GAT) or sync API
- No `tokio::` types in public `rag-core` signatures
- Adapters responsible for `spawn_blocking` if needed

### D4.5: Concurrency Model
- Single-writer model for index updates
- Queries see consistent snapshots
- Lock granularity documented per operation

---

# PART 3: TASKS

## Phase 1: Minimum Viable Portability

### T1.1: Create README.md

**Description**: Create comprehensive README for rag-core crate.

**Pre-Task Consultation**:
- **Tool**: `gemini-3-flash-preview`
- **Prompt**: "Review README outline for rag-core. Does it address CLI, Desktop, and Microservice persona needs? Missing sections?"

**Implementation**:
1. Create `crates/rag-core/README.md`
2. Write sections: Intro, Quick Start, Architecture, Features, Installation
3. Add code examples with `rust,ignore` blocks
4. Link to docs and repository

**Post-Task Validation**:
- **Tool**: `crash` MCP
- **Purpose**: Validate README completeness against persona requirements
- **Tool**: `codex-bridge`
- **Prompt**: "Review README.md for technical accuracy and Rust ecosystem conventions"

**Acceptance**:
- [ ] README renders on GitHub
- [ ] Quick Start example is valid Rust

---

### T1.2: Add Rustdoc to Public API

**Description**: Document all public types, traits, and functions.

**Pre-Task Consultation**:
- **Tool**: `crash` MCP
- **Purpose**: Create prioritized list of types to document (most-used first)
- **Prompt**: Analyze lib.rs exports and rank by usage frequency

**Implementation**:
1. Add `//!` module docs to `lib.rs`
2. Document `RagEngine`, `EmbeddingBackend`, `Rerank` traits
3. Document `DocumentChunk`, `SearchResult`, `RagConfig` types
4. Add `# Examples` to key functions

**Post-Task Validation**:
- **Tool**: `gemini-3-flash-preview`
- **Prompt**: "Review rustdoc for rag-core. Are examples clear? Any missing documentation?"
- **Tool**: Bash: `cargo doc --no-deps && cargo test --doc`

**Acceptance**:
- [ ] `cargo doc` succeeds
- [ ] Doc tests pass

---

### T1.3: Create Usage Examples

**Description**: Add examples/ directory with runnable code.

**Pre-Task Consultation**:
- **Tool**: `gemini-3-flash-preview`
- **Prompt**: "What 3-4 examples would most help CLI, Desktop, and Microservice adopters of rag-core?"

**Implementation**:
1. Create `examples/basic_usage.rs` - MockBackend + search
2. Create `examples/custom_backend.rs` - EmbeddingBackend impl
3. Create `examples/persistence.rs` - save/load cycle
4. Add examples to README

**Post-Task Validation**:
- **Tool**: `crash` MCP
- **Purpose**: Verify examples cover primary use cases
- **Tool**: Bash: `cargo build --examples`

**Acceptance**:
- [ ] All examples compile
- [ ] Examples referenced in README

---

### T1.4: Configure Crate Metadata

**Description**: Complete Cargo.toml for distribution.

**Pre-Task Consultation**:
- **Tool**: `codex-bridge`
- **Prompt**: "Review Cargo.toml for rag-core. What metadata is missing for crates.io publication?"

**Implementation**:
1. Set `publish = true`
2. Add: license, authors, repository, documentation, homepage
3. Add: categories = ["text-processing", "algorithms"]
4. Add: keywords = ["rag", "retrieval", "embedding", "search"]
5. Ensure README is included in package

**Post-Task Validation**:
- **Tool**: Bash: `cargo publish --dry-run`
- **Tool**: `gemini-3-flash-preview`
- **Prompt**: "Are categories and keywords optimal for discoverability?"

**Acceptance**:
- [ ] Dry-run publish succeeds
- [ ] Metadata complete

---

### T1.5: Add CHANGELOG and MSRV

**Description**: Initialize version tracking and toolchain requirements.

**Pre-Task Consultation**:
- **Tool**: `crash` MCP
- **Purpose**: Determine appropriate MSRV based on dependency requirements

**Implementation**:
1. Create `CHANGELOG.md` with Keep-a-Changelog format
2. Document 0.1.0 release
3. Add `rust-version = "1.75"` to Cargo.toml (or appropriate MSRV)
4. Document stability guarantees

**Post-Task Validation**:
- **Tool**: Bash: Test with MSRV: `cargo +1.75 check`
- **Tool**: `codex-bridge`
- **Prompt**: "Is MSRV 1.75 appropriate for rag-core's dependencies?"

**Acceptance**:
- [ ] CHANGELOG exists
- [ ] MSRV documented and tested

---

## Phase 2: Production Hardening

### T2.1: Enhance Error Types

**Description**: Add structured context to all error variants.

**Pre-Task Consultation**:
- **Tool**: `crash` MCP
- **Purpose**: Audit current error types and identify missing context
- **Tool**: `gemini-3-pro-preview`
- **Prompt**: "Review rag-core error.rs. Propose enhanced error structure with path, operation, and source context."

**Implementation**:
1. Refactor `EngineError::Persistence` to include PathBuf and operation
2. Refactor `EmbeddingError` variants with context
3. Implement `std::error::Error` with proper `source()` chains
4. Update all error construction sites

**Post-Task Validation**:
- **Tool**: `codex-bridge`
- **Prompt**: "Review error.rs changes. Are error messages actionable? Is source() properly chained?"
- **Tool**: `crash` MCP
- **Purpose**: Verify all error paths tested

**Acceptance**:
- [ ] All errors include context
- [ ] Error tests pass

---

### T2.2: Add Input Validation

**Description**: Validate embeddings and inputs at ingestion.

**Pre-Task Consultation**:
- **Tool**: `crash` MCP
- **Purpose**: Identify all input validation points
- **Tool**: `gemini-3-flash-preview`
- **Prompt**: "What inputs should rag-core validate? Embeddings, text, dimensions?"

**Implementation**:
1. Add `validate_embedding()` helper
2. Check for NaN/Inf values
3. Validate dimension matches engine expectation
4. Reject empty/whitespace-only text
5. Return `Result` instead of silent skip

**Post-Task Validation**:
- **Tool**: `codex-bridge`
- **Prompt**: "Review validation logic. Any edge cases missed?"
- **Tool**: Bash: Run new validation tests

**Acceptance**:
- [ ] Invalid inputs rejected with clear errors
- [ ] Edge case tests pass

---

### T2.3: Create IndexSet Abstraction

**Description**: Wrap ANN + Lexical indexes in unified structure.

**Pre-Task Consultation**:
- **Tool**: `gemini-3-pro-preview`
- **Prompt**: "Design IndexSet struct API for rag-core. Should expose: add_chunk, remove_chunk, search_candidates, validate_sync."
- **Tool**: `crash` MCP
- **Purpose**: Validate design maintains current consistency model

**Implementation**:
1. Create `search/index_set.rs`
2. Move `AnnIndex` and `LexicalIndex` into search module
3. Define `IndexSet` with atomic `apply_batch` API
4. Implement single authoritative chunk registry
5. Add stable ordering with deterministic tie-breakers
6. Update `RagEngine` to use `IndexSet`
7. Preserve `validate_index_sync` behavior

**Post-Task Validation**:
- **Tool**: `codex-bridge`
- **Prompt**: "Review IndexSet implementation. Does it maintain atomicity? Any consistency risks?"
- **Tool**: Bash: Run all existing tests

**Acceptance**:
- [ ] All tests pass
- [ ] IndexSet encapsulates both indexes

---

### T2.4: Create PersistenceBackend Trait

**Description**: Abstract persistence for pluggable storage.

**Pre-Task Consultation**:
- **Tool**: `gemini-3-pro-preview`
- **Prompt**: "Design PersistenceBackend trait for rag-core. Consider: sync vs async, state structure, error handling."
- **Tool**: `crash` MCP
- **Purpose**: Validate trait design enables SQLite/sled backends

**Implementation**:
1. Create `persistence/mod.rs` with trait definition
2. Define `EngineState` struct for save/load
3. Create `JsonFileBackend` implementing trait
4. Update `RagEngine` to accept optional backend
5. Maintain backward compatibility (default to JSON)

**Post-Task Validation**:
- **Tool**: `codex-bridge`
- **Prompt**: "Review PersistenceBackend trait. Is it generic enough for SQLite? Any object-safety issues?"
- **Tool**: Bash: Run persistence tests

**Acceptance**:
- [ ] Trait defined and documented
- [ ] JsonFileBackend passes tests
- [ ] Backward compatible

---

### T2.5: Add Persistence Round-Trip Tests

**Description**: Test save/load cycle for all persistence backends.

**Pre-Task Consultation**:
- **Tool**: `crash` MCP
- **Purpose**: Design test cases for persistence correctness

**Implementation**:
1. Test: save → load → verify chunks intact
2. Test: save → modify file → load → detect corruption
3. Test: version upgrade handling
4. Test: concurrent save attempts

**Post-Task Validation**:
- **Tool**: `codex-bridge`
- **Prompt**: "Review persistence tests. Any edge cases missing?"

**Acceptance**:
- [ ] Round-trip tests pass
- [ ] Corruption detection works

---

### T2.6: Document Concurrency Contract

**Description**: Clarify threading model and runtime expectations.

**Pre-Task Consultation**:
- **Tool**: `gemini-3-flash-preview`
- **Prompt**: "What concurrency documentation does a Rust library need? Send/Sync, blocking, async considerations."

**Implementation**:
1. Add concurrency section to README
2. Document `Send + Sync` on all traits
3. Note blocking I/O in persistence operations
4. Recommend `Arc<RwLock<RagEngine>>` pattern

**Post-Task Validation**:
- **Tool**: `codex-bridge`
- **Prompt**: "Review concurrency documentation. Any hidden blocking or runtime leakage?"

**Acceptance**:
- [ ] Concurrency documented
- [ ] No hidden blocking in async paths

---

## Phase 3: Enterprise Ready

### T3.1: Implement SqliteBackend

**Description**: Add SQLite as alternative persistence backend.

**Pre-Task Consultation**:
- **Tool**: `gemini-3-pro-preview`
- **Prompt**: "Design SQLite schema for rag-core EngineState. Consider: chunks table, metadata, indexing."
- **Tool**: `crash` MCP
- **Purpose**: Validate schema supports all required operations

**Implementation**:
1. Add `sqlx` or `rusqlite` as optional dependency
2. Create `persistence/sqlite.rs`
3. Implement `PersistenceBackend` trait
4. Feature-gate: `features = ["sqlite"]`
5. Add migration support

**Post-Task Validation**:
- **Tool**: `codex-bridge`
- **Prompt**: "Review SqliteBackend. Transaction safety? Migration strategy?"
- **Tool**: Bash: Run persistence tests with SQLite

**Acceptance**:
- [ ] SQLite backend passes same tests as JSON
- [ ] Feature-gated correctly

---

### T3.2: Add Observability Hooks

**Description**: Integrate metrics and structured tracing.

**Pre-Task Consultation**:
- **Tool**: `gemini-3-flash-preview`
- **Prompt**: "What metrics are valuable for a RAG library? Search latency, chunk count, cache hit rate?"

**Implementation**:
1. Add tracing spans to all major operations
2. Define metrics: `search_latency_seconds`, `chunk_count`, `cache_hits`
3. Add `engine.health()` method
4. Document metrics in README

**Post-Task Validation**:
- **Tool**: `codex-bridge`
- **Prompt**: "Review observability implementation. Any performance impact from tracing?"

**Acceptance**:
- [ ] Tracing spans present
- [ ] Health check functional

---

### T3.3: Create CI/CD Pipeline

**Description**: Dedicated GitHub Actions for rag-core.

**Pre-Task Consultation**:
- **Tool**: `crash` MCP
- **Purpose**: Design CI matrix (MSRV, features, platforms)

**Implementation**:
1. Create `.github/workflows/rag-core.yml`
2. Run tests on push/PR
3. Test MSRV compatibility
4. Run clippy and fmt checks
5. Publish on release tags

**Post-Task Validation**:
- **Tool**: `codex-bridge`
- **Prompt**: "Review CI workflow. Any missing checks? Caching optimized?"

**Acceptance**:
- [ ] CI runs on PR
- [ ] Release automation works

---

### T3.4: Add Performance Benchmarks

**Description**: Establish baseline performance metrics.

**Pre-Task Consultation**:
- **Tool**: `gemini-3-flash-preview`
- **Prompt**: "What benchmarks matter for RAG? Search latency at N chunks, indexing throughput, memory usage?"

**Implementation**:
1. Add `benches/` directory with criterion
2. Benchmark: search latency vs chunk count
3. Benchmark: indexing throughput
4. Document results in README

**Post-Task Validation**:
- **Tool**: `codex-bridge`
- **Prompt**: "Review benchmarks. Are they representative? Any methodology issues?"

**Acceptance**:
- [ ] Benchmarks run with criterion
- [ ] Results documented

---

# PART 4: APPENDICES

## A1: Consultation Protocol

Every task follows this consultation protocol:

### Pre-Task Consultation
1. **CRASH MCP**: Define approach, identify risks, validate assumptions
2. **Gemini (gemini-3-flash-preview)**: Design validation, architecture review
3. **Outcome**: Clear implementation plan

### Post-Task Validation
1. **Codex**: Code review, edge case identification, Rust idiom check
2. **CRASH MCP**: Verify implementation matches requirements
3. **Outcome**: Validated implementation ready for PR

## A2: AI Ensemble Validation Record

| Decision | Tool | Response Summary |
|----------|------|------------------|
| User personas | Gemini Pro | CLI Toolsmith, Desktop Dev, Microservice Architect |
| Storage abstraction | Gemini Pro | Recommended 3-trait split (ChunkStore, VectorIndex, LexicalIndex) |
| Storage validation | Codex | Rejected 3-trait split; recommended IndexSet + PersistenceBackend |
| Final design | CRASH | Adopted Codex recommendation; indexes are derived caches |
| Documentation priority | Gemini Flash | README first, then API docs, then examples |
| Missing tasks | Codex | Crate boundaries, MSRV, concurrency contract, persistence versioning |
| Technical invariants | Codex (final) | Runtime agnosticism, atomic apply_batch, typed errors, schema versioning |
| Trait design | Codex (final) | GAT/sync for runtime independence; identity keys for cache invalidation |

## A3: Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| API breakage on trait changes | Medium | High | Semver, CHANGELOG, deprecation warnings |
| Windows file handling issues | Medium | Medium | Phase 2 task T2.6, CI testing on Windows |
| Persistence format incompatibility | Low | High | Version field, migration path documentation |
| Flaky tests from nondeterminism | Medium | Medium | Seeded RNG, tolerance bands, fixtures |

## A4: Success Metrics

| Metric | Phase 1 Target | Phase 2 Target | Phase 3 Target |
|--------|----------------|----------------|----------------|
| Doc coverage | 100% pub items | 100% pub items | 100% + examples |
| Test coverage | Existing | +10% | +20% |
| External adopters | 1 (internal) | 2-3 | 5+ |
| crates.io downloads | N/A | Published | 100+ |

---

**Document Status**: Ready for implementation
**Next Action**: Approve and begin Phase 1 implementation
