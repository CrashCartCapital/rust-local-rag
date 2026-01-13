# Product Requirements Document: rust-local-rag Codebase Improvement

**Version**: 1.1
**Date**: 2025-01-09
**Authors**: Claude Code (AI-assisted)
**Status**: Draft for Review (pragmatic scope revision)

**Wave 2 PRD**: `analysis/RUST_LOCAL_RAG_PRD_WAVE2.md`

---

## Executive Summary

This PRD specifies a structured improvement initiative for the `rust-local-rag` MCP server codebase. The work addresses **12 simplification opportunities**, **14 bugs**, **9 technical debt items**, **6 modules with test gaps**, and **6 stale documentation items** identified in the [CODEBASE_REVIEW_2025-01-09.md](./CODEBASE_REVIEW_2025-01-09.md).

**Guiding Principles**:
- **Modularity**: Separate concerns into cohesive, loosely-coupled modules
- **Extensibility**: Design for future capability addition without breaking changes
- **Relevance**: Prioritize highest-impact, lowest-risk improvements first
- **Pragmatism**: Minimal disruption to existing functionality
- **Test-Driven Development**: Add targeted tests for changed behavior (avoid brittle network/LLM-dependent tests)

### Implementation Scope for This Change (Wave 1)
To keep this effort **relevant and shippable**, this implementation is intentionally scoped to the highest-impact “quick wins” and correctness fixes that can be landed safely in one change:

**In scope (Wave 1 / this implementation)**:
- Centralize a small set of *tunable* performance parameters (timeouts, cache sizes) into `src/config.rs` with env overrides.
- Fix known correctness hazards (reranker unwrap panic risk, schema default mismatch, unsafe JSON access).
- Replace silent error swallowing with debug logging where it has operational impact.
- Unify HTTP/MCP *job-related* response structs to reduce duplication (search MCP output remains human-readable).
- Update stale documentation that is actively misleading.

**Explicitly deferred (Wave 2+)**:
- `mcp_server.rs` modularization and `worker.rs` decomposition (purely structural; higher churn).
- `rag-core` “dynamic dispatch reranker” refactor (requires making `rag-core::Rerank` object-safe; non-trivial API design change).
- Broad coverage/coverage tooling targets (tarpaulin, >40% coverage), unless required by an adjacent change.

**Revised Estimated Effort (Wave 1)**: ~4–8 hours (fits in a single PR)

---

# PART 1: REQUIREMENTS

## R1: Configuration Management

### R1.1: Centralized Configuration (MUST)
**Problem**: Configuration values scattered across 6+ files with hardcoded defaults.

| Current Location | Value | Purpose |
|------------------|-------|---------|
| `src/embeddings.rs:49` | `1200s` | Batch embedding timeout |
| `src/embeddings.rs:53` | `1000` | LRU cache size |
| `src/reranker.rs:203` | `1` | Concurrency limit |
| `src/reranker.rs:204-205` | `60s` | Reranker timeout |
| `src/reranker.rs:535-536` | `-10.0` | Default logprob fallback |

**Requirement**: All tunable configuration values MUST be:
1. Defined in a single `src/config.rs` module
2. Overridable via environment variables
3. Logged at startup for observability
4. Validated at parse time with graceful error handling

**Acceptance Criteria**:
- [ ] Single `Config` struct contains all tunable values
- [ ] `Config::from_env()` loads from environment with defaults
- [ ] Invalid environment values produce descriptive error messages
- [ ] Active configuration logged at INFO level on startup
- [ ] Server starts successfully with no `RAG_*` env vars set (defaults only)

### R1.2: Environment Variable Naming Convention (SHOULD)
**Requirement**: Environment variables follow `RAG_<CATEGORY>_<SETTING>` pattern.

| Variable | Default | Description |
|----------|---------|-------------|
| `RAG_EMBEDDING_TIMEOUT_SECS` | `1200` | Batch embedding API timeout |
| `RAG_EMBEDDING_CACHE_SIZE` | `1000` | LRU cache entries |
| `RAG_RERANKER_TIMEOUT_SECS` | `60` | Per-candidate reranking timeout |
| `RAG_RERANKER_CONCURRENCY` | `1` | Parallel reranking requests |
| `RAG_DEFAULT_LOGPROB` | `-10.0` | Missing logprob fallback value |

---

## R2: Code Simplification

### R2.1: Generic Type Simplification (DEFERRED)
**Reality Check**: The `rag-core::Rerank` trait currently returns `impl Future`, which is **not object-safe**. Moving to `Box<dyn Rerank>` requires an API change (boxing futures or `async_trait`) across `rag-core`.

**Decision**: Defer to Wave 2+ as a deliberate design refactor, not a “quick win”.

### R2.2: MCP Server Modularization (MUST)
**Problem**: `mcp_server.rs` contains 996 lines with 4 mixed concerns.

**Requirement**: Split into focused modules:
```
src/mcp/
├── mod.rs          # Re-exports and public interface
├── tools.rs        # MCP #[tool] handlers
├── http.rs         # Axum HTTP endpoints
├── responses.rs    # Response formatting with From<> impls
└── models.rs       # Request/response types
```

**Decision**: Defer to Wave 2+ to avoid high-churn changes while landing correctness/config work.

### R2.3: Worker Function Decomposition (SHOULD)
**Problem**: `reindex_documents()` is 326 lines with cyclomatic complexity ~15+.

**Requirement**: Extract helper functions:
```rust
async fn discover_pdfs(&self, base_dir: &Path) -> Result<Vec<PathBuf>>;
async fn process_single_document(&self, pdf: &Path, ...) -> Result<ProcessedDoc>;
fn emit_batch_progress(&self, state: &ProgressState);
async fn finalize_reindex(&self, job_id: &str, stats: &JobStats) -> Result<()>;
```

**Acceptance Criteria**:
- [ ] `reindex_documents()` reduced to ~150 lines
- [ ] Each extracted function has unit test
- [ ] Integration tests continue to pass

### R2.4: Dead Code Removal (MUST)
**Problem**: `EmptyParams` struct marked `#[allow(dead_code)]`.

**Location**: `src/mcp_server.rs:51-55`

**Requirement**: Remove unused code.

**Acceptance Criteria**:
- [ ] `EmptyParams` struct deleted
- [ ] `#[allow(dead_code)]` attribute removed
- [ ] Build succeeds without warnings

### R2.5: Response Struct Unification (SHOULD)
**Problem**: Duplicate JSON construction for HTTP and MCP *job-related* responses.

**Locations**:
- `src/mcp_server.rs:98-119` (MCP search)
- `src/mcp_server.rs:402-432` (HTTP search)

**Requirement**: Extract shared `ReindexResponse` and `JobStatusResponse` structs.

**Acceptance Criteria**:
- [ ] Single struct serves both MCP and HTTP handlers for job status and reindex start
- [ ] Status strings use `JobStatus::as_str()` / `JobType::as_str()` (no `format!("{:?}")`)
- [ ] ~20–40 lines of duplication removed

---

## R3: Test Coverage

### R3.1: Critical Module Coverage (MUST)
**Problem**: 3 modules have ZERO unit tests covering critical functionality.

| Module | Lines | Gap |
|--------|-------|-----|
| `src/embeddings.rs` | 286 | Cache, batch embedding |
| `src/reranker.rs` | 786 | Scoring, logprobs parsing |
| `src/mcp_server.rs` | 996 | Tool handlers (only formatting tested) |

**Requirement**: Achieve minimum test coverage for critical paths.

**Acceptance Criteria**:
- [ ] `embeddings.rs`: 5+ tests covering cache hit/miss, batch vs single, model validation
- [ ] `reranker.rs`: 5+ tests covering scoring, timeout handling, error cases
- [ ] `mcp_server.rs`: 3+ tests covering tool handler logic

### R3.2: Error Scenario Coverage (SHOULD)
**Problem**: All integration tests assume happy path (success).

**Missing scenarios**:
- PDF parsing failures
- Ollama API timeouts/500 errors
- Database constraint violations
- File system permission errors

**Requirement**: Add error scenario integration tests.

**Acceptance Criteria**:
- [ ] Test for PDF parsing failure graceful degradation
- [ ] Test for Ollama timeout handling
- [ ] Test for concurrent job creation race condition

### R3.3: Coverage Target (SHOULD)
**Current**: ~15-20% estimated
**Target**: Add tests proportional to change risk (Wave 1 focuses on targeted unit/integration tests).

---

## R4: Bug Fixes

### R4.1: Critical - Reranker Unwrap Panic (MUST)
**Location**: `src/mcp_server.rs:273`
```rust
let reranker = engine.get_reranker().unwrap();  // PANICS if None
```

**Requirement**: Handle optional reranker gracefully.

**Acceptance Criteria**:
- [ ] `calibrate_reranker` returns error if reranker unavailable
- [ ] No `unwrap()` on `Option` from `get_reranker()`

### R4.2: High - Schema Default Mismatch (MUST)
**Location**: `src/mcp_server.rs:47`
```rust
#[schemars(description = "Number of samples to test (default: 20)")]  // WRONG
pub sample_size: Option<usize>,
// Actual default at line 248: unwrap_or(100)
```

**Requirement**: Align schema description with code behavior.

**Acceptance Criteria**:
- [ ] Schema description says "default: 100"
- [ ] Or code changed to default to 20 (if intentional)

### R4.3: High - JSON Unwrap Without Safety (MUST)
**Location**: `src/embeddings.rs:246`
```rust
.any(|m| m["name"].as_str().unwrap_or("").starts_with(&self.model))
```

**Reality Check**: `serde_json::Value` indexing does not panic here, but the access pattern is **brittle** and can silently mis-handle malformed responses.

**Requirement**: Use a robust JSON access pattern (explicit `get()` / `and_then()` chain) for clarity and correctness.

**Fix**:
```rust
.any(|m| m.get("name").and_then(|n| n.as_str()).map_or(false, |s| s.starts_with(&self.model)))
```

### R4.4: Medium - Silent Error Swallowing (SHOULD)
**Locations**:
- `src/rag_engine.rs:393` - `let _ = std::fs::remove_file(&temp_file);`
- `src/reranker.rs:594` - `let _ = self.score_candidate(...).await;`

**Requirement**: Log silently dropped errors.

**Fix**: Replace `let _ =` with `if let Err(e) = { tracing::debug!(...) }`

---

## R5: Documentation Accuracy

### R5.1: Critical - Reranker Description (MUST)
**Location**: `CLAUDE.md:570`
**Current**: Claims "Phi-4-mini" with "JSON-structured prompts"
**Reality**: Uses Qwen3-Reranker-4B with Yes/No binary classification

**Requirement**: Update to reflect actual implementation.

### R5.2: High - Config Example Mismatch (MUST)
**Location**: `docs/configuration.md` + `CLAUDE.md`
**Requirement**: Keep documentation aligned with *documented defaults* (e.g., `nomic-embed-text`) and clearly state that local `.mcp.json` is user-specific.

### R5.3: Medium - Eval Results Clarification (SHOULD)
**Problem**: `CLAUDE.md:670` claims Qwen3-Reranker-4B but eval report shows phi4-mini was tested.

**Requirement**: Add clarification about historical vs current configuration.

---

# PART 2: DESIGN

## D1: Configuration Module Architecture

### D1.1: Config Struct Design

```rust
// src/config.rs

use std::time::Duration;
use std::num::NonZeroUsize;

/// Central configuration for all tunable parameters.
/// Loaded from environment variables with sensible defaults.
#[derive(Debug, Clone)]
pub struct Config {
    // Embedding service
    pub embedding_timeout: Duration,
    pub embedding_cache_size: NonZeroUsize,

    // Reranker service
    pub reranker_timeout: Duration,
    pub reranker_concurrency: usize,
    pub default_logprob_fallback: f32,

    // Worker/locking
    pub write_lock_threshold: Duration,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            embedding_timeout: Duration::from_secs(1200),
            embedding_cache_size: NonZeroUsize::new(1000).unwrap(),
            reranker_timeout: Duration::from_secs(60),
            reranker_concurrency: 1,
            default_logprob_fallback: -10.0,
            write_lock_threshold: Duration::from_millis(1000),
        }
    }
}

impl Config {
    /// Load configuration from environment variables.
    /// Returns default values for any unset variables.
    pub fn from_env() -> Result<Self, ConfigError> {
        let mut config = Config::default();

        if let Ok(val) = std::env::var("RAG_EMBEDDING_TIMEOUT_SECS") {
            config.embedding_timeout = Duration::from_secs(
                val.parse().map_err(|_| ConfigError::InvalidValue {
                    var: "RAG_EMBEDDING_TIMEOUT_SECS",
                    value: val,
                })?
            );
        }
        // ... similar for other fields

        Ok(config)
    }

    /// Log active configuration at startup.
    pub fn log_active(&self) {
        tracing::info!(
            embedding_timeout_secs = self.embedding_timeout.as_secs(),
            embedding_cache_size = self.embedding_cache_size.get(),
            reranker_timeout_secs = self.reranker_timeout.as_secs(),
            "Configuration loaded"
        );
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("Invalid value '{value}' for {var}")]
    InvalidValue { var: &'static str, value: String },
}
```

### D1.2: Integration Points

```rust
// src/embeddings.rs - BEFORE
let timeout = Duration::from_secs(1200);

// src/embeddings.rs - AFTER
let timeout = config.embedding_timeout;
```

**Dependency Flow**:
```
main.rs -> Config::from_env()
    |
    v
EmbeddingService::new(config)
RerankerService::new(config)
WorkerSupervisor::new(config)
```

---

## D2: Dynamic Dispatch Reranker

**Status**: Deferred to Wave 2+ (requires making `rag-core::Rerank` object-safe; non-trivial API change).

### D2.1: Trait Definition

```rust
// crates/rag-core/src/rerank.rs

use async_trait::async_trait;

/// Trait for scoring search candidates for relevance.
/// Object-safe for dynamic dispatch.
#[async_trait]
pub trait Rerank: Send + Sync {
    /// Score a single candidate's relevance to query.
    /// Returns score in [0.0, 1.0] range.
    async fn score(&self, query: &str, candidate: &SearchCandidate) -> Result<f32, RerankError>;

    /// Score multiple candidates concurrently.
    async fn score_batch(&self, query: &str, candidates: &[SearchCandidate]) -> Vec<Result<f32, RerankError>> {
        // Default implementation: sequential scoring
        let mut results = Vec::with_capacity(candidates.len());
        for c in candidates {
            results.push(self.score(query, c).await);
        }
        results
    }
}
```

### D2.2: Engine Simplification

```rust
// crates/rag-core/src/engine.rs - BEFORE
pub struct RagEngine<B: EmbeddingBackend, R = ()> {
    backend: B,
    reranker: Option<R>,
    // ...
}

// crates/rag-core/src/engine.rs - AFTER
pub struct RagEngine<B: EmbeddingBackend> {
    backend: B,
    reranker: Option<Box<dyn Rerank>>,
    // ...
}

impl<B: EmbeddingBackend> RagEngine<B> {
    pub fn set_reranker(&mut self, reranker: impl Rerank + 'static) {
        self.reranker = Some(Box::new(reranker));
    }

    pub fn get_reranker(&self) -> Option<&dyn Rerank> {
        self.reranker.as_deref()
    }
}
```

### D2.3: Risk Mitigation

**Object Safety**: Ensured by using `async_trait` which boxes futures.

**Performance**: vtable lookup is ~1ns; reranker calls take 50-200ms. Overhead negligible.

---

## D3: MCP Module Architecture

**Status**: Deferred to Wave 2+ (organizational refactor; keep Wave 1 focused on correctness/config).

### D3.1: Module Structure

```
src/mcp/
├── mod.rs          # Public API, re-exports
├── tools.rs        # #[tool] handlers (search, list, stats, reindex, calibrate)
├── http.rs         # Axum routes (/healthz, /readyz, /search, /jobs)
├── responses.rs    # SearchResponse, JobStatusResponse, From<> impls
└── models.rs       # SearchRequest, CalibrateRerankerRequest, etc.
```

### D3.2: Dependency Rules (Enforced)

```
http.rs ──depends-on──> tools.rs ──depends-on──> models.rs
                                       │
responses.rs ──depends-on──────────────┘

models.rs: NO dependencies on other mcp modules (data only)
```

### D3.3: Response Unification

```rust
// src/mcp/responses.rs

#[derive(Debug, Serialize)]
pub struct SearchResponse {
    pub results: Vec<SearchResultItem>,
    pub query: String,
    pub total_found: usize,
}

#[derive(Debug, Serialize)]
pub struct SearchResultItem {
    pub document: String,
    pub page: usize,
    pub section: Option<String>,
    pub text: String,
    pub score: f32,
    pub score_breakdown: ScoreBreakdown,
}

impl From<RagSearchResult> for SearchResponse {
    fn from(result: RagSearchResult) -> Self {
        // Unified conversion logic
    }
}
```

---

## D4: Shared OllamaClient Abstraction

**Status**: Deferred to Wave 2+ (avoid introducing a new abstraction in the same PR as config/correctness fixes).

### D4.1: Client Design

```rust
// src/ollama.rs

pub struct OllamaClient {
    client: reqwest::Client,
    base_url: String,
}

impl OllamaClient {
    pub fn new(base_url: &str, config: &Config) -> Self {
        let client = reqwest::Client::builder()
            .pool_max_idle_per_host(2)
            .connect_timeout(Duration::from_secs(30))
            .build()
            .expect("Failed to build HTTP client");

        Self { client, base_url: base_url.to_string() }
    }

    pub async fn verify_model(&self, model: &str) -> Result<bool, OllamaError>;
    pub async fn embed(&self, model: &str, texts: &[String]) -> Result<Vec<Vec<f32>>, OllamaError>;
    pub async fn generate(&self, model: &str, prompt: &str, options: &GenerateOptions) -> Result<GenerateResponse, OllamaError>;
}
```

### D4.2: Consumer Simplification

```rust
// src/embeddings.rs - BEFORE
pub struct EmbeddingService {
    client: reqwest::Client,
    base_url: String,
    model: String,
    // ... duplicate pooling setup
}

// src/embeddings.rs - AFTER
pub struct EmbeddingService {
    ollama: Arc<OllamaClient>,
    model: String,
    cache: RwLock<LruCache<String, Vec<f32>>>,
}
```

---

## D5: Error Handling Standardization

**Status**: Deferred to Wave 2+ unless required by Wave 1 changes (do not expand error surface unnecessarily).

### D5.1: Error Trait Implementation

```rust
// src/error.rs

#[derive(Debug, thiserror::Error)]
pub enum RagError {
    #[error("Embedding error: {0}")]
    Embedding(#[from] EmbeddingError),

    #[error("Reranker error: {0}")]
    Reranker(#[from] RerankError),

    #[error("Configuration error: {0}")]
    Config(#[from] ConfigError),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}
```

### D5.2: Silent Error Pattern Replacement

```rust
// BEFORE
let _ = std::fs::remove_file(&temp_file);

// AFTER
if let Err(e) = std::fs::remove_file(&temp_file) {
    tracing::debug!(error = %e, path = %temp_file.display(), "Failed to remove temp file");
}
```

---

# PART 3: TASKS (TDD-Driven)

## Sprint 1 (Wave 1): Quick Wins + Critical Bugs (single PR, ~4–8 hours)

### Task 1.1: Fix Critical Reranker Panic (15 min)

**TDD Specification**:
```rust
#[tokio::test]
async fn test_calibrate_reranker_without_reranker_returns_error() {
    // Precondition: RagEngine initialized without reranker
    let engine = RagEngine::new(MockBackend).build();
    let server = RagMcpServer::new(engine);

    // Action: Call calibrate_reranker
    let result = server.calibrate_reranker(CalibrateRerankerRequest {
        query: "test".into(),
        sample_size: None,
    }).await;

    // Assert: Returns descriptive error, not panic
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("reranker not available"));
}
```

**Implementation**:
```rust
// src/mcp_server.rs:273
// BEFORE: let reranker = engine.get_reranker().unwrap();
// AFTER:
let reranker = match engine.get_reranker() {
    Some(r) => r,
    None => return Err(McpError::InvalidRequest("Reranker not available. Configure OLLAMA_RERANK_MODEL to enable calibration.".into())),
};
```

**Verification (MCP Tools)**:
```bash
# Run test
cargo test test_calibrate_reranker_without_reranker_returns_error

# Verify no remaining unwrap on get_reranker
grep -n "get_reranker().unwrap()" src/mcp_server.rs  # Should return nothing
```

---

### Task 1.2: Fix Schema Default Mismatch (5 min)

**TDD Specification**:
```rust
#[test]
fn test_calibrate_request_schema_matches_default() {
    let schema = schemars::schema_for!(CalibrateRerankerRequest);
    let schema_json = serde_json::to_string_pretty(&schema).unwrap();

    // Schema description should match actual default
    assert!(schema_json.contains("default: 100"));
}
```

**Implementation**:
```rust
// src/mcp_server.rs:47
#[schemars(description = "Number of samples to test (default: 100)")]
pub sample_size: Option<usize>,
```

---

### Task 1.3: Remove EmptyParams Dead Code (5 min)

**TDD Specification**: N/A (deletion task)

**Implementation**:
```bash
# Delete lines 51-55 in src/mcp_server.rs
```

**Verification**:
```bash
cargo build 2>&1 | grep -i "dead_code"  # Should show no EmptyParams warning
```

---

### Task 1.4: Fix Documentation - Reranker Description (15 min)

**Implementation**: Edit `CLAUDE.md:570`

**Before**:
```markdown
- **reranker.rs**: LLM-based relevance reranking service using Ollama with Phi-4-mini,
  performs concurrent second-stage scoring of search candidates using JSON-structured
  prompts with Phi chat template
```

**After**:
```markdown
- **reranker.rs**: LLM-based relevance reranking using Ollama with Qwen3-Reranker-4B (configurable via OLLAMA_RERANK_MODEL).
  Uses Yes/No binary classification with logprobs-based scoring for relevance assessment.
  Concurrent second-stage scoring of search candidates with configurable concurrency.
```

**Verification (MCP Tools)**:
```bash
# Verify no phi4/phi-4 references remain in main docs
grep -i "phi-4\|phi4" CLAUDE.md | grep -v "Historical"  # Should return nothing
```

---

### Task 1.5: Create Config Module (1 hour)

**TDD Specification**:
```rust
// tests/config_tests.rs

#[test]
fn test_config_default_values() {
    let config = Config::default();
    assert_eq!(config.embedding_timeout, Duration::from_secs(1200));
    assert_eq!(config.embedding_cache_size.get(), 1000);
    assert_eq!(config.reranker_timeout, Duration::from_secs(60));
    assert_eq!(config.reranker_concurrency, 1);
    assert_eq!(config.default_logprob_fallback, -10.0);
    assert_eq!(config.write_lock_threshold, Duration::from_millis(1000));
}

#[test]
fn test_config_env_override() {
    std::env::set_var("RAG_EMBEDDING_TIMEOUT_SECS", "500");
    let config = Config::from_env().unwrap();
    assert_eq!(config.embedding_timeout, Duration::from_secs(500));
    std::env::remove_var("RAG_EMBEDDING_TIMEOUT_SECS");
}

#[test]
fn test_config_invalid_env_returns_error() {
    std::env::set_var("RAG_EMBEDDING_TIMEOUT_SECS", "not_a_number");
    let result = Config::from_env();
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("RAG_EMBEDDING_TIMEOUT_SECS"));
    std::env::remove_var("RAG_EMBEDDING_TIMEOUT_SECS");
}
```

**Implementation Steps**:
1. Create `src/config.rs` with struct and impl
2. Add `mod config;` to `src/lib.rs`
3. Update `src/embeddings.rs` to accept `&Config`
4. Update `src/reranker.rs` to accept `&Config`
5. Update `src/main.rs` to load config at startup

**Verification (MCP Tools)**:
```bash
# Verify hardcoded values removed
grep -n "Duration::from_secs(1200)" src/embeddings.rs  # Should return nothing
grep -n "1000).unwrap()" src/embeddings.rs  # Should return nothing

# Run config tests
cargo test config
```

---

### Task 1.6: Unify Response Formatting (1 hour)

**TDD Specification**:
```rust
#[test]
fn test_search_response_from_rag_result() {
    let rag_result = RagSearchResult {
        chunks: vec![/* test data */],
        query: "test query".into(),
    };

    let response: SearchResponse = rag_result.into();

    assert_eq!(response.query, "test query");
    assert!(!response.results.is_empty());
}

#[test]
fn test_search_response_serialization() {
    let response = SearchResponse {
        results: vec![],
        query: "test".into(),
        total_found: 0,
    };

    let json = serde_json::to_value(&response).unwrap();
    assert!(json.get("results").is_some());
    assert!(json.get("query").is_some());
}
```

**Implementation Steps**:
1. Create shared response structs for *job status* + *reindex start* (single place)
2. Update MCP handlers to serialize from shared structs
3. Update HTTP handlers to return shared structs
4. Delete duplicated ad-hoc JSON construction

---

### Task 1.7: Fix Documentation - Config Example (10 min)

**Implementation**: Update `CLAUDE.md:940` or `.mcp.json:20` to be consistent.

**Decision**: Update CLAUDE.md to match actual `.mcp.json` configuration.

---

## Sprint 2: Test Infrastructure (Day 2-3, ~8 hours)

**Status**: Deferred to Wave 2+ (beyond the scope of this implementation).

### Task 2.1: Embeddings Module Tests (2 hours)

**TDD Specification**:
```rust
// src/embeddings.rs - add #[cfg(test)] mod tests

mod tests {
    use super::*;

    #[tokio::test]
    async fn test_query_cache_hit() {
        let service = EmbeddingService::new_with_mock();

        // First call - cache miss
        let embedding1 = service.get_query_embedding("test query").await.unwrap();

        // Second call - should hit cache
        let embedding2 = service.get_query_embedding("test query").await.unwrap();

        assert_eq!(embedding1, embedding2);
        assert_eq!(service.cache_hits(), 1);
    }

    #[tokio::test]
    async fn test_batch_embedding_vs_single() {
        let service = EmbeddingService::new_with_mock();
        let texts = vec!["text1".into(), "text2".into()];

        let batch_result = service.embed_texts(&texts).await.unwrap();

        let single1 = service.get_embedding("text1").await.unwrap();
        let single2 = service.get_embedding("text2").await.unwrap();

        assert_eq!(batch_result.len(), 2);
    }

    #[tokio::test]
    async fn test_model_not_found_error() {
        let service = EmbeddingService::new_with_config(
            "http://localhost:11434",
            "nonexistent-model",
            &Config::default(),
        );

        let result = service.verify_model().await;
        assert!(result.is_err());
    }
}
```

**Implementation**: Add mock infrastructure and tests.

**Verification (MCP Tools)**:
```bash
cargo test embeddings::tests --lib
```

---

### Task 2.2: Reranker Module Tests (3 hours)

**TDD Specification**:
```rust
mod tests {
    #[tokio::test]
    async fn test_score_parsing_yes_response() {
        let reranker = RerankerService::new_with_mock(MockResponse::Yes);
        let score = reranker.score("query", &test_candidate()).await.unwrap();
        assert!(score > 0.5);
    }

    #[tokio::test]
    async fn test_score_parsing_no_response() {
        let reranker = RerankerService::new_with_mock(MockResponse::No);
        let score = reranker.score("query", &test_candidate()).await.unwrap();
        assert!(score < 0.5);
    }

    #[tokio::test]
    async fn test_timeout_handling() {
        let config = Config { reranker_timeout: Duration::from_millis(10), ..Default::default() };
        let reranker = RerankerService::new_with_config_and_mock(config, MockResponse::Slow);

        let result = reranker.score("query", &test_candidate()).await;
        assert!(matches!(result, Err(RerankError::Timeout)));
    }

    #[tokio::test]
    async fn test_logprobs_fallback() {
        let reranker = RerankerService::new_with_mock(MockResponse::NoLogprobs);
        let score = reranker.score("query", &test_candidate()).await.unwrap();
        // Should use text-based fallback
        assert!(score >= 0.0 && score <= 1.0);
    }
}
```

---

### Task 2.3: MCP Tool Handler Tests (2 hours)

**TDD Specification**:
```rust
#[tokio::test]
async fn test_search_documents_handler() {
    let server = setup_test_server().await;

    let result = server.search_documents(SearchRequest {
        query: "test".into(),
        top_k: Some(5),
        diversity_factor: None,
        weights: None,
    }).await;

    assert!(result.is_ok());
    let response = result.unwrap();
    assert!(response.content.len() <= 5);
}

#[tokio::test]
async fn test_start_reindex_deduplication() {
    let server = setup_test_server().await;

    // First call creates job
    let result1 = server.start_reindex().await.unwrap();
    let job_id_1 = extract_job_id(&result1);

    // Second call returns existing job (dedup)
    let result2 = server.start_reindex().await.unwrap();
    let job_id_2 = extract_job_id(&result2);

    assert_eq!(job_id_1, job_id_2);
}
```

---

### Task 2.4: Error Scenario Integration Tests (1 hour)

**TDD Specification**:
```rust
// tests/error_scenarios.rs

#[tokio::test]
async fn test_pdf_parsing_failure_graceful() {
    let temp_dir = tempfile::tempdir().unwrap();
    std::fs::write(temp_dir.path().join("corrupt.pdf"), b"not a pdf").unwrap();

    let engine = RagEngine::new_for_test(temp_dir.path());
    let result = engine.index_documents().await;

    // Should complete with warning, not fail entirely
    assert!(result.is_ok());
    assert!(result.unwrap().failures > 0);
}

#[tokio::test]
async fn test_ollama_timeout_handled() {
    let config = Config { embedding_timeout: Duration::from_millis(10), ..Default::default() };
    // Use mock server that delays response
    let service = EmbeddingService::new_with_slow_mock(config);

    let result = service.get_embedding("test").await;
    assert!(matches!(result, Err(EmbeddingError::Timeout)));
}
```

---

## Sprint 3: Structural Refactors (Day 4-5, ~10 hours)

**Status**: Deferred to Wave 2+ (beyond the scope of this implementation).

### Task 3.1: Replace RagEngine Generic (2 hours)

**TDD Specification**:
```rust
#[test]
fn test_rag_engine_without_reranker() {
    let engine: RagEngine<MockBackend> = RagEngine::new(MockBackend);
    assert!(engine.get_reranker().is_none());
}

#[test]
fn test_rag_engine_with_reranker() {
    let mut engine = RagEngine::new(MockBackend);
    engine.set_reranker(MockReranker);
    assert!(engine.get_reranker().is_some());
}

#[test]
fn test_rag_engine_reranker_swap_at_runtime() {
    let mut engine = RagEngine::new(MockBackend);
    engine.set_reranker(MockReranker::new("model1"));
    engine.set_reranker(MockReranker::new("model2"));
    // Should not panic, second reranker replaces first
}
```

**Implementation Steps**:
1. Create `Rerank` trait in `crates/rag-core/src/rerank.rs`
2. Modify `RagEngine` struct definition
3. Consolidate 4 impl blocks into 1
4. Update `RerankerService` to implement `Rerank`
5. Update all call sites

**Verification (MCP Tools)**:
```bash
# Verify impl block reduction
grep -c "impl.*RagEngine" crates/rag-core/src/engine.rs  # Should be 1-2, not 4+

# Distill to verify structure
mcp__mcpjungle-ccc__ai-distiller__distill_file file_path="crates/rag-core/src/engine.rs"
```

---

### Task 3.2: Decompose reindex_documents (2.5 hours)

**TDD Specification**:
```rust
#[tokio::test]
async fn test_discover_pdfs() {
    let temp_dir = setup_test_dir_with_pdfs(3);
    let worker = WorkerSupervisor::new_for_test();

    let pdfs = worker.discover_pdfs(temp_dir.path()).await.unwrap();
    assert_eq!(pdfs.len(), 3);
}

#[tokio::test]
async fn test_process_single_document() {
    let pdf_path = create_test_pdf();
    let worker = WorkerSupervisor::new_for_test();

    let result = worker.process_single_document(&pdf_path).await;
    assert!(result.is_ok());
    assert!(!result.unwrap().chunks.is_empty());
}
```

**Implementation Steps**:
1. Extract `discover_pdfs()` helper
2. Extract `process_single_document()` helper
3. Extract `emit_batch_progress()` helper
4. Extract `finalize_reindex()` helper
5. Update `reindex_documents()` to use helpers

---

### Task 3.3: Modularize mcp_server.rs (3 hours)

**TDD Specification**:
```rust
// Existing tests should continue to pass after split
// Additional test for module isolation:

#[test]
fn test_models_module_no_logic_dependencies() {
    // Verify models.rs only contains data structures
    // by checking it compiles without importing tools/http
    use crate::mcp::models::*;
    let _ = SearchRequest { query: "test".into(), top_k: None, diversity_factor: None, weights: None };
}
```

**Implementation Steps**:
1. Create `src/mcp/` directory
2. Move request/response types to `models.rs`
3. Move formatting to `responses.rs`
4. Move tool handlers to `tools.rs`
5. Move HTTP handlers to `http.rs`
6. Create `mod.rs` with re-exports
7. Update imports throughout codebase

**Verification (MCP Tools)**:
```bash
# Verify no file exceeds 300 lines
wc -l src/mcp/*.rs | awk '$1 > 300 {print "FAIL: " $2 " has " $1 " lines"}'

# Verify no circular dependencies
cargo check 2>&1 | grep -i "cyclic"  # Should return nothing

# Distill new structure
mcp__mcpjungle-ccc__ai-distiller__distill_directory directory_path="src/mcp/"
```

---

### Task 3.4: Convert Priority Unwrap Calls (2 hours)

**TDD Specification**:
```rust
#[test]
fn test_no_unwrap_in_production_code() {
    // This is a static analysis test - run via clippy
    // cargo clippy -- -D clippy::unwrap_used
}
```

**Implementation Steps**:
1. Fix `src/mcp_server.rs:273` (Task 1.1 covers this)
2. Fix `src/embeddings.rs:246` (JSON safety)
3. Fix `src/reranker.rs:508-529` (conditional unwraps → `map_or()`)
4. Fix `src/embeddings.rs:53` (use `expect()` for const)

**Verification**:
```bash
# Run clippy with unwrap lint (informational, not blocking)
cargo clippy -- -W clippy::unwrap_used 2>&1 | grep -c "unwrap"
```

---

## Sprint 4: Technical Debt Cleanup (Week 2, ~8 hours)

**Status**: Deferred to Wave 2+ (beyond the scope of this implementation), except for “silent error” logging which is implemented in Wave 1 when it impacts operations.

### Task 4.1: Create Shared OllamaClient (2 hours)

**TDD Specification**:
```rust
#[tokio::test]
async fn test_ollama_client_verify_model() {
    let client = OllamaClient::new("http://localhost:11434", &Config::default());
    let exists = client.verify_model("nomic-embed-text").await.unwrap();
    // May pass or fail depending on local setup
}

#[tokio::test]
async fn test_ollama_client_embed() {
    let client = OllamaClient::new("http://localhost:11434", &Config::default());
    let embeddings = client.embed("nomic-embed-text", &["test".into()]).await;
    // Structure test, not dependent on local Ollama
    assert!(embeddings.is_ok() || embeddings.is_err());
}
```

**Implementation Steps**:
1. Create `src/ollama.rs` with `OllamaClient`
2. Implement `verify_model()`, `embed()`, `generate()`
3. Refactor `EmbeddingService` to use `OllamaClient`
4. Refactor `RerankerService` to use `OllamaClient`
5. Delete duplicate HTTP client setup

---

### Task 4.2: Add Logging for Silent Errors (1 hour)

**Implementation**:
```rust
// src/rag_engine.rs:393
// BEFORE: let _ = std::fs::remove_file(&temp_file);
// AFTER:
if let Err(e) = std::fs::remove_file(&temp_file) {
    tracing::debug!(error = %e, path = %temp_file.display(), "Failed to cleanup temp file");
}
```

Repeat for:
- `src/reranker.rs:594`
- `src/job_manager.rs:497-499` (test cleanup)

---

### Task 4.3: Implement From<> Traits for Errors (1 hour)

**TDD Specification**:
```rust
#[test]
fn test_rag_error_from_embedding_error() {
    let embedding_err = EmbeddingError::Timeout;
    let rag_err: RagError = embedding_err.into();
    assert!(matches!(rag_err, RagError::Embedding(_)));
}
```

**Implementation**: Add `#[from]` attributes via thiserror.

---

### Task 4.4: Convert Remaining Unwraps (4 hours)

**Scope**: All 61 unwrap calls, prioritizing production code.

**Strategy**:
1. Test code: Convert to `.expect("descriptive message")`
2. Production code: Convert to `?` or `ok_or_else()`

---

# PART 4: MCP TOOL INTEGRATION

## Verification Checkpoints

### After Each Task

| Checkpoint | MCP Tool | Command |
|------------|----------|---------|
| Build succeeds | `Bash` | `cargo build` |
| Tests pass | `Bash` | `cargo test` |
| No new warnings | `Bash` | `cargo build 2>&1 \| grep -c warning` |

### After Sprint 1

| Verification | Tool | Usage |
|--------------|------|-------|
| Config struct correct | `distill_file` | `file_path="src/config.rs"` |
| Dead code removed | `Grep` | `pattern="EmptyParams"` |
| Docs updated | `Grep` | `pattern="phi4-mini" path="CLAUDE.md"` |

### After Sprint 2

| Verification | Tool | Usage |
|--------------|------|-------|
| Test count increased | `Bash` | `cargo test -- --list 2>&1 \| wc -l` |
| Coverage measured | `Bash` | `cargo tarpaulin --out Html` |

### After Sprint 3

| Verification | Tool | Usage |
|--------------|------|-------|
| Module structure | `distill_directory` | `directory_path="src/mcp/"` |
| Generic removed | `Grep` | `pattern="<B.*R.*>" path="crates/rag-core"` |
| Complexity reduced | `Bash` | `wc -l src/worker.rs` |

### After Sprint 4

| Verification | Tool | Usage |
|--------------|------|-------|
| OllamaClient shared | `distill_file` | `file_path="src/ollama.rs"` |
| Silent errors logged | `Grep` | `pattern="let _ =" -A 2` |
| Unwraps reduced | `Grep` | `pattern="\.unwrap\(\)"` count |

---

## AI Consultant Integration

### Design Reviews (Gemini-3-pro)

Use at these decision points:
1. Before implementing `Config` struct - validate field selection
2. Before `RagEngine` generic removal - validate object safety
3. Before module split - validate dependency structure

**Prompt Template**:
```
CONTEXT: Implementing [TASK] for rust-local-rag.
TIMEOUT: 300s.
OUTPUT: JSON with 'approved', 'concerns', 'suggestions' keys.

Design: [PASTE DESIGN]
Question: [SPECIFIC QUESTION]
```

### Code Reviews (Codex)

Use for complex implementations:
1. `reindex_documents()` decomposition
2. `OllamaClient` shared abstraction
3. Error handling standardization

---

# APPENDIX

## A: File Change Summary

| File | Action | Lines Changed |
|------|--------|---------------|
| `src/config.rs` | CREATE | +120 |
| `src/mcp_server.rs` | SPLIT | -996 |
| `src/mcp/mod.rs` | CREATE | +30 |
| `src/mcp/tools.rs` | CREATE | +250 |
| `src/mcp/http.rs` | CREATE | +200 |
| `src/mcp/responses.rs` | CREATE | +80 |
| `src/mcp/models.rs` | CREATE | +60 |
| `src/ollama.rs` | CREATE | +100 |
| `src/embeddings.rs` | MODIFY | -30, +50 |
| `src/reranker.rs` | MODIFY | -20, +40 |
| `src/worker.rs` | MODIFY | -100, +80 |
| `crates/rag-core/src/engine.rs` | MODIFY | -80 |
| `crates/rag-core/src/rerank.rs` | CREATE | +40 |
| `CLAUDE.md` | MODIFY | -5, +15 |

**Net Change**: ~+350 lines (after removing duplicates)

## B: Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Generic removal breaks type inference | Low | Medium | Comprehensive test suite |
| Module split causes import errors | Medium | Low | Incremental migration |
| Config parsing fails in production | Low | High | Graceful defaults, logging |
| Test coverage false confidence | Medium | Medium | Focus on critical paths |

## C: Success Criteria

1. **Build**: `cargo build --release` succeeds with 0 warnings
2. **Tests**: `cargo test` passes with 60+ tests (up from 42)
3. **Coverage**: >40% measured by tarpaulin
4. **Complexity**: No file >400 lines in `src/`
5. **Documentation**: All critical doc issues resolved
6. **Bugs**: 0 critical, <3 high priority remaining

---

**Document Status**: Ready for Review
**Next Steps**: Stakeholder approval, then Sprint 1 execution
