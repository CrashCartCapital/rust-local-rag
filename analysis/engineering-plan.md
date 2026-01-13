# Engineering Plan: rust-local-rag Production Readiness

**Generated**: 2025-12-05
**Revised**: 2025-12-05 (Codex + Gemini review incorporated)
**Based on**: Code Review Report (Draft PRD)
**Codebase State**: 25 tests passing, 4,540 LOC (src/*.rs)

---

## SECTION 1: PRD VERIFICATION SUMMARY

### What Was Accurate

| PRD Claim | Verification | Evidence |
|-----------|--------------|----------|
| 25 tests passing | **CONFIRMED** | `cargo test` output shows 25 passed |
| Blocking I/O in async context | **CONFIRMED** | `src/rag_engine.rs:660-675` uses `std::process::Command` and `std::fs::write` |
| External pdftotext dependency | **CONFIRMED** | `src/rag_engine.rs:668-674` calls `pdftotext` binary |
| Atomic persistence implemented | **CONFIRMED** | `save_to_disk()` uses temp file + `tokio::fs::rename` (lines 1100-1105) |
| Model-partitioned storage | **CONFIRMED** | 14 TDD tests for `chunks_{model}.json` pattern |
| Job-based architecture | **CONFIRMED** | SQLite with WAL mode, atomic transactions |
| Hybrid search (70/30 embedding/BM25) | **CONFIRMED** | Lines 1364-1365: `EMBEDDING_WEIGHT: f32 = 0.7`, `LEXICAL_WEIGHT: f32 = 0.3` |

### What Was INACCURATE

| PRD Claim | Reality | Evidence |
|-----------|---------|----------|
| ~2,600 LOC | **4,540 LOC** | `wc -l src/*.rs` = 4540 total |
| Line 714-744 for blocking I/O | **Lines 654-703** | `extract_pdf_with_pdftotext` function at 659-702 |
| Line 244-277 for write lock starvation | **Lines 244-277** | Correct location, but issue is overstated - see analysis below |
| Exit codes at main.rs:208-217 | **Lines 207-217** | Close but `tokio::select!` block starts at 208 |
| "Write lock held during embedding calls" | **PARTIAL** | Lock IS held during `add_document()` but per-document, not hours-long |

### Gap Analysis: Missing Context from PRD

1. **spawn_blocking Already Used**: The PRD claims "Add spawn_blocking for CPU-intensive tasks" is needed, but `walkdir` traversal already uses `spawn_blocking` (worker.rs:155-167). Only the PDF extraction is blocking.

2. **Per-Document Locking Already Implemented**: The worker.rs code (lines 244-277) acquires write lock per-document, NOT for the entire reindexing operation. The PRD overstates the "hours-long lock" issue.

3. **Tokio fs Already Used for Most Operations**:
   - `tokio::fs::read` used in `load_documents_from_dir` (rag_engine.rs:611)
   - `tokio::fs::write/rename` used in `save_to_disk` (rag_engine.rs:1100-1105)
   - Only `extract_pdf_with_pdftotext` uses sync I/O

4. **Process Exit Handling**: The PRD claims process exits with code 0 on crashes, but `main()` returns `Result<()>`, which Tokio converts to non-zero exit on error. The real issue is that some errors are logged but not propagated.

---

## SECTION 2: REQUIREMENTS (Revised per AI Review)

### 2.1 Functional Requirements

| ID | Requirement | Priority | Testable Outcome |
|----|-------------|----------|------------------|
| R1 | PDF extraction MUST NOT perform blocking filesystem or CPU-bound work on Tokio runtime worker threads; heavy operations MUST run via `tokio::task::spawn_blocking` | P0 | Integration test showing concurrent searches during PDF processing |
| R2 | Write locks on RagEngine SHOULD be held <1 second and MUST NOT be held across async await points that perform I/O or CPU-heavy work | P1 | Timing assertions in integration tests |
| R3 | PDF extraction SHOULD work without external `pdftotext` binary | P2 | Unit test verifying PDF extraction with embedded library |
| R4 | Fatal errors MUST exit with non-zero status code, including failures in background tasks | P1 | Process exit code test via subprocess |
| R5 | Health endpoint MUST indicate server readiness (index loaded, workers started, no critical errors) | P2 | HTTP 200 from `/healthz` when Ollama reachable |
| R6 | Graceful shutdown MUST stop accepting new requests, drain in-flight tasks with bounded timeout, and flush pending writes | P1 | Shutdown test verifying chunks.json integrity |

### 2.2 Constraints from Existing Codebase

1. **MCP Protocol**: Uses `rmcp` crate with `#[tool]` macros - cannot break MCP interface
2. **Tokio Runtime**: All async code uses Tokio with `#[tokio::main]`
3. **Ollama Dependency**: Embedding and reranking require running Ollama instance
4. **File Format**: Must maintain backwards compatibility with `chunks_{model}.json`

### 2.3 Simplicity Decisions

1. **Defer health endpoints to P2**: Current MCP-over-HTTP transport doesn't require Kubernetes-style probes
2. **Keep sync I/O in tests**: Test code using `std::fs` is acceptable since tests run on test threads
3. **Single-worker reindex**: Already limited to 1 concurrent worker (worker.rs:35) - no need to add parallelism

### 2.4 Out of Scope (Deferred)

- Multi-worker parallel reindexing (memory constraints on M2 Max)
- Streaming PDF extraction (complexity vs. value)
- Custom embedding models (Ollama abstraction sufficient)

---

## SECTION 3: DESIGN

### 3.1 High-Level Architecture

No architectural changes required. The existing design is sound:

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   MCP Server    │────▶│  RagEngine       │────▶│  EmbeddingService│
│  (mcp_server.rs)│     │  (Arc<RwLock<>>) │     │  (embeddings.rs) │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               │
                               ▼
                        ┌──────────────────┐
                        │  WorkerSupervisor │
                        │  (worker.rs)     │
                        └──────────────────┘
```

### 3.2 Component Design

#### Component: PDF Extractor (Refactored)

- **Location**: `src/rag_engine.rs` - `extract_pdf_with_pdftotext` function
- **Responsibility**: Extract text from PDF bytes without blocking async executor
- **Current Interface**:
  ```rust
  fn extract_pdf_with_pdftotext(&self, data: &[u8]) -> Result<String>
  ```
- **Proposed Interface**:
  ```rust
  async fn extract_pdf_text_async(&self, data: Vec<u8>) -> Result<String>
  ```
- **Implementation Strategy** (per Gemini/Codex review):

  **CRITICAL**: Use `tokio::task::spawn_blocking` to wrap the existing synchronous function, NOT `tokio::process::Command`. The external `pdftotext` process is a black box - making the wait non-blocking doesn't prevent resource saturation. The idiomatic Tokio pattern is:

  ```rust
  async fn extract_pdf_text_async(&self, data: Vec<u8>) -> Result<String> {
      // Clone self reference for the blocking task
      let extractor = self.clone(); // or use Arc

      tokio::task::spawn_blocking(move || {
          // Entire blocking operation runs on dedicated thread pool
          extractor.extract_pdf_with_pdftotext(&data)
      })
      .await
      .map_err(|e| anyhow::anyhow!("Task join error: {}", e))?
  }
  ```

#### Component: Pure-Rust PDF Fallback (New)

- **Location**: `src/pdf_extractor.rs` (new file)
- **Responsibility**: Extract text from PDFs using embedded library when pdftotext unavailable
- **Interface**:
  ```rust
  pub fn extract_text(data: &[u8]) -> Result<String>
  ```
- **Fallback Strategy** (per Gemini review):
  1. Try pure-Rust extractor first (`pdf-extract` crate)
  2. If extraction fails, log the failure and fall back to `pdftotext` via `spawn_blocking`
  3. This provides deployment flexibility while maintaining robustness
- **Dependencies**: `pdf-extract` crate (CPU-bound, must use `spawn_blocking`)

#### Component: Shutdown Coordinator (Enhancement)

- **Location**: `src/main.rs` - shutdown handling
- **Responsibility**: Ensure clean shutdown with data integrity
- **Current Behavior**: Ctrl+C triggers graceful HTTP server shutdown, but no explicit flush
- **Proposed Enhancement** (per Codex/Gemini review):

  ```rust
  // Robust shutdown sequence with timeouts
  tokio::select! {
      _ = tokio::signal::ctrl_c() => {
          tracing::info!("Shutdown signal received. Stopping new requests...");

          // 1. Stop accepting new work (close listener - handled by axum graceful shutdown)

          // 2. Acquire lock with timeout
          match tokio::time::timeout(
              Duration::from_secs(10),
              rag_state.write()
          ).await {
              Ok(engine) => {
                  // 3. Flush with timeout
                  match tokio::time::timeout(
                      Duration::from_secs(5),
                      engine.save_to_disk()
                  ).await {
                      Ok(Ok(())) => tracing::info!("State saved successfully"),
                      Ok(Err(e)) => tracing::error!("Failed to save state: {}", e),
                      Err(_) => tracing::error!("Save timed out"),
                  }
              },
              Err(_) => {
                  tracing::error!("Could not acquire lock during shutdown. Data may not be saved.");
              }
          }
      },
      // ... other select branches
  }
  ```

### 3.3 Test Design (TDD Specification)

| What to Test | Expected Behavior | Edge Cases |
|--------------|-------------------|------------|
| Async PDF extraction via spawn_blocking | Completes without blocking executor | Empty PDF, corrupted PDF, large PDF (100MB+) |
| Concurrent search during reindex | Searches return results while reindex runs | Lock contention, slow Ollama |
| Pure-Rust PDF extraction | Extracts same text as pdftotext | Scanned PDFs (should fail gracefully), encrypted PDFs |
| Exit code on fatal error | Non-zero exit on Ollama connection failure | Timeout vs. connection refused |
| Shutdown flush with timeout | chunks.json contains all indexed data after SIGTERM | Mid-write interruption, lock contention during shutdown |
| Shutdown lock timeout | Process exits cleanly even if lock unavailable | Worker holding lock during shutdown |

### 3.4 Gotchas and Watch-Outs

1. **spawn_blocking requires `Send` bounds** - ensure data passed to blocking task is `Send`
2. **pdf-extract crate** may not support all PDF variants - keep pdftotext as fallback
3. **Write lock duration** is per-document (~seconds), not a real P0 - profile before major refactor
4. **Temp file cleanup** in sync context is fine inside spawn_blocking
5. **Exit codes**: Tokio main macro already converts `Err` to non-zero exit - verify propagation
6. **Shutdown ordering**: Must stop listener before attempting lock acquisition
7. **JoinError handling**: spawn_blocking can panic - handle the JoinError properly

---

## SECTION 4: TASKS (Revised per AI Review)

### Phase 1: Critical Runtime Fixes (P0)

#### Task 1.1: Wrap PDF extraction in spawn_blocking

- **Test First**: `tests/integration/async_pdf_extraction.rs`
  ```rust
  #[tokio::test]
  async fn test_pdf_extraction_does_not_block_executor() {
      // Spawn multiple search tasks
      let search_handles: Vec<_> = (0..5).map(|_| {
          tokio::spawn(async { /* search operation */ })
      }).collect();

      // Spawn PDF extraction task (should not block searches)
      let extract_handle = tokio::spawn(async { /* PDF extraction */ });

      // Assert searches complete promptly even during extraction
      for (i, handle) in search_handles.into_iter().enumerate() {
          let res = tokio::time::timeout(Duration::from_secs(1), handle).await;
          assert!(res.is_ok(), "Search task {} was blocked and timed out", i);
      }
  }
  ```
- **Implementation**: `src/rag_engine.rs`
  - Keep existing `extract_pdf_with_pdftotext` as sync function
  - Add new `async fn extract_pdf_text_async` that wraps it in `spawn_blocking`
  - **CRITICAL FIX**: Replace `std::process::id()` with `Uuid::new_v4()` or `tempfile::NamedTempFile` for temp filename generation. The current code uses PID which is constant per process - concurrent PDF extraction will cause filename collisions, data corruption, or crashes.
  - Update callers to use the async version
- **Depends on**: None
- **Effort**: 2-3 hours (includes temp file fix)
- **Risk**: Low - spawn_blocking is idiomatic; temp file fix is straightforward

#### Task 1.2: Wrap CPU-intensive chunking in spawn_blocking

- **Test First**: Verify chunking doesn't starve async tasks
- **Implementation**:
  ```rust
  let chunks = tokio::task::spawn_blocking(move || {
      chunk_text(&text, 200) // Pure CPU work
  }).await?;
  ```
- **Depends on**: Task 1.1 (async extraction must complete first)
- **Effort**: 1 hour
- **Risk**: Low - chunking is pure computation, no async APIs needed

### Phase 2: Deployment Readiness (P1)

#### Task 2.1: Verify and fix exit codes on fatal errors

- **Test First**: `tests/integration/exit_codes.rs`
  ```rust
  #[test]
  fn test_exit_code_on_ollama_unreachable() {
      let output = Command::new(env!("CARGO_BIN_EXE_rust-local-rag"))
          .env("OLLAMA_URL", "http://localhost:99999")
          .output()
          .unwrap();
      assert!(!output.status.success());
  }

  #[test]
  fn test_exit_code_propagates_from_background_tasks() {
      // Verify tokio::spawn failures propagate to main
  }
  ```
- **Implementation**: Audit `main()` and ensure all error paths propagate, including background task failures
- **Depends on**: None
- **Effort**: 1-2 hours
- **Risk**: Low - may already work correctly

#### Task 2.2: Implement robust graceful shutdown

- **Test First**: `tests/integration/shutdown.rs`
  ```rust
  #[tokio::test]
  async fn test_shutdown_flushes_data_with_timeout() {
      // Start server, index some data
      // Send SIGTERM
      // Verify chunks.json is complete
  }

  #[tokio::test]
  async fn test_shutdown_exits_cleanly_when_lock_unavailable() {
      // Start server, hold write lock in separate task
      // Send SIGTERM
      // Verify process exits within timeout (doesn't hang)
  }
  ```
- **Implementation**: Add timeout-based shutdown sequence per Section 3.2
  - **Note**: This logic must be integrated with the `axum` web server's lifecycle, likely by passing an async block containing this shutdown sequence to `axum::serve::with_graceful_shutdown`. This prevents race conditions between Axum's shutdown and the flush logic.
  - **Handle PoisonError**: Ensure shutdown logic handles `PoisonError` if the lock was held by a thread that panicked.
- **Depends on**: None
- **Effort**: 2-3 hours
- **Risk**: Medium - requires careful ordering and proper Axum integration

#### Task 2.3: Instrument and verify write lock duration

- **Test First**: Add tracing instrumentation
  ```rust
  #[tokio::test]
  async fn test_write_lock_duration_under_threshold() {
      // Measure time between write().await and drop
      // Assert < 1 second for typical document
  }
  ```
- **Implementation**: Add tracing spans around lock acquisition/release
- **Depends on**: None
- **Effort**: 1-2 hours
- **Risk**: If locks are actually long, triggers new refactor task

### Phase 3: Reliability Improvements (P2)

#### Task 3.1: Add pure-Rust PDF extraction fallback

- **Test First**: `tests/unit/pdf_extractor.rs`
  ```rust
  #[test]
  fn test_extract_simple_pdf() {
      let data = include_bytes!("fixtures/simple.pdf");
      let text = pdf_extractor::extract_text(data).unwrap();
      assert!(text.contains("expected content"));
  }

  #[test]
  fn test_fallback_to_pdftotext_on_failure() {
      // Corrupted PDF triggers fallback
  }
  ```
- **Implementation**:
  1. Evaluate crates: `pdf-extract`, `lopdf`, `pdf`
  2. Create `src/pdf_extractor.rs` module
  3. Modify `extract_pdf_text_async` to try embedded extractor first, fall back to pdftotext
  4. **CRITICAL**: Both paths must use `spawn_blocking` (per Codex review)
- **Depends on**: Task 1.1
- **Effort**: 3-4 hours (includes crate evaluation)
- **Risk**: Medium - PDF libraries vary in coverage

#### Task 3.2: Add health and readiness endpoints

- **Test First**: HTTP client test for `/healthz` and `/readyz`
- **Implementation**:
  1. Add routes to `mcp_server.rs` router
  2. Liveness (`/healthz`): Returns 200 if process is running
  3. Readiness (`/readyz`): Returns 200 when:
     - Index is loaded
     - Background workers are started
     - No critical error flags set
     - Ollama is reachable (optional, may add latency)
- **Depends on**: None
- **Effort**: 2-3 hours
- **Risk**: Low

---

## SECTION 5: IMPLEMENTATION CHECKLIST (TDD Order)

### Phase 1: P0 Critical (Required before production)

- [ ] **1.1** Write test: `test_pdf_extraction_does_not_block_executor`
  - Test: `tests/integration/async_pdf.rs::test_concurrent_pdf_and_search`
  - Implement: `src/rag_engine.rs::extract_pdf_text_async` using `spawn_blocking`
  - Depends on: none

- [ ] **1.2** Wrap `chunk_text` in spawn_blocking
  - Test: Verify via concurrent task completion
  - Implement: `src/rag_engine.rs::add_document`
  - Depends on: 1.1

### Phase 2: P1 Deployment (Required for production)

- [ ] **2.1** Write test: `test_exit_code_on_ollama_unreachable`
  - Test: `tests/integration/exit_codes.rs`
  - Implement: Audit `main()` error propagation
  - Depends on: none

- [ ] **2.2** Write test: `test_shutdown_flushes_data_with_timeout`
  - Test: `tests/integration/shutdown.rs`
  - Implement: `src/main.rs` shutdown handler with timeouts
  - Depends on: none

- [ ] **2.3** Write test: `test_write_lock_duration_under_threshold`
  - Test: `tests/integration/lock_timing.rs`
  - Implement: Add timing instrumentation to `worker.rs`
  - Depends on: none

### Phase 3: P2 Reliability (Nice to have)

- [ ] **3.1** Write test: `test_extract_simple_pdf_pure_rust`
  - Test: `tests/unit/pdf_extractor.rs`
  - Implement: `src/pdf_extractor.rs` (new) with fallback logic
  - Depends on: 1.1

- [ ] **3.2** Write test: `test_health_endpoint_returns_200`
  - Test: `tests/integration/health.rs`
  - Implement: Add routes to `mcp_server.rs`
  - Depends on: none

---

## SECTION 6: RISK ASSESSMENT

### Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Temp file collision (CRITICAL)** | **High** | **Critical** | **Fix in Task 1.1: Replace `process::id()` with `Uuid::new_v4()`** |
| pdf-extract crate doesn't support all PDFs | Medium | Medium | Keep pdftotext as fallback, test with diverse PDFs |
| spawn_blocking thread pool exhaustion | Low | Medium | Monitor thread pool metrics, consider custom pool size |
| Lock duration is actually problematic | Low | High | Profile first (Task 2.3) before major refactor |
| Pure-Rust PDF adds 5MB+ to binary | Medium | Low | Acceptable for deployment simplicity |
| Shutdown timeout too short | Low | Medium | Make timeout configurable via environment |
| Lock poisoning during shutdown | Low | Medium | Handle `PoisonError` in shutdown logic |

### Dependencies

| Crate | Purpose | License | Risk |
|-------|---------|---------|------|
| `pdf-extract` | PDF text extraction | MIT | Actively maintained, 800+ stars |
| `tokio::task` | spawn_blocking | MIT | Part of tokio, no additional dep |

### Rollback Plan

All changes are additive:
1. spawn_blocking: Transparent wrapper, no API changes to callers
2. Pure-Rust PDF: Falls back to pdftotext if extraction fails
3. Health endpoints: New route, doesn't affect existing functionality
4. Graceful shutdown: Timeout-based, will exit cleanly even on failure

---

## SECTION 7: VERIFICATION CRITERIA

### Done When

- [ ] All 25 existing tests still pass
- [ ] New tests for each task pass
- [ ] `cargo clippy` produces no warnings
- [ ] Integration test: concurrent search during reindex works
- [ ] Manual test: reindex 50 PDFs without timeout or lock issues
- [ ] Deployment test: works on fresh machine without pdftotext installed (P2)
- [ ] Shutdown test: SIGTERM results in clean exit with data flushed

### Quality Gates

- [ ] Searched codebase before proposing new code
- [ ] Every task has test defined before implementation
- [ ] No unnecessary complexity introduced
- [ ] External dependencies verified in Cargo.toml
- [ ] Follows existing project patterns (rmcp macros, tracing, etc.)

---

## Appendix A: Codebase Evidence

### A.1: Blocking I/O Location (CONFIRMED)

```rust
// src/rag_engine.rs:659-675
fn extract_pdf_with_pdftotext(&self, data: &[u8]) -> Result<String> {
    use std::process::Command;  // <-- BLOCKING

    let temp_dir = std::env::temp_dir();
    let temp_file = temp_dir.join(format!("temp_pdf_{}.pdf", std::process::id()));

    std::fs::write(&temp_file, data)  // <-- BLOCKING
        .map_err(|e| anyhow::anyhow!("Failed to write temp PDF: {}", e))?;

    let output = Command::new("pdftotext")  // <-- BLOCKING
        .arg("-layout")
        .arg("-enc")
        .arg("UTF-8")
        .arg(&temp_file)
        .arg("-")
        .output();
    let _ = std::fs::remove_file(&temp_file);  // <-- BLOCKING
    // ...
}
```

### A.2: Per-Document Locking Pattern (Correct Design)

```rust
// src/worker.rs:244-277
let result = {
    let mut engine = rag_engine.write().await;  // <-- Lock acquired
    // ...
    engine.add_document(filename, &data, Some(&mut batch_callback)).await
};  // <-- Lock released after this block

// Progress update happens OUTSIDE the lock
progress_state.current_batch = None;
// ...
```

### A.3: Atomic Persistence (Already Implemented)

```rust
// src/rag_engine.rs:1100-1105
// Atomic write: write to temp file, then rename
tokio::fs::write(&temp_path, data)
    .await
    .context("Failed to write index to temporary file")?;
tokio::fs::rename(&temp_path, &final_path)
    .await
    .context("Failed to commit index file (atomic rename)")?;
```

### A.4: spawn_blocking for WalkDir (Already Implemented)

```rust
// src/worker.rs:155-167
let pdf_paths: Vec<_> = tokio::task::spawn_blocking({
    let dir = documents_dir.to_string();
    move || {
        WalkDir::new(&dir)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path().extension().and_then(|s| s.to_str()) == Some("pdf")
            })
            .map(|e| e.path().to_path_buf())
            .collect::<Vec<_>>()
    }
})
.await?;
```

---

## Appendix B: AI Review Summary

### Codex (GPT) Key Recommendations

1. **Strengthen R1**: Clarify that heavy operations MUST run via `spawn_blocking`
2. **Add explicit lock-scope refactor task** if profiling shows issues
3. **Specify readiness semantics**: Index loaded, workers started, no critical errors
4. **Expand graceful shutdown**: Stop requests → signal tasks → timeout → flush

### Gemini Key Recommendations

1. **Use spawn_blocking, NOT tokio::process**: The external pdftotext is a black box - making the wait non-blocking doesn't prevent resource issues
2. **Add shutdown timeouts**: Lock acquisition timeout (10s), save timeout (5s)
3. **Clarify fallback strategy**: Try Rust lib first, fall back to pdftotext on failure
4. **De-prioritize investigation tasks**: "Verify lock duration" is P1, not P0

### Consensus Points (Both Reviewers)

- ✅ spawn_blocking is the correct Tokio pattern for blocking I/O
- ✅ Graceful shutdown needs timeout-based sequence
- ✅ Write lock issue is overstated - per-document locking is reasonable
- ✅ Pure-Rust PDF can be P2 since pdftotext fallback works
- ✅ Exit code verification important but low-risk (may already work)

### Final Review (Gemini 2.5 Pro - Second Pass)

**Assessment: READY TO IMPLEMENT**

Minor refinements applied:
1. **Test improvement**: Changed `.unwrap()` to `assert!(res.is_ok(), ...)` for better failure messages
2. **Axum integration note**: Added guidance to integrate shutdown logic with `axum::serve::with_graceful_shutdown`

**Sign-off**: "The plan is exceptionally thorough. The identified gaps are minor implementation details and do not represent fundamental design flaws."

### Final Review (Gemini 3 Pro Preview)

**Assessment: READY TO IMPLEMENT (Conditional)**

**CRITICAL ISSUE FOUND**: Race condition in temp file naming that previous reviewers missed.

#### The Bug
```rust
// Current code in extract_pdf_with_pdftotext:
let temp_file = temp_dir.join(format!("temp_pdf_{}.pdf", std::process::id()));
```

`std::process::id()` returns the PID, which is **constant for the entire process**. When concurrent PDF extraction is enabled via `spawn_blocking`, multiple threads will write to the **same filename**, causing:
- Race conditions
- File locking errors
- Processing wrong PDF content (data corruption)

#### The Fix
Replace `std::process::id()` with:
- `Uuid::new_v4()` (already in dependencies), OR
- `tempfile::NamedTempFile` crate

#### Additional Refinements
1. **PoisonError handling**: Added to graceful shutdown task
2. **Future optimization**: `tokio::process::Command` + `tokio::fs` is cleaner long-term (acknowledged as P2)

**Sign-off**: "READY TO IMPLEMENT — Conditional on fixing the temp file race condition."

---

*Plan revised 2025-12-05 incorporating Codex, Gemini 2.5 Pro, and Gemini 3 Pro Preview critical findings*
