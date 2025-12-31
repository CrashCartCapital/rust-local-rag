# Test Gap Map Report

## 1. Critical Flows

Based on the codebase analysis, the following critical flows have been identified:

### A. Ingestion & Indexing (Critical)
The pipeline that turns PDFs into searchable chunks.
- **Steps**: PDF Upload/Detection -> Text Extraction (lopdf/pdftotext) -> Chunking (Sentence-aware) -> Embedding (Ollama) -> Storage (Chunks + ANN Index + Lexical Index).
- **Failure Modes**: Corrupt PDF, Extraction failure, Ollama downtime/timeout, Disk write failure, Index sync mismatch (ghost chunks).
- **Risk**: High. If this fails, the system has no knowledge.

### B. Search & Retrieval (Critical)
The user-facing value proposition.
- **Steps**: User Query -> Query Embedding -> ANN Search (Semantic) + Lexical Search (Keyword) -> Reranking (optional) -> Result Formatting.
- **Failure Modes**: Empty results for valid queries, Reranker timeout/failure, Scoring math errors, JSON serialization error.
- **Risk**: High. Poor results or errors destroy user trust.

### C. Job Management (Important)
Background processing to prevent timeouts on large datasets.
- **Steps**: Job Creation (dedupe) -> Background Processing (Worker) -> Status Updates -> Persistence (SQLite).
- **Failure Modes**: Race conditions on job creation (duplicate jobs), Worker crash, State inconsistency (stuck in "inprogress"), Poison pill document crashing the worker.
- **Risk**: Medium-High. Reliability issue.

### D. MCP Server & Tooling (User Facing)
The interface exposed to Claude.
- **Steps**: Request -> Parameter Validation -> Engine Call -> Response Formatting.
- **Failure Modes**: Invalid parameters, Engine lock contention, formatting panic.
- **Risk**: Medium.

## 2. Existing Tests Inventory

| Component | Test Type | Coverage | Strength | Weakness |
|-----------|-----------|----------|----------|----------|
| **RagEngine (Core)** | Unit | High (Logic) | Strong logic tests for Chunking, Indexing, MMR, Math. | **No end-to-end integration.** `add_document` -> `search` flow is never tested together. No mock for EmbeddingService. |
| **JobManager** | Integration | High (DB) | Good coverage of CRUD, status updates, concurrency (race conditions). | |
| **Worker** | Unit/Integration | Low | Only lock metrics tested. | `reindex_documents` logic is complex but untested (mocking FS/RagEngine/JobManager is hard). |
| **MCP Server** | Unit | Low | Only `format_search_results` is tested. | No tests for tool handlers (`search_documents`, `start_reindex`). |
| **Persistence** | Unit | Medium | File paths and atomic writes tested. | **Corrupt index recovery** not tested. |
| **PDF Extraction** | Integration | Medium | `async_pdf` tests non-blocking behavior. | No test for actual text content quality or fallback logic. |

## 3. Gap Map & Prioritization

| Flow | Current Coverage | Confidence | Missing Coverage | Proposed Test | Priority |
|------|------------------|------------|------------------|---------------|----------|
| **Search Pipeline (E2E)** | None (Unit only) | Low | Full flow: Add Doc -> Embed (Mock) -> Index -> Search -> Result. | `tests/rag_integration.rs`: `test_e2e_indexing_and_search` using `wiremock` for Ollama. | **Critical** |
| **Corrupt Index Recovery** | None | Low | System behavior when loading a corrupted/invalid JSON index file. | `tests/rag_persistence.rs`: `test_recover_from_corrupt_index` | **High** |
| **Reranker Fallback** | Manual Code Check | Medium | Fallback to embedding-only if reranker fails/times out. | `tests/rag_integration.rs`: `test_reranker_fallback_on_failure` | **High** |
| **Worker Job Flow** | None | Low | Full job lifecycle: Pending -> InProgress -> Completed. | `tests/worker_integration.rs`: `test_worker_completes_job` | **Medium** |
| **Index Sync** | Unit | Medium | `validate_index_sync` ensures consistency on load. | `tests/rag_persistence.rs`: `test_index_sync_fixes_inconsistencies` | **Medium** |

## 4. Proposed Tests (Implementation Plan)

I will add 3 new integration test files to cover the high-priority gaps.

1.  **`tests/rag_integration.rs`** (The "Golden Path" & Resilience)
    *   **Test 1: `test_e2e_indexing_and_search`**:
        *   Setup `wiremock` to mock Ollama `api/tags` and `api/embed`.
        *   Create `RagEngine`.
        *   Call `add_document` with a mock PDF content (needs write to disk).
        *   Call `search` and verify the document is found.
    *   **Test 2: `test_reranker_fallback_on_failure`**:
        *   Setup `RagEngine` with a reranker that (mock) returns 500 error.
        *   Call `search`.
        *   Verify results are returned (fallback worked) and no panic.

2.  **`tests/rag_persistence.rs`** (Data Safety)
    *   **Test 3: `test_recover_from_corrupt_index`**:
        *   Create a valid index, save it.
        *   Corrupt the file (write garbage).
        *   Initialize new `RagEngine`.
        *   Verify it doesn't panic and starts empty/reindexing.
    *   **Test 4: `test_index_sync_fixes_inconsistencies`**:
        *   Create `RagEngine`, add chunks.
        *   Manually tamper with `chunks` map (remove a chunk but leave it in `ann_index`).
        *   Trigger `validate_index_sync`.
        *   Verify consistency is restored.

3.  **`tests/worker_integration.rs`** (Background Processing)
    *   **Test 5: `test_worker_completes_job`**:
        *   This requires mocking the filesystem or writing real PDFs.
        *   Setup `JobManager` (sqlite memory) and `WorkerSupervisor`.
        *   Send `StartReindex` job.
        *   Wait and verify job status becomes `Completed`.

## 5. Validation

Run `cargo test` to ensure all new tests pass and no regressions in existing tests.
