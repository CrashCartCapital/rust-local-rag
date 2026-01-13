# PRD: rag-core Python Bindings (rag-core-py)

**Version:** 1.0
**Status:** APPROVED_WITH_CHANGES
**Created:** 2025-01-13
**Validated By:** Claude + Codex (gpt-5.2-xhigh) + Gemini (gemini-3-pro-preview)
**Validation Date:** 2025-01-13

---

## Executive Summary

This PRD defines the requirements, design, and implementation tasks for `rag-core-py`, a PyO3-based Python binding layer that exposes the `rag-core` Rust library to Python applications. This enables Python projects to directly import and use RAG functionality (chunking, embedding, hybrid search, reranking) without going through MCP protocol overhead.

**Problem:** The `rag-core` Rust library is currently only consumable by other Rust projects. The primary user (Ryan) works in Python and cannot leverage the library directly.

**Solution:** Create PyO3 bindings with a Pythonic API, supporting both sync and async patterns, allowing Python classes to implement embedding/reranking backends.

---

## Part 1: Requirements

### 1.1 Functional Requirements

| ID | Requirement | Priority | Acceptance Criteria |
|----|-------------|----------|---------------------|
| FR-1 | Python can import `ragcore` module | P0 | `from ragcore import RagEngine` works |
| FR-2 | Create/open RAG engine with custom embedding backend | P0 | `RagEngine.open(path, backend=MyBackend())` |
| FR-3 | Synchronous search API | P0 | `engine.search(query, top_k=5)` returns `list[SearchResult]` |
| FR-4 | Asynchronous search API | P0 | `await engine.asearch(query)` works in asyncio |
| FR-5 | Python classes can implement EmbeddingBackend | P0 | Python `embed()` method called by Rust engine |
| FR-6 | Python classes can implement Reranker | P1 | Python `rerank()` method called during search |
| FR-7 | Document ingestion (upsert) | P0 | `engine.upsert_document(name, text, hash)` |
| FR-8 | Document removal | P1 | `engine.remove_document(name)` |
| FR-9 | Persistence (save/load) | P0 | `engine.save()`, state persists across sessions |
| FR-10 | QuerySpec configuration | P1 | Weights, filters, diversity_factor configurable |
| FR-11 | Score breakdown in results | P1 | `result.scores.embedding`, `.lexical`, `.reranker` |
| FR-12 | Health/stats API | P2 | `engine.stats()` returns document/chunk counts |

### 1.2 Non-Functional Requirements

| ID | Requirement | Priority | Acceptance Criteria |
|----|-------------|----------|---------------------|
| NFR-1 | Type hints (.pyi stubs) | P0 | IDE autocompletion works for all public APIs |
| NFR-2 | Cross-platform wheels | P0 | pip install works on Linux, macOS, Windows |
| NFR-3 | Python 3.10+ support | P0 | abi3 targeting Python 3.10 minimum |
| NFR-4 | Pickle/multiprocessing support | P1 | `RagEngine` can be pickled for multiprocessing |
| NFR-5 | Ctrl-C (SIGINT) handling | P1 | Long operations can be interrupted |
| NFR-6 | Thread safety | P0 | Safe to use from multiple Python threads |
| NFR-7 | Memory safety | P0 | No leaks, no use-after-free at FFI boundary |
| NFR-8 | Error messages | P0 | Clear Python exceptions with actionable messages |

### 1.3 Constraints

| Constraint | Description |
|------------|-------------|
| C-1 | Must not modify `rag-core` public API (additive changes only) |
| C-2 | Rust generics cannot be exposed to Python - use trait objects |
| C-3 | Python callbacks run under GIL - concurrency must be controlled |
| C-4 | Rust panics must not cross FFI boundary - use `catch_unwind` |

### 1.4 Out of Scope

- Direct numpy array support (future enhancement)
- GPU/CUDA integration
- Distributed/multi-process engine (single-process only)
- Streaming search results

---

## Part 2: Design

### 2.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Python Application                           │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  from ragcore import RagEngine, QuerySpec, SearchResult   │  │
│  │  engine = RagEngine.open("./index", backend=MyBackend())  │  │
│  │  results = engine.search("query", top_k=5)                │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ PyO3 FFI
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  rag-core-py (PyO3 Bindings)                     │
│  ┌─────────────────────┐  ┌─────────────────────────────────┐  │
│  │  PyRagEngine        │  │  PyEmbeddingBackendAdapter      │  │
│  │  - search()         │  │  - wraps Py<PyAny>              │  │
│  │  - asearch()        │  │  - impl EmbeddingBackend        │  │
│  │  - upsert_document()│  │  - semaphore concurrency        │  │
│  └─────────────────────┘  └─────────────────────────────────┘  │
│  ┌─────────────────────┐  ┌─────────────────────────────────┐  │
│  │  PyQuerySpec        │  │  Error Mapping                  │  │
│  │  PySearchResult     │  │  RagError → Python exceptions   │  │
│  │  PyScoreBreakdown   │  │  catch_unwind at boundary       │  │
│  └─────────────────────┘  └─────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ Rust crate dependency
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      rag-core (Rust Library)                     │
│  RagEngine<B: EmbeddingBackend, R: Rerank>                      │
│  QuerySpec, SearchResult, DocumentChunk, etc.                   │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Key Design Decisions

#### 2.2.1 Generics Handling

**Decision:** Hide Rust generics behind trait objects.

```rust
// Internal type used by PyRagEngine
type ConcreteEngine = rag_core::RagEngine<
    Arc<dyn rag_core::EmbeddingBackend + Send + Sync>,
    Arc<dyn rag_core::Rerank + Send + Sync>,
>;
```

**Rationale:** Python cannot express Rust generics. Trait objects provide runtime polymorphism with minimal overhead (negligible vs embedding API calls).

#### 2.2.2 Async/Sync Dual API

**Decision:** Expose both sync and async methods.

| Method | Implementation | Use Case |
|--------|---------------|----------|
| `search()` | `py.allow_threads(\|\| runtime.block_on(...))` | Notebooks, scripts |
| `asearch()` | `pyo3_asyncio::tokio::future_into_py(...)` | FastAPI, async services |

**Rationale:** Python ecosystem is split. Forcing async-only would hurt notebook adoption; forcing sync-only would hurt web services.

#### 2.2.3 Python Backend Adapters

**Decision:** Rust adapter structs that call Python methods via `Py<PyAny>`.

```rust
pub struct PyEmbeddingBackendAdapter {
    obj: Py<PyAny>,           // Python object reference
    model_id: String,          // Cached from Python
    dimension: usize,          // Cached from Python
    semaphore: Arc<Semaphore>, // Concurrency control
}

#[async_trait]
impl EmbeddingBackend for PyEmbeddingBackendAdapter {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let _permit = self.semaphore.acquire().await?;
        // Call Python method, handle sync/async responses
    }
}
```

**Rationale:** Allows Python users to implement backends in pure Python while Rust orchestrates. Semaphore prevents overwhelming GIL.

#### 2.2.4 Error Hierarchy

```
ragcore.RagError (base)
├── ragcore.EmbeddingError
├── ragcore.RerankError
├── ragcore.IndexError
├── ragcore.ConfigError
└── ragcore.ValidationError
```

**Mapping:**

| Rust Error | Python Exception |
|------------|------------------|
| `EngineError::Embedding(...)` | `EmbeddingError` |
| `EngineError::Rerank(...)` | `RerankError` |
| `EngineError::Persistence(...)` | `IndexError` |
| `EngineError::Config(...)` | `ConfigError` |
| `EngineError::Validation(...)` | `ValidationError` |

### 2.3 Python API Surface

#### 2.3.1 Core Classes

```python
# ragcore/__init__.py exports

class RagEngine:
    """Core RAG engine with hybrid search."""

    @classmethod
    def open(cls, index_dir: str | Path,
             embedding_backend: EmbeddingBackendProtocol,
             reranker: RerankerProtocol | None = None,
             *, max_backend_concurrency: int = 4) -> RagEngine: ...

    @classmethod
    def create(cls, index_dir: str | Path,
               embedding_backend: EmbeddingBackendProtocol,
               reranker: RerankerProtocol | None = None) -> RagEngine: ...

    def search(self, query: str, *, top_k: int = 10,
               query_spec: QuerySpec | None = None) -> list[SearchResult]: ...

    async def asearch(self, query: str, *, top_k: int = 10,
                      query_spec: QuerySpec | None = None) -> list[SearchResult]: ...

    def upsert_document(self, name: str, text: str,
                        content_hash: str | None = None) -> int: ...

    async def aupsert_document(self, name: str, text: str,
                               content_hash: str | None = None) -> int: ...

    def remove_document(self, name: str) -> bool: ...

    def save(self) -> None: ...

    def stats(self) -> EngineStats: ...

    def list_documents(self) -> list[str]: ...


class QuerySpec:
    """Search configuration."""

    def __init__(self, *,
                 top_k: int = 10,
                 embedding_weight: float = 0.7,
                 lexical_weight: float = 0.3,
                 reranker_weight: float = 0.7,
                 initial_weight: float = 0.3,
                 diversity_factor: float = 0.0,
                 filters: dict[str, Any] | None = None) -> None: ...


class SearchResult:
    """Single search result with score breakdown."""

    id: str
    text: str
    document: str
    chunk_index: int
    page_number: int
    section: str | None
    score: float
    scores: ScoreBreakdown


class ScoreBreakdown:
    """Detailed scoring components."""

    embedding: float | None
    lexical: float | None
    initial: float | None
    reranker: float | None
    total: float


class EngineStats:
    """Engine health and statistics."""

    document_count: int
    chunk_count: int
    embedding_model: str
    embedding_dimension: int
```

#### 2.3.2 Backend Protocols

```python
# ragcore/protocols.py

from typing import Protocol, Sequence

class EmbeddingBackendProtocol(Protocol):
    """Protocol for embedding providers."""

    def model_id(self) -> str: ...
    def dimension(self) -> int: ...

    # Can be sync or async
    def embed(self, text: str) -> Sequence[float]: ...
    def embed_batch(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


class RerankerProtocol(Protocol):
    """Protocol for reranking providers."""

    def rerank(self, query: str,
               candidates: Sequence[RerankerCandidate]) -> Sequence[float]: ...
```

### 2.4 Crate Structure

```
crates/
├── rag-core/                 # Existing Rust library (unchanged)
│   └── src/
│       ├── lib.rs
│       ├── engine.rs
│       ├── traits.rs         # EmbeddingBackend, Rerank
│       └── ...
│
└── rag-core-py/              # NEW: PyO3 bindings
    ├── Cargo.toml
    ├── pyproject.toml        # maturin config
    ├── src/
    │   ├── lib.rs            # PyO3 module definition
    │   ├── engine.rs         # PyRagEngine
    │   ├── adapters.rs       # PyEmbeddingBackendAdapter, PyRerankerAdapter
    │   ├── types.rs          # PyQuerySpec, PySearchResult, PyScoreBreakdown
    │   ├── errors.rs         # Exception hierarchy + mapping
    │   └── conversions.rs    # Python ↔ Rust type conversions
    │
    ├── ragcore/              # Python package (thin wrapper + stubs)
    │   ├── __init__.py       # Re-exports from _native
    │   ├── py.typed          # PEP 561 marker
    │   └── _native.pyi       # Type stubs for IDE support
    │
    └── tests/
        ├── conftest.py       # Fixtures (MockBackend, test data)
        ├── test_engine.py    # Engine operations
        ├── test_search.py    # Search functionality
        ├── test_backends.py  # Python backend adapters
        ├── test_async.py     # Async API parity
        └── test_errors.py    # Error handling
```

### 2.5 Build Configuration

#### Cargo.toml

```toml
[package]
name = "rag-core-py"
version = "0.1.0"
edition = "2021"

[lib]
name = "_native"
crate-type = ["cdylib"]

[dependencies]
rag-core = { path = "../rag-core", features = ["persistence"] }
pyo3 = { version = "0.22", features = ["abi3-py310", "extension-module"] }
pyo3-asyncio = { version = "0.22", features = ["tokio-runtime"] }
tokio = { version = "1", features = ["rt-multi-thread", "sync"] }
async-trait = "0.1"

[dev-dependencies]
pyo3 = { version = "0.22", features = ["auto-initialize"] }
```

#### pyproject.toml

```toml
[build-system]
requires = ["maturin>=1.4,<2.0"]
build-backend = "maturin"

[project]
name = "ragcore"
version = "0.1.0"
requires-python = ">=3.10"
classifiers = [
    "Programming Language :: Rust",
    "Programming Language :: Python :: Implementation :: CPython",
]

[tool.maturin]
features = ["pyo3/extension-module"]
python-source = "ragcore"
module-name = "ragcore._native"
```

---

## Part 3: Implementation Tasks

### 3.1 Task Overview

| Phase | Section | Focus | Tasks | Gate Review |
|-------|---------|-------|-------|-------------|
| Phase 1 | 3.3 | Foundation & TDD Setup | 1.1-1.3 | CRASH + Codex + Gemini |
| Phase 2 | 3.4 | Core Engine Bindings | 2.1-2.4 | CRASH + Codex + Gemini |
| Phase 3 | 3.5 | Backend Adapters | 3.1-3.3 | CRASH + Codex + Gemini |
| Phase 4 | 3.6 | Async Support | 4.1-4.3 | CRASH + Codex + Gemini |
| Phase 5 | 3.7 | Polish & Distribution | 5.1-5.6 | CRASH + Codex + Gemini (FINAL) |

**Total Tasks:** 19 tasks across 5 phases
**Consultations Required:** 38 pre/post task consultations (gemini-3-flash-preview) + 5 phase gate reviews (full ensemble)

### 3.2 Mandatory Consultation Protocol

**CRITICAL:** Every task and phase requires AI ensemble consultation. This is non-negotiable.

#### Per-Task Consultations (gemini-3-flash-preview)

| Stage | Tool | Purpose | Format |
|-------|------|---------|--------|
| **PRE-TASK** | `gemini-3-flash-preview` | Planning validation | "About to implement [task]. Plan: [approach]. Confirm approach is sound or suggest changes." |
| **POST-TASK** | `gemini-3-flash-preview` | Implementation validation | "Completed [task]. Implementation: [summary]. Verify correctness and identify issues." |

#### Phase Gate Reviews (Full Ensemble)

At the end of each phase, before proceeding to the next:

```
□ CRASH MCP: Structured review of phase (3-5 steps)
   - Step 1: Summarize what was implemented
   - Step 2: Identify any deviations from PRD
   - Step 3: List any technical debt or TODOs
   - Step 4: Confirm readiness for next phase

□ Codex (gpt-5.2-xhigh): Deep architecture review
   - Review all code changes from phase
   - Validate against PRD design decisions
   - Check for security/safety issues

□ Gemini (gemini-3-pro-preview): TDD & quality review
   - Verify all tests pass
   - Check test coverage adequacy
   - Validate edge case handling

□ GATE DECISION: All three must approve before next phase
```

#### Consultation Templates

**Pre-Task (Flash):**
```
CONTEXT: Pre-task planning for Claude implementing rag-core-py.
TASK: [Task ID and name]
PLANNED APPROACH: [Your implementation plan]
QUESTIONS:
1. Is this approach sound?
2. Any edge cases to handle?
3. Suggested test cases?
OUTPUT: JSON with 'approved', 'concerns', 'suggestions' keys.
```

**Post-Task (Flash):**
```
CONTEXT: Post-task validation for Claude implementing rag-core-py.
TASK: [Task ID and name]
IMPLEMENTATION SUMMARY: [What was done]
FILES CHANGED: [List of files]
TESTS ADDED: [Test names]
OUTPUT: JSON with 'validated', 'issues', 'improvements' keys.
```

**Phase Gate (Pro + Codex):**
```
CONTEXT: Phase gate review for rag-core-py Phase [N].
PHASE GOAL: [Goal from PRD]
TASKS COMPLETED: [List]
CODE CHANGES: @[paths to key files]
TEST RESULTS: [Summary]
OUTPUT: JSON with 'phase_approved', 'blockers', 'technical_debt', 'ready_for_next' keys.
```

---

### 3.3 Phase 1: Foundation & TDD Setup

**Goal:** Establish project structure, CI/CD, and testing infrastructure.

#### Task 1.1: Create crate structure

**PRE-TASK CONSULT (gemini-3-flash-preview):** □
```
□ Create crates/rag-core-py/
□ Add Cargo.toml with dependencies
□ Add pyproject.toml for maturin
□ Create src/lib.rs with empty module
□ Create ragcore/__init__.py
□ Verify: `maturin develop` succeeds
```
**POST-TASK CONSULT (gemini-3-flash-preview):** □

**TDD:** Write failing test first:
```python
# tests/test_import.py
def test_module_imports():
    import ragcore
    assert hasattr(ragcore, "RagEngine")
```

#### Task 1.2: Set up CI/CD pipeline

**PRE-TASK CONSULT (gemini-3-flash-preview):** □
```
□ Create .github/workflows/python-bindings.yml
□ Configure matrix: ubuntu, macos, windows
□ Add steps: cargo test, maturin build, pytest
□ Add ASAN leak check step (Linux only)
□ Verify: CI runs on PR
```
**POST-TASK CONSULT (gemini-3-flash-preview):** □

**CI Configuration:**
```yaml
name: Python Bindings
on: [push, pull_request]
jobs:
  test:
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install maturin pytest
      - run: cd crates/rag-core-py && maturin develop
      - run: cd crates/rag-core-py && cargo test
      - run: cd crates/rag-core-py && pytest tests/
```

#### Task 1.3: Create MockEmbeddingBackend for testing

**PRE-TASK CONSULT (gemini-3-flash-preview):** □
```
□ Add MockEmbeddingBackend in rag-core (test feature)
□ Returns deterministic embeddings based on text hash
□ Expose via test constructor in bindings
□ Verify: Mock backend usable in pytest
```
**POST-TASK CONSULT (gemini-3-flash-preview):** □

**TDD:**
```python
# tests/conftest.py
@pytest.fixture
def mock_engine(tmp_path):
    return RagEngine.create_mock(str(tmp_path))

# tests/test_engine.py
def test_engine_creates(mock_engine):
    assert mock_engine is not None
    assert mock_engine.stats().chunk_count == 0
```

---

#### PHASE 1 GATE REVIEW

Before proceeding to Phase 2, complete all gate checks:

```
□ CRASH MCP Review (3-5 structured steps)
  - Summarize Phase 1 deliverables
  - Verify crate structure matches PRD
  - Confirm CI/CD pipeline functional
  - Identify any gaps or technical debt

□ Codex Review (gpt-5.2-xhigh, timeout: 700s)
  - Review Cargo.toml dependencies
  - Validate pyproject.toml configuration
  - Check MockEmbeddingBackend design

□ Gemini Review (gemini-3-pro-preview)
  - Verify test infrastructure adequate
  - Confirm maturin build works cross-platform
  - Validate TDD approach

□ GATE DECISION: _____ (PASS / FAIL / PASS_WITH_NOTES)
□ GATE NOTES: _____________________________________
```

---

### 3.4 Phase 2: Core Engine Bindings

**Goal:** Expose RagEngine to Python with basic operations.

#### Task 2.1: Implement PyRagEngine struct

**PRE-TASK CONSULT (gemini-3-flash-preview):** □
```
□ Create #[pyclass] PyRagEngine
□ Store Arc<ConcreteEngine> internally
□ Implement __new__ / constructors
□ Add panic boundary with catch_unwind
□ Verify: Can instantiate from Python
```
**POST-TASK CONSULT (gemini-3-flash-preview):** □

**TDD:**
```rust
// src/engine.rs
#[cfg(test)]
mod tests {
    #[test]
    fn test_engine_creation() {
        // Rust unit test
    }
}
```

```python
def test_engine_create(tmp_path):
    engine = RagEngine.create(str(tmp_path), backend=MockBackend())
    assert isinstance(engine, RagEngine)
```

#### Task 2.2: Implement sync search method

**PRE-TASK CONSULT (gemini-3-flash-preview):** □
```
□ Add #[pymethods] search()
□ Use py.allow_threads for GIL release
□ Convert QuerySpec Python → Rust
□ Convert SearchResult Rust → Python
□ Verify: Search returns results
```
**POST-TASK CONSULT (gemini-3-flash-preview):** □

**TDD:**
```python
def test_search_basic(mock_engine):
    mock_engine.upsert_document("test.txt", "hello world")
    results = mock_engine.search("hello", top_k=5)
    assert len(results) > 0
    assert results[0].document == "test.txt"
```

#### Task 2.3: Implement document operations

**PRE-TASK CONSULT (gemini-3-flash-preview):** □
```
□ Add upsert_document()
□ Add remove_document()
□ Add list_documents()
□ Add save()
□ Verify: CRUD operations work
```
**POST-TASK CONSULT (gemini-3-flash-preview):** □

**TDD:**
```python
def test_document_lifecycle(mock_engine):
    # Upsert
    count = mock_engine.upsert_document("doc.txt", "content here")
    assert count > 0
    assert "doc.txt" in mock_engine.list_documents()

    # Remove
    removed = mock_engine.remove_document("doc.txt")
    assert removed
    assert "doc.txt" not in mock_engine.list_documents()
```

#### Task 2.4: Implement type conversions

**PRE-TASK CONSULT (gemini-3-flash-preview):** □
```
□ Create PyQuerySpec with all fields
□ Create PySearchResult with score breakdown
□ Create PyScoreBreakdown
□ Implement From<> traits for conversion
□ Verify: All fields accessible from Python
```
**POST-TASK CONSULT (gemini-3-flash-preview):** □

**TDD:**
```python
def test_search_result_fields(mock_engine):
    mock_engine.upsert_document("test.txt", "test content")
    results = mock_engine.search("test")

    r = results[0]
    assert isinstance(r.id, str)
    assert isinstance(r.text, str)
    assert isinstance(r.score, float)
    assert r.scores.total == r.score
```

---

#### PHASE 2 GATE REVIEW

Before proceeding to Phase 3, complete all gate checks:

```
□ CRASH MCP Review (3-5 structured steps)
  - Summarize Phase 2 deliverables (PyRagEngine, search, CRUD, types)
  - Verify trait object bounds (Arc<dyn T + Send + Sync + 'static>)
  - Confirm catch_unwind at all FFI boundaries
  - Identify any gaps or technical debt

□ Codex Review (gpt-5.2-xhigh, timeout: 700s)
  - Review PyRagEngine struct design
  - Validate GIL release in search()
  - Check type conversion correctness

□ Gemini Review (gemini-3-pro-preview)
  - Verify all TDD tests pass
  - Check sync API completeness
  - Validate error handling paths

□ GATE DECISION: _____ (PASS / FAIL / PASS_WITH_NOTES)
□ GATE NOTES: _____________________________________
```

---

### 3.5 Phase 3: Backend Adapters

**Goal:** Allow Python classes to implement embedding/reranking backends.

#### Task 3.1: Implement PyEmbeddingBackendAdapter

**PRE-TASK CONSULT (gemini-3-flash-preview):** □
```
□ Create adapter struct with Py<PyAny>
□ Cache model_id and dimension at construction
□ Add semaphore for concurrency control
□ Implement EmbeddingBackend trait
□ Handle both sync and async Python methods
□ Verify: Python backend called from Rust
```
**POST-TASK CONSULT (gemini-3-flash-preview):** □

**TDD:**
```python
class TestBackend:
    def model_id(self) -> str:
        return "test-model"

    def dimension(self) -> int:
        return 3

    def embed(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]

def test_python_backend(tmp_path):
    engine = RagEngine.create(str(tmp_path), backend=TestBackend())
    engine.upsert_document("test.txt", "hello")
    results = engine.search("hello")
    assert len(results) > 0
```

#### Task 3.2: Implement async Python backend support

**PRE-TASK CONSULT (gemini-3-flash-preview):** □
```
□ Detect awaitable return values
□ Use pyo3_asyncio to await Python coroutines
□ Handle mixed sync/async methods
□ Use spawn_blocking for sync backends in async context
□ Verify: Async Python backends work
```
**POST-TASK CONSULT (gemini-3-flash-preview):** □

**TDD:**
```python
class AsyncBackend:
    def model_id(self) -> str:
        return "async-model"

    def dimension(self) -> int:
        return 3

    async def embed(self, text: str) -> list[float]:
        await asyncio.sleep(0.01)  # Simulate async work
        return [0.1, 0.2, 0.3]

@pytest.mark.asyncio
async def test_async_backend(tmp_path):
    engine = RagEngine.create(str(tmp_path), backend=AsyncBackend())
    engine.upsert_document("test.txt", "hello")
    results = await engine.asearch("hello")
    assert len(results) > 0
```

#### Task 3.3: Implement PyRerankerAdapter

**PRE-TASK CONSULT (gemini-3-flash-preview):** □
```
□ Create adapter struct similar to embedding adapter
□ Define RerankerCandidate Python class
□ Implement Rerank trait
□ Handle optional reranker (None = no-op)
□ Verify: Python reranker called during search
```
**POST-TASK CONSULT (gemini-3-flash-preview):** □

**TDD:**
```python
class TestReranker:
    def rerank(self, query: str, candidates: list) -> list[float]:
        return [c.initial_score * 1.1 for c in candidates]

def test_python_reranker(tmp_path):
    engine = RagEngine.create(
        str(tmp_path),
        backend=TestBackend(),
        reranker=TestReranker()
    )
    engine.upsert_document("test.txt", "hello world")
    results = engine.search("hello")
    assert results[0].scores.reranker is not None
```

---

#### PHASE 3 GATE REVIEW

Before proceeding to Phase 4, complete all gate checks:

```
□ CRASH MCP Review (3-5 structured steps)
  - Summarize Phase 3 deliverables (Python backend adapters)
  - Verify GIL handling in callback adapters
  - Confirm semaphore concurrency control works
  - Validate sync/async Python method detection

□ Codex Review (gpt-5.2-xhigh, timeout: 700s)
  - Review PyEmbeddingBackendAdapter design
  - Validate Python::with_gil usage
  - Check spawn_blocking for sync backends in async

□ Gemini Review (gemini-3-pro-preview)
  - Verify callback exception handling
  - Check async Python backend tests
  - Validate reranker integration

□ GATE DECISION: _____ (PASS / FAIL / PASS_WITH_NOTES)
□ GATE NOTES: _____________________________________
```

---

### 3.6 Phase 4: Async Support

**Goal:** Full async API with proper Tokio runtime management.

#### Task 4.1: Implement async search

**PRE-TASK CONSULT (gemini-3-flash-preview):** □
```
□ Add asearch() method using pyo3_asyncio
□ Initialize shared Tokio runtime
□ Handle runtime already exists case
□ Add event loop detection in sync wrappers (error if called from running loop)
□ Verify: Works in asyncio.run()
```
**POST-TASK CONSULT (gemini-3-flash-preview):** □

**TDD:**
```python
@pytest.mark.asyncio
async def test_async_search(mock_engine):
    mock_engine.upsert_document("test.txt", "async test content")
    results = await mock_engine.asearch("async", top_k=5)
    assert len(results) > 0
```

#### Task 4.2: Implement async document operations

**PRE-TASK CONSULT (gemini-3-flash-preview):** □
```
□ Add aupsert_document()
□ Add aremove_document() (if needed)
□ Verify: Async document ops work
```
**POST-TASK CONSULT (gemini-3-flash-preview):** □

#### Task 4.3: Verify sync/async parity

**PRE-TASK CONSULT (gemini-3-flash-preview):** □
```
□ Create parity test suite
□ Verify identical results for sync/async
□ Verify no deadlocks when mixing
□ Add warning for sync-in-async-context
□ Verify: Both APIs produce same results
```
**POST-TASK CONSULT (gemini-3-flash-preview):** □

**TDD:**
```python
@pytest.mark.asyncio
async def test_sync_async_parity(mock_engine):
    mock_engine.upsert_document("test.txt", "parity test")

    sync_results = mock_engine.search("parity", top_k=5)
    async_results = await mock_engine.asearch("parity", top_k=5)

    assert len(sync_results) == len(async_results)
    for s, a in zip(sync_results, async_results):
        assert s.id == a.id
        assert abs(s.score - a.score) < 0.0001
```

---

#### PHASE 4 GATE REVIEW

Before proceeding to Phase 5, complete all gate checks:

```
□ CRASH MCP Review (3-5 structured steps)
  - Summarize Phase 4 deliverables (async API)
  - Verify Tokio runtime management
  - Confirm event loop detection in sync wrappers
  - Validate sync/async parity tests pass

□ Codex Review (gpt-5.2-xhigh, timeout: 700s)
  - Review pyo3_asyncio integration
  - Validate no deadlock scenarios
  - Check cancellation semantics

□ Gemini Review (gemini-3-pro-preview)
  - Verify async tests pass
  - Check sync/async parity completeness
  - Validate error messages for loop detection

□ GATE DECISION: _____ (PASS / FAIL / PASS_WITH_NOTES)
□ GATE NOTES: _____________________________________
```

---

### 3.7 Phase 5: Polish & Distribution

**Goal:** Production-ready package with full IDE support.

#### Task 5.1: Generate type stubs (.pyi)

**PRE-TASK CONSULT (gemini-3-flash-preview):** □
```
□ Create ragcore/_native.pyi manually or with stubgen
□ Add py.typed marker file
□ Verify IDE autocompletion works
□ Add to package_data in pyproject.toml
```
**POST-TASK CONSULT (gemini-3-flash-preview):** □

**Verification:**
```python
# In IDE, should show type hints:
from ragcore import RagEngine
engine: RagEngine = RagEngine.open(...)  # IDE shows signature
```

#### Task 5.2: Implement pickle support

**PRE-TASK CONSULT (gemini-3-flash-preview):** □
```
□ Add __getstate__ / __setstate__ to PyRagEngine
□ Serialize index_dir and backend config
□ Document: Pickling depends on backend being picklable
□ Add validation/warning for non-picklable backends
□ Verify: Works with multiprocessing
```
**POST-TASK CONSULT (gemini-3-flash-preview):** □

**TDD:**
```python
def test_pickle_roundtrip(mock_engine):
    import pickle

    mock_engine.upsert_document("test.txt", "pickle test")

    pickled = pickle.dumps(mock_engine)
    restored = pickle.loads(pickled)

    results = restored.search("pickle")
    assert len(results) > 0
```

#### Task 5.3: Implement signal handling

**PRE-TASK CONSULT (gemini-3-flash-preview):** □
```
□ Add Python::check_signals() calls in long operations
□ Test Ctrl-C interruption
□ Verify: KeyboardInterrupt raised properly
```
**POST-TASK CONSULT (gemini-3-flash-preview):** □

#### Task 5.4: Error handling hardening

**PRE-TASK CONSULT (gemini-3-flash-preview):** □
```
□ Implement full error hierarchy
□ Add catch_unwind at all FFI entry points
□ Verify: No panics escape to Python
□ Add descriptive error messages
```
**POST-TASK CONSULT (gemini-3-flash-preview):** □

**TDD:**
```python
def test_error_handling():
    with pytest.raises(ragcore.ConfigError):
        RagEngine.open("/nonexistent/path", backend=TestBackend())

    with pytest.raises(ragcore.EmbeddingError):
        # Backend that raises
        class FailingBackend:
            def model_id(self): return "fail"
            def dimension(self): return 3
            def embed(self, text): raise RuntimeError("fail")

        engine = RagEngine.create("/tmp/test", backend=FailingBackend())
        engine.upsert_document("test.txt", "content")
```

#### Task 5.5: Documentation and examples

**PRE-TASK CONSULT (gemini-3-flash-preview):** □
```
□ Add docstrings to all public classes/methods
□ Create examples/ directory with usage samples
□ Add README.md for the crate
□ Verify: help(ragcore.RagEngine) shows docs
```
**POST-TASK CONSULT (gemini-3-flash-preview):** □

#### Task 5.6: Release preparation

**PRE-TASK CONSULT (gemini-3-flash-preview):** □
```
□ Set up PyPI publishing workflow (maturin + cibuildwheel)
□ Configure wheel matrix: manylinux (x86_64/aarch64), macOS universal2, Windows
□ Define MSRV and Python version support policy
□ Add CHANGELOG.md
□ Tag v0.1.0 release
□ Verify: pip install ragcore from PyPI works
```
**POST-TASK CONSULT (gemini-3-flash-preview):** □

---

#### PHASE 5 GATE REVIEW (FINAL)

Before declaring project complete:

```
□ CRASH MCP Review (3-5 structured steps)
  - Summarize all Phase 5 deliverables
  - Verify type stubs provide IDE support
  - Confirm pickle/signal/error handling complete
  - Final technical debt assessment

□ Codex Review (gpt-5.2-xhigh, timeout: 700s)
  - Full codebase review against PRD
  - Security audit (FFI boundaries, panic safety)
  - Performance assessment

□ Gemini Review (gemini-3-pro-preview)
  - Final test coverage assessment
  - Documentation completeness
  - Release readiness verification

□ FINAL GATE DECISION: _____ (RELEASE / HOLD / NEEDS_WORK)
□ FINAL NOTES: _____________________________________
```

---

## Part 4: Validation Checklist

### Pre-Implementation Validation

- [x] Architecture reviewed by Codex (gpt-5.2-xhigh) ✓
- [x] TDD strategy validated by Gemini (gemini-3-pro-preview) ✓
- [x] Edge cases identified (GIL, panics, signals) ✓
- [x] CI/CD pipeline designed ✓

### Post-Implementation Validation

- [ ] All tests pass on Linux, macOS, Windows
- [ ] Type stubs provide IDE autocompletion
- [ ] No memory leaks (ASAN clean)
- [ ] Ctrl-C interruption works
- [ ] Pickle/multiprocessing works
- [ ] Error messages are actionable
- [ ] Documentation complete

### Ensemble Sign-Off

| Validator | Status | Notes |
|-----------|--------|-------|
| Claude | APPROVED | Primary implementation orchestrator |
| Codex (gpt-5.2-xhigh) | APPROVED_WITH_CHANGES | Architecture review complete |
| Gemini (gemini-3-pro-preview) | APPROVED_WITH_CHANGES | TDD/edge case review complete |

---

## Part 5: Ensemble Validation Results

### 5.1 Validation Summary

**Final Verdict: APPROVED_WITH_CHANGES**

Both Codex and Gemini validated the PRD as architecturally sound with recommended clarifications.

### 5.2 Gemini Feedback

**Strengths Identified:**
- Excellent Python idioms (Protocols for DI, dual sync/async APIs)
- Robust FFI safety strategy (GIL management, panic handling, signal interruption)
- Sound trait object decision to mask Rust generics
- Comprehensive TDD plan with both Rust and Python tests

**Gaps Identified:**
1. **Pickle ambiguity (Task 5.2):** `RagEngine` pickling depends on the injected backend being picklable. If user's backend isn't picklable, the engine won't be either.
2. **Async Python backends:** The mechanism for Rust/Tokio to await Python coroutines needs precise implementation to avoid deadlocks.

**Recommendations:**
- Refine Task 5.2: Clarify `__getstate__` behavior and add validation/warning for non-picklable backends
- Add model_id validation in `RagEngine.open` to verify backend matches existing index metadata
- In Task 3.2, handle sync Python backends in async context with `spawn_blocking`

### 5.3 Codex Feedback

**Strengths Identified:**
- Trait objects appropriate for stable PyO3 boundary
- Dual sync/async APIs acknowledge real Python usage patterns
- Callback adapters enable Python extensibility
- TDD direction with MockEmbeddingBackend is correct
- Cross-platform CI/CD explicitly considered

**Gaps Identified:**
1. **Trait object safety:** Must explicitly enforce `Arc<dyn Trait + Send + Sync + 'static>` and object safety
2. **GIL/threading policy:** Callbacks must only be invoked with GIL held; must not hold GIL across `.await`
3. **Async bridging:** Need precise runtime ownership story, cancellation semantics, nested loop handling
4. **Python integration tests:** Rust-unit tests alone won't catch GIL issues, refcount leaks, asyncio bugs
5. **Wheel/build risks:** manylinux policy, macOS universal2, Windows MSVC, MSRV pinning need detail

**Recommendations:**
- Standardize on `Arc<dyn EmbeddingBackend + Send + Sync + 'static>` with `async_trait` for object safety
- Define strict rule: Python callbacks via `Python::with_gil`, heavy work outside GIL
- Use `pyo3_asyncio::tokio` for async; sync wrappers should error if called from within running event loop
- Add pytest tests for: sync API, async API, callback exceptions, concurrency, cancellation
- Specify CI matrix with manylinux/universal2/Windows, use maturin/cibuildwheel, define MSRV

### 5.4 Accepted Changes

The following changes are accepted and should be incorporated during implementation:

| Change | Source | Priority | Implementation Note |
|--------|--------|----------|---------------------|
| Explicit trait bounds `Arc<dyn T + Send + Sync + 'static>` | Codex | P0 | Update design section 2.2.1 |
| Backend model_id validation on open | Gemini | P1 | Add to Task 2.1 |
| Pickle depends on backend picklability | Gemini | P1 | Document in Task 5.2 |
| spawn_blocking for sync backends in async | Gemini | P1 | Add to Task 3.2 |
| Event loop detection in sync wrappers | Codex | P1 | Add to Task 4.1 |
| Extended Python integration test suite | Codex | P0 | Add to Task 1.2 |
| Detailed wheel build matrix | Codex | P1 | Expand CI config in Task 1.2 |

### 5.5 Validation Process Documentation

**Tools Used:**
- CRASH MCP: Structured reasoning (4 steps)
- Codex (gpt-5.2-xhigh via codex-bridge): Architecture analysis + final validation
- Gemini (gemini-3-pro-preview): TDD strategy + final validation
- Gemini (gemini-3-flash-preview): Pre-validation quick check

**Methodology:**
1. CRASH Step 1: Plan PRD structure and identify key technical decisions
2. Codex: Deep architecture analysis for PyO3 patterns
3. Gemini Pro: TDD strategy and edge case identification
4. CRASH Step 2-3: Synthesize insights into PRD draft
5. Gemini Flash: Quick pre-validation pass
6. Codex + Gemini Pro (parallel): Final ensemble validation
7. CRASH Step 4: Synthesize validation results

---

## Appendix A: Edge Cases & Mitigations

| Edge Case | Risk | Mitigation |
|-----------|------|------------|
| GIL deadlock | High | Never hold Rust Mutex while acquiring GIL |
| Callback exception | High | Catch Python exceptions, convert to Rust Result |
| Rust panic at FFI | Critical | `catch_unwind` at all entry points |
| Signal handling | Medium | `Python::check_signals()` in loops |
| Memory leak | High | Proper `Py<T>` reference counting |
| Buffer protocol | Low | Start with Vec<f32>, add numpy later |

## Appendix B: Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| pyo3 | 0.22 | Python bindings |
| pyo3-asyncio | 0.22 | Async support |
| tokio | 1.x | Async runtime |
| async-trait | 0.1 | Object-safe async traits |
| maturin | 1.4+ | Build system |

## Appendix C: References

- [PyO3 User Guide](https://pyo3.rs/)
- [pyo3-asyncio Documentation](https://docs.rs/pyo3-asyncio)
- [maturin Documentation](https://www.maturin.rs/)
- [PEP 561 - Type Stubs](https://peps.python.org/pep-0561/)
