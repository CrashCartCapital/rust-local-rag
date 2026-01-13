# Technical Debt Log: rag-core-py

This document tracks technical debt identified during implementation of rag-core-py Python bindings.

---

## Phase 1: Foundation & TDD Setup

**Gate Review Date:** 2025-01-13
**Reviewers:** CRASH MCP, Codex (gpt-5.2-xhigh), Gemini (gemini-3-pro-preview)

### TD-1: MockEmbeddingBackend Hash Stability
| Field | Value |
|-------|-------|
| **ID** | TD-1 |
| **Area** | MockEmbeddingBackend determinism |
| **File** | `crates/rag-core-py/src/mock_backend.rs` |
| **Priority** | Low |
| **Status** | **Resolved** |

**Issue:** `DefaultHasher` is not stable across Rust versions; future toolchain bumps could silently change embeddings and break "golden" tests.

**Resolution:** Switched to SHA-256 (`sha2` crate) for stable, platform-independent hashing. The mock backend now produces deterministic embeddings that are guaranteed stable across Rust versions and platforms.

---

### TD-2: MockEmbeddingBackend Text Correlation
| Field | Value |
|-------|-------|
| **ID** | TD-2 |
| **Area** | MockEmbeddingBackend usefulness |
| **File** | `crates/rag-core-py/src/mock_backend.rs` |
| **Priority** | Low |
| **Status** | **Resolved** |

**Issue:** Hash→pseudo-random vectors are deterministic but do not correlate with textual similarity; future search/ranking tests may feel arbitrary (deterministic noise).

**Resolution:** Added `CorrelatedMockEmbeddingBackend` struct that uses bag-of-words hashing:
- Splits text into whitespace-delimited tokens
- Hashes each token using the existing SHA-256 method
- Sums all token vectors and normalizes
- Similar texts (with shared words) produce similar vectors
- Order-invariant (bag-of-words semantics)

This enables meaningful search/ranking tests where results correlate with actual text similarity.

---

### TD-3: MockEmbeddingBackend Value Distribution
| Field | Value |
|-------|-------|
| **ID** | TD-3 |
| **Area** | MockEmbeddingBackend implementation |
| **File** | `crates/rag-core-py/src/mock_backend.rs` |
| **Priority** | Low |
| **Status** | **Resolved** |

**Issue:** The `seed >> 33` scaling yields values biased toward [-1, 0), then normalization hides it but reduces "randomness quality."

**Resolution:** Now extracts full 32-bit values from SHA-256 hash bytes using `u32::from_le_bytes()`. For dimensions > 8, uses chained hashes with counter suffix. Values are uniformly distributed in [-1, 1] before normalization.

---

### TD-4: Exception Mapping Depth
| Field | Value |
|-------|-------|
| **ID** | TD-4 |
| **Area** | Error handling |
| **Files** | `crates/rag-core-py/src/errors.rs`, `crates/rag-core/src/error.rs` |
| **Priority** | Medium |
| **Status** | **Resolved** |

**Issue:** `engine_error_to_pyerr` is correct/exhaustive for current `rag_core::EngineError` variants, but it stringifies everything (no structured fields; no cause chaining).

**Resolution:** Enhanced `engine_error_to_pyerr()` to extract and attach structured metadata to Python exceptions:

- **Persistence errors**: `path`, `operation`, `source_kind`, `__cause__`
- **Validation errors**: `chunk_id`, `kind`, `expected`, `got` (for dimension mismatch)
- **Embedding errors**: `kind`, `timeout_secs` (for timeout errors)
- **Rerank errors**: `kind` (error/unavailable/invalid_response)
- **Document not found**: `document_name`, `kind`

Python users can now programmatically access error details:
```python
except ragcore.ValidationError as e:
    print(e.chunk_id)  # "chunk-123" or None
    print(e.kind)      # "dimension_mismatch"
    print(e.expected)  # 384
    print(e.got)       # 768
```

---

### TD-5: CI Python 3.13 Coverage
| Field | Value |
|-------|-------|
| **ID** | TD-5 |
| **Area** | CI/CD |
| **Files** | `.github/workflows/python-bindings.yml`, `crates/rag-core-py/pyproject.toml` |
| **Priority** | Low |
| **Status** | **Resolved** |

**Issue:** Python classifiers claim 3.13 support, but CI only runs 3.10–3.12; `extension-module` compilation is only implicitly covered via `pip install .`.

**Resolution:** Added Python 3.13 to the CI test matrix. The `python-tests` job now runs against Python 3.10, 3.11, 3.12, and 3.13 on all platforms (ubuntu, macos, windows).

---

### TD-6: Gitignore for Python Artifacts
| Field | Value |
|-------|-------|
| **ID** | TD-6 |
| **Area** | Repository hygiene |
| **File** | `.gitignore` |
| **Priority** | Low |
| **Status** | **Resolved** |

**Issue:** Local artifacts like `.venv/` and `.pytest_cache/` exist under `crates/rag-core-py/`; root `.gitignore` ignores `*.so` and `__pycache__`, but not `.venv` / `.pytest_cache`.

**Recommendation:** Ignore `.venv/`, `.pytest_cache/`, and common maturin/dist artifacts to reduce accidental commits and confusing diffs.

**Resolution:** Added `.venv/`, `.pytest_cache/`, `dist/`, `*.whl` to `.gitignore`. Also removed `analysis/` from gitignore to track PRD and technical debt files.

---

## Phase 2: Core Engine Bindings

**Start Date:** 2025-01-13

### TD-7: EmbeddingBackend Trait Not Object-Safe
| Field | Value |
|-------|-------|
| **ID** | TD-7 |
| **Area** | Trait object design |
| **Files** | `crates/rag-core/src/traits.rs` |
| **Priority** | High |
| **Status** | **Resolved** |

**Issue:** The `EmbeddingBackend` trait in rag-core uses RPITIT (`impl Future<Output = ...>`) for the `embed()` method, which makes it non-object-safe. This prevents using `Arc<dyn EmbeddingBackend>` directly.

**Impact:** Cannot create a single `PyRagEngine` type that works with arbitrary Python embedding backends without resolving object-safety.

**Resolution:** Implemented dual-trait pattern (additive, non-breaking):

1. **Added `DynEmbeddingBackend` trait** - Object-safe version using explicit `BoxFuture<'a, ...>` returns
2. **Added `DynRerank` trait** - Object-safe version of `Rerank` trait
3. **Blanket impls**: `impl<T: EmbeddingBackend> DynEmbeddingBackend for T` - automatic conversion
4. **Reverse impls**: `impl EmbeddingBackend for Arc<dyn DynEmbeddingBackend>` - enables `BoxedEmbedder` with `RagEngine`
5. **Type aliases**: `BoxedEmbedder`, `BoxedReranker` for convenience

**Key Design Decisions:**
- Used explicit `BoxFuture` instead of `#[async_trait]` macro (per Codex recommendation - cleaner, no proc-macro)
- Reverse impl clones `Arc` and owned strings to create `'static` futures (acceptable overhead for dynamic dispatch path)
- Zero breaking changes to existing `EmbeddingBackend`/`Rerank` users

**Validation:** CRASH analysis + Codex + Gemini Pro review approved the approach.

**Files Changed:**
- `crates/rag-core/Cargo.toml` - Added `futures-core` dependency
- `crates/rag-core/src/traits.rs` - Added ~100 lines for new traits, blanket impls, reverse impls
- `crates/rag-core/src/lib.rs` - Exported new types

---

## Phase 3: Backend Adapters

**Gate Review Date:** 2025-01-13
**Reviewers:** Gemini (gemini-3-pro-preview)

### TD-8: Inspect Module Import per Async Check
| Field | Value |
|-------|-------|
| **ID** | TD-8 |
| **Area** | Performance |
| **File** | `crates/rag-core-py/src/adapters.rs` |
| **Priority** | Low |
| **Status** | **Resolved** |

**Issue:** The `is_coroutine()` helper calls `py.import("inspect")` on every invocation to check if a Python result is a coroutine. While negligible for typical usage (Python imports are cached), it adds minor overhead.

**Resolution:** Implemented thread-local caching for the `inspect.iscoroutine` function reference using `thread_local!` with `RefCell<Option<PyObject>>`. The function is cached on first use and reused for subsequent calls within the same thread.

---

### TD-9: BackendRef Unsafe Send+Sync
| Field | Value |
|-------|-------|
| **ID** | TD-9 |
| **Area** | Thread safety |
| **File** | `crates/rag-core-py/src/adapters.rs` |
| **Priority** | Low |
| **Status** | **Accepted** |

**Issue:** `BackendRef(Py<PyAny>)` has `unsafe impl Send` and `unsafe impl Sync`. This is safe because all access to the Python object is gated through `Python::with_gil()`, but the unsafe marker requires careful maintenance.

**Recommendation:** This is an acceptable pattern for PyO3 interop. The safety invariant is documented in code comments. No action required unless PyO3 provides a safer pattern in future versions.

---

### TD-10: Double-Call for Async Python Methods
| Field | Value |
|-------|-------|
| **ID** | TD-10 |
| **Area** | Correctness |
| **File** | `crates/rag-core-py/src/adapters.rs` |
| **Priority** | Medium |
| **Status** | **Resolved** |

**Issue:** Initial implementation called Python methods twice for async backends - once to detect coroutine type, once to get the actual result.

**Resolution:** Refactored to use enum pattern (`EmbedCallResult`, `EmbedBatchCallResult`, `RerankCallResult`) that stores either the sync result OR the async future from a single Python call. The coroutine check happens on the result of the first call, not a separate call.

**Files Changed:**
- `crates/rag-core-py/src/adapters.rs` - Added enum types and refactored call_embed/call_embed_batch/call_rerank methods

---

## Phase 4: PyRagEngine Integration

**Gate Review Date:** 2025-01-13

### TD-11: Async Python Backends Require Event Loop
| Field | Value |
|-------|-------|
| **ID** | TD-11 |
| **Area** | Async interop |
| **Files** | `crates/rag-core-py/src/adapters.rs`, `crates/rag-core-py/src/engine.rs` |
| **Priority** | Medium |
| **Status** | **Resolved** |

**Issue:** Python backends with async `embed()` or `rerank()` methods fail with "no running event loop" when called from Rust's `block_on()`. The `pyo3_async_runtimes::tokio::into_future()` function requires a Python asyncio event loop to be running, but none exists when invoking from pure Rust context.

**Resolution:** Added `run_coroutine()` helper function with hybrid approach:

1. Checks for running Python event loop via `asyncio.get_running_loop()`
2. If loop exists: uses `into_future()` for true async interop
3. If no loop: uses `asyncio.run()` to execute coroutine in a fresh loop

Applied to `call_embed()`, `call_embed_batch()`, and `call_rerank()` methods.

**Caveats (documented in code):**
- `asyncio.run()` blocks the calling thread (appropriate for spawn_blocking)
- Loop-bound Python objects (aiohttp sessions, etc.) cannot be reused across loops
- Frequent loop creation has performance overhead vs persistent loop

**Tests:** Async Python backends now work. The `tests/test_python_backends.py::TestAsyncPythonBackend` tests can be unskipped.

---

### TD-12: Async Python API Uses spawn_blocking Workaround
| Field | Value |
|-------|-------|
| **ID** | TD-12 |
| **Area** | Async architecture |
| **File** | `crates/rag-core-py/src/engine.rs` |
| **Priority** | Low |
| **Status** | **Accepted** |

**Issue:** The `asearch()` and `aupsert_document()` async methods use `spawn_blocking + block_on` pattern instead of true async. This is because `RagEngine::search()` returns a future that borrows from `&self`, which doesn't satisfy `'static` bounds required by `future_into_py`.

**Impact:** Minor performance overhead (extra thread pool scheduling, blocks a thread during async operations). The Python API remains fully async-compatible and all tests pass.

**Validation:** Reviewed by both Gemini (gemini-3-pro-preview) and Codex (gpt-5.2-xhigh). Both confirmed:
- Pattern is acceptable as a temporary workaround
- Proper fix requires refactoring `rag_core::RagEngine` to support "snapshot" pattern (clone Arc fields before await)
- Current approach is functionally correct

**Recommendation:** In a future iteration, refactor `rag_core::RagEngine` to:
1. Split search into sync "snapshot" + async "work" steps
2. Or make async methods take `self: Arc<Self>` instead of `&self`
3. Or use `Arc<RwLock>` for read-only operations

**Tests:** `tests/test_async_api.py` - 13 tests validating async API parity and concurrency.

---

## Summary

| Phase | Open | Resolved | Accepted | Total |
|-------|------|----------|----------|-------|
| Phase 1 | 0 | 6 | 0 | 6 |
| Phase 2 | 0 | 1 | 0 | 1 |
| Phase 3 | 0 | 2 | 1 | 3 |
| Phase 4 | 0 | 1 | 1 | 2 |
| **Total** | **0** | **10** | **2** | **12** |

---

## Changelog

- **2025-01-13:** Phase 1 gate review - 6 items identified (TD-1 through TD-6)
- **2025-01-13:** TD-6 resolved - Updated .gitignore for Python artifacts
- **2025-01-13:** Phase 2 start - TD-7 identified (RPITIT object-safety issue)
- **2025-01-13:** TD-7 resolved - Implemented dual-trait pattern with DynEmbeddingBackend/DynRerank
- **2025-01-13:** Phase 3 gate review - 3 items identified (TD-8 through TD-10)
- **2025-01-13:** TD-9 accepted - BackendRef unsafe Send+Sync is acceptable pattern
- **2025-01-13:** TD-10 resolved - Refactored to enum pattern to avoid double-calling Python methods
- **2025-01-13:** Phase 4 gate review - TD-11 identified (async Python backends require event loop)
- **2025-01-13:** TD-12 accepted - Async Python API spawn_blocking workaround validated by Gemini/Codex
- **2025-01-13:** TD-1 resolved - Switched to SHA-256 for stable, platform-independent hashing
- **2025-01-13:** TD-3 resolved - Fixed value distribution using full 32-bit extraction from hash bytes
- **2025-01-13:** TD-5 resolved - Added Python 3.13 to CI test matrix
- **2025-01-13:** TD-8 resolved - Implemented thread-local caching for iscoroutine function reference
- **2025-01-13:** TD-2 resolved - Added CorrelatedMockEmbeddingBackend with bag-of-words text correlation
- **2025-01-13:** TD-4 resolved - Enhanced exception mapping with structured attributes and __cause__ chaining
- **2025-01-13:** TD-11 resolved - Hybrid async handling with run_coroutine helper for event loop detection
