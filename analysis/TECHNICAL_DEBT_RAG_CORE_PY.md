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
| **Status** | Open |

**Issue:** `DefaultHasher` is not stable across Rust versions; future toolchain bumps could silently change embeddings and break "golden" tests.

**Recommendation:** Switch to an explicit stable hash (e.g., sha2/blake3/xxhash) or document the determinism scope as "best-effort within current toolchain."

---

### TD-2: MockEmbeddingBackend Text Correlation
| Field | Value |
|-------|-------|
| **ID** | TD-2 |
| **Area** | MockEmbeddingBackend usefulness |
| **File** | `crates/rag-core-py/src/mock_backend.rs` |
| **Priority** | Low |
| **Status** | Open |

**Issue:** Hash→pseudo-random vectors are deterministic but do not correlate with textual similarity; future search/ranking tests may feel arbitrary (deterministic noise).

**Recommendation:** Add an alternate mock mode that produces similarity correlated with token overlap (e.g., hashed bag-of-words / char-ngram) and optionally an error-injection mode for exception-path tests.

---

### TD-3: MockEmbeddingBackend Value Distribution
| Field | Value |
|-------|-------|
| **ID** | TD-3 |
| **Area** | MockEmbeddingBackend implementation |
| **File** | `crates/rag-core-py/src/mock_backend.rs` |
| **Priority** | Low |
| **Status** | Open |

**Issue:** The `seed >> 33` scaling yields values biased toward [-1, 0), then normalization hides it but reduces "randomness quality."

**Recommendation:** Use a full 32-bit slice (e.g., `(seed as u32)`), or convert bytes to `f32` in a well-defined way.

---

### TD-4: Exception Mapping Depth
| Field | Value |
|-------|-------|
| **ID** | TD-4 |
| **Area** | Error handling |
| **Files** | `crates/rag-core-py/src/errors.rs`, `crates/rag-core/src/error.rs` |
| **Priority** | Medium |
| **Status** | Open |

**Issue:** `engine_error_to_pyerr` is correct/exhaustive for current `rag_core::EngineError` variants, but it stringifies everything (no structured fields; no cause chaining).

**Recommendation:** In Phase 2+, consider attaching structured metadata (e.g., `path`, `operation`, `chunk_id`) and/or chaining the original error as `__cause__` where appropriate.

---

### TD-5: CI Python 3.13 Coverage
| Field | Value |
|-------|-------|
| **ID** | TD-5 |
| **Area** | CI/CD |
| **Files** | `.github/workflows/python-bindings.yml`, `crates/rag-core-py/pyproject.toml` |
| **Priority** | Low |
| **Status** | Open |

**Issue:** Python classifiers claim 3.13 support, but CI only runs 3.10–3.12; `extension-module` compilation is only implicitly covered via `pip install .`.

**Recommendation:** Add Python 3.13 to the test matrix (or remove the classifier for now) and consider an explicit `cargo test -p rag-core-py --features extension-module` build/test step.

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
| **Status** | Open |

**Issue:** The `is_coroutine()` helper calls `py.import("inspect")` on every invocation to check if a Python result is a coroutine. While negligible for typical usage (Python imports are cached), it adds minor overhead.

**Recommendation:** Cache the `iscoroutine` function reference in the adapter's inner state during construction, avoiding repeated module lookups.

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
| **Status** | Open |

**Issue:** Python backends with async `embed()` or `rerank()` methods fail with "no running event loop" when called from Rust's `block_on()`. The `pyo3_async_runtimes::tokio::into_future()` function requires a Python asyncio event loop to be running, but none exists when invoking from pure Rust context.

**Impact:** Users cannot use async Python embedding/reranking backends (e.g., those using aiohttp for HTTP calls). Sync Python backends work perfectly.

**Workaround:** Use synchronous Python methods. If async HTTP calls are needed, users can use `asyncio.run()` internally within their sync method wrapper.

**Recommendation:** Investigate proper asyncio event loop management:
1. Create a Python asyncio event loop in the adapter and manage it
2. Use `pyo3_asyncio::tokio::into_future_with_locals()` with proper event loop setup
3. Or document sync-only support as the recommended pattern

**Tests:** `tests/test_python_backends.py::TestAsyncPythonBackend` tests are skipped pending resolution.

---

## Summary

| Phase | Open | Resolved | Accepted | Total |
|-------|------|----------|----------|-------|
| Phase 1 | 5 | 1 | 0 | 6 |
| Phase 2 | 0 | 1 | 0 | 1 |
| Phase 3 | 1 | 1 | 1 | 3 |
| Phase 4 | 1 | 0 | 0 | 1 |
| **Total** | **7** | **3** | **1** | **11** |

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
