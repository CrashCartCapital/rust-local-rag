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
| **Files** | `crates/rag-core/src/traits.rs`, `crates/rag-core-py/src/adapters.rs` |
| **Priority** | High |
| **Status** | Open |

**Issue:** The `EmbeddingBackend` trait in rag-core uses RPITIT (`impl Future<Output = ...>`) for the `embed()` method, which makes it non-object-safe. This prevents using `Arc<dyn EmbeddingBackend>` directly.

**Impact:** Cannot create a single `PyRagEngine` type that works with arbitrary Python embedding backends without resolving object-safety.

**Recommendation Options:**
1. Add `#[async_trait]` to rag-core traits (additive but changes impl signature)
2. Create object-safe bridge traits in rag-core-py using `BoxFuture`
3. Use newtype wrapper pattern with internal `Arc<dyn DynBackend>`

**Current Workaround:** Phase 2 uses concrete `MockEmbeddingBackend` type. Full Python backend integration deferred to Phase 3.

**PRD Reference:** PRD Section 2.2.1 assumes trait objects will work; needs update.

---

## Summary

| Phase | Open | Resolved | Total |
|-------|------|----------|-------|
| Phase 1 | 5 | 1 | 6 |
| Phase 2 | 1 | 0 | 1 |
| **Total** | **6** | **1** | **7** |

---

## Changelog

- **2025-01-13:** Phase 1 gate review - 6 items identified (TD-1 through TD-6)
- **2025-01-13:** TD-6 resolved - Updated .gitignore for Python artifacts
- **2025-01-13:** Phase 2 start - TD-7 identified (RPITIT object-safety issue)
