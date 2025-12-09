# Mini PRD: Per-Query Score Weight Customization

**Status**: ✅ Implemented
**Date**: 2024-12-09
**Revised**: Simplified after ensemble review (Gemini + Codex)
**Implemented**: 2025-12-09

---

## 1. Problem

Weights are cached at startup via `OnceLock`. Users cannot experiment with different weight configurations without restarting the server.

**Goal**: Add optional weight overrides to `search_documents` MCP tool.

---

## 2. Requirements (MVP)

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-1 | MCP `search_documents` accepts optional weight parameters | Must |
| FR-2 | Per-query weights override cached defaults | Must |
| FR-3 | Omitted weights fall back to cached defaults | Must |
| FR-4 | Invalid weights (NaN, Inf, out of range) use defaults | Must |

**Out of Scope (for now)**: HTTP endpoint, TUI display, weight profiles, test coverage targets.

---

## 3. Design

### 3.1 Add One Struct

```rust
/// Optional per-query weight overrides
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct QueryWeights {
    pub embedding: Option<f32>,
    pub lexical: Option<f32>,
    pub reranker: Option<f32>,
    pub initial: Option<f32>,
}
```

### 3.2 Update SearchRequest

```rust
pub struct SearchRequest {
    pub query: String,
    pub top_k: Option<usize>,
    pub diversity_factor: Option<f32>,
    pub weights: Option<QueryWeights>,  // NEW
}
```

### 3.3 Resolve Inline (No Extra Struct)

At start of `search()`:
```rust
// Resolve weights: override if valid, else use cached default
let w_embed = weights
    .and_then(|w| w.embedding)
    .filter(|&v| v.is_finite() && (0.0..=1.0).contains(&v))
    .unwrap_or_else(get_embedding_weight);
let w_lex = weights
    .and_then(|w| w.lexical)
    .filter(|&v| v.is_finite() && (0.0..=1.0).contains(&v))
    .unwrap_or_else(get_lexical_weight);
let w_rerank = weights
    .and_then(|w| w.reranker)
    .filter(|&v| v.is_finite() && (0.0..=1.0).contains(&v))
    .unwrap_or_else(get_reranker_weight);
let w_init = weights
    .and_then(|w| w.initial)
    .filter(|&v| v.is_finite() && (0.0..=1.0).contains(&v))
    .unwrap_or_else(get_initial_score_weight);
```

Then use `w_embed`, `w_lex`, `w_rerank`, `w_init` in scoring instead of calling `get_*_weight()`.

---

## 4. Implementation

### Phase 1: Core Change (~1 hour)

| Task | Notes |
|------|-------|
| Add `QueryWeights` struct to rag_engine.rs | ~10 lines |
| Add `weights` field to `SearchRequest` | ~3 lines |
| Update `search()` signature to accept `Option<&QueryWeights>` | ~2 lines |
| Update `search_with_diversity()` to forward weights | ~2 lines |
| Resolve weights inline at start of `search()` | ~15 lines |
| Replace `get_*_weight()` calls with resolved values | ~4 lines |
| Update MCP `search_documents` to pass weights | ~3 lines |

### Phase 2: Testing (~30-45 min)

| Task | Notes |
|------|-------|
| Test: no overrides = same as current behavior | Basic sanity |
| Test: override one weight, others use defaults | Partial override |
| Test: invalid weight (NaN) falls back to default | Edge case |

---

## 5. Risks

| Risk | Mitigation |
|------|------------|
| Breaking callers | `weights` is `Option`, backward compatible |
| Invalid weights | Filter with `is_finite()` and range check |

---

## 6. Example Usage

```json
{
  "tool": "search_documents",
  "arguments": {
    "query": "machine learning",
    "top_k": 5,
    "weights": {
      "embedding": 0.9,
      "lexical": 0.1
    }
  }
}
```

Only `embedding` and `lexical` are overridden; `reranker` and `initial` use cached defaults.

---

## 7. Future (If Needed)

- HTTP endpoint support
- TUI weight input/display
- Weight profiles ("semantic-heavy", "keyword-focused")
- Logging of effective weights per query

---

**Estimated Time**: 1.5-2 hours total

---

## 8. Implementation Notes (2025-12-09)

### What Was Built

1. **`QueryWeights` struct** in `rag_engine.rs` with schemars JSON schema support
2. **`resolve_weight()` helper** - pure function for testability
3. **`ResolvedWeights` struct** with `from_query_weights()` constructor
4. **Updated `search()` and `search_with_diversity()`** signatures to accept `Option<&QueryWeights>`
5. **Updated `SearchRequest`** in `mcp_server.rs` with optional `weights` field
6. **12 unit tests** covering all resolution scenarios

### Key Design Decisions

- Used `rmcp::schemars::JsonSchema` to resolve schemars version conflict (rmcp uses 1.1.0)
- Created `ResolvedWeights` struct instead of inline resolution for better testability
- `resolve_weight()` is a standalone pure function for easy unit testing
- HTTP endpoint intentionally passes `None` (per-query weights via HTTP out of scope)

### Test Results (MCP Tool Calls)

| Test Case | Result |
|-----------|--------|
| Default (no weights) | ✅ Works as before |
| `reranker=0, initial=1` | ✅ Different results (bibliographies surfaced) |
| `reranker=1, initial=0` | ✅ Reranker-only scoring |
| Partial weights | ✅ Unspecified weights use defaults |
| All 4 weights | ✅ All overrides applied |
| Invalid values (>1, <0, NaN) | ✅ Silently ignored, defaults used |

### Files Modified

- `src/rag_engine.rs`: QueryWeights, resolve_weight, ResolvedWeights, search signatures
- `src/mcp_server.rs`: SearchRequest.weights field, search_documents tool
- `docs/per_query_weights_prd.md`: Status and implementation notes
