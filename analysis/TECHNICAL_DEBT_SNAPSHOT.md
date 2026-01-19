# Technical Debt Snapshot
Date: 2026-01-18 14:30 UTC
Repo: rust-local-rag

## Summary

| Marker | Count |
|--------|-------|
| TODO   | 1     |
| FIXME  | 0     |
| HACK   | 0     |
| XXX    | 0     |
| NOTE   | 2     |

**Top Files by Marker Count**
1. `data/segment.srx` (1)
2. `eval/rag_client.py` (1)
3. `tests/rag_integration.rs` (1)

## High-Risk Markers

### Correctness
- `tests/rag_integration.rs:181`: `// NOTE: The current implementation seems to return Some(relevance) even on fallback,`
  - **Risk**: Search relevance scoring might be misleading when fallback mechanisms are triggered (e.g., pure lexical search), potentially masking the fact that semantic search failed or was skipped.

### API/Integration
- `eval/rag_client.py:148`: `            # NOTE: The MCP server currently returns Markdown-formatted text, not JSON.`
  - **Risk**: The evaluation client expects structured JSON but receives Markdown. This indicates a contract mismatch between the MCP server tools and the client, leading to fragile parsing or potential evaluation errors.

## Findings List

data/segment.srx:5538:<!-- TODO: арт. - артист -->
eval/rag_client.py:148:            # NOTE: The MCP server currently returns Markdown-formatted text, not JSON.
tests/rag_integration.rs:181:    // NOTE: The current implementation seems to return Some(relevance) even on fallback,

## Next 5 Fixes

1. **Fix relevance scoring on fallback** (Correctness)
   - **Description**: Investigate `tests/rag_integration.rs`. Ensure fallback search (lexical only) returns `None` for relevance scores if not applicable, or strictly defines what the score represents to avoid misleading consumers.
   - **Effort**: S
   - **Verification**: `cargo test --test rag_integration`

2. **Standardize MCP server output format** (Integration)
   - **Description**: Address `eval/rag_client.py` note. Update MCP server tools (or the client) to strictly adhere to a JSON schema for search results, avoiding Markdown wrapping that complicates programmatic consumption.
   - **Effort**: M
   - **Verification**: `python -m eval.run evaluate --config baseline -v`

3. **Resolve segmentation rule TODO** (Data)
   - **Description**: Clarify and implement the expansion rule for "арт. - артист" in `data/segment.srx` to improve sentence boundary detection for Russian text (if supported).
   - **Effort**: S
   - **Verification**: Manual inspection of segmentation output or no regression in chunking.
