# Technical Debt Snapshot

- **Timestamp:** 2026-01-26 10:25:19 UTC
- **Repo:** rust-local-rag

## Summary

| Marker | Count |
|--------|-------|
| TODO   | 1     |
| FIXME  | 0     |
| HACK   | 0     |
| XXX    | 0     |
| NOTE   | 13    |

### Top Files by Marker Count
1. crates/rag-core-py/src/engine.rs (2)
2. crates/rag-core-py/examples/custom_backend.py (1)
3. crates/rag-core-py/ragcore/_native.pyi (1)
4. crates/rag-core-py/src/conversions.rs (1)
5. crates/rag-core-py/src/errors.rs (1)
6. crates/rag-core/examples/ollama_backend.rs (1)
7. crates/rag-core/src/relationships.rs (1)
8. crates/rag-core/src/search.rs (1)
9. data/segment.srx (1)
10. eval/rag_client.py (1)

## High-risk markers

### Correctness / Determinism
- `tests/rag_integration.rs:181`: `// NOTE: The current implementation seems to return Some(relevance) even on fallback,`
- `eval/rag_client.py:148`: `# NOTE: The MCP server currently returns Markdown-formatted text, not JSON.`

### Data Correctness
- `data/segment.srx:5538`: `<!-- TODO: арт. - артист -->`

## Findings List

1. `crates/rag-core-py/examples/custom_backend.py:8`: `Note: This example uses a simple hash-based embedding for demonstration.`
2. `crates/rag-core-py/ragcore/_native.pyi:382`: `Note: Python backends must themselves be picklable (module-level classes)`
3. `crates/rag-core-py/src/conversions.rs:10`: `/// Note: rag-core uses page_number=0 as the default/unknown value.`
4. `crates/rag-core-py/src/engine.rs:415`: `// Note: We explicitly set diversity_factor=0.0 when no spec provided,`
5. `crates/rag-core-py/src/engine.rs:589`: `/// Note: Python backends must themselves be picklable for this to work.`
6. `crates/rag-core-py/src/errors.rs:117`: `// Note: rag-core-py always enables persistence feature in rag-core`
7. `crates/rag-core/examples/ollama_backend.rs:25`: `//! Note: This example uses a mock embedder since rag-core doesn't depend on reqwest.`
8. `crates/rag-core/src/relationships.rs:171`: `/// Note: This method does not deduplicate. Adding the same relationship`
9. `crates/rag-core/src/search.rs:31`: `/// Note: This is a re-export of the canonical implementation in tags.rs`
10. `data/segment.srx:5538`: `<!-- TODO: арт. - артист -->`
11. `eval/rag_client.py:148`: `# NOTE: The MCP server currently returns Markdown-formatted text, not JSON.`
12. `src/bin/rag_tui/config.rs:77`: `// Note: In practice, other env vars might be set`
13. `src/bin/rag_tui/main.rs:43`: `// Note: Mouse capture intentionally disabled to allow terminal text selection`
14. `tests/rag_integration.rs:181`: `// NOTE: The current implementation seems to return Some(relevance) even on fallback,`

## Next 5 fixes

1. **Investigate fallback relevance in tests**
   - **Fix:** Check `tests/rag_integration.rs` line 181. Verify if `Some(relevance)` on fallback is intended or a bug. If bug, ensure fallback returns `None` or correct value.
   - **Effort:** Small
   - **Verification:** `cargo test --test rag_integration`

2. **Handle Markdown vs JSON in Eval Client**
   - **Fix:** In `eval/rag_client.py` line 148, if the server is now returning JSON, remove the note and update parsing. If it still returns Markdown and needs JSON, implement a robust parser or update server.
   - **Effort:** Medium
   - **Verification:** `python -m eval.run evaluate --config baseline -v`

3. **Add abbreviation to SRX rules**
   - **Fix:** Add "арт." -> "артист" abbreviation rule to `data/segment.srx` (line 5538) and remove TODO.
   - **Effort:** Small
   - **Verification:** `cargo test` (assuming core tests cover segmentation rules, otherwise manual inspection).

4. **Verify Env Var Handling in TUI Config**
   - **Fix:** Review `src/bin/rag_tui/config.rs` line 77. Determine if other env vars *should* be explicitly handled/logged. If not, remove the note.
   - **Effort:** Small
   - **Verification:** `cargo run --bin rag_tui`

5. **Deduplication in Relationships**
   - **Fix:** In `crates/rag-core/src/relationships.rs` line 171, consider adding a check to prevent duplicate relationships if appropriate, or document why duplicates are allowed clearly.
   - **Effort:** Medium
   - **Verification:** `cargo test -p rag-core`
