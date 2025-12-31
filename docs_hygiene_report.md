# Documentation Hygiene Report

**Date**: 2025-12-31
**Branch**: `chore/docs-hygiene-20251231`
**Auditor**: Claude Code (Opus 4.5)

---

## Summary

| Metric | Count |
|--------|-------|
| **Total Files Audited** | 13 |
| **Files Kept** | 13 |
| **Files Archived** | 0 |
| **Files Deleted** | 0 |
| **Files Updated** | 5 |
| **Up-to-date** | 8 |
| **Marked Historical** | 1 |

---

## Inventory & Decisions

| Path | Type | Last Modified | Decision | Status | Action | Rationale |
|------|------|---------------|----------|--------|--------|-----------|
| `README.md` | README | 2025-12-09 | KEEP | Updated | Updated | Added docs index grouping, added per_query_weights_prd link |
| `CLAUDE.md` | AI Guidance | 2025-12-09 | KEEP | Updated | Updated | Fixed outdated reranker section (was Phi JSON, now Yes/No logprobs), added missing env vars |
| `AGENTS.md` | AI Guidance | 2025-12-16 | KEEP | Up-to-date | None | Recent synthesis, correct references |
| `docs/setup.md` | How-to | 2025-12-09 | KEEP | Updated | Updated | Added missing env vars: OLLAMA_RERANK_MODEL, PROMPTS_DIR, MCP_HTTP_BIND, MCP_HTTP_ENDPOINT |
| `docs/how-to-use.md` | How-to | 2025-12-09 | KEEP | Updated | Updated | Added missing env vars |
| `docs/RAG_EVALUATION_FRAMEWORK_SPEC.md` | Spec | 2025-12-08 | KEEP | Up-to-date | None | Comprehensive spec, accurate |
| `docs/RERANKER_DEBUGGING_POSTMORTEM.md` | Historical | 2025-12-07 | KEEP | Historical | Updated | Added historical disclaimer - describes OLD Phi-4-Mini JSON approach |
| `docs/per_query_weights_prd.md` | PRD | 2025-12-09 | KEEP | Up-to-date | None | Marked as implemented, accurate |
| `eval/README.md` | README | 2025-12-08 | KEEP | Up-to-date | None | Eval framework docs, accurate |
| `eval/ground_truth/README.md` | Dataset docs | 2025-12-08 | KEEP | Up-to-date | None | Ground truth schema, accurate |
| `eval/reports/BASELINE_EVALUATION_SUMMARY.md` | Report | 2025-12-08 | KEEP | Up-to-date | None | Evaluation results, preserved |
| `eval/reports/EXPERT_CORPUS_EVALUATION.md` | Report | 2025-12-08 | KEEP | Up-to-date | None | Expert evaluation, preserved |
| `prompts/reranker.txt` | Prompt | 2025-12-09 | KEEP | Up-to-date | None | Current Yes/No prompt format |

---

## Staleness Reasons Identified

1. **Reranker Architecture Change**: The reranker was migrated from Phi-4-Mini with JSON scoring to Qwen3-Reranker-style Yes/No with logprobs. CLAUDE.md still documented the old approach.

2. **Missing Environment Variables**: Several env vars were added to the codebase but not documented:
   - `PROMPTS_DIR` - Prompt template override directory
   - `MCP_HTTP_BIND` - HTTP health endpoint address
   - `MCP_HTTP_ENDPOINT` - HTTP MCP endpoint path
   - `OLLAMA_RERANK_MODEL` - (was missing from some docs)

3. **Documentation Navigation**: README.md docs section lacked structure (no grouping, missing per_query_weights_prd.md link).

---

## Key Changes Made

### 1. CLAUDE.md - Reranker Section Update
**Before**: Described Phi-4-Mini JSON format with `<|user|>...<|end|><|assistant|>` tokens, JSON pre-fill, 0-100 scoring rubric

**After**: Describes current Yes/No with logprobs approach:
- Binary classification (Yes/No)
- Softmax scoring from logprobs: `exp(yes_lp) / (exp(yes_lp) + exp(no_lp))`
- Stop sequence `\n`, temperature 0.0, num_predict 3

### 2. RERANKER_DEBUGGING_POSTMORTEM.md - Historical Marker
Added disclaimer:
> **HISTORICAL DOCUMENT**: This postmortem documents the debugging process for a **previous** reranker implementation using Phi-4-Mini with JSON scoring (December 2025). The current production system uses **Yes/No binary classification with logprobs-based scoring** (Qwen3-Reranker style).

### 3. Environment Variable Documentation
Added to CLAUDE.md, docs/setup.md, docs/how-to-use.md:
| Variable | Default | Purpose |
|----------|---------|---------|
| `OLLAMA_RERANK_MODEL` | `llama3.1` | LLM for reranking |
| `PROMPTS_DIR` | `./prompts` | Prompt template overrides |
| `MCP_HTTP_BIND` | `127.0.0.1:3046` | Health endpoint |
| `MCP_HTTP_ENDPOINT` | `/mcp` | MCP endpoint path |

### 4. README.md - Documentation Index
Reorganized into sections:
- **Getting Started**: setup.md, how-to-use.md
- **Architecture & Internals**: per_query_weights_prd.md, RAG_EVALUATION_FRAMEWORK_SPEC.md
- **Historical**: RERANKER_DEBUGGING_POSTMORTEM.md

---

## Verification Results

```
$ cargo check
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 5.18s

$ cargo fmt --check
(no output - formatting OK)

$ cargo test
test result: ok. 4 passed; 0 failed
```

All tests pass. No code changes required.

---

## Follow-ups (Not Blocking)

1. **TUI Environment Variables**: The TUI binary (`rag_tui`) has additional env vars (`RAG_TUI_*`, `RAG_*_WEIGHT`) that are not documented anywhere. Consider adding a TUI-specific docs section or separate TUI.md.

2. **EMBEDDING_BATCH_SIZE / EMBEDDING_BATCH_COOLDOWN_MS**: These advanced tuning vars exist in code but are not documented. Low priority - internal optimization knobs.

3. **eval/reports/**: Consider whether evaluation reports should be in .gitignore or preserved. Currently preserved as historical baselines.

4. **docs/tracking/**: Unaudited directory (appears to be internal tracking, no .md files inside). May warrant cleanup in future.

---

## Files Modified

```
M CLAUDE.md                              # Fixed reranker section, added env vars
M README.md                              # Improved docs index
M docs/RERANKER_DEBUGGING_POSTMORTEM.md  # Added historical disclaimer
M docs/how-to-use.md                     # Added missing env vars
M docs/setup.md                          # Added missing env vars
A docs_hygiene_report.md                 # This report
```

---

**Report Complete**
