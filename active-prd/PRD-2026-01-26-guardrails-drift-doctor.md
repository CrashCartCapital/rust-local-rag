---
project: rust-local-rag
date: 2026-01-26
owner: ryanpappal
commit_observed: d2fee7f
status: draft
doc_type: prd
theme: guardrails + drift reduction + self-diagnosis
---

# PRD: Guardrails, Drift Reduction, and “Doctor” Workflow

## Requirements

### R0 — Definitions
- **Loopback URL**: `localhost`, `127.0.0.1`, or `::1` (or an equivalent loopback IP).
- **Non-loopback bind**: binding to any interface/address that can be reached off-machine (e.g., `0.0.0.0`, LAN IP).
- **Explicit override**: a clearly named env var set by the user acknowledging the risk.

### R1 — Privacy Guardrail: `OLLAMA_URL`
If `OLLAMA_URL` is non-loopback, the server must not silently send text to it.

Acceptance criteria:
- Default behavior: **startup fails fast** (clear error) if `OLLAMA_URL` is non-loopback.
- Override: startup proceeds only if `RAG_ALLOW_REMOTE_OLLAMA=1` (name can vary; must be explicit).
- Error message includes the resolved URL and explicitly states that **document/query text will be sent**.

### R2 — Network Exposure Guardrail: `MCP_HTTP_BIND`
If `MCP_HTTP_BIND` is non-loopback, the server must not silently expose search over the network.

Acceptance criteria:
- Default behavior: **startup fails fast** if bind is non-loopback.
- Override: startup proceeds only if `RAG_ALLOW_REMOTE_BIND=1`.
- Error message includes the bind address and states that there is **no auth**.

### R3 — Tests Must Exercise Real Config Paths
Acceptance criteria:
- All tests that intend to set embedding batch size use `RAG_EMBEDDING_BATCH_SIZE` (not `EMBEDDING_BATCH_SIZE`).
- Add at least one integration test that **proves** the batch-size config changes observable Ollama embed request behavior (single vs batch request shape, or request counts).

### R4 — Docs Must Match Runtime Behavior
Acceptance criteria:
- `docs/configuration.md` and `docs/architecture.md` match Streamable HTTP MCP server behavior (`MCP_HTTP_BIND`, `MCP_HTTP_ENDPOINT`).
- Remove or clearly mark `LOG_MAX_MB` as unsupported (preferred: remove until implemented).
- Knob list is authoritative and matches `src/config.rs` + runtime env usage.

### R5 — One Source of Truth for Search Parameter Validation
Acceptance criteria:
- `top_k` clamping and `diversity_factor` clamping are implemented in **one shared function/module** used by both MCP (`src/mcp/tools.rs`) and HTTP (`src/mcp/http.rs`).
- HTTP `/search` supports `weights` consistently with MCP `search_documents` (either both support, or both reject; preferred: both support).
- Add tests verifying MCP and HTTP produce identical clamping for the same inputs.

### R6 — No Fragile Env Parsing Scripts
Acceptance criteria:
- `start-server.sh` is removed, or rewritten to safely load `.env` without `grep|xargs` export parsing.
- Script behavior documented (what it loads, where it expects `.env`, and how it handles spaces).

### R7 — “Doctor” Workflow (Self-Diagnosis)
Acceptance criteria:
- A fast command exists (standalone binary preferred) that runs without starting the full server.
- Outputs a checklist of PASS/FAIL with remediation steps for:
  - `DATA_DIR` writable + expected DB openable
  - `DOCUMENTS_DIR` readable
  - `pdftotext` availability (optional but recommended)
  - `OLLAMA_URL` reachable (`/api/tags`)
  - embedding model present (from `OLLAMA_EMBEDDING_MODEL`)
  - reranker model present (from `OLLAMA_RERANK_MODEL`) if configured
- Exit code: `0` when all checks pass, `>0` otherwise.

---

## Design

### D1 — Guardrails should be pure + testable
Add a small module (proposed: `src/guardrails.rs`) with pure helpers:
- `fn is_loopback_url(url: &str) -> Result<bool, GuardrailError>`
- `fn is_loopback_bind(addr: std::net::SocketAddr) -> bool`
- `fn require_loopback_or_override(kind: GuardrailKind, value: &str) -> Result<(), GuardrailError>`

Integration points (minimal surgery):
- `src/embeddings.rs`: validate `OLLAMA_URL` before any network call.
- `src/main.rs`: validate `MCP_HTTP_BIND` immediately after parsing.

### D2 — Shared search validation should live in the adapter layer (not rag-core)
Because the goal is to avoid cross-surface drift, keep the shared validator near the surfaces:
- Proposed: `src/mcp/validation.rs` with `ValidatedSearch` and `fn validate_search(top_k, diversity, weights)`.
- MCP tool handler and HTTP handler should both call into it.

### D3 — Prefer doc alignment over new log-rotation complexity
`LOG_MAX_MB` is documented but not implemented. Implementing size-based truncation adds code paths and risk.

Default choice:
- Remove `LOG_MAX_MB` from docs until there is a concrete, tested implementation.

### D4 — “Doctor” should be a separate binary
Reason: the doctor needs to be runnable even when guardrails prevent the server from starting.

Proposed:
- Add `src/bin/rag_doctor.rs` (or `src/bin/doctor.rs`).

---

## Tasks

### Section: P0 — Stop the bleeding (trust + safety)
- [x] T0.1 — Add `OLLAMA_URL` guardrail
  - Files: `src/embeddings.rs`, new `src/guardrails.rs`
  - Tests:
    - Unit: remote URL rejected without override.
    - Unit: loopback URL accepted.
    - Unit: remote URL accepted with override.
  - DoD: Meets R1.

- [x] T0.2 — Add `MCP_HTTP_BIND` guardrail
  - Files: `src/main.rs`, new `src/guardrails.rs`
  - Tests:
    - Unit: loopback binds accepted; `0.0.0.0` rejected unless override.
  - DoD: Meets R2.

- [x] T0.3 — Fix “tests that lie” + prove batch-size is honored
  - Files: `tests/rag_integration.rs`, `tests/worker_integration.rs`, any other tests setting `EMBEDDING_BATCH_SIZE`
  - Tests:
    - Update env var name to `RAG_EMBEDDING_BATCH_SIZE`.
    - Add/extend one integration test to assert batch request shape (array `input`) occurs when batch size > 1.
  - DoD: Meets R3.

- [ ] T0.4 — Docs/code alignment pass
  - Files: `docs/configuration.md`, `docs/architecture.md`, `README.md` (if necessary)
  - Changes:
    - Remove/clarify `LOG_MAX_MB`.
    - Ensure transport description matches Streamable HTTP MCP server.
    - Ensure env var list is correct and consistent with `src/config.rs` and runtime env reads.
  - DoD: Meets R4.

### Section: P1 — Reduce drift + remove footguns
- [ ] T1.1 — Unify search validation + add HTTP `weights`
  - Files: `src/mcp/http.rs`, `src/mcp/tools.rs`, `src/mcp/models.rs`, new `src/mcp/validation.rs`
  - Tests:
    - Unit tests for validator clamp behavior.
    - Minimal integration test verifying HTTP and MCP clamp equivalently (table-driven).
  - DoD: Meets R5.

- [ ] T1.2 — Replace/remove `start-server.sh`
  - Files: `start-server.sh` (rewrite) or delete; docs update as needed.
  - DoD: Meets R6.

- [ ] T1.3 — (Optional) Skip symlinks by default during ingestion
  - Motivation: avoid accidental indexing of outside-tree files via symlinks in `DOCUMENTS_DIR`.
  - Files: `src/rag_engine.rs` (doc discovery), docs
  - DoD: Warn and skip symlinked files unless `RAG_ALLOW_SYMLINKS=1`.

### Section: P2 — Compounding ROI
- [ ] T2.1 — Add `rag-doctor` command
  - Files: new `src/bin/rag_doctor.rs`, shared helpers in `src/guardrails.rs` or `src/doctor/`
  - Tests:
    - Unit tests for the pure checks.
    - (Optional) integration test using wiremock for `/api/tags`.
  - DoD: Meets R7.

- [ ] T2.2 — Repo hygiene
  - Files: `.gitignore` (or relevant tooling ignore)
  - DoD: ignore `frontend/node_modules`, `frontend/dist` (and any other bulky derived outputs).

- [ ] T2.3 — Golden eval fixtures
  - Files: `eval/` harness + new fixtures in `eval/fixtures/` (or similar)
  - DoD:
    - 2–3 stable queries with expected doc IDs/sections.
    - One baseline run command documented.

---

## Testing Strategy

1) Put guardrail logic behind pure functions → easy unit tests and clear error messages.
2) Use existing wiremock-based integration tests to prove config wiring (batch request shape).
3) Add table-driven tests for shared validation (MCP + HTTP).

---

## Addenda

### Executive Summary

This PRD turns the two 2026-01-26 status reports into a **surgical, high-ROI** work plan that strengthens the repo’s core promise:

> “Search and analyze PDF documents … without sending data to external services.” (README)

The selected scope focuses on **silent-footgun prevention**, **test trust**, and **interface drift reduction**, while explicitly avoiding heavy refactors (e.g., RagEngine decomposition) unless they become blocking.

#### Outcomes (what “done” looks like)
1) The server **refuses or loudly warns** before it can leak document/query text due to `OLLAMA_URL` or `MCP_HTTP_BIND` misconfiguration.
2) Integration tests stop “lying” (they exercise the real config/env var paths used in production).
3) MCP vs HTTP behavior stops drifting for shared search parameters (top_k, diversity, weights).
4) A fast “doctor” command exists to regain momentum in minutes.

---

### Inputs (Source Reports + Code Anchors)

#### Reports (2026-01-26)
- `STAT-REPORT-rust-local-rag-2026-01-26-112518.md` (commit `d2fee7f`)
- `STAT-REPORT-rust-local-rag-2026-01-26-143000.md` (commit `d2fee7f`)

#### Code anchors (current repo observations)
- `src/embeddings.rs`: POSTs raw text `input` to `{OLLAMA_URL}/api/embed` (privacy boundary).
- `src/main.rs`: binds server to `MCP_HTTP_BIND` default `127.0.0.1:8140`.
- `src/config.rs`: reads `RAG_EMBEDDING_BATCH_SIZE` (tests currently use `EMBEDDING_BATCH_SIZE`).
- `src/mcp/tools.rs` vs `src/mcp/http.rs`: both clamp top_k/diversity, but HTTP lacks `weights` support.
- `docs/configuration.md`: documents `LOG_MAX_MB`, but `src/` has no implementation.
- `start-server.sh`: uses `export $(grep -v '^#' .env | xargs)` (fragile parsing).

---

### Screening: Relevance × Pragmatism × Robustness (Analytics)

#### Rubric
Scores are 1–5 (higher is better) unless noted.

- **Relevance**: improves the repo’s primary value (local-first RAG via MCP).
- **Robustness Impact**: reduces severity/likelihood of “it still works but it’s unsafe/wrong”.
- **Effort**: estimated engineering hours (lower is better).
- **Maintenance Burden**: likelihood the change becomes ongoing overhead (lower is better).

Derived:
- **ROI** ≈ (Relevance + Robustness Impact) / Effort, adjusted down for higher burden.

#### Candidate backlog scoring (screened)
| Candidate | Relevance | Robustness | Effort (h) | Burden | ROI | Decision |
|---|---:|---:|---:|---:|---:|---|
| Guardrails for `OLLAMA_URL` + `MCP_HTTP_BIND` | 5 | 5 | 3 | 1 | 3.3 | **P0** |
| Fix “tests that lie” env mismatch | 4 | 3 | 1 | 1 | 7.0 | **P0** |
| Docs/code drift (transport + `LOG_MAX_MB` + knob list) | 3 | 3 | 1 | 1 | 6.0 | **P0** |
| Shared validation/service layer for MCP + HTTP (incl. `weights`) | 4 | 3 | 3 | 2 | 2.3 | **P1** |
| Replace/remove `start-server.sh` env parsing footgun | 3 | 3 | 2 | 1 | 3.0 | **P1** |
| “doctor” / smoke-check command | 4 | 2 | 4 | 2 | 1.5 | **P2** |
| Ignore bulky derived dirs (`frontend/node_modules`, `frontend/dist`) | 2 | 1 | 1 | 1 | 3.0 | **P2 (opportunistic)** |
| Golden eval fixtures / baseline | 4 | 3 | 2–4 | 2 | 1.8 | **P2** |
| “Two DB distributed transaction gap” | ? | ? | ? | ? | ? | **Reconcile first** |

#### Reconciliation note: “two DB” risk
One report claims separate `jobs.db` and `index.db`. Current code appears to point **both job management and index store** at `sqlite:{DATA_DIR}/jobs.db` (e.g., `src/main.rs`, `src/rag_engine.rs`).

Decision:
- Treat “unify DBs” as **NOT** a current requirement.
- Replace it with a smaller, higher-signal task: **prove** crash-recovery correctness with a targeted test and/or document a recovery procedure (“rerun reindex”).

---

### Rollout / Compatibility

- Guardrails are potentially breaking for users who intentionally point to remote Ollama or bind to LAN.
- Provide explicit opt-outs (`RAG_ALLOW_REMOTE_OLLAMA`, `RAG_ALLOW_REMOTE_BIND`) and document them.
- Prefer failing fast with crisp guidance over “warn and continue” because the failure modes are silent leaks.

---

### Out of Scope (Backlog, not in this PRD)

These were mentioned in one report but are intentionally deferred unless they become blocking:
- Decompose `RagEngine` into services (large refactor, not required for current goals).
- Extraction quality gates (entropy/language metrics) beyond basic validation.
- Prometheus metrics endpoint.
- Reranker prompt-injection hardening (worth revisiting later, but keep current iteration focused).
