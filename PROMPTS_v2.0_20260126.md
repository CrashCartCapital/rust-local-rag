# PROMPTS v2.0 — rust-local-rag

**Version**: 2.0
**Date**: 2026-01-26
**Status**: PROPOSED (pending review)

## Executive Summary: What Changed

### The Problem (v1.0)
- 53 PRs generated in 4 days
- 51 closed as duplicates/low-value (96% rejection rate)
- Only 2 PRs kept: Atomic Upsert (#237), Reranker Sort Stability (#232)

### Root Causes Identified
1. **Parallel Stateless Scanning**: 10 daily prompts with no shared memory
2. **Convergent Behavior**: All agents find the same "obvious" targets
3. **Semantic Redundancy**: 4 prompts for unwrap removal, 6 for chunking
4. **No Deduplication**: No check for existing PRs or recent work
5. **Pattern-Chasing**: "Find something to fix" bias toward easy/low-value changes
6. **No Evidence Gate**: PRs created without proving a real problem exists

### The Fix (v2.0)
1. **Consolidate**: 16 prompts → 5 prompts
2. **Evidence-First**: Failing test required before implementation
3. **Dedup Checks**: Mandatory pre-flight against git history and open PRs
4. **Focus on Value**: Integrity and determinism only; style changes banned
5. **Volume Limits**: Max 2 PRs/week until acceptance rate improves
6. **Backlog System**: Candidates tracked centrally, not re-discovered each run

---

## Architecture Overview

```
DISCOVERY (weekly)     TRIAGE (weekly)      IMPLEMENTATION (as-needed)
      │                      │                       │
      ▼                      ▼                       │
┌─────────────┐        ┌──────────┐                  │
│ Scan for    │───────▶│ Rank by  │                  │
│ candidates  │        │ impact   │                  │
│ (no PRs)    │        │ + dedup  │                  │
└─────────────┘        └────┬─────┘                  │
                            │                        │
                            ▼                        │
                    ┌───────────────┐                │
                    │ Pick top 2-3  │───────────────▶│
                    │ with evidence │                │
                    └───────────────┘                │
                                                     ▼
                                              ┌─────────────┐
                                              │ Implement   │
                                              │ + test      │
                                              │ + PR        │
                                              └─────────────┘
```

---

## Mandatory Pre-Flight Check (ALL PROMPTS)

Every prompt MUST begin with this check. If ANY condition is met, ABORT.

```text
PRE-FLIGHT CHECK (MANDATORY — RUN BEFORE ANY WORK)

1) Check recent commits:
   git log --oneline -20

2) Check open PRs:
   gh pr list --limit 20

3) Check backlog (if exists):
   cat analysis/JULES_BACKLOG.md 2>/dev/null || echo "No backlog"

ABORT CONDITIONS:
- If your intended target appears in recent commits → ABORT (already done)
- If your intended target has an open PR → ABORT (in progress)
- If your intended target is in backlog with status=rejected → ABORT (already declined)
- If your intended target matches a BANNED CATEGORY → ABORT (low value)

BANNED CATEGORIES (DO NOT CREATE PRs FOR):
- Clippy warnings (unless blocking CI)
- Documentation updates (unless factually wrong AND high-traffic doc)
- Tracing/instrumentation additions (unless debugging active issue)
- Unwrap/expect removal (unless proven panic path in production)
- Env parsing hardening (unless proven misconfiguration incident)
- Error message improvements (unless user-reported confusion)
- Test-only changes (unless fixing flaky test)
```

---

## Prompt Library

### jules — weekly

#### W01: Weekly Integrity & Determinism Discovery
- **Title**: Weekly discovery scan for integrity and determinism risks (candidates only, NO PRs)
- **Tags**: discovery, integrity, determinism, weekly
- **Risk**: low (no code changes)
- **Owner**: ryan
- **DoD**:
  - Outputs candidates to `analysis/JULES_BACKLOG.md`
  - Does NOT create PRs or change code
  - Each candidate has: fingerprint, evidence sketch, impact score

```text
W01-IntegrityDeterminismDiscovery

[STRICT MODE — Unattended Cloud Run]

Cloud VM. No LAN. No secrets. No Ollama. Follow `AGENTS.md`.

PURPOSE: Discovery ONLY. You will NOT implement anything or create PRs.
OUTPUT: Append candidates to `analysis/JULES_BACKLOG.md` (create if missing).

FOCUS AREAS (these are the ONLY categories that produced valuable PRs):
1. Data Integrity Risks:
   - Non-atomic operations that could corrupt state
   - Partial writes / torn reads
   - Race conditions in shared state
   - Missing validation before mutation

2. Determinism Risks:
   - Unstable sorting (HashMap iteration, floating-point ties)
   - Time-dependent ordering
   - Random seeds not controlled
   - Non-reproducible test results

DO NOT SCAN FOR:
- Clippy warnings
- Unwrap/expect patterns (unless proven crash)
- Documentation staleness
- Tracing gaps
- Error message wording

DISCOVERY PROCESS:
1) Run pre-flight check (see above). If abort condition met, stop.
2) Scan core modules for integrity/determinism risks:
   - `crates/rag-core/src/engine.rs` (state mutations)
   - `src/reranker.rs` (sorting/ordering)
   - `src/worker.rs` (concurrent operations)
   - `src/job_manager.rs` (state machine)
3) For each candidate, record:
   - Fingerprint: `{module}:{risk_type}:{short_description}`
   - Evidence sketch: How would you prove this is a real problem?
   - Impact: 1-5 (5 = data loss/corruption, 1 = cosmetic)
   - Confidence: 1-5 (5 = proven, 1 = speculative)

OUTPUT FORMAT (append to analysis/JULES_BACKLOG.md):
```markdown
## Candidate: {fingerprint}
- **Discovered**: {date}
- **Status**: pending_triage
- **Module**: {file path}
- **Risk Type**: integrity | determinism
- **Impact**: {1-5}
- **Confidence**: {1-5}
- **Evidence Sketch**: {How to prove this is real}
- **Similar Recent Work**: {git log matches or "none"}
```

STOP CONDITIONS:
- If you find 5+ candidates, stop and output what you have
- If all high-impact areas already have recent PRs/commits, output "No new candidates"
```

---

#### W02: Weekly Triage & Prioritization
- **Title**: Weekly triage of discovery backlog (rank, dedup, select top candidates)
- **Tags**: triage, prioritization, weekly
- **Risk**: low (no code changes)
- **Owner**: ryan
- **DoD**:
  - Reviews `analysis/JULES_BACKLOG.md`
  - Deduplicates against git history and open PRs
  - Ranks remaining candidates
  - Marks top 2-3 as `ready_for_implementation`

```text
W02-WeeklyTriage

[STRICT MODE — Unattended Cloud Run]

Cloud VM. No LAN. No secrets. Follow `AGENTS.md`.

PURPOSE: Triage and rank candidates. You will NOT implement anything.
INPUT: `analysis/JULES_BACKLOG.md`
OUTPUT: Updated backlog with status changes and rankings.

TRIAGE PROCESS:
1) Run pre-flight check. Load backlog.
2) For each candidate with status=pending_triage:

   DEDUP CHECK:
   - Search git log for fingerprint keywords
   - Search open PRs for similar work
   - If duplicate found → status=duplicate, add link

   EVIDENCE CHECK:
   - Can this be proven with a failing test?
   - Is there a concrete repro scenario?
   - If speculative/theoretical → status=deferred, reason="no evidence"

   IMPACT SCORING (must meet threshold):
   - integrity_risk: 0-5
   - determinism_risk: 0-5
   - user_impact: 0-5
   - implementation_risk: 0-5 (lower is better)
   - TOTAL must be >= 12 AND (integrity >= 4 OR determinism >= 4)

3) Rank passing candidates by total score descending.
4) Mark top 2-3 as `status=ready_for_implementation`.
5) All others: `status=backlog` (revisit next week).

OUTPUT: Update each candidate's status and scores in backlog file.

HARD LIMITS:
- Max 3 candidates promoted to ready_for_implementation per week
- If no candidates meet threshold, output "No candidates meet quality bar"
```

---

#### W03: Implementation (Evidence-Gated)
- **Title**: Implement ONE triaged candidate with evidence (failing test first)
- **Tags**: implementation, evidence, weekly
- **Risk**: medium
- **Owner**: ryan
- **DoD**:
  - Selects ONE `ready_for_implementation` candidate from backlog
  - Writes failing test FIRST (evidence gate)
  - Implements minimal fix
  - All verification passes
  - Updates backlog status

```text
W03-EvidenceGatedImplementation

[STRICT MODE — Unattended Cloud Run]

Cloud VM. No LAN. No secrets. No Ollama. Follow `AGENTS.md`.

PURPOSE: Implement exactly ONE candidate that has been triaged and approved.
INPUT: `analysis/JULES_BACKLOG.md` candidates with status=ready_for_implementation

HARD CONSTRAINTS:
- Implement exactly ONE candidate (not more)
- Must write failing test BEFORE implementing fix (evidence gate)
- Max 6 files changed
- If test doesn't fail → candidate was speculative → mark as rejected, do not create PR

PROCESS:
1) Run pre-flight check.
2) Load backlog, select highest-ranked ready_for_implementation candidate.
3) Acquire "lease" by updating status to `in_progress` with timestamp.

EVIDENCE GATE (MANDATORY):
4) Write a test that DEMONSTRATES the problem:
   - For integrity: test that shows corruption/inconsistency without fix
   - For determinism: test that shows non-deterministic ordering without fix
5) Run the test. It MUST FAIL.
   - If test passes → problem doesn't exist → status=rejected, reason="test passed without fix", STOP
   - If test fails → proceed to implementation

IMPLEMENTATION:
6) Implement the minimal fix to make the test pass.
7) Run full verification:
   - `cargo fmt -- --check`
   - `cargo clippy --all-targets --all-features -- -D warnings`
   - `cargo test`
8) Update backlog: status=pr_created, pr_number={number}

PR REQUIREMENTS:
- Title must include backlog fingerprint
- Body must include:
  - Link to backlog entry
  - The failing test (evidence)
  - Explanation of fix
  - Verification commands run
```

---

### jules — monthly (demoted from daily/weekly)

#### M01: Monthly Quality Gate
- **Title**: Monthly Rust quality gate (fmt + clippy + test) — verification only
- **Tags**: hygiene, monthly, rust
- **Risk**: low
- **Owner**: ryan
- **DoD**:
  - Runs verification commands
  - Reports status
  - Only creates PR if CI would fail (blocking issue)

```text
M01-MonthlyQualityGate

[STRICT MODE — Unattended Cloud Run]

Cloud VM. No LAN. No secrets. Follow `AGENTS.md`.

PURPOSE: Verify build health. Only fix BLOCKING issues.

VERIFICATION:
1) `cargo fmt -- --check`
2) `cargo clippy --all-targets --all-features -- -D warnings`
3) `cargo test`

DECISION TREE:
- All pass → Output "Quality gate GREEN" → NO PR
- fmt fails → Run `cargo fmt`, create PR "fix: auto-format"
- clippy fails with errors (not warnings) → Fix errors only, create PR
- test fails → Investigate, fix if obvious, else report

DO NOT CREATE PR FOR:
- Clippy warnings that don't block CI
- Style preferences
- "Improvements" beyond fixing failures
```

---

#### M02: Monthly Docs Reconciliation
- **Title**: Monthly docs accuracy check (verify claims, fix if wrong)
- **Tags**: docs, monthly
- **Risk**: low
- **Owner**: ryan
- **DoD**:
  - Verifies factual claims in README.md
  - Only fixes provably wrong information
  - No style changes, no "improvements"

```text
M02-MonthlyDocsReconciliation

[STRICT MODE — Unattended Cloud Run]

Cloud VM. No LAN. No secrets. Follow `AGENTS.md`.

PURPOSE: Fix factually incorrect documentation. Not style, not completeness.

SCOPE: README.md only (highest traffic doc)

PROCESS:
1) Run pre-flight check.
2) Extract factual claims from README.md:
   - Command examples
   - File paths mentioned
   - Configuration options
3) Verify each claim:
   - Run commands (read-only)
   - Check file paths exist
   - Verify config options work
4) If claim is PROVABLY WRONG:
   - Fix it
   - Document what was wrong and how verified

DO NOT:
- Rewrite for style
- Add new sections
- Expand documentation
- Fix "stale but not wrong" content
```

---

## Removed Prompts (Eliminated as Low-Value)

The following v1.0 prompts are **ELIMINATED** in v2.0:

| v1.0 ID | Title | Reason for Elimination |
|---------|-------|------------------------|
| P0402 | Cargo Quality Gate | Demoted to monthly; daily was overkill |
| P0403 | Chunking Edge Case Tests | 6 duplicate PRs; theoretical edge cases |
| P0404 | Env Parsing Hardening | No real misconfig incidents; theoretical |
| P0406 | Error Envelope One | 5 duplicate PRs; HTTP status bikeshedding |
| P0408 | Error Tracing Instrumentation | 5 duplicate PRs; premature instrumentation |
| P0409 | Corruption Recovery Test | 5 duplicate PRs; testing theoretical failures |
| P0410 | Unwrap/Expect Removal | 5 duplicate PRs; pattern-chasing without bug proof |
| P0411 | TODO/FIXME Snapshot | Meta-docs don't fix code |
| P0412 | TODO Burndown One | Overlaps with other prompts |
| P0413 | Impact Drill | Too broad; caused overlap with everything |
| P0414 | Cargo Update Patch | Created "Aborted" PR; needs human judgment |
| P0415 | Grand Rounds Refactor | Overlaps with P0410, P0416 |
| P0416 | Unwrap Clone Hardening | Overlaps with P0410, P0415 |

### Prompts Consolidated

| v1.0 Prompts | v2.0 Prompt | Rationale |
|--------------|-------------|-----------|
| P0405 (Rerank Tie Breaker) | W01 (Discovery) | Determinism scanning consolidated |
| P0407 (State Machine Invariant) | W01 (Discovery) | Integrity scanning consolidated |
| P0401 (Doc Alignment) | M02 (Monthly Docs) | Demoted to monthly, scoped to factual errors |

---

## Backlog File Format

Create `analysis/JULES_BACKLOG.md` with this structure:

```markdown
# Jules Candidate Backlog

Last updated: {timestamp}

## Ready for Implementation

### {fingerprint}
- **Status**: ready_for_implementation
- **Discovered**: {date}
- **Module**: {path}
- **Risk Type**: integrity | determinism
- **Impact**: {score}
- **Evidence Sketch**: {description}
- **Scores**: integrity={n}, determinism={n}, user_impact={n}, impl_risk={n}, TOTAL={n}

## In Progress

### {fingerprint}
- **Status**: in_progress
- **Lease Acquired**: {timestamp}
- **PR**: (pending)

## Completed

### {fingerprint}
- **Status**: merged | rejected
- **PR**: #{number}
- **Outcome**: {description}

## Deferred

### {fingerprint}
- **Status**: deferred
- **Reason**: {no evidence | low impact | duplicate of X}

## Rejected (Cooldown 30 days)

### {fingerprint}
- **Status**: rejected
- **Date**: {date}
- **Reason**: {test passed without fix | duplicate | banned category}
- **Cooldown Until**: {date + 30 days}
```

---

## Success Metrics

### Target State
- **Acceptance Rate**: > 80% (vs 4% in v1.0)
- **PRs per Week**: 1-2 high-quality (vs 13+ low-quality in v1.0)
- **Duplicate Rate**: < 10% (vs 90%+ in v1.0)
- **Evidence Gate Pass Rate**: 100% (all PRs have failing test first)

### Monitoring
After 4 weeks, review:
1. How many PRs were merged vs closed?
2. Did any duplicates slip through?
3. Were any valuable issues missed by the narrower scope?

---

## Migration Plan

1. **Week 1**: Run W01 (Discovery) only. Populate backlog.
2. **Week 2**: Run W02 (Triage). Validate ranking works.
3. **Week 3**: Run W03 (Implementation) on top candidate.
4. **Week 4**: Review metrics, adjust thresholds.

Do NOT run v1.0 prompts after this date. They are deprecated.
