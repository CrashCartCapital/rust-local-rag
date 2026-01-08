# Dependency Hygiene Report

## Executive Summary
**Date:** 2025-01-08 (Simulated)
**Tools:** `cargo audit`, `cargo update --dry-run`
**Status:** Vulnerabilities Found, Stale Dependencies Detected.

## 1. Security Vulnerabilities
`cargo audit` identified the following vulnerabilities and advisories.

### Critical/Actionable
*   **Crate:** `tracing-subscriber`
    *   **Current Version:** 0.3.19
    *   **Advisory:** [RUSTSEC-2025-0055](https://rustsec.org/advisories/RUSTSEC-2025-0055)
    *   **Description:** Logging user input may result in poisoning logs with ANSI escape sequences.
    *   **Remediation:** Upgrade to `>=0.3.20`.
    *   **Status:** Fix available via `cargo update` (updates to 0.3.22).

### Warnings / Unmaintained
*   **Crate:** `lru`
    *   **Current Version:** 0.12.5
    *   **Advisory:** [RUSTSEC-2026-0002](https://rustsec.org/advisories/RUSTSEC-2026-0002)
    *   **Description:** `IterMut` violates Stacked Borrows.
    *   **Status:** Unsoundness. Upgrade available to v0.16.3 (major version change).
*   **Crate:** `dotenv`
    *   **Current Version:** 0.15.0
    *   **Advisory:** [RUSTSEC-2021-0141](https://rustsec.org/advisories/RUSTSEC-2021-0141)
    *   **Status:** Unmaintained. Recommend replacing with `dotenvy`.
*   **Crate:** `paste`
    *   **Current Version:** 1.0.15
    *   **Advisory:** [RUSTSEC-2024-0436](https://rustsec.org/advisories/RUSTSEC-2024-0436)
    *   **Status:** Unmaintained.
*   **Crate:** `flate2`
    *   **Current Version:** 1.1.7
    *   **Status:** Yanked. Downgrade to 1.1.5 or upgrade if available.
*   **Crate:** `rsa`
    *   **Current Version:** 0.9.9
    *   **Advisory:** [RUSTSEC-2023-0071](https://rustsec.org/advisories/RUSTSEC-2023-0071)
    *   **Status:** No fixed upgrade available. Dependency of `sqlx-mysql` (transitive via `sqlx`).

## 2. Stale Dependencies
The following dependencies have newer major versions available (or 0.x minor bumps which are breaking).

*   **reqwest**: `v0.12.28` -> `v0.13.1`
*   **lopdf**: `v0.34.0` -> `v0.38.0`
*   **lru**: `v0.12.5` -> `v0.16.3`
*   **ratatui**: `v0.29.0` -> `v0.30.0`
*   **rmcp**: `v0.8.5` -> `v0.12.0`
*   **rmcp-macros**: `v0.8.5` -> `v0.12.0`
*   **crossterm**: `v0.28.1` -> `v0.29.0`

## 3. Issue Drafts

### HIGH PRIORITY: Fix Security Vulnerabilities in Dependencies

**Description**
A recent security audit identified vulnerabilities in our dependency tree. Immediate action is required to patch these issues.

**Vulnerabilities:**
1.  **tracing-subscriber (RUSTSEC-2025-0055)**
    *   *Issue:* Log poisoning via ANSI escape sequences.
    *   *Fix:* Update crate to `>=0.3.20`.
    *   *Action:* Run `cargo update -p tracing-subscriber`.

2.  **lru (RUSTSEC-2026-0002)**
    *   *Issue:* Unsoundness in `IterMut`.
    *   *Fix:* Upgrade to latest version (breaking change).
    *   *Action:* Upgrade `lru` in `Cargo.toml`.

3.  **dotenv (RUSTSEC-2021-0141)**
    *   *Issue:* Unmaintained.
    *   *Action:* Migrate to `dotenvy`.

**Acceptance Criteria**
*   [ ] `cargo audit` passes (or ignores known unfixable issues like `rsa`).
*   [ ] `tracing-subscriber` is updated.
*   [ ] `dotenv` is replaced with `dotenvy`.

**Notes**
*   `rsa` vulnerability comes from `sqlx-mysql`. Since we are likely using `sqlite` (implied by `rust-local-rag`), we should check if `sqlx-mysql` can be removed from features.

