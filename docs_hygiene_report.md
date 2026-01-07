# Dependency Hygiene Report

**Date:** 2026-01-07
**Scope:** `rust-local-rag` crate and workspace.

## Summary

*   **Vulnerabilities Found:** 2 (1 Fixable, 1 Unfixable/Mitigated)
*   **Unmaintained/Yanked Crates:** 3
*   **Major Updates Available:** 1 (`reqwest`)

---

## 1. Security Vulnerabilities (`cargo audit`)

### High Priority
**Crate:** `rsa` (v0.9.9)
*   **Advisory ID:** [RUSTSEC-2023-0071](https://rustsec.org/advisories/RUSTSEC-2023-0071)
*   **Title:** Marvin Attack: potential key recovery through timing sidechannels
*   **Severity:** Medium (5.9)
*   **Status:** **No fixed upgrade available.**
*   **Dependency Chain:** `rust-local-rag -> sqlx -> ... -> rsa`
    *   Note: `sqlx` pulls in `rsa` (likely via `sqlx-mysql` -> `rsa` or similar TLS negotiation paths). Even though `rust-local-rag` only requests `sqlite` feature, `sqlx` (v0.8) appears to pull in `sqlx-macros` which may check all drivers or have broader dependencies in the lockfile.
*   **Action Required:** Monitor for `rsa` updates or `sqlx` updates that drop this dependency. See drafted Issue below.

### Fixable via Update
**Crate:** `tracing-subscriber` (v0.3.19)
*   **Advisory ID:** [RUSTSEC-2025-0055](https://rustsec.org/advisories/RUSTSEC-2025-0055)
*   **Title:** Logging user input may result in poisoning logs with ANSI escape sequences
*   **Solution:** Upgrade to >=0.3.20.
*   **Status:** `cargo update` will upgrade this to **v0.3.22**, resolving the issue.

---

## 2. Unmaintained / Yanked Dependencies

*   **`dotenv` (v0.15.0)**: Unmaintained (RUSTSEC-2021-0141).
    *   **Action:** Migrate to [`dotenvy`](https://crates.io/crates/dotenvy).
*   **`paste` (v1.0.15)**: Unmaintained (RUSTSEC-2024-0436).
*   **`flate2` (v1.1.7)**: Yanked.
    *   **Status:** `cargo update` automatically downgrades to v1.1.5 (safe version).

---

## 3. Stale Dependencies (`cargo outdated` / `cargo update`)

### Major Version Updates Available
*   **`reqwest`**: v0.12.19 -> v0.13.1
    *   Breaking changes likely. Review `reqwest` changelog before upgrading.

### Minor/Patch Updates (Applied by `cargo update`)
A `cargo update` will bring in significant bug fixes and minor features:
*   `rsa`: v0.9.9 -> v0.9.10
*   `tracing-subscriber`: v0.3.19 -> v0.3.22 (Fixes vulnerability)
*   `axum`: v0.8.7 -> v0.8.8
*   `tokio`: v1.48.0 -> v1.49.0
*   `flate2`: v1.1.7 -> v1.1.5 (Downgrade from yanked)

---

## Appendix: Issue Drafts

### Issue: Unresolved Vulnerability in `rsa` (via `sqlx`)

**Title:** Security: Unresolved 'Marvin Attack' vulnerability in `rsa` dependency

**Description:**
`cargo audit` reports a vulnerability in the `rsa` crate (v0.9.9) used transitively via `sqlx`.

*   **Advisory:** [RUSTSEC-2023-0071](https://rustsec.org/advisories/RUSTSEC-2023-0071)
*   **Impact:** Potential key recovery through timing sidechannels.
*   **Current State:** No fixed version of `rsa` is currently available.
*   **Dependency Path:** `rust-local-rag` -> `sqlx` -> ... -> `rsa`.

**Action Items:**
1.  Investigate if `sqlx` usage in `rust-local-rag` (SQLite only) actually exercises the vulnerable code paths in `rsa` (likely used for TLS in MySQL/Postgres drivers).
2.  If not relevant to SQLite, consider using `cargo audit` ignore mechanisms or `[patch]` if possible, but preferably wait for upstream fixes.
3.  Monitor `sqlx` and `rsa` for updates.
