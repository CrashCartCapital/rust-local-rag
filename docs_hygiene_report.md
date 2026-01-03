# Dependency Hygiene Report

## Summary
This report analyzes the project's Rust dependencies for security vulnerabilities and stale versions.

- **Date:** 2025-05-15
- **Tooling:** `cargo audit` (v0.22.0), `cargo update --dry-run` (proxy for `cargo outdated`)

## Security Audit Findings

**Vulnerabilities Found:** 2
**Unmaintained Packages:** 2
**Yanked Packages:** 1

### 1. HIGH PRIORITY: `tracing-subscriber` (RUSTSEC-2025-0055)
- **Crate:** `tracing-subscriber` (v0.3.19)
- **Severity:** N/A (Advisory does not list score, but involves log poisoning)
- **Description:** Logging user input may result in poisoning logs with ANSI escape sequences.
- **Solution:** Upgrade to `>=0.3.20`.
- **Action Required:** Update `Cargo.toml` or `Cargo.lock` to pull the latest version. `cargo update` shows 0.3.22 is available.

### 2. MEDIUM PRIORITY: `rsa` (RUSTSEC-2023-0071)
- **Crate:** `rsa` (v0.9.9)
- **Severity:** 5.9 (Medium)
- **Description:** Marvin Attack: potential key recovery through timing sidechannels.
- **Dependency Chain:** `sqlx` -> `sqlx-macros` -> `...` -> `rsa`.
- **Solution:** No fixed upgrade available.
- **Action Required:** Monitor for `sqlx` updates or `rsa` patches. Since this is a local RAG tool, the risk of network-based timing attacks might be lower, but it should be tracked.

### 3. Unmaintained Dependencies
- **`dotenv` (v0.15.0):** Unmaintained since Dec 2021 (RUSTSEC-2021-0141).
  - **Recommendation:** Replace with `dotenvy`.
- **`paste` (v1.0.15):** Unmaintained since Oct 2024 (RUSTSEC-2024-0436).
  - **Recommendation:** Verify necessity or find alternatives.

### 4. Yanked Dependencies
- **`flate2` (v1.1.7):** Yanked.
  - **Action:** Downgrade to v1.1.5 or wait for v1.1.8+. `cargo update` suggests downgrading to v1.1.5.

## Stale Dependency Analysis

### Major Updates Available
- **`reqwest`**: v0.12.19 -> v0.13.1 (Breaking changes likely).
- **`ratatui`**: v0.29.0 (Check for v0.30+ as it releases frequently).
- **`sqlx`**: v0.8.6 (Check for v0.9+ if available).

### Minor Updates (Safe to Apply)
- `tracing-subscriber` v0.3.19 -> v0.3.22 (Fixes security issue).
- `axum` v0.8.7 -> v0.8.8.
- `tokio` v1.48.0 -> v1.49.0.

---

## Issue Draft (High Priority)

**Title:** Security Upgrade: `tracing-subscriber` and Dependency Cleanup

**Description:**
`cargo audit` revealed a vulnerability in `tracing-subscriber` and several hygiene issues.

**Vulnerability:**
- **ID:** RUSTSEC-2025-0055
- **Crate:** `tracing-subscriber` @ 0.3.19
- **Fix:** Update to >= 0.3.20 (0.3.22 available).

**Maintenance Tasks:**
1. Update `tracing-subscriber` to latest minor version.
2. Replace unmaintained `dotenv` with `dotenvy`.
3. Investigate `rsa` vulnerability (RUSTSEC-2023-0071) pulled in via `sqlx`.
4. Run `cargo update` to resolve yanked `flate2` v1.1.7.

**Acceptance Criteria:**
- `cargo audit` passes (except unavoidable `rsa` issue).
- `tracing-subscriber` version is >= 0.3.20.
