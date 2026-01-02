# Dependency Hygiene Report

## Security Audit

### Vulnerabilities

**CRITICAL ATTENTION REQUIRED**

1.  **Crate:** `rsa` (0.9.9)
    *   **ID:** RUSTSEC-2023-0071
    *   **Title:** Marvin Attack: potential key recovery through timing sidechannels
    *   **Severity:** Medium (5.9)
    *   **Status:** No fixed upgrade is available.
    *   **Dependency Chain:** `sqlx-mysql` -> `sqlx` -> `rust-local-rag`

2.  **Crate:** `tracing-subscriber` (0.3.19)
    *   **ID:** RUSTSEC-2025-0055
    *   **Title:** Logging user input may result in poisoning logs with ANSI escape sequences
    *   **Solution:** Upgrade to >=0.3.20 (Available: 0.3.22)

### Unmaintained / Yanked Packages

*   **dotenv (0.15.0):** Unmaintained since 2021 (RUSTSEC-2021-0141). Consider switching to `dotenvy`.
*   **paste (1.0.15):** Unmaintained (RUSTSEC-2024-0436).
*   **flate2 (1.1.7):** Version yanked from crates.io. `cargo update` suggests downgrading to 1.1.5 or upgrading if a newer version exists (checked: 1.1.7 is yanked, 1.1.5 is stable or check for newer).

## Stale Dependencies

Analysis performed using `cargo update --dry-run` and manual inspection (due to `cargo-outdated` installation constraints).

### Major Version Updates Available

*   **reqwest:** v0.12.x -> v0.13.1

### Notable Minor Updates

*   **tracing-subscriber:** v0.3.19 -> v0.3.22 (Fixes security vulnerability)
*   **flate2:** v1.1.7 -> v1.1.5 (Downgrade recommended due to yanked version)

---

## Action Plan (Draft Issue)

**Title:** Security: Resolve vulnerabilities in `rsa` and `tracing-subscriber` and replace unmaintained crates

**Description:**

A security audit revealed the following issues that require immediate attention:

**Vulnerabilities:**
*   `rsa` (via `sqlx`): "Marvin Attack" (RUSTSEC-2023-0071). No fix available in `rsa` yet. We should check if `sqlx` has an update that mitigates this or uses a different backend/version.
*   `tracing-subscriber`: Log poisoning vulnerability (RUSTSEC-2025-0055). **Fix:** Update `tracing-subscriber` to `0.3.22`.

**Maintenance:**
*   Replace `dotenv` with `dotenvy` (maintained fork).
*   Investigate replacement for `paste` if necessary, or acknowledge unmaintained status.
*   Fix `flate2` version (currently using yanked 1.1.7).

**Action Items:**
1.  Run `cargo update -p tracing-subscriber` to get v0.3.22.
2.  Run `cargo update -p flate2` to move off the yanked version.
3.  Replace `dotenv` with `dotenvy` in `Cargo.toml`.
4.  Investigate `sqlx` / `rsa` situation.
