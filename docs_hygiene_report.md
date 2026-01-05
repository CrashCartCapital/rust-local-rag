# Dependency Hygiene Report

**Date**: 2025-10-24
**Auditor**: Jules
**Scope**: Rust dependencies (`Cargo.toml`, `Cargo.lock`)

## 1. Security Vulnerabilities

The following vulnerabilities were identified by `cargo audit`.

### [High Priority] `rsa` (RUSTSEC-2023-0071)
*   **Package**: `rsa` v0.9.9
*   **Severity**: Medium (5.9)
*   **Advisory**: [RUSTSEC-2023-0071](https://rustsec.org/advisories/RUSTSEC-2023-0071)
*   **Description**: Marvin Attack: potential key recovery through timing sidechannels.
*   **Status**: No fixed upgrade is available.
*   **Dependency Chain**: `sqlx-mysql` -> `rust-local-rag`

### [Actionable] `tracing-subscriber` (RUSTSEC-2025-0055)
*   **Package**: `tracing-subscriber` v0.3.19
*   **Severity**: Low/Medium
*   **Advisory**: [RUSTSEC-2025-0055](https://rustsec.org/advisories/RUSTSEC-2025-0055)
*   **Description**: Logging user input may result in poisoning logs with ANSI escape sequences.
*   **Solution**: Upgrade to `>=0.3.20`.
*   **Dependency Chain**: Direct dependency.

## 2. Supply Chain Risks (Unmaintained/Yanked)

### Unmaintained
*   **`dotenv` v0.15.0**: [RUSTSEC-2021-0141](https://rustsec.org/advisories/RUSTSEC-2021-0141). Unmaintained since 2021.
    *   *Recommendation*: Replace with `dotenvy`.
*   **`paste` v1.0.15**: [RUSTSEC-2024-0436](https://rustsec.org/advisories/RUSTSEC-2024-0436). No longer maintained.
    *   *Dependency Chain*: `rmcp`, `ratatui`.

### Yanked
*   **`flate2` v1.1.7**: This version has been yanked from crates.io.
    *   *Dependency Chain*: `lopdf`, `arboard` (via `image` -> `tiff`).

## 3. Stale Dependencies (Major Updates)

The following packages have major version updates available or significant minor updates that were blocked by version constraints.

*   **`reqwest`**: v0.12.28 installed -> **v0.13.1** available.
*   **`tracing-subscriber`**: v0.3.19 installed -> **v0.3.22** available (fixes vulnerability).

## 4. Summary & Action Items

1.  **URGENT**: Upgrade `tracing-subscriber` to `0.3.22` or later to resolve RUSTSEC-2025-0055.
2.  **URGENT**: Investigate usage of `sqlx-mysql` / `rsa`. Since `rsa` has no fix, determine if `sqlx` has an update that moves away from it or if the vulnerability is reachable.
3.  **Maintenance**: Replace `dotenv` with `dotenvy`.
4.  **Monitor**: `paste` is unmaintained but widely used. Monitor for replacement in `rmcp` and `ratatui` or upstream fixes.
5.  **Clean up**: Run `cargo update` to move off yanked `flate2` v1.1.7 (downgrade to v1.1.5 or upgrade if possible).

---

## Issue Draft (High Priority)

**Title**: Security Vulnerability in `rsa` and `tracing-subscriber` dependencies

**Description**:
A security audit revealed two vulnerabilities in our dependency tree.

**1. `rsa` (RUSTSEC-2023-0071)**
*   **Severity**: Medium
*   **Impact**: Potential key recovery through timing sidechannels.
*   **Source**: Pulled in via `sqlx-mysql`.
*   **Action**: Investigate if we can update `sqlx` or disable the feature causing this dependency if not used. Note: There is currently no fixed version of `rsa`.

**2. `tracing-subscriber` (RUSTSEC-2025-0055)**
*   **Severity**: Low/Medium
*   **Impact**: Log poisoning via ANSI escape sequences.
*   **Action**: Upgrade `tracing-subscriber` to `>=0.3.20`.

**3. Unmaintained/Yanked Packages**
*   `dotenv` is unmaintained. Replace with `dotenvy`.
*   `flate2` v1.1.7 is yanked. Run `cargo update` to resolve.

**Recommended Action**:
Run `cargo update` to pull in the latest compatible versions (this should fix `tracing-subscriber` and `flate2`). Evaluate replacing `dotenv`.
