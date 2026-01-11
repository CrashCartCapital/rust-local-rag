# Dependency Hygiene Report

## Summary
- **Vulnerabilities**: 2 found (1 with fix, 1 without fix).
- **Major Updates Available**: `reqwest` (v0.12 -> v0.13).
- **Warnings**: Unmaintained (`dotenv`, `paste`), Unsound (`lru`), Yanked (`flate2`).

## Security Vulnerabilities (High Priority)

### 1. `rsa` (v0.9.9)
- **Advisory ID**: [RUSTSEC-2023-0071](https://rustsec.org/advisories/RUSTSEC-2023-0071)
- **Title**: Marvin Attack: potential key recovery through timing sidechannels
- **Severity**: Medium (5.9)
- **Status**: **No fixed upgrade is available!**
- **Dependency Chain**: `rust-local-rag` -> `sqlx` -> `sqlx-macros` -> `sqlx-mysql` -> `rsa`

### 2. `tracing-subscriber` (v0.3.19)
- **Advisory ID**: [RUSTSEC-2025-0055](https://rustsec.org/advisories/RUSTSEC-2025-0055)
- **Title**: Logging user input may result in poisoning logs with ANSI escape sequences
- **Severity**: Not specified in summary, but implies log injection risk.
- **Status**: Fix available (Upgrade to >=0.3.20).
- **Action**: Run `cargo update` (will upgrade to v0.3.22).

## Warnings

### Unmaintained Crates
- **`dotenv` (v0.15.0)**: Unmaintained since 2021. Consider replacing with `dotenvy`.
- **`paste` (v1.0.15)**: Unmaintained since 2024.

### Unsound Crates
- **`lru` (v0.12.5)**: [RUSTSEC-2026-0002](https://rustsec.org/advisories/RUSTSEC-2026-0002) - `IterMut` violates Stacked Borrows.
  - Dependency Chain: `rust-local-rag` -> `lru` and `ratatui` -> `lru`.

### Yanked Crates
- **`flate2` (v1.1.7)**: Yanked version.
  - **Action**: Run `cargo update` (will downgrade to v1.1.5).

## Stale Dependencies (Major Updates)

The following dependencies have major version updates available:

| Crate | Current | Available |
|-------|---------|-----------|
| `reqwest` | v0.12.28 | v0.13.1 |

---

## Issue Draft

**Title**: Security Vulnerabilities in `rsa` and `tracing-subscriber`

**Description**:
During a routine security audit, the following vulnerabilities were identified:

### 1. Marvin Attack in `rsa` (RUSTSEC-2023-0071)
- **Crate**: `rsa` v0.9.9
- **Impact**: Potential private key recovery via timing side-channels.
- **Context**: Used transitively via `sqlx-mysql`.
- **Mitigation**: Currently, no fixed version is available. We should monitor `sqlx` for updates that might replace or update this dependency, or investigate if `rsa` releases a fix (v0.9.10 is available but advisory claims no fix).

### 2. Log Poisoning in `tracing-subscriber` (RUSTSEC-2025-0055)
- **Crate**: `tracing-subscriber` v0.3.19
- **Impact**: User input could inject ANSI escape sequences into logs.
- **Mitigation**: Upgrade to >=0.3.20.
- **Action Plan**: Run `cargo update` to pull in version 0.3.22 which fixes this issue.

**Recommended Actions**:
1. Run `cargo update` immediately to fix `tracing-subscriber` and remove the yanked `flate2`.
2. Investigate replacement for `dotenv` (e.g., `dotenvy`) and `lru` if possible.
3. Monitor `rsa` advisory for updates.
