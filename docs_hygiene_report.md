# Dependency Hygiene Report

**Date:** 2025-05-15 (Estimated based on environment)
**Tools:** `cargo audit`, `cargo update --dry-run` (fallback for `cargo outdated`)

## Summary
The audit identified **2 vulnerabilities** (1 with a fix available, 1 with no fix) and **3 warnings** (unmaintained/unsound crates). Several dependencies can be updated to newer minor/patch versions.

## Security Vulnerabilities (`cargo audit`)

| Crate | Version | Severity | ID | Status |
|-------|---------|----------|----|--------|
| **tracing-subscriber** | 0.3.19 | Medium | [RUSTSEC-2025-0055](https://rustsec.org/advisories/RUSTSEC-2025-0055) | **Fix Available**: Update to >=0.3.20 |
| **rsa** | 0.9.9 | Medium | [RUSTSEC-2023-0071](https://rustsec.org/advisories/RUSTSEC-2023-0071) | **Unresolved**: No fixed upgrade available. (Dependency of `sqlx-mysql`) |

### Warnings
- **dotenv (0.15.0)**: Unmaintained ([RUSTSEC-2021-0141](https://rustsec.org/advisories/RUSTSEC-2021-0141)).
- **paste (1.0.15)**: Unmaintained ([RUSTSEC-2024-0436](https://rustsec.org/advisories/RUSTSEC-2024-0436)).
- **lru (0.12.5)**: Unsoundness ([RUSTSEC-2026-0002](https://rustsec.org/advisories/RUSTSEC-2026-0002)). Used by `ratatui`.
- **flate2 (1.1.7)**: Yanked. (Found via `cargo audit`).

## Stale Dependencies
*Note: `cargo outdated` timed out, so this list is based on `cargo update --dry-run` which mostly shows minor/patch updates, with limited major version info.*

- **reqwest**: 0.12.19 -> 0.12.28 (Newer major version v0.13.1 available)
- **flate2**: 1.1.7 -> 1.1.5 (Downgrade recommended due to 1.1.7 being yanked)
- **tracing-subscriber**: 0.3.19 -> 0.3.22 (This update fixes RUSTSEC-2025-0055)

---

## Recommended Actions (Issue Draft)

**Title:** [Security] Fix Vulnerabilities in `tracing-subscriber` and Address Unmaintained Dependencies

**Description:**
A security audit of the project dependencies revealed the following issues that require immediate attention.

### 1. `tracing-subscriber` Vulnerability (RUSTSEC-2025-0055)
- **Issue**: Logging user input may result in poisoning logs with ANSI escape sequences.
- **Current Version**: 0.3.19
- **Fix**: Run `cargo update -p tracing-subscriber` to upgrade to v0.3.22 (verified available).

### 2. `rsa` Vulnerability (RUSTSEC-2023-0071)
- **Issue**: Marvin Attack: potential key recovery through timing sidechannels.
- **Context**: Transitive dependency via `sqlx-mysql`.
- **Status**: No fix is currently available in the `rsa` crate. We should monitor `sqlx` updates for a mitigation or alternative.

### 3. Unmaintained/Unsound Dependencies
- **lru**: Used by `ratatui`. Contains unsoundness (RUSTSEC-2026-0002). Check if a newer `ratatui` version removes this dependency.
- **dotenv**: Unmaintained. Consider migrating to `dotenvy`.
- **paste**: Unmaintained.

### Action Plan
1. Run `cargo update` to apply non-breaking fixes (solves `tracing-subscriber` and `flate2`).
2. Investigate `sqlx` updates or configuration to mitigate `rsa` risks.
3. Plan migration from `dotenv` to `dotenvy`.
