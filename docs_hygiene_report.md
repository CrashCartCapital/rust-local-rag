# Dependency Hygiene Report

**Date**: 2026-01-04
**Auditor**: Jules

---

## Security Vulnerabilities (cargo audit)

| Crate | Version | ID | Severity | Status |
|-------|---------|----|----------|--------|
| `rsa` | 0.9.9 | RUSTSEC-2023-0071 | Medium | No fix available (via `sqlx`) |
| `tracing-subscriber` | 0.3.19 | RUSTSEC-2025-0055 | Medium | Fix available (>=0.3.20) |
| `dotenv` | 0.15.0 | RUSTSEC-2021-0141 | Warning | Unmaintained (Use `dotenvy`) |
| `paste` | 1.0.15 | RUSTSEC-2024-0436 | Warning | Unmaintained |
| `flate2` | 1.1.7 | Yanked | Warning | Yanked version in use |

## Stale Dependencies (cargo update --dry-run)

| Crate | Current | Latest Compatible | Latest Available |
|-------|---------|-------------------|------------------|
| `reqwest` | 0.12.19 | 0.12.28 | 0.13.1 |
| `flate2` | 1.1.7 | 1.1.5 (Downgrade) | - |
| `tracing-subscriber` | 0.3.19 | 0.3.22 | - |
| `tokio` | 1.48.0 | 1.49.0 | - |
| `axum` | 0.8.7 | 0.8.8 | - |

*Note: `cargo update` indicates many minor/patch updates are available.*

## Recommendations

1.  **High Priority**:
    -   Update `tracing-subscriber` to fix RUSTSEC-2025-0055 (Available in `0.3.22`).
    -   Investigate `flate2` downgrade (likely to move off yanked version).
    -   Replace `dotenv` with `dotenvy`.

2.  **Monitor**:
    -   `rsa`: Wait for upstream `sqlx` fix or mitigation.
    -   `paste`: Low risk macro crate, but consider alternatives if possible.

3.  **Upgrade**:
    -   Run `cargo update` to apply compatible updates (including `tracing-subscriber` fix).
    -   Consider upgrading `reqwest` to 0.13.x (requires code changes likely).
