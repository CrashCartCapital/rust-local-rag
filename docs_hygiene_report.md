# Docs Hygiene Report

## Security Audit Findings

The following security vulnerabilities were identified by `cargo audit`:

### Vulnerabilities

1.  **Crate:** `tracing-subscriber`
    *   **Version:** 0.3.19
    *   **Advisory ID:** [RUSTSEC-2025-0055](https://rustsec.org/advisories/RUSTSEC-2025-0055)
    *   **Title:** Logging user input may result in poisoning logs with ANSI escape sequences
    *   **Solution:** Upgrade to >=0.3.20. (Latest available is 0.3.22)

2.  **Crate:** `rsa`
    *   **Version:** 0.9.9 (Transitive dependency via `sqlx-mysql`)
    *   **Advisory ID:** [RUSTSEC-2023-0071](https://rustsec.org/advisories/RUSTSEC-2023-0071)
    *   **Title:** Marvin Attack: potential key recovery through timing sidechannels
    *   **Severity:** Medium (5.9)
    *   **Solution:** No fixed upgrade is available!

### Unmaintained / Yanked Packages

*   **`dotenv` (0.15.0):** Unmaintained ([RUSTSEC-2021-0141](https://rustsec.org/advisories/RUSTSEC-2021-0141)).
*   **`paste` (1.0.15):** Unmaintained ([RUSTSEC-2024-0436](https://rustsec.org/advisories/RUSTSEC-2024-0436)).
*   **`flate2` (1.1.7):** Yanked version.

## Stale Dependencies

The following dependencies have major version updates available (or breaking changes in 0.x):

| Package | Current | Latest |
| :--- | :--- | :--- |
| `crossterm` | v0.28.1 | v0.29.0 |
| `lopdf` | v0.34.0 | v0.38.0 |
| `lru` | v0.12.5 | v0.16.2 |
| `ratatui` | v0.29.0 | v0.30.0 |
| `reqwest` | v0.12.19 | v0.13.1 |
| `rmcp` | v0.8.5 | v0.12.0 |
| `rmcp-macros` | v0.8.5 | v0.12.0 |
| `schemars` | v0.9.0 | v1.2.0 |
