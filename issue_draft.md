# Issue Draft: High Priority Security Vulnerabilities

**Title:** Security Vulnerabilities Found in Dependencies (`tracing-subscriber`, `rsa`)

**Description:**

A security audit of the Rust dependencies revealed the following vulnerabilities that require attention.

## 1. `tracing-subscriber` (High Priority)
*   **Vulnerability:** Logging user input may result in poisoning logs with ANSI escape sequences.
*   **Advisory ID:** [RUSTSEC-2025-0055](https://rustsec.org/advisories/RUSTSEC-2025-0055)
*   **Current Version:** 0.3.19
*   **Solution:** Upgrade `tracing-subscriber` to version `0.3.20` or later (latest is `0.3.22`).
*   **Action:** Bump dependency version in `Cargo.toml`.

## 2. `rsa` (Medium Priority - Transitive)
*   **Vulnerability:** Marvin Attack: potential key recovery through timing sidechannels.
*   **Advisory ID:** [RUSTSEC-2023-0071](https://rustsec.org/advisories/RUSTSEC-2023-0071)
*   **Current Version:** 0.9.9
*   **Source:** Transitive dependency via `sqlx-mysql` -> `sqlx` -> `rust-local-rag`.
*   **Solution:** No fixed upgrade is currently available.
*   **Action:** Monitor for updates in `sqlx` or `rsa`, or investigate if the vulnerability impacts the specific usage in this project.
