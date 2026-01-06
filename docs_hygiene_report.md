# Dependency Hygiene Report

## Security Vulnerabilities
Run `cargo audit` to reproduce.

### High/Medium Priority
*   **tracing-subscriber (0.3.19)**
    *   **ID:** [RUSTSEC-2025-0055](https://rustsec.org/advisories/RUSTSEC-2025-0055)
    *   **Issue:** Logging user input may result in poisoning logs with ANSI escape sequences.
    *   **Remediation:** Upgrade to `>=0.3.20`. `cargo update` successfully upgrades this to `0.3.22`.
*   **rsa (0.9.9)**
    *   **ID:** [RUSTSEC-2023-0071](https://rustsec.org/advisories/RUSTSEC-2023-0071)
    *   **Issue:** Marvin Attack: potential key recovery through timing sidechannels.
    *   **Severity:** Medium (5.9).
    *   **Status:** No fixed upgrade available. Transitive dependency via `sqlx-mysql` -> `sqlx`.

## Dependency Health

### Unmaintained / Yanked
*   **dotenv (0.15.0)**
    *   **ID:** [RUSTSEC-2021-0141](https://rustsec.org/advisories/RUSTSEC-2021-0141)
    *   **Status:** Unmaintained.
    *   **Recommendation:** Replace with `dotenvy`.
*   **paste (1.0.15)**
    *   **ID:** [RUSTSEC-2024-0436](https://rustsec.org/advisories/RUSTSEC-2024-0436)
    *   **Status:** Unmaintained.
*   **flate2 (1.1.7)**
    *   **Status:** Yanked from registry.
    *   **Remediation:** `cargo update` resolves this by moving to `1.1.5` (or newer if available).

### Major Version Updates Available
*   **reqwest**: Using `0.12.x`, available `0.13.1`.

## Recommendations
1.  Run `cargo update` immediately to fix `tracing-subscriber` and `flate2`.
2.  Plan migration from `dotenv` to `dotenvy`.
3.  Monitor `sqlx` for `rsa` updates.
