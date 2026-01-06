# ISSUE: Security and Dependency Hygiene Updates

**Priority:** HIGH

**Description:**
A security audit of the codebase has identified vulnerabilities and dependency health issues that need immediate attention.

## Vulnerabilities

### 1. ANSI Escape Sequence Poisoning (tracing-subscriber)
*   **Crate:** `tracing-subscriber` (0.3.19)
*   **Advisory:** RUSTSEC-2025-0055
*   **Fix:** Run `cargo update` to upgrade to `0.3.22`.

### 2. Marvin Attack (rsa)
*   **Crate:** `rsa` (0.9.9)
*   **Advisory:** RUSTSEC-2023-0071
*   **Context:** Transitive dependency via `sqlx`.
*   **Action:** No immediate fix available. Monitor for upstream updates in `sqlx`.

## Maintenance Tasks

### 3. Yanked Dependency (flate2)
*   **Crate:** `flate2` (1.1.7)
*   **Status:** Yanked.
*   **Fix:** Run `cargo update`.

### 4. Unmaintained Dependencies
*   **dotenv:** Replace with `dotenvy`.
*   **paste:** Mark for replacement/removal if possible.

## Action Plan
- [ ] Run `cargo update` to resolve `tracing-subscriber` and `flate2`.
- [ ] Create task to migrate `dotenv` -> `dotenvy`.
