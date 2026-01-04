# Security Vulnerabilities & Deprecated Crates

**Priority**: HIGH
**Labels**: security, dependencies

## Vulnerabilities

### 1. tracing-subscriber (RUSTSEC-2025-0055)
- **Advisory**: https://rustsec.org/advisories/RUSTSEC-2025-0055
- **Severity**: Medium
- **Description**: ANSI escape poisoning.
- **Fix**: Upgrade `tracing-subscriber` to >= 0.3.20 (0.3.22 available).

### 2. rsa (RUSTSEC-2023-0071)
- **Advisory**: https://rustsec.org/advisories/RUSTSEC-2023-0071
- **Severity**: Medium
- **Description**: Marvin Attack (timing sidechannels).
- **Context**: Pulled in via `sqlx` -> `sqlx-macros`.
- **Status**: No fixed upgrade available currently.

## Maintenance Warnings

### 3. dotenv (RUSTSEC-2021-0141)
- **Status**: Unmaintained.
- **Action**: Replace with `dotenvy`.

### 4. paste (RUSTSEC-2024-0436)
- **Status**: Unmaintained.

### 5. flate2
- **Status**: Version 1.1.7 is yanked.
- **Action**: Run `cargo update` to resolve (likely downgrades to 1.1.5 or upgrades if new version exists).
