# Single Source of Truth Docs Pass Report

## Canonical Docs

| Topic | Canonical File | Status |
|---|---|---|
| **Setup** | `docs/setup.md` | Stripped down to Installation only. |
| **Configuration** | `docs/configuration.md` | **New**. Contains all Env Vars, Claude Config, and Model switching details. |
| **Usage** | `docs/usage.md` | Formerly `how-to-use.md`. Focused on MCP Tools and Examples. |
| **Development** | `docs/development.md` | **New**. Contains build commands, repo layout, and testing info. |
| **Architecture** | `docs/architecture.md` | **New**. Contains high-level design, pipeline details, and job system info. |

## Deduplication Actions

*   **Env Vars**: Removed duplication from `README.md`, `setup.md`, and `usage.md`. Centralized in `docs/configuration.md`.
*   **Claude Config**: Removed duplication from `README.md`, `setup.md`. Centralized in `docs/configuration.md`.
*   **Architecture**: Extracted from `README.md` and `CLAUDE.md` into `docs/architecture.md`.
*   **Dev Commands**: Extracted from `README.md` and `AGENTS.md` into `docs/development.md`.
*   **README**: Reduced to a documentation map.

## Contradictions Resolved

*   **Config Drift**: Previously, different files might have listed different default values or partial lists of env vars. Now `docs/configuration.md` is the single source of truth.
*   **Setup Instructions**: `README.md` had a "Quick Start" that duplicated `setup.md`. Now `README.md` points to `docs/setup.md`.

## Follow-ups

*   `CLAUDE.md` and `AGENTS.md` were updated to reference the canonical docs where appropriate, but still contain some context-specific instructions for agents. This is intentional.
