#!/usr/bin/env bash
# Startup script for rust-local-rag with reranker-phi4mini.
#
# Behavior:
# - Loads environment variables from a `.env` file located next to this script (repo root).
# - `.env` must be shell-compatible: `KEY=value` (quote values containing spaces).
# - Uses `source` (not `grep|xargs`) so quoted values and spaces are preserved.
# - Runs `rust-local-rag` from your PATH (e.g., ~/.cargo/bin). For local builds use:
#   `cargo run --bin rust-local-rag`

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"

if [[ -f "$ENV_FILE" ]]; then
  echo "Loading environment from $ENV_FILE"
  set -a
  # shellcheck disable=SC1090
  set +u
  source "$ENV_FILE"
  set -u
  set +a
else
  echo "WARNING: no .env found at $ENV_FILE; using current environment." >&2
fi

if ! command -v rust-local-rag >/dev/null 2>&1; then
  echo "ERROR: rust-local-rag not found in PATH." >&2
  echo "Hint: install with `cargo install --path .` or run with `cargo run --bin rust-local-rag`." >&2
  exit 1
fi

echo "Starting rust-local-rag with reranker-phi4mini:latest..."
echo "Embedding model: ${OLLAMA_EMBEDDING_MODEL:-<unset>}"
echo "Rerank model: ${OLLAMA_RERANK_MODEL:-<unset>}"
echo "Documents: ${DOCUMENTS_DIR:-<unset>}"
echo ""

exec rust-local-rag
