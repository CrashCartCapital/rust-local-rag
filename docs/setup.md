# Rust Local RAG - Setup

## Prerequisites

### 1. Install Rust
Ensure you have the latest stable Rust toolchain installed.

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
```

### 2. Install Ollama
You need Ollama running to handle embeddings and LLM calls.

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh
```

**Start Ollama:**
You must have the Ollama server running.

```bash
# Start in background
ollama serve > /dev/null 2>&1 &

# Verify it's running
curl localhost:11434
```

**Install Embedding Model:**
The system uses `nomic-embed-text` by default.

```bash
ollama pull nomic-embed-text
```

> **Tip:** You can also use `make setup-ollama` (after cloning) to automate starting Ollama and pulling the model.

### 3. Install Poppler (Optional)

The server uses pure-Rust `lopdf` for PDF extraction by default. Poppler is recommended as a robust fallback for complex PDFs.

```bash
# macOS
brew install poppler

# Linux (Ubuntu/Debian)
sudo apt-get install poppler-utils

# Linux (CentOS/RHEL/Fedora)
sudo dnf install poppler-utils
```

## Build and Install

### 1. Clone the repository

```bash
git clone <repository-url>
cd rust-local-rag
```

### 2. Install the binary

You can install the `rust-local-rag` binary to your `~/.cargo/bin` path for global access.

```bash
# Install optimized release version
make install-release
```

*Alternatively, using cargo directly:*

```bash
cargo install --path . --profile release
```

### 3. Verify Installation

Ensure the binary is installed and available in your PATH.

```bash
which rust-local-rag
# Output should be something like: /home/user/.cargo/bin/rust-local-rag
```

Note: `rust-local-rag` is a server application designed to be run by the Claude Desktop MCP client or via `make run`. It does not have a CLI interface for help (`--help` will not work).

## Next Steps

*   **Configuration**: See [Configuration Guide](configuration.md) for setting up Claude Desktop and Environment Variables.
*   **Usage**: See [Usage Guide](usage.md) for how to use the tools in Claude.
*   **Architecture**: See [Architecture Guide](architecture.md) for details on how it works.
