# Rust Local RAG - Setup

## Prerequisites

### 1. Install Rust
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
```

### 2. Install Ollama
```bash
# macOS
brew install ollama

# Linux  
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama
ollama serve

# Install embedding model
ollama pull nomic-embed-text
```

### 3. Install Poppler (Optional - for fallback PDF parsing)

The server uses pure-Rust `lopdf` for PDF extraction by default. Poppler is only needed as a fallback for complex PDFs that lopdf cannot handle.

```bash
# macOS
brew install poppler

# Linux (Ubuntu/Debian)
sudo apt-get install poppler-utils

# Linux (CentOS/RHEL)
sudo yum install poppler-utils
```

## Build and Install

```bash
# Clone and build
git clone <repository-url>
cd rust-local-rag
cargo build --release

# Install globally
cargo install --path .
```

## Next Steps

*   **Configuration**: See [Configuration Guide](configuration.md) for setting up Claude Desktop and Environment Variables.
*   **Usage**: See [Usage Guide](usage.md) for how to use the tools in Claude.
*   **Architecture**: See [Architecture Guide](architecture.md) for details on how it works.
