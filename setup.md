# Rust Local RAG - Setup for Claude Desktop

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

### 3. Install Poppler (for PDF parsing)
```bash
# macOS
brew install poppler

# Linux (Ubuntu/Debian)
sudo apt-get install poppler-utils

# Linux (CentOS/RHEL)
sudo yum install poppler-utils
```

## Setup

### 1. Build and Install
```bash
# Clone and build
git clone <repository-url>
cd rust-local-rag
cargo build --release

# Install globally
cargo install --path .
```

### 2. Create Directories and Add Documents
```bash
# Create required directories
mkdir -p ~/Documents/data
mkdir -p ~/Documents/rag
mkdir -p /tmp/rust-local-rag

# Add your PDF documents
cp your-pdfs/*.pdf ~/Documents/rag/
```

> ℹ️  By default the server writes logs to `/var/log/rust-local-rag` when it can, otherwise `./logs`. Creating `/tmp/rust-local-rag` gives you a convenient writable location that you can point `LOG_DIR` at from your Claude configuration.

## Claude Desktop Integration

### 1. Find Claude Desktop Config
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
- **Linux**: `~/.config/Claude/claude_desktop_config.json`

### 2. Add This Configuration
```json
{
    "mcpServers": {
        "rust-local-rag": {
            "command": "/Users/yourusername/.cargo/bin/rust-local-rag",
            "env": {
                "DATA_DIR": "/Users/yourusername/Documents/data",
                "DOCUMENTS_DIR": "/Users/yourusername/Documents/rag",
                "LOG_DIR": "/tmp/rust-local-rag",
                "LOG_LEVEL": "info",
                "LOG_MAX_MB": "10",
                "OLLAMA_URL": "http://localhost:11434"
            }
        }
    }
}
```

**Important**: Replace `yourusername` with your actual username, or use absolute paths specific to your system.

### 3. Find Your Actual Paths
```bash
# Find your cargo bin directory
echo "$HOME/.cargo/bin/rust-local-rag"

# Verify the binary exists
which rust-local-rag
```

### 4. Restart Claude Desktop

## How PDF Processing Works

The application automatically:

1. **Scans the documents directory** on startup for PDF files
2. **Extracts text** using poppler's `pdftotext` utility
3. **Chunks the text** into manageable segments (typically 500-1000 characters)
4. **Generates embeddings** using Ollama's `nomic-embed-text` model
5. **Stores embeddings** in the data directory for fast retrieval
6. **Indexes documents** for semantic search

Supported formats:
- PDF files (via poppler)
- Text extraction preserves basic formatting
- Each document is split into searchable chunks

## Troubleshooting

### Installation Issues
1. **Rust not found**: Restart terminal after installing Rust
2. **Ollama connection failed**: Ensure `ollama serve` is running
3. **Poppler not found**: Verify installation with `pdftotext --version`

### Claude Desktop Issues
1. **Binary not found**: Check path with `which rust-local-rag`
2. **Permission denied**: Ensure directories are writable
3. **No documents indexed**: Check PDF files exist in `DOCUMENTS_DIR`
4. **Connection failed**: Check logs in `LOG_DIR` directory

### PDF Processing Issues
1. **Text extraction failed**: Ensure PDFs are not password-protected or corrupted
2. **Empty results**: Some PDFs may be image-only (scanned documents)
3. **Slow indexing**: Large documents take time to process on first run

### Log Files
Check application logs for detailed error information:
```bash
# View latest logs
tail -f /tmp/rust-local-rag/rust-local-rag.log

# Check for errors
grep -i error /tmp/rust-local-rag/rust-local-rag.log
```

That's it! Your documents will be automatically indexed and searchable in Claude Desktop.

## Environment Reference

You can customise the server using environment variables or a `.env` file placed alongside the binary. Common options include:

| Variable | Purpose | Default |
|----------|---------|---------|
| `DATA_DIR` | Local storage for embeddings | `./data` |
| `DOCUMENTS_DIR` | Directory scanned for PDFs | `./documents` |
| `LOG_DIR` | Log output directory. Uses `/var/log/rust-local-rag` when writable, otherwise `./logs`. | Auto-detected |
| `LOG_LEVEL` | Logging level (`error`, `warn`, `info`, `debug`, `trace`) | `info` |
| `LOG_MAX_MB` | Maximum log file size before truncation | `5` |
| `OLLAMA_URL` | Base URL for Ollama | `http://localhost:11434` |
| `OLLAMA_EMBEDDING_MODEL` | Embedding model name | `nomic-embed-text` |
| `DEVELOPMENT` / `DEV` | Prefer console logging (development friendly) | _unset_ |
| `CONSOLE_LOGS` | Force console logging regardless of environment | _unset_ |

> 🧹 Logs larger than `LOG_MAX_MB` are automatically truncated by a background task to keep disk usage predictable.
