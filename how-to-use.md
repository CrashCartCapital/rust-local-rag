# Rust Local RAG - Claude Desktop Usage

## Claude Desktop Configuration Template

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
                "OLLAMA_EMBEDDING_MODEL": "nomic-embed-text"
            }
        }
    }
}
```

**Important**: Replace `yourusername` with your actual username. Use absolute paths for reliable operation.

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATA_DIR` | Embeddings storage directory | `./data` |
| `DOCUMENTS_DIR` | PDF documents directory | `./documents` |
| `LOG_DIR` | Log files directory | `./logs` |
| `LOG_LEVEL` | Logging level (error/warn/info/debug) | `info` |
| `LOG_MAX_MB` | Log file size limit in MB | `5` |
| `OLLAMA_EMBEDDING_MODEL` | Ollama embedding model name (must be installed locally) | `nomic-embed-text` |

> 💡 Set `OLLAMA_EMBEDDING_MODEL` to any embedding model you've installed with `ollama pull`. The server verifies your choice at startup and provides guidance if the model is missing.

## Adding Documents

### 1. Add PDFs to Documents Directory
```bash
# Copy PDFs to your documents directory
cp your-file.pdf ~/Documents/rag/

# Or move multiple files
mv /path/to/pdfs/*.pdf ~/Documents/rag/
```

### 2. Restart Claude Desktop
The application will automatically:
- Detect new PDF files
- Extract text using poppler
- Generate embeddings
- Index documents for search

## Available MCP Tools

When configured, Claude Desktop can use these tools:

### 1. Search Documents
Search through your documents using semantic similarity.
- **Tool**: `search_documents`
- **Parameters**: `query` (string), `top_k` (optional number, default: 5)

### 2. List Documents  
Get a list of all indexed documents.
- **Tool**: `list_documents`
- **Parameters**: None

### 3. Get Statistics
View RAG system statistics and status.
- **Tool**: `get_stats` 
- **Parameters**: None

## Usage in Claude

Once configured, you can ask Claude to:

### Document Search Examples
- "Search my documents for information about machine learning"
- "What does my documentation say about API authentication?"
- "Find references to database optimization in my PDFs"

### Document Management Examples  
- "List all the documents you can access"
- "Show me statistics about the document index"
- "How many documents do you have indexed?"

### Analysis Examples
- "Summarize the key points from documents about project requirements"
- "Compare what different documents say about security best practices"
- "Find common themes across all my documentation"

## PDF Processing Details

### Supported PDF Types
- ✅ **Text-based PDFs**: Searchable text content
- ✅ **Mixed content**: PDFs with both text and images
- ⚠️ **Scanned PDFs**: Image-only documents (limited text extraction)
- ❌ **Password-protected**: Encrypted PDFs cannot be processed

### Text Extraction Process
1. **PDF to Text**: Uses poppler's `pdftotext` for reliable extraction
2. **Text Chunking**: Splits documents into ~500-1000 character segments
3. **Embedding Generation**: Creates vector embeddings using Ollama
4. **Indexing**: Stores embeddings for fast semantic search

### Performance Notes
- **First-time indexing**: May take several minutes for large document collections
- **Subsequent startups**: Uses cached embeddings for fast loading
- **Memory usage**: Scales with document collection size
- **Search speed**: Sub-second search responses after indexing

## Troubleshooting

### MCP Server Issues
1. **Server not connecting**:
   ```bash
   # Check binary exists and is executable
   which rust-local-rag
   ls -la ~/.cargo/bin/rust-local-rag
   ```

2. **Check Claude Desktop logs**:
   - **macOS**: `~/Library/Logs/Claude/mcp*.log`
   - **Windows**: `%APPDATA%\Claude\Logs\mcp*.log`
   - **Linux**: `~/.local/share/Claude/logs/mcp*.log`

### Document Processing Issues
1. **Documents not found**:
   ```bash
   # Verify directory exists and contains PDFs
   ls -la ~/Documents/rag/
   file ~/Documents/rag/*.pdf
   ```

2. **PDF processing failures**:
   ```bash
   # Test poppler installation
   pdftotext --version
   
   # Test PDF text extraction manually
   pdftotext ~/Documents/rag/sample.pdf -
   ```

3. **Empty search results**:
   - Check if documents were successfully indexed
   - Verify Ollama is running (`ollama serve`)
   - Check embedding model is installed (`ollama list`)

### Log Analysis
```bash
# View real-time logs
tail -f /tmp/rust-local-rag/rust-local-rag.log

# Search for errors
grep -i "error\|failed\|panic" /tmp/rust-local-rag/rust-local-rag.log

# Check document loading
grep -i "document" /tmp/rust-local-rag/rust-local-rag.log
```

### Configuration Validation
```bash
# Test configuration with manual run
DATA_DIR="~/Documents/data" \
DOCUMENTS_DIR="~/Documents/rag" \
LOG_DIR="/tmp/rust-local-rag" \
LOG_LEVEL="debug" \
rust-local-rag
```

## Performance Optimization

### For Large Document Collections
- Use SSD storage for `DATA_DIR` 
- Increase `LOG_MAX_MB` for detailed logging
- Consider splitting large PDFs into smaller files
- Monitor memory usage during initial indexing

### For Faster Searches
- Keep Ollama running continuously
- Use specific search terms rather than broad queries
- Adjust `top_k` parameter based on needs (lower = faster)

## Security Considerations

- Documents are processed locally (no external API calls)
- Embeddings stored locally in `DATA_DIR`
- Ollama runs locally for embedding generation
- No document content sent to external services
