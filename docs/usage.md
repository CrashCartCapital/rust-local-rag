# Usage Guide

## Quick Start

Once configured (see [Configuration](configuration.md)), restart Claude Desktop. You can now use the `rust-local-rag` MCP tools.

## Available MCP Tools

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

### 4. Start Reindex
Trigger background reindexing of all documents.
- **Tool**: `start_reindex`
- **Parameters**: None
- **Returns**: Job ID for tracking progress

### 5. Get Job Status
Check the status of a background job (like reindexing).
- **Tool**: `get_job_status`
- **Parameters**: `job_id` (string)

### 6. Calibrate Reranker
Measure LLM reranking latencies and get timeout recommendations.
- **Tool**: `calibrate_reranker`
- **Parameters**: `query` (string), `sample_size` (optional number, default: 100)

## Example Prompts

*   "Search my documents for information about machine learning"
*   "What does my documentation say about API authentication?"
*   "Summarize the key points from documents about project requirements"
*   "List all the documents you can access"

## Troubleshooting

See [Configuration](configuration.md) for log locations and troubleshooting configuration issues.
