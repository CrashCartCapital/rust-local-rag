# RAG-TUI Justfile
# Run `just` to see available commands

# Default: show help
default:
    @just --list

# Build the TUI binary
build:
    cargo build --bin rag-tui --release

# Run the TUI (connects to default localhost:3046)
tui:
    cargo run --bin rag-tui --release

# Run the TUI with custom server URL
tui-remote url:
    RAG_TUI_SERVER_URL={{url}} cargo run --bin rag-tui --release

# Start the RAG server in background and launch TUI
up:
    #!/usr/bin/env bash
    set -e
    echo "Starting rust-local-rag server..."
    cargo build --release
    mkdir -p logs
    ./target/release/rust-local-rag > logs/server.log 2>&1 &
    SERVER_PID=$!
    echo "Server started (PID: $SERVER_PID, logs: logs/server.log)"
    sleep 2
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "Server failed to start. Check logs/server.log"
        exit 1
    fi
    echo "Launching TUI..."
    cargo run --bin rag-tui --release

# Start server only (foreground, with logs to console)
server:
    DEV=true cargo run --release

# Start server in background (logs to file)
server-bg:
    #!/usr/bin/env bash
    cargo build --release
    mkdir -p logs
    ./target/release/rust-local-rag > logs/server.log 2>&1 &
    echo "Server started in background (PID: $!, logs: logs/server.log)"

# Tail server logs
logs:
    @tail -f logs/server.log

# Stop background server
server-stop:
    pkill -f "target/release/rust-local-rag" || echo "No server running"

# Check server health
health:
    @curl -s http://localhost:3046/healthz && echo " Server healthy" || echo "Server not responding"

# Get server stats
stats:
    @curl -s http://localhost:3046/stats | jq .

# Run all TUI tests
test:
    cargo test --bin rag-tui

# Run tests with output
test-verbose:
    cargo test --bin rag-tui -- --nocapture

# Quick search from command line (requires jq)
search query:
    @curl -s -X POST http://localhost:3046/search \
        -H "Content-Type: application/json" \
        -d '{"query": "{{query}}", "top_k": 5}' | jq '.results[] | {doc: .document, score: .score, text: .text[:80]}'

# Trigger reindex
reindex:
    @curl -s -X POST http://localhost:3046/reindex | jq .

# Install TUI to ~/.cargo/bin
install:
    cargo install --path . --bin rag-tui

# Clean build artifacts
clean:
    cargo clean
