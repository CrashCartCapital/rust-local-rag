# Rust RAG Project Makefile
# Usage: make <target>

.PHONY: help build check test run clean update upgrade install-tools dev release lint clippy fix fmt watch ollama-start ollama-stop ollama-status ollama-models dev-start clean-all clean-ollama clean-build install install-release install-dev install-production uninstall which-installed logs kill setup ci
.DEFAULT_GOAL := help

help:
	@echo "Available targets:"
	@echo "  build          - Build the project"
	@echo "  check          - Check the project for errors"
	@echo "  test           - Run tests"
	@echo "  run            - Run the application"
	@echo "  clean          - Clean build artifacts only"
	@echo "  clean-build    - Clean Rust build cache"
	@echo "  clean-ollama   - Clean Ollama models (WARNING: Re-download needed)"
	@echo "  clean-all      - Clean everything (build + ask about Ollama)"
	@echo "  install        - Install binary to ~/.cargo/bin (global access)"
	@echo "  install-release- Install optimized release binary globally"
	@echo "  install-dev    - Build, check, and install for development"
	@echo "  install-production- Build optimized and install for production"
	@echo "  uninstall      - Uninstall the binary from system"
	@echo "  which-installed- Show where the binary is installed"
	@echo "  update         - Update dependencies within constraints"
	@echo "  upgrade        - Upgrade to latest versions (including breaking changes)"
	@echo "  install-tools  - Install required tools"
	@echo "  dev            - Development build and check"
	@echo "  release        - Release build"
	@echo "  lint           - Run clippy linter"
	@echo "  clippy         - Run clippy with all checks"
	@echo "  fix            - Auto-fix clippy issues"
	@echo "  fmt            - Format code with rustfmt"
	@echo "  watch          - Watch for changes and run checks automatically"
	@echo "  ollama-start   - Start Ollama server"
	@echo "  ollama-stop    - Stop Ollama server"
	@echo "  ollama-status  - Check Ollama status and models"
	@echo "  ollama-models  - Pull required models"
	@echo "  dev-start      - Start Ollama + run application"
	@echo "  setup-ollama   - Setup Ollama and pull embedding model"
	@echo "  setup          - Complete development environment setup"
	@echo "  ci             - Run CI pipeline (check, lint, test, build)"
	@echo "  logs           - View current log file"
	@echo "  kill           - Kill all rust-local-rag processes"

build:
	cargo build

check:
	cargo check

test:
	cargo test

run:
	DEV=true cargo run

clean:
	cargo clean

clean-build:
	@echo "🧹 Cleaning Rust build cache..."
	@echo "💾 Current size: $$(du -sh ./target 2>/dev/null | cut -f1 || echo '0B')"
	cargo clean
	@echo "✅ Build cache cleaned!"

clean-ollama:
	@echo "🧹 Cleaning Ollama models..."
	@echo "💾 Current size: $$(du -sh ~/.ollama 2>/dev/null | cut -f1 || echo '0B')"
	@echo "⚠️  WARNING: This will delete all Ollama models!"
	@echo "   You'll need to re-download them (~262MB)"
	@echo ""
	@read -p "🤔 Are you sure? [y/N]: " confirm; \
	if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
		make ollama-stop; \
		rm -rf ~/.ollama; \
		echo "✅ Ollama models deleted!"; \
		echo "📥 Run 'make setup-ollama' to reinstall"; \
	else \
		echo "❌ Cleanup cancelled"; \
	fi

clean-all:
	@echo "🧹 COMPREHENSIVE CLEANUP"
	@echo "========================"
	@echo "📊 Current Usage:"
	@echo "  Build cache: $$(du -sh ./target 2>/dev/null | cut -f1 || echo '0B')"
	@echo "  Ollama:      $$(du -sh ~/.ollama 2>/dev/null | cut -f1 || echo '0B')"
	@echo ""
	@echo "🗑️  Cleaning build cache (safe)..."
	@make clean-build
	@echo ""
	@echo "🤖 Clean Ollama models?"
	@make clean-ollama
	@echo ""
	@echo "✅ Cleanup complete!"

update:
	@echo "📦 Updating dependencies within existing constraints..."
	cargo update
	@echo "✅ Dependencies updated!"
	@echo "🔍 Checking for outdated dependencies..."
	-cargo outdated 2>/dev/null || echo "💡 Install cargo-outdated with: cargo install cargo-outdated"

upgrade:
	@echo "⬆️  Upgrading to latest versions (including breaking changes)..."
	-cargo upgrade --incompatible 2>/dev/null || echo "💡 Install cargo-edit with: cargo install cargo-edit"
	@echo "🔧 Verifying build after upgrade..."
	cargo check
	@echo "✅ Upgrade complete!"

install-tools:
	@echo "🛠️  Installing development tools..."
	cargo install cargo-outdated
	cargo install cargo-edit
	cargo install cargo-watch
	@echo "✅ Tools installed!"

dev: check lint build
	@echo "✅ Development build complete!"

release:
	cargo build --release

install:
	@echo "📦 Installing rust-local-rag globally..."
	cargo install --path .
	@echo "✅ Installation complete!"
	@echo "🎯 You can now run: rust-local-rag"
	@echo "📍 Installed at: $$(which rust-local-rag)"

install-release:
	@echo "📦 Installing optimized release version globally..."
	cargo install --path . --profile release
	@echo "✅ Installation complete!"
	@echo "🎯 You can now run: rust-local-rag"
	@echo "📍 Installed at: $$(which rust-local-rag)"

uninstall:
	@echo "🗑️  Uninstalling rust-local-rag..."
	@if cargo install --list | grep -q "rust-local-rag"; then \
		cargo uninstall rust-local-rag; \
		echo "✅ Uninstalled successfully!"; \
	else \
		echo "ℹ️  rust-local-rag is not installed"; \
	fi

which-installed:
	@echo "📍 Checking rust-local-rag installation..."
	@if command -v rust-local-rag >/dev/null 2>&1; then \
		echo "✅ rust-local-rag is installed at: $$(which rust-local-rag)"; \
		echo "📊 Version info:"; \
		ls -la "$$(which rust-local-rag)"; \
	else \
		echo "❌ rust-local-rag is not installed or not in PATH"; \
		echo "💡 Run 'make install' to install it"; \
	fi

lint:
	cargo clippy

clippy:
	cargo clippy -- -D warnings

fix:
	cargo clippy --fix --allow-dirty --allow-staged

fmt:
	cargo fmt

watch:
	cargo watch -x check -x "clippy -- -D warnings" -x test

ollama-start:
	@echo "🚀 Starting Ollama server..."
	@if pgrep -f "ollama serve" > /dev/null; then \
		echo "✅ Ollama is already running"; \
	else \
		echo "Starting Ollama in background..."; \
		nohup ollama serve > /tmp/ollama.log 2>&1 & \
		sleep 3; \
		echo "✅ Ollama started"; \
	fi

ollama-stop:
	@echo "🛑 Stopping Ollama server..."
	@if pgrep -f "ollama serve" > /dev/null; then \
		pkill -f "ollama serve"; \
		echo "✅ Ollama stopped"; \
	else \
		echo "ℹ️  Ollama is not running"; \
	fi

ollama-status:
	@echo "📊 Ollama Status:"
	@if pgrep -f "ollama serve" > /dev/null; then \
		echo "✅ Ollama is running (PID: $$(pgrep -f "ollama serve"))"; \
		echo "📋 Available models:"; \
		ollama list 2>/dev/null || echo "❌ Cannot connect to Ollama"; \
	else \
		echo "❌ Ollama is not running"; \
	fi

ollama-models:
	@echo "📥 Pulling required models..."
	@if ! pgrep -f "ollama serve" > /dev/null; then \
		echo "❌ Ollama is not running. Start it first with 'make ollama-start'"; \
		exit 1; \
	fi
	@echo "Pulling nomic-embed-text model..."
	ollama pull nomic-embed-text
	@echo "✅ Models ready!"

dev-start: ollama-start
	@echo "🚀 Starting development environment..."
	@sleep 2
	@make ollama-models
	@echo "🎯 Starting RAG application..."
	cargo run

setup-ollama:
	@echo "🚀 Setting up Ollama..."
	@if command -v ollama >/dev/null 2>&1; then \
		echo "✅ Ollama is already installed"; \
	else \
		echo "❌ Ollama not found. Please install it first:"; \
		echo "  macOS: brew install ollama"; \
		echo "  Linux: curl -fsSL https://ollama.ai/install.sh | sh"; \
		exit 1; \
	fi
	@echo "🔄 Starting Ollama server..."
	@if ! pgrep -f "ollama serve" > /dev/null; then \
		echo "Starting Ollama in background..."; \
		nohup ollama serve > /tmp/ollama.log 2>&1 & \
		sleep 3; \
	else \
		echo "✅ Ollama server is already running"; \
	fi
	@echo "📥 Pulling nomic-embed-text model..."
	ollama pull nomic-embed-text
	@echo "✅ Ollama setup complete!"

setup: install-tools setup-ollama
	@echo "🎉 Development environment setup complete!"

ci: check lint test build
	@echo "✅ CI pipeline passed!" 

install-dev: dev install
	@echo "🎉 Development build and installation complete!"
	@echo "🚀 Ready to use: rust-local-rag"

install-production: release install-release
	@echo "🎉 Production build and installation complete!"
	@echo "🚀 Ready to use: rust-local-rag" 

logs:
	@echo "📄 Current log file:"
	@if [ -f ./logs/rust-local-rag.log ]; then \
		echo "📊 Size: $$(du -sh ./logs/rust-local-rag.log | cut -f1)"; \
		echo ""; \
		tail -20 ./logs/rust-local-rag.log; \
	else \
		echo "❌ No log file found at ./logs/rust-local-rag.log"; \
	fi

kill:
	@echo "🔪 Killing rust-local-rag processes..."
	@if pgrep -f "rust-local-rag" > /dev/null; then \
		pkill -f "rust-local-rag"; \
		echo "✅ Killed rust-local-rag processes"; \
	else \
		echo "ℹ️  No rust-local-rag processes found"; \
	fi 
