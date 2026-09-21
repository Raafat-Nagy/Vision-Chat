.DEFAULT_GOAL := help
SHELL := /bin/bash

# Only the variables make actually uses. They come from .env when present
# (loaded first); the ?= fallbacks keep `make` working on a fresh clone
# without a .env. App-level settings (MODEL_NAME, LLAMA_API_KEY,
# MAX_TOKENS) are NOT listed here - the Python app reads them itself.
-include .env

# Used by: help banner
PROJECT_NAME ?= VisionChat API
VERSION ?= 1.0.0

# Used by: ui (Chainlit web UI bind address and port)
HOST ?= 0.0.0.0
PORT ?= 8000

# Used by: serve and mock-server (host/port are derived from it below)
LLAMA_BASE_URL ?= http://127.0.0.1:8081/v1

# Used by: serve (Hugging Face repo to download)
MODEL_REPO ?= Qwen/Qwen3-VL-2B-Instruct-GGUF

# Used by: serve. `make install-llama` installs a single multi-command
# binary at ~/.llama-app/llama, so the server is started with
# `llama serve ...` (no bare llama-server on PATH). Override if yours differs.
LLAMA_BIN ?= $(HOME)/.llama-app/llama

# Model server host/port are DERIVED from LLAMA_BASE_URL (single source of truth).
MODEL_SERVER_HOST := $(shell printf '%s' '$(LLAMA_BASE_URL)' | sed -nE 's|^[a-zA-Z]+://([^/:]+).*|\1|p')
MODEL_SERVER_PORT := $(shell printf '%s' '$(LLAMA_BASE_URL)' | sed -nE 's|.*:([0-9]+).*|\1|p')
ifeq ($(strip $(MODEL_SERVER_HOST)),)
MODEL_SERVER_HOST := 127.0.0.1
endif
ifeq ($(strip $(MODEL_SERVER_PORT)),)
MODEL_SERVER_PORT := 8081
endif

.PHONY: help config install install-uv install-llama serve mock-server ui test clean

help: ## Show available targets
	@echo "$(PROJECT_NAME) v$(VERSION) - make targets:"
	@grep -hE '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

config: ## Show the effective configuration
	@echo "--- make variables (from .env or fallbacks) ---"
	@echo "HOST             = $(HOST)"
	@echo "PORT             = $(PORT)"
	@echo "LLAMA_BASE_URL  = $(LLAMA_BASE_URL)"
	@echo "MODEL_REPO       = $(MODEL_REPO)"
	@echo "LLAMA_BIN        = $(LLAMA_BIN)"
	@echo "--- derived from LLAMA_BASE_URL ---"
	@echo "MODEL_SERVER_HOST = $(MODEL_SERVER_HOST)"
	@echo "MODEL_SERVER_PORT = $(MODEL_SERVER_PORT)"
	@echo "--- app settings (read by vision_chat from .env) ---"
	@if [ -f .env ]; then grep -E '^(MODEL_NAME|LLAMA_BASE_URL|LLAMA_API_KEY|MAX_TOKENS|LANGSMITH_TRACING|LANGSMITH_PROJECT)=' .env; else echo "(no .env - app defaults apply)"; fi

install: ## Create .venv and install all dependencies (uv sync)
	uv sync

install-uv: ## Install the uv package manager
	curl -LsSf https://astral.sh/uv/install.sh | sh

install-llama: ## Install llama.cpp via the official installer (~/.llama-app/llama)
	curl -LsSf https://llama.app/install.sh | sh
	@echo
	@echo "Installed llama at ~/.llama-app/llama"
	@echo "To use 'llama' from any terminal, add ~/.local/bin to PATH:"
	@echo "  echo 'export PATH=\$$HOME/.local/bin:\$$PATH' >> ~/.bash_profile"

serve: ## Start the model server: ~/.llama-app/llama serve (host/port from LLAMA_BASE_URL)
	$(LLAMA_BIN) serve --host $(MODEL_SERVER_HOST) --port $(MODEL_SERVER_PORT) -hf $(MODEL_REPO)

mock-server: ## Start the stateless mock llama-server on the LLAMA_BASE_URL port
	uv run python tools/mock_llama_server.py --port $(MODEL_SERVER_PORT)

ui: ## Run the Chainlit web UI (HOST/PORT from .env)
	uv run chainlit run app.py --host $(HOST) --port $(PORT) --headless

test: ## Run the test suite
	uv run pytest

clean: ## Remove caches and build artifacts
	@rm -rf .pytest_cache .ruff_cache dist build *.egg-info .files .chainlit/translations
	@find . -name __pycache__ -type d -prune -exec rm -rf {} +
