.PHONY: install install-dev lint test test-unit test-integration preflight clean help

PYTHON ?= python3
PIP ?= pip

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install package in production mode
	$(PIP) install -e .

install-dev: ## Install with all optional + dev dependencies
	$(PIP) install -e ".[all,dev]"

install-lora: ## Install for LoRA training
	$(PIP) install -e ".[lora]"

install-osft: ## Install for OSFT training
	$(PIP) install -e ".[osft]"

install-backend: ## Install for RAG backend
	$(PIP) install -e ".[backend]"

install-eval: ## Install for evaluation
	$(PIP) install -e ".[eval]"

install-sdg: ## Install for synthetic data generation
	$(PIP) install -e ".[sdg]"

lint: ## Run linter (ruff)
	$(PYTHON) -m ruff check src/ tests/ scripts/
	$(PYTHON) -m ruff format --check src/ tests/ scripts/

format: ## Auto-format code
	$(PYTHON) -m ruff check --fix src/ tests/ scripts/
	$(PYTHON) -m ruff format src/ tests/ scripts/

test: ## Run all tests
	$(PYTHON) -m pytest tests/ -v

test-unit: ## Run unit tests only
	$(PYTHON) -m pytest tests/unit/ -v

test-integration: ## Run integration tests
	$(PYTHON) -m pytest tests/integration/ -v

preflight: ## Validate configuration and bundle readiness
	$(PYTHON) -m rhoai_model_training_lab.cli preflight

clean: ## Clean build artifacts
	rm -rf build/ dist/ *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
