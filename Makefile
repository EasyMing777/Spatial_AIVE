# AIVE — development helpers
#
#   make install     editable install with dev extras
#   make lint        ruff lint + format check
#   make format      auto-format with ruff
#   make type        mypy type checking
#   make test        run the pytest suite
#   make run         run the AIVE pipeline (mock Dreamer)
#   make clean       remove caches

PYTHON ?= python
PACKAGES := pipelines utils dreamer

.PHONY: help install lint format type test run clean

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-8s\033[0m %s\n", $$1, $$2}'

install: ## Editable install with dev extras
	$(PYTHON) -m pip install -e ".[dev]"

lint: ## Run ruff lint + format check
	ruff check $(PACKAGES)
	ruff format --check $(PACKAGES)

format: ## Auto-format with ruff
	ruff format $(PACKAGES)
	ruff check --fix $(PACKAGES)

type: ## Run mypy type checking
	mypy $(PACKAGES)

test: ## Run the pytest suite
	pytest -q

run: ## Run the AIVE pipeline (mock Dreamer)
	bash scripts/run_aive.sh

clean: ## Remove caches and build artifacts
	rm -rf .ruff_cache .mypy_cache .pytest_cache .coverage htmlcov build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
