# Credit default risk pipeline.
# `make help` lists the targets.
#
# Portability note: Make chooses its own shell - /bin/sh on macOS and Linux,
# cmd.exe on Windows unless a POSIX sh is on PATH. So no recipe here may rely on
# shell builtins or Unix tools. Two consequences:
#
#   * environment variables are set with Make's `export`, never with a
#     `VAR=value command` prefix (that syntax is POSIX-shell only and fails
#     under cmd.exe);
#   * targets that would otherwise need grep/awk/rm/find delegate to
#     scripts/devtools.py.
#
# The result is that every target below works identically from bash, zsh,
# PowerShell and cmd.

.PHONY: help setup pipeline ingest features train train-sklearn train-torch \
        serve test lint format docker-build docker-run drift clean deploy-info

PYTHON      ?= python

# Exported by Make itself, so it reaches the recipe regardless of shell.
export PYTHONPATH = src

help: ## Show available targets
	@$(PYTHON) scripts/devtools.py help

setup: ## Install dependencies
	$(PYTHON) -m pip install -r requirements-dev.txt

pipeline: ## Run the full pipeline (ingest -> features -> train -> report)
	$(PYTHON) -m credit_risk.pipeline --stage all

ingest: ## Download source data and load it into the warehouse
	$(PYTHON) -m credit_risk.pipeline --stage ingest

features: ## Build the feature table from SQL
	$(PYTHON) -m credit_risk.pipeline --stage features

train: ## Train both model flavours
	$(PYTHON) -m credit_risk.pipeline --stage train --flavor both

train-sklearn: ## Train only the scikit-learn model
	$(PYTHON) -m credit_risk.pipeline --stage train --flavor sklearn

train-torch: ## Train only the PyTorch model
	$(PYTHON) -m credit_risk.pipeline --stage train --flavor torch

drift: ## Compare served predictions against the training baseline
	$(PYTHON) -m credit_risk.pipeline --stage drift

# Human-readable console logs instead of the Cloud Logging JSON format.
# Set through Make's export rather than a shell prefix - see the note above.
serve: export CR_PLAIN_LOGS = 1
serve: ## Run the API locally on :8080 (docs at /docs)
	$(PYTHON) -m uvicorn credit_risk.serving.app:app --reload --port 8080

test: ## Run the test suite
	$(PYTHON) -m pytest -v -m "not gcp"

lint: ## Check formatting and lint rules
	$(PYTHON) -m ruff check src tests scripts
	$(PYTHON) -m ruff format --check src tests scripts

format: ## Apply formatting and safe lint fixes
	$(PYTHON) -m ruff format src tests scripts
	$(PYTHON) -m ruff check --fix src tests scripts

docker-build: ## Build the serving image
	docker build -f docker/Dockerfile -t credit-risk-api:local .

docker-run: ## Run the serving image locally on :8080
	docker run --rm -p 8080:8080 -e PORT=8080 credit-risk-api:local

deploy-info: ## Print the GCP deployment sequence
	@$(PYTHON) scripts/devtools.py deploy-info

clean: ## Remove generated data, models, reports and caches
	@$(PYTHON) scripts/devtools.py clean
