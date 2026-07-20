# Credit default risk pipeline.
# `make help` lists the targets.

.PHONY: help setup pipeline ingest features train train-sklearn train-torch \
        serve test lint format docker-build docker-run drift clean deploy-info

PYTHON      ?= python
export PYTHONPATH = src

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

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

serve: ## Run the API locally on :8080
	CR_PLAIN_LOGS=1 $(PYTHON) -m uvicorn credit_risk.serving.app:app --reload --port 8080

test: ## Run the test suite
	$(PYTHON) -m pytest -v -m "not gcp"

lint: ## Check formatting and lint rules
	ruff check src tests
	ruff format --check src tests

format: ## Apply formatting and safe lint fixes
	ruff format src tests
	ruff check --fix src tests

docker-build: ## Build the serving image
	docker build -f docker/Dockerfile -t credit-risk-api:local .

docker-run: ## Run the serving image locally on :8080
	docker run --rm -p 8080:8080 -e PORT=8080 credit-risk-api:local

deploy-info: ## Print the GCP deployment sequence
	@echo "1. cd terraform && cp terraform.tfvars.example terraform.tfvars   # fill in project_id"
	@echo "2. terraform init && terraform apply"
	@echo "3. gcloud auth configure-docker \$$(terraform output -raw artifact_registry | cut -d/ -f1)"
	@echo "4. docker build -f docker/Dockerfile -t \$$(terraform output -raw artifact_registry)/api:v1 ."
	@echo "5. docker push \$$(terraform output -raw artifact_registry)/api:v1"
	@echo "6. terraform apply   # rolls the service onto the new image"
	@echo "7. curl \$$(terraform output -raw service_url)/health"

clean: ## Remove generated data, models and reports
	rm -rf data/raw/*.parquet data/processed/* data/models/* reports/*.md reports/*.json
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache
