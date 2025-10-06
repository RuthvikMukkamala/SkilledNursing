.PHONY: install test lint format clean data pipeline dashboard help

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install package and dependencies
	pip install -e ".[dev,analysis]"

install-prod:  ## Install production dependencies only
	pip install -e .

test:  ## Run tests with coverage
	pytest tests/ --cov=src/snf_reit_analysis --cov-report=html --cov-report=term-missing

test-fast:  ## Run tests without coverage
	pytest tests/ -v

lint:  ## Run linting checks
	ruff check src/ tests/
	mypy src/

format:  ## Format code with black and ruff
	black src/ tests/
	ruff check --fix src/ tests/

clean:  ## Clean build artifacts and cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf .pytest_cache .coverage htmlcov/ dist/ build/

clean-data:  ## Clean all data directories (WARNING: deletes all data)
	@echo "WARNING: This will delete all data files!"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf data/raw/* data/interim/* data/processed/* data/external/*; \
		echo "Data directories cleaned"; \
	fi

data-cms:  ## Fetch CMS data only
	python -m snf_reit_analysis.pipelines.etl --source cms

data-bls:  ## Fetch BLS data only
	python -m snf_reit_analysis.pipelines.etl --source bls

data-sec:  ## Fetch SEC data only
	python -m snf_reit_analysis.pipelines.etl --source sec

data:  ## Fetch all data sources
	python -m snf_reit_analysis.pipelines.etl --source all

pipeline:  ## Run full ETL pipeline with all sources
	snf-pipeline --source all

dashboard:  ## Launch Streamlit dashboard
	streamlit run src/snf_reit_analysis/dashboard.py

dev:  ## Run dashboard in development mode with auto-reload
	streamlit run src/snf_reit_analysis/dashboard.py --server.runOnSave true

setup:  ## Initial project setup
	@echo "Setting up SNF REIT Analysis Platform..."
	@if [ ! -f .env ]; then cp .env.example .env; echo "Created .env file - please update with your API keys"; fi
	@mkdir -p data/{raw,interim,processed,external} models/{production,experiments} reports/{figures,outputs}
	@touch data/raw/.gitkeep data/interim/.gitkeep data/processed/.gitkeep data/external/.gitkeep
	@touch models/production/.gitkeep models/experiments/.gitkeep
	@touch reports/figures/.gitkeep reports/outputs/.gitkeep
	@echo "Setup complete! Next steps:"
	@echo "1. Edit .env and add your BLS_API_KEY and SEC_USER_AGENT"
	@echo "2. Run 'make install' to install dependencies"
	@echo "3. Run 'make data' to fetch initial data"
	@echo "4. Run 'make dashboard' to launch the dashboard"

check:  ## Run all quality checks
	@echo "Running format check..."
	@black --check src/ tests/
	@echo "Running lint..."
	@ruff check src/ tests/
	@echo "Running type check..."
	@mypy src/
	@echo "Running tests..."
	@pytest tests/ -v
	@echo "All checks passed!"
