##@ Utility
.PHONY: help
help:  ## Display this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make <target>\033[36m\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)


.PHONY: install
install:  ## Install the virtual environment and install the pre-commit hooks
	@echo "🐱‍👤 Creating virtual environment using uv"
	@uv sync
	@uv run pre-commit install

.PHONY: check
check: ## Run code quality tools
	@echo "🐱‍👤 Checking lock file consistency with 'pyproject.toml'"
	@uv lock --locked
	@echo "🐱‍👤 Linting code: Running pre-commit"
	@uv run pre-commit run -a

.PHONY: test
test: ## Fast tests in parallel, then slow tests serially (-n0)
	@echo "🐱‍👤 Testing code: Running pytest"
	@uv run pytest
	@echo "🐱‍👤 Running slow tests serially (-n0)"
	@uv run pytest -n0 -m 'slow and not llm' -o addopts='--strict-markers'

.PHONY: test-cases
test-cases: ## Run agentic case tests (unit + functional, scripted LLM)
	@echo "🐱‍👤 Running agentic case tests"
	@uv run pytest tests/unit/test_agentic/test_cases tests/functional/test_agentic/test_cases
	@echo "🐱‍👤 Running slow agentic case tests serially (-n0)"
	@uv run pytest tests/unit/test_agentic/test_cases tests/functional/test_agentic/test_cases -n0 -m 'slow and not llm' -o addopts='--strict-markers'

.PHONY: test-llm
test-llm: ## Run @pytest.mark.llm tests (loads repo .env; needs JARL_LLM_PROVIDER)
	@echo "🐱‍👤 Running LLM eval tests (loading .env if present)"
	@uv run --env-file .env pytest -m llm -o addopts='--strict-markers -n3'

##@ App
APP_ENTRY := src/jarl/app/app.py
# Bracket so pgrep/pkill do not match this Makefile recipe.
APP_PATTERN := [s]treamlit run $(APP_ENTRY)

.PHONY: app
app: app-stop ## Start Runner Lab (stops a leftover instance first)
	@echo "🐱‍👤 Starting Runner Lab"
	@uv run streamlit run $(APP_ENTRY)

.PHONY: app-stop
app-stop: ## Stop Runner Lab Streamlit
	@pids=$$(pgrep -f "$(APP_PATTERN)" || true); \
	if [ -n "$$pids" ]; then \
		kill $$pids; \
		echo "🛑 Runner Lab parado"; \
	else \
		echo "No había Runner Lab activo"; \
	fi

# .PHONY: docs
# docs: ## Build docs locally for live preview
# 	@echo "🐱‍👤 Building docs locally for live preview"
# 	@uv run zensical serve --strict --livereload

.DEFAULT_GOAL := help
