# botalpaca — developer shortcuts.
#
# Windows users: `make` works in Git Bash / WSL. Every target also has a plain
# equivalent documented next to it.

PY := .venv/Scripts/python.exe
ifeq ($(OS),Windows_NT)
PY := .venv/Scripts/python.exe
else
PY := .venv/bin/python
endif

.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------- setup
.PHONY: install
install: ## Create the venv and install the project with dev extras
	python -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev]"

.PHONY: env
env: ## Create .env from .env.example if it does not exist
	@if not exist .env (copy .env.example .env && echo "created .env - fill in your secrets") \
		else (echo ".env already exists, leaving it alone")

# ---------------------------------------------------------------- quality
.PHONY: test
test: ## Run the whole test suite
	$(PY) -m pytest

.PHONY: test-cov
test-cov: ## Run the test suite with a coverage report
	$(PY) -m pytest --cov=botalpaca --cov-report=term-missing

.PHONY: lint
lint: ## Lint with ruff
	$(PY) -m ruff check botalpaca tests

.PHONY: format
format: ## Auto-fix lint findings with ruff
	$(PY) -m ruff check --fix botalpaca tests

.PHONY: types
types: ## Static type check with mypy
	$(PY) -m mypy botalpaca

# ---------------------------------------------------------------- run
.PHONY: check
check: ## Validate configuration, database and Alpaca credentials, then exit
	$(PY) -m botalpaca --check

.PHONY: run
run: ## Run the bot (Telegram + background scheduler)
	$(PY) -m botalpaca

.PHONY: run-debug
run-debug: ## Run the bot with DEBUG logging
	$(PY) -m botalpaca --log-level DEBUG

# ---------------------------------------------------------------- database
.PHONY: migrate
migrate: ## Apply all Alembic migrations
	$(PY) -m alembic upgrade head

.PHONY: migrate-down
migrate-down: ## Roll the last migration back
	$(PY) -m alembic downgrade -1

.PHONY: migrate-check
migrate-check: ## Verify the models and the migrations are in sync
	$(PY) -m alembic check

.PHONY: revision
revision: ## Autogenerate a migration: make revision m="add column x"
	$(PY) -m alembic revision --autogenerate -m "$(m)"

# ---------------------------------------------------------------- containers
.PHONY: build
build: ## Build the Docker image
	docker build -t botalpaca:local .

.PHONY: up
up: ## Start the bot with docker compose
	docker compose up -d --build

.PHONY: down
down: ## Stop the stack (keeps the data volume)
	docker compose down

.PHONY: logs
logs: ## Follow the container logs
	docker compose logs -f

# ---------------------------------------------------------------- fly
.PHONY: fly-deploy
fly-deploy: ## Deploy to Fly.io (create the volume first, once)
	fly deploy

.PHONY: fly-vol
fly-vol: ## Create the persistent volume (run once, before the first deploy)
	fly volumes create botalpaca_data --size 1 --region iad

.PHONY: fly-logs
fly-logs: ## Tail the Fly.io logs
	fly logs --app botalpaca

.PHONY: fly-secrets
fly-secrets: ## Set secrets interactively (never committed)
	fly secrets set --app botalpaca
