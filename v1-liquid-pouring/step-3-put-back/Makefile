# step-3-put-back — one entry point for everything.
#
# Run `make` on its own to see what is available. Needs Step 0's detector, Step 1's grasp network,
# Step 2's level network (their `make train`) and Step 2's saved Step 1b runs (`make starts` there).

SHELL := /bin/bash
.DEFAULT_GOAL := help

STARTS ?= seed151_size150 seed152_size150 seed154_size150 seed150_size200 seed153_size200 seed156_size200
START ?= seed150_size200

py = pixi run python -u -m

.PHONY: help run one figures test lint

help: ## Show this help
	@echo ""
	@awk 'BEGIN {FS = ":.*##"} \
		/^##@/ { printf "\n  \033[1m%s\033[0m\n", substr($$0, 5) } \
		/^[a-zA-Z_.-]+:.*?##/ { printf "    \033[36m%-10s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
	@echo ""
	@echo "  Settings: STARTS=\"$(STARTS)\", START=$(START)"
	@echo ""

##@ Runs

run: ## The whole task on the 6 test scenes, 4 at a time, into runs/workflow/ (about 8 minutes)
	$(py) putback.workflow --starts $(STARTS) --workers 4

figures: ## Pictures of one whole run (START) for the docs (about 3 minutes)
	$(py) putback.figures --start $(START) --out ../docs/solutions/images/step-3-put-back

##@ Project

test: ## Run the tests
	pixi run pytest -q

lint: ## Check code style
	pixi run ruff check putback tests
	pixi run ruff format --check putback tests
