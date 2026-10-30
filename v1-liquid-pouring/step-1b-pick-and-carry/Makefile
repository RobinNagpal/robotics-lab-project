# step-1b-pick-and-carry — one entry point for everything.
#
# Run `make` on its own to see what is available. Needs Step 0's trained
# detector and Step 1's trained grasp network next door (their `make train`).

SHELL := /bin/bash
.DEFAULT_GOAL := help

SEED ?= 0
SIZE ?= 0.15
SEEDS ?= 10

py = pixi run python -u -m

.PHONY: help run evaluate figures test lint

help: ## Show this help
	@echo ""
	@awk 'BEGIN {FS = ":.*##"} \
		/^##@/ { printf "\n  \033[1m%s\033[0m\n", substr($$0, 5) } \
		/^[a-zA-Z_.-]+:.*?##/ { printf "    \033[36m%-10s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
	@echo ""
	@echo "  Settings: SEED=$(SEED) SIZE=$(SIZE) for one run; SEEDS=$(SEEDS) scenes for the evaluation"
	@echo ""

##@ Runs

run: ## One run: find the tubes, choose a pair, pick both, meet at the pouring point
	$(py) carry.episode --seed $(SEED) --size $(SIZE)

evaluate: ## Every size in SEEDS task scenes (30 runs by default, about 5 minutes)
	$(py) carry.evaluate --seeds $(SEEDS)

figures: ## Draw the pictures used in the docs, into docs/images/
	$(py) carry.figures

##@ Project

test: ## Run the tests
	pixi run pytest -q

lint: ## Check code style
	pixi run ruff check carry tests
	pixi run ruff format --check carry tests
