# step-3b-put-back-act — one entry point for everything.
#
# The whole task with the learned pour (Step 2b's ACT) instead of the programmed one; the put-back is
# Step 3's code. Run `make` on its own to see what is available. Needs Step 0's detector, Step 1's grasp
# network, Step 2's level network and saved Step 1b runs (`make starts` there), and Step 2b's trained
# policies (`make train` and `make train-mlp` there).

SHELL := /bin/bash
.DEFAULT_GOAL := help

# The same 6 test scenes as Step 3.
STARTS ?= seed151_size150 seed152_size150 seed154_size150 seed150_size200 seed153_size200 seed156_size200
POLICY ?= act

py = pixi run python -u -m

.PHONY: help run test lint

help: ## Show this help
	@echo ""
	@awk 'BEGIN {FS = ":.*##"} \
		/^##@/ { printf "\n  \033[1m%s\033[0m\n", substr($$0, 5) } \
		/^[a-zA-Z_.-]+:.*?##/ { printf "    \033[36m%-10s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
	@echo ""
	@echo "  Settings: STARTS=\"$(STARTS)\", POLICY=$(POLICY) (act or mlp)"
	@echo ""

##@ Runs

run: ## The whole task on the 6 test scenes with POLICY pouring, 4 at a time, into runs/workflow/POLICY/
	$(py) putback_act.workflow --starts $(STARTS) --policy $(POLICY) --workers 4

##@ Project

test: ## Run the tests
	pixi run pytest -q

lint: ## Check code style
	pixi run ruff check putback_act tests
	pixi run ruff format --check putback_act tests
