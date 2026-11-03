# step-2b-pour-act — one entry point for everything.
#
# Run `make` on its own to see what is available. Needs Step 2 next door: its saved starts
# (step-2-pour: make starts) and its trained level network (step-2-pour: make train). Both are only read.

SHELL := /bin/bash
.DEFAULT_GOAL := help

START ?= ../step-2-pour/data/starts/seed150_size200.npy
TARGET ?= 50
POLICY ?= act
WORKERS ?= 4

py = pixi run python -u -m

.PHONY: help demos train train-mlp analyse figures evaluate closed-loop-figures pour test lint

help: ## Show this help
	@echo ""
	@awk 'BEGIN {FS = ":.*##"} \
		/^##@/ { printf "\n  \033[1m%s\033[0m\n", substr($$0, 5) } \
		/^[a-zA-Z_.-]+:.*?##/ { printf "    \033[36m%-10s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
	@echo ""
	@echo "  Settings: START=$(START) TARGET=$(TARGET) POLICY=$(POLICY) WORKERS=$(WORKERS)"
	@echo ""

##@ Data

demos: ## Record the programmed pour twice from each of Step 2's 118 starts (about 50 minutes, 4 workers)
	$(py) demos.collect --workers $(WORKERS)

##@ Policies

train: ## Train ACT (30 minutes on the Mac's GPU)
	$(py) act.train --model act --name act --minutes 30

train-mlp: ## Train the CNN + MLP policy, same data and time (30 minutes)
	$(py) act.train --model mlp --name mlp --minutes 30

analyse: ## Grade both policies tick by tick on the validation and test pours (offline; about 2 minutes)
	$(py) act.analyse --runs act mlp

figures: ## The demonstrations' pictures and the diagrams for the docs
	$(py) demos.figures
	$(py) act.diagrams

##@ Closed loop (built, not yet run: the policy drives Arm B)

evaluate: ## Pour from the 20 test starts with every method, to 50 % (about 25 minutes, 4 workers)
	$(py) act.evaluate --methods programmed programmed-cnn act mlp --workers $(WORKERS)

closed-loop-figures: ## Pictures of the closed-loop pours, after evaluate (--gif runs one more pour)
	$(py) act.figures --gif

##@ Runs

pour: ## One learned pour from START to TARGET % with POLICY
	$(py) act.rollout $(START) --policy $(POLICY) --target $(TARGET)

##@ Project

test: ## Run the tests
	pixi run pytest -q

lint: ## Check code style
	pixi run ruff check demos act tests
	pixi run ruff format --check demos act tests
