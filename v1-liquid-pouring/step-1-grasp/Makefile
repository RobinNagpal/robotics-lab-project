# step-1-grasp — one entry point for everything.
#
# Run `make` on its own to see what is available.

SHELL := /bin/bash
.DEFAULT_GOAL := help

# Scenes per split; each scene gives about 11 pictures, one per tube.
TRAIN ?= 400
VAL   ?= 50
TEST  ?= 100
NAME  ?= grconv
EPOCHS ?= 16
PHYSICS ?= 60

py = pixi run python -u -m

.PHONY: help data train evaluate figures data-side train-side evaluate-side figures-side data-side2 train-side2 evaluate-side2 figures-side2 test lint

help: ## Show this help
	@echo ""
	@awk 'BEGIN {FS = ":.*##"} \
		/^##@/ { printf "\n  \033[1m%s\033[0m\n", substr($$0, 5) } \
		/^[a-zA-Z0-9_.-]+:.*?##/ { printf "    \033[36m%-10s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
	@echo ""
	@echo "  Settings: TRAIN=$(TRAIN) VAL=$(VAL) TEST=$(TEST) scenes; NAME=$(NAME) EPOCHS=$(EPOCHS) PHYSICS=$(PHYSICS)"
	@echo ""

##@ Data

data: ## Render the close-up pictures and label every turn with the fast check
	$(py) grasp.generate_data --split train --count $(TRAIN)
	$(py) grasp.generate_data --split val --count $(VAL)
	$(py) grasp.generate_data --split test --count $(TEST)

##@ Model

train: ## Train the grasp network (settings in docs/top-down/approach.md)
	$(py) grasp.train --name $(NAME) --epochs $(EPOCHS)

evaluate: ## Grade the rule and the model: fast check on every test tube, physics on a sample
	$(py) grasp.evaluate --weights runs/$(NAME)/best.pt --physics $(PHYSICS)

figures: ## Draw the pictures used in the docs, into docs/top-down/images/
	$(py) grasp.figures --run runs/$(NAME)

##@ Side grasps (the second model, docs/side/)

data-side: ## Render the close-ups and label all 24 side directions, into data_side/
	$(py) grasp.generate_data --kind side --split train --count $(TRAIN)
	$(py) grasp.generate_data --kind side --split val --count $(VAL)
	$(py) grasp.generate_data --kind side --split test --count $(TEST)

train-side: ## Train the side-grasp network
	$(py) grasp.train --kind side --data data_side --name side_grconv --epochs $(EPOCHS)

evaluate-side: ## Grade the side-grasp network: fast check on every test tube, physics on a sample
	$(py) grasp.evaluate_side --weights runs/side_grconv/best.pt --physics $(PHYSICS)

figures-side: ## Draw the pictures used in the side docs, into docs/side/images/
	$(py) grasp.figures_side --run runs/side_grconv

##@ Side grasps, version 2 (fingers barely open, directions from the rack, docs/side/improved.md)

data-side2: ## Close-ups turned to the rack, all 24 directions labelled with fingers barely open, into data_side2/
	$(py) grasp.generate_data --kind side2 --split train --count $(TRAIN)
	$(py) grasp.generate_data --kind side2 --split val --count $(VAL)
	$(py) grasp.generate_data --kind side2 --split test --count $(TEST)

train-side2: ## Train the version 2 side-grasp network
	$(py) grasp.train --kind side2 --name side_v2 --epochs $(EPOCHS)

evaluate-side2: ## Grade version 2 (and the version 1 network with only the smoothing, for comparison)
	$(py) grasp.evaluate_side --weights runs/side_grconv/best.pt --smooth --physics 0 --out runs/side_grconv_smoothed
	$(py) grasp.evaluate_side --weights runs/side_v2/best.pt --data data_side2 --version 2 --physics $(PHYSICS)

figures-side2: ## Draw the pictures used in docs/side/improved.md, into docs/side/images/v2/
	$(py) grasp.figures_side --version 2 --run runs/side_v2 --data data_side2 --trial-row 24

##@ Project

test: ## Run the tests
	pixi run pytest -q

lint: ## Check code style
	pixi run ruff check grasp tests
	pixi run ruff format --check grasp tests
