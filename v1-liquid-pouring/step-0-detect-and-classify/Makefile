# step-0-detect-and-classify — one entry point for everything.
#
# Run `make` on its own to see what is available.

SHELL := /bin/bash
.DEFAULT_GOAL := help

TRAIN ?= 800
VAL   ?= 150
TEST  ?= 300
NAME  ?= yolo11s_640_800
MODEL ?= yolo11s-seg.pt
FRACTION ?= 1.0
# 8 epochs of 800 pictures is about 18 minutes on the Mac (yolo11s-seg, imgsz 640).
EPOCHS ?= 8
# The custom CNN (detect_classify/custom_net.py), trained from scratch.
CUSTOM ?= custom_rgb
CUSTOM_EPOCHS ?= 40

py = pixi run python -u -m

.PHONY: help data preview train evaluate train-custom evaluate-custom oracle test lint

help: ## Show this help
	@echo ""
	@awk 'BEGIN {FS = ":.*##"} \
		/^##@/ { printf "\n  \033[1m%s\033[0m\n", substr($$0, 5) } \
		/^[a-zA-Z_.-]+:.*?##/ { printf "    \033[36m%-10s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
	@echo ""
	@echo "  Settings: TRAIN=$(TRAIN) VAL=$(VAL) TEST=$(TEST) pictures; NAME=$(NAME) MODEL=$(MODEL) FRACTION=$(FRACTION) EPOCHS=$(EPOCHS)"
	@echo "  For example: make train NAME=yolo11s_500 FRACTION=0.25"
	@echo ""

##@ Data

data: ## Render the training, validation and test pictures with their labels
	$(py) detect_classify.generate_data --split train --count $(TRAIN) --depth
	$(py) detect_classify.generate_data --split val --count $(VAL) --depth
	$(py) detect_classify.generate_data --split test --count $(TEST) --depth

preview: ## Draw the labels over a few training pictures, into data/preview/
	$(py) detect_classify.preview --split train --count 12

##@ Model

train: ## Train YOLO11-seg (settings in docs/results.md)
	$(py) detect_classify.train --model $(MODEL) --fraction $(FRACTION) --epochs $(EPOCHS) --name $(NAME)

evaluate: ## Grade the trained model on the test pictures: count, class, position
	$(py) detect_classify.evaluate --weights runs/$(NAME)/weights/best.pt

train-custom: ## Train the custom CNN on the same pictures (about 1 minute per epoch)
	$(py) detect_classify.train_custom --name $(CUSTOM) --epochs $(CUSTOM_EPOCHS)

evaluate-custom: ## Grade the custom CNN on the test pictures, the same way as YOLO
	$(py) detect_classify.evaluate --custom runs/$(CUSTOM)/weights/best.pt

oracle: ## Grade the locating maths alone, with the true outlines in place of a detector
	$(py) detect_classify.evaluate --oracle

##@ Project

test: ## Run the tests
	pixi run pytest -q

lint: ## Check code style
	pixi run ruff check detect_classify tests
	pixi run ruff format --check detect_classify tests
