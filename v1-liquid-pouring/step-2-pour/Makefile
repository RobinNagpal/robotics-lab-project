# step-2-pour — one entry point for everything.
#
# Run `make` on its own to see what is available. Needs Step 0's trained detector and Step 1's
# trained grasp network next door (their `make train`), because every pour starts from a Step 1b run.

SHELL := /bin/bash
.DEFAULT_GOAL := help

START ?= data/starts/seed100_size150.npy
TARGET ?= 50

py = pixi run python -u -m

.PHONY: help starts pour closed-loop data summary level-data rule train train-resnet evaluate test lint camera-study

help: ## Show this help
	@echo ""
	@awk 'BEGIN {FS = ":.*##"} \
		/^##@/ { printf "\n  \033[1m%s\033[0m\n", substr($$0, 5) } \
		/^[a-zA-Z_.-]+:.*?##/ { printf "    \033[36m%-10s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
	@echo ""
	@echo "  Settings: START=$(START) TARGET=$(TARGET) for one pour"
	@echo ""

##@ Data

starts: ## Run Step 1b on seeds 100-159, medium and large, and save where each ends (about 65 minutes)
	$(py) pour.start --first 100 --seeds 60 --out data/starts

data: ## Two programmed pours from every start, with the pour camera's pictures (about 20 minutes)
	$(py) pour.data --starts data/starts --out data/pours

summary: ## What is in the data, and how the programmed pours did, into docs/images/
	$(py) pour.summary --pours data/pours --out docs/images

##@ Level network

level-data: ## Shrink and cache the pours' pictures for training (under a minute)
	$(py) level.data

rule: ## Fit the blue-pixel rule on the training pictures
	$(py) level.rule

train: ## Train the custom CNN (about 15 minutes on the Mac's GPU)
	$(py) level.train --model custom --name custom --minutes 15

train-resnet: ## Train the pretrained ResNet-18 to compare, same data and time (about 15 minutes)
	$(py) level.train --model resnet18 --name resnet18 --minutes 15 --lr 3e-4

evaluate: ## Grade the rule and the networks on the test pictures
	$(py) level.evaluate --runs custom resnet18

##@ Runs

pour: ## One programmed pour from START to TARGET %
	$(py) pour.pour $(START) --target $(TARGET)

closed-loop: ## Pour to 50 % on the 20 test starts: true count vs the CNN's reading (about 12 minutes)
	$(py) pour.closed_loop --workers 4

##@ Project

test: ## Run the tests
	pixi run pytest -q

lint: ## Check code style
	pixi run ruff check pour level camera_study tests
	pixi run ruff format --check pour level camera_study tests

##@ Camera study (docs/solutions/step-2-pour-camera.md)

cam = PYTHONPATH=$$PYTHONPATH:camera_study python -u camera_study

camera-study: ## Which fixed camera sees the receiver's level: 20 Step 1b runs, then 576 places (about 15 minutes)
	pixi run bash -c '$(cam)/record.py --out data/camera_study/runs'
	pixi run bash -c '$(cam)/analyse.py --runs data/camera_study/runs --out data/camera_study/out'
	pixi run bash -c '$(cam)/plot.py data/camera_study/out'
