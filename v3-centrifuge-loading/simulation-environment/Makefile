# centrifuge-loading-sim — one entry point for everything.
#
# Run `make` on its own to see what is available.

SHELL := /bin/bash
.DEFAULT_GOAL := help

SEED ?= 0

# MuJoCo Menagerie, pinned (the same commit as v1): the UR5e arm and the Robotiq 2F-85 gripper.
MENAGERIE_URL    := https://github.com/google-deepmind/mujoco_menagerie
MENAGERIE_COMMIT := 8161bba264d7fa7c99ca301e91e7fb44737676ad
MENAGERIE_DIR    := assets/menagerie

# AutoBio, pinned (the same commit as v1 and v2): the Eppendorf 5430 centrifuge and the 50 ml tube.
# Only those files are checked out.
AUTOBIO_URL    := https://github.com/autobio-bench/AutoBio
AUTOBIO_COMMIT := 08504213356cbfe9a82bcd33a43cd282945a45df
AUTOBIO_DIR    := assets/autobio
AUTOBIO_PATHS  := /autobio/model/instrument/centrifuge_eppendorf_5430.xml \
                  /autobio/assets/instrument/centrifuge_eppendorf_5430/ \
                  /autobio/assets/container/centrifuge_50ml_screw_vis/

py = pixi run python -u -m

.PHONY: help setup assets view render info test lint clean-assets

help: ## Show this help
	@echo ""
	@awk 'BEGIN {FS = ":.*##"} \
		/^##@/ { printf "\n  \033[1m%s\033[0m\n", substr($$0, 5) } \
		/^[a-zA-Z_.-]+:.*?##/ { printf "    \033[36m%-13s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
	@echo ""
	@echo "  Settings: SEED=$(SEED) (picks the job: seed 0 pairs, 1 odd one, 2 triangle, 3 mismatch, 4 pairs ...)"
	@echo "  For example: make view SEED=1"
	@echo ""

##@ Scene

view: setup ## Open the scene in MuJoCo's window
	pixi run mjpython -m centrifuge_scene.view --seed $(SEED)

render: setup ## Save pictures from every camera into figures/ (and one with the lid shut)
	$(py) centrifuge_scene.render --seed $(SEED) --out figures --closed

info: setup ## Print the job, every tube's weight, and how fast the scene runs
	$(py) centrifuge_scene.info --seed $(SEED)

##@ Project

setup: assets ## Install the environment and download the models
	@pixi install

assets: $(MENAGERIE_DIR)/.done $(AUTOBIO_DIR)/.done ## Download every model, pinned

$(MENAGERIE_DIR)/.done:
	rm -rf $(MENAGERIE_DIR)
	git clone --quiet --filter=blob:none --no-checkout $(MENAGERIE_URL) $(MENAGERIE_DIR)
	cd $(MENAGERIE_DIR) && git sparse-checkout set universal_robots_ur5e robotiq_2f85 \
		&& git checkout --quiet $(MENAGERIE_COMMIT)
	touch $@

$(AUTOBIO_DIR)/.done:
	rm -rf $(AUTOBIO_DIR)
	git clone --quiet --filter=blob:none --no-checkout $(AUTOBIO_URL) $(AUTOBIO_DIR)
	cd $(AUTOBIO_DIR) && git sparse-checkout set --no-cone $(AUTOBIO_PATHS) \
		&& git checkout --quiet $(AUTOBIO_COMMIT)
	touch $@

test: setup ## Run the tests
	pixi run pytest -q

lint: setup ## Check code style
	pixi run ruff check centrifuge_scene tests
	pixi run ruff format --check centrifuge_scene tests

clean-assets: ## Delete the downloaded models and the meshes made from them
	rm -rf assets
