# liquid-pouring-sim — one entry point for everything.
#
# Run `make` on its own to see what is available.

SHELL := /bin/bash
.DEFAULT_GOAL := help

SEED ?= 0

# MuJoCo Menagerie, pinned: the UR5e arm and the Robotiq 2F-85 gripper.
MENAGERIE_URL    := https://github.com/google-deepmind/mujoco_menagerie
MENAGERIE_COMMIT := 8161bba264d7fa7c99ca301e91e7fb44737676ad
MENAGERIE_DIR    := assets/menagerie

# AutoBio, pinned: the centrifuge tubes and the tube rack. Only the asset
# folders are checked out.
AUTOBIO_URL    := https://github.com/autobio-bench/AutoBio
AUTOBIO_COMMIT := 08504213356cbfe9a82bcd33a43cd282945a45df
AUTOBIO_DIR    := assets/autobio

# The CC0 "Lab Assets" pack by MilkAndBanana: test tubes, beakers, flasks.
GLASSWARE_URL    := https://opengameart.org/sites/default/files/lab_assets.zip
GLASSWARE_SHA256 := 5202bc6759e19c43f513b848e5cf2e28136f71a0eb9355f50b0c0ac4873dd515
GLASSWARE_DIR    := assets/glassware

py = pixi run python -u -m

.PHONY: help setup assets view render info test lint clean-assets

help: ## Show this help
	@echo ""
	@awk 'BEGIN {FS = ":.*##"} \
		/^##@/ { printf "\n  \033[1m%s\033[0m\n", substr($$0, 5) } \
		/^[a-zA-Z_.-]+:.*?##/ { printf "    \033[36m%-13s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
	@echo ""
	@echo "  Settings: SEED=$(SEED) (picks the liquid levels and which tube sits where)"
	@echo "  For example: make view SEED=3"
	@echo ""

##@ Scene

view: setup ## Open the scene in MuJoCo's window and let the liquid settle
	pixi run mjpython -m lab_scene.view --seed $(SEED)

render: setup ## Save pictures of the scene from every camera into figures/
	$(py) lab_scene.render --seed $(SEED) --out figures

info: setup ## Print what is in the scene: objects, sizes, weights, ball counts, speed
	$(py) lab_scene.info --seed $(SEED)

##@ Project

setup: assets ## Install the environment and download the models
	@pixi install

assets: $(MENAGERIE_DIR)/.done $(AUTOBIO_DIR)/.done $(GLASSWARE_DIR)/.done ## Download every model, pinned

$(MENAGERIE_DIR)/.done:
	rm -rf $(MENAGERIE_DIR)
	git clone --quiet --filter=blob:none --no-checkout $(MENAGERIE_URL) $(MENAGERIE_DIR)
	cd $(MENAGERIE_DIR) && git sparse-checkout set universal_robots_ur5e robotiq_2f85 \
		&& git checkout --quiet $(MENAGERIE_COMMIT)
	touch $@

$(AUTOBIO_DIR)/.done:
	rm -rf $(AUTOBIO_DIR)
	git clone --quiet --filter=blob:none --no-checkout $(AUTOBIO_URL) $(AUTOBIO_DIR)
	cd $(AUTOBIO_DIR) && git sparse-checkout set autobio/assets/container autobio/assets/rack \
		&& git checkout --quiet $(AUTOBIO_COMMIT)
	touch $@

$(GLASSWARE_DIR)/.done:
	rm -rf $(GLASSWARE_DIR)
	mkdir -p $(GLASSWARE_DIR)
	curl -sSL -o $(GLASSWARE_DIR)/lab_assets.zip $(GLASSWARE_URL)
	echo "$(GLASSWARE_SHA256)  $(GLASSWARE_DIR)/lab_assets.zip" | shasum -a 256 -c -
	unzip -q $(GLASSWARE_DIR)/lab_assets.zip -d $(GLASSWARE_DIR)
	rm $(GLASSWARE_DIR)/lab_assets.zip
	touch $@

test: setup ## Run the tests
	pixi run pytest -q

lint: setup ## Check code style
	pixi run ruff check lab_scene tests
	pixi run ruff format --check lab_scene tests

clean-assets: ## Delete the downloaded models and the converted meshes
	rm -rf assets
