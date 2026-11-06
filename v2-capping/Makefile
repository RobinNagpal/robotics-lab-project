# v2-capping — one entry point for everything.
#
# Screw caps on the receivers, on top of the v1 lab scene. Run `make` on its own to see what is
# available. Needs the v1 scene's downloaded models first: `make setup` in
# ../v1-liquid-pouring/simulation-environment.

SHELL := /bin/bash
.DEFAULT_GOAL := help

SEED ?= 150
TUBE ?= receiver_150

# AutoBio, pinned (the commit v1 uses for its racks): its screw-cap lab tubes and their matching caps
# (1.5 ml, 15 ml and 50 ml centrifuge tubes, 1.8 ml cryovial). Kept for later; not used by the scene yet.
# Only the tube meshes and their MJCF files are checked out, into assets/ (not committed: no licence file).
AUTOBIO_URL    := https://github.com/autobio-bench/AutoBio
AUTOBIO_COMMIT := 08504213356cbfe9a82bcd33a43cd282945a45df
AUTOBIO_DIR    := assets/autobio

py = pixi run python -u -m

.PHONY: help view render cap-trial uncap uncap-eval uncap-stress grip-test teacher-rec act-train act-one act-eval act-figures autobio-tubes assets test lint

help: ## Show this help
	@echo ""
	@awk 'BEGIN {FS = ":.*##"} \
		/^##@/ { printf "\n  \033[1m%s\033[0m\n", substr($$0, 5) } \
		/^[a-zA-Z_.-]+:.*?##/ { printf "    \033[36m%-14s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
	@echo ""
	@echo "  Settings: SEED=$(SEED), TUBE=$(TUBE) (receiver_100, receiver_150 or receiver_200)"
	@echo ""

##@ Scene

view: ## Open the capped scene in MuJoCo's window
	pixi run mjpython -m capping.view --seed $(SEED)

render: ## Save pictures of the capped scene from every camera into figures/
	$(py) capping.render --seed $(SEED) --out figures

##@ Trial

cap-trial: ## The cap with the real arm and gripper: pull, then quarter turns until it is off (about 30 s)
	$(py) capping.cap_trial --seed $(SEED) --tube $(TUBE)

##@ Uncapping (programmed baseline)

SIZE ?= 0.2
SEEDS ?= 150-159
SIZES ?= 0.2
WORKERS ?= 4

uncap: ## One uncapping run: Arm A holds a capped tube, Arm B unscrews it (about 45 s). SEED=, SIZE=
	$(py) uncap.episode --seed $(SEED) --size $(SIZE)

uncap-eval: ## Uncapping on many scenes (200 mm: 10 runs on 150-159 about 3 min). SEEDS=150-159 SIZES="0.15 0.2"
	$(py) uncap.evaluate --seeds $(SEEDS) --sizes $(SIZES) --workers $(WORKERS)

uncap-stress: ## 200 mm tube, new scenes 200-229, random tube placement and cap tightness (about 7 min)
	$(py) uncap.evaluate --seeds 200-229 --sizes 0.2 --vary --workers $(WORKERS) --out runs/stress

grip-test: ## How much twist Arm A's grip on a capped tube holds, for each grip setting (about 1 min)
	$(py) uncap.grip_test --seed $(SEED) --size $(SIZE) --depth 0.0

##@ Uncapping (learned: ACT unscrews, docs/uncapping-learning.md)

RUN ?= runs/learn/act_A

teacher-rec: ## The teacher on stress scenes 200-229, recorded, to measure against (about 4 min)
	$(py) uncap.evaluate --seeds 200-229 --sizes 0.2 --vary --workers $(WORKERS) --record runs/teacher_A_rec --out runs/teacher_A_rec_eval

act-train: ## Train ACT on the recorded runs in runs/demos (20 min on the Mac's GPU)
	$(py) learn.train --data runs/demos --name act_A --minutes 20

act-one: ## One uncapping run with the learned unscrew (about 45 s). RUN=, SEED= (a stress scene: 200-229)
	$(py) learn.rollout --run $(RUN) --seed $(SEED)

act-eval: ## The learned unscrew on stress scenes 200-229, beside the teacher (about 10 min)
	$(py) learn.evaluate --run $(RUN) --seeds 200-229 --workers $(WORKERS)

act-figures: ## Pictures and animations of one full run with ACT unscrewing (about 1 min). RUN=, SEED=203
	$(py) uncap.figures --act $(RUN) --seed $(if $(filter 150,$(SEED)),203,$(SEED)) --no-grip --out docs/images/uncapping-learned

##@ AutoBio tubes (for later)

autobio-tubes: assets ## The four AutoBio tubes with their caps, unscrewed on our thread: pictures into figures/
	$(py) capping.autobio_tubes --root $(AUTOBIO_DIR)/autobio/assets/container --out figures

##@ Project

assets: $(AUTOBIO_DIR)/.done ## Download the AutoBio tubes and caps, pinned

$(AUTOBIO_DIR)/.done:
	rm -rf $(AUTOBIO_DIR)
	git clone --quiet --filter=blob:none --no-checkout $(AUTOBIO_URL) $(AUTOBIO_DIR)
	cd $(AUTOBIO_DIR) && git sparse-checkout set --no-cone \
		'autobio/assets/container/centrifuge_1-5ml_screw_vis/' \
		'autobio/assets/container/centrifuge_15ml_screw_vis/' \
		'autobio/assets/container/centrifuge_50ml_screw_vis/' \
		'autobio/assets/container/cryovial_1-8ml_vis/' \
		'autobio/model/object/centrifuge_1-5ml_screw.xml' \
		'autobio/model/object/centrifuge_15ml_screw.xml' \
		'autobio/model/object/centrifuge_50ml_screw.xml' \
		'autobio/model/object/cryovial_1-8ml.xml' \
		&& git checkout --quiet $(AUTOBIO_COMMIT)
	touch $@

test: ## Run the tests
	pixi run pytest -q

lint: ## Check code style
	pixi run ruff check capping uncap learn tests
	pixi run ruff format --check capping uncap learn tests
