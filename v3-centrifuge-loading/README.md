# v3: Loading a Centrifuge Safely

One robot arm weighs sample tubes, works out how to place them so a centrifuge's rotor is balanced, fills a
balance tube with water when a sample has no partner, loads the tubes into the rotor's slanted holes and
closes the lid. It uses two AI models from the big robotics labs: **Google's Gemini Robotics-ER 2** to see,
decide and check, and **Physical Intelligence's π0.5** to move the arm.

![The scene](simulation-environment/images/front.png)

**Status (2026-11-23):** the whole job runs end to end, programmed (no learned model, no Gemini yet): on 20
scenes every tube is found (SAM 3), weighed, planned, filled, loaded and the rotor balanced, and the robot's own
report agrees with the examiner ([Step 6 results](step-6-whole-run/docs/results.md)). The lid is left open: as
the centrifuge stands, the arm cannot reach to push it shut ([lid study](step-5-close-and-report/docs/lid-reach.md)).
Gemini waits for its quota; SmolVLA had one short training run for Step 1. Day-to-day state:
[docs/where-we-are.md](docs/where-we-are.md).

## Where to read

| Page | What it covers |
|---|---|
| [Problem statement](docs/problem-statement/problem-statement.md) | the task, the cameras (where each stands, what it sees, why not just overhead), what the arm knows, what counts as success, why we picked it |
| [Steps](docs/solutions/steps.md) | the whole job step by step, and how each step will be built |
| [Models](docs/libraries/models.md) | which AI model is used at which step, and why; open decisions (GPU, credits) |
| [Simulation environment](simulation-environment/README.md) | the scene, how to run it, the four jobs |

## Folders

| Folder | What |
|---|---|
| `simulation-environment/` | the MuJoCo scene (code, tests, docs) |
| `grasp-plan/` | every grip checked against every move of the job, before any step is built |
| `step-0-find/` | find which cells hold which tubes, from three cameras (Gemini Robotics-ER 2) |
| `step-1-weigh/` | weigh each sample (programmed; π0.5) |
| `step-2-plan/` | plan the load (solver; Gemini Robotics-ER 2 with the solver as a tool) |
| `step-3-fill-balance-tube/` | pour water into a balance tube, watching the balance |
| `step-4-load-rotor/` | load each tube into its hole, guided by the wrist force sensor |
| `step-5-close-and-report/` | close the lid, report the balance |
| `step-6-whole-run/` | the whole job, with Gemini Robotics-ER 2 as the coordinator |

Each step folder holds its code and a `docs/` folder with `approach.md` (how it is done) and, once built,
`results.md` (how well it works).

## Quick start

```bash
cd simulation-environment
make setup     # once: installs the environment and downloads the models
make view      # opens the scene (SEED=1 for another job)
make test
```
