# Step 2 Approach: Plan the Load

**Status: part A (the solver) built and measured ([results.md](results.md)); part B not yet.** The overview is in [steps.md](../../docs/solutions/steps.md#step-2-plan-the-load).

## Goal

From the samples' weights (Step 1), decide:

- which hole (0 to 5) each tube goes in;
- whether balance tubes are needed, and for each, the weight it must be filled to;
- the order to load them.

So that the rotor is balanced: the tubes' outward pulls add up to at most 1.0 g
(see [the balance rule](../../simulation-environment/docs/simulation-environment.md#7-the-balance-rule)).

## The rules

- Samples are never changed (no water added to a sample!). Only balance tubes are filled.
- At most two balance tubes (there are two empty tubes), and at most five tubes in all.
- A balance tube gets a cap once filled (Step 3), so its weight includes the cap's 2 g: at least 13.65 g
  (an empty tube and its cap), at most that plus the 34.6 g of water an open tube holds at 35°.
- **Every tube needs an empty hole beside it when it goes in** (the gripper's finger links need the room:
  [grip plan](../../grasp-plan/docs/results.md)). So a full rotor cannot be loaded, and the order matters.
  Balance tubes (open, gripped at the top, with the least room) go in first, while both neighbours are empty.
- Fewest balance tubes first; among plans with the same number: balance tubes with both neighbours empty, then
  the lowest imbalance, then the fewest tubes going in beside a full hole.

## A. The solver (exact, code)

1. Try every placement of the samples alone: for n tubes, 6 x 5 x ... (n factors), at most 720. Compute
   the imbalance of each. If the best is within 1.0 g, that is the plan (the **pairs** and **triangle** jobs).
2. Otherwise try adding one balance tube, then two. A balance tube's weight is free (it will be filled to
   order), so for each placement the best weight is worked out directly: it must cancel the samples' pull,
   so it goes opposite the leftover pull with exactly that weight. Check it is between an empty tube's
   weight and a full one's.
3. For each placement, find a loading order (at most 5! = 120 to try) that keeps the rule above.
4. Output the plan in loading order, each balance tube's target weight, the imbalance, and the worst case if
   each balance tube ends up 0.5 g off (Step 3's limit).

Code: `solver.py` (`plan_load(weights)`), tests in `tests/test_solver.py`, measured by `evaluate.py`.

Examples of what it must find: an **odd one** with its balance tube opposite; three equal samples in holes
0, 2, 4 with no balance tube; a **mismatch** with two balance tubes.

## B. Gemini Robotics-ER 2 with the solver as a tool

Gemini gets the weights and the job description, and can call `plan_load(weights)` (the solver). It decides
which plan to use and **explains it in plain words**, for example:

> *"Samples 0 and 2 weigh 42.01 g and 42.25 g: they go opposite each other (holes 0 and 3). Sample 1
> (46.78 g) has no partner, so I will fill an empty tube to 46.78 g and put it opposite (holes 1 and 4)."*

The explanation is what the demo shows; the numbers come from the solver.

**Also measured:** Gemini with no solver, asked to plan from the weights alone. This shows how often a
language model gets the arithmetic wrong, which is why the tool is there.

## How it will be measured (`results.md`)

On the true weights of 100 seeds (25 per job): plans within 1.0 g; plans with the fewest balance tubes;
for Gemini without the tool, how far out of balance its plans are.
