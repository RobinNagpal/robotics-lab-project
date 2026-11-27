# Step 5 Approach: Close the Lid and Report

**Status: planned.** The overview is in [steps.md](../../docs/solutions/steps.md#step-5-close-the-lid-and-report).

## Goal

The lid latched, and a report a person can trust: "Balanced: 0.08 g out ✓".

## Closing

The lid rests open (99°), held by its gas spring. The arm brings its closed gripper to the lid's front edge
from the open side and pushes it over and down, staying at right angles to the lid as it swings. In the
simulation a 20 N push shuts and latches it in about 1.3 s (the tests check this). The latch catches within
1.7° of shut. The arm then moves clear.

The wrist force sensor shows when the lid is shut (the push stops moving and the force rises).

## Reporting

- **The robot's own number:** the imbalance worked out from the weights it measured (Steps 1 and 3) and the
  holes it used. This is what a real robot could report.
- **The examiner's number:** the same from the simulator's true weights and where the tubes really are
  (`grading.grade`). The two should agree within a few hundredths of a gram.
- **A last look:** Gemini Robotics-ER 2 is shown the rotor picture (taken before the lid shuts) and the
  front picture, and asked: *"Is every hole in this list loaded, and is the lid shut?"*

## How it will be measured (`results.md`)

Lid latched; robot's number minus examiner's number; Gemini's check right or wrong (on runs where a tube is
deliberately left out).
