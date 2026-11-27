# Step 6 Approach: The Whole Run

**Status: planned.** The overview is in [steps.md](../../docs/solutions/steps.md#step-6-the-whole-run).

## Goal

The whole job from a single instruction, *"Load the samples into the centrifuge safely"*, with **Gemini
Robotics-ER 2 as the coordinator**: it decides what to do next, calls the steps as functions, handles
failures, and reports.

## The functions Gemini can call

| Function | Step | Returns |
|---|---|---|
| `find_tubes()` | 0 | the list of tubes, the balance, the holes |
| `weigh(tube)` | 1 | grams |
| `plan_load(weights)` | 2 | holes, balance tubes and their target weights |
| `fill_balance_tube(tube, grams)` | 3 | the weight reached |
| `load(tube, hole)` | 4 | seated or not, with a rotor picture |
| `close_lid()` | 5 | latched or not |
| `look(camera)` | | a picture, to check anything |

Gemini sees the pictures and each function's answer, and decides the next call. This is the way Google
describes Gemini Robotics-ER being used: it plans and calls tools; the tools (here our steps, later a
model like π0.5) do the moving.

## What it should handle

- A tube that did not seat: take it out and try again, or try another hole that keeps the balance.
- A pour that went over: empty and refill the balance tube.
- A weighing that looks wrong (for example a sample lighter than an empty tube): weigh again.

## Compared with

A fixed Python coordinator calling the same functions in a set order. Both on the same 20 seeds: jobs
finished, balance within 1.0 g, time, number of calls, cost.

## The demo

Recorded from the front and rotor cameras, with what Gemini says at each decision shown on screen, ending
on the report: *"Balanced: 0.08 g out ✓"*.
