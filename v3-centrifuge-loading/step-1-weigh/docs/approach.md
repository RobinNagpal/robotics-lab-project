# Step 1 Approach: Weigh Each Sample

**Status: part A (programmed) built and measured: 60 of 60 tubes weighed and put back on 20 seeds
([results.md](results.md)). Part B (π0.5) waits for a GPU.** The overview is in [steps.md](../../docs/solutions/steps.md#step-1-weigh-each-sample).

## Goal

For every sample tube found in Step 0: put it in the balance's holder, read its weight, and put it back in
its cell. Output: `{sample: grams}`, read from the balance's display.

## The moves, for one tube

```
 home ──► above the tube ──► down, fingers either side of the cap ──► close
      ──► straight up out of the rack ──► across to above the balance ──► down into the holder
      ──► open, move up clear ──► wait for a steady reading (changes < 0.01 g for 0.5 s) ──► read
      ──► down, close ──► up ──► back above its cell ──► down ──► open ──► up
```

The grip: from above, the pads on the **cap** (35 mm wide, 12 mm tall), the grip point 4 mm under the cap's
top; the fingers open to 45 mm to come down over it. This is the grip Step 4 needs, and the tube is gripped
the same way everywhere (checked against the whole job before building: [grip plan](../../grasp-plan/docs/results.md)). Gripping under the
cap, as first planned, works here but puts the fingertips 12 mm into the rotor's top in Step 4. Every rack
cell and the holder have clear finger turns (at least 6 of the 12 tried, with 5 mm or more to spare).

## A. Programmed (the baseline, and the teacher for B)

Every move is a straight line for the point between the finger pads, cut into 5 mm pieces, each solved for
joint angles (inverse kinematics) from the last, then followed with gentle speeding up and slowing down (as
v1's Step 1b). Before the arm moves, every piece is checked for collisions on a scratch copy of the scene: the
arm and the tube it holds must stay 2 mm from everything else, or the move is refused and nothing moves. The
gripper closes 2 mm past the cap's width (v1's 4 mm spun the round cap and jolted the simulation: see
results.md). The balance is read like a real balance's filtered display: readings averaged over 0.1 s, steady
when they agree within 0.01 g for 0.5 s.

| File | What |
|---|---|
| `motion.py` | the arm: straight-line moves, the collision check, the gripper, the `on_step` hook for recording |
| `weigh.py` | the task: choosing the fingers' turn, the moves for one tube, reading the balance |
| `evaluate.py` | the measurement on 20 seeds against the simulator's answers |

## B. A learned model: SmolVLA first, then π0.5

**Changed 2026-11-18:** SmolVLA (Hugging Face, 0.45B) is trained first: it is new to this project, small
enough for this Mac or a cheap GPU, and reads the same data as π0.5. π0.5 follows if SmolVLA shows the data
works (it needs a 48 GB GPU, about $1.86 an hour). The demonstrations are recorded:
[dataset.md](dataset.md).

The description below was written for π0.5; SmolVLA works the same way.

π0.5 is a vision-language-action model: it sees camera pictures, reads a sentence, and outputs the next
second of arm movement (a "chunk" of joint targets), over and over until the job is done.

1. **Demonstrations:** run the programmed version on many random seeds (different tubes, cells, weights)
   and record the overhead and wrist pictures, the joint angles, the gripper and the sentence, for example
   *"put the red sample tube in the balance holder"* and *"put the tube back in cell 2 of the sample rack"*.
   The programmed arm is the teacher: no human demonstrations needed.
2. **Format:** LeRobot format (the format openpi reads; AutoBio's centrifuge data uses it too).
3. **Fine-tune** π0.5 from Physical Intelligence's base weights with openpi (low-rank fine-tuning first:
   needs about 24 to 46 GB of GPU memory).
4. **Run** it in the scene: π0.5 drives the arm; the balance reading is taken the same way.

What we want to learn: how many demonstrations it needs, whether one model does both sentences (into the
holder, back to a cell), and whether it copes with cells it never saw in training.

## How it will be measured (`results.md`)

On 20 test seeds, for A and B: weighings that worked (tube in holder, steady reading, back in its cell);
weight read vs the tube's true weight (should be within 0.05 g); tubes knocked or dropped; time per tube.

## Open questions

- Needs a GPU (waiting on AWS credits). Until then, A only.
- Weigh all samples first and then load, or carry each tube from the balance straight to the rotor once the
  plan is known? First version: weigh all, then plan, then load (simpler, and the plan needs every weight).
