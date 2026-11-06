# Uncapping, learned: the details

The exact settings and numbers behind [uncapping-learning.md](uncapping-learning.md), and the history of the
decisions. Read that page first.

## 1. The recorded runs (the teacher)

`uncap/record.py` records Arm B 50 times a second while the program unscrews: what it senses, what the
program commands, and labels for analysis only (the part of the loop, the quarter, and from the
simulator how far the cap has turned and whether it is on its thread). Values are saved clean.

- 200 runs, scenes 300 to 499, `--vary`: each tube up to 2 mm from the middle of its cell and turned at
  random; the cap 0.45 to 1.25 N m tight. 193 right (the 7 others are listed in
  [uncapping-programmed.md](uncapping-programmed.md#4-how-sure-we-are)). 87 MB in `runs/demos/`.
- What one run spends its time on (seconds, mean): finding the cap and coming down 11.7, grip 0.6, turns
  7.5, lift tests 5.2, pressing down after a test 2.6, letting go 3.9, turning back 3.9, gripping again 3.9,
  lifting away 1.8.
- Lift-test pull, all 1,437 tests (read from the finger contacts, as the program does): 9.4 to 15.5 N
  with the cap on, -0.02 to 0.01 N with it off. The wrist sensor agrees (9.5 to 15.3 N on, 0 off, 437 tests).
- Wrist twist, peak per quarter: first quarter (the seal) 0.47 to 1.28 N m; later quarters 0.08 to 0.22.

## 2. What the data study found (2026-11-05)

1. The cap can't come off before quarter 6: 1.5 turns of thread, at most a quarter turn each (the fingers
   can slip, not gain). In every run it came off in quarter 7 or 8.
2. "Off" can't be felt while turning: on the thread or off, the turn differs by only 0.4 N and 0.14 N m,
   under a real sensor's noise.
3. The pause matters, not the test itself: while the arm holds still for a test, the fingers' wound-up
   twist turns the cap a few degrees more. Skip the test without pausing and the cap needs one more quarter.
4. Letting go and gripping again after every quarter is the biggest cost: 11.6 s of each unscrewing.

**Teacher B** (`--test-from 6 --pause 0.7`: lift tests from quarter 6 on, a 0.7 s still pause before
that), on the 30 test scenes: 30 of 30, the same quarters as the program in every scene, 24.3 s from gripping
to "off" (the program 26.3 s), 2.3 lift tests (the program 7.3). Not used for ACT, see
[the main page](uncapping-learning.md#7-why-not-teach-it-to-test-less-yet).

A bug found on the way: Arm B set every cap down about 2 cm too low onto the holder's post (a sign error
in `uncap/episode.py`, fixed). It happens after the unscrewing, so the recorded runs are unaffected.

## 3. The examples (`learn/data.py`)

| Setting | Value |
|---|---|
| Part of each run used | from the first quarter turn to the end of the recording (cap lifted 6 cm) |
| Input moments | 10, every 5th sample (0.1 s apart), the newest now; before the run's start, its first sample repeats |
| Values per moment | 25 (joint angles 6, speeds 6, gripper opening 1, wrist force 3, wrist twist 3, motor torques 6) |
| Wrist zeroing | the mean reading over the first 0.5 s of the recording (the arm moving in free air) |
| Predicted | 25 future samples (0.5 s): 6 joint targets, the gripper command, "done" |
| "done" | 1 for the run's last 5 samples (0.1 s) and after its end (the last command repeats) |
| Scaling | each value scaled by the training runs' mean and spread; a spread is never below its noise |
| Split | training scenes 300-469 (165 runs, 236,663 moments); checking 470-499 (28 runs, 39,824) |

Noise, new for every example, never stored:

| Value | Wobble per reading (1 sigma) | Drift per run (1 sigma) |
|---|---|---|
| wrist force | 1.5 N | 1.0 N |
| wrist twist | 0.1 N m | 0.05 N m |
| motor torques | 0.2 N m | |
| joint speeds | 0.005 rad/s | |
| gripper opening | 0.002 | |
| joint angles | 0.0001 rad | |

## 4. The model and training (`learn/model.py`, `learn/train.py`)

| Setting | Value |
|---|---|
| Tokens | the style (1) + the 10 moments, each projected to 256 numbers, plus a learned position each |
| Transformer | 4 encoder and 4 decoder layers, 8 heads, 1024 wide inside, dropout 0.1 |
| Style (CVAE) | 32 numbers, from a 2-layer encoder over [summary, newest moment, the teacher's 25 moves]; zero when running |
| Loss | L1 on the scaled moves + 0.5 x cross-entropy on "done" + 10 x the style penalty (KL) |
| Weights | 9.01 million |
| Optimiser | AdamW, learning rate 1e-4, weight decay 1e-4, warm-up 300 steps, then down to 0 by the clock |
| Batch, time | 64 examples, 20 minutes on the Mac's GPU (MPS), 22,161 steps |
| Kept | the weights with the smallest checking error (step 14,000) |

| Step | Minutes | Training error | Checking error | Next joint targets (°) | Gripper | done right |
|---|---|---|---|---|---|---|
| 1000 | 1.0 | 0.169 | 0.0404 | 0.41 | 1.37 | 99.85 % |
| 3000 | 2.7 | 0.035 | 0.0263 | 0.21 | 0.59 | 99.93 % |
| 8000 | 7.3 | 0.020 | 0.0191 | 0.18 | 0.71 | 99.96 % |
| 14000 | 12.7 | 0.016 | 0.0168 | 0.20 | 0.83 | 99.98 % |
| 22161 | 20.0 | 0.013 | 0.0176 | 0.21 | 0.86 | 99.98 % |

The training error falls on; the checking error stops falling after about 12 minutes. (The training
error starts higher than the checking error because training adds dropout and samples the style.)

## 5. Running it in the full task (`learn/rollout.py`)

- The program runs as usual until Arm B has gripped the cap (`Uncapper.unscrew` is replaced; nothing
  else changes). The model reads the same recorder the data came from, with the same noise.
- Every 0.02 s: predict the next 25 moves; the command for now is the weighted mean of every prediction
  made for now (weight exp(-0.01 x age), oldest most). Between samples, the joint targets slide to the next
  command, as the program's do.
- "done" above 0.5 for 3 samples in a row: the program carries the cap to the holder (it skips its own
  6 cm lift: `put_cap_away(lifted=True)`). After 60 s without "done", the run stops as failed.
- The program's path checks still guard the moves after the handover: with an untrained model, which said
  "done" at once, the program refused to move ("arm1 would touch receiver_200_thread").

## 6. Measuring (`learn/measure.py`)

Both the program and ACT are measured from their recordings, from the first turn to the end of the
recording (the cap lifted 6 cm). A **grip** is a stretch with the gripper command closed (above 170; about
199 closed, 138 open). A **lift test** is a grip during which the hand rises more than 3 mm (the test lifts
6 mm; a turn raises it only 0.8 mm). The hand's height comes from the commanded joint targets through the
arm's kinematics. The program's own clock ("gripped" to "cap off") gives 26.3 s; this measure adds the lift
away: 28.1 s.

## 7. Every test scene

| Scene | Cap tightness (N m) | Program: s, quarters | ACT: s, grips | ACT right |
|---|---|---|---|---|
| 200 | 0.45 | 26.9, 7 | 27.2, 7 | yes |
| 201 | 0.58 | 26.9, 7 | 31.2, 8 | yes |
| 202 | 1.08 | 30.8, 8 | 31.3, 8 | yes |
| 203 | 0.91 | 26.9, 7 | 30.9, 8 | yes |
| 204 | 0.82 | 26.9, 7 | 31.4, 8 | yes |
| 205 | 0.90 | 26.9, 7 | 30.8, 8 | yes |
| 206 | 0.59 | 26.9, 7 | 31.2, 8 | yes |
| 207 | 0.83 | 30.8, 8 | 30.8, 8 | yes |
| 208 | 1.21 | 30.8, 8 | 30.8, 8 | yes |
| 209 | 0.72 | 26.9, 7 | 30.8, 8 | yes |
| 210 | 0.59 | 26.9, 7 | 26.9, 7 | yes |
| 211 | 0.63 | 26.9, 7 | 31.2, 8 | yes |
| 212 | 0.66 | 26.9, 7 | 31.0, 8 | yes |
| 213 | 1.12 | 30.8, 8 | 30.9, 8 | yes |
| 214 | 0.70 | 26.9, 7 | 31.0, 8 | yes |
| 215 | 0.72 | 26.9, 7 | 30.6, 8 | yes |
| 216 | 0.72 | 26.9, 7 | 31.4, 8 | yes |
| 217 | 0.87 | 26.9, 7 | 30.6, 8 | yes |
| 218 | 0.61 | 26.9, 7 | 26.8, 7 | yes |
| 219 | 0.70 | 26.9, 7 | 30.9, 8 | yes |
| 220 | 1.03 | 30.8, 8 | 26.1, 7 | **no** |
| 221 | 0.61 | 26.9, 7 | 27.1, 7 | yes |
| 222 | 0.63 | 26.9, 7 | 27.4, 7 | yes |
| 223 | 1.17 | 30.8, 8 | 30.7, 8 | yes |
| 224 | 1.03 | 30.8, 8 | 31.2, 8 | yes |
| 225 | 1.02 | 30.8, 8 | 30.6, 8 | yes |
| 226 | 0.91 | 26.9, 7 | 31.3, 8 | yes |
| 227 | 0.91 | 30.8, 8 | 31.2, 8 | yes |
| 228 | 0.59 | 26.9, 7 | 27.0, 7 | yes |
| 229 | 0.81 | 26.9, 7 | 31.3, 8 | yes |

The program needs 8 quarters mostly on tight caps (1 N m and more: its fingers slip a little more on each
turn). ACT turns 0.206 turns per quarter (the program 0.213), so it needs 8 on all but the loosest caps
(0.63 N m and less).

**Scene 220, grip by grip** (the cap's turns at the end of each grip, from the simulator):

| Grip | Program: pull in the test, turned | ACT: pull in the test, turned |
|---|---|---|
| 1-6 | 12.3 to 13.3 N each, 1.23 turns after 6 | 12.2 to 12.7 N each, 1.18 turns after 6 |
| 7 | 15 N (still on), 1.45 | no turn: lifted 54 mm, 5.5 N then 0 (fingers slipped off), 1.18 |
| 8 | about 0 N: off, 1.48 | |

## 8. The decisions, in order

| Date | Decision | Why |
|---|---|---|
| 2026-11-05 | Learn only Arm B's unscrewing, with ACT, from sensors (no camera) | the camera can't see "looser"; the rest works programmed |
| 2026-11-05 | Record teacher A (test every quarter) first; study the data | |
| 2026-11-06 | Don't record teacher B for ACT | a fixed rule: the program does the same; learning can't beat it with every cap screwed fully down |
| 2026-11-06 | Train ACT on teacher A anyway | to check ACT can unscrew from sensors at all, before the harder version |
| next | caps screwed on by different amounts, a teacher that decides from feel | where learning can beat the program |

## 9. Lessons

- On battery this Mac ran the simulation about 6 times slower (12 runs in 19 minutes, against 30 in
  8 on the charger). Run long jobs on the charger.
- Import numpy and the simulation before PyTorch, or a process stops with "OMP: Error #15". Ruff's
  import sorting can reorder them; `learn/evaluate.py` imports PyTorch with `importlib` for this reason.
- Splitting `cap_script` into `unscrew` and `put_cap_away` changed nothing: the 30 test scenes gave the
  same quarters and times as before, to the 0.1 s.
