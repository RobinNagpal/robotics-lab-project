# Step 2: Which Way to Pour (approaches, tests and recommendation)

This page compares the ways to do the pour and records what was measured **before** building it: short tests
to choose the approach. They showed that some of the first plan's assumptions were wrong. What was built
afterwards is in the overview, [step-2-pour.md](step-2-pour.md).

---

## In short

- **Grip first.** The source tube slipped out of position once tilted past about 60 to 80 degrees. Raising the
  finger pads' friction to 1.2 fixed that: on 20 new scenes no tube was lost, and Step 1b went from 41 of 60 runs to
  54 of 60. The 6 runs still failing are mostly a separate problem: fingers passing too close to a neighbouring tube.
- **A recorded motion is not enough.** One fixed motion hit the target in only about 1 pour in 6, because the
  tubes start at different levels.
- **Live feedback works.** Watching the weight while pouring hit the target in 105 of 105 pours, with a perfect
  simulated scale. It has not been tested with noise tuned for, or with a camera.
- **Do not touch the rims.** With today's grips the two grippers collide first, and it is not needed: the lip
  15 mm from the receiver's centre and 20 mm above its rim spills about 1%.

---

## What the words mean

| Word | Meaning |
|---|---|
| **Source** | the full tube, held by Arm B (arm 1) |
| **Receiver** | the empty tube, held by Arm A (arm 2), to be filled to 50% |
| **Lip** | the point of the source's rim nearest the receiver, where the liquid leaves |
| **Tilt** | how far the source is turned from upright: 0 degrees is upright, 90 degrees is lying flat |
| **Open loop** | the robot plays a planned motion and does not check the result while moving |
| **Closed loop** | the robot measures while it moves and changes what it does |
| **Bench** | a test with the real tubes and balls but no arms: the source is turned directly, so one pour takes about 5 s |
| **Real arms** | the full simulation: both arms, the real fingers and tubes, from the end of Step 1b |
| **Pinned** | a test shortcut: the simulator keeps the tube stuck to the fingers at every step, like glue. Only the grip is fake. |
| **Target window** | 48 to 52% full: within 2 balls (small), 3 balls (medium) or 4 balls (large) of the 50% mark |

---

## 1. First problem: the tube slips when tilted

With the real arms and the first settings, **the source slipped on all 6 scenes tried**, starting at about 60 to
80 degrees of tilt. The receiver ended with 0 to 30 balls instead of 74, 73 to 125 balls spilled, and on 3 scenes
the receiver was knocked out of its fingers.

### Is it the arm or the gripper?

No. The UR5e and the Robotiq gripper are both fine choices. The cause is in how the simulation's gripper behaves:

| | Simulated | Real Robotiq 2F-85 |
|---|---|---|
| How hard each pad presses | about 3 N | 20 to 235 N, as set |
| Pad friction | 0.6 to 0.7 | not measured for rubber on glass |

The simulated fingers are told a position, not a force. Once there, the motor pushes with only about 0.2 N, and
the 3 N comes from a small spring in the finger linkage. Closing them further does not press harder: in the one
scene checked, the tube was squeezed out and dropped.

### What was tried

Each was tilted to 100 degrees on the same 6 scenes, away from the receiver, so only the grip was being tested.

| Change | Tube stayed in place | Notes |
|---|---|---|
| Slower tilt | 0 of 6 | |
| Squeeze harder (6 or 8 mm) | 0 of 6 | |
| Grip against twisting (`condim=4`) | 0 of 6 | started slipping earlier, about 10 to 30 degrees on most scenes |
| Grip 5 cm below the rim | 2 of 4 | |
| Tube further out on the pads | 2 of 6 | rims 7 to 12 mm off, and one tube lost |
| Stiffer finger spring, 5 times | 0 of 6 | the pads press 16 to 20 N, but the tube still turned |
| Stiffer finger spring, 20 times | 0 of 6 | the tube is squeezed out |
| **Pad friction 1.2** | **5 of 6** | worst slip about 6 degrees |
| Pad friction 2.0 | 5 of 6 | no better than 1.2 |
| Stiffer spring and friction 1.2 | 6 of 6 | held best here, but spilled more in real pours (up to 16 balls) |

"Stayed in place" means it turned less than 5 degrees in the fingers.

### The fix: pad friction 1.2

The simplest change that works. It is set once for both grippers, as `PAD_FRICTION` in
`simulation-environment/lab_scene/settings.py`.

**Is 1.2 realistic?** It is a likely value for rubber on clean glass, but it was not measured. As a rough
comparison, friction times pressing force is 1.2 x 3 N = 3.6 N here. A real gripper at its lowest force, with a
cautious friction of 0.4, gives 0.4 x 20 N = 8 N. So the simulated grip is still weaker than a real one, not
stronger.

**Why friction worked and more force did not:** the tube is held at a few tiny points, and the most lightly pressed
one lets go first. The full story, with pictures, is in [grip-slip-guide.md](grip-slip-guide.md).

### What it changed

| Test | Pad friction 0.6 to 0.7 | Pad friction 1.2 |
|---|---|---|
| Step 1b, the 10 scenes used to choose 1.2 | 21 of 30 | 30 of 30 |
| **Step 1b, 20 new scenes** | **41 of 60** | **54 of 60** |
| Balls lost on the new scenes | 613 | 0 |
| Real-arm pours, not pinned (6 scenes) | all slipped, 73 to 125 balls spilled | all poured, 0 to 1 ball spilled |

The new scenes (seeds 10 to 29) played no part in choosing 1.2, so they are the fair test. To repeat it, from
`step-1b-pick-and-carry/`: `pixi run python -m carry.evaluate --first 10 --seeds 20 --out runs/v4_new_scenes`.
The older friction was run the same way with `PAD_GEOMS` emptied in `lab_scene/scene.py`.

Why runs failed on the new scenes:

| Reason | Pad friction 0.6 to 0.7 | Pad friction 1.2 |
|---|---|---|
| Tube lost, or 20 mm or more out of place | 11 | 0 |
| Fingers touched a neighbouring tube, or the move was refused because they would | 3 | 5 |
| Rim just over the 5 mm limit (5.0 to 5.8 mm) | 4 | 1 |
| No clear way left to the tube | 1 | 0 |

These count runs, not separate carries: when the size asked for cannot be picked, a run falls back to another
size and can repeat another run's carry. The 3 runs on seed 14 are one lost tube, for example.

In the 6 real-arm pours the stop was decided from the simulator's own ball count, not from a sensor, so they
show the motion works, not how accurate the fill will be. They ended at 76 to 80 balls against a target of 74.
The tube still moved by up to 11 mm in the fingers, and on one scene the two grippers touched briefly.

### What it did not fix

**Touching a neighbouring tube.** Three runs that passed with the old friction failed this way at 1.2, so this was
tested: the scenes with these failures were run again at friction 1.0, 1.1, 1.1999, 1.2001, 1.3 and 1.5.

- **It does not follow friction.** Higher friction was not steadily better or worse. Seed 16 passed at 1.1999 and
  touched at 1.2001, and seed 21 did the opposite. A change of 0.0002 cannot push a tube, so these are near misses:
  the fingers pass a neighbouring tube with almost no room, and tiny differences decide which way it goes.
- **It is a Step 1b clearance problem,** present at both friction settings, and it is not fixed. Of these 5 runs at
  1.2, 3 brush the tube and still finish, and 2 stop because the arm refuses the move.

**One scene stays just over the limit.** Seed 29 on the small tube ended 5.0 to 5.5 mm off at every friction where
it finished. This includes two of the runs that brushed a tube.

**The margin below 1.2 is not known well.** At friction 1.1, seed 11 lost its tube after touching the rack and a
neighbouring tube. That is one run, and what caused it was not checked. No tube was lost in 114 runs from 1.1999
up: 90 at exactly 1.2 and 24 in the repeat tests. Those repeats were chosen because they were hard scenes, so
they are no reason to change the value, and it stays at 1.2.

---

## 2. The steps, checked

| Question | Answer from the tests |
|---|---|
| Can Arm B find the receiver's mouth? | Yes. It can be worked out from Arm A's position and the tube's size. In these tests the position was read from the simulator, so that calculation still has to be built. |
| Should the rims touch? | Not with today's grips. At 90 degrees, the two grippers hit each other when the lip is under about 20 mm above the rim. At 75 degrees it needs 20 to 30 mm. |
| Where should the lip go? | **15 mm from the receiver's centre and 20 mm above its rim.** Not the centre of the mouth. |
| How far does it tilt? | About 90 degrees. Medium tubes stopped between 89.5 and 91.5 degrees. |

**Why not the centre of the mouth?** The balls leave the lip moving sideways, so they land further across than
where they left. On the bench, with the lip 20 mm above the rim:

| Lip's distance from the receiver's centre | Balls spilled |
|---|---|
| 7 mm | 38% |
| 12 mm | 3% |
| 15 mm | 1% |
| 16 to 18 mm | 0% |
| 22 mm | 8% |

**Sideways error.** The lip must also stay in line with the receiver. At 15 mm out and 20 mm up, a sideways error
of 4 mm spills 5%, 6 mm spills 13% and 8 mm spills 31%. Step 1b's own sideways error is usually about 1 mm, and
4.5 mm at worst over 88 finished runs, so most pours are fine and the worst would spill about 5%. Lower is better
for spill: 5 mm above the rim spills 0 to 2% with the lip anywhere from 4 to 15 mm out. The grippers do not allow
that yet.

**The "liquid" does not act like water here.** The first ball left at **81 to 88 degrees for every size and every
starting level**. Real water in a tube 90% full would start at about 55 degrees, which is what the first plan
assumed. The likely reason is that only about 3 balls fit across the tube, so they lock together. Also, at a
fixed tilt the flow stops by itself. A real liquid would behave differently, so what is learned here will not
carry over to real liquid.

---

## 3. The approaches

### A. One fixed motion (open loop)

The same motion for every tube, with no live readings. This is what a model would learn if it only remembered the
sequence ("for the medium tube, go there, tilt this much, for this long"), because it would have nothing to tell
it how full the source is.

### B. A motion chosen from the starting level (open loop)

Look once before pouring, then play the motion that suits that level.

### C. Live feedback (closed loop)

Keep measuring the receiver while pouring, and decide when to slow down and stop. It can be programmed, or learned
as a model that takes the camera picture and the weight and commands the tilt.

### D. Other learned options

| Option | What it does | Fit here |
|---|---|---|
| Learned flow model | predicts how much will still arrive if the arm tips back now | good: it targets "stop early", the hardest part |
| Reinforcement learning | learns by trial and error from a score | possible on the bench (about 1,400 pours in 20 minutes on 6 cores), but the score is hard to design and it does not include the camera |
| ACT, Diffusion Policy | large imitation models from pictures to motion | later: they probably do not fit the 15 to 20 minute budget |

### How they compared

Bench pours, tuned on some seeds and graded on others, **pours within 48 to 52%, out of 35 per size**:

| Method | Small | Medium | Large |
|---|---|---|---|
| A. One fixed motion | 6 | 5 | 6 |
| B. Motion from the exact starting level | 27 | 28 | 20 |
| B. Same, with the level misjudged by 3 points | 15 | 15 | 17 |
| **C. Live weight, no noise** | **35** | **35** | **35** |

- **A fails because of the starting level, not randomness.** The same motion from the same level repeats to
  within about 1.5 balls. But on medium tubes it gave about 54 balls from a 60% source and 97 from a 90% source,
  against a target of 74.
- **B works only if the level is known well.** An error of 3 points cuts it to about half.
- **C spilled no balls.** Just over half of the pours (57 of 105) needed one or two small top-ups, including 33 of
  35 on medium. A pour took about 18 s, or 22 s on medium.
- **C is sensitive to noise as built.** With noise of about 0.016 N per reading, medium fell to 19 of 35, and with
  about 0.064 N to 0 of 35: the simple controller took noise for flow and stopped early. It was not tuned for
  noise, so this is a warning, not a limit.
- **The camera had not been tested at all.** It has since: see [step-2-pour-camera.md](step-2-pour-camera.md).

---

## 4. Standard practice

- **Closed loop is the usual way.** A pour is measured as it happens, usually with a simple feedback rule (PID).
  Reported errors are about [10 ml with torque and touch sensing](https://arxiv.org/abs/2310.18473) and
  [38 ml with vision](https://arxiv.org/abs/1610.02610). These setups differ, so they cannot be compared directly.
  Our window is about ±1 ml on the medium tube, though balls are easier than water.
- **Lab systems usually weigh on a scale.** See this
  [chemistry lab pouring system](https://arxiv.org/pdf/2212.09672) and this
  [dual-arm solid dispenser](https://pubs.rsc.org/en/content/articlelanding/2023/DD/D3DD00075C). Nothing published
  was found on two arms pouring from tube to tube.
- **Learned options exist:** a [learned flow model with planning ahead](https://par.nsf.gov/biblio/10169419);
  [reinforcement learning](https://arxiv.org/abs/2203.04380); and imitation learning, where
  [ACT and Diffusion Policy scored 77% and 73%](https://arxiv.org/pdf/2505.22634) on a simulated pour. No source
  was found showing those two hitting an exact amount.

**The real wrist sensor is too coarse.** The UR5e's has a
[precision of ±3.5 N](https://www.universal-robots.com/manuals/latest/en/datasheets/ur5e/), and the medium target
weighs about 0.24 N, roughly 15 times less. The first plan's "the real UR5e has one built in" is misleading. In
the simulation, the weight reading stands in for a much finer sensor, and should have noise added. So far it was
read from a receiver fixed in place, not from Arm A's wrist.

---

## 5. Recommendation

1. **Grip:** pad friction 1.2, no pinning. Done, and on 20 new scenes no tube was lost (Part 1).
2. **Programmed baseline:** closed loop on the weight, with these steps:

   | Step | What the arm does |
   |---|---|
   | Move in | lip 15 mm from the receiver's centre, 20 mm above its rim, about 78 degrees of tilt: no flow yet |
   | Find the flow | tip at 2 degrees a second until the first balls arrive |
   | Pour | slow to about 0.7 degrees a second |
   | Stop early | start tipping back when weight + flow x reaction time reaches the mark |
   | Tip back | at 40 degrees a second |
   | Check | wait for it to settle, weigh, and top up if short |

3. **Learned model:** camera and weight in, tilt out, in two stages. Which comes first is still to be decided (below).
4. **Grading:** every method on the same scenes with the same window, on scenes not used for tuning.

### Decided

**The learned model:** a CNN that reads level and flow first (built: see [step-2-pour.md](step-2-pour.md)), then
a small policy trained by imitating the controller, because the CNN can be checked on every picture, and if the
policy fails it is clear why.

### Not done yet

- ~~The medium and large pours on the real arms, with a real stop rule.~~ Done, with the true ball count:
  227 of 236 pours within 2 points ([results](../../step-2-pour/docs/results.md)). Small tubes are still to do.
- ~~The camera: whether it can see the level at all with the source in front of it.~~ Done: a new fixed `pour`
  camera sees it in every pose tested ([step-2-pour-camera.md](step-2-pour-camera.md)).
- Noise on the weight, and reading it from Arm A's wrist.
- Clearance around the neighbouring tubes in Step 1b (Part 1). Until it is fixed, about 1 run in 12 will fail
  before the pour starts or brush a tube on the way, for reasons that have nothing to do with pouring.
- Small tubes: friction does not help here. Exactly the same runs fell back to a bigger size at both settings. Of
  the runs asking for small, 11 of 30 used it (1 of 10 on the first scenes, 10 of 20 on the new ones).
