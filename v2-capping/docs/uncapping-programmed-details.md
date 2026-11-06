# Uncapping, programmed: the details

The measurements and reasons behind [uncapping-programmed.md](uncapping-programmed.md) (start there).

**Status (2026-11-05): the programmed uncapping works on the 200 mm tube: 10 of 10 on scenes 150 to
159, and 30 of 30 in a stress test (new scenes 200 to 229, each with the tube off-centre in its cell and
the cap 0.45 to 1.21 N m tight).** Arm A holds the tube with its own grip (no ties). The one weakness
found: if the arms start a little away from home, the steps after the cap is off (the put-back, once the
cap carry) can fail (section 4). This page says what it does, the numbers and what was measured. The cap
itself (thread, friction, the cap trial) is in [capping.md](capping.md).

What changed on 2026-11-05 (sections 3 and 4): the capped tubes stand further apart; 100 mm tubes are
left out, as in v1; Arm A grips the tube between the middles of its pads, tighter, with its fingers
locked (the real 2F-85 is self-locking), so the tube is no longer tied to the hand; Arm B grips the cap
higher. v2 no longer needs any change in v1: it builds v1's scene with its own copy of v1's builder
(`capping/lab_build.py`).

## 1. The task

One arm holds a capped tube, the other unscrews its cap; the cap goes on a holder and the open tube into
the destination rack. It is programmed, with no learning: it is the baseline a learned version will be
compared with, and the teacher it can learn from.

```
Arm A (arm2, beside the capped rack)       Arm B (arm1, beside the source rack)
------------------------------------       ------------------------------------
pick a capped tube (Step 1b's side grasp)  wait at home
carry it to the middle, hold it up
"ready"  -------------------------------->  look down at the cap with the wrist camera
"holding tight"                             fingers down over the cap, grip it
                                            quarter turns until the lift test says "off"
                                            lift the cap away, set it on the cap holder, home
put the open tube in the destination rack (Step 3's put-back), home
```

Everything the arms **decide** comes from what a real arm could sense: Step 0's overhead camera (where the
racks and tubes are, which tubes are capped), the wrist camera's depth picture (where the cap is), and the
force on Arm B's gripper (whether the cap is off). The simulator's own knowledge is used only to **grade**.

```
make uncap SEED=150 SIZE=0.2       one run (about 50 s), printed as JSON
make uncap-eval SEEDS=150-159      200 mm on those scenes (10 runs: about 3 min); SIZES="0.15 0.2" for both
make uncap-stress                  200 mm, scenes 200-229, random placement and tightness (about 7 min)
make grip-test SEED=150 SIZE=0.15  how much twist Arm A's grip holds, for each grip setting (about 1 min)
```

## 2. The files

```
uncap/
  episode.py    the run: the scene, both arms' scripts, the grading
  grip.py       Arm A's firm grip on the tube (section 3)
  grip_test.py  how much twist that grip holds, and why Step 1b's did not
  perceive.py   Step 0 with a cap check (v1's liquid check reads a cap as "full")
  holder.py     the cap holder
  evaluate.py   many runs in parallel, and the summary
```

It reuses v1's code, **read only** (on the PYTHONPATH in `pixi.toml`): Step 0's detector, Step 1's
side-grasp network, Step 1b's coordinator (choosing a grasp, picking, carrying, checked moves) and Step 3's
put-back. From the v2 cap trial it takes the grip on the cap, the quarter turn and the lift test.

## 3. How each part works

### Seeing: which tubes are capped (perceive.py)

v1's Step 0 detector finds capped tubes as well as open ones: on scenes 150 and 153 all six, within
1.3 mm, with the right sizes (the cap's top is 4.6 mm above the rim, well inside the rounding to 100, 150
or 200 mm). But v1 then tells a source from an empty tube by looking straight down into it: a capped tube
shows its cap at the top, which v1 reads as **full**, so the receivers' rack became a second "source
rack". Here the same look first asks **is it capped?** (a surface within 5 mm of the top at its centre).
The rack of capped tubes is the **capped rack**.

The detector also calls the cap holder a rack; the holder is a fixed part with a known place, so anything
found there is left out.

### Arm A: pick and hold (Step 1b's, with three changes)

1. **The cap counts as part of its tube** in Arm A's checks. Step 1b lets the fingers come near the tube
   they pick (the last move slides in past its top); the cap is a separate body, so the fingers stopped
   at it (3.6 mm, inside the 5 mm margin).
2. **Grip low, but above the middle.** Step 1b grips as high as it can (3 cm below the top). Held so,
   Arm A's fingers and wrist sit just under the cap, and Arm B's fingers could not get round it (all 12
   ways blocked). Gripped too low, the tube no longer hangs from the fingers but balances on them and
   swings over (150 mm tube gripped 7 cm down: 47 degrees). So the lowest grip at least 2.5 cm above the
   tube's middle is tried first: **5 cm down for a 150 mm tube, 7 cm for a 200 mm tube**. A 100 mm tube is
   gripped 3 cm down whatever is asked: in its rack, Step 1b keeps the fingers 2 cm above the rack.
3. **The capped tubes start asleep**, as v1's tubes do (`episode.build`). `capping.scene` keeps them
   always awake (needed in the cap trial, where the tube is held by ties to the world). Awake, a tube
   standing in its cell slowly leans over against the cell wall with nothing touching it (5.6 degrees in
   15 s on scene 150), and Arm A's planned way in stopped fitting. Asleep, they stay upright until
   touched; held in Arm A's fingers a tube stays awake.

It holds the tube's top at Step 1b's pouring point (above the middle of the table, 35 cm up).

### Where the tubes stand, and which sizes

- **Spread out** (`capping/scene.py`, `SPREAD_CELLS`). v1 stands its three receivers so that two are
  always diagonally next to each other (51 mm apart). A cap makes a tube's top 28.8 mm wide instead of
  25, and Arm A's closing fingers came within the margin of a neighbour's cap: 4 of 12 runs could not
  pick a tube. Now the three stand at the two ends of one row and the middle of the other (144 and
  80 mm apart). v1's random draw is made first and mapped onto these cells, so the sources and their
  liquid are v1's exactly. Since then Arm A has picked its tube in 8 of 8 runs.
- **No 100 mm tubes** (`episode.SIZES`), as v1 leaves them out: in its rack Step 1b grips a 100 mm tube
  3 cm under its top whatever is asked, so Arm A's hand is right under the cap and Arm B cannot get
  round it.

### Arm A's grip: firm enough for the cap (grip.py)

The cap needs **0.6 N m** to start turning, and Arm A must hold the tube against it. `grip_test.py`
measures this: Arm A picks and holds the tube as in a run, then the tube is twisted about its own axis
by a torque rising 0.3 N m each second until it turns 2 degrees in the hand, and, separately, by a
steady 0.7 N m for 2 s (does it stop turning, or keep going?).

**Step 1b's grip held only 0.05 N m.** Looking at the contacts and the finger joints while twisting
showed three reasons, and each has a fix:

| Why it gave way | Fix | Measured (scene 150, 150 mm) |
|---|---|---|
| Held too deep in the hand: Step 1b's fingers go 11.5 mm past the axis, so the tube sits 12 mm behind the pads' middles, at their inner end, where the 2F-85 wraps round it ("encompassing" grip). Squeeze 8 N | pinch point on the tube's axis (`HOLD_DEPTH = 0`): the tube sits between the pads' middles (within 5 mm on 8 tubes) | squeeze 45 N, holds 0.33 N m |
| The fingertips swing: in the model each fingertip is carried by a link held only by a weak spring (0.05 N m per radian). Twisting pushes one pad along its finger and that fingertip swings away (seen at 0.5 N m: the pad moved 23 mm) | **finger lock**: once closed, every finger joint is held where it is by a stiff spring. The real 2F-85 is self-locking: its fingers stay put with the object pressing on them (Robotiq) | with the next two: see below |
| The contacts creep: MuJoCo's contacts are a little soft, and a held object slowly slides under a steady sideways force well below its friction limit | **noslip** (2 iterations, AutoBio's setting), on only while the cap is turned | |
| | close 10 mm past the tube instead of 4 (about 87 N; the real gripper gives 20 to 235 N) | |

| Grip (scene 150, 150 mm, centred) | Rising twist: holds up to | Steady 0.7 N m for 2 s |
|---|---|---|
| centred only | 0.33 N m | spins |
| + squeeze 10 mm | 0.49 | spins |
| + squeeze 10 mm + noslip | 0.46 | spins (a fingertip swings away) |
| + finger lock + noslip | 0.73 | creeps 1.2 degrees per s |
| **+ squeeze 10 mm + finger lock + noslip** | **1.42** | **0.08 degrees, stops** |

AutoBio's contact settings alone (noslip, stiffer finger contacts) did nothing for Step 1b's deep grip
(0.05 N m): the trouble was the fingers, not the contact. On all 8 tubes (scenes 150 to 153, 150 and
200 mm) the full grip held **1.37 to over 1.5 N m** (2.3 times or more what the cap needs) and turned at
most 0.21 degrees under the steady 0.7 N m. Step 1b's grip on the same tubes: 0.28 to 0.43 N m when
centred, 0.05 N m as Step 1b holds it.

In the run: Arm A picks with the centred grasp; at "ready" it closes tighter, waits 0.6 s, and locks
its fingers (noslip on); once the cap is on the holder it unlocks, eases back to Step 1b's squeeze and
does Step 3's put-back. In the runs the tube turned 0.1 to 2.2 degrees in Arm A's hand while the cap
came off (once 10.6, when Arm B's pad touched the tube: section 4).

### The old simplification (kept for comparison): ties

Before the grip above, the tube was tied to Arm A's hand while the cap was turned (`--ties`, the
function `hold_firm`). What was measured then, twisting the held tube steadily for 2 s:

| How Arm A held it | What happened |
|---|---|
| Step 1b's grip | the tube spun in the fingers at 0.2 N m |
| told to close harder (8 mm past the tube) | it tipped over in the fingers (25 degrees in 5 s) |
| a smooth cylinder for the pads to press on (the cap trial's fix for the cap) | no difference |
| tube between the pads' middles, a higher motor limit | held 0.6 N m on one tube, spun at 0.2 N m on another |

Why: Step 1b holds the tube 11.5 mm deep in the hand, at the inner pads, and the fingertips curl round it:
the gripper's motor reaches its target without pressing (its force 0.2 of 5). A real Robotiq 2F-85 grips
with 20 to 235 N, set for the task; about 25 N with rubber on glass would hold 0.6 N m.

So the tube was tied to Arm A's hand where the fingers held it, by three points (`hold_firm`), while the
cap was turned. That is no longer used; `python -m uncap.episode --ties` still runs it.

### Arm B: find the cap, unscrew it, put it down

- **Look.** The wrist camera's depth picture from 25 cm above where the cap should be; the cap is the
  highest thing within 3 cm of that point, its top every pixel within 3 mm of the highest. It gives the
  cap's middle, its top's height and its size. On scene 150: **0.1 mm** off sideways, 0.0 mm in height,
  14.3 mm across (it is 14.4): the run stops if the size is not a cap's.
- **Grip.** As in the cap trial (fingers down, opened 40 mm, told to close 10 mm past it, about 90 N),
  but higher on the cap: pads 18 mm above the middle of its side (`CAP_GRIP_ABOVE_MIDDLE`; the trial's
  4 mm). Lower, a pad's tip (a little below the cap's bottom, leaning in) touched the tube as the cap
  came off its thread: at 4 mm in 4 of 8 runs, at 8 mm 2, at 12 mm 1; at 15 mm it still grazed the top
  of the tube's collision walls, which reach 1 mm above the drawn rim. At 18 mm nothing touches, and the
  grip still breaks seals of 0.46 to 1.23 N m.
- **Lift test, second look** (`CAP_ONLY`). Under 5 N means the cap is free, but Arm B should then carry
  the cap's weight only (0.04 N). If it reads more (a 200 mm tube and cap weigh 0.41 N), something is
  coming up with the cap: the tube, slipping out of Arm A's hand. The run then stops instead of saying
  "off". (From the AutoBio copy, docs/autobio-uncapping.md: there a failing hold read as "off".) In
  every run so far it read 0.04 N. The way the fingers open is chosen from 12 directions
  (45 degrees first): the first whose whole way down, and a quarter turn there, touches nothing but the
  cap.
- **Unscrew.** The cap trial's loop: turn the wrist 90 degrees anticlockwise, rising a quarter of the
  pitch; **lift test** (6 mm up, hold 0.3 s; under 5 N means off); if not off, down, open, turn back,
  grip again. At most 10 quarters.
- **Put down.** Up 6 cm, across at that height (never lower: the tube is still held up in the middle),
  down onto the holder's post, open, up, home.

### The cap holder (holder.py)

A post on a stand (a rack's height), at (0.25, 0.12) beside Arm B, out of the middle. A post rather than a
tray: Arm B's finger pads reach about 8 mm below the cap's bottom, so on a tray they would hit the tray
first; on a post narrower than the fingers they pass either side. The cap rests on top, right way up,
where it can be gripped the same way to screw it back on.

## 4. What it measured

All with the **200 mm tube** (the 150 mm tube is sometimes gripped only 3 cm under its top, close under
the cap). A run is "right" only if: no failure, the cap on the holder right way up, the tube standing in
a destination cell, nothing touched that should not be, and the tube turned under 5 degrees in Arm A's
hand.

| Test | Runs | Right | Cap off | On holder | In rack | Tube turned in Arm A's hand |
|---|---|---|---|---|---|---|
| Scenes 150 to 159 (`make uncap-eval SEEDS=150-159`) | 10 | **10** | 10 (7 quarters) | 10 | 10 | at most 0.7 degrees |
| **Stress test**, new scenes 200 to 229 (`make uncap-stress`) | 30 | **30** | 30 (7 or 8 quarters) | 30 | 30 | at most 1.4 degrees |

The scenes only differ in which cell each tube stands in, and scenes 150 to 159 already cover all 6
cells the 200 mm tube can take; more seeds alone would repeat them. So the stress test also varies, at
random for each run, what a real lab would: each capped tube stands up to 2 mm off the middle of its cell
(a 30 mm cell, a 25 mm tube) and turned any way, and the cap takes **0.45 to 1.25 N m** to undo (24 mm
caps: 40 to 60 % of the 1.1 to 2.0 N m they are put on with). Measured over the 30: the seal broke at
0.46 to 1.23 N m (Arm B's wrist), the lift test read 12.3 N or more on the thread and 0.04 N off, and
"off" was never said too early.

**One weakness: the arms' starting pose.** With each arm joint also started up to 0.03 rad (2 degrees)
off home (`--vary-arms`), 24 of 30 were right. All 30 caps still came off, with the tube held; the 6
failures were afterwards: Step 3's put-back found no way to put the tube down from the pose Arm A then
held it in (4: every one of its 432 tries "no clean pose" or "would tip past 45 degrees"), the tube
landed 16 mm off its cell (1), and the cap was dropped on the way to the holder (1). The same 6 with the
arms starting at home: 6 of 6 right. A real arm goes home to within 0.03 mm, so this does not happen to
the programmed baseline, but a learned version will not follow exactly the same poses.

Before (2026-11-04, with 100 mm tubes and the tube tied to Arm A's hand): 1 of 12 right.

## 5. Next

1. **The put-back from other poses**: find why Step 3's planner finds no way down when Arm A holds the
   tube in a slightly different pose (and why the cap was dropped once), so the steps after uncapping
   are as sure as the uncapping. Matters for the learned version.
2. **The learned versions** (the plan in the overview): a "cap off?" detector from the force signals,
   then imitation learning of the whole uncap from these runs, compared with this baseline.
3. Later: the 150 mm tube (gripped lower), caps on the sources, screwing caps back on.
