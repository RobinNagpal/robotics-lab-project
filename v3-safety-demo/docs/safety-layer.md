# Track B: the safety layer

27 November 2026

This is Track B of [steps.md](steps.md): the layer that sits between v3's scripts and the arm and slows or holds
the arm when a hand is near. It was built with a "perfect" detector, one that reads the hand's true position from
the simulation; the perfect detector stays as the baseline the trained model is compared with.

**Phase 3 (27 Nov) joined it with Track A** ([integration.md](integration.md)): the zones are now Track A's fixed
ones, the person, the RGB-D camera and the trained YOLO detector plug in through `sensing.py`, and four of the
simplifications listed at the end were closed (detection delay, measured speeds, a signed log, a signed
registry). The sections below say where that changed things.

No v3 file was changed. Everything is in `v3-safety-demo/`.

## What it does, in short

| Situation | What the arm does |
|---|---|
| No hand near | runs exactly as in v3 (same to the last bit: same results on all 20 jobs) |
| Hand in the **warning zone** | slows to 25% speed |
| Hand in the **stop zone** | holds where it is (stops within about 20 ms in the simulation) |
| No hand for **1 second** | carries on from exactly where it was paused |
| Camera frame **missing, late or frozen**; detector **crashes, talks nonsense or is too slow** | holds; carries on after 1 s of good frames with no hand |
| A command **too fast** or **outside the work-area box**; the **emergency stop** | stops and stays stopped until an operator resets it |

## How it reaches the arm without changing v3

Every v3 step moves the arm the same way: it writes joint targets (where each joint should be) into the
simulation's control array, then calls `scene.step(model, data)` to advance the physics by one millisecond. The
call is looked up through the `scene` module each time, so the demo can replace it from outside
(`guard.install()`). Each call is one "tick" of v3's script, and the guard decides how the tick is carried out:

- **run**: one physics step with v3's targets, exactly as v3 does it;
- **slow**: the tick is spread over 4 physics steps, the targets moving a quarter of the way each step. The
  script's whole path is followed, just 4 times slower;
- **hold**: the targets are set to where the arm is at that moment, so it brakes at once, and the physics keeps
  running (the hand keeps moving, liquid keeps settling). The call does not return to v3's script until the arm
  may move again. The script is simply paused. When it may move, the arm first eases back onto the script's path
  (at most 0.5 rad/s, smooth start and stop), and the script then carries on where it was.

After each call, the control array holds v3's own targets again, so v3's code reads back what it wrote. Scratch
copies of the simulation that v3 uses for planning are not touched.

Why hold "where the arm is" and not at v3's last target: v3's arm follows its targets about 0.2 s behind. Holding
the last target let the arm coast on to it: 0.73 s and 41 mm to stop from 0.4 rad/s in a test. Holding where it
is stopped it in 16 ms and 1.6 mm.

## The two layers

**The normal layer (no ML)** checks every command v3 sends (each tick's six joint targets):

- **speed**: no joint faster than the limit, and the point between the fingers not faster than its limit,
  measured over the last 50 ms of commands (so one tiny correction does not count as a burst of speed);
- **work-area box**: the point between the fingers and the elbow must stay inside a box around the bench. The
  check uses a separate copy of the arm's model to work out where the command would put them, so it never
  touches the simulation;
- **measured speeds** (since phase 3): the arm's actual joint speeds and the gripper's actual speed, over the
  last 10 ms of physics, against the same limits times 1.25 (the arm follows its targets with a little
  overshoot). So an arm that moves too fast for any reason is stopped, not only one told to;
- **emergency stop**: `guard.emergency_stop()` at any moment.

Any of these **latches a stop**: the arm holds, `SafetyStop` is raised out of v3's script, and the arm stays
held until `guard.reset_stop()` (an operator's reset). The normal layer does not read the camera or the
detector, and nothing on the ML side can switch it off or overrule it.

**The ML layer** takes a camera frame 10 times per simulated second and asks the detector for the zone. Since
phase 3, a frame's decision takes effect **0.1 s after the frame was taken** (`guard.LATENCY`: the camera's
exposure and read-out, the detector and the rule), as it would on a real system; and before the job starts,
one frame is judged at once (the start-up check: the camera and detector run before the robot's program does,
so a hand already in the cell keeps the arm from ever starting).

- warning zone: slow; stop zone: hold;
- the arm goes back up only after 1 s with nothing worse: 1 s after the last stop-zone frame it may go to slow
  (if the hand is still in the warning zone) or run; 1 s after the last warning frame it may run.

**The watchdog** holds the arm when there is no answer it can trust:

| Fault | How it is spotted |
|---|---|
| frame missing | the camera returns nothing, or raises an error |
| frame late | taken more than 150 ms before it is checked (a 200 ms delay is always caught) |
| frame frozen | its number or time does not move on, or the same picture 3 times running |
| no fresh frame at all | no good decision for 150 ms (decisions keep coming once a frame period apart with the delay too) |
| detector crashed | it raises an error |
| detector talks nonsense | a zone that is not one of the three, a score that is not between 0 and 1, the wrong model version, or not an answer at all |
| detector too slow | more than 250 ms (wall clock) for one frame |

The arm carries on only after 1 s of good frames with no hand.

What the arm does is the **strictest** of the two layers: stop beats hold beats slow beats run.

## The safety log

Every decision is one line of JSON: the simulation time, the frame number, the model version, the score, the zone,
the action and the reason. Normal-layer stops, resets, and model loads or refusals go in too.

Each line carries the hash of the line before it, and its own hash. Changing a line breaks its hash; deleting or
adding one breaks the chain. The last hash (the "head") is also saved in the run's results, so cutting lines off
the end shows up too (`safety_log.verify`). Since phase 3 the hashes are **keyed with a secret** (HMAC-SHA256,
`keys.py`): someone who edits an entry and recomputes every hash after it cannot make them verify without the key.

`guard.replay(entries)` rebuilds every decision from the zones in the log alone and reports any line where the
logged action is not what the rules give. That is the plan's "rebuild decisions from the log" test.

## The model registry

`registry.py` keeps a JSON list of approved model files: version, file name, SHA-256, size and, since phase 3,
the model's **settings** (confidence threshold, picture size), approved with the weights because they change what
the safety function does just as much. Since phase 3 the registry file is also **signed** with the same secret
key: a registry edited by hand (to approve a model, or change a threshold) is refused as a whole. A model is loaded
only through `Registry.read_approved`, which reads the file once, fingerprints those bytes, and hands them back
only if the fingerprint is approved. So the file checked is the file loaded. Anything else raises
`UnapprovedModel`, and the attempt is logged. The registry also records which version is active and the order
versions were made active in, so `rollback()` goes back to the version before. `detector.load(path, registry,
log, build)` is the one way a trained detector is made.

The registry is `models/registry.json`: v1, then v2, approved and activated in that order (v2 active, v1 to roll
back to). The model files themselves (`models/hand_v1.pt`, `hand_v2.pt`) are not committed; their fingerprints
are.

## Files

| File | What |
|---|---|
| `guard.py` | the wrapper around `scene.step`, slow and hold, the behaviour rules, the watchdog, the normal layer, `replay` |
| `detector.py` | the detector interface (`Frame`, `Detection`, `detect`), the perfect detector, the truth camera, `load` |
| `safety_log.py` | the hash-chained log and `verify` |
| `registry.py` | approved models, fingerprints, activate, revoke, rollback (also a command line) |
| `keys.py` | the secret key that signs the log and the registry (made on first use, never committed) |
| `sensing.py` | (phase 3) the person, the RGB-D safety camera, the YOLO detector, the perfect detector |
| `standin.py` | stand-ins used before phase 3, now only by the unit tests: distance-to-arm zones and a scripted hand |
| `run_guarded.py` | v3's whole job (Step 6's `run_seed`, unchanged) with the guard on, and a person if asked |
| `check_scripted.py` | checks the scripted-hand run of phase 1, visit by visit, from its logs |
| `tests/` | unit tests for each part (`pytest tests`) |

## How to run

```bash
cd v3-safety-demo
pixi run python -m pytest tests                                                   # about 40 s
pixi run python run_guarded.py --hand none --workers 2                            # v3's 20 jobs, nobody there
pixi run python run_guarded.py --hand person --detector perfect --seeds 4 5 6 7   # a person, perfect detector
pixi run python run_guarded.py --hand person --detector yolo --seeds 4 5 6 7      # a person, the active model
```

The phase 1 scripted-hand run (below, results 3) used `--hand scripted`, since replaced by Track A's person.

## Results

### 1. With no hand, the guard changes nothing

v3's 20 jobs (Step 6's whole run), guard on, perfect detector, no hand (`results/calibrate_no_hand/`):

- the arm was never slowed or held, and every result is **identical to v3 run without the guard**, to the last
  digit. Checked two ways: the 10 jobs without a balance tube match v3's saved results exactly. The other 10 don't
  match the saved file, because that file is older than the current Step 3 code (saved 23 Nov at 22:50;
  `fill.py`, `cap.py` and `moves.py` changed on 24 Nov). On those, plain v3 run today (no guard, seeds 11 and 13)
  gives exactly the guarded results;
- **19 of 20 loaded and balanced.** The one failure, seed 11 ("no tube stands at [0.23 0.39]" in Step 4), is v3's
  own: plain v3 fails it the same way today. So v3 as it stands is 19 of 20, not the 20 of 20 in its docs. That
  is for v3 to look at; nothing here changes it;
- the guard added no noticeable time (wall-clock times within a few seconds of v3's).

### 2. Setting the limits

The same 20 runs, with the normal layer's limits wide open, measured what v3's jobs actually command:

| What | Most v3 uses | Limit set |
|---|---|---|
| joints 1 to 5 (seeds 0 and 13 measured per joint) | 1.25 rad/s | 1.6 rad/s |
| joint 6 (the last wrist joint) | 3.47 rad/s | 4.0 rad/s |
| speed of the point between the fingers | 0.88 m/s | 1.1 m/s |
| box for the point between the fingers | x −0.43 to 0.70, y −0.69 to 0.63, z 0.08 to 1.24 m | 5 cm larger on each side (z from 0.03 m) |
| box for the elbow | x −0.29 to 0.33, y −0.22 to 0.29, z 0.43 to 0.59 m | 5 cm larger on each side |

All 20 runs' peaks fit inside the set limits. Two things are worth knowing:

- **Joint 6 reaches 3.47 rad/s** in the 10 jobs with a balance tube: a quick turn of the gripper in Step 3, most
  likely the spare cap's screw-on turns (`cap.py`, 60° turns), while the gripper itself hardly moves. A real
  UR5e's joints top out at about 3.14 rad/s, so v3 asks for more than the real arm can do there.
- **The fastest gripper speed and the highest point (1.24 m)** both come from v3 sliding the joints back home from
  up high: the arm swings upright on the way.

Per-joint speeds were measured on 2 seeds only (0 and 13). The scripted-hand run below (seeds 0 to 3, the set
limits on) didn't trip them either. A full 20-seed run with the set limits would confirm the rest.

### 3. With a hand: slows, holds and carries on as planned (phase 1, stand-in zones)

This run used phase 1's stand-ins: a hand point beside the gripper and distance-to-arm zones with placeholder
sizes. It showed the guard's behaviour; the run with Track A's person and the fixed zones is in
[integration.md](integration.md).


Seeds 0 to 3 (one of each job type), the set limits on, a scripted hand every 30 s: 4 s in the warning zone, then
2 s in the stop zone (`results/guarded_scripted/`, checked by `check_scripted.py` from the logs alone):

| | Seed 0 (pairs) | Seed 1 (odd one) | Seed 2 (triangle) | Seed 3 (mismatch) | All |
|---|---|---|---|---|---|
| hand visits | 9 | 16 | 7 | 21 | **53** |
| held at the first stop-zone frame | 9 | 16 | 7 | 21 | **53 of 53** |
| slowed at the first warning frame | 9 | 11 | 7 | 14 | **41 of 53** (the other 12: see below) |
| back to full speed exactly 1 s after the last stop-zone frame | 9 | 16 | 7 | 21 | **53 of 53** |
| longest time to stop / distance | 14 ms / 0.4 mm | 27 ms / 1.2 mm | 14 ms / 0.4 mm | 17 ms / 0.7 mm | **under 30 ms, under 1.5 mm** |
| job loaded and balanced | yes | yes | yes | yes | **4 of 4** |
| imbalance (no hand → with hand) | 0.385 → 0.385 g | 0.41 → 0.41 g | 0.158 → 0.158 g | 0.115 → 0.413 g | all under the 1 g limit |
| balls spilled (no hand → with hand) | 0 → 0 | 5 → 1 | 0 → 0 | 4 → 6 | |
| simulated job time (no hand → with hand) | 226 → 272 s | 417 → 490 s | 164 → 212 s | 496 → 627 s | 20 to 26% longer |
| log chain intact; every decision rebuilt from the log | yes | yes | yes | yes | **yes** |
| normal layer tripped | no | no | no | no | **no** |

- **The 12 warning visits that held instead of slowing** are all in the balance-tube jobs (seeds 1 and 3), during
  Step 3's sideways grip. The scripted hand is placed 27 cm from the gripper, but with the gripper on its side the
  wrist ends up beside the hand point: the true distance was 2.5 cm, in the stop zone. So the perfect detector was
  right and the guard was right to hold. It's a quirk of where the stand-in hand is placed; Track A's hand will
  replace it.
- **The stops are short** because a hold sets the targets to where the arm is: it brakes in under 30 ms from up to
  0.12 m/s. These are simulation numbers; real stopping time and distance need the real arm (plan.md section 6).
- **Holding and slowing hardly change the job.** Seeds 0, 1 and 2 end with exactly the same balance as with no
  hand. Seed 3's second balance tube came out 0.33 g lighter (a hold during the pour) and the rotor 0.41 g out of
  balance instead of 0.12 g, still well inside the 1 g limit. Spills went down on one job and up on another.
- The scripted hand is a single point with no body, so "smallest distance" here is only to the stand-in's arm
  lines (0 cm when the wrist passes right through the point). Contact events come with Track A's hand.

**Done-when for Track B** (steps.md): a v3 job runs with a scripted hand and the arm slows, holds and carries on
as planned. Met on 4 of 4 jobs.

### 4. Unit tests

51 tests (`pytest tests`, about 10 s) covering: running exactly as v3 with nothing near; 25% speed; hold and carry
on after 1 s; holding where the arm is, then easing back; each camera fault (missing, dead, 200 ms late, frozen
number, same picture, not a frame); each detector fault (crash, unknown zone, NaN score, score above 1, wrong
version, not an answer, too slow); too fast per joint and for the gripper; leaving the gripper's box and the
elbow's box; the emergency stop; the stop staying latched until reset; the normal layer stopping the arm while the
detector crashes on every frame; the log catching edits, deletions, insertions, cutting short and rebuilt chains;
the registry refusing unknown, changed and revoked models (and logging it), updates and rollback; the perfect
detector and the stand-in zones.

## For phase 3 (plugging in Track A's parts): done

How these were met is in [integration.md](integration.md). The notes as written before phase 3:


- **Zones.** The guard needs one function, `zone_of(hand_points, arm_points) -> (zone, distance)`, plus
  `arm_points(model, data)`. The stand-in measures from a chain of thick lines through the arm's joints, with
  placeholder sizes (warning 0.35 m, stop 0.15 m). `TruthCamera(hand, arm_points=..., zone_fn=...)` and
  `PerfectDetector(zone_fn)` take Track A's versions directly.
- **The hand.** Track A's hand is a mocap body moved by time. Its driver goes in `guard.before_physics`, which is
  called before every physics step, including while the arm is held. The truth camera wants
  `hand.points(model, data)` (the hand's points, or None when it is parked).
- **The rendering camera** must add a little sensor noise to each picture, as a real camera has. Without it, a
  still scene renders the same picture every time and the watchdog takes that as a frozen camera, and holds.
- **The trained detector** implements `detect(frame) -> Detection(zone, score, model_version)` and is created
  through `detector.load`, so the registry approves it first. Its time per frame is checked against 250 ms.
- **Frame rate**: 10 frames per simulated second, set in `guard.FRAME_HZ`.

## Not done yet, or simplified

Status after phase 3 in **bold**.

- **Closed.** Detector latency is not simulated: the zone from a frame is acted on in the same millisecond. A
  slow detector is caught by the wall-clock budget instead. (Now: 0.1 s from frame to effect, with a start-up
  check.)
- **Open (a v3 decision).** Holding mid-pour (Step 3) stops the arm with the tube tilted, so water may keep
  flowing. On seed 3 it left a balance tube 0.33 g light (still balanced, see Results). A rule such as "tilt back
  upright when held mid-pour" would be a v3 or phase 3 decision.
- **Open.** Seed 3's scripted-hand run took 25 minutes of wall-clock time for 10.5 simulated minutes, about three
  times slower than the other seeds. Not looked into (the Mac was shared with Track A at the time).
- **Closed.** The normal layer checks commands, not the arm's measured speed. (Now it checks both.)
- **Closed.** The log's head is not signed. A real system would sign it or key the hashes with a secret. (Now
  the hashes are keyed, and the registry is signed too.)
