# Where we are (Friday 27 November 2026, evening)

A note to pick up from on Monday 30 November. The details are in the other docs; this page says what is done,
what is waiting, and exactly what to run next.

## In one paragraph

Phases 1 to 3 of [steps.md](steps.md) are done except one step. The person, the RGB-D safety camera and the YOLO
hand detector (Track A) are joined with the safety layer (Track B) and run inside v3's real jobs. With the
perfect detector the joined system works: 93 reaches by a person on the 4 test jobs, the arm stopped within
0.20 s every time, no contact while the robot moved, every job balanced. With the trained detector, the closed
loop found that YOLO v2 can take the robot's own gripper for a dark glove and hold a job for good. The fix is
ready but not trained: **v3 of the detector, waiting for a go-ahead to train (about 18 minutes).**

## What is done

| Part | State | Numbers | Doc |
|---|---|---|---|
| Person, zones, safety camera (phase 1, A) | done | the safety camera sees 100% of both zones | [camera-check.md](camera-check.md) |
| Safety layer (phase 1, B) | done | 51 unit tests at the time; all faults caught | [safety-layer.md](safety-layer.md) |
| Dataset + detector v1 (phase 2) | done | missed 21 of 840 stop-zone hands; false alarm on 2.7% of clear pictures | [detector-results.md](detector-results.md) |
| Depth camera + fixed rule | done | without depth, false alarms on 36% of clear pictures | [detector.md](detector.md) |
| Tracks joined (phase 3) | done | fixed zones; person, camera, YOLO plug into the guard (`sensing.py`) | [integration.md](integration.md) |
| Track B's simplifications closed | done | 0.1 s detection delay + start-up check; measured-speed limits; signed log; signed registry with settings | [integration.md](integration.md) |
| Detector v2 | released (registry) | missed 5 of 840 stop-zone hands; passed the update test (rules 1 to 4) | [integration.md](integration.md) |
| 20 v3 jobs, guard on, nobody there | done | identical to plain v3 on 20 of 20; never slowed or held; 19 of 20 balanced (seed 11 is v3's own failure) | [integration.md](integration.md) |
| Test jobs 4 to 7, a person, perfect detector | done | 93 reaches (40 into the stop zone); 0 contacts while moving; closest 13 cm; 4 of 4 balanced | [integration.md](integration.md) |
| Closed loop with YOLO v2 | **failed, explained** | v2 boxed the robot's gripper; the arm held 236 s; job unfinished | [integration.md](integration.md#the-closed-loop-failure-with-yolo-v2) |
| False alarms with nobody there (new test) | done | v1: 3, v2: 8 of 2,570 pictures | `results/nobody_test_v*.json` |
| Release rule 5 (nobody false alarms) | committed before v3 | under it, v2 would not have replaced v1 | `release_check.py` |
| Hard negatives for v3 | mined | 472 pictures with no hand (the robot in awkward poses) | `data/hands/images/train_negatives/` |

All 82 tests pass (`pixi run python -m pytest tests`, about 40 s); the code passes ruff.

## Waiting for a decision

1. **Train detector v3** (asked, not yet answered). From v2, on 2,272 pictures (1,200 + 600 + the 472 hard
   negatives), 10 epochs, about 18 minutes on the Mac's GPU.

## Next steps, in order (Monday)

Run from `v3-safety-demo/`. Keep to 2 workers per job (4 cores in all); say before any job over 10 minutes.

```bash
# 1. Train v3 (about 18 min, GPU) -- after the go-ahead
pixi run python train_yolo.py --name v3 --epochs 10 --train train train_extra train_negatives \
    --weights runs/v2/weights/best.pt

# 2. The update test, rules 1 to 5 (about 3 min)
pixi run python evaluate.py --weights runs/v3/weights/best.pt --name v3
cp runs/v3/weights/best.pt models/hand_v3.pt
#    (approve it first under a temporary name, or scan with the weights directly; see the note below)
pixi run python nobody_scan.py --version v3 --jobs 4 5 6 7 --name test_v3
pixi run python release_check.py results/evaluation_v2.json results/evaluation_v3.json \
    --nobody results/nobody_test_v2.json results/nobody_test_v3.json --out results/release_v3.json

# 3. If it passes: approve and activate v3 in the signed registry (threshold from evaluation_v3.json)
pixi run python registry.py models/registry.json approve models/hand_v3.pt v3 \
    --note "update: +472 hard negatives (the robot's gripper); passed rules 1-5" --threshold <T> --imgsz 640
pixi run python registry.py models/registry.json activate v3

# 4. The closed loop with YOLO v3: a person on the 4 test jobs, then nobody on all 20 (about 25 + 60 min)
pixi run python run_guarded.py --hand person --detector yolo --seeds 4 5 6 7 --name person_yolo_v3
pixi run python run_guarded.py --hand none --detector yolo --name nobody_yolo_v3_20
```

Note on step 2: `nobody_scan.py` loads a version through the registry, so v3 has to be in the registry to be
scanned. Either approve it (without activating) before the scan and revoke it if it fails, or add a
`--weights` option to the scan for candidates. The second is cleaner: a candidate should not enter the
registry before it passes.

Then, for the closed-loop results with YOLO v3, compare against the perfect detector's run
(`results/person_perfect/`): contacts while moving, time to stop, false holds, jobs balanced. Write them into
[integration.md](integration.md) and [detector-results.md](detector-results.md) (v2 and v3 sections are still
to write there), and update this page.

## Open questions (for later, not blocking)

- **Zone sizing.** The stop zone ends at the arm's reach. From a hand entering it to the arm standing still
  takes up to 0.20 s; ISO 13855 sizing would make the whole bench a stop zone. The better design is speed and
  separation monitoring (ISO/TS 15066): stop by the hand's distance to where the robot is now, which the depth
  camera makes possible. A design decision, written up in [integration.md](integration.md#still-open).
- **v3 (the centrifuge project) is 19 of 20 today, not 20 of 20.** Seed 11 fails in Step 4 in plain v3 too ("no
  tube stands at [0.23 0.39]"); v3's saved results predate its Step 3 changes of 24 November. PR #8's
  description and v3's docs still say 20 of 20.
- **Holding at awkward moments** (mid-pour; a long false hold) costs availability, never safety. Worth a rule
  later, such as "tilt the tube back upright when held mid-pour" (a v3 decision).
- Later phases (manufacturer file, Packages 1 to 3, report, video) are to be done separately, as agreed.

## State of the machine and the repo

- **Branch `v3-safety-demo`**, local only (30 commits on top of `main`, not pushed). Made from
  `v3-centrifuge-loading`, whose own work is pushed to PR #8.
- **Worktree** `../robotics-lab-project-guard` (branch `v3-safety-demo-guard`) was Track B's; its work is merged.
  It can be removed: `git worktree remove ../robotics-lab-project-guard`.
- **Not in git, on this Mac only** (remade by the scripts in the README, except the key):
  - `keys/safety.key`: the secret key that signs the log and the registry. **Without it the registry and the
    logs can't be verified or changed**; keep it, and don't commit it.
  - `models/hand_v1.pt`, `hand_v2.pt` (the registry holds their fingerprints), `runs/` (training output).
  - `data/jobs/` (8 recorded v3 jobs, 19 MB), `data/hands/` (all pictures: train, train_extra,
    train_negatives, val, test; about 430 MB).
  - Raw run logs and example pictures under `results/` (summaries are committed).
- Nothing is running.
