# Steps: building the safety demo

27 November 2026

This breaks [plan.md](plan.md) section 8 into pieces of work small enough to do one at a time, and splits the
first part into two tracks that can run in two Claude sessions at the same time.

The testing (Packages 1 to 3), the manufacturer file, the report and the video come later, once the hand, the
detector and the safety layer exist.

## Order of work

| Phase | What | Track | Needs |
|---|---|---|---|
| 1 | Hand in the scene, safety zones, camera check, sample images | A | nothing |
| 1 | Safety layer around v3's arm, with a "perfect" detector | B | nothing |
| 2 | Render training and test images, train detector v1 | A | phase 1 A |
| 3 | Plug the trained detector into the safety layer, train v2 | A + B | phase 2, phase 1 B |
| 4 | Manufacturer file | either | phase 3 numbers |
| 5 | Packages 1 to 3 | either | phases 3 and 4 |
| 6 | Report and video | either | phase 5 |

## Track A: the hand and the camera (phase 1)

**Status (27 Nov): done; go given** ([camera-check.md](camera-check.md)). The person
is added without changing v3 (same bodies, geoms, cameras, masses; the same physics, tested). A new safety
camera replaces v3's overhead camera, which misses most of the warning zone.

Goal: rendered images where a hand reaching into the cell is clear enough to train a detector on.

1. **Scene wrapper.** `scene_plus.py` builds v3's scene for a seed with `scene.build(seed)` and adds the demo's
   parts without changing any v3 file. v3 builds its scene with `mujoco.MjSpec` inside `build()` and returns only
   the compiled model, so the first check is how to add bodies: export the model to XML and add to it, or wrap
   `MjSpec.compile` while `build()` runs. If neither works cleanly, stop and ask (plan.md section 3).
2. **Forearm and hand.** Capsules and boxes, moved by a mocap body so the physics can't push them around.
   Collisions with the robot are recorded as contact events (they don't push the arm).
3. **Reach paths.** Each episode draws: start time, side and angle of approach, speed, how far in, pause length,
   hand size, skin or glove colour, sleeve colour. A path is reach in, pause, pull back.
4. **Lighting changes**: dim, normal and bright, as an episode setting.
5. **Safety zones.** `zones.py`: a warning zone and a stop zone around the arm's working area, as boxes or
   distances from the arm. The truth for each frame ("hand in warning zone", "hand in stop zone") comes from the
   simulation, and becomes the label for training.
6. **Camera check.** Render from v3's `overhead` camera across the conditions. If the arm or the centrifuge hides
   part of the work area, add one fixed safety camera.
7. **Go / no-go.** A contact sheet of sample images for the user to look at. If the hand isn't clear enough, switch
   to the fallback: a person-sized shape (plan.md section 9).

Done when: the go / no-go images are reviewed.

## Track A: pictures and detector v1 (phase 2)

**Status (27 Nov): v1 trained and evaluated** ([detector-results.md](detector-results.md)): missed 21 of 840
stop-zone hands (2.5%), false alarm on 2.7% of clear pictures. Next: v2 with more hands near the robot.
Approach and decisions: [detector.md](detector.md).

1. **Record real v3 jobs** (`record_jobs.py`): 8 scenes, the state every 0.25 s; seeds 0 to 3 for training,
   4 to 7 for testing only.
2. **Render the pictures** (`dataset.py`): 1,200 training, 300 validation, 3,000 test, with boxes from MuJoCo's
   segmentation and the truth per picture; depth pictures for validation and test.
3. **Train YOLO11n v1** (`train_yolo.py`): 20 epochs, about 25 minutes on the Mac's GPU.
4. **Rule** (`box_zone.py`): box + depth -> zone. Without depth the safe rule raised a false alarm on 36% of
   clear pictures, so the safety camera became an RGB-D camera (decided 27 Nov).
5. **Evaluate** (`evaluate.py`): missed hands and false alarms per condition, 95% bounds, threshold chosen on
   validation only.

Decisions (27 Nov): YOLO11n; dataset sized to train in 20 to 30 minutes; RGB-D camera with a fixed rule; the two
limitations found (arms not detected, hands hidden under the robot) kept as documented gaps
([detector.md](detector.md#known-limitations-found-while-building-keep-for-the-report)).

## Track B: the safety layer (phase 1)

Goal: a full v3 job runs with the safety layer on. It slows and holds the arm when a hand is near, using a
"perfect" detector that reads the hand's true position from the simulation. The trained model replaces it later.
The perfect detector also stays as the baseline the trained model is compared with.

1. **Arm wrapper.** v3's step code moves the arm through `scene.step(model, data, n)` (looked up through the
   module, so it can be wrapped from outside). The wrapper can hold the arm still or slow it down between steps.
   First check: with the wrapper on and no hand, v3's whole run still gives 20 of 20, with the same results.
2. **Detector interface.** `detector.py`: one call that takes a camera frame and returns the zone (clear, warning,
   stop), a score, and the model version. The perfect detector implements it from the simulation's truth.
3. **Behaviour.** Warning zone: arm at 25% speed. Stop zone: arm holds. Clear for 1 second: carry on.
4. **Watchdog.** A late, frozen or missing frame stops the arm.
5. **Normal safety layer (no ML).** Speed limits and a work-area box, checked on every arm command, and an
   emergency stop. It always wins over the detector.
6. **Safety log.** Every decision: time, frame number, model version, score, action. Each entry carries the hash of
   the one before (a hash chain), so an edited or deleted entry shows up.
7. **Model registry.** A list of approved model files and their SHA-256 fingerprints. Loading a model not in the
   list is refused and logged.
8. **Unit tests** for each part.

Done when: a v3 job runs with a scripted hand position (a hand placed in the zone at set times, before Track A's
hand exists) and the arm slows, holds and carries on as planned.

## How the two tracks share work

- **Files.** Track A owns `scene_plus.py`, `hand.py`, `zones.py`, `cameras.py`, `render.py`, `record_jobs.py`,
  `dataset.py`, `train_yolo.py`, `evaluate.py`, `box_zone.py`, `pixi.toml` (the demo's own environment, with
  YOLO), `tests/test_scene_plus.py`, `tests/test_hand_and_zones.py`, `docs/camera-check.md` and
  `docs/detector*.md`. Track B owns `guard.py`
  (wrapper, behaviour, watchdog, normal layer), `detector.py`, `safety_log.py` and `registry.py`, and their tests.
  Neither edits the other's files. `tests/conftest.py` and `ruff.toml` (shared set-up) came with Track A; Track B
  keeps them as they are and adds to `conftest.py` only if it must.
- **Where they meet.** Track B needs only "where is the hand" and "which zone is it in". Track A's calls for that
  (merge `v3-safety-demo` into the Track B branch to get them):
  - `scene_plus.build(seed)`: v3's scene with the person; `scene_plus.patched()` makes v3's step code use it.
  - `hand.place(model, data, wrist_pos or None, quat)`: put the person somewhere (None parks them out of sight).
  - `hand.draw_reach(rng, model, data, t_from, t_to)` and `Reach.wrist(t)`: a random reach over time.
  - `zones.truth(model, data)`: the zone (`zones.Zone.CLEAR`, `WARNING`, `STOP`), how close the person comes to
    the arm's base, and whether they are in view. The perfect detector is this.
  - `zones.distance_to_robot(model, data)`: the gap between person and robot (negative: a contact event).
  - The zones: stop closer than 0.85 m to the arm's base (seen from above), warning 0.85 to 1.15 m, both up to
    1 m above the table.
- **Git.** One branch, `v3-safety-demo`, made from `v3-centrifuge-loading` (the demo imports v3's code, which isn't
  in `main` yet). The second session works in its own git worktree, on a branch made from `v3-safety-demo`, and is
  merged back when its track is done. Two sessions in the same folder would trip over each other's changes.
- **v3 stays unchanged.** Both tracks import v3's code and never edit it (plan.md section 3).
- **Machine load.** The 4-core limit is for the whole Mac, so when both sessions run heavy jobs at once, each uses
  2 workers.
