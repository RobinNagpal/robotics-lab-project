# Plan: a sample ML safety evaluation on the v3 centrifuge simulation

26 November 2026

## 1. Goal

Build one sample of the work we offer notified bodies, so they can see it rather than read about it.

We add one ML safety feature to the v3 centrifuge simulation: a camera model that spots a person's hand reaching
into the robot's area and slows or stops the arm. Then we test it with our three packages, the way a notified body
would ask us to, and write a sample test report in the format we promise on
[dodao.io/ml-safety-evaluation](https://dodao.io/ml-safety-evaluation).

The report and a short video go into the package for notified bodies, next to the 3-page PDF.

## 2. Why v3 is a good fit

- **The task is scripted.** v3 finds the tubes with the camera (SAM 3 in Step 0), then runs a fixed, programmed
  workflow (Steps 1 to 6, 20 of 20 jobs done end to end). So the only ML that keeps people safe will be the one we
  add. That is the exact case the new rule covers: Annex I, Part A, item 5 (an ML component that ensures a safety
  function).
- **It gives a real scope question.** SAM 3 in Step 0 is also ML, but it only finds tubes, so it is a task function,
  not a safety function. The scope check in Package 1 has to say that, which is what notified bodies need to see.
- **It is a realistic setting:** a lab bench where a technician may reach in next to a moving arm.
- **It is fast enough.** MuJoCo can run the thousands of episodes we need on this Mac.

## 3. Rule: v3 is not changed

All new work lives in `robotics-lab-project/v3-safety-demo/`. It imports v3's scene and step code and runs them as
they are.

- **Scene:** v3's `scene.build(seed)` returns a compiled model. We export that model, add our parts (hand, safety
  camera if needed), and rebuild it inside the demo folder.
- **Stopping the arm:** v3's scripts move the arm through the simulation step function. The demo wraps that function
  so the safety layer can slow or hold the arm between steps. v3's scripts run unchanged.
- If either of these turns out to be awkward, we stop and ask before touching any v3 file. The fallback is one small
  hook in v3 that is off by default.

## 4. What we build

### 4.1 A simulated hand and arm

- A forearm and hand (simple shapes first) driven along random paths: reach in, pause, pull back.
- Random choices per episode: start time, approach angle, speed, hand size, glove or skin colour, sleeve colour,
  how far it reaches.
- Contact between the robot and the hand is recorded as a "contact event". We also record the smallest distance
  between them in every episode.

### 4.2 The ML safety feature

- **Camera:** start with v3's fixed `overhead` camera. Add one fixed safety camera only if the overhead view misses
  part of the work area.
- **Model:** a small detector that reports "hand in the warning zone" and "hand in the stop zone". We start from a
  small pretrained model (for example a MobileNet-based detector from torchvision) and fine-tune it on images
  rendered from the simulation. It runs in v3's existing `sam` environment (PyTorch, no NVIDIA GPU needed).
- **Behaviour:**
  - hand in the warning zone: the arm slows to 25% speed;
  - hand in the stop zone: the arm holds where it is;
  - the zone is clear for 1 second: the arm carries on.
- **Watchdog:** if the camera image is late, frozen or missing, the arm stops. That is the safe state.

### 4.3 A normal safety layer (no ML)

- Speed limits and a work-area box, checked on every arm command.
- An emergency stop that works in every step of the job.
- This layer always wins over the ML. An ML failure must never switch it off.

### 4.4 A safety log and model versions

- Every safety decision is logged: time, camera frame number, model version, model score and action taken.
- Each entry carries a hash of the one before it (a hash chain), so an edited or deleted entry shows up.
- Each model file gets a fingerprint (SHA-256) in a small registry. The system refuses a model that isn't in the
  registry.
- We train two versions, v1 and v2 (v2 with more data), to test updates and rollback.

### 4.5 A short "manufacturer file"

Two or three pages, written as if we were the robot maker:

- what the feature does;
- its zones and limits in numbers;
- its risks, such as a missed hand, a false stop, or the model getting worse over time;
- the training and test data;
- the accuracy target;
- how updates are released.

Package 1 reviews this file. We leave a few gaps in it on purpose, so the review has something real to find.

## 5. What we test

Each test maps to one of our three packages and to the clause it supports.

### Package 1: technical file review

| Test | What we check |
|---|---|
| Scope check | Lists every ML function: SAM 3 (task, finds tubes) and the hand detector (safety). Writes a scope memo saying items 5 and 6 apply to the detector only. |
| File review | Checks the manufacturer file against Annex IV points (b), (f), (g), (n). Output: a gap list. |

### Package 2: data, performance and robustness

| Test | What we check |
|---|---|
| Data review | The training data covers every condition in the file (light, glove colour, hand size, angle). Test images never appear in training. Label errors on a sample of 200 images. Data version recorded. |
| Performance | The frozen model on a held-out test set. Missed hands and false stops reported separately, for each condition, each with a 95% confidence bound. With zero misses, about 300 hand images per condition are needed to show a miss rate below 1% (rule of three). |
| Robustness | Dim and very bright light, the arm blocking part of the view, camera noise, dropped frames, 200 ms delay, a frozen image, a dead camera. Pass: the arm slows or stops every time. |

### Package 3: the ML safety requirements

| Test | Clause | Pass |
|---|---|---|
| Movement limits: send commands that are too fast or leave the work-area box, during each step of the job | Annex III 1.2.1 | The normal layer blocks every one |
| Hand in the path during real jobs: Steps 1, 3 and 4, at many random moments | Annex III 1.2.1 | No contact event; smallest distance and stop time recorded |
| Safety log: rebuild decisions from the log, then edit and delete entries | Annex III 1.2.1 | Every decision rebuilt; every change detected |
| Emergency stop in every step; rollback from v2 to v1 | Annex III 1.2.1 | Stops every time; rollback works |
| Separation: crash the detector, make it output rubbish, slow it down | Annex III 1.2.1 | The normal layer and the watchdog still stop the arm |
| Update: run v2 through the same full test set before release | Annex VII (type changes) | Released only if it is no worse than v1 |
| Unapproved model: try to load a model that isn't in the registry | Annex III 1.1.9 | Refused, and the attempt is logged |

## 6. What we don't test

These need hardware, so they stay out:

- real stopping time and distance;
- contact force on a body;
- real camera and lighting effects;
- production checks.

The report lists them as "not tested: needs the real machine".

## 7. What we deliver

1. **Sample test report**, about 6 to 8 pages, in the format on our page:
   - cover with model version, model checksum and data version;
   - scope memo;
   - gap list;
   - results for each requirement, with pass or fail and confidence;
   - deviations;
   - what was not tested.

   It states clearly that this is our own system, in simulation only, and shows our method, not an independent
   assessment.
2. **Short video**, 30 to 60 seconds: the hand reaches in, the arm slows and holds, the log entry appears.
3. **Code** in `v3-safety-demo/`, so every number in the report can be rerun.

## 8. Steps

| # | Step | Rough time | Done when |
|---|---|---|---|
| 1 | Hand and arm in the scene, safety-zone layout, overhead camera check | 2 to 3 days | Rendered images where the hand looks clear enough to train on. **Go / no-go point** (see risks) |
| 2 | Render the training and test images, train detector v1 | 3 to 4 days | v1 meets the accuracy target on held-out images |
| 3 | Safety behaviour, watchdog, normal safety layer, log, model registry; train v2 | 2 to 3 days | A full v3 job runs with the safety layer on and the hand reaching in |
| 4 | Write the manufacturer file | 1 day | File ready for review, with its planted gaps |
| 5 | Run Packages 1 to 3 | 3 to 4 days | Every test in section 5 has a result |
| 6 | Report and video | 2 days | Report reviewed by the manager |

Total: about 3 weeks for one person. This is a rough guess; we will confirm it after step 1.

Machine limits (from the project rules): at most 4 CPU cores for rendering and test runs. Any job longer than about
10 minutes is announced first, with its expected time.

## 9. Risks and fallbacks

| Risk | Fallback |
|---|---|
| A simple-shapes hand doesn't look real enough to train a detector on | Use a person-sized shape entering the cell (a body, not a hand). All tests stay the same. |
| The detector runs too slowly to check every frame | Check at 10 frames per second and record that rate in the file and the report |
| Wrapping v3's scene or step function is awkward | Ask before adding one small, off-by-default hook in v3 |
| Rendering thousands of images takes too long on 4 cores | Lower the image size, and render once and reuse the images |

## 10. Decisions for the manager

1. Go ahead with this plan and the 3-week estimate?
2. Is the fallback (a person-sized shape instead of a hand) acceptable if step 1 fails?
3. Is the sample report only for notified bodies, or also public on our page?
4. Should we also use this work for one of the articles in the package (for example, "How to test an ML safety
   function against Annex III 1.2.1")?
