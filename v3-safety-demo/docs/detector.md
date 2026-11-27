# The hand detector: approach

27 November 2026

How the ML safety feature decides, from the safety camera, whether a hand is in the warning zone or the stop
zone. Results are in [detector-results.md](detector-results.md) once the first model is evaluated.

## In one paragraph

The safety camera is an **RGB-D camera** (colour plus depth) on the ceiling, looking straight down at the bench.
**YOLO** (the ML part) finds hands in the colour picture and draws a box around each. A **fixed rule** (no ML)
then takes the depth inside each box, works out where the hand is in space, and picks the zone from how close it
comes to the arm's base. The ML only finds hands; it never decides the zone on its own, which keeps the decision
easy to check.

## The pieces

| Piece | What it is | File |
|---|---|---|
| Pictures | the safety camera's view of real v3 jobs, with a person reaching in | `record_jobs.py`, `dataset.py` |
| Detector | YOLO11n (Ultralytics), one class "hand", fine-tuned from its COCO weights | `train_yolo.py` |
| Rule | box + depth -> point in space -> zone | `box_zone.py` |
| Check | missed hands and false alarms, per condition, with 95% bounds | `evaluate.py` |

## The pictures

- **Real arm moves.** `record_jobs.py` runs v3's whole job (Steps 1 to 5, unchanged) on 8 scenes and saves the
  scene every 0.25 s: 10,333 moments with the arm carrying tubes, the lid open, tubes in the rotor. Step 0 is
  given the right answer instead of running SAM 3 (the pictures need the arm's moves, not Step 0's).
- **The person** is drawn into a random moment: a random reach (side, speed, how far in, pause), hand size, skin
  or glove colour, sleeve colour, and lighting (dim, normal, bright). One picture in seven has nobody in it.
- **Labels come from the simulation**, not by hand. The hand's box is taken from MuJoCo's segmentation picture
  (exactly the hand's pixels the camera sees; a hand with fewer than 12 pixels showing gets no box). The zone,
  how close the person comes to the arm's base and the gap to the robot are worked out in 3D.
- **Splits never share a job:** training and validation pictures come from seeds 0 to 3, test pictures from
  seeds 4 to 7 (other tubes, other arm moves).

| Split | Pictures | Hand boxes | Clear / warning / stop | Jobs |
|---|---|---|---|---|
| Train | 1,200 | 1,027 | 611 / 243 / 346 | seeds 0 to 3 |
| Validation | 300 | 261 | 151 / 63 / 86 | seeds 0 to 3 |
| Test | 3,000 | 2,538 | 1,572 / 588 / 840 | seeds 4 to 7 |

A hand covers about 900 pixels in a picture (median), at least 24 when it shows at all.

**Why this size.** Training has to fit in 20 to 30 minutes on this Mac (Apple M5 GPU). A timing run measured
about 20 training pictures a second for YOLO11n at 640 pixels, so 1,200 pictures for 20 epochs takes about 23
minutes. The test set costs no training time, so it is larger: enough hands per condition for the Package 2
numbers (about 300 hands with no miss show a miss rate below 1% at 95% confidence).

## Why YOLO

- **Boxes, not just a verdict.** Every decision can log where the hand was in the picture, so a reviewer can
  check it. A classifier that only says "stop" or "slow" is much harder to verify.
- **Small and fast.** YOLO11n has 2.6 million weights, starts from COCO training, and checks a picture in about
  11 ms on the Mac's GPU: every frame can be checked.
- **Known.** v1's Step 0 used YOLO in this repo, and notified bodies will have seen it before.
- **Licence:** Ultralytics YOLO is AGPL-3.0. Fine for this demo; a robot maker shipping it would need
  Ultralytics' commercial licence or another detector. This belongs in the manufacturer file.

## Why the camera needs depth

From one camera looking down, height and distance mix: a hand high above the table looks further from the arm
than it is. Without depth, the only safe rule is the worst case: the hand could be anywhere from the table to 60
cm up, so take the closest that could be. Run with perfect boxes on the 3,000 test pictures:

| Rule (perfect boxes) | Hands in a zone missed | False alarm when nobody is in a zone |
|---|---|---|
| No depth: worst-case height, 0 to 60 cm | 9 of 1,428 | **570 of 1,572 (36%)** |
| Height known to ±10 cm | 10 of 1,428 | 164 of 1,572 (10%) |
| Height known to ±5 cm | 10 of 1,428 | 110 of 1,572 (7%) |

So with a plain camera, the arm would slow down for nothing about a third of the time. With depth, the rule turns
every pixel in the box into a point in space and takes the closest one to the arm's base, leaving out the bare
table around the hand. Real 3D safety cameras measure depth (stereo or time of flight); in the simulation,
MuJoCo renders it. The worst-case rule stays as the fallback for a box with no usable depth.

**Decided (27 Nov):** RGB-D safety camera, YOLO finds the hand, the rule decides the zone. The other option, YOLO
learning the zone itself (three classes: hand clear, warning, stop), was not taken: it would put the decision on
the ML alone.

## Known limitations (found while building; keep for the report)

These are real gaps of this design, written down so that the manufacturer file and the Package 1 review deal
with them openly.

1. **Only hands are detected, not arms.** The zones count any part of the person. In 6 of the 3,000 test
   pictures the forearm or upper arm crosses into the warning zone while the hand stays outside it (a reach that
   comes in at a slant), and a hand detector cannot see that. All 6 are warning-zone cases, none in the stop zone.
   Options if it matters: a second class "arm" in YOLO, or zones defined for hands only (and said so in the
   manufacturer file). **Decided (27 Nov):** keep it as a documented limitation.
2. **Hands hidden under the robot.** In 3 of the 3,000 test pictures a hand in the stop zone is under the robot
   arm and almost invisible from above (1 to 4 pixels showing). No detector can find it in that picture. Track
   B's behaviour covers the common case: the arm only carries on after the zone has been clear for 1 second, so a
   hand that is seen and then slips under the robot keeps the arm held. A hand that is never seen at all is the
   camera's limit; a second camera from another side would close it.
3. **One straight-down camera.** Without depth the zone can't be judged (above). With depth, a hand right above
   another object (a rack, the robot) can make the box include that object's points; that is cautious (more stops,
   not fewer) and is measured in the results.
4. **Simple shapes and simulated sensing.** The hand is capsules and a box; lighting, depth and colours are
   rendered, not measured. The report lists real-camera effects as not tested (plan.md section 6).
