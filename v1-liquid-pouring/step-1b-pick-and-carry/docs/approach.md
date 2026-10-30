# Step 1b Approach: Both Arms Pick and Meet at the Pouring Point

This page has the exact settings, and why each was chosen. The overview is in
[step-1b-pick-and-carry.md](../../docs/solutions/step-1b-pick-and-carry.md); how well it works is in
[results.md](results.md).

Everything runs in the **task's own scene** (`lab_scene.scene`): three racks in a row along y, three source
tubes (100, 150, 200 mm, 60 to 90% full) in the source rack (+y side), three empty receivers of the same
sizes in the empty rack (-y side), both arms at home. Both tubes are taken **from the side**, with Step 1's
side-grasp network, so their openings stay free for the pour.

---

## The whole run

```
 PERCEIVE (coordinator)
   overhead picture ──► Step 0 (YOLO11-seg) ──► racks and tubes, where, how high
                        Step 0.1 ── size of each tube, from its rim height
                        close-up ── liquid or not, looking straight down into each tube
                        roles ───── source rack / empty rack / destination rack

 CHOOSE (coordinator, before anything moves)
   for each size, in the order asked: a receiver and a source of that size
     side-grasp network on each close-up ──► the clear directions, best first
     the arm's own check of each           ──► the first direction whose whole pick passes
     both pass? ── no ──► next size

 MOVE (one arm at a time)
   Arm B (arm 1): pick the source ──► "picked" ──► park it behind the source rack ──► "waiting"
   Arm A (arm 2): plan its pick again ──► pick the receiver ──► "picked"
                  ──► carry to the pouring point ──► hold still ──► "ready"
   Arm B:         carry to the waiting spot ──► find the closest safe distance ──► move in ──► "in place"
```

---

## 1. Perceive

Unchanged from the first version: Step 0's YOLO11-seg (as trained) on the overhead picture at 1280 x 960,
sizes from rim heights, liquid from the wrist camera looking straight down into each tube (an empty tube
shows its bottom, a source its liquid surface more than half-way up), and roles from what each rack holds.

---

## 2. Choose the pair

1. For the first size (in the order asked) with a receiver and a full source: the **side-grasp network**
   scores each tube's close-up (35 cm above the rim). At its best pixel it ranks the 24 directions; those
   under its cut-off (0.10, chosen on Step 1's validation pictures) are dropped.
2. **The arm's own check**, for each remaining direction, best first: the whole pick is planned from where
   the arm is now (up and across with the fingers down, turn them level above the tube, down beside it,
   in) and every pose is checked with the **whole arm**:
   - nothing touched, with a 5 mm margin;
   - the arm never folds onto itself (a link on its own gripper) or touches the table;
   - no joint jumps more than 0.15 rad between two points 5 mm apart;
   - the closing fingers (25 to 100% closed, shifted 3 mm either way, and the pose they take holding a
     full tube) keep **2 cm** from every other tube: the passive finger linkage swings outwards under load.
3. The first direction that passes is used, and **its plan is kept and followed exactly**. If none passes,
   the tube counts as boxed in, and the next size is tried, before anything moves.

**Each arm takes the tube on its own side:** arm 1 rests on the +y side, next to the source rack, so it is
**Arm B** (the source); arm 2 rests on the -y side, next to the empty rack, so it is **Arm A** (the
receiver). The other way round, each arm's routes crossed the other's side of the table.

---

## 3. Move

| Setting | Value | Why |
|---|---|---|
| Every move | a straight line in space, cut into 5 mm pieces, each solved for the arm's joint angles, checked before moving | joint angles slid straight to a goal swept the gripper through a rack (Step 1) |
| Turning the gripper round its tube while moving | tried six ways: the short or long way round while moving, or move then turn on the spot, or turn on the spot then move; the first the arm can follow is used | turning while moving can need a huge wrist swing somewhere on the way |
| Speed | 0.25 m/s empty-handed, 0.12 m/s carrying, 0.05 m/s near a tube; every joint at most 0.8 rad/s; a smooth start and stop | a turn of the wrist covers little distance but a lot of angle |
| Arriving | after each move the arm is given up to 2 s to catch up with its targets | otherwise the next move starts from where the arm should be, not where it is |
| Empty-handed height | 45 cm, or 12 cm above the tallest rim | the gripper, turned level, clears everything |
| Carrying height | the held tube's bottom 4 cm above the tallest rim | |
| Grip | as in Step 1: 3 cm under the rim; the pinch point 11.5 mm past the tube's axis; fingers told 4 mm under 25 mm | |
| Aiming a held tube | its axis sits **15 mm behind the pinch point** | measured over 30 runs: 14 mm empty, 16 mm full (the tube settles back as the fingers close) |

### The pouring point

| | Value | Why |
|---|---|---|
| Receiver's rim | above the middle of the table, **35 cm** up | held from the side, each wrist reaches out over a rack; at 25 cm it would hit the tall tubes |
| Receiver held from | **-y** (arm 2's side) | |
| Source held from | **+y** (arm 1's side), its rim 2 cm higher, along the line between the arms | with the grippers on opposite sides, neither covers the other tube's opening; tipping the source towards the receiver is then a roll of its wrist |
| Closest distance between the tubes' axes | worked out once Arm A is still: tried from 3 cm up, plus 1 cm; it comes to **85 mm** | the two grippers' finger links reach about 7 cm along the line between the tubes |

---

## 4. One arm at a time

The coordinator runs both arms' scripts and the simulation together, but **only one arm moves at a time**:

1. **Arm B** picks the source (Arm A is still at home), and parks it behind the source rack on its own
   side, its gripper turned outwards: "picked", "waiting".
2. **Arm A** plans its pick again against the scene as it is now (Arm B parked), trying the network's
   directions in order; picks the receiver; carries it to the pouring point; holds still: "ready".
3. **Arm B** carries the source to the waiting spot beside the middle, works out the closest safe distance,
   and moves in: "in place".

**Why:** held from the side, both arms reach far across the table. With both moving at once, a move checked
when it started could meet the other arm moving into its way, and in testing the arms touched. Each move is
now checked against an arm that is standing still. If a move is blocked only by the other arm or the tube it
holds, the arm waits (checking again every 0.2 s, for up to 8 s); anything else in the way stops the run with
the reason.

---

## 5. What the simulator is used for, and what not

The arms decide only from cameras and their own joint readings. The simulator's own answers are used only to
grade each run, and to know which body the arm holds, for the collision checks.

---

## 6. How it is graded

Every one of 10 task scenes is run three times, each time asking for a different size first (30 runs).
A run is a **success** when both tubes are in place, of the same size, from the right racks, each rim within
5 mm of where it should be, both within 5 degrees of upright, no ball lost, and neither arm touched anything
but its own tube.

---

## Files

| File | What it does |
|---|---|
| `carry/perceive.py` | Step 0 on the overhead picture, sizes, liquid, roles |
| `carry/look.py` | the straight-down close-up camera |
| `carry/motion.py` | straight-line moves, fingers down or level; solving and checking them (closing fingers too) |
| `carry/episode.py` | the coordinator, both arms' scripts, grading one run |
| `carry/evaluate.py` | the 30 runs and their summary |
| `carry/figures.py` | the pictures in these docs |
| `tests/test_carry.py` | perception, choosing the pair, straight and checked moves, one full run |
