# Step 1b: Both Arms Pick and Meet at the Pouring Point

The second part of Step 1 from the [list of steps](steps.md). [Step 1](step-1-grasp.md) taught the arm
how to take one tube; here both arms take a matching pair and bring them to the pouring point.
It follows the [problem statement](../problem-statement/problem-statement.md).

The code is in [`step-1b-pick-and-carry/`](../../step-1b-pick-and-carry/). Two more pages go deeper:

- [approach.md](../../step-1b-pick-and-carry/docs/approach.md): every setting, and why.
- [results.md](../../step-1b-pick-and-carry/docs/results.md): how well it works.

---

## Goal

Starting from nothing but camera pictures:

1. find the racks and tubes, how big each tube is, and which ones hold liquid;
2. choose a receiver and a source **of the same size**;
3. **Arm A** picks the receiver and holds it upright and still at the **pouring point** (above the middle
   of the table, 35 cm up);
4. **Arm B** picks the source and brings it in beside the receiver, ready for the pour (Step 2).

Both tubes are held **from the side**, with Step 1's side-grasp network: held from above, a tube's opening
is under the gripper's base, and nothing can be poured in or out.

## How it fits together

```
 overhead camera ──► Step 0 (YOLO, as trained) ──► racks and tubes: where, how high
                       + size (Step 0.1), + liquid (wrist camera, straight down), + roles
                                   │
                          COORDINATOR chooses the pair
                          (side-grasp network + the arm's own check, for each tube)
                                   │
   Arm B (arm 1, on the source rack's side)       Arm A (arm 2, on the empty rack's side)
   pick the source, park it on its side
   say "waiting"  ─────────────────────────────►  plan again, pick the receiver
                                                  carry to the pouring point, hold still
   come in beside it, say "in place"  ◄─────────  say "ready"
```

## Who decides: one coordinator

The two arms never talk to each other. **One coordinator** holds the list of what is on the table, gives
each arm its job, and listens for each arm's short messages ("picked", "waiting", "ready", "in place").

- **The pair is chosen before anything moves.** For the size asked for, the coordinator runs the side-grasp
  network on each tube's close-up and checks the network's directions with the whole arm; if either tube has
  no safe way in, it tries the next size.
- **One arm moves at a time.** Held from the side, both arms reach far across the table. Arm B goes first and
  parks the source on its own side; then Arm A picks and holds the receiver at the pouring point; then Arm B
  comes in. With both moving at once, the arms touched in testing.
- **Each arm takes the tube on its own side:** arm 1 rests next to the source rack, arm 2 next to the empty
  rack.

## Result in short

| | |
|---|---|
| Runs | 30 (10 scenes x 3 sizes asked for) |
| Successes | **14 of 30**: medium tubes 13 of 17, large tubes **0 of 12** |
| Liquid lost | none, in any run |
| In the successful runs | receiver within 0.8 mm (median), source within 3.6 mm, both under 3.5 degrees from upright |
| Time until Arm B is in place | about 67 s |

**The medium pair works; the large pair does not yet.** A tall tube has to be held high to clear the other
tubes, with 17 cm of it hanging below the fingers, and at those heights the arms cannot find routes they can
follow smoothly: a joint would have to swing a long way, or a pose is out of reach, or the tube would hit the
arm's own forearm. The planner refuses those moves before making them. Small tubes are rarely reachable from
the side between taller neighbours, so the coordinator usually takes the medium pair instead.

## How it got here

The first version held the tubes **from above** and succeeded 24 times in 30, but a tube held from above
cannot be poured: its opening is under the gripper's base. Moving to side grasps meant a second network
(Step 1) and most of this step's work again: one arm at a time, each arm on its own side, the pouring point
raised from 25 to 35 cm, the closing fingers checked with room to spare, and the held tube checked against
its own arm. Details in [results.md](../../step-1b-pick-and-carry/docs/results.md).

## Simplifications

- **The close-up camera** is on the gripper's axis, looking straight down, with the arms at home.
- **Which body the arm is holding** comes from the simulator, for the collision checks only.
- **A failed pick is not retried**; the run stops and reports why.

## What is next

- **Large tubes:** lower carrying heights where possible, and better starting poses for the arm's joint
  search at high, level-finger poses.
- **Step 2, the pour:** Arm B tips the source towards the receiver (a roll of its wrist), watched by a camera,
  until the receiver is 50% full.
