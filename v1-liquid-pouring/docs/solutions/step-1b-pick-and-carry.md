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
| Successes | **21 of 30** (version 1: 14): medium tubes 13 of 17, large tubes **7 of 12** (version 1: 0) |
| Liquid lost | in 2 runs (one scene asked twice: both tubes fell over at the meeting) |
| In the successful runs | receiver within 1.1 mm (median), source within 3.6 mm, both under 4.4 degrees from upright |
| Time until Arm B is in place | about 68 s |

**Medium and large pairs work now.** Version 1 checked only the pick, and a tall tube's carry then failed
halfway: held high to clear the other tubes, with 17 cm of it below the fingers, the arm met poses where a
joint would have to swing a long way. Version 2 **plans the whole carry before choosing a grasp**, and Step
1's network (version 3) offers each direction at three grip heights. **Small tubes** still fall back to the
medium pair: gripped from the side 8 cm above the table, the arm's wrist would be down among the tall
neighbours. And in 6 runs a tube came out of the fingers or fell over, which version 1 never did.

## How it got here

The first version held the tubes **from above** and succeeded 24 times in 30, but a tube held from above
cannot be poured: its opening is under the gripper's base. Moving to side grasps meant a second network
(Step 1) and most of this step's work again: one arm at a time, each arm on its own side, the pouring point
raised from 25 to 35 cm, the closing fingers checked with room to spare, and the held tube checked against
its own arm (14 of 30). Then version 2: Step 1's side network with grip heights, the fingers coming in barely
open, and the carry planned before choosing (21 of 30). Details in
[results.md](../../step-1b-pick-and-carry/docs/results.md).

## Simplifications

- **The close-up camera** is on the gripper's axis, looking straight down, with the arms at home.
- **Which body the arm is holding** comes from the simulator, for the collision checks only.
- **A failed pick is not retried**; the run stops and reports why.

## What is next

Three open problems; [results.md](../../step-1b-pick-and-carry/docs/results.md) ("Open problems and
suggested fixes") explains each with the fixes suggested.

- **Tubes slipping out or falling over** (6 runs): measured. The tube is not too heavy; it swings out
  about the line between the two fingertip pads, most easily when one pad is left tipped onto its edge.
  Squeezing harder made it worse (0 of 8); closing twice moved the failures to other runs (3 of 8 either
  way). Next: let the simulated pads resist twisting, as real rubber pads do, and move more gently.
- **Small tubes** (9 of 10 fall back): the arm's wrist would be down among the tall tubes. Suggested: pick a
  small tube from above, stand it in an empty spot, and take it again from the side.
- **No scenes held back for grading:** version 2 was tuned on the same 10 scenes it is graded on. Grade on
  20 new scenes as well.
- **Then Step 2, the pour:** Arm B tips the source towards the receiver (a roll of its wrist), watched by a camera,
  until the receiver is 50% full.
