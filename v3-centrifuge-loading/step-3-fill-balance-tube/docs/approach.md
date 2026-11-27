# Step 3 Approach: Fill a Balance Tube

**Status: built (programmed).** The overview is in [steps.md](../../docs/solutions/steps.md#step-3-fill-a-balance-tube).
How well it works: [results.md](results.md).

## Goal

For each balance tube in the plan (Step 2): an empty open tube, filled with water until it weighs what the plan
says, within **0.5 g** (one balance tube) or **0.3 g** (two balance tubes, whose errors could add up).

## Words used here

| Word | Meaning |
|---|---|
| **Balance tube** | an empty tube that is filled with water to match a sample's pull, so the rotor balances |
| **Water tube** | an open tube of water the arm pours from. There are two, one for each balance tube |
| **Lip** | the point of the water tube's rim nearest the balance tube: where the water leaves |
| **Tilt** | how far the water tube is tipped from upright, in degrees: 90° is lying flat |
| **Side grip** | the fingers come in level from the side of a tube, below its rim, so its opening stays free |
| **Burst** | the "water" is small balls, which jam in the tube's neck and then let go several at once |
| **Settled reading** | the balance's reading once nothing is falling: it is then exact |

## The moves, for one balance tube

```
 empty tube (held from above):  supply rack ──► balance holder ──► let go ──► read its empty weight
 water tube (side grip):        supply rack ──► tipped 60°, above the balance ──► down to its pour place
                                ──► pour in rounds, watching the balance (below) ──► tipped back to 60°
                                ──► up, upright ──► back to its cell
 read the balance again (the arm is away)
 balance tube (held from above): holder ──► back to its cell (Step 4 takes it from there to the rotor)
```

With two balance tubes, the second one's target is worked out again from what the first really weighs: two
balance tubes can cancel each other's errors, so the second makes up for the first (Step 2's solver, one weight
left to choose).

## What the arm uses

The bench layout (where the cells and the balance holder are), which cells hold the empty tubes and the water
(Step 0), the target weight (Step 2) and the balance's display. It never reads how many balls are where, nor
where a tube really sits in its fingers.

## The second water tube

The plan first had one water tube. It holds 126 balls (41 g). One balance tube needs up to 106 balls, but the
"mismatch" job's two balance tubes need up to 198 together: 24 of its 25 test seeds needed more water than one
tube holds. Even the solver's lowest-water hole layouts left about half of them short. So the supply rack now
has **two water tubes**, in the corner cells 0 and 3 (the cells beside them empty, for the side grip), and the
two empty tubes in cells 5 and 6, the only cells left. The samples are drawn from the seed before the supply
rack, so every sample's weight is unchanged.

## How the water tube is held to pour

This took most of the work. Four things had to be true at once:

1. **The tube must not turn in the fingers as it is tipped.** The grip plan's grip, fingers from behind (the side
   away from the balance tube), puts the two pads on the tube's left and right: exactly on the line it is tipped
   about. The tube then turns between the pads like a door on its hinge, held only by friction. In the first
   test the full tube stayed nearly upright while the hand tipped to 113°, and all 126 balls missed. The fingers
   must come **from the side**, the pads on the tube's front and back, so tipping presses the tube into them.
   15° off the side was not enough (the tube turned 4°, the lip ended 6 mm to the side): squarely from the side.
2. **The fingers must clear the balance tube.** Held from the side, the front finger is near the balance tube's
   rim. Held 35 mm under its rim, the finger touches it; **40 mm** under the rim clears it from 60° of tilt on.
   45 mm would be better at the pour, but then the hand hits the rack's top plate when it picks the tube.
3. **The wrist must stay away from its singular pose.** The UR5e's wrist has a pose where two of its joints line
   up and a small turn of the hand needs a huge turn of a joint. Pouring away from the arm's base with the hand
   from the side puts the fingers right along those joints. So the water is poured **sideways**, from the racks'
   side of the balance tube towards it: the fingers then point back towards the arm's base, far from that pose.
4. **The tube is tipped about its lip**, which stays **20 mm above the balance tube's rim and 4 mm short of its
   middle** (v1's lesson: aim at the rim; nearer the middle the balls shoot past it).

Picking it from the rack, the fingers come in level from the side with most room (tried every 15° round), the
equal best nearest "away from the arm's base". Every move is checked for collisions before the arm moves,
including the hand against the arm's own links (a turned hand can fold the arm into itself, which Step 1's
fingers-down moves never do).

## The pour

**What the bench test showed** (`tilt_study.py`, tipping at a steady 0.7° a second): nothing leaves until about
82°, not the 50° a liquid would need; then the balls come in bursts of 10 to 20 (3 to 7 g) as the pile in the
neck jams and lets go, and the tube is empty by about 92°. While balls land, each impact kicks the reading by up
to 5 g; within 0.2 s of a burst ending the reading is exact again. One ball is 0.33 g, so the precision asked
for is about one ball.

**So the pour goes in rounds:**

1. Tip fast to 70°, then **creep** at 0.7° a second (0.3° a second when less than 1 g is still to add).
2. When the reading jumps 0.25 g above the last settled reading, a ball has landed:
   - **much still to add** (4 g or more): hold the tilt and let the burst run, unless the reading comes within
     2 g of the target, then tip back;
   - **little still to add:** tip straight back 8°, fast.
3. Wait for the reading to settle, and read it.
4. Done when the reading is within half a ball of the target. Otherwise start the next round from 2° short of
   where the flow last started (the flow starts a little further over each time, as the water tube empties).

Water cannot be taken back out, so the rounds aim to arrive from below. A round that brings two balls when one
was wanted ends 0.33 g over; with two balance tubes the second one's new target makes up for it.

## The balance tube's grip

Held from above by its body, the pinch point 25 mm under the rim, squeezed 1 mm: the 38 mm pads then lie along
the straight body. Held 6 mm under the rim (the grip Step 4 needs in the rotor), they touch only its top 15 mm;
a full tube pivoted in the fingers when it rubbed the rack's hole (0.9 mm wider than the tube), and stood back in
its cell 4.5° off upright. Step 4 will take it from the rack with the grip the rotor needs.

## The cap (the whole run)

Real labs cap every tube before spinning it, so in the whole run (Step 6) Step 3 caps each balance tube once
it is filled (`fill(..., cap=i)`, code in `cap.py`). The water goes to the plan's weight less the cap's 2 g
(the maker's figure). Once the water tube is back in its cell, the arm:

1. takes a spare cap off its peg on the balance's base (fingers down, as it grips a sample's cap);
2. brings it down onto the balance tube in the holder, turning it half a turn over the last 6 mm;
3. opens, rises, reads the balance again (now with the cap: this is the reading the report uses);
4. carries the capped tube back to its cell by the cap, as Step 1 puts a sample back.

Step 4 then loads it like a sample, by its cap. The thread is not simulated: see the simulation environment's
notes ([simulation-environment.md](../../simulation-environment/docs/simulation-environment.md#6-the-liquid-v1s-balls)).

The whole run also stops a pour once it is within 0.4 g under its target (`Filler.close_enough_g`). Step 3's own
evaluation still stops only within half a ball (0.16 g). The last ball is the riskiest round: on seed 13 with
the cap, a round meant to add one ball (0.35 g short) brought 10 (3.3 g over). Step 3 allows 0.5 g either way.

Tested so far on scene 13 only: 3.60 g of water (target 3.95 g), 16.97 g with the cap (target 17.32 g),
loaded by its cap and seen seated by the rotor camera. Not yet run on all 20 scenes.

## What is reused

- Step 1's arm: inverse kinematics, the straight moves checked for collisions, reading the balance.
- v1's pour: tipping about the lip, aiming at the rim, creeping slowly, tipping back early.
- Step 2's solver, for the second balance tube's new target.

## How it is measured (`results.md`)

On 20 seeds that need balance tubes (the odd seeds 1 to 39: 10 with one balance tube, 10 with two), graded
with the simulator's true weights: each balance tube against its target; the rotor's imbalance with every tube's
true weight in its planned hole (within 1.0 g); balls spilled; every tube back in its cell; time.

    cd step-3-fill-balance-tube
    P="pixi run --manifest-path ../simulation-environment/pixi.toml"
    PYTHONPATH=$PWD:$PWD/../step-1-weigh:$PWD/../step-2-plan:$PWD/../simulation-environment \
        $P python evaluate.py --seeds 20 --workers 4
