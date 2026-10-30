"""Side grasps: the gripper comes in sideways, fingers level, so the tube's top stays free for pouring.

A top-down grasp (the rest of this package) leaves the tube's rim 2.5 cm under
the gripper's base, with only a 14 mm slot between the finger links above it:
fine for carrying, useless for pouring. From the side, the gripper's body is
beside the tube and the opening is clear.

A side grasp is described by its **direction**: where the gripper comes from,
seen from above, measured like a turn (from the table's x towards its y). The
gripper's body sits on that side of the tube; unlike a top-down grasp, turning
it by half a turn is a different grasp, so directions run all the way round:
24 of them, 15 degrees apart.

The way in, for every direction:

1. come down **beside** the tube, 6 cm out from its axis, to the grip height;
2. move straight in until the fingers are around the tube;
3. close.

The fast check (SideChecker) tries all three with the open, then closing,
gripper, as scenes.Checker does for top-down grasps.

Version 2 (docs/side/version2.md) changes three things, each switched on by an
argument so that version 1 (runs/side_grconv, used by Step 1b) stays as it was:

- the fingers come in opened only a little wider than the tube (PRESHAPE_GAP),
  not fully open, so they fit between the tube and its row neighbours;
- the 24 directions are counted from the rack's own sides (``frame_yaw``, from
  rack_yaw), so four of them point straight out of the rack;
- the network's score for a direction is mixed with its two neighbours' before
  choosing (smoothed), so the arm aims at the middle of a run of clear
  directions rather than its edge. The camera is the same
close-up from above, set higher (35 cm over the rim) so the picture covers the
gripper's whole way in: everything on the table is an upright tube or a flat
rack, so a height map from above shows all that can block a sideways approach.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np
from lab_scene import kinematics, liquid
from lab_scene import scene as lab
from lab_scene.settings import RACK

from . import rules, scenes, trial

BINS = 24
STEP = 2 * math.pi / BINS
# The close-up for side grasps: higher than for top-down ones (25 cm).
LOOK_HEIGHT = 0.35
# The fingers close 3 cm under the rim, as for top-down grasps. The finger
# links reach 2 cm above and below the pinch point, so the top centimetre of
# the tube and its opening stay clear. Never lower than 2 cm above the rack's
# top plate.
BELOW_RIM = 0.03
# Version 3: the grip heights the network scores, as distances below the rim. Gripping lower never
# makes a tube easier to pick (the fingers come down among the neighbours), but a tall tube held 3 cm
# under its rim hangs 17 cm below the fingers, and carried over the racks the arm then works near
# its wrist's awkward pose (docs/side/version3.md).
HEIGHTS = (0.03, 0.05, 0.07)
ABOVE_RACK = 0.02
# Coming down beside the tube: this far out from its axis, from this far above the grip height.
OUT = 0.06
# Closed on a 25 mm tube, the Robotiq pads sit this far behind the pinch point (towards the
# gripper's base): the finger linkage moves in an arc. So the pinch point goes this far past
# the tube's axis, and the tube ends up centred between the pads.
DEPTH = 0.0115
DROP = 0.12
LIFT = rules.LIFT
# The most any joint may change between two points 2 cm apart on the way in.
MAX_JUMP = 0.5
# The arm turns its fingers level this high up, then moves across at this height.
TRANSFER = 0.45
# Version 2: the gap between the pads on the way in. The tube is 25 mm, so 5 mm of room each
# side; fully open, the pads are 85 mm apart and land on the tubes beside it in its row (36 mm apart).
PRESHAPE_GAP = 0.035
# Version 2: rack top-plate pixels in the close-up are this high above the table, m.
PLATE = (RACK.height - 0.012, RACK.height + 0.012)
# The fewest pixels of plate that count as a rack (a whole rack's plate is about 1,000).
MIN_PLATE = 300


def direction(k: int | np.ndarray) -> float | np.ndarray:
    return k * STEP


def grip_height(top: float, below: float | None = None) -> float:
    """The pinch point's height: ``below`` the rim (default BELOW_RIM), never under rack top + 2 cm."""
    return max(top - (BELOW_RIM if below is None else below), RACK.height + ABOVE_RACK)


def along(a: float) -> np.ndarray:
    """The unit vector from the tube towards where the gripper comes from."""
    return np.array([math.cos(a), math.sin(a), 0.0])


def frame(a: float) -> np.ndarray:
    """The gripper's axes as columns: fingers level and opening sideways, pointing at the tube."""
    z = -along(a)  # the gripper's fingers point along its own z
    y = np.array([0.0, 0.0, 1.0])
    return np.stack([np.cross(y, z), y, z], axis=1)


def pinch_at(xy: np.ndarray, a: float, top: float, below: float | None = None) -> np.ndarray:
    """Where the pinch point goes to hold the tube whose axis is at ``xy`` from direction ``a``."""
    return np.array([xy[0], xy[1], grip_height(top, below)]) - DEPTH * along(a)


def way_in(
    xy: np.ndarray, a: float, top: float, step: float = 0.005, below: float | None = None
) -> list[np.ndarray]:
    """Pinch points along the way in: down beside the tube, then straight in."""
    end = pinch_at(xy, a, top, below)
    out = end + (OUT + DEPTH) * along(a)
    down = [out + [0, 0, h] for h in np.arange(DROP, -1e-9, -step)]
    inward = [out - along(a) * d for d in np.arange(step, OUT + DEPTH + 1e-9, step)]
    return down + inward


def pad_centres(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    fingers: np.ndarray,
    closed: np.ndarray,
    prefix: str,
    closing: float,
) -> float:
    """How far apart a gripper's two pad centres are with its fingers ``closing`` of the way shut.

    ``fingers`` are the gripper's finger joints' qpos addresses, ``closed`` their angles closed on a
    tube (scenes._closed_fingers), ``prefix`` the gripper's geom names' prefix. Leaves the fingers there.
    """
    data.qpos[fingers] = closing * closed
    mujoco.mj_kinematics(model, data)
    return float(np.linalg.norm(data.geom(f"{prefix}left_pad1").xpos - data.geom(f"{prefix}right_pad1").xpos))


def closing_for_gap(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    fingers: np.ndarray,
    closed: np.ndarray,
    prefix: str,
    gap: float,
) -> float:
    """How far closed the fingers are (0 open, 1 closed on the tube) when the pads are ``gap`` apart.

    Closed on the tube the gap is the tube's width; the pads' thickness is the rest of the distance
    between their centres. The fingers are put back as they were.
    """
    before = data.qpos[fingers].copy()
    thickness = pad_centres(model, data, fingers, closed, prefix, 1.0) - 2 * rules.TUBE_RADIUS
    low, high = 0.0, 1.0
    for _ in range(30):
        mid = (low + high) / 2
        if pad_centres(model, data, fingers, closed, prefix, mid) - thickness > gap:
            low = mid
        else:
            high = mid
    data.qpos[fingers] = before
    mujoco.mj_kinematics(model, data)
    return (low + high) / 2


class SideChecker(scenes.Checker):
    """The fast check for side grasps, on the same settled scene and spare gripper.

    Coming in sideways at the grip height, the arm's wrist is right behind
    the gripper at the same height, over whatever stands there; the spare
    gripper's wrist rod stands in for it.
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, preshape: bool = False):
        super().__init__(model, data)
        # How far closed the fingers are on the way in: 0 = fully open (version 1).
        self.way_in_closing = self._closing_for(PRESHAPE_GAP) if preshape else 0.0
        if preshape:
            # With the pads already 5 mm from the tube, the first to touch moves the tube in its cell's
            # play rather than shoving the arm: in physics, 12 of 12 grasps that only the 3 mm shove
            # had called blocked lifted cleanly (docs/side/version2.md).
            self.SHOVE = 0.0
        wrist = model.geom(scenes.WRIST).id
        model.geom_contype[wrist] = model.geom_conaffinity[wrist] = 1
        # MuJoCo also keeps each body's flags, made when the model was built, and skips a body
        # whose flags are off before it looks at its geoms.
        body = model.geom_bodyid[wrist]
        model.body_contype[body] = model.body_conaffinity[body] = 1
        self.is_gripper[wrist] = True

    def _pad_centres(self, closing: float) -> float:
        """How far apart the two pads' centres are, with the fingers ``closing`` of the way shut."""
        out = pad_centres(self.model, self.data, self.fingers, self.closed, "gripper/", closing)
        return out

    def _closing_for(self, gap: float) -> float:
        out = closing_for_gap(self.model, self.data, self.fingers, self.closed, "gripper/", gap)
        self.park()
        return out

    def touches_side(self, pinch: np.ndarray, a: float, closing: float = 0.0) -> list[str]:
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, frame(a).ravel())
        # The hand body sits behind the pinch point, along the fingers.
        self.data.mocap_pos[self.mocap] = pinch + along(a) * self.pinch_below
        self.data.mocap_quat[self.mocap] = quat
        self.data.qpos[self.fingers] = closing * self.closed
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_collision(self.model, self.data)
        hit = []
        for c in self.data.contact[: self.data.ncon]:
            g1, g2 = self.is_gripper[c.geom1], self.is_gripper[c.geom2]
            if g1 != g2:
                other = c.geom2 if g1 else c.geom1
                hit.append(self.model.body(self.model.geom_bodyid[other]).name)
        return hit

    def clear(self, xy: np.ndarray, a: float, top: float, target: str, below: float | None = None) -> bool:
        # The end of the way in first: that is where most side grasps are blocked.
        for pinch in reversed(way_in(xy, a, top, below=below)):
            if self.touches_side(pinch, a, self.way_in_closing):
                return False
        opening = frame(a)[:, 0]
        pinch = pinch_at(xy, a, top, below)
        closing = sorted({max(f, self.way_in_closing) for f in self.CLOSING})
        for shove in (0.0, -self.SHOVE, self.SHOVE):
            for f in closing:
                if not set(self.touches_side(pinch + shove * opening, a, f)) <= {target}:
                    return False
        return True

    def labels(
        self, xy: np.ndarray, top: float, target: str, frame_yaw: float = 0.0, heights: bool = False
    ) -> np.ndarray:
        """For each of the 24 directions, counted from ``frame_yaw``, whether the grasp is clear.

        With ``heights`` (version 3): 24 for each of HEIGHTS in turn, 72 in all. A direction blocked
        at one height is taken as blocked at every lower one, without checking: lower, the fingers are
        further down among the same neighbours (checked against the full test on 15 scenes in
        docs/side/version3.md). That keeps the labelling about as fast as for one height.
        """
        if not heights:
            out = np.array([self.clear(xy, frame_yaw + direction(k), top, target) for k in range(BINS)])
            self.park()
            return out
        rows = []
        still = np.ones(BINS, bool)
        for below in HEIGHTS:
            row = np.zeros(BINS, bool)
            for k in np.nonzero(still)[0]:
                row[k] = self.clear(xy, frame_yaw + direction(k), top, target, below)
            rows.append(row)
            still = row
        self.park()
        return np.concatenate(rows)


def rack_yaw(heights: np.ndarray) -> float:
    """Version 2: the turn of the rack under the middle of a close-up taken with the camera unturned.

    The rack's top plate is a flat rectangle at a known height, with round holes. The plate around
    the middle of the picture is picked out (not the next rack's), its holes are filled, and the
    narrowest-fitting rectangle round it is found by trying every turn half a degree apart. Its
    sides are the rack's. Returns radians, from -45 to 45 degrees (a quarter turn is the same rack).
    """
    plate = (heights > PLATE[0]) & (heights < PLATE[1])
    v, u = np.nonzero(plate)
    if len(u) == 0:
        return 0.0
    centre = (heights.shape[0] - 1) / 2
    # The patch of plate nearest the middle that is big enough to be a rack (slivers at the
    # plate's height show inside holes too).
    mine = None
    for i in np.argsort((u - centre) ** 2 + (v - centre) ** 2)[:400:10]:
        seed = np.zeros_like(plate)
        seed[v[i], u[i]] = True
        patch = _grow(seed, plate)
        if patch.sum() >= MIN_PLATE:
            mine = patch
            break
    if mine is None:
        return 0.0
    # Fill its holes: the outside is everything not on the patch reached from the picture's border.
    border = np.zeros_like(plate)
    border[[0, -1], :] = border[:, [0, -1]] = True
    outside = _grow(border & ~mine, ~mine)
    v, u = np.nonzero(~outside)
    # Picture rows run down the table's -y, columns along its +x (the camera unturned).
    points = np.stack([u, -v], axis=1).astype(np.float64)
    best = None
    for turn in np.radians(np.arange(-45.0, 45.0, 0.5)):
        c, s_ = math.cos(turn), math.sin(turn)
        along, across = points @ [c, s_], points @ [-s_, c]
        area = np.ptp(along) * np.ptp(across)
        if best is None or area < best[0]:
            best = (area, turn)
    return float(best[1])


def _grow(seed: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    """The pixels of ``allowed`` connected to ``seed`` (side by side, not corner to corner)."""
    region = seed & allowed
    while True:
        grown = region.copy()
        grown[1:] |= region[:-1]
        grown[:-1] |= region[1:]
        grown[:, 1:] |= region[:, :-1]
        grown[:, :-1] |= region[:, 1:]
        grown &= allowed
        if (grown == region).all():
            return region
        region = grown


def smoothed(scores: np.ndarray) -> np.ndarray:
    """Version 2: each direction's score mixed with its two neighbours' (directions wrap round).

    Scores (N, 24 x heights, H, W) or (24 x heights, H, W); each height's 24 are mixed on their own.
    A direction at the edge of a run of clear ones, next to a blocked one, scores lower than one in
    the middle; one clear direction alone still stands out.
    """
    axis = scores.ndim - 3
    shape = scores.shape
    blocks = scores.reshape(*shape[:axis], shape[axis] // BINS, BINS, *shape[axis + 1 :])
    mixed = 0.5 * blocks + 0.25 * (np.roll(blocks, 1, axis=axis + 1) + np.roll(blocks, -1, axis=axis + 1))
    return mixed.reshape(shape)


# ------------------------------------------------------------------ the real arm


def solve(
    model: mujoco.MjModel, data: mujoco.MjData, target: np.ndarray, arm: str, a: float, elbow_up: bool = False
) -> np.ndarray:
    """Joint angles with the pinch point at ``target``, fingers level and pointing in from direction ``a``.

    lab_scene.kinematics points the fingers but leaves their roll free; the
    last wrist joint rolls the gripper about the fingers, so it is set
    afterwards to make the fingers open sideways (level). Two rolls half a
    turn apart are both level; the one nearer the arm's current pose is
    taken, or the wrist would flip at every step. Every joint is then moved
    by whole turns into its range, nearest to where it is now.

    With ``elbow_up``, the search starts from several poses and keeps the one
    whose elbow is highest: with the fingers level, the arm can also reach by
    folding its forearm low across the table, into the racks. Later moves
    start from the arm's current pose and stay close to it.
    """
    joints = lab.arm_joint_ids(model, arm)
    now = data.qpos[joints].copy()
    seeds = _seeds(model, data, arm) if elbow_up else [now]
    best = None
    for i, seed in enumerate(seeds):
        q = _solve_from(model, data, target, arm, a, seed, now, centred=elbow_up)
        if q is None:
            if i == 0 and not elbow_up:
                seeds.extend(_seeds(model, data, arm)[1:])  # stuck from here: try the other starting poses
            continue
        work = mujoco.MjData(model)
        work.qpos[:] = data.qpos
        work.qpos[joints] = q
        mujoco.mj_kinematics(model, work)
        elbow = work.body(f"{arm}/forearm_link").xpos[2]
        key = elbow if elbow_up else -np.abs(q - now).sum()
        if best is None or key > best[0]:
            best = (key, q)
        if not elbow_up:
            break
    if best is None:
        raise Unreachable(f"{arm} cannot reach {np.round(target, 3)} from direction {math.degrees(a):.0f}")
    return best[1]


def _solve_from(model, data, target, arm, a, seed, now, centred: bool = False) -> np.ndarray | None:
    """One search from ``seed``. With ``centred``, the wrist joints are put mid-range rather than
    nearest to now: a wrist that starts near a limit has to spin a full turn partway along a path."""
    joints = lab.arm_joint_ids(model, arm)
    work = mujoco.MjData(model)
    work.qpos[:] = data.qpos
    work.qpos[joints] = seed
    q = kinematics.solve(model, work, target, arm, pointing=-along(a))
    limits = np.array([model.jnt_range[model.joint(f"{arm}/{n}").id] for n in lab.ARM_JOINTS])
    # The roll that makes the fingers open level. Turning the last wrist joint turns the
    # gripper about its fingers by the same angle (one way or the other), so the level roll
    # is found directly: the angle that brings the opening's height to zero, or half a turn on.
    work.qpos[joints] = q
    mujoco.mj_kinematics(model, work)
    frame_ = work.site(lab.pinch_site(arm)).xmat.reshape(3, 3)
    theta = math.atan2(frame_[2, 1], frame_[2, 0])
    best = None
    for roll in (theta, theta + math.pi, -theta, -theta + math.pi):
        for turns in range(-2, 3):
            fix = q[5] + roll + 2 * math.pi * turns
            if not limits[5, 0] <= fix <= limits[5, 1]:
                continue
            work.qpos[joints] = q
            work.qpos[joints[5]] = fix
            mujoco.mj_kinematics(model, work)
            tilt = abs(work.site(lab.pinch_site(arm)).xmat.reshape(3, 3)[2, 1])
            cost = (round(float(tilt), 2), abs(fix - now[5]))
            if best is None or cost < best[0]:
                best = (cost, fix)
    if best is None or best[0][0] > 0.02:
        return None
    q = q.copy()
    q[5] = best[1]
    # Whole turns: into range, nearest to now.
    for k in range(6):
        options = q[k] + 2 * np.pi * np.arange(-3, 4)
        options = options[(options >= limits[k, 0]) & (options <= limits[k, 1])]
        if len(options) == 0:
            return None
        near = 0.0 if centred and k >= 3 else now[k]
        q[k] = options[np.argmin(np.abs(options - near))]
    work.qpos[joints] = q
    mujoco.mj_kinematics(model, work)
    if np.linalg.norm(work.site(lab.pinch_site(arm)).xpos - target) > 1e-3:
        return None
    if folds_onto_itself(model, work, arm):
        return None
    return q


def folds_onto_itself(model: mujoco.MjModel, data: mujoco.MjData, arm: str) -> bool:
    """Whether the arm touches itself (a link on its own gripper) or the table (other than its base).

    Some poses reach the point only by folding the wrist back onto the
    forearm; the wrist motors then stall against it.
    """
    mujoco.mj_collision(model, data)
    for c in data.contact[: data.ncon]:
        n1 = model.body(model.geom_bodyid[c.geom1]).name
        n2 = model.body(model.geom_bodyid[c.geom2]).name
        mine1, mine2 = n1.startswith(f"{arm}/"), n2.startswith(f"{arm}/")
        if mine1 and mine2 and (("gripper/" in n1) != ("gripper/" in n2)):
            return True
        if (mine1 and n2 == "world" and not n1.endswith(STANDING)) or (
            mine2 and n1 == "world" and not n2.endswith(STANDING)
        ):
            return True
    return False


# The arm's links that stand on the table anyway.
STANDING = ("/base", "/shoulder_link")


class Unreachable(RuntimeError):
    pass


def _seeds(model: mujoco.MjModel, data: mujoco.MjData, arm: str) -> list[np.ndarray]:
    """Starting poses for the search: where the arm is now, and a few elbow-up poses facing the same way."""
    here = data.qpos[lab.arm_joint_ids(model, arm)].copy()
    out = [here]
    for lift, elbow in ((-1.2, 1.6), (-1.9, 1.9), (-0.8, 1.0)):
        for wrist in (-math.pi / 2, 0.0, math.pi / 2):
            out.append(
                np.array([here[0], lift, elbow, -math.pi / 2 - (lift + elbow) + wrist, -math.pi / 2, here[5]])
            )
    return out


def _line(start: np.ndarray, end: np.ndarray, step: float = 0.02) -> list[np.ndarray]:
    n = max(1, math.ceil(np.linalg.norm(end - start) / step))
    return [start + (end - start) * i / n for i in range(1, n + 1)]


class SideWatch(trial.Watch):
    def arrive(self, arm: str, seconds: float = 2.0) -> None:
        """Wait (up to ``seconds``) until the arm's joints have caught up with their targets."""
        actuators, joints = lab.arm_actuator_ids(self.model, arm), lab.arm_joint_ids(self.model, arm)
        for _ in range(int(seconds / self.model.opt.timestep)):
            if np.abs(self.data.qpos[joints] - self.data.ctrl[actuators]).max() < 0.01:
                return
            self.step()

    def move_side(
        self,
        target: np.ndarray,
        arm: str,
        a: float,
        seconds: float,
        settle: float = 0.3,
        elbow_up: bool = False,
    ) -> None:
        actuators = lab.arm_actuator_ids(self.model, arm)
        start = self.data.ctrl[actuators].copy()
        goal = solve(self.model, self.data, target, arm, a, elbow_up)
        if not elbow_up and np.abs(goal - start).max() > MAX_JUMP:
            # Near a singular pose a short move can need half a turn of a wrist; the arm cannot follow.
            raise Unreachable(f"{arm} would have to swing a joint by {np.abs(goal - start).max():.1f} rad")
        steps = int(seconds / self.model.opt.timestep)
        for i in range(1, steps + 1):
            self.data.ctrl[actuators] = start + (goal - start) * i / steps
            self.step()
        self.wait(settle)


def run(
    seed: int,
    target: str,
    xy: np.ndarray,
    a: float,
    top: float,
    liquid_in: bool = False,
    seen=None,
    preshape: bool = False,
    below: float | None = None,
) -> dict:
    """A side grasp with the real arm and full physics, judged as trial.run judges a top-down one.

    With ``preshape`` (version 2), the fingers close to PRESHAPE_GAP before the arm moves. ``below``
    (version 3): grip this far under the rim instead of BELOW_RIM.
    """
    seen = seen or (lambda *_: None)
    model, data, scene = scenes.build(seed, target if liquid_in else None)
    watch = SideWatch(model, data, target)
    tube = next(t for t in scenes.tubes(scene) if t.name == target)
    container = scenes.random_scene.BY_NAME[tube.kind]
    arm = min(lab.ARM_NAMES, key=lambda n: np.linalg.norm(data.body(f"{n}/base").xpos[:2] - xy))
    start = {i.name: data.body(i.name).xpos.copy() for i in scene.instances}
    balls = len(liquid.balls_in(model, data, container, target))
    path = way_in(xy, a, top, step=0.02, below=below)
    if preshape:
        data.ctrl[lab.gripper_actuator_id(model, arm)] = lab.grip_command(PRESHAPE_GAP, squeeze=0.0)
    try:
        # Straight up first and across, fingers still down; turn them level up there, above
        # where the fingers will come down; then down and in, in straight lines, 2 cm at a time.
        here = data.site(lab.pinch_site(arm)).xpos.copy()
        yaw = trial._open_angle(data, arm)
        high = np.array([here[0], here[1], TRANSFER])
        above = np.array([path[0][0], path[0][1], TRANSFER])
        for p in [high, *_line(high, above, 0.05)]:
            watch.move(p, arm, yaw, 0.25, settle=0.0)
        watch.move_side(above, arm, a, 2.0, elbow_up=True)
        beside = len(_line(above, path[0])) + int(round(DROP / 0.02))
        for i, p in enumerate(_line(above, path[0]) + path):
            watch.move_side(p, arm, a, 0.12, settle=0.0)
            if i == beside:
                watch.arrive(arm)
                seen("beside", model, data)
        watch.arrive(arm)
        watch.wait(0.3)
        seen("in", model, data)
        touched_going_in = sorted(watch.touched)
        data.ctrl[lab.gripper_actuator_id(model, arm)] = lab.grip_command(2 * rules.TUBE_RADIUS)
        watch.wait(0.8)
        seen("closed", model, data)
        touched_closing = sorted(watch.touched - set(touched_going_in))
        force = trial.squeeze_force(model, data, target)
        watch.move_side(pinch_at(xy, a, top, below) + [0, 0, LIFT], arm, a, 1.0)
        seen("lifted", model, data)
        finite = bool(np.isfinite(data.qpos).all())
    except Unreachable as e:
        # The nearest arm cannot get its fingers there level: not a grasp the arm can make.
        return {"success": False, "unreachable": True, "why": str(e), "touched": sorted(watch.touched)}
    except Exception:  # a solver blow-up can surface as a MuJoCo error
        finite = False
    if not finite:
        return {"success": False, "blew_up": True, "touched": sorted(watch.touched)}
    rose = float(data.body(target).xpos[2] - start[target][2])
    upright = float(data.body(target).xmat[8])
    moved = max(float(np.linalg.norm(data.body(k).xpos - p)) for k, p in start.items() if k != target)
    lifted = rose > trial.ROSE and upright > trial.UPRIGHT
    return {
        "success": lifted and not watch.touched,
        "lifted": lifted,
        "touched": sorted(watch.touched),
        "touched_going_in": touched_going_in,
        "touched_closing": touched_closing,
        "blew_up": moved > 1.0,
        "rose": rose,
        "upright": upright,
        "squeeze": force,
        "spilled": balls - len(liquid.balls_in(model, data, container, target)),
        "arm": arm,
    }
