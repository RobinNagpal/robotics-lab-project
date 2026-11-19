"""Check a grip against every place the job puts a tube, before any step is built.

v1 chose its grip for picking and found out at the pour that it covered the tube's opening. Here every grip
is checked against the whole job at once: picking from a rack, the balance holder, all six rotor holes (with
the other holes full), the pour, and the lid. Nothing moves under physics: the arm is put in each pose by
inverse kinematics and the gaps between it and everything else are measured.

Words used below:
- **pinch point**: the point between the finger pads (the site "arm/gripper/pinch"); its z axis runs along
  the fingers, its y axis is the way the fingers close.
- **axial grip**: the fingers point along the tube, from its top (v1's "top-down"). ``depth`` is how far
  below the tube's top the pinch point sits; ``turn`` is which way the fingers close, round the tube.
- **side grip**: the fingers come in across the tube, from the side; the opening stays free (v1's side grasp).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np
from centrifuge_scene import centrifuge, scene
from centrifuge_scene.scale import pan_top
from centrifuge_scene.settings import HOME, LAYOUT, ROTOR, TUBE

UP = np.array([0.0, 0.0, 1.0])
# Gaps are measured up to this; anything farther counts as this.
MARGIN = 0.010
# Less than this is a touch: the real arm, Step 0's positions and the tube in the fingers all have errors.
SAFE = 0.003
# Finger openings (the gap between the pads), m.
OPEN = 0.045  # going down over a cap (35 mm) or letting go
HOLD_CAP = 2 * TUBE.cap_radius
HOLD_BODY = 2 * TUBE.radius
HOLD_RIM = 2 * TUBE.outline[-1][1]
# The hole is wider than the tube near the top (lead-in): radius of the opening a finger cannot enter.
HOLE_OPENING = TUBE.radius + centrifuge.CLEARANCE + centrifuge.LEAD_IN[1]


def rot_about(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * k @ k


def perpendicular(z: np.ndarray) -> np.ndarray:
    """A unit vector at right angles to ``z``: horizontal when it can be."""
    x = np.cross(UP, z)
    if np.linalg.norm(x) < 1e-6:
        x = np.array([1.0, 0.0, 0.0])
    return x / np.linalg.norm(x)


@dataclass(frozen=True)
class TubePose:
    tip: np.ndarray
    rot: np.ndarray  # columns: x, y, z; z runs from the tip to the top
    capped: bool

    @property
    def axis(self) -> np.ndarray:
        return self.rot[:, 2]

    @property
    def length(self) -> float:
        return TUBE.capped_height if self.capped else TUBE.height

    @property
    def top(self) -> np.ndarray:
        return self.tip + self.length * self.axis


def axial_grip(tube: TubePose, depth: float, turn: float, ref: np.ndarray | None = None):
    """(pinch position, pinch rotation) for an axial grip. ``turn`` 0: fingers close along ``ref``."""
    z_t = tube.axis
    e1 = perpendicular(z_t) if ref is None else ref - (ref @ z_t) * z_t
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(z_t, e1)
    z = -z_t
    y = math.cos(turn) * e1 + math.sin(turn) * e2
    return tube.top - depth * z_t, np.column_stack([np.cross(y, z), y, z])


def side_grip(tube: TubePose, below_rim: float, approach: np.ndarray):
    """(pinch position, rotation) for a side grip: fingers come in along ``approach`` (across the tube)."""
    z_t = tube.axis
    z = approach - (approach @ z_t) * z_t
    z /= np.linalg.norm(z)
    y = np.cross(z_t, z)
    return tube.tip + (TUBE.height - below_rim) * z_t, np.column_stack([np.cross(y, z), y, z])


def hold_width(tube: TubePose, depth: float) -> float:
    """The gap the pads close to on a tube gripped ``depth`` below its top."""
    if tube.capped and depth < TUBE.cap_height + 0.020:  # the pads reach the 35 mm cap
        return HOLD_CAP
    return HOLD_RIM if not tube.capped and depth < 0.012 else HOLD_BODY


class World:
    """One seed's scene, with tools to pose the arm and measure gaps."""

    def __init__(self, seed: int):
        self.seed = seed
        self.model, self.truth = scene.build(seed, awake=True)
        m = self.model
        self.data = mujoco.MjData(m)
        scene.reset(m, self.data)
        self.arm_q = scene.arm_joint_ids(m)
        self.arm_dofs = np.array([m.jnt_dofadr[m.jnt_qposadr.tolist().index(a)] for a in self.arm_q])
        self.grip_joints = [j for j in range(m.njnt) if m.joint(j).name.startswith("arm/gripper/")]
        self.grip_q = np.array([m.jnt_qposadr[j] for j in self.grip_joints])
        self.limits = np.array([m.jnt_range[m.jnt_qposadr.tolist().index(a)] for a in self.arm_q])
        self.site = m.site(scene.pinch_site()).id
        names = [m.body(b).name for b in range(m.nbody)]
        self.body_name = names
        self.is_arm = np.array([n.startswith("arm/") for n in names])
        self.is_ball = np.array([n.startswith("ball") for n in names])
        self.tube_of_body = {m.body(t).id: t for t in self.truth.tubes}
        self.rotor = m.body("centrifuge/rotor").id
        self.hole_body = {k: m.body(f"centrifuge/hole{k}").id for k in range(ROTOR.slots)}
        self._grip_states = {}
        self._gripper_points = self._points_on_gripper()
        self.base_qpos = self.data.qpos.copy()

    # --- tubes -------------------------------------------------------------------------------------------

    def tube_pose(self, name: str) -> TubePose:
        b = self.model.body(name).id
        return TubePose(
            self.data.xpos[b].copy(), self.data.xmat[b].reshape(3, 3).copy(), self.truth.capped[name]
        )

    def place(self, name: str, pose: TubePose) -> None:
        scene.move_tube(self.model, self.data, name, pose.tip, pose.rot)

    def seated_pose(self, k: int, capped: bool) -> TubePose:
        s = self.data.site(scene.slot_site(k))
        rot = s.xmat.reshape(3, 3).copy()
        return TubePose(s.xpos + rot[:, 2] * 0.0005, rot, capped)

    def holder_pose(self, capped: bool) -> TubePose:
        return TubePose(np.array([*LAYOUT.scale, pan_top() + 0.0005]), np.eye(3), capped)

    def fill_holes(self, holes: list[int], held: str) -> None:
        """Put tubes (not ``held``; capped samples first) in the given holes."""
        others = [n for n in self.truth.tubes if n not in ("water", held)]
        others.sort(key=lambda n: not self.truth.capped[n])
        if len(others) < len(holes):
            raise ValueError(f"seed {self.seed} has only {len(others)} spare tubes")
        for k, name in zip(holes, others, strict=False):
            self.place(name, self.seated_pose(k, self.truth.capped[name]))

    def set_lid(self, open_deg: float) -> None:
        """The lid resting ``open_deg`` degrees open (98.7 is AutoBio's stop)."""
        from centrifuge_scene.settings import LID_CLOSED

        self.data.qpos[centrifuge.lid_qpos_adr(self.model)] = LID_CLOSED + math.radians(open_deg)
        mujoco.mj_forward(self.model, self.data)

    def fill_rotor(self, except_hole: int, held: str) -> None:
        """Every other hole gets a tube (not ``held``), capped samples first (their caps are the widest)."""
        others = [n for n in self.truth.tubes if n not in ("water", held)]
        holes = [k for k in range(ROTOR.slots) if k != except_hole]
        for k, name in zip(holes, others, strict=False):
            self.place(name, self.seated_pose(k, self.truth.capped[name]))

    # --- the gripper -------------------------------------------------------------------------------------

    def grip_state(self, gap: float) -> np.ndarray:
        """The gripper's joint angles with its pads ``gap`` apart (simulated once, nothing between them)."""
        key = round(gap, 4)
        if key not in self._grip_states:
            d = mujoco.MjData(self.model)
            scene.reset(self.model, d)
            d.ctrl[scene.gripper_actuator_id(self.model)] = scene.grip_command(gap, squeeze=0.0)
            for _ in range(1500):
                mujoco.mj_step(self.model, d)
            self._grip_states[key] = d.qpos[self.grip_q].copy()
        return self._grip_states[key]

    def _points_on_gripper(self) -> list[tuple[int, np.ndarray]]:
        """Points on every colliding gripper part, in its own frame: for the rotor-top check."""
        m = self.model
        out = []
        for g in range(m.ngeom):
            if not self.body_name[m.geom_bodyid[g]].startswith("arm/gripper/") or not m.geom_contype[g]:
                continue
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mid = m.geom_dataid[g]
                a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
                pts = m.mesh_vert[a : a + n].astype(float)
            else:
                h = m.geom_size[g]
                grid = np.linspace(-1, 1, 5)
                pts = np.array([[x, y, z] for x in grid for y in grid for z in grid]) * h
            out.append((g, pts))
        return out

    # --- inverse kinematics ------------------------------------------------------------------------------

    def ik(self, pos: np.ndarray, rot: np.ndarray, start: np.ndarray | None = None) -> np.ndarray | None:
        """Arm angles putting the pinch point at ``pos`` turned to ``rot`` exactly, or None."""
        starts = [] if start is None else [start]
        base = np.array(HOME)
        turn = math.atan2(pos[1], pos[0]) - math.atan2(0.49, -0.134)
        for extra in (0.0, 0.6, -0.6):
            s = base.copy()
            s[0] += turn
            s[5] += extra
            starts.append(s)
        for s in starts:
            q = self._search(pos, rot, s)
            if q is not None:
                return q
        return None

    def _search(self, pos, rot, start, iterations=300):
        m, work = self.model, mujoco.MjData(self.model)
        work.qpos[:] = self.base_qpos
        q = start.copy()
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for _ in range(iterations):
            work.qpos[self.arm_q] = q
            mujoco.mj_kinematics(m, work)
            mujoco.mj_comPos(m, work)
            R = work.site_xmat[self.site].reshape(3, 3)
            err = np.concatenate(
                [pos - work.site_xpos[self.site], 0.5 * sum(np.cross(R[:, i], rot[:, i]) for i in range(3))]
            )
            if np.linalg.norm(err[:3]) < 2e-4 and np.linalg.norm(err[3:]) < 2e-3:
                if np.trace(R.T @ rot) < 2.9:  # pointing the opposite way also zeroes the error
                    return None
                q = q - 2 * np.pi * np.round((q - start) / (2 * np.pi))
                if np.any(q < self.limits[:, 0]) or np.any(q > self.limits[:, 1]):
                    return None
                return q
            mujoco.mj_jacSite(m, work, jacp, jacr, self.site)
            jac = np.vstack([jacp, jacr])[:, self.arm_dofs]
            step = jac.T @ np.linalg.solve(jac @ jac.T + 1e-4 * np.eye(6), err)
            q = q + step * min(1.0, 0.2 / np.abs(step).max())
        return None

    # --- measuring gaps ----------------------------------------------------------------------------------

    def gap(
        self,
        q: np.ndarray,
        opening: float,
        held: str | None = None,
        ignore: tuple[int, ...] = (),
        gripper_may_touch: tuple[int, ...] = (),
    ) -> tuple[float, str]:
        """The smallest gap (m, negative = overlap) between the arm, or the held tube, and anything else.

        ``ignore``: bodies the held tube may touch (the hole or holder it is going into). The rotor's top is
        checked as the solid it really is (its collision model has only webs between the holes).
        """
        m, d = self.model, self.data
        d.qpos[self.arm_q] = q
        d.qpos[self.grip_q] = self.grip_state(opening)
        held_id = m.body(held).id if held else -1
        margin = m.geom_margin.copy()
        mover = self.is_arm[m.geom_bodyid] | (m.geom_bodyid == held_id)
        m.geom_margin[mover] = MARGIN
        mujoco.mj_forward(m, d)
        m.geom_margin[:] = margin
        best, what = MARGIN, ""
        for c in d.contact[: d.ncon]:
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            for a, b in ((b1, b2), (b2, b1)):
                a_arm, b_arm = self.is_arm[a], self.is_arm[b]
                if not (a_arm or a == held_id) or b_arm or b == held_id or self.is_ball[b]:
                    continue
                if a == held_id and b in ignore:
                    continue
                if b in gripper_may_touch and self.body_name[a].startswith("arm/gripper/"):
                    continue
                if b == 0 and self.body_name[a] in ("arm/base", "arm/shoulder_link"):
                    continue  # the arm's base stands on the table
                if c.dist < best:
                    best, what = c.dist, f"{self.body_name[a]} - {self.body_name[b]}"
                break
        top = self.rotor_top_gap()
        if top < best:
            best, what = top, "gripper - rotor top"
        return best, what

    def rotor_top_gap(self) -> float:
        """How far the lowest gripper point over the rotor's solid top is above it (negative: inside)."""
        d = self.data
        R0, p0 = d.xmat[self.rotor].reshape(3, 3), d.xpos[self.rotor]
        frames = centrifuge.slot_frames()
        best = MARGIN
        for g, pts in self._gripper_points:
            world = d.geom_xpos[g] + pts @ d.geom_xmat[g].reshape(3, 3).T
            local = (world - p0) @ R0
            r = np.hypot(local[:, 0], local[:, 1])
            near = (r < ROTOR.radius) & (local[:, 2] < ROTOR.top + MARGIN) & (local[:, 2] > ROTOR.bottom)
            if not near.any():
                continue
            pts_l = local[near]
            in_hole = np.zeros(len(pts_l), bool)
            for bottom, _top, rot in frames:
                rel = pts_l - bottom
                along = rel @ rot[:, 2]
                radial = np.linalg.norm(rel - np.outer(along, rot[:, 2]), axis=1)
                in_hole |= (radial < HOLE_OPENING) & (along > 0)
            h = pts_l[~in_hole, 2] - ROTOR.top
            if len(h):
                best = min(best, float(h.min()))
        return best

    def reset(self) -> None:
        self.data.qpos[:] = self.base_qpos
        mujoco.mj_forward(self.model, self.data)


# --- moves -------------------------------------------------------------------------------------------------


@dataclass
class Check:
    ok: bool
    gap: float  # smallest gap along the move, m
    what: str  # what came closest
    note: str = ""


@dataclass
class Pose:
    """One pose along a move: where the pinch point goes, how open the fingers are, the tube it holds."""

    pos: np.ndarray
    rot: np.ndarray
    opening: float
    held: str | None = None
    held_pose: TubePose | None = None
    ignore: tuple[int, ...] = ()
    may_touch: tuple[int, ...] = ()  # bodies the fingers may touch (the tube just let go)


def walk(world: World, poses: list[Pose]) -> Check:
    """Follow the poses with inverse kinematics, each from the last; the smallest gap on the way."""
    q, best, what = None, MARGIN, ""
    for p in poses:
        q2 = world.ik(p.pos, p.rot, q)
        if q2 is None:
            return Check(False, -1.0, "", "out of reach")
        if q is not None and np.abs(q2 - q).max() > 0.5:
            return Check(False, -1.0, "", "the arm would flip between two poses")
        q = q2
        if p.held:
            world.place(p.held, p.held_pose)
        g, w = world.gap(q, p.opening, p.held, p.ignore, p.may_touch)
        if g < best:
            best, what = g, w
    return Check(best >= SAFE, best, what)


def line(a: np.ndarray, b: np.ndarray, step: float = 0.01) -> list[np.ndarray]:
    n = max(1, int(math.ceil(np.linalg.norm(b - a) / step)))
    return [a + (b - a) * i / n for i in range(n + 1)]


def stand_in_and_let_go(
    world: World,
    name: str,
    seat: TubePose,
    back: float,
    depth: float,
    turn: float,
    ref,
    ignore,
    release: float = OPEN,
) -> Check:
    """Slide a held tube ``back`` m along its axis into ``seat``, open the fingers, back out the same way.

    Used for a rack cell, the balance holder and a rotor hole. Read the other way it is picking the tube
    up: fingers open on the way in, closed on the way out.
    """
    world.reset()
    hold = hold_width(seat, depth)
    _, rot = axial_grip(seat, depth, turn, ref)
    poses = []
    for tip in line(seat.tip + back * seat.axis, seat.tip):
        tp = TubePose(tip, seat.rot, seat.capped)
        poses.append(Pose(axial_grip(tp, depth, turn, ref)[0], rot, hold, name, tp, ignore))
    own = (world.model.body(name).id,)
    for i, tip in enumerate(line(seat.tip, seat.tip + back * seat.axis)):
        tp = TubePose(tip, seat.rot, seat.capped)
        # Just let go, the pads are (release - hold) / 2 from the tube; once 3 cm up they must be clear of it.
        poses.append(Pose(axial_grip(tp, depth, turn, ref)[0], rot, release, may_touch=own if i < 3 else ()))
    return walk(world, poses)
