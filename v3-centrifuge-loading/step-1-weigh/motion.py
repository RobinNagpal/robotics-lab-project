"""Moving the arm: straight lines in space, each checked for collisions before the arm moves.

The programmed arm for Step 1 (and later Step 4), in the style of v1's Step 1b:

- **A move is a straight line** for the pinch point (the point between the finger pads), cut into 5 mm
  pieces. Each piece's end is solved for joint angles (inverse kinematics) from the last one, so the arm
  never swings the long way round. The fingers point straight down; their turn about the vertical (which way
  they close) may change along the line.
- **Before the arm moves**, every piece's end is checked on a scratch copy of the scene with collision tests
  only (no physics): the arm, and the tube it holds, must stay ``SAFE`` away from everything else. If not, the
  move is refused (``Blocked``) and nothing has moved.
- **Then the arm follows** the solved joint angles, speeding up and slowing down gently (a held tube's liquid
  is not jolted), at most ``JOINT_SPEED`` on any joint.

Every simulation step goes through ``Arm.step``, which calls ``on_step`` if one is given: that is where the
demonstrations for π0.5 will be recorded (pictures, joint angles, gripper) without changing this code.

What the planner's copy of the scene contains: the bench as built, with each tube standing where it stands.
That is what the arm knows anyway (the bench layout, and from Step 0 which cells hold a tube); it never reads
a tube's weight or how much liquid it holds.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import mujoco
import numpy as np
from centrifuge_scene import scene
from centrifuge_scene.settings import HOME, TUBE

PIECE = 0.005  # m between solved points along a line
MAX_JUMP = 0.15  # rad: the most any joint may change between two points 5 mm apart
REACHED = 0.002  # m: a solved point further than this from its target counts as out of reach
JOINT_SPEED = 0.7  # rad/s, the fastest any joint is asked to turn
SAFE = 0.002  # m kept between the moving arm (and held tube) and everything else
MARGIN = 0.010  # m: gaps are measured up to this
ARRIVED, CATCH_UP = 0.005, 2.0  # rad: close enough to the targets; s: the most to wait for that

# Finger openings (the gap between the pads), m. The fingers open to 45 mm to come down over a 35 mm cap
# (grasp-plan: every rack cell and the holder have clear turns with this opening).
OPEN = 0.045
HOLD_CAP = 2 * TUBE.cap_radius
# The pinch point sits this far under the cap's top: the 38 mm pads then cover the whole 12 mm cap.
CAP_DEPTH = 0.004
# How far the fingers are told to close past the cap's width (m), and how long they are given. v1 squeezed
# its 29 mm tubes by 4 mm. On the round 35 mm cap that grip slowly spins the tube and drives it down into the
# rack's seat until the simulation jumps (tubes flung on 2 of 20 seeds even with 0.5 s); 2 mm for 0.5 s flung
# none and holds the tube firmly (docs/results.md).
GRIP_SQUEEZE = 0.002
GRIP_SECONDS = 0.5
HOME_PINCH_XY = (-0.134, 0.492)  # where the pinch point is, seen from above, with the arm at home


class Blocked(RuntimeError):
    """A planned move would touch something, or the arm cannot reach."""


def fingers_down(turn: float) -> np.ndarray:
    """The pinch point's rotation with the fingers pointing straight down, closing along ``turn`` (rad).

    The pinch site's z axis runs along the fingers, its y axis is the way they close.
    """
    z = np.array([0.0, 0.0, -1.0])
    y = np.array([math.cos(turn), math.sin(turn), 0.0])
    return np.column_stack([np.cross(y, z), y, z])


def turn_between(a: float, b: float) -> float:
    """The smallest change of finger turn from ``a`` to ``b``. Turns half a circle apart are the same grip."""
    return (b - a + math.pi / 2) % math.pi - math.pi / 2


def smooth(s: float) -> float:
    """0 to 1 with gentle speeding up and slowing down."""
    return s * s * (3 - 2 * s)


@dataclass
class Held:
    """A capped tube in the fingers, held ``CAP_DEPTH`` under its top, upright under the pinch point."""

    body: int  # its body id: only for moving it in the planner's scratch copy

    def tube_qpos(self, pinch: np.ndarray, rot: np.ndarray) -> np.ndarray:
        """The tube's free-joint position (tip, then quaternion) for a pinch point at ``pinch``, ``rot``."""
        up = -rot[:, 2]
        tip = pinch + (CAP_DEPTH - TUBE.capped_height) * up
        tube_rot = np.column_stack([rot[:, 0], -rot[:, 1], up])
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, tube_rot.flatten())
        return np.concatenate([tip, quat])


class Arm:
    """The one arm: plans and checks straight-line moves, follows them, and works the gripper."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        on_step: Callable[[Arm], None] | None = None,
    ):
        self.model, self.data, self.on_step = model, data, on_step
        m = model
        self.joints = scene.arm_joint_ids(m)
        self.dofs = np.array([m.joint(f"{scene.ARM_NAME}/{n}").dofadr[0] for n in scene.ARM_JOINTS])
        self.actuators = scene.arm_actuator_ids(m)
        self.gripper = scene.gripper_actuator_id(m)
        self.limits = np.array([m.joint(f"{scene.ARM_NAME}/{n}").range for n in scene.ARM_JOINTS])
        self.site = m.site(scene.pinch_site()).id
        self.finger_q = np.array(
            [m.jnt_qposadr[j] for j in range(m.njnt) if m.joint(j).name.startswith("arm/gripper/")]
        )
        names = [m.body(b).name for b in range(m.nbody)]
        self.body_names = names
        self.is_arm = np.array([n.startswith("arm/") for n in names])
        self.is_ball = np.array([n.startswith("ball") for n in names])
        self.held: Held | None = None
        self.steps = 0
        self._fingers: dict[float, np.ndarray] = {}
        mujoco.mj_forward(m, data)
        self.turn = self.turn_now()  # the fingers' turn now

    def turn_now(self) -> float:
        """Which way the fingers close now, as an angle about the vertical (fingers pointing down)."""
        y = self.data.site_xmat[self.site].reshape(3, 3)[:, 1]
        return math.atan2(y[1], y[0])

    # --- the simulation ----------------------------------------------------------------------------------

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            scene.step(self.model, self.data)
            self.steps += 1
            if self.on_step:
                self.on_step(self)

    def wait(self, seconds: float) -> None:
        self.step(int(round(seconds / self.model.opt.timestep)))

    @property
    def time(self) -> float:
        return self.steps * self.model.opt.timestep

    def pinch(self) -> np.ndarray:
        return self.data.site_xpos[self.site].copy()

    # --- the gripper -------------------------------------------------------------------------------------

    def open(self, gap: float = OPEN) -> None:
        self.data.ctrl[self.gripper] = scene.grip_command(gap, squeeze=0.0)
        self.held = None
        self.wait(GRIP_SECONDS)

    def close_on(self, body: int, width: float = HOLD_CAP) -> None:
        """Close the fingers on the tube ``body`` (``width`` across where they meet it)."""
        self.data.ctrl[self.gripper] = scene.grip_command(width, GRIP_SQUEEZE)
        self.wait(GRIP_SECONDS)
        self.held = Held(body)

    # --- inverse kinematics ------------------------------------------------------------------------------

    def solve(self, pos: np.ndarray, rot: np.ndarray, start: np.ndarray, iterations: int = 300) -> np.ndarray:
        """Joint angles from ``start`` putting the pinch point at ``pos``, turned to ``rot``, or Blocked."""
        m, work = self.model, mujoco.MjData(self.model)
        work.qpos[:] = self.data.qpos
        q = start.copy()
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for _ in range(iterations):
            work.qpos[self.joints] = q
            mujoco.mj_kinematics(m, work)
            mujoco.mj_comPos(m, work)
            r = work.site_xmat[self.site].reshape(3, 3)
            err = np.concatenate(
                [pos - work.site_xpos[self.site], 0.5 * sum(np.cross(r[:, i], rot[:, i]) for i in range(3))]
            )
            if np.linalg.norm(err[:3]) < 1e-4 and np.linalg.norm(err[3:]) < 1e-3:
                break
            mujoco.mj_jacSite(m, work, jacp, jacr, self.site)
            jac = np.vstack([jacp, jacr])[:, self.dofs]
            step = jac.T @ np.linalg.solve(jac @ jac.T + 1e-4 * np.eye(6), err)
            q = q + step * min(1.0, 0.2 / np.abs(step).max())
        work.qpos[self.joints] = q
        mujoco.mj_kinematics(m, work)
        r = work.site_xmat[self.site].reshape(3, 3)
        if np.linalg.norm(work.site_xpos[self.site] - pos) > REACHED or np.trace(r.T @ rot) < 2.95:
            raise Blocked(f"cannot reach {np.round(pos, 3)}")
        if np.any(q < self.limits[:, 0]) or np.any(q > self.limits[:, 1]):
            raise Blocked(f"{np.round(pos, 3)} is past a joint limit")
        return q

    def solve_any(self, pos: np.ndarray, rot: np.ndarray) -> np.ndarray:
        """As ``solve``, trying from the arm's angles now, then from home turned towards ``pos``."""
        starts = [self.data.qpos[self.joints].copy()]
        here = math.atan2(*HOME_PINCH_XY[::-1])
        for extra in (0.0, 0.6, -0.6):
            s = np.array(HOME)
            s[0] += math.atan2(pos[1], pos[0]) - here
            s[5] += extra
            starts.append(s)
        for s in starts:
            try:
                return self.solve(pos, rot, s)
            except Blocked:
                continue
        raise Blocked(f"cannot reach {np.round(pos, 3)}")

    def plan_line(
        self,
        end: np.ndarray,
        turn: float | None = None,
        start: tuple[np.ndarray, float, np.ndarray] | None = None,
    ) -> list[np.ndarray]:
        """Joint angles every 5 mm from the pinch point to ``end``, turning the fingers to ``turn``.

        From where the arm is now, or from ``start`` = (pinch position, finger turn, joint angles).
        """
        if start is None:
            a, t0, q = self.pinch(), self.turn, self.data.qpos[self.joints].copy()
        else:
            a, t0, q = start
        dt = 0.0 if turn is None else turn_between(t0, turn)
        steps = max(1, math.ceil(np.linalg.norm(end - a) / PIECE), math.ceil(abs(dt) / 0.05))
        out = []
        for i in range(1, steps + 1):
            s = i / steps
            q2 = self.solve(a + (end - a) * s, fingers_down(t0 + dt * s), q)
            if np.abs(q2 - q).max() > MAX_JUMP:
                raise Blocked(f"the arm would flip on the way to {np.round(end, 3)}")
            out.append(q2)
            q = q2
        return out

    # --- checking for collisions ---------------------------------------------------------------------------

    def fingers(self, gap: float) -> np.ndarray:
        """The finger joint angles with the pads ``gap`` apart and nothing between them (simulated once)."""
        key = round(gap, 4)
        if key not in self._fingers:
            d = mujoco.MjData(self.model)
            scene.reset(self.model, d)
            d.ctrl[self.gripper] = scene.grip_command(gap, squeeze=0.0)
            for _ in range(1500):
                mujoco.mj_step(self.model, d)
            self._fingers[key] = d.qpos[self.finger_q].copy()
        return self._fingers[key]

    def clearance(
        self,
        qs: Iterable[np.ndarray],
        allow: Iterable[int] = (),
        fingers: np.ndarray | None = None,
        holding: bool = True,
    ) -> tuple[float, str]:
        """The smallest gap along ``qs`` between the arm (and held tube) and anything else: (m, what).

        ``allow``: bodies that may be touched (the rack or holder the held tube slides into; the tube the open
        fingers come down over or just let go of). ``fingers``: finger joint angles to check with (default: as
        they are now). ``holding`` False checks as if the held tube had been let go.
        """
        m = self.model
        work = mujoco.MjData(m)
        work.qpos[:] = self.data.qpos
        if fingers is not None:
            work.qpos[self.finger_q] = fingers
        allowed = set(allow)
        held = self.held.body if self.held and holding else -1
        if held >= 0:
            adr = m.jnt_qposadr[m.body_jntadr[held]]
        mover = self.is_arm[m.geom_bodyid] | (m.geom_bodyid == held)
        saved = m.geom_margin.copy()
        best, what = MARGIN, ""
        try:
            m.geom_margin[mover] = MARGIN
            for q in qs:
                work.qpos[self.joints] = q
                mujoco.mj_kinematics(m, work)
                if held >= 0:
                    rot = work.site_xmat[self.site].reshape(3, 3)
                    work.qpos[adr : adr + 7] = self.held.tube_qpos(work.site_xpos[self.site], rot)
                    mujoco.mj_kinematics(m, work)
                mujoco.mj_collision(m, work)
                for c in work.contact[: work.ncon]:
                    b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
                    for a, b in ((b1, b2), (b2, b1)):
                        if not (self.is_arm[a] or a == held) or self.is_arm[b] or b == held:
                            continue
                        if self.is_ball[b] or b in allowed:
                            continue
                        if b == 0 and self.body_names[a] in ("arm/base", "arm/shoulder_link"):
                            continue  # the arm's base stands on the table
                        if c.dist < best:
                            best, what = c.dist, f"{self.body_names[a]} - {self.body_names[b]}"
                        break
        finally:
            m.geom_margin[:] = saved
        return best, what

    # --- moving ------------------------------------------------------------------------------------------

    def follow(self, qs: list[np.ndarray]) -> None:
        """Slide the joint targets along ``qs``, speeding up and slowing down gently, then wait to arrive."""
        if not qs:
            return
        path = [self.data.ctrl[self.actuators].copy(), *qs]
        lengths = np.array([np.abs(b - a).max() for a, b in zip(path, path[1:], strict=False)])
        total = float(lengths.sum())
        cum = np.concatenate([[0.0], np.cumsum(lengths)])
        # A smooth start and stop peaks at 1.5 times the average speed.
        seconds = max(0.3, 1.5 * total / JOINT_SPEED)
        n = int(seconds / self.model.opt.timestep)
        for k in range(1, n + 1):
            d = smooth(k / n) * total
            i = min(int(np.searchsorted(cum, d, side="right")) - 1, len(path) - 2)
            f = 0.0 if lengths[i] == 0 else (d - cum[i]) / lengths[i]
            self.data.ctrl[self.actuators] = path[i] + (path[i + 1] - path[i]) * min(1.0, f)
            self.step()
        for _ in range(int(CATCH_UP / self.model.opt.timestep)):
            if np.abs(self.data.qpos[self.joints] - qs[-1]).max() < ARRIVED:
                break
            self.step()

    def move_line(
        self,
        end: np.ndarray,
        turn: float | None = None,
        allow: Iterable[int] = (),
        safe: float = SAFE,
    ) -> float:
        """Check, then make, a straight move of the pinch point to ``end``. Returns the smallest gap (m)."""
        qs = self.plan_line(np.asarray(end, dtype=float), turn)
        gap, what = self.clearance(qs, allow)
        if gap < safe:
            raise Blocked(f"moving to {np.round(end, 3)}: {what} would come {gap * 1000:.1f} mm close")
        self.follow(qs)
        if turn is not None:
            self.turn = self.turn + turn_between(self.turn, turn)
        return gap

    def home(self) -> None:
        """Back to the start pose, by sliding the joints (only from up high, where nothing is near)."""
        self.follow([np.array(HOME)])
        self.turn = self.turn_now()
