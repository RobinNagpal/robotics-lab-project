"""The 50 ml tube's screw thread, made so it runs on a Mac.

AutoBio models the thread as two real helical shapes (signed-distance shapes from its ``mjlab.sdf.thread``
plugin, built for Linux only) that touch with friction 1. MuJoCo's own ``mujoco.sdf.bolt`` and ``nut``
cannot stand in: they are fixed 1 m shapes (a 0.5 m hex head, a 7 cm pitch) and only their thread radius
can be set. So the thread here is made the way ``capping/caps.py`` makes ours, from plain joints:

- **On the thread** the cap is a child of the tube that can only turn about the tube's axis (a hinge)
  and rise along it (a slide), tied together by the pitch: one turn anticlockwise (from above) lifts it
  3 mm. It stops at 0 (screwed down) and at the end of the thread.
- **Off the thread** it is a free body. Until then the free copy is hidden, touches nothing, and is put
  where the threaded one is, moving as it moves, before every step (not welded to it: a weld on a hidden
  body made the simulation blow up). At the end of the thread they swap.

AutoBio's numbers, read from ``centrifuge_50ml_screw.xml``: pitch 3 mm, radius 14 mm. The tube's thread
runs from -1.0 to 1.2 pitches about its centre, the cap's from -1.2 to 1.0 (the plugin's bounding-box
code multiplies these by the pitch), and screwed down they overlap by 5.05 mm: **about 1.7 turns** before
the cap's last thread clears the tube's. That is where the cap comes off here (``TURNS``).

**How hard it is to turn.** AutoBio's cap is not tight: nothing presses its thread together, so turning
it costs only the thread's friction (1) times whatever load is on the thread (the tube's weight when it
hangs from the cap, and any push or pull of the arms). That is copied: each step the friction on the
turn is ``FRICTION x R_EFF x`` the force the thread carried in the last step. A **tight** cap
(``tight=True``, our case) adds the v2 cap's seal: 0.6 N m to start turning, falling to 0.1 N m once it
has turned 20 degrees (numbers from ``capping/caps.py``).

Step a scene with ``step(model, data, thread)``, not ``mujoco.mj_step``; reset it with ``Thread.reset``.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

PITCH = 0.003  # m per turn (AutoBio)
RADIUS = 0.014  # m (AutoBio)
TURNS = 5.05e-3 / PITCH  # 1.68: thread overlap screwed down, in turns
FRICTION = 1.0  # AutoBio's nut-bolt contact pair
FLANK = math.radians(30)  # half the angle of a standard thread's V
R_EFF = RADIUS / math.cos(FLANK)  # m: torque per newton carried, per unit friction
MAX_LOAD_TORQUE = 2.0  # N m: a cap on a sound thread never needs more (a guard against one bad step)
# The tight cap (ours, capping/caps.py).
BREAKAWAY_TORQUE = 0.6  # N m
RUNNING_TORQUE = 0.1  # N m
SEAL_ANGLE = math.radians(20)
END_MARGIN = 0.02 * 2 * math.pi  # rad: this close to the end of the thread, the cap is off
TURN_ARMATURE = 1e-4  # kg m^2 (caps.py: keeps the screw from giving way under the fingers)
RISE_ARMATURE = 0.02  # kg, about the cap's mass
SOLREF = (0.004, 1.0)  # as stiff as AutoBio's thread contact (0.002, raised by MuJoCo to 2 steps)
SOLIMP = (0.99, 0.999, 0.001, 0.5, 2.0)
HIDDEN = 1e-5  # kg: what a cap not in use weighs
# A contact pair cannot be switched off once the model is built (a large negative margin does not do it:
# MuJoCo still makes the contacts). So a hidden cap's geoms in pairs are shrunk to a point on its axis,
# where no finger reaches.
TINY = 1e-6  # m

CAP = "6/centrifuge_50ml_screw_cap"  # on the thread (AutoBio's name)
FREE = "6f/centrifuge_50ml_screw_cap"  # off it
FREE_JOINT = "centrifuge_50ml_screw_cap"  # AutoBio's name for the cap's free joint
TUBE = "7/centrifuge_50ml_screw_body"
CAP_POS = (0.0, 0.0, 0.11965)  # the cap's frame on the tube, screwed down (AutoBio)
CAP_QUAT = (0.0, 1.0, 0.0, 0.0)  # turned over (AutoBio's euler="3.14 0 0")


def add(spec: mujoco.MjSpec, tube_model, turns: float = TURNS) -> None:
    """Put the cap on the tube in ``spec`` (already holding ``TUBE``) on the thread, plus its hidden free
    copy. ``tube_model()`` gives AutoBio's tube model, fresh (attaching takes a body out of its spec)."""
    spec.body(TUBE).add_frame().attach_body(tube_model().body("centrifuge_50ml_screw_cap"), "6/", "")
    cap = spec.body(CAP)
    # The cap's frame is upside down, so the tube's up is its -z.
    open_angle = 2 * math.pi * turns
    turn = cap.add_joint(
        name="cap_turn",
        type=mujoco.mjtJoint.mjJNT_HINGE,
        axis=[0, 0, -1],
        range=[0, math.degrees(open_angle) if spec.compiler.degree else open_angle],
        limited=mujoco.mjtLimited.mjLIMITED_TRUE,
        armature=TURN_ARMATURE,
        solref_limit=list(SOLREF),
        solimp_limit=list(SOLIMP),
        solref_friction=list(SOLREF),
        solimp_friction=list(SOLIMP),
    )
    rise = cap.add_joint(
        name="cap_rise",
        type=mujoco.mjtJoint.mjJNT_SLIDE,
        axis=[0, 0, -1],
        range=[0, PITCH * turns],
        limited=mujoco.mjtLimited.mjLIMITED_TRUE,
        armature=RISE_ARMATURE,
        solref_limit=list(SOLREF),
        solimp_limit=list(SOLIMP),
    )
    spec.add_equality(
        name="cap_screw",
        type=mujoco.mjtEq.mjEQ_JOINT,
        objtype=mujoco.mjtObj.mjOBJ_JOINT,
        name1=rise.name,
        name2=turn.name,
        data=[0, PITCH / (2 * math.pi), 0, 0, 0] + [0] * (mujoco.mjNEQDATA - 5),
        solref=list(SOLREF),
        solimp=list(SOLIMP),
    )
    # The free copy, as AutoBio's free cap: a wrapper body with the free joint, the cap inside it.
    wrap = spec.worldbody.add_body(
        name="centrifuge_50ml_screw_cap_wrap", pos=spec.body("centrifuge_50ml_screw_body_wrap").pos
    )
    wrap.add_freejoint(name=FREE_JOINT)
    wrap.add_frame().attach_body(tube_model().body("centrifuge_50ml_screw_cap"), "6f/", "")
    spec.nuserdata = max(spec.nuserdata, 1)  # userdata[0]: 1 once the cap is off the thread


class Thread:
    """The thread's state in a compiled model: friction each step, and the swap at the end."""

    def __init__(self, model: mujoco.MjModel, tight: bool = False, friction: float = FRICTION):
        self.model = model
        self.tight = tight
        self.friction = friction
        turn = model.joint("cap_turn")
        self.turn_dof, self.turn_adr = int(turn.dofadr[0]), int(turn.qposadr[0])
        self.open_angle = float(model.jnt_range[turn.id][1])
        free = model.joint(FREE_JOINT)
        self.free_adr, self.free_dof = int(free.qposadr[0]), int(free.dofadr[0])
        self.cap = model.body(CAP).id
        self.free = model.body(FREE).id
        self.mass = {
            b: (float(model.body_mass[b]), model.body_inertia[b].copy()) for b in (self.cap, self.free)
        }
        self.geoms = {
            b: [g for g in range(model.ngeom) if model.geom_bodyid[g] == b] for b in (self.cap, self.free)
        }
        gs = [g for v in self.geoms.values() for g in v]
        self.contype = {g: (int(model.geom_contype[g]), int(model.geom_conaffinity[g])) for g in gs}
        self.alpha = {g: float(model.geom_rgba[g, 3]) for g in gs}
        in_pairs = set(model.pair_geom1) | set(model.pair_geom2)
        self.pair_geoms = {b: [g for g in self.geoms[b] if g in in_pairs] for b in (self.cap, self.free)}
        self.size = {g: model.geom_size[g].copy() for v in self.pair_geoms.values() for g in v}
        self.rbound = {g: float(model.geom_rbound[g]) for v in self.pair_geoms.values() for g in v}
        self.max_torque = 0.0
        # The free wrapper's pose relative to the cap: the inverse of the cap's frame in it.
        self._wrap_in_cap = np.zeros(3), np.zeros(4)
        mujoco.mju_negPose(*self._wrap_in_cap, np.array(CAP_POS), np.array(CAP_QUAT))

    # --- state ---
    def on(self, data: mujoco.MjData) -> bool:
        return bool(data.userdata[0] < 0.5)

    def turned(self, data: mujoco.MjData) -> float:
        """Turns undone so far (0 screwed down, ``TURNS`` at the end of the thread)."""
        return float(data.qpos[self.turn_adr]) / (2 * math.pi)

    def cap_body(self, data: mujoco.MjData) -> int:
        return self.cap if self.on(data) else self.free

    def load(self, data: mujoco.MjData) -> float:
        """The force (N) the thread carried between tube and cap in the last step."""
        return float(np.linalg.norm(data.cfrc_int[self.cap, 3:]))

    # --- the swap ---
    def _show(self, body: int, shown: bool) -> None:
        m = self.model
        for g in self.geoms[body]:
            ct, ca = self.contype[g]
            m.geom_contype[g], m.geom_conaffinity[g] = (ct, ca) if shown else (0, 0)
            m.geom_rgba[g, 3] = self.alpha[g] if shown else 0.0
        for g in self.pair_geoms[body]:
            m.geom_size[g] = self.size[g] if shown else TINY
            m.geom_rbound[g] = self.rbound[g] if shown else TINY
        mass, inertia = self.mass[body]
        m.body_mass[body] = mass if shown else HIDDEN
        m.body_inertia[body] = inertia * (1.0 if shown else HIDDEN / mass)
        # MuJoCo skips whole bodies none of whose geoms collide, by flags set at compile time.
        m.body_contype[body] = m.body_conaffinity[body] = 1 if shown else 0

    def _set(self, data: mujoco.MjData, off: bool) -> None:
        self._show(self.free, off)
        self._show(self.cap, not off)
        data.userdata[0] = 1.0 if off else 0.0

    def follow(self, data: mujoco.MjData) -> None:
        """Put the hidden free cap where the threaded one is, moving as it moves (needs data.xpos etc.
        from the last step)."""
        pos, quat = data.xpos[self.cap], data.xquat[self.cap]
        wpos, wquat = np.zeros(3), np.zeros(4)
        mujoco.mju_mulPose(wpos, wquat, pos, quat, *self._wrap_in_cap)
        vel = np.zeros(6)  # angular, linear (world), at the cap's frame
        mujoco.mj_objectVelocity(self.model, data, mujoco.mjtObj.mjOBJ_BODY, self.cap, vel, 0)
        w, v = vel[:3], vel[3:] + np.cross(vel[:3], wpos - pos)
        data.qpos[self.free_adr : self.free_adr + 3] = wpos
        data.qpos[self.free_adr + 3 : self.free_adr + 7] = wquat
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, wquat)
        data.qvel[self.free_dof : self.free_dof + 3] = v
        data.qvel[self.free_dof + 3 : self.free_dof + 6] = rot.reshape(3, 3).T @ w

    def reset(self, data: mujoco.MjData) -> None:
        self._set(data, off=False)
        self.max_torque = 0.0

    # --- stepping ---
    def friction_torque(self, data: mujoco.MjData) -> float:
        torque = min(self.friction * R_EFF * self.load(data), MAX_LOAD_TORQUE)
        if self.tight:
            torque += BREAKAWAY_TORQUE if data.qpos[self.turn_adr] < SEAL_ANGLE else RUNNING_TORQUE
        return torque

    def before_step(self, data: mujoco.MjData) -> None:
        if self.on(data):
            t = self.friction_torque(data)
            self.model.dof_frictionloss[self.turn_dof] = t
            self.max_torque = max(self.max_torque, t)
            self.follow(data)

    def after_step(self, data: mujoco.MjData) -> bool:
        """Swap if the cap has reached the end of its thread. True if it came off just now."""
        if self.on(data) and data.qpos[self.turn_adr] >= self.open_angle - END_MARGIN:
            mujoco.mj_kinematics(self.model, data)
            self.follow(data)
            self._set(data, off=True)
            return True
        return False


def step(model: mujoco.MjModel, data: mujoco.MjData, thread: Thread) -> bool:
    """One step, as AutoBio's ``Manager.step`` (which works out the interaction forces after each step;
    the thread's friction reads them), plus the thread. True if the cap came off in this step."""
    thread.before_step(data)
    mujoco.mj_step(model, data)
    mujoco.mj_rnePostConstraint(model, data)
    return thread.after_step(data)
