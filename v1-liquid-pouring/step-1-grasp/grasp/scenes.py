"""Crowded random scenes, the close-up depth camera, and the fast grasp check.

The scenes are Step 0's random scenes (detect_classify.random_scene), with
fuller racks: in the task's own scene tubes stand three cells apart, and
every turn of the gripper fits, so there would be nothing to learn. Here
tubes stand in touching cells and in corners, and some turns are blocked.

One scene serves everything. It holds both arms (for grasps run with full
physics, trial.py), an open gripper on its own that is put in place by hand
(for the fast check), and the close-up camera. It is let settle first: a
25 mm tube in a 30 mm cell leans a little, and a 200 mm tube leaning one
degree moves its rim by 3.5 mm. The labels, the pictures and the physics
runs all start from that same settled state.

The tubes are left **empty**. Full of the simulation's stiff balls, every
tube rattles and the racks never come to rest (tubes keep leaning up to 8
degrees); empty, they settle within 0.2 degrees. Liquid only changes how
heavy a tube is, so the squeeze is tested separately with full tubes
(``liquid_in``).
"""

from __future__ import annotations

import math

import mujoco
import numpy as np
from detect_classify import random_scene
from detect_classify.camera import Calibration
from lab_scene import liquid
from lab_scene import scene as lab_scene
from lab_scene.settings import ARMS, GRIPPER_XML, TIMESTEP

from . import rules

CROWDED = {"source_rack": (2, 8), "empty_rack": (2, 8), "destination_rack": (0, 4)}

# The close-up camera: straight down, on the gripper's axis, this far above
# the rim Step 0 reported. The picture is 128 x 128 and covers about 20 cm at
# the rim's height (1.6 mm a pixel), enough to see every tube the open
# fingers could touch.
LOOK = "look"
LOOK_HEIGHT = 0.25
LOOK_FOVY = 45.0
LOOK_SIZE = 128
HAND = "hand"
WRIST = "hand_wrist"
PARK = np.array([0.0, 0.0, 3.0])  # where the gripper waits, out of every picture


SETTLE = 0.3  # s


def _arms_gripper_camera(spec: mujoco.MjSpec) -> None:
    """Both arms, as random_scene adds them; an open gripper moved by hand; the close-up camera."""
    for base in ARMS:
        turn = [math.cos(base.yaw / 2), 0, 0, math.sin(base.yaw / 2)]
        frame = spec.worldbody.add_frame(pos=list(base.pos), quat=turn)
        spec.attach(lab_scene.arm_with_gripper(), prefix=f"{base.name}/", frame=frame)
    hand = spec.worldbody.add_body(name=HAND, pos=PARK.tolist(), mocap=True)
    spec.attach(mujoco.MjSpec.from_file(str(GRIPPER_XML)), prefix="gripper/", frame=hand.add_frame())
    # The arm's wrist behind the gripper, as a rod: off (it touches nothing) for top-down
    # checks, where the wrist is above everything; the side-grasp check turns it on.
    hand.add_geom(
        name=WRIST,
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        fromto=[0, 0, 0, 0, 0, -0.20],
        size=[0.045, 0, 0],
        contype=0,
        conaffinity=0,
        group=4,
        rgba=[0, 0, 0, 0],
    )
    # x to the right of the picture is the table's x, up the picture its y.
    spec.worldbody.add_camera(name=LOOK, pos=[0, 0, 1], xyaxes=[1, 0, 0, 0, 1, 0], fovy=LOOK_FOVY)


def build(seed: int, liquid_in: str | None = None):
    """The crowded scene of ``seed``, arms at home, settled. The same every time for one seed.

    Only the tube named ``liquid_in``, if any, holds liquid, at the task's level for its role.
    """
    order = iter(range(10**6))

    def fill(rng: np.random.Generator, role: str) -> float:
        # Called once per tube, in the order the tubes were named (role_number).
        level = random_scene.task_level(rng, role)
        return level if f"{role}_{next(order)}" == liquid_in else 0.0

    model, data, scene = random_scene.build(
        seed, fill=fill, hand=_arms_gripper_camera, tubes_per_rack=CROWDED
    )
    # The simulation's own settings for contact-rich work (lab_scene.scene.build).
    model.opt.timestep = TIMESTEP
    model.opt.solver = mujoco.mjtSolver.mjSOL_CG
    model.opt.iterations = 100
    model.opt.ls_iterations = 10
    lab_scene.reset(model, data)
    for _ in range(int(SETTLE / TIMESTEP)):
        mujoco.mj_step(model, data)
    return model, data, scene


def rim(data: mujoco.MjData, tube) -> tuple[np.ndarray, float]:
    """Where a tube's rim centre is: (x, y) on the table and its height.

    The tube's body starts at its bottom; a leaning tube's rim is off to one side.
    """
    axis = data.body(tube.name).xmat.reshape(3, 3)[:, 2]
    point = data.body(tube.name).xpos + axis * random_scene.BY_NAME[tube.kind].height
    return point[:2].copy(), float(point[2])


def tubes(scene) -> list:
    return [i for i in scene.instances if i.cls == "test_tube"]


def yaw_quat(yaw: float) -> np.ndarray:
    """The gripper's turn with its fingers pointing straight down and opening along ``yaw``."""
    turn = np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
    # The gripper's fingers point along its own +z and open along its x; half
    # a turn about x points them down.
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, turn, np.array([0.0, 1.0, 0.0, 0.0]))
    return out


class Checker:
    """Puts the open gripper along the way down and asks MuJoCo whether it touches anything.

    No physics is run: each pose is one collision test, under a millisecond.
    The liquid is left out of the test; it is inside the tubes, where the
    fingers never go.
    """

    STEP = 0.005  # between poses on the way down, m
    # Points between open (0) and closed (1) at which the closing fingers are tested.
    CLOSING = (0.25, 0.5, 0.75, 1.0)
    # As the fingers close, the first one to reach the tube pushes the arm a
    # little along the finger line (the tube has about 2.5 mm of play in its
    # cell, and the arm gives). The closing fingers are tested this far either way.
    SHOVE = 0.003

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model, self.data = model, data
        self.mocap = model.body_mocapid[model.body(HAND).id]
        gripper = {i for i in range(model.nbody) if model.body(i).name.startswith("gripper/")}
        self.is_gripper = np.array([model.geom_bodyid[g] in gripper for g in range(model.ngeom)])
        balls = np.array(
            [model.body(model.geom_bodyid[g]).name.startswith(liquid.BALL_PREFIX) for g in range(model.ngeom)]
        )
        model.geom_contype[balls] = 0
        model.geom_conaffinity[balls] = 0
        # How far the pinch point is below the hand body, fingers down.
        self.pinch_below = 0.0
        self.place(np.array([0.0, 0.0, 2.0]), 0.0)
        mujoco.mj_kinematics(model, data)
        self.pinch_below = 2.0 - float(data.site("gripper/pinch").xpos[2])
        self.fingers = np.array(
            [model.jnt_qposadr[j] for j in range(model.njnt) if model.joint(j).name.startswith("gripper/")]
        )
        self.closed = _closed_fingers()
        self.park()

    def place(self, pinch: np.ndarray, yaw: float) -> None:
        self.data.mocap_pos[self.mocap] = pinch + [0, 0, self.pinch_below]
        self.data.mocap_quat[self.mocap] = yaw_quat(yaw)

    def park(self) -> None:
        self.data.qpos[self.fingers] = 0.0
        self.data.mocap_pos[self.mocap] = PARK
        mujoco.mj_forward(self.model, self.data)

    def touches(self, pinch: np.ndarray, yaw: float, closing: float = 0.0) -> list[str]:
        """The bodies the gripper touches with its pinch point at ``pinch``.

        ``closing`` runs from 0 (open) to 1 (closed on a tube).
        """
        self.place(pinch, yaw)
        self.data.qpos[self.fingers] = closing * self.closed
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_collision(self.model, self.data)
        hit = []
        for c in self.data.contact[: self.data.ncon]:
            a, b = self.is_gripper[c.geom1], self.is_gripper[c.geom2]
            if a != b:
                other = c.geom2 if a else c.geom1
                hit.append(self.model.body(self.model.geom_bodyid[other]).name)
        return hit

    def clear(self, xy: np.ndarray, yaw: float, top: float, target: str) -> bool:
        """Whether the gripper gets down to the grip height and closes on ``target`` touching nothing else.

        On the Robotiq gripper the finger linkage moves as the fingers close,
        and the arm is shoved a little sideways, so a neighbour the open
        fingers miss on the way down can still be grazed.
        """
        low = rules.grip_height(top)
        # The bottom first: that is where most grasps are blocked.
        for z in np.arange(low, top + rules.APPROACH + 1e-9, self.STEP):
            if self.touches(np.array([xy[0], xy[1], z]), yaw):
                return False
        along = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        for shove in (0.0, -self.SHOVE, self.SHOVE):
            pinch = np.array([xy[0], xy[1], low]) + shove * along
            if not all(set(self.touches(pinch, yaw, f)) <= {target} for f in self.CLOSING):
                return False
        return True

    def labels(self, xy: np.ndarray, top: float, target: str) -> np.ndarray:
        """For each of the 18 turns, whether the grasp is clear. Parks the gripper after."""
        out = np.array([self.clear(xy, rules.bin_yaw(k), top, target) for k in range(rules.BINS)])
        # Out of the way, or the close-up camera would see it hanging over a tube.
        self.park()
        return out


_CLOSED: np.ndarray | None = None


def _closed_fingers() -> np.ndarray:
    """The gripper's joint angles closed on a 25 mm tube, measured once in a tiny simulation."""
    global _CLOSED
    if _CLOSED is None:
        spec = mujoco.MjSpec()
        spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        spec.option.impratio = 10
        hand = spec.worldbody.add_body(name=HAND, pos=[0, 0, 0.5], mocap=True)
        spec.attach(mujoco.MjSpec.from_file(str(GRIPPER_XML)), prefix="gripper/", frame=hand.add_frame())
        model = spec.compile()
        data = mujoco.MjData(model)
        data.mocap_quat[0] = yaw_quat(0.0)
        mujoco.mj_forward(model, data)
        pinch = data.site("gripper/pinch").xpos.copy()
        # A fixed glass rod where the tube would be.
        spec.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[rules.TUBE_RADIUS, 0.05], pos=pinch.tolist()
        )
        model = spec.compile()
        data = mujoco.MjData(model)
        data.mocap_quat[0] = yaw_quat(0.0)
        for _ in range(1500):
            data.ctrl[0] = lab_scene.grip_command(2 * rules.TUBE_RADIUS)
            mujoco.mj_step(model, data)
        joints = [j for j in range(model.njnt) if model.joint(j).name.startswith("gripper/")]
        _CLOSED = np.array([data.qpos[model.jnt_qposadr[j]] for j in joints])
    return _CLOSED


class LookCamera:
    """The close-up depth picture, as a height above the table for each pixel."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model, self.data = model, data
        self.cam = model.camera(LOOK).id
        self.renderer = mujoco.Renderer(model, LOOK_SIZE, LOOK_SIZE)
        self.renderer.enable_depth_rendering()

    def look(
        self, xy: np.ndarray, top: float, height: float = LOOK_HEIGHT, yaw: float = 0.0
    ) -> tuple[np.ndarray, Calibration]:
        """Heights (m) seen from ``height`` above (``xy``, ``top``), and the camera's calibration.

        With ``yaw``, the camera is turned about the vertical by that much: the picture's right
        is then that direction on the table (side grasps, version 2: the rack's own side).
        """
        self.model.cam_pos[self.cam] = [xy[0], xy[1], top + height]
        self.model.cam_quat[self.cam] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera=LOOK)
        depth = self.renderer.render().copy()
        calib = Calibration.from_model(self.model, self.data, LOOK, (LOOK_SIZE, LOOK_SIZE))
        # The camera looks straight down, so depth is the drop from the camera.
        heights = np.clip(calib.pos[2] - depth, 0.0, None).astype(np.float32)
        return heights, calib

    def close(self) -> None:
        self.renderer.close()
