"""The benchtop centrifuge: AutoBio's Eppendorf 5430 housing and lid, with our own six-place rotor.

AutoBio's model comes with a 30 x 1.5 ml rotor. The task uses 50 ml tubes, so that rotor is taken out and
one like Eppendorf's F-35-6-30 is built in its place: six holes leaning 35 degrees. The housing keeps
AutoBio's look and most of its collision boxes; only its floor slab is lowered to make room for the deeper
rotor. The lid keeps AutoBio's hinge and gas spring (it rests open). Closing it past ``LID_LATCH`` lets the
latch catch it: an equality constraint, off until then, holds it shut.

The rotor never spins here, so it is fixed to the housing. Each hole is a ring of thin boxes shaped like the
tube's outline with 0.9 mm to spare and a lead-in at the top; the rotor's solid body, with the six holes cut
out of it, is only drawn.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np
import trimesh

from .settings import ALUMINIUM, CENTRIFUGE_XML, CONVERTED, LID_CLOSED, LID_LATCH, ROTOR, TUBE
from .tubes import ring_boxes

PREFIX = "centrifuge/"
# The rotor's six holes: a site at the bottom of each (where a seated tube's tip rests, with a seated tube's
# axis), and one where each hole's axis meets the rotor's top.
SLOT_SITE = "slot{}"
ENTRY_SITE = "entry{}"
LID_DAMPING = 5.0
# Extra width of a hole over the tube, and the lead-in at its top.
CLEARANCE = 0.0009
LEAD_IN = (0.006, 0.003)  # (height, extra radius)


def slot_frames() -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """For each hole, in the rotor's frame (origin on its axis, at the housing's base): the bottom centre,
    the top centre and the rotation (columns x, y, z: z points out of the hole, along a seated tube)."""
    frames = []
    s, c = math.sin(ROTOR.tilt), math.cos(ROTOR.tilt)
    for k in range(ROTOR.slots):
        phi = 2 * math.pi * k / ROTOR.slots
        out = np.array([math.cos(phi), math.sin(phi), 0.0])
        top = ROTOR.hole_radius_at_top * out + np.array([0, 0, ROTOR.top])
        z = np.array([-s * out[0], -s * out[1], c])  # up and in
        bottom = top - ROTOR.depth * z
        x = np.array([-math.sin(phi), math.cos(phi), 0.0])
        frames.append((bottom, top, np.column_stack([x, np.cross(z, x), z])))
    return frames


def _hole_outline() -> tuple[tuple[float, float], ...]:
    """The inside of a hole, from its bottom up past the top: the tube's outline plus the clearance."""
    heights, radii = zip(*TUBE.outline, strict=True)
    pts = [(z, r + CLEARANCE) for z, r in TUBE.outline if z < ROTOR.depth - LEAD_IN[0]]
    z_lead = ROTOR.depth - LEAD_IN[0]
    r_lead = float(np.interp(z_lead, heights, radii)) + CLEARANCE
    pts += [(z_lead, r_lead), (ROTOR.depth + 0.002, r_lead + LEAD_IN[1])]
    return tuple(pts)


def rotor_mesh_path() -> str:
    """The rotor's body with six blind holes, drawn only. Made once with a mesh boolean and cached."""
    out = CONVERTED / "rotor_f35_6.obj"
    if out.exists():
        return str(out)
    height = ROTOR.top - ROTOR.bottom
    body = trimesh.creation.cylinder(radius=ROTOR.radius, height=height, sections=128)
    body.apply_translation([0, 0, ROTOR.bottom + height / 2])
    knob = trimesh.creation.cylinder(radius=0.022, height=0.024, sections=64)
    knob.apply_translation([0, 0, ROTOR.top + 0.006])
    solid = trimesh.boolean.union([body, knob], engine="manifold")
    holes = []
    for bottom, _, rot in slot_frames():
        length = ROTOR.depth + 0.06
        hole = trimesh.creation.cylinder(radius=ROTOR.hole + 0.0003, height=length, sections=48)
        hole.apply_translation([0, 0, length / 2])
        tf = np.eye(4)
        tf[:3, :3] = rot
        tf[:3, 3] = bottom
        hole.apply_transform(tf)
        holes.append(hole)
    rotor = trimesh.boolean.difference([solid, *holes], engine="manifold")
    CONVERTED.mkdir(parents=True, exist_ok=True)
    rotor.export(out)
    return str(out)


def load() -> mujoco.MjSpec:
    """The centrifuge as a spec, ready to attach: AutoBio's housing and lid, our rotor, the latch."""
    child = mujoco.MjSpec.from_file(str(CENTRIFUGE_XML))
    # AutoBio's file sets no gravity of its own; match the scene's so attaching does not warn.
    child.option.gravity = [0, 0, -9.81]
    lift = 0.0263094  # AutoBio wraps the housing in a frame this high

    # Out with AutoBio's rotor and its meshes, and the housing's floor slab (its top is 125 mm up; our rotor
    # reaches down to 70 mm).
    child.delete(child.body("rotor"))
    for mesh in list(child.meshes):
        if mesh.name.startswith("rotor-"):
            child.delete(mesh)
    housing = child.body("body")
    # AutoBio's hinge damping (20 N m s/rad) makes the lid take 3 to 5 s to shut under a firm push; a real
    # lid takes about a second.
    child.joint("lid").damping[0] = LID_DAMPING
    # Asleep, a shut lid would not notice its latch letting go. One hinge costs nothing to keep awake.
    child.body("lid").sleep = mujoco.mjtSleepPolicy.mjSLEEP_NEVER
    for geom in list(housing.geoms):
        if geom.type == mujoco.mjtGeom.mjGEOM_BOX and abs(geom.size[0] - 0.215) < 1e-3:
            child.delete(geom)
    housing.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.168, 0.215, (ROTOR.bottom - 0.002) / 2],
        pos=[0, -0.002, (ROTOR.bottom - 0.002) / 2 - lift],
        contype=0,
        conaffinity=1,
        group=3,
    )

    # Our rotor, fixed to the housing, its frame on the rotor's axis at the housing's base.
    rotor = housing.add_body(name="rotor", pos=[ROTOR.axis_xy[0], ROTOR.axis_xy[1], -lift])
    child.add_mesh(name="rotor_f35_6", file=rotor_mesh_path())
    rotor.add_geom(
        name="rotor_visual",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="rotor_f35_6",
        rgba=list(ALUMINIUM.rgba),
        contype=0,
        conaffinity=0,
        group=2,
        mass=ROTOR.mass,
    )
    common = dict(friction=[ALUMINIUM.friction, 0.005, 0.0001], group=3, rgba=[0.5, 0.5, 0.5, 0.3], mass=0)
    outline = _hole_outline()
    for k, (bottom, top, rot) in enumerate(slot_frames()):
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, rot.flatten())
        hole = rotor.add_body(name=f"hole{k}", pos=bottom.tolist(), quat=quat.tolist())
        hole.add_site(name=SLOT_SITE.format(k), size=[0.004, 0, 0], group=4)
        hole.add_geom(
            name=f"hole{k}_bottom",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[outline[0][1] + ROTOR.wall, ROTOR.wall / 2, 0],
            pos=[0, 0, -ROTOR.wall / 2],
            **common,
        )
        for i, (half, p, q) in enumerate(ring_boxes(outline, ROTOR.wall, ROTOR.wall_pieces, inward=False)):
            hole.add_geom(
                name=f"hole{k}_wall{i}", type=mujoco.mjtGeom.mjGEOM_BOX, size=half, pos=p, quat=q, **common
            )
        rotor.add_site(name=ENTRY_SITE.format(k), pos=top.tolist(), size=[0.004, 0, 0], group=4)

    # The rotor's top between the holes: a knob in the middle, a web between each two holes, and a rim.
    rotor.add_geom(
        name="rotor_knob",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[0.040, 0.009, 0],
        pos=[0, 0, ROTOR.top - 0.009],
        **common,
    )
    rotor.add_geom(
        name="rotor_cap",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[0.022, 0.012, 0],
        pos=[0, 0, ROTOR.top + 0.006],
        **common,
    )
    for k in range(ROTOR.slots):
        phi = 2 * math.pi * (k + 0.5) / ROTOR.slots
        quat = [math.cos(phi / 2), 0, 0, math.sin(phi / 2)]
        rotor.add_geom(
            name=f"rotor_web{k}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.044, 0.010, 0.004],
            pos=[0.084 * math.cos(phi), 0.084 * math.sin(phi), ROTOR.top - 0.004],
            quat=quat,
            **common,
        )
    rim = 36
    for i in range(rim):
        phi = 2 * math.pi * i / rim
        r_mid = (0.097 + ROTOR.radius) / 2
        rotor.add_geom(
            name=f"rotor_rim{i}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[(ROTOR.radius - 0.097) / 2, ROTOR.radius * math.tan(math.pi / rim) * 1.1, 0.004],
            pos=[r_mid * math.cos(phi), r_mid * math.sin(phi), ROTOR.top - 0.004],
            quat=[math.cos(phi / 2), 0, 0, math.sin(phi / 2)],
            **common,
        )
    return child


def lid_qpos_adr(model: mujoco.MjModel) -> int:
    return model.joint(f"{PREFIX}lid").qposadr[0]


def lid_angle(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    """How far the lid is from closed, in radians: 0 shut, 1.723 fully open."""
    return float(data.qpos[lid_qpos_adr(model)] - LID_CLOSED)


def latch_id(model: mujoco.MjModel) -> int:
    return model.equality(f"{PREFIX}lid-lock").id


def update_latch(model: mujoco.MjModel, data: mujoco.MjData) -> bool:
    """Catch the lid when it is pushed nearly shut, as the real latch does. Returns whether it is latched.

    It catches only a closing lid, so a lid just released (``release_latch``) is free to open.

    Call it after each step (``scene.step`` does). Opening the lid again is ``release_latch``.
    """
    eq = latch_id(model)
    closing = data.qvel[model.joint(f"{PREFIX}lid").dofadr[0]] < -0.05  # rad/s: really moving shut
    if not data.eq_active[eq] and closing and lid_angle(model, data) < LID_LATCH:
        data.eq_active[eq] = 1
    return bool(data.eq_active[eq])


def release_latch(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Let the lid go: the gas spring lifts it clear of the latch (to about 0.2 rad); the arm opens it."""
    data.eq_active[latch_id(model)] = 0


def close_lid(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Shut and latch the lid at once (for tests and pictures; the arm closes it by pushing)."""
    data.qpos[lid_qpos_adr(model)] = LID_CLOSED
    data.qvel[model.joint(f"{PREFIX}lid").dofadr[0]] = 0
    data.eq_active[latch_id(model)] = 1
    mujoco.mj_forward(model, data)
