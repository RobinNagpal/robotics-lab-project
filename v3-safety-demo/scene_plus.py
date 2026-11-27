"""v3's centrifuge scene with the safety demo's parts added, without changing any v3 file.

v3's ``scene.build(seed)`` makes its scene with ``mujoco.MjSpec`` and returns only the compiled model. While
it runs, we hand it a stand-in for the ``mujoco`` module that records each spec it creates. The first one is
the whole scene; we add the person (``hand.py``) and the safety camera to it and compile it again.

Everything added goes on the end of the world body and has no joints and no collisions, so every body, joint
and actuator v3 knows keeps its number, and the physics is the same as v3's.

``patched()`` swaps v3's ``scene.build`` for ours, so v3's step code (which calls ``scene.build``) runs on the
scene with the person in it.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

import mujoco
import numpy as np
from centrifuge_scene import scene
from centrifuge_scene.settings import IMAGE_SIZE

import cameras
import hand

_V3_BUILD = scene.build


class _RecordingMjSpec:
    """Stands in for ``mujoco.MjSpec``: makes real specs, and remembers the ones made by calling it."""

    def __init__(self) -> None:
        self.made: list[mujoco.MjSpec] = []

    def __call__(self, *args, **kwargs) -> mujoco.MjSpec:
        spec = mujoco.MjSpec(*args, **kwargs)
        self.made.append(spec)
        return spec

    def __getattr__(self, name: str):
        return getattr(mujoco.MjSpec, name)  # from_file, from_string and the rest


class _MujocoWithRecorder:
    """Stands in for the ``mujoco`` module inside v3's scene.py: the same, except for ``MjSpec``."""

    def __init__(self, recorder: _RecordingMjSpec) -> None:
        self.MjSpec = recorder

    def __getattr__(self, name: str):
        return getattr(mujoco, name)


def build_spec(seed: int, scenario: str | None = None, awake: bool = False) -> tuple[mujoco.MjSpec, object]:
    """v3's scene for ``seed`` as a spec (not yet compiled with our parts), and its truth."""
    recorder = _RecordingMjSpec()
    real = scene.mujoco
    scene.mujoco = _MujocoWithRecorder(recorder)
    try:
        _, truth = _V3_BUILD(seed, scenario=scenario, awake=awake)
    finally:
        scene.mujoco = real
    if not recorder.made:
        raise RuntimeError("v3's scene.build made no MjSpec: the way it builds the scene has changed")
    return recorder.made[0], truth


FLOOR_RGBA = (0.30, 0.30, 0.32, 1.0)
FLOOR_DEPTH = 0.90  # m below the bench's top


def add_parts(spec: mujoco.MjSpec) -> None:
    """The demo's parts: the person, the safety camera, and a floor under the bench (seen around its
    edges)."""
    hand.add_person(spec)
    cameras.add_safety_camera(spec)
    # On a body of its own: geoms on the world body itself are numbered first, which would shift v3's.
    spec.worldbody.add_body(name="demo_floor").add_geom(
        name="demo_floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[6.0, 6.0, 0.05],
        pos=[0.0, 0.0, -FLOOR_DEPTH],
        rgba=list(FLOOR_RGBA),
        contype=0,
        conaffinity=0,
    )
    # Room for the safety camera's larger picture.
    spec.visual.global_.offwidth = max(IMAGE_SIZE[0], cameras.SAFETY.width)
    spec.visual.global_.offheight = max(IMAGE_SIZE[1], cameras.SAFETY.height)


def build(seed: int, scenario: str | None = None, awake: bool = False) -> tuple[mujoco.MjModel, object]:
    """Same call and result as v3's ``scene.build``, with the person (parked out of sight) and the camera
    added."""
    spec, truth = build_spec(seed, scenario=scenario, awake=awake)
    add_parts(spec)
    return spec.compile(), truth


@contextlib.contextmanager
def patched() -> Iterator[None]:
    """Inside, v3's code that calls ``scene.build`` gets the scene with the demo's parts."""
    scene.build = build
    try:
        yield
    finally:
        scene.build = _V3_BUILD


def same_as_v3(seed: int) -> list[str]:
    """Checks that adding our parts left v3's scene alone; returns the differences found (empty if none)."""
    a, _ = _V3_BUILD(seed)
    b, _ = build(seed)
    problems = []
    for what in ("nq", "nv", "nu", "njnt", "nsensor"):
        if getattr(a, what) != getattr(b, what):
            problems.append(f"{what}: {getattr(a, what)} -> {getattr(b, what)}")
    for kind, count in (
        (mujoco.mjtObj.mjOBJ_BODY, a.nbody),
        (mujoco.mjtObj.mjOBJ_GEOM, a.ngeom),
        (mujoco.mjtObj.mjOBJ_CAMERA, a.ncam),
        (mujoco.mjtObj.mjOBJ_LIGHT, a.nlight),
    ):
        for i in range(count):
            if mujoco.mj_id2name(a, kind, i) != mujoco.mj_id2name(b, kind, i):
                problems.append(f"{kind.name} {i} renamed")
                break
    if not np.array_equal(a.body_mass, b.body_mass[: a.nbody]):
        problems.append("body masses changed")
    return problems


# --- lighting ----------------------------------------------------------------------------------------------

# Each level scales v3's lights by a range; a frame draws a value inside it. Normal is v3's own lighting.
LIGHTING = {"dim": (0.35, 0.55), "normal": (0.85, 1.15), "bright": (1.5, 1.9)}
_V3_LIGHTS: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def set_lighting(model: mujoco.MjModel, scale: float) -> None:
    """Scale every light in the scene (the lamps and the viewer's headlight) by ``scale``; 1 is v3's
    lighting."""
    key = id(model)
    if key not in _V3_LIGHTS:
        _V3_LIGHTS[key] = (
            model.light_diffuse.copy(),
            model.vis.headlight.diffuse.copy(),
            model.vis.headlight.ambient.copy(),
        )
    lamps, head_diffuse, head_ambient = _V3_LIGHTS[key]
    model.light_diffuse[:] = np.clip(lamps * scale, 0, 1)
    model.vis.headlight.diffuse[:] = np.clip(head_diffuse * scale, 0, 1)
    model.vis.headlight.ambient[:] = np.clip(head_ambient * scale, 0, 1)
