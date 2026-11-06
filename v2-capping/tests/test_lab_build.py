"""v2's copy of v1's scene builder (capping/lab_build.py) must build v1's scene exactly."""

import numpy as np
import pytest
from lab_scene import scene as lab

from capping import lab_build, scene

FIELDS = (
    "body_pos",
    "body_quat",
    "body_mass",
    "geom_size",
    "geom_pos",
    "geom_friction",
    "qpos0",
    "jnt_range",
)


@pytest.mark.parametrize("seed", [150, 153])
def test_same_as_v1(seed):
    a, ta = lab.build(seed)
    b, tb = lab_build.build(seed)
    assert (a.nbody, a.ngeom, a.njnt, a.nu) == (b.nbody, b.ngeom, b.njnt, b.nu)
    for f in FIELDS:
        assert np.array_equal(getattr(a, f), getattr(b, f)), f
    assert ta == tb


def test_spread_cells_keep_the_liquid():
    v1, t1 = lab.build(151)
    _, t2 = scene.build(151, spread=True)
    assert t1.levels == t2.levels and t1.source_slots == t2.source_slots
    cells = sorted(t2.receiver_slots.values())
    assert tuple(cells) in {tuple(sorted(c)) for c in scene.SPREAD_CELLS}
