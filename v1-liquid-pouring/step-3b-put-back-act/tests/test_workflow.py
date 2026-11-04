"""The way out Step 3 needs, planned for a learned pour."""

from __future__ import annotations

import numpy as np
import pytest
from putback.workflow import STARTS

from putback_act.workflow import way_in

START = STARTS / "seed150_size200.npy"


@pytest.mark.skipif(not START.exists(), reason="needs Step 2's saved Step 1b runs (make starts there)")
def test_way_in_starts_where_step_1b_left_arm_b():
    from pour import start as starts
    from pour.tilt import Tilter

    path = way_in(START)
    model, data, containers, meta = starts.load(START)
    source = meta["source"]
    tilter = Tilter(model, data, source, containers[source].height, containers[source].radius)
    assert len(path) > 10
    assert np.allclose(path[0], data.qpos[tilter.joints])
    # Small joint steps all the way, so following it backwards is smooth.
    assert max(np.abs(np.diff(np.array(path), axis=0)).max(axis=1)) < 0.2
