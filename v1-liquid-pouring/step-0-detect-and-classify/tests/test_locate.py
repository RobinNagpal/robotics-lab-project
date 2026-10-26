"""The camera maths round-trips, and the labels and locating agree with the scene."""

from __future__ import annotations

import numpy as np
import pytest

from detect_classify import generate_data
from detect_classify.camera import Calibration
from detect_classify.classes import CLASS_ID
from detect_classify.evaluate import true_masks
from detect_classify.locate import locate, own_pixels


def test_project_undoes_deproject():
    rot = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    calib = Calibration(1280, 960, 50.0, np.array([0.4, 0.0, 1.2]), rot)
    u, v, d = np.array([10.0, 640.0, 1200.0]), np.array([20.0, 480.0, 900.0]), np.array([1.0, 1.1, 1.2])
    uv = calib.project(calib.deproject(u, v, d))
    assert np.allclose(uv, np.stack([u, v], axis=1))


@pytest.fixture(scope="module")
def rendered():
    return generate_data.render(seed=12345)


def test_every_labelled_object_has_one_line(rendered):
    _, _, records, _, lines = rendered
    labelled = [r for r in records if r["label"] is not None]
    assert len(labelled) == len(lines)
    for r in labelled:
        assert int(lines[r["label"]].split()[0]) == CLASS_ID[r["cls"]]


def test_true_outlines_locate_within_a_centimetre(rendered, tmp_path):
    _, depth, records, calib, lines = rendered
    label_file = tmp_path / "labels.txt"
    label_file.write_text("\n".join(lines))
    truth = true_masks(label_file, (calib.width, calib.height))
    own = own_pixels(truth)
    by_label = {r["label"]: r for r in records if r["label"] is not None}
    for j, (cls, _) in enumerate(truth):
        where = locate(cls, own[j], depth, calib)
        true = by_label[j]
        assert np.hypot(where.x - true["pos"][0], where.y - true["pos"][1]) < 0.01, true["name"]
