"""Step 0's own code, without asking Gemini: the geometry, reading answers, matching cells, the vote."""

from __future__ import annotations

import json

import numpy as np
import pytest

from bench import CELLS, take_pictures
from evaluate import CONFIGS, Oracle, score, true_table
from find import find_separate, find_together, parse

CAMERAS = ("overhead", "racks_high", "racks_low")


@pytest.fixture(scope="module")
def seed0():
    views, (model, data, truth) = take_pictures(0, CAMERAS)
    return views, true_table(model, data, truth)


def test_a_point_seen_by_a_camera_comes_back_to_the_same_place(seed0):
    views, table = seed0
    for view in views.values():
        cam = view.camera
        for t in table.values():
            u, v, depth = cam.project(t["top"])
            assert np.allclose(cam.unproject(u, v, depth), t["top"], atol=1e-9)
            assert np.allclose(cam.ray(u, v), (t["top"] - cam.pos) / np.linalg.norm(t["top"] - cam.pos))


def test_the_depth_picture_agrees_with_the_geometry(seed0):
    # The overhead camera looks straight down onto the caps: its depth there should match the cap's height.
    # (Over an open tube it looks down inside it, so only capped tubes are checked.)
    views, table = seed0
    cam, depth = views["overhead"].camera, views["overhead"].depth
    for t in (t for t in table.values() if t["kind"] == "sample"):
        u, v, d = cam.project(t["top"])
        assert depth[int(v), int(u)] == pytest.approx(d, abs=0.005)


def test_answers_are_read_with_or_without_fences_and_bad_items_are_skipped():
    text = (
        '```json\n[{"point": [410, 233], "label": "sample red", "kind": "sample", "colour": "red"},'
        ' {"point": [5], "label": "broken"}, {"point": [600, 120], "label": "water blue"}]\n```'
    )
    items = parse(text)
    assert [i["kind"] for i in items] == ["sample", "water"]
    assert items[1]["colour"] == "blue"  # taken from the label when the field is missing
    assert parse("no tubes here") == []


def test_perfect_answers_give_a_perfect_score_for_every_set_of_pictures(seed0):
    views, table = seed0
    oracle = Oracle(views, table)
    for mode, cams in CONFIGS.values():
        subset = {c: views[c] for c in cams}
        found, _, per_camera = (find_separate if mode == "separate" else find_together)(subset, oracle)
        s = score(found, table, per_camera, views)
        assert s["found"] == s["tubes"] and not s["made_up"] and not s["wrong"], (cams, s)
        assert max(s["px"]) < 1.0  # perfect points land on the true tube top, to the pixel


def test_a_tube_seen_by_only_one_of_three_pictures_is_not_believed(seed0):
    views, table = seed0
    oracle = Oracle(views, table)
    ghost = {"point": [500, 500], "kind": "sample", "colour": "red", "label": "sample red"}

    def ask(pngs, prompt):
        answer = oracle(pngs, prompt)
        if pngs[0] == views["racks_low"].png:  # the low camera alone "sees" an extra tube
            answer.text = json.dumps([*json.loads(answer.text), ghost])
        return answer

    found, _, _ = find_separate(views, ask)
    assert set(found) == set(table)


def test_there_are_sixteen_known_cells():
    assert len(CELLS) == 16 and len({c.key for c in CELLS}) == 16
