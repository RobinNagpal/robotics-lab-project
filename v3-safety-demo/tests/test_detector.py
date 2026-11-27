import numpy as np
import pytest

from detector import CLEAR, STOP, WARNING, Frame, PerfectDetector
from standin import ARM_RADIUS, STOP_DISTANCE, WARNING_DISTANCE, distance_to_chain, repeating_visits, zone_of

ARM = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # a straight vertical arm, 1 m tall


@pytest.mark.parametrize(
    "gap, zone",
    [
        (0.01, STOP),
        (STOP_DISTANCE - 0.01, STOP),
        (STOP_DISTANCE + 0.01, WARNING),
        (WARNING_DISTANCE + 0.01, CLEAR),
    ],
)
def test_zone_by_distance_from_the_arm(gap, zone):
    hand = np.array([[ARM_RADIUS + gap, 0.0, 0.5]])
    z, d = zone_of(hand, ARM)
    assert z == zone and d == pytest.approx(gap)


def test_distance_to_the_ends_and_the_nearest_point():
    assert distance_to_chain(np.array([[0, 0, 1.5]]), ARM) == pytest.approx(0.5)
    two = np.array([[3.0, 0, 0.5], [0.2, 0, 0.5]])
    assert distance_to_chain(two, ARM) == pytest.approx(0.2)


def test_perfect_detector():
    det = PerfectDetector()
    assert det.detect(Frame(1, 0.0, truth={"hand": None, "arm": ARM})).zone == CLEAR
    d = det.detect(Frame(2, 0.1, truth={"hand": np.array([[ARM_RADIUS + 0.05, 0, 0.5]]), "arm": ARM}))
    assert (d.zone, d.score, d.model_version) == (STOP, 1.0, "perfect-1")
    with pytest.raises(ValueError):
        det.detect(Frame(3, 0.2, image=np.zeros((4, 4, 3))))


def test_repeating_visits():
    v = repeating_visits(10, 30, 75)
    assert [(x.start, x.end) for x in v] == [(10, 14), (14, 16), (40, 44), (44, 46), (70, 74), (74, 76)]
    assert zone_of(np.array([[ARM_RADIUS + v[0].distance, 0, 0.5]]), ARM)[0] == WARNING
    assert zone_of(np.array([[ARM_RADIUS + v[1].distance, 0, 0.5]]), ARM)[0] == STOP
