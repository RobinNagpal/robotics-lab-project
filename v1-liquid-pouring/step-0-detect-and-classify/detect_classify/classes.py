"""The classes the detector learns, as in docs/solutions/step-0-detect-and-classify.md.

Role (source, receiver or filled tube; which rack is which) is not a class: the
tubes are the same glass test tubes, and the three racks are the same rack.
Tube height (100, 150 or 200 mm) is not a class either, for now.
"""

from __future__ import annotations

CLASSES = ("test_tube", "rack")
CLASS_ID = {name: i for i, name in enumerate(CLASSES)}

# Which simulation container each class is drawn from.
CONTAINER_CLASS = {
    "receiver_100": "test_tube",
    "receiver_150": "test_tube",
    "receiver_200": "test_tube",
    "source_tube_100": "test_tube",
    "source_tube_150": "test_tube",
    "source_tube_200": "test_tube",
}
