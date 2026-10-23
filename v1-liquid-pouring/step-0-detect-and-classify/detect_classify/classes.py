"""The classes the detector learns, as in docs/solutions/step-0-detect-and-classify.md.

Role (source or receiver, which rack is which) is not a class: a source tube
and a receiving tube look the same.
"""

from __future__ import annotations

CLASSES = ("test_tube", "centrifuge_tube", "beaker", "flask", "rack", "tray", "pouring_stand")
CLASS_ID = {name: i for i, name in enumerate(CLASSES)}

# Which simulation container each class is drawn from.
CONTAINER_CLASS = {
    "receiver_100": "test_tube",
    "receiver_150": "test_tube",
    "receiver_200": "test_tube",
    "source_tube": "test_tube",
    "source_centrifuge_tube": "centrifuge_tube",
    "source_beaker": "beaker",
    "source_flask": "flask",
}
