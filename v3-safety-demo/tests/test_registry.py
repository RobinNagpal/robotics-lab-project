import pytest

from detector import Detection, load
from registry import Registry, UnapprovedModel, fingerprint_file
from safety_log import SafetyLog, verify


@pytest.fixture
def files(tmp_path):
    v1 = tmp_path / "detector_v1.bin"
    v2 = tmp_path / "detector_v2.bin"
    v1.write_bytes(b"weights one")
    v2.write_bytes(b"weights two")
    return tmp_path, v1, v2


def test_approved_model_loads_and_is_logged(files):
    d, v1, _ = files
    reg = Registry(d / "registry.json")
    reg.approve(v1, "v1")
    log = SafetyLog()
    blob, entry = reg.read_approved(v1, log)
    assert blob == b"weights one" and entry["version"] == "v1"
    assert entry["sha256"] == fingerprint_file(v1)
    assert log.entries[-1]["action"] == "loaded" and log.entries[-1]["model"] == "v1"
    # The registry survives a reload from disk.
    assert Registry(d / "registry.json").models["v1"]["sha256"] == entry["sha256"]


def test_unknown_or_changed_model_is_refused_and_logged(files):
    d, v1, v2 = files
    reg = Registry(d / "registry.json")
    reg.approve(v1, "v1")
    log = SafetyLog()
    with pytest.raises(UnapprovedModel):
        reg.read_approved(v2, log)
    v1.write_bytes(b"weights one, retrained without approval")
    with pytest.raises(UnapprovedModel):
        reg.read_approved(v1, log)
    assert [e["action"] for e in log.entries] == ["refused", "refused"]
    assert all(e["kind"] == "registry" and e["sha256"] for e in log.entries)
    assert verify(log.entries, head=log.head) == []


def test_revoked_model_is_refused(files):
    d, v1, _ = files
    reg = Registry(d / "registry.json")
    reg.approve(v1, "v1")
    reg.activate("v1")
    reg.revoke("v1")
    assert reg.active is None
    with pytest.raises(UnapprovedModel):
        reg.read_approved(v1)
    with pytest.raises(UnapprovedModel):
        reg.activate("v1")


def test_update_and_rollback(files):
    d, v1, v2 = files
    reg = Registry(d / "registry.json")
    reg.approve(v1, "v1")
    reg.approve(v2, "v2")
    reg.activate("v1")
    reg.activate("v2")
    assert reg.read_active(d)[1]["version"] == "v2"
    assert reg.rollback() == "v1"
    assert Registry(d / "registry.json").active == "v1"
    assert reg.read_active(d)[0] == b"weights one"
    with pytest.raises(ValueError):
        reg.approve(v2, "v2")


def test_rollback_needs_an_earlier_version(files):
    d, v1, _ = files
    reg = Registry(d / "registry.json")
    reg.approve(v1, "v1")
    reg.activate("v1")
    with pytest.raises(UnapprovedModel):
        reg.rollback()


class Fake:
    def __init__(self, version):
        self.version = version

    def detect(self, frame):
        return Detection("clear", 0.9, self.version)


def test_detector_load_goes_through_the_registry(files):
    d, v1, v2 = files
    reg = Registry(d / "registry.json")
    reg.approve(v1, "v1")
    log = SafetyLog()
    det = load(v1, reg, log, lambda blob, version: Fake(version))
    assert det.version == "v1"
    with pytest.raises(UnapprovedModel):
        load(v2, reg, log, lambda blob, version: Fake(version))
    with pytest.raises(ValueError):
        load(v1, reg, log, lambda blob, version: Fake("v9"))
    assert [e["action"] for e in log.entries] == ["loaded", "refused", "loaded"]
