"""The model registry: the list of approved detector model files and their SHA-256 fingerprints.

A model is loaded only through ``Registry.read_approved``: it reads the file's bytes once, fingerprints those
bytes, and hands them back only if the fingerprint is an approved one. So the file checked is the file loaded
(nothing can swap it in between). A refused file raises ``UnapprovedModel`` and, given a safety log, the
attempt
is logged.

The registry also says which approved version is in use (``active``) and keeps the order versions were made
active in, so an update can be rolled back to the version before it.

    reg = Registry("models/registry.json")
    reg.approve("models/detector_v1.pt", "v1", note="first release")
    reg.activate("v1")
    blob, entry = reg.read_approved("models/detector_v1.pt", log)

From the command line:

    python registry.py models/registry.json approve models/detector_v1.pt v1 --note "first release"
    python registry.py models/registry.json activate v1
    python registry.py models/registry.json rollback
    python registry.py models/registry.json show
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


class UnapprovedModel(RuntimeError):
    """A model file whose fingerprint is not in the registry (or whose version was revoked)."""


def fingerprint(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def fingerprint_file(path: str | Path) -> str:
    return fingerprint(Path(path).read_bytes())


class Registry:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if self.path.exists():
            self.data = json.loads(self.path.read_text())
        else:
            self.data = {"models": {}, "active": None, "history": []}

    # --- what is in it
    # ---------------------------------------------------------------------------------------

    @property
    def models(self) -> dict[str, dict]:
        return self.data["models"]

    @property
    def active(self) -> str | None:
        return self.data["active"]

    def find(self, sha256: str) -> dict | None:
        """The approved entry with this fingerprint, if any."""
        for e in self.models.values():
            if e["sha256"] == sha256 and not e.get("revoked"):
                return e
        return None

    # --- changing it ---------------------------------------------------------------------------------------

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=1) + "\n")

    def approve(self, model_path: str | Path, version: str, note: str = "") -> dict:
        blob = Path(model_path).read_bytes()
        if version in self.models:
            raise ValueError(f"version {version!r} is already in the registry")
        entry = {
            "version": version,
            "file": Path(model_path).name,
            "sha256": fingerprint(blob),
            "bytes": len(blob),
            "note": note,
        }
        self.models[version] = entry
        self.save()
        return entry

    def revoke(self, version: str) -> None:
        self.models[version]["revoked"] = True
        if self.active == version:
            self.data["active"] = None
        self.save()

    def activate(self, version: str) -> None:
        e = self.models.get(version)
        if e is None or e.get("revoked"):
            raise UnapprovedModel(f"version {version!r} is not an approved model")
        self.data["active"] = version
        self.data["history"].append(version)
        self.save()

    def rollback(self) -> str:
        """Make the version active before the current one active again, and return it."""
        history = self.data["history"]
        current = self.active
        earlier = [v for v in history if v != current and not self.models[v].get("revoked")]
        if not earlier:
            raise UnapprovedModel("there is no earlier approved version to roll back to")
        self.activate(earlier[-1])
        return earlier[-1]

    # --- loading -------------------------------------------------------------------------------------------

    def read_approved(self, model_path: str | Path, log=None) -> tuple[bytes, dict]:
        """The file's bytes and its registry entry, or ``UnapprovedModel`` (logged if ``log`` is given)."""
        blob = Path(model_path).read_bytes()
        sha = fingerprint(blob)
        entry = self.find(sha)
        if entry is None:
            reason = f"model file {Path(model_path).name} (sha256 {sha[:12]}...) is not an approved model"
            if log is not None:
                log.append(kind="registry", action="refused", reason=reason, sha256=sha, file=str(model_path))
            raise UnapprovedModel(reason)
        if log is not None:
            log.append(
                kind="registry",
                action="loaded",
                reason="",
                model=entry["version"],
                sha256=sha,
                file=str(model_path),
            )
        return blob, entry

    def read_active(self, models_dir: str | Path, log=None) -> tuple[bytes, dict]:
        """The active version's file, from ``models_dir``, checked as ``read_approved``."""
        if self.active is None:
            raise UnapprovedModel("no model version is active")
        blob, entry = self.read_approved(Path(models_dir) / self.models[self.active]["file"], log)
        if entry["version"] != self.active:
            raise UnapprovedModel(
                f"the active model's file holds version {entry['version']}, not {self.active}"
            )
        return blob, entry


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("registry")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("approve")
    a.add_argument("model")
    a.add_argument("version")
    a.add_argument("--note", default="")
    sub.add_parser("activate").add_argument("version")
    sub.add_parser("revoke").add_argument("version")
    sub.add_parser("rollback")
    sub.add_parser("show")
    args = p.parse_args()
    reg = Registry(args.registry)
    match args.cmd:
        case "approve":
            print(reg.approve(args.model, args.version, args.note))
        case "activate":
            reg.activate(args.version)
        case "revoke":
            reg.revoke(args.version)
        case "rollback":
            print("active:", reg.rollback())
    print(json.dumps(reg.data, indent=1))


if __name__ == "__main__":
    main()
