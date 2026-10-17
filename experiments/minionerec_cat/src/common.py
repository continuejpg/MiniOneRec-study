"""MiniOneRec-CAT -- shared paths, config I/O, hashing and the P0 artefact guard.

Design rule enforced here
-------------------------
Every artefact this experiment produces lives under ``EXP_ROOT`` and is keyed by a
*run tag* (``b0``, ``b2``).  The P0 artefacts under ``data/Amazon`` are treated as
**read-only inputs**: :func:`read_p0` only ever opens them for reading, and
:func:`assert_outside_p0` refuses to hand back any path that would overwrite one.

Nothing in this package writes to ``data/Amazon`` or ``checkpoints``.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

# --------------------------------------------------------------------------- #
# Roots
# --------------------------------------------------------------------------- #

#: ``experiments/minionerec_cat`` -- this package's own directory.
EXP_DIR = Path(__file__).resolve().parents[1]

#: Repository root (``experiments/minionerec_cat/src`` -> repo).
REPO_ROOT = EXP_DIR.parents[1]

#: Isolated output root.  Everything we generate goes here and only here.
EXP_ROOT = EXP_DIR / "runs"

#: P0 inputs -- READ ONLY.
P0_DATA = REPO_ROOT / "data" / "Amazon"

CATEGORY = "Industrial_and_Scientific"

#: The five SID tokens a SID may be built from, in order (upstream `generate_indices.py:88`).
SID_PREFIXES: tuple[str, ...] = ("<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>", "<e_{}>")


# --------------------------------------------------------------------------- #
# Run tags
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class RunPaths:
    """Every path one run tag owns.  Nothing outside ``root`` is ever written."""

    tag: str
    root: Path

    # -- inputs (read only) ------------------------------------------------- #
    @property
    def p0_index(self) -> Path:
        return P0_DATA / "index" / f"{CATEGORY}.index.json"

    @property
    def p0_item(self) -> Path:
        return P0_DATA / "index" / f"{CATEGORY}.item.json"

    @property
    def p0_emb(self) -> Path:
        """The *shipped* text embedding.  Never modified; B2 writes a sibling."""
        return P0_DATA / "index" / f"{CATEGORY}.emb-qwen-td.npy"

    def p0_split(self, split: str) -> Path:
        return P0_DATA / split / f"{CATEGORY}_5_2016-10-2018-11.csv"

    @property
    def p0_info(self) -> Path:
        return P0_DATA / "info" / f"{CATEGORY}_5_2016-10-2018-11.txt"

    # -- our own outputs ---------------------------------------------------- #
    @property
    def emb(self) -> Path:
        """The embedding this run quantises.  B0 copies the P0 file here."""
        return self.root / "emb" / f"{CATEGORY}.emb-{self.tag}.npy"

    @property
    def rqvae_ckpt_dir(self) -> Path:
        return self.root / "rqvae"

    @property
    def rqvae_ckpt(self) -> Path:
        return self.rqvae_ckpt_dir / "best_collision_model.pth"

    @property
    def index(self) -> Path:
        return self.root / "index" / f"{CATEGORY}.index.{self.tag}.json"

    @property
    def derived_root(self) -> Path:
        """Root of the SID-derived split CSVs and info file."""
        return self.root / "derived"

    def derived_split(self, split: str) -> Path:
        return self.derived_root / split / f"{CATEGORY}_5_2016-10-2018-11.csv"

    @property
    def derived_info(self) -> Path:
        return self.derived_root / "info" / f"{CATEGORY}_5_2016-10-2018-11.txt"

    @property
    def report_dir(self) -> Path:
        return self.root / "reports"

    @property
    def log_dir(self) -> Path:
        return self.root / "logs"

    def all_output_dirs(self) -> tuple[Path, ...]:
        return (
            self.emb.parent,
            self.rqvae_ckpt_dir,
            self.index.parent,
            self.derived_root / "train",
            self.derived_root / "valid",
            self.derived_root / "test",
            self.derived_root / "info",
            self.report_dir,
            self.log_dir,
        )

    def ensure_dirs(self) -> None:
        for d in self.all_output_dirs():
            d.mkdir(parents=True, exist_ok=True)


def run_paths(tag: str) -> RunPaths:
    """Resolve a run tag to its isolated path set."""
    if not tag or not all(c.isalnum() or c in "_-" for c in tag):
        raise ValueError(f"run tag must be alphanumeric/underscore/dash, got {tag!r}")
    return RunPaths(tag=tag, root=EXP_ROOT / tag)


# --------------------------------------------------------------------------- #
# The P0 guard
# --------------------------------------------------------------------------- #

def assert_outside_p0(path: os.PathLike[str] | str) -> Path:
    """Refuse any write target that would land inside the P0 input tree.

    This is the single choke point that makes "we never overwrite P0" a property
    of the code rather than of our discipline.
    """
    resolved = Path(path).resolve()
    for protected in (P0_DATA.resolve(), (REPO_ROOT / "checkpoints").resolve()):
        if resolved == protected or protected in resolved.parents:
            raise PermissionError(
                f"refusing to write {resolved}: it is inside the read-only P0 tree {protected}"
            )
    return resolved


def guard_write(path: os.PathLike[str] | str) -> Path:
    """Validate then parent-mkdir a write target."""
    p = assert_outside_p0(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# --------------------------------------------------------------------------- #
# Hashing / provenance
# --------------------------------------------------------------------------- #

def sha256_file(path: os.PathLike[str] | str, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_state(repo: Path = REPO_ROOT) -> dict[str, Any]:
    """Best-effort git provenance.  Never raises: a missing repo is just 'unknown'."""
    def run(*args: str) -> str:
        try:
            out = subprocess.run(
                ["git", *args], cwd=repo, capture_output=True, timeout=30
            ).stdout.decode("utf-8", "replace")
            return out.strip()
        except Exception:
            return ""

    return {
        "head": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty_lines": len([l for l in run("status", "--porcelain=v1", "-uall").splitlines() if l]),
    }


def environment_state() -> dict[str, Any]:
    import platform
    state: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": platform.node(),
    }
    try:
        import numpy, pandas, torch  # noqa: PLC0415
        state["numpy"] = numpy.__version__
        state["pandas"] = pandas.__version__
        state["torch"] = torch.__version__
        state["cuda_available"] = bool(torch.cuda.is_available())
        state["cuda_device_count"] = int(torch.cuda.device_count())
        if torch.cuda.is_available():
            state["cuda_device_name"] = torch.cuda.get_device_name(0)
    except Exception as exc:  # pragma: no cover - recorded, never fatal
        state["torch_error"] = repr(exc)
    return state


# --------------------------------------------------------------------------- #
# JSON helpers (ASCII-safe, stable key order)
# --------------------------------------------------------------------------- #

def write_json(path: os.PathLike[str] | str, payload: Any) -> Path:
    p = guard_write(path)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, ensure_ascii=False)
        fh.write("\n")
    return p


def read_json(path: os.PathLike[str] | str) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def make_manifest(
    *,
    stage: str,
    tag: str,
    config: Mapping[str, Any],
    inputs: Mapping[str, os.PathLike[str] | str],
    outputs: Iterable[os.PathLike[str] | str] = (),
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """A provenance record: what ran, against which bytes, producing what."""
    def describe(paths: Iterable[os.PathLike[str] | str]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for p in paths:
            pp = Path(p)
            out[str(pp)] = {
                "exists": pp.exists(),
                "bytes": pp.stat().st_size if pp.exists() else None,
                "sha256": sha256_file(pp) if pp.exists() else None,
            }
        return out

    manifest: dict[str, Any] = {
        "stage": stage,
        "tag": tag,
        "created_utc": utcnow(),
        "config": dict(config),
        "inputs": describe(inputs.values()),
        "outputs": describe(outputs),
        "environment": environment_state(),
        "git": git_state(),
    }
    if extra:
        manifest["extra"] = dict(extra)
    return manifest
