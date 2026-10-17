"""MiniOneRec-CAT / B2 -- build the frozen collaborative item representation.

What this produces
------------------
One row per catalogue item, in **item-id order**, of a low-dimensional vector that
summarises what each item co-occurs with in the TRAIN split.  This is the frozen
target the encoder is aligned to (LETTER's ``cf_embedding``, ``RQ-VAE/main.py:68``).

How it differs from LETTER
--------------------------
LETTER loads a **pre-trained SASRec item embedding** (``--cf_emb
./RQ-VAE/ckpt/<cat>-32d-sasrec.pt``).  We do not train a second model: the
technical decision is to derive the collaborative view from TRAIN co-occurrence
with IDF re-weighting and a truncated SVD, so that B2 adds **no extra training
budget** relative to B0.

Data hygiene
------------
* **TRAIN ONLY.**  The loader reads exactly one CSV and asserts that its path is
  the train split; Valid/Test are never opened.
* The train CSV is a **sliding-window** sample (``st = max(i-10, 0)``,
  ``data/amazon18_data_process.py:259-261``), so the same user interaction
  appears in many consecutive windows.  We therefore dedupe the item *set* per
  row rather than counting row-level pairs; the assumption is stated in
  :func:`build` and in the manifest rather than hidden.

Usage
-----
    python -m experiments.minionerec_cat.src.build_collab_pairs --tag b2
    python -m experiments.minionerec_cat.src.build_collab_pairs --tag b2 --dim 32 --min-degree 3
"""

from __future__ import annotations

import argparse
import csv
import io
import math
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from . import common

#: The only split this module is allowed to read.
TRAIN_SPLIT = "train"


def _assert_train_only(path: Path) -> None:
    """Fail closed if the caller hands us anything but the train split."""
    if TRAIN_SPLIT not in str(path).lower().replace("\\", "/").split("/"):
        raise ValueError(
            f"refusing to build collaborative statistics from {path}: only the "
            f"{TRAIN_SPLIT!r} split may be used"
        )
    if f"/{TRAIN_SPLIT}/" not in str(path).replace("\\", "/"):
        raise ValueError(f"{path} does not look like a {TRAIN_SPLIT} split path")


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #

def collect_item_sets(train_csv: Path, n_items: int) -> tuple[list[set[int]], Counter]:
    """Per-row item sets from the train windows.

    A row contributes ``history_item_id`` plus its target ``item_id``.  Duplicates
    inside one row collapse (the sliding window repeats items across *rows*, not
    within one), which is the documented approximation: we recover window-level
    co-occurrence, **not** the user's full interaction sequence.
    """
    _assert_train_only(train_csv)
    rows: list[set[int]] = []
    degree = Counter()
    with io.open(train_csv, encoding="utf-8", newline="") as fh:
        for rec in csv.DictReader(fh):
            items: set[int] = set()
            raw = rec.get("history_item_id") or ""
            if raw:
                try:
                    items.update(int(x) for x in eval(raw))
                except Exception as exc:  # noqa: BLE001
                    raise ValueError(f"unparsable history_item_id {raw!r}: {exc}") from exc
            items.add(int(rec["item_id"]))
            items = {i for i in items if 1 <= i <= n_items}
            if items:
                rows.append(items)
                for i in items:
                    degree[i] += 1
    return rows, degree


def build_incidence(rows: list[set[int]], n_items: int, *, idf: bool = True) -> sp.csr_matrix:
    """Sparse ``[n_rows, n_items]`` incidence with optional IDF column weighting.

    IDF is the popularity correction: without it a row of blockbuster items would
    dominate every item's vector, and the SVD would mostly recover popularity.
    Weight is ``log((1 + R) / (1 + df)) + 1`` (the standard smooth form).
    """
    n_rows = len(rows)
    indptr = [0]
    indices: list[int] = []
    for s in rows:
        indices.extend(sorted(s))
        indptr.append(len(indices))
    data = np.ones(len(indices), dtype=np.float32)
    m = sp.csr_matrix((data, indices, indptr), shape=(n_rows, n_items + 1))

    if idf and n_rows:
        df = np.asarray((m != 0).sum(axis=0)).ravel().astype(np.float64)
        w = np.log((1.0 + n_rows) / (1.0 + df)) + 1.0
        w[0] = 0.0                       # item id 0 is the PAD slot; never weighted
        m = m @ sp.diags(w.astype(np.float32))
    # L2-normalise each row so long windows do not dominate the second moment.
    norms = np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    m = sp.diags((1.0 / norms).astype(np.float32)) @ m
    return m.tocsr()


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #

def build(
    tag: str,
    *,
    dim: int = 32,
    min_degree: int = 3,
    idf: bool = True,
    seed: int = 0,
    train_csv: Path | None = None,
) -> dict:
    """Build ``cf_embedding[item_id] -> float32[dim]`` and report its coverage."""
    from sklearn.decomposition import TruncatedSVD

    paths = common.run_paths(tag)
    paths.ensure_dirs()
    src = Path(train_csv) if train_csv else paths.p0_split(TRAIN_SPLIT)
    _assert_train_only(src)

    n_items = 3686
    print("=" * 88)
    print(f"MiniOneRec-CAT / {tag}  -- collaborative representation")
    print("=" * 88)
    print(f"  source split : {src}   (TRAIN ONLY)")

    rows, degree = collect_item_sets(src, n_items)
    print(f"  windows      : {len(rows):,}")
    print(f"  items with behavioural evidence : {len(degree):,} / {n_items}")

    inc = build_incidence(rows, n_items, idf=idf)
    print(f"  incidence    : {inc.shape}  nnz={inc.nnz:,}  idf={idf}")

    k = min(dim, min(inc.shape) - 1)
    svd = TruncatedSVD(n_components=k, random_state=seed, algorithm="randomized", n_iter=7)
    emb = svd.fit_transform(inc)                       # [n_rows, k]
    # Item-space factor: co-occurrence profile of each item, i.e. item x window^T x window
    item_space = np.asarray(inc.T @ emb)               # [n_items+1, k]
    explained = float(svd.explained_variance_ratio_.sum())

    # Re-normalise the item vectors (the row normalisation above was per window).
    norms = np.linalg.norm(item_space, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    item_space = item_space / norms

    # ---- coverage masks ---------------------------------------------------- #
    has_evidence = np.zeros(n_items + 1, dtype=bool)
    for item, deg in degree.items():
        if deg >= min_degree:
            has_evidence[item] = True
    no_evidence = sorted(set(range(1, n_items + 1)) - {i for i in degree})
    low_degree = sorted(i for i, d in degree.items() if d < min_degree)

    # Any item without evidence gets the exact zero vector, and is masked out of
    # the loss so a zero row can never be normalised (the NaN/Inf trap).
    item_space[~has_evidence] = 0.0

    vectors = item_space[1:].astype(np.float32)        # drop the PAD row

    out_dir = paths.root / "collab"
    out_dir.mkdir(parents=True, exist_ok=True)
    npy = common.guard_write(out_dir / f"cf_embedding_{tag}.npy")
    np.save(npy, vectors)
    mask_path = common.guard_write(out_dir / f"cf_mask_{tag}.npy")
    np.save(mask_path, has_evidence[1:])

    deg_hist = Counter(degree.values())
    stats = {
        "tag": tag,
        "source_split": str(src),
        "source_split_name": TRAIN_SPLIT,
        "windows": len(rows),
        "n_items": n_items,
        "dim_requested": dim,
        "dim_actual": int(k),
        "idf_weighted": idf,
        "svd_explained_variance_ratio": round(explained, 6),
        "min_degree_for_evidence": min_degree,
        "items_with_evidence": int(has_evidence.sum()),
        "items_masked_out": int((~has_evidence).sum()),
        "items_never_seen_in_train": len(no_evidence),
        "items_never_seen_examples": no_evidence[:20],
        "items_below_min_degree": len(low_degree),
        "degree_histogram_top": dict(sorted(deg_hist.items())[:25]),
        "degree_stats": {
            "min": int(min(degree.values())),
            "max": int(max(degree.values())),
            "mean": round(sum(degree.values()) / len(degree), 4),
            "median": int(sorted(degree.values())[len(degree) // 2]),
        },
        "vector_norms": {
            "min": round(float(np.linalg.norm(vectors, axis=1).min()), 6),
            "max": round(float(np.linalg.norm(vectors, axis=1).max()), 6),
            "zero_rows": int((np.linalg.norm(vectors, axis=1) == 0).sum()),
        },
        "artifacts": {
            "cf_embedding": {"path": str(npy), "sha256": common.sha256_file(npy),
                             "shape": list(vectors.shape), "dtype": str(vectors.dtype)},
            "cf_mask": {"path": str(mask_path), "sha256": common.sha256_file(mask_path),
                        "shape": list(has_evidence[1:].shape)},
        },
        "assumption": (
            "The train CSV is a sliding-window sample, so co-occurrence is recovered at "
            "WINDOW level, not as the user's full interaction sequence. Within one row, "
            "duplicate items collapse to a set."
        ),
    }
    print(f"  dim          : {k} (explained var {explained:.4f})")
    print(f"  evidence     : {stats['items_with_evidence']:,} items kept, "
          f"{stats['items_masked_out']} masked out")
    print(f"                 never seen in train: {stats['items_never_seen_in_train']}, "
          f"degree < {min_degree}: {stats['items_below_min_degree']}")
    print(f"  cf_embedding : {npy}  sha256={stats['artifacts']['cf_embedding']['sha256'][:16]}")
    print(f"  cf_mask      : {mask_path}")

    manifest = common.make_manifest(
        stage="build_collab_pairs",
        tag=tag,
        config={"dim": dim, "min_degree": min_degree, "idf": idf, "seed": seed},
        inputs={"train_csv": src},
        outputs=[npy, mask_path],
        extra={"collab_stats": stats},
    )
    common.write_json(paths.report_dir / "b2_collab_manifest.json", manifest)
    common.write_json(paths.report_dir / "b2_collab_stats.json", stats)
    return stats


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tag", default="b2")
    p.add_argument("--dim", type=int, default=32, help="must match the RQ-VAE e_dim for cosine alignment")
    p.add_argument("--min-degree", type=int, default=3,
                   help="windows an item must appear in before it counts as having collaborative evidence")
    p.add_argument("--no-idf", action="store_true", help="disable popularity re-weighting")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-csv", type=Path, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    build(args.tag, dim=args.dim, min_degree=args.min_degree, idf=not args.no_idf,
          seed=args.seed, train_csv=args.train_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
