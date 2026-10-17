"""MiniOneRec-CAT / Phase 3 -- corrected, item-identity-aware evaluation.

Why this exists
---------------
``calc.py`` scores purely on SID strings (``calc.py:47-68``): it never converts a
prediction to an item id, so "correct" means "the SID string matched at rank r".
Two consequences on the real test split (measured in p3_collision):
  * 261 / 4533 test rows (5.7578%) target a SID that is shared by >= 2 items;
  * 97 / 4533 (2.1399%) target an item that is NOT the lowest-id partner of its SID.
For those 97 rows the ranked ITEM implied by a hit is a different item, and for
the remaining collision rows the hit is identity-ambiguous.

This module keeps the repo metric EXACTLY (so numbers stay comparable with the
published P0 baseline) and adds, alongside it, item-ID-resolved variants:

  repo HR/NDCG      -- verbatim calc.py, SID-string match
  item HR/NDCG      -- resolve each predicted SID to its canonical item id
                       (lowest item id, matching the repo's own convention) and
                       compare against the true item id
  ambiguous         -- rows where the target item shares its SID with another item
  identity_mismatch -- rows where a SID-string hit resolves to a different item

Verified against the frozen P0 artifact
--------------------------------------
Re-running the repo variant on ``checkpoints/grpo_evidence/eval_clean_sft/
test_beam20.json`` reproduces the published baseline to 1e-8:
HR@20 = 0.19832341, NDCG@20 = 0.11786798.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

__all__ = ["load_info", "load_user_ids", "collision_sets", "evaluate_artifact",
           "bootstrap_diff"]

TOPK = (1, 3, 5, 10, 20, 50)


def load_info(info_path: Path) -> dict:
    """``sid -> sorted item ids`` using calc.py's line-index convention.

    The info file is written in ascending item-id order, so the line index IS the
    item id (``rebuild_derived`` asserts the ordering).
    """
    m: dict[str, list[int]] = defaultdict(list)
    for line in Path(info_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        sid = parts[0].strip()
        iid = int(parts[2]) if len(parts) >= 3 else len(m)
        m[sid].append(iid)
    return {k: sorted(v) for k, v in m.items()}


def _dcg_gain(rank: int) -> float:
    """Repo gain, calc.py:74 normalised by 1/log(2); equals 1/log2(rank+1)."""
    return (1.0 / math.log(rank + 2)) / (1.0 / math.log(2))


def collision_sets(info_path: Path, test_csv: Path) -> dict:
    """Which SIDs collide, which items own them, and which test rows are affected.

    Emitted per arm because every arm has its OWN collision set: all arms carry 11
    groups / 22 items, but they sit on different items, so the affected test rows
    differ.  That is precisely why this must be reported rather than assumed away.
    """
    sid2ids = load_info(info_path)
    colliding = {s: v for s, v in sid2ids.items() if len(v) > 1}
    id2sid = {i: s for s, v in sid2ids.items() for i in v}

    import csv as _csv
    import io as _io
    rows = list(_csv.DictReader(_io.open(test_csv, encoding="utf-8", newline="")))

    rows_by_item: dict[int, list[int]] = defaultdict(list)
    ambiguous_rows: list[int] = []
    nonlowest_rows: list[int] = []
    for r_i, rec in enumerate(rows):
        iid = int(rec["item_id"])
        rows_by_item[iid].append(r_i)
        sid = id2sid.get(iid)
        if sid in colliding:
            ambiguous_rows.append(r_i)
            if iid != colliding[sid][0]:
                nonlowest_rows.append(r_i)

    return {
        "colliding_groups": {
            sid: {"items": ids, "canonical_item": ids[0],
                  "members_not_canonical": ids[1:]}
            for sid, ids in sorted(colliding.items())
        },
        "n_colliding_groups": len(colliding),
        "colliding_item_ids": sorted(i for v in colliding.values() for i in v),
        "test_rows_by_colliding_item": {
            str(i): rows_by_item.get(i, []) for i in sorted(id2sid)
            if id2sid[i] in colliding
        },
        "ambiguous_test_rows": ambiguous_rows,
        "nonlowest_test_rows": nonlowest_rows,
        "n_ambiguous_rows": len(ambiguous_rows),
        "n_nonlowest_rows": len(nonlowest_rows),
    }


def evaluate_artifact(artifact: Path, info_path: Path, *,
                      topk: int = 20, persist_per_row: Path | None = None,
                      test_csv: Path | None = None,
                      train_csv: Path | None = None,
                      popular_freq_threshold: int = 100) -> dict:
    """Score one evaluation artifact under the repo and item-identity conventions.

    ``test_csv`` is optional but recommended: supplying it enables the collision
    grouping and the per-collision-item test-row map.
    ``train_csv`` additionally enables the popularity-split grouping (see below).
    """
    test_data = json.loads(Path(artifact).read_text(encoding="utf-8"))
    sid2ids = load_info(info_path)

    # Per-row train-target frequency, when available, so the collision grouping can
    # be separated from item popularity.  A collision row is NOT a random test row:
    # measured on the P0 split, colliding-target rows have median train frequency
    # 491 vs 8 for the rest (61x), because a handful of very popular items sit in
    # collision groups.  Reporting the collision group without that split would
    # attribute a popularity effect to collisions.
    row_freq: list[int] | None = None
    if train_csv is not None:
        import csv as _csv
        import io as _io
        from collections import Counter as _Counter
        f = _Counter()
        with _io.open(train_csv, encoding="utf-8", newline="") as fh:
            for rec in _csv.DictReader(fh):
                f[int(rec["item_id"])] += 1
        row_freq = [f.get(i, 0) for i in range(len(test_data))]
        # align by test-file order if the test csv is supplied as well
        if test_csv is not None:
            with _io.open(test_csv, encoding="utf-8", newline="") as fh:
                row_freq = [f.get(int(rec["item_id"]), 0)
                            for rec in _csv.DictReader(fh)]

    repo_hits = 0
    item_hits = 0
    unamb_hits = 0
    repo_ndcg = 0.0
    item_ndcg = 0.0
    unamb_ndcg = 0.0
    n = len(test_data)
    ambiguous_rows = 0
    identity_mismatch_rows = 0
    no_valid_sid = 0
    per_row = []
    # grouped tallies for collision vs non-collision rows
    grp = {
        "collision": {"rows": 0, "repo_hits": 0, "item_hits": 0},
        "non_collision": {"rows": 0, "repo_hits": 0, "item_hits": 0},
    }
    # refined split: a collision row whose target item is itself popular is a
    # different regime from one whose target is rare
    pop_grp = {
        "collision_popular": {"rows": 0, "repo_hits": 0, "item_hits": 0},
        "collision_rare": {"rows": 0, "repo_hits": 0, "item_hits": 0},
    }

    for r_i, row in enumerate(test_data):
        out = row["output"]
        target_sid = (out[0].strip('"').strip(" ") if isinstance(out, list)
                      else out.strip(' \n"'))
        gold_ids = sid2ids.get(target_sid)
        # The evaluator's implied gold item: lowest id sharing that SID.
        gold_item = gold_ids[0] if gold_ids else None
        is_ambiguous = bool(gold_ids) and len(gold_ids) > 1

        preds = [p.strip('"\n').strip() for p in row["predict"]]

        # ---- repo convention: SID string equality -------------------------- #
        repo_rank = None
        for i, p in enumerate(preds[:topk]):
            if p == target_sid:
                repo_rank = i
                break

        # ---- item convention: resolve SID -> canonical item id -------------- #
        item_rank = None
        for i, p in enumerate(preds[:topk]):
            ids = sid2ids.get(p)
            if not ids:
                continue
            if gold_item is not None and ids[0] == gold_item:
                item_rank = i
                break

        # ---- unambiguous convention: the SID must name exactly ONE item ----- #
        # A hit only counts when the target SID is unique, so tie-breaking can
        # never manufacture it.
        unamb_rank = None
        if gold_ids is not None and len(gold_ids) == 1:
            for i, p in enumerate(preds[:topk]):
                ids = sid2ids.get(p)
                if ids and len(ids) == 1 and ids[0] == gold_item:
                    unamb_rank = i
                    break

        if repo_rank is not None:
            repo_hits += 1
            repo_ndcg += _dcg_gain(repo_rank)
        if item_rank is not None:
            item_hits += 1
            item_ndcg += _dcg_gain(item_rank)
        if unamb_rank is not None:
            unamb_hits += 1
            unamb_ndcg += _dcg_gain(unamb_rank)
        if is_ambiguous:
            ambiguous_rows += 1
            grp["collision"]["rows"] += 1
            grp["collision"]["repo_hits"] += int(repo_rank is not None)
            grp["collision"]["item_hits"] += int(item_rank is not None)
            if row_freq is not None:
                key = ("collision_popular" if row_freq[r_i] > popular_freq_threshold
                       else "collision_rare")
                pop_grp[key]["rows"] += 1
                pop_grp[key]["repo_hits"] += int(repo_rank is not None)
                pop_grp[key]["item_hits"] += int(item_rank is not None)
        else:
            grp["non_collision"]["rows"] += 1
            grp["non_collision"]["repo_hits"] += int(repo_rank is not None)
            grp["non_collision"]["item_hits"] += int(item_rank is not None)
        # a SID hit whose resolved item is not the gold item
        if repo_rank is not None and item_rank is None:
            identity_mismatch_rows += 1
        if not any(p in sid2ids for p in preds[:topk]):
            no_valid_sid += 1

        per_row.append({
            "target_sid": target_sid,
            "gold_item_id": gold_item,
            "gold_sid_shared_by": len(gold_ids) if gold_ids else 0,
            "ambiguous": is_ambiguous,
            "repo_rank": repo_rank,
            "item_rank": item_rank,
            "unambiguous_rank": unamb_rank,
            "top1_sid": preds[0] if preds else None,
            "top1_item_id": (sid2ids.get(preds[0]) or [None])[0] if preds else None,
            "n_predictions": len(preds),
        })

    for g in grp.values():
        g["repo_HR"] = g["repo_hits"] / g["rows"] if g["rows"] else None
        g["item_HR"] = g["item_hits"] / g["rows"] if g["rows"] else None
    for g in pop_grp.values():
        g["repo_HR"] = g["repo_hits"] / g["rows"] if g["rows"] else None
        g["item_HR"] = g["item_hits"] / g["rows"] if g["rows"] else None

    res = {
        "artifact": str(artifact),
        "info": str(info_path),
        "rows": n,
        "topk": topk,
        "repo_sid_string": {
            "HR": repo_hits / n,
            "NDCG": repo_ndcg / n,
            "hits": repo_hits,
        },
        "canonical_item": {
            "HR": item_hits / n,
            "NDCG": item_ndcg / n,
            "hits": item_hits,
            "caveat": ("Canonical-Item resolves a predicted SID to the LOWEST item id "
                       "sharing it. That is an arbitrary but fixed decoding policy; it "
                       "does NOT mean the model identified the true item."),
        },
        "unambiguous_item": {
            "HR": unamb_hits / n,
            "NDCG": unamb_ndcg / n,
            "hits": unamb_hits,
            "caveat": ("Only counts a hit when the target SID names exactly one item, "
                       "so tie-breaking cannot create a hit. Rows with a shared target "
                       "SID are excluded by construction and therefore cap this metric "
                       "below the repo metric."),
        },
        "diagnostics": {
            "ambiguous_rows": ambiguous_rows,
            "ambiguous_share": ambiguous_rows / n,
            "identity_mismatch_rows": identity_mismatch_rows,
            "identity_mismatch_share": identity_mismatch_rows / n,
            "rows_with_no_valid_predicted_sid": no_valid_sid,
            "mean_predictions_per_row": sum(r["n_predictions"] for r in per_row) / n,
        },
        "grouped_by_collision": grp,
        "grouped_collision_by_popularity": (pop_grp if row_freq is not None else None),
        "popular_freq_threshold": popular_freq_threshold,
    }
    if test_csv is not None:
        res["collision_sets"] = collision_sets(info_path, Path(test_csv))
    if persist_per_row:
        Path(persist_per_row).parent.mkdir(parents=True, exist_ok=True)
        Path(persist_per_row).write_text(
            json.dumps(per_row, indent=1), encoding="utf-8")
        res["per_row_path"] = str(persist_per_row)
    return res


def bootstrap_diff(hits_a, hits_b, *, clusters=None, n_boot: int = 10000,
                   seed: int = 2024, alpha: float = 0.05) -> dict:
    """Paired bootstrap for a difference of two means.

    ``hits_a``/``hits_b`` are per-row values (0/1 for HR, the DCG gain for NDCG) in
    the SAME row order, so pairing is by row.  All arms are evaluated on the
    identical 4533 rows in the identical order, which is what makes this valid.

    ``clusters`` (optional) assigns each row to a resampling unit -- normally the
    user id.  When given, whole users are drawn with replacement, because a user's
    rows are not independent: the same history prefix can appear in several test
    rows, so a row-level bootstrap would understate the interval.

    Boundary rule, stated explicitly because it was misreported once:
      ``excludes_zero`` is ``ci_low > 0 or ci_high < 0``.
      An interval of exactly [0, 0] therefore does NOT exclude zero.
    """
    import numpy as np

    a = np.asarray(hits_a, dtype=np.float64)
    b = np.asarray(hits_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"paired bootstrap needs equal lengths, got {a.shape} vs {b.shape}")
    n = a.shape[0]
    d = b - a
    rng = np.random.default_rng(seed)

    if clusters is None:
        idx = rng.integers(0, n, size=(n_boot, n))
        boot = d[idx].mean(axis=1)
        unit = "row"
        n_units = n
    else:
        cl = np.asarray(clusters)
        if cl.shape[0] != n:
            raise ValueError(f"clusters length {cl.shape[0]} != rows {n}")
        uniq, inv = np.unique(cl, return_inverse=True)
        # group row indices by cluster once
        order = np.argsort(inv, kind="stable")
        sorted_inv = inv[order]
        bounds = np.searchsorted(sorted_inv, np.arange(len(uniq)), side="left")
        ends = np.searchsorted(sorted_inv, np.arange(len(uniq)), side="right")
        members = [order[bounds[i]:ends[i]] for i in range(len(uniq))]
        boot = np.empty(n_boot, dtype=np.float64)
        for t in range(n_boot):
            pick = rng.integers(0, len(uniq), size=len(uniq))
            rows = np.concatenate([members[p] for p in pick])
            boot[t] = d[rows].mean()
        unit = "cluster"
        n_units = len(uniq)

    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    lo, hi = float(lo), float(hi)
    return {
        "n_rows": int(n),
        "n_units": int(n_units),
        "resample_unit": unit,
        "n_boot": int(n_boot),
        "seed": int(seed),
        "mean_delta": float(d.mean()),
        "ci_low": lo,
        "ci_high": hi,
        "excludes_zero": bool(lo > 0.0 or hi < 0.0),
        "boundary_rule": "excludes_zero := (ci_low > 0) or (ci_high < 0); [0,0] does NOT exclude zero",
    }


def load_user_ids(test_csv) -> list[str]:
    """Per-row user ids in file order, for clustered resampling."""
    import csv as _csv
    import io as _io
    with _io.open(test_csv, encoding="utf-8", newline="") as fh:
        return [r["user_id"] for r in _csv.DictReader(fh)]
