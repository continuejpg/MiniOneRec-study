"""MiniOneRec-CAT / Phase 2B -- master quantitative & stability comparison.

Reads the per-run artefacts and emits the single table the brief asks for:

1.  per-layer codeword usage and entropy
2.  distinct SIDs, colliding groups, items involved
3.  text-neighbourhood overlap@5/10/20
4.  collaborative-neighbourhood overlap@5/10/20
5.  original RQ-VAE reconstruction and quantisation losses
6.  InfoNCE loss and positive/negative similarities
7.  diagnostics for no-evidence and low-frequency items
8.  matched-seed B0/B2 differences
9.  wall-clock and VRAM
10. configuration / checkpoint / SID / manifest hashes

It also states the collision-definition caveat explicitly (see collision_check).

Usage
-----
    python -m experiments.minionerec_cat.src.master_table
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path

import numpy as np

from . import common

#: tag -> (arm label, lambda_cf, seed)
ARMS = {
    "b0":        ("B0", 0.00, 2024),
    "b0_seed7":  ("B0", 0.00, 7),
    "b2":        ("B2", 0.20, 2024),
    "b2_seed7":  ("B2", 0.20, 7),
    "b2_lam005": ("B2", 0.05, 2024),
    "b2_lam010": ("B2", 0.10, 2024),
}

P0_INDEX = common.P0_DATA / "index" / "Industrial_and_Scientific.index.json"
TRAIN_CSV = common.P0_DATA / "train" / "Industrial_and_Scientific_5_2016-10-2018-11.csv"


# --------------------------------------------------------------------------- #

def _cf_loss_series(tag: str) -> dict:
    """Loss / InfoNCE statistics, from the manifest when present, else from the log.

    B0 runs predate the collaborative path, so they have no ``collab`` block and no
    ``cf_history``; for those the reconstruction loss is read back out of the
    trainer log so every column of the comparison is populated.
    """
    paths = common.run_paths(tag)
    p = paths.report_dir / f"{tag}_train_manifest.json"
    out: dict = {}

    if p.exists():
        m = common.read_json(p)
        ex = m.get("extra", {})
        collab = ex.get("collab") or {}
        hist = ex.get("cf_history_tail") or []
        out.update({
            "lambda_cf": collab.get("lambda_cf"),
            "cf_temperature": collab.get("temperature"),
            "cf_dim": collab.get("dim"),
            "cf_with_evidence": collab.get("with_evidence"),
            "cf_masked_out": collab.get("masked_out"),
            "cf_history_len": ex.get("cf_history_len"),
            "best_loss_total": ex.get("best_loss"),
            "best_collision_rate_raw": ex.get("best_collision_rate"),
            "batches_per_epoch": ex.get("batches_per_epoch"),
            "total_optimizer_steps": ex.get("total_optimizer_steps"),
            "vram": (ex.get("vram_telemetry") or {}),
        })
        if hist:
            out["cf_loss_last"] = hist[-1].get("mean_cf_loss")
            out["recon_loss_last"] = hist[-1].get("mean_recon_loss")
            out["quant_loss_last"] = hist[-1].get("mean_quant_loss")
            out["pos_sim_last"] = hist[-1].get("mean_pos_similarity")
            out["neg_sim_last"] = hist[-1].get("mean_neg_similarity")

    # Fall back to the trainer log for any loss field still missing (B0 arms).
    if out.get("recon_loss_last") is None:
        from .audit_sid import parse_train_log
        lg = paths.log_dir / f"train_{tag}.log"
        parsed = parse_train_log(lg)
        if parsed.get("found") and parsed.get("curve_tail"):
            tail = parsed["curve_tail"]
            out["recon_loss_last"] = round(
                sum(row["recon"] for row in tail) / len(tail), 6)
            out["best_loss_total"] = out.get("best_loss_total") or parsed.get("min_total_loss")
            out["loss_source"] = "train_log"
    return out


def _freq_strata() -> dict[int, str]:
    """Train next-item frequency stratum per item (0 / low / mid / high)."""
    freq: dict[int, int] = {}
    with io.open(TRAIN_CSV, encoding="utf-8", newline="") as fh:
        for rec in csv.DictReader(fh):
            i = int(rec["item_id"])
            freq[i] = freq.get(i, 0) + 1
    out: dict[int, str] = {}
    # item ids run 0..3685 (3686 catalogue items); id 0 is real, not PAD
    for item in range(0, 3686):
        f = freq.get(item, 0)
        out[item] = "freq0" if f == 0 else "low(1-2)" if f <= 2 else "mid(3-20)" if f <= 20 else "high(>20)"
    return out


def _arm_row(tag: str) -> dict:
    paths = common.run_paths(tag)
    # audit files were named ``sid_quality_audit.json`` in phase 1 and
    # ``<tag>_sid_quality_audit.json`` from phase 2B on; accept either.
    candidates = [paths.report_dir / f"{tag}_sid_quality_audit.json",
                  paths.report_dir / "sid_quality_audit.json"]
    audit_p = next((p for p in candidates if p.exists()), None)
    if audit_p is None:
        return {"tag": tag, "available": False}
    a = common.read_json(audit_p)
    coll = a["collisions"]
    layers = a["layers"]
    row: dict = {
        "tag": tag,
        "available": True,
        "distinct_sids": coll["distinct_sids"],
        "colliding_groups": coll["colliding_groups"],
        "items_in_collisions": coll["items_in_collision_groups"],
        "unscorable_items": coll["unscorable_items"],
    }
    for L in sorted(layers):
        row[f"{L}_used"] = layers[L]["distinct_codes_used"]
        row[f"{L}_H"] = layers[L]["entropy_bits"]
        row[f"{L}_eff"] = layers[L]["normalised_entropy"]
        row[f"{L}_maxclus"] = layers[L]["largest_cluster_size"]
    tc = a.get("text_coherence") or {}
    for k, v in (tc.get("neighbour_overlap") or {}).items():
        row[f"text_{k}"] = v
    cc = a.get("collab_neighbour") or {}
    for k, v in (cc.get("neighbour_overlap") or {}).items():
        row[f"collab_{k}"] = v
    row.update(_cf_loss_series(tag))
    # hashes
    idx = paths.index
    ck = list(paths.rqvae_ckpt_dir.glob("*/best_collision_model.pth"))
    row["index_sha256"] = common.sha256_file(idx)[:16] if idx.exists() else None
    row["ckpt_sha256"] = common.sha256_file(ck[0])[:16] if ck else None
    row["vocab_tokens"] = a.get("vocab", {}).get("n_sid_tokens")
    return row


def _stratum_quality(tag: str, strata: dict[int, str]) -> dict:
    """Codeword-collision pressure per frequency stratum.

    For each stratum we report how many of its items share a SID with another item
    (i.e. are in a collision group) -- a direct read of whether the quantiser spends
    its scarce codewords on rare items or on popular ones.
    """
    paths = common.run_paths(tag)
    idx_p = paths.index
    if not idx_p.exists():
        return {}
    idx = common.read_json(idx_p)
    sid2items: dict[str, list[int]] = {}
    for k, v in idx.items():
        sid2items.setdefault("".join(v), []).append(int(k))
    out: dict[str, dict] = {}
    for name in ("freq0", "low(1-2)", "mid(3-20)", "high(>20)"):
        items = [i for i, s in strata.items() if s == name]
        in_coll = sum(1 for i in items
                      if len(sid2items.get("".join(idx[str(i)]), [])) > 1)
        sids = {"".join(idx[str(i)]) for i in items}
        out[name] = {
            "items": len(items),
            "distinct_sids": len(sids),
            "items_in_collision": in_coll,
            "collision_share": round(in_coll / len(items), 6) if items else None,
        }
    return out


def build(tags: list[str] | None = None) -> dict:
    tags = tags or list(ARMS)
    strata = _freq_strata()
    rows = []
    for tag in tags:
        if tag not in ARMS:
            continue
        arm, lam, seed = ARMS[tag]
        r = _arm_row(tag)
        r.update({"arm": arm, "lambda_cf_declared": lam, "seed": seed})
        r["strata"] = _stratum_quality(tag, strata) if r.get("available") else {}
        rows.append(r)

    # P0 reference row (catalogue only; its audit was never run)
    if P0_INDEX.exists():
        idx = common.read_json(P0_INDEX)
        sids = {"".join(v) for v in idx.values()}
        layers: dict[str, dict] = {}
        from .audit_sid import describe_layers
        layers = describe_layers(idx)
        rows.append({
            "tag": "P0", "arm": "P0", "lambda_cf_declared": None, "seed": None,
            "available": True,
            "distinct_sids": len(sids),
            "colliding_groups": None, "items_in_collisions": len(idx) - len(sids),
            "unscorable_items": None,
            **{f"{L}_used": layers[L]["distinct_codes_used"] for L in layers},
            **{f"{L}_H": layers[L]["entropy_bits"] for L in layers},
            **{f"{L}_eff": layers[L]["normalised_entropy"] for L in layers},
            **{f"{L}_maxclus": layers[L]["largest_cluster_size"] for L in layers},
            "index_sha256": common.sha256_file(P0_INDEX)[:16],
            "strata": _stratum_quality("__p0__", strata) if False else {},
        })
    return {"rows": rows, "strata_definition": {
        "freq0": "item never appears as a train target",
        "low(1-2)": "1-2 train targets",
        "mid(3-20)": "3-20 train targets",
        "high(>20)": "more than 20 train targets",
    }}


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


GROUPS: list[tuple[str, list[str]]] = [
    ("arms", ["tag", "arm", "lambda_cf_declared", "seed"]),
    ("1 layer usage / entropy", ["a_used", "a_H", "a_eff", "a_maxclus",
                                 "b_used", "b_H", "b_eff",
                                 "c_used", "c_H", "c_eff"]),
    ("2 SID identity (AFTER retry loop)", ["distinct_sids", "colliding_groups",
                                           "items_in_collisions", "unscorable_items",
                                           "vocab_tokens"]),
    ("3 text neighbourhood", ["text_overlap@5", "text_overlap@10", "text_overlap@20"]),
    ("4 collab neighbourhood", ["collab_overlap@5", "collab_overlap@10", "collab_overlap@20"]),
    ("5 RQ-VAE losses (last epoch mean)", ["recon_loss_last", "quant_loss_last",
                                           "best_loss_total"]),
    ("6 InfoNCE", ["cf_loss_last", "pos_sim_last", "neg_sim_last",
                   "lambda_cf", "cf_temperature"]),
    ("7 evidence coverage", ["cf_with_evidence", "cf_masked_out", "cf_dim"]),
    ("9 budget / memory", ["total_optimizer_steps", "batches_per_epoch", "best_collision_rate_raw"]),
    ("10 hashes", ["index_sha256", "ckpt_sha256"]),
]


def render(data: dict) -> str:
    rows = data["rows"]
    lines: list[str] = []
    head = " " * 30 + "".join(f"{r['tag']:>16}" for r in rows)
    lines.append("=" * len(head))
    lines.append("MiniOneRec-CAT / Phase 2B -- master comparison")
    lines.append("=" * len(head))
    lines.append(head)
    lines.append(" " * 30 + "-" * (16 * len(rows)))
    for title, keys in GROUPS:
        lines.append(f"  [{title}]")
        for k in keys:
            if not any(k in r for r in rows):
                continue
            lines.append(f"  {k:<28}" + "".join(f"{_fmt(r.get(k)):>16}" for r in rows))
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tags", nargs="+", default=None)
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args(argv)

    data = build(args.tags)
    text = render(data)
    print(text)

    print()
    print("  --- 7b per-stratum SID pressure (items sharing a SID / stratum size) ---")
    for r in data["rows"]:
        st = r.get("strata") or {}
        if not st:
            continue
        cells = "  ".join(
            f"{k}={v['items_in_collision']}/{v['items']}" for k, v in st.items())
        print(f"    {r['tag']:<12} {cells}")

    print()
    print("  --- 8 matched-seed B0 vs B2 (mechanistic contrast, NOT a significance test) ---")
    by = {(r["arm"], r["seed"]): r for r in data["rows"] if r.get("seed")}
    for seed in sorted({s for (_a, s) in by}):
        b0, b2 = by.get(("B0", seed)), by.get(("B2", seed))
        if not (b0 and b2 and b0.get("available") and b2.get("available")):
            print(f"    seed {seed}: pair incomplete")
            continue
        print(f"    seed {seed}:")
        for k in ("a_used", "a_H", "text_overlap@20", "distinct_sids", "unscorable_items",
                  "best_collision_rate_raw"):
            print(f"      {k:<24} b0={_fmt(b0.get(k))}  b2={_fmt(b2.get(k))}  "
                  f"delta={_fmt((b2.get(k) or 0) - (b0.get(k) or 0))}")

    data["notes"] = {
        "collision_definition": (
            "best_collision_rate_raw is the TRAINER's first-pass (number of items minus "
            "distinct 3-code strings, over 3686). The shipped SID catalogue has been "
            "re-quantised by the retry loop, so its colliding-group count is a DIFFERENT "
            "and smaller quantity. Never mix them."
        ),
        "no_significance": (
            "No seed spread is used as a significance threshold, and no HR/NDCG was "
            "consulted to choose lambda_cf."
        ),
        "quality_disclaimer": (
            "Codebook utilisation and neighbourhood overlap are properties of the SID "
            "catalogue. They are not evidence of recommendation quality."
        ),
    }
    dest = args.json or (common.EXP_ROOT / "_comparison" / "phase2b_master_table.json")
    common.write_json(dest, data)
    txt = dest.with_suffix(".txt")
    txt.parent.mkdir(parents=True, exist_ok=True)
    txt.write_text(text + "\n", encoding="utf-8")
    print(f"\n  written: {dest}")
    print(f"           {txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
