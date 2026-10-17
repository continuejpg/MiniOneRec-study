"""MiniOneRec-CAT / compare_seeds -- side-by-side SID quality for several runs.

Reads the ``sid_quality_audit.json`` produced by :mod:`audit_sid` for each tag and
emits one comparison table, so the seed-to-seed spread is visible next to the
baseline rather than buried in per-run reports.

The headline number to look at is **the spread between our own seeds**, not the
difference from P0.  Two runs of the *same* recipe differing by more than the
B0-vs-P0 gap would mean the gap is not attributable to the recipe.

Usage
-----
    python -m experiments.minionerec_cat.src.compare_seeds --tags b0 b0_seed7 --baseline p0
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import common


def _row(name: str, audit: dict) -> dict:
    layers = audit.get("layers", {})
    coll = audit.get("collisions", {})
    tc = audit.get("text_coherence", {}) or {}
    rows = {
        "run": name,
        "index_sha256": (audit.get("index_sha256") or "")[:16],
        "distinct_sids": coll.get("distinct_sids"),
        "colliding_groups": coll.get("colliding_groups"),
        "items_in_collisions": coll.get("items_in_collision_groups"),
        "unscorable_items": coll.get("unscorable_items"),
        "levels": sorted((audit.get("sid_lengths", {}).get("length_histogram") or {}).keys()),
    }
    for letter in sorted(layers):
        s = layers[letter]
        rows[f"{letter}_used"] = f"{s['distinct_codes_used']}/{s['codebook_size']}"
        rows[f"{letter}_H"] = s["entropy_bits"]
        rows[f"{letter}_eff"] = s["normalised_entropy"]
    for k, v in sorted((tc.get("neighbour_overlap") or {}).items()):
        rows[k] = v
    tr = audit.get("train_manifest") or {}
    rows["best_loss"] = tr.get("best_loss")
    rows["best_collision_rate"] = tr.get("best_collision_rate")
    return rows


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def compare(tags: list[str], *, baseline: str | None = "p0") -> dict:
    paths = common.run_paths(tags[0])
    audits: dict[str, dict] = {}

    for tag in tags:
        p = common.run_paths(tag).report_dir / "sid_quality_audit.json"
        if not p.exists():
            print(f"  WARN: no audit for tag {tag!r} at {p}; skipping")
            continue
        audits[tag] = common.read_json(p)

    if baseline == "p0":
        # Synthesise a P0 pseudo-audit from the shared catalogue so the table has
        # a reference column without re-running the whole audit.
        from .audit_sid import describe_collisions, describe_layers, describe_sid_lengths, load_index
        idx = load_index(paths.p0_index)
        audits["P0"] = {
            "index_sha256": common.sha256_file(paths.p0_index),
            "layers": describe_layers(idx),
            "collisions": describe_collisions(idx),
            "sid_lengths": describe_sid_lengths(idx),
            "text_coherence": {"neighbour_overlap": None},
            "train_manifest": {},
        }

    rows = [_row(name, a) for name, a in audits.items()]
    keys = ["run", "index_sha256", "distinct_sids", "colliding_groups", "items_in_collisions",
            "unscorable_items", "levels", "best_loss", "best_collision_rate"]
    keys += [k for k in rows[0] if k not in keys and not k.startswith("overlap")]
    keys += sorted(k for k in rows[0] if k.startswith("overlap"))

    width = max(len(k) for k in keys) + 2
    header = " " * width + "".join(f"{r['run']:>16}" for r in rows)
    lines = [header, " " * width + "-" * (16 * len(rows))]
    for k in keys:
        if k == "run":
            continue
        lines.append(f"{k:<{width}}" + "".join(f"{_fmt(r.get(k)):>16}" for r in rows))

    print("=" * len(header))
    print("MiniOneRec-CAT / SID quality comparison")
    print("=" * len(header))
    for l in lines:
        print("  " + l)

    # Seed spread must be computed WITHIN one arm.  Pooling b0 and b2 into a
    # single "spread" would silently report the arm effect as if it were
    # seed noise -- exactly the confusion the brief forbids.
    def _arm_of(run_name: str) -> str:
        return run_name.split("_seed")[0]

    by_arm: dict[str, list[dict]] = {}
    for r in rows:
        if r["run"] == "P0":
            continue
        by_arm.setdefault(_arm_of(r["run"]), []).append(r)

    summary: dict = {"rows": rows, "table": "\n".join(lines), "arms": {}}
    numeric = ["distinct_sids", "colliding_groups", "unscorable_items", "best_loss",
               "best_collision_rate", "a_used", "a_H", "overlap@10"]

    def _spread(group: list[dict]) -> dict:
        out: dict = {}
        for k in numeric:
            vals = []
            for r in group:
                v = r.get(k)
                if isinstance(v, str) and "/" in v:
                    v = int(v.split("/")[0])
                if isinstance(v, (int, float)):
                    vals.append(float(v))
            if len(vals) >= 2:
                out[k] = {"min": min(vals), "max": max(vals), "range": max(vals) - min(vals)}
        return out

    for arm, group in sorted(by_arm.items()):
        summary["arms"][arm] = {
            "runs": [r["run"] for r in group],
            "n_runs": len(group),
            "seed_spread": _spread(group) if len(group) >= 2 else None,
        }

    print()
    for arm, info in sorted(summary["arms"].items()):
        print(f"  --- arm {arm}: {info['n_runs']} run(s) {info['runs']} ---")
        if info["seed_spread"]:
            print("      seed spread (this arm only):")
            for k, v in info["seed_spread"].items():
                print(f"        {k:<22} min={_fmt(v['min'])}  max={_fmt(v['max'])}  "
                      f"range={_fmt(v['range'])}")
        else:
            print("      single run -> no seed spread available (cannot separate "
                  "arm effect from seed noise)")

    # Cross-arm contrast, reported as a contrast and NOT as a significance test.
    if "b0" in by_arm and "b2" in by_arm:
        b0 = by_arm["b0"][0]
        b2 = by_arm["b2"][0]
        contrast = {}
        for k in numeric:
            a, b = b0.get(k), b2.get(k)
            if isinstance(a, str) and "/" in a:
                a = int(a.split("/")[0])
            if isinstance(b, str) and "/" in b:
                b = int(b.split("/")[0])
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                contrast[k] = {"b0": a, "b2": b, "delta": b - a}
        summary["b2_vs_b0_contrast"] = contrast
        print()
        print("  --- B2 vs B0 (same seed 2024): mechanistic contrast, NOT a significance test ---")
        for k, v in contrast.items():
            print(f"    {k:<22} b0={_fmt(v['b0'])}  b2={_fmt(v['b2'])}  delta={_fmt(v['delta'])}")

    seed_spread_note = (
        "Seed spread is computed per arm. With a single B2 run the B2-vs-B0 "
        "contrast CANNOT be separated from seed noise, and no significance claim "
        "is made from it."
    )
    summary["seed_spread_note"] = seed_spread_note
    print()
    print("  " + seed_spread_note)

    summary["disclaimer"] = (
        "B0 and P0 are NOT a same-configuration comparison: P0's quantiser checkpoint was "
        "never shipped, so its training budget, seed and exact hyper-parameters are unknown. "
        "These are catalogue-level properties only; no HR/NDCG claim is made."
    )
    print("  NOTE: " + summary["disclaimer"])
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tags", nargs="+", default=["b0", "b0_seed7"])
    p.add_argument("--baseline", default="p0")
    p.add_argument("--json", type=Path, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = compare(args.tags, baseline=args.baseline or None)
    out = args.json or (common.EXP_ROOT / "_comparison" / "sid_quality_comparison.json")
    common.write_json(out, summary)
    print(f"  report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
