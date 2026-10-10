#!/usr/bin/env python3
"""
analysis/analyze_sid_prefix_matched_control.py

Read-only. Trains nothing. Supersedes the branch-size control in
analyze_sid_prefix_control.py, which used a DIFFERENT control variable per depth
(prefix-1 support for depth=1, prefix-2 support for depth=2) and therefore did not
compare depths under a common control.

Fix applied here
----------------
Every test sample gets ONE control variable, regardless of its LCP depth:

    target_prefix2_support = # unique catalogue SIDs sharing the target SID's
                             first two tokens

depths 0, 1 and 2 are then compared inside the same target_prefix2_support
buckets, and additionally inside the same frequency band.

Structural constraint (not a bug): a target whose prefix-2 branch contains only
itself (support == 1) cannot share two tokens with any history item, so
depth == 2 has zero samples in that bucket. Cells that are structurally
impossible are reported as 0 / "n/a" and never filled in.

No causal claim is made. Output: analysis/results/sid_prefix_matched_control.json
"""
import ast
import csv
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np

CODE = "/root/autodl-tmp/code"
RUNS = "/root/autodl-tmp/runs"
DATA = f"{CODE}/data/Amazon"
CAT = "Industrial_and_Scientific"

TRAIN_CSV = f"{DATA}/train/{CAT}_5_2016-10-2018-11.csv"
TEST_CSV = f"{DATA}/test/{CAT}_5_2016-10-2018-11.csv"
INDEX_JSON = f"{DATA}/index/{CAT}.index.json"
OUT_DIR = f"{CODE}/analysis/results"

# audited provenance (see analysis/audit_predictions.py)
MODELS = {
    "SFT_full":         f"{RUNS}/eval_clean_sft/test_beam20.json",
    "GRPO_1.5ep":       f"{RUNS}/eval_grpo_step26274/test_beam20.json",
    "GRPO_2.0ep":       f"{RUNS}/eval_grpo_baseline/test_beam20.json",
    "GRPO_0.25ep_orig": f"{RUNS}/grpo_short025/eval_beam20.json",
    "GRPO_0.25ep_fast": f"{RUNS}/grpo_fast025/eval_beam20.json",
}
# models reported in the main tables (as requested)
REPORT_MODELS = ["SFT_full", "GRPO_0.25ep_orig", "GRPO_1.5ep", "GRPO_2.0ep"]

BEAM = 20
TOPK = [1, 3, 5, 10, 20]
SUP_BUCKETS = [(1, 1, "1"), (2, 5, "2-5"), (6, 20, "6-20"),
               (21, 100, "21-100"), (101, 10 ** 9, ">100")]
FREQ_BANDS = [(0, 5, "freq 0-5"), (6, 20, "freq 6-20"), (21, 10 ** 9, "freq >20")]
MIN_CELL = 30

PROBLEMS = []
_SID_RE = None


def problem(m):
    PROBLEMS.append(m)
    print(f"  !! {m}")


def sid_tokens(s):
    global _SID_RE
    if _SID_RE is None:
        _SID_RE = re.compile(r"<[^<>]+>")
    return _SID_RE.findall(s)


def lcp_depth(target_sid, hist_sids):
    tt = sid_tokens(target_sid)
    best = 0
    for h in hist_sids:
        ht = sid_tokens(h)
        d = 0
        for a, b in zip(tt, ht):
            if a != b:
                break
            d += 1
        if d > best:
            best = d
    return best


def hit_metrics(preds, target_sid):
    rank = -1
    for i, p in enumerate(preds):
        if p == target_sid:
            rank = i
            break
    out = {f"HR@{k}": (1.0 if 0 <= rank < k else 0.0) for k in TOPK}
    out["NDCG@20"] = (1.0 / math.log2(rank + 2)) if 0 <= rank < 20 else 0.0
    return out


def agg(sub, model):
    n = len(sub)
    keys = [f"HR@{k}" for k in TOPK] + ["NDCG@20"]
    if n == 0:
        return {"n": 0, **{k: None for k in keys}}
    return {"n": n, **{k: round(float(np.mean([s[model][k] for s in sub])), 6) for k in keys}}


def pctstr(v):
    return f"{v*100:8.3f}" if v is not None else "     n/a"


def medstr(v):
    return f"{v:8.1f}" if v is not None else "     n/a"


def table(headers, rows, title):
    print()
    print("=" * 150)
    print(title)
    print("=" * 150)
    w = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0)) for i, h in enumerate(headers)]
    print("  ".join(str(h).ljust(x) for h, x in zip(headers, w)))
    print("-" * 150)
    for r in rows:
        print("  ".join(str(c).ljust(x) for c, x in zip(r, w)))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 150)
    print("SID prefix-affinity -- MATCHED control on target_prefix2_support (offline, read-only)")
    print("=" * 150)

    train = list(csv.DictReader(open(TRAIN_CSV, encoding="utf-8")))
    test = list(csv.DictReader(open(TEST_CSV, encoding="utf-8")))
    idx = json.load(open(INDEX_JSON, encoding="utf-8"))
    item2sid = {int(k): "".join(v) for k, v in idx.items()}
    cat_sids = set(item2sid.values())
    print(f"[data] train={len(train)} test={len(test)} items={len(item2sid)} uniqueSIDs={len(cat_sids)}")

    # ---- the single control variable: catalogue SIDs sharing target's prefix-2
    p2_sids = defaultdict(set)
    for sid in item2sid.values():
        tk = sid_tokens(sid)
        if len(tk) >= 2:
            p2_sids[tk[0] + tk[1]].add(sid)
    print(f"[ctrl] distinct prefix-2 keys in catalogue: {len(p2_sids)}")

    preds = {}
    for name, path in MODELS.items():
        if not os.path.exists(path):
            problem(f"{name}: missing {path}")
            continue
        d = json.load(open(path, encoding="utf-8"))
        if len(d) != len(test):
            problem(f"{name}: {len(d)} rows != {len(test)}")
            continue
        preds[name] = [x["predict"] for x in d]
    report_models = [m for m in REPORT_MODELS if m in preds]
    print(f"[pred] reporting models: {report_models}")
    if not report_models:
        print("*** no usable predictions ***")
        return 2

    # ---- per-sample
    train_freq = Counter(r["item_id"] for r in train)
    samples = []
    for i, r in enumerate(test):
        tsid = r["item_sid"]
        tk = sid_tokens(tsid)
        sup = len(p2_sids.get(tk[0] + tk[1], ())) if len(tk) >= 2 else 0
        hist = [str(x) for x in ast.literal_eval(r["history_item_sid"])]
        f = train_freq.get(r["item_id"], 0)
        rec = {
            "idx": i, "target_item": r["item_id"], "target_sid": tsid,
            "freq": f, "depth": lcp_depth(tsid, hist), "prefix2_support": sup,
        }
        rec["band"] = next((nm for lo, hi, nm in FREQ_BANDS if lo <= f <= hi), "?")
        for m in report_models:
            rec[m] = hit_metrics(preds[m][i], tsid)
        samples.append(rec)

    sup_all = [s["prefix2_support"] for s in samples]
    print(f"[ctrl] prefix2_support: min={min(sup_all)} max={max(sup_all)} "
          f"median={int(np.median(sup_all))}")

    # ============================================================ 1+2 by support bucket
    r12, rows12 = [], []
    for lo, hi, bname in SUP_BUCKETS:
        sub_b = [s for s in samples if lo <= s["prefix2_support"] <= hi]
        for depth in [0, 1, 2]:
            sub = [s for s in sub_b if s["depth"] == depth]
            entry = {
                "support_bucket": bname, "support_range": [lo, hi if hi < 10 ** 9 else None],
                "depth": depth, "n": len(sub),
                "freq_median": (float(np.median([s["freq"] for s in sub])) if sub else None),
                "freq_mean": (round(float(np.mean([s["freq"] for s in sub])), 4) if sub else None),
                "models": {m: agg(sub, m) for m in report_models},
                "structurally_possible": len(sub) > 0,
            }
            r12.append(entry)
            row = [bname, f"depth={depth}", len(sub), medstr(entry["freq_median"])]
            for m in report_models:
                row += [pctstr(entry["models"][m]["HR@20"]), pctstr(entry["models"][m]["NDCG@20"])]
            rows12.append(row)
    table(["prefix2 support", "depth", "n", "freq med"]
          + sum([[f"{m} HR@20", f"{m} NDCG@20"] for m in report_models], []),
          rows12, "1+2. HR@20 / NDCG@20 by target_prefix2_support bucket x LCP depth "
                  "(SAME control variable for every depth)")

    # ============================================================ 3 triple split
    r3, rows3 = [], []
    for lo_f, hi_f, bname in FREQ_BANDS:
        for lo_s, hi_s, sname in SUP_BUCKETS:
            for depth in [0, 1, 2]:
                sub = [s for s in samples
                       if lo_f <= s["freq"] <= hi_f
                       and lo_s <= s["prefix2_support"] <= hi_s
                       and s["depth"] == depth]
                if len(sub) < MIN_CELL:
                    continue
                entry = {
                    "freq_band": bname, "support_bucket": sname, "depth": depth, "n": len(sub),
                    "freq_median": float(np.median([s["freq"] for s in sub])),
                    "models": {m: agg(sub, m) for m in report_models},
                }
                r3.append(entry)
                row = [bname, sname, f"depth={depth}", len(sub), medstr(entry["freq_median"])]
                for m in report_models:
                    row += [pctstr(entry["models"][m]["HR@20"]), pctstr(entry["models"][m]["NDCG@20"])]
                rows3.append(row)
    table(["freq band", "prefix2 sup", "depth", "n", "freq med"]
          + sum([[f"{m} HR@20", f"{m} NDCG@20"] for m in report_models], []),
          rows3, f"3. freq band x prefix2_support x depth (only cells with n >= {MIN_CELL})")

    # ============================================================ 5 verdict
    print()
    print("=" * 150)
    print("5. VERDICT -- within the same prefix2_support bucket, is depth=2 still > depth=0/1?")
    print("=" * 150)
    verdict = []
    for lo, hi, bname in SUP_BUCKETS:
        d2 = next((e for e in r12 if e["support_bucket"] == bname and e["depth"] == 2), None)
        d1 = next((e for e in r12 if e["support_bucket"] == bname and e["depth"] == 1), None)
        d0 = next((e for e in r12 if e["support_bucket"] == bname and e["depth"] == 0), None)
        if not d2 or d2["n"] == 0:
            note = ("depth=2 structurally impossible here (branch has no other member, "
                    "so no history item can share two tokens)")
            print(f"\n  support={bname:8s} : {note}")
            verdict.append({"support_bucket": bname, "comparable": False, "note": note})
            continue
        for m in report_models:
            h2 = d2["models"][m]["HR@20"]
            h1 = d1["models"][m]["HR@20"] if d1 and d1["n"] else None
            h0 = d0["models"][m]["HR@20"] if d0 and d0["n"] else None
            s1 = f"{h1*100:6.3f}" if h1 is not None else "   n/a"
            s0 = f"{h0*100:6.3f}" if h0 is not None else "   n/a"
            dl = f"{(h2-h1)*100:+7.3f}" if h1 is not None else "    n/a"
            print(f"  support={bname:8s} depth=2 n={d2['n']:4d}  {m:18s} "
                  f"depth0={s0}  depth1={s1}  depth2={h2*100:6.3f}   (d2-d1={dl})")
        verdict.append({
            "support_bucket": bname, "comparable": True,
            "n": {"depth0": d0["n"] if d0 else 0, "depth1": d1["n"] if d1 else 0, "depth2": d2["n"]},
            "HR@20": {m: {"depth0": (d0["models"][m]["HR@20"] if d0 and d0["n"] else None),
                          "depth1": (d1["models"][m]["HR@20"] if d1 and d1["n"] else None),
                          "depth2": d2["models"][m]["HR@20"]} for m in report_models},
        })

    out = {
        "meta": {
            "train_csv": TRAIN_CSV, "test_csv": TEST_CSV, "index_json": INDEX_JSON,
            "test_rows": len(test), "beam": BEAM,
            "control_variable": ("target_prefix2_support = # unique catalogue SIDs sharing "
                                 "the target SID's first two tokens (same variable for "
                                 "every depth level)"),
            "supersedes": "analysis/results/sid_prefix_control.json (per-depth control variables)",
            "models": {m: MODELS[m] for m in report_models},
            "min_cell": MIN_CELL,
            "caveat": ("All results are correlational; no causal claim is made. "
                       "The random-hit-rate / relative-to-random argument used in the "
                       "earlier revision has been removed and is not used anywhere here."),
            "structural_note": ("depth=2 has zero samples in the support=1 bucket because a "
                                "branch containing a single SID cannot share two tokens with "
                                "any history item. That cell is left empty, not filled."),
        },
        "control_variable_distribution": {
            b: sum(1 for s in samples if lo <= s["prefix2_support"] <= hi)
            for lo, hi, b in SUP_BUCKETS
        },
        "table_1_2_support_x_depth": r12,
        "table_3_freq_x_support_x_depth": r3,
        "verdict": verdict,
        "problems": PROBLEMS,
    }
    p = os.path.join(OUT_DIR, "sid_prefix_matched_control.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n[save] {p}")
    if PROBLEMS:
        print("[problems]")
        for x in PROBLEMS:
            print(f"  - {x}")
    print("[done]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
