#!/usr/bin/env python3
"""
analysis/analyze_sid_value.py

Offline, read-only analysis of an already-trained MiniOneRec reproduction.
Trains nothing. Modifies no training/evaluation logic. Reuses the existing
SID-level beam-20 hit rule from calc.py so every number is comparable to the
project's reported metrics.

Three analyses
--------------
1. Popularity-stratified evaluation
   Target frequency = number of times the target item appears as the next-item
   label in the TRAIN split. Strata: 0 / 1 / 2 / 3-5 / 6-20 / >20.
   f == 0 is labelled `interaction-unseen` (NOT cold-start: the item may well
   appear inside training histories, it simply never appears as a next-item label).

2. SID prefix-affinity analysis
   For each test row, deepest LCP depth (0/1/2/3) between the target SID and any
   SID in the user's history. depth == 3 (target already in own history) is
   reported separately and excluded from the trend reading.

3. Collision-aware analysis
   Catalogue-level collision stats + per-sample slices
   (all / unique-SID-only / collision-affected).

Inputs are discovered, not assumed. If a required prediction file is missing the
script reports exactly what is missing and continues with what it has.

Run:  /root/miniconda3/bin/python analysis/analyze_sid_value.py
"""
import ast
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict

import numpy as np

# --------------------------------------------------------------------------- paths
CODE = "/root/autodl-tmp/code"
RUNS = "/root/autodl-tmp/runs"
DATA = f"{CODE}/data/Amazon"
CAT = "Industrial_and_Scientific"

TRAIN_CSV = f"{DATA}/train/{CAT}_5_2016-10-2018-11.csv"
TEST_CSV = f"{DATA}/test/{CAT}_5_2016-10-2018-11.csv"
INDEX_JSON = f"{DATA}/index/{CAT}.index.json"

OUT_DIR = f"{CODE}/analysis/results"

# ---------------------------------------------------------------------------
# Prediction files -> checkpoint provenance.
#
# Each entry below was audited by recomputing global HR@1/5/10/20 + NDCG@20 over
# all 4,533 test rows and matching the result against the recorded metrics of the
# corresponding run. `epoch` is the *trained fraction* reported by the run's own
# train.log ('epoch' field), i.e. the data volume actually consumed, NOT the
# scheduler name. See analysis/audit_predictions.py for the audit.
#
# NOTE (corrected 2026-10-05): the file under runs/eval_grpo_step26274/ is the
# 1.5-epoch checkpoint-26274 of the grpo_baseline run -- it is NOT a 0.25-epoch
# run, which is what this script previously claimed.
# ---------------------------------------------------------------------------
MODELS = {
    "SFT_full":
        (f"{RUNS}/eval_clean_sft/test_beam20.json",
         "SFT full (36,259 train pairs)"),
    "GRPO_1.5ep":
        (f"{RUNS}/eval_grpo_step26274/test_beam20.json",
         "GRPO 1.5ep (checkpoint-26274 of grpo_baseline)"),
    "GRPO_2.0ep":
        (f"{RUNS}/eval_grpo_baseline/test_beam20.json",
         "GRPO 2.0ep (grpo_baseline/final_checkpoint)"),
    "GRPO_0.25ep_orig":
        (f"{RUNS}/grpo_short025/eval_beam20.json",
         "GRPO 0.25ep original scheduling (grpo_short025)"),
    "GRPO_0.25ep_fast":
        (f"{RUNS}/grpo_fast025/eval_beam20.json",
         "GRPO 0.25ep optimized scheduling (grpo_fast025)"),
    "SFT_preunify":
        (f"{RUNS}/eval_industrial/final_result_Industrial_and_Scientific.json",
         "SFT full, evaluated BEFORE prompt unification (legacy protocol)"),
}

# SASRec checkpoint used to reconstruct per-sample top-20 candidates (inference only).
SASREC_CKPT = f"{RUNS}/sasrec_baseline/d_h64_neg100.pt"
SASREC_CFG = dict(hidden=64, state=10, dropout=0.3, heads=1)

BEAM = 20
TOPK = [1, 3, 5, 10, 20]

# Which model is used as the reference for the "delta vs SASRec" column.
REF_MODEL = "SFT_full"

STRATA = [(0, 0, "interaction-unseen (f=0)"), (1, 1, "f=1"), (2, 2, "f=2"),
          (3, 5, "f=3-5"), (6, 20, "f=6-20"), (21, 10 ** 9, "f>20")]

PROBLEMS = []


def problem(msg):
    PROBLEMS.append(msg)
    print(f"  !! {msg}")


# --------------------------------------------------------------------------- io
def read_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_sid_list(s):
    """history_item_sid is stored as a python list literal."""
    v = ast.literal_eval(s)
    return [str(x) for x in v]


_SID_TOKEN_RE = None


def sid_tokens(sid):
    """Split a SID into its atomic tokens:
    '<a_223><b_80><c_216>' -> ['<a_223>', '<b_80>', '<c_216>'].

    Token-wise comparison is required. A character-wise comparison would treat
    '<a_2>' as a prefix of '<a_223>', which is incorrect.
    """
    global _SID_TOKEN_RE
    if _SID_TOKEN_RE is None:
        import re
        _SID_TOKEN_RE = re.compile(r"<[^<>]+>")
    return _SID_TOKEN_RE.findall(sid)


def lcp_depth(target_sid, hist_sids):
    """Deepest common TOKEN prefix between the target SID and any history SID."""
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


def load_index():
    idx = json.load(open(INDEX_JSON, encoding="utf-8"))
    item2sid = {int(k): "".join(v) for k, v in idx.items()}
    sid2items = defaultdict(list)
    for it, sid in item2sid.items():
        sid2items[sid].append(it)
    return item2sid, dict(sid2items)


# --------------------------------------------------------------------------- metrics
def hit_metrics(preds, target_sid):
    """Reproduce calc.py: first exact SID match wins; credit HR@k for k > rank."""
    rank = -1
    for i, p in enumerate(preds):
        if p == target_sid:
            rank = i
            break
    out = {}
    for k in TOPK:
        out[f"HR@{k}"] = 1.0 if 0 <= rank < k else 0.0
    out["NDCG@20"] = (1.0 / math.log2(rank + 2)) if 0 <= rank < 20 else 0.0
    return out


def aggregate(rows, model):
    """rows: list of per-sample dicts containing model -> metric dict."""
    n = len(rows)
    agg = {"n": n}
    keys = [f"HR@{x}" for x in TOPK] + ["NDCG@20"]
    if n == 0:
        for k in keys:
            agg[k] = float("nan")
        return agg
    for k in keys:
        agg[k] = float(np.mean([r[model][k] for r in rows]))
    return agg


# --------------------------------------------------------------------------- SASRec
def sasrec_top20(test_rows, item2sid):
    """Run the already-trained SASRec checkpoint for top-20 item candidates."""
    if not os.path.exists(SASREC_CKPT):
        problem(f"SASRec checkpoint not found: {SASREC_CKPT}")
        return None
    try:
        import torch
        sys.path.insert(0, CODE)
        from sasrec import SASRec
    except Exception as e:
        problem(f"cannot import torch/sasrec for SASRec inference: {type(e).__name__}: {e}")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_items = max(item2sid) + 1
    PAD = n_items
    L = SASREC_CFG["state"]

    model = SASRec(SASREC_CFG["hidden"], n_items, L, SASREC_CFG["dropout"], device,
                   num_heads=SASREC_CFG["heads"]).to(device)
    sd = torch.load(SASREC_CKPT, map_location=device)
    model.load_state_dict(sd)
    model.eval()
    print(f"  [sasrec] loaded {os.path.basename(SASREC_CKPT)}  device={device}")

    states, lens = [], []
    for r in test_rows:
        h = ast.literal_eval(r["history_item_id"])
        h = [int(x) for x in h][-L:]
        states.append([PAD] * (L - len(h)) + h)
        lens.append(max(1, len(h)))

    preds = []
    with torch.no_grad():
        for i in range(0, len(states), 512):
            sb = torch.LongTensor(states[i:i + 512]).to(device)
            lb = torch.LongTensor(lens[i:i + 512]).to(device)
            out = model.forward_eval(sb, lb)
            if out.dim() == 1:
                out = out.unsqueeze(0)
            top = torch.topk(out[:, :n_items], k=BEAM, dim=1).indices.cpu().numpy()
            for row in top:
                preds.append([item2sid[int(it)] for it in row])
    return preds


# --------------------------------------------------------------------------- printing
def table(headers, rows, title):
    print()
    print("=" * 118)
    print(title)
    print("=" * 118)
    widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
              for i, h in enumerate(headers)]
    print("  ".join(str(h).ljust(w) for h, w in zip(headers, widths)))
    print("-" * 118)
    for r in rows:
        print("  ".join(str(c).ljust(w) for c, w in zip(r, widths)))


def fmt(v):
    return f"{v*100:7.3f}" if isinstance(v, float) else str(v)


# --------------------------------------------------------------------------- main
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 118)
    print("SID value analysis -- offline, read-only")
    print("=" * 118)

    # ---------------------------------------------------------------- inputs
    for p in [TRAIN_CSV, TEST_CSV, INDEX_JSON]:
        if not os.path.exists(p):
            problem(f"required data file missing: {p}")
    train = read_csv(TRAIN_CSV)
    test = read_csv(TEST_CSV)
    item2sid, sid2items = load_index()
    print(f"[data] train rows      = {len(train)}")
    print(f"[data] test rows       = {len(test)}")
    print(f"[data] catalogue items = {len(item2sid)}")
    print(f"[data] unique SIDs     = {len(sid2items)}")

    # ---------------------------------------------------------------- load predictions
    preds = {}
    for name, (path, label) in MODELS.items():
        if not os.path.exists(path):
            problem(f"prediction file for {name} missing: {path}")
            continue
        d = json.load(open(path, encoding="utf-8"))
        if len(d) != len(test):
            problem(f"{name}: prediction rows ({len(d)}) != test rows ({len(test)})")
            continue
        preds[name] = [x["predict"] for x in d]
        print(f"[pred] {name:12s} <- {path}  ({len(d)} rows, beam={len(d[0]['predict'])})")

    if "SFT_full" not in preds and "GRPO_2.0ep" not in preds:
        print("\n*** no usable MiniOneRec prediction files -- cannot run analysis ***")
        return 2

    # ---------------------------------------------------------------- alignment check
    print("\n[align] verifying prediction order against test CSV ...")
    ref = REF_MODEL if REF_MODEL in preds else next(iter(preds))
    pd_ref = json.load(open(MODELS[ref][0], encoding="utf-8"))
    mism = sum(1 for r, p in zip(test, pd_ref) if r["item_sid"] != p["output"].strip())
    if mism:
        problem(f"alignment: {mism}/{len(test)} rows where CSV item_sid != prediction output")
    else:
        print(f"  OK: {ref} predictions align with test CSV row order (0 mismatches)")

    # ---------------------------------------------------------------- SASRec
    print("\n[sasrec] reconstructing top-20 candidates from trained checkpoint (inference only) ...")
    sr_preds = sasrec_top20(test, item2sid)
    if sr_preds is not None:
        preds["SASRec"] = sr_preds

    model_names = [m for m in (["SASRec"] + list(MODELS.keys())) if m in preds]
    print(f"[pred] usable models: {model_names}")

    if REF_MODEL not in preds:
        problem(f"reference model {REF_MODEL} not available; delta column will be n/a")
    print(f"[pred] reference for delta column: {REF_MODEL}")

    # ---------------------------------------------------------------- per-sample table
    print("\n[perf] computing per-sample metrics ...")
    train_freq = Counter(r["item_id"] for r in train)
    samples = []
    for i, r in enumerate(test):
        target_sid = r["item_sid"]
        tgt_item = r["item_id"]
        f = train_freq.get(tgt_item, 0)

        hist = parse_sid_list(r["history_item_sid"])
        hist_sids = list(hist)
        depth = lcp_depth(target_sid, hist_sids)

        sid_group = sid2items.get(target_sid, [])
        stratum = None
        for lo, hi, nm in STRATA:
            if lo <= f <= hi:
                stratum = nm
                break
        rec = {
            "idx": i,
            "target_item": tgt_item,
            "target_sid": target_sid,
            "freq": f,
            "stratum": stratum,
            "depth": depth,
            "n_sid_siblings": len(sid_group),
            "collision": len(sid_group) > 1,
        }
        for m in model_names:
            rec[m] = hit_metrics(preds[m][i], target_sid)
        samples.append(rec)

    # ================================================================ ANALYSIS 1
    a1, rows = [], []
    for lo, hi, name in STRATA:
        sub = [s for s in samples if lo <= s["freq"] <= hi]
        entry = {"stratum": name, "freq_range": [lo, hi if hi < 10 ** 9 else None], "n": len(sub),
                 "models": {}}
        for m in model_names:
            a = aggregate(sub, m)
            entry["models"][m] = {k: round(float(a[k]), 6) for k in a if k != "n"}
        delta = None
        if REF_MODEL in preds and "SASRec" in preds and sub:
            delta = (float(np.mean([s[REF_MODEL]["HR@20"] for s in sub]))
                     - float(np.mean([s["SASRec"]["HR@20"] for s in sub])))
        entry["delta_ref_minus_SASRec_HR@20"] = (round(delta, 6) if delta is not None else None)
        entry["delta_ref_model"] = REF_MODEL
        a1.append(entry)

        row = [name, len(sub)]
        for m in model_names:
            row += [fmt(entry["models"][m]["HR@20"]), fmt(entry["models"][m]["NDCG@20"])]
        row.append(f"{delta*100:+7.3f}" if delta is not None else "n/a")
        rows.append(row)

    hdr = ["stratum", "n"]
    for m in model_names:
        hdr += [f"{m} HR@20", f"{m} NDCG@20"]
    hdr.append(f"{REF_MODEL}-SASRec")
    table(hdr, rows, "ANALYSIS 1  Popularity-stratified evaluation (test n=%d)" % len(samples))

    # detailed HR@1/5/10/20 per stratum
    a1b, rows2 = [], []
    for lo, hi, name in STRATA:
        sub = [s for s in samples if s["stratum"] == name]
        for m in model_names:
            a = aggregate(sub, m)
            a1b.append({"stratum": name, "model": m,
                        **{k: (round(float(v), 6) if isinstance(v, float) and not np.isnan(v) else v)
                           for k, v in a.items()}})
            rows2.append([name, m, a["n"]]
                         + [fmt(a.get(f"HR@{k}", float("nan"))) for k in [1, 5, 10, 20]]
                         + [fmt(a.get("NDCG@20", float("nan")))])
    table(["stratum", "model", "n", "HR@1", "HR@5", "HR@10", "HR@20", "NDCG@20"],
          rows2, "ANALYSIS 1b  Full HR/NDCG by stratum")

    # ================================================================ ANALYSIS 2
    a2, rows3 = [], []
    for depth, label in [(0, "depth=0 (no shared prefix)"), (1, "depth=1 (<a_x>)"),
                         (2, "depth=2 (<a_x><b_y>)"), (3, "depth=3 (target in history)")]:
        sub = [s for s in samples if s["depth"] == depth]
        freq = [s["freq"] for s in sub]
        entry = {"depth": depth, "label": label, "n": len(sub),
                 "freq_mean": round(float(np.mean(freq)), 4) if freq else None,
                 "freq_median": float(np.median(freq)) if freq else None,
                 "models": {}}
        for m in model_names:
            a = aggregate(sub, m)
            entry["models"][m] = {k: round(float(a[k]), 6) for k in a if k != "n"}
        a2.append(entry)

        row = [label, len(sub),
               f"{entry['freq_mean']:.2f}" if entry["freq_mean"] is not None else "-",
               f"{entry['freq_median']:.0f}" if entry["freq_median"] is not None else "-"]
        for m in model_names:
            row += [fmt(entry["models"][m]["HR@20"]), fmt(entry["models"][m]["NDCG@20"])]
        rows3.append(row)
    table(["LCP depth", "n", "freq mean", "freq med"]
          + sum([[f"{m} HR@20", f"{m} NDCG@20"] for m in model_names], []),
          rows3, "ANALYSIS 2  SID prefix affinity (depth=3 reported separately, excluded from trend)")

    # ================================================================ ANALYSIS 3
    n_items = len(item2sid)
    n_sids = len(sid2items)
    coll_groups = {s: it for s, it in sid2items.items() if len(it) > 1}
    coll_items = sum(len(v) for v in coll_groups.values())
    n_coll_samples = sum(1 for s in samples if s["collision"])

    print()
    print("=" * 118)
    print("ANALYSIS 3a  Catalogue collision statistics")
    print("=" * 118)
    print(f"  catalogue items                 : {n_items}")
    print(f"  unique SIDs                     : {n_sids}")
    print(f"  collision SID groups            : {len(coll_groups)}")
    print(f"  items involved in collisions    : {coll_items}")
    print(f"  collision rate (item level)     : {(n_items - n_sids)/n_items*100:.4f}%")
    print(f"  test samples with colliding SID : {n_coll_samples} / {len(samples)}"
          f"  ({n_coll_samples/len(samples)*100:.2f}%)")

    rows4, a3b = [], []
    for label, sub in [("all", samples),
                       ("unique-SID-only", [s for s in samples if not s["collision"]]),
                       ("collision-affected", [s for s in samples if s["collision"]])]:
        freq = [s["freq"] for s in sub]
        entry = {"slice": label, "n": len(sub),
                 "freq_mean": round(float(np.mean(freq)), 4) if freq else None,
                 "freq_median": float(np.median(freq)) if freq else None,
                 "pct_in_f_gt_20": round(float(np.mean([f > 20 for f in freq])), 6) if freq else None,
                 "models": {}}
        row = [label, len(sub),
               f"{entry['freq_mean']:.2f}" if freq else "-",
               f"{entry['pct_in_f_gt_20']*100:.1f}%" if freq else "-"]
        for m in model_names:
            a = aggregate(sub, m)
            entry["models"][m] = {k: (round(float(a[k]), 6) if not np.isnan(a[k]) else None)
                                  for k in a if k != "n"}
            row += [fmt(entry["models"][m]["HR@20"]), fmt(entry["models"][m]["NDCG@20"])]
        a3b.append(entry)
        rows4.append(row)
    table(["slice", "n", "freq mean", "% in f>20"]
          + sum([[f"{m} HR@20", f"{m} NDCG@20"] for m in model_names], []),
          rows4, "ANALYSIS 3b  Collision-aware slices")

    # ================================================================ provenance self-check
    # Recompute global metrics per file and compare with the recorded official
    # numbers, so a mislabelled prediction file cannot silently enter the analysis.
    OFFICIAL = {
        "SFT_full":         (0.19832341, 0.11786798),
        "GRPO_1.5ep":       (0.16170307, 0.10470676),
        "GRPO_2.0ep":       (0.16236488, 0.10447064),
        "GRPO_0.25ep_orig": (0.17626296, 0.10885475),
        "GRPO_0.25ep_fast": (0.17538054, 0.10824989),
    }
    EPOCH = {
        "SFT_full": "full SFT (all 36,259 train rows)",
        "GRPO_1.5ep": "1.5 epoch = checkpoint-26274 of the grpo_baseline run",
        "GRPO_2.0ep": "2.0 epoch = grpo_baseline/final_checkpoint",
        "GRPO_0.25ep_orig": "0.25 epoch, original scheduling (grpo_short025)",
        "GRPO_0.25ep_fast": "0.25 epoch, optimized scheduling (grpo_fast025)",
        "SFT_preunify": "full SFT evaluated BEFORE prompt unification (legacy protocol)",
    }
    prov, rows5 = [], []
    print()
    print("=" * 118)
    print("PROVENANCE SELF-CHECK  (global metrics recomputed from each prediction file)")
    print("=" * 118)
    for name in model_names:
        if name == "SASRec":
            hr20 = float(np.mean([s["SASRec"]["HR@20"] for s in samples]))
            nd20 = float(np.mean([s["SASRec"]["NDCG@20"] for s in samples]))
            prov.append({"model": name, "path": SASREC_CKPT, "label": MODELS[name][1]
                         if name in MODELS else "reconstructed SASRec baseline",
                         "epoch": "best of 4-config sweep",
                         "HR@20": round(hr20, 8), "NDCG@20": round(nd20, 8),
                         "official_HR@20": None, "official_NDCG@20": None,
                         "delta_L1": None, "verdict": "n/a (new baseline, no official reference)"})
            rows5.append([name, "reconstructed", "n/a", f"{hr20*100:.5f}%", f"{nd20*100:.5f}%",
                          "-", "-"])
            continue
        hr20 = float(np.mean([s[name]["HR@20"] for s in samples]))
        nd20 = float(np.mean([s[name]["NDCG@20"] for s in samples]))
        off = OFFICIAL.get(name)
        if off:
            d = abs(hr20 - off[0]) + abs(nd20 - off[1])
            ok = "MATCH" if d < 1e-6 else f"MISMATCH (d={d:.2e})"
        else:
            d, ok = None, "no official reference"
        prov.append({"model": name, "path": MODELS[name][0], "label": MODELS[name][1],
                     "epoch": EPOCH.get(name), "HR@20": round(hr20, 8),
                     "NDCG@20": round(nd20, 8),
                     "official_HR@20": off[0] if off else None,
                     "official_NDCG@20": off[1] if off else None,
                     "delta_L1": (round(d, 10) if d is not None else None), "verdict": ok})
        rows5.append([name, EPOCH.get(name, "-")[:34], f"{hr20:.8f}", f"{hr20*100:.5f}%",
                      f"{nd20*100:.5f}%", f"{off[0]:.8f}" if off else "-", ok])
    table(["model", "epoch / checkpoint", "HR@20", "HR@20 %", "NDCG@20 %", "official HR@20", "verdict"],
          rows5, "Provenance self-check")

    # ================================================================ save
    result = {
        "meta": {
            "train_csv": TRAIN_CSV, "test_csv": TEST_CSV, "index_json": INDEX_JSON,
            "train_rows": len(train), "test_rows": len(test),
            "beam": BEAM, "topk": TOPK,
            "hit_rule": "first exact SID match; HR@k credited when rank<k (same as calc.py)",
            "frequency_definition": "count of target item as next-item label in TRAIN split",
            "models": {m: (MODELS[m][0] if m in MODELS else
                           f"{SASREC_CKPT} [reconstructed by inference]")
                       for m in model_names},
            "sasrec_ckpt": SASREC_CKPT if "SASRec" in preds else None,
            "sasrec_inference_config": SASREC_CFG if "SASRec" in preds else None,
            "sasrec_note": ("per-sample top-20 candidates for SASRec were reconstructed by "
                            "running the already-trained checkpoint (d_h64_neg100, the best "
                            "of the 4-config sweep); no training was performed. This is "
                            "required because the sweep only stored aggregate metrics."),
            "caveat": "All results are correlational. No causal claim is made.",
            "label_correction": (
                "runs/eval_grpo_step26274/test_beam20.json is the 1.5-epoch "
                "checkpoint-26274 of the grpo_baseline run. It was previously (and "
                "incorrectly) labelled as a 0.25-epoch run in an earlier revision of "
                "this script. The two genuine 0.25-epoch runs live in "
                "runs/grpo_short025/eval_beam20.json (original scheduling) and "
                "runs/grpo_fast025/eval_beam20.json (optimized scheduling)."),
        },
        "model_provenance": prov,
        "analysis1_popularity_strata": a1,
        "analysis1b_full_metrics": a1b,
        "analysis2_prefix_affinity": a2,
        "analysis3a_catalogue_collisions": {
            "catalogue_items": n_items, "unique_sids": n_sids,
            "collision_sid_groups": len(coll_groups),
            "items_in_collisions": coll_items,
            "collision_rate_item_level": round((n_items - n_sids) / n_items, 6),
            "test_samples_collision_affected": n_coll_samples,
            "test_samples_total": len(samples),
        },
        "analysis3b_collision_slices": a3b,
        "problems": PROBLEMS,
    }
    out = os.path.join(OUT_DIR, "sid_value_analysis.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\n[save] {out}")

    if PROBLEMS:
        print("\n[problems encountered]")
        for p in PROBLEMS:
            print(f"  - {p}")
    print("\n[done]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
