#!/usr/bin/env python3
"""
analysis/analyze_sid_prefix_control.py

Read-only follow-up to analyze_sid_value.py. Trains nothing.

Question under test
-------------------
The previous analysis found HR@20 rising sharply with SID prefix-affinity depth
(0 -> 1 -> 2). That could be a mechanical artefact: a target whose prefix is
shared by only a handful of catalogue SIDs sits in a *small catalogue branch*,
and a beam of 20 candidates is more likely to land in it regardless of any
semantic effect. This script tests that explanation.

Per test sample it computes
  prefix1 / prefix2            : token prefixes of the target SID
  support_sid_p1 / _p2         : # unique catalogue SIDs sharing that prefix
  support_item_p1 / _p2        : # catalogue items sharing that prefix
  depth                        : max LCP depth vs the user's history (0/1/2/3)

Reports
  A  per-depth summary (n, freq mean/median, support mean/median/p25/p75,
     HR@20, NDCG@20 for SASRec / SFT_full / GRPO_0.25ep_orig / GRPO_1.5ep / GRPO_2.0ep)
     depth=3 listed separately, excluded from trend statements.
  B  depth=1/2 bucketed by the RELEVANT prefix support size, to check whether
     HR@20 in those depths is explained by "branch smaller than beam width".
  C  depth=0/1/2 compared inside coarse frequency bands (0-5 / 6-20 / >20) to
     reduce the popularity confound.
  D  LegalRate@20 and DuplicateRate@20 for every prediction file.

No causal claim is made anywhere. Output:
  analysis/results/sid_prefix_control.json
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

# audited provenance (see analyze_sid_value.py / audit_predictions.py)
MODELS = {
    "SFT_full":         f"{RUNS}/eval_clean_sft/test_beam20.json",
    "GRPO_1.5ep":       f"{RUNS}/eval_grpo_step26274/test_beam20.json",
    "GRPO_2.0ep":       f"{RUNS}/eval_grpo_baseline/test_beam20.json",
    "GRPO_0.25ep_orig": f"{RUNS}/grpo_short025/eval_beam20.json",
    "GRPO_0.25ep_fast": f"{RUNS}/grpo_fast025/eval_beam20.json",
    "SFT_preunify":     f"{RUNS}/eval_industrial/final_result_Industrial_and_Scientific.json",
}
SASREC_CKPT = f"{RUNS}/sasrec_baseline/d_h64_neg100.pt"
SASREC_CFG = dict(hidden=64, state=10, dropout=0.3, heads=1)

BEAM = 20
TOPK = [1, 3, 5, 10, 20]
FREQ_BANDS = [(0, 5, "freq 0-5"), (6, 20, "freq 6-20"), (21, 10 ** 9, "freq >20")]

PROBLEMS = []


def problem(m):
    PROBLEMS.append(m)
    print(f"  !! {m}")


_SID_RE = None


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


def hit_metrics(preds, target_sid, legal_set):
    rank = -1
    for i, p in enumerate(preds):
        if p == target_sid:
            rank = i
            break
    out = {f"HR@{k}": (1.0 if 0 <= rank < k else 0.0) for k in TOPK}
    out["NDCG@20"] = (1.0 / math.log2(rank + 2)) if 0 <= rank < 20 else 0.0
    out["_rank"] = rank
    return out


def pct(v):
    return f"{v*100:7.3f}" if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v)


def num(v):
    return f"{v:8.2f}" if isinstance(v, (int, float)) else str(v)


def table(headers, rows, title):
    print()
    print("=" * 140)
    print(title)
    print("=" * 140)
    w = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0)) for i, h in enumerate(headers)]
    print("  ".join(str(h).ljust(x) for h, x in zip(headers, w)))
    print("-" * 140)
    for r in rows:
        print("  ".join(str(c).ljust(x) for c, x in zip(r, w)))


def agg(sub, model):
    n = len(sub)
    keys = [f"HR@{k}" for k in TOPK] + ["NDCG@20"]
    if n == 0:
        return {"n": 0, **{k: float("nan") for k in keys}}
    return {"n": n, **{k: float(np.mean([s[model][k] for s in sub])) for k in keys}}


def dist(vals):
    if not vals:
        return dict(mean=None, median=None, p25=None, p75=None, min=None, max=None)
    a = np.array(vals, dtype=float)
    return dict(mean=round(float(a.mean()), 4), median=round(float(np.median(a)), 4),
                p25=round(float(np.percentile(a, 25)), 4), p75=round(float(np.percentile(a, 75)), 4),
                min=round(float(a.min()), 4), max=round(float(a.max()), 4))


# --------------------------------------------------------------------------- sasrec
def sasrec_top20(test_rows, item2sid):
    if not os.path.exists(SASREC_CKPT):
        problem(f"SASRec checkpoint missing: {SASREC_CKPT}")
        return None
    try:
        import torch
        sys.path.insert(0, CODE)
        from sasrec import SASRec
    except Exception as e:
        problem(f"cannot import torch/sasrec: {type(e).__name__}: {e}")
        return None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_items = max(item2sid) + 1
    PAD, L = n_items, SASREC_CFG["state"]
    m = SASRec(SASREC_CFG["hidden"], n_items, L, SASREC_CFG["dropout"], device,
               num_heads=SASREC_CFG["heads"]).to(device)
    m.load_state_dict(torch.load(SASREC_CKPT, map_location=device))
    m.eval()
    states, lens = [], []
    for r in test_rows:
        h = [int(x) for x in ast.literal_eval(r["history_item_id"])][-L:]
        states.append([PAD] * (L - len(h)) + h)
        lens.append(max(1, len(h)))
    preds = []
    with torch.no_grad():
        for i in range(0, len(states), 512):
            sb = torch.LongTensor(states[i:i + 512]).to(device)
            lb = torch.LongTensor(lens[i:i + 512]).to(device)
            out = m.forward_eval(sb, lb)
            if out.dim() == 1:
                out = out.unsqueeze(0)
            top = torch.topk(out[:, :n_items], k=BEAM, dim=1).indices.cpu().numpy()
            preds += [[item2sid[int(x)] for x in row] for row in top]
    return preds


# --------------------------------------------------------------------------- main
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 140)
    print("SID prefix-affinity CONFOUND CONTROL -- offline, read-only")
    print("=" * 140)

    train = list(csv.DictReader(open(TRAIN_CSV, encoding="utf-8")))
    test = list(csv.DictReader(open(TEST_CSV, encoding="utf-8")))
    idx = json.load(open(INDEX_JSON, encoding="utf-8"))
    item2sid = {int(k): "".join(v) for k, v in idx.items()}
    cat_sids = set(item2sid.values())
    print(f"[data] train rows={len(train)}  test rows={len(test)}  "
          f"catalogue items={len(item2sid)}  unique SIDs={len(cat_sids)}")

    # catalogue branch support
    p1_sids, p2_sids = defaultdict(set), defaultdict(set)
    p1_items, p2_items = Counter(), Counter()
    for it, sid in item2sid.items():
        tk = sid_tokens(sid)
        if len(tk) >= 1:
            p1_sids[tk[0]].add(sid); p1_items[tk[0]] += 1
        if len(tk) >= 2:
            k2 = tk[0] + tk[1]
            p2_sids[k2].add(sid); p2_items[k2] += 1

    # ---------------------------------------------------------------- predictions
    preds = {}
    for name, path in MODELS.items():
        if not os.path.exists(path):
            problem(f"{name}: prediction missing {path}")
            continue
        d = json.load(open(path, encoding="utf-8"))
        if len(d) != len(test):
            problem(f"{name}: {len(d)} rows != test {len(test)}")
            continue
        preds[name] = [x["predict"] for x in d]
    sr = sasrec_top20(test, item2sid)
    if sr:
        preds["SASRec"] = sr
    print(f"[pred] models: {list(preds.keys())}")

    # ---------------------------------------------------------------- per-sample
    train_freq = Counter(r["item_id"] for r in train)
    samples = []
    for i, r in enumerate(test):
        tsid = r["item_sid"]
        tk = sid_tokens(tsid)
        p1 = tk[0] if len(tk) >= 1 else None
        p2 = (tk[0] + tk[1]) if len(tk) >= 2 else None
        f = train_freq.get(r["item_id"], 0)
        hist = [str(x) for x in ast.literal_eval(r["history_item_sid"])]
        rec = {
            "idx": i, "target_item": r["item_id"], "target_sid": tsid,
            "freq": f,
            "depth": lcp_depth(tsid, hist),
            "support_sid_p1": len(p1_sids.get(p1, ())),
            "support_item_p1": p1_items.get(p1, 0),
            "support_sid_p2": len(p2_sids.get(p2, ())) if p2 else 0,
            "support_item_p2": p2_items.get(p2, 0) if p2 else 0,
        }
        rec["band"] = next((nm for lo, hi, nm in FREQ_BANDS if lo <= f <= hi), "?")
        for m in preds:
            rec[m] = hit_metrics(preds[m][i], tsid, cat_sids)
        samples.append(rec)

    model_names = [m for m in ["SASRec"] + list(MODELS) if m in preds]

    # ============================================================== A: per-depth
    a1, rows = [], []
    for depth in [0, 1, 2, 3]:
        sub = [s for s in samples if s["depth"] == depth]
        if not sub:
            continue
        sup = "support_sid_p2" if depth >= 2 else "support_sid_p1"
        entry = {
            "depth": depth, "n": len(sub),
            "freq": dist([s["freq"] for s in sub]),
            "support_sid_p1": dist([s["support_sid_p1"] for s in sub]),
            "support_sid_p2": dist([s["support_sid_p2"] for s in sub]),
            "support_item_p1": dist([s["support_item_p1"] for s in sub]),
            "support_item_p2": dist([s["support_item_p2"] for s in sub]),
            "relevant_support": sup, "models": {},
        }
        for m in model_names:
            entry["models"][m] = {k: round(float(v), 6) for k, v in agg(sub, m).items()}
        a1.append(entry)

        r = [f"depth={depth}", len(sub), num(entry["freq"]["mean"]), num(entry["freq"]["median"])]
        for key in ["support_sid_p1", "support_sid_p2"]:
            d = entry[key]
            r += [num(d["mean"]), num(d["median"]), num(d["p25"]), num(d["p75"])]
        for m in model_names:
            r += [pct(entry["models"][m]["HR@20"]), pct(entry["models"][m]["NDCG@20"])]
        rows.append(r)

    hdr = ["depth", "n", "freq mean", "freq med",
           "sidP1 mean", "sidP1 med", "sidP1 p25", "sidP1 p75",
           "sidP2 mean", "sidP2 med", "sidP2 p25", "sidP2 p75"]
    for m in model_names:
        hdr += [f"{m} HR@20", f"{m} NDCG@20"]
    table(hdr, rows, "A. Per-depth summary (depth=3 listed separately; not used for trend claims)")

    # ============================================================== B: support buckets
    a2, rowsB = [], []
    for depth in [1, 2]:
        key = "support_sid_p1" if depth == 1 else "support_sid_p2"
        sub_all = [s for s in samples if s["depth"] == depth]
        buckets = [(0, 5, "<=5"), (6, 20, "6-20 (<=beam)"), (21, 100, "21-100"),
                   (101, 10 ** 9, ">100")]
        for lo, hi, label in buckets:
            sub = [s for s in sub_all if lo <= s[key] <= hi]
            entry = {"depth": depth, "support_bucket": label, "support_key": key, "n": len(sub),
                     "freq": dist([s["freq"] for s in sub]),
                     "support": dist([s[key] for s in sub]), "models": {}}
            for m in model_names:
                entry["models"][m] = {k: round(float(v), 6) for k, v in agg(sub, m).items()}
            a2.append(entry)
            r = [f"depth={depth}", key.replace("support_sid_", ""), label, len(sub),
                 num(entry["freq"]["mean"]), num(entry["support"]["median"])]
            for m in model_names:
                r += [pct(entry["models"][m]["HR@20"]), pct(entry["models"][m]["NDCG@20"])]
            rowsB.append(r)
    table(["depth", "support key", "support bucket", "n", "freq mean", "support med"]
          + sum([[f"{m} HR@20", f"{m} NDCG@20"] for m in model_names], []),
          rowsB, "B. depth=1/2 bucketed by RELEVANT prefix support size "
                 "(does 'branch <= beam' explain the HR?)")

    # ============================================================== C: within freq band
    a3, rowsC = [], []
    for lo, hi, band in FREQ_BANDS:
        for depth in [0, 1, 2]:
            sub = [s for s in samples if s["band"] == band and s["depth"] == depth]
            entry = {"freq_band": band, "depth": depth, "n": len(sub), "models": {}}
            for m in model_names:
                entry["models"][m] = {k: round(float(v), 6) for k, v in agg(sub, m).items()}
            a3.append(entry)
            r = [band, f"depth={depth}", len(sub)]
            for m in model_names:
                r += [pct(entry["models"][m]["HR@20"]), pct(entry["models"][m]["NDCG@20"])]
            rowsC.append(r)
    table(["freq band", "depth", "n"]
          + sum([[f"{m} HR@20", f"{m} NDCG@20"] for m in model_names], []),
          rowsC, "C. depth=0/1/2 WITHIN coarse frequency bands (popularity confound reduced)")

    # ============================================================== D: legal / duplicate
    a4, rowsD = [], []
    for name, plist in preds.items():
        legal, dup, per_sample_uniq = [], [], []
        for pr in plist:
            legal.append(sum(1 for x in pr if x in cat_sids) / len(pr))
            dup.append((len(pr) - len(set(pr))) / len(pr))
            per_sample_uniq.append(len(set(pr)))
        a4.append({"model": name, "n": len(plist), "beam": len(plist[0]),
                   "LegalRate@20": round(float(np.mean(legal)), 6),
                   "DuplicateRate@20": round(float(np.mean(dup)), 6),
                   "unique_candidates_mean": round(float(np.mean(per_sample_uniq)), 4)})
        rowsD.append([name, len(plist), len(plist[0]),
                      f"{np.mean(legal)*100:.4f}%", f"{np.mean(dup)*100:.4f}%",
                      f"{np.mean(per_sample_uniq):.3f}"])
    table(["model", "n", "beam", "LegalRate@20", "DuplicateRate@20", "mean unique cand."],
          rowsD, "D. LegalRate@20 and DuplicateRate@20 per prediction file "
                 "(legal = prediction is a catalogue SID)")

    # ============================================================== save
    out = {
        "meta": {
            "test_csv": TEST_CSV, "train_csv": TRAIN_CSV, "index_json": INDEX_JSON,
            "test_rows": len(test), "beam": BEAM,
            "models": {m: MODELS.get(m, SASREC_CKPT) for m in model_names},
            "depth_definition": "max token-level LCP between target SID and any history SID",
            "support_definition": "unique catalogue SIDs sharing the target's prefix-1 / prefix-2",
            "caveat": "All results are correlational. No causal claim is made.",
            "purpose": ("test whether the depth=2 HR@20 advantage is explained by small "
                        "catalogue branches (prefix support smaller than beam width)"),
        },
        "A_per_depth": a1,
        "B_support_buckets": a2,
        "C_within_freq_band": a3,
        "D_legal_duplicate_rates": a4,
        "problems": PROBLEMS,
    }
    p = os.path.join(OUT_DIR, "sid_prefix_control.json")
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
