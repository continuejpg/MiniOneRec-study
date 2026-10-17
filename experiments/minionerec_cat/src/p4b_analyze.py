"""Phase 4B -- C0 vs C1 final Valid comparison (NO test used)."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

D = Path("/root/autodl-tmp")
CAT = "Industrial_and_Scientific"
W = D / "p3_armdata/S0"
EV = D / "p4b_eval"
OUT = D / "p4b_analysis"
OUT.mkdir(parents=True, exist_ok=True)
STRATA = ("freq0", "low(1-2)", "mid(3-20)", "high(>20)")


def gain(r):
    return (1.0 / math.log(r + 2)) / (1.0 / math.log(2))


def load_info(p):
    m = defaultdict(list)
    for line in p.read_text().splitlines():
        if line.strip():
            f = line.split("\t")
            m[f[0].strip()].append(int(f[2]))
    return {k: sorted(v) for k, v in m.items()}


def rows(p):
    with io.open(p, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def st(f):
    return "freq0" if f == 0 else "low(1-2)" if f <= 2 else "mid(3-20)" if f <= 20 else "high(>20)"


freq = Counter(int(r["item_id"]) for r in rows(W / "derived/train" / f"{CAT}_5_2016-10-2018-11.csv"))
src = rows(W / "derived/valid" / f"{CAT}_5_2016-10-2018-11.csv")
info = load_info(W / "derived/info" / f"{CAT}_5_2016-10-2018-11.txt")
universe = set(info)
users = [r["user_id"] for r in src]


def bstrap(a, b, clusters=None, n_boot=10000, seed=2024):
    import numpy as np
    a = np.asarray(a, float); b = np.asarray(b, float)
    d = b - a; n = d.shape[0]
    rng = np.random.default_rng(seed)
    if clusters is None:
        boot = d[rng.integers(0, n, size=(n_boot, n))].mean(axis=1); unit, nu = "row", n
    else:
        cl = np.asarray(clusters); uniq, inv = np.unique(cl, return_inverse=True)
        o = np.argsort(inv, kind="stable"); si = inv[o]
        mem = [o[np.searchsorted(si, i, "left"):np.searchsorted(si, i, "right")] for i in range(len(uniq))]
        boot = np.empty(n_boot)
        for t in range(n_boot):
            p = rng.integers(0, len(uniq), size=len(uniq))
            boot[t] = d[np.concatenate([mem[x] for x in p])].mean()
        unit, nu = "cluster", len(uniq)
    lo, hi = float(np.quantile(boot, .025)), float(np.quantile(boot, .975))
    return {"delta_pp": round(100 * float(d.mean()), 4), "ci_low_pp": round(100 * lo, 4),
            "ci_high_pp": round(100 * hi, 4), "excludes_zero": bool(lo > 0 or hi < 0),
            "unit": unit, "n_units": int(nu)}


res, recs = {}, {}
for A in ("C0", "C1"):
    art = EV / f"{A}_valid_beam20.json"
    data = json.loads(art.read_text())
    hits = nd = 0.0
    legal = tot = 0
    per = []
    strat = {k: {"n": 0, "hit": 0} for k in STRATA}
    surf = Counter()
    top1 = Counter()
    for i, row in enumerate(data):
        out = row["output"]
        tsid = (out[0].strip('"').strip(" ") if isinstance(out, list) else out.strip(' \n"'))
        tid = int(src[i]["item_id"]); f = freq.get(tid, 0); s = st(f)
        preds = [p.strip('"\n').strip() for p in row["predict"]][:20]
        tot += len(preds); legal += sum(1 for p in preds if p in universe)
        r = next((k for k, p in enumerate(preds) if p == tsid), None)
        strat[s]["n"] += 1
        if r is not None:
            hits += 1; nd += gain(r); strat[s]["hit"] += 1
        for pos, p in enumerate(preds):
            ids = info.get(p)
            if ids:
                surf[ids[0]] += 1
                if pos == 0:
                    top1[ids[0]] += 1
        per.append({"row_index": i, "user_id": src[i]["user_id"], "target_item_id": tid,
                    "target_sid": tsid, "target_next_freq": f, "stratum": s,
                    "repo_rank": r, "predicted_sids": preds,
                    "decoded_canonical_item_ids": [info.get(p, [None])[0] for p in preds]})
    n = len(data)
    res[A] = {
        "arm": A, "rows": n, "hits": int(hits),
        "HR@20": hits / n, "NDCG@20": nd / n,
        "legal_sid_rate": legal / tot,
        "assert_hits_over_rows_equals_HR": abs(hits / n - hits / n) < 1e-15,
        "by_stratum": {k: {"rows": v["n"], "hits": v["hit"],
                           "HR@20": v["hit"] / v["n"] if v["n"] else None} for k, v in strat.items()},
        "distinct_items_surfaced_top20": len(surf),
        "global_item_coverage": len(surf) / 3686,
        "distinct_items_top1": len(top1),
        "surfaced_freq_mix": {k: sum(c for it, c in surf.items() if st(freq.get(it, 0)) == k)
                              for k in STRATA},
        "slots_total": sum(surf.values()),
        "artifact_sha256": hashlib.sha256(art.read_bytes()).hexdigest()[:16],
    }
    recs[A] = per

print("=" * 100)
print("1. OVERALL (VALID, beam20) -- hits/rows == HR asserted")
print("=" * 100)
for A in ("C0", "C1"):
    r = res[A]
    print(f"  {A}: rows={r['rows']} hits={r['hits']} HR@20={r['HR@20']:.8f} "
          f"({100*r['HR@20']:.4f}%)  NDCG@20={r['NDCG@20']:.8f}  "
          f"legal={100*r['legal_sid_rate']:.2f}%  assert={r['assert_hits_over_rows_equals_HR']}")
print(f"  C1-C0: HR {(res['C1']['HR@20']-res['C0']['HR@20'])*100:+.4f} pp   "
      f"NDCG {(res['C1']['NDCG@20']-res['C0']['NDCG@20'])*100:+.4f} pp")

print()
print("=" * 100)
print("2. BY TARGET-FREQUENCY STRATUM (VALID HR@20)")
print("=" * 100)
print(f"  {'stratum':<12}{'rows':>7}{'C0 HR@20':>12}{'C1 HR@20':>12}{'C1-C0 (pp)':>13}")
for k in STRATA:
    a = res["C0"]["by_stratum"][k]; b = res["C1"]["by_stratum"][k]
    d = (b["HR@20"] - a["HR@20"]) * 100 if a["HR@20"] is not None and b["HR@20"] is not None else float("nan")
    print(f"  {k:<12}{a['rows']:>7}{a['HR@20']:>12.5f}{b['HR@20']:>12.5f}{d:>+13.4f}")

print()
print("=" * 100)
print("3. TOP-20 EXPOSURE BY ITEM TRAIN-FREQUENCY (did the mix shift?)")
print("=" * 100)
for k in STRATA:
    a = res["C0"]["surfaced_freq_mix"][k] / res["C0"]["slots_total"]
    b = res["C1"]["surfaced_freq_mix"][k] / res["C1"]["slots_total"]
    print(f"  {k:<12} C0={100*a:>6.2f}%  C1={100*b:>6.2f}%  shift={100*(b-a):>+6.2f} pp")

print()
print("=" * 100)
print("4. GLOBAL ITEM COVERAGE (distinct items appearing in any Top-20)")
print("=" * 100)
for A in ("C0", "C1"):
    r = res[A]
    print(f"  {A}: {r['distinct_items_surfaced_top20']}/3686 = "
          f"{100*r['global_item_coverage']:.2f}%   (Top-1: {r['distinct_items_top1']})")

print()
print("=" * 100)
print("5. PAIRED BOOTSTRAP C1-C0 (same 4,532 rows)")
print("=" * 100)
ha = [1 if r["repo_rank"] is not None else 0 for r in recs["C0"]]
hb = [1 if r["repo_rank"] is not None else 0 for r in recs["C1"]]
na = [gain(r["repo_rank"]) if r["repo_rank"] is not None else 0.0 for r in recs["C0"]]
nb = [gain(r["repo_rank"]) if r["repo_rank"] is not None else 0.0 for r in recs["C1"]]
boot = {"HR_row": bstrap(ha, hb), "HR_user": bstrap(ha, hb, users),
        "NDCG_row": bstrap(na, nb), "NDCG_user": bstrap(na, nb, users)}
for k, v in boot.items():
    print(f"  {k:<12} delta={v['delta_pp']:+.4f} pp  95%CI=[{v['ci_low_pp']:+.4f}, "
          f"{v['ci_high_pp']:+.4f}]  excludes0={v['excludes_zero']}  {v['unit']}({v['n_units']})")

# transition counts
both = sum(1 for a, b in zip(ha, hb) if a and b)
w = sum(1 for a, b in zip(ha, hb) if not a and b)
l = sum(1 for a, b in zip(ha, hb) if a and not b)
print(f"\n  transitions: both_hit={both}  C0fail->C1hit={w}  C0hit->C1fail={l}  both_fail={len(ha)-both-w-l}")

out = {"C0": res["C0"], "C1": res["C1"], "bootstrap": boot,
       "transitions": {"both_hit": both, "c1_wins": w, "c1_losses": l},
       "protocol": {"epochs": 1, "eff_batch": 64, "micro_bs": 8, "accum": 8,
                    "lr": 3e-4, "cutoff": 512, "seed": 42,
                    "alpha_C0": 0.0, "alpha_C1": 0.5,
                    "sid": "B0 seed 2024 (b5a7d78b0f50e6b5)",
                    "test_used": False}}
(OUT / "p4b_final_metrics.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
for A in ("C0", "C1"):
    (OUT / f"{A}_valid_per_row.json").write_text(json.dumps(recs[A], indent=1), encoding="utf-8")
print(f"\n  written under {OUT}")
