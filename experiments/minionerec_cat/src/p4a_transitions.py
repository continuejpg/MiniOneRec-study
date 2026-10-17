"""Phase 4A -- S0/S1 per-sample transitions, SID-structure effect, and mechanism tests.

Everything is computed on VALID unless stated otherwise.  Test is used only as a
descriptive cross-check and is never used to select anything.
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

D = Path("/root/autodl-tmp")
OUT = D / "p4a"
STRATA = ("freq0", "low(1-2)", "mid(3-20)", "high(>20)")


def load(n):
    return json.loads((OUT / n).read_text())


expo = load("p4a_exposure.json")
per_item = {int(k): v for k, v in expo.items()}
summary = load("p4a_summary.json")
S0 = load("S0_valid_per_row.json")
S1 = load("S1_valid_per_row.json")
S2 = load("S2_valid_per_row.json")


def st(f):
    return "freq0" if f == 0 else "low(1-2)" if f <= 2 else "mid(3-20)" if f <= 20 else "high(>20)"


def gain(r):
    return (1.0 / math.log(r + 2)) / (1.0 / math.log(2))


print("=" * 100)
print("4. S0 vs S1 PER-SAMPLE TRANSITIONS (VALID)")
print("=" * 100)
cats = {"S0_fail_S1_hit": [], "S0_hit_S1_fail": [], "both_fail": [], "both_hit": []}
for a, b in zip(S0, S1):
    ha, hb = a["repo_rank"] is not None, b["repo_rank"] is not None
    k = ("both_hit" if ha and hb else "S0_hit_S1_fail" if ha else
         "S0_fail_S1_hit" if hb else "both_fail")
    cats[k].append((a, b))

n = len(S0)
print(f"  {'category':<18}{'n':>7}{'share':>9}   freq mix (freq0/low/mid/high)"
      f"   mean_cluster_a  mean_next_freq")
for k, v in cats.items():
    if not v:
        continue
    mix = Counter(st(a["target_next_freq"]) for a, _ in v)
    mixs = "/".join(str(mix.get(s, 0)) for s in STRATA)
    ca = sum(per_item[a["target_item_id"]]["n_sid_owners"] for a, _ in v) / len(v)
    nf = sum(a["target_next_freq"] for a, _ in v) / len(v)
    print(f"  {k:<18}{len(v):>7}{100*len(v)/n:>8.2f}%   {mixs:<26} {ca:>12.3f}  {nf:>13.2f}")

# rank improvement among pairs where both hit
improved = sum(1 for a, b in zip(S0, S1)
               if a["repo_rank"] is not None and b["repo_rank"] is not None
               and b["repo_rank"] < a["repo_rank"])
worsened = sum(1 for a, b in zip(S0, S1)
               if a["repo_rank"] is not None and b["repo_rank"] is not None
               and b["repo_rank"] > a["repo_rank"])
both = len(cats["both_hit"])
print(f"\n  among {both} rows where both hit: S1 ranks better in {improved}, worse in {worsened}, "
      f"equal in {both-improved-worsened}")
print(f"  net HR change = {(len(cats['S0_fail_S1_hit'])-len(cats['S0_hit_S1_fail']))/n:+.5f}")

print()
print("  --- the rows S1 WINS (S0 fail -> S1 hit) ---")
w = cats["S0_fail_S1_hit"]
if w:
    print(f"    n={len(w)}  stratum mix={dict(Counter(a['stratum'] for a,_ in w))}")
    print(f"    S1 rank distribution: {dict(sorted(Counter(b['repo_rank'] for _,b in w).items()))}")
    print(f"    ambiguous rows among them: {sum(1 for a,_ in w if a['ambiguous'])}")
    print(f"    mean train next-freq: {sum(a['target_next_freq'] for a,_ in w)/len(w):.2f}")
print("  --- the rows S1 LOSES (S0 hit -> S1 fail) ---")
l = cats["S0_hit_S1_fail"]
if l:
    print(f"    n={len(l)}  stratum mix={dict(Counter(a['stratum'] for a,_ in l))}")
    print(f"    S0 rank distribution: {dict(sorted(Counter(a['repo_rank'] for a,_ in l).items()))}")
    print(f"    ambiguous rows among them: {sum(1 for a,_ in l if a['ambiguous'])}")
    print(f"    mean train next-freq: {sum(a['target_next_freq'] for a,_ in l)/len(l):.2f}")

print()
print("=" * 100)
print("4b. BOTH-FAIL rows: the residual long-tail problem")
print("=" * 100)
bf = cats["both_fail"]
print(f"  rows={len(bf)} ({100*len(bf)/n:.2f}% of valid)")
print(f"  stratum mix: {dict(Counter(a['stratum'] for a,_ in bf))}")
print(f"  distinct target items: {len({a['target_item_id'] for a,_ in bf})}")
print(f"  mean train next-freq: {sum(a['target_next_freq'] for a,_ in bf)/len(bf):.2f}")
amb = sum(1 for a, _ in bf if a["ambiguous"])
print(f"  ambiguous (shared-SID) rows: {amb} ({100*amb/len(bf):.2f}% of both-fail)")
print(f"  among non-ambiguous both-fail, mean next-freq = "
      f"{sum(a['target_next_freq'] for a,_ in bf if not a['ambiguous'])/max(sum(1 for a,_ in bf if not a['ambiguous']),1):.2f}")

print()
print("=" * 100)
print("4c. SID first-layer cluster size vs outcome (VALID, S1)")
print("=" * 100)
clus = defaultdict(lambda: {"n": 0, "hit": 0})
for r in S1:
    c = per_item[r["target_item_id"]]["n_sid_owners"]
    clus[c]["n"] += 1
    clus[c]["hit"] += int(r["repo_rank"] is not None)
print(f"  {'sid_owners':<12}{'rows':>7}{'HR@20':>10}")
for c in sorted(clus):
    v = clus[c]
    print(f"  {c:<12}{v['n']:>7}{v['hit']/v['n']:>10.5f}")

print()
print("=" * 100)
print("4d. HYPOTHESIS TEST: does SID structure matter beyond train frequency?")
print("=" * 100)
# Within the mid stratum, split by whether the item's SID is shared (collision)
mid = [r for r in S1 if st(r["target_next_freq"]) == "mid(3-20)"]
for label, sel in (("SID shared (ambiguous)", lambda r: r["ambiguous"]),
                   ("SID unique", lambda r: not r["ambiguous"])):
    g = [r for r in mid if sel(r)]
    if not g:
        continue
    hr = sum(1 for r in g if r["repo_rank"] is not None) / len(g)
    print(f"  mid stratum, {label:<24} n={len(g):>5} HR@20={hr:.5f} "
          f"mean_next_freq={sum(r['target_next_freq'] for r in g)/len(g):.2f}")
print("  (note: collision items are heavily popularity-selected, so this is confounded)")

print()
print("=" * 100)
print("5. MECHANISM SUMMARY (VALID)")
print("=" * 100)
v = summary["S1_valid"]
for k in STRATA:
    b = v["by_stratum"][k]
    cov = v["candidate_coverage_by_stratum"][k]["coverage"]
    print(f"  {k:<10} coverage={100*cov:>5.1f}%  HR@20={b['HR@20']:.5f}  "
          f"HR@1={b['HR@1']:.5f}  HR@5={b['HR@5']:.5f}")
    # conditional hit rate given surfaced
    print(f"             target in top20 {b['target_in_top20']}/{b['rows']} "
          f"-> conditional on coverage, {100*b['HR@20']/max(cov,1e-9):.2f}% of covered rows hit")

json.dump({"transitions": {k: len(v) for k, v in cats.items()},
           "s1_rank_improved": improved, "s1_rank_worsened": worsened,
           "both_hit": both},
          open(OUT / "p4a_transitions.json", "w"), indent=1)
print(f"\n  written {OUT/'p4a_transitions.json'}")
