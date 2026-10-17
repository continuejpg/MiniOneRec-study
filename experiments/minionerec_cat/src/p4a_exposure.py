"""Phase 4A -- quantify exposure bias and separate coverage from ranking."""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

D = Path("/root/autodl-tmp")
OUT = D / "p4a"
STRATA = ("freq0", "low(1-2)", "mid(3-20)", "high(>20)")


def st(f):
    return "freq0" if f == 0 else "low(1-2)" if f <= 2 else "mid(3-20)" if f <= 20 else "high(>20)"


def gain(r):
    return (1.0 / math.log(r + 2)) / (1.0 / math.log(2))


expo = {int(k): v for k, v in json.loads((OUT / "p4a_exposure.json").read_text()).items()}
summ = json.loads((OUT / "p4a_summary.json").read_text())

print("=" * 100)
print("EXPOSURE BIAS: how many Top-20 slots each stratum receives vs how many rows it owns")
print("=" * 100)
for S in ("S0", "S1", "S2"):
    v = summ[f"{S}_valid"]
    total_slots = sum(v["surfaced_freq_mix"].values())
    total_rows = sum(v["by_stratum"][k]["rows"] for k in STRATA)
    print(f"\n  --- {S} (VALID) ---")
    print(f"  {'stratum':<12}{'rows':>7}{'row%':>8}{'slots':>9}{'slot%':>8}"
          f"{'exposure_ratio':>16}{'items_avail':>13}{'items_surfaced':>15}")
    for k in STRATA:
        rows = v["by_stratum"][k]["rows"]
        slots = v["surfaced_freq_mix"].get(k, 0)
        cov = v["candidate_coverage_by_stratum"][k]
        rowp = 100 * rows / total_rows
        slotp = 100 * slots / total_slots
        ratio = slotp / rowp if rowp else float("nan")
        print(f"  {k:<12}{rows:>7}{rowp:>7.2f}%{slots:>9}{slotp:>7.2f}%"
              f"{ratio:>16.3f}{cov['items_in_stratum']:>13}{cov['items_surfaced_at_least_once']:>15}")

print()
print("=" * 100)
print("COVERAGE vs RANKING: decompose the HR@20 shortfall")
print("=" * 100)
for S in ("S0", "S1"):
    v = summ[f"{S}_valid"]
    print(f"\n  --- {S} (VALID) ---")
    print(f"  {'stratum':<12}{'coverage':>10}{'HR@20':>10}{'HR@20|covered':>15}"
          f"{'HR@1|covered':>14}{'best-case HR':>14}")
    for k in STRATA:
        b = v["by_stratum"][k]
        cov = v["candidate_coverage_by_stratum"][k]["coverage"]
        hr = b["HR@20"]
        cond = hr / cov if cov else float("nan")
        cond1 = b["HR@1"] / cov if cov else float("nan")
        print(f"  {k:<12}{100*cov:>9.2f}%{hr:>10.5f}{cond:>15.5f}{cond1:>14.5f}{cov:>14.4f}")
    print("  reading: coverage = ceiling if every covered row hit; HR@20|covered = realized")
    print("           rate among covered rows. A large gap => RANKING problem; a low")
    print("           coverage with high conditional rate => EXPOSURE problem.")

print()
print("=" * 100)
print("BEST RANK ACHIEVED BY THE TARGET when it IS surfaced (VALID)")
print("=" * 100)
for S in ("S0", "S1"):
    rows = json.loads((OUT / f"{S}_valid_per_row.json").read_text())
    print(f"\n  --- {S} ---")
    for k in STRATA:
        ranks = [r["repo_rank"] for r in rows if r["stratum"] == k and r["repo_rank"] is not None]
        if not ranks:
            print(f"  {k:<12} never surfaced (0 rows)")
            continue
        ranks.sort()
        print(f"  {k:<12} surfaced={len(ranks):>5}  median_rank={ranks[len(ranks)//2]:>3}  "
              f"mean_rank={sum(ranks)/len(ranks):>6.2f}  in_top1={sum(1 for x in ranks if x==0):>4}  "
              f"in_top5={sum(1 for x in ranks if x<5):>4}  in_top10={sum(1 for x in ranks if x<10):>4}")

print()
print("=" * 100)
print("CANDIDATE-LIST COMPOSITION: what fraction of each Top-20 belongs to each stratum")
print("=" * 100)
for S in ("S0", "S1", "S2"):
    v = summ[f"{S}_valid"]
    mix = v["surfaced_freq_mix"]
    tot = sum(mix.values())
    print(f"  {S}: " + "  ".join(f"{k}={100*mix.get(k,0)/tot:.2f}%" for k in STRATA))
print("  (uniform would be: " + "  ".join(
    f"{k}={100*summ['S0_valid']['by_stratum'][k]['rows']/sum(summ['S0_valid']['by_stratum'][x]['rows'] for x in STRATA):.2f}%"
    for k in STRATA) + ")")

# how many distinct items does a typical Top-20 draw on
print()
print("=" * 100)
print("ITEM REACH")
print("=" * 100)
for S in ("S0", "S1", "S2"):
    v = summ[f"{S}_valid"]
    print(f"  {S}: distinct items in Top20={v['distinct_items_surfaced_top20']} of 3686 "
          f"({100*v['distinct_items_surfaced_top20']/3686:.1f}%)   "
          f"in Top1={v['distinct_items_surfaced_top1']}   "
          f"mean exposures per surfaced item={v['mean_exposure_per_surfaced_item']}")
