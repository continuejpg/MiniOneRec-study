#!/usr/bin/env python3
"""
analysis/analyze_pair_identity_retention.py

Corrects the pair-prefix retention definition used in the shuffled-SID preflight.

BUG IN THE PREVIOUS DEFINITION
------------------------------
The earlier metric was  sum_P C(|P|,2)_after / sum_P C(|P|,2)_before  where P runs
over prefix groups. Because the permutation only re-orders SID *strings* within a
frequency bucket, every prefix group keeps its exact SID set, hence every |P| is
unchanged, hence the ratio is identically 100%. It measures nothing about whether
the SAME item pairs still share a prefix -- the group owners are almost entirely
different.

CORRECT DEFINITION
------------------
  BEFORE_pairs = { unordered item pairs (i,j) whose ORIGINAL SIDs share prefix of
                   length d }
  AFTER_pairs  = { unordered item pairs (i,j) whose SHUFFLED SIDs share prefix of
                   length d }
  retention = |BEFORE ∩ AFTER| / |BEFORE|
  jaccard   = |BEFORE ∩ AFTER| / |BEFORE ∪ AFTER|

Reported for d = 1, 2 over all items and over singleton items only.

Read-only. Trains nothing. Does not modify the permutation or the data.
Writes analysis/results/shuffled_sid_preflight.json (adds the corrected section).
"""
import json
import os
import re
import sys
from collections import defaultdict
from itertools import combinations

CODE = "/root/autodl-tmp/code"
DATA = f"{CODE}/data/Amazon"
SH = f"{CODE}/analysis/shuffled_sid"
RES = f"{CODE}/analysis/results"
PREFLIGHT = f"{RES}/shuffled_sid_preflight.json"
CAT = "Industrial_and_Scientific"

# TokenExtender (sft.py:31-39) rebuilds the index name as stem + ".index.json",
# so the shuffled index must follow the "<Category>.index.json" convention.
INDEX_NAME = f"{CAT}.index.json"

_SRE = re.compile(r"<[^<>]+>")


def toks(s):
    return _SRE.findall(s)


def pair_set(sid_of_item, items, depth):
    """Unordered item pairs whose SIDs share a token prefix of length `depth`."""
    groups = defaultdict(list)
    for i in items:
        tk = toks(sid_of_item[i])
        if len(tk) >= depth:
            groups[tuple(tk[:depth])].append(i)
    out = set()
    for members in groups.values():
        if len(members) > 1:
            out.update(combinations(sorted(members), 2))
    return out


def main():
    orig = {int(k): "".join(v) for k, v in
            json.load(open(f"{DATA}/index/{CAT}.index.json", encoding="utf-8")).items()}
    new = {int(k): "".join(v) for k, v in
           json.load(open(f"{SH}/{INDEX_NAME}", encoding="utf-8")).items()}

    # reuse the frozen-collision list recorded by the strict build
    mapping = json.load(open(f"{SH}/mapping.json", encoding="utf-8"))
    coll_items = sorted(mapping["frozen_collision_items"])
    all_items = sorted(orig)
    single_items = [i for i in all_items if i not in set(coll_items)]

    print("=" * 104)
    print("PAIR-IDENTITY PREFIX RETENTION (corrected definition)")
    print("=" * 104)
    print(f"all items      = {len(all_items)}")
    print(f"singleton items= {len(single_items)}")
    print(f"frozen collision items = {len(coll_items)}")

    results = {}
    for d in [1, 2]:
        for label, subset in [("all", all_items), ("singleton", single_items)]:
            B = pair_set(orig, subset, d)
            A = pair_set(new, subset, d)
            inter = B & A
            union = B | A
            ret = len(inter) / len(B) if B else 0.0
            jac = len(inter) / len(union) if union else 0.0
            results[f"depth{d}_{label}"] = {
                "depth": d, "item_set": label, "n_items": len(subset),
                "BEFORE_pairs": len(B), "AFTER_pairs": len(A),
                "intersection": len(inter), "union": len(union),
                "retention": round(ret, 6), "jaccard": round(jac, 6),
                "n_removed": len(B - A), "n_added": len(A - B),
            }
            print(f"\n--- depth={d}  [{label}]  n_items={len(subset)} ---")
            print(f"  |BEFORE_pairs| = {len(B):>10,}")
            print(f"  |AFTER_pairs|  = {len(A):>10,}")
            print(f"  |intersection| = {len(inter):>10,}")
            print(f"  |union|        = {len(union):>10,}")
            print(f"  retention      = {ret*100:8.4f}%   (intersection / BEFORE)")
            print(f"  Jaccard        = {jac*100:8.4f}%   (intersection / union)")
            print(f"  removed={len(B-A):,}  added={len(A-B):,}")

    # ---------------------------------------------------------------- self retention (unchanged)
    def self_ret(subset, d):
        return sum(1 for i in subset
                   if toks(orig[i])[:d] == toks(new[i])[:d]) / len(subset)

    self_stats = {
        "exact_SID_retention_all_items": round(
            sum(1 for i in all_items if orig[i] == new[i]) / len(all_items), 6),
        "exact_SID_retention_singleton_items": round(
            sum(1 for i in single_items if orig[i] == new[i]) / len(single_items), 6),
        "prefix1_self_retention_all": round(self_ret(all_items, 1), 6),
        "prefix1_self_retention_singleton": round(self_ret(single_items, 1), 6),
        "prefix2_self_retention_all": round(self_ret(all_items, 2), 6),
        "prefix2_self_retention_singleton": round(self_ret(single_items, 2), 6),
    }
    print("\n" + "=" * 104)
    print("SELF RETENTION (kept from before)")
    print("=" * 104)
    for k, v in self_stats.items():
        print(f"  {k:42s} = {v*100:8.4f}%")

    # ---------------------------------------------------------------- merge into preflight
    pf = json.load(open(PREFLIGHT, encoding="utf-8"))
    pf["pair_identity_retention"] = {
        "definition": ("unordered item pairs whose SIDs share a token prefix of length d; "
                       "BEFORE from the original mapping, AFTER from the shuffled mapping; "
                       "retention = |B∩A|/|B|, jaccard = |B∩A|/|B∪A|"),
        "supersedes": ("the earlier 'pair prefix-d RETENTION' entries in "
                       "intervention_strength, which were |after|/|before| over prefix-group "
                       "pair counts and were identically 100% by construction; they measure "
                       "group-size preservation, NOT pair identity"),
        "results": results,
    }
    pf["intervention_strength_corrected"] = self_stats
    with open(PREFLIGHT, "w", encoding="utf-8") as f:
        json.dump(pf, f, indent=2)
    print(f"\n[save] {PREFLIGHT}")
    print("[done]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
