#!/usr/bin/env python3
"""
analysis/build_shuffled_sid_strict.py

STRICT popularity-stratified shuffled-SID causal control.

Difference vs build_shuffled_sid.py
-----------------------------------
The earlier version permuted collision groups atomically, which forced equal-size
collision groups to trade SID strings -- so the exact SID multiset was NOT
preserved. This version freezes all collision-involved items (31 items / 15 SIDs)
at their original SIDs and permutes ONLY the 3655 singleton items.

Because the singleton SIDs form a closed set (verified: no singleton SID is held
by a collision item, and every singleton SID is unique), a bucket-local
derangement of singleton SIDs preserves:
    * the exact SID set,
    * the exact SID multiset (per-string multiplicity),
    * the collision groups and their item membership,
    * the trie structure at every depth,
    * the SID token vocabulary,
while still destroying the item <-> SID semantic correspondence for 99.16% of items.

Guarantee: every singleton gets a SID from its own frequency bucket, so the
popularity stratification is preserved exactly. The 31 collision items are frozen
(a deliberate, reported limitation of the intervention).

CPU only. Trains nothing. Never writes outside analysis/.
Outputs:
    analysis/shuffled_sid/<Category>.index.json | train.csv | valid.csv | test.csv
    analysis/shuffled_sid/mapping.json
    analysis/results/shuffled_sid_preflight.json
"""
import ast
import csv
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import CATEGORY, DATA_ROOT, SHUFFLED_DIR, RESULTS_DIR  # noqa: E402

DATA = DATA_ROOT
CAT = CATEGORY

TRAIN = f"{DATA}/train/{CAT}_5_2016-10-2018-11.csv"
VALID = f"{DATA}/valid/{CAT}_5_2016-10-2018-11.csv"
TEST = f"{DATA}/test/{CAT}_5_2016-10-2018-11.csv"
INDEX = f"{DATA}/index/{CAT}.index.json"

OUT_DIR = SHUFFLED_DIR        # same dir, regenerated
RES_DIR = RESULTS_DIR
SEED = 42

# TokenExtender (sft.py:31-39) does NOT use the path you pass to --sid_index_path
# verbatim. It rebuilds the filename as
#     dirname(sid_index_path) / basename(sid_index_path).split('.')[0] + ".index.json"
# so the index file MUST be named "<stem>.index.json" (upstream convention is
# "<Category>.index.json"). Naming it "index.json" makes the stem "index" and the
# rebuild produces "index.index.json" -> FileNotFoundError.
# data.py opens index_file directly and is unaffected; only TokenExtender is.
INDEX_NAME = f"{CAT}.index.json"

BUCKETS = [(0, 0, "f=0"), (1, 1, "f=1"), (2, 2, "f=2"),
           (3, 5, "f=3-5"), (6, 20, "f=6-20"), (21, 10 ** 9, "f=>20")]

_SID_RE = None


def sid_tokens(s):
    global _SID_RE
    if _SID_RE is None:
        _SID_RE = re.compile(r"<[^<>]+>")
    return _SID_RE.findall(s)


def bucket_of(f):
    for lo, hi, nm in BUCKETS:
        if lo <= f <= hi:
            return nm
    return "?"


def trie_prefixes(sids, depth):
    return {tuple(sid_tokens(s)[:depth]) for s in sids if len(sid_tokens(s)) >= depth}


def pair_share_rate(sid_of_item, items, depth):
    """Fraction of item PAIRS that share a token prefix of length `depth`."""
    groups = defaultdict(list)
    for i in items:
        tk = sid_tokens(sid_of_item[i])
        if len(tk) >= depth:
            groups[tuple(tk[:depth])].append(i)
    same = sum(len(v) * (len(v) - 1) // 2 for v in groups.values())
    total = len(items) * (len(items) - 1) // 2
    return (same / total) if total else 0.0, same, total


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(RES_DIR, exist_ok=True)
    report = {"seed": SEED, "scheme": "strict: collision items frozen, singletons permuted",
              "checks": [], "problems": []}

    def check(name, ok, detail=""):
        report["checks"].append({"check": name, "pass": bool(ok), "detail": detail})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
        return ok

    def problem(m):
        report["problems"].append(m)
        print(f"  !! {m}")

    print("=" * 104)
    print("BUILD STRICT popularity-stratified shuffled-SID control (CPU only)")
    print("=" * 104)

    # ---------------------------------------------------------------- load
    idx = json.load(open(INDEX, encoding="utf-8"))
    item2sid = {int(k): "".join(v) for k, v in idx.items()}
    n_items = len(item2sid)
    sid2items = defaultdict(list)
    for i, s in item2sid.items():
        sid2items[s].append(i)

    coll_sids = {s for s, v in sid2items.items() if len(v) > 1}
    coll_items = sorted(i for s in coll_sids for i in sid2items[s])
    single_items = sorted(set(item2sid) - set(coll_items))
    print(f"[in] items={n_items}  collision SIDs={len(coll_sids)}  "
          f"collision items={len(coll_items)}  singletons={len(single_items)}")

    freq = Counter(r["item_id"] for r in
                   csv.DictReader(open(TRAIN, encoding="utf-8")))

    # ---------------------------------------------------------------- mapping
    new_sid_of_item = {i: item2sid[i] for i in coll_items}   # frozen
    old_sid_to_new = {s: s for s in coll_sids}               # frozen

    rng = random.Random(SEED)
    per_bucket = []
    for lo, hi, nm in BUCKETS:
        s_items = sorted(i for i in single_items if lo <= freq.get(str(i), 0) <= hi)
        n = len(s_items)
        if n == 0:
            per_bucket.append({"bucket": nm, "n_singletons": 0, "note": "empty"})
            continue
        units_sid = [item2sid[i] for i in s_items]
        distinctive = len(set(units_sid))
        slots = units_sid[:]
        rng.shuffle(slots)
        repairs = 0
        if distinctive > 1:
            for i in range(n):
                if slots[i] == units_sid[i]:
                    for j in range(n):
                        if j != i and slots[j] != units_sid[i] and slots[i] != units_sid[j]:
                            slots[i], slots[j] = slots[j], slots[i]
                            repairs += 1
                            break
        fixed = sum(1 for i in range(n) if slots[i] == units_sid[i])
        if fixed:
            problem(f"bucket {nm}: {fixed} singleton(s) kept their SID")
        for i, it in enumerate(s_items):
            new_sid_of_item[it] = slots[i]
            old_sid_to_new[units_sid[i]] = slots[i]
        per_bucket.append({"bucket": nm, "n_singletons": n, "distinctive_sids": distinctive,
                           "repairs": repairs, "fixed_points": fixed})
        print(f"  bucket {nm:>7}: singletons={n:5d}  repairs={repairs:3d}  fixed={fixed}")

    if len(new_sid_of_item) != n_items:
        problem(f"mapping incomplete {len(new_sid_of_item)} != {n_items}")

    # ---------------------------------------------------------------- write
    with open(f"{OUT_DIR}/{INDEX_NAME}", "w", encoding="utf-8") as f:
        json.dump({str(i): sid_tokens(new_sid_of_item[i]) for i in sorted(item2sid)}, f)
    with open(f"{OUT_DIR}/mapping.json", "w", encoding="utf-8") as f:
        json.dump({"seed": SEED,
                   "scheme": "strict: 31 collision-involved items frozen at original SID; "
                             "3655 singleton items deranged within train-frequency buckets",
                   "frozen_collision_items": coll_items,
                   "frozen_collision_sids": sorted(coll_sids),
                   "item_to_original_sid": {str(i): item2sid[i] for i in sorted(item2sid)},
                   "item_to_shuffled_sid": {str(i): new_sid_of_item[i] for i in sorted(item2sid)},
                   "per_bucket": per_bucket}, f, indent=1)

    def remap(src, dst):
        rows = list(csv.DictReader(open(src, encoding="utf-8")))
        with open(dst, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                n = dict(r)
                n["history_item_sid"] = str([old_sid_to_new[x] for x in
                                             ast.literal_eval(r["history_item_sid"])])
                n["item_sid"] = new_sid_of_item[int(r["item_id"])]
                w.writerow(n)
        return rows

    n_train = len(remap(TRAIN, f"{OUT_DIR}/train.csv"))
    n_valid = len(remap(VALID, f"{OUT_DIR}/valid.csv"))
    n_test = len(remap(TEST, f"{OUT_DIR}/test.csv"))
    print(f"[out] train={n_train} valid={n_valid} test={n_test}")

    # ================================================================ INVARIANTS
    print("\n" + "=" * 104)
    print("INVARIANTS")
    print("=" * 104)
    new_idx = json.load(open(f"{OUT_DIR}/{INDEX_NAME}", encoding="utf-8"))
    new_item2sid = {int(k): "".join(v) for k, v in new_idx.items()}
    new_sid2items = defaultdict(list)
    for i, s in new_item2sid.items():
        new_sid2items[s].append(i)

    check("item count unchanged", len(new_item2sid) == n_items, f"{len(new_item2sid)}")
    check("item identity set unchanged", set(new_item2sid) == set(item2sid))

    o_set, n_set = set(item2sid.values()), set(new_item2sid.values())
    check("exact SID SET identical", o_set == n_set,
          f"|orig|={len(o_set)} |new|={len(n_set)} |sym diff|={len(o_set ^ n_set)}")

    o_ms, n_ms = Counter(item2sid.values()), Counter(new_item2sid.values())
    check("exact SID MULTISET identical (per-string multiplicity)",
          o_ms == n_ms,
          f"distinct={len(o_ms)}; differing keys={len(set(o_ms) ^ set(n_ms)) + sum(1 for k in o_ms if o_ms[k] != n_ms.get(k, 0))}")

    o_groups = {s: sorted(v) for s, v in sid2items.items() if len(v) > 1}
    n_groups = {s: sorted(v) for s, v in new_sid2items.items() if len(v) > 1}
    check("15 collision groups with identical item membership",
          o_groups == n_groups, f"{len(n_groups)} groups identical={o_groups == n_groups}")

    for d in [1, 2, 3]:
        o, n = trie_prefixes(list(item2sid.values()), d), trie_prefixes(list(new_item2sid.values()), d)
        check(f"exact trie structure at depth {d} identical", o == n, f"|prefixes|={len(n)}")

    o_tok = sorted({t for s in item2sid.values() for t in sid_tokens(s)})
    n_tok = sorted({t for s in new_item2sid.values() for t in sid_tokens(s)})
    check("SID token vocabulary identical", o_tok == n_tok, f"{len(n_tok)} tokens")

    for sp, src, dst, n in [("train", TRAIN, f"{OUT_DIR}/train.csv", n_train),
                            ("valid", VALID, f"{OUT_DIR}/valid.csv", n_valid),
                            ("test", TEST, f"{OUT_DIR}/test.csv", n_test)]:
        a, b = (list(csv.DictReader(open(src, encoding="utf-8"))),
                list(csv.DictReader(open(dst, encoding="utf-8"))))
        check(f"{sp}: row count unchanged", len(a) == len(b), f"{len(b)}")
        check(f"{sp}: user_id/item_id/item_title/history_item_id unchanged",
              all(x["user_id"] == y["user_id"] and x["item_id"] == y["item_id"]
                  and x["item_title"] == y["item_title"]
                  and x["history_item_id"] == y["history_item_id"] for x, y in zip(a, b)))

    bad = bad2 = 0
    for src, dst in [(TRAIN, f"{OUT_DIR}/train.csv"), (VALID, f"{OUT_DIR}/valid.csv"),
                     (TEST, f"{OUT_DIR}/test.csv")]:
        for a, b in zip(csv.DictReader(open(src, encoding="utf-8")),
                        csv.DictReader(open(dst, encoding="utf-8"))):
            if b["item_sid"] != new_item2sid[int(b["item_id"])]:
                bad += 1
            exp = [old_sid_to_new[x] for x in ast.literal_eval(a["history_item_sid"])]
            if ast.literal_eval(b["history_item_sid"]) != exp:
                bad2 += 1
    check("all CSV history/target SIDs consistent with the new mapping", bad == 0 and bad2 == 0,
          f"target mismatches={bad} history mismatches={bad2}")

    single_fixed = sum(1 for i in single_items if new_item2sid[i] == item2sid[i])
    check("singleton fixed points = 0", single_fixed == 0,
          f"{single_fixed}/{len(single_items)}")
    coll_changed = sum(1 for i in coll_items if new_item2sid[i] != item2sid[i])
    check("only the 31 collision items keep their original SID", coll_changed == 0,
          f"{coll_changed} collision items changed (expected 0)")

    # ================================================================ INTERVENTION STRENGTH
    print("\n" + "=" * 104)
    print("INTERVENTION STRENGTH")
    print("=" * 104)
    all_items = sorted(item2sid)
    orig_map = item2sid
    rows_is = []
    is_stats = {}

    def add(name, value, extra=None):
        rows_is.append([name, f"{value*100:.4f}%" if isinstance(value, float) else str(value),
                        extra or ""])
        is_stats[name] = value

    for label, subset in [("all items", all_items), ("singleton items only", single_items)]:
        ret = sum(1 for i in subset if new_item2sid[i] == orig_map[i]) / len(subset)
        add(f"exact-SID retention [{label}]", ret, f"n={len(subset)}")

    for d in [1, 2]:
        o_r = sum(1 for i in all_items
                  if sid_tokens(orig_map[i])[:d] == sid_tokens(new_item2sid[i])[:d]) / len(all_items)
        add(f"prefix-{d} retention (self) [all items]", o_r, f"n={len(all_items)}")
        o_rs = sum(1 for i in single_items
                   if sid_tokens(orig_map[i])[:d] == sid_tokens(new_item2sid[i])[:d]) / len(single_items)
        add(f"prefix-{d} retention (self) [singleton items]", o_rs, f"n={len(single_items)}")

    for d in [1, 2]:
        ob, oc, ot = pair_share_rate(orig_map, all_items, d)
        nb, nc, nt = pair_share_rate(new_item2sid, all_items, d)
        add(f"pair prefix-{d} share rate BEFORE [all items]", ob, f"{oc}/{ot} pairs")
        add(f"pair prefix-{d} share rate AFTER  [all items]", nb, f"{nc}/{nt} pairs")
        add(f"pair prefix-{d} RETENTION [all items]", (nb / ob) if ob else 0.0,
            "after/before")
        ob2, oc2, ot2 = pair_share_rate(orig_map, single_items, d)
        nb2, nc2, nt2 = pair_share_rate(new_item2sid, single_items, d)
        add(f"pair prefix-{d} share rate BEFORE [singleton items]", ob2, f"{oc2}/{ot2} pairs")
        add(f"pair prefix-{d} share rate AFTER  [singleton items]", nb2, f"{nc2}/{nt2} pairs")
        add(f"pair prefix-{d} RETENTION [singleton items]", (nb2 / ob2) if ob2 else 0.0,
            "after/before")

    w = [max(len(str(r[i])) for r in rows_is) for i in range(3)]
    print("  " + "metric".ljust(w[0]) + "  " + "value".rjust(12) + "  detail")
    print("  " + "-" * (w[0] + 30))
    for r in rows_is:
        print("  " + str(r[0]).ljust(w[0]) + "  " + str(r[1]).rjust(12) + "  " + str(r[2]))

    report.update({
        "n_items": n_items, "n_singletons": len(single_items),
        "n_collision_items": len(coll_items), "n_collision_sids": len(coll_sids),
        "frozen_collision_items": coll_items,
        "singleton_fixed_points": single_fixed,
        "collision_items_changed": coll_changed,
        "per_bucket": per_bucket,
        "intervention_strength": is_stats,
        "outputs": {k: f"{OUT_DIR}/{v}" for k, v in
                    [("index", INDEX_NAME), ("train", "train.csv"),
                     ("valid", "valid.csv"), ("test", "test.csv"), ("mapping", "mapping.json")]},
        "token_extender_note": (
            "TokenExtender (sft.py:31-39) rebuilds the index filename as "
            "basename(sid_index_path).split('.')[0] + '.index.json'. The index file is "
            "therefore named '" + INDEX_NAME + "' so the rebuild resolves to itself. "
            "data.py opens --sid_index_path directly and is unaffected."),
    })
    n_fail = sum(1 for c in report["checks"] if not c["pass"])
    report["summary"] = {"checks": len(report["checks"]), "failed": n_fail,
                         "problems": len(report["problems"])}
    with open(f"{RES_DIR}/shuffled_sid_preflight.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 104)
    print(f"INVARIANTS: {len(report['checks'])-n_fail}/{len(report['checks'])} PASS, {n_fail} FAIL"
          f" | problems: {len(report['problems'])}")
    print(f"[save] {RES_DIR}/shuffled_sid_preflight.json")
    print("=" * 104)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
