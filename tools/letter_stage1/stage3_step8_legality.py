#!/usr/bin/env python3
"""
Stage 3 / step 8 -- legality / duplication audit of the formal LETTER eval output.

Checks the required protocol properties on all 4533 samples x 20 candidates:
  LegalRate        : every emitted SID is a catalogue SID, 4 tokens, no early EOS,
                     no 5th-level token
  DuplicateRate    : how often the same SID appears twice within one sample's top-20
"""
import ast
import json
import os
import re
import sys

import os as _os
_S = _os.environ.get("LEGALITY_STAGE", "3")
if _S == "3":
    RESULT = "runs/letter_sid_qwen05b_eval/letter_result_Industrial_and_Scientific.json"
    INFO = "artifacts/letter_stage3/data/info/Industrial_and_Scientific_5_2016-10-2018-11.txt"
    TEST = "artifacts/letter_stage3/data/test/Industrial_and_Scientific_5_2016-10-2018-11.csv"
else:
    RESULT = "runs/content_only_sid_qwen05b_eval/content_only_result_Industrial_and_Scientific.json"
    INFO = "artifacts/letter_stage4_content_only/data/info/Industrial_and_Scientific_5_2016-10-2018-11.txt"
    TEST = "artifacts/letter_stage4_content_only/data/test/Industrial_and_Scientific_5_2016-10-2018-11.csv"
LEVELS = "abcd"


def main():
    print("=" * 84)
    print("STEP 8 -- LEGALITY / DUPLICATION AUDIT (LETTER beam20, 4533 samples)")
    print("=" * 84)

    catalogue = set()
    with open(INFO, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                catalogue.add(line.split("\t")[0].strip())
    print(f"  catalogue SIDs = {len(catalogue)}")

    with open(TEST, encoding="utf-8", newline="") as f:
        import csv
        rr = csv.reader(f)
        hdr = next(rr)
        test_rows = [r for r in rr]
    targets = [r[hdr.index("item_sid")] for r in test_rows]
    print(f"  test rows      = {len(test_rows)}")

    d = json.load(open(RESULT, encoding="utf-8"))
    print(f"  result samples = {len(d)}")
    print(f"  keys           = {list(d[0].keys())}")

    n_pred = 0
    bad_depth, bad_legal, bad_5th, bad_empty = 0, 0, 0, 0
    dup_in_sample = 0
    dup_pairs_total = 0
    cand_counts = []
    hit1 = 0

    for i, s in enumerate(d):
        preds = s["predict"]
        cand_counts.append(len(preds))
        seen = set()
        for p in preds:
            n_pred += 1
            p = p.strip()
            if not p:
                bad_empty += 1
                continue
            # token-level structure
            levels = re.findall(r"<([a-z])_(\d+)>", p)
            letters = [lv[0] for lv in levels]
            if len(levels) != 4:
                bad_depth += 1
            if len(levels) > 4 or any(l not in LEVELS for l in letters):
                bad_5th += 1
            if p not in catalogue:
                bad_legal += 1
            if p in seen:
                dup_pairs_total += 1
            seen.add(p)
        if len(seen) != len(preds):
            dup_in_sample += 1
        if preds and preds[0].strip() == targets[i].strip():
            hit1 += 1

    print()
    print(f"  total candidates           = {n_pred}")
    print(f"  candidates per sample      = {sorted(set(cand_counts))}")
    print()
    print(f"  dup pairs within sample    = {dup_pairs_total}")
    print(f"  samples with any duplicate = {dup_in_sample}")
    print(f"  bad token depth (!= 4)     = {bad_depth}")
    print(f"  bad 5th-level / bad letter = {bad_5th}")
    print(f"  empty predictions          = {bad_empty}")
    print(f"  non-catalogue predictions  = {bad_legal}")
    print()
    legal_rate = 100.0 * (n_pred - bad_legal - bad_depth - bad_5th - bad_empty) / n_pred
    dup_rate = 100.0 * dup_pairs_total / n_pred
    print(f"  LegalRate       = {legal_rate:.4f}%   "
          f"({n_pred - bad_legal - bad_depth - bad_5th - bad_empty}/{n_pred})")
    print(f"  DuplicateRate   = {dup_rate:.4f}%   (duplicate candidates / all candidates)")
    print(f"  top-1 exact hits (sanity)  = {hit1}/{len(d)} "
          f"({100.0*hit1/len(d):.4f}%)")

    fails = []
    if bad_legal or bad_depth or bad_5th or bad_empty:
        fails.append("legality")
    if dup_pairs_total:
        fails.append("duplicates")
    print()
    print(f"  failures = {fails}")
    print(f"  RESULT: {'ALL PASS' if not fails else 'FAIL'}")

    json.dump({"n_samples": len(d), "n_candidates": n_pred,
               "legal_rate": legal_rate, "duplicate_rate": dup_rate,
               "bad_depth": bad_depth, "bad_5th_level": bad_5th,
               "bad_legal": bad_legal, "bad_empty": bad_empty,
               "dup_pairs_total": dup_pairs_total,
               "samples_with_duplicate": dup_in_sample,
               "top1_exact_hits": hit1},
              open(_os.environ.get("LEGALITY_OUT", "artifacts/letter_stage3/eval_legality_audit.json"), "w"), indent=2)
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
