#!/usr/bin/env python3
"""
audit_predictions.py -- read-only audit of every beam-20 prediction file.

For each candidate prediction file:
  - score it against the split it was actually evaluated on (the clean test CSV
    for every historical run; the SHUFFLED test CSV for the shuffled-SID arm,
    whose target SIDs are the shuffled ones)
  - compute the HR@1/3/5/10/20 and NDCG@20 grid with the SAME first-exact-SID-match
    rule as calc.py
  - print path, n, beam width
  - match against the known official metrics to identify which run it came from

Note on the shuffled-SID arm (runs/eval_shuffled_sid/test_beam20.json):
  its model was trained with a 2-epoch budget and completed all 2496 steps, but the
  run was INTERRUPTED after step 2180 and RESUMED from checkpoint-2125, so steps
  2125-2180 were retrained. The `train_loss` printed by the resume segment
  (0.0600) is a resumed-segment statistic and is NOT the full-training loss; the
  all-step mean is 0.8027 (clean SFT: 0.7712).

Prints what is found. Trains nothing. Modifies no existing file.
"""
import ast
import csv
import glob
import json
import math
import os

ROOT = "/root/autodl-tmp"
RUNS = f"{ROOT}/runs"
CODE = f"{ROOT}/code"
CAT = "Industrial_and_Scientific"
TEST_CSV = f"{CODE}/data/Amazon/test/{CAT}_5_2016-10-2018-11.csv"
# The shuffled-SID arm is scored against the SHUFFLED test split: its target SIDs
# are the shuffled ones, so the clean targets would be meaningless.
SHUF_TEST_CSV = f"{CODE}/analysis/shuffled_sid/test.csv"

# Known official results (ground truth). Historical labels are unchanged; the
# shuffled-SID arm was appended on 2026-10-05 and its `test_csv` is overridden
# because it is evaluated on the shuffled split.
KNOWN = [
    ("SFT                      ", 0.19832341, 0.11786798),
    ("GRPO 1.5ep               ", 0.16170307, 0.10470676),
    ("GRPO 2.0ep               ", 0.16236488, 0.10447064),
    ("GRPO original 0.25ep     ", 0.17626296, 0.10885475),
    ("GRPO fast 0.25ep         ", 0.17538054, 0.10824989),
    ("Shuffled-SID SFT         ", 0.10765497, 0.08157358),
]

# per-artefact target split overrides
TEST_CSV_OVERRIDE = {
    f"{RUNS}/eval_shuffled_sid/test_beam20.json": SHUF_TEST_CSV,
}

TOPK = [1, 3, 5, 10, 20]


def ncdcg(rank):
    return 1.0 / math.log2(rank + 2)


def metrics(preds, targets):
    hr = {k: 0 for k in TOPK}
    nd = 0.0
    for p, t in zip(preds, targets):
        r = -1
        for i, x in enumerate(p):
            if x == t:
                r = i
                break
        for k in TOPK:
            if 0 <= r < k:
                hr[k] += 1
        if 0 <= r < 20:
            nd += ncdcg(r)
    n = len(targets)
    return {f"HR@{k}": hr[k] / n for k in TOPK}, nd / n


def main():
    with open(TEST_CSV, encoding="utf-8") as f:
        test = list(csv.DictReader(f))
    clean_targets = [r["item_sid"] for r in test]
    print(f"clean test rows = {len(test)}")

    with open(SHUF_TEST_CSV, encoding="utf-8") as f:
        shuf_test = list(csv.DictReader(f))
    shuf_targets = [r["item_sid"] for r in shuf_test]
    print(f"shuffled test rows = {len(shuf_test)}")

    # ---------------------------------------------------------------- discover
    pats = [
        f"{RUNS}/*/test_beam20.json",
        f"{RUNS}/*/eval_beam20.json",
        f"{RUNS}/*/*/test_beam20.json",
        f"{RUNS}/eval_*/final_result_*.json",
    ]
    found = []
    for p in pats:
        found += glob.glob(p)
    found = sorted(set(found))

    print(f"\ndiscovered {len(found)} candidate prediction files\n")
    rows = []
    for p in found:
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception as e:
            print(f"  SKIP {p}: {type(e).__name__}")
            continue
        if not isinstance(d, list) or not d or "predict" not in d[0]:
            print(f"  SKIP {p}: no 'predict' field")
            continue
        preds = [x["predict"] for x in d]
        n = len(preds)
        beam = len(preds[0]) if n else 0

        # pick the split this artefact was scored against
        tgt_csv = TEST_CSV_OVERRIDE.get(p, TEST_CSV)
        targets = shuf_targets if tgt_csv == SHUF_TEST_CSV else clean_targets
        split = "shuffled" if tgt_csv == SHUF_TEST_CSV else "clean"

        note = f"split={split}"
        if n != len(targets):
            note += f"  !! n={n} != split {len(targets)}"
        hr, nd = metrics(preds, targets)

        # closest known match
        best, bestd = None, 1e9
        for label, khr, knd in KNOWN:
            dist = abs(hr["HR@20"] - khr) + abs(nd - knd)
            if dist < bestd:
                best, bestd = label.strip(), dist
        match = (f"{best}  (d={bestd:.2e})" if bestd < 5e-4
                 else f"(no match, closest {best} d={bestd:.2e})")

        rows.append((os.path.relpath(p, ROOT), n, beam,
                     hr["HR@1"], hr["HR@3"], hr["HR@5"], hr["HR@10"], hr["HR@20"],
                     nd, match, note))

    print("=" * 150)
    print("PER-FILE RECOMPUTED GLOBAL METRICS (scored against each artefact's own split)")
    print("=" * 150)
    for r in rows:
        path, n, beam, h1, h3, h5, h10, h20, nd, match, note = r
        print(f"\npath      : {path}")
        print(f"n / beam  : {n} / {beam}   {note}")
        print(f"HR@1      : {h1*100:.5f}%")
        print(f"HR@3      : {h3*100:.5f}%")
        print(f"HR@5      : {h5*100:.5f}%")
        print(f"HR@10     : {h10*100:.5f}%")
        print(f"HR@20     : {h20*100:.5f}%   (as fraction {h20:.8f})")
        print(f"NDCG@20   : {nd*100:.5f}%   (as fraction {nd:.8f})")
        print(f"identity  : {match}")

    print()
    print("=" * 150)
    print("OFFICIAL REFERENCE")
    print("=" * 150)
    for label, khr, knd in KNOWN:
        print(f"  {label} HR@20={khr:.8f}  NDCG@20={knd:.8f}")

    # ---------------------------------------------------------------- checkpoint inventory
    print()
    print("=" * 150)
    print("CHECKPOINT / ARTIFACT INVENTORY (to establish epoch identity)")
    print("=" * 150)
    for d in sorted(glob.glob(f"{RUNS}/*/")):
        name = os.path.basename(d.rstrip("/"))
        ck = sorted(glob.glob(f"{d}checkpoint-*"))
        has_final = os.path.isdir(f"{d}final_checkpoint")
        logs = glob.glob(f"{d}*.log")
        extra = [os.path.basename(x) for x in glob.glob(f"{d}*.json") if "tokenizer" not in x
                 and os.path.basename(x) not in ("config.json", "generation_config.json")]
        print(f"\n{name}/")
        print(f"   checkpoint-*      : {[os.path.basename(c) for c in ck] or '-'}")
        print(f"   final_checkpoint  : {has_final}")
        print(f"   logs              : {[os.path.basename(l) for l in logs] or '-'}")
        print(f"   json              : {extra or '-'}")


if __name__ == "__main__":
    main()
