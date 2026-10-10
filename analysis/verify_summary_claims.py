#!/usr/bin/env python3
"""
Verification for the experiment summary rewrite. Read-only.

V1. The 317 test rows whose TARGET is unchanged by the shuffle: are they exactly
    the rows whose target item is one of the 31 intentionally frozen
    collision-involved items?
V2. The 54 rows whose INPUT (prompt) is unchanged: same question.
V3. Machine-check every headline HR@20 / NDCG@20 against the prediction artifacts.
"""
import csv
import importlib.util
import io
import json
import os
import sys
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import CATEGORY, INFO, RUN_ROOT, SHUFFLED_DIR, SHUFFLED_TEST, TEST, repo, run  # noqa: E402

CAT = CATEGORY
RUNS = RUN_ROOT

CLEAN_PRED = run("eval_clean_sft", "test_beam20.json")
SHUF_PRED = run("eval_shuffled_sid", "test_beam20.json")
CLEAN_TEST = TEST
SHUF_TEST = SHUFFLED_TEST
MAPPING = os.path.join(SHUFFLED_DIR, "mapping.json")
CLEAN_INFO = INFO

print("=" * 100)
print("V1/V2. ARE THE UNCHANGED ROWS THE FROZEN COLLISION ITEMS?")
print("=" * 100)

clean = json.load(open(CLEAN_PRED, encoding="utf-8"))
shuf = json.load(open(SHUF_PRED, encoding="utf-8"))
ct = list(csv.DictReader(open(CLEAN_TEST, encoding="utf-8")))
st = list(csv.DictReader(open(SHUF_TEST, encoding="utf-8")))
mapping = json.load(open(MAPPING, encoding="utf-8"))
frozen = {str(i) for i in mapping["frozen_collision_items"]}

assert len(clean) == len(shuf) == len(ct) == len(st) == 4533

# row alignment: user_id and item_id must be identical between the two splits
align = sum(1 for a, b in zip(ct, st)
            if a["user_id"] == b["user_id"] and a["item_id"] == b["item_id"])
print(f"\nrow alignment (user_id + item_id identical): {align}/4533")

same_input = [i for i in range(4533) if clean[i]["input"] == shuf[i]["input"]]
same_output = [i for i in range(4533) if clean[i]["output"] == shuf[i]["output"]]
print(f"unchanged INPUT  rows: {len(same_input)}  -> prompt changed "
      f"{(4533-len(same_input))/4533*100:.2f}%")
print(f"unchanged OUTPUT rows: {len(same_output)} -> target changed "
      f"{(4533-len(same_output))/4533*100:.2f}%")

frozen_rows = {i for i in range(4533) if ct[i]["item_id"] in frozen}
print(f"\ntest rows whose target item_id is a frozen collision item: {len(frozen_rows)}")

for label, idxs in [("unchanged OUTPUT", same_output), ("unchanged INPUT", same_input)]:
    S = set(idxs)
    print(f"\n--- {label}: {len(S)} rows ---")
    print(f"    S subset of frozen_rows            : {S <= frozen_rows}")
    print(f"    frozen_rows subset of S            : {frozen_rows <= S}")
    print(f"    S == frozen_rows                   : {S == frozen_rows}")
    print(f"    |S ∩ frozen_rows|                  : {len(S & frozen_rows)}")
    print(f"    |S \\ frozen_rows| (NOT frozen)     : {len(S - frozen_rows)}")
    if S - frozen_rows:
        ex = sorted(S - frozen_rows)[:8]
        print(f"      examples (row, item_id, clean target, shuffled target):")
        for i in ex:
            print(f"        row {i:5d}  item {ct[i]['item_id']:>5s}  "
                  f"{ct[i]['item_sid']}  ->  {st[i]['item_sid']}")

# independent check: is the target SID unchanged iff the target item is frozen?
pred_unchanged = {i for i in range(4533) if ct[i]["item_sid"] == st[i]["item_sid"]}
print(f"\nrows where the target SID string is unchanged: {len(pred_unchanged)}")
print(f"    equals frozen_rows : {pred_unchanged == frozen_rows}")
print(f"    equals same_output : {pred_unchanged == set(same_output)}")
print(f"    |pred_unchanged \\ frozen_rows| : {len(pred_unchanged - frozen_rows)}")

# history: how many rows have an unchanged history?
same_hist = sum(1 for a, b in zip(ct, st)
                if a["history_item_sid"] == b["history_item_sid"])
print(f"\nrows with unchanged history_item_sid: {same_hist}")

print("\n" + "=" * 100)
print("V3. MACHINE CHECK OF EVERY HEADLINE METRIC AGAINST ITS ARTIFACT")
print("=" * 100)

spec = importlib.util.spec_from_file_location("calc_real", repo("calc.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def grid(pred, info):
    buf = io.StringIO()
    with redirect_stdout(buf):
        mod.gao(path=pred, item_path=info)
    out = buf.getvalue()
    hr = ndcg = None
    for line in out.splitlines():
        if line.startswith("NDCG:"):
            ndcg = [float(x) for x in line.split("\t")[-1].strip().strip("[]").split()]
        elif line.startswith("HR"):
            hr = [float(x) for x in line.split("\t")[-1].strip().strip("[]").split()]
    return hr, ndcg


SOURCES = [
    ("Clean SFT", CLEAN_PRED, CLEAN_INFO, "clean",
     (0.19832341, 0.11786798)),
    ("GRPO 0.25ep original", f"{RUNS}/grpo_short025/eval_beam20.json", CLEAN_INFO, "clean",
     (0.17626296, 0.10885475)),
    ("GRPO 0.25ep optimized", f"{RUNS}/grpo_fast025/eval_beam20.json", CLEAN_INFO, "clean",
     (0.17538054, 0.10824989)),
    ("GRPO 1.5ep", f"{RUNS}/eval_grpo_step26274/test_beam20.json", CLEAN_INFO, "clean",
     (0.16170307, 0.10470676)),
    ("GRPO 2.0ep", f"{RUNS}/eval_grpo_baseline/test_beam20.json", CLEAN_INFO, "clean",
     (0.16236488, 0.10447064)),
    ("Shuffled-SID SFT", SHUF_PRED, CLEAN_INFO, "shuffled",
     (0.10765497, 0.08157358)),
    ("LEGACY eval_industrial", f"{RUNS}/eval_industrial/final_result_{CAT}.json",
     CLEAN_INFO, "clean", None),
]

print(f"\n{'run':24s} {'split':9s} {'HR@20':>12s} {'NDCG@20':>12s} {'artifact HR':>12s} "
      f"{'artifact NDCG':>13s}  verdict")
ok_all = True
for name, pred, info, split, exp in SOURCES:
    hr, ndcg = grid(pred, info)
    if exp is None:
        print(f"{name:24s} {split:9s} {hr[-1]:>12.8f} {ndcg[-1]:>12.8f} "
              f"{'-':>12s} {'-':>13s}  (legacy, no official reference)")
        continue
    d1, d2 = abs(hr[-1] - exp[0]), abs(ndcg[-1] - exp[1])
    v = "MATCH" if (d1 < 1e-8 and d2 < 1e-8) else f"MISMATCH {d1:.1e}/{d2:.1e}"
    if v != "MATCH":
        ok_all = False
    print(f"{name:24s} {split:9s} {hr[-1]:>12.8f} {ndcg[-1]:>12.8f} "
          f"{exp[0]:>12.8f} {exp[1]:>13.8f}  {v}")
    print(f"{'':24s} full HR   = {hr}")
    print(f"{'':24s} full NDCG = {ndcg}")

print(f"\nALL HEADLINE METRICS MATCH: {ok_all}")
sys.exit(0 if ok_all else 1)
