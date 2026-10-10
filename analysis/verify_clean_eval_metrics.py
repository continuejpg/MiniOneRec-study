#!/usr/bin/env python3
"""
Independent metric reproduction for the eval audit.

Runs the ACTUAL calc.py code (imported, not transcribed) on the clean-SFT
prediction file, so the reproduction is independent of any re-implementation.
Also prints the protocol facts an auditor needs.
"""
import importlib.util
import io
import json
import math
import os
import sys
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import CATEGORY, INFO, run, repo  # noqa: E402

CAT = CATEGORY
PRED = run("eval_clean_sft", "test_beam20.json")
INFO = INFO

TARGET_HR20 = 0.19832341
TARGET_NDCG20 = 0.11786798

# ---- import the real calc.py -------------------------------------------------
spec = importlib.util.spec_from_file_location("calc_real", repo("calc.py"))
calc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calc)

print("=" * 96)
print("INDEPENDENT REPRODUCTION USING THE ACTUAL calc.py (imported, unmodified)")
print("=" * 96)
print(f"  calc.py      : {repo('calc.py')}")
print(f"  predictions  : {PRED}")
print(f"  item_path    : {INFO}")

buf = io.StringIO()
try:
    with redirect_stdout(buf):
        calc.gao(path=PRED, item_path=INFO)
    out = buf.getvalue()
except Exception as e:
    print(f"\n  calc.gao raised {type(e).__name__}: {e}")
    out = buf.getvalue()

print("\n--- raw stdout of calc.gao ---")
for line in out.splitlines():
    if line.strip() and not line.startswith("100%") and "it/s" not in line:
        print("   " + line)

# ---- parse the emitted arrays ------------------------------------------------
import re
hr = ndcg = None
for line in out.splitlines():
    if line.startswith("NDCG:"):
        ndcg = [float(x) for x in re.findall(r"[0-9.]+", line.split("\t")[-1])]
    elif line.startswith("HR\t") or line.startswith("HR "):
        hr = [float(x) for x in re.findall(r"[0-9.]+", line.split("\t")[-1])]

print("\n--- parsed ---")
print(f"  HR   = {hr}")
print(f"  NDCG = {ndcg}")

if hr and ndcg:
    print("\n--- gate ---")
    e1 = abs(hr[-1] - TARGET_HR20)
    e2 = abs(ndcg[-1] - TARGET_NDCG20)
    print(f"  HR@20   = {hr[-1]:.12f}  target {TARGET_HR20:.12f}  |diff| = {e1:.3e}")
    print(f"  NDCG@20 = {ndcg[-1]:.12f}  target {TARGET_NDCG20:.12f}  |diff| = {e2:.3e}")
    ok = e1 < 1e-8 and e2 < 1e-8
    print(f"\n  GATE: {'PASS' if ok else 'FAIL'}")
else:
    print("\n  *** could not parse calc.py output ***")
    ok = False

# ---- protocol facts ---------------------------------------------------------
print("\n" + "=" * 96)
print("PROTOCOL FACTS RECORDED IN THE ARTEFACT")
print("=" * 96)
d = json.load(open(PRED, encoding="utf-8"))
print(f"  samples                 : {len(d)}")
print(f"  candidates per sample   : {sorted({len(x['predict']) for x in d})}")
print(f"  target convention       : sample['output'].strip() == target SID")
print(f"  output ends with newline: {all(x['output'].endswith(chr(10)) for x in d)}")
print(f"  duplicate candidates    : "
      f"{sum(1 for x in d if len(set(x['predict'])) < len(x['predict']))}")

sys.exit(0 if ok else 1)
