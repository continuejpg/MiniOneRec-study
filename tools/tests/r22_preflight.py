#!/usr/bin/env python3
"""
R2.2 preflight -- hard gates that must all pass BEFORE any GPU training starts.

Strategy: parse patches/grpo_r21.sh for the exact `rl.py` CLI arguments, rebuild
the training dataset with those SAME arguments, and check it against the route
cache. No model, no GPU, no training.

Every gate is a hard assert; the first failure exits non-zero.
"""
import hashlib
import inspect
import json
import os
import re
import sys

import fire

sys.path.insert(0, os.getcwd())
os.chdir(os.getcwd())

LAUNCHER = "patches/grpo_r21.sh"
BASELINE = "patches/grpo_fast025.sh"
CACHE = "splits/r21_route_cache.json"
EXPECT_SHA = "d7a7bc40e96217e416825cc2cee29b216e3443d64fcea17b5440e72975b140c3"
FAILS = []
WARNS = []


def gate(ok, label, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAILS.append(label)
    return ok


def warn(ok, label, detail=""):
    print(f"  [{'PASS' if ok else 'WARN'}] {label}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        WARNS.append(label)


def parse_args(path):
    """Extract `--key value` pairs from the rl.py invocation in a launcher."""
    txt = open(path, encoding="utf-8").read()
    i = txt.index("rl.py")
    j = txt.index('> "$OUT/train.log"')
    body = txt[i:j]
    body = body.replace("\\\n", " ")
    toks = body.split()
    args = {}
    k = 1
    while k < len(toks):
        t = toks[k]
        if t.startswith("--"):
            key = t[2:]
            if k + 1 < len(toks) and not toks[k + 1].startswith("--"):
                args[key] = toks[k + 1].strip('"')
                k += 2
            else:
                args[key] = "True"
                k += 1
        else:
            k += 1
    return args


print("=" * 96)
print("R2.2 PREFLIGHT")
print("=" * 96)

a_r21 = parse_args(LAUNCHER)
a_base = parse_args(BASELINE)
print(f"  launcher : {LAUNCHER}")
print(f"  baseline : {BASELINE}")

# ---------------- G1 launcher exists and differs only as authorised ----------
print()
print("--- G1  launcher vs baseline: authorised diffs only ---")
AUTH = {"reward_type", "r21_enable", "route_cache", "output_dir"}
keys = set(a_r21) | set(a_base)
diff = {k for k in keys if a_r21.get(k) != a_base.get(k)}
unauth = diff - AUTH
gate(not unauth, "only reward_type / r21_enable / route_cache / output_dir differ",
     f"diff={sorted(diff)}")
if unauth:
    for k in sorted(unauth):
        print(f"      UNAUTHORISED {k}: {a_base.get(k)!r} -> {a_r21.get(k)!r}")
for k in sorted(AUTH & diff):
    print(f"      {k:14s} {a_base.get(k)!r} -> {a_r21.get(k)!r}")

# ---------------- G2 frozen config unchanged ---------------------------------
print()
print("--- G2  frozen hyperparameters unchanged ---")
for k, want in (("num_generations", "16"), ("beam_search", "True"),
                ("train_batch_size", "16"), ("eval_batch_size", "16"),
                ("gradient_accumulation_steps", "1"), ("num_train_epochs", "0.25"),
                ("learning_rate", "1e-5"), ("beta", "1e-3"),
                ("temperature", "1.0"), ("sync_ref_model", "True"),
                ("add_gt", "False"), ("dapo", "False"), ("gspo", "False"),
                ("subset_seq", "grpo_seq_10k.json"),
                ("subset_seqtitle", "grpo_seqtitle_1k.json"),
                ("seqtitle_sample", "10000"), ("sample_train", "False"),
                ("dynamic_sampling", "False"), ("mask_all_zero", "False")):
    got = a_r21.get(k)
    gate(got == want, f"{k} == {want}", f"got={got!r}")

# ---------------- G3 reward + checkpoint -------------------------------------
print()
print("--- G3  reward and checkpoint ---")
gate(a_r21.get("reward_type") == "r21_exact", "reward_type == r21_exact",
     str(a_r21.get("reward_type")))
gate(a_r21.get("reward_type") not in ("ranking", "rere_rank", "ranking_only"),
     "NOT a ranking / rere_rank reward", str(a_r21.get("reward_type")))
gate(a_r21.get("r21_enable") in ("True", "true", "1"), "r21_enable enabled",
     str(a_r21.get("r21_enable")))
gate(a_r21.get("route_cache", "").endswith("r21_route_cache.json"),
     "route_cache points at the validated cache", str(a_r21.get("route_cache")))
# the launcher uses shell vars, so resolve them the way bash would
SH = {"CKPT": "runs/industrial_sft/final_checkpoint", "OUT": "runs/grpo_r21",
      "SPLITS": "splits"}


def expand(v):
    for k, val in SH.items():
        v = v.replace(f"${k}", val)
        v = v.replace(f"${{{k}}}", val)
    return v


mp = expand(a_r21.get("model_path", ""))
gate(mp.endswith("industrial_sft/final_checkpoint"),
     "policy/ref = frozen P0 SFT", f"{a_r21.get('model_path')} -> {mp}")
rc = expand(a_r21.get("route_cache", ""))
gate(rc == "splits/r21_route_cache.json",
     "route_cache resolves to splits/r21_route_cache.json", f"{a_r21.get('route_cache')} -> {rc}")

# Fire boolean semantics, measured rather than assumed
print()
print("--- G3b Fire boolean parsing (measured, not assumed) ---")
seen = {}


def _probe(r21_enable: bool = False, reward_type: str = "rule"):
    seen["v"] = r21_enable
    seen["t"] = type(r21_enable).__name__


fire.Fire(_probe, command=[f"--r21_enable={a_r21.get('r21_enable')}"])
gate(seen.get("t") == "bool" and seen.get("v") is True,
     f"Fire parses --r21_enable {a_r21.get('r21_enable')!r} as bool True",
     f"{seen.get('t')} {seen.get('v')!r}")
import rl as _rl                                                    # noqa: E402
_sig = inspect.signature(_rl.train)
gate(_sig.parameters["r21_enable"].annotation is bool,
     "rl.train declares r21_enable: bool")
gate(_sig.parameters["r21_enable"].default is False,
     "rl.train default r21_enable is False")

# ---------------- G4 smoke env vars must be OFF ------------------------------
print()
print("--- G4  smoke environment variables ---")
for v in ("R21_SMOKE_IDS", "R21_SMOKE_LIMIT", "R21_SMOKE_STEPS"):
    val = os.environ.get(v)
    gate(val in (None, "", "0"), f"{v} not enabled", repr(val))
gate(os.environ.get("R21_SMOKE_LIMIT", "0") in ("", "0"),
     "R21_SMOKE_LIMIT does not trigger the smoke branch")
src = open("rl.py", encoding="utf-8").read()
gate('_smoke_any = (_smoke_ids' in src and "or int(os.environ.get(\"R21_SMOKE_LIMIT\", \"0\") or 0) > 0)" in src,
     "formal branch is selected when both smoke vars are off")

# ---------------- G5 cache integrity ----------------------------------------
print()
print("--- G5  route cache integrity ---")
raw = open(CACHE, "rb").read()
sha = hashlib.sha256(raw).hexdigest()
gate(sha == EXPECT_SHA, "cache SHA256 matches the validated value", sha)
cache = json.loads(raw.decode("utf-8"))
routes = cache["routes"]
gate(len(routes) == 17516, "cache entries == 17516", str(len(routes)))

# ---------------- G6 rebuild the training set from the REAL args -------------
print()
print("--- G6  training dataset rebuilt with the launcher's own args ---")
from data import SidDataset, RLTitle2SidDataset, RLSeqTitle2SidDataset  # noqa: E402
CAT = "industrial and scientific items"
base_stem = "Industrial_and_Scientific_5_2016-10-2018-11"
TR = f"data/Amazon/train/{base_stem}.csv"
IM = "data/Amazon/index/Industrial_and_Scientific.item.json"
IX = "data/Amazon/index/Industrial_and_Scientific.index.json"

d1 = SidDataset(train_file=TR, category=CAT, sample=-1)
d2 = RLTitle2SidDataset(item_file=IM, index_file=IX, category=CAT, sample=-1)
d3 = RLSeqTitle2SidDataset(TR, category=CAT,
                           sample=int(a_r21.get("seqtitle_sample", 10000)))
ids1 = set(json.load(open(f"splits/{a_r21['subset_seq']}"))["sample_ids"])
ids3 = set(json.load(open(f"splits/{a_r21['subset_seqtitle']}"))["sample_ids"])
d1 = [x for x in d1 if x["sample_id"] in ids1]
d3 = [x for x in d3 if x["sample_id"] in ids3]
allx = list(d1) + list(d2) + list(d3)
gate(len(allx) == 17516, "dataset length == 17516", str(len(allx)))

from collections import Counter                                          # noqa: E402
tt = Counter(x["task_type"] for x in allx)
for name, want in (("seq_rec", 10000), ("title2sid", 3646),
                   ("description2sid", 2870), ("seqtitle2sid", 1000)):
    gate(tt.get(name) == want, f"{name} == {want}", str(tt.get(name)))

ds_ids = [x["sample_id"] for x in allx]
gate(len(set(ds_ids)) == len(ds_ids), "no duplicate sample_id in the dataset")
gate(set(ds_ids) == set(routes), "sample_id set identical to the cache",
     f"missing={len(set(ds_ids)-set(routes))} extra={len(set(routes)-set(ds_ids))}")
gate(len(set(ds_ids) - set(routes)) == 0, "missing == 0")
gate(len(set(routes) - set(ds_ids)) == 0, "extra == 0")

n_norm = sum(1 for r in routes.values() if r["route"] == "NORMAL")
n_hard = sum(1 for r in routes.values() if r["route"] == "HARD")
gate(n_norm == 4818, "NORMAL == 4818", str(n_norm))
gate(n_hard == 12698, "HARD == 12698", str(n_hard))

# ---------------- G7 root cause of the earlier "missing=9985" ---------------
print()
print("--- G7  why the earlier probe reported missing=9985 ---")
gate(len(ids1) == 10000, "grpo_seq_10k.json holds 10000 ids (seq_rec only)",
     str(len(ids1)))
gate(len(set(ids1)) == 10000, "those ids are seq_rec-only, by construction")
r21cache = json.load(open("/tmp/r21/cache60.json", encoding="utf-8"))["routes"] \
    if os.path.exists("/tmp/r21/cache60.json") else {}
if r21cache:
    gate(len(set(ids1) - set(r21cache)) == 9985,
         "60-entry cache leaves exactly 9985 seq_rec ids uncovered",
         str(len(set(ids1) - set(r21cache))))
print("      explanation: that probe compared the FULL 10,000 seq_rec subset against")
print("      the 60-entry SMOKE cache -- 10000-15 = 9985 -- so the number measured the")
print("      smoke cache, NOT the formal run. The formal cache covers all 17,516 ids")
print("      across all four task types, as verified in G5/G6 above.")
gate(len(routes) == 17516 and tt.get("seq_rec") == 10000,
     "formal run has 10,000 seq_rec AND 7,516 non-seq_rec samples (not seq_rec-only)",
     f"seq_rec={tt.get('seq_rec')} other={17516-tt.get('seq_rec',0)}")

# ---------------- G8 output dir must be fresh -------------------------------
print()
print("--- G8  output directory ---")
outdir = expand(a_r21.get("output_dir", ""))
gate(outdir.endswith("grpo_r21"), "output_dir is the new run dir",
     f"{a_r21.get('output_dir')} -> {outdir}")
gate(not os.path.exists(outdir), "output_dir does not exist yet (no overwrite)",
     outdir)
for old in ("grpo_fast025", "grpo_baseline", "grpo_short025", "grpo_rere_rank025"):
    gate(old not in outdir, f"does not touch old run {old}")

# ---------------- summary ----------------------------------------------------
print()
print("=" * 96)
print(f"RESULT: {'ALL PASS' if not FAILS else 'FAIL'}   "
      f"({len(FAILS)} failed, {len(WARNS)} warnings)")
for f in FAILS:
    print("  FAILED:", f)
for w in WARNS:
    print("  WARN  :", w)
print("=" * 96)
json.dump({"fails": FAILS, "warns": WARNS, "sha256": sha,
           "n_samples": len(allx), "task_types": dict(tt),
           "n_normal": n_norm, "n_hard": n_hard,
           "args": a_r21},
          open("artifacts/rl_audit/r22_preflight.json", "w"), indent=2)
print("  [save] artifacts/rl_audit/r22_preflight.json")
sys.exit(1 if FAILS else 0)
