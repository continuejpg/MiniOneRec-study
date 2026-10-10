#!/usr/bin/env python3
"""
R2.1g -- strict validation of the full route cache.

Verifies, against the authoritative training-set construction in rl.py:
  * 17516 entries, sample_id set IDENTICAL to the training set (no dup/missing/extra)
  * all four task_types covered
  * route fields valid; only h in {0,1}; NORMAL<->h=0, HARD<->h=1
  * HARD hint == first SID token of that sample's ground-truth SID
  * per-task route ratios, SHA256
"""
import collections
import hashlib
import json
import os
import sys

sys.path.insert(0, os.getcwd())
os.chdir(os.getcwd())
from data import SidDataset, RLTitle2SidDataset, RLSeqTitle2SidDataset  # noqa: E402

CAT = "industrial and scientific items"
TR = "data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv"
IM = "data/Amazon/index/Industrial_and_Scientific.item.json"
IX = "data/Amazon/index/Industrial_and_Scientific.index.json"
CACHE = "splits/r21_route_cache.json"
FAILS = []


def ck(ok, label, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAILS.append(label)


print("=" * 92)
print("R2.1g FULL ROUTE CACHE VALIDATION")
print("=" * 92)

# ---- rebuild the exact training sample_id -> (task_type, GT) mapping ----
d1 = SidDataset(train_file=TR, category=CAT, sample=-1)
d2 = RLTitle2SidDataset(item_file=IM, index_file=IX, category=CAT, sample=-1)
d3 = RLSeqTitle2SidDataset(TR, category=CAT, sample=10000)
ids1 = set(json.load(open("splits/grpo_seq_10k.json"))["sample_ids"])
ids3 = set(json.load(open("splits/grpo_seqtitle_1k.json"))["sample_ids"])
d1 = [x for x in d1 if x["sample_id"] in ids1]
d3 = [x for x in d3 if x["sample_id"] in ids3]

gt_task = {}
for x in list(d1) + list(d2) + list(d3):
    gt_task[x["sample_id"]] = (x["task_type"], x["completion"])
print(f"  training set rebuilt: {len(gt_task)} samples")
print(f"    seq_rec        : {sum(1 for v in gt_task.values() if v[0]=='seq_rec')}")
print(f"    title2sid      : {sum(1 for v in gt_task.values() if v[0]=='title2sid')}")
print(f"    description2sid: {sum(1 for v in gt_task.values() if v[0]=='description2sid')}")
print(f"    seqtitle2sid   : {sum(1 for v in gt_task.values() if v[0]=='seqtitle2sid')}")

raw = open(CACHE, "rb").read()
sha = hashlib.sha256(raw).hexdigest()
cache = json.loads(raw.decode("utf-8"))
routes = cache["routes"]
print(f"\n  cache: {CACHE}  sha256={sha}  bytes={len(raw)}")

print()
print("--- 1. size and identity ---")
ck(cache["stats"]["n_samples"] == 17516, "stats.n_samples == 17516",
   str(cache["stats"]["n_samples"]))
ck(len(routes) == 17516, "routes entries == 17516", str(len(routes)))
ds_ids, rc_ids = set(gt_task), set(routes)
ck(len(ds_ids) == len(gt_task), "no duplicate sample_id in the training set")
ck(len(rc_ids) == len(routes), "no duplicate sample_id in the cache")
ck(ds_ids == rc_ids, "sample_id sets IDENTICAL",
   f"missing={len(ds_ids-rc_ids)} extra={len(rc_ids-ds_ids)}")
if ds_ids - rc_ids:
    print("    missing e.g.", sorted(ds_ids - rc_ids)[:5])
if rc_ids - ds_ids:
    print("    extra   e.g.", sorted(rc_ids - ds_ids)[:5])

print()
print("--- 2. task_type coverage ---")
tt = collections.Counter(gt_task[s][0] for s in routes)
for t in ("seq_rec", "title2sid", "description2sid", "seqtitle2sid"):
    ck(tt.get(t, 0) > 0, f"task_type {t} present", str(tt.get(t, 0)))
ck(set(tt) == {"seq_rec", "title2sid", "description2sid", "seqtitle2sid"},
   "no unexpected task_type", str(sorted(set(tt))))

print()
print("--- 3. route field validity ---")
bad_route = [s for s, r in routes.items() if r.get("route") not in ("NORMAL", "HARD")]
ck(not bad_route, "every route in {NORMAL, HARD}", f"bad={len(bad_route)}")
bad_h = [s for s, r in routes.items() if r.get("h") not in (0, 1)]
ck(not bad_h, "every h in {0, 1}", f"bad={len(bad_h)}")
mism = [s for s, r in routes.items()
        if (r["route"] == "NORMAL" and (r["h"] != 0 or r.get("hint", "") != ""))
        or (r["route"] == "HARD" and r["h"] != 1)]
ck(not mism, "NORMAL<->h=0<->empty hint and HARD<->h=1", f"mismatch={len(mism)}")
hs = collections.Counter(r["h"] for r in routes.values())
ck(set(hs) <= {0, 1}, "only h=0 / h=1 used", str(dict(hs)))

print()
print("--- 4. HARD hint == first SID token of the GT ---")
import re
SIDRE = re.compile(r"<[a-z]_\d+>")
bad_hint, checked = [], 0
for s, r in routes.items():
    if r["route"] != "HARD":
        continue
    gt = gt_task[s][1]
    first = SIDRE.findall(gt)[0] if SIDRE.findall(gt) else None
    checked += 1
    if r.get("hint") != first:
        bad_hint.append((s, r.get("hint"), first, gt))
ck(not bad_hint, f"HARD hint == GT first SID token ({checked} checked)",
   f"bad={len(bad_hint)}" + (f" e.g. {bad_hint[:2]}" if bad_hint else ""))

print()
print("--- 5. per-task route ratios ---")
per = collections.defaultdict(lambda: collections.Counter())
for s, r in routes.items():
    per[gt_task[s][0]][r["route"]] += 1
print(f"    {'task':17s} {'NORMAL':>8s} {'HARD':>8s} {'total':>8s} {'HARD%':>8s}")
tot_n = tot_h = 0
for t in ("seq_rec", "title2sid", "description2sid", "seqtitle2sid"):
    n, h = per[t]["NORMAL"], per[t]["HARD"]
    tot_n += n
    tot_h += h
    print(f"    {t:17s} {n:8d} {h:8d} {n+h:8d} {100.0*h/(n+h):7.2f}%")
print(f"    {'TOTAL':17s} {tot_n:8d} {tot_h:8d} {tot_n+tot_h:8d} {100.0*tot_h/(tot_n+tot_h):7.2f}%")
ck(tot_n == cache["stats"]["n_normal"] and tot_h == cache["stats"]["n_hard"],
   "recomputed totals match cache stats",
   f"{tot_n}/{tot_h} vs {cache['stats']['n_normal']}/{cache['stats']['n_hard']}")

print()
print("--- 6. metadata ---")
m = cache["meta"]
ck(m.get("seed") == 42, "seed == 42", str(m.get("seed")))
ck(m.get("G") == 16, "G == 16", str(m.get("G")))
ck(m.get("checkpoint") == "runs/industrial_sft/final_checkpoint",
   "frozen SFT checkpoint", str(m.get("checkpoint")))
ck(m.get("subset") == "splits", "original subset preserved", str(m.get("subset")))
ck(m.get("reward") == "exact_match_only", "reward = exact_match_only",
   str(m.get("reward")))

print()
print("=" * 92)
print(f"RESULT: {'ALL PASS' if not FAILS else 'FAIL'}")
for f in FAILS:
    print("  FAILED:", f)
print(f"SHA256 = {sha}")
print("=" * 92)

json.dump({"sha256": sha, "bytes": len(raw), "n_samples": len(routes),
           "n_normal": tot_n, "n_hard": tot_h,
           "per_task": {t: dict(per[t]) for t in per},
           "fails": FAILS},
          open("artifacts/rl_audit/r21g_cache_validation.json", "w"), indent=2)
print("  [save] artifacts/rl_audit/r21g_cache_validation.json")
