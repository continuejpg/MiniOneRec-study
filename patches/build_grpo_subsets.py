#!/usr/bin/env python3
"""
build_grpo_subsets.py -- create FIXED, reproducible GRPO training subsets.

Run once on the server from the code root:
    python build_grpo_subsets.py

Writes (never overwrites unless --force):
    splits/grpo_seq_10k.json
    splits/grpo_seqtitle_1k.json
    splits/grpo_manifest.json

Each file stores the namespaced sample_ids produced by the patched data.py, so
that every later run (baseline AND optimized) trains on byte-identical data.
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

DATA = os.path.join(ROOT, "data", "Amazon")
CAT = "Industrial_and_Scientific"
TRAIN = os.path.join(DATA, "train", f"{CAT}_5_2016-10-2018-11.csv")
INDEX = os.path.join(DATA, "index", f"{CAT}.index.json")
ITEM = os.path.join(DATA, "index", f"{CAT}.item.json")
CATEGORY = "industrial and scientific items"

SEED = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="overwrite existing subset files")
    args = ap.parse_args()

    outdir = os.path.join(ROOT, "splits")
    os.makedirs(outdir, exist_ok=True)

    targets = [
        ("seq", "grpo_seq_10k.json", 10000),
        ("seq_title", "grpo_seqtitle_1k.json", 1000),
    ]
    for _, fname, _ in targets:
        p = os.path.join(outdir, fname)
        if os.path.exists(p) and not args.force:
            print(f"*** {p} already exists. Refusing to overwrite (use --force).")
            print("*** Existing subsets are the frozen baseline; regenerating them invalidates comparisons.")
            return 2

    from data import SidDataset, RLTitle2SidDataset, RLSeqTitle2SidDataset

    print("building datasets (upstream defaults) ...")
    ds_seq = SidDataset(TRAIN, category=CATEGORY, sample=10000, seed=SEED)
    ds_title = RLTitle2SidDataset(item_file=ITEM, index_file=INDEX, category=CATEGORY, sample=-1, seed=SEED)
    ds_seqt = RLSeqTitle2SidDataset(TRAIN, category=CATEGORY, sample=1000, seed=SEED)

    print(f"  seq_rec        : {len(ds_seq)}")
    print(f"  title2sid+desc : {len(ds_title)}")
    print(f"  seqtitle2sid   : {len(ds_seqt)}")

    manifest = {
        "seed": SEED,
        "category": CAT,
        "created_from": {
            "train_file": os.path.relpath(TRAIN, ROOT),
            "index_file": os.path.relpath(INDEX, ROOT),
            "item_file": os.path.relpath(ITEM, ROOT),
        },
        "tasks": {
            "seq_rec": {"dataset": "SidDataset", "requested_sample": 10000, "actual": len(ds_seq),
                        "subset_file": "grpo_seq_10k.json"},
            "title2sid": {"dataset": "RLTitle2SidDataset", "requested_sample": -1,
                          "actual": len(ds_title), "subset_file": None, "note": "full, fixed"},
            "seqtitle2sid": {"dataset": "RLSeqTitle2SidDataset", "requested_sample": 1000,
                             "actual": len(ds_seqt), "subset_file": "grpo_seqtitle_1k.json"},
        },
        "total_samples": len(ds_seq) + len(ds_title) + len(ds_seqt),
    }

    for ds, key, fname, want in [
        (ds_seq, "seq", "grpo_seq_10k.json", 10000),
        (ds_seqt, "seq_title", "grpo_seqtitle_1k.json", 1000),
    ]:
        ids = [x["sample_id"] for x in ds]
        assert len(ids) == len(set(ids)), f"duplicate sample_id inside {key}"
        assert all(i.startswith(key + ":") for i in ids), f"bad namespace in {key}"
        payload = {
            "namespace": key,
            "seed": SEED,
            "requested": want,
            "count": len(ids),
            "sample_ids": ids,
        }
        p = os.path.join(outdir, fname)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=1)
        print(f"  wrote {p}  ({len(ids)} ids)")

    pm = os.path.join(outdir, "grpo_manifest.json")
    with open(pm, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"  wrote {pm}")
    print(f"\ntotal samples for GRPO baseline: {manifest['total_samples']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
