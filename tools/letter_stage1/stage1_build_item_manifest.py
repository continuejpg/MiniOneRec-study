#!/usr/bin/env python3
"""
LETTER-SID Stage 1 -- step 2: freeze the catalogue mapping.

Builds artifacts/letter_stage1/item_manifest.json, which is the single source of
truth that ties row index -> item_id for every downstream embedding artifact.

Design rules (from the Stage-1 brief):
  * the item_id space is an EXPLICIT integer range, never an implicit dict order
  * assert item ids == set(range(3686))   -- fail loud, never infer from "same count"
  * cross-check index.json / item.json / info / train / valid / test

READ-ONLY with respect to existing data: this script only reads the formal
catalogue files and writes into artifacts/letter_stage1/.
"""
import argparse
import csv
import hashlib
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_text(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/Amazon")
    ap.add_argument("--category", default="Industrial_and_Scientific")
    ap.add_argument("--out-dir", default="artifacts/letter_stage1")
    ap.add_argument("--expect-items", type=int, default=3686)
    args = ap.parse_args()

    D = args.data_root
    CAT = args.category
    INDEX = os.path.join(D, "index", f"{CAT}.index.json")
    ITEMJSON = os.path.join(D, "index", f"{CAT}.item.json")
    INFO = os.path.join(D, "info", f"{CAT}_5_2016-10-2018-11.txt")
    SPLITS = {s: os.path.join(D, s, f"{CAT}_5_2016-10-2018-11.csv")
              for s in ("train", "valid", "test")}

    problems = []

    def chk(ok, label, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  -- {detail}" if detail else ""))
        if not ok:
            problems.append(label)

    print("=" * 84)
    print("STAGE 1 / step 2 -- freeze catalogue mapping")
    print("=" * 84)

    for p in (INDEX, ITEMJSON, INFO, *SPLITS.values()):
        if not os.path.exists(p):
            print(f"REFUSE: missing required input {p}")
            return 1

    N = args.expect_items

    # ---------------------------------------------------------------- index.json
    idx = json.load(open(INDEX, encoding="utf-8"))
    idx_ids = sorted(int(k) for k in idx)
    chk(len(idx) == N, "index.json item count", str(len(idx)))
    chk(idx_ids == list(range(N)),
        "index.json keys == set(range(N))  [EXPLICIT, not dict-order]",
        f"{idx_ids[0]}..{idx_ids[-1]}, n={len(idx_ids)}")
    chk(all(len(v) == 3 for v in idx.values()),
        "every index SID is exactly 3 tokens",
        str(sorted({len(v) for v in idx.values()})))

    # ---------------------------------------------------------------- item.json
    ij = json.load(open(ITEMJSON, encoding="utf-8"))
    ij_ids = sorted(int(k) for k in ij)
    chk(len(ij) == N, "item.json item count", str(len(ij)))
    chk(ij_ids == list(range(N)), "item.json keys == set(range(N))")

    # ---------------------------------------------------------------- info
    rows = [l.split("\t") for l in open(INFO, encoding="utf-8").read().splitlines() if l.strip()]
    chk(len(rows) == N, "info line count", str(len(rows)))
    info_ids = [int(r[2].strip()) for r in rows]
    chk(sorted(info_ids) == list(range(N)),
        "info column-3 ids == set(range(N))",
        f"min={min(info_ids)} max={max(info_ids)} unique={len(set(info_ids))}")
    # info row order must equal item_id order (downstream code reads info positionally)
    chk(info_ids == list(range(N)),
        "info row order == item_id order (positional identity)",
        "row i <-> item_id i" if info_ids == list(range(N)) else f"first mismatch at {next(i for i,(a,b) in enumerate(zip(info_ids, range(N))) if a!=b)}")

    # ---------------------------------------------------------------- splits
    split_seen = {}
    for name, path in SPLITS.items():
        ids = set()
        with open(path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                ids.add(int(r["item_id"]))
                for x in r["history_item_id"].strip("[]").split(","):
                    if x.strip():
                        ids.add(int(x.strip()))
        chk(ids <= set(range(N)),
            f"{name}.csv item_ids within range(N)",
            f"n_distinct={len(ids)} min={min(ids)} max={max(ids)}")
        split_seen[name] = ids
        if name == "train":
            train_ids = ids

    # ---------------------------------------------------------------- alignment
    # the four sources must agree on the same integer space
    chk(set(idx_ids) == set(ij_ids) == set(info_ids) == set(range(N)),
        "index / item.json / info agree on the SAME id space")

    train_unseen = sorted(set(range(N)) - train_ids)
    print()
    print(f"  train-seen items   = {len(train_ids)}")
    print(f"  train-UNSEEN items = {len(train_unseen)}")
    if train_unseen:
        print(f"    (these have no collaborative supervision; rows preserved, masked later)")
        print(f"    count = {len(train_unseen)}  first 10 = {train_unseen[:10]}")

    if problems:
        print()
        print(f"REFUSE: {len(problems)} failed check(s): {problems}")
        return 1

    # ---------------------------------------------------------------- write
    manifest = {
        "category": CAT,
        "n_items": N,
        "row_to_item_id": "identity",          # row i  <->  item_id i
        "item_ids": list(range(N)),
        "sid_of_item": [ "".join(idx[str(i)]) for i in range(N) ],
        "train_seen_item_ids": sorted(train_ids),
        "train_unseen_item_ids": train_unseen,
        "n_train_seen": len(train_ids),
        "n_train_unseen": len(train_unseen),
        "sources": {
            "index_json": INDEX,
            "index_json_sha256": sha256_file(INDEX),
            "item_json": ITEMJSON,
            "item_json_sha256": sha256_file(ITEMJSON),
            "info": INFO,
            "info_sha256": sha256_file(INFO),
            "splits": {k: {"path": v, "sha256": sha256_file(v)} for k, v in SPLITS.items()},
        },
        "assertions": {
            "index_ids_eq_range": True,
            "item_json_ids_eq_range": True,
            "info_ids_eq_range": True,
            "info_row_order_is_identity": info_ids == list(range(N)),
            "splits_within_range": True,
        },
    }
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, "item_manifest.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # stable hash over the identity mapping itself (for embedding meta binding)
    payload = json.dumps({"n": N, "ids": list(range(N)),
                          "sid": manifest["sid_of_item"]}, sort_keys=True)
    manifest_hash = sha256_text(payload)

    print()
    print(f"  [save] {out}")
    print(f"  item_manifest hash (identity mapping) = {manifest_hash}")
    print(f"  file sha256 = {sha256_file(out)}")
    print()
    print("RESULT: MANIFEST OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
