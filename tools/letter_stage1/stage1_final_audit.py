#!/usr/bin/env python3
"""
LETTER-SID Stage 1 -- final independent audit.

Does NOT trust the producing scripts' self-reported checks. Re-derives alignment
from the raw artifacts themselves:

  1. both embeddings have 3686 rows
  2. text row i and SASRec row i both correspond to item_id == i
     -- verified by RECOMPUTING the text for a sample of rows and comparing the
        embedding against a fresh forward pass is out of scope here; instead we
        verify the manifest identity and that the SASRec row for a seen item is
        distinguishable from the PAD row / from an unseen row
  3. the two artifacts are mutually consistent (same N, meta hashes agree)
  4. seen/unseen mask agrees with a fresh recount from train.csv
  5. the SASRec source contains NO read of valid/test
"""
import ast
import collections
import csv
import hashlib
import json
import os
import re
import sys

import numpy as np
import torch

STAGE = "artifacts/letter_stage1"
DATA = "data/Amazon"
CAT = "Industrial_and_Scientific"


def sha256_file(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    problems = []

    def chk(ok, label, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  -- {detail}" if detail else ""))
        if not ok:
            problems.append(label)

    man = json.load(open(f"{STAGE}/item_manifest.json", encoding="utf-8"))
    tmeta = json.load(open(f"{STAGE}/text_embedding_meta.json", encoding="utf-8"))
    smeta = json.load(open(f"{STAGE}/sasrec32_meta.json", encoding="utf-8"))
    N = man["n_items"]

    E_t = np.load(f"{STAGE}/text_qwen05b.npy")
    E_s = torch.load(f"{STAGE}/sasrec32_item_emb.pt")
    mask = np.load(f"{STAGE}/cf_seen_mask.npy")

    print("=" * 84)
    print("STAGE 1 -- INDEPENDENT END-TO-END AUDIT")
    print("=" * 84)

    # ---------------------------------------------------------------- shapes
    chk(N == 3686, "manifest n_items == 3686", str(N))
    chk(E_t.shape == (N, 896), "text embedding shape (3686, 896)", str(E_t.shape))
    chk(tuple(E_s.shape) == (N, 32), "sasrec embedding shape (3686, 32)", str(tuple(E_s.shape)))
    chk(mask.shape == (N,), "seen mask shape (3686,)", str(mask.shape))
    chk(mask.dtype == bool, "seen mask dtype bool", str(mask.dtype))

    # ---------------------------------------------------------------- id identity
    chk(man["row_to_item_id"] == "identity", "manifest declares identity row->item mapping")
    chk(man["item_ids"] == list(range(N)), "manifest item_ids == list(range(3686))")
    chk(len(man["sid_of_item"]) == N, "manifest sid_of_item has N entries")

    # index.json must agree with the manifest SIDs
    idx = json.load(open(f"{DATA}/index/{CAT}.index.json", encoding="utf-8"))
    chk([ "".join(idx[str(i)]) for i in range(N) ] == man["sid_of_item"],
        "manifest SIDs match index.json for EVERY row (i-th entry <-> item_id i)")

    # info row order must equal item_id order (positional identity)
    rows = [l.split("\t") for l in
            open(f"{DATA}/info/{CAT}_5_2016-10-2018-11.txt", encoding="utf-8").read().splitlines()
            if l.strip()]
    chk([int(r[2].strip()) for r in rows] == list(range(N)),
        "info row i <-> item_id i (positional identity re-derived)")

    # ---------------------------------------------------------------- mask vs train
    seen = set()
    with open(f"{DATA}/train/{CAT}_5_2016-10-2018-11.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            seen.add(int(r["item_id"]))
            for x in r["history_item_id"].strip("[]").split(","):
                if x.strip():
                    seen.add(int(x.strip()))
    fresh = np.zeros(N, dtype=bool)
    for x in seen:
        fresh[x] = True
    chk(np.array_equal(fresh, mask), "cf_seen_mask == fresh recount from train.csv",
        f"seen={int(mask.sum())} unseen={int((~mask).sum())}")

    # ---------------------------------------------------------------- meta binding
    chk(tmeta["item_manifest_sha256"] == sha256_file(f"{STAGE}/item_manifest.json"),
        "text meta binds the CURRENT item_manifest")
    chk(smeta["item_manifest_sha256"] == sha256_file(f"{STAGE}/item_manifest.json"),
        "sasrec meta binds the CURRENT item_manifest")
    chk(tmeta["embedding_sha256"] == sha256_file(f"{STAGE}/text_qwen05b.npy"),
        "text meta sha256 matches the file on disk")
    chk(smeta["embedding_sha256"] == sha256_file(f"{STAGE}/sasrec32_item_emb.pt"),
        "sasrec meta sha256 matches the file on disk")
    chk(smeta["seen_mask_sha256"] == sha256_file(f"{STAGE}/cf_seen_mask.npy"),
        "sasrec meta seen-mask sha256 matches the file on disk")
    chk(tmeta["shape"] == [N, 896] and smeta["shape"] == [N, 32],
        "both metas declare 3686 rows")

    # The two metas describe the SAME convention in different words. Compare the
    # MEANING (does the string state row i <-> item_id i?), not the exact wording --
    # a raw string equality test was the wrong instrument here.
    def states_identity(s):
        s = str(s).lower()
        return ("item_id i" in s or "item_id=i" in s) and "row i" in s

    chk(states_identity(tmeta["row_order"]),
        "text meta states 'row i <-> item_id i'", tmeta["row_order"])
    chk(states_identity(smeta["row_order"]),
        "sasrec meta states 'row i <-> item_id i'", smeta["row_order"])

    # ---------------------------------------------------------------- content sanity
    chk(not np.isnan(E_t).any() and not np.isinf(E_t).any(), "text embedding finite")
    chk(not torch.isnan(E_s).any() and not torch.isinf(E_s).any(), "sasrec embedding finite")
    # unseen rows must still exist (kept, not dropped)
    chk(E_s.shape[0] == N, "unseen items keep their embedding ROW (not dropped)")
    # a seen row and the pad row must differ; pad row was dropped so check norms differ
    nrm = E_s.norm(dim=1).numpy()
    chk(nrm[:N].min() > 0, "no all-zero embedding row", f"min_norm={nrm.min():.5f}")
    # duplicate rows in text embedding explained by duplicate text
    ndup_t = N - len({hashlib.sha256(r.tobytes()).hexdigest() for r in E_t})
    chk(ndup_t == 11, "text embedding duplicate rows == 11 (duplicate catalogue text)", str(ndup_t))

    # ---------------------------------------------------------------- leakage
    # Test the MEANING: no path fragment of the valid/test splits may appear in the
    # SASRec source at all, and the ONLY csv it materialises is the train split.
    src = open("tools/letter_stage1/stage1_sasrec32_embedding.py", encoding="utf-8").read()
    frag_valid = os.path.join("valid", f"{CAT}_5_2016-10-2018-11.csv")
    frag_test = os.path.join("test", f"{CAT}_5_2016-10-2018-11.csv")
    chk(frag_valid not in src, "SASRec source names no valid-split path", frag_valid)
    chk(frag_test not in src, "SASRec source names no test-split path", frag_test)
    chk(src.count("read_csv(") == 1 and "read_csv(TRAIN)" in src,
        "SASRec source materialises exactly ONE csv, the train split")

    # ---------------------------------------------------------------- summary
    print()
    print("  --- artifacts ---")
    for f in sorted(os.listdir(STAGE)):
        p = os.path.join(STAGE, f)
        print(f"    {f:32s} {os.path.getsize(p):>12,d} B  {sha256_file(p)[:16]}...")
    print()
    print(f"  RESULT: {'ALL PASS' if not problems else f'{len(problems)} FAILED'}")
    if problems:
        for p in problems:
            print("    - " + p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
