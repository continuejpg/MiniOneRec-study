#!/usr/bin/env python3
"""
analysis/build_shuffled_info.py

Generate the shuffled-SID twin of data/Amazon/info/<Category>_..._.txt.

Why this file is required
-------------------------
evaluate.py builds the Semantic-ID constrained-decoding trie EXCLUSIVELY from
--info_file (evaluate.py:61-98: it reads field 1 of each line, tokenises
"### Response:\\n<sid>", and fills hash_dict). The test CSV is used only for the
user history / target SIDs. So a shuffled-SID evaluation must be given an info
file whose field 1 carries the SHUFFLED SIDs; otherwise the trie would still
constrain decoding to the ORIGINAL item<->SID mapping and the intervention would
be undone at decoding time.

Field 2 (item title) and field 3 (item_id) are item attributes and are copied
verbatim -- the intervention must not touch them.

Input :  data/Amazon/info/<Category>_5_2016-10-2018-11.txt   (original, CRLF)
         analysis/shuffled_sid/<Category>.index.json          (shuffled mapping)
Output:  analysis/shuffled_sid/<Category>_shuffled.info.txt

Also mirrors the file into data/Amazon/info/ so that evaluate.py's own
`if item_path.endswith('.txt'): item_path = item_path[:-4]` logic in calc.py gets
a clean stem. CPU only, trains nothing.
"""
import csv
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import (  # noqa: E402
    CATEGORY, INDEX, INFO, SHUFFLED_DIR, SHUFFLED_INFO,
)

CAT = CATEGORY

SRC_INFO = INFO
SHUF_INDEX = os.path.join(SHUFFLED_DIR, f"{CATEGORY}.index.json")
ORIG_INDEX = INDEX

# Authoritative output location: the data tree, matching
# scripts/eval_shuffled_sid.sh and .gitignore.
OUT_SHUF = SHUFFLED_INFO

# Legacy location from an earlier run. The file recorded in
# analysis/results/*.json lives at OUT_SHUF; this path is only cross-checked
# when it happens to exist, and is never the primary write target.
OUT_LEGACY = os.path.join(SHUFFLED_DIR, f"{CATEGORY}_shuffled.info.txt")
OUT_MIRROR = OUT_SHUF


def main():
    item2sid = {int(k): "".join(v) for k, v in
                json.load(open(SHUF_INDEX, encoding="utf-8")).items()}
    orig2sid = {int(k): "".join(v) for k, v in
                json.load(open(ORIG_INDEX, encoding="utf-8")).items()}

    raw = open(SRC_INFO, "rb").read()
    crlf = raw.count(b"\r\n")
    lines = raw.decode("utf-8").split("\r\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    print(f"[in]  {SRC_INFO}")
    print(f"      lines={len(lines)}  CRLF endings={crlf}")

    out_lines = []
    n_replaced = 0
    bad = 0
    for ln in lines:
        parts = ln.split("\t")
        if len(parts) < 3:
            bad += 1
            out_lines.append(ln)
            continue
        old_sid, title, item_id = parts[0], parts[1], parts[2]
        try:
            new_sid = item2sid[int(item_id)]
        except (KeyError, ValueError):
            bad += 1
            out_lines.append(ln)
            continue
        if old_sid != orig2sid.get(int(item_id)):
            bad += 1
        if new_sid != old_sid:
            n_replaced += 1
        out_lines.append(f"{new_sid}\t{title}\t{item_id}")

    payload = "\r\n".join(out_lines) + "\r\n"
    os.makedirs(os.path.dirname(OUT_SHUF), exist_ok=True)
    with open(OUT_SHUF, "w", encoding="utf-8", newline="") as f:
        f.write(payload)
    print(f"[out] {OUT_SHUF}  ({os.path.getsize(OUT_SHUF):,} bytes)")

    # a stale copy from an earlier run must agree; if it disagrees, say so loudly
    if os.path.exists(OUT_LEGACY) and os.path.abspath(OUT_LEGACY) != os.path.abspath(OUT_SHUF):
        same_legacy = open(OUT_LEGACY, "rb").read() == open(OUT_SHUF, "rb").read()
        print(f"[legacy] {OUT_LEGACY}  identical={same_legacy}")
        if not same_legacy:
            print("[legacy] WARNING: stale copy differs from the authoritative file; "
                  "delete it or re-run to refresh")

    # ------------------------------------------------------------------ checks
    print(f"\n[check] lines written          = {len(out_lines)}  (expect {len(lines)})")
    print(f"[check] SID field replaced     = {n_replaced}")
    print(f"[check] malformed / unexpected = {bad}")

    # The STRICT shuffle freezes the 31 collision-involved items, so exactly the
    # 3655 singleton SIDs get permuted among themselves. The per-string SID
    # multiset therefore DOES change; what is preserved is
    #   (a) the SID SET (a permutation of the same 3670 SIDs),
    #   (b) the multiplicity structure,
    #   (c) the SID token vocabulary (this is what drives the decoding trie).
    # NOTE: open() applies universal-newline translation, so the decoded text
    # never contains "\r\n" even though the file on disk is CRLF. splitlines()
    # is therefore the correct parser; split("\r\n") returns a single chunk.
    def field1(p):
        return [l.split("\t")[0] for l in
                open(p, encoding="utf-8").read().splitlines() if l.strip()]
    def field23(p):
        return [tuple(l.split("\t")[1:3]) for l in
                open(p, encoding="utf-8").read().splitlines() if l.strip()]

    fa, fb = field1(SRC_INFO), field1(OUT_SHUF)
    ca, cb = Counter(fa), Counter(fb)

    ok_set = set(fa) == set(fb)
    print(f"[check] SID SET identical      = {ok_set}  "
          f"(|orig|={len(set(fa))} |new|={len(set(fb))})")

    ok_mult = Counter(ca.values()) == Counter(cb.values())
    print(f"[check] multiplicity structure = {ok_mult}  "
          f"{dict(sorted(Counter(ca.values()).items()))}")

    same = field23(SRC_INFO) == field23(OUT_SHUF)
    print(f"[check] title+item_id identical= {same}")

    changed = sum(1 for x, y in zip(fa, fb) if x != y)
    print(f"[check] lines whose SID changed= {changed}/{len(lines)} "
          f"({changed/len(lines)*100:.4f}%)")

    def toks(p):
        return sorted({t for l in open(p, encoding="utf-8").read().splitlines()
                       if l.strip() for t in
                       __import__("re").findall(r"<[^<>]+>", l.split("\t")[0])})
    ta, tb = toks(SRC_INFO), toks(OUT_SHUF)
    print(f"[check] SID token vocab equal  = {ta == tb}  ({len(tb)} tokens)")

    ok = ok_set and ok_mult and same and (ta == tb)
    print(f"\n[verdict] {'OK -- file is safe to use' if ok else 'MISMATCH -- do not use'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
