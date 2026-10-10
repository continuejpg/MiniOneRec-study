#!/usr/bin/env python3
"""
Stage 2.5 -- build the LETTER 4-level info file and a small test slice.

The info file is the ONLY catalogue input evaluate.py needs:
    evaluate.py:64-71 reads column 0 of every line as the SID and wraps it as
    f'### Response:\\n{sid}'. It never reads the index.json directly.

Writes into artifacts/letter_stage2/ (no formal data is touched):
    letter_info.txt          <sid>\t<title>\t<item_id>
    letter_test_smoke.csv    first N rows of the ORIGINAL test split with
                             item_sid / history_item_sid remapped to LETTER SIDs
"""
import argparse
import ast
import csv
import json
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="Industrial_and_Scientific")
    ap.add_argument("--data-root", default="data/Amazon")
    ap.add_argument("--index", default="artifacts/letter_stage2/letter_index.json")
    ap.add_argument("--out-dir", default="artifacts/letter_stage2")
    ap.add_argument("--n-samples", type=int, default=64)
    args = ap.parse_args()

    CAT = args.category
    idx = json.load(open(args.index, encoding="utf-8"))
    N = len(idx)
    sid = {int(k): "".join(v) for k, v in idx.items()}
    assert sorted(sid) == list(range(N)), "index id space is not 0..N-1"
    depth = len(idx["0"])
    print(f"  LETTER index: N={N} depth={depth}")

    # ---------------- info file ----------------
    item_json = json.load(open(
        f"{args.data_root}/index/{CAT}.item.json", encoding="utf-8"))
    assert sorted(int(k) for k in item_json) == list(range(N))
    info_path = os.path.join(args.out_dir, "letter_info.txt")
    with open(info_path, "w", encoding="utf-8", newline="\n") as f:
        for i in range(N):
            d = item_json[str(i)]
            title = d.get("title", "")
            if isinstance(title, list):
                title = " ".join(title)
            title = str(title).replace("\t", " ").replace("\n", " ").strip()
            f.write(f"{sid[i]}\t{title}\t{i}\n")
    print(f"  [save] {info_path}")

    # ---------------- test slice ----------------
    src = f"{args.data_root}/test/{CAT}_5_2016-10-2018-11.csv"
    rows = list(csv.DictReader(open(src, encoding="utf-8")))
    out_csv = os.path.join(args.out_dir, "letter_test_smoke.csv")
    fieldnames = list(rows[0].keys())
    written = 0
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows[:args.n_samples]:
            h = ast.literal_eval(r["history_item_id"])
            r2 = dict(r)
            r2["history_item_sid"] = str([sid[int(x)] for x in h])
            r2["item_sid"] = sid[int(r["item_id"])]
            w.writerow(r2)
            written += 1
    print(f"  [save] {out_csv}  rows={written}")

    # ---------------- self-checks ----------------
    info_lines = [l for l in open(info_path, encoding="utf-8").read().splitlines() if l.strip()]
    depths = {len(l.split("\t")[0].split("><")) for l in info_lines}
    print(f"  info lines = {len(info_lines)}  distinct sid depths = {depths}")
    print(f"  RESULT: {'OK' if len(info_lines) == N and depths == {depth} else 'FAIL'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
