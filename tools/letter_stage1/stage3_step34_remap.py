#!/usr/bin/env python3
"""
Stage 3 / steps 3+4 -- remap the Amazon splits to LETTER SIDs, byte-faithfully.

Corrected approach
------------------
The previous version re-serialised every CSV record, which mis-handled the source
file's line terminators. This version works at the BYTE level inside each raw
record:

  * parse the raw record only to locate the two SID fields as (start, end) byte
    spans inside the record's own bytes
  * splice the new SID text into exactly those spans
  * leave every other byte of the record -- including its terminator -- untouched

That makes "only the SID representation changed" a property of the construction
rather than something to be checked afterwards. It is checked afterwards anyway.

Outputs into artifacts/letter_stage3/data/:
    train|valid|test/<BASE>.csv
    info/<BASE>.txt
    index/Industrial_and_Scientific.index.json   (copy; the filename is a hard
                                                  constraint of sft.py)
    index/Industrial_and_Scientific.item.json    (copy of the ORIGINAL item meta)
"""
import argparse
import ast
import csv
import hashlib
import json
import os
import re
import shutil
import sys

LETTER_INDEX = "artifacts/letter_stage2/letter_index.json"
SRC_ROOT = "data/Amazon"
OUT_ROOT = "artifacts/letter_stage3/data"
CATEGORY = "Industrial_and_Scientific"
BASE = f"{CATEGORY}_5_2016-10-2018-11"
SPLITS = ("train", "valid", "test")
SID_COLUMNS = ("history_item_sid", "item_sid")
KEEP_COLUMNS = ("user_id", "item_title", "history_item_title",
                "item_id", "history_item_id")


def sha256(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def split_records(raw):
    """Split a CSV body into records on UNQUOTED newlines only.

    Some title fields are quoted AND contain embedded newlines, so a naive
    raw.split(b"\\n") fragments a single record into several. This walks the bytes
    tracking quote state, exactly like a CSV reader, and keeps each record's own
    terminator attached.

    Returns (records, trailing) where `trailing` is b"" if the file ended with a
    terminator, else None.
    """
    recs, start, i, in_q = [], 0, 0, False
    n = len(raw)
    while i < n:
        c = raw[i:i + 1]
        if in_q:
            if c == b'"':
                if raw[i + 1:i + 2] == b'"':
                    i += 2
                    continue
                in_q = False
            i += 1
            continue
        if c == b'"':
            in_q = True
            i += 1
            continue
        if c == b"\n":
            recs.append(raw[start:i + 1])
            i += 1
            start = i
            continue
        i += 1
    if start < n:
        recs.append(raw[start:n])
        return recs, None
    return recs, b""


def field_spans(raw):
    """Byte spans of each field inside a raw CSV record (record keeps its EOL)."""
    r = raw
    n = len(r)
    spans, i, start = [], 0, 0
    in_q = False
    while i < n:
        c = r[i:i + 1]
        if in_q:
            if c == b'"':
                if r[i + 1:i + 2] == b'"':
                    i += 2
                    continue
                in_q = False
            i += 1
            continue
        if c == b'"':
            in_q = True
            i += 1
            continue
        if c == b",":
            spans.append((start, i))
            i += 1
            start = i
            continue
        if c in (b"\r", b"\n"):
            spans.append((start, i))
            return spans, i
        i += 1
    spans.append((start, n))
    return spans, n


def unquote(b):
    """Decode a CSV field's bytes to its logical string value.

    Field spans keep the surrounding double quotes (and any doubled inner quotes),
    so '[117]' arrives as b'"[117]"'. csv.reader handles both.
    """
    s = b.decode("utf-8")
    if s.startswith('"'):
        return next(csv.reader([s]))[0]
    return s


def quote_field(v):
    if any(ch in v for ch in (",", '"', "\n", "\r")):
        return '"' + v.replace('"', '""') + '"'
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--letter-index", default=LETTER_INDEX)
    ap.add_argument("--src-root", default=SRC_ROOT)
    ap.add_argument("--out-root", default=OUT_ROOT)
    args = ap.parse_args()

    idx = json.load(open(args.letter_index, encoding="utf-8"))
    n_items = len(idx)
    sid_of = {int(k): "".join(v) for k, v in idx.items()}
    assert sorted(sid_of) == list(range(n_items)), "letter index id space broken"
    depth = len(idx["0"])
    sid_set = set(sid_of.values())

    print("=" * 86)
    print("STEPS 3+4 -- byte-faithful remap to LETTER SIDs + integrity audit")
    print("=" * 86)
    print(f"  letter index : {args.letter_index}  items={n_items} depth={depth}")
    print(f"  sha256       = {sha256(args.letter_index)}")

    report = {"letter_index": args.letter_index,
              "letter_index_sha256": sha256(args.letter_index),
              "letter_n_items": n_items, "letter_depth": depth,
              "splits": {}}
    fails = []

    def chk(ok, label, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  -- {detail}" if detail else ""))
        if not ok:
            fails.append(label)
        return ok

    for split in SPLITS:
        src = os.path.join(args.src_root, split, f"{BASE}.csv")
        os.makedirs(os.path.join(args.out_root, split), exist_ok=True)
        dst = os.path.join(args.out_root, split, f"{BASE}.csv")

        print()
        print("-" * 86)
        print(f"  {split}: {src}")
        raw = open(src, "rb").read()
        recs, trailing = split_records(raw)
        n_rec = len(recs)                      # header + data
        header = next(csv.reader([recs[0].decode("utf-8")]))
        hl = {h: i for i, h in enumerate(header)}
        for c in KEEP_COLUMNS + SID_COLUMNS:
            if c not in hl:
                print(f"REFUSE: column {c!r} missing")
                sys.exit(1)
        n_rows = n_rec - 1
        print(f"    records={n_rec} (1 header + {n_rows} rows)  "
              f"trailing_newline={trailing is not None}")

        out = [recs[0]]
        keep_src = {c: [] for c in KEEP_COLUMNS}
        keep_dst = {c: [] for c in KEEP_COLUMNS}
        n_hist = 0
        for r in range(1, n_rec):
            rec = recs[r]
            spans, _end = field_spans(rec)
            if len(spans) != len(header):
                print(f"REFUSE: row {r} has {len(spans)} fields, header has {len(header)}")
                sys.exit(1)
            fields = [unquote(rec[a:b]) for a, b in spans]
            for c in KEEP_COLUMNS:
                keep_src[c].append(fields[hl[c]])

            target_id = int(fields[hl["item_id"]].strip())
            try:
                hist_ids = ast.literal_eval(fields[hl["history_item_id"]])
            except Exception as e:  # noqa: BLE001
                print(f"REFUSE: row {r}: cannot parse history_item_id "
                      f"({type(e).__name__}: {e})")
                print(f"        nspans={len(spans)} nheader={len(header)} "
                      f"in_q_end={in_q}")
                for i, (a, b) in enumerate(spans):
                    print(f"          [{i}] {header[i]:20s} = {fields[i][:80]!r}")
                print(f"        record head: {rec[:200]!r}")
                sys.exit(1)
            if target_id not in sid_of:
                print(f"REFUSE: item_id {target_id} (row {r}) unmapped")
                sys.exit(1)
            new_target = sid_of[target_id]
            new_hist = []
            for h in hist_ids:
                if int(h) not in sid_of:
                    print(f"REFUSE: history item_id {h} (row {r}) unmapped")
                    sys.exit(1)
                new_hist.append(sid_of[int(h)])
                n_hist += 1

            # splice: rebuild the record from spans, replacing only SID fields
            pieces, last = [], 0
            for i, (a, b) in enumerate(spans):
                pieces.append(rec[last:a])
                name = header[i]
                if name == "item_sid":
                    pieces.append(quote_field(new_target).encode("utf-8"))
                elif name == "history_item_sid":
                    pieces.append(quote_field(str(new_hist)).encode("utf-8"))
                else:
                    pieces.append(rec[a:b])
                last = b
            pieces.append(rec[last:])          # includes the EOL
            out.append(b"".join(pieces))
        # derive every column from the spliced rows (used by the checks below)
        keep_dst = {c: [] for c in KEEP_COLUMNS}
        for r in range(1, len(out)):
            sp, _ = field_spans(out[r])
            f = [unquote(out[r][a:b]) for a, b in sp]
            for c in KEEP_COLUMNS:
                keep_dst[c].append(f[hl[c]])

        body = b"".join(out)
        open(dst, "wb").write(body)

        # ---------------- assertions ----------------
        chk(len(out) - 1 == n_rows, f"{split}: row count preserved", f"{n_rows}")
        for c in KEEP_COLUMNS:
            chk(keep_src[c] == keep_dst[c], f"{split}: {c} byte-identical",
                f"{len(keep_src[c])} values")
        # strongest structural check: the only differing bytes are inside the two
        # SID columns -- verified by re-splicing the ORIGINAL SID values back in
        # and requiring full byte equality with the source.
        # strongest structural check: re-splice the ORIGINAL SID *values* (in the
        # original quoting style, taken from the source record's own bytes) back
        # into the rewritten record and require full byte equality with the source.
        restored = [recs[0]]      # header is copied verbatim; include it
        for r in range(1, len(out)):
            sp, _ = field_spans(out[r])
            old_sp, _ = field_spans(recs[r])
            pieces, last = [], 0
            for i, (a, b) in enumerate(sp):
                pieces.append(out[r][last:a])
                if header[i] in SID_COLUMNS:
                    oa, ob = old_sp[i]
                    pieces.append(recs[r][oa:ob])        # original bytes verbatim
                else:
                    pieces.append(out[r][a:b])
                last = b
            pieces.append(out[r][last:])
            restored.append(b"".join(pieces))
        rb = b"".join(restored)
        if rb != raw:
            k = next(i for i in range(min(len(rb), len(raw))) if rb[i] != raw[i])
            print(f"    DEBUG restore mismatch at byte {k} of {len(raw)} "
                  f"(restored len {len(rb)})")
            print(f"      src : {raw[max(0,k-90):k+90]!r}")
            print(f"      rest: {rb[max(0,k-90):k+90]!r}")
        chk(rb == raw,
            f"{split}: restoring original SIDs reproduces the source BYTE-FOR-BYTE")

        with open(dst, encoding="utf-8", newline="") as f:
            rr = csv.reader(f)
            h2 = next(rr)
            rows2 = [x for x in rr]
        chk(h2 == header, f"{split}: header unchanged")
        chk(len(rows2) == n_rows, f"{split}: re-read row count", f"{len(rows2)}")

        bd = bn = bh = 0
        for row in rows2:
            ts = row[hl["item_sid"]]
            if len(re.findall(r"<[^<>]+>", ts)) != depth:
                bd += 1
            if ts not in sid_set:
                bn += 1
            for hs in ast.literal_eval(row[hl["history_item_sid"]]):
                if len(re.findall(r"<[^<>]+>", hs)) != depth:
                    bh += 1
        chk(bd == 0, f"{split}: target SIDs all depth {depth}", f"{bd} bad")
        chk(bn == 0, f"{split}: target SIDs all in letter_index", f"{bn} not found")
        chk(bh == 0, f"{split}: history SIDs all depth {depth}", f"{bh} bad")

        report["splits"][split] = {
            "src": src, "dst": dst, "rows": n_rows,
            "src_sha256": sha256(src), "dst_sha256": sha256(dst),
            "history_sid_entries": n_hist,
        }
        print(f"    src_sha256={sha256(src)[:16]}  dst_sha256={sha256(dst)[:16]}")

    # ---------------- info ----------------
    print()
    print("-" * 86)
    src_info = os.path.join(args.src_root, "info", f"{BASE}.txt")
    os.makedirs(os.path.join(args.out_root, "info"), exist_ok=True)
    dst_info = os.path.join(args.out_root, "info", f"{BASE}.txt")
    rawi = open(src_info, "rb").read()
    recs = rawi.split(b"\n")
    tail = recs.pop() if recs and recs[-1] == b"" else None
    outi, bad = [], 0
    for i, lb in enumerate(recs):
        s = sid_of[i]
        if len(re.findall(r"<[^<>]+>", s)) != depth:
            bad += 1
        parts = lb.rstrip(b"\r").split(b"\t")
        parts[0] = s.encode("utf-8")
        outi.append(b"\t".join(parts))
    bodyi = b"\r\n".join(outi) + (b"\r\n" if tail is not None else b"")
    open(dst_info, "wb").write(bodyi)
    chk(len(outi) == n_items, "info: line count == letter items", f"{len(outi)}")
    chk(bad == 0, "info: every SID depth correct", f"{bad} bad")
    chk(rawi.count(b"\t") == bodyi.count(b"\t"), "info: tab count preserved")
    report["info"] = {"src": src_info, "dst": dst_info,
                      "src_sha256": sha256(src_info), "dst_sha256": sha256(dst_info),
                      "lines": len(outi)}

    # ---------------- index + item meta ----------------
    print()
    print("-" * 86)
    os.makedirs(os.path.join(args.out_root, "index"), exist_ok=True)
    dst_idx = os.path.join(args.out_root, "index", f"{CATEGORY}.index.json")
    shutil.copyfile(args.letter_index, dst_idx)
    dst_item = os.path.join(args.out_root, "index", f"{CATEGORY}.item.json")
    src_item = os.path.join(args.src_root, "index", f"{CATEGORY}.item.json")
    shutil.copyfile(src_item, dst_item)
    chk(sha256(dst_idx) == sha256(args.letter_index), "index copy byte-identical")
    chk(sha256(dst_item) == sha256(src_item),
        "item meta copy byte-identical to ORIGINAL")
    report["index"] = {"dst": dst_idx, "sha256": sha256(dst_idx)}
    report["item_meta"] = {"dst": dst_item, "sha256": sha256(dst_item)}

    # ---------------- coverage ----------------
    print()
    print("=" * 86)
    print("GLOBAL AUDIT")
    print("=" * 86)
    used = set()
    for split in SPLITS:
        with open(report["splits"][split]["dst"], encoding="utf-8", newline="") as f:
            rr = csv.reader(f)
            h = next(rr)
            hh = {x: i for i, x in enumerate(h)}
            for row in rr:
                used.add(row[hh["item_sid"]])
                used.update(ast.literal_eval(row[hh["history_item_sid"]]))
    chk(len(used) <= n_items, "distinct SIDs used <= catalogue", f"{len(used)}/{n_items}")
    chk(used <= sid_set, "every used SID exists in letter_index",
        f"{len(used - sid_set)} unknown")
    report["audit"] = {"distinct_sids_used": len(used), "letter_n_items": n_items,
                       "failures": fails}

    json.dump(report, open("artifacts/letter_stage3/remap_audit.json", "w"), indent=2)
    print()
    print(f"  failures = {fails}")
    print(f"  RESULT: {'ALL PASS' if not fails else 'FAIL'}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
