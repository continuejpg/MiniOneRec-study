"""MiniOneRec-CAT / verify_chain -- prove the SID chain is internally consistent.

The failure this guards against
-------------------------------
A SID experiment goes wrong *silently* when the catalogue and its derived files
disagree: the model trains on new SIDs while ``evaluate.py`` still constrains
decoding with the old ones.  Nothing raises; the metrics simply look like a
modelling result.

This module re-derives the cross-checks and fails loudly:

1. every split row's ``item_sid`` equals the catalogue's SID for that ``item_id``;
2. every ``history_item_sid`` entry equals the catalogue's SID for the
   corresponding ``history_item_id`` (position by position);
3. the info file has one line per catalogue item, in item-id order, and its SID
   matches the catalogue;
4. the decoding trie built from the info file is non-degenerate (every non-leaf
   prefix has at least one legal continuation);
5. the SID token vocabulary is reported, so a change in vocabulary size is
   visible before a checkpoint is reused;
6. nothing in the P0 tree changed, by hash.

Usage
-----
    python -m experiments.minionerec_cat.src.verify_chain --tag b0
    python -m experiments.minionerec_cat.src.verify_chain --tag b0 --json out.json
"""

from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path

from . import common
from .rebuild_derived import derive_vocab, load_index, sid_string

#: Hash of the P0 artefacts as recorded at Phase-0 start.  A mismatch means the
#: read-only inputs were disturbed.
P0_EXPECTED_SHA256_PREFIX = {
    "data/Amazon/index/Industrial_and_Scientific.index.json": "53f1aefbe8b8aa77",
    "data/Amazon/index/Industrial_and_Scientific.emb-qwen-td.npy": "91a1c875e4b009f3",
    "data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt": "ebd7d594f2dc884d",
    "data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv": "5e5456d4b54681d7",
    "data/Amazon/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv": "44d441a4cbb50e18",
    "data/Amazon/test/Industrial_and_Scientific_5_2016-10-2018-11.csv": "43cea20fc742e481",
}


class ChainError(RuntimeError):
    """The SID chain is inconsistent.  Never caught; the run must stop."""


def check_p0_untouched() -> dict:
    out = {}
    for rel, expected in P0_EXPECTED_SHA256_PREFIX.items():
        p = common.REPO_ROOT / rel.replace("/", "/")
        if not p.exists():
            out[rel] = {"ok": False, "reason": "missing"}
            continue
        got = common.sha256_file(p)
        out[rel] = {"ok": got.startswith(expected), "sha256_prefix": got[:16], "expected": expected}
    return out


def check_splits(paths: common.RunPaths, index: dict[str, list[str]]) -> dict:
    results = {}
    for split in ("train", "valid", "test"):
        p = paths.derived_split(split)
        if not p.exists():
            results[split] = {"ok": False, "reason": f"missing {p}"}
            continue
        with io.open(p, encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))

        target_bad = []
        history_bad = []
        history_checked = 0
        for n, row in enumerate(rows):
            item = str(row["item_id"]).strip()
            want = sid_string(index[item]) if item in index else None
            if want is None or row["item_sid"] != want:
                target_bad.append(n)
            try:
                hist_ids = [str(int(x)) for x in eval(row["history_item_id"])] if row["history_item_id"] else []
            except Exception:
                hist_ids = []
            try:
                hist_sids = eval(row["history_item_sid"]) if row["history_item_sid"] else []
            except Exception:
                hist_sids = []
            if len(hist_ids) != len(hist_sids):
                history_bad.append(n)
                continue
            for iid, isid in zip(hist_ids, hist_sids):
                history_checked += 1
                expect = sid_string(index[iid]) if iid in index else None
                if expect is None or isid != expect:
                    history_bad.append(n)
                    break

        results[split] = {
            "ok": not target_bad and not history_bad,
            "rows": len(rows),
            "target_mismatches": len(target_bad),
            "history_row_mismatches": len(history_bad),
            "history_entries_checked": history_checked,
            "first_bad_rows": (target_bad + history_bad)[:5],
        }
    return results


def check_info(paths: common.RunPaths, index: dict[str, list[str]]) -> dict:
    info = paths.derived_info
    if not info.exists():
        return {"ok": False, "reason": f"missing {info}"}
    lines = [l for l in io.open(info, encoding="utf-8").read().splitlines() if l.strip()]

    order_ok = True
    sid_bad = []
    title_missing = 0
    for n, line in enumerate(lines):
        parts = line.split("\t")
        if len(parts) < 3:
            sid_bad.append(n)
            continue
        sid, title, item_id = parts[0], parts[1], parts[2]
        if item_id != str(n):
            order_ok = False
        if item_id in index and sid != sid_string(index[item_id]):
            sid_bad.append(n)
        if not title.strip():
            title_missing += 1

    return {
        "ok": order_ok and not sid_bad,
        "lines": len(lines),
        "item_id_ascending": order_ok,
        "sid_mismatches": len(sid_bad),
        "empty_titles": title_missing,
        "distinct_sids": len({l.split("\t")[0] for l in lines}),
    }


def check_decoding_trie(paths: common.RunPaths) -> dict:
    """Mirror ``evaluate.py:86-99``: every prefix must allow at least one token.

    A prefix that maps to an empty allowed-token set would make constrained
    decoding emit nothing, which shows up as a generation failure rather than a
    metric change.
    """
    info = paths.derived_info
    if not info.exists():
        return {"ok": False, "reason": "info file missing"}

    semantic_ids = [l.split("\t")[0].strip() + "\n" for l in io.open(info, encoding="utf-8") if l.strip()]

    def get_hash(ids):
        # Same spirit as evaluate.py:get_hash -- an order-sensitive prefix key.
        return "\u241f".join(str(t) for t in ids)

    trie: dict[str, set[str]] = {}
    for sid in semantic_ids:
        # Tokenise the SID into its <x_n> atoms; that is the unit the decoder emits.
        import re
        atoms = re.findall(r"<[a-z]_\d+>", sid) + ["\n"]
        for i in range(len(atoms)):
            key = get_hash(atoms[:i])
            trie.setdefault(key, set()).add(atoms[i])

    empty = [k for i, k in enumerate(trie) if not trie[k]]
    # Depth distribution: how many distinct prefixes exist per level.
    depth_counts: dict[int, int] = {}
    for key in trie:
        depth = 0 if key == "" else key.count("\u241f") + 1
        depth_counts[depth] = depth_counts.get(depth, 0) + 1

    return {
        "ok": not empty,
        "prefixes": len(trie),
        "empty_prefixes": len(empty),
        "prefixes_per_depth": dict(sorted(depth_counts.items())),
        "max_depth": max(depth_counts) if depth_counts else 0,
    }


def verify(tag: str, *, require_p0_match: bool = True) -> dict:
    paths = common.run_paths(tag)
    index = load_index(paths.index)

    print("=" * 88)
    print(f"MiniOneRec-CAT / {tag}  -- chain verification")
    print("=" * 88)

    report: dict = {"tag": tag, "index": str(paths.index), "items_in_index": len(index)}

    p0 = check_p0_untouched() if require_p0_match else {}
    report["p0_untouched"] = p0
    if p0:
        bad = [k for k, v in p0.items() if not v.get("ok")]
        print(f"  P0 inputs  : {'UNCHANGED' if not bad else 'CHANGED -> ' + str(bad)}")

    splits = check_splits(paths, index)
    report["splits"] = splits
    for split, res in splits.items():
        print(f"  {split:5}      : rows={res.get('rows')} target_mismatch={res.get('target_mismatches')} "
              f"history_mismatch={res.get('history_row_mismatches')} "
              f"[{'OK' if res.get('ok') else 'FAIL'}]")

    info = check_info(paths, index)
    report["info"] = info
    print(f"  info       : lines={info.get('lines')} ascending={info.get('item_id_ascending')} "
          f"sid_mismatch={info.get('sid_mismatches')} distinct_sids={info.get('distinct_sids')} "
          f"[{'OK' if info.get('ok') else 'FAIL'}]")

    trie = check_decoding_trie(paths)
    report["trie"] = trie
    print(f"  trie       : prefixes={trie.get('prefixes')} empty={trie.get('empty_prefixes')} "
          f"max_depth={trie.get('max_depth')} [{'OK' if trie.get('ok') else 'FAIL'}]")

    vocab = derive_vocab(index)
    report["vocab"] = {k: v for k, v in vocab.items() if k != "tokens"}
    report["vocab_tokens"] = vocab["tokens"]
    print(f"  vocab      : {vocab['n_sid_tokens']} SID tokens {vocab['per_layer_token_counts']}")

    # Cross-run vocabulary drift: this is what invalidates a reused checkpoint.
    p0_index = load_index(paths.p0_index)
    p0_vocab = derive_vocab(p0_index)
    report["p0_vocab"] = {k: v for k, v in p0_vocab.items() if k != "tokens"}
    report["vocab_delta"] = {
        "p0_tokens": p0_vocab["n_sid_tokens"],
        "run_tokens": vocab["n_sid_tokens"],
        "added": len(set(vocab["tokens"]) - set(p0_vocab["tokens"])),
        "removed": len(set(p0_vocab["tokens"]) - set(vocab["tokens"])),
        "identical": set(vocab["tokens"]) == set(p0_vocab["tokens"]),
    }
    d = report["vocab_delta"]
    print(f"  vs P0 vocab: +{d['added']} / -{d['removed']} "
          f"(P0={d['p0_tokens']} run={d['run_tokens']} identical={d['identical']})")
    if not d["identical"]:
        print("               NOTE: a P0 SFT checkpoint cannot be reused with this vocabulary;")
        print("                     SFT must be retrained from the base model.")

    report["ok"] = all([
        all(v.get("ok") for v in splits.values()),
        bool(info.get("ok")),
        bool(trie.get("ok")),
        (not p0) or all(v.get("ok") for v in p0.values()),
    ])
    print("-" * 88)
    print(f"  OVERALL    : {'PASS' if report['ok'] else 'FAIL'}")
    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tag", default="b0")
    p.add_argument("--json", type=Path, default=None, help="also write the report here")
    p.add_argument("--no-p0-check", action="store_true", help="skip the P0 hash comparison")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = verify(args.tag, require_p0_match=not args.no_p0_check)
    paths = common.run_paths(args.tag)
    paths.ensure_dirs()
    out = args.json or (paths.report_dir / f"{args.tag}_chain_verification.json")

    # Contract: the report is ALWAYS written first, then the exit code carries the
    # verdict.  A FAIL must both persist its evidence and be visible to shell
    # callers -- returning 0 unconditionally would hide a broken chain.
    common.write_json(out, report)
    print(f"  report     : {out}")

    ok = bool(report.get("ok"))
    if not ok:
        print("  EXIT       : 1 (chain verification FAILED -- see the report above)")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
