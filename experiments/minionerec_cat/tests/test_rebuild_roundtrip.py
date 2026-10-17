#!/usr/bin/env python3
"""MiniOneRec-CAT / Phase-0 safety test.

Two layers, deliberately separated so the rebuild logic can be verified WITHOUT
the unpublished Amazon corpus:

  UNIT        a synthetic 4-item catalogue (one collision group, 2 splits rows) is
              pushed through ``load_index`` / ``rebuild_info`` / ``rebuild_split`` /
              ``derive_vocab``, plus the write-guard.  Every expectation is
              hand-computable, so this layer needs no external asset.  Always runs.

  INTEGRATION the real P0 assets.  The single most important correctness question
              for the whole pipeline is:

                  If I feed the **P0** SID catalogue through my rebuild chain,
                  do I get the **P0** derived files back?

              If yes, the rebuild is a faithful re-lettering and any difference in a
              B0/B2 run comes from the SID catalogue, not from the tooling.  It must
              show the info file reproduced **byte-for-byte** (and 3670 distinct
              SIDs), and the split CSVs reproduced row-for-row with the SID columns
              equal and the preserved columns equal.

              The P0 corpus is third-party data and is NOT published with this
              repository, so this layer SKIPS with an explicit reason when absent.
              A skip is never reported as a pass.

Run:
    python experiments/minionerec_cat/tests/test_rebuild_roundtrip.py
    CAT_REQUIRE_INTEGRATION=1 python .../test_rebuild_roundtrip.py   # absence = FAIL
"""

from __future__ import annotations

import csv
import io
import json
import os
import shutil
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]          # experiments/minionerec_cat
REPO = EXP.parents[1]                              # repo root
sys.path.insert(0, str(REPO))

from experiments.minionerec_cat.src import common                    # noqa: E402
from experiments.minionerec_cat.src.rebuild_derived import (         # noqa: E402
    derive_vocab, load_index, rebuild_info, rebuild_split, sid_string,
)

FAILURES: list[str] = []
SKIPPED: list[str] = []
P0_SCRATCH = common.run_paths("_selftest_p0roundtrip")


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def skip(name: str, reason: str) -> None:
    print(f"  [SKIP] {name}")
    print(f"         reason: {reason}")
    SKIPPED.append(name)


def banner(txt: str) -> None:
    print()
    print("=" * 92)
    print(txt)
    print("=" * 92)


# --------------------------------------------------------------------------- #
# UNIT layer -- synthetic catalogue
# --------------------------------------------------------------------------- #

SYN_INDEX = {
    "0": ["<a_1>", "<b_1>", "<c_1>"],
    "1": ["<a_1>", "<b_1>", "<c_2>"],
    "2": ["<a_1>", "<b_1>", "<c_2>"],     # collides with item 1
    "3": ["<a_2>", "<b_1>", "<c_1>"],
}
SYN_TITLES = {k: f"synthetic item {k}" for k in SYN_INDEX}


def _write_synthetic(root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    index_p = root / "index.json"
    item_p = root / "item.json"
    split_p = root / "split.csv"
    out_info = root / "out_info.txt"
    out_split = root / "out_split.csv"

    index_p.write_text(json.dumps(SYN_INDEX), encoding="utf-8")
    item_p.write_text(json.dumps({k: {"title": v} for k, v in SYN_TITLES.items()}),
                      encoding="utf-8")

    # history_item_id is a Python-literal list, as pandas writes it
    rows = [
        {"user_id": "u0", "item_title": SYN_TITLES["3"],
         "history_item_id": "['0', '1']", "item_id": "3",
         "history_item_title": "['synthetic item 0', 'synthetic item 1']",
         "item_sid": "", "history_item_sid": ""},
        {"user_id": "u1", "item_title": SYN_TITLES["2"],
         "history_item_id": "['3']", "item_id": "2",
         "history_item_title": "['synthetic item 3']",
         "item_sid": "", "history_item_sid": ""},
    ]
    fields = list(rows[0])
    with io.open(split_p, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return {"index": index_p, "item": item_p, "split": split_p,
            "out_info": out_info, "out_split": out_split}


def run_unit() -> None:
    root = EXP / "runs" / "_selftest_rebuild_unit"
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    try:
        fx = _write_synthetic(root)

        banner("1. UNIT -- load_index validates the catalogue")
        index = load_index(fx["index"])
        check("four items loaded", len(index) == 4, str(len(index)))
        check("keys are normalised to plain integer strings",
              sorted(index) == ["0", "1", "2", "3"], str(sorted(index)))
        check("token lists preserved",
              index["0"] == ["<a_1>", "<b_1>", "<c_1>"], str(index["0"]))

        bad = root / "bad_index.json"
        bad.write_text(json.dumps({"0": ["not-a-token"]}), encoding="utf-8")
        try:
            load_index(bad)
            check("malformed token rejected", False, "no exception")
        except ValueError:
            check("malformed token rejected", True)
        try:
            load_index(root / "does_not_exist.json")
            check("missing catalogue raises FileNotFoundError", False, "no exception")
        except FileNotFoundError:
            check("missing catalogue raises FileNotFoundError", True)

        banner("2. UNIT -- sid_string / rebuild_info")
        check("sid_string concatenates in order",
              sid_string(["<a_1>", "<b_1>", "<c_2>"]) == "<a_1><b_1><c_2>")
        info = rebuild_info(fx["item"], fx["out_info"], index)
        check("one info line per catalogue item", info["lines"] == 4, str(info["lines"]))
        check("no item missing from the index", info["missing_from_index"] == 0)
        check("distinct SIDs collapse the collision", info["distinct_sids"] == 3,
              str(info["distinct_sids"]))
        check("duplicate lines reported", info["duplicate_sid_lines"] == 1,
              str(info["duplicate_sid_lines"]))
        produced = fx["out_info"].read_text(encoding="utf-8").splitlines()
        check("format is SID<TAB>title<TAB>item_id",
              produced[0] == "<a_1><b_1><c_1>\tsynthetic item 0\t0", produced[0])
        check("lines are in ascending item-id order",
              [ln.split("\t")[2] for ln in produced] == ["0", "1", "2", "3"],
              str([ln.split("\t")[2] for ln in produced]))

        banner("3. UNIT -- rebuild_split rewrites only the SID columns")
        res = rebuild_split(fx["split"], fx["out_split"], index)
        check("row count conserved", res["rows"] == res["rows_after"] == 2,
              f"{res['rows']} -> {res['rows_after']}")
        check("no unknown target item", res["unknown_target_items"] == 0,
              str(res["unknown_target_examples"]))
        check("no unknown history reference", res["unknown_history_refs"] == 0)
        with io.open(fx["split"], encoding="utf-8", newline="") as fh:
            src_rows = list(csv.DictReader(fh))
        with io.open(fx["out_split"], encoding="utf-8", newline="") as fh:
            out_rows = list(csv.DictReader(fh))
        check("item_sid re-lettered to the target item's SID",
              out_rows[0]["item_sid"] == "<a_2><b_1><c_1>" and
              out_rows[1]["item_sid"] == "<a_1><b_1><c_2>",
              f"{out_rows[0]['item_sid']} / {out_rows[1]['item_sid']}")
        check("history_item_sid is a repr'd list of SIDs",
              out_rows[0]["history_item_sid"]
              == repr(["<a_1><b_1><c_1>", "<a_1><b_1><c_2>"]),
              out_rows[0]["history_item_sid"])
        check("preserved columns untouched",
              all(a[c] == b[c] for a, b in zip(src_rows, out_rows)
                  for c in ("user_id", "item_id", "history_item_id",
                            "item_title", "history_item_title")))

        banner("4. UNIT -- derive_vocab")
        v = derive_vocab(index)
        check("distinct SID tokens counted", v["n_sid_tokens"] == 5,
              str(v["n_sid_tokens"]))
        check("per-layer counts a/b/c == 2/1/2",
              v["per_layer_token_counts"] == {"a": 2, "b": 1, "c": 2},
              str(v["per_layer_token_counts"]))

        banner("5. UNIT -- the write guard must actually refuse")
        try:
            common.assert_outside_p0(common.REPO_ROOT / "data" / "Amazon" / "index" / "x.json")
            check("guard refuses a write inside data/Amazon", False, "no exception")
        except PermissionError:
            check("guard refuses a write inside data/Amazon", True)
        try:
            common.assert_outside_p0(common.REPO_ROOT / "checkpoints" / "x")
            check("guard refuses a write inside checkpoints/", False, "no exception")
        except PermissionError:
            check("guard refuses a write inside checkpoints/", True)
        try:
            common.assert_outside_p0(root / "ok.json")
            check("guard allows a write inside the experiment scratch", True)
        except Exception as exc:                                    # noqa: BLE001
            check("guard allows a write inside the experiment scratch", False, repr(exc))
    finally:
        shutil.rmtree(root, ignore_errors=True)


# --------------------------------------------------------------------------- #
# INTEGRATION layer -- real P0 assets (not published)
# --------------------------------------------------------------------------- #

def p0_status() -> tuple[list[str], list[str]]:
    present, missing = [], []
    for label, p in (("P0 SID catalogue", P0_SCRATCH.p0_index),
                     ("P0 item metadata", P0_SCRATCH.p0_item),
                     ("P0 info file", P0_SCRATCH.p0_info),
                     ("P0 train split", P0_SCRATCH.p0_split("train")),
                     ("P0 valid split", P0_SCRATCH.p0_split("valid")),
                     ("P0 test split", P0_SCRATCH.p0_split("test"))):
        (present if Path(p).exists() else missing).append(f"{label}: {p}")
    return present, missing


def run_integration() -> None:
    banner("6. INTEGRATION -- P0 catalogue through the rebuild chain")
    index = load_index(P0_SCRATCH.p0_index)
    print(f"  P0 index: {len(index):,} items")

    info = rebuild_info(P0_SCRATCH.p0_item, P0_SCRATCH.derived_info, index)
    src_bytes = P0_SCRATCH.p0_info.read_bytes()
    new_bytes = P0_SCRATCH.derived_info.read_bytes()
    check("info.txt byte-identical to P0", src_bytes == new_bytes,
          f"src={len(src_bytes)} new={len(new_bytes)} sha={info['sha256'][:16]}")
    check("info line count == catalogue size", info["lines"] == len(index),
          f"{info['lines']} vs {len(index)}")
    check("info has no item missing from the index", info["missing_from_index"] == 0)
    check("info SIDs distinct count matches P0", info["distinct_sids"] == 3670,
          f"got {info['distinct_sids']}")

    for split in ("train", "valid", "test"):
        src = P0_SCRATCH.p0_split(split)
        dst = P0_SCRATCH.derived_split(split)
        res = rebuild_split(src, dst, index)
        check(f"{split}: row count conserved", res["rows"] == res["rows_after"],
              f"{res['rows']} -> {res['rows_after']}")
        check(f"{split}: no target item missing from the index",
              res["unknown_target_items"] == 0, str(res["unknown_target_examples"]))
        check(f"{split}: no history reference missing from the index",
              res["unknown_history_refs"] == 0)
        with io.open(src, encoding="utf-8", newline="") as fh:
            a = list(csv.DictReader(fh))
        with io.open(dst, encoding="utf-8", newline="") as fh:
            b = list(csv.DictReader(fh))
        check(f"{split}: SID columns identical to P0",
              all(x["item_sid"] == y["item_sid"]
                  and x["history_item_sid"] == y["history_item_sid"]
                  for x, y in zip(a, b)))
        check(f"{split}: user/item/history identity identical to P0",
              all(all(x[c] == y[c] for c in ("user_id", "item_id", "history_item_id"))
                  for x, y in zip(a, b)))
        try:
            title_eq = all(
                eval(x["history_item_title"]) == eval(y["history_item_title"])
                and x["item_title"] == y["item_title"]
                for x, y in zip(a, b))
        except Exception:                                          # noqa: BLE001
            title_eq = False
        check(f"{split}: titles semantically identical to P0", title_eq)

    v_new = derive_vocab(index)
    check("vocab token count == 560 for P0", v_new["n_sid_tokens"] == 560,
          f"got {v_new['n_sid_tokens']}")
    check("vocab per-layer a/b/c == 48/256/256",
          v_new["per_layer_token_counts"] == {"a": 48, "b": 256, "c": 256},
          str(v_new["per_layer_token_counts"]))


def main() -> int:
    run_unit()

    present, missing = p0_status()
    require = os.environ.get("CAT_REQUIRE_INTEGRATION") == "1"
    banner("INTEGRATION layer availability")
    for p in present:
        print(f"  [FOUND]   {p}")
    for p in missing:
        print(f"  [MISSING] {p}")
    if missing:
        print()
        if require:
            for p in missing:
                check(f"required asset present -- {p.split(':')[0]}", False, p)
        else:
            skip("6. P0 catalogue through the rebuild chain",
                 "requires the unpublished P0 Amazon corpus (third-party data, "
                 "deliberately not redistributed)")
            print()
            print("  This integration test asserts byte-identical info reproduction and")
            print("  3670 distinct SIDs on the real P0 catalogue. It cannot pass without")
            print("  those assets and is NOT simulated. Set CAT_REQUIRE_INTEGRATION=1 to")
            print("  make their absence a failure.")
    else:
        run_integration()

    banner("RESULT")
    if FAILURES:
        print(f"RESULT: FAIL ({len(FAILURES)}) -> {FAILURES}")
        return 1
    if SKIPPED:
        print(f"RESULT: UNIT PASS -- INTEGRATION SKIPPED ({len(SKIPPED)} sections)")
        print("        every unit assertion holds; the P0 assertions were NOT run")
        return 0
    print("RESULT: PASS -- the rebuild chain faithfully reproduces the P0 derived artefacts "
          "(unit + integration)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
