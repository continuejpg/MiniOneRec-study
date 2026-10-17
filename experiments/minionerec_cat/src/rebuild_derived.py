"""MiniOneRec-CAT / rebuild_derived -- propagate a new SID catalogue through EVERY consumer.

The problem this solves
-----------------------
``index.json`` is only the *source* of the SID assignment.  Four other artefacts
are derived from it, and the pipeline breaks silently if any one of them is stale:

===========================  =====================================================
artefact                     consumer / why it must be regenerated
===========================  =====================================================
``train/*.csv``              ``data.SidSFTDataset`` reads ``item_sid`` and
                             ``history_item_sid`` (``data.py:404,413``)
``valid/*.csv``              ``sft.py:204`` builds the eval set from it
``test/*.csv``               ``evaluate.py:143`` builds the scored cohort
``info/*.txt``               ``evaluate.py:61-99`` builds the **decoding trie**
                             and ``calc.py:23`` builds the **metric universe**
===========================  =====================================================

Replacing ``index.json`` alone therefore leaves the model training on new SIDs
while decoding is constrained to old ones -- it fails silently and looks like a
modelling result.

What this module does
---------------------
Rewrites the three split CSVs and the info file from a SID catalogue, changing
**only** the two SID columns.  Every other byte-relevant field (``user_id``,
``item_id``, ``history_item_id``, the two title columns, and the row order) is
carried through untouched, and the row count is asserted equal.

It deliberately does **not** go through upstream ``convert_dataset.py``: that
script re-derives the splits from ``*.inter`` files which are not in this
repository, and re-running it would silently change the split instead of only
re-lettering the SIDs.

Usage
-----
    python -m experiments.minionerec_cat.src.rebuild_derived --tag b0
    python -m experiments.minionerec_cat.src.rebuild_derived --tag b0 --index <path>
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
from pathlib import Path

from . import common

#: The columns a SID catalogue is allowed to change.
SID_COLUMNS = ("item_sid", "history_item_sid")

#: Columns that must survive byte-for-byte.
PRESERVED_COLUMNS = ("user_id", "item_title", "history_item_id", "item_id", "history_item_title")

_TOKEN_RE = re.compile(r"<([a-z])_(\d+)>")


def load_index(path: Path) -> dict[str, list[str]]:
    """Read a SID catalogue and validate its shape."""
    if not path.exists():
        raise FileNotFoundError(f"SID catalogue not found: {path}")
    raw = common.read_json(path)
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"{path} is not a non-empty JSON object")
    out: dict[str, list[str]] = {}
    for key, value in raw.items():
        if not str(key).isdigit():
            raise ValueError(f"{path}: key {key!r} is not an integer item id")
        if not isinstance(value, list) or not value:
            raise ValueError(f"{path}: value for item {key} must be a non-empty token list")
        for token in value:
            if not isinstance(token, str) or not _TOKEN_RE.fullmatch(token):
                raise ValueError(f"{path}: item {key} has a malformed SID token {token!r}")
        out[str(int(key))] = [str(t) for t in value]
    return out


def sid_string(tokens: list[str]) -> str:
    """Concatenate tokens exactly as ``convert_dataset.semantic_tokens_to_id`` does."""
    return "".join(tokens)


# --------------------------------------------------------------------------- #
# Split CSVs
# --------------------------------------------------------------------------- #

def rebuild_split(src: Path, dst: Path, index: dict[str, list[str]]) -> dict:
    """Rewrite one split CSV with the new SIDs, changing nothing else."""
    if not src.exists():
        raise FileNotFoundError(f"P0 split not found: {src}")

    with io.open(src, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    missing = [c for c in SID_COLUMNS if c not in fieldnames]
    if missing:
        raise ValueError(f"{src}: missing SID column(s) {missing}; got {fieldnames}")
    for c in PRESERVED_COLUMNS:
        if c not in fieldnames:
            raise ValueError(f"{src}: missing preserved column {c!r}; got {fieldnames}")

    unknown_targets: list[int] = []
    unknown_history = 0
    n_rows = len(rows)

    for row in rows:
        target = str(row["item_id"]).strip()
        if target not in index:
            unknown_targets.append(int(target))
            row["item_sid"] = ""
        else:
            row["item_sid"] = sid_string(index[target])

        history_raw = row["history_item_id"]
        try:
            history = [str(int(x)) for x in eval(history_raw)] if history_raw else []
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"{src}: unparsable history_item_id {history_raw!r}: {exc}") from exc

        new_history: list[str] = []
        for item in history:
            if item in index:
                new_history.append(sid_string(index[item]))
            else:
                unknown_history += 1
        # Repr of a list of strings, matching pandas' object-column serialisation.
        row["history_item_sid"] = repr(new_history)

    out = common.guard_write(dst)
    with io.open(out, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Row count must be conserved: the SID change is a re-lettering, not a filter.
    with io.open(out, encoding="utf-8", newline="") as fh:
        n_out = sum(1 for _ in csv.DictReader(fh))
    if n_out != n_rows:
        raise RuntimeError(f"{out}: wrote {n_out} rows, expected {n_rows}")

    return {
        "source": str(src),
        "output": str(out),
        "rows": n_rows,
        "rows_after": n_out,
        "unknown_target_items": len(unknown_targets),
        "unknown_target_examples": sorted(set(unknown_targets))[:10],
        "unknown_history_refs": unknown_history,
        "sha256": common.sha256_file(out),
    }


# --------------------------------------------------------------------------- #
# info file -- the decoding trie AND calc.py's metric universe
# --------------------------------------------------------------------------- #

def rebuild_info(item_path: Path, dst: Path, index: dict[str, list[str]]) -> dict:
    """``SID<TAB>title<TAB>item_id`` in item-id order, one line per catalogue item.

    Line order defines ``calc.py``'s ``item_ids = range(len(item_names))``, so the
    order must follow the numeric item id, not dict iteration order.
    """
    items = common.read_json(item_path)
    if not isinstance(items, dict):
        raise ValueError(f"{item_path} is not an object")

    ids = sorted((int(k) for k in items), key=int)
    lines: list[str] = []
    missing: list[int] = []
    for i in ids:
        key = str(i)
        sid = index.get(key)
        if sid is None:
            missing.append(i)
            continue
        title = items[key].get("title", f"Item_{key}")
        lines.append(f"{sid_string(sid)}\t{title}\t{key}")

    out = common.guard_write(dst)
    with io.open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    sids = [l.split("\t")[0] for l in lines]
    return {
        "source": str(item_path),
        "output": str(out),
        "lines": len(lines),
        "catalogue_items": len(ids),
        "missing_from_index": len(missing),
        "missing_examples": missing[:10],
        "distinct_sids": len(set(sids)),
        "duplicate_sid_lines": len(sids) - len(set(sids)),
        "sha256": common.sha256_file(out),
    }


# --------------------------------------------------------------------------- #
# Vocabulary -- what ``sft.py``'s TokenExtender will add to the tokenizer
# --------------------------------------------------------------------------- #

def derive_vocab(index: dict[str, list[str]]) -> dict:
    """Reproduce ``sft.TokenExtender.get_new_tokens`` (``sft.py:49-53``) exactly."""
    tokens: set[str] = set()
    for value in index.values():
        tokens.update(value)
    per_layer: dict[str, int] = {}
    for token in tokens:
        m = _TOKEN_RE.fullmatch(token)
        if m:
            per_layer[m.group(1)] = per_layer.get(m.group(1), 0) + 1
    return {
        "n_sid_tokens": len(tokens),
        "per_layer_token_counts": dict(sorted(per_layer.items())),
        "tokens": sorted(tokens),
    }


# --------------------------------------------------------------------------- #

def rebuild(tag: str, *, index_path: Path | None = None) -> dict:
    paths = common.run_paths(tag)
    paths.ensure_dirs()
    src_index = Path(index_path) if index_path else paths.index

    print("=" * 88)
    print(f"MiniOneRec-CAT / {tag}  -- rebuild SID-derived artefacts")
    print("=" * 88)
    print(f"  SID catalogue : {src_index}")

    index = load_index(src_index)
    print(f"  items in index: {len(index):,}")

    splits = {}
    for split in ("train", "valid", "test"):
        splits[split] = rebuild_split(paths.p0_split(split), paths.derived_split(split), index)
        s = splits[split]
        print(f"  {split:5} : rows={s['rows']:,} -> {s['rows_after']:,}  "
              f"unknown_target={s['unknown_target_items']}  "
              f"unknown_history={s['unknown_history_refs']}")

    info = rebuild_info(paths.p0_item, paths.derived_info, index)
    print(f"  info  : lines={info['lines']:,}  distinct_sids={info['distinct_sids']:,}  "
          f"missing={info['missing_from_index']}  dup_lines={info['duplicate_sid_lines']}")

    vocab = derive_vocab(index)
    vocab_path = common.write_json(paths.derived_root / "sid_vocab.json", vocab)
    print(f"  vocab : {vocab['n_sid_tokens']} SID tokens  per-layer={vocab['per_layer_token_counts']}")
    print(f"          -> {vocab_path}")

    manifest = common.make_manifest(
        stage="rebuild_derived",
        tag=tag,
        config={"index_path": str(src_index)},
        inputs={
            "index": src_index,
            "p0_item": paths.p0_item,
            "p0_train": paths.p0_split("train"),
            "p0_valid": paths.p0_split("valid"),
            "p0_test": paths.p0_split("test"),
        },
        outputs=[paths.derived_split(s) for s in ("train", "valid", "test")]
        + [paths.derived_info, vocab_path],
        extra={"splits": splits, "info": info, "vocab_summary": {
            "n_sid_tokens": vocab["n_sid_tokens"],
            "per_layer_token_counts": vocab["per_layer_token_counts"],
        }},
    )
    common.write_json(paths.report_dir / f"{tag}_rebuild_manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tag", default="b0")
    p.add_argument("--index", type=Path, default=None,
                   help="SID catalogue to use; defaults to the run's own index")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rebuild(args.tag, index_path=args.index)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
