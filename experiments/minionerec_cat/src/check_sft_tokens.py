"""MiniOneRec-CAT / Phase 3, review item D -- REAL tokenisation of the SFT dataset.

Why the earlier check was insufficient
--------------------------------------
I previously compared only the *SID string* structure (3 tokens, identical history
length histogram across arms) and concluded truncation exposure was equal.  That is
a proxy: the SFT datasets tokenise a full instruction prompt plus the item titles and
SID atoms, so the true sequence length depends on the tokenizer, on the per-arm SID
token strings, and on the title text.  This script measures it directly.

What it does
------------
Builds the THREE datasets sft.py actually concatenates, with the real tokenizer and
the real per-arm artefacts, and reports per arm:

  * the full prompt+target token-length distribution
  * the exact truncation RATE at cutoff_len (how many samples exceed it)
  * the number of target tokens LOST to truncation (labels[-max_len:] can drop the
    supervised tail, which is the failure that actually damages training)
  * the train row total, to confirm the composition figure independently

Requirements: the Qwen2.5-0.5B tokenizer directory.  It is not available on this
machine, so the script reports that as a blocker instead of guessing.

Usage
-----
    python -m experiments.minionerec_cat.src.check_sft_tokens --tokenizer <dir>
    python -m experiments.minionerec_cat.src.check_sft_tokens --tokenizer <dir> --cutoff 512
"""

from __future__ import annotations

import argparse
import io
import json
import statistics
import sys
from pathlib import Path

from . import common

REPO = Path(__file__).resolve().parents[3]
BASE = "Industrial_and_Scientific_5_2016-10-2018-11.csv"

#: tag -> (arm label).  These are the three arms under test in phase 3.
ARMS = [("S0", "b0"), ("S1", "b2_lam010"), ("S2", "b2")]


def _import_datasets():
    """sft.py's dataset classes live at the repo root and need ``datasets``."""
    sys.path.insert(0, str(REPO))
    try:
        import datasets  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"BLOCKER: `datasets` is not importable ({exc!r}). sft.py imports it at "
            f"line 8, so the real SFT datasets cannot be constructed here."
        )
    from data import FusionSeqRecDataset, SidItemFeatDataset, SidSFTDataset
    return SidSFTDataset, SidItemFeatDataset, FusionSeqRecDataset


def _lengths(ds):
    """Token length and supervised-tail length for every sample of a dataset."""
    out = []
    for item in ds.inputs:
        if item is None:
            continue
        ids = item["input_ids"]
        labels = item.get("labels")
        n_sup = sum(1 for l in labels if l != -100) if labels else 0
        out.append((len(ids), n_sup))
    return out


def measure(tag: str, tokenizer_dir: Path, cutoff: int) -> dict:
    SidSFTDataset, SidItemFeatDataset, FusionSeqRecDataset = _import_datasets()
    from transformers import AutoTokenizer

    paths = common.run_paths(tag)
    tok = AutoTokenizer.from_pretrained(str(tokenizer_dir), trust_remote_code=True)
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"

    index_file = paths.index
    # replicate sft.py:149-159 -- register this arm's SID tokens
    idx = json.loads(Path(index_file).read_text(encoding="utf-8"))
    new_tokens = sorted({t for sids in idx.values() for t in sids})
    n_before = len(tok)
    tok.add_tokens(new_tokens)
    n_after = len(tok)
    eos_id = tok.eos_token_id

    train_csv = paths.derived_split("train")
    item_file = common.P0_DATA / "index" / "Industrial_and_Scientific.item.json"

    datasets = {
        "SidSFTDataset": SidSFTDataset(
            train_file=str(train_csv), tokenizer=tok, max_len=cutoff,
            sample=-1, seed=42, category="industrial and scientific items"),
        "SidItemFeatDataset": SidItemFeatDataset(
            item_file=str(item_file), index_file=str(index_file), tokenizer=tok,
            max_len=cutoff, sample=-1, seed=42,
            category="industrial and scientific items"),
        "FusionSeqRecDataset": FusionSeqRecDataset(
            train_file=str(train_csv), item_file=str(item_file),
            index_file=str(index_file), tokenizer=tok, max_len=cutoff,
            sample=-1, seed=42, category="industrial and scientific items"),
    }

    per_ds = {}
    all_lengths = []
    all_sup = []
    for name, ds in datasets.items():
        lens = _lengths(ds)
        lengths = [l for l, _ in lens]
        sup = [s for _, s in lens]
        # truncation: sequence longer than cutoff -> the head is dropped by
        # input_ids[-max_len:], so the PROMPT is truncated, not the target...
        over = sum(1 for l in lengths if l > cutoff)
        # ...but labels[-max_len:] can still cut the supervised tail if the
        # target sits beyond the window.  Detect samples whose supervised tokens
        # would be clipped.
        tail_lost = sum(1 for l, s in lens if s and l > cutoff and (l - cutoff) > (l - s))
        per_ds[name] = {
            "n_samples": len(lengths),
            "len_min": min(lengths) if lengths else None,
            "len_max": max(lengths) if lengths else None,
            "len_mean": round(statistics.fmean(lengths), 2) if lengths else None,
            "len_p50": int(statistics.median(lengths)) if lengths else None,
            "len_p95": (int(sorted(lengths)[int(0.95 * (len(lengths) - 1))])
                        if lengths else None),
            "len_p99": (int(sorted(lengths)[int(0.99 * (len(lengths) - 1))])
                        if lengths else None),
            "over_cutoff": over,
            "over_cutoff_rate": over / len(lengths) if lengths else None,
            "supervised_tail_lost": tail_lost,
            "supervised_tokens_mean": round(statistics.fmean(sup), 3) if sup else None,
        }
        all_lengths.extend(lengths)
        all_sup.extend(sup)

    n = len(all_lengths)
    over = sum(1 for l in all_lengths if l > cutoff)
    return {
        "tag": tag,
        "cutoff_len": cutoff,
        "tokenizer_dir": str(tokenizer_dir),
        "vocab_before_add": n_before,
        "vocab_after_add": n_after,
        "sid_tokens_added": len(new_tokens),
        "eos_token_id": eos_id,
        "datasets": per_ds,
        "combined": {
            "n_samples": n,
            "len_min": min(all_lengths),
            "len_max": max(all_lengths),
            "len_mean": round(statistics.fmean(all_lengths), 2),
            "len_p50": int(statistics.median(all_lengths)),
            "len_p95": int(sorted(all_lengths)[int(0.95 * (n - 1))]),
            "len_p99": int(sorted(all_lengths)[int(0.99 * (n - 1))]),
            "over_cutoff": over,
            "over_cutoff_rate": over / n,
            "supervised_tokens_mean": round(statistics.fmean(all_sup), 3),
        },
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tokenizer", type=Path, required=True,
                   help="path to the Qwen2.5-0.5B tokenizer directory")
    p.add_argument("--cutoff", type=int, default=512)
    p.add_argument("--tags", nargs="+", default=None)
    args = p.parse_args(argv)

    if not Path(args.tokenizer).exists():
        raise SystemExit(f"BLOCKER: tokenizer directory not found: {args.tokenizer}")

    arms = [(l, t) for l, t in ARMS if args.tags is None or t in args.tags]
    out = {}
    print("=" * 100)
    print("D. REAL tokenisation length / truncation per arm")
    print("=" * 100)
    for label, tag in arms:
        r = measure(tag, args.tokenizer, args.cutoff)
        out[tag] = r
        c = r["combined"]
        print(f"\n  --- {label} ({tag})  cutoff={args.cutoff} ---")
        print(f"    vocab {r['vocab_before_add']} -> {r['vocab_after_add']} "
              f"(+{r['sid_tokens_added']} SID tokens)")
        for name, d in r["datasets"].items():
            print(f"    {name:<20} n={d['n_samples']:>6,}  "
                  f"len p50={d['len_p50']} p95={d['len_p95']} p99={d['len_p99']} "
                  f"max={d['len_max']}  over={d['over_cutoff']} "
                  f"({100*(d['over_cutoff_rate'] or 0):.3f}%)  "
                  f"sup_tail_lost={d['supervised_tail_lost']}")
        print(f"    COMBINED             n={c['n_samples']:>6,}  "
              f"len p50={c['len_p50']} p95={c['len_p95']} p99={c['len_p99']} "
              f"max={c['len_max']}")
        print(f"    truncation rate      {c['over_cutoff']} / {c['n_samples']} "
              f"= {100*c['over_cutoff_rate']:.4f}%")
        print(f"    mean supervised tokens {c['supervised_tokens_mean']}")

    dest = common.EXP_ROOT / "_comparison" / "p3_sft_token_lengths.json"
    common.write_json(dest, out)
    print(f"\n  written: {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
