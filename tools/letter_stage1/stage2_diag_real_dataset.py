#!/usr/bin/env python3
"""Definitive test using the REAL EvalSidDataset input_ids.

No more reasoning: build the actual dataset the evaluation uses, take the actual
input_ids, and check whether sent[-pi:] hashes to the trie root key that the
official trie builder produces. Run for P0 (depth 3) and TX (depth 4).
"""
import hashlib
import json
import os
import sys

from transformers import AutoTokenizer

sys.path.insert(0, os.getcwd())
from sft import TokenExtender
from data import EvalSidDataset

MODEL = "/root/autodl-tmp/models/Qwen2.5-0.5B"
CAT = "Industrial_and_Scientific"
TEST = f"data/Amazon/test/{CAT}_5_2016-10-2018-11.csv"
PATHS = {"P0-3level": "data/Amazon/index/Industrial_and_Scientific.index.json",
         "TX-4level": "artifacts/letter_stage2/letter_index.json"}
SCRATCH = "/tmp/pi_scratch"


def gh(t):
    return hashlib.md5(str(list(t)).encode()).hexdigest()


def build_tok(path):
    tok = AutoTokenizer.from_pretrained(MODEL)
    os.makedirs(SCRATCH, exist_ok=True)
    link = os.path.join(SCRATCH, f"{CAT}.index.json")
    if os.path.islink(link) or os.path.exists(link):
        os.remove(link)
    os.symlink(os.path.abspath(path), link)
    tok.add_tokens(sorted(TokenExtender(data_path=SCRATCH, dataset=CAT).get_new_tokens()))
    return tok


def main():
    print("=" * 88)
    print("REAL EvalSidDataset input_ids vs OFFICIAL trie root key")
    print("=" * 88)
    for tag, path in PATHS.items():
        tok = build_tok(path)
        idx = json.load(open(path, encoding="utf-8"))
        seqs = ["".join(idx[str(i)]) for i in range(len(idx))]
        depth = len(tok(seqs[0]).input_ids)
        print(f"\n--- {tag}: depth={depth}")

        # REAL evaluation dataset (test=True path, returns input_ids only)
        ds = EvalSidDataset(train_file=TEST, tokenizer=tok, max_len=2048,
                            test=True, category="industrial and scientific items")
        ex = ds[0]
        ids = list(ex["input_ids"])
        print(f"    real EvalSidDataset[0] input_ids len = {len(ids)}")
        print(f"    last 6 tokens = {tok.convert_ids_to_tokens(ids[-6:])}")
        print(f"    target sid    = {seqs[0]!r} -> {tok(seqs[0]).input_ids}")

        for pi in (2, 3, 4, 5):
            # official trie build
            hd = {}
            for s in seqs:
                ID = list(tok(s).input_ids) + [tok.eos_token_id]
                for i in range(pi, len(ID)):
                    hn = gh(ID[:i]) if i == pi else gh(ID[pi:i])
                    hd.setdefault(hn, set()).add(ID[i])
            root = hd.get(gh(ids[-pi:]), None)
            print(f"    pi={pi}: keys={len(hd):>6d}  root(sent[-{pi}:])="
                  f"{'HIT' if root else 'MISS':4s}  -> {root if root else ''}")

    print()
    print("  The prompt's last tokens come from the instruction text, so the decoder's")
    print("  step-0 hash key depends on them -- that is what prefix_index must match.")


if __name__ == "__main__":
    main()
