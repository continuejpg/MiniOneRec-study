#!/usr/bin/env python3
"""Measure the ACTUAL wrapper/entry tokenisation; stop assuming."""
import json
import os
import sys

from transformers import AutoTokenizer

sys.path.insert(0, os.getcwd())
from sft import TokenExtender

MODEL = "/root/autodl-tmp/models/Qwen2.5-0.5B"
CAT = "Industrial_and_Scientific"
SCRATCH = "/tmp/pi_scratch"
CASES = {"P0": "data/Amazon/index/Industrial_and_Scientific.index.json",
         "LETTER": "artifacts/letter_stage2/letter_index.json"}

WRAPPERS = ["### Response:\n", "### Response:", "### Response", "### Response: \n"]


def load(path):
    tok = AutoTokenizer.from_pretrained(MODEL)
    os.makedirs(SCRATCH, exist_ok=True)
    link = os.path.join(SCRATCH, f"{CAT}.index.json")
    if os.path.islink(link) or os.path.exists(link):
        os.remove(link)
    os.symlink(os.path.abspath(path), link)
    tok.add_tokens(sorted(TokenExtender(data_path=SCRATCH, dataset=CAT).get_new_tokens()))
    return tok


for tag, path in CASES.items():
    tok = load(path)
    idx = json.load(open(path, encoding="utf-8"))
    sid = "".join(idx["0"])
    sid_ids = tok(sid, add_special_tokens=False).input_ids
    depth = len(sid_ids)
    print("=" * 80)
    print(f"{tag}: SID={sid!r}  bare tokens={sid_ids}  depth={depth}")
    full = f"### Response:\n{sid}"
    fids = tok(full, add_special_tokens=False).input_ids
    print(f"  full entry tokens = {fids}  n={len(fids)}")
    print(f"  decoded pieces    = {tok.convert_ids_to_tokens(fids)}")
    print(f"  implied wrapper len = {len(fids) - depth}")
    for w in WRAPPERS:
        wids = tok(w, add_special_tokens=False).input_ids
        print(f"    wrapper {w!r:20s} -> {len(wids)} tokens {tok.convert_ids_to_tokens(wids)}")
    # what if we tokenize wrapper and sid SEPARATELY and concatenate?
    wid = tok("### Response:\n", add_special_tokens=False).input_ids
    print(f"  concat(encode(wrapper), encode(sid)) len = {len(wid)+depth}")
    print()
