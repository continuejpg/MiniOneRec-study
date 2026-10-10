#!/usr/bin/env python3
"""Measure SidItemFeatDataset size for P0 vs LETTER. No assumptions."""
import os
import sys

from transformers import AutoTokenizer

sys.path.insert(0, os.getcwd())
from data import SidItemFeatDataset

BASE = "/root/autodl-tmp/models/Qwen2.5-0.5B"
CATEGORY = "Industrial_and_Scientific"
STEM = f"{CATEGORY}_5_2016-10-2018-11"

CASES = {
    "P0": {
        "item": f"data/Amazon/index/{CATEGORY}.item.json",
        "index": f"data/Amazon/index/{CATEGORY}.index.json",
        "tok": "runs/industrial_sft/final_checkpoint",
    },
    "LETTER": {
        "item": f"artifacts/letter_stage3/data/index/{CATEGORY}.item.json",
        "index": f"artifacts/letter_stage3/data/index/{CATEGORY}.index.json",
        "tok": "artifacts/letter_stage3/tokenizer",
    },
}

for tag, c in CASES.items():
    tok = AutoTokenizer.from_pretrained(c["tok"])
    ds = SidItemFeatDataset(item_file=c["item"], index_file=c["index"],
                            tokenizer=tok, max_len=512, sample=-1, seed=42,
                            category="industrial and scientific items")
    print(f"  [{tag}] tok_len={len(tok):,}  SidItemFeat size = {len(ds)}")
    print(f"         item_file={c['item']}")
    print(f"         index_file={c['index']}")
