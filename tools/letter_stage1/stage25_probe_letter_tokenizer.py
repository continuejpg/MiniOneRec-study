#!/usr/bin/env python3
"""Are the LETTER SID tokens representable by the P0 SFT model's vocabulary?

evaluate.py loads tokenizer AND model from the same --base_model path. The P0 SFT
checkpoint carries a tokenizer with the 560 P0 SID tokens (vocab 152225). LETTER
adds up to 841 tokens total. If any LETTER token id >= 152225 the P0 model's
embedding table cannot represent it.
"""
import json
import os
import sys

from transformers import AutoTokenizer

CKPT = "runs/industrial_sft/final_checkpoint"
TX_INDEX = "artifacts/letter_stage2/letter_index.json"
P0_INDEX = "data/Amazon/index/Industrial_and_Scientific.index.json"

tok = AutoTokenizer.from_pretrained(CKPT)
print(f"  tokenizer from {CKPT}: len={len(tok)}  vocab_size(config)={tok.vocab_size}")

cfg = json.load(open(os.path.join(CKPT, "config.json"), encoding="utf-8"))
print(f"  model config vocab_size = {cfg.get('vocab_size')}")

added = tok.get_added_vocab()
print(f"  added tokens in checkpoint tokenizer = {len(added)}")

tx = json.load(open(TX_INDEX, encoding="utf-8"))
p0 = json.load(open(P0_INDEX, encoding="utf-8"))
tx_tokens = sorted({t for v in tx.values() for t in v})
p0_tokens = sorted({t for v in p0.values() for t in v})
print(f"  LETTER distinct SID tokens = {len(tx_tokens)}")
print(f"  P0     distinct SID tokens = {len(p0_tokens)}")
print(f"  shared                     = {len(set(tx_tokens) & set(p0_tokens))}")
print(f"  LETTER-only                = {len(set(tx_tokens) - set(p0_tokens))}")

print()
print("  --- how does the checkpoint tokenizer encode LETTER SIDs? ---")
for s in ["<a_48>", "<b_191>", "<c_62>", "<d_198>",
          "<a_236>", "<b_231>", "<c_226>"]:
    ids = tok(s, add_special_tokens=False).input_ids
    print(f"    {s:10s} -> {len(ids)} token(s) {ids[:6]}{'...' if len(ids)>6 else ''}")

print()
print("  --- after add_tokens(LETTER tokens): what ids do they get? ---")
n_new = tok.add_tokens(tx_tokens)
print(f"    add_tokens returned {n_new}; len now {len(tok)}")
mx = 0
over = []
for s in tx_tokens:
    i = tok.convert_tokens_to_ids(s)
    mx = max(mx, i)
    if i >= cfg.get("vocab_size", len(tok)):
        over.append((s, i))
print(f"    max LETTER token id = {mx}")
print(f"    model config vocab_size = {cfg.get('vocab_size')}")
print(f"    tokens OUT OF MODEL RANGE (>= config vocab_size) = {len(over)}")
if over:
    print(f"      examples: {over[:5]}")
print()
depth_ids = [tok.convert_tokens_to_ids(f"<d_{i}>") for i in range(256)]
print(f"    <d_*> ids min={min(depth_ids)} max={max(depth_ids)}")
inrange = sum(1 for i in depth_ids if i < cfg.get("vocab_size"))
print(f"    <d_*> within model vocab = {inrange}/256")
