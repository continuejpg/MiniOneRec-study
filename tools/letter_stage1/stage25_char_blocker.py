#!/usr/bin/env python3
"""
Stage 2.5 -- characterise the LETTER smoke blocker precisely and size the fix.

Blocker: evaluate.py loads model AND tokenizer from the same --base_model. The P0
SFT checkpoint has an embedding table of 152,225 rows. Registering LETTER's SID
tokens pushes the tokenizer to 152,543, so 318 token ids fall outside the table
and the CUDA embedding lookup asserts.

This script produces (a) the exact out-of-range id list and (b) the token-budget
arithmetic needed by whichever remap strategy is chosen next stage.
"""
import json
import os
import sys

from transformers import AutoTokenizer

sys.path.insert(0, os.getcwd())

CKPT = "runs/industrial_sft/final_checkpoint"
MODEL_VOCAB = json.load(open(os.path.join(CKPT, "config.json"),
                             encoding="utf-8"))["vocab_size"]
BASE = "/root/autodl-tmp/models/Qwen2.5-0.5B"
TX = "artifacts/letter_stage2/letter_index.json"
P0I = "data/Amazon/index/Industrial_and_Scientific.index.json"

tok = AutoTokenizer.from_pretrained(CKPT)
print("=" * 84)
print("LETTER SID TOKEN BUDGET")
print("=" * 84)
print(f"  model embedding rows (config vocab_size) = {MODEL_VOCAB:,}")
print(f"  checkpoint tokenizer len before add      = {len(tok):,}")

base_tok = AutoTokenizer.from_pretrained(BASE)
print(f"  base Qwen2.5-0.5B tokenizer len          = {len(base_tok):,}")
print(f"  rows reserved but unused at load         = {MODEL_VOCAB - len(base_tok):,}")

tx = json.load(open(TX, encoding="utf-8"))
p0 = json.load(open(P0I, encoding="utf-8"))
tx_tok = sorted({t for v in tx.values() for t in v})
p0_tok = sorted({t for v in p0.values() for t in v})
shared = sorted(set(tx_tok) & set(p0_tok))
tx_only = sorted(set(tx_tok) - set(p0_tok))
print()
print(f"  LETTER distinct SID tokens = {len(tx_tok):,}")
print(f"    already in checkpoint tok = {len(shared):,}")
print(f"    NEW (need new rows)       = {len(tx_only):,}")

after = AutoTokenizer.from_pretrained(CKPT)
n_added = after.add_tokens(tx_tok)
over = sorted(t for t in tx_tok
              if after.convert_tokens_to_ids(t) >= MODEL_VOCAB)
print()
print(f"  add_tokens returned {n_added}; tokenizer len = {len(after):,}")
print(f"  tokens with id >= {MODEL_VOCAB:,}  = {len(over):,}")
print(f"  max LETTER token id           = {max(after.convert_tokens_to_ids(t) for t in tx_tok):,}")
print(f"  OVERFLOW (max id - vocab)     = "
      f"{max(after.convert_tokens_to_ids(t) for t in tx_tok) - MODEL_VOCAB + 1:,} rows short")

# per-level view (convert_tokens_to_ids may return None for tokens not present)
for lvl, pre in (("a", "<a_"), ("b", "<b_"), ("c", "<c_"), ("d", "<d_")):
    ids = [after.convert_tokens_to_ids(f"{pre}{i}>") for i in range(256)]
    ids = [i for i in ids if isinstance(i, int) and i >= 0]
    if not ids:
        print(f"    level {lvl}: no tokens resolvable")
        continue
    inr = sum(1 for i in ids if i < MODEL_VOCAB)
    print(f"    level {lvl}: n={len(ids):>3d}  ids {min(ids):,}..{max(ids):,}   "
          f"within model vocab = {inr}/{len(ids)}")

print()
print("=" * 84)
print("REMAP BUDGET (for the next stage, not executed here)")
print("=" * 84)
# candidate rows that a LETTER-only vocabulary could occupy
free_rows = MODEL_VOCAB - len(base_tok)
retired_p0_rows = len(p0_tok) - len(shared)   # P0 tokens LETTER does not need
print(f"  rows available = MODEL_VOCAB - base_tokenizer_len = {free_rows:,}")
print(f"  of which P0-token rows that LETTER does not reuse = {retired_p0_rows:,}")
print(f"  LETTER tokens needing a row                       = {len(tx_tok):,}")
print(f"  FIT? {len(tx_tok)} <= {free_rows}  -> {len(tx_tok) <= free_rows}")
print()
print("  NOTE: of the shared tokens only those P0 SIDs actually trained by the P0")
print("  SFT run have learned embeddings. A LETTER-only remap would retrain the")
print("  tokenizer/sft anyway, so the cheapest correct path is to retrain the")
print("  LETTER tokenizer under a vocabulary budget that fits the model.")
