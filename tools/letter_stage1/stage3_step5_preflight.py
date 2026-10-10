#!/usr/bin/env python3
"""
Stage 3 / step 5 -- SFT preflight.

Re-confirms the depth-aware pipeline on the LETTER catalogue and then runs a
2-example forward/backward smoke on the RESIZED pristine base so that an embedding
index overflow (the Stage 2.5 failure) is caught before the 2-epoch run.

Nothing here judges recommendation quality -- the base model is untrained for the
SID task and its outputs are meaningless by construction.
"""
import ast
import json
import os
import re
import sys

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.getcwd())
from sid_utils import infer_prefix_index, build_sid_trie, get_hash, split_sid_tokens  # noqa: E402
from data import SidSFTDataset, SidItemFeatDataset, FusionSeqRecDataset  # noqa: E402

BASE = "/root/autodl-tmp/models/Qwen2.5-0.5B"
import os as _os
STAGE = _os.environ.get("PREFLIGHT_STAGE", "3")
TOKDIR = ("artifacts/letter_stage3/tokenizer" if STAGE == "3"
          else "artifacts/letter_stage4_content_only/tokenizer")
D = ("artifacts/letter_stage3/data" if STAGE == "3"
     else "artifacts/letter_stage4_content_only/data")
CATEGORY = "Industrial_and_Scientific"
STEM = f"{CATEGORY}_5_2016-10-2018-11"
WRAPPER = "### Response:\n"

fails = []


def chk(ok, label, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        fails.append(label)
    return ok


print("=" * 84)
print("STEP 5 -- SFT PREFLIGHT")
print("=" * 84)

# ---------------------------------------------------------------- tokenizer
tok = AutoTokenizer.from_pretrained(TOKDIR)
print(f"  tokenizer from {TOKDIR}  len={len(tok):,}")

info_path = f"{D}/info/{STEM}.txt"
idx_path = f"{D}/index/{CATEGORY}.index.json"
item_path = f"{D}/index/{CATEGORY}.item.json"
sids = []
with open(info_path, encoding="utf-8") as f:
    for line in f:
        if line.strip():
            sids.append(line.split("\t")[0].strip())
idx = json.load(open(idx_path, encoding="utf-8"))

# ---- every catalogue SID must be 4 atomic tokens ---------------------------
bad_len, bad_atomic = [], []
for i, s in enumerate(sids):
    enc = tok.encode(s, add_special_tokens=False)
    if len(enc) != 4:
        bad_len.append((i, len(enc)))
    if [tok.convert_ids_to_tokens(e) for e in enc] != split_sid_tokens(s):
        bad_atomic.append(i)
chk(not bad_len, "all catalogue SIDs encode to exactly 4 atomic tokens",
    f"{len(sids)} checked, {len(bad_len)} bad")
chk(not bad_atomic, "SID tokens are atomic (no subword splitting)",
    f"{len(bad_atomic)} bad")

# ---- depth inference + trie -------------------------------------------------
entries = [f"{WRAPPER}{s}" for s in sids]
# evaluate.py appends "\n" (semantic_ids = [.. + "\n"]); mirror it here
entries_nl = [f"{WRAPPER}{s}" for s in sids]
pi, depth, wlen = infer_prefix_index(entries_nl, tok, WRAPPER)
chk(depth == 4, "LETTER inferred depth == 4", f"depth={depth} prefix_index={pi}")
trie, pi2, _ = build_sid_trie(entries_nl, tok, wrapper=WRAPPER)

eos = tok.eos_token_id
hit = only_eos = 0
for s in sids:
    sid_ids = tok(s, add_special_tokens=False).input_ids
    al = trie.get(get_hash(sid_ids))
    if al is not None:
        hit += 1
        if set(al) == {eos}:
            only_eos += 1
chk(hit == len(sids), f"full SID prefix present in trie", f"{hit}/{len(sids)}")
chk(only_eos == len(sids), "full SID -> ONLY EOS", f"{only_eos}/{len(sids)}")

early = 0
for s in sids:
    sid_ids = tok(s, add_special_tokens=False).input_ids
    for k in range(depth):
        if eos in trie.get(get_hash(sid_ids[:k]), ()):
            early += 1
chk(early == 0, "no incomplete prefix emits EOS", f"{early} violations")

cat = {tuple(tok(s, add_special_tokens=False).input_ids) for s in sids}
allt = sorted({t for v in idx.values() for t in v})
viol = tested = 0
for s in sids[:200]:
    sid_ids = tok(s, add_special_tokens=False).input_ids
    for pos in range(depth):
        for cand in allt[:8]:
            ci = tok.convert_tokens_to_ids(cand)
            mut = tuple(sid_ids[:pos] + [ci] + sid_ids[pos + 1:])
            tested += 1
            if mut in cat:
                continue
            if eos in trie.get(get_hash(mut), ()):
                viol += 1
chk(viol == 0, "non-catalogue continuation cannot reach EOS",
    f"{viol} violations / {tested} mutations")

# ---------------------------------------------------------------- datasets
print()
print("  --- dataset construction: expect 36259 + SidItemFeat + 36259 ---")
d1 = SidSFTDataset(train_file=f"{D}/train/{STEM}.csv", tokenizer=tok,
                   max_len=512, sample=-1, seed=42,
                   category="industrial and scientific items")
d2 = SidItemFeatDataset(item_file=item_path, index_file=idx_path, tokenizer=tok,
                        max_len=512, sample=-1, seed=42,
                        category="industrial and scientific items")
d3 = FusionSeqRecDataset(train_file=f"{D}/train/{STEM}.csv", item_file=item_path,
                         index_file=idx_path, tokenizer=tok, max_len=512,
                         sample=-1, seed=42,
                         category="industrial and scientific items")
chk(len(d1) == 36259, "SidSFT size == 36259", str(len(d1)))
chk(len(d2) > 7000, f"SidItemFeat size = {len(d2)} (catalogue-dependent; P0 was 7316, T was 7320)", str(len(d2)))
chk(len(d3) == 36259, "FusionSeqRec size == 36259", str(len(d3)))
chk(len(d1) + len(d2) + len(d3) == 36259 + len(d2) + 36259, "total == 36259 + SidItemFeat + 36259",
    str(len(d1) + len(d2) + len(d3)))

v = SidSFTDataset(train_file=f"{D}/valid/{STEM}.csv", tokenizer=tok, max_len=512,
                  sample=-1, seed=42, category="industrial and scientific items")
chk(len(v) == 4532, "valid SidSFT size == 4532", str(len(v)))

# every training example must carry only in-range token ids
max_id = len(tok) - 1
mx = 0
for ds, nm in ((d1, "SidSFT"), (d2, "SidItemFeat"), (d3, "FusionSeqRec")):
    for k in range(min(200, len(ds))):
        ex = ds[k]
        mx = max(mx, max(ex["input_ids"]), max(ex["labels"]))
chk(mx <= max_id, "sampled example token ids within vocab",
    f"max_id={mx} < {max_id+1}")

# ---------------------------------------------------------------- forward smoke
print()
print("  --- resized-base forward/backward smoke ---")
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16)
n_add = model.resize_token_embeddings(len(tok))
in_r = model.get_input_embeddings().weight.shape[0]
out_r = model.get_output_embeddings().weight.shape[0]
chk(in_r == len(tok) and out_r == len(tok),
    "resized rows == tokenizer len", f"in={in_r} out={out_r} tok={len(tok)}")
model = model.to(dev)
model.train()

batch = [d1[0], d1[1]]
L = max(len(x["input_ids"]) for x in batch)
pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
input_ids = torch.tensor([[pad] * (L - len(x["input_ids"])) + x["input_ids"] for x in batch]).to(dev)
labels = torch.tensor([[-100] * (L - len(x["labels"])) + x["labels"] for x in batch]).to(dev)
am = torch.tensor([[0] * (L - len(x["input_ids"])) + [1] * len(x["input_ids"]) for x in batch]).to(dev)

try:
    out = model(input_ids=input_ids, attention_mask=am, labels=labels)
    loss = out.loss
    loss.backward()
    finite = torch.isfinite(loss).item()
    chk(True, "forward/backward ran without embedding index overflow", "")
    chk(finite, "loss is finite", f"loss={loss.item():.6f}")
    grad_ok = any(p.grad is not None and torch.isfinite(p.grad).all().item()
                  for p in model.parameters() if p.requires_grad)
    chk(grad_ok, "gradients finite")
    print(f"    batch={len(batch)} seq_len={L} vocab={len(tok)} "
          f"loss={loss.item():.6f}")
except RuntimeError as e:  # noqa: BLE001
    chk(False, "forward/backward ran without embedding index overflow",
        f"{type(e).__name__}: {str(e)[:200]}")

print()
print(f"  failures = {fails}")
print(f"  RESULT: {'ALL PASS' if not fails else 'FAIL'}")
sys.exit(0 if not fails else 1)
