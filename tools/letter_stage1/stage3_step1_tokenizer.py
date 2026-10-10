#!/usr/bin/env python3
"""
Stage 3 / step 1 -- fix the LETTER tokenizer from the PRISTINE Qwen2.5-0.5B base.

Deliberately does NOT load the P0 SFT tokenizer. Reports the exact token-budget
arithmetic required by the brief, and asserts that every catalogue SID encodes to
exactly 4 atomic tokens.
"""
import hashlib
import json
import os
import sys

from transformers import AutoTokenizer, AutoModelForCausalLM

BASE = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/models/Qwen2.5-0.5B"
INDEX = sys.argv[2] if len(sys.argv) > 2 else "artifacts/letter_stage2/letter_index.json"
OUT = sys.argv[3] if len(sys.argv) > 3 else "artifacts/letter_stage3"
os.makedirs(OUT, exist_ok=True)


def sha256(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


print("=" * 84)
print("STEP 1 -- LETTER tokenizer from pristine base")
print("=" * 84)
print(f"  base model = {BASE}")

if not os.path.isdir(BASE):
    print(f"REFUSE: {BASE} is not a directory")
    sys.exit(1)

# ---- base model / tokenizer file hashes -----------------------------------
files = {}
for name in ("config.json", "tokenizer.json", "tokenizer_config.json",
             "vocab.json", "merges.txt", "model.safetensors", "generation_config.json"):
    p = os.path.join(BASE, name)
    if os.path.exists(p):
        files[name] = {"bytes": os.path.getsize(p), "sha256": sha256(p)}
for k, v in sorted(files.items()):
    print(f"    {k:26s} {v['bytes']:>14,d} B  {v['sha256'][:16]}")

TOK = AutoTokenizer.from_pretrained(BASE)
base_len = len(TOK)
print(f"\n  base tokenizer len            = {base_len:,}")

# ---- collect LETTER SID tokens -------------------------------------------
idx = json.load(open(INDEX, encoding="utf-8"))
N = len(idx)
sids = {int(k): "".join(v) for k, v in idx.items()}
assert sorted(sids) == list(range(N)), "index id space is not 0..N-1"

all_tokens = sorted({t for s in sids.values() for t in
                     __import__("re").findall(r"<[^<>]+>", s)})
print(f"  LETTER distinct SID tokens    = {len(all_tokens):,}")

already = [t for t in all_tokens if TOK.convert_tokens_to_ids(t) is not None
           and TOK.convert_tokens_to_ids(t) != TOK.unk_token_id]
print(f"  already-existing in base      = {len(already):,}")

n_added = TOK.add_tokens(all_tokens)
final_len = len(TOK)
print(f"  newly added                   = {n_added:,}")
print(f"  final tokenizer len           = {final_len:,}")

ids = [TOK.convert_tokens_to_ids(t) for t in all_tokens]
print(f"  SID token id range            = {min(ids):,} .. {max(ids):,}")
print(f"  max SID token id              = {max(ids):,}")

# ---- every SID must be exactly 4 atomic tokens ---------------------------
bad_len, bad_atomic = [], []
for i in range(N):
    enc = TOK.encode(sids[i], add_special_tokens=False)
    if len(enc) != 4:
        bad_len.append((i, len(enc)))
    levels = __import__("re").findall(r"<[^<>]+>", sids[i])
    if [TOK.convert_ids_to_tokens(e) for e in enc] != levels:
        bad_atomic.append(i)

print()
print(f"  SIDs with != 4 tokens         = {len(bad_len)}   {bad_len[:5]}")
print(f"  SIDs whose tokens != the 4 atomic levels = {len(bad_atomic)}")

# ---- model + resize ------------------------------------------------------
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype="auto")
in_rows = model.get_input_embeddings().weight.shape[0]
out_emb = model.get_output_embeddings()
out_rows = out_emb.weight.shape[0] if out_emb is not None else None
tied = (model.get_input_embeddings().weight.data_ptr()
        == getattr(out_emb, "weight", None).data_ptr()) if out_emb is not None else None
cfg_vocab = model.config.vocab_size

print()
print(f"  BEFORE resize: config.vocab_size={cfg_vocab:,} "
      f"input_rows={in_rows:,} output_rows={out_rows:,} tied={tied}")
model.resize_token_embeddings(final_len)
in_rows2 = model.get_input_embeddings().weight.shape[0]
out_rows2 = model.get_output_embeddings().weight.shape[0]
print(f"  AFTER  resize: input_rows={in_rows2:,} output_rows={out_rows2:,}")

checks = {
    "tokenizer_len == input_rows": final_len == in_rows2,
    "tokenizer_len == output_rows": final_len == out_rows2,
    "input_rows == output_rows": in_rows2 == out_rows2,
    "all SIDs are 4 tokens": not bad_len,
    "all SID tokens atomic": not bad_atomic,
}
print()
for k, v in checks.items():
    print(f"  [{'PASS' if v else 'FAIL'}] {k}")

if not all(checks.values()):
    print("\nRESULT: FAIL")
    sys.exit(1)

# ---- persist the LETTER tokenizer ---------------------------------------
tok_dir = os.path.join(OUT, "tokenizer")
os.makedirs(tok_dir, exist_ok=True)
TOK.save_pretrained(tok_dir)
tok_files = {}
for name in sorted(os.listdir(tok_dir)):
    p = os.path.join(tok_dir, name)
    if os.path.isfile(p):
        tok_files[name] = {"bytes": os.path.getsize(p), "sha256": sha256(p)}
        print(f"  [save] {p}  {tok_files[name]['bytes']:>12,d} B  "
              f"{tok_files[name]['sha256'][:16]}")

report = {
    "base_model": BASE,
    "base_tokenizer_len": base_len,
    "letter_sid_token_count": len(all_tokens),
    "already_existing_count": len(already),
    "newly_added_count": n_added,
    "final_tokenizer_len": final_len,
    "max_sid_token_id": max(ids),
    "model_input_rows_after_resize": in_rows2,
    "model_output_rows_after_resize": out_rows2,
    "config_vocab_size_before_resize": cfg_vocab,
    "tied_embeddings": tied,
    "base_files": files,
    "letter_tokenizer_files": tok_files,
    "checks": checks,
    "n_items": N,
}
json.dump(report, open(os.path.join(OUT, "step1_tokenizer_report.json"), "w"), indent=2)
print()
print("  RESULT: ALL PASS")
