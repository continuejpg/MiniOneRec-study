#!/usr/bin/env python3
"""Pin down prefix_index semantics from the tokenizer's actual output.

Claim to test: prefix_index does NOT mean "skip the tokenizer's synthetic
prefix"; it means "start building the trie at token index prefix_index of the
SID sequence". If true, a 4-token SID is only constrained from its last level,
which would be a genuine depth defect (not merely an unused-variable issue).
"""
import json

from transformers import AutoTokenizer

MODEL = "/root/autodl-tmp/models/Qwen2.5-0.5B"
P0 = "data/Amazon/index/Industrial_and_Scientific.index.json"
TX = "artifacts/letter_stage2/letter_index.json"

tok = AutoTokenizer.from_pretrained(MODEL)
print("=" * 84)
print("TOKENISER BEHAVIOUR ON SID STRINGS")
print("=" * 84)
for s in ("<a_236>", "<a_236><b_231><c_226>", "<a_48><b_191><c_62><d_198>"):
    ids = tok(s).input_ids
    toks = tok.convert_ids_to_tokens(ids)
    print(f"  {s!r}")
    print(f"    n_ids={len(ids)}  ids={ids}")
    print(f"    tokens={toks}")

print()
print("=" * 84)
print("WHAT prefix_index ACTUALLY INDEXES")
print("=" * 84)
for tag, path in (("P0 3-level", P0), ("TX 4-level", TX)):
    idx = json.load(open(path, encoding="utf-8"))
    seqs = ["".join(idx[str(i)]) for i in range(len(idx))]
    lens = {len(tok(s).input_ids) for s in seqs}
    print(f"  {tag}: tokenised SID length(s) = {sorted(lens)}   n_items={len(seqs)}")

print()
print("  evaluate.py:94  for i in range(prefix_index, len(ID))  adds hash(ID[prefix_index:i]) -> ID[i]")
print("  With prefix_index=3:")
for tag, path in (("P0 3-level", P0), ("TX 4-level", TX)):
    idx = json.load(open(path, encoding="utf-8"))
    seqs = ["".join(idx[str(i)]) for i in range(len(idx))]
    ID = tok(seqs[0]).input_ids
    print(f"    {tag}: first SID tokenises to {len(ID)} ids -> loop runs i in "
          f"range(3,{len(ID)}) = {list(range(3, len(ID)+1))}")
    print(f"      => trie keys built = {max(0, len(ID)+1-3)} per item; "
          f"only the LAST {max(0, len(ID)+1-3)} level(s) are constrained")

print()
print("  => CORRECT prefix_index should exclude only the tokenizer's synthetic")
print("     prefix (the leading <|im_start|>-style token), i.e. the index of the")
print("     FIRST real SID token. Measure it:")
for s in ("<a_236>", "<a_236><b_231><c_226>", "<a_48><b_191><c_62><d_198>"):
    ids = tok(s).input_ids
    # a real SID token is one that is registered as an added token
    added = tok.get_added_vocab()
    first = next((i for i, v in enumerate(ids) if v in added.values()), None)
    print(f"    {s!r}: ids={ids}  first_SID_token_index={first}")
