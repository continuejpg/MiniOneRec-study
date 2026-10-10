#!/usr/bin/env python3
"""prefix_index semantics WITH the SID-extended tokenizer (the real pipeline).

Previously I measured with a raw tokenizer that splits '<a_236>' into 6 subword
pieces. The real pipeline calls TokenExtender -> tokenizer.add_tokens(SID tokens)
so each SID level is ONE atomic id. This changes everything about prefix_index.
"""
import json
import os
import sys

from transformers import AutoTokenizer

sys.path.insert(0, os.getcwd())
from sft import TokenExtender  # noqa: E402

MODEL = "/root/autodl-tmp/models/Qwen2.5-0.5B"
P0 = "data/Amazon/index/Industrial_and_Scientific.index.json"
TX = "artifacts/letter_stage2/letter_index.json"


def load_tok(index_path, scratch):
    tok = AutoTokenizer.from_pretrained(MODEL)
    os.makedirs(scratch, exist_ok=True)
    link = os.path.join(scratch, "Industrial_and_Scientific.index.json")
    if os.path.islink(link) or os.path.exists(link):
        os.remove(link)
    os.symlink(os.path.abspath(index_path), link)
    te = TokenExtender(data_path=scratch, dataset="Industrial_and_Scientific")
    n = tok.add_tokens(sorted(te.get_new_tokens()))
    return tok, n


def get_hash(t):
    import hashlib
    return hashlib.md5(str(list(t)).encode()).hexdigest()


for tag, path in (("P0 3-level", P0), ("TX 4-level", TX)):
    print("=" * 84)
    print(f"{tag}  ({path})")
    print("=" * 84)
    tok, nadd = load_tok(path, "/tmp/pi_scratch")
    print(f"  tokenizer len after add_tokens = {len(tok)}  (added {nadd})")
    idx = json.load(open(path, encoding="utf-8"))
    seqs = ["".join(idx[str(i)]) for i in range(len(idx))]
    lens = sorted({len(tok(s).input_ids) for s in seqs})
    print(f"  tokenised SID length(s) = {lens}")
    s0 = seqs[0]
    ID = tok(s0).input_ids
    print(f"  example {s0!r}")
    print(f"    input_ids = {ID}   n={len(ID)}")
    print(f"    decoded pieces = {tok.convert_ids_to_tokens(ID)}")

    # evaluate.py:92-102 logic with prefix_index=3
    for pi in (0, 1, 2, 3):
        hd = {}
        for s in seqs[:50]:
            ids = tok(s).input_ids + [tok.eos_token_id]
            for i in range(pi, len(ids)):
                hn = get_hash(ids[:i]) if i == pi else get_hash(ids[pi:i])
                hd.setdefault(hn, set()).add(ids[i])
        # can a full SID be reached greedily from its own prefix?
        full = tok(s0).input_ids + [tok.eos_token_id]
        reach = True
        for i in range(pi, len(full)):
            hn = get_hash(full[:i]) if i == pi else get_hash(full[pi:i])
            if full[i] not in hd.get(hn, set()):
                reach = False
                break
        print(f"    prefix_index={pi}: keys(50 items)={len(hd):>5d}  "
              f"item0 fully reachable={reach}")
    print()
