#!/usr/bin/env python3
"""
Deterministic reachability test for constrained SID decoding.

With scores = 0 the mask ALONE decides the output, so the argmax path is exactly
the set of SIDs the trie permits. We sweep prefix_index and measure, for both the
P0 (3-level) and Treatment (4-level) indexes:

  * how many of the N catalogue SIDs are reachable
  * whether the emitted sequence is a legal SID

This turns "is prefix_index=3 correct for 4 levels?" into a measurement.
"""
import hashlib
import json
import os
import sys

import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.getcwd())
from sft import TokenExtender                       # noqa: E402
from LogitProcessor import ConstrainedLogitsProcessor  # noqa: E402

MODEL = "/root/autodl-tmp/models/Qwen2.5-0.5B"
PATHS = {"P0-3level": "data/Amazon/index/Industrial_and_Scientific.index.json",
         "TX-4level": "artifacts/letter_stage2/letter_index.json"}
SCRATCH = "/tmp/pi_scratch"


def get_hash(t):
    return hashlib.md5(str(list(t)).encode()).hexdigest()


def build_tok(path):
    tok = AutoTokenizer.from_pretrained(MODEL)
    os.makedirs(SCRATCH, exist_ok=True)
    link = os.path.join(SCRATCH, "Industrial_and_Scientific.index.json")
    if os.path.islink(link) or os.path.exists(link):
        os.remove(link)
    os.symlink(os.path.abspath(path), link)
    tok.add_tokens(sorted(TokenExtender(data_path=SCRATCH,
                                        dataset="Industrial_and_Scientific").get_new_tokens()))
    return tok


def build_trie(seqs, tok, pi):
    hd = {}
    for s in seqs:
        ids = tok(s).input_ids + [tok.eos_token_id]
        for i in range(pi, len(ids)):
            hn = get_hash(ids[:i]) if i == pi else get_hash(ids[pi:i])
            hd.setdefault(hn, set()).add(ids[i])
    return {k: sorted(v) for k, v in hd.items()}


def constrained_argmax(tok, hd, pi, prompt_ids, max_new=8):
    """Greedy decode with scores=0 so ONLY the mask decides."""
    NB = 4
    ccc = ConstrainedLogitsProcessor(
        prefix_allowed_tokens_fn=lambda b, ids: hd.get(get_hash(ids), []),
        num_beams=NB, base_model=MODEL, eos_token_id=tok.eos_token_id)
    cur = prompt_ids.repeat(NB, 1)
    out = []
    for _ in range(max_new):
        scores = torch.zeros(NB, tok.vocab_size + len(tok), dtype=torch.float)
        m = ccc(cur, scores)
        nxt = int(m[0].argmax().item())
        out.append(nxt)
        if nxt == tok.eos_token_id:
            break
        cur = torch.cat([cur, torch.full((NB, 1), nxt, dtype=cur.dtype)], dim=1)
    return out


def main():
    print("=" * 88)
    print("CONSTRAINED-DECODING REACHABILITY vs prefix_index")
    print("=" * 88)
    for tag, path in PATHS.items():
        tok = build_tok(path)
        idx = json.load(open(path, encoding="utf-8"))
        seqs = ["".join(idx[str(i)]) for i in range(len(idx))]
        legal = set(tuple(tok(s).input_ids + [tok.eos_token_id]) for s in seqs)
        N = len(seqs)
        print(f"\n--- {tag}: {N} items, SID depth = {len(tok(seqs[0]).input_ids)}, "
              f"tokenizer len = {len(tok)}")
        # a prompt that ends with 3 SID tokens (3-level) / 4 (4-level)
        prompt = torch.tensor([[tok(seqs[0]).input_ids[0]]])
        for pi in (0, 1, 2, 3, 4):
            hd = build_trie(seqs, tok, pi)
            gen = constrained_argmax(tok, hd, pi, prompt)
            gen_core = tuple(g for g in gen if g != tok.eos_token_id)
            is_legal = (gen_core in {tuple(x[:-1]) for x in legal})
            # reachability: try every catalogue SID as the prompt tail
            reach = 0
            for s in seqs[:300]:
                pr = torch.tensor([tok(s).input_ids[:1]])
                g = constrained_argmax(tok, hd, pi, pr)
                core = tuple(x for x in g if x != tok.eos_token_id)
                if core in {tuple(x[:-1]) for x in legal}:
                    reach += 1
            print(f"    prefix_index={pi}: trie_keys={len(hd):>6d}  "
                  f"greedy_emit={len(gen)} tok  legal={is_legal}  "
                  f"reachable(300 prompts)={reach}/300")


if __name__ == "__main__":
    main()
