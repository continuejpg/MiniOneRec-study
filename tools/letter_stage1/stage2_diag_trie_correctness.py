#!/usr/bin/env python3
"""
Decisive test: does the constrained-decoder trie actually permit the GROUND-TRUTH
next token, and how tight is it?

For each catalogue SID we replay the decoder's own bookkeeping (LogitProcessor
uses hash_key = sent[-count:]) and ask, at every generation step, whether the true
next token is inside the allowed set. This measures CORRECTNESS (recall) directly
and is far more informative than a greedy argmax walk.
"""
import hashlib
import json
import os
import sys
from collections import Counter

from transformers import AutoTokenizer

sys.path.insert(0, os.getcwd())
from sft import TokenExtender  # noqa: E402

MODEL = "/root/antodl/PLACEHOLDER"
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
    return hd


def replay(tok, hd, pi, prompt_ids, sids):
    """Replay the decoder bookkeeping and check the true token is allowed."""
    total = ok = 0
    tight = []
    for ids in sids:
        cur = list(prompt_ids)              # decoder context
        for count in range(len(ids)):
            key = cur[-pi:] if count == 0 else cur[-count:]
            al = hd.get(get_hash(key), [])
            total += 1
            if ids[count] in al:
                ok += 1
            tight.append(len(al))
            cur = cur + [ids[count]]
    return ok, total, tight


def main():
    print("=" * 88)
    print("TRIE CORRECTNESS: is the ground-truth next token always allowed?")
    print("=" * 88)
    for tag, path in PATHS.items():
        tok = build_tok(path)
        idx = json.load(open(path, encoding="utf-8"))
        seqs = ["".join(idx[str(i)]) for i in range(len(idx))]
        N = len(seqs)
        depth = len(tok(seqs[0]).input_ids)
        sids = [tok(s).input_ids for s in seqs]
        print(f"\n--- {tag}: depth={depth}  n_items={N}  tokenizer_len={len(tok)}")
        for pi in (2, 3, 4, 5):
            hd = build_trie(seqs, tok, pi)
            # a realistic prompt: ends with (depth) SID tokens
            prompt = sids[0][:depth] + sids[1][:1]      # arbitrary but SID-shaped
            ok, total, tight = replay(tok, hd, pi, prompt, sids[:200])
            t = Counter(tight)
            print(f"    pi={pi}: keys={len(hd):>6d}  truth_allowed={ok}/{total} "
                  f"({100*ok/total:6.2f}%)  allowed_set_size median="
                  f"{sorted(tight)[len(tight)//2]:>5d} max={max(tight):>6d}")
    print()
    print("  Reading: truth_allowed = % of generation steps where the trie permits the")
    print("  correct token. <100% means the trie is TOO RESTRICTIVE (blocks valid SIDs).")


if __name__ == "__main__":
    main()
