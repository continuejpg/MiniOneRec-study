#!/usr/bin/env python3
"""
FINAL, precise test of the SID-depth assumption.

Mechanism (derived from the code, then verified here):
  * evaluate.py:94 builds trie keys as  hash(ID[:pi])  and  hash(ID[pi:i])
    where ID = [sid_token_0 ... sid_token_{d-1}, EOS]
  * LogitProcessor at generation step 0 uses  hash_key = prompt_ids[-pi:]
  * therefore the ROOT key matches ONLY IF the prompt's last `pi` tokens are
    exactly the first `pi` SID tokens of the item
  * the real prompt (data.py build_recommendation_prompt + ", ".join(history_sids))
    ends with a history SID, i.e. exactly `d` SID tokens

  => prefix_index must equal the SID DEPTH, not 3.

This script tests reachability of the GROUND-TRUTH SID from the REAL prompt, for
pi = depth and pi = 3, on both the P0 (d=3) and Treatment (d=4) indexes.
"""
import hashlib
import json
import os
import sys

from transformers import AutoTokenizer

sys.path.insert(0, os.getcwd())
from sft import TokenExtender                     # noqa: E402
from data import build_recommendation_prompt      # noqa: E402

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
        ID = list(tok(s).input_ids)
        ID.append(tok.eos_token_id)
        for i in range(pi, len(ID)):
            hn = get_hash(ID[:i]) if i == pi else get_hash(ID[pi:i])
            hd.setdefault(hn, set()).add(ID[i])
    return hd


def replay_decoder(tok, hd, pi, prompt_ids, sid_ids):
    """Exact LogitProcessor bookkeeping; returns (steps_ok, steps_total)."""
    cur = list(prompt_ids)
    ok = tot = 0
    for count in range(len(sid_ids) + 1):          # +1 includes EOS
        key = cur[-pi:] if count == 0 else cur[-count:]
        al = hd.get(get_hash(key), [])
        truth = sid_ids[count] if count < len(sid_ids) else tok.eos_token_id
        tot += 1
        if truth in al:
            ok += 1
        cur = cur + [truth]
    return ok, tot


def main():
    print("=" * 88)
    print("REAL-PROMPT SID REACHABILITY vs prefix_index")
    print("=" * 88)
    for tag, path in PATHS.items():
        tok = build_tok(path)
        idx = json.load(open(path, encoding="utf-8"))
        seqs = ["".join(idx[str(i)]) for i in range(len(idx))]
        depth = len(tok(seqs[0]).input_ids)
        sids = [tok(s).input_ids for s in seqs]
        print(f"\n--- {tag}: depth={depth}  n_items={len(seqs)}")

        # REAL prompt construction: history SIDs joined by ", "
        for pi in sorted({3, depth}):
            hd = build_trie(seqs, tok, pi)
            ok = tot = 0
            for i in range(min(300, len(seqs))):
                hist = seqs[i:i + 3] if i + 3 <= len(seqs) else seqs[:3]
                prompt = build_recommendation_prompt(", ".join(hist))
                pid = tok(prompt).input_ids
                a, b = replay_decoder(tok, hd, pi, pid, sids[i])
                ok += a
                tot += b
            print(f"    pi={pi}: keys={len(hd):>6d}   truth_allowed={ok}/{tot} "
                  f"({100*ok/tot:6.2f}%)")
        print(f"    => depth={depth}; the prompt ends with {depth} SID tokens, so only")
        print(f"       pi={depth} can match the trie root key.")

    print()
    print("CONCLUSION: prefix_index is a function of SID DEPTH. The repository")
    print("hardcodes 3 (and 4 for gpt2). A 4-level SID therefore requires")
    print("prefix_index=4 in evaluate.py AND LogitProcessor.py, otherwise the")
    print("prompt-tail key never matches the trie and the constraint silently")
    print("degrades to 'no valid tokens' -> forced EOS.")


if __name__ == "__main__":
    main()
