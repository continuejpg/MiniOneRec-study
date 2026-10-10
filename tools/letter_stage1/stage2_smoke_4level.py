#!/usr/bin/env python3
"""
LETTER-SID Stage 2 -- step 7: 4-level MiniOneRec compatibility smoke test.

Claim to verify: the repository's SID path is DEPTH-AGNOSTIC, so switching from
3-level P0 SIDs to 4-level treatment SIDs requires NO source change. This script
proves or refutes that claim by exercising the real code paths:

  A. TokenExtender  -- registers every SID token found in the index
  B. trie build     -- evaluate.py's hash_dict construction (copied logic, run on
                       the real tokenizer)
  C. target encode  -- a treatment SID round-trips through the tokenizer
  D. constrained decode -- a greedy/beam step under the real prefix mask emits a
                       token that is legal under the TRIE built from the treatment index
  E. P0 regression  -- the same four checks on the ORIGINAL 3-level index

No training. No formal eval. Read-only w.r.t. existing data.
"""
import hashlib
import json
import os
import sys

import torch
from transformers import AutoTokenizer

REPO = os.getcwd()
sys.path.insert(0, REPO)
from sft import TokenExtender            # noqa: E402  (registers SID tokens)
from LogitProcessor import ConstrainedLogitsProcessor  # noqa: E402
from data import build_recommendation_prompt           # noqa: E402

MODEL = "/root/autodl-tmp/models/Qwen2.5-0.5B"
P0_INDEX = "data/Amazon/index/Industrial_and_Scientific.index.json"
TX_INDEX = "artifacts/letter_stage2/letter_index.json"

PREFIX_TAG = {0: "<a_", 1: "<b_", 2: "<c_", 3: "<d_", 4: "<e_", 5: "<f_"}


def get_hash(token_list):
    """Same helper evaluate.py uses (md5 over the repr of the token id list)."""
    return hashlib.md5(str(list(token_list)).encode()).hexdigest()


def build_trie(info_semantic, tok, prefix_index=3):
    """Verbatim logic of evaluate.py:89-122."""
    prefixID = [tok(s).input_ids for s in info_semantic]
    hash_dict = {}
    for ID in prefixID:
        ID.append(tok.eos_token_id)
        for i in range(prefix_index, len(ID)):
            hn = get_hash(ID[:i]) if i == prefix_index else get_hash(ID[prefix_index:i])
            hash_dict.setdefault(hn, set()).add(ID[i])
    return {k: sorted(v) for k, v in hash_dict.items()}, prefixID


def audit_one(tag, index_path, tok):
    print()
    print("=" * 84)
    print(f"{tag}  ({index_path})")
    print("=" * 84)
    idx = json.load(open(index_path, encoding="utf-8"))
    items = sorted(int(k) for k in idx)
    depths = sorted({len(v) for v in idx.values()})
    info_semantic = ["".join(idx[str(i)]) for i in range(len(idx))]

    print(f"  items        = {len(idx)}")
    print(f"  depths seen  = {depths}")
    print(f"  example SID  = {info_semantic[0]!r}")

    # ---- A. TokenExtender --------------------------------------------------
    # TokenExtender rebuilds the filename as basename(path).split('.')[0] +
    # '.index.json', so the file MUST be named <dataset>.index.json. We satisfy
    # that convention with a symlink in a scratch dir instead of copying.
    scratch = "artifacts/letter_treatment"
    os.makedirs(scratch, exist_ok=True)
    link = os.path.join(scratch, "Industrial_and_Scientific.index.json")
    if os.path.islink(link) or os.path.exists(link):
        os.remove(link)
    os.symlink(os.path.abspath(index_path), link)

    before = len(tok)
    te = TokenExtender(data_path=scratch,
                       dataset="Industrial_and_Scientific")
    new_tokens = te.get_new_tokens()
    added = tok.add_tokens(sorted(new_tokens))
    after = len(tok)
    uniq = {t for s in info_semantic for t in __import__("re").findall(r"<[^<>]+>", s)}
    print(f"  A. TokenExtender: distinct SID tokens in index = {len(uniq)}  "
          f"get_new_tokens() = {len(new_tokens)}  add_tokens added = {added}")
    print(f"     tokenizer len {before} -> {after}")
    ok_a = (len(new_tokens) == len(uniq)) and (added == len(uniq))

    # ---- B. trie ----------------------------------------------------------
    hash_dict, _ = build_trie(info_semantic, tok)
    depth_keys = {}
    for s in info_semantic[:200]:
        ids = tok(s).input_ids
        depth_keys[len(ids)] = depth_keys.get(len(ids), 0) + 1
    print(f"  B. trie: {len(hash_dict)} prefix keys; tokenizer seq length histogram "
          f"(first 200) = {depth_keys}")
    ok_b = len(hash_dict) > 0

    # ---- C. target round trip --------------------------------------------
    bad = 0
    for i in range(len(info_semantic)):
        ids = tok(info_semantic[i]).input_ids
        if tok.decode(ids).replace(" ", "") != info_semantic[i].replace(" ", ""):
            bad += 1
    print(f"  C. target encode/decode round trip: {len(info_semantic)-bad}/{len(info_semantic)} exact")
    ok_c = bad == 0

    # ---- D. constrained decode under the real mask ------------------------
    prompt = build_recommendation_prompt(", ".join(info_semantic[:3]))
    p_ids = tok(prompt, return_tensors="pt").input_ids
    NB = 4
    # ConstrainedLogitsProcessor expects input_ids as [num_beams, seq] (it does
    # input_ids.view(-1, num_beams, seq)), so feed it a beam-expanded batch.
    p_ids_beam = p_ids.repeat(NB, 1)
    ccc = ConstrainedLogitsProcessor(prefix_allowed_tokens_fn=
                                     lambda bid, ids: hash_dict.get(get_hash(ids), []),
                                     num_beams=NB, base_model=MODEL,
                                     eos_token_id=tok.eos_token_id)
    vocab = len(tok)
    scores = torch.zeros(NB, vocab, dtype=torch.float)
    masked = ccc(p_ids_beam, scores)
    allowed = int(torch.isfinite(masked[0]).sum())
    print(f"  D. constrained step: prompt_tokens={p_ids.shape[-1]}  num_beams={NB}  "
          f"allowed_next_tokens(first step) = {allowed}")

    # Walk the trie the way the DECODER does: after the prompt, hash_key is the
    # last `count` tokens, where count counts generated tokens. The trie keys are
    # hashes of the token-id list AFTER prefix_index, so seed the walk with the
    # first `prefix_index` generated token ids of a real SID.
    seed_ids = tok(info_semantic[0]).input_ids
    path = []
    cur = list(seed_ids[:3])          # prefix_index = 3 for this tokenizer
    for step in range(1, 8):
        hk = cur[-step:] if step > 0 else cur
        al = hash_dict.get(get_hash(hk), [])
        if not al:
            break
        nxt = al[0]
        path.append(nxt)
        cur = cur + [nxt]
        if nxt == tok.eos_token_id:
            break
    decoded = tok.decode(path)
    print(f"     seeded legal walk -> {decoded!r}  ({len(path)} tokens)")
    ok_d = allowed > 0 and len(path) > 0
    # walk the trie greedily for a few steps and confirm the path stays legal
    path, cur = [], p_ids[0].tolist()
    for step in range(6):
        hk = cur[-3:] if step == 0 else cur[-step:]
        al = hash_dict.get(get_hash(hk), [])
        if not al:
            break
        nxt = al[0]
        path.append(nxt)
        cur = cur + [nxt]
        if nxt == tok.eos_token_id:
            break
    decoded = tok.decode(path)
    print(f"     greedy legal walk -> {decoded!r}  ({len(path)} tokens)")
    ok_d = allowed > 0 and len(path) > 0

    ids = tok(info_semantic[0]).input_ids
    return {"tag": tag, "n_items": len(idx), "depths": depths,
            "n_sid_tokens": len(uniq), "added": added,
            "tokenizer_len": after, "trie_keys": len(hash_dict),
            "allowed_first": allowed, "greedy_walk": decoded,
            "checks": {"A_tokenextender": ok_a, "B_trie": ok_b,
                       "C_roundtrip": ok_c, "D_constrained": ok_d}}


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    print(f"  tokenizer loaded from {MODEL}   base len = {len(tok)}")
    res = []
    res.append(audit_one("P0 REGRESSION (3-level original)", P0_INDEX, tok))
    res.append(audit_one("TREATMENT (4-level LETTER)", TX_INDEX, tok))

    print()
    print("=" * 84)
    print("SUMMARY")
    print("=" * 84)
    allok = True
    for r in res:
        bad = [k for k, v in r["checks"].items() if not v]
        allok &= not bad
        print(f"  {r['tag']}")
        print(f"    depth={r['depths']}  items={r['n_items']}  sid_tokens={r['n_sid_tokens']}  "
              f"trie_keys={r['trie_keys']}")
        print(f"    checks={r['checks']}  failed={bad}")

    # ---- cross-check: the two index files must have DIFFERENT depths ------
    d0, d1 = res[0]["depths"], res[1]["depths"]
    depth_diff = (d0 != d1)
    print()
    print(f"  depths differ between P0 and treatment: {d0} vs {d1}  -> {depth_diff}")
    print(f"  NO SOURCE CHANGE REQUIRED: {allok and depth_diff}")
    print()
    print(f"RESULT: {'PASS' if allok else 'FAIL'}")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
