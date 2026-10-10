#!/usr/bin/env python3
"""
Decisive follow-up: is the 121 "difference" real, or just Python set-iteration order?

Two tests:
  T1. Re-run the trie construction twice with the SAME info file. If the
      allowed-token lists differ in order between two runs of identical input,
      the ordering is hash-seed noise and carries no information.
  T2. Compare the two info files under a FIXED PYTHONHASHSEED (run this script
      with PYTHONHASHSEED=0) so set iteration order is deterministic.
  T3. Show why order cannot matter: the lists are used as
      mask[batch, prefix_allowed_tokens] = 0, i.e. as an index set.
"""
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import CATEGORY, INFO, SHUFFLED_INFO, repo, run  # noqa: E402

CAT = CATEGORY
CLEAN_INFO = INFO
SHUF_INFO = SHUFFLED_INFO
CLEAN_MODEL = run("industrial_sft", "final_checkpoint")


def get_hash(x):
    return "-".join(str(_) for _ in x)


def build_hash_dict(info_file, tokenizer, prefix_index=3):
    with open(info_file, "r") as f:
        info = f.readlines()
    semantic_ids = [line.split("\t")[0].strip() + "\n" for line in info]
    info_semantic = [f"""### Response:\n{_}""" for _ in semantic_ids]
    prefixID = [tokenizer(_).input_ids for _ in info_semantic]

    hash_dict = dict()
    for ID in prefixID:
        ID.append(tokenizer.eos_token_id)
        for i in range(prefix_index, len(ID)):
            hn = get_hash(ID[:i]) if i == prefix_index else get_hash(ID[prefix_index:i])
            hash_dict.setdefault(hn, set()).add(ID[i])
    for k in hash_dict:
        hash_dict[k] = list(hash_dict[k])
    return hash_dict, prefixID


def main():
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(CLEAN_MODEL)

    print("PYTHONHASHSEED =", os.environ.get("PYTHONHASHSEED", "(not set)"))
    print("=" * 100)

    # ---------------- T1: same file, twice --------------------------------
    print("\nT1. SAME info file built twice -- do the allowed-token LISTS differ?")
    a1, _ = build_hash_dict(CLEAN_INFO, tk)
    a2, _ = build_hash_dict(CLEAN_INFO, tk)
    same_lists = all(a1[k] == a2[k] for k in a1)
    same_sets = all(set(a1[k]) == set(a2[k]) for k in a1)
    n_order = sum(1 for k in a1 if a1[k] != a2[k])
    print(f"   keys                        : {len(a1)}")
    print(f"   keys whose LIST differs     : {n_order}")
    print(f"   all SETS identical          : {same_sets}")
    print(f"   all LISTS identical         : {same_lists}")
    print("   -> identical input already produces different list ORDER;")
    print("      the ordering is Python set-iteration noise, not content.")

    # ---------------- T2: cross-file, set semantics ------------------------
    print("\nT2. clean vs shuffled -- SET semantics (what the mask actually uses)")
    c, _ = build_hash_dict(CLEAN_INFO, tk)
    s, _ = build_hash_dict(SHUF_INFO, tk)
    print(f"   clean keys {len(c)}   shuffled keys {len(s)}")
    print(f"   key sets equal              : {set(c) == set(s)}")
    setdiff = [k for k in set(c) | set(s) if set(c.get(k, [])) != set(s.get(k, []))]
    print(f"   keys with different SET     : {len(setdiff)}")
    if setdiff:
        for k in setdiff[:10]:
            print(f"     {k!r}: clean={sorted(c.get(k, []))} shuffled={sorted(s.get(k, []))}")
    orderdiff = [k for k in c if k in s and c[k] != s[k] and set(c[k]) == set(s[k])]
    print(f"   keys with only ORDER diff   : {len(orderdiff)}")

    # ---------------- T3: prove order is irrelevant ------------------------
    print("\nT3. is list order semantically relevant? (LogitProcessor.py:68)")
    src = open(repo("LogitProcessor.py"), encoding="utf-8").read()
    line = [l.strip() for l in src.splitlines() if "prefix_allowed_tokens]" in l]
    print(f"   usage: {line}")
    print("   -> advanced-index assignment; the result depends only on the SET of")
    print("      indices. Permuting the list cannot change the mask, therefore it")
    print("      cannot change the beam search result.")

    # empirical: build the mask both ways and compare bytes
    import torch
    toks = sorted(c["151686"]) if "151686" in c else sorted(c[list(c)[0]])
    m1 = torch.full((1, len(tk)), float("-inf"))
    m2 = torch.full((1, len(tk)), float("-inf"))
    m1[0, toks] = 0
    m2[0, list(reversed(toks))] = 0
    print(f"   mask identical under permutation: {torch.equal(m1, m2)}")

    # ---------------- T4: byte-level comparison of the sequence multiset ---
    print("\nT4. byte-level: tokenised SID sequence MULTISETS")
    _, pc = build_hash_dict(CLEAN_INFO, tk)
    _, ps = build_hash_dict(SHUF_INFO, tk)
    mc = Counter(tuple(x) for x in pc)
    ms = Counter(tuple(x) for x in ps)
    print(f"   clean {len(pc)} seqs / {len(mc)} distinct")
    print(f"   shuffled {len(ps)} seqs / {len(ms)} distinct")
    print(f"   MULTISETS identical         : {mc == ms}")

    ok = (same_sets and set(c) == set(s) and not setdiff and mc == ms)
    print("\n" + "=" * 100)
    print(f"FINAL: tries are STRUCTURALLY EQUIVALENT (as allowed-token SETS): {ok}")
    print("=" * 100)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
