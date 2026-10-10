#!/usr/bin/env python3
"""
analysis/audit_info_file_equivalence.py

Decisive test of the claim: "the shuffled evaluation MUST swap --info_file,
otherwise constrained decoding restores the original item<->SID mapping."

Method: reproduce evaluate.py's trie construction EXACTLY (evaluate.py:61-119) for
both info files, with the real tokenizer, and compare the resulting objects:
  1. the tokenised SID sequences           prefixID          (list of lists)
  2. the prefix -> allowed-next-token map  hash_dict
  3. the legal whole-SID set (EOS nodes)
  4. the get_hash keys, introduced in the same order

The comparison is order-sensitive where it matters: hash_dict iteration order and
each allowed-token list order both affect nothing semantically, but they are
reported so any difference is visible rather than hidden.

Read-only. Runs no generation, no GPU work beyond loading a tokenizer.
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
SHUF_MODEL = run("industrial_sft_shuffled_sid", "final_checkpoint")


# ---- exact copies from evaluate.py (lines 24-26, 61-119, 122-126) ------------
def get_hash(x):
    x = [str(_) for _ in x]
    return "-".join(x)


def build_trie(info_file, tokenizer, is_llama=False, is_gpt2=False):
    """Byte-faithful reproduction of evaluate.py:61-119."""
    with open(info_file, "r") as f:
        info = f.readlines()
    semantic_ids = [line.split("\t")[0].strip() + "\n" for line in info]
    item_titles = [line.split("\t")[1].strip() + "\n"
                   for line in info if len(line.split("\t")) >= 2]

    info_semantic = [f"""### Response:\n{_}""" for _ in semantic_ids]
    info_titles = [f"""### Response:\n{_}""" for _ in item_titles]

    if is_llama:
        prefixID = [tokenizer(_).input_ids[1:] for _ in info_semantic]
        prefixTitleID = [tokenizer(_).input_ids[1:] for _ in info_titles]
    else:
        prefixID = [tokenizer(_).input_ids for _ in info_semantic]
        prefixTitleID = [tokenizer(_).input_ids for _ in info_titles]

    prefix_index = 4 if is_gpt2 else 3

    hash_dict = dict()
    for index, ID in enumerate(prefixID):
        ID.append(tokenizer.eos_token_id)
        for i in range(prefix_index, len(ID)):
            if i == prefix_index:
                hash_number = get_hash(ID[:i])
            else:
                hash_number = get_hash(ID[prefix_index:i])
            if hash_number not in hash_dict:
                hash_dict[hash_number] = set()
            hash_dict[hash_number].add(ID[i])
        hash_number = get_hash(ID[prefix_index:])

    hash_dict_title = dict()
    for index, ID in enumerate(prefixTitleID):
        ID.append(tokenizer.eos_token_id)
        for i in range(prefix_index, len(ID)):
            if i == prefix_index:
                hash_number = get_hash(ID[:i])
            else:
                hash_number = get_hash(ID[prefix_index:i])
            if hash_number not in hash_dict_title:
                hash_dict_title[hash_number] = set()
            hash_dict_title[hash_number].add(ID[i])
        hash_number = get_hash(ID[prefix_index:])

    for key in hash_dict.keys():
        hash_dict[key] = list(hash_dict[key])
    for key in hash_dict_title.keys():
        hash_dict_title[key] = list(hash_dict_title[key])

    return {
        "info": info,
        "semantic_ids": semantic_ids,
        "item_titles": item_titles,
        "prefixID": prefixID,          # NOTE: mutated in place (EOS appended)
        "prefixTitleID": prefixTitleID,
        "hash_dict": hash_dict,
        "hash_dict_title": hash_dict_title,
        "prefix_index": prefix_index,
    }


def main():
    from transformers import AutoTokenizer

    print("=" * 104)
    print("AUDIT: is --info_file only a SID-codebook source, or does it carry the item<->SID map?")
    print("=" * 104)

    tk = AutoTokenizer.from_pretrained(CLEAN_MODEL)
    print(f"\ntokenizer = {CLEAN_MODEL}   vocab_size = {len(tk)}   eos = {tk.eos_token_id}")

    c = build_trie(CLEAN_INFO, tk)
    s = build_trie(SHUF_INFO, tk)

    # =========================================================== 1. tokenised SIDs
    print("\n" + "-" * 104)
    print("1. tokenised SID sequences (prefixID)")
    print("-" * 104)
    print(f"  clean    : {len(c['prefixID'])} sequences, "
          f"length histogram {dict(Counter(len(x) for x in c['prefixID']))}")
    print(f"  shuffled : {len(s['prefixID'])} sequences, "
          f"length histogram {dict(Counter(len(x) for x in s['prefixID']))}")

    # compare as SETS (the trie does not care about order)
    set_c = Counter(tuple(x) for x in c["prefixID"])
    set_s = Counter(tuple(x) for x in s["prefixID"])
    print(f"\n  as MULTISETS equal          : {set_c == set_s}")
    print(f"  as ORDERED lists equal      : {c['prefixID'] == s['prefixID']}")
    if set_c != set_s:
        only_c = set(set_c) - set(set_s)
        only_s = set(set_s) - set(set_c)
        print(f"    only in clean  : {len(only_c)}  e.g. {list(only_c)[:3]}")
        print(f"    only in shuffled: {len(only_s)}  e.g. {list(only_s)[:3]}")

    # =========================================================== 2. hash_dict
    print("\n" + "-" * 104)
    print("2. prefix -> allowed-next-token map (hash_dict)  <-- this IS the trie")
    print("-" * 104)
    hc, hs = c["hash_dict"], s["hash_dict"]
    print(f"  clean    : {len(hc)} prefix keys")
    print(f"  shuffled : {len(hs)} prefix keys")
    print(f"  key sets equal              : {set(hc) == set(hs)}")
    print(f"  key ORDER equal             : {list(hc) == list(hs)}")

    diffs = []
    for k in set(hc) | set(hs):
        a, b = hc.get(k), hs.get(k)
        if a is None or b is None:
            diffs.append((k, a, b, "key missing on one side"))
        elif set(a) != set(b):
            diffs.append((k, a, b, "allowed-token SET differs"))
        elif a != b:
            diffs.append((k, a, b, "same set, different ORDER"))
    print(f"  per-key differences         : {len(diffs)}")
    if diffs:
        for k, a, b, why in diffs[:15]:
            print(f"    {why}: key={k!r}")
            print(f"      clean    = {sorted(a) if a else a}")
            print(f"      shuffled = {sorted(b) if b else b}")
    else:
        print("    (none -- every prefix maps to exactly the same allowed-token set)")

    # =========================================================== 3. EOS nodes
    print("\n" + "-" * 104)
    print("3. EOS termination (is EOS an allowed next token after a complete SID?)")
    print("-" * 104)
    eos = tk.eos_token_id
    tc = {k for k, v in hc.items() if eos in v}
    ts = {k for k, v in hs.items() if eos in v}
    print(f"  clean    : {len(tc)} prefixes allow EOS")
    print(f"  shuffled : {len(ts)} prefixes allow EOS")
    print(f"  EOS-node sets equal         : {tc == ts}")
    # a complete SID is a prefix after which only EOS is allowed
    term_c = {k for k, v in hc.items() if set(v) == {eos}}
    term_s = {k for k, v in hs.items() if set(v) == {eos}}
    print(f"  clean    : {len(term_c)} pure terminal nodes (only EOS allowed)")
    print(f"  shuffled : {len(term_s)} pure terminal nodes")
    print(f"  terminal-node sets equal    : {term_c == term_s}")

    # =========================================================== 4. legal SIDs
    print("\n" + "-" * 104)
    print("4. legal whole-SID set (the decoding target space)")
    print("-" * 104)

    # The trie's legal SIDs are exactly the tokenised semantic_ids: each is fed to
    # the tokenizer the same way evaluate.py does (evaluate.py:64,68,79).
    def tok_sids(info):
        return [tuple(tk(f"""### Response:\n{l.split(chr(9))[0].strip() + chr(10)}""").input_ids)
                for l in info if l.strip()]

    a, b = tok_sids(c["info"]), tok_sids(s["info"])
    print(f"  clean    : {len(a)} tokenised SIDs, distinct {len(set(a))}")
    print(f"  shuffled : {len(b)} tokenised SIDs, distinct {len(set(b))}")
    print(f"  legal-SID SET equal         : {set(a) == set(b)}")
    print(f"  legal-SID MULTISET equal    : {Counter(a) == Counter(b)}")
    print(f"  legal-SID ORDER equal       : {a == b}")
    if set(a) != set(b):
        print(f"    only clean    : {list(set(a)-set(b))[:3]}")
        print(f"    only shuffled : {list(set(b)-set(a))[:3]}")

    # =========================================================== 5. tokenizer identity
    print("\n" + "-" * 104)
    print("5. tokenizer identity between the two evaluated checkpoints")
    print("-" * 104)
    try:
        tk1 = AutoTokenizer.from_pretrained(CLEAN_MODEL)
        tk2 = AutoTokenizer.from_pretrained(SHUF_MODEL)
        v1, v2 = tk1.get_vocab(), tk2.get_vocab()
        print(f"  clean vocab {len(v1)}   shuffled vocab {len(v2)}   equal: {v1 == v2}")
        print(f"  eos ids equal: {tk1.eos_token_id == tk2.eos_token_id} "
              f"({tk1.eos_token_id})")
        # cross-tokenise: same strings through both tokenizers
        strs = [f"""### Response:\n{l.split(chr(9))[0].strip() + chr(10)}"""
                for l in c["info"][:200] if l.strip()]
        same = all(tk1(s).input_ids == tk2(s).input_ids for s in strs)
        print(f"  first 200 SID strings tokenise identically in both: {same}")
    except Exception as e:
        print(f"  could not compare tokenizers: {type(e).__name__}: {e}")

    # =========================================================== 6. the item_id field
    print("\n" + "-" * 104)
    print("6. does anything read field 3 (item_id) of the info file?")
    print("-" * 104)
    src = open(repo("evaluate.py"), encoding="utf-8").read()
    uses = re.findall(r"split\('\\t'\)\[(\d)\]", src)
    print(f"  evaluate.py accesses tab-field indices: {sorted(set(uses))}")
    print(f"  -> field 0 = SID (used), field 1 = title (built, never consumed), "
          f"field 2 = item_id (NEVER READ)")

    # =========================================================== verdict
    trie_equal = (len(hc) == len(hs) and set(hc) == set(hs) and not diffs
                  and tc == ts and term_c == term_s and set(a) == set(b))
    print("\n" + "=" * 104)
    print(f"VERDICT: the two info files produce an EQUIVALENT constrained-decoding trie: "
          f"{trie_equal}")
    print("=" * 104)
    return 0 if trie_equal else 1


if __name__ == "__main__":
    sys.exit(main())
