#!/usr/bin/env python3
"""
analysis/verify_info_equivalence_final.py

Consolidated, self-contained re-verification of the --info_file audit.

Rebuilds the constrained-decoding trie exactly as evaluate.py:61-119 does, for the
original and the shuffled info file, and reports:
  1. legal SID sequence set
  2. prefix -> allowed-next-token mapping   (as SETS, the semantics the mask uses)
  3. EOS termination nodes
  4. the exact nature of any residual ordering difference
  5. proof that ordering cannot affect the mask

Read-only, no generation, no GPU compute beyond a tokenizer load.
Run with PYTHONHASHSEED=0 for determinism.
"""
import hashlib
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


def get_hash(x):
    return "-".join(str(_) for _ in x)


def build(info_file, tokenizer):
    """Byte-faithful reproduction of evaluate.py:61-119 (+122-126)."""
    info = open(info_file, "r").readlines()
    semantic_ids = [l.split("\t")[0].strip() + "\n" for l in info]
    item_titles = [l.split("\t")[1].strip() + "\n" for l in info
                   if len(l.split("\t")) >= 2]

    info_semantic = [f"""### Response:\n{_}""" for _ in semantic_ids]
    info_titles = [f"""### Response:\n{_}""" for _ in item_titles]
    prefixID = [tokenizer(_).input_ids for _ in info_semantic]
    prefixTitleID = [tokenizer(_).input_ids for _ in info_titles]
    prefix_index = 3                       # not gpt2 => 3 (evaluate.py:81-84)

    hash_dict = {}
    for ID in prefixID:
        ID.append(tokenizer.eos_token_id)
        for i in range(prefix_index, len(ID)):
            hn = get_hash(ID[:i]) if i == prefix_index else get_hash(ID[prefix_index:i])
            hash_dict.setdefault(hn, set()).add(ID[i])
    for k in hash_dict:
        hash_dict[k] = list(hash_dict[k])
    return {"prefixID": prefixID, "hash_dict": hash_dict,
            "semantic_ids": semantic_ids}


def main():
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(CLEAN_MODEL)
    print(f"PYTHONHASHSEED = {os.environ.get('PYTHONHASHSEED', '(unset)')}")
    print(f"tokenizer      = {CLEAN_MODEL}  vocab={len(tk)}  eos={tk.eos_token_id}")

    c = build(CLEAN_INFO, tk)
    s = build(SHUF_INFO, tk)

    print("\n" + "=" * 96)
    print("A. info_file ROLE -- static check of evaluate.py")
    print("=" * 96)
    src = open(repo("evaluate.py"), encoding="utf-8").read()
    n_ref = len(re.findall(r"\binfo_file\b", src))
    fields = sorted(set(re.findall(r"split\('\\t'\)\[(\d)\]", src)))
    print(f"  occurrences of the token 'info_file' in evaluate.py : {n_ref}")
    print(f"  tab-field indices accessed                          : {fields}")
    print(f"  field 0 = SID string, field 1 = title, field 2 = item_id")
    print(f"  -> item_id (field 2) read anywhere?                 : {'2' in fields}")
    print(f"  -> 'index_file' / 'indices' used in evaluate.py?    : "
          f"{('index_file' in src) or ('indices' in src)}")
    ds = open(repo("data.py"), encoding="utf-8").read()
    ev = re.search(r"class EvalSidDataset.*?(?=\nclass )", ds, re.S).group(0)
    print(f"  EvalSidDataset refs info_file/item_file/index_file  : "
          f"{[t for t in ('info_file','item_file','index_file','indices') if t in ev]}")

    print("\n" + "=" * 96)
    print("B. TRIE EQUIVALENCE")
    print("=" * 96)

    # ---- 1. legal SID sequence set
    mc = Counter(tuple(x) for x in c["prefixID"])
    ms = Counter(tuple(x) for x in s["prefixID"])
    legal_c = {tuple(x) for x in c["prefixID"]}
    legal_s = {tuple(x) for x in s["prefixID"]}
    print("\n  1. legal SID sequence set")
    print(f"     clean    : {len(c['prefixID'])} seqs, {len(legal_c)} distinct")
    print(f"     shuffled : {len(s['prefixID'])} seqs, {len(legal_s)} distinct")
    print(f"     SET equal      : {legal_c == legal_s}")
    print(f"     MULTISET equal : {mc == ms}")

    # ---- 2. prefix -> allowed next token  (SET semantics)
    hc, hs = c["hash_dict"], s["hash_dict"]
    print("\n  2. prefix -> allowed-next-token mapping")
    print(f"     clean    : {len(hc)} prefix keys")
    print(f"     shuffled : {len(hs)} prefix keys")
    print(f"     key SET equal            : {set(hc) == set(hs)}")
    setdiff = [k for k in set(hc) | set(hs)
               if set(hc.get(k, [])) != set(hs.get(k, []))]
    print(f"     keys with different SET  : {len(setdiff)}")
    for k in setdiff[:10]:
        print(f"       {k!r}")
        print(f"         clean    = {sorted(hc.get(k, []))}")
        print(f"         shuffled = {sorted(hs.get(k, []))}")
    orderdiff = [k for k in hc if k in hs and hc[k] != hs[k] and set(hc[k]) == set(hs[k])]
    print(f"     keys differing only in ORDER : {len(orderdiff)}")
    # confirm every order-diff key is a permutation of the same multiset
    bad_perm = [k for k in orderdiff if Counter(hc[k]) != Counter(hs[k])]
    print(f"     ...of which not a permutation: {len(bad_perm)}")
    if orderdiff:
        k = orderdiff[0]
        print(f"     example key {k!r}")
        print(f"       clean    {hc[k]}")
        print(f"       shuffled {hs[k]}")
        print(f"       sorted equal: {sorted(hc[k]) == sorted(hs[k])}")

    # ---- 3. EOS termination
    eos = tk.eos_token_id
    ec = {k for k, v in hc.items() if eos in v}
    es = {k for k, v in hs.items() if eos in v}
    tc = {k for k, v in hc.items() if set(v) == {eos}}
    ts = {k for k, v in hs.items() if set(v) == {eos}}
    print("\n  3. EOS termination nodes")
    print(f"     prefixes allowing EOS      : clean {len(ec)}  shuffled {len(es)}  equal {ec == es}")
    print(f"     pure terminal (EOS only)   : clean {len(tc)}  shuffled {len(ts)}  equal {tc == ts}")

    # ---- 4. ordering cannot matter
    print("\n  4. does allowed-token ORDER change the mask?")
    import torch
    key = orderdiff[0] if orderdiff else list(hc)[0]
    a = hc[key]
    m1 = torch.full((1, len(tk)), float("-inf"))
    m2 = torch.full((1, len(tk)), float("-inf"))
    m1[0, a] = 0
    m2[0, list(reversed(a))] = 0
    print(f"     LogitProcessor.py:68  mask[b, prefix_allowed_tokens] = 0  (index assignment)")
    print(f"     mask identical under list permutation : {torch.equal(m1, m2)}")
    print(f"     digest of mask built from clean list   : "
          f"{hashlib.md5(m1.numpy().tobytes()).hexdigest()}")
    m3 = torch.full((1, len(tk)), float("-inf"))
    m3[0, hs[key]] = 0
    print(f"     digest of mask built from shuffled list: "
          f"{hashlib.md5(m3.numpy().tobytes()).hexdigest()}")
    print(f"     -> byte-identical masks                : {torch.equal(m1, m3)}")

    # ---- 5. tokenizers
    print("\n  5. tokenizer identity between the two evaluated checkpoints")
    t1, t2 = AutoTokenizer.from_pretrained(CLEAN_MODEL), AutoTokenizer.from_pretrained(SHUF_MODEL)
    print(f"     vocab equal: {t1.get_vocab() == t2.get_vocab()}  "
          f"eos equal: {t1.eos_token_id == t2.eos_token_id}")

    equivalent = (legal_c == legal_s and mc == ms and set(hc) == set(hs)
                  and not setdiff and ec == es and tc == ts and not bad_perm)
    print("\n" + "=" * 96)
    print(f"VERDICT: trie structurally equivalent (semantically identical) : {equivalent}")
    print("=" * 96)
    return 0 if equivalent else 1


if __name__ == "__main__":
    sys.exit(main())
