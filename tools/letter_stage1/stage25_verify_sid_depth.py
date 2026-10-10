#!/usr/bin/env python3
"""
Stage 2.5 verification -- SID depth inference + trie semantics, P0 and LETTER.

Everything here is derived from the REAL call-site semantics:

  evaluate.py:64-71 reads an info file, takes column 0 of each line as the SID,
  and wraps it as  entry = f'### Response:\\n{sid}'.  The trie is then built as

      ID  = tokenizer(entry).input_ids + [EOS]
      for i in range(prefix_index, len(ID)):
          key = hash(ID[:i]) if i == prefix_index else hash(ID[prefix_index:i])
          trie[key].add(ID[i])

  and the decoder, having generated the full SID into the context, asks
      hash(context[-prefix_index:])

  With prefix_index == token length of '### Response:\\n' the context tail after
  generating the whole SID is exactly the SID itself, so the lookup lands on the
  key hash(wrapper_ids ++ sid_ids) whose only allowed token is EOS. That is what
  forces termination at the right depth and is asserted below.

Run:
    python tools/letter_stage1/stage25_verify_sid_depth.py
"""
import hashlib
import json
import os
import re
import sys

from transformers import AutoTokenizer

sys.path.insert(0, os.getcwd())
from sid_utils import (get_hash, infer_prefix_index, split_sid_tokens)  # noqa: E402

MODEL = "/root/autodl-tmp/models/Qwen2.5-0.5B"
CAT = "Industrial_and_Scientific"
WRAPPER = "### Response:\n"

CASES = {
    "P0": {
        "info": f"data/Amazon/info/{CAT}_5_2016-10-2018-11.txt",
        "index": f"data/Amazon/index/{CAT}.index.json",
    },
    "LETTER": {
        "info": "artifacts/letter_stage2/letter_info.txt",
        "index": "artifacts/letter_stage2/letter_index.json",
    },
}

results = {}


def read_info_sids(path):
    """Exactly evaluate.py's extraction: column 0 of every non-empty line."""
    sids = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                sids.append(line.split("\t")[0].strip())
    return sids


def build_trie(entries, tok, prefix_index):
    trie = {}
    for e in entries:
        ID = list(tok(e, add_special_tokens=False).input_ids)
        ID.append(tok.eos_token_id)
        for i in range(prefix_index, len(ID)):
            k = get_hash(ID[:i]) if i == prefix_index else get_hash(ID[prefix_index:i])
            trie.setdefault(k, set()).add(ID[i])
    return trie


def audit(tag, spec, tok):
    print("=" * 86)
    print(f"[{tag}]  info={spec['info']}")
    print("=" * 86)
    out = {"tag": tag}
    fails = []

    def chk(ok, label, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  -- {detail}" if detail else ""))
        if not ok:
            fails.append(label)
        return ok

    sids = read_info_sids(spec["info"])
    idx = json.load(open(spec["index"], encoding="utf-8"))
    entries = None  # built below, after we know the tokenizer is SID-extended

    # ---- 0. tokenizer must treat SID levels as atomic tokens ----------------
    # The caller registers the catalogue's own SID tokens, exactly as sft.py does.
    all_tokens = sorted({t for v in idx.values() for t in split_sid_tokens("".join(v))})
    added = tok.add_tokens(all_tokens)
    print(f"  registered {added} new SID tokens (catalogue has {len(all_tokens)} distinct)")

    entries = [f"{WRAPPER}{s}" for s in sids]

    # ---- 1. depth inference ------------------------------------------------
    try:
        prefix_index, depth, wlen = infer_prefix_index(entries, tok, WRAPPER)
        out["inferred_depth"] = depth
        out["prefix_index"] = prefix_index
        out["wrapper_tokens"] = wlen
        print(f"  inferred_depth = {depth}   prefix_index = {prefix_index}   "
              f"wrapper_tokens = {wlen}")
    except Exception as e:  # noqa: BLE001
        out["inferred_depth"] = None
        chk(False, "infer_prefix_index", f"{type(e).__name__}: {e}")
        out["fails"] = fails
        return out

    out["n_items"] = len(sids)
    chk(len(sids) == len(idx), "info lines == index entries",
        f"{len(sids)} vs {len(idx)}")

    # ---- 2. encode/decode --------------------------------------------------
    bad = []
    for i, s in enumerate(sids):
        ids = tok(s, add_special_tokens=False).input_ids
        if tok.decode(ids).replace(" ", "") != s.replace(" ", ""):
            bad.append(i)
    out["encode_decode_pass"] = not bad
    chk(not bad, "encode/decode round trip", f"{len(sids)-len(bad)}/{len(sids)} exact")

    # ---- 3. trie ----------------------------------------------------------
    trie = build_trie(entries, tok, prefix_index)
    eos = tok.eos_token_id
    out["trie_keys"] = len(trie)

    # 3a. full SID -> EOS
    hit = 0
    only_eos = 0
    for s in sids:
        sid_ids = tok(s, add_special_tokens=False).input_ids
        key = get_hash(sid_ids)          # == decoder's context[-prefix_index:]
        al = trie.get(key)
        if al is not None:
            hit += 1
            if set(al) == {eos}:
                only_eos += 1
    out["full_sid_to_eos_pass"] = (hit == len(sids) and only_eos == len(sids))
    chk(hit == len(sids), "full SID prefix present in trie", f"{hit}/{len(sids)}")
    chk(only_eos == len(sids), "full SID -> ONLY EOS", f"{only_eos}/{len(sids)}")

    # 3b. incomplete prefix must not reach EOS
    early = []
    for s in sids:
        sid_ids = tok(s, add_special_tokens=False).input_ids
        for k in range(depth):
            if eos in trie.get(get_hash(sid_ids[:k]), ()):
                early.append((s, k))
    out["early_eos_rejected"] = not early
    chk(not early, "no incomplete prefix can emit EOS", f"{len(early)} violations")

    # 3c. non-catalogue continuation rejected
    catalogue = {tuple(tok(s, add_special_tokens=False).input_ids) for s in sids}
    other = [t for t in all_tokens if tok.convert_tokens_to_ids(t) != eos]
    violations = 0
    tested = 0
    for s in sids[:200]:
        sid_ids = tok(s, add_special_tokens=False).input_ids
        for pos in range(depth):
            for cand in other[:8]:
                ci = tok.convert_tokens_to_ids(cand)
                mutated = tuple(sid_ids[:pos] + [ci] + sid_ids[pos + 1:])
                tested += 1
                if mutated in catalogue:
                    continue
                # the decoder reaches this mutated sequence only if every step was
                # allowed; the final step consults hash(mutated)
                if eos in trie.get(get_hash(mutated), ()):
                    violations += 1
    out["invalid_path_rejected"] = violations == 0
    chk(violations == 0, "non-catalogue SID cannot reach EOS",
        f"{violations} violations over {tested} mutations")

    out["fails"] = fails
    return out


def main():
    ok_all = True
    for tag, spec in CASES.items():
        if not os.path.exists(spec["info"]) or not os.path.exists(spec["index"]):
            print(f"[{tag}] SKIP: missing {spec['info']} or {spec['index']}")
            ok_all = False
            results[tag] = {"tag": tag, "fails": ["missing input"], "inferred_depth": None}
            continue
        tok = AutoTokenizer.from_pretrained(MODEL)
        r = audit(tag, spec, tok)
        results[tag] = r
        if r.get("fails"):
            ok_all = False
        print()

    print("=" * 86)
    print("STAGE 2.5 VERIFICATION SUMMARY")
    print("=" * 86)
    for tag in ("P0", "LETTER"):
        r = results.get(tag, {})
        print(f"[{tag}]")
        print(f"  items                 = {r.get('n_items')}")
        print(f"  inferred_depth        = {r.get('inferred_depth')}")
        print(f"  prefix_index          = {r.get('prefix_index')}")
        print(f"  trie_keys             = {r.get('trie_keys')}")
        print(f"  encode_decode_pass    = {r.get('encode_decode_pass')}")
        print(f"  full_sid_to_eos_pass  = {r.get('full_sid_to_eos_pass')}")
        print(f"  early_eos_rejected    = {r.get('early_eos_rejected')}")
        print(f"  invalid_path_rejected = {r.get('invalid_path_rejected')}")
        print(f"  fails                 = {r.get('fails')}")
        print()

    p0 = results.get("P0", {})
    tx = results.get("LETTER", {})
    exp = (p0.get("inferred_depth") == 3 and tx.get("inferred_depth") == 4)
    print(f"  expected depths 3 / 4 ................ {exp}")
    print(f"  RESULT: {'ALL PASS' if (ok_all and exp) else 'FAIL'}")
    json.dump(results, open("/tmp/s25/verify_results.json", "w"), indent=2)
    return 0 if (ok_all and exp) else 1


if __name__ == "__main__":
    sys.exit(main())
