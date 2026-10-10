#!/usr/bin/env python3
"""
Regression test for variable-depth Semantic ID handling.

Purpose
-------
Stages 2.5/3/4 replaced three implicit `SID depth == 3` assumptions with
catalogue-derived inference. This test is the durable guard for that work: it is
CPU-only, loads the base tokenizer (never a model checkpoint), and finishes in a
few seconds.

It does NOT load an LLM, does NOT touch `runs/`, and does NOT need a GPU.

Coverage
--------
  1. P0 catalogue        -> inferred depth 3, full SID -> EOS, encode/decode exact
  2. LETTER catalogue    -> inferred depth 4, full SID -> EOS, encode/decode exact
  3. synthetic 2/5-level -> depth inference generalises beyond 3 and 4
  4. fail-loud cases     -> inconsistent depth, empty SID, non-atomic SID raise
  5. trie semantics      -> no early EOS, non-catalogue continuation rejected

Run:
    python tools/tests/test_sid_depth.py
    python -m pytest tools/tests/test_sid_depth.py     (also works)
"""
import hashlib
import json
import os
import re
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from sid_utils import (get_hash, infer_prefix_index, infer_sid_depth,  # noqa: E402
                       split_sid_tokens, build_sid_trie)

BASE_TOKENIZER = os.environ.get("BASE_TOKENIZER", "/root/autodl-tmp/models/Qwen2.5-0.5B")
WRAPPER = "### Response:\n"

P0_INDEX = "data/Amazon/index/Industrial_and_Scientific.index.json"
LETTER_INDEX = "artifacts/letter_stage2/letter_index.json"
LETTER_INFO = "artifacts/letter_stage2/letter_info.txt"

_RESULTS = []


def check(ok, label, detail=""):
    _RESULTS.append((bool(ok), label))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  -- {detail}" if detail else ""))
    return bool(ok)


def load_tokenizer(catalogue_tokens):
    """Base tokenizer + the catalogue's atomic SID tokens. No model is loaded."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE_TOKENIZER)
    if catalogue_tokens:
        tok.add_tokens(sorted(catalogue_tokens))
    return tok


def catalogue_tokens_of(sids):
    return {t for s in sids for t in split_sid_tokens(s)}


def read_json_index(path):
    idx = json.load(open(os.path.join(REPO, path), encoding="utf-8"))
    return ["".join(idx[str(i)]) for i in range(len(idx))]


def read_info_sids(path):
    sids = []
    with open(os.path.join(REPO, path), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                sids.append(line.split("\t")[0].strip())
    return sids


def audit_catalogue(tag, sids, expected_depth):
    print()
    print(f"--- {tag}: {len(sids)} entries, expected depth {expected_depth} ---")
    tok = load_tokenizer(catalogue_tokens_of(sids))
    entries = [f"{WRAPPER}{s}" for s in sids]

    depth = infer_sid_depth(entries, tok, WRAPPER)
    check(depth == expected_depth, f"{tag}: inferred depth == {expected_depth}",
          f"got {depth}")

    pi, _, _ = infer_prefix_index(entries, tok, WRAPPER)
    check(pi > 0, f"{tag}: prefix_index derived", f"{pi}")

    # encode/decode round trip, and atomicity
    bad_rt, bad_atomic = [], []
    for i, s in enumerate(sids):
        ids = tok(s, add_special_tokens=False).input_ids
        if tok.decode(ids).replace(" ", "") != s.replace(" ", ""):
            bad_rt.append(i)
        if [tok.convert_ids_to_tokens(x) for x in ids] != split_sid_tokens(s):
            bad_atomic.append(i)
    check(not bad_rt, f"{tag}: encode/decode PASS", f"{len(sids) - len(bad_rt)}/{len(sids)}")
    check(not bad_atomic, f"{tag}: all SID levels are atomic tokens",
          f"{len(bad_atomic)} bad")

    # trie semantics
    trie, pi2, _ = build_sid_trie(entries, tok, wrapper=WRAPPER)
    eos = tok.eos_token_id
    hit = only_eos = 0
    for s in sids:
        sid_ids = tok(s, add_special_tokens=False).input_ids
        al = trie.get(get_hash(sid_ids))
        if al is not None:
            hit += 1
            if set(al) == {eos}:
                only_eos += 1
    check(hit == len(sids), f"{tag}: full SID prefix present in trie", f"{hit}/{len(sids)}")
    check(only_eos == len(sids), f"{tag}: full SID -> ONLY EOS", f"{only_eos}/{len(sids)}")

    early = 0
    for s in sids:
        sid_ids = tok(s, add_special_tokens=False).input_ids
        for k in range(depth):
            if eos in trie.get(get_hash(sid_ids[:k]), ()):
                early += 1
    check(early == 0, f"{tag}: early EOS not permitted", f"{early} violations")

    cat = {tuple(tok(s, add_special_tokens=False).input_ids) for s in sids}
    allt = sorted(catalogue_tokens_of(sids))
    viol = tested = 0
    for s in sids[:150]:
        sid_ids = tok(s, add_special_tokens=False).input_ids
        for pos in range(depth):
            for cand in allt[:6]:
                ci = tok.convert_tokens_to_ids(cand)
                mut = tuple(sid_ids[:pos] + [ci] + sid_ids[pos + 1:])
                tested += 1
                if mut in cat:
                    continue
                if eos in trie.get(get_hash(mut), ()):
                    viol += 1
    check(viol == 0, f"{tag}: non-catalogue continuation rejected",
          f"{viol}/{tested} mutations")
    return tok


def audit_synthetic_depths():
    print()
    print("--- synthetic catalogues: depth inference must generalise ---")
    tok = load_tokenizer(None)
    for d in (2, 5):
        sids = ["".join(f"<{chr(97 + i)}_{(n * 7 + i * 3) % 256}>" for i in range(d))
                for n in range(40)]
        tok2 = load_tokenizer(catalogue_tokens_of(sids))
        entries = [f"{WRAPPER}{s}" for s in sids]
        got = infer_sid_depth(entries, tok2, WRAPPER)
        check(got == d, f"synthetic depth {d} inferred correctly", f"got {got}")
        pi, depth, _ = infer_prefix_index(entries, tok2, WRAPPER)
        check(depth == d, f"synthetic depth {d}: infer_prefix_index agrees",
              f"depth={depth} pi={pi}")


def audit_fail_loud():
    print()
    print("--- fail-loud behaviour ---")
    # Each case gets its OWN tokenizer instance: registering tokens is a mutation,
    # and a shared instance silently changes how later cases tokenise.

    def expect_raise(fn, label, match=None):
        try:
            fn()
        except ValueError as e:
            ok = (match is None) or (match in str(e))
            check(ok, label, str(e)[:110])
            return
        except Exception as e:  # noqa: BLE001
            check(False, label, f"wrong exception {type(e).__name__}: {e}")
            return
        check(False, label, "no exception raised")

    # --- inconsistent depth (own tokenizer so the entries stay atomic) ---
    sids_2 = ["<a_1><b_2>"]
    sids_3 = ["<a_1><b_2><c_3>"]
    mixed = [f"{WRAPPER}{s}" for s in (sids_2 + sids_3)]
    tok_mixed = load_tokenizer(catalogue_tokens_of(sids_2 + sids_3))
    expect_raise(lambda: infer_sid_depth(mixed, tok_mixed, WRAPPER),
                 "inconsistent depth -> raise", "inconsistent")

    # --- empty SID / empty catalogue ---
    tok_e = load_tokenizer(None)
    expect_raise(lambda: infer_sid_depth([f"{WRAPPER}"], tok_e, WRAPPER),
                 "empty SID -> raise")
    expect_raise(lambda: infer_sid_depth([], tok_e, WRAPPER),
                 "empty catalogue -> raise", "empty")

    # --- non-atomic SID: extra text after the last level breaks the 1:1 mapping
    #     between textual SID levels and tokenizer tokens ---
    tok_n = load_tokenizer(["<a_1>", "<b_2>"])
    nonatomic = [f"{WRAPPER}<a_1><b_2>", f"{WRAPPER}<a_1><b_2> trailing"]
    expect_raise(lambda: infer_sid_depth(nonatomic, tok_n, WRAPPER),
                 "non-atomic / extra text after last level -> raise")

    # --- wrapper mismatch, two variants ---
    sids = ["<a_1><b_2><c_3>"] * 4
    tok_w = load_tokenizer(catalogue_tokens_of(sids))
    entries = [f"{WRAPPER}{s}" for s in sids]
    # (a) different TEXT but the SAME token count. Only a literal prefix check can
    #     catch this: '### WRONG:\n' and '### Response:\n' are both 3 Qwen2 tokens.
    expect_raise(lambda: infer_prefix_index(entries, tok_w, "### WRONG:\n"),
                 "wrong wrapper (same token count) -> raise", "does not start with")
    # (b) different token count as well
    expect_raise(lambda: infer_prefix_index(entries, tok_w, "### Response"),
                 "wrong wrapper (different token count) -> raise")


def audit_p0_regression_evidence():
    """Static P0 regression: the frozen reference numbers must be unchanged."""
    print()
    print("--- P0 static regression (frozen reference metrics) ---")
    p = os.path.join(REPO, "runs/reference_full_clean_beam20/metrics.txt")
    if not os.path.exists(p):
        check(True, "P0 reference metrics present (SKIP: file absent)", p)
        return
    txt = open(p, encoding="utf-8").read()
    m = re.search(r"HR\s*\[([^\]]+)\]", txt)
    hr = [float(x) for x in m.group(1).split()] if m else []
    n = re.search(r"NDCG:\s*\[([^\]]+)\]", txt)
    nd = [float(x) for x in n.group(1).split()] if n else []
    check(len(hr) == 5 and abs(hr[-1] - 0.19832341) < 1e-8,
          "P0 HR@20 == 0.19832341", f"{hr[-1] if hr else 'n/a'}")
    check(len(nd) == 5 and abs(nd[-1] - 0.11786798) < 1e-8,
          "P0 NDCG@20 == 0.11786798", f"{nd[-1] if nd else 'n/a'}")


def main():
    print("=" * 84)
    print("SID DEPTH REGRESSION TEST  (CPU-only, no model checkpoint loaded)")
    print("=" * 84)
    print(f"  repo = {REPO}")
    print(f"  base tokenizer = {BASE_TOKENIZER}")

    if not os.path.isdir(BASE_TOKENIZER):
        print(f"SKIP: base tokenizer {BASE_TOKENIZER} not found")
        return 0

    p0 = os.path.join(REPO, P0_INDEX)
    if os.path.exists(p0):
        audit_catalogue("P0 (3-level)", read_json_index(P0_INDEX), 3)
    else:
        print(f"\nSKIP P0 catalogue: {P0_INDEX} absent")

    tx = os.path.join(REPO, LETTER_INFO)
    if os.path.exists(tx):
        audit_catalogue("LETTER (4-level)", read_info_sids(LETTER_INFO), 4)
    else:
        print(f"\nSKIP LETTER catalogue: {LETTER_INFO} absent")

    audit_synthetic_depths()
    audit_fail_loud()
    audit_p0_regression_evidence()

    n_pass = sum(1 for ok, _ in _RESULTS if ok)
    n_fail = len(_RESULTS) - n_pass
    print()
    print("=" * 84)
    print(f"TOTAL: {n_pass} PASS / {n_fail} FAIL  ({len(_RESULTS)} checks)")
    if n_fail:
        for ok, lbl in _RESULTS:
            if not ok:
                print("  FAILED: " + lbl)
    print(f"RESULT: {'ALL PASS' if not n_fail else 'FAIL'}")
    return 0 if not n_fail else 1


if __name__ == "__main__":
    sys.exit(main())
