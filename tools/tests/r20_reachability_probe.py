#!/usr/bin/env python3
"""
R2.0 -- Reachability feasibility probe (READ-ONLY).

Does NOT modify training code, does NOT start training or eval. Loads the frozen
original SFT checkpoint, reproduces the existing constrained rollout exactly, and
measures how reachable the ground-truth SID is as a function of how much of it is
hinted.

Design note (this is the crux of the feasibility question)
---------------------------------------------------------
For the hint to be *free* -- i.e. not to receive policy credit, and not to corrupt
the GRPO ratio -- the hinted tokens must live inside the PROMPT, not inside the
completion. That is the only layout in which:

  * completion_ids contains exactly the sampled suffix,
  * per_token_logps is computed over the suffix only,
  * ref_per_token_logps is computed over the same suffix,
  * completion_mask needs no change at all,
  * `_prepare_inputs`'s prompt_length bookkeeping stays consistent.

This probe therefore builds the hinted prompt as
    <standard prompt ending in "### Response:\n"> + <h hinted GT SID tokens>
and asks the model to continue. The constrained trie is left untouched: the
decoder's own hash lookup starts from the hinted prefix and permits only legal
continuations.

Two levels of measurement are reported separately:
  ORACLE reachability  -- does the catalogue contain a SID extending the hint?
                          Deterministic, no model involved, exact upper bound.
  MODEL  reachability  -- does the sampled rollout actually contain the GT?
                          This is the number the GRPO reward sees.

Run:
    python tools/tests/r20_reachability_probe.py
"""
import collections
import csv
import json
import math
import os
import statistics
import sys

import torch
from transformers import (AutoTokenizer, AutoModelForCausalLM, GenerationConfig,
                          LogitsProcessorList, TemperatureLogitsWarper)

REPO = os.getcwd()
sys.path.insert(0, REPO)
import rere_reward as rr                                    # noqa: E402
from sid_utils import infer_prefix_index, get_hash           # noqa: E402
from LogitProcessor import ConstrainedLogitsProcessor        # noqa: E402
from data import SidDataset                                  # noqa: E402

CKPT = "runs/industrial_sft/final_checkpoint"
CAT = "Industrial_and_Scientific"
TRAIN = f"data/Amazon/train/{CAT}_5_2016-10-2018-11.csv"
INDEX = f"data/Amazon/index/{CAT}.index.json"
INFO = f"data/Amazon/info/{CAT}_5_2016-10-2018-11.txt"
INDEX_TRAIN = "splits/grpo_seq_10k.json"

G = 16
TEMP = 1.0
MAXNEW = 128
N_SAMPLES = 64
WRAPPER = "### Response:\n"
PROMPT_TAIL = " in chronological order. Can you predict the next possible item that the user may expect?"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------- catalogue
def load_catalogue():
    idx = json.load(open(INDEX, encoding="utf-8"))
    sid_of = {int(k): "".join(v) for k, v in idx.items()}
    by_id = {int(k): [t for t in v] for k, v in idx.items()}
    sids = set(sid_of.values())
    # prefix reachability: prefix (tuple of levels) -> True if any SID extends it
    prefixes = collections.defaultdict(int)
    for lv in by_id.values():
        for h in range(0, len(lv) + 1):
            prefixes[tuple(lv[:h])] += 1
    return sid_of, by_id, sids, dict(prefixes)


def load_trie(tok):
    sids = []
    with open(INFO, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                sids.append(line.split("\t")[0].strip())
    entries = [f"{WRAPPER}{s}\n" for s in sids]
    pi, depth, _ = infer_prefix_index(entries, tok, WRAPPER)
    hd = {}
    for e in entries:
        ID = list(tok(e, add_special_tokens=False).input_ids)
        ID.append(tok.eos_token_id)
        for i in range(pi, len(ID)):
            hn = get_hash(ID[:i]) if i == pi else get_hash(ID[pi:i])
            hd.setdefault(hn, set()).add(ID[i])
    return {k: sorted(v) for k, v in hd.items()}, pi, depth


def history_sid_str(row, sid_of):
    import ast
    h = ast.literal_eval(row["history_item_id"])
    return ", ".join(sid_of[int(x)] for x in h)


def build_hinted_prompt(instruction_ids, history_str, hint_tokens, tok):
    """instruction + history + PROMPT_TAIL, then the hinted GT SID tokens.

    Returns (prompt_text, n_hint_tokens). The hint is placed AFTER the standard
    prompt so that the trie constrains the continuation and the completion holds
    only the sampled suffix.
    """
    base = instruction_ids + history_str + PROMPT_TAIL
    if not hint_tokens:
        return base, 0
    hint = "".join(hint_tokens)
    return base + hint, len(tok(hint, add_special_tokens=False).input_ids)


def main():
    os.makedirs("artifacts/rl_audit", exist_ok=True)
    print("=" * 100)
    print("R2.0 REACHABILITY PROBE (read-only, no training code modified)")
    print("=" * 100)

    sid_of, by_id, all_sids, prefixes = load_catalogue()
    id_of_sid = {v: k for k, v in sid_of.items()}
    print(f"  catalogue: {len(sid_of)} items, {len(all_sids)} unique SID")

    tok = AutoTokenizer.from_pretrained(CKPT)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    hd, pi, depth = load_trie(tok)
    print(f"  tokenizer len={len(tok)}  prefix_index={pi}  depth={depth}  "
          f"trie keys={len(hd)}")

    # ---- 64 training samples via SidDataset (same class the trainer uses) ----
    ds = SidDataset(train_file=TRAIN, max_len=512, sample=-1,
                    seed=42, category="industrial and scientific items")
    subset = json.load(open(INDEX_TRAIN, encoding="utf-8"))
    ids = subset["sample_ids"] if isinstance(subset, dict) and "sample_ids" in subset else None
    print(f"  SidDataset len={len(ds)}   subset file sample_ids={len(ids) if ids else 'n/a'}")

    # take the first N_SAMPLES samples from the frozen seq subset if usable
    picked = []
    if ids:
        want = set(ids[:N_SAMPLES])
        for i in range(len(ds)):
            ex = ds[i]
            if ex.get("sample_id") in want:
                picked.append(ex)
            if len(picked) >= N_SAMPLES:
                break
    if len(picked) < N_SAMPLES:
        picked = [ds[i] for i in range(N_SAMPLES)]
    print(f"  picked {len(picked)} training samples")

    # ---- instruction prefix, taken from the dataset's own prompt text ----
    # SidDataset builds: pre-instruction + pre-prompt + "### Response:\n"
    # Recover the "instruction + history" head by locating the wrapper.
    def split_prompt(p):
        k = p.rfind(WRAPPER)
        return p[:k], p[k:]

    # ---------- ORACLE reachability (no model) ----------
    print()
    print("=" * 100)
    print("A. ORACLE REACHABILITY (deterministic; upper bound for any model)")
    print("=" * 100)
    oracle = {0: 0, 1: 0, 2: 0}
    oracle_unresolved_at = {2: []}
    for ex in picked:
        tgt = ex["completion"].strip()
        lv = by_id[id_of_sid[tgt]]
        for h in (0, 1, 2):
            if prefixes.get(tuple(lv[:h]), 0) > 0:
                oracle[h] += 1
    n = len(picked)
    for h in (0, 1, 2):
        print(f"  h={h}: a catalogue SID extends the {h}-token prefix in "
              f"{oracle[h]}/{n} samples  ({100.0*oracle[h]/n:.1f}%)")
    print("  (= 100% by construction for h=0/1/2, since the GT itself extends it)")

    # ---------- MODEL reachability ----------
    model = AutoModelForCausalLM.from_pretrained(CKPT, torch_dtype=torch.bfloat16,
                                                 device_map="auto").eval()
    gc = GenerationConfig(max_new_tokens=MAXNEW, length_penalty=1.0,
                          num_beams=G, num_return_sequences=G,
                          pad_token_id=tok.pad_token_id,
                          eos_token_id=tok.eos_token_id,
                          top_k=None, top_p=None, temperature=TEMP, do_sample=True)
    print()
    print(f"  generation mode = {gc.get_generation_mode()}  (unchanged)")

    results = {}
    for h in (0, 1, 2):
        print()
        print("=" * 100)
        print(f"B. MODEL REACHABILITY  h={h}  (hint = first {h} GT SID token(s))")
        print("=" * 100)
        hit_groups = 0
        legal_ok = legal_bad = 0
        stds = []
        per_sample_min_h = {}
        for si, ex in enumerate(picked):
            tgt = ex["completion"].strip()
            head, tail = split_prompt(ex["prompt"])
            # hint tokens = first h levels of the GT SID.
            # The hint goes where the SFT target itself starts, i.e. immediately
            # AFTER "### Response:\n" -- that is the answer slot, so the prompt
            # keeps the exact format the model was SFT'd on and the hint is a
            # natural prefix of the target rather than trailing noise.
            lv = by_id[id_of_sid[tgt]]
            hint = lv[:h]
            prompt_text = head + tail + "".join(hint)
            enc = tok(prompt_text, return_tensors="pt", add_special_tokens=False)
            pids = enc["input_ids"].to(DEVICE)
            am = enc["attention_mask"].to(DEVICE)
            # Runtime injection of the count_0 fix. NO source file is modified:
            # this subclass lives only inside the probe process. It demonstrates
            # the minimal change found by the interface check (count_0 = h) end to
            # end, without touching LogitProcessor.py or minionerec_trainer.py.
            class _HintedConstrainedLogitsProcessor(ConstrainedLogitsProcessor):
                def __init__(self, *a, count_0=0, **kw):
                    super().__init__(*a, **kw)
                    self.count = int(count_0)

            ccc = _HintedConstrainedLogitsProcessor(
                prefix_allowed_tokens_fn=lambda b, i: hd.get(get_hash(i), []),
                num_beams=G, base_model=CKPT, eos_token_id=tok.eos_token_id,
                count_0=h)
            lp = LogitsProcessorList([TemperatureLogitsWarper(temperature=TEMP), ccc])
            with torch.no_grad():
                out = model.generate(pids, attention_mask=am, generation_config=gc,
                                     logits_processor=lp,
                                     return_dict_in_generate=True, output_scores=True)
            hint_len = len(tok("".join(hint), add_special_tokens=False).input_ids) if hint else 0
            gen = out.sequences[:, pids.shape[1]:]
            txt = tok.batch_decode(gen, skip_special_tokens=True)
            # completion holds ONLY the sampled suffix; rebuild full SID to test GT
            fulls = []
            for t in txt:
                c = t.split("Response:\n")[-1]
                # strip any leaked wrapper text and keep only SID tokens
                import re as _re
                toks = _re.findall(r"<[a-z]_\d+>", c)
                fulls.append("".join(hint) + "".join(toks))
            flags = [f == tgt for f in fulls]
            if any(flags):
                hit_groups += 1
            legal_ok += sum(1 for f in fulls if f in all_sids)
            legal_bad += sum(1 for f in fulls if f not in all_sids)
            total, rule, rank = rr.group_rewards(flags)
            stds.append(statistics.pstdev(total))

        print(f"  samples                = {n}")
        print(f"  GT-hit groups          = {hit_groups}  "
              f"({100.0*hit_groups/n:.1f}%)")
        print(f"  reward_std   mean      = {statistics.mean(stds):.6f}   "
              f"median = {statistics.median(stds):.6f}")
        print(f"  reward_std == 0 groups = {sum(1 for s in stds if s == 0)}")
        tot_c = legal_ok + legal_bad
        print(f"  candidates legal       = {legal_ok}/{tot_c} "
              f"({100.0*legal_ok/tot_c:.2f}%)")
        results[h] = {"hit_groups": hit_groups, "ratio": hit_groups / n,
                      "std_mean": statistics.mean(stds),
                      "std_median": statistics.median(stds),
                      "zero_std_groups": sum(1 for s in stds if s == 0),
                      "legal_frac": legal_ok / tot_c}

    print()
    print("=" * 100)
    print("C. VERDICT NUMBERS")
    print("=" * 100)
    for h in (0, 1, 2):
        r = results[h]
        print(f"  h={h}: GT-hit group ratio = {r['ratio']:.4f} "
              f"({r['hit_groups']}/{n})   reward_std median = {r['std_median']:.6f}")

    hard = n - results[0]["hit_groups"]
    rec1 = max(0, results[1]["hit_groups"] - results[0]["hit_groups"])
    rec2 = max(0, results[2]["hit_groups"] - results[0]["hit_groups"])
    unresolved = n - results[2]["hit_groups"]
    print()
    print(f"  h=0 miss (hard cases)        = {hard}/{n}")
    print(f"  recovered by h=1 (vs h=0)    = {rec1}   "
          f"({100.0*rec1/hard if hard else 0:.1f}% of hard cases)")
    print(f"  recovered by h=2 (vs h=0)    = {rec2}   "
          f"({100.0*rec2/hard if hard else 0:.1f}% of hard cases)")
    print(f"  unresolved at h=2            = {unresolved}/{n} "
          f"({100.0*unresolved/n:.1f}%)")

    json.dump({"n_samples": n, "results": {str(k): v for k, v in results.items()},
               "hard_cases": hard, "recovered_h1": rec1, "recovered_h2": rec2,
               "unresolved_h2": unresolved},
              open("artifacts/rl_audit/r20_reachability.json", "w"), indent=2)
    print(f"\n  [save] artifacts/rl_audit/r20_reachability.json")


if __name__ == "__main__":
    main()
