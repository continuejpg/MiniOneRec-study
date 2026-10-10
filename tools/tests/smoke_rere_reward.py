#!/usr/bin/env python3
"""
Pre-training smoke for the ReRe ranking reward (R1).

Uses REAL prompts and the REAL rollout configuration -- it does not change the
generation config, the constrained trie, or the model. For each sampled prompt it
prints the group's 16 candidates with R_rule, R_rank and R_total, plus the group
reward std, and asserts the invariants that training depends on:

  * at least one GT-hit group is present in the sample
  * every GT-miss group has reward std > 0 (the whole point of R1)
  * GT-miss groups are no longer zero-variance

Writes a small JSON report under artifacts/rl_audit/. No source file is modified.
"""
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
import rere_reward as rr                                       # noqa: E402
from sid_utils import infer_prefix_index, get_hash              # noqa: E402
from LogitProcessor import ConstrainedLogitsProcessor           # noqa: E402
from data import EvalSidDataset                                 # noqa: E402

CKPT = "runs/industrial_sft/final_checkpoint"
CAT = "Industrial_and_Scientific"
TEST = f"data/Amazon/test/{CAT}_5_2016-10-2018-11.csv"
INFO = f"data/Amazon/info/{CAT}_5_2016-10-2018-11.txt"
G = 16
TEMP = 1.0
MAXNEW = 128
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def build_hash_dict(info_path, tok):
    sids = []
    with open(info_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                sids.append(line.split("\t")[0].strip())
    entries = [f"### Response:\n{s}\n" for s in sids]
    pi, depth, _ = infer_prefix_index(entries, tok, "### Response:\n")
    hd = {}
    for e in entries:
        ID = list(tok(e, add_special_tokens=False).input_ids)
        ID.append(tok.eos_token_id)
        for i in range(pi, len(ID)):
            hn = get_hash(ID[:i]) if i == pi else get_hash(ID[pi:i])
            hd.setdefault(hn, set()).add(ID[i])
    return {k: sorted(v) for k, v in hd.items()}, pi, depth


def main():
    os.makedirs("artifacts/rl_audit", exist_ok=True)
    print("=" * 100)
    print("R1 PRE-TRAINING SMOKE -- ReRe ranking reward on REAL rollout")
    print("=" * 100)

    tok = AutoTokenizer.from_pretrained(CKPT)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    hd, pi, depth = build_hash_dict(INFO, tok)
    print(f"  tokenizer len={len(tok)}  prefix_index={pi}  depth={depth}  "
          f"trie keys={len(hd)}")

    # 32 prompts: enough to see GT hits (production hit rate is low)
    with open(TEST, encoding="utf-8", newline="") as f:
        rdr = csv.reader(f)
        hdr = next(rdr)
        rows = [r for r in rdr]
    n_prompts = 32
    idxs = list(range(0, n_prompts))
    ds = EvalSidDataset(train_file=TEST, tokenizer=tok, max_len=2560, test=True,
                        category="industrial and scientific items")
    batch = [ds[i] for i in idxs]
    L = max(len(b["input_ids"]) for b in batch)
    pad = tok.pad_token_id
    input_ids = torch.tensor(
        [[pad] * (L - len(b["input_ids"])) + b["input_ids"] for b in batch]).to(DEVICE)
    attn = torch.tensor(
        [[0] * (L - len(b["input_ids"])) + [1] * len(b["input_ids"]) for b in batch]).to(DEVICE)
    print(f"  prompts={n_prompts}  prompt_len={L}")

    model = AutoModelForCausalLM.from_pretrained(CKPT, torch_dtype=torch.bfloat16,
                                                 device_map="auto").eval()

    # EXACT copy of ReReTrainer's beam_search branch -- unchanged
    gc = GenerationConfig(max_new_tokens=MAXNEW, length_penalty=1.0,
                          num_beams=G, num_return_sequences=G,
                          pad_token_id=tok.pad_token_id,
                          eos_token_id=tok.eos_token_id,
                          top_k=None, top_p=None, temperature=TEMP, do_sample=True)
    print(f"  generation mode = {gc.get_generation_mode()}")
    ccc = ConstrainedLogitsProcessor(
        prefix_allowed_tokens_fn=lambda b, ids: hd.get(get_hash(ids), []),
        num_beams=G, base_model=CKPT, eos_token_id=tok.eos_token_id)
    lp = LogitsProcessorList([TemperatureLogitsWarper(temperature=TEMP), ccc])

    with torch.no_grad():
        out = model.generate(input_ids, attention_mask=attn, generation_config=gc,
                             logits_processor=lp, return_dict_in_generate=True,
                             output_scores=True)
    gen = out.sequences[:, L:]
    texts = tok.batch_decode(gen, skip_special_tokens=True)
    texts = [t.split("Response:\n")[-1].strip() for t in texts]
    scores = out.sequences_scores.float().tolist() if out.sequences_scores is not None else [0.0] * len(texts)

    print()
    print("  --- per-group reward (REAL rollout output) ---")
    n_hit = n_miss = 0
    miss_std, hit_std = [], []
    bad_miss = []
    shown_hit = shown_miss = 0
    report = {"generation_mode": str(gc.get_generation_mode()), "groups": []}

    for g in range(n_prompts):
        sl = slice(g * G, (g + 1) * G)
        grp = [t.strip() for t in texts[sl]]
        tgt = rows[idxs[g]][hdr.index("item_sid")].strip()
        flags = [c == tgt for c in grp]
        total, rule, rank = rr.group_rewards(flags)
        std = statistics.pstdev(total)
        gt_pos = [r for r in range(G) if flags[r]]
        if gt_pos:
            n_hit += 1
            hit_std.append(std)
        else:
            n_miss += 1
            miss_std.append(std)
            if std <= 0:
                bad_miss.append(g)
        # show the first 2 hit groups and the first 6 miss groups
        show = (gt_pos and shown_hit < 2) or ((not gt_pos) and shown_miss < 6)
        if show:
            if gt_pos:
                shown_hit += 1
            else:
                shown_miss += 1
            print()
            print(f"  group {g:2d}  target={tgt}  GT_rank={gt_pos if gt_pos else 'MISS'}"
                  f"  reward_std={std:.6f}")
            print(f"    {'r':>3} {'seq_score':>10}  {'candidate':<28} {'R_rule':>7} "
                  f"{'R_rank':>8} {'R_total':>8}")
            for r in range(G):
                mark = " <== GT" if flags[r] else ""
                print(f"    {r:>3} {scores[sl.start + r]:>10.4f}  {grp[r]:<28} "
                      f"{rule[r]:>7.4f} {rank[r]:>8.4f} {total[r]:>8.4f}{mark}")

        report["groups"].append({
            "group": g, "target": tgt, "gt_rank": gt_pos,
            "reward_std": std, "sum_r_rank": sum(rank),
            "candidates": grp, "rule": rule, "rank": rank, "total": total})

    print()
    print("=" * 100)
    print("SMOKE SUMMARY")
    print("=" * 100)
    print(f"  prompts            = {n_prompts}")
    print(f"  GT-hit groups      = {n_hit}  ({100.0*n_hit/n_prompts:.1f}%)")
    print(f"  GT-miss groups     = {n_miss}  ({100.0*n_miss/n_prompts:.1f}%)")
    if hit_std:
        print(f"  hit  group std     = min {min(hit_std):.6f}  median "
              f"{statistics.median(hit_std):.6f}  max {max(hit_std):.6f}")
    if miss_std:
        print(f"  miss group std     = min {min(miss_std):.6f}  median "
              f"{statistics.median(miss_std):.6f}  max {max(miss_std):.6f}")
    sums = [sum(gr["rank"]) for gr in report["groups"]]
    print(f"  sum(R_rank) per group: min {min(sums):.12f}  max {max(sums):.12f}")
    all_totals = [x for gr in report["groups"] for x in gr["total"]]
    print(f"  mean R_total       = {statistics.mean(all_totals):.6f}")

    fails = []
    if n_hit == 0:
        fails.append("no GT-hit group in the smoke sample")
    if bad_miss:
        fails.append(f"GT-miss groups with std<=0: {bad_miss}")
    report["n_hit"] = n_hit
    report["n_miss"] = n_miss
    report["miss_std_min"] = min(miss_std) if miss_std else None
    report["hit_std_min"] = min(hit_std) if hit_std else None
    report["failures"] = fails
    json.dump(report, open("artifacts/rl_audit/r1_reward_smoke.json", "w"), indent=2)
    print(f"\n  [save] artifacts/rl_audit/r1_reward_smoke.json")
    print()
    print(f"  failures = {fails}")
    print(f"  RESULT: {'ALL PASS' if not fails else 'FAIL'}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
