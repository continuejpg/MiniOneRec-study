#!/usr/bin/env python3
"""
READ-ONLY diagnostic: what does the existing GRPO rollout generation actually
return, and is the index a legitimate rank?

Reproduces the rollout generation configuration EXACTLY as ReReTrainer builds it
(minionerec_trainer.py:478-491 + :684-694) and inspects the raw output of
`generate()` without changing anything:

  * scores=True   -> sequences_scores
  * output_scores -> per-step scores
  * return_dict_in_generate

For 16 prompts (num_generations=16 each) it answers:
  A. is the returned order sorted by final sequence score (descending)?
  B. can `i % 16` be read as a generation rank?
  C. if not, what is the minimal explicit rank computation?

Writes nothing except a small JSON report under artifacts/ (read-only w.r.t. code).
"""
import json
import math
import os
import sys

import torch
from transformers import (AutoTokenizer, AutoModelForCausalLM, GenerationConfig,
                          LogitsProcessorList, TemperatureLogitsWarper)

REPO = os.getcwd()
sys.path.insert(0, REPO)
from sid_utils import infer_prefix_index, get_hash          # noqa: E402
from LogitProcessor import ConstrainedLogitsProcessor       # noqa: E402
from data import EvalSidDataset                             # noqa: E402

CKPT = "runs/industrial_sft/final_checkpoint"
CAT = "Industrial_and_Scientific"
TEST = f"data/Amazon/test/{CAT}_5_2016-10-2018-11.csv"
INFO = f"data/Amazon/info/{CAT}_5_2016-10-2018-11.txt"
G = 16                       # num_generations
N_PROMPTS = 4                # 4 prompts x 16 = 64 sequences
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


def seq_logprob_from_scores(scores, seq, prompt_len, eos_id):
    """Mean log-prob of the generated part, from the per-step score tensors."""
    lp = []
    for step, sc in enumerate(scores):
        pos = prompt_len + step
        if pos >= seq.shape[0]:
            break
        tokid = int(seq[pos].item())
        logp = torch.log_softmax(sc[0].float(), dim=-1)[tokid].item()
        lp.append(logp)
        if tokid == eos_id:
            break
    return (sum(lp) / len(lp) if lp else float("nan")), len(lp)


def main():
    print("=" * 84)
    print("ROLLOUT GENERATION DIAGNOSTIC (read-only)")
    print("=" * 84)

    tok = AutoTokenizer.from_pretrained(CKPT)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    hd, pi, depth = build_hash_dict(INFO, tok)
    print(f"  tokenizer len={len(tok)}  prefix_index={pi}  depth={depth}  "
          f"trie keys={len(hd)}")

    ds = EvalSidDataset(train_file=TEST, tokenizer=tok, max_len=2560,
                        test=True, category="industrial and scientific items")
    idxs = [0, 100, 500, 1000]
    batch = [ds[i] for i in idxs]
    L = max(len(b["input_ids"]) for b in batch)
    pad = tok.pad_token_id
    input_ids = torch.tensor(
        [[pad] * (L - len(b["input_ids"])) + b["input_ids"] for b in batch]).to(DEVICE)
    attn = torch.tensor(
        [[0] * (L - len(b["input_ids"])) + [1] * len(b["input_ids"]) for b in batch]).to(DEVICE)
    print(f"  prompts={len(batch)}  padded prompt_len={L}")

    model = AutoModelForCausalLM.from_pretrained(CKPT, torch_dtype=torch.bfloat16,
                                                 device_map="auto").eval()

    # EXACT copy of ReReTrainer's beam_search branch
    gc = GenerationConfig(max_new_tokens=MAXNEW, length_penalty=1.0,
                          num_beams=G, num_return_sequences=G,
                          pad_token_id=tok.pad_token_id,
                          eos_token_id=tok.eos_token_id,
                          top_k=None, top_p=None, temperature=TEMP, do_sample=True)
    print(f"\n  generation mode = {gc.get_generation_mode()}")
    ccc = ConstrainedLogitsProcessor(
        prefix_allowed_tokens_fn=lambda b, ids: hd.get(get_hash(ids), []),
        num_beams=G, base_model=CKPT, eos_token_id=tok.eos_token_id)
    lp = LogitsProcessorList([TemperatureLogitsWarper(temperature=TEMP), ccc])

    with torch.no_grad():
        out = model.generate(input_ids, attention_mask=attn, generation_config=gc,
                             logits_processor=lp, return_dict_in_generate=True,
                             output_scores=True)

    seqs = out.sequences
    print(f"  returned sequences shape = {tuple(seqs.shape)}   "
          f"(expect {len(batch)*G} x ...)")
    has_ss = out.sequences_scores is not None
    print(f"  sequences_scores present = {has_ss}")
    if has_ss:
        print(f"    shape={tuple(out.sequences_scores.shape)}")
    print(f"  transition_scores attr   = "
          f"{hasattr(out, 'transition_scores')}")

    gen = seqs[:, L:]
    texts = tok.batch_decode(gen, skip_special_tokens=True)
    texts = [t.split("Response:\n")[-1].strip() for t in texts]

    rep = {"generation_mode": str(gc.get_generation_mode()),
           "num_beams": G, "num_return_sequences": G, "do_sample": True,
           "sequences_shape": list(seqs.shape),
           "sequences_scores_present": bool(has_ss),
           "groups": []}

    print()
    print("=" * 84)
    print("A. IS THE RETURNED ORDER SORTED BY SEQUENCE SCORE (DESC)?")
    print("=" * 84)
    n_sorted = 0
    for g in range(len(batch)):
        sl = slice(g * G, (g + 1) * G)
        grp_txt = texts[sl]
        if has_ss:
            sc = out.sequences_scores[sl].float().tolist()
        else:
            sc = [seq_logprob_from_scores(out.scores, seqs[i], L, tok.eos_token_id)[0]
                  for i in range(sl.start, sl.stop)]
        desc = all(sc[i] >= sc[i + 1] - 1e-9 for i in range(len(sc) - 1))
        asc = all(sc[i] <= sc[i + 1] + 1e-9 for i in range(len(sc) - 1))
        n_sorted += int(desc)
        uniq = len(set(grp_txt))
        print(f"  prompt {g}: sorted_desc={desc}  sorted_asc={asc}  "
              f"unique_candidates={uniq}/{G}")
        print(f"     scores[:6] = {[round(x,4) for x in sc[:6]]}")
        print(f"     scores[-3:] = {[round(x,4) for x in sc[-3:]]}")
        print(f"     cand[0:3]  = {grp_txt[:3]}")
        rep["groups"].append({"prompt": g, "sorted_desc": bool(desc),
                              "sorted_asc": bool(asc), "unique": uniq,
                              "scores": [float(x) for x in sc]})

    print()
    print(f"  groups sorted descending by score: {n_sorted}/{len(batch)}")

    print()
    print("=" * 84)
    print("B. DOES THE RETURNED ORDER MATCH DECODED-STRING SORT ORDER?")
    print("=" * 84)
    for g in range(len(batch)):
        sl = slice(g * G, (g + 1) * G)
        grp = texts[sl]
        lex = sorted(grp)
        print(f"  prompt {g}: order==lexicographic_sorted? {grp == lex}")

    print()
    print("=" * 84)
    print("C. ARE PUTATIVE GT HITS SPREAD NON-MONOTONICALLY ACROSS THE 16 SLOTS?")
    print("=" * 84)
    print("  (if hits appear at scattered indices, the index carries no rank meaning)")
    import csv
    with open(TEST, encoding="utf-8", newline="") as f:
        rr = csv.reader(f)
        hdr = next(rr)
        rows = [r for r in rr]
    for g, i in enumerate(idxs):
        tgt = rows[i][hdr.index("item_sid")].strip()
        sl = slice(g * G, (g + 1) * G)
        grp = [t.strip() for t in texts[sl]]
        hit_pos = [j for j, t in enumerate(grp) if t == tgt]
        print(f"  prompt {g}: target={tgt}  hit_positions={hit_pos}  "
              f"hit_count={len(hit_pos)}")

    out_json = "artifacts/rl_audit/rollout_order_diagnostic.json"
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    json.dump(rep, open(out_json, "w"), indent=2)
    print(f"\n  [save] {out_json}  (analysis artifact only; no code changed)")


if __name__ == "__main__":
    main()
