#!/usr/bin/env python3
"""Stop reasoning; RUN the real constrained generation and observe.

For each index (P0 3-level, TX 4-level) build the official trie, then run the
repository's own generate() path on a few real test rows with the matching
checkpoint and report:
  * whether generation completes
  * how many of the emitted candidates are legal catalogue SIDs
  * the emitted sequence length
This is the only test that can settle the prefix_index question, because the
decoder's hash_key bookkeeping is stateful across steps.
"""
import hashlib
import json
import os
import sys

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList
from transformers.generation.logits_process import TemperatureLogitsWarper

sys.path.insert(0, os.getcwd())
from sft import TokenExtender
from data import EvalSidDataset
from LogitProcessor import ConstrainedLogitsProcessor

MODEL_LLM = "runs/industrial_sft/final_checkpoint"
BASE = "/root/autodl-tmp/models/Qwen2.5-0.5B"
CAT = "Industrial_and_Scientific"
TEST = f"data/Amazon/test/{CAT}_5_2016-10-18-11.csv"
TEST = f"data/Amazon/test/{CAT}_5_2016-10-2018-11.csv"
SCRATCH = "/tmp/pi_scratch"
CASES = {"P0-3level": "data/Amazon/index/Industrial_and_Scientific.index.json",
         "TX-4level": "artifacts/letter_stage2/letter_index.json"}


def gh(t):
    return hashlib.md5(str(list(t)).encode()).hexdigest()


def load(path):
    tok = AutoTokenizer.from_pretrained(MODEL_LLM if os.path.isdir(MODEL_LLM) else BASE)
    os.makedirs(SCRATCH, exist_ok=True)
    link = os.path.join(SCRATCH, f"{CAT}.index.json")
    if os.path.islink(link) or os.path.exists(link):
        os.remove(link)
    os.symlink(os.path.abspath(path), link)
    tok.add_tokens(sorted(TokenExtender(data_path=SCRATCH, dataset=CAT).get_new_tokens()))
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


def main():
    print("=" * 88)
    print("REAL CONSTRAINED GENERATION (authoritative test)")
    print("=" * 88)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_LLM, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    print(f"  model = {MODEL_LLM}   device = {dev}")

    for tag, path in CASES.items():
        tok = load(path)
        idx = json.load(open(path, encoding="utf-8"))
        seqs = ["".join(idx[str(i)]) for i in range(len(idx))]
        legal = {tuple(tok(s).input_ids) for s in seqs}
        depth = len(tok(seqs[0]).input_ids)
        print(f"\n--- {tag}: depth={depth}  legal_SIDs={len(legal)}  vocab={len(tok)}")

        # official trie, prefix_index as the repo hardcodes (3)
        pi = 3
        hd = {}
        for s in seqs:
            ID = list(tok(s).input_ids) + [tok.eos_token_id]
            for i in range(pi, len(ID)):
                hn = gh(ID[:i]) if i == pi else gh(ID[pi:i])
                hd.setdefault(hn, set()).add(ID[i])
        hd = {k: sorted(v) for k, v in hd.items()}

        ds = EvalSidDataset(train_file=TEST, tokenizer=tok, max_len=2048,
                            test=True, category="industrial and scientific items")
        NB = 20
        ccc = ConstrainedLogitsProcessor(
            prefix_allowed_tokens_fn=lambda b, ids: hd.get(gh(ids), []),
            num_beams=NB, base_model=BASE, eos_token_id=tok.eos_token_id)
        lp = LogitsProcessorList([TemperatureLogitsWarper(temperature=1.0), ccc])

        hits = 0
        lens = []
        for i in range(5):
            ids = torch.tensor([ds[i]["input_ids"]]).to(dev)
            am = torch.ones_like(ids)
            with torch.no_grad():
                out = model.generate(ids, attention_mask=am, num_beams=NB,
                                     num_return_sequences=NB, max_new_tokens=8,
                                     do_sample=True, temperature=1.0,
                                     pad_token_id=tok.eos_token_id,
                                     logits_processor=lp)
            gen = out[:, ids.shape[1]:]
            for row in gen:
                core = []
                for v in row.tolist():
                    if v == tok.eos_token_id:
                        break
                    core.append(v)
                lens.append(len(core))
                if tuple(core) in legal:
                    hits += 1
        print(f"    generated {len(lens)} candidates over 5 prompts")
        print(f"    legal catalogue SIDs : {hits}/{len(lens)}")
        print(f"    emitted length hist  : "
              f"{ {k: lens.count(k) for k in sorted(set(lens))} }")
        print(f"    expected length      : {depth}")


if __name__ == "__main__":
    main()
