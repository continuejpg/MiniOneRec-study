#!/usr/bin/env python3
"""FINAL: reproduce evaluate.py's deterministic generation EXACTLY, instrumented.

evaluate.py builds GenerationConfig(num_beams, length_penalty, num_return_sequences,
max_new_tokens, top_k=None, top_p=None) WITHOUT do_sample -> deterministic.

We additionally wrap the logits processor to log, per step, the hash_key it used
and how many tokens were allowed. That reveals the true prefix_index contract.
"""
import hashlib
import json
import os
import sys

import torch
from transformers import (AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList,
                          GenerationConfig)

sys.path.insert(0, os.getcwd())
from sft import TokenExtender
from data import EvalSidDataset
from LogitProcessor import ConstrainedLogitsProcessor

LLM = "runs/industrial_sft/final_checkpoint"
BASE = "/root/autodl-tmp/models/Qwen2.5-0.5B"
CAT = "Industrial_and_Scientific"
TEST = f"data/Amazon/test/{CAT}_5_2016-10-2018-11.csv"
SCRATCH = "/tmp/pi_scratch"
CASES = {"P0-3level": "data/Amazon/index/Industrial_and_Scientific.index.json",
         "TX-4level": "artifacts/letter_stage2/letter_index.json"}


def gh(t):
    return hashlib.md5(str(list(t)).encode()).hexdigest()


def load(path):
    tok = AutoTokenizer.from_pretrained(LLM)
    os.makedirs(SCRATCH, exist_ok=True)
    link = os.path.join(SCRATCH, f"{CAT}.index.json")
    if os.path.islink(link) or os.path.exists(link):
        os.remove(link)
    os.symlink(os.path.abspath(path), link)
    tok.add_tokens(sorted(TokenExtender(data_path=SCRATCH, dataset=CAT).get_new_tokens()))
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


class Spy(ConstrainedLogitsProcessor):
    def __call__(self, input_ids, scores):
        sent = input_ids.view(-1, self._num_beams, input_ids.shape[-1])[0][0]
        key = sent[-self.prefix_index:] if self.count == 0 else sent[-self.count:]
        hits = len(self._prefix_allowed_tokens_fn(0, key.tolist()))
        if self.count < 8:
            print(f"      step {self.count}: count={self.count} key_len={len(key)} "
                  f"allowed={hits}  tail={key.tolist()[-4:]}")
        return super().__call__(input_ids, scores)


def main():
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(LLM, torch_dtype=torch.bfloat16,
                                                 device_map="auto").eval()
    print(f"  model = {LLM}")
    for tag, path in CASES.items():
        tok = load(path)
        idx = json.load(open(path, encoding="utf-8"))
        seqs = ["".join(idx[str(i)]) for i in range(len(idx))]
        legal = {tuple(tok(s).input_ids) for s in seqs}
        depth = len(tok(seqs[0]).input_ids)
        pi = 3
        hd = {}
        for s in seqs:
            ID = list(tok(s).input_ids) + [tok.eos_token_id]
            for i in range(pi, len(ID)):
                hn = gh(ID[:i]) if i == pi else gh(ID[pi:i])
                hd.setdefault(hn, set()).add(ID[i])
        hd = {k: sorted(v) for k, v in hd.items()}
        print(f"\n--- {tag}: depth={depth} pi={pi} keys={len(hd)} "
              f"(unique SID = {len(set(seqs))})")

        ds = EvalSidDataset(train_file=TEST, tokenizer=tok, max_len=2048, test=True,
                            category="industrial and scientific items")
        NB = 4
        enc = [ds[0]]
        L = max(len(e["input_ids"]) for e in enc)
        pad = [[tok.pad_token_id] * (L - len(e["input_ids"])) + e["input_ids"] for e in enc]
        ids = torch.tensor(pad).to(dev)
        am = torch.tensor([[0] * (L - len(e["input_ids"])) + [1] * len(e["input_ids"])
                           for e in enc]).to(dev)
        gc = GenerationConfig(num_beams=NB, length_penalty=0, num_return_sequences=NB,
                              pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id,
                              max_new_tokens=6, top_k=None, top_p=None)
        clp = Spy(prefix_allowed_tokens_fn=lambda b, i: hd.get(gh(i), []),
                  num_beams=NB, base_model=BASE, eos_token_id=tok.eos_token_id)
        with torch.no_grad():
            out = model.generate(ids, attention_mask=am, generation_config=gc,
                                 logits_processor=LogitsProcessorList([clp]))
        gen = out[:, ids.shape[1]:]
        okc = 0
        for row in gen:
            core = []
            for v in row.tolist():
                if v == tok.eos_token_id:
                    break
                core.append(v)
            if tuple(core) in legal:
                okc += 1
        print(f"    legal candidates: {okc}/{len(gen)}")

        # ---- corrected walk: the decoder's key at step k is the last k tokens ----
        s = tok(seqs[0]).input_ids
        print(f"    examine: can the trie constrain the FULL prefix?")
        for k in range(1, depth + 2):
            key = s[:k]
            hits = hd.get(gh(key), None)
            print(f"      key=hash(SID[:{k}]) -> "
                  f"{'HIT ' + str(sorted(hits)[:3]) if hits else 'MISS'}")


if __name__ == "__main__":
    main()
