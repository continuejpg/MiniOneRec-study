#!/usr/bin/env python3
"""Print the EXACT hash keys so we stop guessing."""
import hashlib
import json
import os
import sys

from transformers import AutoTokenizer

sys.path.insert(0, os.getcwd())
from sft import TokenExtender
from data import build_recommendation_prompt

MODEL = "/root/autodl-tmp/models/Qwen2.5-0.5B"
P0 = "data/Amazon/index/Industrial_and_Scientific.index.json"
SCRATCH = "/tmp/pi_scratch"


def gh(t):
    return hashlib.md5(str(list(t)).encode()).hexdigest()


tok = AutoTokenizer.from_pretrained(MODEL)
os.makedirs(SCRATCH, exist_ok=True)
link = os.path.join(SCRATCH, "Industrial_and_Scientific.index.json")
if os.path.islink(link) or os.path.exists(link):
    os.remove(link)
os.symlink(os.path.abspath(P0), link)
tok.add_tokens(sorted(TokenExtender(data_path=SCRATCH,
                                    dataset="Industrial_and_Scientific").get_new_tokens()))

idx = json.load(open(P0, encoding="utf-8"))
seqs = ["".join(idx[str(i)]) for i in range(len(idx))]
sid0 = tok(seqs[0]).input_ids
print(f"  SID0 = {seqs[0]!r}")
print(f"  SID0 ids = {sid0}")
print(f"  EOS id   = {tok.eos_token_id}")
print()

# ---- trie keys exactly as evaluate.py builds them (pi=3) ----
pi = 3
ID = list(sid0) + [tok.eos_token_id]
print(f"  evaluate.py build, pi={pi}: ID = {ID}")
for i in range(pi, len(ID)):
    hn = gh(ID[:i]) if i == pi else gh(ID[pi:i])
    print(f"    i={i}: key({('ID[:%d]' % i) if i==pi else ('ID[%d:%d]'%(pi,i))}) = {hn[:12]} -> allowed {ID[i]}")
root_key = gh(ID[:pi])
print(f"  ROOT key = gh(ID[:{pi}]) = {root_key[:12]}")
print()

# ---- what the decoder sees at step 0 ----
hist = seqs[:3]
prompt = build_recommendation_prompt(", ".join(hist))
pid = tok(prompt).input_ids
print(f"  prompt tail (last 8 ids) = {pid[-8:]}")
print(f"  prompt tail decoded      = {tok.convert_ids_to_tokens(pid[-8:])}")
print(f"  step0 key = gh(pid[-{pi}:]) = {gh(pid[-pi:])[:12]}")
print(f"  MATCH ROOT KEY ? {gh(pid[-pi:]) == root_key}")
print()
print(f"  sid0 first {pi} ids      = {sid0[:pi]}")
print(f"  gh(sid0[:{pi}])          = {gh(sid0[:pi])[:12]}")
print(f"  prompt ends with sid0?   {pid[-pi:] == sid0[:pi]}")
print()
# is the last history SID actually seqs[2]?
last = seqs[2]
print(f"  last history SID = {last!r}  ids={tok(last).input_ids}")
print(f"  prompt tail == last history SID ids ? {pid[-pi:] == tok(last).input_ids}")
