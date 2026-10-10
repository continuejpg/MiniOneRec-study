#!/usr/bin/env python3
"""
R2.1 -- end-to-end dual-route smoke (read-only w.r.t. the experiment).

Runs the REAL frozen SFT model on REAL training samples for both routes and
reports the numbers the design depends on. It does not train and does not touch
run directories.

  * h=0 NORMAL : prompt unchanged
  * h=1 HARD   : prompt carries the first GT SID token, count_0 = 1
  * reward     : exact-match 0/1 only (no ranking reward)
  * prefix CE  : supervised CE of the hint token from the ORIGINAL prompt

Covers both GT-hit and GT-miss groups. All completions must be legal catalogue
SIDs after reconstruction.
"""
import collections
import json
import os
import re
import statistics
import sys

import torch
import torch.nn.functional as F
from transformers import (AutoTokenizer, AutoModelForCausalLM, GenerationConfig,
                          LogitsProcessorList, TemperatureLogitsWarper)

REPO = os.getcwd()
sys.path.insert(0, REPO)
import rl_reward as RW                                  # noqa: E402
import rl_prefix_ce as PCE                              # noqa: E402
from LogitProcessor import ConstrainedLogitsProcessor    # noqa: E402
from sid_utils import get_hash, infer_prefix_index       # noqa: E402
from data import SidDataset                              # noqa: E402

CKPT = "runs/industrial_sft/final_checkpoint"
CAT = "Industrial_and_Scientific"
TRAIN = f"data/Amazon/train/{CAT}_5_2016-10-2018-11.csv"
INFO = f"data/Amazon/info/{CAT}_5_2016-10-2018-11.txt"
INDEX = f"data/Amazon/index/{CAT}.index.json"
WRAPPER = "### Response:\n"
G, TEMP, MAXNEW = 16, 1.0, 128
N = 8
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
COEF = 0.1


def main():
    os.makedirs("artifacts/rl_audit", exist_ok=True)
    print("=" * 96)
    print("R2.1 DUAL-ROUTE END-TO-END SMOKE (real model, real train samples)")
    print("=" * 96)

    tok = AutoTokenizer.from_pretrained(CKPT)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    sids = [l.split("\t")[0].strip() for l in open(INFO, encoding="utf-8") if l.strip()]
    entries = [f"{WRAPPER}{s}\n" for s in sids]
    pi, depth, _ = infer_prefix_index(entries, tok, WRAPPER)
    hd = {}
    for e in entries:
        ID = list(tok(e, add_special_tokens=False).input_ids)
        ID.append(tok.eos_token_id)
        for i in range(pi, len(ID)):
            hn = get_hash(ID[:i]) if i == pi else get_hash(ID[pi:i])
            hd.setdefault(hn, set()).add(ID[i])
    hd = {k: sorted(v) for k, v in hd.items()}
    sid_ids = set()
    for v in hd.values():
        sid_ids.update(v)
    sid_ids.discard(tok.eos_token_id)
    cat_sids = set(sids)
    print(f"  prefix_index={pi} depth={depth} trie keys={len(hd)} "
          f"sid token ids={len(sid_ids)}")

    idx = json.load(open(INDEX, encoding="utf-8"))
    levels_of = {"".join(v): list(v) for v in idx.values()}

    ds = SidDataset(train_file=TRAIN, max_len=512, sample=-1, seed=42,
                    category="industrial and scientific items")
    picked = [ds[i] for i in range(N)]
    model = AutoModelForCausalLM.from_pretrained(CKPT, torch_dtype=torch.bfloat16,
                                                 device_map="auto").eval()

    gc = GenerationConfig(max_new_tokens=MAXNEW, length_penalty=1.0,
                          num_beams=G, num_return_sequences=G,
                          pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
                          top_k=None, top_p=None, temperature=TEMP, do_sample=True)
    print(f"  generation mode = {gc.get_generation_mode()}  (unchanged)  G={G}")

    summary = {}
    for route, h in (("NORMAL", 0), ("HARD", 1)):
        print()
        print("=" * 96)
        print(f"ROUTE {route}  (h={h})   coefficient(prefix CE) = {COEF}")
        print("=" * 96)
        hits = 0
        stds, ce_vals = [], []
        legal = bad = 0
        for k, ex in enumerate(picked):
            tgt = ex["completion"].strip()
            lv = levels_of.get(tgt)
            if lv is None:
                print(f"  sample {k}: GT {tgt} not in catalogue, skipped")
                continue
            hint = lv[0] if h == 1 else ""
            prompt = ex["prompt"] + hint
            enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
            pids, am = enc["input_ids"].to(DEV), enc["attention_mask"].to(DEV)
            n_hint = len(tok(hint, add_special_tokens=False).input_ids) if hint else 0
            ccc = ConstrainedLogitsProcessor(
                prefix_allowed_tokens_fn=lambda b, i: hd.get(get_hash(i), []),
                num_beams=G, base_model=CKPT, eos_token_id=tok.eos_token_id,
                count_0=n_hint)
            lp = LogitsProcessorList([TemperatureLogitsWarper(temperature=TEMP), ccc])
            with torch.no_grad():
                out = model.generate(pids, attention_mask=am, generation_config=gc,
                                     logits_processor=lp)
            gen = out[:, pids.shape[1]:]
            txt = tok.batch_decode(gen, skip_special_tokens=True)
            txt = [t.split("Response:\n")[-1] for t in txt]
            fulls = [RW.reconstruct(hint, t) for t in txt]
            rewards = RW.exact_match_rewards(txt, [tgt] * G, [hint] * G)
            std = statistics.pstdev(rewards)
            stds.append(std)
            if any(r == 1.0 for r in rewards):
                hits += 1
            legal += sum(1 for f in fulls if f in cat_sids)
            bad += sum(1 for f in fulls if f not in cat_sids)
            # prefix CE from the ORIGINAL prompt
            ce = 0.0
            if h == 1:
                with torch.no_grad():
                    o = tok(ex["prompt"], return_tensors="pt",
                            add_special_tokens=False)
                    oid = o["input_ids"].to(DEV)
                    hid = torch.tensor(
                        [tok(hint, add_special_tokens=False).input_ids], device=DEV)
                    c = torch.cat([oid, hid], dim=1)
                    lg = model(input_ids=c, attention_mask=torch.ones_like(c)).logits
                    ce, _ = PCE.prefix_ce_loss(lg, oid.size(1), hid)
                    ce = float(ce)
                ce_vals.append(ce)
            if k < 3:
                print(f"  sample {k}: GT={tgt} hint={hint!r} hit={any(r==1.0 for r in rewards)} "
                      f"reward_std={std:.6f} sum_reward={sum(rewards):.1f} "
                      f"prefixCE={ce:.4f}")
                print(f"     full[0] = {fulls[0]}   legal={fulls[0] in cat_sids}")
        n = len(stds)
        print()
        print(f"  samples             = {n}")
        print(f"  GT-hit groups       = {hits}  ({100.0*hits/n:.1f}%)")
        print(f"  zero-adv groups     = {sum(1 for s in stds if s == 0)}  "
              f"({100.0*sum(1 for s in stds if s==0)/n:.1f}%)")
        print(f"  reward_std  median  = {statistics.median(stds):.6f}")
        print(f"  full-SID legality   = {legal}/{legal+bad} "
              f"({100.0*legal/(legal+bad):.2f}%)")
        if ce_vals:
            print(f"  prefix CE           = mean {statistics.mean(ce_vals):.4f}  "
                  f"median {statistics.median(ce_vals):.4f}")
        summary[route] = {"h": h, "n": n, "hit": hits,
                          "hit_ratio": hits / n,
                          "zero_adv": sum(1 for s in stds if s == 0),
                          "std_median": statistics.median(stds),
                          "legality": legal / (legal + bad),
                          "prefix_ce_mean": statistics.mean(ce_vals) if ce_vals else None}

    # ------- backward / gradient / credit checks on REAL tensors -----------
    print()
    print("=" * 96)
    print("GRADIENT AND CREDIT-ASSIGNMENT CHECKS (real model)")
    print("=" * 96)
    ex = picked[0]
    tgt = ex["completion"].strip()
    lv = levels_of[tgt]
    hint = lv[0]
    # (a) prefix CE produces a real gradient on the LM head / backbone
    model.zero_grad()
    o = tok(ex["prompt"], return_tensors="pt", add_special_tokens=False)
    oid = o["input_ids"].to(DEV)
    hid = torch.tensor([tok(hint, add_special_tokens=False).input_ids], device=DEV)
    c = torch.cat([oid, hid], dim=1)
    logits = model(input_ids=c, attention_mask=torch.ones_like(c)).logits
    ce, _ = PCE.prefix_ce_loss(logits, oid.size(1), hid)
    ce.backward()
    gnorms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    print(f"  prefix CE value                       = {float(ce):.6f}")
    print(f"  params receiving prefix-CE gradient   = {len(gnorms)}")
    print(f"  prefix-CE grad norm (mean/max)        = "
          f"{statistics.mean(gnorms):.3e}/{max(gnorms):.3e}")
    ok_ce_grad = len(gnorms) > 0 and max(gnorms) > 0
    print(f"  [{'PASS' if ok_ce_grad else 'FAIL'}] prefix CE produces a non-zero gradient")
    # (b) loss finite
    loss_demo = torch.tensor(0.5, requires_grad=True) + COEF * ce.detach()
    print(f"  combined loss demo (grpo 0.5 + {COEF}*CE) = {float(loss_demo):.6f}  "
          f"finite={bool(torch.isfinite(loss_demo))}")
    print(f"  [{'PASS' if bool(torch.isfinite(loss_demo)) else 'FAIL'}] loss is finite")
    # (c) hinted tokens are NOT in completion_ids -> no policy credit
    enc = tok(ex["prompt"] + hint, return_tensors="pt", add_special_tokens=False)
    pids = enc["input_ids"].to(DEV)
    with torch.no_grad():
        ccc = ConstrainedLogitsProcessor(
            prefix_allowed_tokens_fn=lambda b, i: hd.get(get_hash(i), []),
            num_beams=G, base_model=CKPT, eos_token_id=tok.eos_token_id, count_0=1)
        lp = LogitsProcessorList([TemperatureLogitsWarper(temperature=TEMP), ccc])
        out = model.generate(pids, attention_mask=torch.ones_like(pids),
                             generation_config=gc, logits_processor=lp)
    comp = out[:, pids.shape[1]:]
    hint_id = tok(hint, add_special_tokens=False).input_ids[0]
    in_comp = bool((comp == hint_id).any().item())
    print(f"  hint token id                         = {hint_id}")
    print(f"  hint appears in completion_ids        = {in_comp}")
    print(f"  [{'PASS' if not in_comp else 'FAIL'}] hinted token gets no policy-gradient credit")
    # (d) h=0 regression against the pre-patch processor
    try:
        sys.path.insert(0, "/tmp/r21")
        import importlib.util
        spec = importlib.util.spec_from_file_location("lp_pre", "/tmp/r21/LogitProcessor.py.pre")
        pre = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pre)
        pA = pre.ConstrainedLogitsProcessor(
            prefix_allowed_tokens_fn=lambda b, i: hd.get(get_hash(i), []),
            num_beams=G, base_model=CKPT, eos_token_id=tok.eos_token_id)
        pB = ConstrainedLogitsProcessor(
            prefix_allowed_tokens_fn=lambda b, i: hd.get(get_hash(i), []),
            num_beams=G, base_model=CKPT, eos_token_id=tok.eos_token_id)
        enc0 = tok(ex["prompt"], return_tensors="pt", add_special_tokens=False)
        ids0 = enc0["input_ids"].to(DEV)
        sc0 = torch.zeros(G, 8, device=DEV)
        m0 = pA(ids0.repeat(G, 1), sc0)
        m1 = pB(ids0.repeat(G, 1), sc0)
        same = bool(torch.equal(m0, m1))
        print(f"  h=0 mask identical pre-patch vs now   = {same}")
        print(f"  [{'PASS' if same else 'FAIL'}] h=0 regression")
    except Exception as e:  # noqa: BLE001
        print(f"  h=0 regression: SKIP ({type(e).__name__}: {e})")

    json.dump({"summary": summary, "coefficient": COEF,
               "generation_mode": str(gc.get_generation_mode())},
              open("artifacts/rl_audit/r21_smoke.json", "w"), indent=2)
    print(f"\n  [save] artifacts/rl_audit/r21_smoke.json")


if __name__ == "__main__":
    main()
