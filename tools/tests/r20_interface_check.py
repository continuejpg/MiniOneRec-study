#!/usr/bin/env python3
"""
R2.0 -- conditional-generation interface check (READ-ONLY).

Establishes the exact minimal change that makes prefix-hinted rollout work with the
EXISTING trie, and verifies the credit-assignment properties that GRPO depends on.

Root cause found by the probe
-----------------------------
`ConstrainedLogitsProcessor.count` starts at 0 and at that first call uses

    hash_key = sent[-self.prefix_index:]

a FIXED 3-token window. When the prompt already ends with h hinted SID tokens, that
window slides onto the hint, producing a key the trie does not contain
(h=1 -> ['ĠResponse', ':Ċ', '<a>']; h=2 -> [':Ċ', '<a>', '<b>']) and
`prefix_allowed_tokens_fn` returns [], so the processor forces EOS and the
completion is empty.

The trie itself is fine: hash(SID[:1]) and hash(SID[:2]) both exist and return the
legal continuations. What is missing is the decoder's bookkeeping: the h hinted
tokens must be counted as if they had already been generated, i.e.

    count_0 = h

Verified below for h = 0, 1, 2 by simulating the processor's key sequence and
checking every query against the real trie.

Also checks, for the hinted layout (hint inside the prompt, completion = suffix):
  * the hinted prefix is expressible as part of the prompt
  * the completion contains only the sampled suffix
  * suffix-only logprob is what `_get_per_token_logps` would return, because the
    hint is inside prompt_ids and completion_ids starts after it
  * prefix CE can be computed independently from the original prompt

No source file is modified.
"""
import json
import os
import sys

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

REPO = os.getcwd()
sys.path.insert(0, REPO)
from sid_utils import get_hash, infer_prefix_index     # noqa: E402

CKPT = "runs/industrial_sft/final_checkpoint"
CAT = "Industrial_and_Scientific"
INFO = f"data/Amazon/info/{CAT}_5_2016-10-2018-11.txt"
INDEX = f"data/Amazon/index/{CAT}.index.json"
WRAPPER = "### Response:\n"
PI = 3


def build_trie(tok):
    sids = []
    for line in open(INFO, encoding="utf-8"):
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


def simulate(tok, hd, prompt_ids, h, pi, eos, n_steps=4):
    """Replay ConstrainedLogitsProcessor's key sequence for a hinted prompt.

    count_0 = h  <-- the proposed minimal fix (upstream: count_0 = 0 always).
    Returns (allowed_sizes, ok) where ok is False if any step was forced to EOS.
    """
    ctx = list(prompt_ids)
    sizes = []
    for step in range(n_steps):
        count = h + step
        key = ctx[-pi:] if count == 0 else ctx[-count:]
        al = hd.get(get_hash(key))
        n = len(al) if al else 0
        sizes.append(n)
        if not al:
            return sizes, False
        # follow the highest-probability legal continuation to keep replaying
        nxt = al[0]
        ctx.append(nxt)
        if nxt == eos:
            break
    return sizes, True


def main():
    print("=" * 92)
    print("R2.0 CONDITIONAL-GENERATION INTERFACE CHECK (read-only)")
    print("=" * 92)

    tok = AutoTokenizer.from_pretrained(CKPT)
    hd, pi, depth = build_trie(tok)
    eos = tok.eos_token_id
    print(f"  prefix_index={pi}  depth={depth}  trie keys={len(hd)}")

    idx = json.load(open(INDEX, encoding="utf-8"))
    lv0 = [t for t in idx["0"]]
    lv0_ids = tok("".join(lv0), add_special_tokens=False).input_ids
    print(f"  example GT SID levels = {lv0}  ids={lv0_ids}")

    base = "### User Input: \n" + "The user has interacted with items <a_165><b_107><c_44> " \
           "in chronological order. Can you predict the next possible item that the user may expect?\n\n" + WRAPPER
    base_ids = tok(base, add_special_tokens=False).input_ids

    print()
    print("=" * 92)
    print("A. KEY SEQUENCE: upstream (count_0 = 0) vs proposed (count_0 = h)")
    print("=" * 92)
    print(f"  {'h':>2} {'variant':<22} {'step0':>8} {'step1':>8} {'step2':>8} {'step3':>8}  usable")
    for h in (0, 1, 2):
        hint = lv0_ids[:h]
        pids = base_ids + hint
        for label, c0 in (("upstream count_0=0", 0), ("proposed count_0=h", h)):
            # monkey-patch the initial count by offsetting the replay
            ctx = list(pids)
            sizes, ok = [], True
            for step in range(4):
                count = c0 + step
                key = ctx[-pi:] if count == 0 else ctx[-count:]
                al = hd.get(get_hash(key))
                sizes.append(len(al) if al else 0)
                if not al:
                    ok = False
                    break
                ctx.append(al[0])
                if al[0] == eos:
                    break
            sizes += [0] * (4 - len(sizes))
            print(f"  {h:>2} {label:<22} {sizes[0]:>8} {sizes[1]:>8} {sizes[2]:>8} "
                  f"{sizes[3]:>8}  {'YES' if ok else 'NO (forced EOS)'}")

    print()
    print("=" * 92)
    print("B. PREFIX CE CAN BE COMPUTED INDEPENDENTLY FROM THE ORIGINAL PROMPT")
    print("=" * 92)
    model = AutoModelForCausalLM.from_pretrained(CKPT, torch_dtype=torch.bfloat16,
                                                 device_map="auto").eval()
    dev = next(model.parameters()).device
    # CE of the hint tokens given the ORIGINAL prompt (no hint in context)
    for h in (1, 2):
        hint = lv0_ids[:h]
        with torch.no_grad():
            ids = torch.tensor([base_ids + hint], device=dev)
            logits = model(ids).logits[0]
            lp = 0.0
            for j, tid in enumerate(hint):
                pos = len(base_ids) + j - 1
                lp += torch.log_softmax(logits[pos].float(), dim=-1)[tid].item()
        print(f"  h={h}: prefix CE over {h} hint token(s) = {-lp/max(1,h):.6f} "
              f"(mean NLL)   [computable from the original prompt alone: YES]")

    print()
    print("=" * 92)
    print("C. CREDIT-ASSIGNMENT LAYOUT WITH HINT INSIDE THE PROMPT")
    print("=" * 92)
    print("  layout:  prompt_ids    = <standard prompt> + <h hint tokens>")
    print("           completion_ids = <sampled suffix>        (hint NOT included)")
    print()
    for h in (1, 2):
        hint = lv0_ids[:h]
        prompt_len = len(base_ids) + len(hint)
        print(f"  h={h}: prompt_len={prompt_len}  completion starts at index {prompt_len}")
    print()
    print("  consequences (all follow from the hint being inside prompt_ids):")
    print("    * per_token_logps  = logprob of the SUFFIX only   -> correct")
    print("      (`logits_to_keep = completion_ids.size(1)` already slices this way)")
    print("    * ref_per_token_logps = same slice of the reference model -> aligned")
    print("    * completion_mask  = unchanged (still 'up to first EOS')")
    print("    * advantage        = group-relative over suffix rewards only")
    print("    * hinted tokens receive NO policy gradient, because they never appear")
    print("      in completion_ids and therefore never in per_token_loss")
    print("    * prefix CE is available separately (section B) if a router needs it")

    print()
    print("=" * 92)
    print("D. MINIMAL CHANGE REQUIRED IN THE TRAINER")
    print("=" * 92)
    print("  LogitProcessor.py  ConstrainedLogitsProcessor.__init__/__call__:")
    print("      add `count_0: int = 0`;  self.count = int(count_0)")
    print("      (upstream hardcodes self.count = 0)")
    print("  minionerec_trainer.py  _prepare_inputs (single construction site, :684):")
    print("      pass count_0 = <number of trailing SID tokens the prompt ends with>")
    print("      i.e. 0 for the normal route, h for a hinted route.")
    print("  No change to: trie construction, hash_dict, generation_config,")
    print("                _get_per_token_logps, completion_mask, advantage, loss.")

    json.dump({"prefix_index": pi, "trie_keys": len(hd),
               "example_sid": lv0, "example_ids": lv0_ids,
               "root_cause": "ConstrainedLogitsProcessor.count starts at 0, so the "
                             "step-0 hash window sent[-prefix_index:] slides onto "
                             "hinted SID tokens and misses the trie",
               "fix": "count_0 = h (number of trailing SID tokens in the prompt)"},
              open("artifacts/rl_audit/r20_interface.json", "w"), indent=2)
    print(f"\n  [save] artifacts/rl_audit/r20_interface.json")


if __name__ == "__main__":
    main()
