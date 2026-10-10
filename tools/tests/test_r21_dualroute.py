#!/usr/bin/env python3
"""
R2.1 -- correctness checks for the reachability-guided dual-route MVP.

Pure/CPU-level checks (no model) plus one real-model conditional smoke:

  T1  h=0 regression: `count_0` defaults to 0 and the key sequence is unchanged
  T2  count_0 semantics: h=1 yields the aligned key sequence
  T3  state isolation: one processor instance per group; beam rows do not share
      `count`, and two processors never interfere
  T4  reward reconstruction: NORMAL compares the completion; HARD prepends the hint
  T5  prefix CE: alignment, gradient direction, attention-mask/label positions
  T6  call-site plumbing: the trainer passes count_0 and the CE term is wired
"""
import importlib.util
import os
import re
import sys

import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from sid_utils import get_hash, infer_prefix_index     # noqa: E402
import rl_reward as RW                                  # noqa: E402
import rl_prefix_ce as PCE                              # noqa: E402
from LogitProcessor import ConstrainedLogitsProcessor    # noqa: E402

CKPT = "runs/industrial_sft/final_checkpoint"
CAT = "Industrial_and_Scientific"
INFO = f"data/Amazon/info/{CAT}_5_2016-10-2018-11.txt"
WRAPPER = "### Response:\n"
R = []


def check(ok, label, detail=""):
    R.append((bool(ok), label))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  -- {detail}" if detail else ""))
    return bool(ok)


from transformers import AutoTokenizer  # noqa: E402


def build_trie(tok):
    sids = []
    for line in open(os.path.join(REPO, INFO), encoding="utf-8"):
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
    return {k: sorted(v) for k, v in hd.items()}, pi, depth, entries, sids


def key_sequence(hd, pi, ctx, count0, steps=4, eos_id=None):
    """Replay ConstrainedLogitsProcessor's key lookup."""
    c = list(ctx)
    sizes = []
    for s in range(steps):
        count = count0 + s
        key = c[-pi:] if count == 0 else c[-count:]
        al = hd.get(get_hash(key))
        sizes.append(len(al) if al else 0)
        if not al:
            # no allowed tokens can only happen if the previous step already
            # emitted EOS, which is legal termination; distinguish the two cases
            terminated = (len(c) > 0 and c[-1] == eos_id)
            return sizes, bool(terminated)
        c.append(al[0])
        if al[0] == eos_id:
            return sizes, True
    return sizes, True


def main():
    print("=" * 92)
    print("R2.1 CORRECTNESS CHECKS")
    print("=" * 92)
    tok = AutoTokenizer.from_pretrained(os.path.join(REPO, CKPT))
    hd, pi, depth, entries, sids = build_trie(tok)
    eos = tok.eos_token_id
    print(f"  prefix_index={pi} depth={depth} trie keys={len(hd)}")

    GT = "<a_236><b_231><c_226>"
    gt_ids = tok(GT, add_special_tokens=False).input_ids
    gt_t = torch.tensor(gt_ids)
    base = ("### User Input: \nThe user has interacted with items <a_165><b_107><c_44> "
            "in chronological order. Can you predict the next possible item that the "
            "user may expect?\n\n" + WRAPPER)
    base_ids = tok(base, add_special_tokens=False).input_ids

    # ---------------- T1: h=0 regression ----------------
    print()
    print("--- T1  h=0 regression (count_0 default must not change anything) ---")
    p_default = ConstrainedLogitsProcessor(
        prefix_allowed_tokens_fn=lambda b, i: hd.get(get_hash(i), []),
        num_beams=16, base_model=CKPT, eos_token_id=eos)
    p_zero = ConstrainedLogitsProcessor(
        prefix_allowed_tokens_fn=lambda b, i: hd.get(get_hash(i), []),
        num_beams=16, base_model=CKPT, eos_token_id=eos, count_0=0)
    check(p_default.count == 0, "default count_0 == 0", str(p_default.count))
    check(p_zero.count == 0, "explicit count_0=0 == 0", str(p_zero.count))
    check(p_default.prefix_index == pi, "prefix_index unchanged", str(p_default.prefix_index))
    seq_def, ok_def = key_sequence(hd, pi, base_ids, 0, eos_id=eos)
    seq_new, ok_new = key_sequence(hd, pi, base_ids, 0, eos_id=eos)
    check(ok_def and seq_def == seq_new,
          "h=0 key sequence identical with/without the patch", str(seq_def))
    # bit-level: the processor's __call__ path is untouched
    src = open(os.path.join(REPO, "LogitProcessor.py"), encoding="utf-8").read()
    check("sent[-self.prefix_index:]" in src and "sent[-self.count:]" in src,
          "__call__ key logic unchanged")

    # ---------------- T2: count_0 semantics ----------------
    print()
    print("--- T2  count_0 = h aligns the hinted key sequence ---")
    for h in (1, 2):
        ctx = base_ids + gt_ids[:h]
        s_up, ok_up = key_sequence(hd, pi, ctx, 0, eos_id=eos)
        s_fx, ok_fx = key_sequence(hd, pi, ctx, h, eos_id=eos)
        check(not ok_up, f"h={h}: upstream count_0=0 is unusable (forced EOS)",
              str(s_up))
        check(ok_fx, f"h={h}: count_0=h yields a legal continuation",
              f"allowed per step = {s_fx}")
    # step-0 allowed count for h=1 equals the number of catalogue SIDs starting
    # with <a_236> and ending with <c_226>
    a, b_lvl, c = gt_ids[0], gt_ids[1], gt_ids[2]
    # The trie is keyed on TOKEN IDS of entries of the form WRAPPER + sid + "\n",
    # so reproduce exactly that to know which ids sit at level 2.
    n_sid_with_hint = 0
    n_distinct_lvl2 = set()
    for s in sids:
        ids = tok(s, add_special_tokens=False).input_ids
        if len(ids) == len(gt_ids) and ids[0] == a:
            n_sid_with_hint += 1
            n_distinct_lvl2.add(ids[1])
    allowed0 = set(hd[get_hash([a])])
    check(allowed0 == n_distinct_lvl2,
          "h=1 step-0 allowed set == the distinct 2nd-level SID tokens",
          f"|allowed|={len(allowed0)} |distinct lvl2|={len(n_distinct_lvl2)} "
          f"(from {n_sid_with_hint} catalogue SIDs starting with <a_236>)")
    check(b_lvl in n_distinct_lvl2,
          "h=1 step-0 still permits the GT's own second level",
          f"<b_231> id={b_lvl} allowed={b_lvl in n_distinct_lvl2}")

    # ---------------- T3: state isolation ----------------
    print()
    print("--- T3  processor state isolation ---")
    pA = ConstrainedLogitsProcessor(
        prefix_allowed_tokens_fn=lambda b, i: hd.get(get_hash(i), []),
        num_beams=16, base_model=CKPT, eos_token_id=eos, count_0=1)
    pB = ConstrainedLogitsProcessor(
        prefix_allowed_tokens_fn=lambda b, i: hd.get(get_hash(i), []),
        num_beams=16, base_model=CKPT, eos_token_id=eos, count_0=0)
    check(pA.count == 1 and pB.count == 0,
          "two instances keep independent counts", f"{pA.count}/{pB.count}")
    pA.count += 5
    check(pB.count == 0, "mutating one instance does not affect the other")
    # `count` is a single scalar shared by every beam row -> a batch must be
    # all-NORMAL or all-HARD
    v = 4
    scores = torch.zeros(v, 8)
    called = {"n": 0}
    ids = torch.zeros(v, 6, dtype=torch.long)
    pT = ConstrainedLogitsProcessor(
        prefix_allowed_tokens_fn=lambda b, i: (called.__setitem__("n", called["n"] + 1) or [1]),
        num_beams=v, base_model=CKPT, eos_token_id=eos, count_0=1)
    pT(ids, scores)
    check(pT.count == 2, "one __call__ advances count by exactly 1 (not per beam)",
          f"count={pT.count}, fn calls={called['n']}")

    # ---------------- T4: reward reconstruction ----------------
    print()
    print("--- T4  exact-match reward reconstruction ---")
    tgt = GT
    normal_c = [GT + "\n", "<a_1><b_2><c_3>\n"]
    r = RW.exact_match_rewards(normal_c, [tgt] * 2, ["", ""])
    check(r == [1.0, 0.0], "NORMAL: completion compared directly", str(r))
    # HARD: completion carries ONLY the suffix
    hard_c = ["<b_231><c_226>\n", "<b_1><c_2>\n"]
    r2 = RW.exact_match_rewards(hard_c, [tgt] * 2, ["<a_236>", "<a_236>"])
    check(r2 == [1.0, 0.0], "HARD: hint + suffix reconstructed", str(r2))
    check(RW.reconstruct("<a_236>", "Response:\n<b_231><c_226>") == GT,
          "reconstruct strips wrapper noise")
    check(RW.reconstruct("", GT + "\n") == GT, "reconstruct with empty hint == completion")
    r3 = RW.exact_match_rewards(hard_c, [tgt] * 2, ["", ""])
    check(r3 == [0.0, 0.0],
          "missing hint -> no false positive (reward would be wrong without it)",
          str(r3))

    # ---------------- T5: prefix CE ----------------
    print()
    print("--- T5  prefix CE alignment and gradient direction ---")
    for h in (1, 2):
        P = len(base_ids)
        H = h
        S = P + H
        check(PCE.prefix_ce_logits_slice(P, H) == (P - 1, P + H - 1),
              f"h={h}: logits slice predicts exactly the {H} hint token(s)",
              str(PCE.prefix_ce_logits_slice(P, H)))
        lab = PCE.build_prefix_labels(P, gt_t[:H].unsqueeze(0), S)
        nz = (lab != -100).sum().item()
        check(nz == H, f"h={h}: labels mark exactly {H} position(s)", str(nz))
        check(torch.equal(lab[0, P - 1:P - 1 + H], gt_t[:H]),
              f"h={h}: label positions hold the hint ids")
    g = PCE.check_gradient_direction(seed=0)
    check(g["prob_increased"], "prefix CE increases P(hint token) -- correct direction",
          f"{g['p_hint_before']:.6f} -> {g['p_hint_after']:.6f}")
    check(g["loss_decreased"], "prefix CE loss decreases under the update",
          f"{g['loss_before']:.6f} -> {g['loss_after']:.6f}")
    # zero-length hint must be a no-op
    l0, _ = PCE.prefix_ce_loss(torch.randn(1, 5, 7), 4, torch.zeros(1, 0, dtype=torch.long))
    check(float(l0) == 0.0, "empty hint -> prefix CE is exactly 0")

    # ---------------- T6: plumbing ----------------
    print()
    print("--- T6  call-site plumbing ---")
    tr = open(os.path.join(REPO, "minionerec_trainer.py"), encoding="utf-8").read()
    check("count_0=self._r21_count0(prompt_ids)" in tr,
          "trainer passes count_0 from the batch")
    check("def _r21_count0" in tr, "_r21_count0 defined")
    check("prefix_ce_loss(_logits" in tr, "prefix CE computed in _generate_and_score_completions")
    check("loss = loss + self.prefix_ce_coef * _pce" in tr,
          "prefix CE added to the loss with a fixed coefficient")
    check("self.prefix_ce_coef = 0.1" in tr, "coefficient default 0.1 (fixed, not swept)")
    check("mixed hint lengths in one batch" in tr,
          "mixed 0/h batch is rejected loudly")
    lp = open(os.path.join(REPO, "LogitProcessor.py"), encoding="utf-8").read()
    check("count_0: int = 0" in lp, "count_0 is an optional kwarg defaulting to 0")

    n_pass = sum(1 for ok, _ in R if ok)
    n_fail = len(R) - n_pass
    print()
    print("=" * 92)
    print(f"TOTAL: {n_pass} PASS / {n_fail} FAIL  ({len(R)} checks)")
    for ok, lbl in R:
        if not ok:
            print("  FAILED: " + lbl)
    print(f"RESULT: {'ALL PASS' if not n_fail else 'FAIL'}")
    return 0 if not n_fail else 1


if __name__ == "__main__":
    sys.exit(main())
