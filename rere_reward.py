#!/usr/bin/env python3
"""
ReRe ranking reward -- pure, importable, side-effect-free.

Reference: the ranking reward of ReRe (Eq.7-9), as specified for this experiment.

    Eq.7  R_rule      = +1 for the ground-truth item, 0 for every other item
    Eq.8  R_hat_rank  =  0            for the ground-truth item
                        -1/log(r + 2) for a NON-ground-truth item at zero-based
                                      rank r
    Eq.9  R_rank      = -R_hat_rank_i / sum_j R_hat_rank_j     (over the group)
    final             R_i = R_rule_i + R_rank_i

Why `r` needs no extra computation
----------------------------------
The GRPO rollout runs with `num_beams=16, num_return_sequences=16, do_sample=True`,
which transformers 4.57.1 dispatches to `GenerationMode.BEAM_SAMPLE` (verified from
`GenerationConfig.get_generation_mode()`). `generate(..., return_dict_in_generate=
True, output_scores=True)` returns each prompt's G sequences **already sorted by
sequence score descending**, so the zero-based position inside the group *is* the
rank and `rho = r + 1`. No re-ranking, no score plumbing.

Why the denominator must be per-group
-------------------------------------
`sum_j R_hat_rank_j` is taken over the CURRENT group, whose GT entry contributes
0. Normalising a fixed 16-entry vector and then zeroing the GT slot would use a
different denominator whenever a GT is present -- that is the specific mistake the
group-wise form in Eq.9 avoids.

This module deliberately imports nothing. It is used by `rl.py` and is directly
unit-tested by `tools/tests/test_rere_reward.py` without a model, a tokenizer, a
GPU or trl.
"""
import math

LOG_BASE = 2.0          # matches the existing ndcg_rule_reward implementation


def raw_rank_rewards(is_gt, base=LOG_BASE):
    """Eq.8 for one group.

    is_gt : sequence of bool, True where the candidate equals the ground truth,
            ordered by descending sequence score (rank 0 first).

    Returns a list of floats: 0.0 at ground-truth positions, -1/log_base(r+2)
    elsewhere.
    """
    out = []
    for r, gt in enumerate(is_gt):
        out.append(0.0 if gt else -1.0 / math.log(r + 2.0, base))
    return out


def normalise_rank_rewards(raw):
    """Eq.9 for one group: divide by the group's own sum of raw rank rewards.

    A degenerate group whose raw rewards sum to 0 (only possible if every
    candidate is the ground truth, which cannot happen for distinct SIDs) yields
    zeros instead of NaN.
    """
    denom = float(sum(raw))
    if denom == 0.0:
        return [0.0] * len(raw)
    return [-x / denom for x in raw]


def group_rewards(is_gt, base=LOG_BASE):
    """Eq.7 + Eq.9 for one group.

    Returns (total, rule, rank):
        rule[i]  = 1.0 if is_gt[i] else 0.0
        rank[i]  = normalised Eq.8 term
        total[i] = rule[i] + rank[i]
    """
    raw = raw_rank_rewards(is_gt, base=base)
    rnk = normalise_rank_rewards(raw)
    rule = [1.0 if g else 0.0 for g in is_gt]
    total = [a + b for a, b in zip(rule, rnk)]
    return total, rule, rnk


def group_flatten(flags_per_group, base=LOG_BASE):
    """Apply `group_rewards` to a list of per-group `is_gt` flag lists.

    Returns (totals, rules, ranks) flattened in group-major order, matching the
    (B*G,) layout TRL expects from a reward function.
    """
    T, R, K = [], [], []
    for flags in flags_per_group:
        t, r, k = group_rewards(flags, base=base)
        T.extend(t)
        R.extend(r)
        K.extend(k)
    return T, R, K
