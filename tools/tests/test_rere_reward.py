#!/usr/bin/env python3
"""
Unit test for the ReRe ranking reward (Eq.7-9).

Pure-function test: no model, no tokenizer, no GPU, no trl, no dataset. Runs in
milliseconds.

Covers the invariants required before any training is allowed to start:

  1. GT present  -> R_rank(GT) == 0 and R_total(GT) == 1
  2. every wrong candidate -> R_rank < 0
  3. harder negative penalised more: reward is non-increasing in rank, and the
     rank-0 wrong candidate is strictly the worst
  4. per group: sum(R_rank) == -1
  5. GT miss: std(R_total) > 0 and not all rewards are zero
  6. GT hit: argmax(R_total) == GT position
  7. deterministic: repeated calls are bit-identical

Run:
    python tools/tests/test_rere_reward.py
"""
import os
import statistics
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
import rere_reward as rr  # noqa: E402

G = 16
_RESULTS = []


def check(ok, label, detail=""):
    _RESULTS.append((bool(ok), label))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  -- {detail}" if detail else ""))
    return bool(ok)


def flags_with_gt_at(pos, g=G):
    f = [False] * g
    if pos is not None:
        f[pos] = True
    return f


def audit_group(name, pos):
    print()
    print(f"--- {name}: GT at rank {pos} ---")
    flags = flags_with_gt_at(pos)
    total, rule, rank = rr.group_rewards(flags)

    print(f"    rank : {'  '.join(f'{i:>7d}' for i in range(G))}")
    print(f"    rule : {'  '.join(f'{x:>7.4f}' for x in rule)}")
    print(f"    rank : {'  '.join(f'{x:>7.4f}' for x in rank)}")
    print(f"    total: {'  '.join(f'{x:>7.4f}' for x in total)}")

    # invariant 4: group-normalised rank rewards sum to -1
    s = sum(rank)
    check(abs(s + 1.0) < 1e-12, f"{name}: sum(R_rank) == -1", f"{s:.15f}")

    # invariant 2: all wrong candidates have negative rank reward
    wrong_neg = all(rank[i] < 0 for i in range(G) if i != pos)
    check(wrong_neg, f"{name}: every wrong candidate has R_rank < 0",
          f"min={min(rank[i] for i in range(G) if i != pos):.4f} "
          f"max={max(rank[i] for i in range(G) if i != pos):.4f}")

    if pos is not None:
        # invariant 1
        check(rank[pos] == 0.0, f"{name}: R_rank(GT) == 0", f"{rank[pos]}")
        check(abs(total[pos] - 1.0) < 1e-12, f"{name}: R_total(GT) == 1",
              f"{total[pos]:.15f}")
        # invariant 6
        am = max(range(G), key=lambda i: total[i])
        check(am == pos, f"{name}: argmax(R_total) == GT position",
              f"argmax={am} gt={pos}")
    else:
        # invariant 5 (miss group)
        sd = statistics.pstdev(total)
        check(sd > 0, f"{name}: GT-miss reward_std > 0", f"std={sd:.6f}")
        check(any(x != 0.0 for x in total), f"{name}: not all-zero rewards")

    # invariant 3: monotone penalty for wrong candidates
    wrong_idx = [i for i in range(G) if i != pos]
    wrong_vals = [rank[i] for i in wrong_idx]
    mono = all(wrong_vals[k] <= wrong_vals[k + 1] + 1e-15
               for k in range(len(wrong_vals) - 1))
    check(mono, f"{name}: R_rank non-decreasing in rank (earlier = worse)",
          f"rank{wrong_idx[0]}={wrong_vals[0]:.4f} -> "
          f"rank{wrong_idx[-1]}={wrong_vals[-1]:.4f}")
    strictly = wrong_vals[0] < wrong_vals[-1]
    check(strictly, f"{name}: rank ordering is strict (not flat)",
          f"{wrong_vals[0]:.6f} < {wrong_vals[-1]:.6f}")

    return total, rule, rank


def audit_eq9_denominator():
    """The specific mistake Eq.9 forbids: normalising a fixed vector then
    zeroing the GT slot must give a DIFFERENT result from the group-wise form
    whenever the GT is present."""
    print()
    print("--- Eq.9 denominator must be group-wise, not a fixed vector ---")
    raw_all_wrong = rr.raw_rank_rewards([False] * G)
    naive_fixed = rr.normalise_rank_rewards(raw_all_wrong)     # denominator over 16

    flags = flags_with_gt_at(0)
    _, _, rank_groupwise = rr.group_rewards(flags)             # denominator over 15

    # the naive vector put at position r for r=1..15 would be naive_fixed[r]
    naive_applied = [0.0] + [naive_fixed[r] for r in range(1, G)]
    diff = max(abs(naive_applied[r] - rank_groupwise[r]) for r in range(G))
    check(diff > 1e-6,
          "group-wise normalisation differs from fixed-vector normalisation",
          f"max abs diff = {diff:.6f}")
    print(f"    group-wise denominator = sum over 15 negatives = "
          f"{sum(rr.raw_rank_rewards(flags)):.6f}")
    print(f"    fixed denominator      = sum over 16 negatives = "
          f"{sum(raw_all_wrong):.6f}")
    print(f"    e.g. rank1: group-wise={rank_groupwise[1]:.6f}  "
          f"fixed={naive_applied[1]:.6f}")


def audit_determinism():
    print()
    print("--- determinism ---")
    flags = flags_with_gt_at(7)
    a = rr.group_rewards(flags)
    b = rr.group_rewards(flags)
    check(a == b, "repeated calls are bit-identical")
    # also across a freshly built flag list
    c = rr.group_rewards([i == 7 for i in range(G)])
    check(a == c, "independent construction gives identical results")


def audit_flatten():
    print()
    print("--- group_flatten layout (B*G, group-major) ---")
    groups = [flags_with_gt_at(0), flags_with_gt_at(None), flags_with_gt_at(15)]
    T, R, K = rr.group_flatten(groups)
    check(len(T) == len(R) == len(K) == 3 * G, "flattened length == B*G",
          f"{len(T)}")
    ok = True
    for gi, flags in enumerate(groups):
        t, r, k = rr.group_rewards(flags)
        if T[gi * G:(gi + 1) * G] != t or R[gi * G:(gi + 1) * G] != r \
                or K[gi * G:(gi + 1) * G] != k:
            ok = False
    check(ok, "each group occupies a contiguous block of G entries")
    sums = [sum(K[gi * G:(gi + 1) * G]) for gi in range(len(groups))]
    check(all(abs(s + 1.0) < 1e-12 for s in sums),
          "every flattened group still sums R_rank to -1", str([round(s, 12) for s in sums]))


def audit_expected_values():
    """Pin the exact numeric values so a silent formula change is caught."""
    print()
    print("--- exact values (regression pin) ---")
    import math
    raw = rr.raw_rank_rewards([False] * G)
    exp0 = -1.0 / math.log2(2)          # rank 0  -> -1.000000
    exp1 = -1.0 / math.log2(3)          # rank 1  -> -0.630930
    exp15 = -1.0 / math.log2(17)        # rank 15 -> -0.244651
    check(abs(raw[0] - exp0) < 1e-12, "raw[0] == -1/log2(2) == -1",
          f"{raw[0]:.6f}")
    check(abs(raw[1] - exp1) < 1e-12, "raw[1] == -1/log2(3)",
          f"{raw[1]:.6f}")
    check(abs(raw[15] - exp15) < 1e-12, "raw[15] == -1/log2(17)",
          f"{raw[15]:.6f}")
    # For an all-wrong group the raw sum is the geometric sum over all 16 ranks
    # (== -6.105999), NOT -1: Eq.9 is what turns it into -1.
    raw_sum = sum(raw)
    check(abs(raw_sum + 6.105999) < 1e-5,
          "all-wrong RAW sum == -6.105999 (before Eq.9 normalisation)",
          f"{raw_sum:.6f}")
    norm = rr.normalise_rank_rewards(raw)
    check(abs(sum(norm) + 1.0) < 1e-12,
          "all-wrong NORMALISED sum == -1 (after Eq.9)",
          f"{sum(norm):.15f}")
    check(abs(norm[0] + 0.163773) < 1e-6,
          "normalised[0] == -0.163773 (matches the earlier audit value)",
          f"{norm[0]:.6f}")


def main():
    print("=" * 84)
    print("ReRe RANKING REWARD -- UNIT TEST (pure, no model, no trl)")
    print("=" * 84)
    print(f"  module = {rr.__file__}")
    print(f"  G      = {G}   log base = {rr.LOG_BASE}")

    audit_group("Case A", 0)
    audit_group("Case B", 7)
    audit_group("Case C", 15)
    audit_group("Case D", None)
    audit_eq9_denominator()
    audit_determinism()
    audit_flatten()
    audit_expected_values()

    n_pass = sum(1 for ok, _ in _RESULTS if ok)
    n_fail = len(_RESULTS) - n_pass
    print()
    print("=" * 84)
    print(f"TOTAL: {n_pass} PASS / {n_fail} FAIL  ({len(_RESULTS)} checks)")
    if n_fail:
        for ok, lbl in _RESULTS:
            if not ok:
                print("  FAILED: " + lbl)
    print(f"RESULT: {'ALL PASS' if not n_fail else 'FAIL'}")
    return 0 if not n_fail else 1


if __name__ == "__main__":
    sys.exit(main())
