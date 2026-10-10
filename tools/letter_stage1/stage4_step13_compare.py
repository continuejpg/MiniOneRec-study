#!/usr/bin/env python3
"""Stage 4 / step 13 -- assemble the P0 / B / T comparison table."""
import json

P0 = {"hr": [0.06926980, 0.10103684, 0.12111185, 0.15376131, 0.19832341],
      "ndcg": [0.06926980, 0.08789727, 0.09613706, 0.10663781, 0.11786798]}
B = {"hr": [0.07081403, 0.09508052, 0.11228767, 0.13765718, 0.16765939],
     "ndcg": [0.07081403, 0.08494025, 0.09202226, 0.10023594, 0.10784649]}
T = {"hr": [0.07434370, 0.09905140, 0.11780278, 0.14537834, 0.17383631],
     "ndcg": [0.07434370, 0.08840169, 0.09609076, 0.10495113, 0.11208197]}
KS = [1, 3, 5, 10, 20]


def pp(x, y):
    return (x - y) * 100.0


def rel(x, y):
    return (x - y) / y * 100.0 if y else float("nan")


print("=" * 92)
print("STAGE 4 / STEP 13 -- FINAL COMPARISON")
print("=" * 92)
print()
print("  raw metrics")
print(f"  {'k':>4} | {'P0 HR':>10} {'B HR':>10} {'T HR':>10} | "
      f"{'P0 NDCG':>10} {'B NDCG':>10} {'T NDCG':>10}")
print("  " + "-" * 82)
for i, k in enumerate(KS):
    print(f"  {k:>4} | {P0['hr'][i]:>10.8f} {B['hr'][i]:>10.8f} {T['hr'][i]:>10.8f} | "
          f"{P0['ndcg'][i]:>10.8f} {B['ndcg'][i]:>10.8f} {T['ndcg'][i]:>10.8f}")

print()
print("  B vs P0   (absolute pp / relative %)")
print(f"  {'metric':>10} | {'abs pp':>12} {'rel %':>12}")
print("  " + "-" * 42)
for i, k in enumerate(KS):
    print(f"  {'HR@'+str(k):>10} | {pp(B['hr'][i], P0['hr'][i]):>+12.4f} "
          f"{rel(B['hr'][i], P0['hr'][i]):>+12.4f}")
for i, k in enumerate(KS):
    print(f"  {'NDCG@'+str(k):>10} | {pp(B['ndcg'][i], P0['ndcg'][i]):>+12.4f} "
          f"{rel(B['ndcg'][i], P0['ndcg'][i]):>+12.4f}")

print()
print("  T vs B    (absolute pp / relative %)")
print(f"  {'metric':>10} | {'abs pp':>12} {'rel %':>12}")
print("  " + "-" * 42)
for i, k in enumerate(KS):
    print(f"  {'HR@'+str(k):>10} | {pp(T['hr'][i], B['hr'][i]):>+12.4f} "
          f"{rel(T['hr'][i], B['hr'][i]):>+12.4f}")
for i, k in enumerate(KS):
    print(f"  {'NDCG@'+str(k):>10} | {pp(T['ndcg'][i], B['ndcg'][i]):>+12.4f} "
          f"{rel(T['ndcg'][i], B['ndcg'][i]):>+12.4f}")

print()
print("=" * 92)
print("CASE CLASSIFICATION (per the Stage-4 rule; single seed, no significance claim)")
print("=" * 92)
p0h, bh, th = P0["hr"][-1], B["hr"][-1], T["hr"][-1]
p0n, bn, tn = P0["ndcg"][-1], B["ndcg"][-1], T["ndcg"][-1]
print(f"  P0  HR@20={p0h:.8f}  NDCG@20={p0n:.8f}")
print(f"  B   HR@20={bh:.8f}  NDCG@20={bn:.8f}")
print(f"  T   HR@20={th:.8f}  NDCG@20={tn:.8f}")
print()
print(f"  B > P0  ? HR {bh > p0h}   NDCG {bn > p0n}")
print(f"  B > T   ? HR {bh > th}   NDCG {bn > tn}")
print(f"  T > B   ? HR {th > bh}   NDCG {tn > bn}")
print(f"  T > P0  ? HR {th > p0h}   NDCG {tn > p0n}")
print()

lbl = {True: "YES", False: "NO"}
print(f"  CONTENT_ONLY_BEATS_P0      = {lbl[bh > p0h and bn > p0n]}")
print(f"  LETTER_BEATS_CONTENT_ONLY  = {lbl[th > bh and tn > bn]}")

json.dump({"P0": P0, "B": B, "T": T, "ks": KS,
           "B_vs_P0_HR20_pp": pp(bh, p0h), "B_vs_P0_HR20_rel": rel(bh, p0h),
           "B_vs_P0_NDCG20_pp": pp(bn, p0n), "B_vs_P0_NDCG20_rel": rel(bn, p0n),
           "T_vs_B_HR20_pp": pp(th, bh), "T_vs_B_HR20_rel": rel(th, bh),
           "T_vs_B_NDCG20_pp": pp(tn, bn), "T_vs_B_NDCG20_rel": rel(tn, bn),
           "content_only_beats_p0": bool(bh > p0h and bn > p0n),
           "letter_beats_content_only": bool(th > bh and tn > bn)},
          open("artifacts/letter_stage4_content_only/final_comparison.json", "w"),
          indent=2)
