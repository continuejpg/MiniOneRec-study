#!/usr/bin/env python3
"""
analysis/audit_shuffled_eval.py

Final read-only provenance audit of the shuffled-SID beam20 evaluation.

Checks
  1. the artefact exists and holds 4533 samples
  2. exactly 20 predictions per sample
  3. the full HR/NDCG grid recomputed with the REAL calc.py (imported, unmodified)
  4. HR@20 / NDCG@20 match the recorded values to < 1e-8
  5. prediction legality, unique-SID count per sample, DuplicateRate@20
  6. clean-SFT vs shuffled-SID absolute and relative deltas

No training, no evaluation, no GPU work. By default it writes
analysis/results/shuffled_eval_provenance.json; pass --check-only to run the
identical checks without writing anything.
"""
import argparse
import importlib.util
import io
import json
import os
import sys
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import (  # noqa: E402
    CATEGORY, INFO, INDEX, RESULTS_DIR,
    SHUFFLED_INFO, SHUFFLED_TEST, repo, run,
)

CAT = CATEGORY
SHUF_PRED = run("eval_shuffled_sid", "test_beam20.json")
CLEAN_PRED = run("eval_clean_sft", "test_beam20.json")
SHUF_INFO = SHUFFLED_INFO
CLEAN_INFO = INFO
SHUF_INDEX = os.path.join(os.path.dirname(SHUFFLED_TEST), f"{CATEGORY}.index.json")
ORIG_INDEX = INDEX
TEST_CSV = SHUFFLED_TEST
OUT_JSON = os.path.join(RESULTS_DIR, "shuffled_eval_provenance.json")

# recorded values the audit must reproduce
EXPECT_NDCG = [0.06132804, 0.07315633, 0.07599527, 0.07921379, 0.08157358]
EXPECT_HR = [0.06132804, 0.08140304, 0.08824178, 0.09816898, 0.10765497]
TOL = 1e-8

CLEAN_HR20 = 0.19832341
CLEAN_NDCG20 = 0.11786798

results = {"checks": [], "problems": []}


def check(name, ok, detail=""):
    results["checks"].append({"check": name, "pass": bool(ok), "detail": detail})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    return ok


def run_calc(pred_path, info_path):
    """Import the real calc.py and call gao(); parse the printed arrays."""
    spec = importlib.util.spec_from_file_location("calc_real", repo("calc.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    buf = io.StringIO()
    with redirect_stdout(buf):
        mod.gao(path=pred_path, item_path=info_path)
    out = buf.getvalue()
    hr = ndcg = None
    for line in out.splitlines():
        if line.startswith("NDCG:"):
            ndcg = [float(x) for x in line.split("\t")[-1].strip().strip("[]").split()]
        elif line.startswith("HR"):
            hr = [float(x) for x in line.split("\t")[-1].strip().strip("[]").split()]
    return hr, ndcg, out


def main(check_only=False):
    print("=" * 100)
    print("SHUFFLED-SID beam20 EVALUATION -- READ-ONLY PROVENANCE AUDIT")
    if check_only:
        print("MODE: --check-only (no file is written)")
    print("=" * 100)

    # ---------------------------------------------------------------- 1
    print("\n1. artefact")
    if not os.path.exists(SHUF_PRED):
        check("artefact exists", False, SHUF_PRED)
        print("*** cannot continue ***")
        return 1
    sz = os.path.getsize(SHUF_PRED)
    check("artefact exists", True, f"{SHUF_PRED} ({sz:,} bytes)")
    d = json.load(open(SHUF_PRED, encoding="utf-8"))
    check("4533 samples", len(d) == 4533, f"{len(d)}")
    check("schema keys == ['input','output','predict']",
          list(d[0].keys()) == ["input", "output", "predict"], str(list(d[0].keys())))

    # ---------------------------------------------------------------- 2
    print("\n2. predictions per sample")
    ns = sorted({len(x["predict"]) for x in d})
    check("exactly 20 predictions per sample", ns == [20], f"observed {ns}")

    # ---------------------------------------------------------------- 3+4
    print("\n3+4. full HR/NDCG via the REAL calc.py (imported, unmodified)")
    hr, ndcg, raw = run_calc(SHUF_PRED, CLEAN_INFO)
    print(f"  calc.py stdout tail:")
    for line in raw.splitlines()[-4:]:
        if line.strip() and "it/s" not in line:
            print(f"    {line}")
    print(f"\n  HR   = {hr}")
    print(f"  NDCG = {ndcg}")

    for i, (e_h, e_n) in enumerate(zip(EXPECT_HR, EXPECT_NDCG)):
        dh, dn = abs(hr[i] - e_h), abs(ndcg[i] - e_n)
        results.setdefault("grid", []).append(
            {"k": [1, 3, 5, 10, 20][i], "HR": hr[i], "NDCG": ndcg[i],
             "expected_HR": e_h, "expected_NDCG": e_n,
             "abs_diff_HR": dh, "abs_diff_NDCG": dn})
    e_hr = abs(hr[-1] - EXPECT_HR[-1])
    e_nd = abs(ndcg[-1] - EXPECT_NDCG[-1])
    check("HR grid matches to <1e-8",
          all(abs(a - b) < TOL for a, b in zip(hr, EXPECT_HR)),
          f"max |diff| = {max(abs(a-b) for a,b in zip(hr, EXPECT_HR)):.3e}")
    check("NDCG grid matches to <1e-8",
          all(abs(a - b) < TOL for a, b in zip(ndcg, EXPECT_NDCG)),
          f"max |diff| = {max(abs(a-b) for a,b in zip(ndcg, EXPECT_NDCG)):.3e}")
    check("HR@20 < 1e-8", e_hr < TOL, f"|diff| = {e_hr:.3e}")
    check("NDCG@20 < 1e-8", e_nd < TOL, f"|diff| = {e_nd:.3e}")

    # ---------------------------------------------------------------- 5
    print("\n5. legality / duplicates")
    orig = json.load(open(ORIG_INDEX, encoding="utf-8"))
    shuf = json.load(open(SHUF_INDEX, encoding="utf-8"))
    legal_orig = {"".join(v) for v in orig.values()}
    legal_shuf = {"".join(v) for v in shuf.values()}
    check("original and shuffled legal SID sets identical", legal_orig == legal_shuf,
          f"{len(legal_orig)} vs {len(legal_shuf)}")

    tot = sum(len(x["predict"]) for x in d)
    in_legal_o = sum(1 for x in d for p in x["predict"] if p.strip() in legal_orig)
    in_legal_s = sum(1 for x in d for p in x["predict"] if p.strip() in legal_shuf)
    legal_rate = in_legal_o / tot
    check("LegalRate@20 == 100% (vs ORIGINAL catalogue SIDs)",
          in_legal_o == tot, f"{in_legal_o}/{tot} = {legal_rate*100:.4f}%")
    check("LegalRate@20 == 100% (vs SHUFFLED catalogue SIDs)",
          in_legal_s == tot, f"{in_legal_s}/{tot} = {in_legal_s/tot*100:.4f}%")

    uniq = [len({p.strip() for p in x["predict"]}) for x in d]
    dup_rate = sum((20 - u) / 20 for u in uniq) / len(uniq)
    n_dup_samples = sum(1 for u in uniq if u < 20)
    check("DuplicateRate@20 == 0", abs(dup_rate) < 1e-12,
          f"{dup_rate*100:.4f}%  ({n_dup_samples}/{len(d)} samples with a repeat)")
    check("unique SIDs per sample == 20 for all rows", set(uniq) == {20},
          f"min={min(uniq)} max={max(uniq)} mean={sum(uniq)/len(uniq):.4f}")

    # target consistency: output must equal the shuffled test CSV item_sid
    import csv
    rows = list(csv.DictReader(open(TEST_CSV, encoding="utf-8")))
    m = sum(1 for a, b in zip(d, rows) if a["output"].strip() == b["item_sid"])
    check("sample['output'] == shuffled test.csv item_sid for every row",
          m == len(rows), f"{m}/{len(rows)}")

    # history SIDs in the prompt must be shuffled-catalogue SIDs
    hist_ok = all(h in legal_shuf
                  for x in d for h in
                  __import__("re").findall(r"<a_\d+><b_\d+><c_\d+>", x["input"]))
    check("all history SIDs in the prompt are shuffled-catalogue SIDs", hist_ok)

    # how many prompts actually differ from the clean run?
    clean = json.load(open(CLEAN_PRED, encoding="utf-8"))
    same_in = sum(1 for a, b in zip(d, clean) if a["input"] == b["input"])
    same_out = sum(1 for a, b in zip(d, clean) if a["output"] == b["output"])
    results["intervention_effective"] = {
        "rows_with_identical_input": same_in,
        "rows_with_identical_output": same_out,
        "rows": len(d),
    }
    check("the prompt/target actually changed vs clean (intervention live)",
          same_in < len(d) and same_out < len(d),
          f"identical input rows={same_in}, identical output rows={same_out}")

    # ---------------------------------------------------------------- 6
    print("\n6. clean SFT vs shuffled SID -- metrics from the two artefacts")
    chr_, cnd, _ = run_calc(CLEAN_PRED, CLEAN_INFO)
    print(f"  clean    HR   = {chr_}")
    print(f"  clean    NDCG = {cnd}")

    ks = [1, 3, 5, 10, 20]
    delta = []
    print(f"\n  {'K':>4} {'clean HR':>11} {'shuf HR':>11} {'abs d':>10} {'rel d':>9} "
          f"| {'clean NDCG':>11} {'shuf NDCG':>11} {'abs d':>10} {'rel d':>9}")
    for i, k in enumerate(ks):
        dh = hr[i] - chr_[i]
        dn = ndcg[i] - cnd[i]
        rh = dh / chr_[i] * 100 if chr_[i] else float("nan")
        rn = dn / cnd[i] * 100 if cnd[i] else float("nan")
        print(f"  {k:>4} {chr_[i]:>11.6f} {hr[i]:>11.6f} {dh:>+10.6f} {rh:>+8.2f}% "
              f"| {cnd[i]:>11.6f} {ndcg[i]:>11.6f} {dn:>+10.6f} {rn:>+8.2f}%")
        delta.append({"K": k, "clean_HR": chr_[i], "shuffled_HR": hr[i],
                      "abs_delta_HR": dh, "rel_delta_HR_pct": rh,
                      "clean_NDCG": cnd[i], "shuffled_NDCG": ndcg[i],
                      "abs_delta_NDCG": dn, "rel_delta_NDCG_pct": rn})

    check("clean artefact still reproduces its official HR@20",
          abs(chr_[-1] - CLEAN_HR20) < 1e-7, f"{chr_[-1]:.12f} vs {CLEAN_HR20}")
    check("clean artefact still reproduces its official NDCG@20",
          abs(cnd[-1] - CLEAN_NDCG20) < 1e-7, f"{cnd[-1]:.12f} vs {CLEAN_NDCG20}")

    # relative HR@20 drop
    drop = (chr_[-1] - hr[-1]) / chr_[-1] * 100
    print(f"\n  HR@20   : {chr_[-1]:.6f} -> {hr[-1]:.6f}   drop {drop:.2f}%")
    print(f"  NDCG@20 : {cnd[-1]:.6f} -> {ndcg[-1]:.6f}   "
          f"drop {(cnd[-1]-ndcg[-1])/cnd[-1]*100:.2f}%")

    n_fail = sum(1 for c in results["checks"] if not c["pass"])
    results.update({
        "artefact": SHUF_PRED,
        "n_samples": len(d),
        "predictions_per_sample": 20,
        "metrics": {"K": ks, "HR": hr, "NDCG": ndcg},
        "expected": {"HR": EXPECT_HR, "NDCG": EXPECT_NDCG},
        "legality": {"legal_rate_vs_original": legal_rate,
                     "legal_rate_vs_shuffled": in_legal_s / tot,
                     "duplicate_rate": dup_rate,
                     "unique_per_sample": {"min": min(uniq), "max": max(uniq),
                                           "mean": sum(uniq) / len(uniq)}},
        "clean_vs_shuffled": delta,
        "training_provenance_caveat": {
            "run": "/root/autodl-tmp/runs/industrial_sft_shuffled_sid",
            "completed_steps": 2496,
            "interrupted_after_step": 2180,
            "resumed_from": "checkpoint-2125",
            "retrained_steps": "2125-2180",
            "resume_log": "/root/autodl-tmp/runs/industrial_sft_shuffled_sid/resume_from_2125.log",
            "train_loss_reported_by_resume_segment": 0.060042200884662375,
            "train_loss_all_step_mean": 0.802715,
            "clean_sft_train_loss_all_step_mean": 0.7712313783569977,
            "warning": ("The 0.0600 train_loss printed by the resume segment is a "
                        "RESUMED-SEGMENT statistic, not the full-training loss. Use the "
                        "all-step mean 0.8027 for any comparison with clean SFT (0.7712). "
                        "Because steps 2125-2180 were retrained, the two arms do NOT share "
                        "a bit-identical optimisation trajectory; the final LR reconverged "
                        "to the identical value 1.21163166e-07 and the loss trajectory is "
                        "continuous across the resume."),
        },
        "summary": {"checks": len(results["checks"]), "failed": n_fail,
                    "problems": len(results["problems"])},
    })
    if check_only:
        print("\n" + "=" * 100)
        print(f"AUDIT: {len(results['checks'])-n_fail}/{len(results['checks'])} PASS, {n_fail} FAIL")
        print("[check-only] nothing written")
        print("=" * 100)
        return 1 if n_fail else 0

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 100)
    print(f"AUDIT: {len(results['checks'])-n_fail}/{len(results['checks'])} PASS, {n_fail} FAIL")
    print(f"[save] {OUT_JSON}")
    print("=" * 100)
    return 1 if n_fail else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check-only", action="store_true",
                    help="run the identical checks but write no file")
    args = ap.parse_args()
    sys.exit(main(check_only=args.check_only))
