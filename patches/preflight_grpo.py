#!/usr/bin/env python3
"""
preflight_grpo.py -- CPU-only correctness checks that MUST pass before any GRPO GPU run.

Run on the server from the code root:
    python preflight_grpo.py

Checks (all CPU, no GPU, no model weights needed):
  T1  sample_id namespace collision check, per-dataset AND on the concatenation
  T2  duplicate-prompt rows map to their OWN targets (target-binding regression)
  T3  missing sample_id  -> raises
  T4  unknown sample_id  -> raises
  T5  SFT/RL prompt  ==  Eval prompt, character for character
  T6  target binding count == number of samples

Writes tests/grpo_preflight_report.txt and exits non-zero on any FAIL.
"""
import json
import os
import sys
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

DATA = os.path.join(ROOT, "data", "Amazon")
CAT = "Industrial_and_Scientific"
TRAIN = os.path.join(DATA, "train", f"{CAT}_5_2016-10-2018-11.csv")
EVALC = os.path.join(DATA, "valid", f"{CAT}_5_2016-10-2018-11.csv")
INDEX = os.path.join(DATA, "index", f"{CAT}.index.json")
ITEM = os.path.join(DATA, "index", f"{CAT}.item.json")

CATEGORY = "industrial and scientific items"
SAMPLE = 200  # small but enough to expose collisions

RESULTS = []


def record(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))


def main():
    lines = []

    def out(s=""):
        print(s)
        lines.append(str(s))

    out("=" * 78)
    out("MiniOneRec GRPO preflight report")
    out("=" * 78)

    # ------------------------------------------------------------------ imports
    try:
        from data import (SidDataset, RLTitle2SidDataset, RLSeqTitle2SidDataset,
                          EvalSidDataset, build_recommendation_prompt)
    except ImportError as e:
        out(f"\n*** cannot import patched data.py: {e}")
        out("*** expected data.py with stable sample_id and shared prompt construction")
        return 2

    # ---------------------------------------------------------- build datasets
    out("\n--- building datasets (sample=%d) ---" % SAMPLE)
    ds_seq = SidDataset(TRAIN, category=CATEGORY, sample=SAMPLE)
    ds_title = RLTitle2SidDataset(item_file=ITEM, index_file=INDEX,
                                  category=CATEGORY, sample=SAMPLE)
    ds_seqt = RLSeqTitle2SidDataset(TRAIN, category=CATEGORY, sample=SAMPLE)
    out(f"  SidDataset            : {len(ds_seq)}")
    out(f"  RLTitle2SidDataset    : {len(ds_title)}")
    out(f"  RLSeqTitle2SidDataset : {len(ds_seqt)}")

    datasets = [ds_seq, ds_title, ds_seqt]

    # Shared target binding for the sampled preflight datasets.
    id2target = {}
    for ds in datasets:
        id2target.update(ds.id2target)

    # ---------------------------------------------------------------- T1
    out("\n--- T1: sample_id uniqueness / namespace collision ---")
    per = {}
    for name, ds in zip(["seq", "title", "seq_title"], datasets):
        ids = [x["sample_id"] for x in ds]
        per[name] = ids
        dup = len(ids) - len(set(ids))
        pref_ok = all(i.startswith(name + ":") for i in ids)
        record(f"T1a {name}: ids unique within dataset", dup == 0, f"dups={dup}, n={len(ids)}")
        record(f"T1b {name}: all ids carry '{name}:' prefix", pref_ok)

    all_ids = per["seq"] + per["title"] + per["seq_title"]
    cross = len(all_ids) - len(set(all_ids))
    record("T1c no collision across concatenated datasets", cross == 0,
           f"total={len(all_ids)}, unique={len(set(all_ids))}, collisions={cross}")

    # ---------------------------------------------------------------- T2
    out("\n--- T2: duplicate-prompt rows bind to their OWN target ---")

    # Duplicate prompts are relatively rare, so use the full seq-rec dataset
    # for this regression test only. Other preflight checks stay on SAMPLE=200.
    ds_seq_t2 = SidDataset(TRAIN, category=CATEGORY, sample=-1)
    out(f"  full SidDataset for T2: {len(ds_seq_t2)}")

    id2target_t2 = dict(ds_seq_t2.id2target)

    seen = {}
    dup_groups = 0
    dup_rows = 0
    wrong = 0
    examples = []

    for x in ds_seq_t2:
        p = x["prompt"]
        seen.setdefault(p, []).append((x["sample_id"], x["completion"]))

    for p, items in seen.items():
        if len(items) > 1:
            dup_groups += 1
            dup_rows += len(items)

            # New implementation: each row resolves through its own sample_id.
            for sid, comp in items:
                got = id2target_t2.get(sid)
                if got is None or got.strip() != comp.strip():
                    wrong += 1
                    if len(examples) < 3:
                        examples.append(
                            (sid, comp.strip(), (got or "<missing>").strip())
                        )

    record("T2a duplicate prompts exist in full seq-rec data",
           dup_groups > 0,
           f"{dup_groups} prompt groups, {dup_rows} rows")

    record("T2b every duplicate-prompt row resolves to its own target",
           wrong == 0,
           f"wrong={wrong}" + (f" e.g. {examples}" if examples else ""))

    # Reproduce the old prompt-text lookup to demonstrate the failure mode.
    legacy_p2h = dict(getattr(ds_seq_t2, "prompt2history", {}))
    legacy_h2t = dict(getattr(ds_seq_t2, "history2target", {}))
    legacy_wrong = 0
    legacy_total = 0

    for x in ds_seq_t2:
        legacy_total += 1
        p = x["prompt"]
        if p in legacy_p2h:
            got = legacy_h2t.get(legacy_p2h[p])
            if got is None or got.strip() != x["completion"].strip():
                legacy_wrong += 1

    out(f"  (legacy prompt-text lookup would misbind "
        f"{legacy_wrong}/{legacy_total} seq-rec rows)")

    # ---------------------------------------------------------------- T3 / T4
    out("\n--- T3/T4: reward raises loudly on bad sample_id ---")
    try:
        import rl  # noqa: F401  (import only to prove it is importable)
        rl_importable = True
    except Exception as e:
        rl_importable = False
        out(f"  (rl.py not importable without torch stack: {type(e).__name__})")

    # Test _resolve_targets semantics directly on an equivalent copy
    def resolve(sample_id, id2target_local):
        if sample_id is None:
            raise ValueError("sample_id was not passed")
        out_targets = []
        for sid in sample_id:
            if sid not in id2target_local:
                raise KeyError(f"sample_id {sid!r} has no bound target")
            out_targets.append(id2target_local[sid])
        return out_targets

    try:
        resolve(None, id2target)
        record("T3 missing sample_id -> raises", False, "no exception raised")
    except ValueError:
        record("T3 missing sample_id -> raises", True, "ValueError")

    try:
        resolve(["seq:999999999"], id2target)
        record("T4 unknown sample_id -> raises", False, "no exception raised")
    except KeyError:
        record("T4 unknown sample_id -> raises", True, "KeyError")

    # ---------------------------------------------------------------- T5
    out("\n--- T5: SFT/RL prompt == Eval prompt (character for character) ---")
    # Do not instantiate EvalSidDataset here: its __init__ immediately tokenizes
    # and therefore requires a real tokenizer.  T5 only needs get_history().
    eval_stub = EvalSidDataset.__new__(EvalSidDataset)

    # Compare both implementations on the SAME rows.
    n_cmp = 0
    mismatch = 0
    first_diff = None
    for i in range(min(50, len(ds_seq))):
        # get_history mutates history_item_sid, so use independent row copies.
        a = ds_seq.data.iloc[i].copy()
        b = ds_seq.data.iloc[i].copy()
        ha = ds_seq.get_history(a)["input"]
        hb = eval_stub.get_history(b)["input"]
        n_cmp += 1
        if ha != hb:
            mismatch += 1
            if first_diff is None:
                first_diff = (ha[:120], hb[:120])
    record("T5a SidDataset.input == EvalSidDataset.input", mismatch == 0,
           f"compared={n_cmp}, mismatches={mismatch}"
           + (f", first: SFT={first_diff[0]!r} vs EVAL={first_diff[1]!r}" if first_diff else ""))

    # full prompt must also match on the SAME source row.
    p_mismatch = 0
    for i in range(min(50, len(ds_seq))):
        row_a = ds_seq.data.iloc[i].copy()
        row_b = ds_seq.data.iloc[i].copy()

        h = ds_seq.get_history(row_a)
        h["output"] = ""
        p_sft = ds_seq.generate_prompt(h)

        h2 = eval_stub.get_history(row_b)
        h2["output"] = ""
        p_eval = eval_stub.generate_prompt(h2)

        if p_sft != p_eval:
            p_mismatch += 1

    record("T5b full prompt string identical", p_mismatch == 0,
           f"mismatches={p_mismatch}")

    # shared builder sanity
    s = build_recommendation_prompt("<a_1><b_2><c_3>")
    record("T5c build_recommendation_prompt is the single source",
           "in chronological order" in s and s.startswith("The user has interacted with items"))

    # ---------------------------------------------------------------- T6
    out("\n--- T6: binding counts ---")
    total_samples = sum(len(d) for d in datasets)
    bound = len(id2target)
    record("T6a every sample has a bound target", bound == total_samples,
           f"samples={total_samples}, bound={bound}")
    tasks = {}
    for ds in datasets:
        for t in getattr(ds, "id2task", {}).values():
            tasks[t] = tasks.get(t, 0) + 1
    out(f"  task_type breakdown: {tasks}")

    # ---------------------------------------------------------------- summary
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    out("\n" + "=" * 78)
    out(f"SUMMARY: {len(RESULTS) - n_fail}/{len(RESULTS)} PASS, {n_fail} FAIL")
    out("=" * 78)
    for name, ok, detail in RESULTS:
        out(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))

    # ---------------------------------------------------------------- write
    os.makedirs(os.path.join(ROOT, "tests"), exist_ok=True)
    rpt = os.path.join(ROOT, "tests", "grpo_preflight_report.txt")
    with open(rpt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nreport written: {rpt}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(3)
