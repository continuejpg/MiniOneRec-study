#!/usr/bin/env python3
"""
analysis/benchmark_inference.py

Inference quality-latency benchmark for the generative recommender:
how does recommendation quality trade off against inference cost, across
beam width and batch size?

=============================================================================
DESIGN
=============================================================================

1. PROTOCOL IDENTITY (no second protocol)
   The generation path is executed by the repository's OWN committed
   `evaluate.py`, imported and called in-process via `evaluate.main(...)`.
   Prompts, SID parsing, the constrained-decoding trie (built from the
   ORIGINAL INFO file), the logits processor and the generate() call are NOT
   re-implemented here. Every knob is forwarded through evaluate.main()'s own
   parameters.

2. ONE evaluate.main() CALL PER CONFIGURATION
   The model/tokenizer are loaded once per configuration. Warmup is the
   `warmup_batches` leading batches INSIDE that single call, executed on the
   same model instance and discarded before the measured pass. The measured
   pass then traverses the FULL dataloader and produces every prediction.

3. MEASUREMENT CONTRACT
   For each measured batch, inside evaluate.py:
       torch.cuda.synchronize()
       t0 = time.perf_counter()
       model.generate(...)
       torch.cuda.synchronize()
       elapsed = time.perf_counter() - t0
   Warmup batches pass on_generate=None and therefore never reach that block.

4. METRIC IDENTITY
   HR/NDCG are computed by the ORIGINAL calc.py, unmodified, from the
   prediction file and its `output` targets. calc.py's `item_path` argument
   builds `item_dict`, which is used ONLY to count out-of-universe candidates
   (the printed CC number); it never participates in HR/NDCG. Verified
   empirically: the same prediction file scored against the full 3686-line
   codebook and against a 7-line codebook yields bit-identical HR and NDCG.
   Consequence: subset scoring needs NO subset codebook, and the SID universe
   is never shrunk here.

5. FAIL-LOUD
   * refuses to overwrite an existing output directory
   * refuses to report a cutoff above the beam width
   * refuses to run without CUDA
   * source-target misalignment, wrong sample count, wrong candidate count,
     LegalRate != 1 or DuplicateRate != 0 are all HARD failures

=============================================================================
CUTOFF RULE (enforced in code)
=============================================================================
   available cutoffs = [c for c in (1,3,5,10,20) if c <= num_beams]

=============================================================================
MEASUREMENT SEMANTICS (also written into results.json under "semantics")
=============================================================================
   latency_p50/p95          model.generate-only batch latency (ms)
   request_latency_p50/p95  generate-only single-request latency; batch_size==1
   generation_seconds       sum of measured generate calls only. excludes model
                            load, warmup, JSON writing and calc.py
   samples_per_second       num_samples / generation_seconds
   peak_vram_gb             torch.cuda.max_memory_allocated() during the
                            MEASURED GENERATION WINDOW only, i.e. PyTorch peak
                            allocated memory, after reset_peak_memory_stats()
                            has run post-warmup. NOT total device memory.
   peak_device_used_gb      separately named, different quantity: max device
                            used memory sampled at 50 ms (includes other
                            processes and the CUDA context)

=============================================================================
USAGE
=============================================================================
    python analysis/benchmark_inference.py --mode quality
    python analysis/benchmark_inference.py --mode efficiency
    python analysis/benchmark_inference.py --mode all
    python analysis/benchmark_inference.py --preflight      # static, no GPU
"""
import argparse
import csv
import io
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import threading
import time

# --------------------------------------------------------------------------- #
# import paths
#
# The benchmark lives in <repo>/analysis/ but the protocol it drives lives in
# <repo>/evaluate.py. Both directories must be importable, and neither may
# depend on the shell's cwd:
#
#   <repo>/analysis  -> `import _paths`               (this file's own dir)
#   <repo>           -> `import evaluate as EV`       (the protocol under test)
#
# The repo root is derived from _paths.PROJECT_ROOT, which is itself computed
# from this file's __file__ (and honours a PROJECT_ROOT env override), so
# `python /any/cwd/.../analysis/benchmark_inference.py` works unchanged and no
# absolute server path is ever hard-coded.
# --------------------------------------------------------------------------- #
_ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _ANALYSIS_DIR not in sys.path:
    sys.path.insert(0, _ANALYSIS_DIR)

from _paths import (  # noqa: E402
    CATEGORY, INFO, PROJECT_ROOT, RUN_ROOT, TEST, repo, run,
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --------------------------------------------------------------------------- #
# formal protocol constants -- identical to scripts/eval_seq_only_clean.sh
# --------------------------------------------------------------------------- #
FORMAL_BATCH_SIZE = 8
FORMAL_BEAM = 20
FORMAL_SEED = 42
FORMAL_LENGTH_PENALTY = 0.0
FORMAL_K = 0
FORMAL_MAX_NEW_TOKENS = 256
FORMAL_TIMING_CALLBACK = None
FORMAL_WARMUP_BATCHES = 0

ALL_CUTOFFS = (1, 3, 5, 10, 20)

# benchmark warmup: leading batches inside the single EV.main() call
BENCH_WARMUP_BATCHES = 3

# quality sweep: beam -> cutoffs actually reported
QUALITY_SWEEP = {
    5: (1, 3, 5),
    10: (1, 3, 5, 10),
    20: (1, 3, 5, 10, 20),
    50: (1, 3, 5, 10, 20),
}

# efficiency sweep: fixed beam, varying batch
EFFICIENCY_BEAM = 20
EFFICIENCY_BATCHES = (1, 8, 32)
EFFICIENCY_SUBSET = 512          # first N test rows; identical for every batch size
EFFICIENCY_SUBSET_NAME = f"efficiency_subset_{EFFICIENCY_SUBSET}.csv"
EFFICIENCY_SUBSET_REL = f"_inputs/{EFFICIENCY_SUBSET_NAME}"
EFFICIENCY_SUBSET_SELECTION = "first_512_in_original_csv_order"
BATCH_INVARIANCE_BASELINE = 1    # reference arm for the invariance diagnostic

# reference: formal clean SFT beam20 result (protocol-identity check)
REFERENCE = {"HR@20": 0.19832341, "NDCG@20": 0.11786798}
REFERENCE_TOL = 1e-6

MAIN_MODEL = run("industrial_sft", "final_checkpoint")
OUT_DIR = run("inference_benchmark")

SEMANTICS = {
    "latency_p50_ms": "model.generate-only batch latency, nearest-rank percentile",
    "latency_p95_ms": "model.generate-only batch latency, nearest-rank percentile",
    "request_latency_p50_ms": "generate-only single-request latency (batch_size==1 only)",
    "request_latency_p95_ms": "generate-only single-request latency (batch_size==1 only)",
    "generation_seconds": ("sum of measured model.generate calls only; excludes model "
                           "load, warmup batches, JSON writing and calc.py"),
    "samples_per_second": "num_samples / generation_seconds",
    "peak_vram_gb": ("torch.cuda.max_memory_allocated() over the MEASURED GENERATION "
                     "WINDOW only (PyTorch peak allocated); reset_peak_memory_stats() "
                     "runs after warmup and before the measured loop. NOT total "
                     "device memory and NOT other processes' memory"),
    "peak_device_used_gb": ("separately named proxy: max device used memory sampled at "
                            "50 ms, including the CUDA context and any other process. "
                            "Not comparable with peak_vram_gb"),
    "hr_ndcg": ("computed by the original calc.py from prediction['predict'] and "
                "prediction['output']; calc.py's item universe only affects its CC "
                "counter and never HR/NDCG, so no SID universe is ever shrunk"),
    "warmup": (f"the first {BENCH_WARMUP_BATCHES} batches inside the single "
               "evaluate.main() call, executed with timing_callback=None and discarded"),
}

# --------------------------------------------------------------------------- #
# protocol import -- the single place `evaluate` is imported
# --------------------------------------------------------------------------- #


def import_evaluate():
    """Import the protocol under test using the benchmark's OWN sys.path.

    Returns (module, error_string). Exactly one of the two is None.

    Why this exists as a function
    -----------------------------
    A previous revision imported evaluate only inside run_once() via a bare
    `import evaluate as EV`. That failed at runtime with
    `ModuleNotFoundError: No module named 'evaluate'`, because <repo> was not on
    sys.path -- while --preflight still reported 99 PASS, since its only check
    was a source-text assertion that the line existed. The preflight never
    actually imported anything.

    Both --preflight and run_once now go through this function, so the check
    cannot drift from the real import path. Importing evaluate.py has no side
    effects: it defines helpers and main() and never loads a model, touches
    CUDA or generates anything.

    sys.modules is cleared first so that the import is genuinely re-evaluated
    against the CURRENT sys.path rather than served from a cached module.
    """
    import importlib

    sys.modules.pop("evaluate", None)
    try:
        mod = importlib.import_module("evaluate")
    except BaseException as e:                    # noqa: BLE001 - report anything
        return None, f"{type(e).__name__}: {e}"
    if not callable(getattr(mod, "main", None)):
        return None, "module 'evaluate' has no callable main()"
    return mod, None


# --------------------------------------------------------------------------- #
# static rules (pure functions -- also exercised by --preflight)
# --------------------------------------------------------------------------- #


def available_cutoffs(num_beams):
    """Cutoffs a beam-k run is allowed to report."""
    return tuple(c for c in ALL_CUTOFFS if c <= num_beams)


def parse_metrics(metrics_txt):
    """Parse calc.py stdout: lines 'HR\\t[...]' and 'NDCG:\\t[...]'."""
    hr = ndcg = None
    for line in metrics_txt.splitlines():
        line = line.strip()
        if line.startswith("HR") and "[" in line:
            hr = [float(x) for x in re.findall(r"[-\d.eE+]+", line[line.index("["):])]
        elif line.startswith("NDCG") and "[" in line:
            ndcg = [float(x) for x in re.findall(r"[-\d.eE+]+", line[line.index("["):])]
    return hr, ndcg


def score_predictions(pred_path, calc_py):
    """Score a prediction artifact with the ORIGINAL calc.py.

    The item universe is always the ORIGINAL INFO file: calc.py's item_dict
    never influences HR/NDCG, so shrinking it would only change the CC counter
    while making the numbers harder to interpret.
    """
    r = subprocess.run([sys.executable, calc_py, "--path", pred_path,
                        "--item_path", INFO],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"calc.py failed on {pred_path}:\n{r.stderr[-2000:]}")
    return parse_metrics(r.stdout), r.stdout


def load_legal(info_path):
    return {l.split("\t")[0].strip()
            for l in open(info_path, encoding="utf-8").read().splitlines() if l.strip()}


def load_predictions(pred_path):
    return json.load(open(pred_path, encoding="utf-8"))


def read_test_targets(n=None):
    """Targets of the ORIGINAL test split, in file order (all rows when n is None).

    Justification for row-order alignment: evaluate.py builds its dataset with
    EvalSidDataset -> CSVBaseDataset, which does `pd.read_csv(train_file)` and
    only reshuffles when `sample > 0`. The benchmark always passes sample=-1
    (FORMAL_K convention kept: K=0, no sampling), so the dataset order equals
    the CSV order and prediction row i corresponds to CSV row i.

    NOTE: this reads the ORIGINAL TEST file only. Use read_targets_from() when
    the configuration being checked evaluates something else.
    """
    return read_targets_from(TEST, n)


def read_targets_from(csv_path, n=None):
    """item_sid targets of an arbitrary test CSV, in file order."""
    import csv as _csv
    with open(csv_path, encoding="utf-8") as f:
        rows = list(_csv.DictReader(f))
    if n is not None:
        rows = rows[:n]
    return [r["item_sid"].strip() for r in rows]


def write_efficiency_subset(out_root):
    """Materialise the fixed efficiency workload from the ORIGINAL test split.

    Writes the FIRST `EFFICIENCY_SUBSET` rows, in original CSV order, with the
    header and every column preserved byte-for-byte, to
    <out_root>/_inputs/efficiency_subset_512.csv.

    This file is a provenance artifact of the benchmark: it is what the
    efficiency configurations actually evaluate, so it is kept (not deleted)
    and its SHA256 is recorded in results.json.

    Why this exists
    ---------------
    An earlier revision only *declared* EFFICIENCY_SUBSET = 512 and then passed
    the ORIGINAL test CSV to evaluate.main(), so the efficiency arm silently
    evaluated all 4533 rows and the integrity gate failed with
    `sample count 4533 != expected 512`. The subset has to be a real file that
    is really handed to the protocol.

    Returns the subset path.
    """
    subset_dir = os.path.join(out_root, "_inputs")
    os.makedirs(subset_dir, exist_ok=True)
    subset_path = os.path.join(subset_dir, EFFICIENCY_SUBSET_NAME)

    with open(TEST, encoding="utf-8", newline="") as f:
        lines = f.read().splitlines(keepends=True)
    if len(lines) < EFFICIENCY_SUBSET + 1:
        raise RuntimeError(
            f"original test split has only {len(lines) - 1} data rows, "
            f"cannot take {EFFICIENCY_SUBSET}")

    # header + first N data rows, original order, everything preserved
    with open(subset_path, "w", encoding="utf-8", newline="") as f:
        f.writelines(lines[:EFFICIENCY_SUBSET + 1])

    # ---- assertions: the subset must be exactly what was asked for --------
    import csv as _csv
    with open(TEST, encoding="utf-8", newline="") as f:
        orig_rows = list(_csv.reader(f))
    with open(subset_path, encoding="utf-8", newline="") as f:
        sub_rows = list(_csv.reader(f))

    if len(sub_rows) - 1 != EFFICIENCY_SUBSET:
        raise RuntimeError(
            f"efficiency subset has {len(sub_rows) - 1} data rows, "
            f"expected {EFFICIENCY_SUBSET}")
    if sub_rows[0] != orig_rows[0]:
        raise RuntimeError("efficiency subset header differs from the original test")
    if sub_rows[1:] != orig_rows[1:EFFICIENCY_SUBSET + 1]:
        raise RuntimeError(
            "efficiency subset rows are not identical to the first "
            f"{EFFICIENCY_SUBSET} rows of the original test split")

    return subset_path


def efficiency_subset_provenance(subset_path):
    """The block recorded in results.json so the workload is self-describing."""
    import hashlib
    h = hashlib.sha256(open(subset_path, "rb").read()).hexdigest()
    return {
        "source_test": os.path.relpath(TEST, PROJECT_ROOT).replace("\\", "/"),
        "path": os.path.relpath(subset_path, PROJECT_ROOT).replace("\\", "/"),
        "rows": EFFICIENCY_SUBSET,
        "selection": EFFICIENCY_SUBSET_SELECTION,
        "sha256": h,
    }


# results.json provenance, filled in by main() before the first write
EFFICIENCY_PROVENANCE = None


def check_source_target_alignment(preds, expected_targets):
    """Hard gate: prediction[i]['output'].strip() must equal target[i]."""
    if len(preds) != len(expected_targets):
        return [f"row count {len(preds)} != expected {len(expected_targets)}"]
    mism = [i for i, (p, t) in enumerate(zip(preds, expected_targets))
            if p["output"].strip() != t]
    if mism:
        head = ", ".join(f"{i}:{preds[i]['output'].strip()!r}!={expected_targets[i]!r}"
                         for i in mism[:5])
        return [f"{len(mism)} source-target misalignment(s); first: {head}"]
    return []


def check_integrity(preds, info_path, expected_n, expected_targets):
    """Hard integrity checks -- any failure is fatal for the configuration."""
    problems = []
    legal = load_legal(info_path)
    n = len(preds)
    if n != expected_n:
        problems.append(f"sample count {n} != expected {expected_n}")
    cand_counts = [len(x["predict"]) for x in preds]
    if len(set(cand_counts)) != 1:
        problems.append(f"candidate counts not constant: min={min(cand_counts)} "
                        f"max={max(cand_counts)}")
    tot = sum(cand_counts)
    ok = sum(1 for x in preds for p in x["predict"] if p in legal)
    dup = sum(len(x["predict"]) - len(set(x["predict"])) for x in preds)
    legal_rate = ok / tot if tot else 0.0
    dup_rate = dup / tot if tot else 0.0
    if legal_rate != 1.0:
        problems.append(f"LegalRate != 1 ({legal_rate})")
    if dup_rate != 0.0:
        problems.append(f"DuplicateRate != 0 ({dup_rate})")
    # target alignment is checked BEFORE anything downstream relies on row order
    problems.extend(check_source_target_alignment(preds, expected_targets))
    return {"num_samples": n, "num_candidates": tot,
            "candidates_per_sample": cand_counts[0] if cand_counts else 0,
            "legal_rate": legal_rate, "duplicate_rate": dup_rate}, problems


def percentile(values, p):
    """Nearest-rank percentile (no interpolation)."""
    if not values:
        return None
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round(p / 100.0 * (len(s) - 1)))))
    return s[k]


class DeviceUsedSampler:
    """Poll device used memory on a background thread.

    Reports a SEPARATELY NAMED quantity (peak_device_used_gb). This is NOT
    peak_vram_gb: peak_vram_gb comes from torch.cuda.max_memory_allocated()
    over the measured window, which is the PyTorch-allocated peak.
    """

    def __init__(self, interval_s=0.05):
        self.interval_s = interval_s
        self.peak_mib = None
        self._stop = False
        self._thread = None

    def start(self):
        import torch

        if not torch.cuda.is_available():
            return
        try:
            _free, total = torch.cuda.mem_get_info()
            self.peak_mib = (total - _free) / (1024 ** 2)
        except Exception:
            self.peak_mib = 0.0

        def loop():
            while not self._stop:
                try:
                    _free, total = torch.cuda.mem_get_info()
                    mib = (total - _free) / (1024 ** 2)
                except Exception:
                    break
                if mib > self.peak_mib:
                    self.peak_mib = mib
                time.sleep(self.interval_s)

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop = True
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        return self.peak_mib


# --------------------------------------------------------------------------- #
# atomic output
# --------------------------------------------------------------------------- #


def atomic_write_text(path, text):
    """Same-directory temp file + os.replace, so readers never see a partial file."""
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def render_results(records, out_root):
    """Return (json_text, csv_text) for the current record set."""
    doc = {"semantics": SEMANTICS,
           "reference": REFERENCE}
    if EFFICIENCY_PROVENANCE is not None:
        doc["efficiency_subset"] = EFFICIENCY_PROVENANCE
    doc["configurations"] = records
    json_text = json.dumps(doc, indent=2)
    cols = ["mode", "tag", "status", "beam", "batch_size", "num_samples",
            "num_candidates", "HR@1", "HR@3", "HR@5", "HR@10", "HR@20",
            "NDCG@1", "NDCG@3", "NDCG@5", "NDCG@10", "NDCG@20",
            "generation_seconds", "samples_per_second", "batches",
            "latency_p50_ms", "latency_p95_ms", "latency_mean_ms",
            "request_latency_p50_ms", "request_latency_p95_ms",
            "peak_vram_gb", "peak_device_used_gb",
            "legal_rate", "duplicate_rate", "prediction_json"]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for r in records:
        w.writerow(r)
    return json_text, buf.getvalue()


def write_results(records, out_root):
    """Incrementally persist results after EVERY configuration.

    Atomicity: each file is written to a same-directory temp file and moved
    into place with os.replace(), so a reader sees either the previous complete
    version or the new complete version. If a write fails, the previous
    complete file is left untouched and the error is reported; nothing is
    deleted.
    """
    json_text, csv_text = render_results(records, out_root)
    wrote = []
    for name, text in (("results.json", json_text), ("results.csv", csv_text)):
        p = os.path.join(out_root, name)
        try:
            atomic_write_text(p, text)
            wrote.append(p)
        except OSError as e:
            print(f"    WARNING: could not persist {p}: {e}", file=sys.stderr)
    if wrote:
        print(f"    [incremental save] {len(records)} configuration(s) -> "
              + ", ".join(os.path.basename(w) for w in wrote))
    return wrote


# --------------------------------------------------------------------------- #
# one measured configuration
# --------------------------------------------------------------------------- #


def run_once(*, mode, beam, batch_size, num_samples, out_root, tag, do_quality=True,
             test_data_path=None, expected_targets_n=None):
    """Execute the formal protocol ONCE for this configuration.

    A single evaluate.main() call performs warmup (leading batches, discarded)
    and then the full measured pass, so the model is loaded exactly once.

    `test_data_path` is passed through to evaluate.main() verbatim and is ALSO
    the single source of the expected targets, so a configuration can never be
    checked against a file other than the one it actually evaluated.
    """
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available -- refusing to benchmark on CPU")

    # the protocol under test, resolved through the same helper --preflight uses
    EV, err = import_evaluate()
    if EV is None:
        raise RuntimeError(f"cannot import evaluate from the benchmark context: {err}")

    if test_data_path is None:
        test_data_path = TEST
    expected_n = num_samples if expected_targets_n is None else expected_targets_n

    run_dir = os.path.join(out_root, tag)
    pred_path = os.path.join(run_dir, "test_beam%d.json" % beam)
    os.makedirs(run_dir, exist_ok=False)

    batch_latencies = []          # ms, one entry per MEASURED model.generate call
    batch_sizes = []              # required samples per measured call

    def on_generate(n, ms):
        batch_latencies.append(ms)
        batch_sizes.append(n)

    device_sampler = DeviceUsedSampler()
    device_sampler.start()

    t0 = time.perf_counter()
    EV.main(
        base_model=MAIN_MODEL,
        info_file=INFO,
        category=CATEGORY,
        test_data_path=test_data_path,
        result_json_data=pred_path,
        batch_size=batch_size,
        K=FORMAL_K,
        seed=FORMAL_SEED,
        length_penalty=FORMAL_LENGTH_PENALTY,
        max_new_tokens=FORMAL_MAX_NEW_TOKENS,
        num_beams=beam,
        timing_callback=on_generate,
        warmup_batches=BENCH_WARMUP_BATCHES,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - t0
    peak_device_mib = device_sampler.stop()

    # peak_vram_gb: PyTorch peak allocated over the MEASURED window.
    # evaluate.py resets this statistic after warmup and before the measured
    # loop, and no allocation happens between the last measured batch and this
    # read, so this value describes the measured generation window.
    peak_alloc_gb = None
    if torch.cuda.is_available():
        peak_alloc_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    gen_seconds = sum(batch_latencies) / 1000.0

    preds = load_predictions(pred_path)
    # alignment is read from the VERY file that was handed to evaluate.main()
    expected_targets = read_targets_from(test_data_path, expected_targets_n)
    integ, problems = check_integrity(preds, INFO, expected_n, expected_targets)
    if problems:
        raise RuntimeError("integrity check failed: " + "; ".join(problems))

    rec = {
        "mode": mode,
        "tag": tag,
        "status": "ok",
        "beam": beam,
        "batch_size": batch_size,
        **integ,
        "test_data_path": os.path.relpath(test_data_path, PROJECT_ROOT).replace("\\", "/"),
        "source_target_alignment": f"{len(preds)}/{len(expected_targets)}",
        "generation_seconds": gen_seconds,
        "wall_seconds_including_warmup_and_load": wall_seconds,
        "samples_per_second": integ["num_samples"] / gen_seconds if gen_seconds else None,
        "batches": len(batch_latencies),
        "latency_p50_ms": percentile(batch_latencies, 50),
        "latency_p95_ms": percentile(batch_latencies, 95),
        "latency_mean_ms": statistics.fmean(batch_latencies) if batch_latencies else None,
        "peak_vram_gb": peak_alloc_gb,
        "peak_device_used_gb": (peak_device_mib / 1024.0) if peak_device_mib else None,
        "prediction_json": os.path.relpath(pred_path, PROJECT_ROOT).replace("\\", "/"),
    }
    if batch_size == 1:
        rec["request_latency_p50_ms"] = rec["latency_p50_ms"]
        rec["request_latency_p95_ms"] = rec["latency_p95_ms"]

    if do_quality:
        (hr, ndcg), raw = score_predictions(pred_path, repo("calc.py"))
        cut = available_cutoffs(beam)
        if hr is None or len(hr) < len(cut):
            raise RuntimeError(f"calc.py did not return {len(cut)} cutoffs:\n{raw[-1500:]}")
        for i, c in enumerate(ALL_CUTOFFS):
            if c in cut:
                rec[f"HR@{c}"] = hr[i]
                rec[f"NDCG@{c}"] = ndcg[i]
            # cutoffs above the beam are deliberately absent, not zero
        with open(os.path.join(run_dir, "metrics.txt"), "w", encoding="utf-8") as f:
            f.write(raw)

    with open(os.path.join(run_dir, "per_batch_latency.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["batch_index", "num_samples", "latency_ms"])
        for i, (n, ms) in enumerate(zip(batch_sizes, batch_latencies)):
            w.writerow([i, n, f"{ms:.3f}"])

    return rec


# --------------------------------------------------------------------------- #
# batch-invariance diagnostic
# --------------------------------------------------------------------------- #


def batch_invariance(records):
    """Compare semantic predictions across batch sizes on the SAME subset.

    Compares, per row and in order: output.strip() and the predict list.
    Deliberately NOT a file-digest comparison.

    HR/NDCG for each arm come from the SAME calc.py invocation path used for
    the full runs (score_predictions), against the ORIGINAL INFO universe, so
    the metric definition is identical to the formal one. The only difference
    is the number of rows scored (the efficiency subset), which is stated
    explicitly so the numbers are not confused with the full 4533-row result.
    """
    eff = [r for r in records if r.get("mode") == "efficiency" and r.get("status") == "ok"]
    by_bs = {r["batch_size"]: r for r in eff}
    base_bs = BATCH_INVARIANCE_BASELINE
    if base_bs not in by_bs:
        return {"available": False,
                "reason": f"baseline batch_size {base_bs} not in efficiency results"}
    base = load_predictions(os.path.join(PROJECT_ROOT, by_bs[base_bs]["prediction_json"]))
    others = sorted(b for b in by_bs if b != base_bs)

    out = {"available": True, "baseline_batch_size": base_bs,
           "rows_compared": len(base),
           "metric": "original calc.py HR/NDCG, ORIGINAL INFO universe",
           "note": (f"HR/NDCG below are computed on the {len(base)}-row efficiency "
                    "subset with the SAME metric definition as the formal runs; "
                    "they are comparable ACROSS batch sizes but are NOT the full "
                    "4533-row formal numbers"),
           "comparisons": {}, "subset_rows": len(base), "subset_metrics": {}}

    (bhr, bnd) = score_predictions(
        os.path.join(PROJECT_ROOT, by_bs[base_bs]["prediction_json"]),
        repo("calc.py"))[0]
    out["subset_metrics"][f"batch{base_bs}"] = {"HR": bhr, "NDCG": bnd}

    for b in others:
        other = load_predictions(os.path.join(PROJECT_ROOT, by_bs[b]["prediction_json"]))
        if len(other) != len(base):
            out["comparisons"][f"{base_bs}_vs_{b}"] = {
                "status": "FAIL", "reason": f"sample count {len(other)} != {len(base)}"}
            continue
        mism = [i for i, (x, y) in enumerate(zip(base, other))
                if x["output"].strip() != y["output"].strip()
                or x["predict"] != y["predict"]]
        rate = 1.0 - len(mism) / len(base)
        (ohr, ond) = score_predictions(
            os.path.join(PROJECT_ROOT, by_bs[b]["prediction_json"]),
            repo("calc.py"))[0]
        out["subset_metrics"][f"batch{b}"] = {"HR": ohr, "NDCG": ond}
        out["comparisons"][f"{base_bs}_vs_{b}"] = {
            "status": "MATCH" if not mism else "DIVERGENT",
            "prediction_exact_match_rate": rate,
            "mismatch_rows": len(mism),
            "mismatch_row_indices_head": mism[:20],
        }

    return out


# --------------------------------------------------------------------------- #
# static preflight (CPU only -- no torch import, no CUDA)
# --------------------------------------------------------------------------- #


def preflight():
    """Static, CPU-only checks. Uses the AST so prose cannot be mistaken for code."""
    import ast

    checks = []

    def chk(ok, label, detail=""):
        checks.append((bool(ok), label, detail))

    src_path = os.path.abspath(__file__)
    src = open(src_path, encoding="utf-8").read()
    tree = ast.parse(src)

    pf = next((n for n in tree.body
               if isinstance(n, ast.FunctionDef) and n.name == "preflight"), None)
    pf_range = (pf.lineno, pf.end_lineno) if pf is not None else (0, 0)
    tested = [n for n in tree.body
              if n is not pf and not (pf_range[0] <= getattr(n, "lineno", 0) <= pf_range[1])]

    call_names, literals, func_names = set(), set(), set()
    for node in tested:
        for n in ast.walk(node):
            if isinstance(n, ast.FunctionDef):
                func_names.add(n.name)
            if isinstance(n, ast.Call):
                if isinstance(n.func, ast.Name):
                    call_names.add(n.func.id)
                elif isinstance(n.func, ast.Attribute):
                    call_names.add(n.func.attr)
            if isinstance(n, ast.Constant) and isinstance(n.value, str):
                literals.add(n.value)
            if isinstance(n, ast.JoinedStr):
                for part in n.values:
                    if isinstance(part, ast.Constant) and isinstance(part.value, str):
                        literals.add(part.value)

    # ---- 0. metric identity: no SID-universe shrinking ---------------------
    chk("filter_codebook_to_targets" not in func_names and
        "filter_codebook_to_targets" not in call_names,
        "no subset-codebook shrinking logic exists (removed)")
    chk("subset_codebook" not in "".join(literals),
        "no subset codebook artifact is produced or referenced")
    chk("item_path" in src and "--item_path" in literals,
        "calc.py is invoked with its own item_path argument")
    chk(not any("target SID" in s and "universe" in s.lower() for s in literals),
        "no stale 'shrink the target-SID universe' claim remains")
    score_fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef)
                     and n.name == "score_predictions"), None)
    chk(score_fn is not None, "score_predictions() exists")
    if score_fn is not None:
        ssrc = ast.get_source_segment(src, score_fn)
        chk("--item_path" in ssrc and "INFO" in ssrc,
            "score_predictions always uses the ORIGINAL INFO universe")
        chk("pred_path" in ssrc and "--path" in ssrc,
            "score_predictions takes the prediction file as --path")

    # ---- 0b. target alignment is an unconditional hard gate ----------------
    chk("check_source_target_alignment" in func_names,
        "check_source_target_alignment() exists")
    chk("check_source_target_alignment(preds, expected_targets)" in src,
        "check_integrity() calls the alignment gate rather than returning early")
    integ_fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef)
                     and n.name == "check_integrity"), None)
    isrc = ast.get_source_segment(src, integ_fn)
    chk("problems.extend(check_source_target_alignment" in isrc,
        "alignment problems are appended to `problems` (unconditional)")
    chk(isrc.index("check_source_target_alignment") < isrc.index("return {"),
        "alignment is evaluated before check_integrity returns")
    chk(not isrc.rstrip().endswith("return [], []"),
        "check_integrity never returns an empty problem list unconditionally")
    chk("if problems:\n        raise RuntimeError(\"integrity check failed: \""
        in src.replace("\r\n", "\n"),
        "any integrity problem (incl. misalignment) raises -> config FAILS")
    chk("read_test_targets" in func_names and "item_sid" in literals,
        "targets are read from the test CSV's item_sid column")
    chk('f"{len(preds)}/{len(expected_targets)}"' in src,
        "the record reports the source-target alignment count")

    # ---- 1. warmup: one EV.main() per configuration ------------------------
    chk(BENCH_WARMUP_BATCHES == 3, "benchmark warmup_batches == 3",
        str(BENCH_WARMUP_BATCHES))
    chk(FORMAL_WARMUP_BATCHES == 0, "formal warmup_batches default == 0",
        str(FORMAL_WARMUP_BATCHES))
    chk(FORMAL_TIMING_CALLBACK is None, "formal timing_callback default is None")
    ev_main_calls = [n for n in ast.walk(tree)
                     if isinstance(n, ast.Call)
                     and isinstance(n.func, ast.Attribute)
                     and n.func.attr == "main"
                     and isinstance(n.func.value, ast.Name)
                     and n.func.value.id == "EV"]
    chk(len(ev_main_calls) == 1,
        "exactly ONE EV.main() CALL SITE in this file (AST)", str(len(ev_main_calls)))
    run_once_fn = next(n for n in tree.body
                       if isinstance(n, ast.FunctionDef) and n.name == "run_once")
    run_once_src = ast.get_source_segment(src, run_once_fn)
    kw = {k.arg for c in ev_main_calls for k in c.keywords}
    chk({"warmup_batches", "timing_callback", "num_beams", "batch_size"} <= kw,
        "run_once() forwards warmup_batches / timing_callback / beam / batch")
    callers = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)
               and n.name != "run_once"
               and any(isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
                       and c.func.attr == "main"
                       and isinstance(c.func.value, ast.Name) and c.func.value.id == "EV"
                       for c in ast.walk(n))]
    chk(callers == [], "only run_once() invokes EV.main()", str(callers))

    # ---- 2. timing contract ------------------------------------------------
    ev_src = open(repo("evaluate.py"), encoding="utf-8").read()
    syn = ev_src.index("torch.cuda.synchronize()", ev_src.index("timing hook"))
    t0 = ev_src.index("_t0 = time.perf_counter()")
    gen = ev_src.index("generation_output = model.generate(")
    syn2 = ev_src.index("torch.cuda.synchronize()", gen)
    cb = ev_src.index("on_generate(len(encodings)")
    chk(syn < t0 < gen < syn2 < cb,
        "evaluate.py: sync -> t0 -> generate -> sync -> callback",
        f"{syn} < {t0} < {gen} < {syn2} < {cb}")
    chk("if on_generate is not None and torch.cuda.is_available()" in ev_src,
        "the pre-generate synchronize is guarded by the callback being present")

    # ---- 3. peak VRAM: measured-window max_memory_allocated ----------------
    chk("peak_alloc_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)" in src,
        "peak_vram_gb comes from torch.cuda.max_memory_allocated()")
    chk('"peak_vram_gb": peak_alloc_gb' in src,
        "peak_vram_gb is assigned from that value")
    chk(ev_src.index("reset_peak_memory_stats")
        < ev_src.index("for idx, encodings in enumerate(tqdm(new_encodings))"),
        "evaluate.py resets peak stats AFTER warmup and BEFORE the measured loop")
    warm_i = ev_src.index("warmup_batches and warmup_batches > 0")
    reset_i = ev_src.index("reset_peak_memory_stats")
    meas_i = ev_src.index("for idx, encodings in enumerate(tqdm(new_encodings))")
    chk(warm_i < reset_i < meas_i,
        "order is warmup -> reset_peak_memory_stats -> measured loop")
    chk("peak_vram_gb" in SEMANTICS and "meAsured" not in SEMANTICS["peak_vram_gb"],
        "peak_vram_gb semantics documented")
    chk("MEASURED GENERATION" in SEMANTICS["peak_vram_gb"]
        and "NOT total" in SEMANTICS["peak_vram_gb"],
        "peak_vram_gb is documented as measured-window, not device total")
    chk("peak_device_used_gb" in SEMANTICS
        and "peak_device_used_gb" != "peak_vram_gb",
        "device-level sampler is reported under a SEPARATE name")
    chk('"peak_device_used_gb": (peak_device_mib / 1024.0)' in src,
        "device-level value is not written into peak_vram_gb")
    chk("reset_peak_memory_stats" not in run_once_src,
        "the benchmark does not reset peak stats itself (evaluate.py owns ordering)")

    # ---- 4. warmup cannot contaminate measurements -------------------------
    chk("on_generate=None" in ev_src and 'desc="warmup (discarded)"' in ev_src,
        "evaluate.py warmup batch passes on_generate=None and is discarded")
    chk(warm_i < meas_i, "warmup block precedes the measured loop")
    chk("if warmup_batches and warmup_batches > 0:" in ev_src,
        "warmup is skipped entirely when warmup_batches == 0")

    # ---- 5. incremental atomic persistence ---------------------------------
    chk("write_results" in call_names, "write_results is called")
    chk("os.replace(tmp, path)" in src, "atomic write uses os.replace(tmp, path)")
    chk("os.fsync" in src, "atomic write fsyncs before replace")
    chk(".tmp." in src, "temp file is written in the same directory as the target")
    main_fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                   and n.name == "main")
    main_src = ast.get_source_segment(src, main_fn)
    chk(main_src.count("write_results(") >= 3,
        "main() persists once per configuration + on failure + at the end",
        f"{main_src.count('write_results(')} call sites")
    chk("except Exception" in main_src and "status" in main_src,
        "main() catches configuration failures and records status")
    chk('"status": "ok"' in src, "successful configurations are marked status=ok")
    chk('"status": "FAILED"' in src, "failed configurations are marked status=FAILED")
    chk("break" in main_src, "a failed configuration stops the remaining ones")

    # ---- 6. batch invariance compares semantics ----------------------------
    bi = next((n for n in tree.body
               if isinstance(n, ast.FunctionDef) and n.name == "batch_invariance"), None)
    chk(bi is not None, "batch_invariance() exists")
    if bi is not None:
        bsrc = ast.get_source_segment(src, bi)
        chk('x["output"].strip() != y["output"].strip()' in bsrc,
            "invariance compares output.strip() per row")
        chk('x["predict"] != y["predict"]' in bsrc,
            "invariance compares the predict list per row")
        chk("prediction_exact_match_rate" in bsrc, "invariance reports match rate")
        chk("mismatch_rows" in bsrc, "invariance reports mismatch row count")
        bi_bare = ast.unparse(ast.parse(
            "".join(l for l in bsrc.splitlines(keepends=True)
                    if not l.strip().startswith("#"))))
        bi_calls = {n.func.attr for n in ast.walk(ast.parse(bi_bare))
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        chk("md5" not in bi_calls and "sha256" not in bi_calls
            and "hashlib" not in bi_bare,
            "invariance performs no file-digest call (semantic compare only)")
        chk("score_predictions(" in bsrc,
            "invariance uses the SAME calc.py scoring path as the formal runs")
        chk("filter_codebook_to_targets" not in bsrc,
            "invariance no longer scores against a shrunk codebook")
        chk("subset_metrics" in bsrc, "invariance reports per-batch HR/NDCG")
    chk(BATCH_INVARIANCE_BASELINE == 1, "invariance baseline is batch_size 1")
    chk(set(EFFICIENCY_BATCHES) >= {1, 8, 32},
        "both batch1-vs-batch8 and batch1-vs-batch32 are covered",
        str(EFFICIENCY_BATCHES))
    chk('"status": "DIVERGENT"' in src and "does not fail the benchmark" in src,
        "prediction divergence only warns")

    # ---- 7. unique targets <= rows (the sanity property that was violated) --
    try:
        tgt = read_test_targets(EFFICIENCY_SUBSET)
        uniq = len({t for t in tgt if t})
        chk(uniq <= len(tgt),
            "unique target count <= subset row count",
            f"{uniq} <= {len(tgt)}")
        chk(len(tgt) == EFFICIENCY_SUBSET,
            "read_test_targets returns exactly the requested row count",
            f"{len(tgt)}")
    except Exception as e:
        chk(False, "read_test_targets works", f"{type(e).__name__}: {e}")

    # ---- 8. cutoff rule ----------------------------------------------------
    chk(available_cutoffs(5) == (1, 3, 5), "beam=5 -> cutoffs (1,3,5)")
    chk(available_cutoffs(10) == (1, 3, 5, 10), "beam=10 -> cutoffs (1,3,5,10)")
    chk(available_cutoffs(20) == (1, 3, 5, 10, 20), "beam=20 -> all cutoffs")
    chk(20 not in available_cutoffs(5) and 20 not in available_cutoffs(10),
        "HR@20/NDCG@20 are NOT reported when beam < 20")
    chk(not any(c > b for b in QUALITY_SWEEP for c in QUALITY_SWEEP[b]),
        "no sweep entry requests a cutoff above its beam")
    chk(all(c == available_cutoffs(b) for b, c in QUALITY_SWEEP.items()),
        "declared sweep table equals the computed cutoff rule")
    chk(EFFICIENCY_BEAM == FORMAL_BEAM, "efficiency beam == formal beam 20")
    chk(EFFICIENCY_SUBSET % max(EFFICIENCY_BATCHES) == 0,
        "efficiency subset divides evenly by every batch size",
        f"{EFFICIENCY_SUBSET} % {max(EFFICIENCY_BATCHES)} == 0")

    # ---- 9. hard-fail vs warn boundaries -----------------------------------
    chk("LegalRate != 1" in src, "LegalRate != 1 is a hard failure")
    chk("DuplicateRate != 0" in src, "DuplicateRate != 0 is a hard failure")
    chk("sample count" in src, "sample-count mismatch is a hard failure")
    chk("misalignment" in src, "source-target misalignment is a hard failure")
    chk("integrity check failed" in src, "integrity problems raise (fail-loud)")

    # ---- 10. protocol identity --------------------------------------------
    for forbidden in ("ConstrainedLogitsProcessor", "GenerationConfig",
                      "LogitsProcessorList", "EvalSidDataset", "hash_dict",
                      "get_hash"):
        chk(forbidden not in call_names and forbidden not in literals,
            f"benchmark does not re-implement {forbidden}")
    chk("main" in call_names and "EV.main(" in src,
        "the benchmark calls evaluate.main() directly (single protocol)")
    chk("import evaluate as EV" in src, "evaluate is imported, not copied")
    chk("import_evaluate()" in src and "EV, err = import_evaluate()" in src,
        "run_once() resolves evaluate through import_evaluate(), not a bare import")
    chk("num_beams=beam" in src and "batch_size=batch_size" in src,
        "all protocol knobs are forwarded through evaluate.main()'s own parameters")
    ev_main = next((n for n in ast.walk(ast.parse(ev_src))
                    if isinstance(n, ast.FunctionDef) and n.name == "main"), None)
    if ev_main is not None:
        args = [a.arg for a in ev_main.args.args]
        defaults = [ast.unparse(d) for d in ev_main.args.defaults]
        chk(args[-2:] == ["timing_callback", "warmup_batches"]
            and defaults[-2:] == ["None", "0"],
            "evaluate.py defaults are timing_callback=None, warmup_batches=0",
            f"{args[-2:]} = {defaults[-2:]}")
    chk("ConstrainedLogitsProcessor" in ev_src and "logits_processor" in ev_src,
        "evaluate.py still builds its own constrained logits processor")

    # ---- 7. efficiency subset is a REAL workload, not just an expected count --
    # Previous revision only declared EFFICIENCY_SUBSET = 512 and still handed
    # the ORIGINAL test CSV to evaluate.main(), so the efficiency arm evaluated
    # 4533 rows and the integrity gate failed. These checks make that class of
    # bug impossible to reintroduce silently.
    chk("efficiency_test = write_efficiency_subset(" in src,
        "main() materialises the efficiency subset before running")
    chk("efficiency_test, EFFICIENCY_SUBSET))" in src,
        "efficiency plan entries carry the subset path AND its row count")
    chk("TEST, None))" in src,
        "quality plan entries carry the ORIGINAL test path")
    run_src = ast.get_source_segment(src, run_once_fn)
    chk("test_data_path=test_data_path" in run_src,
        "run_once() hands its test_data_path straight to evaluate.main()")
    chk("read_targets_from(test_data_path" in run_src,
        "alignment targets are read from the SAME path that was evaluated")
    chk('"test_data_path": os.path.relpath(test_data_path' in src,
        "each record records the test_data_path it actually used")
    chk(EFFICIENCY_SUBSET == 512, "EFFICIENCY_SUBSET == 512", str(EFFICIENCY_SUBSET))
    chk(EFFICIENCY_SUBSET_REL.endswith(EFFICIENCY_SUBSET_NAME),
        "subset artifact name derives from EFFICIENCY_SUBSET", EFFICIENCY_SUBSET_REL)
    chk("efficiency_subset" in src and '"sha256"' in src,
        "results.json records the efficiency_subset provenance block")

    # CPU-only materialisation self-check: actually build the subset in a temp
    # dir and confirm the resulting FILE has 512 data rows identical to the
    # original's first 512, with an identical header.
    _selfcheck_dir = None
    try:
        import shutil
        # A deterministic path next to this file. mkdtemp() would give a random
        # name, and some sandboxed filesystems refuse to create subdirectories
        # inside a directory they did not see created explicitly -- which would
        # make this self-check fail for reasons unrelated to the benchmark.
        _selfcheck_dir = os.path.join(_ANALYSIS_DIR, ".bench_subset_check")
        shutil.rmtree(_selfcheck_dir, ignore_errors=True)
        shutil.rmtree(_selfcheck_dir, ignore_errors=True)
        sp = write_efficiency_subset(_selfcheck_dir)
        import csv as _csv
        with open(sp, encoding="utf-8", newline="") as f:
            sub = list(_csv.reader(f))
        with open(TEST, encoding="utf-8", newline="") as f:
            org = list(_csv.reader(f))
        chk(os.path.basename(sp) == EFFICIENCY_SUBSET_NAME,
            "materialised subset filename is the declared one",
            os.path.basename(sp))
        chk(len(sub) - 1 == EFFICIENCY_SUBSET,
            "materialised subset file has exactly 512 data rows",
            f"{len(sub) - 1} data rows")
        chk(sub[0] == org[0],
            "materialised subset header matches the original test header",
            f"{len(sub[0])} columns")
        chk(sub[1:] == org[1:EFFICIENCY_SUBSET + 1],
            "subset rows are byte-identical to the original's first 512 rows")
        chk(len(read_targets_from(sp)) == EFFICIENCY_SUBSET,
            "read_targets_from(subset) returns 512 targets",
            str(len(read_targets_from(sp))))
        chk(len(read_targets_from(TEST)) == len(org) - 1,
            "read_targets_from(ORIGINAL TEST) returns every row",
            str(len(read_targets_from(TEST))))
        shutil.rmtree(_selfcheck_dir, ignore_errors=True)
        chk(not os.path.exists(_selfcheck_dir)
            or not os.listdir(_selfcheck_dir),
            "self-check cleaned up after itself (no artifact left behind)")
    except Exception as e:                                    # noqa: BLE001
        chk(False, "efficiency subset materialisation self-check",
            f"{type(e).__name__}: {e}")

    # ---- 11. inputs / outputs ---------------------------------------------
    chk(os.path.exists(INFO), "ORIGINAL INFO exists", INFO)
    chk(os.path.exists(TEST), "original test split exists", TEST)
    chk(os.path.exists(repo("calc.py")), "calc.py exists")
    chk(os.path.exists(repo("evaluate.py")), "evaluate.py exists")
    chk(os.path.abspath(OUT_DIR).startswith(os.path.abspath(RUN_ROOT)),
        "output path is inside RUN_ROOT", OUT_DIR)
    chk(not any("analysis/results" in s for s in literals),
        "no string literal references analysis/results/ (committed provenance untouched)")
    chk(not os.path.exists(OUT_DIR),
        "benchmark output dir does not exist yet (will not overwrite)", OUT_DIR)
    chk("exist_ok=False" in src, "per-configuration dir is created fail-loud")

    # ---- 11b. RUNTIME import of the protocol under test --------------------
    # Not a text assertion this time: this actually imports evaluate.py through
    # the benchmark's own configured sys.path. It cannot pass by accident from a
    # convenient cwd, because the path entry it relies on is the one set up at
    # the top of THIS file. evaluate.py only defines helpers and main() at import
    # time -- no model is loaded, no CUDA call is made, nothing is generated.
    chk(PROJECT_ROOT in sys.path, "repo root is on sys.path (protocol importable)",
        PROJECT_ROOT)
    chk(_ANALYSIS_DIR in sys.path, "analysis/ (own dir) is on sys.path", _ANALYSIS_DIR)
    ev_mod, ev_err = import_evaluate()
    chk(ev_mod is not None,
        "runtime import evaluate succeeds from benchmark execution context",
        "imported from " + str(getattr(ev_mod, "__file__", "?")) if ev_mod
        else f"FAILED: {ev_err}")
    chk(ev_mod is not None and os.path.abspath(ev_mod.__file__) ==
        os.path.abspath(repo("evaluate.py")),
        "the imported evaluate is the repository's own evaluate.py",
        os.path.abspath(repo("evaluate.py")))
    chk(ev_mod is not None and callable(getattr(ev_mod, "main", None)),
        "imported evaluate exposes a callable main()")
    chk(ev_mod is None or not hasattr(ev_mod, "__version__"),
        "importing evaluate loads no model (no version attribute)")

    # ---- 12. recorded measurement semantics --------------------------------
    for key in ("latency_p50_ms", "request_latency_p50_ms", "generation_seconds",
                "peak_vram_gb", "peak_device_used_gb", "hr_ndcg", "warmup"):
        chk(key in SEMANTICS, f"measurement semantics documented for {key}")
    chk("excludes model" in SEMANTICS["generation_seconds"],
        "generation_seconds semantics state the exclusions")
    chk("never participate" in SEMANTICS["hr_ndcg"]
        or "never" in SEMANTICS["hr_ndcg"],
        "hr_ndcg semantics state that the item universe does not affect the metric")

    model_ok = os.path.exists(os.path.join(MAIN_MODEL, "model.safetensors"))

    print("=" * 100)
    print("STATIC PREFLIGHT: inference quality-latency benchmark")
    print(f"  PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"  RUN_ROOT     = {RUN_ROOT}")
    print(f"  CATEGORY     = {CATEGORY}")
    print(f"  MAIN_MODEL   = {MAIN_MODEL}")
    print("=" * 100)
    n_ok = 0
    for ok, label, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  -- {detail}" if detail else ""))
        n_ok += ok
    n_fail = len(checks) - n_ok

    print()
    if model_ok:
        print("  [ OK ] main model checkpoint present")
    else:
        print("  [WARN] main model checkpoint MISSING (informational, not a check failure):")
        print(f"         {MAIN_MODEL}")
        print("         the benchmark targets the FORMAL clean SFT run; it must be")
        print("         restored before --mode quality/efficiency can execute.")
        print("         No substitute model is used.")
    print()
    print("=" * 100)
    print(f"RESULT: {n_ok} PASS, {n_fail} FAIL"
          + ("" if model_ok else "   (+1 informational WARN: main model missing)"))
    print("=" * 100)
    return 0 if n_fail == 0 else 1


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--mode", choices=("quality", "efficiency", "all"), default="all")
    ap.add_argument("--preflight", action="store_true",
                    help="run static CPU-only checks and exit")
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()

    # preflight is pure static analysis: it must NOT create the output dir.
    if args.preflight:
        return preflight()

    if os.path.exists(args.out_dir):
        print(f"REFUSE: output dir already exists: {args.out_dir}", file=sys.stderr)
        print("        move it aside or pass --out-dir", file=sys.stderr)
        return 2
    os.makedirs(args.out_dir)

    # ------------------------------------------------------------------ subset
    # Materialise the efficiency workload BEFORE any run, so the file the
    # efficiency configurations evaluate is a recorded provenance artifact and
    # its hash is in results.json from the very first write.
    global EFFICIENCY_PROVENANCE
    efficiency_test = None
    if args.mode in ("efficiency", "all"):
        efficiency_test = write_efficiency_subset(args.out_dir)
        EFFICIENCY_PROVENANCE = efficiency_subset_provenance(efficiency_test)
        print(f"  efficiency workload: {EFFICIENCY_PROVENANCE['path']}  "
              f"rows={EFFICIENCY_PROVENANCE['rows']}  "
              f"sha256={EFFICIENCY_PROVENANCE['sha256'][:16]}...")

    records = []
    failed = None

    plan = []
    if args.mode in ("quality", "all"):
        for beam, cut in QUALITY_SWEEP.items():
            plan.append(("quality", beam, FORMAL_BATCH_SIZE, 4533,
                         f"quality_beam{beam}_batch{FORMAL_BATCH_SIZE}", cut, True,
                         TEST, None))
    if args.mode in ("efficiency", "all"):
        for bs in EFFICIENCY_BATCHES:
            plan.append(("efficiency", EFFICIENCY_BEAM, bs, EFFICIENCY_SUBSET,
                         f"efficiency_beam{EFFICIENCY_BEAM}_batch{bs}", None, False,
                         efficiency_test, EFFICIENCY_SUBSET))

    for mode, beam, bs, n, tag, cut, do_quality, tpath, tgt_n in plan:
        print(f"\n=== [{mode}] {tag}  beam={beam} batch={bs} samples={n}"
              + (f"  cutoffs={list(cut)}" if cut else ""))
        print(f"    test_data_path = {os.path.relpath(tpath, PROJECT_ROOT)}")
        try:
            rec = run_once(mode=mode, beam=beam, batch_size=bs, num_samples=n,
                           out_root=args.out_dir, tag=tag, do_quality=do_quality,
                           test_data_path=tpath, expected_targets_n=tgt_n)
        except Exception as e:
            rec = {"mode": mode, "tag": tag, "status": "FAILED",
                   "beam": beam, "batch_size": bs, "num_samples": n,
                   "error": f"{type(e).__name__}: {e}"}
            records.append(rec)
            write_results(records, args.out_dir)
            failed = rec
            print(f"    FAILED: {rec['error']}", file=sys.stderr)
            print("    stopping: remaining configurations will not run.",
                  file=sys.stderr)
            break

        print("    " + json.dumps({k: v for k, v in rec.items()
                                   if k in ("HR@1", "HR@3", "HR@5", "HR@10", "HR@20",
                                            "NDCG@1", "NDCG@3", "NDCG@5", "NDCG@10",
                                            "NDCG@20", "source_target_alignment",
                                            "generation_seconds", "samples_per_second",
                                            "latency_p50_ms", "latency_p95_ms",
                                            "peak_vram_gb", "legal_rate",
                                            "duplicate_rate")}))

        if beam == FORMAL_BEAM and bs == FORMAL_BATCH_SIZE and do_quality:
            rec["reference_match"] = all(
                abs(rec.get(k, float("nan")) - v) < REFERENCE_TOL
                for k, v in REFERENCE.items())
            print(f"    reference check vs formal clean SFT: "
                  f"{'MATCH' if rec['reference_match'] else 'MISMATCH'}  "
                  f"(HR@20 {rec.get('HR@20')} vs {REFERENCE['HR@20']}, "
                  f"NDCG@20 {rec.get('NDCG@20')} vs {REFERENCE['NDCG@20']})")

        records.append(rec)
        write_results(records, args.out_dir)

    if args.mode in ("efficiency", "all") and not failed:
        print("\n=== batch-invariance diagnostic (same subset, semantic compare) ===")
        try:
            bi = batch_invariance(records)
        except Exception as e:
            bi = {"available": False, "reason": f"{type(e).__name__}: {e}"}
        for k, v in bi.get("comparisons", {}).items():
            mark = {"MATCH": "OK  ", "DIVERGENT": "WARN", "FAIL": "FAIL"}.get(v["status"], "?")
            print(f"  [{mark}] {k}: match_rate={v.get('prediction_exact_match_rate')} "
                  f"mismatch_rows={v.get('mismatch_rows')} {v.get('reason','')}")
            if v["status"] == "DIVERGENT":
                print("         prediction differs across batch sizes -- WARNING only, "
                      "does not fail the benchmark")
        for k, v in bi.get("subset_metrics", {}).items():
            print(f"  subset metrics {k}: HR={v['HR']}  NDCG={v['NDCG']}")
        if bi.get("note"):
            print(f"  note: {bi['note']}")
        records.append({"mode": "diagnostic", "tag": "batch_invariance",
                        "status": "ok", "batch_invariance": bi})
        write_results(records, args.out_dir)

    print(f"\n[save] {os.path.join(args.out_dir, 'results.json')}")
    print(f"[save] {os.path.join(args.out_dir, 'results.csv')}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
