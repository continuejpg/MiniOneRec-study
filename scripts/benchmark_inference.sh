#!/bin/bash
# =============================================================================
# scripts/benchmark_inference.sh -- inference quality-latency benchmark
#
# Measures the quality-vs-cost trade-off of the generative recommender across
# beam width and batch size, on the FORMAL clean SFT model.
#
#   quality sweep    : beam in {5,10,20,50}, batch 8, full test (4533)
#   efficiency sweep : beam 20, batch in {1,8,32}, first 512 test rows
#
# Protocol identity: the benchmark calls evaluate.py's OWN main(), so prompts,
# SID parsing, the constrained-decoding trie (ORIGINAL INFO) and the logits
# processor are exactly the formal ones. Nothing is re-implemented.
#
# Output: $RUN_ROOT/inference_benchmark/{results.json,results.csv,...}
#         Refuses to run if that directory already exists (never overwrites).
#
# Does NOT train, does NOT touch the formal eval outputs, does NOT modify
# analysis/results/.
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# main benchmark model: the formal clean SFT checkpoint
MAIN_MODEL="$RUN_ROOT/industrial_sft/final_checkpoint"
OUT="$RUN_ROOT/inference_benchmark"

MODE="${MODE:-all}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"

echo "=============================================================================="
echo " Inference quality-latency benchmark"
echo "   PROJECT_ROOT = $PROJECT_ROOT"
echo "   RUN_ROOT     = $RUN_ROOT"
echo "   CATEGORY     = $CATEGORY"
echo "   MAIN_MODEL   = $MAIN_MODEL"
echo "   OUT          = $OUT"
echo "   MODE         = $MODE"
echo "=============================================================================="
echo

# -----------------------------------------------------------------------------
# read-only preflight (no GPU work)
# -----------------------------------------------------------------------------
fail=0
chk() { if [ -e "$1" ]; then printf '  OK      %s\n' "$1"; else printf '  MISSING %s\n' "$1"; fail=1; fi; }
refuse() { printf '  REFUSE  %s\n' "$1"; fail=1; }

echo "=== preflight ==="
chk "$PROJECT_ROOT/evaluate.py"
chk "$PROJECT_ROOT/calc.py"
chk "$PROJECT_ROOT/analysis/benchmark_inference.py"
chk "$INFO"
chk "$TEST"
chk "$ITEM_META"

# the main benchmark model MUST be the formal clean SFT run
if [ -d "$MAIN_MODEL" ]; then
    chk "$MAIN_MODEL/config.json"
    chk "$MAIN_MODEL/model.safetensors"
else
    echo "  MISSING $MAIN_MODEL"
    echo "          the benchmark targets the FORMAL clean SFT checkpoint;"
    echo "          it must be restored before this can run. Refusing to"
    echo "          substitute GRPO or shuffled models."
    fail=1
fi

# output must not exist: never silently overwrite a previous benchmark
if [ -e "$OUT" ]; then
    refuse "output dir already exists: $OUT (move it aside before rerunning)"
else
    echo "  OK      output dir is free: $OUT"
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    refuse "nvidia-smi not found -- no GPU on this host"
fi

# -----------------------------------------------------------------------------
# Correctness / provenance gate -- delegated to the benchmark's own --preflight
#
# This used to be a bare `grep -q 'analysis/results'` over the benchmark source.
# That was a false positive: the benchmark's preflight ASSERTS that it never
# references analysis/results/, and the assertion text itself contains the
# substring, so the grep matched the guard rather than a violation and the
# committed launcher refused to run at all.
#
# It is NOT replaced by another string grep. The check now lives where it can be
# done properly: analysis/benchmark_inference.py --preflight walks its own AST
# and asserts, among 99 checks, that no string literal references
# analysis/results/, that the protocol is not re-implemented, and that the
# cutoff / timing / VRAM / alignment rules hold. The shell only requires that
# this preflight exits 0 AND reports zero failures.
# -----------------------------------------------------------------------------
echo
echo "--- benchmark static preflight (AST-based correctness gate) ---"
preflight_log="$(mktemp)"
"$PY" "$PROJECT_ROOT/analysis/benchmark_inference.py" --preflight 2>&1 | tee "$preflight_log"
pf_rc=${PIPESTATUS[0]}
echo
if [ "$pf_rc" -ne 0 ]; then
    echo "  REFUSE  benchmark --preflight exited $pf_rc (expected 0)"
    fail=1
elif ! grep -qE '^RESULT: [0-9]+ PASS, 0 FAIL' "$preflight_log"; then
    echo "  REFUSE  benchmark --preflight did not report '0 FAIL'"
    fail=1
else
    echo "  OK      benchmark --preflight: $(grep -E '^RESULT:' "$preflight_log")"
fi
rm -f "$preflight_log"

if [ "$fail" -ne 0 ]; then
    echo "=== preflight FAILED -- not starting benchmark ==="
    exit 1
fi
echo
echo "=== preflight OK ==="
echo

if [ "$PREFLIGHT_ONLY" = "1" ]; then
    # the preflight above has already run and its result is the exit condition
    echo "PREFLIGHT_ONLY=1 -- static checks done, stopping before the benchmark."
    exit 0
fi

# -----------------------------------------------------------------------------
# benchmark
# -----------------------------------------------------------------------------
"$PY" -u "$PROJECT_ROOT/analysis/benchmark_inference.py" --mode "$MODE" --out-dir "$OUT"
rc=$?

echo
echo "=== done (exit $rc) ==="
if [ -f "$OUT/results.csv" ]; then
    echo "--- results.csv ---"
    cat "$OUT/results.csv"
fi
exit $rc
