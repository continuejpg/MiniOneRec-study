#!/bin/bash
# =============================================================================
# eval_seq_only_clean.sh -- beam-20 evaluation of the seq_only CLEAN arm.
#
# Copies the authoritative formal protocol exactly (see scripts/eval_shuffled_sid.sh):
#   evaluate.py: batch_size 8, num_beams 20, max_new_tokens 256,
#                length_penalty 0, seed 42, K 0
#   calc.py    : the ORIGINAL metric implementation, unmodified
#
# This arm differs from its Shuffled sibling in exactly THREE paths:
#   model path, test path, output path.
# Both arms use the ORIGINAL INFO file, so the constrained decoding space
# (legal SID codebook / trie) is identical.
#
# Does NOT run by itself -- execute it explicitly.
# =============================================================================
set -euo pipefail

export WANDB_MODE=disabled
export HF_ENDPOINT=https://hf-mirror.com

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

MODEL="$RUN_ROOT/seq_only_clean_sft/final_checkpoint"
TEST_CSV="$TEST"
EVAL_INFO="$INFO"
OUT="$RUN_ROOT/eval_seq_only_clean"
RESULT="$OUT/test_beam20.json"

# ---- protocol constants (identical to the authoritative clean protocol) -----
BATCH_SIZE=8
NUM_BEAMS=20
MAX_NEW_TOKENS=256
LENGTH_PENALTY=0
SEED=42
K=0

# ---- preflight (no GPU work) -------------------------------------------------
echo "=== preflight ==="
fail=0
chk() { if [ -e "$1" ]; then printf '  OK      %s\n' "$1"; else printf '  MISSING %s\n' "$1"; fail=1; fi; }

chk "$PROJECT_ROOT/evaluate.py"
chk "$PROJECT_ROOT/calc.py"
chk "$MODEL"
chk "$MODEL/config.json"
chk "$MODEL/model.safetensors"
chk "$TEST_CSV"
chk "$EVAL_INFO"
chk "$ITEM_META"

echo "  info  = $EVAL_INFO  (ORIGINAL -- decoding space unchanged)"
if [ -e "$RESULT" ]; then
    echo "  REFUSE  output already exists: $RESULT"; fail=1
else
    echo "  OK      output dir is free: $OUT"
fi

if [ "$fail" -ne 0 ]; then
    echo "=== preflight FAILED -- not starting evaluation ==="
    exit 1
fi
echo "=== preflight OK ==="
echo

# -----------------------------------------------------------------------------
# Step 1: generate predictions -- IDENTICAL protocol to the authoritative run
# -----------------------------------------------------------------------------
mkdir -p "$OUT"

CUDA_VISIBLE_DEVICES=0 "$PY" -u "$PROJECT_ROOT/evaluate.py" \
    --base_model       "$MODEL" \
    --info_file        "$EVAL_INFO" \
    --category         "$CATEGORY" \
    --test_data_path   "$TEST_CSV" \
    --result_json_data "$RESULT" \
    --batch_size       $BATCH_SIZE \
    --K                $K \
    --seed             $SEED \
    --length_penalty   $LENGTH_PENALTY \
    --max_new_tokens   $MAX_NEW_TOKENS \
    --num_beams        $NUM_BEAMS \
    2>&1 | tee "$OUT/evaluate.log"

# -----------------------------------------------------------------------------
# Step 2: metrics -- ORIGINAL calc.py, unmodified
# -----------------------------------------------------------------------------
"$PY" "$PROJECT_ROOT/calc.py" \
    --path      "$RESULT" \
    --item_path "$EVAL_INFO" \
    2>&1 | tee "$OUT/metrics.txt"

echo
echo "=== done ==="
cat "$OUT/metrics.txt"
