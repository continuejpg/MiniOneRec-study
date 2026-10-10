#!/bin/bash
# =============================================================================
# eval_seq_only_shuffled.sh -- beam-20 evaluation of the seq_only SHUFFLED arm.
#
# Copies the authoritative formal protocol exactly (see scripts/eval_shuffled_sid.sh):
#   evaluate.py: batch_size 8, num_beams 20, max_new_tokens 256,
#                length_penalty 0, seed 42, K 0
#   calc.py    : the ORIGINAL metric implementation, unmodified
#
# IMPORTANT: the shuffled arm is evaluated with the ORIGINAL INFO file, exactly
# like the Clean arm. --info_file supplies ONLY the legal SID codebook that
# builds the constrained-decoding trie; it does not carry the item<->SID
# assignment. Using the original file therefore keeps the decoding space
# byte-identical between the two arms.
# The shuffled consistency-info file
# (data/Amazon/info/<Category>_shuffled.info.txt) is an AUDIT artifact only and
# is deliberately NOT used here.
#
# This arm differs from its Clean sibling in exactly THREE paths:
#   model path, test path, output path.
#
# Does NOT run by itself -- execute it explicitly.
# =============================================================================
set -euo pipefail

export WANDB_MODE=disabled
export HF_ENDPOINT=https://hf-mirror.com

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

SHUF_DIR="$PROJECT_ROOT/analysis/shuffled_sid"

MODEL="$RUN_ROOT/seq_only_shuffled_sft/final_checkpoint"
TEST_CSV="$SHUF_DIR/test.csv"
EVAL_INFO="$INFO"                      # ORIGINAL info -- see header note
SHUF_INFO="$DATA_ROOT/info/${CATEGORY}_shuffled.info.txt"
OUT="$RUN_ROOT/eval_seq_only_shuffled"
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
echo "  note  : shuffled consistency-info at $SHUF_INFO is audit-only, not used here"
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
