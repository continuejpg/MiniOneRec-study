#!/bin/bash
# =============================================================================
# sft_seq_only_shuffled.sh -- sequence-only SFT, SHUFFLED-SID arm.
#
# Same sequence-only objective as sft_seq_only_clean.sh:
#   trains ONLY SidSFTDataset:  history_item_sid -> item_sid
#   removes EXPLICIT TEXT supervision:
#     - SidItemFeatDataset  (title <-> SID)
#     - FusionSeqRecDataset (history SID -> target title)
#
# The ONLY difference vs the Clean arm is the item<->SID assignment:
#   Clean    : original index.json / train.csv / valid.csv
#   Shuffled : strict popularity-stratified shuffled index / train / valid
#             (generated deterministically by analysis/build_shuffled_sid_strict.py, seed 42)
#
# Exactly FOUR arguments differ: --train_file, --eval_file, --sid_index_path,
# --output_dir. Everything else is identical.
#
# Does NOT start training by itself -- execute it explicitly.
# =============================================================================
set -euo pipefail

export WANDB_MODE=disabled
export NCCL_IB_DISABLE=1
export HF_ENDPOINT=https://hf-mirror.com

# portable paths: PROJECT_ROOT / RUN_ROOT / DATA_ROOT / CATEGORY / PY
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# this stage needs the upstream Qwen2.5-0.5B weights -- deliberately no default
: "${BASE_MODEL:?set BASE_MODEL to the Qwen2.5-0.5B model path}"

# ---- intervention (shuffled) paths -- the ONLY recipe difference -------------
SHUF_DIR="$PROJECT_ROOT/analysis/shuffled_sid"
SHUF_INDEX="$SHUF_DIR/${CATEGORY}.index.json"
SHUF_TRAIN="$SHUF_DIR/train.csv"
SHUF_VALID="$SHUF_DIR/valid.csv"

OUT="$RUN_ROOT/seq_only_shuffled_sft"

# must not collide with the full-recipe runs
for forbidden in "$RUN_ROOT/industrial_sft" "$RUN_ROOT/industrial_sft_shuffled_sid"; do
    if [ "$(cd "$(dirname "$OUT")" && pwd)/$(basename "$OUT")" = \
         "$(cd "$(dirname "$forbidden")" 2>/dev/null && pwd)/$(basename "$forbidden")" ]; then
        echo "REFUSE: output_dir would collide with $forbidden" >&2
        exit 1
    fi
done

# ---- preflight (no GPU work) -------------------------------------------------
echo "=== preflight ==="
fail=0
check_exists() {
    if [ -e "$1" ]; then printf '  OK      %s\n' "$1"; else printf '  MISSING %s\n' "$1"; fail=1; fi
}
check_exists "$BASE_MODEL"
check_exists "$ITEM_META"
check_exists "$SHUF_INDEX"
check_exists "$SHUF_TRAIN"
check_exists "$SHUF_VALID"
check_exists "$PROJECT_ROOT/sft.py"

# TokenExtender (sft.py:31-39) rebuilds the filename from
# basename(--sid_index_path).split('.')[0]; the stem must be "<Category>".
if [ "$(basename "$INDEX")" = "${CATEGORY}.index.json" ]; then
    echo "  OK      original index basename is ${CATEGORY}.index.json"
else
    echo "  REFUSE  original index basename is $(basename "$INDEX")"; fail=1
fi
if [ "$(basename "$SHUF_INDEX")" = "${CATEGORY}.index.json" ]; then
    echo "  OK      shuffled index basename is ${CATEGORY}.index.json"
else
    echo "  REFUSE  shuffled index basename is $(basename "$SHUF_INDEX"), expected ${CATEGORY}.index.json"
    fail=1
fi

if [ -e "$OUT/final_checkpoint/model.safetensors" ]; then
    echo "  REFUSE  output dir already holds a finished run: $OUT"; fail=1
else
    echo "  OK      output dir is free: $OUT"
fi

if [ "$fail" -ne 0 ]; then
    echo "=== preflight FAILED -- not starting training ==="
    exit 1
fi
echo "=== preflight OK ==="
echo

# -----------------------------------------------------------------------------
# Training -- seq_only SHUFFLED arm
# -----------------------------------------------------------------------------
mkdir -p "$OUT"

torchrun --nproc_per_node 1 \
    sft.py \
    --base_model       "$BASE_MODEL" \
    --train_file       "$SHUF_TRAIN" \
    --eval_file        "$SHUF_VALID" \
    --output_dir       "$OUT" \
    --category         "$CATEGORY" \
    --sid_index_path   "$SHUF_INDEX" \
    --item_meta_path   "$ITEM_META" \
    --sft_mode         seq_only \
    --sample           -1 \
    --num_epochs       2 \
    --batch_size       64 \
    --micro_batch_size 16 \
    --learning_rate    3e-4 \
    --cutoff_len       512 \
    --seed             42 \
    --train_from_scratch False \
    --freeze_LLM       False \
    2>&1 | tee "$OUT/train.log"
