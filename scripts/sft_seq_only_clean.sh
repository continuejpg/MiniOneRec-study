#!/bin/bash
# =============================================================================
# sft_seq_only_clean.sh -- sequence-only SFT, CLEAN arm.
#
# Sequence-only definition (see sft.py --sft_mode):
#   trains ONLY SidSFTDataset:  history_item_sid -> item_sid
#   removes EXPLICIT TEXT supervision:
#     - SidItemFeatDataset  (title <-> SID)
#     - FusionSeqRecDataset (history SID -> target title)
#   It does NOT remove all text information: the semantic IDs themselves are
#   produced by upstream RQ-VAE text quantization.
#
# This arm differs from its Shuffled sibling in EXACTLY FOUR arguments:
#   --train_file, --eval_file, --sid_index_path, --output_dir
# Everything else (model, batch, lr, epochs, seed, cutoff, precision) is identical.
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

OUT="$RUN_ROOT/seq_only_clean_sft"

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
check_exists "$TRAIN"
check_exists "$VALID"
check_exists "$PROJECT_ROOT/sft.py"

# TokenExtender (sft.py:31-39) rebuilds the filename from
# basename(--sid_index_path).split('.')[0]; the stem must be "<Category>".
if [ "$(basename "$INDEX")" = "${CATEGORY}.index.json" ]; then
    echo "  OK      index basename is ${CATEGORY}.index.json"
else
    echo "  REFUSE  index basename is $(basename "$INDEX"), expected ${CATEGORY}.index.json"
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
# Training -- seq_only CLEAN arm
# -----------------------------------------------------------------------------
mkdir -p "$OUT"

torchrun --nproc_per_node 1 \
    sft.py \
    --base_model       "$BASE_MODEL" \
    --train_file       "$TRAIN" \
    --eval_file        "$VALID" \
    --output_dir       "$OUT" \
    --category         "$CATEGORY" \
    --sid_index_path   "$INDEX" \
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
