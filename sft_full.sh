#!/bin/bash
# propagate torchrun's failure code through the `| tee` pipeline below.
# Deliberately NOT set -e / set -u: this keeps the historical failure semantics
# otherwise unchanged.
set -o pipefail

export WANDB_MODE=disabled
export NCCL_IB_DISABLE=1
export HF_ENDPOINT=https://hf-mirror.com

# portable paths: PROJECT_ROOT / RUN_ROOT / DATA_ROOT / CATEGORY / PY
# override any of them from the environment or from a .env file
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/scripts/common.sh"

# this stage needs the upstream Qwen2.5-0.5B weights -- deliberately no default
: "${BASE_MODEL:?set BASE_MODEL to the Qwen2.5-0.5B model path}"

OUT="$RUN_ROOT/industrial_sft"
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
