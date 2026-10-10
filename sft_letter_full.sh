#!/bin/bash
# =============================================================================
# Stage 3 -- formal LETTER-SID Qwen2.5-0.5B full-parameter SFT.
#
# Recipe is a STRICT COPY of the P0 Clean SFT launcher (sft_full.sh):
#   num_epochs 2, batch_size 64, micro_batch_size 16 (= grad accum 4),
#   learning_rate 3e-4, cutoff_len 512, seed 42, no gradient checkpointing,
#   bf16 + AdamW torch + linear scheduler + warmup_steps 20 (all set inside sft.py).
#
# Only the SID representation, the index, and the output directory differ from P0.
# The base model is the SAME pristine Qwen2.5-0.5B checkpoint P0 started from --
# deliberately NOT runs/industrial_sft/final_checkpoint.
# =============================================================================
set -o pipefail

export WANDB_MODE=disabled
export NCCL_IB_DISABLE=1
export HF_ENDPOINT=https://hf-mirror.com

# Stage 3: protobuf was upgraded 3.x -> 6.33.6 when k-means-constrained was
# installed for the LETTER tokenizer. tensorboard 2.11.2 -- imported by
# transformers' TensorBoardCallback -- requires protobuf < 4 and otherwise raises
#     TypeError: Descriptors cannot be created directly.
# This is protobuf's own documented workaround #2 (pure-Python implementation).
# Neither a source file nor a package is modified.
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Stage 3: wandb also reaches tensorboard; every formal run sets WANDB_MODE=disabled,
# so an inert wandb stub is injected via sitecustomize.py. Reversible: drop the dir.
export PYTHONPATH=/tmp/s3/guard:${PYTHONPATH:-}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

D=/root/autodl-tmp/code
BASE_MODEL=/root/autodl-tmp/models/Qwen2.5-0.5B
CATEGORY=Industrial_and_Scientific
STEM=${CATEGORY}_5_2016-10-2018-11
LD=$D/artifacts/letter_stage3/data
OUT=$D/runs/letter_sid_qwen05b_sft
mkdir -p "$OUT"

echo "=== base model : $BASE_MODEL"
echo "=== train file : $LD/train/$STEM.csv"
echo "=== valid file : $LD/valid/$STEM.csv"
echo "=== index file : $LD/index/$CATEGORY.index.json"
echo "=== item meta  : $LD/index/$CATEGORY.item.json"
echo "=== output dir : $OUT"
echo "=== start      : $(date '+%Y-%m-%d %H:%M:%S')"

date +%s > /tmp/s3/sft_t0.txt

torchrun --nproc_per_node 1 \
    sft.py \
    --base_model       "$BASE_MODEL" \
    --train_file       "$LD/train/$STEM.csv" \
    --eval_file        "$LD/valid/$STEM.csv" \
    --output_dir       "$OUT" \
    --category         "$CATEGORY" \
    --sid_index_path   "$LD/index/$CATEGORY.index.json" \
    --item_meta_path   "$LD/index/$CATEGORY.item.json" \
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

echo "=== exit       : $?"
echo "=== end        : $(date '+%Y-%m-%d %H:%M:%S')"
