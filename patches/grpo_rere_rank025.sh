#!/bin/bash
# =====================================================================
# grpo_rere_rank025.sh -- R1: ReRe group-wise ranking reward, 0.25 epoch.
#
# STRICT COPY of the optimized 0.25-epoch baseline (patches/grpo_fast025.sh).
# The ONLY differences are:
#     --reward_type  ranking  ->  rere_rank
#     --output_dir   grpo_fast025 -> grpo_rere_rank025
#
# Everything else is byte-identical so the two runs stay comparable:
#   same SFT checkpoint, same frozen 17,516-sample subsets, same seed,
#   same optimizer / LR / beta=1e-3 / G=16 / batch-accumulation,
#   same BEAM_SAMPLE rollout, same constrained trie, same 0.25 epoch.
#
# `ranking` (rule_reward + ndcg_rule_reward) is left untouched in rl.py, so the
# original baseline remains reproducible.
# =====================================================================
set -e
export WANDB_MODE=disabled
export NCCL_IB_DISABLE=1
export HF_ENDPOINT=https://hf-mirror.com

# protobuf 6.x (installed with k-means-constrained) breaks tensorboard 2.11.2,
# which transformers imports; and wandb is stubbed since WANDB_MODE=disabled.
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH=/tmp/s3/guard:${PYTHONPATH:-}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../scripts/common.sh"
CKPT="$RUN_ROOT/industrial_sft/final_checkpoint"
OUT="$RUN_ROOT/grpo_rere_rank025"
mkdir -p "$OUT"

echo "=== R1 ReRe-rank GRPO (0.25 epoch, reward_type=rere_rank) ==="
date -Is | tee "$OUT/started_at.txt"
echo "policy/ref : $CKPT"
echo "output     : $OUT"
echo "reward     : rere_rank  (ReRe Eq.7-9, group-wise)"
echo "disk before:"
df -h "$RUN_ROOT" | tail -1 | tee -a "$OUT/started_at.txt"

nohup accelerate launch --num_processes 1 \
    rl.py \
    --model_path              "$CKPT" \
    --train_file              "$TRAIN" \
    --eval_file               "$VALID" \
    --info_file               "$INFO" \
    --category                "$CATEGORY" \
    --sid_index_path          "$INDEX" \
    --item_meta_path          "$ITEM_META" \
    --subset_dir              "$SPLITS" \
    --subset_seq              grpo_seq_10k.json \
    --subset_seqtitle         grpo_seqtitle_1k.json \
    --output_dir              "$OUT" \
    --reward_type             rere_rank \
    --num_generations         16 \
    --beam_search             True \
    --train_batch_size        16 \
    --eval_batch_size         16 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs        0.25 \
    --learning_rate           1e-5 \
    --beta                    1e-3 \
    --temperature             1.0 \
    --eval_step               999999 \
    --save_steps_frac         999999 \
    --sample_train            False \
    --seqtitle_sample         10000 \
    --dynamic_sampling        False \
    --mask_all_zero           False \
    --sync_ref_model          True \
    --test_during_training    False \
    --add_gt                  False \
    --dapo                    False \
    --gspo                    False \
    > "$OUT/train.log" 2>&1 &

PID=$!
echo "PID: $PID" | tee -a "$OUT/started_at.txt"
echo "$PID" > "$OUT/pid.txt"
echo
echo "launched in background. monitor with:"
echo "  tail -f $OUT/train.log"
echo "to stop: kill \$(cat $OUT/pid.txt)"
