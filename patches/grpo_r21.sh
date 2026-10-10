#!/bin/bash
# =====================================================================
# grpo_r21.sh -- R2.2 reachability-guided dual-route GRPO (0.25 epoch).
# Run from the repository root (paths come from scripts/common.sh), AFTER:
#   1) frozen subset files exist under splits/
#   2) python patches/preflight_grpo.py passes
#
# Data (frozen, identical for every future run):
#   seq_rec       : 10,000  (splits/grpo_seq_10k.json)
#   title2sid+desc:  6,516  (full, fixed)
#   seqtitle2sid  :  1,000  (splits/grpo_seqtitle_1k.json)
#   -> 17,516 samples
#
# Reward: exact-match 0/1 only (--reward_type r21_exact). NOT the R1 rank reward.
#   NORMAL (h=0, 4,818) : prompt unchanged, completion compared to GT directly
#   HARD   (h=1, 12,698): first GT SID token placed in the prompt; full SID is
#                         reconstructed as hint + sampled suffix before comparison
# Offline route cache: splits/r21_route_cache.json
#   sha256 d7a7bc40e96217e416825cc2cee29b216e3443d64fcea17b5440e72975b140c3
# Everything else is a strict copy of grpo_fast025.sh (UNCHANGED: SFT checkpoint,
# seed, G, beam_sample, KL beta, LR, batch, grad-accum, subsets, 0.25 epoch).
# =====================================================================
set -e
export WANDB_MODE=disabled
export NCCL_IB_DISABLE=1
export HF_ENDPOINT=https://hf-mirror.com

# portable paths: PROJECT_ROOT / RUN_ROOT / DATA_ROOT / CATEGORY / PY
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../scripts/common.sh"
CKPT="$RUN_ROOT/industrial_sft/final_checkpoint"
OUT="$RUN_ROOT/grpo_r21"
mkdir -p "$OUT"

echo "=== GRPO baseline (fixed subsets) ==="
date -Is | tee "$OUT/started_at.txt"
echo "policy/ref : $CKPT"
echo "output     : $OUT"
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
    --reward_type             r21_exact \
    --r21_enable              True \
    --route_cache             "$SPLITS/r21_route_cache.json" \
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
echo "  grep -oE \"'[a-z_/]*zero_adv[a-z_/]*': [0-9.eE+-]+|'reward': [0-9.eE+-]+|'reward_std': [0-9.eE+-]+|'kl': [0-9.eE+-]+|'peak_vram_gb': [0-9.eE+-]+\" $OUT/train.log | tail -20"
echo "  nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader"
echo
echo "to stop: kill \$(cat $OUT/pid.txt)"
