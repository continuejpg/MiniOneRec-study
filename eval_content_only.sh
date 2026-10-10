#!/bin/bash
# =============================================================================
# Stage 3 -- formal LETTER-SID beam20 evaluation.
#
# Protocol is a STRICT COPY of the P0 evaluation (eval_industrial.sh):
#   test samples 4533, num_beams 20, num_return_sequences 20, constrained
#   decoding, length_penalty 0.0, max_new_tokens 256, batch_size 8,
#   same calc.py.
#
# Everything SID-related comes from the LETTER artefacts -- checkpoint, tokenizer
# (saved inside the checkpoint), info and test. The ORIGINAL/P0 info is NEVER used.
# =============================================================================
set -o pipefail

export WANDB_MODE=disabled
export HF_ENDPOINT=https://hf-mirror.com

# Stage 3: protobuf 6.33.6 (pulled in by k-means-constrained) breaks
# tensorboard 2.11.2, which transformers imports via TensorBoardCallback.
# protobuf's documented workaround #2: pure-Python implementation.
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
# inert wandb stub (see sitecustomize.py) -- reversible, no source change
export PYTHONPATH=/tmp/s3/guard:${PYTHONPATH:-}

D=/root/autodl-tmp/code
cat=Industrial_and_Scientific
CKPT=$D/runs/content_only_sid_qwen05b_sft/final_checkpoint
LD=$D/artifacts/letter_stage4_content_only/data
STEM=${cat}_5_2016-10-2018-11

OUT=$D/runs/content_only_sid_qwen05b_eval
mkdir -p "$OUT"

echo "=== checkpoint : $CKPT"
echo "=== info       : $LD/info/$STEM.txt"
echo "=== test       : $LD/test/$STEM.csv"
echo "=== output     : $OUT"
echo "=== start      : $(date '+%Y-%m-%d %H:%M:%S')"

cd "$D"

/root/miniconda3/bin/python -u ./evaluate.py \
    --base_model      "$CKPT" \
    --info_file       "$LD/info/$STEM.txt" \
    --category        "$cat" \
    --test_data_path  "$LD/test/$STEM.csv" \
    --result_json_data "$OUT/content_only_result_${cat}.json" \
    --batch_size      8 \
    --num_beams       20 \
    --max_new_tokens  256 \
    --length_penalty  0.0 \
    2>&1 | tee "$OUT/evaluate.log"

echo "=== evaluate exit: $?"

/root/miniconda3/bin/python -u ./calc.py \
    --path "$OUT/content_only_result_${cat}.json" \
    --item_path "$LD/info/$STEM.txt" \
    2>&1 | tee "$OUT/metrics.txt"

echo "=== calc exit: $?"
echo "=== end        : $(date '+%Y-%m-%d %H:%M:%S')"
