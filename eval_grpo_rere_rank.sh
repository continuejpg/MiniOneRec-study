#!/bin/bash
# =====================================================================
# eval_grpo_rere_rank.sh -- formal beam20 eval for the R1 ReRe-rank GRPO run.
#
# Protocol is a STRICT COPY of the existing GRPO baseline evaluation
# (eval_industrial.sh / eval_grpo_baseline): same 4,533 test samples,
# same info file, beam 20, max_new_tokens 256, length_penalty 0.0,
# batch_size 8, same calc.py.
#
# The ONLY differences are the checkpoint and the output directory.
# =====================================================================
set -o pipefail

export WANDB_MODE=disabled
export HF_ENDPOINT=https://hf-mirror.com
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH=/tmp/s3/guard:${PYTHONPATH:-}

D=/root/autodl-tmp/code
cat=Industrial_and_Scientific
CKPT=$D/runs/grpo_rere_rank025/final_checkpoint
OUT=$D/runs/eval_grpo_rere_rank025
mkdir -p "$OUT"

echo "=== checkpoint : $CKPT"
echo "=== info       : $D/data/Amazon/info/${cat}_5_2016-10-2018-11.txt"
echo "=== test       : $D/data/Amazon/test/${cat}_5_2016-10-2018-11.csv"
echo "=== output     : $OUT"
echo "=== start      : $(date '+%Y-%m-%d %H:%M:%S')"

cd "$D"

/root/miniconda3/bin/python -u ./evaluate.py \
    --base_model      "$CKPT" \
    --info_file       "$D/data/Amazon/info/${cat}_5_2016-10-2018-11.txt" \
    --category        "$cat" \
    --test_data_path  "$D/data/Amazon/test/${cat}_5_2016-10-2018-11.csv" \
    --result_json_data "$OUT/test_beam20.json" \
    --batch_size      8 \
    --num_beams       20 \
    --max_new_tokens  256 \
    --length_penalty  0.0 \
    2>&1 | tee "$OUT/evaluate.log"

echo "=== evaluate exit: $?"

/root/miniconda3/bin/python -u ./calc.py \
    --path "$OUT/test_beam20.json" \
    --item_path "$D/data/Amazon/info/${cat}_5_2016-10-2018-11.txt" \
    2>&1 | tee "$OUT/metrics.txt"

echo "=== calc exit: $?"
echo "=== end        : $(date '+%Y-%m-%d %H:%M:%S')"
