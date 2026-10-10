#!/bin/bash
export WANDB_MODE=disabled
export HF_ENDPOINT=https://hf-mirror.com

D=/root/autodl-tmp
cat=Industrial_and_Scientific
# [R2.2] evaluate the trained R2 checkpoint on the VALIDATION split.
# Protocol otherwise byte-identical to eval_industrial.sh.
CKPT=/root/autodl-tmp/code/runs/grpo_r21/final_checkpoint

OUT=/root/autodl-tmp/runs/eval_grpo_r21_valid
mkdir -p $OUT

echo "=== 评测 checkpoint: $CKPT ==="
echo "=== 输出目录: $OUT ==="

# 单卡直接评测,不用 split.py 拆分
python -u ./evaluate.py \
    --base_model      "$CKPT" \
    --info_file       "$D/code/data/Amazon/info/${cat}_5_2016-10-2018-11.txt" \
    --category        $cat \
    --test_data_path  "$D/code/data/Amazon/valid/${cat}_5_2016-10-2018-11.csv" \
    --result_json_data "$OUT/final_result_${cat}.json" \
    --batch_size      8 \
    --num_beams       20 \
    --max_new_tokens  256 \
    --length_penalty  0.0 \
    2>&1 | tee $OUT/evaluate.log

echo ""
echo "=== 计算指标 ==="
python ./calc.py \
    --path "$OUT/final_result_${cat}.json" \
    --item_path "$D/code/data/Amazon/info/${cat}_5_2016-10-2018-11.txt" \
    2>&1 | tee $OUT/metrics.txt

echo ""
echo "=== 完成 ==="
cat $OUT/metrics.txt
