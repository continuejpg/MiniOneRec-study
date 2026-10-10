#!/bin/bash
export WANDB_MODE=disabled
export HF_ENDPOINT=https://hf-mirror.com

D=/root/autodl-tmp
cat=Industrial_and_Scientific
CKPT=$D/runs/industrial_sft/final_checkpoint

OUT=$D/runs/eval_industrial
mkdir -p $OUT

echo "=== 评测 checkpoint: $CKPT ==="
echo "=== 输出目录: $OUT ==="

# 单卡直接评测,不用 split.py 拆分
python -u ./evaluate.py \
    --base_model      "$CKPT" \
    --info_file       "$D/code/data/Amazon/info/${cat}_5_2016-10-2018-11.txt" \
    --category        $cat \
    --test_data_path  "$D/code/data/Amazon/test/${cat}_5_2016-10-2018-11.csv" \
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
