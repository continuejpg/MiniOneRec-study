#!/bin/bash
# Phase 4B -- Valid beam-20 evaluation for C0/C1 (NO test evaluation this round).
set -u
export PATH=/root/miniconda3/bin:$PATH
export OMP_NUM_THREADS=16 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export HF_HOME=/root/autodl-tmp/hf_home

D=/root/autodl-tmp
CODE=$D/code
CAT=Industrial_and_Scientific
RUNS=$D/p4b_runs
EV=$D/p4b_eval
mkdir -p "$EV"

for ARM in C0 C1; do
  W=$D/p3_armdata/S0
  echo
  echo "################################################################"
  echo "  VALID EVAL $ARM"
  echo "################################################################"
  if [ ! -f "$RUNS/$ARM/final_checkpoint/model.safetensors" ]; then
    echo "  no model yet -- skipping"; continue
  fi
  cd "$CODE" || exit 1
  START=$(date +%s)
  CUDA_VISIBLE_DEVICES=0 timeout 5400 python -u evaluate.py \
    --base_model       "$RUNS/$ARM/final_checkpoint" \
    --info_file        "$W/derived/info/${CAT}_5_2016-10-2018-11.txt" \
    --category         "$CAT" \
    --test_data_path   "$W/derived/valid/${CAT}_5_2016-10-2018-11.csv" \
    --result_json_data "$EV/${ARM}_valid_beam20.json" \
    --batch_size       8 --num_beams 20 --max_new_tokens 256 --length_penalty 0.0 \
    > "$EV/${ARM}_valid_eval.log" 2>&1
  echo "  exit=$?  elapsed=$(( $(date +%s)-START ))s"
  grep -aE "inferred SID depth|Traceback|Error" "$EV/${ARM}_valid_eval.log" | head -3 | sed 's/^/    /'
  [ -f "$EV/${ARM}_valid_beam20.json" ] && \
    echo "    rows=$(python -c "import json;print(len(json.load(open('$EV/${ARM}_valid_beam20.json'))))")  sha=$(sha256sum $EV/${ARM}_valid_beam20.json | cut -c1-16)"
done

cd "$EV" && sha256sum *.json > SHA256SUMS 2>/dev/null && cat SHA256SUMS | sed 's/^/  /'
df -h $D | tail -1
