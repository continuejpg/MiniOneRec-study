#!/bin/bash
# Phase 4B -- discard the VOID C1 run and redo it with the weight-propagation fix.
set -u
export PATH=/root/miniconda3/bin:/usr/bin:/bin:/usr/sbin:/sbin
export OMP_NUM_THREADS=16
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export HF_HOME=/root/autodl-tmp/hf_home HF_DATASETS_CACHE=/root/autodl-tmp/hf_home/datasets
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True P3_REPO_ROOT=/root/autodl-tmp/code

D=/root/autodl-tmp
CODE=$D/code
CAT=Industrial_and_Scientific
W=$D/p3_armdata/S0
RUNS=$D/p4b_runs
LOGS=$RUNS/logs
EV=$D/p4b_eval
ARCH=$D/p4b_archive
mkdir -p "$LOGS" "$EV" "$ARCH"
free_gb() { df --output=avail -BG "$D" | tail -1 | tr -dc '0-9'; }

echo "########## 1. quarantine the VOID C1 (weights never reached the loss) ##########"
if [ -f "$RUNS/C1/final_checkpoint/model.safetensors" ]; then
  mkdir -p "$ARCH/void_C1"
  echo "  old C1 model sha: $(sha256sum $RUNS/C1/final_checkpoint/model.safetensors | cut -c1-16)"
  echo "  old C0 model sha: $(sha256sum $RUNS/C0/final_checkpoint/model.safetensors | cut -c1-16)"
  echo "  (identical => C1 was inert) ; recording evidence then removing the void run"
  cp "$RUNS/C1/final_checkpoint/config.json" "$ARCH/void_C1/config.json" 2>/dev/null
  cp "$RUNS/C1/p4b_weights.json" "$ARCH/void_C1/weights.json" 2>/dev/null
  echo "void C1: weights.json differed but train_loss and model hash were identical to C0" \
    > "$ARCH/void_C1/README.txt"
  rm -rf "$RUNS/C1"
  echo "  removed void C1 run dir"
fi
mv "$EV/C1_valid_beam20.json" "$ARCH/void_C1/C1_valid_beam20.json" 2>/dev/null && \
  echo "  quarantined the void C1 eval artifact"
df -h $D | tail -1

echo
echo "########## 2. retrain C1 alpha=0.5 (fix applied)  free=$(free_gb) GiB ##########"
mkdir -p "$RUNS/C1"
cd "$CODE" || exit 1
START=$(date +%s)
torchrun --nproc_per_node 1 /root/p4b_run.py \
  --arm C1 --alpha 0.5 --base_model "$D/models/Qwen2.5-0.5B" \
  --train_csv "$W/derived/train/${CAT}_5_2016-10-2018-11.csv" \
  --eval_file "$W/derived/valid/${CAT}_5_2016-10-2018-11.csv" \
  --output_dir "$RUNS/C1" \
  --sid_index_path "$W/${CAT}.index.json" \
  --item_meta_path "$W/${CAT}.item.json" \
  --sample -1 --num_epochs 1 --batch_size 64 --micro_batch_size 8 \
  --learning_rate 3e-4 --cutoff_len 512 --seed 42 --equivalence_probe 0 \
  > "$LOGS/C1.log" 2>&1
echo "  exit=$?  elapsed=$(( $(date +%s)-START ))s  free=$(free_gb) GiB"
grep -aE "remove_unused_columns|train dataset components" "$LOGS/C1.log" | head -3 | sed 's/^/    /'
tr '\r' '\n' < "$LOGS/C1.log" | grep -aE "'train_runtime'" | tail -1 | sed 's/^/    /'
echo "  --- did the weighted loss actually differ from stock? (sample of probe lines) ---"
grep -aoE "weighted=[0-9.]+ stock=[0-9.]+ diff=[0-9.e-]+" "$LOGS/C1.log" | head -3 | sed 's/^/    /'
echo "  --- NaN/Inf ---"
tr '\r' '\n' < "$LOGS/C1.log" | grep -aoE "'loss': [-0-9.eE+]+|'grad_norm': [-0-9.eE+]+" \
  | sed "s/[^0-9.eE+-]//g" > "$RUNS/C1/reported_values.txt"
python - "$RUNS/C1/reported_values.txt" <<'PYEOF' | sed 's/^/    /'
import math, sys
v=[float(x) for x in open(sys.argv[1]) if x.strip()]
print(f"values={len(v)} non_finite={sum(1 for x in v if not math.isfinite(x))} min={min(v):.4f} max={max(v):.4f}")
PYEOF

echo
echo "########## 3. reload + Valid beam20 ##########"
cd "$CODE" || exit 1
python - <<'PYEOF'
import json, torch, pathlib
from transformers import AutoModelForCausalLM, AutoTokenizer
ck = pathlib.Path("/root/autodl-tmp/p4b_runs/C1/final_checkpoint")
tok = AutoTokenizer.from_pretrained(str(ck))
mdl = AutoModelForCausalLM.from_pretrained(str(ck), dtype=torch.bfloat16)
finite = all(torch.isfinite(v.float()).all().item() for v in mdl.state_dict().values())
print(f"  C1 reloaded: vocab={len(tok)} embed={mdl.get_input_embeddings().weight.shape[0]} "
      f"params={sum(p.numel() for p in mdl.parameters()):,} all_finite={finite}")
PYEOF
CUDA_VISIBLE_DEVICES=0 timeout 5400 python -u evaluate.py \
  --base_model "$RUNS/C1/final_checkpoint" \
  --info_file  "$W/derived/info/${CAT}_5_2016-10-2018-11.txt" \
  --category   "$CAT" \
  --test_data_path "$W/derived/valid/${CAT}_5_2016-10-2018-11.csv" \
  --result_json_data "$EV/C1_valid_beam20.json" \
  --batch_size 8 --num_beams 20 --max_new_tokens 256 --length_penalty 0.0 \
  > "$EV/C1_valid_eval.log" 2>&1
echo "  eval exit=$?"
grep -aE "inferred SID depth|Traceback" "$EV/C1_valid_eval.log" | head -2 | sed 's/^/    /'

echo
echo "########## 4. hashes ##########"
for A in C0 C1; do
  printf "  %s model sha: %s\n" "$A" "$(sha256sum $RUNS/$A/final_checkpoint/model.safetensors | cut -c1-16)"
done
echo "  --- the two models MUST now differ ---"
if [ "$(sha256sum $RUNS/C0/final_checkpoint/model.safetensors | cut -d' ' -f1)" = \
     "$(sha256sum $RUNS/C1/final_checkpoint/model.safetensors | cut -d' ' -f1)" ]; then
  echo "  *** STILL IDENTICAL -- the intervention is STILL inert ***"
else
  echo "  OK: models differ, the intervention had an effect"
fi
cd "$EV" && sha256sum *.json > SHA256SUMS 2>/dev/null && cat SHA256SUMS | sed 's/^/  /'
df -h $D | tail -1
