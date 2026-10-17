#!/bin/bash
# Phase 4B -- run C0 (alpha=0) then C1 (alpha=0.5), 1 epoch each, identical otherwise.
# Disk-safe: train one, archive + verify reload, then prune only THIS round's
# optimizer states before starting the next.
set -u
export PATH=/root/miniconda3/bin:$PATH
export OMP_NUM_THREADS=16
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export HF_HOME=/root/autodl-tmp/hf_home
export HF_DATASETS_CACHE=/root/autodl-tmp/hf_home/datasets
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export P3_REPO_ROOT=/root/autodl-tmp/code

D=/root/autodl-tmp
CODE=$D/code
CAT=Industrial_and_Scientific
BASE=$D/models/Qwen2.5-0.5B
W=$D/p3_armdata/S0                 # B0 seed-2024 SID, same as S0 in phase 3
RUNS=$D/p4b_runs
LOGS=$RUNS/logs
ARCH=$D/p4b_archive
MINGB=8
mkdir -p "$LOGS" "$ARCH"

free_gb() { df --output=avail -BG "$D" | tail -1 | tr -dc '0-9'; }

echo "#################### PREFLIGHT ####################"
echo "  disk free: $(free_gb) GiB (threshold $MINGB)"
/root/miniconda3/bin/python - <<'PYEOF'
import hashlib, json, pathlib
D = pathlib.Path("/root/autodl-tmp"); A = D/"code/data/Amazon"
P0 = {"index/Industrial_and_Scientific.index.json":"53f1aefbe8b8aa77",
      "info/Industrial_and_Scientific_5_2016-10-2018-11.txt":"ebd7d594f2dc884d",
      "train/Industrial_and_Scientific_5_2016-10-2018-11.csv":"5e5456d4b54681d7",
      "valid/Industrial_and_Scientific_5_2016-10-2018-11.csv":"44d441a4cbb50e18",
      "test/Industrial_and_Scientific_5_2016-10-2018-11.csv":"43cea20fc742e481"}
ok = True
for rel, exp in P0.items():
    g = hashlib.sha256((A/rel).read_bytes()).hexdigest()[:16]
    ok &= g == exp
    print(f"    [{'OK ' if g==exp else 'BAD'}] {g}  {rel}")
m = D/"models/Qwen2.5-0.5B/model.safetensors"
g = hashlib.sha256(m.read_bytes()).hexdigest()[:16]
ok &= g == "88c142557820ccad"
print(f"    [{'OK ' if g=='88c142557820ccad' else 'BAD'}] {g}  base model")
# B0 SID hash must equal S0's
idx = D/"p3_armdata/S0/Industrial_and_Scientific.index.json"
h = hashlib.sha256(idx.read_bytes()).hexdigest()[:16]
print(f"    SID index (B0 seed2024) = {h}  (S0 used b5a7d78b0f50e6b5)")
ok &= h == "b5a7d78b0f50e6b5"
print(f"  PREFLIGHT: {'PASS' if ok else 'FAIL'}")
raise SystemExit(0 if ok else 1)
PYEOF
[ $? -ne 0 ] && { echo "  PREFLIGHT FAILED"; exit 1; }

for spec in "C0:0.0" "C1:0.5"; do
  ARM=${spec%%:*}; ALPHA=${spec##*:}
  OUT=$RUNS/$ARM
  echo
  echo "================================================================"
  echo "  $ARM   alpha=$ALPHA   free=$(free_gb) GiB"
  echo "================================================================"
  if [ -f "$OUT/final_checkpoint/model.safetensors" ]; then echo "  already done"; continue; fi
  [ "$(free_gb)" -lt "$MINGB" ] && { echo "  DISK TOO LOW -- stopping"; break; }

  mkdir -p "$OUT"
  cd "$CODE" || exit 1
  START=$(date +%s)
  torchrun --nproc_per_node 1 /root/p4b_run.py \
    --arm "$ARM" --alpha "$ALPHA" \
    --base_model "$BASE" \
    --train_csv  "$W/derived/train/${CAT}_5_2016-10-2018-11.csv" \
    --eval_file  "$W/derived/valid/${CAT}_5_2016-10-2018-11.csv" \
    --output_dir "$OUT" \
    --sid_index_path "$W/${CAT}.index.json" \
    --item_meta_path "$W/${CAT}.item.json" \
    --sample -1 --num_epochs 1 --batch_size 64 --micro_batch_size 8 \
    --learning_rate 3e-4 --cutoff_len 512 --seed 42 \
    > "$LOGS/${ARM}.log" 2>&1
  RC=$?
  END=$(date +%s)
  cp "$OUT/p4b_weights.json" "$ARCH/${ARM}_weights.json" 2>/dev/null
  cp "$OUT/p4b_alpha0_probe.json" "$ARCH/${ARM}_probe.json" 2>/dev/null
  echo "  exit=$RC  elapsed=$((END-START))s ($(( (END-START)/60 )) min)  free=$(free_gb) GiB"
  tr '\r' '\n' < "$LOGS/${ARM}.log" | grep -aE "train dataset components|'train_runtime'|EarlyStopping|Traceback|out of memory|No space" | tail -6 | sed 's/^/    /'
  echo "  --- NaN/Inf scan ---"
  tr '\r' '\n' < "$LOGS/${ARM}.log" | grep -aoE "'loss': [-0-9.eE+]+|'grad_norm': [-0-9.eE+]+" \
    | sed "s/[^0-9.eE+-]//g" > "$OUT/reported_values.txt"
  /root/miniconda3/bin/python - "$OUT/reported_values.txt" <<'PYEOF' | sed 's/^/    /'
import math, sys
v = [float(x) for x in open(sys.argv[1]) if x.strip()]
bad = [x for x in v if not math.isfinite(x)]
print(f"values={len(v)} non_finite={len(bad)} min={min(v):.4f} max={max(v):.4f}")
PYEOF
  echo "  run dir: $(du -sh $OUT | cut -f1)"
done

echo
echo "#################### DONE ####################"
df -h $D | tail -1
du -sh $RUNS/* 2>/dev/null | sed 's/^/  /'
