#!/bin/bash
# MiniOneRec-CAT / B0 on the AutoDL instance (RTX 4090).
#
# Intended to be copied to the instance and run there, e.g.
#   bash /root/autodl-tmp/code/experiments/minionerec_cat/scripts/run_b0_remote.sh
#
# Differences from the local launcher:
#   * uses the instance's python (/root/miniconda3/bin/python, torch 2.6.0+cu118)
#   * defaults to device cuda:0
#   * writes only under the instance copy of experiments/minionerec_cat/runs/
#
# It performs a fail-fast environment check first, because the instance can be
# sitting in a state where nvidia-smi reports "No devices were found" while SSH
# still works -- training would then silently fall back to CPU.

set -o pipefail

TAG="${TAG:-b0}"
PY="${PY:-/root/miniconda3/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
EPOCHS="${EPOCHS:-300}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
EVAL_STEP="${EVAL_STEP:-10}"
SEED="${SEED:-2024}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SKIP_ENV_CHECK="${SKIP_ENV_CHECK:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$CAT_ROOT/../.." && pwd)"

echo "=============================================================="
echo " MiniOneRec-CAT / B0 on AutoDL    tag=$TAG"
echo "=============================================================="
echo "  REPO_ROOT : $REPO_ROOT"
echo "  python    : $PY"
echo "  device    : $DEVICE"
echo "--------------------------------------------------------------"

if [ "$SKIP_ENV_CHECK" != "1" ]; then
    echo "[env] GPU check"
    if ! nvidia-smi >/dev/null 2>&1; then
        echo "  FAIL: nvidia-smi not usable. If it prints 'No devices were found' the"
        echo "        instance has no GPU allocated (powered off / released). Start it"
        echo "        from the AutoDL console, then re-run. This script will not"
        echo "        silently train on CPU."
        exit 4
    fi
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/^/  /'
    "$PY" - <<'PYEOF' || exit 4
import sys
try:
    import torch
except Exception as exc:
    raise SystemExit(f"  FAIL: cannot import torch: {exc!r}")
print(f"  torch {torch.__version__}  cuda_available={torch.cuda.is_available()}  devices={torch.cuda.device_count()}")
if not torch.cuda.is_available():
    raise SystemExit("  FAIL: torch reports CUDA unavailable; refusing to train on CPU here")
PYEOF
    echo "[env] disk"
    df -h "$REPO_ROOT" | tail -1 | sed 's/^/  /'
    avail_kb=$(df -Pk "$REPO_ROOT" | awk 'NR==2{print $4}')
    if [ "${avail_kb:-0}" -lt 2000000 ]; then
        echo "  WARN: less than ~2 GB free under $REPO_ROOT; RQ-VAE artefacts are small"
        echo "        but the SFT stage that follows needs much more."
    fi
fi

cd "$REPO_ROOT" || { echo "cannot cd to $REPO_ROOT"; exit 2; }

echo "[stage 0] P0 input integrity"
"$PY" - <<'PYEOF' || exit 3
from experiments.minionerec_cat.src.verify_chain import check_p0_untouched
res = check_p0_untouched()
bad = {k: v for k, v in res.items() if not v.get("ok")}
for k, v in res.items():
    print(f"   {'OK ' if v.get('ok') else 'BAD'} {v.get('sha256_prefix','?')}  {k}")
if bad:
    raise SystemExit("REFUSING TO RUN: P0 inputs differ from the recorded hashes")
print("   all P0 inputs match the recorded hashes")
PYEOF

echo "[stage 1] train RQ-VAE"
"$PY" -m experiments.minionerec_cat.src.train_rqvae \
    --tag "$TAG" --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" \
    --eval-step "$EVAL_STEP" --seed "$SEED" --device "$DEVICE" \
    --num-workers "$NUM_WORKERS"
rc=$?
if [ "$rc" -ne 0 ]; then echo "[stage 1] FAILED rc=$rc"; exit "$rc"; fi

echo "[stage 2] extract SID catalogue"
"$PY" -m experiments.minionerec_cat.src.extract_index --tag "$TAG" --seed "$SEED" --device "$DEVICE"
rc=$?
if [ "$rc" -ne 0 ]; then echo "[stage 2] FAILED rc=$rc"; exit "$rc"; fi

echo "[stage 3] rebuild derived artefacts"
"$PY" -m experiments.minionerec_cat.src.rebuild_derived --tag "$TAG"
rc=$?
if [ "$rc" -ne 0 ]; then echo "[stage 3] FAILED rc=$rc"; exit "$rc"; fi

echo "[stage 4] verify chain"
"$PY" -m experiments.minionerec_cat.src.verify_chain --tag "$TAG"
rc=$?

echo "--------------------------------------------------------------"
if [ "$rc" -eq 0 ]; then
    echo " B0 COMPLETE on AutoDL.  tag=$TAG"
    echo "   index : $CAT_ROOT/runs/$TAG/index/Industrial_and_Scientific.index.$TAG.json"
    echo "   info  : $CAT_ROOT/runs/$TAG/derived/info/Industrial_and_Scientific_5_2016-10-2018-11.txt"
    echo "   splits: $CAT_ROOT/runs/$TAG/derived/{train,valid,test}/"
    echo " Pull the reports back with scripts/pull_from_autodl.sh"
else
    echo " B0 VERIFICATION FAILED rc=$rc"
fi
exit "$rc"
