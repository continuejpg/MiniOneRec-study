#!/bin/bash
# MiniOneRec-CAT / B0 -- text-only RQ-VAE, end to end, fully isolated.
#
# Runs the three stages in order:
#   1. train the RQ-VAE on the frozen item-text embeddings
#   2. extract the SID catalogue from the trained checkpoint
#   3. rebuild EVERY SID-derived artefact (splits + info + vocab) and verify
#
# Nothing outside experiments/minionerec_cat/runs/<TAG>/ is written.  The P0
# inputs are read-only and are hash-checked at the end.
#
# Usage:
#   bash experiments/minionerec_cat/scripts/run_b0.sh
#   TAG=b0_nightly EPOCHS=600 bash experiments/minionerec_cat/scripts/run_b0.sh
#   DRY_RUN=1 bash experiments/minionerec_cat/scripts/run_b0.sh

set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$CAT_ROOT/../.." && pwd)"

TAG="${TAG:-b0}"
EPOCHS="${EPOCHS:-300}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
LR="${LR:-1e-3}"
EVAL_STEP="${EVAL_STEP:-10}"
E_DIM="${E_DIM:-32}"
SEED="${SEED:-2024}"
DEVICE="${DEVICE:-cuda:0}"
NUM_WORKERS="${NUM_WORKERS:-0}"
PY="${PY:-python}"
DRY_RUN="${DRY_RUN:-0}"

RUN_ROOT="$CAT_ROOT/runs/$TAG"
LOG_DIR="$RUN_ROOT/logs"
mkdir -p "$LOG_DIR"

echo "=============================================================="
echo " MiniOneRec-CAT / B0   tag=$TAG"
echo "=============================================================="
echo "  REPO_ROOT : $REPO_ROOT"
echo "  RUN_ROOT  : $RUN_ROOT"
echo "  python    : $($PY -V 2>&1)"
echo "  epochs=$EPOCHS batch=$BATCH_SIZE lr=$LR eval_step=$EVAL_STEP e_dim=$E_DIM seed=$SEED device=$DEVICE"
echo "  DRY_RUN   : $DRY_RUN"
echo "--------------------------------------------------------------"

cd "$REPO_ROOT" || { echo "cannot cd to $REPO_ROOT"; exit 2; }

# ---- stage 0: refuse to start if the P0 inputs are not the ones we recorded --
if [ "$DRY_RUN" != "1" ]; then
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
fi

# ---- stage 1: train ----------------------------------------------------------
echo "[stage 1] train RQ-VAE"
ARGS=(--tag "$TAG" --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --lr "$LR"
      --eval-step "$EVAL_STEP" --e-dim "$E_DIM" --seed "$SEED"
      --device "$DEVICE" --num-workers "$NUM_WORKERS")
if [ "$DRY_RUN" = "1" ]; then
    "$PY" -m experiments.minionerec_cat.src.train_rqvae "${ARGS[@]}" --dry-run \
        2>&1 | tee "$LOG_DIR/b0_train.log"
    echo "[stage 1] dry run complete; stopping before index extraction"
    exit 0
fi
"$PY" -m experiments.minionerec_cat.src.train_rqvae "${ARGS[@]}" 2>&1 | tee "$LOG_DIR/b0_train.log"
rc=${PIPESTATUS[0]}
if [ "$rc" -ne 0 ]; then echo "[stage 1] FAILED rc=$rc"; exit "$rc"; fi

# ---- stage 2: extract the SID catalogue -------------------------------------
echo "[stage 2] extract SID catalogue"
"$PY" -m experiments.minionerec_cat.src.extract_index --tag "$TAG" --seed "$SEED" \
    2>&1 | tee "$LOG_DIR/b0_index.log"
rc=${PIPESTATUS[0]}
if [ "$rc" -ne 0 ]; then echo "[stage 2] FAILED rc=$rc"; exit "$rc"; fi

# ---- stage 3: rebuild every SID-derived artefact -----------------------------
echo "[stage 3] rebuild derived artefacts"
"$PY" -m experiments.minionerec_cat.src.rebuild_derived --tag "$TAG" \
    2>&1 | tee "$LOG_DIR/b0_rebuild.log"
rc=${PIPESTATUS[0]}
if [ "$rc" -ne 0 ]; then echo "[stage 3] FAILED rc=$rc"; exit "$rc"; fi

# ---- stage 4: verify the chain ----------------------------------------------
echo "[stage 4] verify chain"
"$PY" -m experiments.minionerec_cat.src.verify_chain --tag "$TAG" \
    2>&1 | tee "$LOG_DIR/b0_verify.log"
rc=${PIPESTATUS[0]}
echo "--------------------------------------------------------------"
if [ "$rc" -eq 0 ]; then
    echo " B0 COMPLETE.  tag=$TAG"
    echo "   derived train : $RUN_ROOT/derived/train"
    echo "   derived valid : $RUN_ROOT/derived/valid"
    echo "   derived test  : $RUN_ROOT/derived/test"
    echo "   info          : $RUN_ROOT/derived/info"
    echo "   SFT launch with --sid_index_path $RUN_ROOT/index/Industrial_and_Scientific.index.$TAG.json"
else
    echo " B0 VERIFICATION FAILED rc=$rc -- do NOT start SFT on this SID set"
fi
exit "$rc"
