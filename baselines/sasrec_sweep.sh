#!/bin/bash
# SASRec configuration sweep -- find a fairly-trained baseline, not a straw man.
#
# Paths come from scripts/common.sh (override PROJECT_ROOT / RUN_ROOT / PY from
# the environment). The 4-config sweep below is intentionally unchanged.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../scripts/common.sh"

cd "$PROJECT_ROOT"
B="$PROJECT_ROOT/baselines/sasrec_baseline.py"
L="$RUN_ROOT/sasrec_sweep.log"
mkdir -p "$RUN_ROOT"
: > "$L"

"$PY" -c "import torch" 2>/dev/null || {
  echo "REFUSE: '$PY' cannot import torch. Set PY to an interpreter that can." >&2
  exit 1
}

run () {
  echo "" | tee -a "$L"
  echo "############################ $1 ############################" | tee -a "$L"
  shift
  "$PY" -u "$B" "$@" 2>/dev/null | grep -vE "^\s*$" | tail -32 | tee -a "$L"
}

run "A: hidden=64  heads=1  full-softmax  100ep"  --tag a_h64_full    --hidden 64  --heads 1 --objective full --epochs 100
run "B: hidden=64  heads=2  full-softmax  200ep"  --tag b_h64_full2   --hidden 64  --heads 2 --objective full --epochs 200
run "C: hidden=128 heads=2  full-softmax  200ep"  --tag c_h128_full   --hidden 128 --heads 2 --objective full --epochs 200
run "D: hidden=64  heads=1  neg=100       200ep"  --tag d_h64_neg100  --hidden 64  --heads 1 --objective neg --neg 100 --epochs 200

echo "" | tee -a "$L"
echo "############################ SWEEP DONE ############################" | tee -a "$L"
