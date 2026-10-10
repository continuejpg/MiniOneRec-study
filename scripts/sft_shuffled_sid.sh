#!/bin/bash
# =============================================================================
# sft_shuffled_sid.sh -- SFT on the STRICT popularity-stratified shuffled-SID data
#
# Causal control for: "does the item <-> Semantic-ID correspondence carry the
# recommendation signal?"
#
# This script is a byte-for-byte functional clone of sft_full.sh (the script that
# produced runs/industrial_sft) with EXACTLY THREE arguments changed:
#
#     --train_file       data/Amazon/train/...csv  ->  analysis/shuffled_sid/train.csv
#     --eval_file        data/Amazon/valid/...csv  ->  analysis/shuffled_sid/valid.csv
#     --sid_index_path   data/Amazon/index/....index.json -> analysis/shuffled_sid/<cat>.index.json
#     --output_dir       runs/industrial_sft        ->  runs/industrial_sft_shuffled_sid
#
# Everything else (base model, batch sizes, LR, epochs, cutoff_len, seed,
# freeze_LLM, train_from_scratch, scheduler knobs, dtypes) is IDENTICAL.
#
# --output_dir is a write target, not a recipe parameter: it is changed only so
# that the clean run is never overwritten.
#
# --item_meta_path is deliberately KEPT at the ORIGINAL file. Titles/descriptions
# are item attributes, not SID assignments; the intervention must not touch them.
# See analysis/results/clean_vs_shuffled_sft_recipe.md for the full argument diff.
#
# SINGLE RTX 4090, no gradient accumulation change needed.
# This script does NOT start training by itself -- it is meant to be invoked
# explicitly. Nothing after the preflight below runs unless you execute it.
# =============================================================================

set -euo pipefail

export WANDB_MODE=disabled
export NCCL_IB_DISABLE=1
export HF_ENDPOINT=https://hf-mirror.com

# portable paths: PROJECT_ROOT / RUN_ROOT / DATA_ROOT / CATEGORY / PY
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# this stage needs the upstream Qwen2.5-0.5B weights -- deliberately no default
: "${BASE_MODEL:?set BASE_MODEL to the Qwen2.5-0.5B model path}"

# ---- original (clean) paths, kept for reference / the unchanged arguments ----
ORIG_INDEX="$INDEX"

# ---- intervention (shuffled) paths -- the ONLY recipe difference -------------
# NOTE on the index filename: TokenExtender (sft.py:31-39) does NOT open
# --sid_index_path verbatim. It rebuilds the name as
#     dirname(path) / basename(path).split('.')[0] + ".index.json"
# so the file must be called "<stem>.index.json" (upstream convention:
# "<Category>.index.json"). A file named "index.json" would give the stem "index"
# and the rebuild would look for "index.index.json" and crash.
# data.py opens --sid_index_path directly and is unaffected by the name.
SHUF_DIR="$PROJECT_ROOT/analysis/shuffled_sid"
SHUF_INDEX="$SHUF_DIR/${CATEGORY}.index.json"
SHUF_TRAIN="$SHUF_DIR/train.csv"
SHUF_VALID="$SHUF_DIR/valid.csv"

OUT="$RUN_ROOT/industrial_sft_shuffled_sid"
CLEAN_RUN="$RUN_ROOT/industrial_sft"

# -----------------------------------------------------------------------------
# Preflight: refuse to start if anything is off. No GPU work happens here.
# -----------------------------------------------------------------------------
echo "=== preflight ==="
fail=0
check_exists() {
    if [ -e "$1" ]; then printf '  OK      %s\n' "$1"; else printf '  MISSING %s\n' "$1"; fail=1; fi
}
check_exists "$BASE_MODEL"
check_exists "$ITEM_META"
check_exists "$SHUF_INDEX"
check_exists "$SHUF_TRAIN"
check_exists "$SHUF_VALID"
check_exists "$PROJECT_ROOT/sft.py"

# INDEX must keep the "<Category>.index.json" stem: TokenExtender rebuilds the
# filename from `basename(--sid_index_path).split('.')[0]` (sft.py:31-39/152-153).
if [ "$(basename "$INDEX")" = "${CATEGORY}.index.json" ]; then
    echo "  OK      index basename is ${CATEGORY}.index.json"
else
    echo "  REFUSE  index basename is $(basename "$INDEX"), expected ${CATEGORY}.index.json"
    fail=1
fi

# the clean run must exist (we are controlling against it) and must be untouched
if [ -e "$CLEAN_RUN/final_checkpoint/model.safetensors" ]; then
    echo "  OK      clean run present: $CLEAN_RUN"
else
    echo "  MISSING clean run: $CLEAN_RUN (control baseline absent)"; fail=1
fi

# output dir must NOT already contain a completed run
if [ -e "$OUT/final_checkpoint/model.safetensors" ]; then
    echo "  REFUSE  output dir already holds a finished run: $OUT"
    echo "          move it aside or pick a new --output_dir before rerunning"
    fail=1
else
    echo "  OK      output dir is free: $OUT"
fi

# the shuffled index must add the SAME number of new tokens as the original,
# otherwise the tokenizer/embedding layout would differ between the two arms
python - "$ORIG_INDEX" "$SHUF_INDEX" <<'PY' || fail=1
import json, os, sys
o = json.load(open(sys.argv[1], encoding="utf-8"))
s = json.load(open(sys.argv[2], encoding="utf-8"))
ot = sorted({t for v in o.values() for t in v})
st = sorted({t for v in s.values() for t in v})
print(f"  orig new_tokens={len(ot)}  shuffled new_tokens={len(st)}  identical={ot == st}")
if ot != st:
    print("  REFUSE  shuffled index has a different token vocabulary than the original")
    sys.exit(1)
if set(o) != set(s):
    print("  REFUSE  shuffled index item-key set differs from the original")
    sys.exit(1)

# reproduce exactly what TokenExtender will compute and require it to resolve
p = sys.argv[2]
rebuilt = os.path.join(os.path.dirname(p), os.path.basename(p).split(".")[0] + ".index.json")
print(f"  TokenExtender would open: {rebuilt}")
if not os.path.exists(rebuilt):
    print("  REFUSE  TokenExtender's rebuilt index path does not exist "
          "(the file must be named '<stem>.index.json')")
    sys.exit(1)
if os.path.abspath(rebuilt) != os.path.abspath(p):
    print("  REFUSE  TokenExtender would open a DIFFERENT file than --sid_index_path")
    sys.exit(1)
print("  OK      TokenExtender resolved path == --sid_index_path")
PY

if [ "$fail" -ne 0 ]; then
    echo "=== preflight FAILED -- not starting training ==="
    exit 1
fi
echo "=== preflight OK ==="
echo

# -----------------------------------------------------------------------------
# Training -- IDENTICAL to sft_full.sh except the three swapped paths.
# -----------------------------------------------------------------------------
mkdir -p "$OUT"

torchrun --nproc_per_node 1 \
    sft.py \
    --base_model       "$BASE_MODEL" \
    --train_file       "$SHUF_TRAIN" \
    --eval_file        "$SHUF_VALID" \
    --output_dir       "$OUT" \
    --category         "$CATEGORY" \
    --sid_index_path   "$SHUF_INDEX" \
    --item_meta_path   "$ITEM_META" \
    --sample           -1 \
    --num_epochs       2 \
    --batch_size       64 \
    --micro_batch_size 16 \
    --learning_rate    3e-4 \
    --cutoff_len       512 \
    --seed             42 \
    --train_from_scratch False \
    --freeze_LLM       False \
    2>&1 | tee "$OUT/train.log"
