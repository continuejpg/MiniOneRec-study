#!/bin/bash
# =============================================================================
# eval_shuffled_sid.sh -- beam=20 test evaluation of the shuffled-SID SFT model
#
# Causal control for: "does the item <-> Semantic-ID correspondence carry the
# recommendation signal?"
#
# This script reproduces the FORMAL clean-SFT evaluation protocol
# (the one that produced runs/eval_clean_sft/test_beam20.json,
#  HR@20 = 0.19832341  NDCG@20 = 0.11786798) with ONLY the following replaced:
#
#   model       runs/industrial_sft/final_checkpoint
#            -> runs/industrial_sft_shuffled_sid/final_checkpoint
#   test data   data/Amazon/test/<cat>_...csv
#            -> analysis/shuffled_sid/test.csv
#   output      runs/eval_clean_sft/...
#            -> runs/eval_shuffled_sid/...
#
# Everything else is byte-identical to the clean protocol:
#   --batch_size 8  --K 0  --seed 42  --length_penalty 0
#   --max_new_tokens 256  --num_beams 20
#   --info_file   data/Amazon/info/<cat>_...txt        <-- ORIGINAL, unchanged
# and the metric step uses the ORIGINAL calc.py unmodified.
#
# ON --info_file (corrected 2026-10-05)
#   An earlier revision swapped --info_file for a shuffled twin, claiming that
#   otherwise "constrained decoding would restore the original item<->SID mapping
#   and undo the intervention". THAT CLAIM WAS WRONG and is withdrawn.
#   Audited facts (analysis/audit_info_file_equivalence.py):
#     * evaluate.py reads the info file ONLY to obtain field 0 (the SID string)
#       and build the constrained-decoding trie. The item_id field is NEVER read;
#       the title field is built but never consumed.
#     * the trie is a function of the SID CODEBOOK alone. The strict shuffle
#       changes item -> SID assignment while keeping the codebook identical, so
#       original and shuffled info files yield the SAME trie:
#         9684 prefix keys, identical key sets, 0 differing allowed-token sets
#         (121 keys differ only in set-iteration ORDER, which the advanced-index
#         assignment at LogitProcessor.py:68 ignores), 3670 identical EOS
#         terminal nodes, identical legal-SID multiset.
#     * calc.py uses the info file only as a membership set -- also identical.
#   The original info file is therefore used, which REMOVES one changed item from
#   the evaluation protocol. The shuffled info is kept for consistency auditing
#   but is not required. INFO_MODE=shuffled uses it anyway; results must match.
#
# --item_meta_path is not an evaluate.py argument at all; evaluate.py needs only
# model + test CSV + info file. Nothing else is required.
#
# This script does NOT run by itself; execute it explicitly.
# =============================================================================

set -euo pipefail

export WANDB_MODE=disabled
export HF_ENDPOINT=https://hf-mirror.com

# INFO_MODE: "original" (default, matches the clean protocol exactly) or
#            "shuffled" (audit only -- the trie is provably identical).
INFO_MODE=${INFO_MODE:-original}

# portable paths: PROJECT_ROOT / RUN_ROOT / DATA_ROOT / CATEGORY / PY
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# ---- clean (reference) paths ------------------------------------------------
CLEAN_MODEL="$RUN_ROOT/industrial_sft/final_checkpoint"
CLEAN_OUT="$RUN_ROOT/eval_clean_sft"
CLEAN_INFO="$INFO"
CLEAN_TEST="$TEST"

# ---- intervention paths -----------------------------------------------------
SHUF_MODEL="$RUN_ROOT/industrial_sft_shuffled_sid/final_checkpoint"
SHUF_TEST="$PROJECT_ROOT/analysis/shuffled_sid/test.csv"
SHUF_INFO="$DATA_ROOT/info/${CATEGORY}_shuffled.info.txt"
OUT="$RUN_ROOT/eval_shuffled_sid"
RESULT="$OUT/test_beam20.json"

# ---- which info file (see the header note) ----------------------------------
case "$INFO_MODE" in
  original) EVAL_INFO=$CLEAN_INFO ;;
  shuffled) EVAL_INFO=$SHUF_INFO ;;
  *) echo "INFO_MODE must be 'original' or 'shuffled' (got '$INFO_MODE')" >&2; exit 2 ;;
esac

# ---- protocol constants (identical to clean) --------------------------------
BATCH_SIZE=8
NUM_BEAMS=20
MAX_NEW_TOKENS=256
LENGTH_PENALTY=0
SEED=42
K=0

# -----------------------------------------------------------------------------
# Preflight -- refuses to start on any inconsistency. No GPU work here.
# -----------------------------------------------------------------------------
echo "=== preflight ==="
fail=0
chk() { if [ -e "$1" ]; then printf '  OK      %s\n' "$1"; else printf '  MISSING %s\n' "$1"; fail=1; fi; }

chk "$PROJECT_ROOT/evaluate.py"
chk "$PROJECT_ROOT/calc.py"
chk "$SHUF_MODEL"
chk "$SHUF_MODEL/config.json"
chk "$SHUF_MODEL/model.safetensors"
chk "$SHUF_TEST"
chk "$CLEAN_INFO"
chk "$EVAL_INFO"
chk "$CLEAN_OUT/test_beam20.json"

echo "  INFO_MODE = $INFO_MODE  ->  --info_file $EVAL_INFO"

# must not clobber the clean evaluation
if [ -e "$RESULT" ]; then
    echo "  REFUSE  output already exists: $RESULT"
    fail=1
else
    echo "  OK      output dir is free: $OUT"
fi
if [ -e "$CLEAN_OUT/test_beam20.json" ]; then
    echo "  OK      clean eval intact: $CLEAN_OUT/test_beam20.json"
fi

python - "$CLEAN_INFO" "$SHUF_INFO" "$EVAL_INFO" "$SHUF_TEST" "$SHUF_MODEL" "$PROJECT_ROOT/analysis/shuffled_sid/${CATEGORY}.index.json" <<'PY' || fail=1
import json, re, sys
clean_info, shuf_info, eval_info, shuf_test, model, shuf_index = sys.argv[1:7]
SRE = re.compile(r"<[^<>]+>")
ok = True

def lines(p):
    return [l for l in open(p, encoding="utf-8").read().splitlines() if l.strip()]

# 1. The info file supplies ONLY the legal SID codebook (see the header note).
#    Verify the codebook -- not the item<->SID assignment -- is identical between
#    the original and the shuffled info file, so the decoding trie cannot differ.
ci, si = lines(clean_info), lines(shuf_info)
tc = sorted({t for l in ci for t in SRE.findall(l.split("\t")[0])})
ts = sorted({t for l in si for t in SRE.findall(l.split("\t")[0])})
sc = sorted({l.split("\t")[0].strip() for l in ci})
ss = sorted({l.split("\t")[0].strip() for l in si})
print(f"  {'OK ' if len(ci)==len(si) else 'FAIL'} info lines: clean={len(ci)} shuffled={len(si)}")
print(f"  {'OK ' if tc==ts else 'FAIL'} info SID token vocab identical: {len(ts)} tokens")
print(f"  {'OK ' if sc==ss else 'FAIL'} legal SID codebook identical: {len(ss)} distinct SIDs")
if not (len(ci) == len(si) and tc == ts and sc == ss):
    ok = False
# the info file actually handed to evaluate.py must be one of the two audited ones
print(f"  {'OK ' if eval_info in (clean_info, shuf_info) else 'FAIL'} --info_file is an audited file")

# 2. test CSV row count
import csv
rows = list(csv.DictReader(open(shuf_test, encoding="utf-8")))
print(f"  {'OK ' if len(rows)==4533 else 'FAIL'} shuffled test rows = {len(rows)} (expect 4533)")
if len(rows) != 4533:
    ok = False

# 3. model tokenizer vocab size must be 152225
try:
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(model)
    n = len(tk)
    print(f"  {'OK ' if n==152225 else 'FAIL'} tokenizer vocab_size = {n} (expect 152225)")
    if n != 152225:
        ok = False
except Exception as e:
    print(f"  FAIL  tokenizer load: {type(e).__name__}: {e}")
    ok = False

# 4. every shuffled-test SID must exist in the shuffled index (trie consistency)
idx = json.load(open(shuf_index, encoding="utf-8"))
sids = {"".join(v) for v in idx.values()}
bad = sum(1 for r in rows
          if r["item_sid"] not in sids
          or any(h not in sids for h in eval(r["history_item_sid"])))
print(f"  {'OK ' if bad==0 else 'FAIL'} shuffled test SIDs consistent with shuffled index ({bad} bad)")
if bad:
    ok = False

sys.exit(0 if ok else 1)
PY

if [ "$fail" -ne 0 ]; then
    echo "=== preflight FAILED -- not starting evaluation ==="
    exit 1
fi
echo "=== preflight OK ==="
echo

# -----------------------------------------------------------------------------
# Step 1: generate predictions -- IDENTICAL protocol to the clean run
# -----------------------------------------------------------------------------
mkdir -p "$OUT"

CUDA_VISIBLE_DEVICES=0 "$PY" -u "$PROJECT_ROOT/evaluate.py" \
    --base_model       "$SHUF_MODEL" \
    --info_file        "$EVAL_INFO" \
    --category         "$CATEGORY" \
    --test_data_path   "$SHUF_TEST" \
    --result_json_data "$RESULT" \
    --batch_size       $BATCH_SIZE \
    --K                $K \
    --seed             $SEED \
    --length_penalty   $LENGTH_PENALTY \
    --max_new_tokens   $MAX_NEW_TOKENS \
    --num_beams        $NUM_BEAMS \
    2>&1 | tee "$OUT/evaluate.log"

# -----------------------------------------------------------------------------
# Step 2: metrics -- ORIGINAL calc.py, unmodified
# -----------------------------------------------------------------------------
# item_path is used by calc.py only as a SID membership set; it must contain the
# same SIDs the generator was allowed to emit, i.e. the same file passed above.
"$PY" "$PROJECT_ROOT/calc.py" \
    --path      "$RESULT" \
    --item_path "$EVAL_INFO" \
    2>&1 | tee "$OUT/metrics.txt"

echo
echo "=== done ==="
cat "$OUT/metrics.txt"
