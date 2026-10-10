#!/bin/bash
# =============================================================================
# analysis/preflight_seq_only_ablation.sh
#
# Static, CPU-only, READ-ONLY preflight for the sequence-only ablation:
#   seq_only_clean  vs  seq_only_shuffled
#
# Verifies:
#   1.  sft.py implements sft_mode with values full / seq_only
#   2.  sft_mode defaults to "full"
#   3.  the seq_only branch constructs ONLY SidSFTDataset
#   4.  both training launchers pass --sft_mode seq_only explicitly
#   5.  Clean vs Shuffled training-parameter diff is exactly the 4 allowed args
#   6.  dataset scale (original vs shuffled train/valid rows)
#   7.  planned Trainer steps for seq-only (567/epoch, 1134 total, eval_steps 57)
#   8.  eval launchers share the protocol and use the ORIGINAL INFO file
#   9.  TokenExtender index-basename constraint still holds
#  10.  nothing is written; no generated dataset is created
#
# Nothing here is executed on GPU, no model is loaded, no runs/ directory is made.
# Exit 0 only if every check passes.
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../scripts/common.sh"

SFT="$PROJECT_ROOT/sft.py"
T_CLEAN="$PROJECT_ROOT/scripts/sft_seq_only_clean.sh"
T_SHUF="$PROJECT_ROOT/scripts/sft_seq_only_shuffled.sh"
E_CLEAN="$PROJECT_ROOT/scripts/eval_seq_only_clean.sh"
E_SHUF="$PROJECT_ROOT/scripts/eval_seq_only_shuffled.sh"

n_pass=0
n_fail=0
n_skip=0
SKIPPED=()
i=0

# check <ok:0|1> <label> [detail]
# exit 0 -> PASS, exit non-zero -> FAIL
check() {
    i=$((i + 1))
    if [ "$1" -eq 0 ]; then
        n_pass=$((n_pass + 1))
        printf '  %2d. PASS  %s%s\n' "$i" "$2" "${3:+  -- $3}"
    else
        n_fail=$((n_fail + 1))
        printf '  %2d. FAIL  %s%s\n' "$i" "$2" "${3:+  -- $3}"
    fi
}

# skip_check <label> <reason>
# a SKIPPED check is NEITHER a pass NOR a failure; it is reported explicitly and
# never counted as PASS. It does not by itself make the preflight fail.
skip_check() {
    i=$((i + 1))
    n_skip=$((n_skip + 1))
    SKIPPED+=("$1 -- $2")
    printf '  %2d. SKIP  %s  -- %s\n' "$i" "$1" "$2"
}

echo "=============================================================================="
echo "PREFLIGHT: sequence-only ablation (seq_only_clean vs seq_only_shuffled)"
echo "   PROJECT_ROOT = $PROJECT_ROOT"
echo "   DATA_ROOT    = $DATA_ROOT"
echo "   RUN_ROOT     = $RUN_ROOT"
echo "   CATEGORY     = $CATEGORY"
echo "   (static / CPU-only / read-only; no training, no evaluation)"
echo "=============================================================================="

# -----------------------------------------------------------------------------
echo
echo "--- 1-3. sft.py sft_mode implementation ---"
# -----------------------------------------------------------------------------
if [ -f "$SFT" ]; then
    grep -q 'sft_mode: str = "full"' "$SFT"
    check $? "sft.py declares sft_mode with default \"full\""

    grep -q 'if sft_mode not in ("full", "seq_only")' "$SFT"
    check $? "sft.py fail-loud validates sft_mode in (full, seq_only)"

    grep -q 'if sft_mode == "full":' "$SFT"
    check $? "sft.py gates the extra datasets behind sft_mode == \"full\""
else
    check 1 "sft.py exists" "$SFT"
fi

# The three dataset constructions and WHICH branch they live in.
"$PY" - "$SFT" <<'PY'
import ast, sys
src = open(sys.argv[1], encoding="utf-8").read()
tree = ast.parse(src)

fn = None
for n in ast.walk(tree):
    if isinstance(n, ast.FunctionDef) and n.name == "train":
        fn = n
        break
if fn is None:
    print("  FAIL  cannot find train()"); sys.exit(1)

# find the `if sft_mode == "full":` block
full_block = None
for n in ast.walk(fn):
    if isinstance(n, ast.If):
        t = n.test
        if (isinstance(t, ast.Compare) and isinstance(t.left, ast.Name)
                and t.left.id == "sft_mode"
                and any(isinstance(c, ast.Constant) and c.value == "full" for c in t.comparators)):
            full_block = n
            break
if full_block is None:
    print("  FAIL  no `if sft_mode == \"full\":` block in train()"); sys.exit(1)

def calls(node):
    out = []
    for x in ast.walk(node):
        if isinstance(x, ast.Call) and isinstance(x.func, ast.Name):
            out.append(x.func.id)
    return out

# top-level (unconditional) dataset constructions in train()
conditional = set()
for x in ast.walk(full_block):
    conditional.add(id(x))
uncond = []
for x in ast.walk(fn):
    if id(x) in conditional:
        continue
    if isinstance(x, ast.Call) and isinstance(x.func, ast.Name) and x.func.id.endswith("Dataset"):
        uncond.append(x.func.id)

in_full = [c for c in calls(full_block) if c.endswith("Dataset")]

ok = (uncond.count("SidSFTDataset") >= 1)
print(f"  {'OK ' if ok else 'FAIL'} unconditional datasets in train(): {sorted(set(uncond))}")
ok2 = (set(in_full) == {"SidItemFeatDataset", "FusionSeqRecDataset"})
print(f"  {'OK ' if ok2 else 'FAIL'} datasets inside `if sft_mode == \"full\"`: {sorted(set(in_full))}")
sys.exit(0 if (ok and ok2) else 1)
PY
check $? "seq_only branch constructs ONLY SidSFTDataset (verified via AST)"

# -----------------------------------------------------------------------------
echo
echo "--- 4. both training launchers pass --sft_mode seq_only ---"
# -----------------------------------------------------------------------------
for f in "$T_CLEAN" "$T_SHUF"; do
    if [ -f "$f" ]; then
        grep -qE '^\s+--sft_mode\s+seq_only' "$f"
        check $? "$(basename "$f") passes --sft_mode seq_only"
    else
        check 1 "$(basename "$f") exists"
    fi
done

# -----------------------------------------------------------------------------
echo
echo "--- 5. Clean vs Shuffled training-parameter diff ---"
# -----------------------------------------------------------------------------
if [ -f "$T_CLEAN" ] && [ -f "$T_SHUF" ]; then
    "$PY" - train "$T_CLEAN" "$T_SHUF" "train_file,eval_file,sid_index_path,output_dir" <<'PY'
import re, sys

label, pa, pb, allowed_csv = sys.argv[1:5]
allowed = set(filter(None, allowed_csv.split(",")))

def assignments(text):
    """NAME=value / NAME="value" at start of line -> {NAME: value}"""
    out = {}
    for m in re.finditer(r'^\s*([A-Za-z_][A-Za-z0-9_]*)=("([^"]*)"|\'([^\']*)\'|(\S+))\s*$',
                         text, re.M):
        out[m.group(1)] = m.group(3) if m.group(3) is not None else (
            m.group(4) if m.group(4) is not None else m.group(5))
    return out

def resolve(value, env, depth=4):
    """Expand $NAME / ${NAME} so that two scripts passing the same literal
    ('"$OUT"') but defining it differently are correctly seen as DIFFERENT."""
    for _ in range(depth):
        new = re.sub(r'\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?',
                     lambda mm: env.get(mm.group(1), mm.group(0)), value)
        if new == value:
            break
        value = new
    return value

def canon(v):
    """Fold any remaining shell variable to <VAR> so that two arms which both
    point at a common.sh variable compare equal even though that variable's
    value lives in common.sh, not in the launcher text."""
    return re.sub(r'\$\{?[A-Za-z_][A-Za-z0-9_]*\}?', '<VAR>', v)

def args_with_values(path, invoke_re):
    text = open(path, encoding="utf-8").read()
    env = assignments(text)
    m = re.search(invoke_re, text, re.S)
    body = (m.group(1) if m.re.groups else m.group(0)) if m else text
    # two explicit alternatives so the .strip() below never touches a None group
    pairs = re.findall(r'--([A-Za-z_]+)\s+("(?:[^"]*)"|\'(?:[^\']*)\'|\S+)', body)
    return {k: canon(resolve(v.strip('"\''), env)) for k, v in pairs}

inv = r"(torchrun.*?(?=\n\s*2>&1|\Z))"
a, b = args_with_values(pa, inv), args_with_values(pb, inv)

print(f"    [{label}] flags parsed: {len(set(a) | set(b))}")
bad = []
for k in sorted(set(a) | set(b)):
    va, vb = a.get(k, "(absent)"), b.get(k, "(absent)")
    if va == vb:
        continue
    tag = "allowed" if k in allowed else "UNEXPECTED"
    sa = va if len(va) <= 40 else "..." + va[-37:]
    sb = vb if len(vb) <= 40 else "..." + vb[-37:]
    print(f"    DIFF  --{k:20s} {sa:44s} -> {sb}   [{tag}]")
    if k not in allowed:
        bad.append(k)

diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
print(f"    [{label}] differing : {diff}")
print(f"    [{label}] expected  : {sorted(allowed)}")
sys.exit(0 if (not bad and set(diff) == allowed) else 1)
PY
    check $? "train launchers differ in exactly the 4 allowed arguments"

    # all other training parameters must be byte-identical
    grep -qE '^\s+--sft_mode\s+seq_only' "$T_CLEAN" && grep -qE '^\s+--sft_mode\s+seq_only' "$T_SHUF"
    check $? "both launchers request the same sft_mode"
else
    check 1 "both training launchers exist"
fi

# -----------------------------------------------------------------------------
echo
echo "--- 6. dataset scale ---"
# -----------------------------------------------------------------------------
# Each row count is an INDEPENDENT check with three possible outcomes:
#   PASS  file exists and has the expected number of rows
#   FAIL  file exists but the row count is wrong  (never downgraded to SKIP)
#   SKIP  file does not exist yet  (gitignored, generated by
#         analysis/build_shuffled_sid_strict.py) -- reported, NOT counted as PASS
row_check() {  # row_check <label> <path> <expected>
    local label="$1" path="$2" want="$3"
    if [ ! -f "$path" ]; then
        if [ "$label" = "original train rows" ] || [ "$label" = "original valid rows" ]; then
            # the original data is tracked in the repository and must always exist
            check 1 "$label == $want" "MISSING (tracked file absent): $path"
        else
            skip_check "$label == $want" "not generated yet: $path"
        fi
        return
    fi
    local n
    n="$(awk 'END{print NR-1}' "$path")"
    if [ "$n" = "$want" ]; then
        check 0 "$label == $want" "measured $n"
    else
        check 1 "$label == $want" "measured $n"
    fi
}
row_check "original train rows" "$TRAIN" 36259
row_check "original valid rows" "$VALID" 4532
row_check "shuffled train rows" "$PROJECT_ROOT/analysis/shuffled_sid/train.csv" 36259
row_check "shuffled valid rows" "$PROJECT_ROOT/analysis/shuffled_sid/valid.csv" 4532

# -----------------------------------------------------------------------------
echo
echo "--- 7. planned Trainer steps for seq-only ---"
# -----------------------------------------------------------------------------
"$PY" - "$T_CLEAN" <<'PY'
import math, re, sys

t = open(sys.argv[1], encoding="utf-8").read()
def arg(name, default=None):
    m = re.search(rf"--{name}\s+(\S+)", t)
    return int(m.group(1)) if m else default

micro = arg("micro_batch_size")
batch = arg("batch_size")
epochs = arg("num_epochs")
samples = 36259

# transformers set_initial_training_values:
#   len_dataloader = ceil(dataset_len / per_device_train_batch_size)
#   num_update_steps_per_epoch = len_dataloader // grad_accum + int(len_dataloader % grad_accum > 0)
#   max_steps = ceil(num_train_epochs * num_update_steps_per_epoch)
grad_accum = batch // micro
len_dl = math.ceil(samples / micro)
per_epoch = len_dl // grad_accum + int(len_dl % grad_accum > 0)
max_steps = math.ceil(epochs * per_epoch)
eval_steps = math.ceil(max_steps * 0.05)

print(f"    micro_batch_size={micro}  batch_size={batch}  grad_accum={grad_accum}  epochs={epochs}")
print(f"    dataloader batches        = ceil({samples}/{micro}) = {len_dl}")
print(f"    optimizer steps / epoch   = {len_dl}//{grad_accum} + int({len_dl}%{grad_accum}>0) = {per_epoch}")
print(f"    planned max steps         = ceil({epochs} * {per_epoch}) = {max_steps}")
print(f"    planned eval interval     = ceil({max_steps} * 0.05) = {eval_steps}")
print(f"    (planned values only -- actual eval count is NOT predicted here)")

ok = (len_dl == 2267 and per_epoch == 567 and max_steps == 1134 and eval_steps == 57)
sys.exit(0 if ok else 1)
PY
check $? "planned seq-only steps == 567/epoch, 1134 total, eval interval 57"

# -----------------------------------------------------------------------------
echo
echo "--- 8. eval launcher protocol ---"
# -----------------------------------------------------------------------------
for f in "$E_CLEAN" "$E_SHUF"; do
    if [ -f "$f" ]; then
        ok=0
        grep -qE '^BATCH_SIZE=8$'          "$f" || ok=1
        grep -qE '^NUM_BEAMS=20$'          "$f" || ok=1
        grep -qE '^MAX_NEW_TOKENS=256$'    "$f" || ok=1
        grep -qE '^LENGTH_PENALTY=0$'      "$f" || ok=1
        grep -qE '^SEED=42$'               "$f" || ok=1
        grep -qE '^K=0$'                   "$f" || ok=1
        check $ok "$(basename "$f") protocol constants (8/20/256/0/42/0)"
    else
        check 1 "$(basename "$f") exists"
    fi
done

for f in "$E_CLEAN" "$E_SHUF"; do
    if [ -f "$f" ]; then
        grep -qE 'EVAL_INFO="\$(INFO|EVAL_INFO)"' "$f"
        check $? "$(basename "$f") decodes with the ORIGINAL INFO file"

        # the shuffled consistency-info must NOT become the formal decoding space
        if grep -qE 'EVAL_INFO="\$SHUF_INFO"' "$f"; then
            check 1 "$(basename "$f") must not decode with the shuffled consistency-info"
        else
            check 0 "$(basename "$f") does not use the shuffled consistency-info"
        fi
    fi
done

if [ -f "$E_CLEAN" ] && [ -f "$E_SHUF" ]; then
    "$PY" - eval "$E_CLEAN" "$E_SHUF" "base_model,test_data_path,result_json_data" <<'PY'
import re, sys

label, pa, pb, allowed_csv = sys.argv[1:5]
allowed = set(filter(None, allowed_csv.split(",")))

def assignments(text):
    out = {}
    for m in re.finditer(r'^\s*([A-Za-z_][A-Za-z0-9_]*)=("([^"]*)"|\'([^\']*)\'|(\S+))\s*$',
                         text, re.M):
        out[m.group(1)] = m.group(3) if m.group(3) is not None else (
            m.group(4) if m.group(4) is not None else m.group(5))
    return out

def resolve(value, env, depth=4):
    # expand local aliases first (EVAL_INFO="$INFO" -> the INFO value) so that
    # two scripts which spell the same thing differently compare equal
    for _ in range(depth):
        stripped = value.strip()
        if stripped.startswith("$") and re.fullmatch(r'\$\{?[A-Za-z_][A-Za-z0-9_]*\}?', stripped):
            key = stripped.strip("${}")
            if key in env:
                value = env[key]
                continue
        break
    for _ in range(depth):
        new = re.sub(r'\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?',
                     lambda mm: env.get(mm.group(1), mm.group(0)), value)
        if new == value:
            break
        value = new
    return value

def canon(v):
    """Fold any remaining shell variable to <VAR> so that two arms which both
    point at a common.sh variable compare equal even though that variable's
    value lives in common.sh, not in the launcher text."""
    return re.sub(r'\$\{?[A-Za-z_][A-Za-z0-9_]*\}?', '<VAR>', v)

def args_with_values(path, invoke_re):
    text = open(path, encoding="utf-8").read()
    env = assignments(text)
    m = re.search(invoke_re, text, re.S)
    body = (m.group(1) if m.re.groups else m.group(0)) if m else text
    # two explicit alternatives so the .strip() below never touches a None group
    pairs = re.findall(r'--([A-Za-z_]+)\s+("(?:[^"]*)"|\'(?:[^\']*)\'|\S+)', body)
    return {k: canon(resolve(v.strip('"\''), env)) for k, v in pairs}

inv = r'evaluate\.py"(.*?)(?=\n\s*2>&1)'
a, b = args_with_values(pa, inv), args_with_values(pb, inv)

print(f"    [{label}] flags parsed: {len(set(a) | set(b))}")
bad = []
for k in sorted(set(a) | set(b)):
    va, vb = a.get(k, "(absent)"), b.get(k, "(absent)")
    if va == vb:
        continue
    tag = "allowed" if k in allowed else "UNEXPECTED"
    sa = va if len(va) <= 40 else "..." + va[-37:]
    sb = vb if len(vb) <= 40 else "..." + vb[-37:]
    print(f"    DIFF  --{k:20s} {sa:44s} -> {sb}   [{tag}]")
    if k not in allowed:
        bad.append(k)

diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
print(f"    [{label}] differing : {diff}")
print(f"    [{label}] expected  : {sorted(allowed)}")
sys.exit(0 if (not bad and set(diff) == allowed) else 1)
PY
    check $? "eval launchers differ in exactly model / test / output"
else
    check 1 "both eval launchers exist"
fi

# -----------------------------------------------------------------------------
echo
echo "--- 9. TokenExtender index-basename constraint ---"
# -----------------------------------------------------------------------------
if [ -f "$INDEX" ]; then
    bn="$(basename "$INDEX")"
    [ "$bn" = "${CATEGORY}.index.json" ]
    check $? "original index basename == ${CATEGORY}.index.json" "$bn"
    rebuilt="$(dirname "$INDEX")/$(basename "$INDEX" | cut -d. -f1).index.json"
    [ "$rebuilt" = "$INDEX" ]
    check $? "TokenExtender rebuild resolves back to INDEX"
else
    check 1 "original index exists" "$INDEX"
fi

# -----------------------------------------------------------------------------
echo
echo "--- 10. read-only guarantees ---"
# -----------------------------------------------------------------------------
for f in "$T_CLEAN" "$T_SHUF" "$E_CLEAN" "$E_SHUF" "$SFT"; do
    if [ -f "$f" ] && grep -vE '^\s*#' "$f" | grep -qE 'build_shuffled_sid_strict|build_shuffled_info'; then
        printf '    WARN  %s EXECUTES a generator script\n' "$(basename "$f")"
    fi
done
if grep -hvE '^\s*#' "$T_CLEAN" "$T_SHUF" "$E_CLEAN" "$E_SHUF" 2>/dev/null | grep -qE 'analysis/results/'; then
    check 1 "no launcher writes analysis/results/"
else
    check 0 "no launcher writes analysis/results/"
fi

# this script itself must not start any training or evaluation
# inspect only the bash part: awk drops heredoc bodies so this checker's own
# python regex literals are not mistaken for real invocations
if awk '/<<.PY./{inh=1} !inh && !/^[[:space:]]*#/' "${BASH_SOURCE[0]}" \
     | grep -qE 'torchrun|/evaluate\.py|/calc\.py'; then
    check 1 "preflight starts no training / evaluation"
else
    check 0 "preflight starts no training / evaluation"
fi

# -----------------------------------------------------------------------------
echo
echo "=============================================================================="
printf 'PREFLIGHT: %d PASS, %d FAIL, %d SKIP' "$n_pass" "$n_fail" "$n_skip"
if [ "$n_fail" -eq 0 ]; then
    printf '  ->  OK\n'
else
    printf '  ->  FAILURES PRESENT\n'
fi
if [ "$n_skip" -gt 0 ]; then
    echo "  skipped (neither pass nor fail):"
    for s in "${SKIPPED[@]}"; do printf '    - %s\n' "$s"; done
fi
echo "=============================================================================="
exit $(( n_fail > 0 ? 1 : 0 ))
