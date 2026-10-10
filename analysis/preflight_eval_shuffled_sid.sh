#!/bin/bash
# Static preflight for scripts/eval_shuffled_sid.sh -- NO GPU evaluation is started.
#
# Paths come from scripts/common.sh. Every check that existed before is kept
# unchanged; only the path SOURCE changed.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../scripts/common.sh"

S="$PROJECT_ROOT/scripts/eval_shuffled_sid.sh"
SHUF_DIR_SELF="$PROJECT_ROOT/analysis/shuffled_sid"
SHUF_MODEL="$RUN_ROOT/industrial_sft_shuffled_sid/final_checkpoint"
SHUF_INFO="$DATA_ROOT/info/${CATEGORY}_shuffled.info.txt"
EVAL_OUT="$RUN_ROOT/eval_shuffled_sid/test_beam20.json"
CLEAN_EVAL_OUT="$RUN_ROOT/eval_clean_sft/test_beam20.json"
fail=0
note() { printf '  %-55s %s\n' "$1" "$2"; }

echo "=============================================================================="
echo "STATIC PREFLIGHT: eval_shuffled_sid.sh   (no evaluation is launched)"
echo "  PROJECT_ROOT = $PROJECT_ROOT"
echo "  DATA_ROOT    = $DATA_ROOT"
echo "  RUN_ROOT     = $RUN_ROOT"
echo "  CATEGORY     = $CATEGORY"
echo "=============================================================================="

echo
echo "--- 1. bash -n (shell syntax) ---"
if bash -n "$S"; then note "bash -n" "PASS"; else note "bash -n" "FAIL"; fail=1; fi

echo
echo "--- 2. line endings / shebang / exec bit ---"
if grep -q $'\r' "$S"; then note "CRLF present" "FAIL"; fail=1; else note "LF only" "PASS"; fi
head -1 "$S" | grep -q '^#!/bin/bash' && note "shebang" "PASS" || { note "shebang" "FAIL"; fail=1; }
[ -x "$S" ] && note "executable" "PASS" || { note "executable" "FAIL"; fail=1; }

echo
echo "--- 3. required paths ---"
for p in \
  "$PROJECT_ROOT/evaluate.py" \
  "$PROJECT_ROOT/calc.py" \
  "$PROJECT_ROOT/LogitProcessor.py" \
  "$SHUF_MODEL/config.json" \
  "$SHUF_MODEL/model.safetensors" \
  "$SHUF_DIR_SELF/test.csv" \
  "$SHUF_DIR_SELF/${CATEGORY}.index.json" \
  "$SHUF_INFO" \
  "$INFO" \
  "$CLEAN_EVAL_OUT"
do
  if [ -e "$p" ]; then note "$p" "PASS"; else note "$p" "FAIL"; fail=1; fi
done

echo
echo "--- 4. shuffled data properties ---"
"$PY" - "$PROJECT_ROOT" "$DATA_ROOT" "$RUN_ROOT" "$CATEGORY" <<'PY' || fail=1
import csv, json, os, re, sys
CODE, DATA_ROOT, RUN_ROOT, CAT = sys.argv[1:5]
SH = CODE + "/analysis/shuffled_sid"
MODEL = RUN_ROOT + "/industrial_sft_shuffled_sid/final_checkpoint"
BASE = CAT + "_5_2016-10-2018-11"
SRE = re.compile(r"<[^<>]+>")
ok = True

def L(p):
    return [l for l in open(p, encoding="utf-8").read().splitlines() if l.strip()]

# (a) shuffled test row count
rows = list(csv.DictReader(open(SH + "/test.csv", encoding="utf-8")))
good = len(rows) == 4533
print("  {:<55} {}".format("shuffled test row count = %d (expect 4533)" % len(rows),
                           "PASS" if good else "FAIL"))
ok &= good

# (b) tokenizer vocab size
try:
    from transformers import AutoTokenizer
    n = len(AutoTokenizer.from_pretrained(MODEL))
    good = n == 152225
    print("  {:<55} {}".format("tokenizer vocab_size = %d (expect 152225)" % n,
                               "PASS" if good else "FAIL"))
    ok &= good
except Exception as e:
    print("  {:<55} FAIL {}".format("tokenizer load", type(e).__name__))
    ok = False

# (c) shuffled index SID token vocab == clean index SID token vocab
def toks_from_index(p):
    d = json.load(open(p, encoding="utf-8"))
    return sorted({t for v in d.values() for t in (v if isinstance(v, list) else SRE.findall(v))})
a = toks_from_index(DATA_ROOT + "/index/%s.index.json" % CAT)
b = toks_from_index(SH + "/%s.index.json" % CAT)
good = a == b
print("  {:<55} {}".format("clean vs shuffled index SID tokens (%d)" % len(b),
                           "PASS" if good else "FAIL"))
ok &= good

# (d) info files: line count, token vocab, titles+ids verbatim
ci = L(DATA_ROOT + "/info/%s.txt" % BASE)
si = L(DATA_ROOT + "/info/%s_shuffled.info.txt" % CAT)
good = len(ci) == len(si)
print("  {:<55} {}".format("info line counts %d == %d" % (len(ci), len(si)),
                           "PASS" if good else "FAIL"))
ok &= good
tc = sorted({t for l in ci for t in SRE.findall(l.split("\t")[0])})
ts = sorted({t for l in si for t in SRE.findall(l.split("\t")[0])})
good = tc == ts
print("  {:<55} {}".format("info SID token vocab identical (%d)" % len(ts),
                           "PASS" if good else "FAIL"))
ok &= good
sc = sorted({l.split("\t")[0].strip() for l in ci})
ss = sorted({l.split("\t")[0].strip() for l in si})
good = sc == ss
print("  {:<55} {}".format("legal SID codebook identical (%d)" % len(ss),
                           "PASS" if good else "FAIL"))
ok &= good
# the item<->SID assignment SHOULD differ -- that is the intervention itself
nc = sum(1 for a, b in zip(ci, si) if a.split("\t")[0] != b.split("\t")[0])
good = nc > 0
print("  {:<55} {}".format("shuffled info SID assignment differs (%d lines)" % nc,
                           "PASS" if good else "FAIL"))
ok &= good

# (e) test SIDs consistent with the shuffled index
idx = json.load(open(SH + "/%s.index.json" % CAT, encoding="utf-8"))
valid = {"".join(v) for v in idx.values()}
bad = sum(1 for r in rows if r["item_sid"] not in valid
          or any(h not in valid for h in eval(r["history_item_sid"])))
good = bad == 0
print("  {:<55} {}".format("test SIDs consistent with shuffled index (%d bad)" % bad,
                           "PASS" if good else "FAIL"))
ok &= good

# (f) no candidate can be out-of-vocabulary: every info SID is a 3-token code
badlen = sum(1 for l in si if len(SRE.findall(l.split("\t")[0])) != 3)
good = badlen == 0
print("  {:<55} {}".format("all info SIDs are 3-token codes (%d bad)" % badlen,
                           "PASS" if good else "FAIL"))
ok &= good

sys.exit(0 if ok else 1)
PY

echo
echo "--- 5. output must not clobber the clean evaluation ---"
if [ -e "$EVAL_OUT" ]; then
  note "$EVAL_OUT exists" "REFUSE"; fail=1
else
  note "$EVAL_OUT is free" "PASS"
fi
if [ -e "$CLEAN_EVAL_OUT" ]; then
  note "clean eval result present" "PASS"
else
  note "clean eval result missing" "FAIL"; fail=1
fi

echo
echo "--- 6. parameter diff vs the clean protocol (only paths may differ) ---"
"$PY" - "$PROJECT_ROOT" <<'PY' || fail=1
import re, sys
PROJECT_ROOT = sys.argv[1]
S = open(PROJECT_ROOT + "/scripts/eval_shuffled_sid.sh", encoding="utf-8").read()

# protocol constants the script sets
consts = dict(re.findall(r"^(BATCH_SIZE|NUM_BEAMS|MAX_NEW_TOKENS|LENGTH_PENALTY|SEED|K)=(\S+)$",
                         S, re.M))
EXPECT = {"BATCH_SIZE": "8", "NUM_BEAMS": "20", "MAX_NEW_TOKENS": "256",
          "LENGTH_PENALTY": "0", "SEED": "42", "K": "0"}
print("  protocol constants:")
okc = True
for k, v in EXPECT.items():
    got = consts.get(k)
    tag = "PASS" if got == v else "FAIL"
    if got != v:
        okc = False
    print("    {:<16} = {:<6} expect {:<6} {}".format(k, got, v, tag))

# the evaluate.py call: match the portable invocation
#   CUDA_VISIBLE_DEVICES=0 "$PY" -u "$PROJECT_ROOT/evaluate.py"
call = re.search(r'evaluate\.py"(.*?)(?=\n\s*2>&1)', S, re.S)
if call is None:
    print("  FAIL  could not locate the evaluate.py invocation")
    sys.exit(1)
call = call.group(1)
# A value is either a double-quoted string or a bare token. Quotes are KEPT so the
# comparison is exact. A bare token must not swallow the line-continuation
# backslash that terminates every argument line, otherwise "$K \" is captured as
# "$K\" and never equals "$K".
VAL = r'("[^"]*"|[^\s\\]+)'
flags = dict(re.findall(r"--([A-Za-z_]+)\s+" + VAL, call))
print("\n  evaluate.py invocation (parsed):")
for k in sorted(flags):
    print("    --{:<18} = {}".format(k, flags[k]))

EXPECT_FLAGS = {"batch_size": "$BATCH_SIZE", "num_beams": "$NUM_BEAMS",
                "max_new_tokens": "$MAX_NEW_TOKENS", "length_penalty": "$LENGTH_PENALTY",
                "seed": "$SEED", "K": "$K", "category": '"$CATEGORY"'}
okf = True
print("\n  protocol flags:")
for k, v in EXPECT_FLAGS.items():
    got = flags.get(k)
    tag = "PASS" if got == v else "FAIL"
    if got != v:
        okf = False
    print("    --{:<18} = {:<20} expect {:<20} {}".format(k, str(got), v, tag))

EXPECT_INTERVENTION = {"base_model": '"$SHUF_MODEL"',
                       "test_data_path": '"$SHUF_TEST"',
                       "result_json_data": '"$RESULT"'}
print("\n  intervention flags:")
oki = True
for k, v in EXPECT_INTERVENTION.items():
    got = flags.get(k)
    tag = "PASS" if got == v else "FAIL"
    if got != v:
        oki = False
    print("    --{:<18} = {:<20} expect {:<20} {}".format(k, str(got), v, tag))

# --info_file must be routed through $EVAL_INFO, which defaults to the ORIGINAL
# file (the trie depends on the SID codebook only; see the script header).
got = flags.get("info_file")
good = got == '"$EVAL_INFO"'
if not good:
    oki = False
print("    --{:<18} = {:<20} expect {:<20} {}".format(
    "info_file", str(got), '"$EVAL_INFO"', "PASS" if good else "FAIL"))

# the default INFO_MODE must be "original"
m = re.search(r'INFO_MODE=\$\{INFO_MODE:-(\w+)\}', S)
mode = m.group(1) if m else None
good = mode == "original"
if not good:
    oki = False
print("    {:<21} = {:<20} expect {:<20} {}".format(
    "INFO_MODE default", str(mode), "original", "PASS" if good else "FAIL"))

present = set(flags)
extra = present - set(EXPECT_FLAGS) - set(EXPECT_INTERVENTION) - {"info_file"}
if extra:
    print("\n    UNEXPECTED EXTRA FLAGS: {}".format(sorted(extra)))
okd = not extra
print("\n  VERDICT: {}".format(
    "MATCH -- only model/data/output differ; --info_file stays the ORIGINAL"
    if (okc and okf and oki and okd) else "MISMATCH"))
sys.exit(0 if (okc and okf and oki and okd) else 1)
PY

echo
echo "--- 7. no evaluation currently running ---"
if pgrep -af "evaluate.py|calc.py" | grep -v pgrep > /dev/null; then
  echo "  WARNING  a process is already running:"; pgrep -af "evaluate.py|calc.py" | grep -v pgrep
else
  note "no evaluate.py / calc.py running" "PASS"
fi

echo
echo "--- 8. GPU state (informational; nothing is launched) ---"
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader | sed 's/^/  /'

echo
echo "=============================================================================="
if [ "$fail" -eq 0 ]; then echo "PREFLIGHT: ALL PASS"; else echo "PREFLIGHT: FAILURES PRESENT"; fi
echo "=============================================================================="
exit $fail
