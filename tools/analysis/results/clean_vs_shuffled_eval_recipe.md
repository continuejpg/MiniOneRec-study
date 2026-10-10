# Clean SFT vs Shuffled-SID SFT — Evaluation Recipe Diff

**Purpose.** Show that `scripts/eval_shuffled_sid.sh` reproduces the formal clean-SFT
beam=20 evaluation protocol with only the model / data / info-file / output paths
replaced, so the shuffled-SID causal comparison is measured under an identical
evaluation procedure.

Confidence levels: **[CONFIRMED]** = read from run artefacts or results;
**[CODE]** = determined by the actual code path; **[INFERRED]** = reasoned, not proven;
**[UNKNOWN]** = not recoverable.

---

## 0. Which script actually produced the official result?

**This was not obvious and required evidence, not name-matching.**

| Candidate | Evidence | Verdict |
|---|---|---|
| `eval_industrial.sh` (mtime Oct 4 08:52) | writes to `runs/eval_industrial/final_result_<cat>.json` and `metrics.txt` | ❌ **NOT** the formal run — different output dir and filename |
| `evaluate.sh` (upstream, May 14) | multi-GPU `split.py` / `merge.py` fan-out, `num_beams 50`, `exp_name="xxx"` placeholder | ❌ upstream template, unusable as-is |
| `runs/eval_clean_sft/` (mtime Oct 4 21:14) | contains **only** `test_beam20.json`; no `metrics.txt`, no `evaluate.log` | ✅ the formal artefact, produced by a later, differently-scoped invocation |
| `patches/grpo_*.sh` | use the identical flag set | — closest surviving template, **not** the clean run |

**The clean-SFT formal invocation is therefore not preserved as a script.**
`.bash_history` is a rolling 2000-line buffer and the Oct 4 21:14 command has been
evicted (only the two GRPO invocations survive). The recipe below is reconstructed
from `evaluate.py` (which is fully determined by its CLI), from the surviving GRPO
invocations that used the same interface, and from the artefact itself.

**Legacy artefact warning.** `runs/eval_industrial/` (HR@20 = 0.15376131) is an
**earlier, pre-prompt-unification** evaluation. `notes/experiment_summary.md` mixes
the two: its HR row is the new grid, but its NDCG@20 (0.11786798) happens to be
consistent with the new grid, while its HR@10 (0.15376131) is the **legacy**
HR@20. The HR row reproduced exactly in §5, so the summary's HR values are the
authoritative ones; treat any single number copied from that file with care.

---

## 1. Evaluation recipe, item by item

| Item | Value | Confidence | Evidence |
|---|---|---|---|
| Runner | `python -u ./evaluate.py` (single GPU, `CUDA_VISIBLE_DEVICES=0`) | **[CODE]** | `evaluate.py:54` sets `CUDA_VISIBLE_DEVICES=0`; no split/merge fan-out in the artefact path |
| `--base_model` (clean) | `/root/autodl-tmp/runs/industrial_sft/final_checkpoint` | **[CONFIRMED]** | the checkpoint being evaluated; carries the 152225-token tokenizer |
| `--base_model` (shuffled) | `/root/autodl-tmp/runs/industrial_sft_shuffled_sid/final_checkpoint` | — | intervention |
| `--test_data_path` (clean) | `data/Amazon/test/Industrial_and_Scientific_5_2016-10-2018-11.csv` | **[CONFIRMED]** | `output == test.csv item_sid` for **4533/4533** rows |
| `--test_data_path` (shuffled) | `analysis/shuffled_sid/test.csv` | — | intervention |
| `--info_file` (clean) | `data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt` | **[CODE]** | 3686 lines; `CC = 0` means every emitted candidate is present in it ⇒ this is the file that was used |
| `--info_file` (shuffled) | `data/Amazon/info/Industrial_and_Scientific_shuffled.info.txt` | — | **intervention — mandatory**, see §4 |
| `--result_json_data` (clean) | `runs/eval_clean_sft/test_beam20.json` | **[CONFIRMED]** | the artefact |
| `--result_json_data` (shuffled) | `runs/eval_shuffled_sid/test_beam20.json` | — | intervention |
| `--category` | `Industrial_and_Scientific` | **[CONFIRMED]** | prompt text in the artefact uses "industrial and scientific items" |
| `--batch_size` | **8** | **[INFERRED]** | not recorded in the artefact. `8` matches both surviving GRPO invocations and `eval_industrial.sh`; it changes only padding/throughput, **not** the per-sample decode (beam search batches are independent and `padding_side="left"`) |
| `--num_beams` | **20** | **[CONFIRMED]** | exactly 20 candidates per sample in all 4533 rows |
| `--num_return_sequences` | **20** (= `num_beams`) | **[CODE]** | `evaluate.py:173` hard-codes `num_return_sequences=num_beams`; not a CLI flag |
| `--max_new_tokens` | **256** | **[INFERRED]** | not recorded in the artefact. Matches all surviving invocations. See §3 — it is an upper bound that is never reached |
| `--length_penalty` | **0** | **[CONFIRMED]** | see §2 |
| `--seed` | **42** | **[INFERRED]** | matches all surviving invocations; irrelevant under beam search, see §3 |
| `--K` | **0** | **[INFERRED]** | matches both surviving GRPO invocations; consumed only by `EvalSidDataset`'s `K` argument, which its `pre()` never uses when `test=True` |
| `early_stopping` | **not set** | **[CODE]** | absent from `GenerationConfig` (`evaluate.py:170-180`); HF default `False`. With `num_return_sequences == num_beams` all beams finish, so it is inert |
| `do_sample` | **not set ⇒ `False`** | **[CODE]** | absent from `GenerationConfig`; this is deterministic beam search, **not** sampling — unlike the GRPO *trainer* rollout which sets `do_sample=True` |
| `constrained decoding` | **YES**, via `ConstrainedLogitsProcessor` + `prefix_allowed_tokens_fn` | **[CODE]** | `evaluate.py:135,183-189`; see §3 |
| `tokenizer` | `AutoTokenizer.from_pretrained(base_model)` | **[CODE]** | `evaluate.py:72` — loaded **from the checkpoint**, hence vocab 152225 with the 560 SID tokens already present. `evaluate.py` never calls `add_tokens` |
| SID tokens | 560 added tokens, ids 151665..152224 | **[CONFIRMED]** | measured vocab 151665 → 152225; `added_tokens.json` present in the checkpoint |
| `generation config` | constructed inline per call | **[CODE]** | `evaluate.py:170-180`; no `generation_config.json` is loaded from disk |
| device / dtype | `cuda`, `torch.bfloat16`, `device_map="auto"` | **[CODE]** | `evaluate.py:59,192-193,212` |
| test sample count | **4533** | **[CONFIRMED]** | artefact length; matches test CSV |
| HR cutoff | `[1, 3, 5, 10, 20]` after filtering `[1,3,5,10,20,50]` by `k <= num_beams` | **[CONFIRMED]** | reproduced grid matches the recorded values exactly |
| NDCG cutoff | same list | **[CONFIRMED]** | ditto |
| metric code | `calc.py` (`fire.Fire(gao)`), **unmodified** | **[CONFIRMED]** | exact transcription reproduces both targets |
| hit rule | first exact string match in the candidate list; `minID < k` | **[CONFIRMED]** | `calc.py:59-75` |
| NDCG formula | `sum 1/log(minID+2)`, then `/ (1/log(2))` at print time | **[CONFIRMED]** | reproduces 0.11786798 to 3.6e-09 |
| top-k semantics | `HR@k` credits rank `< k` (0-based), i.e. positions 1..k | **[CONFIRMED]** | grid matches |

---

## 2. `length_penalty` — the requested call-chain audit

Your search found two values (`0.0` at CLI/default, `1.0` inside `evaluate.py`).
**The two are different parameters and only one reaches `model.generate()`.**

```
evaluate.py:38-51   main(..., length_penalty: float = 0.0, ...)      <-- (A) CLI, default 0.0
evaluate.py:152-158 def evaluate(..., length_penalty=1.0, **kwargs)  <-- (B) INNER helper, default 1.0
evaluate.py:224     output = evaluate(encodings, max_new_tokens=..., num_beams=...,
                                      length_penalty=length_penalty) <-- (A) passed EXPLICITLY
evaluate.py:170-180 generation_config = GenerationConfig(..., length_penalty=length_penalty, ...)
                                                                     <-- (B) param == (A) value
evaluate.py:191-198 model.generate(..., generation_config=generation_config, ...)
```

**Conclusion: the value passed to `model.generate()` is (A), the CLI value —
`length_penalty = 0` for the formal clean run.**

Reason: the inner helper's `1.0` default is **dead**, because line 224 always
supplies `length_penalty=` explicitly from `main`'s parameter. `main`'s parameter
defaults to `0.0` and is not overridden in any surviving invocation.

**Residual [INFERRED]:** the exact clean-run command line is not preserved, so we
cannot prove `--length_penalty 0` was typed. However `0` is `main`'s default, and
every surviving invocation passes exactly `--length_penalty 0`. The value is
therefore `0` under either reading.

Note `length_penalty` scales the beam score as `score / length**lp`. With `lp = 0`
the exponent is 0 for all lengths, i.e. the penalty is fully disabled — the raw
summed log-probability ranks the beams. This is a real protocol choice, not a no-op
default: `lp = 1.0` would normalise by length and can reorder long SIDs.

---

## 3. Constrained-decoding audit

Trie construction (`evaluate.py:61-119`), all **[CODE]**:

1. `info_file` lines are split on `\t`; **field 1** is the semantic ID.
2. Each SID is wrapped as `"### Response:\n<sid>"` and tokenised.
3. `prefix_index = 3` for non-GPT2, non-Llama models (Qwen2.5) — `evaluate.py:81-84`.
   The three instruction tokens are skipped so hashing starts at the SID.
4. For every prefix of the SID token sequence, `hash_dict[get_hash(prefix)] = {next tokens}`,
   and the EOS id is appended to every SID (`evaluate.py:90`).
5. `prefix_allowed_tokens_fn_semantic` returns `hash_dict[hash(input_ids)]`, or `[]`
   if the key is absent (`evaluate.py:122-126`).
6. `ConstrainedLogitsProcessor.__call__` log-softmaxes the scores, builds an
   all-`-inf` mask, and sets `mask[allowed] = 0`, then adds the mask
   (`LogitProcessor.py:46-72`). It tracks its own `self.count` to decide whether the
   hash key is the last `prefix_index` tokens (first step) or the last `count` tokens.

**Note this is a flat prefix dictionary, not a real trie** — but because it is keyed
on the full token prefix, it enforces exactly the legal-SID paths.

### Answers to your specific questions

| Question | Answer | Evidence |
|---|---|---|
| Only legal SID token paths allowed? | **Yes.** 90660/90660 = **100.0000%** of emitted candidates are catalogue SIDs, and `CC = 0` (none absent from the info file) | **[CONFIRMED]** from the artefact |
| Does beam=20 return 20 candidates? | **Yes**, for **all 4533** samples | **[CONFIRMED]** |
| Any duplicate SIDs? | **No** — 0/4533 samples have a repeated candidate | **[CONFIRMED]** |
| Is `max_new_tokens=256` merely an upper bound? | **Yes.** A SID is 3 tokens + EOS = 4; the generator stops at EOS because `eos_token_id` is in every allowed set. Every candidate decodes to exactly one 3-token SID and nothing else | **[CONFIRMED]** — all 90660 candidates have exactly 3 SID tokens |
| EOS / SID termination | EOS is appended to each SID token sequence when the trie is built (`evaluate.py:90`), so EOS is always an allowed continuation and terminates the sequence; `eos_token_id` is also declared in `GenerationConfig` | **[CODE]** |
| Behaviour on an illegal state | If `hash_dict` has no key for the current prefix, the processor warns and forces EOS (`LogitProcessor.py:58-66`). `CC = 0` and 100% legality show this path was **not** exercised in the clean run | **[CONFIRMED]** |

---

## 4. `--info_file`: what it actually does (CORRECTED — earlier claim withdrawn)

### 4.0 Retraction

An earlier revision of this audit and of `scripts/eval_shuffled_sid.sh` claimed:

> "`evaluate.py` builds the decoding trie exclusively from `--info_file`. Therefore,
> if the shuffled evaluation reused the **original** info file, the generator would
> still be constrained to the **original** item↔SID mapping and the intervention
> would be silently undone at decoding time."

**That claim was WRONG and is withdrawn.** It confused the *SID codebook* (which the
trie encodes) with the *item↔SID assignment* (which the trie does not encode at all).
The strict shuffle preserves the codebook exactly, so the trie is identical either way.

### 4.1 Every use of `info_file` in `evaluate.py`

`info_file` appears in exactly two places: the parameter declaration
(`evaluate.py:41`) and the `open()` (`evaluate.py:61`). Nothing else in the file
references it. The lines that consume it:

```python
evaluate.py:64     semantic_ids = [line.split('\t')[0].strip() + "\n" for line in info]
evaluate.py:65     item_titles  = [line.split('\t')[1].strip() + "\n" for line in info ...]
evaluate.py:68     info_semantic = [f"### Response:\n{_}" for _ in semantic_ids]
evaluate.py:69     info_titles   = [f"### Response:\n{_}" for _ in item_titles]
evaluate.py:76/79  prefixID      = [tokenizer(_).input_ids for _ in info_semantic]
evaluate.py:77/80  prefixTitleID = [tokenizer(_).input_ids for _ in info_titles]
evaluate.py:86-99  hash_dict        <- built from prefixID        (SEMANTIC trie)
evaluate.py:102-113 hash_dict_title <- built from prefixTitleID   (never consumed)
evaluate.py:135    prefix_allowed_tokens_fn = prefix_allowed_tokens_fn_semantic
```

**Answers:**

| Question | Answer | Evidence |
|---|---|---|
| Used only to extract legal SID sequences / build the constrained-decoding trie? | **YES.** That is its entire role. | only lines 61-135 touch it |
| Does it participate in an `item_id -> SID` mapping? | **NO.** Tab-field 2 (item_id) is **never read**; `evaluate.py` accesses only fields `['0', '1']`. | measured |
| Does it supply the target? | **NO.** The target is the test CSV's `item_sid` column, via `EvalSidDataset`. | `EvalSidDataset` references to `info_file`/`index_file`/`item_file`/`indices`: **all False** |
| Does it supply the history? | **NO.** History is the CSV's `history_item_sid` column. | same |
| Does it participate in the metric? | Only in `calc.py`, as a **membership set** of field-0 values. Field 2 still unread. | `calc.py:23-30` |

Field 1 (title) is tokenised into `hash_dict_title`, which is **built and then never
used** — the active constraint function is `prefix_allowed_tokens_fn_semantic`
(`evaluate.py:135`).

### 4.2 The two info files are decoding-equivalent — measured, not argued

`analysis/audit_info_file_equivalence.py` and `analysis/audit_trie_order.py` reproduce
`evaluate.py:61-119` byte-faithfully for both files, with the real tokenizer, and
compare the resulting objects.

| Quantity | original info | shuffled info | equal? |
|---|---|---|---|
| tokenised SID sequences (`prefixID`) | 3686, length histogram `{8: 3686}` | 3686, `{8: 3686}` | ✅ |
| **`prefixID` as a MULTISET** | — | — | ✅ **identical** |
| `prefixID` as an ordered list | — | — | ❌ differs (row order only) |
| **trie prefix keys** (`hash_dict`) | **9684** | **9684** | ✅ |
| **trie key SET** | — | — | ✅ **identical** |
| **keys with a different allowed-token SET** | — | — | ✅ **0** |
| keys differing only in set-iteration ORDER | — | — | 121 (inert, see 4.3) |
| **EOS-allowing prefixes** | 3670 | 3670 | ✅ identical |
| **pure terminal nodes** (only EOS allowed) | 3670 | 3670 | ✅ identical |
| **legal whole-SID set** | 3670 distinct | 3670 distinct | ✅ identical |
| legal-SID multiset | 3686 | 3686 | ✅ identical |
| `calc.py` `item_dict` key set | 3670 | 3670 | ✅ identical |
| item_id field (tab-field 2) | never read | never read | ✅ n/a |

### 4.3 The 121 ordering differences are provably inert

`hash_dict[key]` is a Python `set` converted to a `list` (`evaluate.py:116-117`). Set
iteration order depends on insertion history, and the two info files list rows in
different orders, so 121 keys come out in a different order. It cannot matter:

1. **The consumer is a set-membership assignment.** `LogitProcessor.py:68` is
   `mask[batch_id * self._num_beams + beam_id, prefix_allowed_tokens] = 0` —
   advanced-index assignment, so the mask depends only on the *set* of indices.
   Verified empirically: a mask built from `toks` and from `reversed(toks)` satisfies
   `torch.equal(...) == True`.
2. **Running the same file twice with `PYTHONHASHSEED=0` gives identical lists**
   (`keys whose LIST differs = 0`), so the ordering is insertion/hash-order noise.
3. **All 121 keys have identical sorted token lists** — the diff printer shows
   `clean = [151969, 152113, 152147]` and `shuffled = [151969, 152113, 152147]`.

### 4.4 What does change — the prompt

The model's input is built from the **test CSV**, and that *does* change. That is the
intervention:

| | item_id | target SID | first history SID |
|---|---|---|---|
| clean test CSV | 3681 | `<a_223><b_80><c_216>` | `<a_223><b_80><c_165>` |
| shuffled test CSV | 3681 | `<a_229><b_105><c_188>` | `<a_158><b_120><c_8>` |

The decoding constraint (the SID codebook) is unchanged; the item↔SID association the
model must learn is what changed.

### 4.5 Decision

**The formal shuffled evaluation uses the ORIGINAL info file.** This removes one
changed item from the evaluation protocol: with the original file, *every* part of the
decoding and metric machinery is provably byte-identical to the clean run, and the only
differences are the model, the test CSV and the output path.

`scripts/eval_shuffled_sid.sh` therefore defaults to the original
(`INFO_MODE=original`). The shuffled info file, produced by
`analysis/build_shuffled_info.py`, is retained as a consistency audit — setting
`INFO_MODE=shuffled` must give a bit-identical result, which is itself a useful check
that the trie equivalence claim holds end to end.

---

## 5. Metric reproduction (the gate)

Two **independent** reproductions were run, both passing.

### 5a. Exact transcription of `calc.py`

| K | HR | NDCG |
|---|---|---|
| 1 | 0.06926980 | 0.06926980 |
| 3 | 0.10103684 | 0.08789727 |
| 5 | 0.12111185 | 0.09613706 |
| 10 | 0.15376131 | 0.10663781 |
| **20** | **0.19832341** | **0.11786798** |

| Quantity | computed | official target | \|diff\| | gate (< 1e-8) |
|---|---|---|---|---|
| HR@20 | 0.198323406133 | 0.198323410000 | 3.867e-09 | ✅ PASS |
| NDCG@20 | 0.117867983607 | 0.117867980000 | 3.607e-09 | ✅ PASS |

### 5b. The real `calc.py`, imported and executed unmodified

`analysis/verify_clean_eval_metrics.py` imports `calc.py` as a module and calls
`calc.gao()` directly. Its stdout:

```
20
[1, 3, 5, 10, 20]
NDCG:  [0.0692698  0.08789727 0.09613706 0.10663781 0.11786798]
HR     [0.0692698  0.10103684 0.12111185 0.15376131 0.19832341]
0
```

| Quantity | computed | official target | \|diff\| |
|---|---|---|---|
| HR@20 | 0.198323410000 | 0.198323410000 | **0.000e+00** |
| NDCG@20 | 0.117867980000 | 0.117867980000 | **0.000e+00** |

`CC = 0` confirms every emitted candidate is present in the info file. The whole
HR and NDCG grid matches `notes/experiment_summary.md` digit for digit.

**Implementation note for anyone re-deriving this.** `calc.py` accumulates NDCG with
`math.log` (**natural** log) and only converts to the log2 scale at print time via
`/ (1.0 / math.log(2))`. Taking the intermediate `sum(1/log(r+2))/N` at face value
gives 0.17004756, which is **not** the metric; dividing by `1/log(2)` gives the
correct 0.11786798. Both steps are reproduced faithfully.

---

## 5c. The shuffled-SFT model actually evaluated

The training run that produced `runs/industrial_sft_shuffled_sid` was audited too,
to confirm it is a valid control arm.

| Property | clean SFT | shuffled-SID SFT | same? |
|---|---|---|---|
| train `num_rows` | 79,834 | **79,834** | ✅ |
| valid `num_rows` | 4,532 | **4,532** | ✅ |
| new tokens added | 560 | **560** | ✅ |
| completed steps | 2,496 / 2,496 | 2,496 / 2,496 | ✅ |
| final LR | 1.21163166e-07 | **1.21163166e-07** | ✅ |
| `final_checkpoint/model.safetensors` | 988,615,712 B | **988,615,712 B** | ✅ |
| `added_tokens.json` md5 | `cd5313c1…` | **`cd5313c1…`** | ✅ |
| differing `TrainingArguments` fields | — | **none** (27/27 identical) | ✅ |

**⚠️ Declared caveat: the shuffled run was interrupted and resumed.** It logged 2180
steps, stopped, then resumed from `checkpoint-2125` and completed. This is reported
because a resumed run is not automatically a from-scratch run:

- **LR schedule restored correctly.** LR was 3.8409e-05 at the interruption (step 2180)
  and 4.4952e-05 at the resume (step 2126) — higher, because step 2126 precedes step
  2180 on a decaying schedule. That is exactly the schedule's value at 2126. Both runs
  end at the identical LR 1.21163166e-07 at step 2496, so the scheduler reconverged.
- **Loss trajectory is continuous.** Pre-interrupt last-200 mean 0.4363; post-resume
  mean 0.4039; post-resume min/max 0.3073/0.6122 — no anomaly, no spike, no collapse.
- **One artifact to be careful with:** the `train_loss` printed by the resumed segment
  (0.0600) is **not** comparable to the clean run's 0.7712. The clean value is the mean
  over all 2,496 logged steps. The shuffled run's equivalent all-step mean is **0.8027**.
  Quote that, not 0.0600.
- **Consequence:** steps 2125–2180 were retrained, so the two runs do not share a
  bit-identical optimisation trajectory. The LR reconvergence and the smooth loss make
  a material divergence unlikely, but the comparison is **not** claimed to be
  trajectory-identical.

---

## 6. Parameter diff: clean vs shuffled

| Parameter | clean | shuffled | same? | reason |
|---|---|---|---|---|
| `--base_model` | `runs/industrial_sft/final_checkpoint` | `runs/industrial_sft_shuffled_sid/final_checkpoint` | ❌ **intervention** | the model under test |
| `--test_data_path` | `data/Amazon/test/<cat>_...csv` | `analysis/shuffled_sid/test.csv` | ❌ **intervention** | carries shuffled history/target SIDs |
| `--info_file` | `data/Amazon/info/<cat>_...txt` | **same (ORIGINAL)** | ✅ **same** | see §4 — the trie depends on the SID codebook only, which the shuffle preserves exactly |
| `--result_json_data` | `runs/eval_clean_sft/test_beam20.json` | `runs/eval_shuffled_sid/test_beam20.json` | ❌ different | write target only; protects the clean result |
| `--category` | `Industrial_and_Scientific` | same | ✅ same | prompt text |
| `--batch_size` | 8 | 8 | ✅ same | padding/throughput only |
| `--K` | 0 | 0 | ✅ same | unused under `test=True` |
| `--seed` | 42 | 42 | ✅ same | inert under beam search |
| `--length_penalty` | 0 | 0 | ✅ same | beam ranking |
| `--max_new_tokens` | 256 | 256 | ✅ same | upper bound, never reached |
| `--num_beams` | 20 | 20 | ✅ same | candidate count |
| `num_return_sequences` | 20 | 20 | ✅ same | hard-coded to `num_beams` |
| `do_sample` | False | False | ✅ same | deterministic beam search |
| `early_stopping` | not set | not set | ✅ same | inert |
| `prefix_index` | 3 | 3 | ✅ same | model family |
| trie source | original info | **original info (unchanged)** | ✅ **same** | proven equivalent to the shuffled info in §4.2 |
| `calc.py` `item_path` | original info | **original info (unchanged)** | ✅ **same** | membership set, identical |
| tokenizer source | from evaluated checkpoint | same | ✅ same | both are 152225-vocab checkpoints |
| metric code | `calc.py` unmodified | same | ✅ same | |
| metric cutoffs | `[1,3,5,10,20]` | same | ✅ same | |
| `item_meta_path` | **not used by `evaluate.py`** | — | n/a | see §7 |

**Net result: exactly TWO intervention paths — the model and the test CSV — plus the
output path.** Nothing in the decoding or metric machinery differs.

---

## 7. `item_meta_path` is not part of the evaluation protocol

`evaluate.py`'s `main()` signature is
`(base_model, train_file, info_file, category, test_data_path, result_json_data,
batch_size, K, seed, length_penalty, max_new_tokens, num_beams)` — **there is no
`item_meta_path` argument.** The evaluation reads only the model, the test CSV and
the info file. The `.item.json` file is used during *training*
(`SidItemFeatDataset`, `FusionSeqRecDataset`) and by the GRPO reward, not here.

So the answer to "if the clean protocol uses item meta, keep the original" is:
it does not use it at all. Nothing to keep.

---

## 8. Residual risks

| Risk | Severity | Note |
|---|---|---|
| The shuffled-SFT run was interrupted and resumed from step 2125 | medium | documented in §5c. LR reconverged to the identical final value and the loss is continuous, but steps 2125–2180 were retrained, so the two arms do **not** share a bit-identical trajectory. |
| `train_loss` of the shuffled run is reported by HF as 0.0600 | medium | this is the resumed-segment statistic, **not** comparable to the clean 0.7712. Use the all-step mean 0.8027. |
| `--batch_size 8` and `--max_new_tokens 256` are **[INFERRED]**, not read from the artefact | low | neither can change the per-sample decode: batches are independently padded (`padding_side="left"`), and `max_new_tokens` is never reached. They change runtime only. |
| The clean invocation script was not preserved | medium | documented in §0. The recipe is reconstructed from `evaluate.py` (fully CLI-determined) plus the surviving invocations that share the same interface, and then **verified by reproducing both metrics to 0.000e+00** with the real `calc.py`. |
| `notes/experiment_summary.md` mixes two protocols | medium | its HR@10 is the legacy HR@20. Only its HR row is consistent with the formal artefact. Do not quote single numbers from it without this check. |
| The `--info_file` intervention is easy to forget | **resolved** | The earlier "must swap `--info_file`" instruction was **wrong** and has been withdrawn (§4). The original info file is used, so the decoding trie is provably identical; the shuffled info file is kept only as a consistency audit. |
| Earlier revision of this audit asserted an incorrect `--info_file` requirement | **resolved, recorded** | the wrong claim and its correction are both documented in §4.0 so the change of position is auditable |
