# Clean SFT vs Shuffled-SID SFT — Recipe Diff

**Purpose.** Verify that `scripts/sft_shuffled_sid.sh` differs from the clean SFT run
in the SID assignment / data paths only, so the shuffled-SID causal experiment has a
single primary intervention variable.

**Evidence base.** The clean recipe was reconstructed from actual run artefacts, not
from the upstream `sft.sh` template (which is a multi-GPU placeholder with
`your_model_path` / `output_dir/xxx` and would not reproduce the run):

| Source | What it establishes |
|---|---|
| `runs/industrial_sft/training_args.bin` | the exact `TrainingArguments` object of the run |
| `runs/industrial_sft/train.log` | `num_rows`, step counts, LR trajectory, token count |
| `runs/industrial_sft/config.json` | saved model config (vocab 152225, 24 layers, hidden 896) |
| `code/sft_full.sh` | the real launcher actually used for `runs/industrial_sft` |
| `code/sft.py` | how CLI flags map onto `TrainingArguments` and the datasets |
| `code/data.py` | how `SidSFTDataset` / `SidItemFeatDataset` / `FusionSeqRecDataset` build samples |

Confidence levels used below: **[CONFIRMED]** = read directly from run artefacts;
**[DERIVED]** = determined by code path / measured arithmetic, not printed by the run;
**[UNKNOWN]** = not recoverable from the available evidence.

---

## 1. Command-line arguments

| Argument | clean value | shuffled value | same? | reason |
|---|---|---|---|---|
| `--base_model` | `/root/autodl-tmp/models/Qwen2.5-0.5B` | same | ✅ same | backbone must be identical |
| `--train_file` | `data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv` | `analysis/shuffled_sid/train.csv` | ❌ **different** | **intervention** |
| `--eval_file` | `data/Amazon/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv` | `analysis/shuffled_sid/valid.csv` | ❌ **different** | **intervention** (must carry the same shuffled mapping) |
| `--output_dir` | `runs/industrial_sft` | `runs/industrial_sft_shuffled_sid` | ❌ different | write target only; protects the clean run. Not a recipe parameter |
| `--category` | `Industrial_and_Scientific` | same | ✅ same | prompt verbosity only |
| `--sid_index_path` | `data/Amazon/index/Industrial_and_Scientific.index.json` | `analysis/shuffled_sid/index.json` | ❌ **different** | **intervention** (drives new tokens, `sid2title`, `title2sid`) |
| `--item_meta_path` | `data/Amazon/index/Industrial_and_Scientific.item.json` | **same (original)** | ✅ same | titles/descriptions are item attributes, not SID assignments — must NOT change |
| `--sample` | `-1` | same | ✅ same | no subsampling; row counts must match exactly |
| `--num_epochs` | `2` | same | ✅ same | |
| `--batch_size` | `64` | same | ✅ same | |
| `--micro_batch_size` | `16` | same | ✅ same | |
| `--learning_rate` | `3e-4` | same | ✅ same | |
| `--cutoff_len` | `512` | same | ✅ same | |
| `--seed` | `42` | same | ✅ same | |
| `--train_from_scratch` | `False` | same | ✅ same | |
| `--freeze_LLM` | `False` | same | ✅ same | |

`--wandb_project` / `--wandb_run_name` / `--resume_from_checkpoint` / `--group_by_length`
were **not passed** in the clean run and are not passed here.

---

## 2. Environment variables exported by the launcher

| Variable | clean | shuffled | same? | reason |
|---|---|---|---|---|
| `WANDB_MODE` | `disabled` | same | ✅ same | identical logging behaviour |
| `NCCL_IB_DISABLE` | `1` | same | ✅ same | identical comms setup |
| `HF_ENDPOINT` | `https://hf-mirror.com` | same | ✅ same | identical model resolution |

**[CONFIRMED]** from `code/sft_full.sh`.

---

## 3. Derived `TrainingArguments` (from `training_args.bin`)

`code/sft.py:231-253` maps the CLI flags onto `TrainingArguments`. The resulting object
was read back from the run.

| Field | clean value | shuffled value | same? | provenance |
|---|---|---|---|---|
| `per_device_train_batch_size` | 16 (= `micro_batch_size`) | 16 | ✅ same | **[CONFIRMED]** bin + code |
| `per_device_eval_batch_size` | 16 | 16 | ✅ same | **[CONFIRMED]** bin |
| `gradient_accumulation_steps` | **4** (= `batch_size // micro_batch_size`, `world_size=1`) | 4 | ✅ same | **[CONFIRMED]** bin (=4) + **[DERIVED]** formula `sft.py:125` |
| effective batch size | **64** | 64 | ✅ same | **[DERIVED]** 16 × 4 |
| `num_train_epochs` | 2 | 2 | ✅ same | **[CONFIRMED]** bin |
| `learning_rate` | 3e-4 | 3e-4 | ✅ same | **[CONFIRMED]** bin |
| `lr_scheduler_type` | **linear** | linear | ✅ same | **[CONFIRMED]** bin. The custom cosine scheduler at `sft.py:68-86` is **dead code** — `optimizers=` is commented out at `sft.py:258` |
| `warmup_steps` | 20 | 20 | ✅ same | **[CONFIRMED]** bin |
| `warmup_ratio` | 0.0 | 0.0 | ✅ same | **[CONFIRMED]** bin |
| `bf16` / `fp16` | True / False | same | ✅ same | **[CONFIRMED]** bin |
| `optim` | `adamw_torch` | same | ✅ same | **[CONFIRMED]** bin |
| `adam_beta1/2`, `adam_epsilon` | 0.9 / 0.999 / 1e-8 | same | ✅ same | **[CONFIRMED]** bin (defaults) |
| `weight_decay` | 0.0 | 0.0 | ✅ same | **[CONFIRMED]** bin |
| `max_grad_norm` | 1.0 | 1.0 | ✅ same | **[CONFIRMED]** bin |
| `label_smoothing_factor` | 0.0 | 0.0 | ✅ same | **[CONFIRMED]** bin |
| `max_steps` | -1 | -1 | ✅ same | **[CONFIRMED]** bin |
| `eval_strategy` / `eval_steps` | `steps` / 0.05 | same | ✅ same | **[CONFIRMED]** bin |
| `save_strategy` / `save_steps` | `steps` / 0.05 | same | ✅ same | **[CONFIRMED]** bin |
| `save_total_limit` | 1 | 1 | ✅ same | **[CONFIRMED]** bin |
| `load_best_model_at_end` | True | True | ✅ same | **[CONFIRMED]** bin |
| `metric_for_best_model` / `greater_is_better` | `loss` / False | same | ✅ same | **[CONFIRMED]** bin |
| `logging_steps` / `logging_strategy` | 1 / `steps` | same | ✅ same | **[CONFIRMED]** bin |
| `cutoff_len` | **not a `TrainingArguments` field** | — | n/a | **[CONFIRMED]** absent from `training_args.bin`; consumed as the datasets' `max_len` — see §4 |
| `seed` | 42 | 42 | ✅ same | **[CONFIRMED]** bin |
| `data_seed` | None | None | ✅ same | **[CONFIRMED]** bin |
| `group_by_length` | False | False | ✅ same | **[CONFIRMED]** bin |
| `torch_compile` | False | False | ✅ same | **[CONFIRMED]** bin |
| `gradient_checkpointing` | False | False | ✅ same | **[CONFIRMED]** bin |
| `remove_unused_columns` | True | True | ✅ same | **[CONFIRMED]** bin |
| `report_to` | `None` (passed as `report_to=None`) | `None` | ✅ same | **[CONFIRMED]** `sft.py:252`; note the saved bin records `['tensorboard','wandb']` — see §6 |
| `logging_dir` | run-specific timestamped path | run-specific timestamped path | ⚠️ differs | auto-generated by Trainer from the run timestamp; carries no semantic content. **[UNKNOWN]** whether it has any effect given `report_to=None` — treated as a harmless artefact |
| `run_name` | `''` | `''` | ✅ same | **[CONFIRMED]** bin |

---

## 4. Code-level settings not expressible as CLI flags

These come from `code/sft.py` and apply to both arms automatically because **the same
`sft.py` is used unchanged**.

| Setting | value | same? | provenance |
|---|---|---|---|
| model dtype at load | `torch.bfloat16` | ✅ same | `sft.py:137` |
| `tokenizer.pad_token` | `eos_token` | ✅ same | `sft.py:145-146` |
| `tokenizer.padding_side` | `left` | ✅ same | `sft.py:147` |
| new tokens added | **560** | ✅ same | **[CONFIRMED]** log + measured vocab 151665 → 152225 |
| `resize_token_embeddings` | `len(tokenizer)` → 152225 | ✅ same | `sft.py:159`; both indices yield the identical 560-token vocabulary |
| `mean_resizing` | default (False) | ✅ same | log warning present in clean run; identical code path |
| train sub-datasets (order) | `[SidSFTDataset, SidItemFeatDataset, FusionSeqRecDataset]` | ✅ same | `sft.py:193-203` |
| `ConcatDataset` order | as above | ✅ same | `sft.py:203` |
| val dataset | `SidSFTDataset(eval_file, sample=sample)` | ✅ same | `sft.py:204` |
| train rows | **79,834** | must equal 79,834 | **[CONFIRMED]** log `num_rows: 79834` |
| val rows | **4,532** | must equal 4,532 | **[CONFIRMED]** log `num_rows: 4532` |
| dataset shuffle | `shuffle(seed=42)` on train, `shuffle(seed=seed)`+`shuffle(seed=42)` on val | ✅ same | `sft.py:219-221`; same row counts ⇒ identical permutation |
| `sample_frac` | 1 | ✅ same | `sft.py:217` |
| collator | `DataCollatorForSeq2Seq(pad_to_multiple_of=8, return_tensors='pt', padding=True)` | ✅ same | `sft.py:254-256` |
| callbacks | `EarlyStoppingCallback(early_stopping_patience=3)` | ✅ same | `sft.py:257` |
| `model.config.use_cache` | `False` | ✅ same | `sft.py:260` |
| `cache` setting | not set | ✅ same | — |
| gradient checkpointing | off | ✅ same | **[CONFIRMED]** bin |
| `torch.backends.cudnn.deterministic` | `True` | ✅ same | `sft.py:65` |
| `torch.backends.cudnn.benchmark` | `False` | ✅ same | `sft.py:66` |
| launch wrapper | `torchrun --nproc_per_node 1` | ✅ same | `sft_full.sh`; `WORLD_SIZE=1` ⇒ `ddp=False` ⇒ **no** gradient-accumulation division |
| steps/epoch | `floor(79834/64)` = 1247 | must equal 1247 | **[DERIVED]**; log shows 2496 steps over 2 epochs |
| `cutoff_len` → datasets' `max_len` | 512 | 512 | ✅ same | `sft.py:193-204` passes `max_len=cutoff_len` to all four datasets; `pre()` truncates with `tokens[-max_len:]`. Clean log contains **0** "exceeds max_len" warnings, i.e. no sample was truncated |

---

## 5. Data-intervention files: what changes, what must not

| File | role | clean | shuffled | action |
|---|---|---|---|---|
| `index.json` | SID vocabulary + `sid2title`/`title2sid` keys | original | `analysis/shuffled_sid/index.json` | **REPLACE** |
| `train.csv` | sequence corpus (`history_item_sid`, `item_sid`) | original | `analysis/shuffled_sid/train.csv` | **REPLACE** |
| `valid.csv` | val corpus | original | `analysis/shuffled_sid/valid.csv` | **REPLACE** |
| `item.json` (`--item_meta_path`) | `title`, `description` keyed by `item_id` | original | **original, unchanged** | **KEEP** |
| `info/*.txt` | eval-time item list | original | **original, unchanged** | **KEEP** |

**Why `--item_meta_path` must stay original.** `SidItemFeatDataset` and
`FusionSeqRecDataset` read `title`/`description` from the item file and key them by SID
taken from the index (`data.py:733-741`, `data.py:1198-1210`). Swapping only the index
therefore produces the auxiliary tasks
`sid2title: <shuffled SID> -> <the real title of that item>` and
`title2sid: <real title> -> <shuffled SID>`,
which is exactly the intended treatment. Replacing the item file as well would confound
the intervention with a title/description change.

**Item-level consistency is guaranteed.** `item_id` is unchanged everywhere; only the
SID assigned to each `item_id` moved. Because `history_item_sid` in the shuffled CSVs was
rewritten through the same `old_sid -> new_sid` map, the sequence corpus and the index
agree by construction (verified: 0 mismatches over all 45,324 rows = 36,259 + 4,532 + 4,533).

---

## 6. Residual differences (declared, not hidden)

| Item | clean | shuffled | impact |
|---|---|---|---|
| `output_dir` | `runs/industrial_sft` | `runs/industrial_sft_shuffled_sid` | none on the optimisation trajectory |
| `run_name` | `''` | `''` | none |
| `logging_dir` | timestamped path from the clean run | new timestamped path | **[UNKNOWN]** — auto-generated. `report_to=None`, so no logger consumes it. Treated as inert |
| `report_to` in saved `training_args.bin` | records `['tensorboard','wandb']` | will record `None` | bookkeeping artefact: `train.log` shows the progress-bar format consistent with `report_to=None`. Does not affect optimisation |
| wall-clock / step timing | 1532 s, 1.629 steps/s | expected similar | hardware-level, not a recipe parameter |

---

## 7. Does this satisfy "the only primary intervention variable is SID assignment"?

**Yes, with one declared scope limitation.**

- Every optimiser-visible hyperparameter is identical (§3).
- Every code-level setting is identical because `sft.py` is unchanged (§4).
- The random-stream alignment is identical: the same `seed=42`, the same row counts
  (79,834 / 4,532), and therefore the same `shuffle(seed=42)` permutation. Only the SID
  *content* of each row differs.
- The item→SID mapping is the only quantity that changes. `item_id`, `user_id`, titles,
  descriptions, sequence lengths, popularity per `item_id`, and the SID vocabulary are
  all invariant.
- The SID vocabulary being identical (same 560 tokens, same sorted order) means the
  tokenizer expansion and the newly initialised embedding rows are aligned between the
  two arms.

**Declared limitation.** The strict shuffle freezes the 31 collision-involved items at
their original SIDs, so 0.84 % of items are not intervened on
(`singleton exact-SID retention = 0.0000 %`, `all-item exact-SID retention = 0.8410 %`).
This is the deliberate price of preserving the exact SID multiset, trie structure and
collision membership. It bounds the interpretation: a null result cannot be attributed
to the 31 frozen items, and a positive result cannot be explained by them.

**[UNKNOWN] items.** No parameter in this recipe could not be reconstructed: every
optimiser-visible value in §3 is confirmed from `training_args.bin`, and every code-level
value in §4 from `sft.py`. The only object not recoverable from evidence is the exact
`logging_dir` string of the clean run's Trainer instance, which is inert here.
