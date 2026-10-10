# Reproducibility

Everything needed to rebuild, re-evaluate and audit this repository — plus an
explicit list of what **cannot** be reproduced from Git alone.

---

## 1. Environment

Verified on the machine that produced every number in
[`rl_posttraining_results.md`](rl_posttraining_results.md).

| component | version |
|---|---|
| Python | 3.10.8 (miniconda `base`) |
| torch | 2.6.0+cu118 |
| transformers | 4.57.1 |
| trl | 0.24.0 |
| datasets | 4.2.0 |
| accelerate | 1.10.1 |
| numpy | 2.2.6 |
| pandas | 2.2.2 |
| fire | 0.7.1 |
| CUDA | 11.8 |
| GPU | 1× NVIDIA GeForce RTX 4090 (24,564 MiB) |

`peft` is **not installed** on the reference machine and is not required.

### 1.1 Two environment workarounds that are easy to miss

1. **protobuf / tensorboard clash.**
   `transformers.integrations.integration_utils` imports
   `torch.utils.tensorboard`, which raises
   `TypeError: Descriptors cannot be created directly` with protobuf 6.x.
   Workaround (protobuf's own documented option #2):

   ```bash
   export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
   ```

2. **inert `wandb` stub.** Runs are launched with `WANDB_MODE=disabled`, but the
   import path still touches `wandb`. A permissive stub is placed on
   `PYTHONPATH` (`tools/letter_stage1/sitecustomize.py`):

   ```bash
   export PYTHONPATH=/tmp/s3/guard
   ```

   Both are environment-only; no training code depends on them.

### 1.2 Disk

The reference run needed ~66 GB used / 35 GB free. Each GRPO checkpoint is
~1.98 GB (`model.safetensors`) plus optimizer state in `checkpoint-*/`.

---

## 2. Data

Amazon **Industrial and Scientific**, 5-core, 2016-10 → 2018-11.

| file | role | rows |
|---|---|---|
| `data/Amazon/train/..._5_2016-10-2018-11.csv` | train | 36,259 |
| `data/Amazon/valid/..._5_2016-10-2018-11.csv` | **validation** | 4,532 |
| `data/Amazon/test/..._5_2016-10-2018-11.csv` | **test** | 4,533 |
| `data/Amazon/info/..._5_2016-10-2018-11.txt` | SID ↔ title ↔ item_id | 3,686 |
| `data/Amazon/index/<CAT>.index.json` | SID → level tokens | — |
| `data/Amazon/index/<CAT>.item.json` | item metadata | — |

> The **validation and test splits differ** and their metrics must never be
> mixed. All runs before R2 were evaluated on **test**; the three-way comparison
> in the results doc is on **validation**.

### 2.1 Semantic ID

3 levels, level cardinality **48 / 256 / 256**, **3,686 items → 3,670 unique
SIDs** (15 collision groups, 31 items). `INDEX` must be named
`<stem>.index.json` — `sft.py`'s `TokenExtender` rebuilds the filename from the
stem, so renaming it breaks the run. `scripts/common.sh` documents this.

`TokenExtender` adds every `<x_N>` as a **single atomic token**; the tokenizer
becomes 152,225 entries. Constrained decoding depends on this (a multi-token SID
would make the trie keys wrong).

### 2.2 Preprocessing

```bash
bash convert_dataset.sh          # raw → csv splits
bash convert_dataset_gpr.py      # (upstream) GPR variant
```

The frozen GRPO subsets are already committed:

| file | samples |
|---|---|
| `splits/grpo_seq_10k.json` | 10,000 `seq_rec` |
| `splits/grpo_seqtitle_1k.json` | 1,000 `seqtitle2sid` |

Training set = those two subsets + the **full** `RLTitle2SidDataset`
(`title2sid` 3,646 + `description2sid` 2,870) = **17,516 samples**.

---

## 3. Launchers

All launchers source `scripts/common.sh`, which resolves
`PROJECT_ROOT / RUN_ROOT / DATA_ROOT / CATEGORY / PY / BASE / TRAIN / VALID /
TEST / INFO / ITEM_META / INDEX / SPLITS`. **`RUN_ROOT` defaults to
`$PROJECT_ROOT/runs`** — on the reference machine `PROJECT_ROOT` is
`/root/autodl-tmp/code`, so GRPO outputs land in
`/root/autodl-tmp/code/runs/...`, *not* `/root/autodl-tmp/runs/...`. Older runs
predate `common.sh` and live in `/root/autodl-tmp/runs/...`. This is the single
most common source of "file not found" when re-running evaluation.

| purpose | command |
|---|---|
| SFT (best model) | `bash sft_full.sh` |
| SFT, LETTER SID variant | `bash sft_letter_full.sh` |
| SFT, content-only SID variant | `bash sft_content_only_full.sh` |
| GRPO preflight | `python patches/preflight_grpo.py` |
| GRPO, optimized 0.25 ep | `bash patches/grpo_fast025.sh` |
| GRPO, earlier 0.25 ep | `bash patches/grpo_short025.sh` |
| R1 rank reward | `bash patches/grpo_rere_rank025.sh` |
| R2 dual route | `bash patches/grpo_r21.sh` |
| eval, test split (any ckpt) | `bash eval_industrial.sh` |
| eval, validation split | `bash patches/eval_grpo_r21_valid.sh` |

### 3.1 R2 launcher — three-step order matters

```bash
# 1) build the offline route cache (~900 s, GPU, read-only; NOT committed)
python rl_router.py --out splits/r21_route_cache.json

# 2) validate it (hard gates; refuses to start on any failure)
python tools/tests/r22_preflight.py

# 3) train
bash patches/grpo_r21.sh
```

`rl_router.py` routes **exactly** the `rl.py` training set (same three dataset
classes, same subset files, same seed). In **formal mode** `rl.py` aborts with
`KeyError` if the cache does not cover every training `sample_id` — it will not
silently shrink the training set. The only way to narrow the scope is to set
`R21_SMOKE_IDS` / `R21_SMOKE_LIMIT`, which are **off by default** and print a
`SMOKE mode` banner when used.

---

## 4. Seeds and evaluation parameters

| item | value |
|---|---|
| global seed | **42** |
| `num_generations` (G) | 16 |
| generation mode | `BEAM_SAMPLE` (`num_beams=16`, `do_sample=True`) |
| temperature | 1.0 |
| `train_batch_size` / `grad_accum` | 16 / 1 |
| `num_train_epochs` | 0.25 → 4,379 steps |
| `learning_rate` / `beta` | 1e-5 / 1e-3 |
| `sync_ref_model` | True |
| offline eval | `num_beams=20`, `max_new_tokens=256`, `length_penalty=0.0`, `batch_size=8` |

**Evaluation protocol is fixed:** `evaluate.py` → `calc.py`. Changing beam width
or `length_penalty` changes the metrics and invalidates comparisons.

---

## 5. Tests

CPU-only, no GPU, no training. Current status on the frozen tree:

| test | checks | result |
|---|---|---|
| `tools/tests/test_sid_depth.py` | 28 | **28 PASS / 0 FAIL** |
| `tools/tests/test_rere_reward.py` | 39 | **39 PASS / 0 FAIL** |
| `tools/tests/test_r21_dualroute.py` | 35 | **35 PASS / 0 FAIL** |
| `scripts/audit.sh` | 50 | 49 PASS / **1 FAIL** (see below) / 1 SKIP |

```bash
python tools/tests/test_sid_depth.py
python tools/tests/test_rere_reward.py
python tools/tests/test_r21_dualroute.py
bash   scripts/audit.sh
```

### 5.1 Known pre-existing failure — not introduced here

`scripts/audit.sh` reports **one** failure: a CRLF check on shell scripts that
shipped from upstream with Windows line endings:

```
data/amazon18_data_process.sh
data/amazon23_data_process.sh
rq/text2emb/amazon_text2emb.sh
rq/generate_indices_plus.sh
rq/rqkmeans_const*
```

These files are untouched upstream artifacts. The check is **left red on
purpose** — "fixing" it would create diff noise unrelated to this work. One
artifact-gate check is SKIPped (requires artifacts not present in a fresh
clone).

### 5.2 R2-specific verification drivers

These are acceptance/smoke drivers, not unit tests; they need a GPU:

| script | what it proves |
|---|---|
| `tools/tests/r20_reachability_probe.py` | oracle + model reachability at h=0/1/2 |
| `tools/tests/r20_interface_check.py` | `count_0` root cause and key sequences |
| `tools/tests/r21c_train_smoke.py` | real `rl.py`→Trainer→generate→loss→backward |
| `tools/tests/r21f_acceptance.py` | reward recomputation, suffix credit, gradients |
| `tools/tests/r21g_validate.py` | route-cache completeness (17516/17516) |
| `tools/tests/r22_preflight.py` | launcher + cache + reward gates |
| `tools/tests/stage6_rollout_order_diag.py` | proves `BEAM_SAMPLE` + sorted-desc order |

---

## 6. Artifacts and SHA256

`runs/`, `artifacts/` and `splits/r21_route_cache.json` are **not committed**.
Hashes below allow verifying a locally rebuilt copy.

| artifact | bytes | SHA256 |
|---|---|---|
| `splits/r21_route_cache.json` | 1,561,855 | `d7a7bc40e96217e416825cc2cee29b216e3443d64fcea17b5440e72975b140c3` |
| `splits/grpo_seq_10k.json` (committed) | 206,997 | `a2983d7f1fe3defb3803e79ceef4828128e9f5fc30034977e5e192ec13000919` |
| `splits/grpo_seqtitle_1k.json` (committed) | 18,988 | `e56775d11d556383708db95a8527c83424f05ea6c42a95f4301a97daabcba567` |
| P0 SFT weights | 988,615,712 | `c362ed183daeb290299a475932ee9067cc0189e8199fac3fa080b054c12bffe8` |
| optimized GRPO weights | 1,977,199,248 | `b4d527c10b3dbb018e6e3213da6713be7d2145e0c731b88bb53d2303dd2c5360` |
| R1 weights | 1,977,199,248 | `7ebdf9c151ed266d31b859eb5a217cadc0951d8c30223ad0c8af894c2319d007` |
| R2 weights | 1,977,199,248 | `a983f46b83e51f2972ee2b7d0a3dbe315c87671fc3e66ca3ba10c80ac642f978` |
| `runs/grpo_r21/train.log` | 3,095,062 | `a61dda8334948d44ac2f1374a737433bf13f9d58a4c1f39385bd82268d761f0b` |
| SFT test metrics | 196 | `c78052962ba98afac9706107abe20688399df0e2f41d41885f0766e56e82a407` |
| SFT validation metrics | 196 | `0996efcddf62436ce1f0a24229c8ed1f88404e18e26c06521c37e093e747be38` |
| old-GRPO validation metrics | 196 | `f1bbd7b83e8faa64a31c32a2031f868010c1fd064d98b5ab8995beae99ad0868` |

### 6.1 Paths

```
<PROJECT_ROOT>=/root/autodl-tmp/code        RUN_ROOT=<PROJECT_ROOT>/runs

runs/industrial_sft/final_checkpoint                 P0 SFT  (BEST)
runs/grpo_rere_rank025/{final_checkpoint,checkpoint-4379}   R1
runs/grpo_r21/{final_checkpoint,checkpoint-4379}     R2
runs/reference_full_clean_beam20/metrics.txt         P0 SFT test metrics
runs/eval_grpo_rere_rank025/metrics.txt              R1 test metrics
runs/eval_grpo_r21_valid/metrics.txt                 R2 validation metrics

/root/autodl-tmp/runs/grpo_fast025/checkpoint-4379   optimized GRPO (legacy root)
/root/autodl-tmp/runs/eval_valid_sft/metrics.txt     SFT validation metrics
/root/autodl-tmp/runs/eval_valid_grpo_old/metrics.txt old-GRPO validation metrics
```

---

## 7. What Git alone does **not** reproduce

| item | why | how to obtain |
|---|---|---|
| **Model weights** (all 4 checkpoints) | ~1–2 GB each, excluded by `.gitignore` | re-run the launcher (GPU hours) |
| **`splits/r21_route_cache.json`** | 1.5 MB generated artifact, excluded | `python rl_router.py --out splits/r21_route_cache.json` (~900 s GPU) |
| **Raw Amazon CSVs** | dataset licensing | download Amazon 2018 + `convert_dataset.sh` |
| **LETTER upstream source** | license not declared upstream | fetch separately; see `rq/letter/UPSTREAM_MANIFEST.json` |
| **Training logs** | multi-MB, excluded with `runs/` | re-run, or use the SHA256s above |
| **`artifacts/`** | excluded; exploratory JSON/PNG | re-run the probes |

### 7.1 Provenance notes

1. **old-GRPO test metrics (HR@20 0.17538054 / NDCG@20 0.10824989).**
   No `metrics.txt` exists for that run - `/root/autodl-tmp/runs/grpo_fast025/`
   keeps `eval_beam20.json` (the raw prediction artifact) and `train.log`. The
   numbers are **traceable** to
   [`notes/experiment_summary.md`](../notes/experiment_summary.md) §5, which lists
   the full grid
   (`0.06750496 / 0.09596294 / 0.11030223 / 0.14096625 / 0.17538054 / 0.10824989`)
   and states that every headline number was machine-re-derived from its artifact
   rather than copied. `analysis/verify_summary_claims.py` re-checks them.

2. **Test-split baselines vs validation baselines.** The test-split numbers in
   `notes/experiment_summary.md` and README §2 predate the validation harness and
   were produced by a different evaluation invocation than the R2-era validation
   numbers. Within each family the protocol is identical and verified; **across
   families it is not comparable.**

3. **R1 / R2 checkpoints, route cache and logs are not in Git.** See §7 table.
