# REPRODUCE.md — how to rebuild every number

> ★ **Read this first.** The **local fork** and the **remote training repo** are **not the
> same code**. `sft.py`, `data.py`, `evaluate.py` and `LogitProcessor.py` **differ**, and
> `sid_utils.py` exists only remotely. **Only the remote files produced the SFT/eval
> results.** Quoting a single commit hash would overstate reproducibility.
> Exact hashes: `SOURCE_VERSION_MANIFEST.json`.

| | Local fork | Remote training |
|---|---|---|
| Path | `D:\IT\CODE\PYTHON\project1\MiniOneRec` | `/root/autodl-tmp/code` |
| HEAD | `7892d6bdb3856491` (`main`) | `61ad0c8fc184fbc2` (`freeze/v1-docs-and-rl`) |
| Python / torch | 3.10.19 / 2.10.0.dev+cu128 | 3.10.8 / 2.6.0+cu118 |
| GPU | RTX 5070 Laptop (8 151 MiB) | RTX 4090 (24 564 MiB) |
| Identical | `calc.py`, `split.py`, `merge.py` | |
| **Divergent** | `sft.py`, `data.py`, `evaluate.py`, `LogitProcessor.py` | |
| **Remote only** | — | `sid_utils.py` |

## 0. Environment notes that will otherwise bite you

- **`PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` is required on the training host.**
  `tensorboard 2.11.2` requires `protobuf<4` but the image ships `6.33.6`, and
  `transformers.Trainer.__init__` constructs a `TensorBoardCallback` unconditionally, which
  imports `tensorboard` and raises *"Descriptors cannot be created directly"*. `sft.py` sets
  `report_to=None`, so nothing is written — this is a pure import-time incompatibility, and
  the env var fixes it without touching any source.
- **index filenames must be canonical.** `sft.py:151-154` derives the index filename as
  `basename(sid_index_path).split('.')[0] + '.index.json'`. A per-arm name like
  `X.index.b0.json` resolves to `X.index.b0.index.json` and fails. Stage each arm in its own
  directory with the file named `<CAT>.index.json` and `<CAT>.item.json` beside it.
- **Set `HF_HOME` / `HF_DATASETS_CACHE` onto the data volume.** The container's `/` is ~30 G
  and was at 86 % during this project.
- **Root vs data volume.** `/root` is on `/` (overlay); `/root/autodl-tmp` is a separate xfs
  volume. `du -sh /root` mixes them unless you pass `-x`.

## 1. Hash-pinned inputs (must never change)

| File | sha256 (16) |
|---|---|
| `data/Amazon/index/Industrial_and_Scientific.index.json` | `53f1aefbe8b8aa77` |
| `data/Amazon/index/Industrial_and_Scientific.emb-qwen-td.npy` | `91a1c875e4b009f3` |
| `data/Amazon/info/…_5_2016-10-2018-11.txt` | `ebd7d594f2dc884d` |
| `data/Amazon/train/…csv` | `5e5456d4b54681d7` |
| `data/Amazon/valid/…csv` | `44d441a4cbb50e18` |
| `data/Amazon/test/…csv` | `43cea20fc742e481` |

Base model: `Qwen2.5-0.5B/model.safetensors` sha `88c142557820ccad` (988 097 824 B).
Verify with `experiments/minionerec_cat/src/verify_chain.py --tag <tag>` — it checks all of
these and exits **0 on PASS, 1 on FAIL**, always writing the report first.

## 2. RQ-VAE / SID catalogue

```bash
# budget MUST come from the launcher, never the argparse defaults:
#   launcher  epochs=10000 batch_size=20480  ->  1 step/epoch -> 10 000 steps
#   parser    epochs=5000  batch_size=2048   ->  2 steps/epoch -> 16.7x short
python -m experiments.minionerec_cat.src.train_rqvae --tag b0 --seed 2024 --num-workers 0 --device cuda:0
python -m experiments.minionerec_cat.src.extract_index --tag b0 --seed 2024 --device cuda:0 --batch-size 4096
python -m experiments.minionerec_cat.src.rebuild_derived --tag b0
python -m experiments.minionerec_cat.src.verify_chain --tag b0      # expect PASS + exit 0
python -m experiments.minionerec_cat.src.audit_sid --tag b0 --coherence-sample 800
```
B2 additionally needs the collaborative table and then the same chain:
```bash
python -m experiments.minionerec_cat.src.build_collab_pairs --tag b2 --dim 32 --min-degree 3
python -m experiments.minionerec_cat.src.train_rqvae --tag b2 --seed 2024 --num-workers 0 --device cuda:0 \
  --lambda-cf 0.2 --cf-embedding experiments/minionerec_cat/runs/b2/collab/cf_embedding_b2.npy \
  --cf-mask experiments/minionerec_cat/runs/b2/collab/cf_mask_b2.npy --cf-temperature 1.0
```
λ was chosen by gradient-norm balance, reproducible via
`python -m experiments.minionerec_cat.src.lambda_probe --tag b0 --cf-tag b2`.

**Sanity check the round trip:** `tests/test_rebuild_roundtrip.py` feeds the **P0** catalogue
through the rebuild and reproduces `info.txt` **byte-for-byte**. That assertion needs the
private P0 corpus, so it lives in the test's **INTEGRATION** layer; without the corpus the
test runs its 23 synthetic **UNIT** assertions and reports `INTEGRATION SKIPPED` (see §9).

## 3. Qwen SFT (remote training host only)

```bash
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python HF_HOME=/root/autodl-tmp/hf_home
torchrun --nproc_per_node 1 sft.py \
  --base_model /root/autodl-tmp/models/Qwen2.5-0.5B \
  --train_file <ARM>/derived/train/Industrial_and_Scientific_5_2016-10-2018-11.csv \
  --eval_file  <ARM>/derived/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv \
  --output_dir <OUT> --category Industrial_and_Scientific \
  --sid_index_path <ARM>/Industrial_and_Scientific.index.json \
  --item_meta_path <ARM>/Industrial_and_Scientific.item.json \
  --sample -1 --num_epochs 2 --batch_size 64 --micro_batch_size 8 \
  --learning_rate 3e-4 --cutoff_len 512 --seed 42 \
  --train_from_scratch False --freeze_LLM False
```
Expected: the log prints `train dataset components: 36259 + 7321 + 36259 = 79839`, and
**2 496 optimizer steps** for 2 epochs (`1 248` for 1 epoch).

★ **Early stopping is disabled identically for every arm** by launching through
`p4b_train_freq.py`, which filters `EarlyStoppingCallback` from the callback list at runtime —
`sft.py` itself is **not** modified. Validation-based best-checkpoint selection is retained.

## 4. Evaluation (remote, per arm, explicit beam settings)

```bash
CUDA_VISIBLE_DEVICES=0 python -u evaluate.py \
  --base_model <OUT>/final_checkpoint \
  --info_file <ARM>/derived/info/Industrial_and_Scientific_5_2016-10-2018-11.txt \
  --category Industrial_and_Scientific \
  --test_data_path <ARM>/derived/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv \
  --result_json_data <OUT>_beam20.json \
  --batch_size 8 --num_beams 20 --max_new_tokens 256 --length_penalty 0.0
```
★ `--num_beams 20` **must be written explicitly**: `evaluate.sh:49` hardcodes `50`, and the
frozen protocol for every reported number is **20**. Expect the log to print
`[evaluate] inferred SID depth = 3, prefix_index = 3`.

## 5. Metrics

```bash
python calc.py --path <result>.json --item_path <ARM>/derived/info/…txt
```
Validation of the implementation (run on the frozen P0 artifact, expect Δ ≈ 4e-09):
```bash
python experiments/minionerec_cat/tests/test_eval_item_metric.py
```
The test has two layers. Its **UNIT** layer builds a synthetic catalogue and asserts
`load_info`, `collision_sets`, `evaluate_artifact` and the bootstrap boundary rule — it runs
anywhere, with only numpy. Its **INTEGRATION** layer reproduces the published baseline on the
frozen P0 artifact; that needs private assets, so it reports `INTEGRATION SKIPPED` and lists
the missing paths when they are absent. Neither layer simulates a pass.

## 6. Phase 4B frequency-weighted loss

```bash
torchrun --nproc_per_node 1 p4b_train_freq.py --arm C1 --alpha 0.5 \
  --base_model /root/autodl-tmp/models/Qwen2.5-0.5B \
  --train_csv …/train/…csv --eval_file …/valid/…csv --output_dir <OUT> \
  --sid_index_path …/Industrial_and_Scientific.index.json \
  --item_meta_path …/Industrial_and_Scientific.item.json \
  --num_epochs 1 --batch_size 64 --micro_batch_size 8 --learning_rate 3e-4 \
  --cutoff_len 512 --seed 42 --equivalence_probe 0
```
★ **Two things to verify, or the run is meaningless:**
1. the driver log contains
   `[p4b] set TrainingArguments.remove_unused_columns=False (required so '_weight' reaches compute_loss)`
   — without it `Trainer` **silently drops** the `_weight` column and the intervention is
   **inert** (this actually happened; see `PROJECT_REPORT.md` §4.6);
2. `weighted=` differs from `stock=` in the probe lines.

**Correctness tests:**
```bash
python experiments/minionerec_cat/tests/test_collab_loss.py    # 33/33 PASS (fully standalone)
```
It asserts α=0 equivalence (identical loss, identical reconstruction, **max gradient diff
0.000e+00** over 15 parameter tensors) and that no gradient reaches the frozen target.

## 7. What cannot be reproduced

| Item | Why |
|---|---|
| **P0's SID / quantiser** | the RQ-VAE checkpoint was never published; searched local + AutoDL, 0 hits |
| **P0's exact protocol** | budget, seed and hyper-parameters are unknown |
| **Removed optimizer state** (if cleanup Option B runs) | optimizer state is not re-derivable, so those runs cannot be *resumed* — their models and evaluations remain intact |
| **The pre-existing 1.5B / GRPO lines** | belong to an earlier engagement; only their artifacts are present |

## 8. Expected values for a regression check

| artifact | value |
|---|---|
| `b0` best_loss / first-pass collision | 0.272513 / 0.117743 |
| `b0` layer-a codewords / vocab | 61/256 / 573 |
| P0 eval reproduction | HR@20 0.19832341, NDCG@20 0.11786798 (Δ ≈ 4e-09) |
| S0 / S1 / S2 TEST HR@20 | 0.19038165 / 0.19479373 / 0.18376351 |
| Phase-3 VALID hits (S0/S1/S2) | 860 / 862 / 838 of 4 532 |
| **C0 / C1 VALID HR@20** | **0.16769638 / 0.16240071** |
| **C0 / C1 VALID NDCG@20** | **0.10862407 / 0.10422123** |
| `test_collab_loss.py` | **33/33 PASS** — fully standalone (numpy + torch) |
| `test_rebuild_roundtrip.py` | **23/23 UNIT PASS**; P0 integration SKIPPED without the private corpus |
| `test_eval_item_metric.py` | **34/34 UNIT PASS**; P0 integration SKIPPED without the private assets |

Set `CAT_REQUIRE_INTEGRATION=1` to turn a missing private asset into a failure instead of a
skip. Absence is reported explicitly; it is never counted as a pass.
