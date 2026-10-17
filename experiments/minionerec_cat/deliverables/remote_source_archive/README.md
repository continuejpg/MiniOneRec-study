# remote_source_archive — the ACTUAL training code

**Why this directory exists.** Phase 5A established that the local fork and the AutoDL
training repo are **not the same code**. `sft.py`, `data.py`, `evaluate.py` and
`LogitProcessor.py` **diverge**, and `sid_utils.py` exists **only** remotely. Every SFT and
every beam-20 evaluation in this project was produced by the **remote** files, so the local
git HEAD alone would misrepresent reproducibility. This is that remote source, byte-exact.

## Files

| file | sha256 (first 16) | role |
|---|---|---|
| `sft.py` | `03cdea6d757380c8` | SFT entry point; adds the SID tokens, resizes embeddings, builds the 3-dataset mix |
| `data.py` | `ef02a7cb8715768a` | `SidSFTDataset` / `SidItemFeatDataset` / `FusionSeqRecDataset` / `EvalSidDataset` |
| `evaluate.py` | `d68af11fa4efb954` | constrained beam-search evaluation; builds the SID prefix trie |
| `LogitProcessor.py` | `789808bad88780e4` | `ConstrainedLogitsProcessor` — forces every emitted token into the catalogue trie |
| `sid_utils.py` | `000ed72a0d96a5c6` | **remote-only**: infers SID depth / prefix index from the catalogue instead of hardcoding it |
| `calc.py` | `95a77747b797e755` | HR/NDCG metric (identical to the local fork) |
| `split.py` | `e7a771e58256bb39` | shards the test file for multi-GPU evaluation |
| `merge.py` | `fc0b5e1751912f99` | merges those shards back before scoring |
| `REMOTE_ENVIRONMENT.txt` | — | container, git, python stack, GPU/driver, required env vars, pip freeze |

**Verification:** every file above was fetched from `/root/autodl-tmp/code/` and its local
sha256 re-checked against the digest recorded **on the remote** — **8/8 OK**. The copies here
are readable locally, so the training code can be inspected without the instance.

## What differs from the local fork

| file | difference | why it matters |
|---|---|---|
| `sft.py` | remote adds `sft_mode` (`full` vs `seq_only`) and prints the per-dataset component sizes | the printed `36259 + 7321 + 36259 = 79839` line is the composition evidence quoted throughout the report |
| `evaluate.py` | remote calls `sid_utils.infer_prefix_index` and adds a discard-warmup pass | SID depth is derived from the catalogue rather than hardcoded — it printed `inferred SID depth = 3, prefix_index = 3` for every arm |
| `LogitProcessor.py` | remote accepts explicit `prefix_index` and `count_0` | `count_0=0` reproduces the original behaviour exactly |
| `data.py` | remote is a superset | the three dataset classes used here behave identically |

## Required environment variables (see `REMOTE_ENVIRONMENT.txt`)

```
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python   # tensorboard 2.11.2 requires protobuf<4; image has 6.33.6
HF_HOME=/root/autodl-tmp/hf_home                # keep caches off the 30 G system disk
HF_DATASETS_CACHE=/root/autodl-tmp/hf_home/datasets
WANDB_MODE=disabled
```

## Credentials

This archive contains **source code and environment metadata only**. No SSH keys, tokens,
passwords or private paths are included; the archive was scanned for eight credential
patterns and is clean.
