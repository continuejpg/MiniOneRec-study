# MiniOneRec-CAT — Phase 0 runbook

Everything here lives under `experiments/minionerec_cat/`. **The P0 tree
(`data/Amazon/**`, `checkpoints/**`) is opened read-only and is protected in code
by `common.assert_outside_p0`, which raises rather than allowing a write.**

---

## 1. Why this directory exists

The project goal changed from *reproducing* MiniOneRec to *improving* it via
Semantic ID construction. That requires a SID pipeline we can actually re-run,
which the P0 assets do not provide:

| P0 asset | State | Consequence |
|---|---|---|
| `index.json` (SID catalogue) | present (3 686 items, 3 670 SIDs) | usable as a frozen reference |
| `rq/` quantiser code | present | reusable as-is |
| **RQ-VAE checkpoint** | **absent, everywhere** | **P0's SID cannot be regenerated** |
| `*.inter` raw splits | absent | `convert_dataset.py` cannot be re-run |
| `.emb-qwen-td.npy` | present (3 686 × 2 560) | the quantiser input |

So the experiment needs its own **B0** (text-only control) before **B2**
(behaviour-aware) can be compared against anything.

---

## 2. The SID derivation chain

Replacing `index.json` alone is **not enough** — four other artefacts are derived
from it and the pipeline fails *silently* if any is stale:

```
index.json  (SID catalogue -- source of truth)
    │
    ├─► train/*.csv   item_sid + history_item_sid   ── data.SidSFTDataset (data.py:404,413)
    ├─► valid/*.csv   item_sid + history_item_sid   ── sft.py:204 (eval set)
    ├─► test/*.csv    item_sid                      ── evaluate.py:143 (scored cohort)
    ├─► info/*.txt    SID <TAB> title <TAB> item_id ── evaluate.py:61-99  (decoding trie)
    │                                                └─ calc.py:23       (metric universe)
    └─► SID token vocabulary                        ── sft.py:49-53 TokenExtender
```

`rebuild_derived` regenerates **all** of these together and `verify_chain`
cross-checks them. A stale `info.txt` would constrain decoding to the old SIDs
while the model trains on new ones — which looks like a modelling result.

---

## 3. Layout

```
experiments/minionerec_cat/
├── configs/b0_b2_protocol.json     the declared protocol, arms and guarantees
├── docs/PHASE0_RUNBOOK.md          this file
├── src/
│   ├── common.py                   paths, hashing, the P0 write guard
│   ├── train_rqvae.py              stage 1: train the RQ-VAE
│   ├── extract_index.py            stage 2: checkpoint -> SID catalogue
│   ├── rebuild_derived.py          stage 3: catalogue -> splits + info + vocab
│   └── verify_chain.py             stage 4: cross-check the chain
├── scripts/
│   ├── run_b0.sh                   local end-to-end launcher
│   ├── run_b0_remote.sh            AutoDL end-to-end launcher
│   └── sync_to_autodl.sh           --check / --push / --pull
├── tests/test_rebuild_roundtrip.py the P0 round-trip safety test
└── runs/<tag>/                     ALL generated artefacts  (git-ignored)
    ├── emb/<cat>.emb-<tag>.npy     the embedding copy this run quantises
    ├── rqvae/<timestamp>/best_collision_model.pth
    ├── index/<cat>.index.<tag>.json
    ├── derived/{train,valid,test,info}/
    ├── reports/*.json              manifests, dry run, chain verification
    └── logs/*.log
```

---

## 4. Running B0

### 4.1 Local (validated in Phase 0)

```bash
cd <repo>
python -m experiments.minionerec_cat.src.train_rqvae --tag b0 --dry-run   # validate only
bash experiments/minionerec_cat/scripts/run_b0.sh                        # full chain
```

The full chain is ~3–10 min on CPU for 3 686 items; a GPU only shortens it. It
does **not** need a GPU, which matters because the instance currently has none.

### 4.2 AutoDL (staged; requires a GPU-backed instance)

```bash
# 1. read-only preflight from the workstation
bash experiments/minionerec_cat/scripts/sync_to_autodl.sh --check

# 2. push the code (generated artefacts excluded)
bash experiments/minionerec_cat/scripts/sync_to_autodl.sh --push

# 3. on the instance
ssh -p <port> root@<host>
cd /root/autodl-tmp/code
bash experiments/minionerec_cat/scripts/run_b0_remote.sh

# 4. pull the reports back
bash experiments/minionerec_cat/scripts/sync_to_autodl.sh --pull
```

`run_b0_remote.sh` **refuses to train** when `nvidia-smi` reports no devices, so
it cannot silently fall back to CPU.

---

## 5. What Phase 0 actually proved

| Check | Result |
|---|---|
| P0 catalogue through the rebuild chain | **26/26 PASS** — `info.txt` reproduced **byte-for-byte** |
| RQ-VAE training path | works; smoke run trained, saved a schema-compatible checkpoint |
| `extract_index` | works; 20-round collision retry honoured |
| `rebuild_derived` on a *new* catalogue | 36 259 / 4 532 / 4 533 rows conserved, 0 unknown references |
| `verify_chain` on a new catalogue | PASS — SIDs consistent, trie non-degenerate, vocab drift reported |
| P0 write guard | correctly refuses `data/Amazon/**` and `checkpoints/**` |
| P0 inputs after all of the above | **hash-unchanged** |

The decisive test is `tests/test_rebuild_roundtrip.py`: feeding the **P0**
catalogue through the rebuild must return the **P0** files. It does — which is
what makes a B0/B2 difference attributable to the catalogue rather than the
tooling.

---

## 6. Reference hashes (P0, must never change)

```
53f1aefbe8b8aa77  data/Amazon/index/Industrial_and_Scientific.index.json
91a1c875e4b009f3  data/Amazon/index/Industrial_and_Scientific.emb-qwen-td.npy
ebd7d594f2dc884d  data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt
5e5456d4b54681d7  data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv
44d441a4cbb50e18  data/Amazon/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv
43cea20fc742e481  data/Amazon/test/Industrial_and_Scientific_5_2016-10-2018-11.csv
```

`verify_chain --tag <t>` re-checks these on every run and `run_b0.sh` stage 0
refuses to start if any differs.

---

## 7. Phase 1 blockers

1. **B2 embedding does not exist yet.** `build_collab_embedding` must produce
   `runs/b2/emb/<cat>.emb-fused-b2.npy` from **train-only** co-occurrence.
2. **Scale mismatch.** Text rows have L2 norm ≈ 92 (`emb-qwen-td.npy` is not
   normalised); collaborative vectors will be ~O(1). Fusion must align scales or
   the text term dominates.
3. **Cold items.** 39 items never appear in train (37 are test targets). They
   must fall back to the text-only vector and the count must be reported.
4. **No re-training variance baseline.** A single-seed comparison cannot
   distinguish a SID effect from re-training noise. A second B0 seed is required
   before any B2 claim.
5. **AutoDL instance has no GPU allocated** (`/dev/nvidia0` absent,
   `nvidia-smi` → "No devices were found"), and the root volume had only ~4.3 GB
   free. Both must be resolved before SFT.
