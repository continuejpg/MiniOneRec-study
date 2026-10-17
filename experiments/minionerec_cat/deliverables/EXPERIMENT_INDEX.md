# EXPERIMENT_INDEX.md — MiniOneRec-CAT

Every row traces to a concrete artifact on disk. Nothing here is estimated.

**Environments (never conflate them):**

| | Local fork | Remote training (AutoDL) |
|---|---|---|
| Path | `D:\IT\CODE\PYTHON\project1\MiniOneRec` | `/root/autodl-tmp/code` |
| HEAD | `7892d6bdb3856491` (`main`) | `61ad0c8fc184fbc2` (`freeze/v1-docs-and-rl`) |
| Role | authoring, analysis, RQ-VAE training | **all Qwen SFT + all beam-20 evaluation** |
| GPU | RTX 5070 Laptop 8151 MiB | **RTX 4090 24564 MiB** |

`sft.py` / `data.py` / `evaluate.py` / `LogitProcessor.py` **differ between the two**;
`calc.py` / `split.py` / `merge.py` are identical. Full hashes:
`SOURCE_VERSION_MANIFEST.json`.

---

## 1. P0 — historical baseline (frozen reference, not reproducible)

| Property | Value |
|---|---|
| Published HR@20 / NDCG@20 | **0.19832341 / 0.11786798** (`README.md:63`) |
| Eval artifact | `checkpoints/grpo_evidence/eval_clean_sft/test_beam20.json`, sha `51903f9babf883fb` |
| Metric reproduction | re-implementing `calc.py` reproduces both values to **Δ ≈ 4e−09** |
| **RQ-VAE checkpoint** | ★ **absent everywhere** (searched local + remote, 0 hits) |
| Consequence | P0's SID **cannot be regenerated**; P0 is a **frozen reference only** |

---

## 2. CAT quantisation (RQ-VAE) — Phase 1 / 2 / 2B

Common config: 3 levels × 256 codewords, `e_dim` 32, **10 000 optimizer steps**
(launcher `epochs=10000`, `batch_size=20480` ⇒ 1 step/epoch), text embedding
`emb-qwen-td.npy` sha `91a1c875e4b009f3`.

| tag | λ_cf | seed | best_loss | best_collision (raw first pass) | layer-a codewords | vocab | index sha | ckpt sha |
|---|---|---|---|---|---|---|---|---|
| `_b0_undertrained_600step` | 0 | 2024 | 1.2142 | 0.7927 | **5 / 256** | — | `4778e8526c719b97` | kept as evidence |
| `b0` | 0 | 2024 | 0.272513 | **0.117743** | **61 / 256** | 573 | `b5a7d78b0f50e6b5` | `197125ca3960bb98` |
| `b0_seed7` | 0 | 7 | 0.217508 | 0.110418 | 76 / 256 | 587 | `462482ffe4a78adb` | `b3850e71d170de14` |
| `b2_lam005` | 0.05 | 2024 | 0.604928 | 0.092512 | 141 / 256 | 653 | `80429c86e74f3d3a` | `ebbbafa2365e4389` |
| `b2_lam010` | 0.10 | 2024 | 1.1027 | 0.106891 | 92 / 256 | 604 | `e7ec9b78be3013c7` | `91f88fbf73de474f` |
| `b2` | 0.20 | 2024 | 2.011992 | **0.066739** | **163 / 256** | 675 | `7e9c2b9c2f420c41` | `d44cddacbfc53683` |
| `b2_seed7` | 0.20 | 7 | 2.033036 | 0.055073 | 175 / 256 | 687 | `bc251e64399c593c` | `53712a966ba2d315` |

**λ selection was pre-registered, not tuned on any metric:** gradient-norm balance at
initialisation gave ‖∇L_rqvae‖ = 0.0934, ‖∇L_cf‖ = 0.4727 ⇒ λ* = **0.1977**; LETTER's
own default α = 0.1 sits in the same order of magnitude. Test was never consulted.

### ★ Collision-rate definition (must not be mixed)
`best_collision_rate` is the **trainer's first-pass** rate,
`(3686 − distinct 3-code strings)/3686` (`rq/trainer.py:127-152`), with no retry. The
**shipped catalogue** is produced *after* up to 20 Sinkhorn retry rounds
(`rq/generate_indices.py:106-133`) and collapses to **11 groups / 22 items
(0.00298426)** for **every** arm — so the shipped count cannot distinguish arms.
Verified A == B == D for all six arms (`runs/_comparison/collision_definition_check.json`).

---

## 3. Phase 3 — formal recommendation experiment (TEST)

Protocol: Qwen2.5-0.5B from the hash-pinned base, 2 epochs, eff batch 64 (micro 16/accum 4
in the launcher), lr 3e-4, cutoff 512, seed 42, bf16, `load_best_model_at_end`.
Beam 20 / max_new_tokens 256 / length_penalty 0.0 / batch 8. Same 4 533 test rows.

| arm | SID source | λ_cf | HR@20 | NDCG@20 | model sha | eval sha |
|---|---|---|---|---|---|---|
| **S0** | `b0` seed 2024 | 0 | **0.19038165** | **0.11791880** | `ceb6808e85260a2c` | `133c0b020d9e9dc5` |
| **S1** | `b2_lam010` | 0.10 | **0.19479373** | **0.12124519** | `47e291e094231fa0` | `68249656bd93b5b4` |
| **S2** | `b2` | 0.20 | **0.18376351** | **0.11444928** | `e52bc398af02f759` | `9ff03aa1a10f9708` |

**Contrasts vs S0 (absolute pp) and paired bootstrap (4 533 rows):**

| contrast | HR Δ | 95 % CI (row) | 95 % CI (user-clustered) | NDCG Δ | 95 % CI (row) | 95 % CI (user) |
|---|---|---|---|---|---|---|
| S1−S0 | **+0.4412** | [−0.2647, +1.1692] ❌ | [−0.2697, +1.1430] ❌ | **+0.3326** | [−0.0055, +0.6879] ❌ | [−0.0062, +0.6753] ❌ |
| S2−S0 | **−0.6618** | [−1.3904, +0.0662] ❌ | [−1.3810, +0.0645] ❌ | **−0.3470** | [−0.7056, +0.0064] ❌ | [−0.6839, −0.0135] ✓ |

❌ = CI contains zero. **7 of 8 intervals contain zero**; the single exclusion is the
user-clustered S2 NDCG, which is **negative** and one of eight comparisons.

**Per-sample transitions (S1 vs S0):** both-hit 729, S1-wins 133, S1-loses 131;
among the 729, S1 ranks better 163, worse 169, equal 397 — i.e. **the arms re-order hits
rather than find more of them**.

**Training:** 2 496 optimizer steps each, ~32 min each, peak VRAM 10 218 MiB,
**0 non-finite values** in 2 451–2 490 logged values per arm.

**Diagnostics:** legal SID 100 %, 0 duplicate rows, 261 ambiguous rows (5.76 %),
**identity_mismatch = 0** ⇒ repo SID-level == Canonical-Item.

**Phase-3 VALID recomutation (used by Phase 4A):**

| arm | rows | hits | HR@20 | NDCG@20 |
|---|---|---|---|---|
| S0 | 4 532 | 860 | **0.18976169** | 0.12578678 |
| S1 | 4 532 | 862 | **0.19020300** | 0.12530986 |
| S2 | 4 532 | 838 | **0.18490733** | 0.12270049 |

Artifacts: `experiments/minionerec_cat/runs/_comparison/phase3/`

---

## 4. Phase 4A — long-tail diagnosis (VALID only)

Corrected definitions and results:

| stratum | valid rows | S0 HR@20 | S1 HR@20 | S2 HR@20 |
|---|---|---|---|---|
| freq0 | 149 | 0.02013 | 0.00000 | 0.02685 |
| low(1-2) | 448 | 0.02455 | 0.03348 | 0.02232 |
| mid(3-20) | 2 719 | 0.10188 | 0.10482 | 0.09746 |
| high(>20) | 1 216 | 0.46793 | 0.46217 | 0.45970 |

**Exposure bias (VALID, S0):** slots-per-stratum ÷ rows-per-stratum = freq0 **0.007×**,
low **0.031×**, mid 0.535×, high **2.519×** — a 360× spread.
Global item coverage (S0) **2 611 / 3 686 = 70.8 %** — a **different quantity** from HR@20
(items vs requests); they must not be subtracted.

**Corrected collision definitions:** ambiguous rows (target SID shared) = **207 VALID /
261 TEST**; non-lowest-id rows = **76 VALID / 97 TEST**; 11 groups / 22 items per arm.

**Exposure statistics (train supervision per item):** next-item mean 0.00 / 1.68 / 7.28 /
45.00 for freq0 / low / mid / high; **every item has ≥ 1 total supervision** (auxiliary
tasks), so "zero-supervision item" does not exist.

**Timeline of the Phase-4A correction (mine):** the machine-readable artifacts were
always split-correct; **my report text** pasted the TEST row into the VALID row. Caught and
corrected in Phase 5A. See `PROJECT_REPORT.md` §10.

Artifacts: `/root/autodl-tmp/p4a`, `/root/autodl-tmp/p4a_valid`.

---

## 5. Phase 4B — frequency-aware loss (VALID only; TEST never run)

**Weight:** `raw = (1 + train_next_item_freq)^(-0.5)`, normalised on train targets,
clipped to [0.25, 4.0]. Applied to **SidSFTDataset + FusionSeqRecDataset**; weight 1 for
**SidItemFeatDataset**. Prompt labels stay −100.
**Aggregation:** `loss_w = Σ w_i·Σ_t l_it / Σ w_i·n_i` (n_i = supervised tokens/sample).

Weight distribution at α = 0.5: overall mean 1.0002, min 0.25, max 2.6005;
freq0 2.6005, low 1.6104, mid 0.9716, **high 0.4414**.

### ★ FINAL VALID RESULTS (the numbers this project stands on)

| metric | **C0 (α=0)** | **C1 (α=0.5)** | C1−C0 |
|---|---|---|---|
| rows | 4 532 | 4 532 | — |
| hits | **760** | **736** | −24 |
| **HR@20** | **0.16769638** | **0.16240071** | **−0.5296 pp** |
| **NDCG@20** | **0.10862407** | **0.10422123** | **−0.4403 pp** |

**Paired bootstrap C1−C0 (same rows/order):**

| metric | unit | Δ (pp) | 95 % CI | excludes 0 |
|---|---|---|---|---|
| HR@20 | row (4 532) | −0.5296 | [−1.2136, +0.1545] | ❌ |
| HR@20 | user (2 011) | −0.5296 | [−1.2815, +0.2435] | ❌ |
| NDCG@20 | row | −0.4403 | [−0.7412, −0.1213] | ✓ (negative) |
| NDCG@20 | user | −0.4403 | [−0.7829, −0.0799] | ✓ (negative) |

**Per stratum (user-clustered CI, all eight exclude zero):**

| stratum | rows / users | C0 HR@20 | C1 HR@20 | HR Δ (pp) | HR 95 % CI | NDCG Δ (pp) | NDCG 95 % CI |
|---|---|---|---|---|---|---|---|
| freq0 | 149 / 112 | 0.00000 | **0.03356** | **+3.3557** | [+0.6944, +6.4935] | **+0.8352** | [+0.1725, +1.6156] |
| low(1-2) | 448 / 373 | 0.01116 | **0.02455** | **+1.3393** | [+0.4367, +2.4719] | **+1.3851** | [+0.3633, +2.7204] |
| mid(3-20) | 2 719 / 1 508 | 0.07503 | **0.09526** | **+2.0228** | [+1.1029, +3.0737] | **+0.7426** | [+0.3647, +1.1791] |
| **high(>20)** | 1 216 / 782 | 0.45312 | 0.37911 | **−7.4013** | [−9.2295, −5.6650] | **−3.9141** | [−4.7108, −3.1468] |

**Exposure shift (Top-20 slots by item train-frequency):** freq0 +0.02 pp, low +0.64 pp,
**mid +29.65 pp** (25.49 → 55.14), **high −30.31 pp** (74.39 → 44.08).
**Global item coverage:** C0 1 898 (51.49 %) → C1 **2 413 (65.46 %)**; Top-1 758 → 1 027.

**Training:** 1 248 steps each; C0 1 116 s / train_loss 9.175849; C1 1 258 s /
10.720048 (higher because rare items carry more weight — not divergence);
0 non-finite values in 1 248 logged values each.

### ★ The inert-C1 bug (must be reported, not hidden)
The first C1 run produced a model **byte-identical** to C0 (`5a6db35ceb74389d`) and the
same eval file. Root cause: `transformers.Trainer` defaults to
`remove_unused_columns=True`, which **silently drops** any dataset column not in
`model.forward`'s signature — so `_weight` never reached `compute_loss` and the override
fell through to the stock loss. Fix: set `remove_unused_columns=False`. After the fix
`compute_loss` receives the weights and weighted ≠ stock on **all 20 753** sampled batches
(mean |Δ| 0.1836, max 2.8660). The void run is quarantined at
`/root/autodl-tmp/p4b_archive/void_C1/`.

Artifacts: `experiments/minionerec_cat/runs/_comparison/phase4/`
(`p4b_final_metrics.json`, `p4b_stratum_bootstrap.json`, `p4b_record.json`).

---

## 6. Metric implementation — verified, not assumed

Re-implementing `calc.py:34-81` on the frozen P0 artifact reproduces the published
baseline to **Δ ≈ 4e−09** with **0 unmapped predictions** and **20 beams**.
NDCG gain `1/log(rank+2)` normalised by `1/log 2` is **algebraically identical** to the
standard binary `1/log2(rank+1)`. Collision convention: `calc.py` builds
`sid → [item ids]` but only uses it to count `CC`; scoring is pure string equality, so
the canonical item is the **lowest item id** for that SID.

---

## 7. Test-set discipline

| phase | TEST used? |
|---|---|
| Phase 1–2B (RQ-VAE) | no — SID catalogue metrics only |
| Phase 3 | **yes, once**, after all configs were frozen |
| Phase 4A | no (VALID only; TEST read descriptively for cross-check) |
| **Phase 4B** | ★ **never run** |

No configuration was ever changed after seeing TEST.
